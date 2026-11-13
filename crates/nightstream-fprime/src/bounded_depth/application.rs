//! A native application in place of a step package's own
//! (`StepPackage.Package.application`). The application is each step
//! circuit's first child, so its region leads every column range: its witness
//! words in the shared words, its locals from the first local, its pieces from
//! the first piece, its fresh columns from `first_fresh` and its blocks' S-box
//! outputs from `sbox_start`. Substitution drops the region, moves every later
//! column, and puts the application's rows first. Its variables are the
//! step's current state, its witness words, its next state and one local per
//! recipe (`ApplicationCircuit`'s variable order).

use std::ops::ControlFlow;
use std::sync::Arc;

use neo_ccs::crypto::poseidon2_goldilocks as poseidon2;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

use super::words::PUBLIC_WORDS;
use super::{BoundedDepthPackages, Linear, Piece, Region, Row, StepPackage};
use crate::application_records::ApplicationRecords;
use crate::package::PackageError;

/// The shared words before the states: the encoded digest, the five counter
/// words and the two key digests (`StepWords::columns`).
const STATES_START: usize = PUBLIC_WORDS + 5 + 8;
const WITNESS_START: usize = STATES_START + 12;
const BLOCK_LOCALS: usize = 1096;
const BLOCK_OUTPUTS: usize = 1080;
const SBOX_COLUMNS: usize = 150;
const RING_DEGREE: usize = 54;
const IDENTITY_DOMAIN: &[u8] = b"Nightstream/OptionB/application/v1";

fn moved(column: usize, by: i128) -> Option<usize> {
    usize::try_from(column as i128 + by).ok()
}

fn piece_columns(piece: &Piece) -> Vec<usize> {
    let linear = |linear: &Linear| {
        linear
            .terms
            .iter()
            .map(|&(column, _)| column)
            .collect::<Vec<_>>()
    };
    match piece {
        Piece::Rows(rows) => rows
            .iter()
            .flat_map(|row| [&row.a, &row.b, &row.c])
            .flat_map(linear)
            .collect(),
        Piece::Ring {
            cells,
            start,
            quotient,
            lanes,
        } => (*start..start + RING_DEGREE * cells)
            .chain(*quotient..quotient + RING_DEGREE * cells)
            .chain(lanes.iter().flat_map(linear))
            .collect(),
        Piece::Block { start, lanes } => (*start..start + BLOCK_LOCALS)
            .chain(lanes.iter().flat_map(linear))
            .collect(),
    }
}

fn map_linear(linear: &Linear, map: &impl Fn(usize) -> Option<usize>) -> Option<Linear> {
    Some(Linear {
        constant: linear.constant,
        terms: linear
            .terms
            .iter()
            .map(|&(column, coefficient)| Some((map(column)?, coefficient)))
            .collect::<Option<_>>()?,
    })
}

fn map_piece(piece: &Piece, map: &impl Fn(usize) -> Option<usize>) -> Option<Piece> {
    let lanes = |lanes: &[Linear]| {
        lanes
            .iter()
            .map(|lane| map_linear(lane, map))
            .collect::<Option<Vec<_>>>()
    };
    Some(match piece {
        Piece::Rows(rows) => Piece::Rows(
            rows.iter()
                .map(|row| {
                    Some(Row {
                        a: map_linear(&row.a, map)?,
                        b: map_linear(&row.b, map)?,
                        c: map_linear(&row.c, map)?,
                    })
                })
                .collect::<Option<_>>()?,
        ),
        Piece::Ring {
            cells,
            start,
            quotient,
            lanes: ring,
        } => Piece::Ring {
            cells: *cells,
            start: map(*start)?,
            quotient: map(*quotient)?,
            lanes: lanes(ring)?,
        },
        Piece::Block { start, lanes: block } => Piece::Block {
            start: map(*start)?,
            lanes: lanes(block)?,
        },
    })
}

/// What a native application places natively: its Poseidon2 blocks, over
/// the application's variables.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeShape {
    pub blocks: Vec<NativeBlock>,
}

/// One permutation as a block (`PackagePiece.block`): its 1,096 locals from
/// variable `first_local` in the order of `Gadgets/Poseidon2/Permutation.lean`,
/// the last 16 its output, and its input lanes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBlock {
    pub first_local: usize,
    pub lanes: [Linear; 16],
}

impl StepPackage {
    /// The column of application variable `variable` under this package's
    /// region.
    fn variable_column(&self, variable: usize) -> Result<usize, PackageError> {
        let Region {
            witness_words,
            local_start,
            locals,
            ..
        } = self.application;
        let column = match variable {
            v if v < 4 => STATES_START + 4 + v,
            v if v < 4 + witness_words => WITNESS_START + v - 4,
            v if v < 8 + witness_words => STATES_START + 8 + v - 4 - witness_words,
            v if v < 8 + witness_words + locals => local_start + v - 8 - witness_words,
            _ => return Err(PackageError::Input("application variable outside its region")),
        };
        Ok(column)
    }

    /// Fills the native application's locals in recipe order: each is the
    /// product of its recipe row's A and B sides (`ApplicationCircuit::execute`).
    pub(super) fn fill_application(
        &self,
        records: &ApplicationRecords,
        columns: &mut [Goldilocks],
    ) -> Result<(), PackageError> {
        let mut walk = records.walk();
        for recipe in 0..records.recipe_count() {
            let row = walk.recipe_row(recipe)?;
            let [a, b, _] = walk.evaluate_row(row, |variable| {
                Ok::<_, PackageError>(columns[self.variable_column(variable)?].as_canonical_u64())
            })?;
            columns[self.application.local_start + recipe] = Goldilocks::from_u64(a) * Goldilocks::from_u64(b);
        }
        Ok(())
    }

    /// This package with `records` in place of its application and `shape`'s
    /// blocks in place of their rows. The pieces and witness batches outside
    /// the region keep their order; a dropped piece may read only the shared
    /// words and the region, and a kept piece or batch may not read the
    /// region. The application's blocks lead the S-box columns. The slots
    /// follow Lean's rule on the new pieces (`native_slots`).
    fn with_application(
        &self,
        records: &Arc<ApplicationRecords>,
        witness_words: usize,
        shape: &NativeShape,
    ) -> Result<Self, PackageError> {
        let old = self.application;
        if old.pieces > self.pieces.len() || old.local_start < WITNESS_START + old.witness_words {
            return Err(PackageError::Malformed("bounded-depth application region"));
        }
        let blocks = self.pieces[..old.pieces]
            .iter()
            .filter(|piece| matches!(piece, Piece::Block { .. }))
            .count();
        let words = witness_words as i128 - old.witness_words as i128;
        let locals = words + records.recipe_count() as i128 - old.locals as i128;
        let fresh = locals - old.fresh as i128;
        let sbox = fresh + (SBOX_COLUMNS as i128) * (shape.blocks.len() as i128 - blocks as i128);
        let region_locals = old.local_start..old.local_start + old.locals;
        let region_fresh = self.first_fresh..self.first_fresh + old.fresh;
        let region_sbox = self.sbox_start..self.sbox_start + SBOX_COLUMNS * blocks;
        let map = |column: usize| match column {
            c if c < WITNESS_START => Some(c),
            c if c < WITNESS_START + old.witness_words => None,
            c if c < old.local_start => moved(c, words),
            c if region_locals.contains(&c) => None,
            c if c < self.first_fresh => moved(c, locals),
            c if region_fresh.contains(&c) => None,
            c if c < self.sbox_start => moved(c, fresh),
            c if region_sbox.contains(&c) => None,
            c => moved(c, sbox),
        };
        let inside = |column: usize| {
            column < old.local_start || region_locals.contains(&column) || region_fresh.contains(&column)
        };
        if !self.pieces[..old.pieces]
            .iter()
            .all(|piece| piece_columns(piece).into_iter().all(inside))
        {
            return Err(PackageError::Malformed(
                "bounded-depth application piece reads past its region",
            ));
        }
        let mut package = Self {
            first_fresh: moved(self.first_fresh, locals).ok_or(PackageError::Malformed("bounded-depth columns"))?,
            sbox_start: moved(self.sbox_start, fresh).ok_or(PackageError::Malformed("bounded-depth columns"))?,
            application: Region {
                witness_words,
                local_start: moved(old.local_start, words).ok_or(PackageError::Malformed("bounded-depth columns"))?,
                locals: records.recipe_count(),
                pieces: 1 + shape.blocks.len(),
                fresh: 0,
            },
            native: Some(Arc::clone(records)),
            pieces: Vec::new(),
            witness: Vec::new(),
            slots: Vec::new(),
            ..self.clone()
        };
        let mut pieces = package.native_pieces(records, shape)?;
        for piece in &self.pieces[old.pieces..] {
            pieces.push(
                map_piece(piece, &map).ok_or(PackageError::Malformed("bounded-depth piece reads the application"))?,
            );
        }
        package.pieces = pieces;
        package.witness = self
            .witness
            .iter()
            .filter(|batch| !region_locals.contains(&batch.start))
            .map(|batch| {
                batch.map_columns(&map).ok_or(PackageError::Malformed(
                    "bounded-depth witness batch reads the application",
                ))
            })
            .collect::<Result<_, _>>()?;
        package.slots = package.native_slots();
        Ok(package)
    }

    /// The application's pieces: one piece of every row outside the blocks,
    /// then one block per permutation of `shape`. The blocks take ascending,
    /// disjoint locals; a block drops the recipe rows of its locals; no kept
    /// row and no lane reads a dead local (`Frame.Framed`), and a lane reads
    /// only variables before its block.
    fn native_pieces(&self, records: &ApplicationRecords, shape: &NativeShape) -> Result<Vec<Piece>, PackageError> {
        let first = 8 + self.application.witness_words;
        let end = first + records.recipe_count();
        let mut dead = vec![false; end];
        let mut dropped = vec![false; records.row_count()];
        let mut free = first;
        let mut blocks = Vec::with_capacity(shape.blocks.len());
        for block in &shape.blocks {
            let start = block.first_local;
            if start < free || end < BLOCK_LOCALS || start > end - BLOCK_LOCALS {
                return Err(PackageError::Malformed(
                    "bounded-depth native block outside the free locals",
                ));
            }
            let lanes = block
                .lanes
                .iter()
                .map(|lane| {
                    if lane
                        .terms
                        .iter()
                        .any(|&(variable, _)| variable >= start || dead[variable])
                    {
                        return Err(PackageError::Malformed(
                            "bounded-depth native lane reads past its inputs",
                        ));
                    }
                    self.columns_of(lane)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for recipe in start - first..start - first + BLOCK_LOCALS {
                let row = dropped
                    .get_mut(records.recipe_row(recipe)?)
                    .ok_or(PackageError::Malformed("bounded-depth native block row"))?;
                if std::mem::replace(row, true) {
                    return Err(PackageError::Malformed("bounded-depth native block row in two blocks"));
                }
            }
            dead[start..start + BLOCK_OUTPUTS].fill(true);
            blocks.push(Piece::Block {
                start: self.variable_column(start)?,
                lanes,
            });
            free = start + BLOCK_LOCALS;
        }
        let rows = self.application_rows(records, &dropped, &dead)?;
        Ok(std::iter::once(Piece::Rows(rows)).chain(blocks).collect())
    }

    fn columns_of(&self, linear: &Linear) -> Result<Linear, PackageError> {
        Ok(Linear {
            constant: linear.constant,
            terms: linear
                .terms
                .iter()
                .map(|&(variable, coefficient)| Ok((self.variable_column(variable)?, coefficient)))
                .collect::<Result<_, PackageError>>()?,
        })
    }

    /// The application's rows outside the blocks over the step's columns.
    fn application_rows(
        &self,
        records: &ApplicationRecords,
        dropped: &[bool],
        dead: &[bool],
    ) -> Result<Vec<Row>, PackageError> {
        (0..records.row_count())
            .filter(|&row| !dropped[row])
            .map(|row| {
                let header = records.row_header(row)?;
                let mut sides = header.constants.map(|constant| Linear {
                    constant: Goldilocks::from_u64(constant),
                    terms: Vec::new(),
                });
                let _: ControlFlow<()> = records.visit_terms(row, |term| {
                    if dead.get(term.variable) == Some(&true) {
                        return Err(PackageError::Malformed(
                            "bounded-depth application row reads a dead local",
                        ));
                    }
                    sides[term.form.index()].terms.push((
                        self.variable_column(term.variable)?,
                        Goldilocks::from_u64(term.coefficient),
                    ));
                    Ok(ControlFlow::Continue(()))
                })?;
                let [a, b, c] = sides;
                Ok(Row { a, b, c })
            })
            .collect()
    }

    /// The slot runs of this package by Lean's rule (`StepPackage.runsOf`).
    /// The bit candidates are its `bit` hints and every live column of the
    /// application: its witness words and its locals that are not a block's
    /// dead locals. A candidate gets a bit slot only if a row forces it to 0
    /// or 1.
    fn native_slots(&self) -> Vec<(usize, usize, u8)> {
        let region = self.application;
        let dead: Vec<_> = self
            .blocks()
            .take(region.pieces - 1)
            .map(|block| block.start()..block.outputs())
            .collect();
        let mut bits = self.bit_columns();
        bits.extend(WITNESS_START..WITNESS_START + region.witness_words);
        bits.extend(
            (region.local_start..region.local_start + region.locals)
                .filter(|column| !dead.iter().any(|locals| locals.contains(column))),
        );
        self.rule_slots(&bits)
    }
}

impl BoundedDepthPackages {
    /// Both packages with a native application in place of Lean's. The
    /// identity binds the packages' identity, the witness word count, the
    /// application's rows and recipe rows, then the shape: each block's first
    /// local and lanes.
    pub fn with_application(
        &self,
        records: Arc<ApplicationRecords>,
        witness_words: usize,
        shape: &NativeShape,
    ) -> Result<Self, PackageError> {
        let mut input = poseidon2::Poseidon2Hasher::default();
        let mut update = |word: u64| input.update(&[Goldilocks::from_u64(word)]);
        IDENTITY_DOMAIN
            .iter()
            .for_each(|&byte| update(u64::from(byte)));
        self.identity.into_iter().for_each(&mut update);
        update(witness_words as u64);
        update(records.row_count() as u64);
        for row in 0..records.row_count() {
            let header = records.row_header(row)?;
            for form in 0..3 {
                update(header.constants[form]);
                update(header.term_counts[form] as u64);
            }
            let _: ControlFlow<()> = records.visit_terms(row, |term| {
                update(term.variable as u64);
                update(term.coefficient);
                Ok(ControlFlow::Continue(()))
            })?;
        }
        update(records.recipe_count() as u64);
        for recipe in 0..records.recipe_count() {
            update(records.recipe_row(recipe)? as u64);
        }
        update(shape.blocks.len() as u64);
        for block in &shape.blocks {
            update(block.first_local as u64);
            for lane in &block.lanes {
                update(lane.constant.as_canonical_u64());
                update(lane.terms.len() as u64);
                for &(variable, coefficient) in &lane.terms {
                    update(variable as u64);
                    update(coefficient.as_canonical_u64());
                }
            }
        }
        Ok(Self {
            light: self
                .light
                .with_application(&records, witness_words, shape)?,
            refresh: self
                .refresh
                .with_application(&records, witness_words, shape)?,
            identity: input.finalize().map(|word| word.as_canonical_u64()),
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_application.rs"]
mod tests;
