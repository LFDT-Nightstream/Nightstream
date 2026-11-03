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
use super::{Linear, OptionBPackages, Piece, Region, Row, StepPackage};
use crate::application_records::ApplicationRecords;
use crate::package::PackageError;

/// The shared words before the states: the encoded digest, the five counter
/// words and the two key digests (`StepWords::columns`).
const STATES_START: usize = PUBLIC_WORDS + 5 + 8;
const WITNESS_START: usize = STATES_START + 12;
const BLOCK_LOCALS: usize = 1096;
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
        for recipe in 0..records.recipe_count() {
            let row = records.recipe_row(recipe)?;
            let [a, b, _] = records.evaluate_row(row, |variable| {
                Ok::<_, PackageError>(columns[self.variable_column(variable)?].as_canonical_u64())
            })?;
            columns[self.application.local_start + recipe] = Goldilocks::from_u64(a) * Goldilocks::from_u64(b);
        }
        Ok(())
    }

    /// This package with `records` in place of its application. The pieces,
    /// witness batches and slots outside the region keep their order; a
    /// dropped piece may read only the shared words and the region, and a kept
    /// piece or batch may not read the region.
    fn with_application(&self, records: &Arc<ApplicationRecords>, witness_words: usize) -> Result<Self, PackageError> {
        let old = self.application;
        if old.pieces > self.pieces.len() || old.local_start < WITNESS_START + old.witness_words {
            return Err(PackageError::Malformed("option-B application region"));
        }
        let blocks = self.pieces[..old.pieces]
            .iter()
            .filter(|piece| matches!(piece, Piece::Block { .. }))
            .count();
        let words = witness_words as i128 - old.witness_words as i128;
        let locals = words + records.recipe_count() as i128 - old.locals as i128;
        let fresh = locals - old.fresh as i128;
        let sbox = fresh - (SBOX_COLUMNS * blocks) as i128;
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
                "option-B application piece reads past its region",
            ));
        }
        let mut package = Self {
            first_fresh: moved(self.first_fresh, locals).ok_or(PackageError::Malformed("option-B columns"))?,
            sbox_start: moved(self.sbox_start, fresh).ok_or(PackageError::Malformed("option-B columns"))?,
            application: Region {
                witness_words,
                local_start: moved(old.local_start, words).ok_or(PackageError::Malformed("option-B columns"))?,
                locals: records.recipe_count(),
                pieces: 1,
                fresh: 0,
            },
            native: Some(Arc::clone(records)),
            pieces: Vec::new(),
            witness: Vec::new(),
            slots: Vec::new(),
            ..self.clone()
        };
        let mut pieces = vec![Piece::Rows(package.application_rows(records)?)];
        for piece in &self.pieces[old.pieces..] {
            pieces.push(map_piece(piece, &map).ok_or(PackageError::Malformed("option-B piece reads the application"))?);
        }
        package.pieces = pieces;
        package.witness = self
            .witness
            .iter()
            .filter(|batch| !region_locals.contains(&batch.start))
            .map(|batch| {
                batch
                    .map_columns(&map)
                    .ok_or(PackageError::Malformed("option-B witness batch reads the application"))
            })
            .collect::<Result<_, _>>()?;
        package.slots = self.mapped_slots(&map, &package)?;
        Ok(package)
    }

    /// The application's rows over the step's columns.
    fn application_rows(&self, records: &ApplicationRecords) -> Result<Vec<Row>, PackageError> {
        (0..records.row_count())
            .map(|row| {
                let header = records.row_header(row)?;
                let mut sides = header.constants.map(|constant| Linear {
                    constant: Goldilocks::from_u64(constant),
                    terms: Vec::new(),
                });
                let _: ControlFlow<()> = records.visit_terms(row, |term| {
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

    /// The slot runs of `next`: every kept column keeps its kind, and the new
    /// witness words and locals are field elements.
    fn mapped_slots(
        &self,
        map: &impl Fn(usize) -> Option<usize>,
        next: &Self,
    ) -> Result<Vec<(usize, usize, u8)>, PackageError> {
        const NO_SLOT: u8 = 3;
        let end = next.sbox_start + SBOX_COLUMNS * next.block_count();
        let mut codes = vec![NO_SLOT; end];
        for &(start, count, kind) in &self.slots {
            for column in start..start + count {
                if let Some(target) = map(column) {
                    *codes
                        .get_mut(target)
                        .ok_or(PackageError::Malformed("option-B slot run"))? = kind;
                }
            }
        }
        let region = next.application;
        for column in (WITNESS_START..WITNESS_START + region.witness_words)
            .chain(region.local_start..region.local_start + region.locals)
        {
            codes[column] = 2;
        }
        let mut runs: Vec<(usize, usize, u8)> = Vec::new();
        for (column, &kind) in codes.iter().enumerate().skip(self.public_columns) {
            match runs.last_mut() {
                _ if kind == NO_SLOT => {}
                Some((start, count, last)) if *last == kind && *start + *count == column => *count += 1,
                _ => runs.push((column, 1, kind)),
            }
        }
        Ok(runs)
    }
}

impl StepPackage {
    /// This package with `witness_words` application witness words and no
    /// application rows: the finish package, which has no application, keeps
    /// the step's shared words.
    pub fn with_witness_words(&self, witness_words: usize) -> Result<Self, PackageError> {
        let empty = Arc::new(crate::application_records::ApplicationRecordsWriter::new()?.finish()?);
        self.with_application(&empty, witness_words)
    }
}

impl OptionBPackages {
    /// Both packages with a native application in place of Lean's. The
    /// identity binds the packages' identity, the witness word count and the
    /// application's rows and recipe rows.
    pub fn with_application(
        &self,
        records: Arc<ApplicationRecords>,
        witness_words: usize,
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
        Ok(Self {
            light: self.light.with_application(&records, witness_words)?,
            refresh: self.refresh.with_application(&records, witness_words)?,
            identity: input.finalize().map(|word| word.as_canonical_u64()),
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/option_b_application.rs"]
mod tests;
