//! Lean `StepPackage.Package.application`.

use std::collections::HashMap;
use std::sync::Arc;

use neo_ccs::crypto::poseidon2;
use neo_math::Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::packed_boolean::PackedBooleans;
use super::{BoundedDepthPackages, Linear, Piece, Region, Row, StepPackage};
use crate::application_records::{ApplicationRecords, ApplicationWalk};
use crate::package::{PackageError, BLOCK_LOCALS, BLOCK_OUTPUTS, PUBLIC_WORDS, RING_DEGREE, SBOX_ROWS as SBOX_COLUMNS};

const STATE_WORDS: usize = 4;
const STATES_START: usize = PUBLIC_WORDS + 5 + 8;
pub(super) const WITNESS_START: usize = STATES_START + 12;
const IDENTITY_DOMAIN: &[u8] = b"Nightstream/OptionB/application/combined-native-experiment/v1";

fn moved(column: usize, by: i128) -> Option<usize> {
    usize::try_from(column as i128 + by).ok()
}

pub(super) fn piece_columns(piece: &Piece) -> Vec<usize> {
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeShape {
    pub blocks: Vec<NativeBlock>,
}

/// Lean `PackagePiece.block`, with locals in the order of `Gadgets/Poseidon2/Permutation.lean`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBlock {
    pub first_local: usize,
    pub lanes: [Linear; 16],
}

impl StepPackage {
    fn variable_column(&self, variable: usize) -> Result<usize, PackageError> {
        let Region {
            witness_words,
            local_start,
            locals,
            ..
        } = self.application;
        let column = match variable {
            v if v < STATE_WORDS => STATES_START + STATE_WORDS + v,
            v if v < STATE_WORDS + witness_words => WITNESS_START + v - STATE_WORDS,
            v if v < 2 * STATE_WORDS + witness_words => {
                STATES_START + 2 * STATE_WORDS + v - STATE_WORDS - witness_words
            }
            v if v < 2 * STATE_WORDS + witness_words + locals => local_start + v - 2 * STATE_WORDS - witness_words,
            _ => return Err(PackageError::Input("application variable outside its region")),
        };
        Ok(column)
    }

    pub(super) fn fill_application(
        &self,
        records: &ApplicationRecords,
        columns: &mut [Fq],
    ) -> Result<(), PackageError> {
        let mut walk = records.walk();
        for recipe in 0..records.recipe_count() {
            let row = walk.recipe_row(recipe)?;
            let [a, b, _] = walk.evaluate_row(row, |variable| {
                Ok::<_, PackageError>(columns[self.variable_column(variable)?].as_canonical_u64())
            })?;
            columns[self.application.local_start + recipe] = Fq::from_u64(a) * Fq::from_u64(b);
        }
        Ok(())
    }

    /// Lean `Export.Generic.BooleanInference.derived_valid`.
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
        let region_locals = old.local_start..old.local_start + old.locals;
        let region_fresh = self.first_fresh..self.first_fresh + old.fresh;
        let region_sbox = self.sbox_start..self.sbox_start + SBOX_COLUMNS * blocks;
        let inside = |column: usize| {
            column < old.local_start
                || region_locals.contains(&column)
                || region_fresh.contains(&column)
                || (self.packed_booleans.start..self.packed_booleans.end()).contains(&column)
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
            circuit: self.circuit,
            public_columns: self.public_columns,
            width: self.width,
            first_fresh: moved(self.first_fresh, locals).ok_or(PackageError::Malformed("bounded-depth columns"))?,
            sbox_start: moved(self.sbox_start, fresh).ok_or(PackageError::Malformed("bounded-depth columns"))?,
            application: Region {
                witness_words,
                local_start: moved(old.local_start, words).ok_or(PackageError::Malformed("bounded-depth columns"))?,
                locals: records.recipe_count(),
                pieces: 0,
                fresh: 0,
            },
            native: Some(Arc::clone(records)),
            packed_booleans: PackedBooleans::default(),
            field_lowering: super::field_lowering::FieldLowering::default(),
            guard_fusion: super::guard_fusion::GuardFusion::default(),
            word_packing: super::word_packing::WordPacking::default(),
            pieces: Vec::new(),
            witness: Vec::new(),
            slots: Vec::new(),
        };
        let (mut pieces, block_rows) = package.native_pieces(records, shape)?;
        package.application.pieces = pieces.len();
        let kept_blocks = pieces.len() - 1;
        let sbox = fresh + SBOX_COLUMNS as i128 * (kept_blocks as i128 - blocks as i128);
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
        let (bits, forced) = package.slot_facts()?;
        package.slots = package.rule_slots(&bits, &forced);
        package.pack_native_booleans()?;
        let (packed_bits, packed_forced) = package.slot_facts()?;
        package.slots = package.rule_slots(&packed_bits, &packed_forced);
        package.field_lowering = super::field_lowering::FieldLowering::checked(&package)?;
        package.slots = package.rule_slots(&packed_bits, &packed_forced);
        package.guard_fusion = super::guard_fusion::GuardFusion::checked(&package, &block_rows)?;
        package.word_packing = super::word_packing::WordPacking::checked(&package)?;
        package.check_packed_slots()?;
        Ok(package)
    }

    /// `SPEC.md` section 7.10, step 5.
    fn check_packed_slots(&self) -> Result<(), PackageError> {
        let layout = self.layout()?;
        let packed = &self.packed_booleans;
        let reads = packed
            .products
            .iter()
            .flat_map(|product| [product.left, product.right, product.output])
            .chain(
                packed
                    .zeros
                    .iter()
                    .flat_map(|zero| [zero.left.column, zero.right.column]),
            )
            .chain(packed.start..packed.end());
        for column in reads {
            if layout
                .slot(column)
                .is_none_or(|(_, kind)| kind != super::layout::Kind::Bit)
            {
                return Err(PackageError::Malformed("bounded-depth packed product slot"));
            }
        }
        Ok(())
    }

    fn pack_native_booleans(&mut self) -> Result<(), PackageError> {
        let layout = self.layout()?;
        let region = self.application;
        let end = self
            .blocks()
            .map(|block| block.start() + BLOCK_LOCALS)
            .max();
        let mut in_block = vec![false; end.unwrap_or(0)];
        for block in self.blocks() {
            in_block[block.start()..block.start() + BLOCK_LOCALS].fill(true);
        }
        let eligible = |column| {
            ((WITNESS_START..WITNESS_START + region.witness_words).contains(&column)
                || (region.local_start..region.local_start + region.locals).contains(&column))
                && in_block.get(column) != Some(&true)
                && layout
                    .slot(column)
                    .is_some_and(|(_, kind)| kind == super::layout::Kind::Bit)
        };
        let start = self.sbox_start + SBOX_COLUMNS * self.block_count();
        let Some(Piece::Rows(rows)) = self.pieces.first_mut() else {
            return Err(PackageError::Malformed("bounded-depth native application rows"));
        };
        self.packed_booleans = PackedBooleans::lower(rows, start, self.width, eligible);
        self.packed_booleans.validate(rows, start, self.width)?;
        Ok(())
    }

    /// Lean `Frame.Framed`.
    fn native_pieces(
        &self,
        records: &ApplicationRecords,
        shape: &NativeShape,
    ) -> Result<(Vec<Piece>, Vec<bool>), PackageError> {
        let first = 8 + self.application.witness_words;
        let end = first + records.recipe_count();
        let mut dead = vec![false; end];
        let mut dropped = vec![false; records.row_count()];
        let mut walk = records.walk();
        let mut free = first;
        let mut blocks = Vec::with_capacity(shape.blocks.len());
        let mut constants = HashMap::new();
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
                    self.columns_of(lane, &constants)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for recipe in start - first..start - first + BLOCK_LOCALS {
                let row = dropped
                    .get_mut(walk.recipe_row(recipe)?)
                    .ok_or(PackageError::Malformed("bounded-depth native block row"))?;
                if std::mem::replace(row, true) {
                    return Err(PackageError::Malformed("bounded-depth native block row in two blocks"));
                }
            }
            dead[start..start + BLOCK_OUTPUTS].fill(true);
            if lanes.iter().all(|lane| lane.terms.is_empty()) {
                let input = std::array::from_fn(|lane| lanes[lane].constant);
                for (lane, value) in poseidon2::permute_state(input).into_iter().enumerate() {
                    constants.insert(start + BLOCK_OUTPUTS + lane, value);
                }
            } else {
                blocks.push(Piece::Block {
                    start: self.variable_column(start)?,
                    lanes,
                });
            }
            free = start + BLOCK_LOCALS;
        }
        let rows = self.application_rows(&mut walk, &dropped, &dead, &constants)?;
        Ok((std::iter::once(Piece::Rows(rows)).chain(blocks).collect(), dropped))
    }

    fn columns_of(&self, linear: &Linear, constants: &HashMap<usize, Fq>) -> Result<Linear, PackageError> {
        let mut result = Linear {
            constant: linear.constant,
            terms: Vec::with_capacity(linear.terms.len()),
        };
        for &(variable, coefficient) in &linear.terms {
            self.add_application_term(&mut result, variable, coefficient, constants)?;
        }
        Ok(result)
    }

    fn add_application_term(
        &self,
        side: &mut Linear,
        variable: usize,
        coefficient: Fq,
        constants: &HashMap<usize, Fq>,
    ) -> Result<(), PackageError> {
        if let Some(&value) = constants.get(&variable) {
            side.constant += coefficient * value;
        } else {
            side.terms
                .push((self.variable_column(variable)?, coefficient));
        }
        Ok(())
    }

    fn application_rows(
        &self,
        walk: &mut ApplicationWalk<'_>,
        dropped: &[bool],
        dead: &[bool],
        constants: &HashMap<usize, Fq>,
    ) -> Result<Vec<Row>, PackageError> {
        let mut rows = Vec::with_capacity(dropped.iter().filter(|&&dropped| !dropped).count());
        let mut terms = Vec::new();
        for row in (0..dropped.len()).filter(|&row| !dropped[row]) {
            let header = walk.row(row, &mut terms)?;
            let mut sides = [0, 1, 2].map(|form| Linear {
                constant: Fq::from_u64(header.constants[form]),
                terms: Vec::with_capacity(header.term_counts[form]),
            });
            for term in &terms {
                if dead.get(term.variable) == Some(&true) {
                    return Err(PackageError::Malformed(
                        "bounded-depth application row reads a dead local",
                    ));
                }
                self.add_application_term(
                    &mut sides[term.form.index()],
                    term.variable,
                    Fq::from_u64(term.coefficient),
                    constants,
                )?;
            }
            let [a, b, c] = sides;
            rows.push(Row { a, b, c });
        }
        Ok(rows)
    }

    /// `SPEC.md` section 7.9, step 5.
    fn slot_facts(&self) -> Result<(Vec<usize>, Vec<usize>), PackageError> {
        let region = self.application;
        let end = self
            .blocks()
            .take(region.pieces - 1)
            .map(|block| block.first_output())
            .max();
        let mut dead = vec![false; end.unwrap_or(0)];
        for block in self.blocks().take(region.pieces - 1) {
            dead[block.start()..block.first_output()].fill(true);
        }
        let live = |column: &usize| dead.get(*column) != Some(&true);
        let mut bits = self.bit_columns();
        bits.extend(WITNESS_START..WITNESS_START + region.witness_words);
        bits.extend((region.local_start..region.local_start + region.locals).filter(live));
        bits.extend(self.packed_booleans.start..self.packed_booleans.end());
        let mut eligible = vec![false; self.source_end()];
        for column in (WITNESS_START..WITNESS_START + region.witness_words)
            .chain(region.local_start..region.local_start + region.locals)
            .filter(live)
        {
            eligible[column] = true;
        }
        let seeds = self
            .packed_booleans
            .sources
            .iter()
            .copied()
            .chain(self.packed_booleans.start..self.packed_booleans.end())
            .collect::<Vec<_>>();
        let mut inferred = match self.pieces.first() {
            Some(Piece::Rows(rows)) => {
                super::boolean::inferred_from_rows(self.packed_booleans.original_rows(rows)?, &eligible, &seeds)
            }
            _ => Vec::new(),
        };
        inferred.extend(seeds);
        Ok((bits, inferred))
    }
}

/// Lean `Lifecycle.BoundedDepth.ApplicationIdentity.Application.words`.
fn record_words(
    identity: [u64; 4],
    records: &ApplicationRecords,
    witness_words: usize,
    shape: &NativeShape,
    update: &mut impl FnMut(u64),
) -> Result<(), PackageError> {
    IDENTITY_DOMAIN
        .iter()
        .for_each(|&byte| update(u64::from(byte)));
    identity.into_iter().for_each(&mut *update);
    update(witness_words as u64);
    update(records.row_count() as u64);
    let mut walk = records.walk();
    let mut terms = Vec::new();
    for row in 0..records.row_count() {
        let header = walk.row(row, &mut terms)?;
        for form in 0..3 {
            update(header.constants[form]);
            update(header.term_counts[form] as u64);
        }
        for term in &terms {
            update(term.variable as u64);
            update(term.coefficient);
        }
    }
    update(records.recipe_count() as u64);
    for recipe in 0..records.recipe_count() {
        update(walk.recipe_row(recipe)? as u64);
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
    Ok(())
}

/// Lean `Lifecycle.BoundedDepth.ApplicationIdentity.Layout.words`.
fn layout_words(package: &StepPackage, update: &mut impl FnMut(u64)) -> Result<(), PackageError> {
    let Some(Piece::Rows(rows)) = package.pieces.first() else {
        return Err(PackageError::Malformed("bounded-depth native application rows"));
    };
    package.guard_fusion.bind(rows, update);
    package.word_packing.bind(rows, update);
    update(package.packed_booleans.start as u64);
    update(package.packed_booleans.sources.len() as u64);
    for &column in &package.packed_booleans.sources {
        update(column as u64);
    }
    update(package.packed_booleans.removed.len() as u64);
    for &(row, _) in &package.packed_booleans.removed {
        update(row as u64);
    }
    update(package.packed_booleans.products.len() as u64);
    for product in &package.packed_booleans.products {
        for column in [product.row, product.left, product.right, product.output] {
            update(column as u64);
        }
    }
    update(package.packed_booleans.zeros.len() as u64);
    for zero in &package.packed_booleans.zeros {
        for word in [
            zero.row,
            zero.left.column,
            usize::from(zero.left.complement),
            zero.right.column,
            usize::from(zero.right.complement),
        ] {
            update(word as u64);
        }
    }
    update(package.slots.len() as u64);
    for &(start, count, kind) in &package.slots {
        update(start as u64);
        update(count as u64);
        update(u64::from(kind));
    }
    Ok(())
}

impl BoundedDepthPackages {
    fn substitute(
        &self,
        records: &Arc<ApplicationRecords>,
        witness_words: usize,
        shape: &NativeShape,
        update: &mut (impl FnMut(u64) + Send),
    ) -> Result<[StepPackage; 2], PackageError> {
        let (words, (light, refresh)) = rayon::join(
            || record_words(self.identity, records, witness_words, shape, update),
            || {
                rayon::join(
                    || self.light.with_application(records, witness_words, shape),
                    || self.refresh.with_application(records, witness_words, shape),
                )
            },
        );
        words?;
        let packages = [light?, refresh?];
        for package in &packages {
            layout_words(package, update)?;
        }
        Ok(packages)
    }

    pub fn application_identity_words(
        &self,
        records: &Arc<ApplicationRecords>,
        witness_words: usize,
        shape: &NativeShape,
    ) -> Result<Vec<u64>, PackageError> {
        let mut words = Vec::new();
        self.substitute(records, witness_words, shape, &mut |word| words.push(word))?;
        Ok(words)
    }

    /// Lean `Lifecycle.BoundedDepth.ApplicationIdentity.Application.Canonical`, `Export.Generic.NativePower`.
    pub fn with_application(
        &self,
        records: Arc<ApplicationRecords>,
        witness_words: usize,
        shape: &NativeShape,
    ) -> Result<Self, PackageError> {
        let mut input = poseidon2::Poseidon2Hasher::default();
        let mut reduced = None;
        let [light, refresh] = self.substitute(&records, witness_words, shape, &mut |word| {
            if word >= Fq::ORDER_U64 {
                reduced.get_or_insert(word);
            }
            input.update(&[Fq::from_u64(word)]);
        })?;
        if let Some(value) = reduced {
            return Err(PackageError::NonCanonicalField {
                location: "application identity",
                value,
            });
        }
        Ok(Self {
            light,
            refresh,
            identity: input.finalize().map(|word| word.as_canonical_u64()),
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bounded_depth_application.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/application_identity_lean.rs"]
mod identity_lean;
