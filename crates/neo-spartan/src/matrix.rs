//! The CCS matrices as runs, slots and blocks: the two-level scatter of
//! `docs/reviews/compression-akita/M2-DESIGN.md`.
//!
//! Owns: the run and slot tables built from `MatrixRows`, the structure the
//! verifier keeps, the per-proof columns and weights, the leaves of the six
//! matrix fraction trees, and the verifier's checks of their leaf claims.
//! Does not own the setup oracles, the commitments or the transcript.
//!
//! - A run `k` is one `GeometricRowRun`: row `i_k`, matrix `j(k)`, initial
//!   coefficient `m_k`, and a slot. Runs are sorted by matrix, then row.
//!   Padding runs have row, slot and coefficient zero and matrix tag zero.
//! - A slot `σ` is a distinct (class, start column); a class is a run shape
//!   `(len ≤ D, ratio)`. Slots are sorted by class, lane, then block, so the
//!   class and lane of a slot are step functions of `σ`.
//! - Early columns (before λ): `e_k = χ_r(i_k)`,
//!   `U_j(σ) = Σ_{k: j(k) = j, σ_k = σ} m_k·e_k`, and the row counts `mult`.
//! - Late weights: `Ū(σ) = Σ_j π_j(U_j(σ))` with the Eval_A projections `π_j`,
//!   and `Ω^A(b) = Σ_σ Ū(σ)·(Ĥ_0(σ)·[b_σ = b] + Ĥ_1(σ)·[b_σ + 1 = b])`, where
//!   `Ĥ_t` is the lane split of the slot's class at its lane.
//!
//! Trees (logUp; each pair has equal fraction sums):
//! - lookup: runs `(1, β - (i + γ·e_re + γ²·e_im))`, rows
//!   `(mult(i), β - (i + γ·Re χ_r(i) + γ²·Im χ_r(i)))`;
//! - scatter: runs `(m·(e_re + μ·e_im), β - (σ + 2^s·j))`, pairs
//!   `(U_j,re(σ) + μ·U_j,im(σ), β - (σ + 2^s·j))`;
//! - blocks: pairs `(Ū(σ)·Ĥ_t(σ), β - (b_σ + t))`, blocks `(Ω^A(b), β - b)`.

use std::collections::BTreeSet;
use std::ops::ControlFlow;

use neo_ccs::GeometricRowRun;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows};
use neo_reductions::PiCcsError;
use p3_field_v08::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::field::{eq_table, gl, re_im, Ext, Gl};
use crate::gkr::{Tree, TreeClaim, TreeShape};
use crate::mle::{chi, identity, lane_split, step};
use crate::Error;

/// A run shape. The ratio is canonical, and one for single-coordinate runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct Class {
    pub(crate) len: u64,
    pub(crate) ratio: u64,
}

/// Slots `first..` (up to the next segment) share one class and one lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Segment {
    pub(crate) first: u64,
    pub(crate) class: u32,
    pub(crate) lane: u32,
}

/// The matrices as the verifier knows them: sizes, the first run of each
/// matrix, the run classes and the slot segments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Structure {
    pub(crate) matrices: usize,
    pub(crate) row_variables: usize,
    pub(crate) run_variables: usize,
    pub(crate) slot_variables: usize,
    /// The first run of each matrix, then the run count.
    pub(crate) run_bounds: Vec<u64>,
    pub(crate) classes: Vec<Class>,
    pub(crate) segments: Vec<Segment>,
    pub(crate) slots: u64,
}

/// The prover's run and slot tables.
pub(crate) struct Tables {
    structure: Structure,
    /// Per run, in run order.
    row: Vec<u32>,
    slot: Vec<u32>,
    coefficient: Vec<Gl>,
    /// Per slot: the start column.
    start: Vec<u64>,
}

/// Per-proof columns that depend only on `r`.
pub(crate) struct Early {
    /// `e_k = χ_r(i_k)` per run: real, imaginary.
    pub(crate) e: [Vec<Gl>; 2],
    /// `U_j` per matrix: real, imaginary, per slot.
    pub(crate) u: Vec<[Vec<Gl>; 2]>,
    /// Runs per row; padding runs count at row zero.
    pub(crate) mult: Vec<Gl>,
}

/// The lookup challenges `(β, γ)` and the scatter challenges `(β, μ)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EarlyChallenges {
    pub(crate) lookup: [Ext; 2],
    pub(crate) scatter: [Ext; 2],
}

/// Column values the prover sends at the early leaf points.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct EarlyValues {
    /// `e_re`, `e_im` and the row column `i` at the run point.
    pub(crate) runs: [Ext; 3],
    /// `U_j` (real, imaginary) per matrix at the slot part of the pair point.
    pub(crate) pairs: Vec<[Ext; 2]>,
    /// The slot blocks at the slot part of the run point.
    pub(crate) block: Ext,
}

/// What the early leaf checks leave to openings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EarlyOpenings {
    pub(crate) run_point: Vec<Ext>,
    pub(crate) pair_point: Vec<Ext>,
    pub(crate) row_point: Vec<Ext>,
    pub(crate) mult: Ext,
    /// The setup run columns `i`, `σ`, `m` at the run point.
    pub(crate) setup: [Ext; 3],
}

/// What the late leaf checks leave to openings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LateOpenings {
    pub(crate) pair_point: Vec<Ext>,
    pub(crate) block: Ext,
    pub(crate) block_point: Vec<Ext>,
    pub(crate) omega: Ext,
}

fn bits(count: u64) -> usize {
    (count.next_power_of_two().trailing_zeros() as usize).max(1)
}

impl Structure {
    pub(crate) fn matrix_variables(&self) -> usize {
        self.matrices.next_power_of_two().trailing_zeros() as usize
    }

    /// The structure as transcript words.
    pub(crate) fn words(&self) -> Vec<Gl> {
        let sizes = [
            self.matrices,
            self.row_variables,
            self.run_variables,
            self.slot_variables,
            self.classes.len(),
            self.segments.len(),
        ]
        .map(|value| value as u64);
        let classes = self
            .classes
            .iter()
            .flat_map(|class| [class.len, class.ratio]);
        let segments = self
            .segments
            .iter()
            .flat_map(|segment| [segment.first, u64::from(segment.class), u64::from(segment.lane)]);
        sizes
            .into_iter()
            .chain([self.slots])
            .chain(self.run_bounds.iter().copied())
            .chain(classes)
            .chain(segments)
            .map(Gl::from_u64)
            .collect()
    }

    /// `[H0, H1]` of every class for the lane weights `tau`.
    pub(crate) fn lane_tables(&self, tau: &[Ext; D]) -> Vec<[[Ext; D]; 2]> {
        self.classes
            .iter()
            .map(|class| lane_split(tau, class.len as usize, Gl::new(class.ratio)))
            .collect()
    }

    /// Early trees, in order: lookup runs, lookup rows, scatter runs, scatter pairs.
    pub(crate) fn early_shapes(&self) -> [TreeShape; 4] {
        [
            TreeShape {
                depth: self.run_variables,
                factors: 0,
            },
            TreeShape {
                depth: self.row_variables,
                factors: 1,
            },
            TreeShape {
                depth: self.run_variables,
                factors: 2,
            },
            TreeShape {
                depth: self.slot_variables + self.matrix_variables(),
                factors: 1,
            },
        ]
    }

    /// Late trees, in order: block pairs, blocks.
    pub(crate) fn late_shapes(&self, block_variables: usize) -> [TreeShape; 2] {
        [
            TreeShape {
                depth: self.slot_variables + 1,
                factors: 2,
            },
            TreeShape {
                depth: block_variables,
                factors: 1,
            },
        ]
    }

    /// Check the early roots and leaf claims against `values`.
    pub(crate) fn check_early(
        &self,
        r: &[K],
        challenges: EarlyChallenges,
        claims: &[TreeClaim],
        values: &EarlyValues,
    ) -> Result<EarlyOpenings, Error> {
        let [lookup_runs, lookup_rows, scatter_runs, scatter_pairs] = claims else {
            return Err(Error::Rejected("early tree count"));
        };
        if values.pairs.len() != self.matrices {
            return Err(Error::Rejected("early values shape"));
        }
        check_pair(lookup_runs, lookup_rows, "row lookup")?;
        check_pair(scatter_runs, scatter_pairs, "slot scatter")?;
        let [beta, gamma] = challenges.lookup;
        let [e_re, e_im, row] = values.runs;
        if lookup_runs.values[0] != beta - (row + gamma * e_re + gamma * gamma * e_im) {
            return Err(Error::Rejected("row lookup run leaf"));
        }
        let row_point = &lookup_rows.point;
        let chi = chi(r, row_point);
        let [mult, denominator] = lookup_rows.values[..] else {
            return Err(Error::Rejected("row lookup row leaf"));
        };
        if denominator != beta - (identity(row_point) + gamma * chi.re + gamma * gamma * chi.im) {
            return Err(Error::Rejected("row lookup row leaf"));
        }
        let [beta, mu] = challenges.scatter;
        let [coefficient, e, denominator] = scatter_runs.values[..] else {
            return Err(Error::Rejected("slot scatter run leaf"));
        };
        if e != e_re + mu * e_im {
            return Err(Error::Rejected("slot scatter run leaf"));
        }
        let run_point = &scatter_runs.point;
        let slot = beta - denominator - self.matrix_step(run_point) * self.slot_scale();
        let pair_point = &scatter_pairs.point;
        let (slot_part, matrix_part) = pair_point.split_at(self.slot_variables);
        let eq = eq_table(matrix_part);
        let combined: Ext = values
            .pairs
            .iter()
            .zip(&eq)
            .map(|(&[re, im], &weight)| weight * (re + mu * im))
            .sum();
        if scatter_pairs.values[..] != [combined, beta - identity(pair_point)] {
            return Err(Error::Rejected("slot scatter pair leaf"));
        }
        Ok(EarlyOpenings {
            run_point: run_point.clone(),
            pair_point: slot_part.to_vec(),
            row_point: row_point.clone(),
            mult,
            setup: [row, slot, coefficient],
        })
    }

    /// Check the late roots and leaf claims against the `U_j` values sent at
    /// the slot part of the pair point.
    pub(crate) fn check_late(
        &self,
        beta: Ext,
        eval_a: &[[Ext; 2]],
        lanes: &[[[Ext; D]; 2]],
        claims: &[TreeClaim],
        pairs: &[[Ext; 2]],
    ) -> Result<LateOpenings, Error> {
        let [block_pairs, blocks] = claims else {
            return Err(Error::Rejected("late tree count"));
        };
        if pairs.len() != self.matrices || eval_a.len() != self.matrices {
            return Err(Error::Rejected("late values shape"));
        }
        check_pair(block_pairs, blocks, "block scatter")?;
        let (slot_part, t) = block_pairs.point.split_at(self.slot_variables);
        let t = t[0];
        let ubar: Ext = pairs
            .iter()
            .zip(eval_a)
            .map(|(&[re, im], &[w_re, w_im])| w_re * re + w_im * im)
            .sum();
        let lane = (Ext::ONE - t) * self.slot_step(slot_part, lanes, 0) + t * self.slot_step(slot_part, lanes, 1);
        let [weight, split, denominator] = block_pairs.values[..] else {
            return Err(Error::Rejected("block scatter pair leaf"));
        };
        if weight != ubar || split != lane {
            return Err(Error::Rejected("block scatter pair leaf"));
        }
        let [omega, block_denominator] = blocks.values[..] else {
            return Err(Error::Rejected("block scatter block leaf"));
        };
        if block_denominator != beta - identity(&blocks.point) {
            return Err(Error::Rejected("block scatter block leaf"));
        }
        Ok(LateOpenings {
            pair_point: slot_part.to_vec(),
            block: beta - denominator - t,
            block_point: blocks.point.clone(),
            omega,
        })
    }

    fn slot_scale(&self) -> Gl {
        Gl::from_u64(1 << self.slot_variables)
    }

    /// `j~` over runs: matrix `j` on its run range, zero on padding runs.
    fn matrix_step(&self, point: &[Ext]) -> Ext {
        let values: Vec<Ext> = (0..self.matrices)
            .map(|j| Ext::from(Gl::from_usize(j)))
            .collect();
        step(
            point,
            &self.run_bounds[..self.matrices],
            &values,
            self.run_bounds[self.matrices],
        )
    }

    /// `Ĥ_t~` over slots: the class's lane table at the slot's lane.
    fn slot_step(&self, point: &[Ext], lanes: &[[[Ext; D]; 2]], t: usize) -> Ext {
        let starts: Vec<u64> = self.segments.iter().map(|segment| segment.first).collect();
        let values: Vec<Ext> = self
            .segments
            .iter()
            .map(|segment| lanes[segment.class as usize][t][segment.lane as usize])
            .collect();
        step(point, &starts, &values, self.slots)
    }
}

/// `P_x·Q_y = P_y·Q_x` with both denominators nonzero.
fn check_pair(x: &TreeClaim, y: &TreeClaim, name: &'static str) -> Result<(), Error> {
    let ([px, qx], [py, qy]) = (x.root, y.root);
    if qx == Ext::ZERO || qy == Ext::ZERO || px * qy != py * qx {
        return Err(Error::Rejected(name));
    }
    Ok(())
}

/// `χ_r(i)` for every `i < 2^variables`; the higher coordinates of `r` see bit zero.
fn chi_rows(r: &[K], variables: usize) -> Vec<K> {
    let tail: K = r[variables..]
        .iter()
        .map(|&value| <K as p3_field::PrimeCharacteristicRing>::ONE - value)
        .product();
    neo_ccs::utils::tensor_point(&r[..variables])
        .into_iter()
        .map(|value| value * tail)
        .collect()
}

/// `Σ_x eq[x]·table[x]` for a base-field table.
fn evaluate(eq: &[Ext], table: &[Gl]) -> Ext {
    eq.par_iter()
        .zip(table)
        .map(|(&weight, &value)| weight * value)
        .sum()
}

struct Run {
    row: u32,
    start: u64,
    class: Class,
    initial: Gl,
}

/// Collects runs per matrix in visit order.
struct Collect {
    runs: Vec<Vec<Run>>,
    longest: usize,
    end: usize,
}

impl MatrixRowSink for Collect {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        let len = run.len();
        self.longest = self.longest.max(len);
        self.end = self.end.max(run.column_start() + len);
        let ratio = if len == 1 { Gl::ONE } else { gl(*run.ratio()) };
        self.runs[matrix].push(Run {
            row: row as u32,
            start: run.column_start() as u64,
            class: Class {
                len: len as u64,
                ratio: ratio.as_canonical_u64(),
            },
            initial: gl(*run.initial()),
        });
        Ok(())
    }

    fn finish_matrix_row(&mut self, _row: usize, _matrix: usize) -> Result<ControlFlow<()>, PiCcsError> {
        Ok(ControlFlow::Continue(()))
    }
}

impl Tables {
    pub(crate) fn from_rows(rows: &dyn MatrixRows) -> Result<Self, Error> {
        let shape = rows.shape();
        if shape.rows > u32::MAX as usize || !shape.columns.is_multiple_of(D) || shape.matrices == 0 {
            return Err(Error::Shape("matrix source"));
        }
        let mut collect = Collect {
            runs: (0..shape.matrices).map(|_| Vec::new()).collect(),
            longest: 0,
            end: 0,
        };
        rows.visit_rows(0..shape.rows, &mut collect)
            .map_err(|_| Error::Shape("matrix rows"))?;
        if collect.longest > D || collect.end > shape.columns {
            return Err(Error::Shape(
                "a matrix run is longer than one block or leaves the columns",
            ));
        }
        let mut run_bounds = vec![0u64];
        for runs in &collect.runs {
            run_bounds.push(run_bounds.last().expect("nonempty") + runs.len() as u64);
        }
        let runs: Vec<Run> = collect.runs.into_iter().flatten().collect();
        if runs.len() > u32::MAX as usize {
            return Err(Error::Shape("matrix run count"));
        }

        let classes: Vec<Class> = runs
            .iter()
            .map(|run| run.class)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        // Slot key: class, lane, then block, high bits first.
        let key = |run: &Run| {
            let class = classes.binary_search(&run.class).expect("collected") as u64;
            (class << 40) | ((run.start % D as u64) << 32) | (run.start / D as u64)
        };
        let mut keys: Vec<u64> = runs.par_iter().map(key).collect();
        keys.par_sort_unstable();
        keys.dedup();
        let slot: Vec<u32> = runs
            .par_iter()
            .map(|run| keys.binary_search(&key(run)).expect("collected") as u32)
            .collect();
        let start: Vec<u64> = keys
            .iter()
            .map(|key| (key & 0xffff_ffff) * D as u64 + ((key >> 32) & 0xff))
            .collect();
        let mut segments: Vec<Segment> = Vec::new();
        for (index, key) in keys.iter().enumerate() {
            let (class, lane) = ((key >> 40) as u32, ((key >> 32) & 0xff) as u32);
            if segments
                .last()
                .is_none_or(|last| (last.class, last.lane) != (class, lane))
            {
                segments.push(Segment {
                    first: index as u64,
                    class,
                    lane,
                });
            }
        }
        let structure = Structure {
            matrices: shape.matrices,
            row_variables: bits(shape.rows as u64),
            run_variables: bits(runs.len() as u64),
            slot_variables: bits(keys.len() as u64),
            run_bounds,
            classes,
            segments,
            slots: keys.len() as u64,
        };
        Ok(Self {
            structure,
            row: runs.iter().map(|run| run.row).collect(),
            slot,
            coefficient: runs.iter().map(|run| run.initial).collect(),
            start,
        })
    }

    pub(crate) fn structure(&self) -> &Structure {
        &self.structure
    }

    /// Setup oracle `k` of the run columns, zero-padded to `2^variables`:
    /// rows, slots, coefficients, then the slot blocks.
    pub(crate) fn setup_column(&self, k: usize, variables: usize) -> Vec<Gl> {
        let mut column: Vec<Gl> = match k {
            0 => self.row.iter().map(|&row| Gl::from_u32(row)).collect(),
            1 => self.slot.iter().map(|&slot| Gl::from_u32(slot)).collect(),
            2 => self.coefficient.clone(),
            3 => self
                .start
                .iter()
                .map(|&start| Gl::from_u64(start / D as u64))
                .collect(),
            _ => unreachable!("four run columns"),
        };
        column.resize(1 << variables, Gl::ZERO);
        column
    }

    pub(crate) fn early(&self, r: &[K]) -> Early {
        let structure = &self.structure;
        let rows = chi_rows(r, structure.row_variables);
        let row = |k: usize| self.row.get(k).map_or(0, |&row| row as usize);
        let (re, im) = (0..1usize << structure.run_variables)
            .into_par_iter()
            .map(|k| {
                let [re, im] = re_im(rows[row(k)]);
                (re, im)
            })
            .unzip();
        let slots = 1usize << structure.slot_variables;
        let u = (0..structure.matrices)
            .into_par_iter()
            .map(|j| {
                let mut column = [vec![Gl::ZERO; slots], vec![Gl::ZERO; slots]];
                for k in structure.run_bounds[j] as usize..structure.run_bounds[j + 1] as usize {
                    let [re, im] = re_im(rows[self.row[k] as usize]);
                    let slot = self.slot[k] as usize;
                    column[0][slot] += self.coefficient[k] * re;
                    column[1][slot] += self.coefficient[k] * im;
                }
                column
            })
            .collect();
        let mut mult = vec![0u64; 1 << structure.row_variables];
        for &row in &self.row {
            mult[row as usize] += 1;
        }
        mult[0] += (1u64 << structure.run_variables) - self.row.len() as u64;
        Early {
            e: [re, im],
            u,
            mult: mult.into_iter().map(Gl::from_u64).collect(),
        }
    }

    /// Early trees, in the order of `Structure::early_shapes`.
    pub(crate) fn early_trees(&self, early: &Early, r: &[K], challenges: EarlyChallenges) -> Vec<Tree> {
        let structure = &self.structure;
        let runs = 1usize << structure.run_variables;
        let row = |k: usize| self.row.get(k).map_or(0, |&row| row);
        let [beta, gamma] = challenges.lookup;
        let lookup_runs = Tree {
            factors: Vec::new(),
            denominator: (0..runs)
                .into_par_iter()
                .map(|k| beta - (gamma * early.e[0][k] + gamma * gamma * early.e[1][k] + Gl::from_u32(row(k))))
                .collect(),
        };
        let rows = chi_rows(r, structure.row_variables);
        let lookup_rows = Tree {
            factors: vec![early.mult.iter().map(|&count| Ext::from(count)).collect()],
            denominator: rows
                .par_iter()
                .enumerate()
                .map(|(i, &value)| {
                    let [re, im] = re_im(value);
                    beta - (gamma * re + gamma * gamma * im + Gl::from_usize(i))
                })
                .collect(),
        };
        let [beta, mu] = challenges.scatter;
        let scale = structure.slot_scale();
        let mut tag = vec![Gl::ZERO; runs];
        for j in 0..structure.matrices {
            let range = structure.run_bounds[j] as usize..structure.run_bounds[j + 1] as usize;
            tag[range.clone()]
                .par_iter_mut()
                .zip(&self.slot[range])
                .for_each(|(tag, &slot)| *tag = Gl::from_u32(slot) + scale * Gl::from_usize(j));
        }
        let scatter_runs = Tree {
            factors: vec![
                (0..runs)
                    .into_par_iter()
                    .map(|k| Ext::from(self.coefficient.get(k).copied().unwrap_or(Gl::ZERO)))
                    .collect(),
                (0..runs)
                    .into_par_iter()
                    .map(|k| mu * early.e[1][k] + early.e[0][k])
                    .collect(),
            ],
            denominator: tag.into_par_iter().map(|tag| beta - tag).collect(),
        };
        let slots = 1usize << structure.slot_variables;
        let pairs = slots << structure.matrix_variables();
        let scatter_pairs = Tree {
            factors: vec![(0..pairs)
                .into_par_iter()
                .map(|x| match early.u.get(x / slots) {
                    Some([re, im]) => mu * im[x % slots] + re[x % slots],
                    None => Ext::ZERO,
                })
                .collect()],
            denominator: (0..pairs)
                .into_par_iter()
                .map(|x| beta - Gl::from_usize(x))
                .collect(),
        };
        vec![lookup_runs, lookup_rows, scatter_runs, scatter_pairs]
    }

    /// The column values at the early leaf points of `claims`.
    pub(crate) fn early_values(&self, early: &Early, claims: &[TreeClaim]) -> EarlyValues {
        let structure = &self.structure;
        let run_eq = eq_table(&claims[0].point);
        let runs = [
            evaluate(&run_eq, &early.e[0]),
            evaluate(&run_eq, &early.e[1]),
            evaluate(&run_eq, &self.setup_column(0, structure.run_variables)),
        ];
        drop(run_eq);
        let pair_eq = eq_table(&claims[3].point[..structure.slot_variables]);
        let block_eq = eq_table(&claims[0].point[..structure.slot_variables]);
        EarlyValues {
            runs,
            pairs: self.slot_values(early, &pair_eq),
            block: evaluate(&block_eq, &self.blocks()),
        }
    }

    /// `U_j` (real, imaginary) at the point of `eq`.
    pub(crate) fn slot_values(&self, early: &Early, eq: &[Ext]) -> Vec<[Ext; 2]> {
        early
            .u
            .iter()
            .map(|[re, im]| [evaluate(eq, re), evaluate(eq, im)])
            .collect()
    }

    /// The slot blocks, zero-padded to `2^slot_variables`: the prover's copy
    /// of setup column 3.
    pub(crate) fn blocks(&self) -> Vec<Gl> {
        self.setup_column(3, self.structure.slot_variables)
    }

    /// `Ū(σ) = Σ_j (w_re,j·Re U_j(σ) + w_im,j·Im U_j(σ))`.
    pub(crate) fn slot_weights(&self, early: &Early, eval_a: &[[Ext; 2]]) -> Vec<Ext> {
        (0..1usize << self.structure.slot_variables)
            .into_par_iter()
            .map(|slot| {
                early
                    .u
                    .iter()
                    .zip(eval_a)
                    .map(|([re, im], &[w_re, w_im])| w_re * re[slot] + w_im * im[slot])
                    .sum()
            })
            .collect()
    }

    /// The Eval_A weight of every coordinate `D·b + l`, for `blocks` blocks:
    /// `Σ` over slots that cover it of `Ū(σ)·ratio^t`.
    pub(crate) fn column_weights(&self, ubar: &[Ext], blocks: usize) -> Vec<[Ext; D]> {
        let mut weights = vec![[Ext::ZERO; D]; blocks];
        for (range, segment) in self.slot_ranges() {
            let class = self.structure.classes[segment.class as usize];
            let ratio = Gl::new(class.ratio);
            for slot in range {
                let mut weight = ubar[slot];
                for column in self.start[slot]..self.start[slot] + class.len {
                    weights[column as usize / D][column as usize % D] += weight;
                    weight *= ratio;
                }
            }
        }
        weights
    }

    /// `Ω^A(b)` for `b < 2^block_variables`.
    pub(crate) fn block_weights(&self, ubar: &[Ext], lanes: &[[[Ext; D]; 2]], block_variables: usize) -> Vec<Ext> {
        let mut omega = vec![Ext::ZERO; 1 << block_variables];
        for (range, segment) in self.slot_ranges() {
            let [head, tail] = lanes[segment.class as usize].map(|table| table[segment.lane as usize]);
            for slot in range {
                let block = (self.start[slot] / D as u64) as usize;
                omega[block] += ubar[slot] * head;
                if tail != Ext::ZERO {
                    omega[block + 1] += ubar[slot] * tail;
                }
            }
        }
        omega
    }

    /// Late trees, in the order of `Structure::late_shapes`.
    pub(crate) fn late_trees(&self, ubar: &[Ext], lanes: &[[[Ext; D]; 2]], omega: &[Ext], beta: Ext) -> Vec<Tree> {
        let slots = 1usize << self.structure.slot_variables;
        let mut split = vec![Ext::ZERO; 2 * slots];
        for (range, segment) in self.slot_ranges() {
            let [head, tail] = lanes[segment.class as usize].map(|table| table[segment.lane as usize]);
            for slot in range {
                split[slot] = head;
                split[slot + slots] = tail;
            }
        }
        let blocks = self.blocks();
        let block_pairs = Tree {
            factors: vec![(0..2 * slots).map(|x| ubar[x % slots]).collect(), split],
            denominator: (0..2 * slots)
                .into_par_iter()
                .map(|x| beta - (blocks[x % slots] + Gl::from_usize(x / slots)))
                .collect(),
        };
        let block_tree = Tree {
            factors: vec![omega.to_vec()],
            denominator: (0..omega.len()).map(|b| beta - Gl::from_usize(b)).collect(),
        };
        vec![block_pairs, block_tree]
    }

    /// The slots of each segment.
    fn slot_ranges(&self) -> impl Iterator<Item = (std::ops::Range<usize>, Segment)> + '_ {
        let segments = &self.structure.segments;
        segments.iter().enumerate().map(move |(s, &segment)| {
            let end = segments
                .get(s + 1)
                .map_or(self.structure.slots, |next| next.first);
            (segment.first as usize..end as usize, segment)
        })
    }
}
