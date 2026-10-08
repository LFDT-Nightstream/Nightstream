//! The two-level scatter against a direct per-run scatter (M1's method), and
//! the six matrix fraction trees through the batched GKR, with tampering.

use std::ops::{ControlFlow, Range};

use neo_ccs::GeometricRowRun;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape};
use neo_reductions::PiCcsError;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing as _;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing};

use super::word;
use crate::field::{eq_table, project, Ext, Gl};
use crate::gkr::{prove, verify};
use crate::hash::challenger;
use crate::matrix::{EarlyChallenges, Tables};
use crate::sumcheck::evaluate;

const BLOCKS: usize = 6;
const ROWS: usize = 20;
const MATRICES: usize = 4;
const POINT: usize = 10;

/// Explicit runs: `(row, matrix, start, len, initial, ratio)`.
struct Runs(Vec<(usize, usize, usize, usize, u64, u64)>);

impl MatrixRows for Runs {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: ROWS,
            columns: BLOCKS * D,
            matrices: MATRICES,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        for row in rows {
            for matrix in 0..MATRICES {
                for &(_, _, start, len, initial, ratio) in self.0.iter().filter(|run| run.0 == row && run.1 == matrix) {
                    let run = GeometricRowRun::new(row, start, len, F::from_u64(initial), F::from_u64(ratio));
                    sink.push_run(row, matrix, run)?;
                }
                if let ControlFlow::Break(()) = sink.finish_matrix_row(row, matrix)? {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

/// Both classes, runs that spill into the next block, a run that ends at the
/// last column, and slots shared by several runs.
fn runs() -> Runs {
    let mut runs = Vec::new();
    for k in 0..60u64 {
        let row = (word(1, k) % ROWS as u64) as usize;
        let matrix = (word(2, k) % MATRICES as u64) as usize;
        let initial = 1 + word(3, k) % 1000;
        if k % 3 == 0 {
            let start = (word(4, k) % (BLOCKS * D - 41) as u64) as usize;
            runs.push((row, matrix, start, 41, initial, 3));
        } else {
            let start = (word(5, k) % (BLOCKS * D) as u64) as usize;
            runs.push((row, matrix, start, 1, initial, 1 + word(6, k) % 5));
        }
    }
    // Spill from lane 30 into the next block; end at the last column; share slots.
    runs.push((3, 1, D + 30, 41, 7, 3));
    runs.push((9, 2, D + 30, 41, 11, 3));
    runs.push((4, 0, BLOCKS * D - 41, 41, 5, 3));
    runs.push((5, 3, 2 * D, 1, 13, 1));
    runs.push((6, 3, 2 * D, 1, 17, 2));
    Runs(runs)
}

fn ext(seed: u64, index: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed, 4 * index + i as u64)))
}

fn point() -> Vec<K> {
    (0..POINT as u64)
        .map(|t| K::from_coeffs([F::from_u64(word(7, t)), F::from_u64(word(8, t))]))
        .collect()
}

fn chi_at(r: &[K], index: usize) -> K {
    r.iter()
        .enumerate()
        .map(|(t, &value)| if (index >> t) & 1 == 1 { value } else { K::ONE - value })
        .product()
}

fn eval_a() -> Vec<[Ext; 2]> {
    (0..MATRICES as u64)
        .map(|j| [ext(9, 2 * j), ext(9, 2 * j + 1)])
        .collect()
}

/// M1's scatter: every run adds `π_j(χ_r(row))·m·ratio^t` at `start + t`.
fn direct_weights(runs: &Runs, r: &[K], eval_a: &[[Ext; 2]]) -> Vec<[Ext; D]> {
    let mut weights = vec![[Ext::ZERO; D]; BLOCKS];
    for &(row, matrix, start, len, initial, ratio) in &runs.0 {
        let mut weight = project(chi_at(r, row), eval_a[matrix]) * Gl::from_u64(initial);
        for column in start..start + len {
            weights[column / D][column % D] += weight;
            weight *= Gl::from_u64(if len == 1 { 1 } else { ratio });
        }
    }
    weights
}

#[test]
fn two_level_scatter_matches_the_direct_scatter() {
    let runs = runs();
    let tables = Tables::from_rows(&runs).unwrap();
    let structure = tables.structure();
    assert_eq!(structure.classes.len(), 2);
    let r = point();
    let eval_a = eval_a();
    let early = tables.early(&r);
    let ubar = tables.slot_weights(&early, &eval_a);
    let direct = direct_weights(&runs, &r, &eval_a);
    assert_eq!(tables.column_weights(&ubar, BLOCKS), direct);

    let tau: [Ext; D] = std::array::from_fn(|l| ext(10, l as u64));
    let lanes = structure.lane_tables(&tau);
    let omega = tables.block_weights(&ubar, &lanes, 3);
    for (block, &value) in omega.iter().enumerate() {
        let expected: Ext = direct.get(block).map_or(Ext::ZERO, |weights| {
            weights.iter().zip(&tau).map(|(&w, &t)| w * t).sum()
        });
        assert_eq!(value, expected, "block {block}");
    }
    // Control: without the spill tables the blocks after a spill are wrong.
    let no_spill: Vec<[[Ext; D]; 2]> = lanes
        .iter()
        .map(|&[head, _]| [head, [Ext::ZERO; D]])
        .collect();
    assert_ne!(tables.block_weights(&ubar, &no_spill, 3), omega);
}

#[test]
fn runs_longer_than_a_block_are_refused() {
    let mut runs = runs();
    runs.0.push((0, 0, 0, D + 1, 1, 2));
    assert!(Tables::from_rows(&runs).is_err());
}

fn challenges() -> EarlyChallenges {
    EarlyChallenges {
        lookup: [ext(20, 0), ext(20, 1)],
        scatter: [ext(20, 2), ext(20, 3)],
    }
}

#[test]
fn matrix_trees_verify_and_leave_true_openings() {
    let runs = runs();
    let tables = Tables::from_rows(&runs).unwrap();
    let structure = tables.structure();
    let r = point();
    let early = tables.early(&r);
    let lift = |table: &[Gl]| {
        table
            .iter()
            .map(|&value| Ext::from(value))
            .collect::<Vec<Ext>>()
    };

    // Early trees.
    let trees = tables.early_trees(&early, &r, challenges());
    let (proof, prover) = prove(trees, &mut challenger(Poseidon2Transcript::new_v1_1()));
    let values = tables.early_values(&early, &prover);
    let claims = verify(
        &proof,
        &structure.early_shapes(),
        &mut challenger(Poseidon2Transcript::new_v1_1()),
    )
    .unwrap();
    let openings = structure
        .check_early(&r, challenges(), &claims, &values)
        .unwrap();
    assert_eq!(openings.mult, evaluate(&lift(&early.mult), &openings.row_point));
    for (k, &value) in openings.setup.iter().enumerate() {
        let column = lift(&tables.setup_column(k, structure.run_variables));
        assert_eq!(value, evaluate(&column, &openings.run_point), "setup column {k}");
    }
    let slot_point = &openings.run_point[..structure.slot_variables];
    assert_eq!(values.block, evaluate(&lift(&tables.blocks()), slot_point));
    let embedded = lift(&tables.setup_column(3, structure.run_variables));
    let tail: Ext = openings.run_point[structure.slot_variables..]
        .iter()
        .map(|&x| Ext::ONE - x)
        .product();
    assert_eq!(evaluate(&embedded, &openings.run_point), values.block * tail);
    for (j, [re, im]) in early.u.iter().enumerate() {
        assert_eq!(
            values.pairs[j],
            [
                evaluate(&lift(re), &openings.pair_point),
                evaluate(&lift(im), &openings.pair_point)
            ]
        );
    }

    // Every sent value is bound.
    for index in 0..3 {
        let mut tampered = values.clone();
        tampered.runs[index] += Ext::ONE;
        assert!(
            structure
                .check_early(&r, challenges(), &claims, &tampered)
                .is_err(),
            "run value {index}"
        );
    }
    for j in 0..MATRICES {
        let mut tampered = values.clone();
        tampered.pairs[j][1] += Ext::ONE;
        assert!(
            structure
                .check_early(&r, challenges(), &claims, &tampered)
                .is_err(),
            "pair value {j}"
        );
    }

    // Late trees.
    let eval_a = eval_a();
    let ubar = tables.slot_weights(&early, &eval_a);
    let tau: [Ext; D] = std::array::from_fn(|l| ext(10, l as u64));
    let lanes = structure.lane_tables(&tau);
    let omega = tables.block_weights(&ubar, &lanes, 3);
    let beta = ext(21, 0);
    let late = |omega: &[Ext]| {
        let (proof, prover) = prove(
            tables.late_trees(&ubar, &lanes, omega, beta),
            &mut challenger(Poseidon2Transcript::new_v1_1()),
        );
        let pairs = tables.slot_values(&early, &eq_table(&prover[0].point[..structure.slot_variables]));
        let claims = verify(
            &proof,
            &structure.late_shapes(3),
            &mut challenger(Poseidon2Transcript::new_v1_1()),
        )
        .unwrap();
        (claims, pairs)
    };
    let (claims, pairs) = late(&omega);
    let openings = structure
        .check_late(beta, &eval_a, &lanes, &claims, &pairs)
        .unwrap();
    assert_eq!(openings.block, evaluate(&lift(&tables.blocks()), &openings.pair_point));
    assert_eq!(openings.omega, evaluate(&omega, &openings.block_point));
    let mut tampered = pairs.clone();
    tampered[2][0] += Ext::ONE;
    assert!(structure
        .check_late(beta, &eval_a, &lanes, &claims, &tampered)
        .is_err());
    // A wrong block weight breaks the block scatter roots.
    let mut forged = omega.clone();
    forged[1] += Ext::ONE;
    let (claims, pairs) = late(&forged);
    assert!(structure
        .check_late(beta, &eval_a, &lanes, &claims, &pairs)
        .is_err());
}

#[test]
fn wrong_early_columns_break_the_roots() {
    let runs = runs();
    let tables = Tables::from_rows(&runs).unwrap();
    let structure = tables.structure();
    let r = point();
    let run = |early: &crate::matrix::Early| {
        let (proof, prover) = prove(
            tables.early_trees(early, &r, challenges()),
            &mut challenger(Poseidon2Transcript::new_v1_1()),
        );
        let values = tables.early_values(early, &prover);
        let claims = verify(
            &proof,
            &structure.early_shapes(),
            &mut challenger(Poseidon2Transcript::new_v1_1()),
        )
        .unwrap();
        structure.check_early(&r, challenges(), &claims, &values)
    };
    assert!(run(&tables.early(&r)).is_ok());
    let mut wrong_e = tables.early(&r);
    wrong_e.e[0][5] += Gl::ONE;
    assert!(run(&wrong_e).is_err());
    let mut wrong_u = tables.early(&r);
    wrong_u.u[1][1][0] += Gl::ONE;
    assert!(run(&wrong_u).is_err());
    let mut wrong_mult = tables.early(&r);
    wrong_mult.mult[3] += Gl::ONE;
    assert!(run(&wrong_mult).is_err());
}
