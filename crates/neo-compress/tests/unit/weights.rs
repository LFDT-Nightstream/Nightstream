//! `matrix_part` equals its definition on random rows and on the rows of the
//! light and refresh circuits:
//! `Σ_claims c · Σ_runs ω^(κ+1+j) · eq(r, row) · Σ_k initial · ratio^k · Φ(start + k)`,
//! where each entry of matrix `j` is a geometric run of columns
//! (`MatrixRows::visit_rows`) and `Φ(64b + h) = eq(β, b) · conjugate(u)[h]`.

use std::ops::{ControlFlow, Range};

use neo_ajtai::Commitment;
use neo_ccs::{GeometricRowRun, Mat};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{
    FactoredEntries, FactoredRows, FactoredSlots, MatrixRowSink, MatrixRows, MatrixShape,
};
use neo_reductions::PiCcsError;
use nightstream_fprime::{
    load_bounded_depth_packages, PlanRows, SlotSpans, StepPackage, BOUNDED_DEPTH_PACKAGES_IDENTITY, PUBLIC_WORDS,
};
use p3_field::PrimeCharacteristicRing;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{matrix_part, Challenges, Uses, KAPPA};
use crate::support::{self, claim_of, conjugate, equality, random_extension, random_witness, Rows, MATRICES};
use crate::{key_blocks, layout, CeClaim, Norm, Statement};

/// The runs of one claim's rows against `eq(r, ·)`, the powers of `ω` and `Φ`.
struct Direct<'a> {
    rows: &'a [K],
    omega: &'a [K],
    blocks: &'a [K],
    form: &'a [K; D],
    sum: K,
}

impl MatrixRowSink for Direct<'_> {
    fn push_run(&mut self, row: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        let weight = self.rows[row] * self.omega[matrix];
        let mut term = *run.initial();
        for column in run.column_start()..run.column_start() + run.len() {
            self.sum += weight * (self.blocks[column / D] * self.form[column % D]).scale_base(term);
            term *= *run.ratio();
        }
        Ok(())
    }

    fn finish_matrix_row(&mut self, _: usize, _: usize) -> Result<ControlFlow<()>, PiCcsError> {
        Ok(ControlFlow::Continue(()))
    }
}

fn direct(statements: &[Statement<'_>], challenges: &Challenges, uses: &Uses<'_>) -> K {
    let form = conjugate(&uses.form);
    statements
        .iter()
        .zip(&uses.claims)
        .map(|(statement, claim)| {
            let rows = statement
                .rows
                .factored()
                .expect("rows")
                .expect("factored rows");
            let shape = rows.shape();
            let row_weights: Vec<K> = (0..shape.rows)
                .into_par_iter()
                .map(|row| equality(&statement.claim.r, row))
                .collect();
            let blocks: Vec<K> = (0..claim.blocks)
                .into_par_iter()
                .map(|block| equality(claim.point, block))
                .collect();
            let omega: Vec<K> = (0..shape.matrices)
                .map(|matrix| challenges.omega.exp_u64((KAPPA + 1 + matrix) as u64))
                .collect();
            let part = |chunk: Range<usize>| {
                let mut sink = Direct {
                    rows: &row_weights,
                    omega: &omega,
                    blocks: &blocks,
                    form: &form,
                    sum: K::ZERO,
                };
                rows.visit_rows(chunk, &mut sink).expect("runs");
                sink.sum
            };
            let chunk = 1 << 10;
            let sum: K = (0..shape.rows.div_ceil(chunk))
                .into_par_iter()
                .map(|index| part(index * chunk..((index + 1) * chunk).min(shape.rows)))
                .sum();
            claim.coefficient * sum
        })
        .sum()
}

fn random_challenges(rng: &mut ChaCha8Rng, tables: usize, span_vars: usize) -> Challenges {
    let mut extensions = |count: usize| -> Vec<K> { (0..count).map(|_| random_extension(rng)).collect() };
    let [omega, delta, gamma, mu] = extensions(4).try_into().expect("four challenges");
    Challenges {
        taus: std::array::from_fn(|index| F::from_u64(index as u64 + 7)),
        omega,
        delta,
        gamma,
        mu,
        psi: extensions(4).try_into().expect("one per lookup"),
        xi: extensions(tables),
        lambda: extensions(tables),
        beta: extensions(span_vars),
    }
}

fn assert_matches_direct(rng: &mut ChaCha8Rng, statements: &[Statement<'_>]) {
    let blocks = statements
        .iter()
        .map(Statement::blocks)
        .max()
        .expect("a claim");
    let layout = layout(statements, key_blocks(blocks).expect("key blocks")).expect("layout");
    let challenges = random_challenges(rng, layout.tables().count(), layout.span_vars);
    let rho: Vec<K> = (0..layout.witness_vars)
        .map(|_| random_extension(rng))
        .collect();
    let uses = Uses::new(&layout, &challenges, &rho);
    assert_eq!(
        matrix_part(statements, &challenges, &uses).expect("matrix part"),
        direct(statements, &challenges, &uses)
    );
}

/// Random rows of slots up to 9 columns, a second claim on the rows of the
/// first, and rows of slots up to 200 columns, which meet up to 5 blocks.
#[test]
fn the_matrix_part_is_its_definition_on_random_rows() {
    let instances = support::instances();
    let mut rng = ChaCha8Rng::seed_from_u64(0x7765_6967);
    let point: Vec<K> = (0..support::POINT_VARS)
        .map(|_| random_extension(&mut rng))
        .collect();
    let shared = claim_of(&instances[0].rows, &instances[0].witness, &point);
    let wide = Rows::random_with_widest(&mut rng, 150, 70, 200);
    let wide_claim = claim_of(&wide, &random_witness(&mut rng, 70, 1), &point);
    let mut statements = support::statements(&instances);
    statements.extend([
        Statement {
            claim: &shared,
            rows: &instances[0].rows,
            norm: Norm::Caller,
        },
        Statement {
            claim: &wide_claim,
            rows: &wide,
            norm: Norm::Caller,
        },
    ]);
    assert_matches_direct(&mut rng, &statements);
}

/// The rows of a step circuit as `nightstream` reads them.
struct Circuit {
    rows: PlanRows,
    spans: SlotSpans,
    blocks: usize,
}

impl Circuit {
    fn new(package: &StepPackage) -> Self {
        let layout = package.layout().expect("layout");
        Self {
            rows: package.plan_rows().expect("rows"),
            spans: layout.spans(),
            blocks: layout.width().div_ceil(D),
        }
    }

    fn rows(&self) -> FactoredRows<'_> {
        FactoredRows::new(
            self.shape(),
            FactoredEntries {
                offsets: self.rows.offsets(),
                slots: self.rows.columns(),
                coefficients: self.rows.coefficients(),
            },
            FactoredSlots {
                starts: &self.spans.starts,
                widths: &self.spans.widths,
                ratio: self.spans.ratio,
            },
        )
        .expect("factored rows")
    }
}

impl MatrixRows for Circuit {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows.row_count(),
            columns: self.blocks * D,
            matrices: MATRICES,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.rows().visit_rows(rows, sink)
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        Ok(Some(self.rows()))
    }
}

/// A claim of the shape of a step circuit's claim, with random values.
fn random_claim(rng: &mut ChaCha8Rng) -> CeClaim {
    let mut field = |count: usize| -> Vec<F> { (0..count).map(|_| F::new(rng.next_u64())).collect() };
    let commitment = Commitment::new(KAPPA, field(KAPPA * D)).expect("commitment");
    let public = Mat::from_row_major(D, PUBLIC_WORDS / D, field(PUBLIC_WORDS));
    let mut extensions = |count: usize| -> Vec<K> { (0..count).map(|_| random_extension(rng)).collect() };
    CeClaim {
        c: commitment,
        X: public,
        r: extensions(28),
        eval_k: extensions(D),
        eval_a: (0..MATRICES).map(|_| extensions(D)).collect(),
        m_in: PUBLIC_WORDS,
        fold_digest: [0; 32],
        adv: None,
    }
}

/// The three claims of a final proof: light, refresh, then a fresh claim of
/// each circuit in turn.
#[test]
fn the_matrix_part_is_its_definition_on_the_step_circuits() {
    let artifact = |name: &str| {
        std::fs::read(format!(
            "{}/../../formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-{name}-v1.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap_or_else(|error| panic!("read the {name} package: {error}"))
    };
    let packages = load_bounded_depth_packages(
        &artifact("light"),
        &artifact("refresh"),
        BOUNDED_DEPTH_PACKAGES_IDENTITY,
    )
    .expect("packages");
    let circuits = [Circuit::new(packages.light()), Circuit::new(packages.refresh())];
    drop(packages);
    let mut rng = ChaCha8Rng::seed_from_u64(0x7374_6570);
    let claims: [CeClaim; 3] = std::array::from_fn(|_| random_claim(&mut rng));
    let table = Norm::Table { bound: 10_368 };
    for fresh in &circuits {
        let statements = [
            Statement {
                claim: &claims[0],
                rows: &circuits[0],
                norm: table,
            },
            Statement {
                claim: &claims[1],
                rows: &circuits[1],
                norm: table,
            },
            Statement {
                claim: &claims[2],
                rows: fresh,
                norm: Norm::Caller,
            },
        ];
        assert_matches_direct(&mut rng, &statements);
    }
}
