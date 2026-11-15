//! `matrix_part` equals the output of its Lean model
//! (`Spec.Compression.Verifier.matrixPass`) on the sample relations that Lean
//! emits, and it equals its definition on random rows and on the rows of the
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

use super::{matrix_part, Challenges, Matrices, Uses, COLUMNS_PER_TASK, KAPPA};
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

/// Slots in no order, whose columns overlap and cross the tasks of
/// `Matrices::add_columns`.
#[test]
fn the_column_weights_are_their_definition_across_tasks() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x636f_6c73);
    let columns = 3 * COLUMNS_PER_TASK + 77;
    let (starts, widths): (Vec<u32>, Vec<u8>) = (0..20_000)
        .map(|_| {
            let width = rng.next_u32() as u8;
            (rng.next_u32() % (columns - usize::from(width) + 1) as u32, width)
        })
        .unzip();
    let weights: Vec<K> = starts.iter().map(|_| random_extension(&mut rng)).collect();
    let ratio = F::from_u64(7);
    let mut expected = vec![K::ZERO; columns];
    for ((&start, &width), &weight) in starts.iter().zip(&widths).zip(&weights) {
        let mut term = weight;
        for column in &mut expected[start as usize..][..usize::from(width)] {
            *column += term;
            term = term.scale_base(ratio);
        }
    }
    let slots = FactoredSlots {
        starts: &starts,
        widths: &widths,
        ratio,
    };
    let matrices = Matrices::of_slots(slots, weights, columns);
    let mut actual = vec![K::ZERO; columns];
    for (task, output) in actual.chunks_mut(COLUMNS_PER_TASK).enumerate() {
        matrices.add_columns(task, output);
    }
    assert!(actual == expected);
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

/// Factored rows from the tables of a Lean relation.
struct LeanRows {
    shape: MatrixShape,
    offsets: Vec<usize>,
    slots: Vec<u32>,
    coefficients: Vec<F>,
    starts: Vec<u32>,
    widths: Vec<u8>,
    ratio: F,
}

impl LeanRows {
    /// `[rows, columns, ratio, starts, widths, entries]`, with the pairs
    /// `[slot, coefficient]` of each row and matrix in `entries`.
    fn new(relation: &serde_json::Value) -> Self {
        let number = |value: &serde_json::Value| value.as_u64().expect("a natural number");
        let numbers = |value: &serde_json::Value| -> Vec<u64> {
            let numbers = value.as_array().expect("numbers").iter();
            numbers.map(number).collect()
        };
        let (mut offsets, mut slots, mut coefficients) = (vec![0], Vec::new(), Vec::new());
        let rows = relation[5].as_array().expect("rows");
        for row in rows {
            let matrices = row.as_array().expect("matrices");
            assert_eq!(matrices.len(), MATRICES);
            for matrix in matrices {
                for entry in matrix.as_array().expect("entries") {
                    slots.push(number(&entry[0]) as u32);
                    coefficients.push(F::from_u64(number(&entry[1])));
                }
                offsets.push(slots.len());
            }
        }
        assert_eq!(rows.len() as u64, number(&relation[0]));
        Self {
            shape: MatrixShape {
                rows: rows.len(),
                columns: number(&relation[1]) as usize,
                matrices: MATRICES,
            },
            offsets,
            slots,
            coefficients,
            starts: numbers(&relation[3])
                .into_iter()
                .map(|start| start as u32)
                .collect(),
            widths: numbers(&relation[4])
                .into_iter()
                .map(|width| width as u8)
                .collect(),
            ratio: F::from_u64(number(&relation[2])),
        }
    }

    fn rows(&self) -> FactoredRows<'_> {
        FactoredRows::new(
            self.shape,
            FactoredEntries {
                offsets: &self.offsets,
                slots: &self.slots,
                coefficients: &self.coefficients,
            },
            FactoredSlots {
                starts: &self.starts,
                widths: &self.widths,
                ratio: self.ratio,
            },
        )
        .expect("factored rows")
    }
}

impl MatrixRows for LeanRows {
    fn shape(&self) -> MatrixShape {
        self.shape
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.rows().visit_rows(rows, sink)
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        Ok(Some(self.rows()))
    }
}

/// Lean emits the output of its model of `matrix_part` on sample relations:
/// slots across one and two block borders, two claims on one relation, and
/// claims on two relations of one size.
#[test]
fn the_matrix_part_is_the_lean_matrix_pass() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts/nightstream-fprime-matrix-pass-parity-v1.json");
    let file: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("Lean matrix pass file")).expect("JSON");
    assert_eq!(file[0].as_u64(), Some(1), "Lean matrix pass schema");
    let scalars = |value: &serde_json::Value| -> Vec<K> {
        let scalars = value.as_array().expect("values of K").iter();
        let word = |word: &serde_json::Value| F::from_u64(word.as_u64().expect("u64"));
        scalars
            .map(|pair| neo_math::from_complex(word(&pair[0]), word(&pair[1])))
            .collect()
    };
    let cases = file[1].as_array().expect("cases");
    assert_eq!(cases.len(), 3);
    for case in cases {
        let relations: Vec<LeanRows> = case[0]
            .as_array()
            .expect("relations")
            .iter()
            .map(LeanRows::new)
            .collect();
        assert!(relations.iter().any(|rows| {
            let wide =
                |(&start, &width): (&u32, &u8)| start as usize / D + 2 <= (start as usize + width as usize - 1) / D;
            rows.starts.iter().zip(&rows.widths).any(wide)
        }));
        let claims: Vec<(usize, CeClaim, Norm)> = case[1]
            .as_array()
            .expect("claims")
            .iter()
            .map(|claim| {
                let norm = match claim[2].as_u64().expect("the norm word") {
                    0 => Norm::Caller,
                    word => Norm::Table { bound: word - 1 },
                };
                let relation = claim[0].as_u64().expect("relation") as usize;
                let claim = CeClaim {
                    c: Commitment::zeros(KAPPA),
                    X: Mat::from_row_major(D, 0, Vec::new()),
                    r: scalars(&claim[1]),
                    eval_k: Vec::new(),
                    eval_a: vec![Vec::new(); MATRICES],
                    m_in: 0,
                    fold_digest: [0; 32],
                    adv: None,
                };
                (relation, claim, norm)
            })
            .collect();
        let statements: Vec<Statement<'_>> = claims
            .iter()
            .map(|(relation, claim, norm)| Statement {
                claim,
                rows: &relations[*relation],
                norm: *norm,
            })
            .collect();
        let sizes: Vec<(usize, Norm)> = statements
            .iter()
            .map(|statement| (statement.blocks(), statement.norm))
            .collect();
        let layout = crate::Layout::new(&sizes, 512).expect("layout");
        let [mu, omega, delta] = scalars(&case[2]).try_into().expect("three challenges");
        let challenges = Challenges {
            taus: [F::ZERO; 4],
            omega,
            delta,
            gamma: K::ZERO,
            mu,
            psi: [K::ZERO; 4],
            xi: Vec::new(),
            lambda: Vec::new(),
            beta: Vec::new(),
        };
        let rho = scalars(&case[3]);
        assert_eq!(rho.len(), layout.witness_vars);
        let uses = Uses::new(&layout, &challenges, &rho);
        let expected = scalars(&serde_json::Value::Array(vec![case[4].clone()]))[0];
        assert_eq!(
            matrix_part(&statements, &challenges, &uses).expect("matrix part"),
            expected
        );
        assert_eq!(direct(&statements, &challenges, &uses), expected);
    }
}
