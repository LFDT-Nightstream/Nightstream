//! The shrink layer: closed forms against brute force, and programs proved
//! and verified, with every proof part mutated.

use std::time::Instant;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::D;
use p3_challenger_v08::{CanObserve, FieldChallenger};
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use p3_symmetric_v08::Permutation;

use super::layer1::{relation_of, setup_toy, transcript};
use super::verifier::statement_input;
use super::word;
use crate::circuit::block::Kind;
use crate::circuit::block::{C, KINDS, MATRICES, X};
use crate::circuit::record::{Recorder, Sink, Term, Trace};
use crate::circuit::{algebra, Backend};
use crate::device::Accelerator;
use crate::field::{eq_table, fold_low, Ext, Gl};
use crate::hash::{permutation, seed, Challenger};
use crate::shrink::layout::{diagonal, Layout, Shape};
use crate::shrink::{prove, prove_outer, verify, Program, Shrink, ShrinkProof};
use crate::sumcheck;
use crate::verifier::ProofView;
use crate::{Claim, ClaimWords, Proof, R1cs, Relation};

const BITS: f64 = 100.0;

fn ext(seed: u64, index: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(seed, 4 * index + i as u64)))
}

fn points(seed: u64, count: usize) -> Vec<Ext> {
    (0..count).map(|i| ext(seed, i as u64)).collect()
}

fn eq_at(point: &[Ext], index: u64) -> Ext {
    if index >> point.len() != 0 {
        return Ext::ZERO;
    }
    eq_table(point)[index as usize]
}

#[test]
fn diagonal_matches_brute_force() {
    let x = points(1, 7);
    let y = points(2, 6);
    for (ox, oy, count) in [(0, 0, 5), (3, 9, 11), (12, 0, 40), (0, 20, 44), (7, 7, 1), (5, 2, 0)] {
        let direct: Ext = (0..count)
            .map(|b| eq_at(&x, ox + b) * eq_at(&y, oy + b))
            .sum();
        assert_eq!(diagonal(&[(&x, ox), (&y, oy)], count), direct, "{ox} {oy} {count}");
        let single: Ext = (0..count).map(|b| eq_at(&x, ox + b)).sum();
        assert_eq!(diagonal(&[(&x, ox)], count), single, "{ox} {count}");
    }
    // Control: one block more is a different sum.
    let direct: Ext = (0..11).map(|b| eq_at(&x, 3 + b) * eq_at(&y, 9 + b)).sum();
    assert_ne!(diagonal(&[(&x, 3), (&y, 9)], 12), direct);
}

#[test]
fn block_part_matches_every_entry() {
    let shape = Shape {
        blocks: [3],
        glue_rows: 70,
        glue_cells: 90,
        publics: 5,
    };
    let layout = Layout::new(shape);
    let rx = points(3, layout.row_variables);
    let ry = points(4, layout.cell_variables + 1);
    let rho: [Ext; MATRICES] = std::array::from_fn(|j| ext(5, j as u64));
    let (rows, columns) = (eq_table(&rx), eq_table(&ry));
    let mut direct = Ext::ZERO;
    for kind in KINDS {
        for block in 0..shape.blocks[kind.index()] {
            for entry in kind.entries() {
                direct += rho[entry.matrix]
                    * rows[layout.block_row(kind, block, entry.row)]
                    * columns[layout.entry_column(kind, block, entry.cell)]
                    * entry.coefficient;
            }
        }
    }
    assert_eq!(layout.block_part(&rx, &ry, &rho), direct);
    // Control: one block fewer on the same cubes.
    let fewer = Layout::new(Shape {
        blocks: [2],
        glue_rows: shape.glue_rows + 192,
        glue_cells: shape.glue_cells + 192,
        ..shape
    });
    assert_eq!(
        (fewer.row_variables, fewer.cell_variables),
        (layout.row_variables, layout.cell_variables)
    );
    assert_ne!(fewer.block_part(&rx, &ry, &rho), direct);
}

/// The outer sum-check summed directly: the full `eq` table, every table in
/// `Ext`, and `eq·P` evaluated at every message abscissa.
fn textbook_outer(
    tables: &[Vec<Gl>],
    tau: &[Ext],
    challenger: &mut Challenger,
) -> (Vec<Vec<Ext>>, Vec<Ext>, [Ext; MATRICES]) {
    let mut tables: Vec<Vec<Ext>> = tables
        .iter()
        .map(|table| table.iter().map(|&value| Ext::from(value)).collect())
        .collect();
    let mut eq = eq_table(tau);
    let (mut rounds, mut point) = (Vec::new(), Vec::new());
    while eq.len() > 1 {
        let mut sums = [Ext::ZERO; 9];
        for k in 0..eq.len() / 2 {
            let mut e = eq[2 * k];
            let de = eq[2 * k + 1] - e;
            let mut at: [Ext; MATRICES] = std::array::from_fn(|j| tables[j][2 * k]);
            let step: [Ext; MATRICES] = std::array::from_fn(|j| tables[j][2 * k + 1] - at[j]);
            for (t, sum) in sums.iter_mut().enumerate() {
                if t != 1 {
                    *sum += e * (at[X].exp_u64(7) + at[1] * at[2] + at[3] * at[4] + at[5] * at[6] - at[C]);
                }
                e += de;
                for j in 0..MATRICES {
                    at[j] += step[j];
                }
            }
        }
        let mut message = vec![sums[0]];
        message.extend_from_slice(&sums[2..]);
        let r = sumcheck::send(&message, challenger);
        fold_low(&mut eq, r);
        for table in &mut tables {
            fold_low(table, r);
        }
        rounds.push(message);
        point.push(r);
    }
    (rounds, point, std::array::from_fn(|j| tables[j][0]))
}

#[test]
fn outer_sum_check_matches_the_textbook_prover() {
    // Random tables break the zero claim, so the prover may not rely on it; a
    // zero `τ` coordinate makes `ℓ(1) = 0`.
    for (m, zero) in [(0, None), (1, None), (2, Some(1)), (5, None), (12, None), (12, Some(7))] {
        let tables: Vec<Vec<Gl>> = (0..MATRICES as u64)
            .map(|j| {
                (0..1u64 << m)
                    .map(|x| Gl::from_u64(word(100 + j, x)))
                    .collect()
            })
            .collect();
        let mut tau = points(200 + m as u64, m);
        if let Some(i) = zero {
            tau[i] = Ext::ZERO;
        }
        let mut challenger = Challenger::new(permutation().clone());
        challenger.observe(Gl::from_u64(m as u64));
        let mut textbook = challenger.clone();
        let fast = prove_outer(tables.clone(), &tau, &mut challenger);
        assert_eq!(fast, textbook_outer(&tables, &tau, &mut textbook), "m = {m}");
        let draw: Ext = challenger.sample_algebra_element();
        assert_eq!(draw, textbook.sample_algebra_element(), "m = {m}");
    }
}

/// Knowledge of `secret` with `P(P(secret, s1, 0, ...))[0] = s0`, plus some
/// products, an extension inverse, a bit split and a fixed R1CS system.
struct Toy {
    key: Vec<u64>,
    statement: Vec<Gl>,
    secret: Option<Gl>,
}

impl Toy {
    fn digest(secret: Gl, tag: Gl) -> Gl {
        let mut state = [Gl::ZERO; WIDTH];
        state[0] = secret;
        state[1] = tag;
        permutation().permute(permutation().permute(state))[0]
    }

    fn honest(secret: u64, tag: u64) -> (Self, Self) {
        let (secret, tag) = (Gl::from_u64(secret), Gl::from_u64(tag));
        let statement = vec![Self::digest(secret, tag), tag];
        let prover = Self {
            key: vec![1, 2],
            statement: statement.clone(),
            secret: Some(secret),
        };
        (
            prover,
            Self {
                key: vec![1, 2],
                statement,
                secret: None,
            },
        )
    }
}

impl Program for Toy {
    fn statement(&self) -> Vec<u64> {
        self.statement.iter().map(Gl::as_canonical_u64).collect()
    }

    fn key(&self) -> Vec<u64> {
        self.key.clone()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), crate::Error> {
        let words: Vec<B::F> = self
            .statement
            .iter()
            .map(|word| b.public(word.as_canonical_u64()))
            .collect();
        let secret = algebra::private(b, self.secret.unwrap_or(Gl::ZERO));
        let zero = b.constant(0);
        let mut state = [zero; WIDTH];
        state[0] = secret;
        state[1] = words[1];
        let state = b.permute(state);
        let state = b.permute(state);
        b.assert_equal(state[0], words[0], "digest")?;
        let square = b.mul(secret, secret);
        let value = b.ext([square, secret, words[1]]);
        let value = b.ext_mul(value, value);
        b.ext_inverse(value, "a nonzero value")?;
        b.bits(state[3], 64, "bits")?;
        let cube = b.mul(square, secret);
        let one = b.constant(1);
        b.r1cs(&cubes(), &[secret, square, cube, one, words[1]], "toy system")?;
        Ok(())
    }
}

/// Over `[s, s², s³, 1, t]`: `s·s = s²`, `s²·s = s³`, `(s + 3)·1 = s + 3`,
/// and `(s + t − t + t)·1 = s + t` (one column three times in one side).
fn cubes() -> R1cs {
    let minus = |c: u64| Gl::ORDER_U64 - c;
    let mut system = R1cs::new(5);
    system.push(&[(0, 1)], &[(0, 1)], &[(1, 1)]);
    system.push(&[(1, 1)], &[(0, 1)], &[(2, 1)]);
    system.push(&[(0, 1), (3, 3)], &[(3, 1)], &[(0, 1), (3, 3)]);
    system.push(&[(0, 1), (4, 1), (4, minus(1)), (4, 1)], &[(3, 1)], &[(0, 1), (4, 1)]);
    system
}

#[test]
fn native_r1cs_accepts_its_rows_and_rejects_a_changed_column() {
    let s = Gl::from_u64(5);
    let columns = [s, s * s, s * s * s, Gl::ONE, Gl::from_u64(9)];
    assert!(crate::Native.r1cs(&cubes(), &columns, "rows").is_ok());
    // `t` is free: it enters only the identity row.
    for column in 0..4 {
        let mut changed = columns;
        changed[column] += Gl::ONE;
        assert!(
            crate::Native.r1cs(&cubes(), &changed, "rows").is_err(),
            "column {column}"
        );
        // The constraint reading notes the failure and records every row.
        let mut recorder = Recorder::new(Trace::default());
        let forms: Vec<_> = changed
            .iter()
            .map(|value| recorder.private(value.as_canonical_u64()))
            .collect();
        recorder.r1cs(&cubes(), &forms, "rows").unwrap();
        let (trace, failure) = recorder.finish();
        assert_eq!(failure, Some("rows"));
        assert_eq!(trace.rows.len(), cubes().rows());
    }
}

/// A trace that records R1CS rows one at a time, by the `Sink` default.
struct RowByRow(Trace);

impl Sink for RowByRow {
    fn row(&mut self, products: &[(Vec<Term>, Vec<Term>)], linear: &[Term]) {
        self.0.row(products, linear);
    }
    fn cell(&mut self, value: Gl) {
        self.0.cell(value);
    }
    fn block(&mut self, kind: Kind, cells: &[Gl]) {
        self.0.block(kind, cells);
    }
    fn public(&mut self, value: Gl) {
        self.0.public(value);
    }
}

/// The trace builds R1CS rows in parallel; the rows equal the row-by-row
/// default, for columns of several terms and with constants.
#[test]
fn parallel_r1cs_rows_equal_the_default() {
    fn record<S: Sink>(sink: S) -> S {
        let mut b = Recorder::new(sink);
        let s = b.private(5);
        let square = b.private(25);
        let shifted = b.add(s, square);
        let shifted = b.sub(shifted, square);
        let cube = b.private(125);
        let one = b.constant(1);
        let t = b.private(9);
        let t = b.scale(t, 1);
        // `t` is free (it enters only the identity row), so `t + 3` holds.
        let three = b.constant(3);
        let t = b.add(t, three);
        b.r1cs(&cubes(), &[shifted, square, cube, one, t], "rows")
            .unwrap();
        b.r1cs(&cubes(), &[s, square, cube, one, t], "rows")
            .unwrap();
        let (sink, failure) = b.finish();
        assert_eq!(failure, None);
        sink
    }
    let parallel = record(Trace::default());
    assert_eq!(parallel.rows.len(), 2 * cubes().rows());
    assert!(parallel
        .entries
        .iter()
        .any(|&(wire, _)| wire == crate::circuit::record::ONE));
    assert_eq!(parallel, record(RowByRow(Trace::default())).0);
}

/// A check that fails between constants records rows that cannot hold, so
/// the shrink's equations carry the failure, not only the run.
#[test]
fn failing_constant_checks_record_rows() {
    let mut b = Recorder::new(Trace::default());
    let five = b.constant(5);
    b.assert_zero(five, "constant").unwrap();
    let zero = b.constant(0);
    b.ext_inverse([zero; 3], "constant inverse").unwrap();
    let (trace, failure) = b.finish();
    assert_eq!(failure, Some("constant"));
    let failing = (0..trace.rows.len())
        .filter(|&row| trace.row_value(row) != Gl::ZERO)
        .count();
    assert_eq!(failing, 2, "one row per failed check");
}

fn shrink_of(program: &impl Program) -> Shrink {
    Shrink::new(Shape::derive(program).unwrap(), BITS).unwrap()
}

fn mutations(proof: &ShrinkProof) -> Vec<(&'static str, ShrinkProof)> {
    let mut cases = Vec::new();
    let mut push = |name, change: &dyn Fn(&mut ShrinkProof)| {
        let mut changed = proof.clone();
        change(&mut changed);
        cases.push((name, changed));
    };
    push("first outer round", &|p| p.outer[0][0] += Ext::ONE);
    push("last outer round", &|p| p.outer.last_mut().unwrap()[7] += Ext::ONE);
    for j in 0..MATRICES {
        push("matrix value", &move |p| p.values[j] += Ext::ONE);
    }
    push("first inner round", &|p| p.inner[0][1] += Ext::ONE);
    push("last inner round", &|p| p.inner.last_mut().unwrap()[0] += Ext::ONE);
    push("opening value", &|p| {
        let mut current = p.opening.evals[0].current().to_vec();
        current[0] += Ext::ONE;
        p.opening.evals[0] = p3_sumcheck_v08::OpeningBatch::new(current, Vec::new());
    });
    cases
}

#[test]
fn toy_program_proves_verifies_and_rejects_mutations() {
    let (prover, verifier) = Toy::honest(17, 4);
    let shrink = shrink_of(&verifier);
    assert_eq!(Shape::derive(&prover).unwrap(), Shape::derive(&verifier).unwrap());
    assert!(shrink.security_bits() >= BITS);
    let started = Instant::now();
    let proof = prove(&shrink, &prover, &Accelerator::cpu()).unwrap();
    let proved = started.elapsed();
    let started = Instant::now();
    verify(&shrink, &verifier, &proof).unwrap();
    eprintln!(
        "toy shrink: prove {proved:?}, verify {:?}, {} bytes",
        started.elapsed(),
        bincode::serialize(&proof).unwrap().len()
    );
    for (name, changed) in mutations(&proof) {
        assert!(verify(&shrink, &verifier, &changed).is_err(), "{name}");
    }
    let (_, other) = Toy::honest(18, 4);
    assert!(verify(&shrink, &other, &proof).is_err(), "other statement");
    // Same shape and statement, another key: other challenges.
    let other_key = Toy {
        key: vec![1, 3],
        ..Toy::honest(17, 4).1
    };
    assert_eq!(Shape::derive(&other_key).unwrap(), Shape::derive(&verifier).unwrap());
    assert!(verify(&shrink, &other_key, &proof).is_err(), "other key");
    // A false statement has no witness.
    let false_statement = Toy {
        key: vec![1, 2],
        statement: vec![Gl::ONE, Gl::from_u64(4)],
        secret: Some(Gl::from_u64(17)),
    };
    assert!(matches!(
        prove(&shrink, &false_statement, &Accelerator::cpu()),
        Err(crate::Error::Witness("digest"))
    ));
}

const TOY_PROOF_BYTES: (usize, u64) = (20070, 2147401111984846564);

/// The prover runs on one thread: with more, the bytes differ from run to run.
#[test]
fn toy_proof_bytes_are_pinned() {
    let (prover, verifier) = Toy::honest(17, 4);
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let proof = one_thread
        .install(|| prove(&shrink_of(&verifier), &prover, &Accelerator::cpu()))
        .unwrap();
    let bytes = bincode::serialize(&proof).unwrap();
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    assert_eq!((bytes.len(), hash), TOY_PROOF_BYTES);
}

/// The toy's key and statement with no checks: a smaller shape.
struct Short(Toy);

impl Program for Short {
    fn statement(&self) -> Vec<u64> {
        self.0.statement()
    }

    fn key(&self) -> Vec<u64> {
        self.0.key()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), crate::Error> {
        for word in self.statement() {
            b.public(word);
        }
        Ok(())
    }
}

/// A shape smaller than the program's run, as in a stale pinned key: the
/// verifier rejects at the count check instead of indexing past its eq
/// tables.
#[test]
fn verify_rejects_a_run_past_the_shape() {
    let short = Short(Toy::honest(17, 4).1);
    let shrink = shrink_of(&short);
    let proof = prove(&shrink, &short, &Accelerator::cpu()).unwrap();
    verify(&shrink, &short, &proof).unwrap();
    let (_, toy) = Toy::honest(17, 4);
    assert!(matches!(
        verify(&shrink, &toy, &proof),
        Err(crate::Error::Rejected(
            "the program's shape run differs from the shrink shape"
        ))
    ));
}

/// The layer-1 verifier as a shrink program.
pub(super) struct Layer1<'a> {
    relation: &'a Relation<'a>,
    seed: [Gl; 4],
    statement: ClaimWords<Gl>,
    proof: Option<&'a Proof>,
}

impl<'a> Layer1<'a> {
    pub(super) fn new(relation: &'a Relation<'a>, seed: [Gl; 4], claim: &Claim, proof: Option<&'a Proof>) -> Self {
        Self {
            relation,
            seed,
            statement: ClaimWords::new(&relation.shape, claim).unwrap(),
            proof,
        }
    }

    /// A shape run: a zero seed, zero statement words and no proof. A run
    /// never branches on a value, so every run records these rows.
    pub(super) fn shape(relation: &'a Relation<'a>) -> Self {
        let shape = &relation.shape;
        let pair = [Gl::ZERO; 2];
        let statement = ClaimWords {
            commitment: vec![[Gl::ZERO; D]; shape.kappa],
            public: vec![[Gl::ZERO; D]; shape.public_blocks],
            point: vec![pair; shape.point_variables],
            eval_k: [pair; D],
            eval_a: vec![[pair; D]; shape.matrices],
        };
        Self {
            relation,
            seed: [Gl::ZERO; 4],
            statement,
            proof: None,
        }
    }
}

impl Program for Layer1<'_> {
    fn key(&self) -> Vec<u64> {
        self.relation.key_words()
    }

    fn statement(&self) -> Vec<u64> {
        let mut words = self.seed.to_vec();
        words.extend(self.statement.words());
        words.iter().map(Gl::as_canonical_u64).collect()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), crate::Error> {
        let seed = self.seed.map(|word| b.public(word.as_canonical_u64()));
        let statement = statement_input(b, &self.statement);
        let view = ProofView::read(b, self.relation, self.proof)?;
        crate::verifier::verify(b, self.relation, seed, &statement, &view)
    }
}

#[test]
fn layer1_verifier_shrinks() {
    let (toy, setup, _scratch) = setup_toy(8, 4, 6);
    let relation = relation_of(&toy, setup.key());
    let inner = crate::prove(
        &relation,
        &setup,
        &Accelerator::cpu(),
        transcript(1),
        &toy.claim,
        &toy.witness,
    )
    .unwrap();
    let honest = seed(transcript(1));
    let prover = Layer1::new(&relation, honest, &toy.claim, Some(&inner));
    let verifier = Layer1::new(&relation, honest, &toy.claim, None);
    let shape = Shape::derive(&verifier).unwrap();
    let shrink = Shrink::new(shape, BITS).unwrap();
    let started = Instant::now();
    let proof = prove(&shrink, &prover, &Accelerator::cpu()).unwrap();
    let proved = started.elapsed();
    let started = Instant::now();
    verify(&shrink, &verifier, &proof).unwrap();
    let layout = Layout::new(shape);
    eprintln!(
        "layer-1 shrink: {shape:?}, 2^{} rows, 2^{} cells, prove {proved:?}, verify {:?}, {} bytes",
        layout.row_variables,
        layout.cell_variables,
        started.elapsed(),
        bincode::serialize(&proof).unwrap().len()
    );
    let other = Layer1::new(&relation, seed(transcript(2)), &toy.claim, None);
    assert!(verify(&shrink, &other, &proof).is_err(), "other seed");
}
