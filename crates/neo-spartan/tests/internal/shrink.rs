//! The shrink layer: closed forms against brute force, and programs proved
//! and verified, with every proof part mutated.

use std::time::Instant;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_symmetric_v08::Permutation;

use super::layer1::{relation_of, setup_toy, transcript};
use super::verifier::statement_input;
use super::word;
use crate::circuit::Backend;
use crate::field::{eq_table, Ext, Gl};
use crate::hash::{permutation, seed};
use crate::shrink::layout::{diagonal, entries, Layout, Shape, MATRICES};
use crate::shrink::{prove, verify, Program, Shrink, ShrinkProof};
use crate::verifier::ProofView;
use crate::{Claim, Proof, Relation, Statement};

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
        blocks: 3,
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
    for block in 0..shape.blocks {
        for entry in entries() {
            direct += rho[entry.matrix]
                * rows[layout.block_index(block, entry.row)]
                * columns[layout.entry_column(block, entry.cell)]
                * entry.coefficient;
        }
    }
    assert_eq!(layout.block_part(&rx, &ry, &rho), direct);
    // Control: one block fewer on the same cubes.
    let fewer = Layout::new(Shape {
        blocks: 2,
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

/// Knowledge of `secret` with `P(P(secret, s1, 0, ...))[0] = s0`, plus some
/// products, an extension inverse and a bit split.
struct Toy {
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
            statement: statement.clone(),
            secret: Some(secret),
        };
        (
            prover,
            Self {
                statement,
                secret: None,
            },
        )
    }
}

impl Program for Toy {
    fn statement(&self) -> Vec<Gl> {
        self.statement.clone()
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), crate::Error> {
        let words: Vec<B::F> = self.statement.iter().map(|&word| b.public(word)).collect();
        let secret = b.private(self.secret.unwrap_or(Gl::ZERO));
        let zero = b.constant(Gl::ZERO);
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
        Ok(())
    }
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
    let proof = prove(&shrink, &prover).unwrap();
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
    // A false statement has no witness.
    let false_statement = Toy {
        statement: vec![Gl::ONE, Gl::from_u64(4)],
        secret: Some(Gl::from_u64(17)),
    };
    assert!(matches!(
        prove(&shrink, &false_statement),
        Err(crate::Error::Witness("digest"))
    ));
}

/// The layer-1 verifier as a shrink program.
struct Layer1<'a> {
    relation: &'a Relation<'a>,
    seed: [Gl; 4],
    statement: Statement<Gl>,
    proof: Option<&'a Proof>,
}

impl<'a> Layer1<'a> {
    fn new(relation: &'a Relation<'a>, seed: [Gl; 4], claim: &Claim, proof: Option<&'a Proof>) -> Self {
        Self {
            relation,
            seed,
            statement: Statement::new(&relation.shape, claim).unwrap(),
            proof,
        }
    }
}

impl Program for Layer1<'_> {
    fn statement(&self) -> Vec<Gl> {
        let mut words = self.seed.to_vec();
        words.extend(self.statement.words());
        words
    }

    fn run<B: Backend>(&self, b: &mut B) -> Result<(), crate::Error> {
        let seed = self.seed.map(|word| b.public(word));
        let statement = statement_input(b, &self.statement);
        let view = ProofView::read(b, self.relation, self.proof)?;
        crate::verifier::verify(b, self.relation, seed, &statement, &view)
    }
}

#[test]
fn layer1_verifier_shrinks() {
    let (toy, setup, _scratch) = setup_toy(8, 4, 6);
    let relation = relation_of(&toy, setup.key());
    let inner = crate::prove(&relation, &setup, transcript(1), &toy.claim, &toy.witness).unwrap();
    let honest = seed(transcript(1));
    let prover = Layer1::new(&relation, honest, &toy.claim, Some(&inner));
    let verifier = Layer1::new(&relation, honest, &toy.claim, None);
    let shape = Shape::derive(&verifier).unwrap();
    let shrink = Shrink::new(shape, BITS).unwrap();
    let started = Instant::now();
    let proof = prove(&shrink, &prover).unwrap();
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
