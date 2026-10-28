//! The complete layer-1 argument on CE(B) claims built by repository code.

use std::time::Instant;

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ccs::Mat;
use neo_math::{F, K};
use neo_reductions::superneo_eval::CachedMatrixRows;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing as _, PrimeField64 as _};
use p3_sumcheck_v08::OpeningBatch;
use p3_whir::pcs::proof::QueryOpenings;

use super::{commit, word, Scratch, Toy};
use crate::circuit::Native;
use crate::device::Accelerator;
use crate::field::{Ext, Gl};
use crate::prover::witness_table;
use crate::ring::{self, Mixing};
use crate::{prove, verify, Claim, ClaimWords, Key, Proof, Relation, Setup, LANES};

/// The honest parent bound for 17 folded sources: 17 · T · (b - 1) = 17 · 216.
pub(super) const BOUND: u32 = 3672;
const BITS: f64 = 100.0;

pub(super) fn transcript(extra: u64) -> Poseidon2Transcript {
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&[F::from_u64(9), F::from_u64(extra)]);
    transcript
}

/// A toy claim with its setup files in a scratch directory.
pub(super) fn setup_toy(seed: u64, blocks: usize, rows: usize) -> (Toy, Setup, Scratch) {
    let toy = Toy::new(seed, blocks, rows, 4, BOUND);
    let scratch = Scratch::new(&format!("layer1-{seed}"));
    let setup = Setup::build(&CachedMatrixRows::new(&toy.cache).unwrap(), &scratch.0).unwrap();
    (toy, setup, scratch)
}

pub(super) fn relation_of<'k>(toy: &Toy, key: &'k Key) -> Relation<'k> {
    Relation::new(key, toy.public_blocks, toy.point_variables, BOUND, BITS).unwrap()
}

fn holds(setup: &Setup, relation: &Relation<'_>, claim: &Claim, witness: &Mat<F>) -> bool {
    let shape = &relation.shape;
    let statement = ClaimWords::new(shape, claim).unwrap();
    let lambda = Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(77, i as u64)));
    let mixing = Mixing::new(&mut Native, lambda, shape);
    let early = setup.tables.early(&claim.r);
    let ubar = setup.tables.slot_weights(&early, &mixing.eval_a());
    let eval_a = setup.tables.column_weights(&ubar, shape.blocks);
    let weights = ring::block_weights(
        shape,
        &claim.r,
        &mixing,
        &eval_a,
        &ring::key_weights(shape, &mixing, &crate::device::Accelerator::cpu()),
    );
    let z = witness_table(shape, witness).unwrap();
    let (_, remainder) = ring::divide(&weights, &z, LANES, &ring::targets(&mut Native, &statement, &mixing));
    remainder.iter().all(|&value| value == Ext::ZERO)
}

pub(super) fn claim_mutations(claim: &Claim) -> Vec<(&'static str, Claim)> {
    let bump = |value: K, part: usize| {
        let mut parts = neo_math::KExtensions::as_coeffs(&value);
        parts[part] += F::ONE;
        <K as neo_math::KExtensions>::from_coeffs(parts)
    };
    let mut cases = Vec::new();
    let mut changed = claim.clone();
    changed.c.data_mut()[0] += F::ONE;
    cases.push(("commitment row 0", changed));
    let mut changed = claim.clone();
    let last = changed.c.data().len() - 1;
    changed.c.data_mut()[last] += F::ONE;
    cases.push(("commitment last row", changed));
    let mut changed = claim.clone();
    changed.X[(5, 0)] += F::ONE;
    cases.push(("public input", changed));
    let mut changed = claim.clone();
    changed.eval_k[0] = bump(changed.eval_k[0], 0);
    cases.push(("Eval_K Re", changed));
    let mut changed = claim.clone();
    changed.eval_k[53] = bump(changed.eval_k[53], 1);
    cases.push(("Eval_K Im", changed));
    let mut changed = claim.clone();
    let matrix = changed.eval_a.len() - 1;
    changed.eval_a[matrix][27] = bump(changed.eval_a[matrix][27], 1);
    cases.push(("Eval_A Im", changed));
    let mut changed = claim.clone();
    changed.r[0] = bump(changed.r[0], 0);
    cases.push(("point", changed));
    cases
}

/// One mutation of every proof part outside the WHIR openings' internal data,
/// except the GKR sum-check messages, which `gkr.rs` changes.
pub(super) fn proof_mutations(proof: &Proof) -> Vec<(&'static str, Proof)> {
    let mut cases = Vec::new();
    let mut push = |name, change: &dyn Fn(&mut Proof)| {
        let mut changed = proof.clone();
        change(&mut changed);
        cases.push((name, changed));
    };
    push("histogram", &|p| {
        let from = p.histogram.iter().position(|&count| count > 0).unwrap();
        p.histogram[from] -= 1;
        p.histogram[from + 1] += 1;
    });
    push("early GKR child", &|p| p.early.layers[2].children[1][0] += Ext::ONE);
    push("early GKR root", &|p| p.early.roots[3][0] += Ext::ONE);
    push("early run value", &|p| p.early_values.runs[2] += Ext::ONE);
    push("early pair value", &|p| p.early_values.pairs[1][0] += Ext::ONE);
    push("early block value", &|p| p.early_values.block += Ext::ONE);
    push("quotient low", &|p| p.quotient[0] += Ext::ONE);
    push("quotient high", &|p| p.quotient[52] += Ext::ONE);
    push("linear round", &|p| p.linear[0][0] += Ext::ONE);
    push("last linear round", &|p| {
        let last = p.linear.len() - 1;
        p.linear[last][1] += Ext::ONE;
    });
    push("key value", &|p| p.finals[0] += Ext::ONE);
    push("block weight value", &|p| p.finals[1] += Ext::ONE);
    push("z value", &|p| p.finals[2] += Ext::ONE);
    push("key partial", &|p| p.partials[0][3] += Ext::ONE);
    push("run partial", &|p| p.partials[1][7] += Ext::ONE);
    push("leaf value", &|p| p.leaves[4].values[9] += Gl::ONE);
    push("leaf path", &|p| p.leaves[0].path[1][2] += Gl::ONE);
    push("swapped leaves", &|p| p.leaves.swap(0, 1));
    push("late GKR child", &|p| p.late.layers[1].children[0][2] += Ext::ONE);
    push("late pair value", &|p| p.late_values[3][1] += Ext::ONE);
    push("P1 opening value", &|p| bump_eval(&mut p.p1_opening, 0));
    push("P1 query value", &|p| bump_eval(&mut p.p1_opening, 5));
    push("P0 opening value", &|p| bump_eval(&mut p.p0_opening, 6));
    push("P0 root", &|p| p.p0 = moved_root(&p.p0));
    push("P1 root", &|p| p.p1 = moved_root(&p.p1));
    push("late GKR root", &|p| p.late.roots[0][1] += Ext::ONE);
    cases
}

/// One mutation of each value part of the WHIR openings: OOD answers,
/// sum-check messages, round roots, query rows, Merkle siblings and final
/// polynomials. The proof-of-work witnesses and cap lengths are not changed.
/// The Lean replay does not run WHIR's checks, so only the native and
/// recorded verifiers get these.
pub(super) fn whir_mutations(proof: &Proof) -> Vec<(&'static str, Proof)> {
    let mut cases = Vec::new();
    let mut push = |name, change: &dyn Fn(&mut Proof)| {
        let mut changed = proof.clone();
        change(&mut changed);
        cases.push((name, changed));
    };
    // A WHIR part is changed in each opening that has it, and every part is
    // in at least one opening.
    let mut whir = |names: [&'static str; 2], change: &dyn Fn(&mut crate::pcs::Opening) -> bool| {
        let in_p1 = change(&mut proof.p1_opening.clone());
        let in_p0 = change(&mut proof.p0_opening.clone());
        assert!(in_p1 || in_p0, "no opening has {}", names[0]);
        if in_p1 {
            push(names[0], &|p| assert!(change(&mut p.p1_opening)));
        }
        if in_p0 {
            push(names[1], &|p| assert!(change(&mut p.p0_opening)));
        }
    };
    whir(["P1 WHIR initial OOD", "P0 WHIR initial OOD"], &|o| {
        bump(o.whir.initial_ood_answers.first_mut())
    });
    whir(["P1 WHIR initial sum-check", "P0 WHIR initial sum-check"], &|o| {
        bump(
            o.whir
                .initial_sumcheck
                .polynomial_evaluations
                .first_mut()
                .map(|e| &mut e[1]),
        )
    });
    whir(
        ["P1 WHIR round root", "P0 WHIR round root"],
        &|o| match o.whir.rounds.first_mut() {
            Some(round) => {
                round.commitment = Some(moved_root(round.commitment.as_ref().unwrap()));
                true
            }
            None => false,
        },
    );
    whir(["P1 WHIR round OOD", "P0 WHIR round OOD"], &|o| {
        bump(
            o.whir
                .rounds
                .first_mut()
                .and_then(|r| r.ood_answers.first_mut()),
        )
    });
    whir(["P1 WHIR round query row", "P0 WHIR round query row"], &|o| {
        o.whir
            .rounds
            .first_mut()
            .is_some_and(|r| bump_query_row(&mut r.openings))
    });
    whir(["P1 WHIR round sibling", "P0 WHIR round sibling"], &|o| {
        o.whir
            .rounds
            .first_mut()
            .is_some_and(|r| bump_sibling(&mut r.openings))
    });
    whir(["P1 WHIR round sum-check", "P0 WHIR round sum-check"], &|o| {
        bump(
            o.whir
                .rounds
                .first_mut()
                .and_then(|r| r.sumcheck.polynomial_evaluations.first_mut())
                .map(|e| &mut e[0]),
        )
    });
    whir(["P1 WHIR final polynomial", "P0 WHIR final polynomial"], &|o| {
        bump(
            o.whir
                .final_poly
                .as_mut()
                .and_then(|poly| poly.as_mut_slice().first_mut()),
        )
    });
    whir(["P1 WHIR final query row", "P0 WHIR final query row"], &|o| {
        bump_query_row(&mut o.whir.final_openings)
    });
    whir(["P1 WHIR final sibling", "P0 WHIR final sibling"], &|o| {
        bump_sibling(&mut o.whir.final_openings)
    });
    whir(["P1 WHIR final sum-check", "P0 WHIR final sum-check"], &|o| {
        bump(
            o.whir
                .final_sumcheck
                .as_mut()
                .and_then(|data| data.polynomial_evaluations.first_mut())
                .map(|e| &mut e[0]),
        )
    });
    cases
}

/// Add one to `value`; false when the part is absent.
fn bump(value: Option<&mut Ext>) -> bool {
    value.map(|value| *value += Ext::ONE).is_some()
}

fn bump_eval(opening: &mut crate::pcs::Opening, batch: usize) {
    let mut current = opening.evals[batch].current().to_vec();
    current[0] += Ext::ONE;
    opening.evals[batch] = OpeningBatch::new(current, opening.evals[batch].next().to_vec());
}

fn moved_root(commitment: &crate::pcs::Commitment) -> crate::pcs::Commitment {
    let mut root = commitment.roots()[0];
    root[1] += Gl::ONE;
    p3_merkle_tree_v08::MerkleCap::new(vec![root])
}

type Queries = QueryOpenings<Gl, Ext, crate::pcs::MultiProof>;

fn bump_query_row(openings: &mut Queries) -> bool {
    match openings {
        QueryOpenings::Base(opening) => opening
            .rows
            .first_mut()
            .and_then(|row| row.first_mut())
            .map(|value| *value += Gl::ONE)
            .is_some(),
        QueryOpenings::Extension(opening) => bump(opening.rows.first_mut().and_then(|row| row.first_mut())),
    }
}

fn bump_sibling(openings: &mut Queries) -> bool {
    let siblings = match openings {
        QueryOpenings::Base(opening) => &mut opening.proof.sibling_hashes,
        QueryOpenings::Extension(opening) => &mut opening.proof.sibling_hashes,
    };
    siblings
        .first_mut()
        .map(|hash| hash[0] += Gl::ONE)
        .is_some()
}

#[test]
fn commitment_helper_matches_the_repository() {
    let witness = Mat::from_row_major(
        neo_math::D,
        3,
        (0..neo_math::D * 3)
            .map(|i| [F::ZERO, F::ONE, -F::ONE][(word(5, i as u64) % 3) as usize])
            .collect(),
    );
    assert_eq!(
        commit(&witness),
        commit_production_signed_unit_prefix_matrix(&witness).unwrap()
    );
}

#[test]
fn batched_row_holds_exactly_for_a_true_claim() {
    let (toy, setup, _scratch) = setup_toy(1, 4, 6);
    let relation = relation_of(&toy, setup.key());
    assert!(holds(&setup, &relation, &toy.claim, &toy.witness));
    for (name, claim) in claim_mutations(&toy.claim) {
        assert!(!holds(&setup, &relation, &claim, &toy.witness), "{name}");
    }
    let mut witness = toy.witness.clone();
    witness[(53, 3)] += F::ONE;
    assert!(!holds(&setup, &relation, &toy.claim, &witness), "carrier tail lane");
}

#[test]
fn keys_derive_reopen_and_bind() {
    let (toy, setup, scratch) = setup_toy(5, 4, 6);
    let rows = CachedMatrixRows::new(&toy.cache).unwrap();
    let key = Key::derive(&rows).unwrap();
    assert_eq!(&key, setup.key());
    assert_eq!(Key::from_bytes(&key.to_bytes()).unwrap(), key);
    assert!(Setup::open(&rows, &scratch.0, &key).is_ok());
    let (other, other_setup, _other_scratch) = setup_toy(6, 4, 6);
    assert_ne!(other_setup.key(), &key);
    assert!(Setup::open(&rows, &scratch.0, other_setup.key()).is_err());
    let other_rows = CachedMatrixRows::new(&other.cache).unwrap();
    assert!(Setup::open(&other_rows, &scratch.0, &key).is_err());
}

/// A pinned key is decoded, not trusted: a key whose structure the verifier
/// would index out of range does not decode.
#[test]
fn malformed_keys_do_not_decode() {
    let toy = Toy::new(5, 4, 6, 4, BOUND);
    let key = Key::derive(&CachedMatrixRows::new(&toy.cache).unwrap()).unwrap();
    let mut class = key.clone();
    class.structure.segments[0].class = class.structure.classes.len() as u32;
    let mut lane = key.clone();
    lane.structure.segments[0].lane = neo_math::D as u32;
    let mut runs = key.clone();
    runs.structure.run_bounds.reverse();
    for (malformed, what) in [(class, "segment class"), (lane, "segment lane"), (runs, "run bounds")] {
        assert!(Key::from_bytes(&malformed.to_bytes()).is_err(), "{what}");
    }
}

/// A root word `w + p` names the same field element as `w`, so it would be a
/// second encoding of one key. It does not decode. The root's first word
/// leads the key's encoding.
#[test]
fn non_canonical_root_words_do_not_decode() {
    let toy = Toy::new(5, 4, 6, 4, BOUND);
    let mut bytes = Key::derive(&CachedMatrixRows::new(&toy.cache).unwrap())
        .unwrap()
        .to_bytes();
    bytes[..8].copy_from_slice(&1u64.to_le_bytes());
    assert!(Key::from_bytes(&bytes).is_ok(), "canonical root word");
    bytes[..8].copy_from_slice(&(1 + Gl::ORDER_U64).to_le_bytes());
    assert!(Key::from_bytes(&bytes).is_err(), "root word w + p");
}

/// `Relation::new` gives the sizes of the Lean `parentShape_of`
/// (`formal/compression-lean/CompressionLean/Final/RustSetup.lean`): `D = 54`,
/// six lane variables, `κ = 22`, the key's matrices and blocks, and the public
/// blocks and bound that the caller passes. The parent passes five public
/// blocks and 3672 (`nightstream`, `parent_relation_arguments_are_the_lean_constants`).
#[test]
fn relation_shape_has_the_lean_constants() {
    let toy = Toy::new(5, 6, 6, 4, BOUND);
    let key = Key::derive(&CachedMatrixRows::new(&toy.cache).unwrap()).unwrap();
    let relation = Relation::new(&key, 5, toy.point_variables, BOUND, BITS).unwrap();
    let shape = &relation.shape;
    assert_eq!((neo_math::D, crate::LANE_VARIABLES, shape.kappa), (54, 6, 22));
    assert_eq!((shape.matrices, shape.blocks), (4, 6));
    assert_eq!((shape.public_blocks, shape.norm_bound), (5, 3672));
}

#[test]
fn true_claims_verify_and_every_tampered_part_rejects() {
    let (toy, setup, _scratch) = setup_toy(2, 4, 6);
    let relation = relation_of(&toy, setup.key());
    assert!(relation.security_bits() >= BITS);
    let proof = prove(
        &relation,
        &setup,
        &Accelerator::cpu(),
        transcript(1),
        &toy.claim,
        &toy.witness,
    )
    .unwrap();
    let accepts = |claim: &Claim, proof: &Proof, extra: u64| verify(&relation, transcript(extra), claim, proof).is_ok();
    assert!(accepts(&toy.claim, &proof, 1));
    let decoded = Proof::from_bytes(&proof.to_bytes()).unwrap();
    assert!(accepts(&toy.claim, &decoded, 1));

    assert!(!accepts(&toy.claim, &proof, 2), "fold transcript");
    for (name, claim) in claim_mutations(&toy.claim) {
        assert!(!accepts(&claim, &proof, 1), "{name}");
    }
    for (name, changed) in proof_mutations(&proof)
        .into_iter()
        .chain(whir_mutations(&proof))
    {
        assert!(!accepts(&toy.claim, &changed, 1), "{name}");
    }
    let mut bytes = proof.to_bytes();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 1;
    assert!(
        Proof::from_bytes(&bytes).map_or(true, |changed| !accepts(&toy.claim, &changed, 1)),
        "proof byte"
    );

    // The same proof under the key of other matrices of the same sizes.
    let (_, other, _other_scratch) = setup_toy(7, 4, 6);
    let other_relation = relation_of(&toy, other.key());
    assert!(
        verify(&other_relation, transcript(1), &toy.claim, &proof).is_err(),
        "other key"
    );
}

#[test]
fn a_false_witness_gives_a_proof_that_rejects() {
    let (toy, setup, _scratch) = setup_toy(3, 4, 6);
    let relation = relation_of(&toy, setup.key());
    for (block, lane) in [(0, 0), (2, 17), (3, 53)] {
        let mut witness = toy.witness.clone();
        witness[(lane, block)] += F::ONE;
        let proof = prove(
            &relation,
            &setup,
            &Accelerator::cpu(),
            transcript(1),
            &toy.claim,
            &witness,
        )
        .unwrap();
        assert!(
            verify(&relation, transcript(1), &toy.claim, &proof).is_err(),
            "block {block} lane {lane}"
        );
    }
    let mut witness = toy.witness.clone();
    witness[(1, 1)] = F::from_u64(u64::from(BOUND) + 1);
    assert!(
        prove(
            &relation,
            &setup,
            &Accelerator::cpu(),
            transcript(1),
            &toy.claim,
            &witness
        )
        .is_err(),
        "norm bound"
    );
}

#[test]
fn larger_relation_round_trips() {
    let start = Instant::now();
    let (toy, setup, _scratch) = setup_toy(4, 1 << 10, 1 << 10);
    let built = start.elapsed();
    let relation = relation_of(&toy, setup.key());
    let start = Instant::now();
    let proof = prove(
        &relation,
        &setup,
        &Accelerator::cpu(),
        transcript(1),
        &toy.claim,
        &toy.witness,
    )
    .unwrap();
    let proved = start.elapsed();
    let start = Instant::now();
    verify(&relation, transcript(1), &toy.claim, &proof).unwrap();
    eprintln!(
        "2^{} cube: setup {built:?}, prove {proved:?}, verify {:?}, {} proof bytes, {} queries, {:.1} bits",
        relation.shape.cube_variables(),
        start.elapsed(),
        proof.to_bytes().len(),
        relation.queries,
        relation.security_bits()
    );
}
