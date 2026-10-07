//! The complete layer-1 argument on CE(B) claims built by repository code.

use std::time::Instant;

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ccs::Mat;
use neo_math::{F, K};
use neo_reductions::superneo_eval::CachedMatrixRows;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use p3_field_v08::{BasedVectorSpace, PrimeCharacteristicRing as _};

use super::{commit, word, Toy};
use crate::field::{Ext, Gl};
use crate::ring::{self, Mixing};
use crate::{prove, verify, witness_table, Claim, Proof, Relation, Statement, LANES};

/// The honest parent bound for 17 folded sources: 17 · T · (b - 1) = 17 · 216.
const BOUND: u32 = 3672;
const BITS: f64 = 100.0;

fn transcript(extra: u64) -> Poseidon2Transcript {
    let mut transcript = Poseidon2Transcript::new_v1_1();
    transcript.absorb_v1_1(&[F::from_u64(9), F::from_u64(extra)]);
    transcript
}

fn holds(relation: &Relation<'_>, claim: &Claim, witness: &Mat<F>) -> bool {
    let shape = &relation.shape;
    let statement = Statement::new(shape, claim).unwrap();
    let lambda = Ext::from_basis_coefficients_fn(|i| Gl::from_u64(word(77, i as u64)));
    let mixing = Mixing::new(lambda, shape);
    let weights = ring::block_weights(relation.matrices, shape, &statement, &mixing).unwrap();
    let z = witness_table(shape, witness).unwrap();
    let (_, remainder) = ring::divide(&weights, &z, LANES, &ring::targets(&statement, &mixing));
    remainder.iter().all(|&value| value == Ext::ZERO)
}

fn claim_mutations(claim: &Claim) -> Vec<(&'static str, Claim)> {
    let bump = |value: K, part: usize| {
        let mut parts = neo_math::KExtensions::as_coeffs(&value);
        parts[part] += F::ONE;
        <K as neo_math::KExtensions>::from_coeffs(parts)
    };
    let mut cases = Vec::new();
    let mut changed = claim.clone();
    changed.c.data[0] += F::ONE;
    cases.push(("commitment row 0", changed));
    let mut changed = claim.clone();
    let last = changed.c.data.len() - 1;
    changed.c.data[last] += F::ONE;
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
    let toy = Toy::new(1, 4, 6, 4, BOUND);
    let rows = CachedMatrixRows::new(&toy.cache).unwrap();
    let relation = Relation::new(&rows, toy.public_blocks, toy.point_variables, BOUND, BITS).unwrap();
    assert!(holds(&relation, &toy.claim, &toy.witness));
    for (name, claim) in claim_mutations(&toy.claim) {
        assert!(!holds(&relation, &claim, &toy.witness), "{name}");
    }
    let mut witness = toy.witness.clone();
    witness[(53, 3)] += F::ONE;
    assert!(!holds(&relation, &toy.claim, &witness), "carrier tail lane");
}

#[test]
fn true_claims_verify_and_every_tampered_part_rejects() {
    let toy = Toy::new(2, 4, 6, 4, BOUND);
    let rows = CachedMatrixRows::new(&toy.cache).unwrap();
    let relation = Relation::new(&rows, toy.public_blocks, toy.point_variables, BOUND, BITS).unwrap();
    assert!(relation.security_bits() >= BITS);
    let proof = prove(&relation, transcript(1), &toy.claim, &toy.witness).unwrap();
    let accepts = |claim: &Claim, proof: &Proof, extra: u64| verify(&relation, transcript(extra), claim, proof).is_ok();
    assert!(accepts(&toy.claim, &proof, 1));
    let decoded = Proof::from_bytes(&proof.to_bytes()).unwrap();
    assert!(accepts(&toy.claim, &decoded, 1));

    assert!(!accepts(&toy.claim, &proof, 2), "fold transcript");
    for (name, claim) in claim_mutations(&toy.claim) {
        assert!(!accepts(&claim, &proof, 1), "{name}");
    }

    let mut changed = proof.clone();
    let from = changed
        .histogram
        .iter()
        .position(|&count| count > 0)
        .unwrap();
    changed.histogram[from] -= 1;
    changed.histogram[from + 1] += 1;
    assert!(!accepts(&toy.claim, &changed, 1), "histogram");
    let mut changed = proof.clone();
    changed.gkr.layers[2].children[1] += Ext::ONE;
    assert!(!accepts(&toy.claim, &changed, 1), "GKR child");
    let mut changed = proof.clone();
    changed.quotient[0] += Ext::ONE;
    assert!(!accepts(&toy.claim, &changed, 1), "quotient low");
    let mut changed = proof.clone();
    changed.quotient[52] += Ext::ONE;
    assert!(!accepts(&toy.claim, &changed, 1), "quotient high");
    let mut changed = proof.clone();
    changed.linear[0][0] += Ext::ONE;
    assert!(!accepts(&toy.claim, &changed, 1), "linear round");
    let mut changed = proof.clone();
    let last = changed.linear.len() - 1;
    changed.linear[last][1] += Ext::ONE;
    assert!(!accepts(&toy.claim, &changed, 1), "last linear round");
    let mut bytes = proof.to_bytes();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 1;
    assert!(
        Proof::from_bytes(&bytes).map_or(true, |changed| !accepts(&toy.claim, &changed, 1)),
        "proof byte"
    );
}

#[test]
fn a_false_witness_gives_a_proof_that_rejects() {
    let toy = Toy::new(3, 4, 6, 4, BOUND);
    let rows = CachedMatrixRows::new(&toy.cache).unwrap();
    let relation = Relation::new(&rows, toy.public_blocks, toy.point_variables, BOUND, BITS).unwrap();
    for (block, lane) in [(0, 0), (2, 17), (3, 53)] {
        let mut witness = toy.witness.clone();
        witness[(lane, block)] += F::ONE;
        let proof = prove(&relation, transcript(1), &toy.claim, &witness).unwrap();
        assert!(
            verify(&relation, transcript(1), &toy.claim, &proof).is_err(),
            "block {block} lane {lane}"
        );
    }
    let mut witness = toy.witness.clone();
    witness[(1, 1)] = F::from_u64(u64::from(BOUND) + 1);
    assert!(
        prove(&relation, transcript(1), &toy.claim, &witness).is_err(),
        "norm bound"
    );
}

#[test]
fn larger_relation_round_trips() {
    let toy = Toy::new(4, 1 << 10, 1 << 10, 4, BOUND);
    let rows = CachedMatrixRows::new(&toy.cache).unwrap();
    let relation = Relation::new(&rows, toy.public_blocks, toy.point_variables, BOUND, BITS).unwrap();
    let start = Instant::now();
    let proof = prove(&relation, transcript(1), &toy.claim, &toy.witness).unwrap();
    let proved = start.elapsed();
    let start = Instant::now();
    verify(&relation, transcript(1), &toy.claim, &proof).unwrap();
    eprintln!(
        "2^{} cube: prove {proved:?}, verify {:?}, {} proof bytes, {:.1} bits",
        relation.shape.cube_variables(),
        start.elapsed(),
        proof.to_bytes().len(),
        relation.security_bits()
    );
}
