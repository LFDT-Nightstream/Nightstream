//! Fixed Construction-2 NIFS contract tests.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::nifs::{prove_fixed, verify_fixed, Error, FixedNifsAccumulator};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

#[test]
fn canonical_zero_is_fixed_k_and_round_trips_one_fresh_instance() {
    let prep = support::toy_preprocessing();
    let zero = FixedNifsAccumulator::canonical_zero(prep.params(), prep.structure(), prep.combine_b_pows(), D)
        .expect("canonical zero accumulator");
    assert_eq!(zero.claims().len(), prep.params().k_rho() as usize);
    assert!(zero.claims().iter().all(|claim| {
        claim.c.data.iter().all(|&value| value == F::ZERO)
            && claim.X.to_dense_vec().iter().all(|&value| value == F::ZERO)
    }));

    let fresh = support::toy_instance(&prep, 7);
    let fresh_claim = fresh.claim.clone();
    let mut prover_transcript = Transcript::session();
    let (next, proof) = prove_fixed(
        &mut prover_transcript,
        prep.params(),
        prep.structure(),
        prep.optimized_cache(),
        prep.commitment_scheme(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &zero,
    )
    .expect("fixed NIFS prover");

    let zero_verifier = FixedNifsAccumulator::from_verifier_running(
        prep.params(),
        prep.structure(),
        prep.combine_b_pows(),
        zero.running().claims_only(),
    )
    .expect("verifier-side zero");
    let mut verifier_transcript = Transcript::session();
    let verified = verify_fixed(
        &mut verifier_transcript,
        prep.params(),
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claim,
        &zero_verifier,
        &proof,
    )
    .expect("fixed NIFS verifier");

    assert_eq!(next.claims(), verified.claims());
    assert_eq!(next.claims().len(), prep.params().k_rho() as usize);
}

#[test]
fn fixed_interface_rejects_variable_arity_running_state() {
    let prep = support::toy_preprocessing();
    let zero = FixedNifsAccumulator::canonical_zero(prep.params(), prep.structure(), prep.combine_b_pows(), D)
        .expect("canonical zero accumulator");
    let mut malformed = zero.into_running();
    malformed.claims.pop();
    malformed.witnesses.pop();

    let error =
        FixedNifsAccumulator::from_prover_running(prep.params(), prep.structure(), prep.combine_b_pows(), malformed)
            .expect_err("variable-arity accumulator must fail");
    assert!(matches!(error, Error::FixedShape { .. }));
}

#[test]
fn fixed_interface_rejects_a_forged_decomposition_parent_cache() {
    let prep = support::toy_preprocessing();
    let zero = FixedNifsAccumulator::canonical_zero(prep.params(), prep.structure(), prep.combine_b_pows(), D)
        .expect("canonical zero accumulator");
    let mut malformed = zero.into_running();
    malformed
        .parent_authority
        .as_mut()
        .expect("fixed accumulator parent")
        .c
        .data[0] += F::ONE;

    FixedNifsAccumulator::from_prover_running(prep.params(), prep.structure(), prep.combine_b_pows(), malformed)
        .expect_err("the derived parent cache must be checked against all k formal CE claims");
}
