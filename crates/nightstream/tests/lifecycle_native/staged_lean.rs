//! Numeric checker inputs and observed C/R/D values from a verified native fold.
//! Schema: neo-fold-clean/tests/nifs/stage1_values.rs. No expected values enter here.

use super::*;
use neo_math::KExtensions;
use neo_reductions::{engines::pi_ccs_joint::ProtocolTrace, optimized_engine::optimized_verify_with_trace};
use neo_transcript::Poseidon2Transcript;

fn words(values: &[K]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|value| value.to_limbs_u64().into())
        .collect()
}

fn fields(values: &[F]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

fn public_words(value: &Mat<F>) -> Vec<u64> {
    assert_eq!((value.rows(), value.cols()), (D, 5));
    (0..270)
        .map(|index| value[(index % D, index / D)].as_canonical_u64())
        .collect()
}

fn claim_value(claim: &CeClaim, combined: bool) -> Value {
    assert_eq!((claim.c.d, claim.c.kappa, claim.m_in), (D, 22, 270));
    assert_eq!(claim.c.data.len(), 22 * D);
    assert_eq!(claim.r.len(), 28);
    assert_eq!(claim.eval_k.len(), D.next_power_of_two());
    assert_eq!(claim.eval_a.len(), 14);
    assert!(claim.adv.is_none());
    assert!(claim.eval_k[D..].iter().all(|&value| value == K::ZERO));
    for family in &claim.eval_a {
        assert_eq!(family.len(), D.next_power_of_two());
        assert!(family[D..].iter().all(|&value| value == K::ZERO));
    }
    json!([
        fields(&claim.c.data),
        public_words(&claim.X),
        words(&claim.r),
        words(&claim.eval_k[..D]),
        claim
            .eval_a
            .iter()
            .map(|family| words(&family[..D]))
            .collect::<Vec<_>>(),
        u64::from(combined)
    ])
}

fn running_value(claims: &[CeClaim]) -> Value {
    assert_eq!(claims.len(), 16);
    let values: Vec<_> = claims
        .iter()
        .map(|claim| claim_value(claim, false))
        .collect();
    assert!(values.iter().all(|value| value[2] == values[0][2]));
    json!([
        values[0][2],
        values.iter().map(|value| &value[0]).collect::<Vec<_>>(),
        values.iter().map(|value| &value[1]).collect::<Vec<_>>(),
        values.iter().map(|value| &value[3]).collect::<Vec<_>>(),
        values.iter().map(|value| &value[4]).collect::<Vec<_>>()
    ])
}

fn ccs_input(fresh: &CcsClaim, running: &[CeClaim], proof: &pi_ccs::Proof) -> Value {
    assert_eq!((fresh.c.d, fresh.c.kappa, fresh.m_in), (D, 22, 270));
    assert_eq!(fresh.c.data.len(), 22 * D);
    assert_eq!(fresh.x.len(), 270);
    assert!(fresh.adv.is_none());
    assert_eq!(proof.sumcheck.sumcheck_rounds.len(), 28);
    assert!(proof
        .sumcheck
        .sumcheck_rounds
        .iter()
        .all(|round| round.len() == 10));
    assert_eq!(proof.outputs.len(), 17);
    let outputs: Vec<_> = proof
        .outputs
        .iter()
        .map(|claim| claim_value(claim, false))
        .collect();
    json!([
        2,
        fields(&fresh.c.data),
        fields(&fresh.x),
        proof
            .sumcheck
            .sumcheck_rounds
            .iter()
            .map(|round| words(round))
            .collect::<Vec<_>>(),
        outputs.iter().map(|value| &value[3]).collect::<Vec<_>>(),
        outputs.iter().map(|value| &value[4]).collect::<Vec<_>>(),
        running_value(running)
    ])
}

fn ccs_phase(proof: &pi_ccs::Proof, trace: &ProtocolTrace) -> Value {
    let terminal = &trace.terminal_components;
    assert!(proof
        .outputs
        .iter()
        .all(|claim| claim.r == trace.round_challenges));
    json!([
        1,
        words(&trace.alpha),
        words(&[trace.gamma])[0],
        fields(&trace.pre_sumcheck_state),
        words(&trace.round_challenges),
        trace
            .round_states
            .iter()
            .map(|state| fields(state))
            .collect::<Vec<_>>(),
        words(&proof.outputs[0].r),
        words(&[trace.initial_claim])[0],
        words(&trace.round_claims),
        words(&[
            terminal.eval_k,
            terminal.eval_a,
            terminal.ccs,
            terminal.norm,
            terminal.terminal,
            trace.terminal_claim
        ]),
        proof
            .outputs
            .iter()
            .map(|claim| fields(&claim.c.data))
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|claim| public_words(&claim.X))
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|claim| words(&claim.eval_k[..D]))
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|claim| claim
                .eval_a
                .iter()
                .map(|family| words(&family[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        fields(&trace.outgoing_state)
    ])
}

pub(super) fn export(
    directory: &Path,
    package: &PreparedLifecycle,
    fresh: &CcsClaim,
    running: &RunningInstance,
    proof: &NifsProof,
    verified_state: [F; 8],
    verified_absorbed: usize,
) {
    let mut ccs_transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
    let (accepted, trace) = optimized_verify_with_trace(
        &mut ccs_transcript,
        params(package).inner(),
        &package.structure,
        std::slice::from_ref(fresh),
        &running.claims,
        &proof.pi_ccs.outputs,
        &proof.pi_ccs.sumcheck,
    )
    .unwrap();
    assert!(accepted, "fresh Lean export requires an accepted C trace");
    let input = ccs_input(fresh, &running.claims, &proof.pi_ccs);
    let children = running_value(&proof.pi_dec.children);
    let observed = json!({
        "schema": 1,
        "structural_identifier": package.package.structural_identifier(),
        "package_identity": package.package_identity(),
        "pi_ccs_input": input,
        "pi_ccs_phase": ccs_phase(&proof.pi_ccs, &trace),
        "pi_rlc_parent": claim_value(&proof.pi_rlc.combined, true),
        "children": children,
        "outgoing_state": fields(&verified_state),
        "absorbed": verified_absorbed,
    });
    save(&directory.join("pi_ccs_input.json"), &input);
    save(&directory.join("children.json"), &children);
    save(&directory.join("actual_result.json"), &observed);
}

#[path = "staged_lean_tests.rs"]
mod tests;
