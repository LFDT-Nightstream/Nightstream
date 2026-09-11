//! Existing numeric claim schemas and C trace values. These encode observed values only.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CeClaim, Mat};
use neo_fold_clean::paper::pi_ccs;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::engines::pi_ccs_joint::ProtocolTrace;
use nightstream_fprime::{
    PI_CCS_V1_1_MATRIX_COUNT as MATRICES, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS as PUBLIC,
    PI_DEC_V1_1_CHILD_COUNT as CHILDREN,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};

type Claim = CeClaim<Commitment, F, K>;

pub fn words(values: &[K]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|value| <[u64; 2]>::from(value.to_limbs_u64()))
        .collect()
}

pub fn fields(values: &[F]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect()
}

pub fn public_words(value: &Mat<F>) -> Vec<u64> {
    (0..PUBLIC)
        .map(|column| value[(column % D, column / D)].as_canonical_u64())
        .collect()
}

pub fn claim_value(claim: &Claim, combined: bool) -> Value {
    assert_eq!((claim.c.d, claim.c.kappa, claim.m_in), (D, 22, PUBLIC));
    assert_eq!(claim.c.data.len(), 22 * D);
    assert_eq!(claim.r.len(), 28);
    assert_eq!(claim.eval_k.len(), D.next_power_of_two());
    assert_eq!(claim.eval_a.len(), MATRICES);
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

pub fn running_value(children: &[Claim]) -> Value {
    assert_eq!(children.len(), CHILDREN);
    json!([
        words(&children[0].r),
        children
            .iter()
            .map(|child| fields(&child.c.data))
            .collect::<Vec<_>>(),
        children
            .iter()
            .map(|child| public_words(&child.X))
            .collect::<Vec<_>>(),
        children
            .iter()
            .map(|child| words(&child.eval_k[..D]))
            .collect::<Vec<_>>(),
        children
            .iter()
            .map(|child| child
                .eval_a
                .iter()
                .map(|family| words(&family[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    ])
}

pub fn ccs_phase(proof: &pi_ccs::Proof, trace: &ProtocolTrace, valid: bool) -> Value {
    let public_width = PUBLIC;
    let terminal = &trace.terminal_components;
    json!([
        u64::from(valid),
        words(&trace.alpha),
        words(&[trace.gamma])[0],
        trace
            .pre_sumcheck_state
            .map(|value| value.as_canonical_u64()),
        words(&trace.round_challenges),
        trace
            .round_states
            .iter()
            .map(|state| state.map(|value| value.as_canonical_u64()))
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
            .map(|output| output
                .c
                .data
                .iter()
                .map(|value| value.as_canonical_u64())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|output| (0..public_width)
                .map(|column| output.X[(column % D, column / D)].as_canonical_u64())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|output| words(&output.eval_k[..D]))
            .collect::<Vec<_>>(),
        proof
            .outputs
            .iter()
            .map(|output| output
                .eval_a
                .iter()
                .map(|matrix| words(&matrix[..D]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        trace.outgoing_state.map(|value| value.as_canonical_u64())
    ])
}

pub fn ccs_input(fresh: &CcsClaim<Commitment, F>, running: &[Claim], proof: &pi_ccs::Proof) -> Value {
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
        running_value(running)
    ])
}
