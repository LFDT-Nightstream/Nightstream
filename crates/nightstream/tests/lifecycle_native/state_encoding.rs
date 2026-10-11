//! The state hash binds only the running parent public input. The prover and
//! the decider accept only its canonical child split.

use super::*;
use crate::folding::{CcsInstance, CcsWitness};
use crate::lifecycle::{
    pi_ccs_v1_2_prior_children, serialize_pi_ccs_v1_2_state_preimage, PiCcsV1_2PackageBridgeError, Stage1Envelope,
    VerifyError,
};

const PARENT_COORDINATES: usize = 270;

fn zero_running() -> Vec<CeClaim> {
    let claim = CeClaim {
        c: Commitment::zeros(D, 22),
        X: Mat::zero(D, PARENT_COORDINATES / D, F::ZERO),
        r: vec![K::ZERO; 28],
        eval_k: vec![K::ZERO; D],
        eval_a: vec![vec![K::ZERO; D]; 4],
        m_in: PARENT_COORDINATES,
        fold_digest: [0; 32],
        adv: None,
    };
    vec![claim; 16]
}

fn preimage(running: &[CeClaim]) -> Result<Vec<u64>, PiCcsV1_2PackageBridgeError> {
    serialize_pi_ccs_v1_2_state_preimage([F::ONE; 4], 1, [F::ZERO; 4], [F::ZERO; 4], running)
}

#[test]
fn the_state_hash_rejects_a_second_split_of_the_same_parent() {
    let mut canonical = zero_running();
    canonical[0].X[(0, 0)] = F::ONE;
    // 1 = -1 + 2 · 1 is a second signed binary split of the same parent.
    let mut other = zero_running();
    other[0].X[(0, 0)] = F::NEG_ONE;
    other[1].X[(0, 0)] = F::ONE;

    preimage(&canonical).unwrap();
    assert!(matches!(
        preimage(&other),
        Err(PiCcsV1_2PackageBridgeError::NonCanonicalChildren)
    ));
    assert!(matches!(
        pi_ccs_v1_2_prior_children(&other),
        Err(PiCcsV1_2PackageBridgeError::NonCanonicalChildren)
    ));
}

#[test]
fn the_state_hash_rejects_a_child_outside_the_digit_set() {
    let mut running = zero_running();
    running[0].X[(0, 0)] = F::from_u64(2);
    assert!(matches!(
        preimage(&running),
        Err(PiCcsV1_2PackageBridgeError::NonCanonicalChildren)
    ));
}

/// The terminal decider must reject a second split before any commitment or
/// relation check. With canonical children the same envelope reaches the
/// later public-input check, so the first rejection comes from the split.
#[test]
fn verify_rejects_a_second_split_of_the_running_parent() {
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let source = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    let binding = source.production_verifier_binding().unwrap();
    let package =
        PreparedLifecycle::from_package(source.into(), binding, crate::engine::Backend::Optimized, 114).unwrap();
    let structure = package.structure();
    let params = Params::for_ccs_shape(
        structure.domain_rows(),
        structure.m,
        structure.t(),
        structure.max_degree(),
        114,
    )
    .unwrap();
    let blocks = structure.m.div_ceil(D);
    let state = Stage1State::new(1, [F::ZERO; 4], [F::ONE; 4]);
    // Each child digit sits in both the claim and its witness, so the
    // public-projection check passes.
    let envelope = |digits: &[(usize, F)]| {
        let mut running = RunningInstance::canonical_zero(&params, structure, PARENT_COORDINATES).unwrap();
        for &(child, digit) in digits {
            let mut positive = vec![0; blocks];
            let mut negative = vec![0; blocks];
            if digit == F::ONE {
                positive[0] = 1;
            } else {
                negative[0] = 1;
            }
            running.witnesses[child] =
                Mat::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap();
            let mut x = Mat::zero(D, PARENT_COORDINATES / D, F::ZERO);
            x[(0, 0)] = digit;
            running.claims[child].X = x;
        }
        let fresh = CcsInstance {
            claim: CcsClaim {
                c: Commitment::zeros(D, params.kappa() as usize),
                x: vec![F::ZERO; PARENT_COORDINATES],
                m_in: PARENT_COORDINATES,
                adv: None,
            },
            witness: CcsWitness {
                w: vec![],
                Z: Mat::virtual_constant(D, blocks, F::ZERO),
            },
        };
        Stage1Envelope::from_parts(state.clone(), running, fresh)
    };

    let canonical = package.verify(&state, &envelope(&[(0, F::ONE)]));
    assert!(
        matches!(
            canonical,
            Err(VerifyError::Fresh(
                "public input differs from the recomputed terminal state hash"
            ))
        ),
        "{canonical:?}"
    );
    let second = package.verify(&state, &envelope(&[(0, F::NEG_ONE), (1, F::ONE)]));
    assert!(
        matches!(
            second,
            Err(VerifyError::StateHash(
                PiCcsV1_2PackageBridgeError::NonCanonicalChildren
            ))
        ),
        "{second:?}"
    );
}

#[test]
fn prior_children_are_child_major_digits() {
    let mut running = zero_running();
    // Lean column 54 is ring coefficient 0 of public column 1.
    running[0].X[(0, 1)] = F::NEG_ONE;
    running[2].X[(0, 1)] = F::NEG_ONE;
    running[3].X[(1, 0)] = F::ONE;
    let words = pi_ccs_v1_2_prior_children(&running).unwrap();
    assert_eq!(words.len(), 16 * PARENT_COORDINATES);
    assert_eq!(words[54], F::NEG_ONE.as_canonical_u64());
    assert_eq!(words[2 * PARENT_COORDINATES + 54], F::NEG_ONE.as_canonical_u64());
    assert_eq!(words[3 * PARENT_COORDINATES + 1], 1);
    assert_eq!(
        words.iter().filter(|word| **word != 0).count(),
        3,
        "only the three set digits are nonzero"
    );
}
