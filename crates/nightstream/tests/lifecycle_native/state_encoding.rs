//! The state hash binds only the running parent public input. The prover and
//! the decider accept only its canonical child split.

use crate::folding::CeClaim;
use crate::lifecycle::{
    check_pi_ccs_v1_1_canonical_children, pi_ccs_v1_1_prior_children, serialize_pi_ccs_v1_1_state_preimage,
};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

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

fn preimage(running: &[CeClaim]) -> Vec<u64> {
    serialize_pi_ccs_v1_1_state_preimage([F::ONE; 4], 1, [F::ZERO; 4], [F::ZERO; 4], running).unwrap()
}

#[test]
fn the_decider_rejects_a_second_split_of_the_same_parent() {
    let mut canonical = zero_running();
    canonical[0].X[(0, 0)] = F::ONE;
    // 1 = -1 + 2 · 1 is a second signed binary split of the same parent.
    let mut other = zero_running();
    other[0].X[(0, 0)] = F::NEG_ONE;
    other[1].X[(0, 0)] = F::ONE;

    assert_eq!(preimage(&canonical), preimage(&other), "the hash binds only the parent");
    check_pi_ccs_v1_1_canonical_children(&canonical).unwrap();
    assert!(check_pi_ccs_v1_1_canonical_children(&other).is_err());
    assert!(pi_ccs_v1_1_prior_children(&other).is_err());
}

#[test]
fn the_decider_rejects_a_parent_beyond_the_split_bound() {
    let mut running = zero_running();
    running[0].X[(0, 0)] = F::from_u64(1 << 16);
    assert!(check_pi_ccs_v1_1_canonical_children(&running).is_err());
}

#[test]
fn prior_children_are_child_major_digits_then_parent_signs() {
    let mut running = zero_running();
    // Lean column 54 is ring coefficient 0 of public column 1.
    running[0].X[(0, 1)] = F::NEG_ONE;
    running[2].X[(0, 1)] = F::NEG_ONE;
    running[3].X[(1, 0)] = F::ONE;
    let words = pi_ccs_v1_1_prior_children(&running).unwrap();
    assert_eq!(words.len(), 17 * PARENT_COORDINATES);
    assert_eq!(words[54], F::NEG_ONE.as_canonical_u64());
    assert_eq!(words[2 * PARENT_COORDINATES + 54], F::NEG_ONE.as_canonical_u64());
    assert_eq!(words[3 * PARENT_COORDINATES + 1], 1);
    let signs = &words[16 * PARENT_COORDINATES..];
    assert_eq!(signs[54], 1, "the parent -5 is negative");
    assert_eq!(signs[1], 0, "the parent 8 is nonnegative");
    assert_eq!(signs.iter().sum::<u64>(), 1);
}
