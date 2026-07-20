use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::{
    delayed_beta_power_selector, delayed_claimed_initial_sum, delayed_terminal_rhs, DelayedProjectionChallenges,
    DelayedProjectionConfig, DelayedProjectionInput,
};
use p3_field::PrimeCharacteristicRing;

fn claim(y_zcol: Vec<K>) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment::zeros(D, 1),
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::ZERO],
        s_col: vec![K::ZERO],
        y_ring: Vec::new(),
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol,
        m_in: 1,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
        adv: None,
    }
}

#[test]
fn beta_power_selector_matches_boolean_exponents() {
    let beta = K::from(F::from_u64(7));
    for lane in 0..64usize {
        let point: Vec<K> = (0..6)
            .map(|bit| if ((lane >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
            .collect();
        let expected = (0..lane).fold(K::ONE, |power, _| power * beta);
        assert_eq!(delayed_beta_power_selector(beta, &point), expected);
    }
}

#[test]
fn honest_cube_sum_and_terminal_specialization_match_parent_projection() {
    let beta = K::from(F::from_u64(3));
    let weight = K::from(F::from_u64(11));
    let mut child0 = vec![K::ZERO; 64];
    let mut child1 = vec![K::ZERO; 64];
    child0[2] = K::from(F::from_u64(5));
    child1[2] = K::from(F::from_u64(7));
    let radix = 2u32;
    let mut parent = vec![K::ZERO; 64];
    parent[2] = child0[2] + K::from(F::from_u64(radix as u64)) * child1[2];
    let old_s = [K::ONE];
    let config = DelayedProjectionConfig {
        input: DelayedProjectionInput {
            s_col: &old_s,
            y_zcol: &parent,
        },
        challenges: DelayedProjectionChallenges {
            producer_beta: beta,
            batch_weight: weight,
        },
    };
    let outputs = [claim(vec![K::ZERO; 64]), claim(child0), claim(child1)];
    let lane_two = [K::ZERO, K::ONE, K::ZERO, K::ZERO, K::ZERO, K::ZERO];
    let terminal =
        delayed_terminal_rhs(config, &old_s, &lane_two, &outputs, 1, radix).expect("well-shaped delayed terminal");
    assert_eq!(terminal, delayed_claimed_initial_sum(config));
}
