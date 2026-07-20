use neo_ccs::{CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::oracle::{
    BlockLaneNcChallenges, BlockLaneNcOracle, BlockLaneNcPending, BLOCK_LANE_NC_BLOCK_VARIABLES,
    BLOCK_LANE_NC_LANE_VARIABLES,
};
use neo_reductions::sumcheck::{poly_eval_k, RoundOracle};
use p3_field::PrimeCharacteristicRing;

fn identity_left(rows: usize, columns: usize) -> Mat<F> {
    let mut matrix = Mat::zero(rows, columns, F::ZERO);
    for index in 0..rows.min(columns) {
        matrix.set(index, index, F::ONE);
    }
    matrix
}

fn fixture() -> (
    CcsStructure<F>,
    Vec<CcsWitness<F>>,
    Vec<Mat<F>>,
    BlockLaneNcChallenges,
    BlockLaneNcPending,
) {
    let logical_width = 2 * D;
    let structure = CcsStructure::new(vec![identity_left(D, logical_width)], SparsePoly::new(1, Vec::new()))
        .expect("small CCS structure");

    let mut fresh_z = Mat::zero(D, 2, F::ZERO);
    fresh_z.set(3, 1, F::ONE);
    let fresh = vec![CcsWitness {
        w: vec![F::ZERO; logical_width],
        Z: fresh_z,
    }];

    let mut child_zero = Mat::zero(D, 2, F::ZERO);
    child_zero.set(0, 0, F::ONE);
    let mut child_one = Mat::zero(D, 2, F::ZERO);
    child_one.set(1, 0, F::ONE);
    let running = vec![child_zero, child_one];

    let challenges = BlockLaneNcChallenges {
        beta_block: std::array::from_fn(|index| K::from(F::from_u64(7 + index as u64))),
        beta_lane: std::array::from_fn(|index| K::from(F::from_u64(31 + index as u64))),
        gamma: K::from(F::from_u64(71)),
        producer_beta: K::from(F::from_u64(5)),
        batch_weight: K::from(F::from_u64(11)),
    };
    let mut parent_y = [K::ZERO; D];
    parent_y[0] = K::ONE;
    parent_y[1] = K::from(F::from_u64(2));
    let pending = BlockLaneNcPending {
        old_block: [K::ZERO; BLOCK_LANE_NC_BLOCK_VARIABLES],
        parent_y,
    };
    (structure, fresh, running, challenges, pending)
}

#[test]
fn raw_block_lane_oracle_closes_all_rounds_and_computes_padding() {
    let (structure, fresh, running, challenges, pending) = fixture();
    let mut oracle = BlockLaneNcOracle::new(&structure, &fresh, &running, challenges, Some(pending))
        .expect("well-shaped raw block-lane oracle");
    let mut claimed = oracle.initial_sum();

    for round in 0..oracle.num_rounds() {
        let coefficients = oracle.round_coefficients();
        assert_eq!(
            coefficients[0] + poly_eval_k(&coefficients, K::ONE),
            claimed,
            "sumcheck invariant at round {round}"
        );
        let challenge = K::from(F::from_u64(101 + round as u64));
        claimed = poly_eval_k(&coefficients, challenge);
        oracle.fold(challenge);

        if round + 1 == BLOCK_LANE_NC_BLOCK_VARIABLES {
            let rows = oracle.block_projected_source_rows();
            assert_eq!(rows.len(), fresh.len() + running.len());
            for row in rows {
                assert!(row[D..].iter().all(|value| *value == K::ZERO));
            }
        }
    }

    assert_eq!(
        oracle.num_rounds(),
        BLOCK_LANE_NC_BLOCK_VARIABLES + BLOCK_LANE_NC_LANE_VARIABLES
    );
    assert_eq!(claimed, oracle.finalized_value());
    assert_eq!(oracle.finalized_running_values().len(), running.len());
}

#[test]
fn raw_child_mutation_breaks_the_delayed_initial_claim() {
    let (structure, fresh, mut running, challenges, pending) = fixture();
    running[0].set(0, 0, F::ZERO);
    let oracle = BlockLaneNcOracle::new(&structure, &fresh, &running, challenges, Some(pending))
        .expect("well-shaped mutated raw oracle");
    let coefficients = oracle.round_coefficients();

    assert_ne!(
        coefficients[0] + poly_eval_k(&coefficients, K::ONE),
        oracle.initial_sum(),
        "the delayed source must be the raw running matrix, not a carried evaluation"
    );
}

#[test]
fn zero_batch_weight_is_the_explicit_mixing_root_branch() {
    let (structure, fresh, mut running, mut challenges, pending) = fixture();
    running[0].set(0, 0, F::ZERO);
    challenges.batch_weight = K::ZERO;
    let oracle = BlockLaneNcOracle::new(&structure, &fresh, &running, challenges, Some(pending))
        .expect("well-shaped zero-weight oracle");
    let coefficients = oracle.round_coefficients();

    assert_eq!(oracle.initial_sum(), K::ZERO);
    assert_eq!(coefficients[0] + poly_eval_k(&coefficients, K::ONE), K::ZERO);
}
