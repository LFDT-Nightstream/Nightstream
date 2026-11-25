use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::F;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::*;
use crate::layout::LOOKUPS;
use crate::support::{random_extension, Rows, KAPPA, MATRICES};
use crate::weights::Challenges;
use crate::{CeClaim, Norm, Statement};

#[test]
fn packed_lane_moments_equal_dense_production_weights_through_six_binds() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x19b4);
    let blocks = BLOCKS_PER_TASK + 3;
    let padded = blocks.next_power_of_two();
    let rows = Rows::random(&mut rng, 28, blocks);
    let r: Vec<K> = (0..LANE_VARS + padded.ilog2() as usize)
        .map(|_| random_extension(&mut rng))
        .collect();
    let claim = CeClaim {
        c: Commitment::new(KAPPA, vec![F::ZERO; KAPPA * D]).unwrap(),
        X: Mat::virtual_constant(D, 2, F::ZERO),
        r,
        eval_k: vec![K::ZERO; D],
        eval_a: vec![vec![K::ZERO; D]; MATRICES],
        m_in: 2 * D,
        fold_digest: [0; 32],
        adv: None,
    };
    let statement = Statement {
        claim: &claim,
        rows: &rows,
        norm: Norm::BoundByCaller,
    };
    let challenges = Challenges {
        taus: [F::ZERO; LOOKUPS],
        omega: random_extension(&mut rng),
        delta: random_extension(&mut rng),
        gamma: random_extension(&mut rng),
        mu: random_extension(&mut rng),
        psi: [K::ZERO; LOOKUPS],
        xi: Vec::new(),
        lambda: Vec::new(),
        beta: Vec::new(),
    };
    let key: Vec<K> = (0..blocks * D)
        .map(|_| random_extension(&mut rng))
        .collect();
    let positive: Vec<u64> = (0..blocks).map(|_| rng.next_u64()).collect();
    let negative: Vec<u64> = positive.iter().map(|&p| rng.next_u64() & !p).collect();
    let masks = Masks {
        positive: &positive,
        negative: &negative,
    };
    let source = LaneSource::new(&statement, &challenges, blocks).unwrap();
    let point = Point::new(challenges.delta);

    let mut witness = vec![K::ZERO; padded * D];
    let mut weights = vec![K::ZERO; padded * D];
    let mut columns = vec![K::ZERO; COLUMNS_PER_TASK];
    for (task, weights) in weights[..blocks * D]
        .chunks_mut(COLUMNS_PER_TASK)
        .enumerate()
    {
        let columns = &mut columns[..weights.len()];
        source.columns(task, columns);
        for (local, (weights, columns)) in weights
            .chunks_exact_mut(D)
            .zip(columns.chunks_exact(D))
            .enumerate()
        {
            let block = task * BLOCKS_PER_TASK + local;
            let element = source.element(columns, &key[block * D..][..D]);
            let element = std::array::from_fn(|lane| element[lane] + source.blocks[block] * source.lanes[lane]);
            for (lane, (weight, value)) in weights.iter_mut().zip(point.lanes(&element)).enumerate() {
                let public = source
                    .public()
                    .get(block * D + lane)
                    .copied()
                    .unwrap_or(K::ZERO);
                *weight = (value + public) * challenges.mu;
                witness[block * D + lane] =
                    K::from_u64(positive[block] >> lane & 1) - K::from_u64(negative[block] >> lane & 1);
            }
        }
    }

    let mut start = LaneStart::new(masks, &key, source, padded, challenges.delta);
    let rho: Vec<K> = (0..LANE_VARS).map(|_| random_extension(&mut rng)).collect();
    let bind = |values: &[K], at: K| -> Vec<K> {
        values
            .chunks_exact(2)
            .map(|pair| (K::ONE - at) * pair[0] + at * pair[1])
            .collect()
    };
    for &coordinate in &rho {
        let expected = [K::ZERO, K::TWO].map(|sample| dot(&bind(&witness, sample), &bind(&weights, sample)));
        assert_eq!(start.round(), [expected[0], expected[1], K::ZERO]);
        start.bind(coordinate);
        witness = bind(&witness, coordinate);
        weights = bind(&weights, coordinate);
    }
    let tail = [
        random_extension(&mut rng),
        random_extension(&mut rng),
        random_extension(&mut rng),
    ];
    witness.extend(tail);
    weights.extend(tail);
    assert_eq!(start.materialize(&rho, &tail, &tail), (witness, weights));
}
