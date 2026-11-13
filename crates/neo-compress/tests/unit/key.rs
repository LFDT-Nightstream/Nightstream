//! The reduction of the Ajtai key's part of the weights to one value of the
//! key polynomial, against key elements from an independent SHAKE128 path,
//! and proofs under the key polynomial of another seed.

use crate::support::{claim_of_seed, instances, key, layout, random_extension, statements};
use neo_ajtai::nightstream_fprime_setup::{coefficient_block, PRODUCTION_SEED};
use neo_math::{D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{fold, prefix_equality, production_element, row_point, table, weight_at, weight_table, ProverKey};
use crate::akita::SCHEDULES;
use crate::layout::LANE_VARS;
use crate::ring::{dot, Point};
use crate::sumcheck;
use crate::weights::{self, equality, powers, Use, Uses, KAPPA};
use crate::{commit, verify, Error};

/// The blocks of the light relation of the Poseidon2 hash chain, which a
/// test of `nightstream` compares with the relation.
const LIGHT_BLOCKS: usize = 74_530;

fn random_point(rng: &mut ChaCha8Rng, count: usize) -> Vec<K> {
    (0..count).map(|_| random_extension(rng)).collect()
}

fn vars_for(count: usize) -> usize {
    count.next_power_of_two().trailing_zeros() as usize
}

/// The polynomial with the evaluation table `table` at `point`.
fn value(table: &[K], point: &[K]) -> K {
    table
        .par_iter()
        .zip(equality(point, table.len()))
        .map(|(&cell, weight)| cell * weight)
        .sum()
}

/// Claims that read `counts` blocks each of a key polynomial of `blocks`
/// blocks: the key's part of the weights from `coefficient_block`, by the
/// formula of its definition, is the sum that the prover computes from the
/// folded table. The sum-check reduces it to the value of the folded table
/// at one point, which is `scale` times the value of the key polynomial at
/// that point and the row point.
fn the_reduction_gives_the_shake_value(blocks: usize, counts: &[usize], seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let block_vars = vars_for(blocks);
    let (omega, delta) = (random_extension(&mut rng), random_extension(&mut rng));
    let rho = random_point(&mut rng, LANE_VARS + block_vars);
    let lanes: [K; D] = equality(&rho[..LANE_VARS], D).try_into().expect("lanes");
    let uses = Uses {
        form: Point::new(delta).lanes(&lanes),
        claims: counts
            .iter()
            .map(|&count| Use {
                coefficient: random_extension(&mut rng),
                point: &rho[LANE_VARS..LANE_VARS + vars_for(count)],
                blocks: count,
            })
            .collect(),
    };

    let rows = powers(omega, KAPPA);
    let block = |index: usize| -> K {
        (0..KAPPA)
            .map(|row| {
                let element = coefficient_block(&PRODUCTION_SEED, row as u32, index as u64);
                let lanes: K = uses
                    .form
                    .iter()
                    .zip(element)
                    .map(|(&weight, cell)| weight * F::new(cell))
                    .sum();
                rows[row] * lanes
            })
            .sum()
    };
    let by_block: Vec<K> = (0..blocks).into_par_iter().map(block).collect();
    let direct: K = uses
        .claims
        .iter()
        .map(|claim| claim.coefficient * dot(&equality(claim.point, claim.blocks), &by_block))
        .sum();

    let table = table(blocks, production_element, F::new);
    let folded = fold(&table, omega);
    assert_eq!(weights::key_part(&uses, &folded), direct, "the prover's sum");

    // A prefix of the blocks is not a value of the folded table: the value
    // at `(β, 0, …)` reads each block of the power of two above the prefix.
    for claim in &uses.claims {
        if !claim.blocks.is_power_of_two() {
            let full = dot(&equality(claim.point, 1 << claim.point.len()), &by_block);
            assert_ne!(
                full,
                dot(&equality(claim.point, claim.blocks), &by_block),
                "{} blocks",
                claim.blocks
            );
        }
    }

    let rounds = LANE_VARS + block_vars;
    let (messages, point, folded_value) = sumcheck::prove_product(
        &mut Poseidon2Transcript::new_v1_2(),
        folded.clone(),
        weight_table(&uses),
        direct,
        rounds,
    );
    assert_eq!(messages.len(), rounds);
    let (verifier_point, claim) =
        sumcheck::verify(&mut Poseidon2Transcript::new_v1_2(), &messages, direct).expect("the rounds are consistent");
    assert_eq!(verifier_point, point);
    assert_eq!(
        claim,
        folded_value * weight_at(&uses, &point),
        "the end of the sum-check"
    );
    assert_eq!(
        weight_at(&uses, &point),
        value(&weight_table(&uses), &point),
        "the verifier's weight"
    );
    assert_eq!(folded_value, value(&folded, &point));

    let (row_point, scale) = row_point(omega).expect("row point");
    let row_values: Vec<K> = table
        .chunks(blocks * D)
        .map(|row| {
            let cells: Vec<K> = row.par_iter().map(|&cell| K::from(cell)).collect();
            value(&cells, &point)
        })
        .collect();
    assert_eq!(row_values.len(), KAPPA);
    assert_eq!(
        folded_value,
        scale * dot(&equality(&row_point, KAPPA), &row_values),
        "the folded table is the key polynomial at the row point"
    );

    assert!(matches!(
        sumcheck::verify(&mut Poseidon2Transcript::new_v1_2(), &messages, direct + K::ONE),
        Err(Error::Rejected(_))
    ));
}

#[test]
fn the_reduction_of_a_small_key_gives_the_shake_value() {
    the_reduction_gives_the_shake_value(16, &[16, 5, 11, 1], 1);
    the_reduction_gives_the_shake_value(8, &[7], 2);
}

#[test]
fn the_reduction_of_the_light_relation_gives_the_shake_value() {
    the_reduction_gives_the_shake_value(LIGHT_BLOCKS.next_power_of_two(), &[LIGHT_BLOCKS], 3);
}

/// `prefix_equality` against the table of `eq(point, j) · [j < count]`, for
/// each count and for points with fewer coordinates than the table.
#[test]
fn the_prefix_weight_is_the_extension_of_the_equality_table_of_a_prefix() {
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    for (vars, at_vars) in [(0, 0), (0, 3), (1, 1), (3, 3), (3, 5), (4, 4)] {
        let point = random_point(&mut rng, vars);
        let at = random_point(&mut rng, at_vars);
        for count in 0..=(1 << vars) + 1 {
            let mut table = equality(&point, count.min(1 << vars));
            table.resize(1 << at_vars, K::ZERO);
            assert_eq!(
                prefix_equality(&point, count, &at),
                value(&table, &at),
                "{vars} variables, count {count}, at {at_vars} variables"
            );
        }
    }
}

#[test]
fn the_powers_of_omega_are_a_scaled_equality_table() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for omega in [K::ZERO, K::ONE, K::TWO, random_extension(&mut rng)] {
        let (point, scale) = row_point(omega).expect("row point");
        let table: Vec<K> = equality(&point, 32)
            .into_iter()
            .map(|weight| scale * weight)
            .collect();
        assert_eq!(table, powers(omega, 32));
    }
    assert!(matches!(row_point(K::NEG_ONE), Err(Error::Challenge)));
}

/// The claims of a prover that holds the key polynomial of another seed are
/// about that key: their proof verifies under the verification key of that
/// polynomial and not under the key of the production seed. The honest
/// proof does not verify under the other key.
#[test]
fn a_proof_under_the_key_of_another_seed_does_not_verify() {
    let seed = [7u8; 32];
    let other = ProverKey::of_elements(&SCHEDULES[0], |row, pair| {
        [2 * pair, 2 * pair + 1].map(|block| coefficient_block(&seed, row, block))
    })
    .expect("key");
    assert_ne!(other.verification_key(), key().verification_key());
    assert_eq!(other.verification_key().blocks(), key().verification_key().blocks());

    let mut instances = instances();
    for instance in &mut instances {
        instance.claim = claim_of_seed(&seed, &instance.rows, &instance.witness, &instance.claim.r.clone());
    }
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    let prove = |key: &ProverKey| {
        commit(key, layout(&instances), &witnesses)
            .expect("commitment")
            .prove(&mut Poseidon2Transcript::new_v1_2(), &statements(&instances))
            .expect("the prover does not check its witness")
            .0
    };
    let accepts = |key: &ProverKey, proof| {
        verify(
            &mut Poseidon2Transcript::new_v1_2(),
            key.verification_key(),
            &statements(&instances),
            proof,
        )
    };
    let proof = prove(&other);
    accepts(&other, &proof).expect("the claims hold under the other key");
    assert!(matches!(accepts(key(), &proof), Err(Error::Rejected(_))));
    // The claims do not hold under the production key.
    assert!(matches!(accepts(key(), &prove(key())), Err(Error::Rejected(_))));
}
