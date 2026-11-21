use crate::support::{claim_of_seed, instances, key, layout, random_extension, statements};
use neo_ajtai::nightstream_fprime_setup::{coefficient_block, ExpandedKey, PRODUCTION_SEED};
use neo_math::{D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::sync::Arc;

use akita_serialization::AkitaSerialize;
use akita_types::{AkitaSetupSeed, SetupPrefixVerifierRegistry};

use super::{block_weights, fold, prefix_equality, row_point, weight_at, ProverKey, VerificationKey};
use crate::akita::verifier::{commitment, setup_prefixes};
use crate::akita::{Schedules, SCHEDULES};
use crate::layout::LANE_VARS;
use crate::ring::{dot, Point};
use crate::sumcheck;
use crate::weights::{self, equality, powers, Use, Uses, KAPPA};
use crate::{commit, verify, Error};

const LIGHT_BLOCKS: usize = 74_146;

fn random_point(rng: &mut ChaCha8Rng, count: usize) -> Vec<K> {
    (0..count).map(|_| random_extension(rng)).collect()
}

fn vars_for(count: usize) -> usize {
    count.next_power_of_two().trailing_zeros() as usize
}

fn value(table: &[K], point: &[K]) -> K {
    table
        .par_iter()
        .zip(equality(point, table.len()))
        .map(|(&cell, weight)| cell * weight)
        .sum()
}

/// `H` of SPEC.md Algorithm 9, step 10, as a table, lane first.
fn weight_table(uses: &Uses) -> Vec<K> {
    block_weights(uses)
        .iter()
        .flat_map(|&block| uses.form.map(|lane| lane * block))
        .collect()
}

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

    let key = ExpandedKey::production(blocks as u64);
    let folded = fold(&key, omega, blocks);
    let blocks_for_key = block_weights(&uses);
    assert_eq!(
        weights::key_part(&uses.form, &blocks_for_key, &folded),
        direct,
        "the prover's sum"
    );

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
        folded[..(D * counts.iter().copied().max().unwrap())].to_vec(),
        &uses.form,
        blocks_for_key,
        direct,
        rounds,
        |lanes, values| super::fold_remaining_blocks(&key, omega, lanes, values),
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
    let row_values: Vec<K> = key
        .words()
        .chunks(blocks * D)
        .map(|row| {
            let cells: Vec<K> = row.par_iter().map(|&cell| K::from(F::new(cell))).collect();
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
    the_reduction_gives_the_shake_value(32, &[1], 4);
}

#[test]
fn the_reduction_of_the_light_relation_gives_the_shake_value() {
    the_reduction_gives_the_shake_value(LIGHT_BLOCKS.next_power_of_two(), &[LIGHT_BLOCKS], 3);
}

#[test]
fn the_fold_is_the_sum_of_the_key_rows_times_the_powers_of_omega() {
    let mut rng = ChaCha8Rng::seed_from_u64(6);
    let omega = random_extension(&mut rng);
    let rows = powers(omega, KAPPA);
    for blocks in [2, 18, 34] {
        let key = ExpandedKey::of_pairs(blocks, |row, pair| {
            let word = |lane: usize| match (u64::from(row) + pair + lane as u64) % 5 {
                0 => u64::MAX - lane as u64,
                1 => 0,
                _ => rng_word(row, pair, lane),
            };
            [std::array::from_fn(word), std::array::from_fn(|lane| word(lane + D))]
        });
        let cells = blocks as usize * D;
        let expected: Vec<K> = (0..cells)
            .map(|cell| {
                (0..KAPPA)
                    .map(|row| rows[row] * F::new(key.words()[row * cells + cell]))
                    .sum()
            })
            .collect();
        assert!(fold(&key, omega, blocks as usize) == expected, "{blocks} blocks");
    }
}

fn rng_word(row: u32, pair: u64, lane: usize) -> u64 {
    let mut rng = ChaCha8Rng::seed_from_u64((u64::from(row) << 48) ^ (pair << 8) ^ lane as u64);
    rand::RngCore::next_u64(&mut rng)
}

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
    let quarter = (F::ORDER_U64 - 1) / 4;
    let half = F::ORDER_U64.div_ceil(2);
    let root = (2..)
        .map(|x: u64| {
            random_extension(&mut ChaCha8Rng::seed_from_u64(x))
                .exp_u64(quarter)
                .exp_u64(half)
        })
        .find(|root| root.exp_u64(4) == K::NEG_ONE)
        .expect("a root of order 8");
    for power in 1..8 {
        assert!(
            matches!(row_point(root.exp_u64(power)), Err(Error::Challenge)),
            "power {power}"
        );
    }
}

#[test]
fn a_proof_under_the_key_of_another_seed_does_not_verify() {
    let seed = [7u8; 32];
    let other = other_key(&seed);
    assert_ne!(other.verification_key(), key().verification_key());
    assert_eq!(other.verification_key().blocks(), key().verification_key().blocks());
    assert!(matches!(
        VerificationKey::from_bytes(&other.verification_key().to_bytes()),
        Err(Error::Key(_))
    ));

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
    assert!(matches!(accepts(key(), &prove(key())), Err(Error::Rejected(_))));
}

#[test]
fn the_rows_of_another_key_under_the_honest_commitment_do_not_verify() {
    let seed = [7u8; 32];
    let other = other_key(&seed);
    let mut forged = ProverKey::new(key().verification_key().blocks()).expect("key");
    assert_eq!(forged.verification_key(), key().verification_key());
    forged.key = other.key.clone();

    let mut instances = instances();
    for instance in &mut instances {
        instance.claim = claim_of_seed(&seed, &instance.rows, &instance.witness, &instance.claim.r.clone());
    }
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    let proof = commit(&forged, layout(&instances), &witnesses)
        .expect("commitment")
        .prove(&mut Poseidon2Transcript::new_v1_2(), &statements(&instances));
    match proof {
        Ok((proof, _)) => {
            let result = verify(
                &mut Poseidon2Transcript::new_v1_2(),
                key().verification_key(),
                &statements(&instances),
                &proof,
            );
            assert!(matches!(result, Err(Error::Rejected(_))), "{result:?}");
        }
        Err(error) => assert!(matches!(error, Error::Akita(_)), "{error}"),
    }
}

fn other_key(seed: &[u8; 32]) -> ProverKey {
    let blocks = SCHEDULES[0].blocks() as u64;
    let key = ExpandedKey::of_pairs(blocks, |row, pair| {
        [2 * pair, 2 * pair + 1].map(|block| coefficient_block(seed, row, block))
    });
    ProverKey::of_key(&SCHEDULES[0], Arc::new(key)).expect("key")
}

fn derive(schedules: &'static Schedules) -> VerificationKey {
    ProverKey::of_key(schedules, ExpandedKey::production(schedules.blocks() as u64))
        .expect("key")
        .verification
}

fn the_pinned_key_is_the_key_of_the_public_seeds(schedules: &'static Schedules) {
    let key = derive(schedules);
    assert!(
        key.to_bytes() == schedules.key,
        "artifacts/key{}.vk is not the derived key",
        schedules.key_vars
    );
    assert_eq!(VerificationKey::from_bytes(schedules.key).expect("pinned key"), key);
    assert_eq!(VerificationKey::new(schedules.blocks()).expect("pinned key"), key);
}

#[test]
fn the_small_pinned_key_is_the_key_of_the_public_seeds() {
    the_pinned_key_is_the_key_of_the_public_seeds(&SCHEDULES[0]);
    assert_eq!(key().verification_key().to_bytes(), SCHEDULES[0].key);
}

#[test]
#[ignore = "6 GB: the key polynomial of 2^19 blocks."]
fn the_large_pinned_key_is_the_key_of_the_public_seeds() {
    the_pinned_key_is_the_key_of_the_public_seeds(&SCHEDULES[1]);
}

fn with_parts(key: &VerificationKey, commitment: &[u8], setup_prefixes: &[u8]) -> Vec<u8> {
    VerificationKey::of_parts(key.schedules, commitment.to_vec(), setup_prefixes.to_vec()).to_bytes()
}

#[test]
fn only_a_pinned_key_decodes() {
    for schedules in &SCHEDULES {
        let key = VerificationKey::from_bytes(schedules.key).expect("pinned key");
        assert_eq!(key.to_bytes(), schedules.key);
        let rejected = |bytes: &[u8]| matches!(VerificationKey::from_bytes(bytes), Err(Error::Key(_)));

        let mut well_formed = 0;
        for at in 0..key.commitment.len() {
            let mut changed = key.commitment.clone();
            changed[at] ^= 1;
            well_formed += usize::from(commitment(&changed).is_ok());
            assert!(
                rejected(&with_parts(&key, &changed, &key.setup_prefixes)),
                "commitment byte {at}"
            );
        }
        assert!(well_formed > 0, "no changed commitment is well formed");

        let registry = setup_prefixes(&key.setup_prefixes).expect("registry");
        let mut other = SetupPrefixVerifierRegistry::new(AkitaSetupSeed::shake256_paged_v1([9; 32]));
        for (_, slot) in registry.iter() {
            other.insert(slot.clone()).expect("slot");
        }
        let mut bytes = Vec::new();
        other.serialize_compressed(&mut bytes).expect("bytes");
        assert!(rejected(&with_parts(&key, &key.commitment, &bytes)), "another seed");

        let mut well_formed = 0;
        for at in 0..key.setup_prefixes.len() {
            let mut changed = key.setup_prefixes.clone();
            changed[at] ^= 1;
            well_formed += usize::from(setup_prefixes(&changed).is_ok());
            assert!(
                rejected(&with_parts(&key, &key.commitment, &changed)),
                "setup prefix byte {at}"
            );
        }
        if !registry.is_empty() {
            assert!(well_formed > 0, "no changed registry is well formed");
        }
    }
}

#[test]
fn circuits_that_fit_one_schedule_share_its_live_prover_key() {
    let a = ProverKey::shared(1).expect("small key");
    let b = ProverKey::shared(a.verification_key().blocks()).expect("same schedule");
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert_eq!(a.verification_key(), &VerificationKey::new(1).unwrap());
}
