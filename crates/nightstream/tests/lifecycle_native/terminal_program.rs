//! Parts of the generic final program against their native counterparts:
//! the PiRLC challenge decoder and the terminal statement (canonical split,
//! state hash, encoded public input).

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{from_complex, D, F, K};
use neo_reductions::common::decode_pi_rlc_coefficients;
use neo_spartan::{Backend, Native};
use nightstream_fprime::{
    PI_CCS_V1_1_MATRIX_COUNT, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::k::Kw;
use super::pi_rlc;
use super::state::{self, State};
use super::words::{Evaluations, FinalWords, COLUMNS, ROWS};
use crate::folding::CeClaim;
use crate::lifecycle::{encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage};

/// Deterministic pseudo-random words; the tests need variety, not secrecy.
fn word(seed: u64, index: u64) -> u64 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    (x ^ (x >> 29)) % F::ORDER_U64
}

fn field(seed: u64, index: u64) -> F {
    F::from_u64(word(seed, index))
}

fn ext(seed: u64, index: u64) -> K {
    from_complex(field(seed, 2 * index), field(seed, 2 * index + 1))
}

#[test]
fn rho_decoder_matches_the_sampler() {
    let p = F::ORDER_U64 - 1;
    let mut digests: Vec<[u64; 4]> = (0..24)
        .map(|seed| std::array::from_fn(|l| word(seed, l as u64)))
        .collect();
    digests.extend([[0; 4], [p; 4], [p, 0, p, 0], [1, 0, 0, p]]);
    for digest in digests {
        let b = &mut Native;
        let words = digest.map(|w| b.private(w));
        let ours = pi_rlc::decode(b, words).unwrap();
        let theirs = decode_pi_rlc_coefficients(&digest.map(F::from_u64));
        for (lane, (&value, &expected)) in ours.iter().zip(&theirs).enumerate() {
            let expected = b.constant(F::from_i64(i64::from(expected)).as_canonical_u64());
            assert!(
                b.assert_equal(value, expected, "digit").is_ok(),
                "{digest:?} lane {lane}"
            );
        }
    }
}

/// The balanced base-2 digits of `value` over sixteen children.
fn split(value: i64) -> [F; PI_DEC_V1_1_CHILD_COUNT] {
    let sign = if value < 0 { -F::ONE } else { F::ONE };
    std::array::from_fn(|j| {
        if (value.unsigned_abs() >> j) & 1 == 1 {
            sign
        } else {
            F::ZERO
        }
    })
}

fn running(seed: u64) -> Vec<CeClaim> {
    let parents: Vec<i64> = (0..PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS as u64)
        .map(|j| (word(seed, j) % (1 << 17)) as i64 - (1 << 16) + 1)
        .collect();
    let digits: Vec<[F; PI_DEC_V1_1_CHILD_COUNT]> = parents.iter().map(|&value| split(value)).collect();
    let point: Vec<K> = (0..PI_CCS_V1_1_ROUND_COUNT as u64)
        .map(|t| ext(seed + 1, t))
        .collect();
    let padded = |values: Vec<K>| {
        let mut values = values;
        values.resize(D.next_power_of_two(), K::ZERO);
        values
    };
    (0..PI_DEC_V1_1_CHILD_COUNT)
        .map(|child| {
            let s = seed + 10 * (child as u64 + 2);
            let mut c = Commitment::zeros(D, ROWS);
            for (i, value) in c.data.iter_mut().enumerate() {
                *value = field(s, i as u64);
            }
            let x: Vec<F> = (0..D * COLUMNS)
                .map(|i| digits[(i % COLUMNS) * D + i / COLUMNS][child])
                .collect();
            CeClaim {
                c,
                X: Mat::from_row_major(D, COLUMNS, x),
                r: point.clone(),
                eval_k: padded((0..D as u64).map(|l| ext(s + 1, l)).collect()),
                eval_a: (0..PI_CCS_V1_1_MATRIX_COUNT as u64)
                    .map(|m| padded((0..D as u64).map(|l| ext(s + 2 + m, l)).collect()))
                    .collect(),
                m_in: PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS,
                fold_digest: [0; 32],
                adv: None,
            }
        })
        .collect()
}

fn words_of<B: Backend>(b: &mut B, claims: &[CeClaim]) -> FinalWords<B> {
    let w = |b: &mut B, value: F| b.private(value.as_canonical_u64());
    let kw = |b: &mut B, value: K| -> Kw<B> {
        let [re, im] = neo_math::KExtensions::as_coeffs(&value);
        [w(b, re), w(b, im)]
    };
    FinalWords {
        commitments: claims
            .iter()
            .map(|claim| {
                (0..ROWS)
                    .map(|row| std::array::from_fn(|lane| w(b, claim.c.col(row)[lane])))
                    .collect()
            })
            .collect(),
        digits: claims
            .iter()
            .map(|claim| {
                (0..PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS)
                    .map(|j| w(b, claim.X[(j % D, j / D)]))
                    .collect()
            })
            .collect(),
        evaluations: claims
            .iter()
            .map(|claim| Evaluations {
                eval_k: std::array::from_fn(|lane| kw(b, claim.eval_k[lane])),
                eval_a: claim
                    .eval_a
                    .iter()
                    .map(|values| std::array::from_fn(|lane| kw(b, values[lane])))
                    .collect(),
            })
            .collect(),
        point: claims[0].r.iter().map(|&value| kw(b, value)).collect(),
        fresh: Vec::new(),
        rounds: Vec::new(),
        outputs: Vec::new(),
    }
}

#[test]
fn statement_matches_the_native_hash_and_encoding() {
    let claims = running(5);
    let context = [11, 12, 13, 14];
    let (iteration, z0, current) = (9u64, [1, 2, 3, 4].map(F::from_u64), [5, 6, 7, 8].map(F::from_u64));
    let preimage =
        serialize_pi_ccs_v1_1_state_preimage(context.map(F::from_u64), iteration, z0, current, &claims).unwrap();
    let digest = pi_ccs_v1_1_state_hash(&preimage).unwrap();
    let encoded = encode_pi_ccs_v1_1_public_input(digest).unwrap();

    let b = &mut Native;
    let words = words_of(b, &claims);
    let state = State {
        iteration: b.public(iteration),
        z0: z0.map(|v| b.public(v.as_canonical_u64())),
        current: current.map(|v| b.public(v.as_canonical_u64())),
    };
    let parent = state::canonical_parent(b, &words.digits).unwrap();
    let ours = state::state_hash(b, context, &state, &words, &parent);
    for (value, expected) in ours.into_iter().zip(digest) {
        let expected = b.constant(expected);
        b.assert_equal(value, expected, "state digest").unwrap();
    }
    let public = state::encode(b, ours).unwrap();
    for (value, expected) in public.into_iter().zip(encoded) {
        let expected = b.constant(expected);
        b.assert_equal(value, expected, "encoded public input")
            .unwrap();
    }

    // Controls: mixed signs in one coordinate, and a changed point.
    let mut mixed = claims.clone();
    let (lane, column) = (0..D * COLUMNS)
        .map(|i| (i % D, i / D))
        .find(|&(lane, column)| mixed[0].X[(lane, column)] != F::ZERO)
        .unwrap();
    let value = mixed[0].X[(lane, column)];
    let other = mixed
        .iter()
        .position(|claim| claim.X[(lane, column)] == F::ZERO)
        .unwrap();
    mixed[other].X[(lane, column)] = -value;
    let words = words_of(b, &mixed);
    assert!(state::canonical_parent(b, &words.digits).is_err());
    let mut moved = claims;
    for claim in &mut moved {
        claim.r[0] += K::ONE;
    }
    let words = words_of(b, &moved);
    let parent = state::canonical_parent(b, &words.digits).unwrap();
    let changed = state::state_hash(b, context, &state, &words, &parent);
    let expected = b.constant(digest[0]);
    assert!(b
        .assert_equal(changed[0], expected, "state digest")
        .is_err());
}
