//! The Π_RLC draw decoder of SPEC.md section 5.3 against exact integer arithmetic.

use neo_math::{D, F};
use neo_params::{nightstream_p64_k16 as profile, NeoParams};
use neo_reductions::common::{decode_pi_rlc_coefficients, read_pi_rlc_draw, PI_RLC_DRAW_LANES};
use neo_reductions::{sample_rot_rhos_n_typed, RotRing};
use neo_transcript::Poseidon2Transcript;
use num_bigint::BigUint;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn challenge_set_size() -> BigUint {
    BigUint::from(profile::CHALLENGE_ALPHABET.len()).pow(D as u32)
}

/// `N = Σ_k draw[k] · q^k`, `R = N mod 5^D`, and coefficient `h` is `⌊R / 5^h⌋ mod 5` mapped
/// through the alphabet `[−2, −1, 0, 1, 2]`.
fn reference(draw: &[u64]) -> Vec<i8> {
    let q = BigUint::from(F::ORDER_U64);
    let integer = draw
        .iter()
        .rev()
        .fold(BigUint::ZERO, |high, &lane| high * &q + BigUint::from(lane));
    let mut reduced = integer % challenge_set_size();
    (0..D)
        .map(|_| {
            let digit = u32::try_from(&reduced % 5u32).unwrap();
            reduced /= 5u32;
            profile::CHALLENGE_ALPHABET[digit as usize]
        })
        .collect()
}

fn decode(draw: [u64; PI_RLC_DRAW_LANES]) -> Vec<i8> {
    decode_pi_rlc_coefficients(&draw.map(F::from_u64)).to_vec()
}

/// The exact statistical distance between `N mod M` for a uniform `N < q^lanes` and the
/// uniform law on `[0, M)` is `r·(M − r) / (M · q^lanes)` with `r = q^lanes mod M`.
/// The result is whether that distance is below `2^−132`.
fn distance_is_below_2_pow_minus_132(lanes: u32) -> bool {
    let draws = BigUint::from(F::ORDER_U64).pow(lanes);
    let set = challenge_set_size();
    let remainder = &draws % &set;
    let numerator = &remainder * (&set - &remainder);
    (numerator << 132) < set * draws
}

#[test]
fn five_lanes_meet_the_distance_bound_and_four_do_not() {
    assert_eq!(PI_RLC_DRAW_LANES, 5);
    assert!(distance_is_below_2_pow_minus_132(PI_RLC_DRAW_LANES as u32));
    assert!(!distance_is_below_2_pow_minus_132(4));
}

#[test]
fn decoder_matches_the_integer_rule_at_the_boundaries() {
    let top = F::ORDER_U64 - 1;
    let last = PI_RLC_DRAW_LANES - 1;
    let lane = |index: usize, value: u64| {
        let mut draw = [0; PI_RLC_DRAW_LANES];
        draw[index] = value;
        draw
    };
    for draw in [
        [0; PI_RLC_DRAW_LANES],
        [top; PI_RLC_DRAW_LANES],
        lane(0, 1),
        lane(0, top),
        lane(last, 1),
        lane(last, top),
        core::array::from_fn(|index| (last - index) as u64),
    ] {
        assert_eq!(decode(draw), reference(&draw), "draw {draw:?}");
    }
    assert_eq!(decode([0; PI_RLC_DRAW_LANES]), vec![-2; D]);
    let mut one = vec![-2; D];
    one[0] = -1;
    assert_eq!(decode(lane(0, 1)), one);
}

#[test]
fn decoder_matches_the_integer_rule_on_random_draws_and_reads_each_lane() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5049_524c);
    for _ in 0..2_000 {
        let draw: [u64; PI_RLC_DRAW_LANES] = core::array::from_fn(|_| rng.next_u64() % F::ORDER_U64);
        let decoded = decode(draw);
        assert_eq!(decoded, reference(&draw), "draw {draw:?}");
        for lane in 0..PI_RLC_DRAW_LANES {
            let mut changed = draw;
            changed[lane] = (changed[lane] + 1) % F::ORDER_U64;
            assert_ne!(decode(changed), decoded, "lane {lane} of {draw:?}");
        }
    }
}

#[test]
fn sampler_reads_the_draw_after_the_block_and_permutes_once() {
    const COUNT: usize = 3;
    let params = NeoParams::nightstream_p64_k16();
    let mut initial = Poseidon2Transcript::new_v1_2();
    initial.absorb_v1_2(&[F::from_u64(11), F::from_u64(13)]);

    let mut sampled = initial.clone();
    let rhos = sample_rot_rhos_n_typed(&mut sampled, &params, &RotRing::profile(), COUNT).unwrap();

    let mut replay = initial;
    for (source, rho) in rhos.iter().enumerate() {
        replay.absorb_v1_2(&[F::from_u64(4), F::from_u64(source as u64)]);
        let entered = replay.state();
        let draw = read_pi_rlc_draw(&mut replay);
        assert_eq!(draw.as_slice(), &entered[..PI_RLC_DRAW_LANES]);
        let expected = reference(&draw.map(|lane| lane.as_canonical_u64()));
        for (row, coefficient) in expected.into_iter().enumerate() {
            assert_eq!(
                rho.as_mat()[(row, 0)],
                F::from_i8(coefficient),
                "source {source}, row {row}"
            );
        }
    }
    assert_eq!(sampled.state(), replay.state());
}
