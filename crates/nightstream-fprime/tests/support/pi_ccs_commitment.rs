//! Independent indexed SHAKE128 and signed-integer Ajtai evaluation.
//! Spec.AjtaiSetupV1 fixes the element input, the lane bytes, and the 256-bit
//! reduction. No production hash, key expander, or commitment routine is called.

use rayon::prelude::*;
use serde_json::{json, Value};

const DEGREE: usize = 54;
const MODULUS: u128 = 18_446_744_069_414_584_321;
const RATE: usize = 168;
pub const SETUP_ID: &[u8] = b"nightstream-ajtai-shake128-wide256-v1";

/// FIPS 202 example: the first 32 bytes of SHAKE128 of the empty message.
const SHAKE128_EMPTY: [u8; 32] = [
    0x7f, 0x9c, 0x2b, 0xa4, 0xe8, 0x8f, 0x82, 0x7d, 0x61, 0x60, 0x45, 0x50, 0x76, 0x05, 0x85, 0x3e, 0xd7, 0x3b, 0x80,
    0x93, 0xf6, 0xef, 0xbc, 0x88, 0xeb, 0x1a, 0x6e, 0xac, 0xfa, 0x66, 0xef, 0x26,
];

/// Round constants of step iota (FIPS 202, Section 3.2.5).
const ROUND_CONSTANTS: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rotation offsets of step rho (FIPS 202, Table 2), at lane `x + 5y`.
const RHO: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

fn keccak_f(state: &mut [u64; 25]) {
    for round_constant in ROUND_CONSTANTS {
        let parity: [u64; 5] = std::array::from_fn(|x| (0..5).fold(0, |sum, y| sum ^ state[x + 5 * y]));
        for x in 0..5 {
            let theta = parity[(x + 4) % 5] ^ parity[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[x + 5 * y] ^= theta;
            }
        }
        // rho and pi: lane (x, y) moves to (y, 2x + 3y).
        let mut moved = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                moved[y + 5 * ((2 * x + 3 * y) % 5)] = state[x + 5 * y].rotate_left(RHO[x + 5 * y]);
            }
        }
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] = moved[x + 5 * y] ^ (!moved[(x + 1) % 5 + 5 * y] & moved[(x + 2) % 5 + 5 * y]);
            }
        }
        state[0] ^= round_constant;
    }
}

/// SHAKE128 (FIPS 202, Section 6.2) of `message`, `length` output bytes.
pub fn shake128(message: &[u8], length: usize) -> Vec<u8> {
    let mut padded = message.to_vec();
    padded.push(0x1f);
    padded.resize(padded.len().div_ceil(RATE) * RATE, 0);
    *padded.last_mut().unwrap() |= 0x80;
    let mut state = [0u64; 25];
    for block in padded.chunks_exact(RATE) {
        for (lane, bytes) in state.iter_mut().zip(block.chunks_exact(8)) {
            *lane ^= u64::from_le_bytes(bytes.try_into().unwrap());
        }
        keccak_f(&mut state);
    }
    let mut output = Vec::with_capacity(length.div_ceil(RATE) * RATE);
    loop {
        for lane in &state[..RATE / 8] {
            output.extend(lane.to_le_bytes());
        }
        if output.len() >= length {
            output.truncate(length);
            return output;
        }
        keccak_f(&mut state);
    }
}

/// All `32 * DEGREE` output bytes of key element `(row, block)`.
pub fn element_bytes(seed: &[u8; 32], row: u32, block: u64) -> Vec<u8> {
    let mut input = SETUP_ID.to_vec();
    input.extend(seed);
    input.extend(row.to_le_bytes());
    input.extend(block.to_le_bytes());
    shake128(&input, 32 * DEGREE)
}

/// Coefficient `lane` is bytes `32 * lane .. 32 * lane + 32`, read as one
/// little-endian 256-bit integer and reduced modulo q.
pub fn coefficients(seed: &[u8; 32], row: u32, block: u64) -> [u64; DEGREE] {
    let bytes = element_bytes(seed, row, block);
    std::array::from_fn(|lane| {
        bytes[32 * lane..32 * lane + 32]
            .chunks_exact(8)
            .rev()
            .fold(0u128, |value, limb| {
                ((value << 64) | u128::from(u64::from_le_bytes(limb.try_into().unwrap()))) % MODULUS
            }) as u64
    })
}

/// Check the schema-4 Lean setup fixture for `seed`: the FIPS 202 example,
/// the indexed coefficients, and one complete key element.
pub fn check_lean_setup_vectors(setup: &Value, seed: &[u8; 32]) {
    assert_eq!(setup[0], 4, "Lean SHAKE128 setup schema");
    assert_eq!(setup[1], json!(SETUP_ID));
    assert_eq!(setup[4], json!(seed));
    assert_eq!(shake128(&[], 32), SHAKE128_EMPTY, "FIPS 202 SHAKE128 example");
    assert_eq!(setup[3], json!(SHAKE128_EMPTY));
    let samples: Vec<[u64; 4]> = serde_json::from_value(setup[5].clone()).expect("Lean indexed coefficients");
    assert!(!samples.is_empty());
    for [row, block, lane, expected] in samples {
        assert_eq!(
            coefficients(seed, row.try_into().unwrap(), block)[lane as usize],
            expected
        );
    }
    // All lanes, including those that cross a 168-byte rate boundary.
    assert_eq!(setup[7], json!(element_bytes(seed, 1, 32_768)));
}

pub fn commitment_row(seed: &[u8; 32], row: u32, carrier: &[u8]) -> [u64; DEGREE] {
    assert_eq!(carrier.len() % DEGREE, 0);
    // One convolution coefficient has at most carrier.len() terms. The
    // descending Phi81 reduction can combine at most three such sums.
    assert!((carrier.len() as u128)
        .checked_mul(MODULUS - 1)
        .and_then(|bound| bound.checked_mul(3))
        .is_some_and(|bound| bound < i128::MAX as u128));
    let mut raw = carrier
        .par_chunks_exact(DEGREE)
        .enumerate()
        .filter(|(_, values)| values.iter().any(|&value| value != 0))
        .map(|(block, values)| {
            let key = coefficients(seed, row, block as u64);
            let mut product = [0i128; 2 * DEGREE - 1];
            for (lane, &coefficient) in key.iter().enumerate() {
                let coefficient = i128::from(coefficient);
                for (power, &value) in values.iter().enumerate() {
                    match value {
                        0 => {}
                        1 => product[lane + power] += coefficient,
                        255 => product[lane + power] -= coefficient,
                        _ => panic!("commitment opening is not a signed unit"),
                    }
                }
            }
            product
        })
        .reduce(
            || [0i128; 2 * DEGREE - 1],
            |left, right| std::array::from_fn(|index| left[index] + right[index]),
        );
    for power in (DEGREE..raw.len()).rev() {
        let coefficient = raw[power];
        raw[power - DEGREE] -= coefficient;
        raw[power - DEGREE / 2] -= coefficient;
    }
    std::array::from_fn(|index| raw[index].rem_euclid(MODULUS as i128) as u64)
}
