//! Independent indexed ChaCha20 and signed-integer Ajtai evaluation.
//! Spec.AjtaiSetupV1 fixes the nonce, counter, and 256-bit reduction.
//! No production RNG, key expander, or commitment routine is called.

use rayon::prelude::*;

const DEGREE: usize = 54;
const MODULUS: u128 = 18_446_744_069_414_584_321;

fn quarter(state: &mut [u32; 16], indices: [usize; 4]) {
    let [ai, bi, ci, di] = indices;
    let [mut a, mut b, mut c, mut d] = indices.map(|index| state[index]);
    a = a.wrapping_add(b);
    d = (d ^ a).rotate_left(16);
    c = c.wrapping_add(d);
    b = (b ^ c).rotate_left(12);
    a = a.wrapping_add(b);
    d = (d ^ a).rotate_left(8);
    c = c.wrapping_add(d);
    b = (b ^ c).rotate_left(7);
    state[ai] = a;
    state[bi] = b;
    state[ci] = c;
    state[di] = d;
}

pub fn block_words(seed: &[u8; 32], row: u32, block: u64, lane: u32) -> [u32; 16] {
    let mut initial = [0u32; 16];
    for (index, bytes) in b"expand 32-byte k".chunks_exact(4).enumerate() {
        initial[index] = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    for (index, bytes) in seed.chunks_exact(4).enumerate() {
        initial[index + 4] = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    initial[12..].copy_from_slice(&[lane, row, block as u32, (block >> 32) as u32]);
    let mut state = initial;
    for _ in 0..10 {
        for column in 0..4 {
            quarter(&mut state, [column, column + 4, column + 8, column + 12]);
        }
        for column in 0..4 {
            quarter(
                &mut state,
                [
                    column,
                    4 + (column + 1) % 4,
                    8 + (column + 2) % 4,
                    12 + (column + 3) % 4,
                ],
            );
        }
    }
    std::array::from_fn(|index| state[index].wrapping_add(initial[index]))
}

pub fn coefficient(seed: &[u8; 32], row: u32, block: u64, lane: u32) -> u64 {
    let words = block_words(seed, row, block, lane);
    // Four 64-bit limbs give the same little-endian 256-bit integer as
    // Lean's eight 32-bit limbs. The accumulator is reduced at each limb.
    (0..4).rev().fold(0u128, |value, limb| {
        let word = u64::from(words[2 * limb]) | (u64::from(words[2 * limb + 1]) << 32);
        ((value << 64) | u128::from(word)) % MODULUS
    }) as u64
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
            let mut product = [0i128; 2 * DEGREE - 1];
            for lane in 0..DEGREE {
                let coefficient = i128::from(coefficient(seed, row, block as u64, lane as u32));
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
