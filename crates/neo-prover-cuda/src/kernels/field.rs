//! Goldilocks, `K` = Goldilocks[u] / (u^2 - 7) and Poseidon2 width 16 for
//! device code; the CUDA port of the arithmetic of `nightstream.metal`.
//! Each function takes words that may be non-canonical where the Metal
//! function does, and returns the same canonical word.

pub const GOLDILOCKS_MODULUS: u64 = 0xffff_ffff_0000_0001;
pub const GOLDILOCKS_EPSILON: u64 = 0xffff_ffff;
pub const LIMB_MASK: u64 = 0xffff_ffff;
pub const RING_DEGREE: u64 = 54;
pub const RING_PRODUCT_COEFFICIENTS: u64 = 107;

pub const POSEIDON_WIDTH: usize = 16;
pub const POSEIDON_RATE: usize = 12;
pub const POSEIDON_DIGEST: usize = 4;
const POSEIDON_EXTERNAL_HALF_ROUNDS: usize = 4;
const POSEIDON_INTERNAL_ROUNDS: usize = 22;
const POSEIDON_RC_INTERNAL: usize = POSEIDON_EXTERNAL_HALF_ROUNDS * POSEIDON_WIDTH;
const POSEIDON_RC_TERMINAL: usize = POSEIDON_RC_INTERNAL + POSEIDON_INTERNAL_ROUNDS;
const POSEIDON_RC_DIAG: usize = POSEIDON_RC_TERMINAL + POSEIDON_EXTERNAL_HALF_ROUNDS * POSEIDON_WIDTH;

#[inline(always)]
pub fn gl_from_word(value: u64) -> u64 {
    if value >= GOLDILOCKS_MODULUS {
        value - GOLDILOCKS_MODULUS
    } else {
        value
    }
}

#[inline(always)]
pub fn gl_add(lhs: u64, rhs: u64) -> u64 {
    let (mut sum, carry) = lhs.overflowing_add(rhs);
    if carry {
        sum = sum.wrapping_add(GOLDILOCKS_EPSILON);
    }
    if sum >= GOLDILOCKS_MODULUS {
        sum -= GOLDILOCKS_MODULUS;
    }
    sum
}

#[inline(always)]
pub fn gl_sub(lhs: u64, rhs: u64) -> u64 {
    let (difference, borrow) = lhs.overflowing_sub(rhs);
    if borrow {
        difference.wrapping_sub(GOLDILOCKS_EPSILON)
    } else {
        difference
    }
}

/// `lo + hi·2^64` reduced, for any 128-bit value.
#[inline(always)]
pub fn gl_reduce_wide(lo: u64, hi: u64) -> u64 {
    let hi_hi = hi >> 32;
    let hi_lo = hi & GOLDILOCKS_EPSILON;
    let (mut reduced_lo, borrow) = lo.overflowing_sub(hi_hi);
    if borrow {
        reduced_lo = reduced_lo.wrapping_sub(GOLDILOCKS_EPSILON);
    }
    let folded_hi = (hi_lo << 32).wrapping_sub(hi_lo);
    gl_add(reduced_lo, folded_hi)
}

#[inline(always)]
pub fn mul_wide(lhs: u64, rhs: u64) -> (u64, u64) {
    let product = (lhs as u128) * (rhs as u128);
    (product as u64, (product >> 64) as u64)
}

#[inline(always)]
pub fn gl_mul(lhs: u64, rhs: u64) -> u64 {
    let (lo, hi) = mul_wide(lhs, rhs);
    gl_reduce_wide(lo, hi)
}

/// `lo + hi·2^64` reduced, for a sum whose high word is below 2^32.
#[inline(always)]
pub fn gl_reduce_sum(lo: u64, hi: u64) -> u64 {
    gl_add(gl_from_word(lo), hi.wrapping_mul(GOLDILOCKS_EPSILON))
}

#[inline(always)]
pub fn gl_sbox(value: u64) -> u64 {
    let square = gl_mul(value, value);
    let cube = gl_mul(square, value);
    gl_mul(gl_mul(cube, cube), value)
}

#[inline(always)]
pub fn gl_mul_by_7(value: u64) -> u64 {
    let twice = gl_add(value, value);
    let four_times = gl_add(twice, twice);
    gl_add(gl_add(four_times, twice), value)
}

/// A `K` value `c0 + c1·u`, stored interleaved as `c0, c1`.
#[derive(Clone, Copy)]
pub struct Kx {
    pub c0: u64,
    pub c1: u64,
}

pub const KX_ZERO: Kx = Kx { c0: 0, c1: 0 };
pub const KX_ONE: Kx = Kx { c0: 1, c1: 0 };

#[inline(always)]
pub fn kx_add(lhs: Kx, rhs: Kx) -> Kx {
    Kx {
        c0: gl_add(lhs.c0, rhs.c0),
        c1: gl_add(lhs.c1, rhs.c1),
    }
}

#[inline(always)]
pub fn kx_sub(lhs: Kx, rhs: Kx) -> Kx {
    Kx {
        c0: gl_sub(lhs.c0, rhs.c0),
        c1: gl_sub(lhs.c1, rhs.c1),
    }
}

#[inline(always)]
pub fn kx_mul(lhs: Kx, rhs: Kx) -> Kx {
    let c0_product = gl_mul(lhs.c0, rhs.c0);
    let c1_product = gl_mul(lhs.c1, rhs.c1);
    let cross = gl_sub(
        gl_sub(gl_mul(gl_add(lhs.c0, lhs.c1), gl_add(rhs.c0, rhs.c1)), c0_product),
        c1_product,
    );
    Kx {
        c0: gl_add(c0_product, gl_mul_by_7(c1_product)),
        c1: cross,
    }
}

/// `K` value `index` of an interleaved table.
///
/// # Safety
///
/// `values` points to at least `2·index + 2` readable words.
#[inline(always)]
pub unsafe fn load_k(values: *const u64, index: u64) -> Kx {
    // SAFETY: the caller guarantees both words are readable.
    unsafe {
        Kx {
            c0: gl_from_word(*values.add(2 * index as usize)),
            c1: gl_from_word(*values.add(2 * index as usize + 1)),
        }
    }
}

/// Store `value` as `K` value `index` of an interleaved table.
///
/// # Safety
///
/// `values` points to at least `2·index + 2` writable words.
#[inline(always)]
pub unsafe fn store_k(values: *mut u64, index: u64, value: Kx) {
    // SAFETY: the caller guarantees both words are writable.
    unsafe {
        *values.add(2 * index as usize) = value.c0;
        *values.add(2 * index as usize + 1) = value.c1;
    }
}

#[inline(always)]
fn poseidon_mat4(x0: u64, x1: u64, x2: u64, x3: u64) -> [u64; 4] {
    let t01 = gl_add(x0, x1);
    let t23 = gl_add(x2, x3);
    let t0123 = gl_add(t01, t23);
    let t01123 = gl_add(t0123, x1);
    let t01233 = gl_add(t0123, x3);
    [
        gl_add(t01123, t01),
        gl_add(gl_add(t01123, x2), x2),
        gl_add(t01233, t23),
        gl_add(gl_add(t01233, x0), x0),
    ]
}

#[inline(always)]
fn poseidon_mds_light(state: &mut [u64; POSEIDON_WIDTH]) {
    let a = poseidon_mat4(state[0], state[1], state[2], state[3]);
    let b = poseidon_mat4(state[4], state[5], state[6], state[7]);
    let c = poseidon_mat4(state[8], state[9], state[10], state[11]);
    let d = poseidon_mat4(state[12], state[13], state[14], state[15]);
    for lane in 0..4 {
        let mixed = gl_add(gl_add(a[lane], b[lane]), gl_add(c[lane], d[lane]));
        state[lane] = gl_add(a[lane], mixed);
        state[4 + lane] = gl_add(b[lane], mixed);
        state[8 + lane] = gl_add(c[lane], mixed);
        state[12 + lane] = gl_add(d[lane], mixed);
    }
}

/// # Safety
///
/// `round_constants` points to `poseidon2::RC_WORDS` readable words.
#[inline(always)]
unsafe fn poseidon_external_round(state: &mut [u64; POSEIDON_WIDTH], round_constants: *const u64, base: usize) {
    for (lane, word) in state.iter_mut().enumerate() {
        // SAFETY: `base + lane` is a round-constant index (caller).
        *word = gl_sbox(gl_add(*word, unsafe { *round_constants.add(base + lane) }));
    }
    poseidon_mds_light(state);
}

/// The Poseidon2 width-16 permutation with the round constants in the
/// layout of `crate::poseidon2::round_constant_words`.
///
/// # Safety
///
/// `round_constants` points to `poseidon2::RC_WORDS` readable words.
#[inline(always)]
pub unsafe fn poseidon_permute(state: &mut [u64; POSEIDON_WIDTH], round_constants: *const u64) {
    poseidon_mds_light(state);
    for round in 0..POSEIDON_EXTERNAL_HALF_ROUNDS {
        // SAFETY: the caller's round constants cover every round.
        unsafe { poseidon_external_round(state, round_constants, round * POSEIDON_WIDTH) };
    }
    for round in 0..POSEIDON_INTERNAL_ROUNDS {
        // SAFETY: as above.
        let constant = unsafe { *round_constants.add(POSEIDON_RC_INTERNAL + round) };
        state[0] = gl_sbox(gl_add(state[0], constant));
        let mut sum = 0;
        for &word in state.iter() {
            sum = gl_add(sum, word);
        }
        for (lane, word) in state.iter_mut().enumerate() {
            // SAFETY: as above.
            let diagonal = unsafe { *round_constants.add(POSEIDON_RC_DIAG + lane) };
            *word = gl_add(gl_mul(*word, diagonal), sum);
        }
    }
    for round in 0..POSEIDON_EXTERNAL_HALF_ROUNDS {
        // SAFETY: as above.
        unsafe { poseidon_external_round(state, round_constants, POSEIDON_RC_TERMINAL + round * POSEIDON_WIDTH) };
    }
}
