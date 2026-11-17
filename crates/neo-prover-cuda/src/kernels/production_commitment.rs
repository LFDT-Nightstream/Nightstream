//! The kernels of `production_commitment.metal`: SHAKE128 key rows, the
//! signed-unit Ajtai sums, and the commitment key weights.

use cuda_device::{cuda_module, kernel, thread, DynamicSharedArray};

use crate::kernels::field::{
    gl_add, gl_mul, gl_reduce_sum, gl_sub, GOLDILOCKS_MODULUS, RING_DEGREE, RING_PRODUCT_COEFFICIENTS,
};

const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];
/// SHAKE128 absorbs and squeezes 168 bytes, 21 lanes, per permutation.
const SHAKE128_RATE_LANES: usize = 21;

/// One Keccak-f[1600] round on lanes `a[5y + x]`: theta, rho and pi, chi
/// and iota (FIPS 202, Section 3.3). Every lane index is fixed, so the
/// state can stay in registers.
#[inline(always)]
fn keccak_round(a: &mut [u64; 25], round_constant: u64) {
    let c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    let c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    let c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    let c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    let c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    let d0 = c4 ^ c1.rotate_left(1);
    let d1 = c0 ^ c2.rotate_left(1);
    let d2 = c1 ^ c3.rotate_left(1);
    let d3 = c2 ^ c4.rotate_left(1);
    let d4 = c3 ^ c0.rotate_left(1);
    let b00 = a[0] ^ d0;
    let b10 = (a[6] ^ d1).rotate_left(44);
    let b20 = (a[12] ^ d2).rotate_left(43);
    let b30 = (a[18] ^ d3).rotate_left(21);
    let b40 = (a[24] ^ d4).rotate_left(14);
    let b01 = (a[3] ^ d3).rotate_left(28);
    let b11 = (a[9] ^ d4).rotate_left(20);
    let b21 = (a[10] ^ d0).rotate_left(3);
    let b31 = (a[16] ^ d1).rotate_left(45);
    let b41 = (a[22] ^ d2).rotate_left(61);
    let b02 = (a[1] ^ d1).rotate_left(1);
    let b12 = (a[7] ^ d2).rotate_left(6);
    let b22 = (a[13] ^ d3).rotate_left(25);
    let b32 = (a[19] ^ d4).rotate_left(8);
    let b42 = (a[20] ^ d0).rotate_left(18);
    let b03 = (a[4] ^ d4).rotate_left(27);
    let b13 = (a[5] ^ d0).rotate_left(36);
    let b23 = (a[11] ^ d1).rotate_left(10);
    let b33 = (a[17] ^ d2).rotate_left(15);
    let b43 = (a[23] ^ d3).rotate_left(56);
    let b04 = (a[2] ^ d2).rotate_left(62);
    let b14 = (a[8] ^ d3).rotate_left(55);
    let b24 = (a[14] ^ d4).rotate_left(39);
    let b34 = (a[15] ^ d0).rotate_left(41);
    let b44 = (a[21] ^ d1).rotate_left(2);
    a[0] = b00 ^ (!b10 & b20) ^ round_constant;
    a[1] = b10 ^ (!b20 & b30);
    a[2] = b20 ^ (!b30 & b40);
    a[3] = b30 ^ (!b40 & b00);
    a[4] = b40 ^ (!b00 & b10);
    a[5] = b01 ^ (!b11 & b21);
    a[6] = b11 ^ (!b21 & b31);
    a[7] = b21 ^ (!b31 & b41);
    a[8] = b31 ^ (!b41 & b01);
    a[9] = b41 ^ (!b01 & b11);
    a[10] = b02 ^ (!b12 & b22);
    a[11] = b12 ^ (!b22 & b32);
    a[12] = b22 ^ (!b32 & b42);
    a[13] = b32 ^ (!b42 & b02);
    a[14] = b42 ^ (!b02 & b12);
    a[15] = b03 ^ (!b13 & b23);
    a[16] = b13 ^ (!b23 & b33);
    a[17] = b23 ^ (!b33 & b43);
    a[18] = b33 ^ (!b43 & b03);
    a[19] = b43 ^ (!b03 & b13);
    a[20] = b04 ^ (!b14 & b24);
    a[21] = b14 ^ (!b24 & b34);
    a[22] = b24 ^ (!b34 & b44);
    a[23] = b34 ^ (!b44 & b04);
    a[24] = b44 ^ (!b04 & b14);
}

#[inline(always)]
fn keccak_f(state: &mut [u64; 25]) {
    for &round_constant in &KECCAK_RC {
        keccak_round(state, round_constant);
    }
}

/// For `x = 2^32`, `x^2 = x - 1` and `x^6 = 1`, so a 32-byte chunk reduces
/// to `a + b·x`.
#[inline(always)]
fn wide_coefficient(q0: u64, q1: u64, q2: u64, q3: u64) -> u64 {
    let w = |word: u64, high: bool| i64::from(if high { (word >> 32) as u32 } else { word as u32 });
    let a = w(q0, false) - w(q1, false) - w(q1, true) + w(q2, true) + w(q3, false);
    let b = w(q0, true) + w(q1, false) - w(q2, false) - w(q2, true) + w(q3, true);
    let field = |value: i64| {
        if value < 0 {
            GOLDILOCKS_MODULUS - value.unsigned_abs()
        } else {
            value as u64
        }
    };
    gl_add(field(a), gl_mul(field(b), 1 << 32))
}

/// Key element `(row, column)`: the `RING_DEGREE` coefficients of
/// SHAKE128(setup ID ‖ seed ‖ row_u32_le ‖ column_u64_le), 81 bytes, at
/// `out[lane·stride]`. `prefix` holds bytes 0..69 as nine lanes.
///
/// # Safety
///
/// `prefix` points to nine words and `out` to `RING_DEGREE·stride` words.
#[inline(always)]
pub(crate) unsafe fn production_ajtai_element(prefix: *const u64, row: u32, column: u64, out: *mut u64, stride: usize) {
    let mut state = [0u64; 25];
    for (lane, word) in state.iter_mut().enumerate().take(9) {
        // SAFETY: the prefix holds nine lanes (caller).
        *word = unsafe { *prefix.add(lane) };
    }
    state[8] |= (u64::from(row) & 0xff_ffff) << 40;
    state[9] = (u64::from(row) >> 24) | (column << 8);
    // Byte 80 is the last column byte; byte 81 starts the SHAKE padding.
    state[10] = (column >> 56) | (0x1f << 8);
    state[SHAKE128_RATE_LANES - 1] ^= 0x80 << 56;
    // The last four lanes shift through q0..q3.
    let (mut q1, mut q2, mut q3) = (0, 0, 0);
    let (mut words, mut lane) = (0, 0);
    loop {
        keccak_f(&mut state);
        for &squeezed in state.iter().take(SHAKE128_RATE_LANES) {
            let q0 = q1;
            q1 = q2;
            q2 = q3;
            q3 = squeezed;
            words += 1;
            if words == 4 {
                // SAFETY: `lane < RING_DEGREE` (caller's bound).
                unsafe { *out.add(lane * stride) = wide_coefficient(q0, q1, q2, q3) };
                words = 0;
                lane += 1;
                if lane == RING_DEGREE as usize {
                    return;
                }
            }
        }
    }
}

struct HalfSums {
    low_lo: u64,
    low_hi: u64,
    high_lo: u64,
    high_hi: u64,
}

/// # Safety
///
/// `key` points to one key element: `RING_DEGREE` words.
#[inline(always)]
unsafe fn add_bits(mut bits: u32, offset: u32, key: *const u64, lane: u32, sums: &mut HalfSums) {
    while bits != 0 {
        let shift = offset + bits.trailing_zeros();
        bits &= bits - 1;
        let low = lane >= shift;
        let index = if low {
            lane - shift
        } else {
            lane + RING_DEGREE as u32 - shift
        };
        // SAFETY: `index < RING_DEGREE` because `shift < RING_DEGREE`.
        let value = unsafe { *key.add(index as usize) };
        let (lo, hi) = (value & 0xffff_ffff, value >> 32);
        if low {
            sums.low_lo += lo;
            sums.low_hi += hi;
        } else {
            sums.high_lo += lo;
            sums.high_hi += hi;
        }
    }
}

/// # Safety
///
/// As `add_bits`.
#[inline(always)]
unsafe fn add_digits(digits: u64, key: *const u64, lane: u32, sums: &mut HalfSums) {
    // SAFETY: the caller's key element.
    unsafe {
        add_bits(digits as u32, 0, key, lane, sums);
        add_bits((digits >> 32) as u32, 32, key, lane, sums);
    }
}

#[inline(always)]
fn join_halves(lo: u64, hi: u64) -> u64 {
    let low = lo.wrapping_add(hi << 32);
    let high = (hi >> 32) + u64::from(low < lo);
    gl_reduce_sum(low, high)
}

#[cuda_module]
pub mod production_commitment_kernels {
    use super::*;

    /// Key row `row` at `count` columns: element `position` is
    /// `(row, columns[position])`, written at `slab[position·RING_DEGREE]`.
    #[kernel]
    pub fn production_key_row(prefix: *const u64, columns: *const u32, slab: *mut u64, count: u64, row: u64) {
        let position = thread::index_1d().get();
        if position as u64 >= count {
            return;
        }
        // SAFETY: the host gives nine prefix lanes, `count` columns and
        // `count` elements of slab.
        unsafe {
            production_ajtai_element(
                prefix,
                row as u32,
                u64::from(*columns.add(position)),
                slab.add(position * RING_DEGREE as usize),
                1,
            );
        }
    }

    /// Partial sums of one key row's signed ring products: block `group`
    /// of witness block `w` sums the columns `group·per_group ..` of its
    /// witnesses. Shape: `[count, witnesses, groups, per_group, row,
    /// witnesses_per_group]`. Each witness takes 64 threads, one per lane.
    #[kernel]
    pub fn production_ajtai_accumulate(
        key: *const u64,
        masks: *const [u64; 2],
        partials: *mut u64,
        key_columns: *const u32,
        shape: [u64; 6],
    ) {
        let scratch: *mut u64 = DynamicSharedArray::<u64>::get();
        let (count, witnesses, groups, columns_per_group, witnesses_per_group) =
            (shape[0], shape[1], shape[2], shape[3], shape[5]);
        let group_index = u64::from(thread::blockIdx_x());
        let thread_index = thread::threadIdx_x();
        let group = group_index % groups;
        let lane = thread_index % 64;
        let witness = (group_index / groups) * witnesses_per_group + u64::from(thread_index / 64);
        let active = witness < witnesses && u64::from(lane) < RING_DEGREE;
        let mut positive = HalfSums {
            low_lo: 0,
            low_hi: 0,
            high_lo: 0,
            high_hi: 0,
        };
        let mut negative = HalfSums {
            low_lo: 0,
            low_hi: 0,
            high_lo: 0,
            high_hi: 0,
        };
        let first = group * columns_per_group;
        let end = (first + columns_per_group).min(count);
        if active {
            for position in first..end {
                // SAFETY: the host gives `witnesses·count` masks, `count`
                // key columns and the key element of each column.
                unsafe {
                    let digits = *masks.add((witness * count + position) as usize);
                    let element = key.add(*key_columns.add(position as usize) as usize * RING_DEGREE as usize);
                    add_digits(digits[0], element, lane, &mut positive);
                    add_digits(digits[1], element, lane, &mut negative);
                }
            }
        }
        // SAFETY: the launch gives each witness `RING_PRODUCT_COEFFICIENTS`
        // words of shared memory.
        let raw = unsafe { scratch.add((thread_index / 64) as usize * RING_PRODUCT_COEFFICIENTS as usize) };
        if active {
            // SAFETY: as above.
            unsafe {
                *raw.add(lane as usize) = gl_sub(
                    join_halves(positive.low_lo, positive.low_hi),
                    join_halves(negative.low_lo, negative.low_hi),
                );
                if u64::from(lane) + RING_DEGREE < RING_PRODUCT_COEFFICIENTS {
                    *raw.add(lane as usize + RING_DEGREE as usize) = gl_sub(
                        join_halves(positive.high_lo, positive.high_hi),
                        join_halves(negative.high_lo, negative.high_hi),
                    );
                }
            }
        }
        thread::sync_threads();
        if active {
            let lane = lane as usize;
            let degree = RING_DEGREE as usize;
            // SAFETY: as above; `partials` holds `witnesses·groups` elements.
            unsafe {
                let value = if lane < degree / 2 {
                    let mut value = gl_sub(*raw.add(lane), *raw.add(lane + degree));
                    if lane + 81 < RING_PRODUCT_COEFFICIENTS as usize {
                        value = gl_add(value, *raw.add(lane + 81));
                    }
                    value
                } else {
                    gl_sub(*raw.add(lane), *raw.add(lane + degree / 2))
                };
                *partials.add(((witness * groups + group) * RING_DEGREE) as usize + lane) = value;
            }
        }
    }

    /// `output[i] = Σ_group partials[witness, group, coefficient]` for
    /// `i < count`.
    #[kernel]
    pub fn production_ajtai_sum_groups(partials: *const u64, output: *mut u64, groups: u64, count: u64) {
        let index = thread::index_1d().get() as u64;
        if index >= count {
            return;
        }
        let witness = index / RING_DEGREE;
        let coefficient = index % RING_DEGREE;
        let mut value = 0;
        for group in 0..groups {
            // SAFETY: the host gives `groups` partials per witness.
            value = gl_add(value, unsafe {
                *partials.add(((witness * groups + group) * RING_DEGREE + coefficient) as usize)
            });
        }
        // SAFETY: the output holds `count` words.
        unsafe { *output.add(index as usize) = value };
    }

    /// One key row's part of `Σ_i λ_i·a_i`: `out[3e + c] = (out[3e + c] +)
    /// λ[c]·row[e]` for every coefficient `e < count`; `first` starts the
    /// sum. `λ` is a cubic-extension value as its three base coordinates.
    #[kernel]
    pub fn production_key_weights(row: *const u64, out: *mut u64, power: [u64; 3], first: u64, count: u64) {
        let index = thread::index_1d().get();
        if index as u64 >= count {
            return;
        }
        // SAFETY: the host gives `count` row words and `3·count` outputs.
        unsafe {
            let value = *row.add(index);
            for (c, &lambda) in power.iter().enumerate() {
                let term = gl_mul(value, lambda);
                let at = out.add(3 * index + c);
                *at = if first != 0 { term } else { gl_add(*at, term) };
            }
        }
    }
}
