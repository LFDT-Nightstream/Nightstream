//! The diagonal sums of the shrink verifier's block part,
//! `Σ_{b < count} Π_tracks eq(point, offset + b)`, by a walk over the bits of `b`, low bit
//! first. The walk's state is each track's carry and whether the low bits of `b` are below
//! those of `count`.
//!
//! Each function is one of `formal/compression-lean/CompressionLean/Ports/Shrink/Diagonal.lean`,
//! and `CompressionLeanRust/Diagonal.lean` proves it equal to that copy.

use crate::algebra::index_bit;
use crate::field::{ext_add, ext_mul, ext_reduce, ext_sub, Ext, EXT_ONE, EXT_ZERO};

/// A point and an offset: `b` reads as the index `offset + b` of `point`.
#[derive(Clone)]
pub struct Track {
    pub point: Vec<Ext>,
    pub offset: usize,
}

/// The number of bits of `value`.
fn bit_length(value: usize) -> usize {
    (usize::BITS as usize).wrapping_sub(value.leading_zeros() as usize)
}

/// Bit `t` of `value`, as a number.
fn bit_at(value: usize, t: usize) -> usize {
    if index_bit(value, t) {
        1
    } else {
        0
    }
}

/// `2^tracks.len()`, the carry states.
fn carry_states(tracks: &[Track]) -> usize {
    let mut states: usize = 1;
    let mut i = 0;
    while i < tracks.len() {
        states = states.checked_mul(2).unwrap();
        i = i.wrapping_add(1);
    }
    states
}

/// The bits that the walk reads: those of `count`, of every offset, and one per coordinate.
pub fn walk_bits(tracks: &[Track], count: usize) -> usize {
    let mut bits = bit_length(count);
    let mut i = tracks.len();
    while 0 < i {
        i = i.wrapping_sub(1);
        bits = tracks[i]
            .point
            .len()
            .max(bit_length(tracks[i].offset))
            .max(bits);
    }
    bits
}

/// The factor of index bit `bit` at coordinate `t`. Past the point's coordinates, the index
/// must not have the bit.
pub fn track_factor(point: &[Ext], t: usize, bit: usize) -> Ext {
    if t < point.len() {
        if bit == 1 {
            ext_reduce(point[t])
        } else {
            ext_sub(EXT_ONE, point[t])
        }
    } else if bit == 1 {
        EXT_ZERO
    } else {
        EXT_ONE
    }
}

/// The product of the tracks' factors at bit `t` for bit `b` of `b`, and the next carries; bit
/// `i` of `carry` is track `i`'s carry.
pub fn tracks_step(tracks: &[Track], t: usize, b: usize, carry: usize) -> (Ext, usize) {
    let mut factor = EXT_ONE;
    let mut next_carry: usize = 0;
    let mut i = tracks.len();
    while 0 < i {
        i = i.wrapping_sub(1);
        let sum = bit_at(tracks[i].offset, t)
            .checked_add(b)
            .unwrap()
            .checked_add(bit_at(carry, i))
            .unwrap();
        let bit = if sum.is_multiple_of(2) { 0 } else { 1 };
        factor = ext_mul(track_factor(&tracks[i].point, t, bit), factor);
        next_carry = sum
            .wrapping_shr(1)
            .checked_add(next_carry.checked_mul(2).unwrap())
            .unwrap();
    }
    (factor, next_carry)
}

/// Whether the low bits of `b` are below those of `count` after one more bit.
pub fn next_below(count_bit: usize, b: usize, below: usize) -> usize {
    if b < count_bit {
        1
    } else if count_bit < b {
        0
    } else {
        below
    }
}

/// The walk's state `(target, target_below)` after bit `t`, from `states` before it.
fn step_sum(
    tracks: &[Track],
    count_bit: usize,
    t: usize,
    states: &[[Ext; 2]],
    target: usize,
    target_below: usize,
) -> Ext {
    let carries = carry_states(tracks);
    let mut total = EXT_ZERO;
    let mut carry = 0;
    while carry < carries {
        let mut below = 0;
        while below < 2 {
            let mut b = 0;
            while b < 2 {
                let (factor, next_carry) = tracks_step(tracks, t, b, carry);
                if next_carry == target && next_below(count_bit, b, below) == target_below {
                    total = ext_add(total, ext_mul(states[carry][below], factor));
                }
                b = b.wrapping_add(1);
            }
            below = below.wrapping_add(1);
        }
        carry = carry.wrapping_add(1);
    }
    total
}

/// The walk's states after bit `t` from `states` before it, by carry, then below or not.
pub fn walk_step(tracks: &[Track], count: usize, t: usize, states: &[[Ext; 2]]) -> Vec<[Ext; 2]> {
    let carries = carry_states(tracks);
    let count_bit = bit_at(count, t);
    let mut next = Vec::new();
    let mut target = 0;
    while target < carries {
        let at = step_sum(tracks, count_bit, t, states, target, 0);
        let below = step_sum(tracks, count_bit, t, states, target, 1);
        next.push([at, below]);
        target = target.wrapping_add(1);
    }
    next
}

/// The walk's states after `width` bits.
pub fn walk_states(tracks: &[Track], count: usize, width: usize) -> Vec<[Ext; 2]> {
    let carries = carry_states(tracks);
    let mut states = Vec::new();
    let mut carry = 0;
    while carry < carries {
        if carry == 0 {
            states.push([EXT_ONE, EXT_ZERO]);
        } else {
            states.push([EXT_ZERO, EXT_ZERO]);
        }
        carry = carry.wrapping_add(1);
    }
    let mut t = 0;
    while t < width {
        states = walk_step(tracks, count, t, &states);
        t = t.wrapping_add(1);
    }
    states
}

/// `Σ_{b < count} Π_tracks eq(point, offset + b)`, where `eq(point, i)` is zero for an `i` of
/// more bits than the point has coordinates.
pub fn diagonal(tracks: &[Track], count: usize) -> Ext {
    let states = walk_states(tracks, count, walk_bits(tracks, count));
    states[0][1]
}
