//! Closed-form multilinear extensions that the succinct verifier evaluates.
//!
//! Owns every public weight the layer-1 verifier computes without reading the
//! key or the matrices: the scaled-eq point of a power weight, interval and
//! step functions over a sorted index, the identity, `χ_r` at an `Ext` point,
//! the lane-split tables of a geometric run, and the Eval_K (Pad) weight by
//! long division. Points are low-bit first. Each formula has a brute-force
//! test in `tests/internal/mle.rs`.

use neo_math::{D, K};
use p3_field_v08::{Field, PrimeCharacteristicRing};

use crate::field::{Ext, Gl, Kx};

/// `Σ_{x < 2^k} c^x f(x) = κ · f~(x_c)` with `κ = Π_t (1 + c^{2^t})` and
/// `x_t = c^{2^t} / (1 + c^{2^t})`. `None` when some `1 + c^{2^t}` is zero.
pub(crate) fn scaled_eq_point(c: Ext, bits: usize) -> Option<(Ext, Vec<Ext>)> {
    let mut kappa = Ext::ONE;
    let mut point = Vec::with_capacity(bits);
    let mut power = c;
    for _ in 0..bits {
        let denominator = Ext::ONE + power;
        kappa *= denominator;
        point.push(power * denominator.try_inverse()?);
        power = power.square();
    }
    Some((kappa, point))
}

/// `Σ_{x < bound} eq(point, x)`; `bound` may equal `2^point.len()`.
pub(crate) fn less_than(point: &[Ext], bound: u64) -> Ext {
    let variables = point.len();
    if bound >> variables != 0 {
        return Ext::ONE;
    }
    let mut total = Ext::ZERO;
    let mut prefix = Ext::ONE;
    for t in (0..variables).rev() {
        if (bound >> t) & 1 == 1 {
            total += prefix * (Ext::ONE - point[t]);
            prefix *= point[t];
        } else {
            prefix *= Ext::ONE - point[t];
        }
    }
    total
}

/// The MLE of a step function: `values[s]` on `[starts[s], starts[s + 1])`,
/// with `starts[0] = 0` and the last segment ending at `end`; zero after.
pub(crate) fn step(point: &[Ext], starts: &[u64], values: &[Ext], end: u64) -> Ext {
    assert_eq!(starts.len(), values.len());
    let mut total = Ext::ZERO;
    for (s, &value) in values.iter().enumerate() {
        let stop = starts.get(s + 1).copied().unwrap_or(end);
        total += value * (less_than(point, stop) - less_than(point, starts[s]));
    }
    total
}

/// `Σ_t 2^t point_t`, the MLE of `x ↦ x`.
pub(crate) fn identity(point: &[Ext]) -> Ext {
    point
        .iter()
        .rev()
        .fold(Ext::ZERO, |acc, &coordinate| acc.double() + coordinate)
}

/// `χ_r` at an `Ext` point of the low `point.len()` bits; the remaining
/// coordinates of `r` see bit 0: `Π_{t<n} eq(r_t, point_t) · Π_{t≥n} (1 - r_t)`.
pub(crate) fn chi(r: &[K], point: &[Ext]) -> Kx {
    assert!(point.len() <= r.len());
    let mut total = Kx::ONE;
    for (t, &value) in r.iter().enumerate() {
        let r = Kx::from_k(value);
        let factor = match point.get(t) {
            // eq(r, x) = (1 - x) + r·(2x - 1)
            Some(&x) => {
                let mut factor = r.scale(x.double() - Ext::ONE);
                factor.re += Ext::ONE - x;
                factor
            }
            None => Kx {
                re: Ext::ONE - r.re,
                im: -r.im,
            },
        };
        total = total.mul(factor);
    }
    total
}

/// The lane split of a geometric run class `(length ≤ D, ratio)`. For a run
/// starting at lane `l` of block `b` and any `g(c) = E(⌊c/D⌋)·τ_{c mod D}`,
/// `Σ_{k<length} ratio^k g(D·b + l + k) = E(b)·H0[l] + E(b + 1)·H1[l]`.
pub(crate) fn lane_split(tau: &[Ext; D], length: usize, ratio: Gl) -> [[Ext; D]; 2] {
    assert!((1..=D).contains(&length));
    let ratio_power = ratio.exp_u64(length as u64);
    let mut head = [Ext::ZERO; D];
    head[D - 1] = tau[D - 1];
    for lane in (0..D - 1).rev() {
        let mut value = tau[lane] + head[lane + 1] * ratio;
        if lane + length < D {
            value -= tau[lane + length] * ratio_power;
        }
        head[lane] = value;
    }
    let mut prefix = [Ext::ZERO; D];
    let mut acc = Ext::ZERO;
    let mut power = Gl::ONE;
    for (k, &value) in tau.iter().enumerate() {
        acc += value * power;
        prefix[k] = acc;
        power *= ratio;
    }
    let tail = std::array::from_fn(|lane| match (lane + length).checked_sub(D + 1) {
        Some(last) => prefix[last] * ratio.exp_u64((D - lane) as u64),
        None => Ext::ZERO,
    });
    [head, tail]
}

/// `Σ_{b<blocks} eq(s_block, b) Σ_{l<D} τ_l χ_r(D·b + l)`, by long division of
/// the coordinate by `D`, most significant bit first. The state is the
/// remainder and whether the quotient prefix is already below `blocks`'s.
pub(crate) fn eval_k(r: &[K], s_block: &[Ext], tau: &[Ext; D], blocks: usize) -> Kx {
    let block_bits = s_block.len();
    assert!(blocks <= 1 << block_bits);
    let coordinate_bits = (D << block_bits).next_power_of_two().trailing_zeros() as usize;
    assert!(coordinate_bits <= r.len());
    // states[remainder][below]
    let mut states = vec![[Kx::ZERO; 2]; D];
    states[0][0] = Kx::ONE;
    for t in (0..coordinate_bits).rev() {
        let r_t = Kx::from_k(r[t]);
        let r_factor = [
            Kx {
                re: Ext::ONE - r_t.re,
                im: -r_t.im,
            },
            r_t,
        ];
        let bound_bit = (blocks >> t) & 1;
        let mut next = vec![[Kx::ZERO; 2]; D];
        for (remainder, accumulated) in states.iter().enumerate() {
            for (below, &value) in accumulated.iter().enumerate() {
                if value == Kx::ZERO {
                    continue;
                }
                for bit in 0..2 {
                    let doubled = 2 * remainder + bit;
                    let quotient = usize::from(doubled >= D);
                    if t >= block_bits && quotient == 1 {
                        continue;
                    }
                    let next_below = match (below, quotient.cmp(&bound_bit)) {
                        (1, _) => 1,
                        (_, std::cmp::Ordering::Less) => 1,
                        (_, std::cmp::Ordering::Equal) => 0,
                        (_, std::cmp::Ordering::Greater) => continue,
                    };
                    let mut factor = r_factor[bit];
                    if t < block_bits {
                        let s = s_block[t];
                        factor = factor.scale(if quotient == 1 { s } else { Ext::ONE - s });
                    }
                    let slot = &mut next[doubled - D * quotient][next_below];
                    *slot = slot.add(value.mul(factor));
                }
            }
        }
        states = next;
    }
    let mut total = Kx::ZERO;
    for (remainder, accumulated) in states.iter().enumerate() {
        total = total.add(accumulated[1].scale(tau[remainder]));
    }
    for &value in &r[coordinate_bits..] {
        let r_t = Kx::from_k(value);
        total = total.mul(Kx {
            re: Ext::ONE - r_t.re,
            im: -r_t.im,
        });
    }
    total
}
