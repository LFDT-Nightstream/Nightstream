//! Closed-form multilinear extensions that the succinct verifier evaluates,
//! written once over `Backend`.
//!
//! Owns every public weight the layer-1 verifier computes without reading the
//! key or the matrices: the scaled-eq point of a power weight, interval and
//! step functions over a sorted index, the identity, `χ_r` at an `Ext` point,
//! the lane-split tables of a geometric run, and the Eval_K (Pad) weight by
//! long division. Points are low-bit first. Each formula has a brute-force
//! test in `tests/internal/mle.rs`.

use neo_math::D;
use p3_field_v08::PrimeCharacteristicRing;

use crate::circuit::algebra::{self, Kx, K};
use crate::circuit::Backend;
use crate::field::Gl;
use crate::Error;

/// `Σ_{x < 2^k} c^x f(x) = κ · f~(x_c)` with `κ = Π_t (1 + c^{2^t})` and
/// `x_t = c^{2^t} / (1 + c^{2^t})`. Rejects when some `1 + c^{2^t}` is zero.
pub(crate) fn scaled_eq_point<B: Backend>(b: &mut B, c: B::E, bits: usize) -> Result<(B::E, Vec<B::E>), Error> {
    let one = algebra::ext_one(b);
    let mut kappa = one;
    let mut point = Vec::with_capacity(bits);
    let mut power = c;
    for _ in 0..bits {
        let denominator = b.ext_add(one, power);
        kappa = b.ext_mul(kappa, denominator);
        let inverse = b.ext_inverse(denominator, "a pole of the lane weights")?;
        point.push(b.ext_mul(power, inverse));
        power = b.ext_mul(power, power);
    }
    Ok((kappa, point))
}

/// `Σ_{x < bound} eq(point, x)`; `bound` may equal `2^point.len()`.
pub(crate) fn less_than<B: Backend>(b: &mut B, point: &[B::E], bound: u64) -> B::E {
    let variables = point.len();
    if bound >> variables != 0 {
        return algebra::ext_one(b);
    }
    let mut total = algebra::ext_zero(b);
    let mut prefix = algebra::ext_one(b);
    for t in (0..variables).rev() {
        let low = algebra::one_minus(b, point[t]);
        if (bound >> t) & 1 == 1 {
            let term = b.ext_mul(prefix, low);
            total = b.ext_add(total, term);
            prefix = b.ext_mul(prefix, point[t]);
        } else {
            prefix = b.ext_mul(prefix, low);
        }
    }
    total
}

/// The MLE of a step function: `values[s]` on `[starts[s], starts[s + 1])`,
/// with `starts[0] = 0` and the last segment ending at `end`; zero after.
pub(crate) fn step<B: Backend>(b: &mut B, point: &[B::E], starts: &[u64], values: &[B::E], end: u64) -> B::E {
    assert_eq!(starts.len(), values.len());
    let mut total = algebra::ext_zero(b);
    let mut below = less_than(b, point, starts.first().copied().unwrap_or(end));
    for (s, &value) in values.iter().enumerate() {
        let stop = starts.get(s + 1).copied().unwrap_or(end);
        let upper = less_than(b, point, stop);
        let interval = b.ext_sub(upper, below);
        let term = b.ext_mul(value, interval);
        total = b.ext_add(total, term);
        below = upper;
    }
    total
}

/// `Σ_t 2^t point_t`, the MLE of `x ↦ x`.
pub(crate) fn identity<B: Backend>(b: &mut B, point: &[B::E]) -> B::E {
    let mut total = algebra::ext_zero(b);
    for &coordinate in point.iter().rev() {
        let doubled = b.ext_add(total, total);
        total = b.ext_add(doubled, coordinate);
    }
    total
}

/// `χ_r` at an `Ext` point of the low `point.len()` bits; the remaining
/// coordinates of `r` see bit 0: `Π_{t<n} eq(r_t, point_t) · Π_{t≥n} (1 - r_t)`.
pub(crate) fn chi<B: Backend>(b: &mut B, r: &[K<B>], point: &[B::E]) -> Kx<B> {
    assert!(point.len() <= r.len());
    let mut total = Kx::one(b);
    for (t, &value) in r.iter().enumerate() {
        let r = Kx::from_k(b, value);
        let factor = match point.get(t) {
            // eq(r, x) = (1 - x) + r·(2x - 1)
            Some(&x) => {
                let twice = b.ext_add(x, x);
                let one = algebra::ext_one(b);
                let slope = b.ext_sub(twice, one);
                let mut factor = r.scale(b, slope);
                let low = algebra::one_minus(b, x);
                factor.re = b.ext_add(factor.re, low);
                factor
            }
            None => r.one_minus(b),
        };
        total = total.mul(b, factor);
    }
    total
}

/// The lane split of a geometric run class `(length ≤ D, ratio)`. For a run
/// starting at lane `l` of block `b` and any `g(c) = E(⌊c/D⌋)·τ_{c mod D}`,
/// `Σ_{k<length} ratio^k g(D·b + l + k) = E(b)·H0[l] + E(b + 1)·H1[l]`.
pub(crate) fn lane_split<B: Backend>(b: &mut B, tau: &[B::E; D], length: usize, ratio: Gl) -> [[B::E; D]; 2] {
    assert!((1..=D).contains(&length));
    let ratio_power = ratio.exp_u64(length as u64);
    let mut head = [tau[D - 1]; D];
    for lane in (0..D - 1).rev() {
        let carried = algebra::ext_scale_constant(b, head[lane + 1], ratio);
        let mut value = b.ext_add(tau[lane], carried);
        if lane + length < D {
            let dropped = algebra::ext_scale_constant(b, tau[lane + length], ratio_power);
            value = b.ext_sub(value, dropped);
        }
        head[lane] = value;
    }
    let mut prefix = [tau[0]; D];
    let mut acc = algebra::ext_zero(b);
    let mut power = Gl::ONE;
    for (k, &value) in tau.iter().enumerate() {
        let term = algebra::ext_scale_constant(b, value, power);
        acc = b.ext_add(acc, term);
        prefix[k] = acc;
        power *= ratio;
    }
    let zero = algebra::ext_zero(b);
    let tail = std::array::from_fn(|lane| match (lane + length).checked_sub(D + 1) {
        Some(last) => algebra::ext_scale_constant(b, prefix[last], ratio.exp_u64((D - lane) as u64)),
        None => zero,
    });
    [head, tail]
}

/// `Σ_{b<blocks} eq(s_block, b) Σ_{l<D} τ_l χ_r(D·b + l)`, by long division of
/// the coordinate by `D`, most significant bit first. The state is the
/// remainder and whether the quotient prefix is already below `blocks`'s;
/// only structurally reachable states are kept.
pub(crate) fn eval_k<B: Backend>(b: &mut B, r: &[K<B>], s_block: &[B::E], tau: &[B::E; D], blocks: usize) -> Kx<B> {
    let block_bits = s_block.len();
    assert!(blocks <= 1 << block_bits);
    let coordinate_bits = (D << block_bits).next_power_of_two().trailing_zeros() as usize;
    assert!(coordinate_bits <= r.len());
    let zero = Kx::zero(b);
    // states[remainder][below], with a structural reachability flag.
    let mut states: Vec<[Option<Kx<B>>; 2]> = vec![[None, None]; D];
    states[0][0] = Some(Kx::one(b));
    for t in (0..coordinate_bits).rev() {
        let r_t = Kx::from_k(b, r[t]);
        let r_factor = [r_t.one_minus(b), r_t];
        let bound_bit = (blocks >> t) & 1;
        let mut next: Vec<[Option<Kx<B>>; 2]> = vec![[None, None]; D];
        for (remainder, accumulated) in states.iter().enumerate() {
            for (below, value) in accumulated.iter().enumerate() {
                let Some(value) = *value else { continue };
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
                        let weight = if quotient == 1 { s } else { algebra::one_minus(b, s) };
                        factor = factor.scale(b, weight);
                    }
                    let product = value.mul(b, factor);
                    let slot = &mut next[doubled - D * quotient][next_below];
                    *slot = Some(match *slot {
                        Some(existing) => existing.add(b, product),
                        None => product,
                    });
                }
            }
        }
        states = next;
    }
    let mut total = zero;
    for (remainder, accumulated) in states.iter().enumerate() {
        if let Some(value) = accumulated[1] {
            let term = value.scale(b, tau[remainder]);
            total = total.add(b, term);
        }
    }
    for &value in &r[coordinate_bits..] {
        let factor = Kx::from_k(b, value).one_minus(b);
        total = total.mul(b, factor);
    }
    total
}
