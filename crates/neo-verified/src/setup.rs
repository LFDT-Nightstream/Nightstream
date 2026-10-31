//! The verifier's check of one queried setup leaf, over any backend.
//!
//! A leaf holds `2^FOLD` coset values of every setup oracle and its authentication path,
//! bottom first. Its index is a little-endian list of bits.
//!
//! `formal/compression-lean/CompressionLeanRust/Setup.lean` proves `path_ok`, run on `Native`,
//! equal to `Replay.pathOk`, the path check that the Merkle binding reads, and `check`, run on
//! the copies' `Build`, equal to `Circuit.setupCheck`.

use crate::algebra::{constant, dot, ext_one, ext_scale_constant, ext_zero, power_from_bits, scale};
use crate::backend::{Backend, Rejected};
use crate::degrees::FOLD;
use crate::field::{gl_inv, gl_pow, reduce, two_adic_generator};
use crate::hash::{hash_leaf, merkle_root, Digest};

/// The coset values of one oracle in a leaf.
const LEAF: usize = 1 << FOLD;

/// Plonky3's `Goldilocks::GENERATOR`.
const GENERATOR: u64 = 7;

/// Whether the leaf with `values` at the index with little-endian `bits` reaches `root` along
/// `path`.
pub fn path_ok<B: Backend>(
    b: &mut B,
    root: Digest<B>,
    bits: &[B::F],
    values: &[B::F],
    path: &[Digest<B>],
) -> Result<(), Rejected> {
    let digest = hash_leaf(b, values);
    let computed = merkle_root(b, digest, bits, path);
    b.assert_equal(computed[0], root[0], "setup leaf path")?;
    b.assert_equal(computed[1], root[1], "setup leaf path")?;
    b.assert_equal(computed[2], root[2], "setup leaf path")?;
    b.assert_equal(computed[3], root[3], "setup leaf path")
}

/// Check the leaf with `values` and `path` at the index with little-endian `bits` against
/// `root`. The groups of per-oracle weights cover the oracles in order. For each group, return
/// the folded value `Σ_u α^[u]·Ô_u(q)` of the combined oracle: the value at `q` of its fold
/// by `alpha`.
#[allow(clippy::too_many_arguments)]
pub fn check<B: Backend>(
    b: &mut B,
    root: Digest<B>,
    variables: usize,
    bits: &[B::F],
    values: &[B::F],
    path: &[Digest<B>],
    groups: &[Vec<B::E>],
    alpha: &[B::E; FOLD],
) -> Result<Vec<B::E>, Rejected> {
    let mut oracles = 0usize;
    let mut g = 0;
    while g < groups.len() {
        oracles = oracles.checked_add(groups[g].len()).unwrap();
        g = g.wrapping_add(1);
    }
    assert!(oracles.checked_mul(LEAF).unwrap() == values.len());
    let order = gl_inv(two_adic_generator(variables.checked_add(1).unwrap()));
    path_ok(b, root, bits, values, path)?;
    // Ô_u(q) = 8^{-1}·y^{-u}·Σ_v ω_8^{-uv}·Ô(y·ω_8^v), so the folded value is
    // Σ_v weight[v]·Ô(y·ω_8^v) with weight[v] = Σ_u α^[u]·8^{-1}·y^{-u}·ω_8^{-uv}.
    // y^{-1} = g^{-1}·ω^{-index}, with ω of order 2^{n+1}.
    let rotation = power_from_bits(b, order, bits);
    let y_inverse = scale(b, rotation, gl_inv(reduce(GENERATOR)));
    let mut monomials = Vec::new();
    monomials.push(ext_one(b));
    let mut i = 0;
    while i < FOLD {
        let mut scaled = Vec::new();
        let mut j = 0;
        while j < monomials.len() {
            scaled.push(b.ext_mul(monomials[j], alpha[i]));
            j = j.wrapping_add(1);
        }
        let mut k = 0;
        while k < scaled.len() {
            monomials.push(scaled[k]);
            k = k.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    let mut step = constant(b, gl_inv(reduce(LEAF as u64)));
    let mut coefficients = Vec::new();
    let mut m = 0;
    while m < monomials.len() {
        coefficients.push(b.ext_scale(monomials[m], step));
        step = b.mul(step, y_inverse);
        m = m.wrapping_add(1);
    }
    let omega = two_adic_generator(FOLD);
    let mut weight = Vec::new();
    let mut v = 0;
    while v < LEAF {
        let mut total = ext_zero(b);
        let mut u = 0;
        while u < coefficients.len() {
            // `(LEAF - u)·v mod LEAF`, with `LEAF` a power of two.
            let exponent = LEAF.wrapping_sub(u).checked_mul(v).unwrap() & LEAF.wrapping_sub(1);
            let term = ext_scale_constant(b, coefficients[u], gl_pow(omega, exponent as u64));
            total = b.ext_add(total, term);
            u = u.wrapping_add(1);
        }
        weight.push(total);
        v = v.wrapping_add(1);
    }
    let mut oracle = 0usize;
    let mut folded = Vec::new();
    let mut g = 0;
    while g < groups.len() {
        let zero = ext_zero(b);
        let mut combined = Vec::new();
        let mut v = 0;
        while v < LEAF {
            combined.push(zero);
            v = v.wrapping_add(1);
        }
        let mut k = 0;
        while k < groups[g].len() {
            let start = oracle.checked_mul(LEAF).unwrap();
            oracle = oracle.wrapping_add(1);
            let mut v = 0;
            while v < LEAF {
                let term = b.ext_scale(groups[g][k], values[start.wrapping_add(v)]);
                let sum = b.ext_add(combined[v], term);
                combined[v] = sum;
                v = v.wrapping_add(1);
            }
            k = k.wrapping_add(1);
        }
        folded.push(dot(b, &weight, &combined));
        g = g.wrapping_add(1);
    }
    Ok(folded)
}
