//! The linear conjuncts of CE(B) as ring rows over `R_F = F[X] / Φ81`.
//!
//! Owns: the one ordered list of rows (commitment rows, Eval_K Re/Im, Eval_A_j
//! Re/Im, public blocks), their batching by powers of λ into one weight
//! `ω_b ∈ Ext[X]` per block, the batched target `Ȳ`, and the quotient by `Φ81`.
//! Every row has the form `Σ_b g_b ⋆ z_b = y`; prover and verifier build the
//! weights with the same function.
//!
//! - Commitment row `i`: `g_b = a_{i,b}` (Ajtai key element), `y = c_i`.
//! - Eval_K (Pad) and Eval_A_j: `g_b = bar(w_b)`, with `w = χ_r` over the
//!   carrier or `w = M_jᵀ χ_r`; the Re and Im coordinates are separate rows.
//!   The Eval_A part of `w` comes from `matrix.rs`.
//! - Public block `j`: `g_b = [b = j]`, `y = X_j`.

use std::sync::OnceLock;

use neo_ajtai::nightstream_fprime_setup::{coefficient_block, PRODUCTION_SEED};
use neo_math::{ring::superneo_bar_matrix, D, F, K};
use p3_field::PrimeCharacteristicRing as _;
use p3_field_v08::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::circuit::algebra;
use crate::circuit::Backend;
use crate::circuit::Native;
use crate::field::{gl, re_im, Ext, Gl};
use crate::{ClaimWords, Shape};

/// Coefficients of `Σ_b ω_b ⋆ z_b` before reduction: degree below `2D - 1`.
const PRODUCT: usize = 2 * D - 1;
/// `Φ81 = X^54 + X^27 + 1`.
const PHI_MIDDLE: usize = 27;

/// Commitment rows, Eval_K (2), Eval_A (2 per matrix), public blocks.
pub(crate) fn row_count(shape: &Shape) -> usize {
    shape.kappa + 2 + 2 * shape.matrices + shape.public_blocks
}

/// Powers of λ in row order. The order here is the only row list.
pub(crate) struct Mixing<E> {
    powers: Vec<E>,
    kappa: usize,
    matrices: usize,
}

impl<E: Copy> Mixing<E> {
    pub(crate) fn new<B: Backend<E = E>>(b: &mut B, lambda: E, shape: &Shape) -> Self {
        let mut powers = vec![algebra::ext_one(b)];
        for _ in 1..row_count(shape) {
            let last = *powers.last().expect("one power");
            powers.push(b.ext_mul(last, lambda));
        }
        Self {
            powers,
            kappa: shape.kappa,
            matrices: shape.matrices,
        }
    }

    pub(crate) fn commitment(&self, row: usize) -> E {
        self.powers[row]
    }

    pub(crate) fn eval_k(&self) -> [E; 2] {
        [self.powers[self.kappa], self.powers[self.kappa + 1]]
    }

    /// The Eval_A projection weights of every matrix.
    pub(crate) fn eval_a(&self) -> Vec<[E; 2]> {
        (0..self.matrices)
            .map(|matrix| {
                let first = self.kappa + 2 + 2 * matrix;
                [self.powers[first], self.powers[first + 1]]
            })
            .collect()
    }

    pub(crate) fn public(&self, block: usize) -> E {
        self.powers[self.kappa + 2 + 2 * self.matrices + block]
    }
}

/// `A_λ(b) = Σ_i λ^i·a_{i,b}` for every block: the commitment rows' part of
/// `ω_b`, one pass over the key.
pub(crate) fn key_weights(shape: &Shape, mixing: &Mixing<Ext>) -> Vec<[Ext; D]> {
    (0..shape.blocks)
        .into_par_iter()
        .map(|block| {
            let mut out = [Ext::ZERO; D];
            for key_row in 0..shape.kappa {
                let power = mixing.commitment(key_row);
                let key = coefficient_block(&PRODUCTION_SEED, key_row as u32, block as u64);
                for (out, &coefficient) in out.iter_mut().zip(&key) {
                    *out += power * Gl::new(coefficient);
                }
            }
            out
        })
        .collect()
}

/// `ω_b` for every block `b < shape.blocks`, from the evaluation point `r`,
/// the Eval_A weight of every coordinate (`eval_a`) and the key part
/// (`key_weights`).
pub(crate) fn block_weights(
    shape: &Shape,
    r: &[K],
    mixing: &Mixing<Ext>,
    eval_a: &[[Ext; D]],
    key: &[[Ext; D]],
) -> Vec<[Ext; D]> {
    let chi = Chi::new(r);
    let eval_k = mixing.eval_k();
    let bar = bar_rows();
    (0..shape.blocks)
        .into_par_iter()
        .map(|block| {
            let weight: [Ext; D] = std::array::from_fn(|lane| {
                project(&mut Native, re_im(chi.at(D * block + lane)), eval_k) + eval_a[block][lane]
            });
            let mut out = key[block];
            for (row, terms) in bar.iter().enumerate() {
                for &(column, sign) in terms {
                    out[row] += weight[column] * sign;
                }
            }
            if block < shape.public_blocks {
                out[0] += mixing.public(block);
            }
            out
        })
        .collect()
}

/// `τ_l = Σ_p bar[p][l]·ζ^p`, so that `<bar(w), (ζ^p)_p> = Σ_l τ_l·w_l`.
pub(crate) fn tau<B: Backend>(b: &mut B, zeta: B::E) -> [B::E; D] {
    let mut tau = [algebra::ext_zero(b); D];
    let mut power = algebra::ext_one(b);
    for terms in bar_rows() {
        for &(column, sign) in terms {
            let term = algebra::ext_scale_constant(b, power, sign);
            tau[column] = b.ext_add(tau[column], term);
        }
        power = b.ext_mul(power, zeta);
    }
    tau
}

/// `Ȳ = Σ_rows λ^row · y_row`, as a ring element.
pub(crate) fn targets<B: Backend>(b: &mut B, statement: &ClaimWords<B::F>, mixing: &Mixing<B::E>) -> [B::E; D] {
    let eval_a = mixing.eval_a();
    std::array::from_fn(|lane| {
        let mut total = algebra::ext_zero(b);
        for (row, coefficients) in statement.commitment.iter().enumerate() {
            let term = b.ext_scale(mixing.commitment(row), coefficients[lane]);
            total = b.ext_add(total, term);
        }
        let term = project(b, statement.eval_k[lane], mixing.eval_k());
        total = b.ext_add(total, term);
        for (values, &weights) in statement.eval_a.iter().zip(&eval_a) {
            let term = project(b, values[lane], weights);
            total = b.ext_add(total, term);
        }
        for (block, values) in statement.public.iter().enumerate() {
            let term = b.ext_scale(mixing.public(block), values[lane]);
            total = b.ext_add(total, term);
        }
        total
    })
}

/// `weights[0]·Re(value) + weights[1]·Im(value)`: one F-linear projection of
/// a `K` value. A `K` equation over F-valued data holds exactly when both
/// coordinates hold, so distinct weights per coordinate lose nothing.
pub(crate) fn project<B: Backend>(b: &mut B, value: algebra::K<B>, weights: [B::E; 2]) -> B::E {
    let re = b.ext_scale(weights[0], value[0]);
    let im = b.ext_scale(weights[1], value[1]);
    b.ext_add(re, im)
}

/// `Σ_b ω_b ⋆ z_b - Ȳ = Q·Φ81 + remainder` over `Ext[X]`. The remainder is
/// zero exactly when the batched row holds; the prover does not check it.
pub(crate) fn divide(weights: &[[Ext; D]], z: &[Gl], lanes: usize, target: &[Ext; D]) -> ([Ext; D - 1], [Ext; D]) {
    let mut product = weights
        .par_iter()
        .enumerate()
        .fold(
            || [Ext::ZERO; PRODUCT],
            |mut acc, (block, weight)| {
                for lane in 0..D {
                    let value = z[lanes * block + lane];
                    if value != Gl::ZERO {
                        for (k, &w) in weight.iter().enumerate() {
                            acc[k + lane] += w * value;
                        }
                    }
                }
                acc
            },
        )
        .reduce(|| [Ext::ZERO; PRODUCT], |a, b| std::array::from_fn(|k| a[k] + b[k]));
    for (coefficient, &value) in product.iter_mut().zip(target) {
        *coefficient -= value;
    }
    let mut quotient = [Ext::ZERO; D - 1];
    for degree in (D..PRODUCT).rev() {
        let q = product[degree];
        quotient[degree - D] = q;
        product[degree] -= q;
        product[degree - PHI_MIDDLE] -= q;
        product[degree - D] -= q;
    }
    (quotient, std::array::from_fn(|k| product[k]))
}

/// `Ȳ(ζ) + Q(ζ)·Φ81(ζ)`: the value of `Σ_b ω_b(ζ)·z_b(ζ)` the quotient claims.
pub(crate) fn lifted_target<B: Backend>(b: &mut B, target: &[B::E; D], quotient: &[B::E], zeta: B::E) -> B::E {
    let mut middle = zeta;
    for _ in 1..PHI_MIDDLE {
        middle = b.ext_mul(middle, zeta);
    }
    let top = b.ext_mul(middle, middle);
    let one = algebra::ext_one(b);
    let sum = b.ext_add(top, middle);
    let phi = b.ext_add(sum, one);
    let target = algebra::horner(b, target, zeta);
    let quotient = algebra::horner(b, quotient, zeta);
    let lifted = b.ext_mul(quotient, phi);
    b.ext_add(target, lifted)
}

/// `Ω_b = ω_b(ζ)` for every block.
pub(crate) fn evaluate(weights: &[[Ext; D]], zeta: Ext) -> Vec<Ext> {
    weights
        .par_iter()
        .map(|weight| horner(weight, zeta))
        .collect()
}

fn horner(coefficients: &[Ext], x: Ext) -> Ext {
    coefficients
        .iter()
        .rev()
        .fold(Ext::ZERO, |acc, &c| acc * x + c)
}

/// The `bar` transform as signed sparse rows (every entry is 0 or ±1).
fn bar_rows() -> &'static [Vec<(usize, Gl)>; D] {
    static ROWS: OnceLock<[Vec<(usize, Gl)>; D]> = OnceLock::new();
    ROWS.get_or_init(|| {
        let matrix = superneo_bar_matrix();
        std::array::from_fn(|row| {
            (0..D)
                .filter(|&column| matrix[row][column] != F::ZERO)
                .map(|column| (column, gl(matrix[row][column])))
                .collect()
        })
    })
}

/// `χ_r(c)` for `c < 2^r.len()` from a low and a high tensor table.
struct Chi {
    low: Vec<K>,
    high: Vec<K>,
    split: usize,
}

impl Chi {
    fn new(point: &[K]) -> Self {
        let split = point.len() / 2;
        Self {
            low: neo_ccs::utils::tensor_point(&point[..split]),
            high: neo_ccs::utils::tensor_point(&point[split..]),
            split,
        }
    }

    fn at(&self, index: usize) -> K {
        self.low[index & ((1 << self.split) - 1)] * self.high[index >> self.split]
    }
}
