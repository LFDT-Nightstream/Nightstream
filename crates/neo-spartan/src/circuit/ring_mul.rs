//! The product `a·b mod Φ81` of two ring elements in coefficient form, as one
//! constraint block.
//!
//! Owns: the block's cells and rows. Cells: `a` (54), `b` (54), the values
//! `p_k = a(k)·b(k)` at the points `k < 107`, and the 54 output coefficients.
//! Row `k < 107`: `(Σ_j a_j·k^j)·(Σ_j b_j·k^j) = p_k`. Row `107 + i`:
//! `out_i = Σ_k W_ik·p_k`, where `W` interpolates the product (degree 106)
//! from its values and reduces it modulo `Φ81 = X^54 + X^27 + 1`.

use std::sync::OnceLock;

use neo_math::D;
use p3_field_v08::{Field, PrimeCharacteristicRing};

use super::block::{Entry, A0, B0, C};
use crate::field::Gl;

/// Evaluation points of the product: its degree is `2D - 2`.
const POINTS: usize = 2 * D - 1;
pub(crate) const CELLS: usize = 2 * D + POINTS + D;
pub(crate) const ROWS: usize = POINTS + D;
/// The first output cell.
pub(crate) const OUTPUT: usize = 2 * D + POINTS;
/// `Φ81 = X^54 + X^27 + 1`.
const PHI_MIDDLE: usize = 27;

/// `coefficients mod Φ81`, for at most `POINTS` coefficients.
fn reduce(mut coefficients: Vec<Gl>) -> [Gl; D] {
    for degree in (D..coefficients.len()).rev() {
        let value = coefficients[degree];
        coefficients[degree - PHI_MIDDLE] -= value;
        coefficients[degree - D] -= value;
    }
    std::array::from_fn(|i| coefficients[i])
}

/// `a·b mod Φ81` by schoolbook multiplication.
pub(crate) fn multiply(a: &[Gl], b: &[Gl]) -> [Gl; D] {
    let mut product = vec![Gl::ZERO; POINTS];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            product[i + j] += x * y;
        }
    }
    reduce(product)
}

/// `W[i][k]`: the output coefficient `i` per unit value at point `k`.
fn weights() -> &'static Vec<[Gl; POINTS]> {
    static WEIGHTS: OnceLock<Vec<[Gl; POINTS]>> = OnceLock::new();
    WEIGHTS.get_or_init(|| {
        // Π_j (x - j), low coefficient first.
        let mut full = vec![Gl::ONE];
        for j in 0..POINTS {
            let mut next = vec![Gl::ZERO; full.len() + 1];
            for (degree, &c) in full.iter().enumerate() {
                next[degree + 1] += c;
                next[degree] -= c * Gl::from_usize(j);
            }
            full = next;
        }
        let mut weights = vec![[Gl::ZERO; POINTS]; D];
        for k in 0..POINTS {
            // L_k = Π_{j≠k} (x - j) / (k - j), by synthetic division.
            let point = Gl::from_usize(k);
            let mut quotient = vec![Gl::ZERO; POINTS];
            let mut carry = Gl::ZERO;
            for degree in (0..POINTS).rev() {
                carry = full[degree + 1] + carry * point;
                quotient[degree] = carry;
            }
            let denominator: Gl = (0..POINTS)
                .filter(|&j| j != k)
                .map(|j| point - Gl::from_usize(j))
                .product();
            let scale = denominator.inverse();
            let reduced = reduce(quotient.into_iter().map(|c| c * scale).collect());
            for (row, &value) in weights.iter_mut().zip(&reduced) {
                row[k] = value;
            }
        }
        weights
    })
}

/// `Σ_j values_j·k^j`.
fn evaluate(values: &[Gl], k: usize) -> Gl {
    let point = Gl::from_usize(k);
    values
        .iter()
        .rev()
        .fold(Gl::ZERO, |acc, &v| acc * point + v)
}

/// The cell values of one product of `a` and `b`.
pub(crate) fn trace(a: &[Gl], b: &[Gl]) -> Vec<Gl> {
    let mut cells = Vec::with_capacity(CELLS);
    cells.extend_from_slice(a);
    cells.extend_from_slice(b);
    let products: Vec<Gl> = (0..POINTS)
        .map(|k| evaluate(a, k) * evaluate(b, k))
        .collect();
    cells.extend_from_slice(&products);
    for row in weights() {
        cells.push(row.iter().zip(&products).map(|(&w, &p)| w * p).sum());
    }
    cells
}

/// Every nonzero of the template.
pub(crate) fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    let mut push = |matrix, row, cell, coefficient: Gl| {
        if coefficient != Gl::ZERO {
            entries.push(Entry {
                matrix,
                row,
                cell: Some(cell),
                coefficient,
            });
        }
    };
    for k in 0..POINTS {
        let point = Gl::from_usize(k);
        let mut power = Gl::ONE;
        for j in 0..D {
            push(A0, k, j, power);
            push(B0, k, D + j, power);
            power *= point;
        }
        push(C, k, 2 * D + k, Gl::ONE);
    }
    for (i, row) in weights().iter().enumerate() {
        push(C, POINTS + i, OUTPUT + i, Gl::ONE);
        for (k, &w) in row.iter().enumerate() {
            push(C, POINTS + i, 2 * D + k, -w);
        }
    }
    entries
}
