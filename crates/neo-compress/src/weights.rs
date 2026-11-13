//! The public weights `G` of the witness polynomial `W` and the value that
//! `Σ_x W(x) · G(x)` must have (SPEC.md section 6).
//!
//! For each claim, `G` holds on the witness's cells the weights of its ring
//! equations at `δ`, batched with powers of `ω`: the κ rows of the Ajtai
//! opening, then `Eval_K`, then each `Eval_A`, and on the public cells the
//! powers of `γ`. On the cells of a table's counts it holds the negated
//! lookup inverses. `table` is the prover's `G`, and `evaluate` is the
//! verifier's `G(ρ)`.

use neo_ajtai::nightstream_fprime_setup::{production_key_pair, PRODUCTION_VERIFIER_ROWS};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{EqualityWeights, FactoredRows};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::layout::{Cells, Layout, LOOKUPS};
use crate::lookup::Inverses;
use crate::ring::{conjugate, dot, Point};
use crate::{Error, Statement};

pub(crate) const KAPPA: usize = PRODUCTION_VERIFIER_ROWS as usize;
const LANE_VARS: usize = D.trailing_zeros() as usize;

/// The verifier's challenges after both commitments.
pub(crate) struct Challenges {
    /// The lookup challenges, drawn before the inverses are committed.
    pub(crate) taus: [F; LOOKUPS],
    /// Batches the ring equations and the public input of one claim.
    pub(crate) omega: K,
    /// The point at which each ring equation is read.
    pub(crate) delta: K,
    /// Batches the cells of the public input.
    pub(crate) gamma: K,
    /// Batches the claims.
    pub(crate) mu: K,
    /// Batches the lookups of the challenges.
    pub(crate) psi: [K; LOOKUPS],
    /// For each table, the weight of its inverse check and of its sum.
    pub(crate) xi: Vec<K>,
    pub(crate) lambda: Vec<K>,
    /// The point of the inverse check.
    pub(crate) beta: Vec<K>,
}

pub(crate) fn powers(base: K, count: usize) -> Vec<K> {
    std::iter::successors(Some(K::ONE), |&power| Some(power * base))
        .take(count)
        .collect()
}

/// `eq(point, i)` for each `i` below `count`: bit `b` of `i` goes with
/// `point[b]`, and `i` has no bit from `point.len()`.
pub(crate) fn equality(point: &[K], count: usize) -> Vec<K> {
    let vars = count.next_power_of_two().trailing_zeros() as usize;
    let (low, high) = point.split_at(vars);
    let mut table = vec![high.iter().map(|&coordinate| K::ONE - coordinate).product()];
    for &coordinate in low {
        let width = table.len();
        table.resize(2 * width, K::ZERO);
        let (zeros, ones) = table.split_at_mut(width);
        for (zero, one) in zeros.iter_mut().zip(ones) {
            *one = *zero * coordinate;
            *zero -= *one;
        }
    }
    table.truncate(count);
    table
}

/// What one claim adds to the weights besides the Ajtai rows.
struct Evaluations<'a> {
    rows: FactoredRows<'a>,
    /// `ω^κ · conjugate(eq(r_lo, ·))`, the lane part of `Eval_K`.
    lanes: [K; D],
    /// `eq(r_hi, j)` for each block `j`.
    blocks: Vec<K>,
    /// `Lᵀ` of `eq(r, ·)` with the `Eval_A` powers of `ω`.
    slots: Vec<K>,
    /// The weight of public cell 0.
    public: K,
}

impl<'a> Evaluations<'a> {
    fn new(statement: &'a Statement<'a>, cells: &Cells, omega: K) -> Result<Self, Error> {
        let claim = statement.claim;
        let rows = statement
            .rows
            .factored()?
            .ok_or(Error::Statement("the matrix rows have no factored form"))?;
        let (low, high) = claim.r.split_at(LANE_VARS);
        let low: [K; D] = equality(low, D).try_into().expect("D lane weights");
        let row_weights = EqualityWeights::new(&claim.r);
        let matrices = claim.eval_a.len();
        let omega = powers(omega, KAPPA + 2 + matrices);
        Ok(Self {
            rows,
            lanes: conjugate(&low).map(|weight| weight * omega[KAPPA]),
            blocks: equality(high, cells.blocks),
            slots: rows.slot_weights(&|row| row_weights.at(row), &omega[KAPPA + 1..][..matrices]),
            public: omega[KAPPA + 1 + matrices],
        })
    }
}

/// `prepare` of each claim's `Evaluations`, on other threads than the pass
/// over the Ajtai key that `key` makes.
fn with_key_pass<'a, T: Send>(
    statements: &'a [Statement<'a>],
    layout: &Layout,
    challenges: &Challenges,
    key: impl FnOnce(usize) -> Vec<K> + Send,
    prepare: impl Fn(Evaluations<'a>, &Cells) -> T + Sync,
) -> Result<(Vec<K>, Vec<T>), Error> {
    let max_blocks = layout
        .cells
        .iter()
        .map(|cells| cells.blocks)
        .max()
        .unwrap_or(0);
    let (ajtai, prepared) = rayon::join(
        || key(max_blocks),
        || {
            statements
                .par_iter()
                .zip(&layout.cells)
                .map(|(statement, cells)| Ok(prepare(Evaluations::new(statement, cells, challenges.omega)?, cells)))
                .collect::<Result<Vec<_>, Error>>()
        },
    );
    Ok((ajtai, prepared?))
}

/// The value that the weights of claim `statement` give on its witness: its
/// ring equations at `δ` and its public input at `γ`.
fn target(statement: &Statement<'_>, omega: &[K], point: &Point, gamma: K) -> K {
    let claim = statement.claim;
    let commitment = claim
        .c
        .data()
        .chunks_exact(D)
        .map(|row| point.value_base(row));
    let evaluations = std::iter::once(&claim.eval_k)
        .chain(&claim.eval_a)
        .map(|family| point.value(&family[..D]));
    let public = claim.X.to_dense_vec();
    let columns = claim.X.cols();
    let public = (0..claim.m_in).rev().fold(K::ZERO, |value, cell| {
        value * gamma + public[(cell % D) * columns + cell / D]
    });
    dot(
        omega,
        &commitment
            .chain(evaluations)
            .chain([public])
            .collect::<Vec<_>>(),
    )
}

/// `Σ_claims μ^claim · target`.
pub(crate) fn targets(statements: &[Statement<'_>], challenges: &Challenges) -> K {
    let point = Point::new(challenges.delta);
    statements
        .iter()
        .zip(powers(challenges.mu, statements.len()))
        .map(|(statement, mu)| {
            let omega = powers(challenges.omega, KAPPA + 2 + statement.claim.eval_a.len());
            mu * target(statement, &omega, &point, challenges.gamma)
        })
        .sum()
}

/// For each block below `blocks`, `width` values that `fill` makes from the
/// block's κ key elements. One pass regenerates the key from SHAKE128.
fn key_pass(blocks: usize, width: usize, fill: impl Fn(&[[F; D]; KAPPA], &mut [K]) + Sync) -> Vec<K> {
    let mut output = vec![K::ZERO; blocks * width];
    output
        .par_chunks_mut(2 * width)
        .enumerate()
        .for_each(|(pair, output)| {
            let columns = [2 * pair as u64, (2 * pair + 1).min(blocks - 1) as u64];
            let keys: [[[u64; D]; 2]; KAPPA] = std::array::from_fn(|row| production_key_pair(row as u32, columns));
            for (slot, output) in output.chunks_mut(width).enumerate() {
                fill(&std::array::from_fn(|row| keys[row][slot].map(F::new)), output);
            }
        });
    output
}

fn split(values: &[K]) -> (Vec<F>, Vec<F>) {
    values
        .iter()
        .map(|value| {
            let [real, imaginary] = value.as_coeffs();
            (real, imaginary)
        })
        .unzip()
}

/// `Σ weights[i] · values[i]` with base-field `values`.
fn combine(weights: &(Vec<F>, Vec<F>), values: impl Iterator<Item = F> + Clone) -> K {
    K::from_coeffs([
        F::sum_of_products(weights.0.iter().copied().zip(values.clone())),
        F::sum_of_products(weights.1.iter().copied().zip(values)),
    ])
}

/// The count cells of a table: cell `t` has `−λ · Σ_k ψ_k / (τ_k − (t − bound))`.
fn count_weights(bound: u64, lambda: K, challenges: &Challenges) -> Result<Vec<K>, Error> {
    let inverses = Inverses::new(bound, &challenges.taus)?;
    Ok(inverses
        .combine(&challenges.psi)
        .into_iter()
        .map(|weight| -(lambda * weight))
        .collect())
}

/// The prover's `G`, on the prefix of `W` that holds cells.
pub(crate) fn table(statements: &[Statement<'_>], layout: &Layout, challenges: &Challenges) -> Result<Vec<K>, Error> {
    let point = Point::new(challenges.delta);
    let omega = split(&powers(challenges.omega, KAPPA));
    let (ajtai, prepared) = with_key_pass(
        statements,
        layout,
        challenges,
        // Block j, lane h: (X^h · Σ_i ω^i A_ij)(δ), the same for each claim.
        |blocks| {
            key_pass(blocks, D, |keys, output| {
                let batched: [K; D] = std::array::from_fn(|lane| combine(&omega, keys.iter().map(|key| key[lane])));
                output.copy_from_slice(&point.lanes(&batched));
            })
        },
        |evaluations, _| {
            let columns = evaluations.rows.column_weights(&evaluations.slots);
            (evaluations, columns)
        },
    )?;
    let mut table = vec![K::ZERO; layout.witness_len];
    let mut tables = 0;
    for (((statement, cells), (evaluations, columns)), mu) in statements
        .iter()
        .zip(&layout.cells)
        .zip(prepared)
        .zip(powers(challenges.mu, statements.len()))
    {
        let witness = &mut table[cells.witness.range()][..cells.blocks * D];
        witness
            .par_chunks_mut(D)
            .zip(columns.par_chunks(D))
            .zip(ajtai.par_chunks(D))
            .zip(&evaluations.blocks)
            .for_each(|(((output, columns), ajtai), &block)| {
                let columns = conjugate(columns.try_into().expect("D columns"));
                let element: [K; D] = std::array::from_fn(|lane| evaluations.lanes[lane] * block + columns[lane]);
                for ((output, lane), &ajtai) in output.iter_mut().zip(point.lanes(&element)).zip(ajtai) {
                    *output = mu * (lane + ajtai);
                }
            });
        let mut public = mu * evaluations.public;
        for output in &mut witness[..statement.claim.m_in] {
            *output += public;
            public *= challenges.gamma;
        }
        if let Some((bound, counts)) = cells.table {
            let weights = count_weights(bound, challenges.lambda[tables], challenges)?;
            table[counts.range()][..weights.len()].copy_from_slice(&weights);
            tables += 1;
        }
    }
    Ok(table)
}

/// The verifier's `G(ρ)`, in time linear in the key and in the matrices: one
/// pass over the Ajtai key and one over the rows of each claim. A holographic
/// verifier replaces this function only.
pub(crate) fn evaluate(
    statements: &[Statement<'_>],
    layout: &Layout,
    challenges: &Challenges,
    rho: &[K],
) -> Result<K, Error> {
    let point = Point::new(challenges.delta);
    // P ↦ (E · P)(δ) for E = Σ_h eq(ρ_lo, h) X^h, the lane part of each block.
    let lane_weights: [K; D] = equality(&rho[..LANE_VARS], D)
        .try_into()
        .expect("D lane weights");
    let form = point.lanes(&lane_weights);
    let conjugate_form = conjugate(&form);
    let omega = powers(challenges.omega, KAPPA);
    let split_form = split(&form);
    let (ajtai, prepared) = with_key_pass(
        statements,
        layout,
        challenges,
        |blocks| {
            key_pass(blocks, 1, |keys, output| {
                output[0] = keys
                    .iter()
                    .zip(&omega)
                    .map(|(key, &omega)| omega * combine(&split_form, key.iter().copied()))
                    .sum();
            })
        },
        // The matrices and `Eval_K` of the claim, and `eq(ρ_hi, j)` of each block.
        |evaluations, cells| {
            let blocks = equality(&rho[LANE_VARS..cells.witness.vars], cells.blocks);
            let mut projection = vec![K::ZERO; cells.blocks * D];
            projection
                .par_chunks_mut(D)
                .zip(&blocks)
                .for_each(|(output, &block)| {
                    for (output, &weight) in output.iter_mut().zip(&conjugate_form) {
                        *output = block * weight;
                    }
                });
            let matrices = dot(&evaluations.slots, &evaluations.rows.projected_slot_values(&projection));
            let pad = dot(&form, &evaluations.lanes) * dot(&blocks, &evaluations.blocks);
            (matrices + pad, evaluations.public, blocks)
        },
    )?;
    let mut total = K::ZERO;
    let mut tables = 0;
    for (((statement, cells), (evaluations, public, blocks)), mu) in statements
        .iter()
        .zip(&layout.cells)
        .zip(prepared)
        .zip(powers(challenges.mu, statements.len()))
    {
        let public = public
            * equality(&rho[..cells.witness.vars], statement.claim.m_in)
                .iter()
                .rev()
                .fold(K::ZERO, |value, &weight| value * challenges.gamma + weight);
        let witness = dot(&blocks, &ajtai[..cells.blocks]) + evaluations + public;
        total += mu * cells.witness.selector(rho) * witness;
        if let Some((bound, counts)) = cells.table {
            let weights = count_weights(bound, challenges.lambda[tables], challenges)?;
            total += counts.selector(rho) * dot(&equality(&rho[..counts.vars], weights.len()), &weights);
            tables += 1;
        }
    }
    Ok(total)
}
