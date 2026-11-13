//! The public weights `G` of the witness polynomial `W` and the value that
//! `Σ_x W(x) · G(x)` must have (SPEC.md section 6).
//!
//! For each claim, `G` holds on the witness's cells the weights of its ring
//! equations at `δ`, batched with powers of `ω`: the κ rows of the Ajtai
//! opening, then `Eval_K`, then each `Eval_A`, and on the public cells the
//! powers of `γ`. On the cells of a table's counts it holds the negated
//! lookup inverses. `table` is the prover's `G`. The verifier's `G(ρ)` has
//! three parts: `key_part`, which the proof carries and `key` proves,
//! `matrix_part` and `closed_part`.

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{EqualityWeights, FactoredRows, FactoredSlots};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::layout::{Layout, LANE_VARS, LOOKUPS};
use crate::lookup::Inverses;
use crate::ring::{conjugate, dot, Point};
use crate::{Error, Statement};

pub(crate) const KAPPA: usize = PRODUCTION_VERIFIER_ROWS as usize;

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

/// The multilinear extension of `j ↦ eq(point, j) · [j < count]` at `at`,
/// where `eq(point, j)` is zero when `j` has a bit from `point.len()`. An
/// index is below `count` exactly when, at the highest bit where they
/// differ, the index has 0 and `count` has 1.
pub(crate) fn prefix_equality(point: &[K], count: usize, at: &[K]) -> K {
    let (low, high) = at.split_at(point.len());
    let high: K = high.iter().map(|&coordinate| K::ONE - coordinate).product();
    let terms: Vec<[K; 2]> = point
        .iter()
        .zip(low)
        .map(|(&point, &at)| [(K::ONE - point) * (K::ONE - at), point * at])
        .collect();
    // below[k] = Σ over the bits below k of eq(point, ·) · eq(at, ·).
    let mut below = vec![K::ONE; terms.len() + 1];
    for (bit, [zero, one]) in terms.iter().enumerate() {
        below[bit + 1] = below[bit] * (*zero + *one);
    }
    if count >> terms.len() != 0 {
        return below[terms.len()] * high;
    }
    let (mut sum, mut equal) = (K::ZERO, K::ONE);
    for (bit, [zero, one]) in terms.iter().enumerate().rev() {
        if count >> bit & 1 == 1 {
            sum += equal * *zero * below[bit];
            equal *= *one;
        } else {
            equal *= *zero;
        }
    }
    sum * high
}

/// The powers of `ω` that batch the equations of `statement`.
fn omega_powers(statement: &Statement<'_>, omega: K) -> Vec<K> {
    powers(omega, KAPPA + 2 + statement.claim.eval_a.len())
}

/// The matrices of one claim in the weights: `Lᵀ` of `eq(r, ·)` with the
/// `Eval_A` powers of `ω`.
struct Matrices<'a> {
    rows: FactoredRows<'a>,
    slots: Vec<K>,
}

impl<'a> Matrices<'a> {
    fn new(statement: &'a Statement<'a>, omega: &[K]) -> Result<Self, Error> {
        let rows = statement
            .rows
            .factored()?
            .ok_or(Error::Statement("the matrix rows have no factored form"))?;
        let row_weights = EqualityWeights::new(&statement.claim.r);
        let matrices = statement.claim.eval_a.len();
        let slots = rows.slot_weights(&|row| row_weights.at(row), &omega[KAPPA + 1..][..matrices]);
        Ok(Self { rows, slots })
    }
}

/// `Eval_K` and the public input of one claim in the weights. Block `j` of
/// the witness has the weight `eq(r_hi, j)` in `Eval_K`.
struct Evaluation<'a> {
    /// `ω^κ · conjugate(eq(r_lo, ·))`, the lane part of `Eval_K`.
    lanes: [K; D],
    r_high: &'a [K],
    /// The weight of public cell 0.
    public: K,
}

impl<'a> Evaluation<'a> {
    fn new(statement: &Statement<'a>, omega: &[K]) -> Self {
        let (low, r_high) = statement.claim.r.split_at(LANE_VARS);
        let low: [K; D] = equality(low, D).try_into().expect("D lane weights");
        Self {
            lanes: conjugate(&low).map(|weight| weight * omega[KAPPA]),
            r_high,
            public: omega[KAPPA + 1 + statement.claim.eval_a.len()],
        }
    }
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
            mu * target(
                statement,
                &omega_powers(statement, challenges.omega),
                &point,
                challenges.gamma,
            )
        })
        .sum()
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

/// The prover's `G`, on the prefix of `W` that holds cells. `folded` is
/// `F = Σ_i ω^i · A_i` of the Ajtai key (`key::ProverKey::fold`): lane `h` of
/// block `j` gets `(X^h · F_j)(δ)`, the same for each claim.
pub(crate) fn table(
    statements: &[Statement<'_>],
    layout: &Layout,
    challenges: &Challenges,
    folded: &[K],
) -> Result<Vec<K>, Error> {
    let point = Point::new(challenges.delta);
    let prepared = statements
        .par_iter()
        .zip(&layout.cells)
        .map(|(statement, cells)| {
            let omega = omega_powers(statement, challenges.omega);
            let matrices = Matrices::new(statement, &omega)?;
            let evaluation = Evaluation::new(statement, &omega);
            let blocks = equality(evaluation.r_high, cells.blocks);
            Ok((evaluation, blocks, matrices.rows.column_weights(&matrices.slots)))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut table = vec![K::ZERO; layout.witness_len];
    let mut tables = 0;
    for (((statement, cells), (evaluation, blocks, columns)), mu) in statements
        .iter()
        .zip(&layout.cells)
        .zip(prepared)
        .zip(powers(challenges.mu, statements.len()))
    {
        let witness = &mut table[cells.witness.range()][..cells.blocks * D];
        witness
            .par_chunks_mut(D)
            .zip(columns.par_chunks(D))
            .zip(folded.par_chunks(D))
            .zip(&blocks)
            .for_each(|(((output, columns), key), &block)| {
                let columns = conjugate(columns.try_into().expect("D columns"));
                let element: [K; D] =
                    std::array::from_fn(|lane| evaluation.lanes[lane] * block + columns[lane] + key[lane]);
                for (output, lane) in output.iter_mut().zip(point.lanes(&element)) {
                    *output = mu * lane;
                }
            });
        let mut public = mu * evaluation.public;
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

/// How the witness of each claim enters `G(ρ)`. A linear form `P_j` on the
/// lanes of each block `j` of a claim adds
/// `c · Σ_{j < blocks} eq(β, j) · (E · P_j)(δ)` for `E = Σ_h eq(ρ_lo, h) X^h`.
pub(crate) struct Uses<'a> {
    /// `u` with `Σ_h u(h) · P[h] = (E · P)(δ)`.
    pub(crate) form: [K; D],
    pub(crate) claims: Vec<Use<'a>>,
}

/// The witness of one claim in `G(ρ)`.
pub(crate) struct Use<'a> {
    /// `c`: `μ^claim` times the selector of the claim's witness slice at `ρ`.
    pub(crate) coefficient: K,
    /// `β`: the coordinates of `ρ` that select a block of the slice.
    pub(crate) point: &'a [K],
    pub(crate) blocks: usize,
}

impl<'a> Uses<'a> {
    pub(crate) fn new(layout: &Layout, challenges: &Challenges, rho: &'a [K]) -> Self {
        let lanes: [K; D] = equality(&rho[..LANE_VARS], D)
            .try_into()
            .expect("D lane weights");
        let claims = layout
            .cells
            .iter()
            .zip(powers(challenges.mu, layout.cells.len()))
            .map(|(cells, mu)| Use {
                coefficient: mu * cells.witness.selector(rho),
                point: &rho[LANE_VARS..cells.witness.vars],
                blocks: cells.blocks,
            })
            .collect();
        Self {
            form: Point::new(challenges.delta).lanes(&lanes),
            claims,
        }
    }
}

/// The part of `G(ρ)` that the Ajtai key gives, from `F = Σ_i ω^i · A_i`:
/// `Σ_claims c · Σ_{j < blocks} eq(β, j) · Σ_h u(h) · F(h, j)`. Only the
/// prover computes it. The verifier takes it from the proof, and the
/// sum-check of `key` proves it against the committed key polynomial.
pub(crate) fn key_part(uses: &Uses<'_>, folded: &[K]) -> K {
    uses.claims
        .iter()
        .map(|claim| {
            let blocks: K = folded
                .par_chunks(D)
                .zip(equality(claim.point, claim.blocks))
                .map(|(key, weight)| weight * dot(&uses.form, key))
                .sum();
            claim.coefficient * blocks
        })
        .sum()
}

/// The part of `G(ρ)` that the matrices of the claims give:
/// `Σ_claims c · Σ_row eq(r, row) · Σ_j ω^(κ+1+j) · Σ_entries coefficient · Γ(slot)`,
/// in one pass over the rows of each relation. Claims of one relation have
/// the same blocks, so the same `β` and `Γ`, and share the pass. A verifier
/// that holds a commitment of the matrices replaces the pass over the rows
/// only, as the proof replaces `key_part`.
pub(crate) fn matrix_part(statements: &[Statement<'_>], challenges: &Challenges, uses: &Uses<'_>) -> Result<K, Error> {
    let form = conjugate(&uses.form);
    let mut relations: Vec<(&Statement<'_>, &Use<'_>, Vec<(K, &[K])>)> = Vec::new();
    for (statement, claim) in statements.iter().zip(&uses.claims) {
        let rows = (claim.coefficient, &statement.claim.r[..]);
        match relations.iter_mut().find(|(other, other_claim, _)| {
            std::ptr::addr_eq(other.rows, statement.rows) && other_claim.point == claim.point
        }) {
            Some((_, _, claims)) => claims.push(rows),
            None => relations.push((statement, claim, vec![rows])),
        }
    }
    let mut total = K::ZERO;
    for (statement, claim, claims) in relations {
        let rows = statement
            .rows
            .factored()?
            .ok_or(Error::Statement("the matrix rows have no factored form"))?;
        let omega = omega_powers(statement, challenges.omega);
        let omega = &omega[KAPPA + 1..][..statement.claim.eval_a.len()];
        let slots = slot_values(rows.slots(), &equality(claim.point, claim.blocks), &form);
        let weights = RowWeights::new(&claims);
        total += (0..rows.shape().rows)
            .into_par_iter()
            .with_min_len(1 << 12)
            .map(|row| weights.at(row) * rows.weighted_row(row, omega, &slots))
            .sum::<K>();
    }
    Ok(total)
}

/// `Σ c · eq(r, row)` over the claims `(c, r)` of one relation. Each
/// `eq(r, ·)` is a table of the low bits of `row` times a table of the high
/// bits, and `c` scales the high table.
struct RowWeights(Vec<(Vec<K>, Vec<K>, usize)>);

impl RowWeights {
    fn new(claims: &[(K, &[K])]) -> Self {
        Self(
            claims
                .iter()
                .map(|&(coefficient, point)| {
                    let (low, high) = point.split_at(point.len() / 2);
                    let high = equality(high, 1 << high.len())
                        .into_iter()
                        .map(|weight| coefficient * weight);
                    (equality(low, 1 << low.len()), high.collect(), low.len())
                })
                .collect(),
        )
    }

    fn at(&self, row: usize) -> K {
        self.0
            .iter()
            .map(|(low, high, bits)| low[row & ((1 << bits) - 1)] * high[row >> bits])
            .sum()
    }
}

/// `Γ(s) = Σ_{k < width} ratio^k · Φ(start + k)` for each slot `s` of
/// columns `start..start + width`, where `Φ(64b + h) = eq(β, b) · form[h]`
/// and `blocks[b] = eq(β, b)`. With `(b, h) = divmod(start, 64)`,
/// `Γ(s) = Σ_t eq(β, b + t) · windows[width][t][h]`, so a slot costs one
/// product for each block it meets, not one for each column.
pub(crate) fn slot_values(slots: FactoredSlots<'_>, blocks: &[K], form: &[K; D]) -> Vec<K> {
    let widest = slots.widths.iter().copied().max().map_or(0, usize::from);
    // windows[w][t][h] = Σ over k < w with ⌊(h + k)/64⌋ = t of ratio^k · form[(h + k) mod 64].
    let mut windows: Vec<Vec<[K; D]>> = vec![Vec::new()];
    let mut power = F::ONE;
    for k in 0..widest {
        let mut window = windows[k].clone();
        window.resize(k.div_ceil(D) + 1, [K::ZERO; D]);
        for (lane, column) in (k..k + D).enumerate() {
            window[column / D][lane] += form[column % D].scale_base(power);
        }
        power *= slots.ratio;
        windows.push(window);
    }
    slots
        .starts
        .par_iter()
        .zip(slots.widths)
        .with_min_len(1 << 12)
        .map(|(&start, &width)| {
            let (block, lane) = (start as usize / D, start as usize % D);
            let width = usize::from(width);
            windows[width]
                .iter()
                .take((lane + width).div_ceil(D))
                .zip(&blocks[block..])
                .map(|(window, &weight)| weight * window[lane])
                .sum()
        })
        .collect()
}

/// The parts of `G(ρ)` with a closed form: `Eval_K` and the public input of
/// each claim, and the counts of each table.
pub(crate) fn closed_part(
    statements: &[Statement<'_>],
    layout: &Layout,
    challenges: &Challenges,
    uses: &Uses<'_>,
    rho: &[K],
) -> Result<K, Error> {
    let mut total = K::ZERO;
    let mut tables = 0;
    for ((statement, cells), claim) in statements.iter().zip(&layout.cells).zip(&uses.claims) {
        let evaluation = Evaluation::new(statement, &omega_powers(statement, challenges.omega));
        let pad = dot(&uses.form, &evaluation.lanes) * prefix_equality(claim.point, claim.blocks, evaluation.r_high);
        let public = evaluation.public
            * equality(&rho[..cells.witness.vars], statement.claim.m_in)
                .iter()
                .rev()
                .fold(K::ZERO, |value, &weight| value * challenges.gamma + weight);
        total += claim.coefficient * (pad + public);
        if let Some((bound, counts)) = cells.table {
            let weights = count_weights(bound, challenges.lambda[tables], challenges)?;
            total += counts.selector(rho) * dot(&equality(&rho[..counts.vars], weights.len()), &weights);
            tables += 1;
        }
    }
    Ok(total)
}

#[cfg(test)]
#[path = "../tests/unit/weights.rs"]
mod tests;
