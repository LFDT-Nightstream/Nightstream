//! The public weights `G` of the witness polynomial `W` and the value that
//! `Σ_x W(x) · G(x)` must have (SPEC.md section 6).

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::{EqualityWeights, FactoredSlots};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::layout::{Layout, LANE_VARS, LOOKUPS};
use crate::lookup::Inverses;
use crate::ring::{conjugate, dot, Point};
use crate::{Error, Statement};

pub(crate) const KAPPA: usize = PRODUCTION_VERIFIER_ROWS as usize;

pub(crate) struct Challenges {
    pub(crate) taus: [F; LOOKUPS],
    pub(crate) omega: K,
    pub(crate) delta: K,
    pub(crate) gamma: K,
    pub(crate) mu: K,
    pub(crate) psi: [K; LOOKUPS],
    pub(crate) xi: Vec<K>,
    pub(crate) lambda: Vec<K>,
    pub(crate) beta: Vec<K>,
}

pub(crate) fn powers(base: K, count: usize) -> Vec<K> {
    std::iter::successors(Some(K::ONE), |&power| Some(power * base))
        .take(count)
        .collect()
}

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

pub(crate) fn prefix_equality(point: &[K], count: usize, at: &[K]) -> K {
    let (low, high) = at.split_at(point.len());
    let high: K = high.iter().map(|&coordinate| K::ONE - coordinate).product();
    let terms: Vec<[K; 2]> = point
        .iter()
        .zip(low)
        .map(|(&point, &at)| [(K::ONE - point) * (K::ONE - at), point * at])
        .collect();
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

fn omega_powers(statement: &Statement<'_>, omega: K) -> Vec<K> {
    powers(omega, KAPPA + 2 + statement.claim.eval_a.len())
}

const COLUMNS_PER_TASK: usize = 1 << 16;

/// `Mᵀ = Gᵀ · Lᵀ` (Lean `SlotLayout.layout`, `LogicalPlan.plan`).
struct Matrices<'a> {
    slots: FactoredSlots<'a>,
    weights: Vec<K>,
    met: Vec<Vec<Vec<u32>>>,
}

impl<'a> Matrices<'a> {
    fn new(statement: &'a Statement<'a>, omega: &[K]) -> Result<Self, Error> {
        let rows = statement
            .rows
            .factored()?
            .ok_or(Error::Statement("the matrix rows have no factored form"))?;
        let row_weights = EqualityWeights::new(&statement.claim.r);
        let matrices = statement.claim.eval_a.len();
        let weights = rows.slot_weights(&|row| row_weights.at(row), &omega[KAPPA + 1..][..matrices]);
        Ok(Self::of_slots(rows.slots(), weights, rows.shape().columns))
    }

    fn of_slots(slots: FactoredSlots<'a>, weights: Vec<K>, columns: usize) -> Self {
        let tasks = columns.div_ceil(COLUMNS_PER_TASK);
        let met = slots
            .starts
            .par_iter()
            .zip(slots.widths)
            .enumerate()
            .with_min_len(1 << 16)
            .fold(
                || vec![Vec::new(); tasks],
                |mut met, (slot, (&start, &width))| {
                    let end = start as usize + usize::from(width);
                    for task in start as usize / COLUMNS_PER_TASK..end.div_ceil(COLUMNS_PER_TASK) {
                        met[task].push(slot as u32);
                    }
                    met
                },
            )
            .collect();
        Self { slots, weights, met }
    }

    fn add_columns(&self, task: usize, output: &mut [K]) {
        let first = task * COLUMNS_PER_TASK;
        for &slot in self.met.iter().flat_map(|met| &met[task]) {
            let slot = slot as usize;
            let mut term = self.weights[slot];
            for column in (self.slots.starts[slot] as usize..).take(usize::from(self.slots.widths[slot])) {
                if let Some(cell) = column
                    .checked_sub(first)
                    .and_then(|column| output.get_mut(column))
                {
                    *cell += term;
                }
                term = term.scale_base(self.slots.ratio);
            }
        }
    }
}

struct Evaluation<'a> {
    lanes: [K; D],
    r_high: &'a [K],
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

fn count_weights(bound: u64, lambda: K, challenges: &Challenges) -> Result<Vec<K>, Error> {
    let inverses = Inverses::new(bound, &challenges.taus)?;
    Ok(inverses
        .combine(&challenges.psi)
        .into_iter()
        .map(|weight| -(lambda * weight))
        .collect())
}

pub(crate) fn table(
    statements: &[Statement<'_>],
    layout: &Layout,
    challenges: &Challenges,
    folded: &[K],
) -> Result<Vec<K>, Error> {
    let point = Point::new(challenges.delta);
    let prepare = || {
        statements
            .par_iter()
            .zip(&layout.cells)
            .map(|(statement, cells)| {
                let omega = omega_powers(statement, challenges.omega);
                let matrices = Matrices::new(statement, &omega)?;
                let evaluation = Evaluation::new(statement, &omega);
                let blocks = equality(evaluation.r_high, cells.blocks);
                Ok((evaluation, blocks, matrices))
            })
            .collect::<Result<Vec<_>, Error>>()
    };
    let (prepared, mut table) = rayon::join(prepare, || vec![K::ZERO; layout.witness_len]);
    let mut tables = 0;
    for (((statement, cells), (evaluation, blocks, matrices)), mu) in statements
        .iter()
        .zip(&layout.cells)
        .zip(prepared?)
        .zip(powers(challenges.mu, statements.len()))
    {
        let witness = &mut table[cells.witness.range()][..cells.blocks * D];
        witness
            .par_chunks_mut(COLUMNS_PER_TASK)
            .enumerate()
            .for_each_init(Vec::new, |columns, (task, output)| {
                columns.clear();
                columns.resize(output.len(), K::ZERO);
                matrices.add_columns(task, columns);
                let first = task * COLUMNS_PER_TASK;
                let (key, blocks) = (&folded[first..], &blocks[first / D..]);
                for (((output, columns), key), &block) in output
                    .chunks_mut(D)
                    .zip(columns.chunks(D))
                    .zip(key.chunks(D))
                    .zip(blocks)
                {
                    let columns = conjugate(columns.try_into().expect("D columns"));
                    let element: [K; D] =
                        std::array::from_fn(|lane| evaluation.lanes[lane] * block + columns[lane] + key[lane]);
                    for (output, lane) in output.iter_mut().zip(point.lanes(&element)) {
                        *output = mu * lane;
                    }
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

pub(crate) struct Uses<'a> {
    pub(crate) form: [K; D],
    pub(crate) claims: Vec<Use<'a>>,
}

pub(crate) struct Use<'a> {
    pub(crate) coefficient: K,
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

pub(crate) fn slot_values(slots: FactoredSlots<'_>, blocks: &[K], form: &[K; D]) -> Vec<K> {
    let widest = slots.widths.iter().copied().max().map_or(0, usize::from);
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
