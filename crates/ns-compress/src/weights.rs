//! The public weights `G` of the witness polynomial `W` and the value that
//! `Σ_x W(x) · G(x)` must have (SPEC.md section 6).

use std::ops::ControlFlow;

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS;
use neo_ccs::GeometricRowRun;
use neo_math::{KExtensions, ProductSum, D, F, K};
use neo_reductions::superneo_eval::{EqualityWeights, FactoredSlots, MatrixRowSink};
use neo_reductions::PiCcsError;
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

pub(crate) const COLUMNS_PER_TASK: usize = 1 << 16;
const _: () = assert!(COLUMNS_PER_TASK.is_multiple_of(D));

/// `Mᵀ = Gᵀ · Lᵀ` (Lean `SlotLayout.layout`, `LogicalPlan.plan`).
struct Matrices<'a> {
    slots: FactoredSlots<'a>,
    weights: Vec<K>,
    slots_per_task: Vec<Vec<Vec<u32>>>,
    runs: Option<std::borrow::Cow<'a, neo_reductions::superneo_eval::SlotRuns>>,
    powers: [F; 256],
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
        Ok(Self::with_runs(
            rows.slots(),
            weights,
            rows.shape().columns,
            rows.slot_runs(),
        ))
    }

    #[cfg(test)]
    fn of_slots(slots: FactoredSlots<'a>, weights: Vec<K>, columns: usize) -> Self {
        let runs =
            neo_reductions::superneo_eval::SlotRuns::new(slots.starts, slots.widths).map(std::borrow::Cow::Owned);
        Self::with_runs(slots, weights, columns, runs)
    }

    fn with_runs(
        slots: FactoredSlots<'a>,
        weights: Vec<K>,
        columns: usize,
        runs: Option<std::borrow::Cow<'a, neo_reductions::superneo_eval::SlotRuns>>,
    ) -> Self {
        let tasks = columns.div_ceil(COLUMNS_PER_TASK);
        let mut powers = [F::ONE; 256];
        for i in 1..powers.len() {
            powers[i] = powers[i - 1] * slots.ratio;
        }
        let slots_per_task = if runs.is_some() {
            Vec::new()
        } else {
            slots
                .starts
                .par_iter()
                .zip(slots.widths)
                .enumerate()
                .with_min_len(1 << 16)
                .fold(
                    || vec![Vec::new(); tasks],
                    |mut part, (slot, (&start, &width))| {
                        if width == 0 {
                            return part;
                        }
                        let end = start as usize + usize::from(width);
                        for task in start as usize / COLUMNS_PER_TASK..end.div_ceil(COLUMNS_PER_TASK) {
                            part[task].push(slot as u32);
                        }
                        part
                    },
                )
                .collect()
        };
        Self {
            slots,
            weights,
            slots_per_task,
            runs,
            powers,
        }
    }

    fn add_columns(&self, task: usize, output: &mut [K]) {
        let first = task * COLUMNS_PER_TASK;
        if let Some(runs) = &self.runs {
            crate::column_runs::write(runs, &self.weights, first, output, &self.powers);
            return;
        }
        output.fill(K::ZERO);
        for &slot in self.slots_per_task.iter().flat_map(|part| &part[task]) {
            let slot = slot as usize;
            let weight = self.weights[slot];
            if weight == K::ZERO {
                continue;
            }
            let start = self.slots.starts[slot] as usize;
            let width = usize::from(self.slots.widths[slot]);
            let lo = start.max(first);
            let hi = (start + width).min(first + output.len());
            if lo < hi {
                for (value, &scale) in output[lo - first..hi - first]
                    .iter_mut()
                    .zip(&self.powers[lo - start..])
                {
                    *value += weight.scale_base(scale);
                }
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

pub(crate) struct LaneSource<'a> {
    matrices: Matrices<'a>,
    pub(crate) lanes: [K; D],
    pub(crate) blocks: Vec<K>,
    public: Vec<K>,
    pub(crate) scale: K,
}

impl<'a> LaneSource<'a> {
    pub(crate) fn new(statement: &'a Statement<'a>, challenges: &Challenges, blocks: usize) -> Result<Self, Error> {
        let omega = omega_powers(statement, challenges.omega);
        let matrices = Matrices::new(statement, &omega)?;
        let evaluation = Evaluation::new(statement, &omega);
        Ok(Self {
            matrices,
            lanes: evaluation.lanes,
            blocks: equality(evaluation.r_high, blocks),
            public: powers(challenges.gamma, statement.claim.m_in)
                .into_iter()
                .map(|power| power * evaluation.public)
                .collect(),
            scale: challenges.mu,
        })
    }

    pub(crate) fn columns(&self, task: usize, values: &mut [K]) {
        self.matrices.add_columns(task, values);
    }

    pub(crate) fn element(&self, columns: &[K], key: &[K]) -> [K; D] {
        let columns = conjugate(columns.try_into().expect("D columns"));
        std::array::from_fn(|lane| key[lane] + columns[lane])
    }

    pub(crate) fn projection_windows(&self, form: &[K; D]) -> Option<crate::column_runs::Windows> {
        self.matrices
            .runs
            .as_ref()
            .map(|r| crate::column_runs::windows(r, &conjugate(form), &self.matrices.powers))
    }
    pub(crate) fn project_task(
        &self,
        task: usize,
        output: &mut [K],
        columns: &mut Vec<K>,
        folded: &[K],
        form: &[K; D],
        evaluation: K,
        windows: Option<&crate::column_runs::Windows>,
    ) {
        let start = task * COLUMNS_PER_TASK;
        let blocks = &self.blocks[start / D..];
        let key = &folded[start..][..output.len() * D];
        if let (Some(runs), Some(windows)) = (&self.matrices.runs, windows) {
            crate::column_runs::project(runs, &self.matrices.weights, start, output, windows);
            for ((out, key), &block) in output.iter_mut().zip(key.chunks_exact(D)).zip(blocks) {
                let dot = key
                    .iter()
                    .zip(form)
                    .fold(ProductSum::default(), |s, (&v, &f)| s.add(v, f))
                    .value();
                *out += dot + block * evaluation;
            }
        } else {
            columns.resize(output.len() * D, K::ZERO);
            self.columns(task, columns);
            for (((out, key), columns), &block) in output
                .iter_mut()
                .zip(key.chunks_exact(D))
                .zip(columns.chunks_exact(D))
                .zip(blocks)
            {
                *out = dot(&self.element(columns, key), form) + block * evaluation;
            }
        }
    }

    pub(crate) fn public(&self) -> &[K] {
        &self.public
    }
}

pub(crate) fn count_tail(layout: &Layout, challenges: &Challenges, start: usize) -> Result<Vec<K>, Error> {
    let mut weights = vec![K::ZERO; layout.witness_len - start];
    for (table, (_, _, bound, counts)) in layout.tables().enumerate() {
        let cells = count_weights(bound, challenges.lambda[table], challenges)?;
        weights[counts.offset - start..][..cells.len()].copy_from_slice(&cells);
    }
    Ok(weights)
}

pub(crate) fn table(
    statements: &[Statement<'_>],
    layout: &Layout,
    challenges: &Challenges,
    fold_key: impl FnOnce() -> Vec<K> + Send,
    omit_zero_prefix: bool,
) -> Result<(Vec<K>, Vec<K>), Error> {
    let point = Point::new(challenges.delta);
    let transform = crate::ring::LaneTransform::new(&point);
    let prefix = if omit_zero_prefix {
        layout.cells[0].witness.len()
    } else {
        0
    };
    let prepare = || {
        statements
            .par_iter()
            .zip(&layout.cells)
            .skip(usize::from(omit_zero_prefix))
            .map(|(statement, cells)| {
                let omega = omega_powers(statement, challenges.omega);
                let matrices = Matrices::new(statement, &omega)?;
                let evaluation = Evaluation::new(statement, &omega);
                let blocks = equality(evaluation.r_high, cells.blocks);
                Ok((evaluation, blocks, matrices))
            })
            .collect::<Result<Vec<_>, Error>>()
    };
    let ((prepared, mut table), folded) = rayon::join(
        || rayon::join(prepare, || vec![K::ZERO; layout.witness_len - prefix]),
        fold_key,
    );
    let mut prepared = prepared?.into_iter();
    let mut tables = 0;
    for ((statement, cells), mu) in statements
        .iter()
        .zip(&layout.cells)
        .zip(powers(challenges.mu, statements.len()))
    {
        if cells.witness.offset >= prefix {
            let (evaluation, blocks, matrices) = prepared.next().expect("prepared weights for the claim");
            let start = cells.witness.offset - prefix;
            let witness = &mut table[start..][..cells.blocks * D];
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
                        for (output, lane) in output.iter_mut().zip(transform.lanes(&element)) {
                            *output = mu * lane;
                        }
                    }
                });
            let mut public = mu * evaluation.public;
            for output in &mut witness[..statement.claim.m_in] {
                *output += public;
                public *= challenges.gamma;
            }
        }
        if let Some((bound, counts)) = cells.table {
            let weights = count_weights(bound, challenges.lambda[tables], challenges)?;
            table[counts.offset - prefix..][..weights.len()].copy_from_slice(&weights);
            tables += 1;
        }
    }
    Ok((table, folded))
}

struct MatrixAt<'a> {
    coefficients: &'a [K],
    lanes: [K; D],
    blocks: EqualityWeights,
    value: K,
}

impl MatrixRowSink for MatrixAt<'_> {
    fn push_run(&mut self, _: usize, matrix: usize, run: GeometricRowRun<F>) -> Result<(), PiCcsError> {
        run.for_each_term(|_, column, coefficient| {
            self.value += (self.lanes[column % D] * self.blocks.at(column / D) * self.coefficients[matrix])
                .scale_base(coefficient);
        });
        Ok(())
    }

    fn finish_matrix_row(&mut self, _: usize, _: usize) -> Result<ControlFlow<()>, PiCcsError> {
        Ok(ControlFlow::Continue(()))
    }
}

pub(crate) fn first_slice_at(
    statement: &Statement<'_>,
    challenges: &Challenges,
    folded: &[K],
    rho: &[K],
) -> Result<(K, K), Error> {
    let (lanes, block_point) = rho.split_at(LANE_VARS);
    let lanes: [K; D] = equality(lanes, D).try_into().expect("D lane weights");
    let form = Point::new(challenges.delta).lanes(&lanes);
    let block_count = statement.blocks();
    let uses = Uses {
        form,
        claims: vec![Use {
            coefficient: K::ONE,
            point: block_point,
            blocks: block_count,
        }],
    };
    let omega = omega_powers(statement, challenges.omega);
    let evaluation = Evaluation::new(statement, &omega);
    let key = key_part(&form, &equality(block_point, block_count), folded);
    let pad = dot(&form, &evaluation.lanes) * prefix_equality(block_point, block_count, evaluation.r_high);
    let public = evaluation.public
        * equality(rho, statement.claim.m_in)
            .iter()
            .rev()
            .fold(K::ZERO, |value, &weight| value * challenges.gamma + weight);
    let matrix = if statement
        .claim
        .r
        .iter()
        .all(|&coordinate| coordinate == K::ZERO)
    {
        let mut sink = MatrixAt {
            coefficients: &omega[KAPPA + 1..][..statement.claim.eval_a.len()],
            lanes: conjugate(&form),
            blocks: EqualityWeights::new(block_point),
            value: K::ZERO,
        };
        let rows = statement
            .rows
            .factored()?
            .ok_or(Error::Statement("the matrix rows have no factored form"))?;
        rows.visit_rows(0..1, &mut sink)?;
        sink.value
    } else {
        matrix_part(std::slice::from_ref(statement), challenges, &uses)?
    };
    Ok((key + pad + public + matrix, key))
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

pub(crate) fn key_part(form: &[K; D], blocks: &[K], folded: &[K]) -> K {
    folded
        .par_chunks(D)
        .zip(blocks)
        .map(|(key, &block)| {
            let lane_sum = key
                .iter()
                .zip(form)
                .fold(ProductSum::default(), |sum, (&key, &form)| sum.add(form, key));
            block * lane_sum.value()
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
