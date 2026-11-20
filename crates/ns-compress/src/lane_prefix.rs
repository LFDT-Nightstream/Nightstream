//! Lane variables in the first sum-check of SPEC.md section 6.

use neo_math::{D, K};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use crate::layout::LANE_VARS;
use crate::ring::{dot, Point};
use crate::weights::{equality, LaneSource, COLUMNS_PER_TASK};

type Ring = [K; D];
const GROUP_BITS: usize = 4;
const GROUPS: usize = D / GROUP_BITS;
const PATTERNS: usize = (1 << GROUP_BITS) - 1;
const BLOCKS_PER_TASK: usize = COLUMNS_PER_TASK / D;

#[derive(Clone, Copy)]
pub(crate) struct Masks<'a> {
    positive: &'a [u64],
    negative: &'a [u64],
}

impl<'a> Masks<'a> {
    pub(crate) fn new((positive, negative): (&'a [u64], &'a [u64])) -> Self {
        Self { positive, negative }
    }

    pub(crate) fn suitable((positive, negative): (&'a [u64], &'a [u64])) -> Option<Self> {
        if positive.len() < 1 << 14 {
            return None;
        }
        let nonzero_groups = |mask: u64| {
            ((mask | (mask >> 1) | (mask >> 2) | (mask >> 3)) & 0x1111_1111_1111_1111).count_ones() as usize
        };
        let groups: usize = positive
            .par_iter()
            .zip(negative)
            .map(|(&positive, &negative)| nonzero_groups(positive) + nonzero_groups(negative))
            .sum();
        (groups <= 20 * positive.len()).then_some(Self::new((positive, negative)))
    }

    fn at(self, block: usize) -> (u64, u64) {
        (self.positive[block], self.negative[block])
    }
}

#[derive(Clone)]
struct Bucket {
    element: Ring,
    evaluation: K,
}

struct Buckets(Vec<Bucket>);

impl Buckets {
    fn new() -> Self {
        Self(vec![
            Bucket {
                element: [K::ZERO; D],
                evaluation: K::ZERO
            };
            GROUPS * PATTERNS
        ])
    }

    fn add(&mut self, (positive, negative): (u64, u64), element: &Ring, evaluation: K) {
        for (group, bins) in self.0.chunks_exact_mut(PATTERNS).enumerate() {
            let positive = (positive >> (GROUP_BITS * group)) as usize & PATTERNS;
            let negative = (negative >> (GROUP_BITS * group)) as usize & PATTERNS;
            if positive != 0 {
                let bucket = &mut bins[positive - 1];
                for (value, &coefficient) in bucket.element.iter_mut().zip(element) {
                    *value += coefficient;
                }
                bucket.evaluation += evaluation;
            }
            if negative != 0 {
                let bucket = &mut bins[negative - 1];
                for (value, &coefficient) in bucket.element.iter_mut().zip(element) {
                    *value -= coefficient;
                }
                bucket.evaluation -= evaluation;
            }
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (left, right) in self.0.iter_mut().zip(other.0) {
            for (left, right) in left.element.iter_mut().zip(right.element) {
                *left += right;
            }
            left.evaluation += right.evaluation;
        }
        self
    }

    fn moments(self, point: &Point, evaluation: &Ring) -> Vec<Ring> {
        let mut rows = vec![[K::ZERO; D]; D];
        let mut evaluations = [K::ZERO; D];
        let evaluation = point.lanes(evaluation);
        for (group, bins) in self.0.chunks_exact(PATTERNS).enumerate() {
            for (bin, coefficients) in bins.iter().enumerate() {
                let mut pattern = bin + 1;
                while pattern != 0 {
                    let lane = GROUP_BITS * group + pattern.trailing_zeros() as usize;
                    for (value, &coefficient) in rows[lane].iter_mut().zip(&coefficients.element) {
                        *value += coefficient;
                    }
                    evaluations[lane] += coefficients.evaluation;
                    pattern &= pattern - 1;
                }
            }
        }
        for (row, contract) in rows.iter_mut().zip(evaluations) {
            *row = point.lanes(row);
            for (value, pad) in row.iter_mut().zip(evaluation) {
                *value += contract * pad;
            }
        }
        rows
    }
}

pub(crate) struct LaneStart<'a> {
    masks: Masks<'a>,
    folded_key: &'a [K],
    source: LaneSource<'a>,
    padded_blocks: usize,
    point: Point,
    rows: Vec<Ring>,
    width: usize,
}

impl<'a> LaneStart<'a> {
    pub(crate) fn new(
        masks: Masks<'a>,
        folded_key: &'a [K],
        source: LaneSource<'a>,
        padded_blocks: usize,
        delta: K,
    ) -> Self {
        let point = Point::new(delta);
        let blocks = masks.positive.len();
        let tasks = blocks.div_ceil(BLOCKS_PER_TASK);
        let per_job = tasks.div_ceil(rayon::current_num_threads());
        let bins = (0..tasks.div_ceil(per_job))
            .into_par_iter()
            .map(|job| {
                let mut columns = vec![K::ZERO; COLUMNS_PER_TASK];
                let mut bins = Buckets::new();
                for task in job * per_job..((job + 1) * per_job).min(tasks) {
                    let start = task * BLOCKS_PER_TASK;
                    let end = (start + BLOCKS_PER_TASK).min(blocks);
                    let columns = &mut columns[..(end - start) * D];
                    source.columns(task, columns);
                    for (block, columns) in (start..end).zip(columns.chunks_exact(D)) {
                        let mask = masks.at(block);
                        if mask != (0, 0) {
                            let key = &folded_key[block * D..][..D];
                            bins.add(mask, &source.element(columns, key), source.blocks[block]);
                        }
                    }
                }
                bins
            })
            .reduce_with(Buckets::merge)
            .expect("a packed witness has blocks");
        let mut rows = bins.moments(&point, &source.lanes);
        for (block, public) in source.public().chunks(D).enumerate() {
            let (positive, negative) = masks.at(block);
            for (lane, row) in rows.iter_mut().enumerate() {
                let (add, subtract) = ((positive >> lane) & 1 != 0, (negative >> lane) & 1 != 0);
                if add {
                    for (value, &weight) in row.iter_mut().zip(public) {
                        *value += weight;
                    }
                }
                if subtract {
                    for (value, &weight) in row.iter_mut().zip(public) {
                        *value -= weight;
                    }
                }
            }
        }
        for cell in rows.iter_mut().flatten() {
            *cell *= source.scale;
        }
        Self {
            masks,
            folded_key,
            source,
            padded_blocks,
            point,
            rows,
            width: D,
        }
    }

    pub(crate) fn round(&self) -> [K; 3] {
        let mut sums = [K::ZERO; 3];
        for low in (0..self.width).step_by(2) {
            let high = low + 1;
            let (a, b) = (self.rows[low][low], self.rows[low][high]);
            let (c, d) = (self.rows[high][low], self.rows[high][high]);
            sums[0] += a;
            sums[1] += a - (b + c).double() + d.double().double();
        }
        sums
    }

    pub(crate) fn bind(&mut self, point: K) {
        let bind = |low, high| low + (high - low) * point;
        for row in 0..self.width / 2 {
            for col in 0..self.width / 2 {
                let a = bind(self.rows[2 * row][2 * col], self.rows[2 * row][2 * col + 1]);
                let b = bind(self.rows[2 * row + 1][2 * col], self.rows[2 * row + 1][2 * col + 1]);
                self.rows[row][col] = bind(a, b);
            }
        }
        self.width /= 2;
    }

    pub(crate) fn materialize(self, rho: &[K], tail_values: &[K], tail_weights: &[K]) -> (Vec<K>, Vec<K>) {
        assert_eq!(rho.len(), LANE_VARS);
        let coefficients: Ring = equality(rho, D).try_into().expect("D lane coefficients");
        let form = self.point.lanes(&coefficients);
        let windows = self.source.projection_windows(&form);
        let evaluation = dot(&self.source.lanes, &form);
        let mut subsets = [[K::ZERO; 256]; D / 8];
        for (subset, weights) in subsets.iter_mut().zip(coefficients.chunks_exact(8)) {
            for mask in 1usize..256 {
                subset[mask] = subset[mask & (mask - 1)] + weights[mask.trailing_zeros() as usize];
            }
        }
        let blocks = self.masks.positive.len();
        let mut values = vec![K::ZERO; self.padded_blocks + tail_values.len()];
        let mut weights = vec![K::ZERO; self.padded_blocks + tail_weights.len()];
        values[..blocks]
            .par_chunks_mut(BLOCKS_PER_TASK)
            .zip(weights[..blocks].par_chunks_mut(BLOCKS_PER_TASK))
            .enumerate()
            .for_each_init(Vec::new, |columns, (task, (values, weights))| {
                self.source.project_task(
                    task,
                    weights,
                    columns,
                    self.folded_key,
                    &form,
                    evaluation,
                    windows.as_ref(),
                );
                let start = task * BLOCKS_PER_TASK;
                for (offset, (value, weight)) in values.iter_mut().zip(weights).enumerate() {
                    let block = start + offset;
                    let (positive, negative) = self.masks.at(block);
                    *value = positive
                        .to_le_bytes()
                        .into_iter()
                        .zip(negative.to_le_bytes())
                        .zip(&subsets)
                        .map(|((positive, negative), subset)| subset[positive as usize] - subset[negative as usize])
                        .sum();
                    let public = self.source.public().get(block * D..).unwrap_or(&[]);
                    let public = dot(&public[..public.len().min(D)], &coefficients);
                    *weight = self.source.scale * (public + *weight);
                }
            });
        values[self.padded_blocks..].copy_from_slice(tail_values);
        weights[self.padded_blocks..].copy_from_slice(tail_weights);
        (values, weights)
    }
}

#[cfg(test)]
#[path = "../tests/unit/lane_prefix.rs"]
mod tests;
