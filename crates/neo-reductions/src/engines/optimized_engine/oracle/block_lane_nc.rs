//! Production block-by-lane norm-check oracle.
//!
//! This leaf owns the CPU evaluation of the canonical 19-block-bit followed
//! by 6-lane-bit polynomial.  Its source values are read directly from the
//! packed witness matrices.  The optional delayed term reads the same raw
//! running matrices; it never accepts `CeClaim::y_zcol` or another carried
//! evaluation as source authority.

use super::*;
use crate::error::PiCcsError;

/// Fixed production block domain: `2^19` blocks cover the 265,535 live
/// carrier blocks.
pub const BLOCK_LANE_NC_BLOCK_VARIABLES: usize = 19;
/// Fixed production lane domain: `2^6` lanes contain 54 live coefficient
/// lanes and ten computed zero lanes.
pub const BLOCK_LANE_NC_LANE_VARIABLES: usize = 6;
/// The combined polynomial is quartic, so every round serializes five
/// low-to-high coefficients.
pub const BLOCK_LANE_NC_ROUND_COEFFICIENTS: usize = 5;

const BLOCK_DOMAIN: usize = 1usize << BLOCK_LANE_NC_BLOCK_VARIABLES;
const LANE_DOMAIN: usize = 1usize << BLOCK_LANE_NC_LANE_VARIABLES;

/// Verifier-owned coins for the canonical combined NC polynomial.
///
/// Arrays make the production arities part of the Rust type.  Transcript
/// integration must construct this value only after binding the parent and
/// authoritative child matrices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLaneNcChallenges {
    pub beta_block: [K; BLOCK_LANE_NC_BLOCK_VARIABLES],
    pub beta_lane: [K; BLOCK_LANE_NC_LANE_VARIABLES],
    pub gamma: K,
    pub producer_beta: K,
    pub batch_weight: K,
}

/// One-fold-delayed parent projection carried by the recursive state.
///
/// `parent_y` contains only the 54 live coefficients.  The remaining ten
/// lane leaves are computed zeros.  This value supplies the initial claim;
/// it is not used as the raw-child source in the delayed polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLaneNcPending {
    pub old_block: [K; BLOCK_LANE_NC_BLOCK_VARIABLES],
    pub parent_y: [K; D],
}

/// CPU oracle for the production combined block-by-lane NC polynomial.
///
/// Source order is fresh witnesses followed by running witnesses.  The
/// ordinary NC term uses the paper-relative weights `gamma^source`, beginning
/// with `gamma^0`.  When `pending` is present, the delayed term uses base-two
/// radix weights over the raw running matrices.
///
/// The oracle never materializes the `2^25` product table.  Before the first
/// challenge it reads the packed matrices directly.  The first block fold
/// materializes only the live folded support, and every later fold shrinks
/// that support in place.
pub struct BlockLaneNcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    raw_sources: Vec<&'a Mat<F>>,
    fresh_count: usize,
    source_gamma: Vec<K>,
    running_radix: Vec<K>,

    round_idx: usize,
    block_support: usize,
    lane_support: usize,
    source_tables: Option<Vec<Vec<K>>>,
    block_rows: Option<Vec<[K; LANE_DOMAIN]>>,

    eq_beta_block: Vec<K>,
    eq_beta_lane: Vec<K>,
    eq_old_block: Option<Vec<K>>,
    beta_power_lane: Option<Vec<K>>,
    batch_weight: K,
    initial_sum: K,
}

impl<'a, F> BlockLaneNcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    /// Construct the fixed production oracle from authoritative packed
    /// matrices.  Matrix validation is fail-closed against the CCS logical
    /// width and the fixed 19-block domain.
    pub fn new(
        s: &CcsStructure<F>,
        fresh_witnesses: &'a [CcsWitness<F>],
        running_witnesses: &'a [Mat<F>],
        challenges: BlockLaneNcChallenges,
        pending: Option<BlockLaneNcPending>,
    ) -> Result<Self, PiCcsError> {
        if s.m == 0 {
            return Err(PiCcsError::InvalidInput(
                "block-lane NC logical width must be non-zero".into(),
            ));
        }
        if fresh_witnesses.is_empty() && running_witnesses.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "block-lane NC requires at least one raw witness matrix".into(),
            ));
        }

        let block_support = s.m.div_ceil(D);
        if block_support > BLOCK_DOMAIN {
            return Err(PiCcsError::InvalidInput(format!(
                "block-lane NC needs {block_support} blocks, exceeding the fixed {BLOCK_DOMAIN}-block domain"
            )));
        }
        if D != 54 || D > LANE_DOMAIN {
            return Err(PiCcsError::InvalidInput(format!(
                "block-lane NC requires exactly 54 live lanes inside the {LANE_DOMAIN}-lane domain, got D={D}"
            )));
        }

        let mut raw_sources = Vec::with_capacity(fresh_witnesses.len() + running_witnesses.len());
        for witness in fresh_witnesses {
            crate::common::validate_superneo_witness_mat(&witness.Z, s.m)?;
            raw_sources.push(&witness.Z);
        }
        for witness in running_witnesses {
            crate::common::validate_superneo_witness_mat(witness, s.m)?;
            raw_sources.push(witness);
        }

        let mut source_gamma = Vec::with_capacity(raw_sources.len());
        let mut gamma_power = K::ONE;
        for _ in 0..raw_sources.len() {
            source_gamma.push(gamma_power);
            gamma_power *= challenges.gamma;
        }

        let mut running_radix = Vec::with_capacity(running_witnesses.len());
        let radix = K::from(F::from_u64(2));
        let mut radix_power = K::ONE;
        for _ in running_witnesses {
            running_radix.push(radix_power);
            radix_power *= radix;
        }

        let eq_beta_block = chi_tail_weights(&challenges.beta_block);
        let eq_beta_lane = chi_tail_weights(&challenges.beta_lane);

        let (eq_old_block, beta_power_lane, initial_sum) = match pending {
            Some(pending) => {
                let mut beta_powers = Vec::with_capacity(LANE_DOMAIN);
                let mut power = K::ONE;
                for _ in 0..LANE_DOMAIN {
                    beta_powers.push(power);
                    power *= challenges.producer_beta;
                }

                let mut parent_projection = K::ZERO;
                for coefficient in pending.parent_y.iter().rev() {
                    parent_projection = parent_projection * challenges.producer_beta + *coefficient;
                }
                (
                    Some(chi_tail_weights(&pending.old_block)),
                    Some(beta_powers),
                    challenges.batch_weight * parent_projection,
                )
            }
            None => (None, None, K::ZERO),
        };

        Ok(Self {
            raw_sources,
            fresh_count: fresh_witnesses.len(),
            source_gamma,
            running_radix,
            round_idx: 0,
            block_support,
            lane_support: D,
            source_tables: None,
            block_rows: None,
            eq_beta_block,
            eq_beta_lane,
            eq_old_block,
            beta_power_lane,
            batch_weight: challenges.batch_weight,
            initial_sum,
        })
    }

    /// The public initial claim: zero on the base step, otherwise the sampled
    /// batch weight times the old parent's compact 54-coefficient evaluation.
    pub fn initial_sum(&self) -> K {
        self.initial_sum
    }

    /// Whether this step carries a delayed parent projection.
    pub fn has_pending_projection(&self) -> bool {
        self.eq_old_block.is_some()
    }

    #[inline]
    fn interpolate(lo: K, hi: K, point: K) -> K {
        lo + (hi - lo) * point
    }

    #[inline]
    fn evaluate_coefficients(coefficients: &[K; BLOCK_LANE_NC_ROUND_COEFFICIENTS], point: K) -> K {
        coefficients
            .iter()
            .rev()
            .fold(K::ZERO, |value, coefficient| value * point + *coefficient)
    }

    fn source_block_pair(&self, source: usize, tail: usize, lane: usize) -> (K, K) {
        let low_block = 2 * tail;
        let high_block = low_block + 1;
        if let Some(tables) = &self.source_tables {
            let table = &tables[source];
            let low = if low_block < self.block_support {
                table[low_block * D + lane]
            } else {
                K::ZERO
            };
            let high = if high_block < self.block_support {
                table[high_block * D + lane]
            } else {
                K::ZERO
            };
            (low, high)
        } else {
            let matrix = self.raw_sources[source];
            let low = if low_block < self.block_support {
                K::from(matrix[(lane, low_block)])
            } else {
                K::ZERO
            };
            let high = if high_block < self.block_support {
                K::from(matrix[(lane, high_block)])
            } else {
                K::ZERO
            };
            (low, high)
        }
    }

    fn source_lane_pair(&self, source: usize, tail: usize) -> (K, K) {
        let table = &self
            .source_tables
            .as_ref()
            .expect("lane phase requires fully block-folded raw matrices")[source];
        let low_lane = 2 * tail;
        let high_lane = low_lane + 1;
        let low = table.get(low_lane).copied().unwrap_or(K::ZERO);
        let high = table.get(high_lane).copied().unwrap_or(K::ZERO);
        (low, high)
    }

    #[inline]
    fn accumulate_source_polynomials(&self, source_pairs: impl Iterator<Item = (usize, K, K)>) -> ([K; 4], [K; 2]) {
        let mut cubic = [K::ZERO; 4];
        let mut running = [K::ZERO; 2];
        let three = K::from(F::from_u64(3));

        for (source, low, high) in source_pairs {
            let delta = high - low;
            let low_sq = low * low;
            let delta_sq = delta * delta;
            let gamma = self.source_gamma[source];

            cubic[0] += gamma * (low_sq * low - low);
            cubic[1] += gamma * (three * low_sq * delta - delta);
            cubic[2] += gamma * (three * low * delta_sq);
            cubic[3] += gamma * (delta_sq * delta);

            if source >= self.fresh_count {
                let radix = self.running_radix[source - self.fresh_count];
                running[0] += radix * low;
                running[1] += radix * delta;
            }
        }
        (cubic, running)
    }

    fn block_round_coefficients(&self) -> [K; BLOCK_LANE_NC_ROUND_COEFFICIENTS] {
        let mut coefficients = [K::ZERO; BLOCK_LANE_NC_ROUND_COEFFICIENTS];
        let active_tails = self.block_support.div_ceil(2);

        for tail in 0..active_tails {
            let beta_low = self.eq_beta_block[2 * tail];
            let beta_delta = self.eq_beta_block[2 * tail + 1] - beta_low;
            let old_pair = self.eq_old_block.as_ref().map(|table| {
                let low = table[2 * tail];
                (low, table[2 * tail + 1] - low)
            });

            for lane in 0..D {
                let (cubic, running) = self.accumulate_source_polynomials((0..self.raw_sources.len()).map(|source| {
                    let (low, high) = self.source_block_pair(source, tail, lane);
                    (source, low, high)
                }));

                let lane_weight = self.eq_beta_lane[lane];
                coefficients[0] += lane_weight * beta_low * cubic[0];
                coefficients[1] += lane_weight * (beta_low * cubic[1] + beta_delta * cubic[0]);
                coefficients[2] += lane_weight * (beta_low * cubic[2] + beta_delta * cubic[1]);
                coefficients[3] += lane_weight * (beta_low * cubic[3] + beta_delta * cubic[2]);
                coefficients[4] += lane_weight * beta_delta * cubic[3];

                if let (Some((old_low, old_delta)), Some(beta_powers)) = (old_pair, self.beta_power_lane.as_ref()) {
                    let delayed_weight = self.batch_weight * beta_powers[lane];
                    coefficients[0] += delayed_weight * old_low * running[0];
                    coefficients[1] += delayed_weight * (old_low * running[1] + old_delta * running[0]);
                    coefficients[2] += delayed_weight * old_delta * running[1];
                }
            }
        }
        coefficients
    }

    fn lane_round_coefficients(&self) -> [K; BLOCK_LANE_NC_ROUND_COEFFICIENTS] {
        let mut coefficients = [K::ZERO; BLOCK_LANE_NC_ROUND_COEFFICIENTS];
        let active_tails = self.lane_support.div_ceil(2);
        let block_weight = self.eq_beta_block[0];
        let old_block_weight = self.eq_old_block.as_ref().map(|table| table[0]);

        for tail in 0..active_tails {
            let beta_low = self.eq_beta_lane[2 * tail];
            let beta_delta = self.eq_beta_lane[2 * tail + 1] - beta_low;
            let power_pair = self.beta_power_lane.as_ref().map(|table| {
                let low = table[2 * tail];
                (low, table[2 * tail + 1] - low)
            });
            let (cubic, running) = self.accumulate_source_polynomials((0..self.raw_sources.len()).map(|source| {
                let (low, high) = self.source_lane_pair(source, tail);
                (source, low, high)
            }));

            coefficients[0] += block_weight * beta_low * cubic[0];
            coefficients[1] += block_weight * (beta_low * cubic[1] + beta_delta * cubic[0]);
            coefficients[2] += block_weight * (beta_low * cubic[2] + beta_delta * cubic[1]);
            coefficients[3] += block_weight * (beta_low * cubic[3] + beta_delta * cubic[2]);
            coefficients[4] += block_weight * beta_delta * cubic[3];

            if let (Some(old_weight), Some((power_low, power_delta))) = (old_block_weight, power_pair) {
                let delayed_weight = self.batch_weight * old_weight;
                coefficients[0] += delayed_weight * power_low * running[0];
                coefficients[1] += delayed_weight * (power_low * running[1] + power_delta * running[0]);
                coefficients[2] += delayed_weight * power_delta * running[1];
            }
        }
        coefficients
    }

    /// Exact low-to-high quartic coefficients for the current round.
    pub fn round_coefficients(&self) -> [K; BLOCK_LANE_NC_ROUND_COEFFICIENTS] {
        assert!(
            self.round_idx < BLOCK_LANE_NC_BLOCK_VARIABLES + BLOCK_LANE_NC_LANE_VARIABLES,
            "block-lane NC requested coefficients after its final round"
        );
        if self.round_idx < BLOCK_LANE_NC_BLOCK_VARIABLES {
            self.block_round_coefficients()
        } else {
            self.lane_round_coefficients()
        }
    }

    fn fold_full_table(table: &mut Vec<K>, point: K) {
        debug_assert!(table.len() >= 2 && table.len().is_multiple_of(2));
        let next_len = table.len() / 2;
        for index in 0..next_len {
            table[index] = Self::interpolate(table[2 * index], table[2 * index + 1], point);
        }
        table.truncate(next_len);
    }

    fn fold_block_sources(&mut self, point: K) {
        let next_support = self.block_support.div_ceil(2);
        match &mut self.source_tables {
            Some(tables) => {
                for table in tables {
                    for block in 0..next_support {
                        for lane in 0..D {
                            let low = table[(2 * block) * D + lane];
                            let high = if 2 * block + 1 < self.block_support {
                                table[(2 * block + 1) * D + lane]
                            } else {
                                K::ZERO
                            };
                            table[block * D + lane] = Self::interpolate(low, high, point);
                        }
                    }
                    table.truncate(next_support * D);
                }
            }
            None => {
                let mut tables = Vec::with_capacity(self.raw_sources.len());
                for matrix in &self.raw_sources {
                    let mut table = vec![K::ZERO; next_support * D];
                    for block in 0..next_support {
                        for lane in 0..D {
                            let low = K::from(matrix[(lane, 2 * block)]);
                            let high = if 2 * block + 1 < self.block_support {
                                K::from(matrix[(lane, 2 * block + 1)])
                            } else {
                                K::ZERO
                            };
                            table[block * D + lane] = Self::interpolate(low, high, point);
                        }
                    }
                    tables.push(table);
                }
                self.source_tables = Some(tables);
            }
        }
        self.block_support = next_support;
    }

    fn fold_lane_sources(&mut self, point: K) {
        let next_support = self.lane_support.div_ceil(2);
        for table in self
            .source_tables
            .as_mut()
            .expect("lane phase requires fully block-folded raw matrices")
        {
            for lane in 0..next_support {
                let low = table[2 * lane];
                let high = table.get(2 * lane + 1).copied().unwrap_or(K::ZERO);
                table[lane] = Self::interpolate(low, high, point);
            }
            table.truncate(next_support);
        }
        self.lane_support = next_support;
    }

    /// Raw source MLE values at the fully transcript-derived block/lane point,
    /// in fresh-then-running order.
    pub fn finalized_source_values(&self) -> Vec<K> {
        assert_eq!(
            self.round_idx,
            BLOCK_LANE_NC_BLOCK_VARIABLES + BLOCK_LANE_NC_LANE_VARIABLES,
            "block-lane NC source point is not final"
        );
        self.source_tables
            .as_ref()
            .expect("final source tables missing")
            .iter()
            .map(|table| {
                assert_eq!(table.len(), 1, "final source table must be scalar");
                table[0]
            })
            .collect()
    }

    /// Authoritative running-child MLE values at the final block/lane point.
    /// These values come from the raw matrices folded by this oracle; they are
    /// not copied from the claims transported by the prover.
    pub fn finalized_running_values(&self) -> Vec<K> {
        self.finalized_source_values()
            .into_iter()
            .skip(self.fresh_count)
            .collect()
    }

    /// Per-source raw MLE rows at the transcript-derived block point. The
    /// first 54 lanes come from the authoritative matrices; the ten Boolean
    /// padding lanes are verifier-computed zeros. This snapshot is retained
    /// before the subsequent six lane folds consume the live rows.
    pub fn block_projected_source_rows(&self) -> &[[K; LANE_DOMAIN]] {
        assert!(
            self.round_idx >= BLOCK_LANE_NC_BLOCK_VARIABLES,
            "block-lane NC block point is not final"
        );
        self.block_rows
            .as_deref()
            .expect("block-lane NC block-row snapshot missing")
    }

    /// Running-child suffix of [`Self::block_projected_source_rows`].
    pub fn block_projected_running_rows(&self) -> &[[K; LANE_DOMAIN]] {
        &self.block_projected_source_rows()[self.fresh_count..]
    }

    /// Exact combined polynomial value at the fully folded point.  This is
    /// computed from raw matrices and verifier challenges, not a carried
    /// `y_zcol` value.
    pub fn finalized_value(&self) -> K {
        let sources = self.finalized_source_values();
        let mut ordinary = K::ZERO;
        let mut running = K::ZERO;
        for (source, value) in sources.into_iter().enumerate() {
            ordinary += self.source_gamma[source] * (value * value * value - value);
            if source >= self.fresh_count {
                running += self.running_radix[source - self.fresh_count] * value;
            }
        }
        let ordinary = self.eq_beta_block[0] * self.eq_beta_lane[0] * ordinary;
        let delayed = match (&self.eq_old_block, &self.beta_power_lane) {
            (Some(old_block), Some(beta_power)) => self.batch_weight * old_block[0] * beta_power[0] * running,
            _ => K::ZERO,
        };
        ordinary + delayed
    }
}

impl<'a, F> RoundOracle for BlockLaneNcOracle<'a, F>
where
    F: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<F>,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let coefficients = self.round_coefficients();
        points
            .iter()
            .map(|point| Self::evaluate_coefficients(&coefficients, *point))
            .collect()
    }

    fn num_rounds(&self) -> usize {
        BLOCK_LANE_NC_BLOCK_VARIABLES + BLOCK_LANE_NC_LANE_VARIABLES
    }

    fn degree_bound(&self) -> usize {
        BLOCK_LANE_NC_ROUND_COEFFICIENTS - 1
    }

    fn fold(&mut self, point: K) {
        assert!(
            self.round_idx < self.num_rounds(),
            "block-lane NC folded after its final round"
        );
        if self.round_idx < BLOCK_LANE_NC_BLOCK_VARIABLES {
            Self::fold_full_table(&mut self.eq_beta_block, point);
            if let Some(old_block) = &mut self.eq_old_block {
                Self::fold_full_table(old_block, point);
            }
            self.fold_block_sources(point);
            if self.round_idx + 1 == BLOCK_LANE_NC_BLOCK_VARIABLES {
                let tables = self
                    .source_tables
                    .as_ref()
                    .expect("last block fold must materialize source tables");
                let mut rows = Vec::with_capacity(tables.len());
                for table in tables {
                    assert_eq!(table.len(), D, "last block fold must leave one live 54-lane row");
                    let mut row = [K::ZERO; LANE_DOMAIN];
                    row[..D].copy_from_slice(table);
                    rows.push(row);
                }
                self.block_rows = Some(rows);
            }
        } else {
            Self::fold_full_table(&mut self.eq_beta_lane, point);
            if let Some(beta_power) = &mut self.beta_power_lane {
                Self::fold_full_table(beta_power, point);
            }
            self.fold_lane_sources(point);
        }
        self.round_idx += 1;
    }
}
