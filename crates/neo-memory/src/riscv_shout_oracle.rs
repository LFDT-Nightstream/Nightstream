//! RISC-V-specific Shout helpers.
//!
//! This module contains RISC-V-specific lookup oracles that should not live in the
//! generic Twist/Shout oracle implementations.

use std::collections::HashMap;

use neo_math::K;
use neo_reductions::error::PiCcsError;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

use crate::riscv_lookups::{compute_op, evaluate_opcode_mle, uninterleave_bits, RiscvOpcode};
use crate::twist_oracle::build_eq_table;

/// Address-domain lookup oracle for Shout with an implicit RISC-V opcode table.
///
/// This is a SPARSE variant of the generic address-domain lookup oracle that avoids
/// materializing the full `2^ell_addr` weight table (and thus supports `ell_addr = 64`
/// for RV32).
///
/// The proven identity is:
/// ```text
/// Σ_a Table(a) · weight(a) = claimed_sum
/// ```
/// where `Table` is defined by `(opcode, xlen)` and `weight` is derived from
/// the committed `(addr_bits, has_lookup)` columns and `r_cycle`.
pub struct RiscvAddressLookupOracleSparse {
    opcode: RiscvOpcode,
    xlen: usize,
    ell_total: usize,
    rounds_remaining: usize,
    /// Bound challenges for already-eliminated address bits (little-endian order).
    bound_prefix: Vec<K>,
    /// Sparse truth table of `weight` on the remaining Boolean hypercube.
    /// Keys are indices in little-endian order over the remaining variables.
    weights: HashMap<u64, K>,
}

impl RiscvAddressLookupOracleSparse {
    fn ensure_supported(opcode: RiscvOpcode, xlen: usize) -> Result<(), PiCcsError> {
        if xlen != 32 {
            return Err(PiCcsError::InvalidInput(
                "RISC-V implicit Shout tables currently support xlen=32 only".into(),
            ));
        }
        match opcode {
            RiscvOpcode::And
            | RiscvOpcode::Xor
            | RiscvOpcode::Or
            | RiscvOpcode::Add
            | RiscvOpcode::Sub
            | RiscvOpcode::Eq
            | RiscvOpcode::Neq
            | RiscvOpcode::Slt
            | RiscvOpcode::Sltu => Ok(()),
            _ => Err(PiCcsError::InvalidInput(format!(
                "RISC-V implicit Shout table MLE not implemented for opcode {opcode:?} at xlen={xlen}"
            ))),
        }
    }

    /// Construct the oracle and its claimed sum from decoded time-domain columns.
    pub fn new(
        opcode: RiscvOpcode,
        xlen: usize,
        addr_bits: &[Vec<K>],
        has_lookup: &[K],
        r_cycle: &[K],
    ) -> Result<(Self, K), PiCcsError> {
        Self::ensure_supported(opcode, xlen)?;

        let ell_total = xlen
            .checked_mul(2)
            .ok_or_else(|| PiCcsError::InvalidInput("RISC-V implicit Shout: 2*xlen overflow".into()))?;
        if addr_bits.len() != ell_total {
            return Err(PiCcsError::InvalidInput(format!(
                "RISC-V implicit Shout: addr_bits.len()={} != 2*xlen={}",
                addr_bits.len(),
                ell_total
            )));
        }
        if ell_total > 64 {
            return Err(PiCcsError::InvalidInput(
                "RISC-V implicit Shout: ell_total > 64 not supported (key does not fit u64)".into(),
            ));
        }

        let pow2_cycle = 1usize << r_cycle.len();
        if has_lookup.len() != pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "RISC-V implicit Shout: has_lookup.len()={} != 2^ell_cycle={pow2_cycle}",
                has_lookup.len()
            )));
        }
        for (i, col) in addr_bits.iter().enumerate() {
            if col.len() != pow2_cycle {
                return Err(PiCcsError::InvalidInput(format!(
                    "RISC-V implicit Shout: addr_bits[{i}].len()={} != 2^ell_cycle={pow2_cycle}",
                    col.len()
                )));
            }
        }

        let eq_cycle_table = build_eq_table(r_cycle);

        // Build sparse weight table: weight[a] = Σ_t eq(r_cycle, t) * has_lookup(t) for steps hitting address a.
        let mut weights: HashMap<u64, K> = HashMap::new();
        for t in 0..pow2_cycle {
            let gate = has_lookup[t];
            if gate == K::ZERO {
                continue;
            }

            // Decode address bits at time t into a u64 key.
            let mut addr_t: u64 = 0;
            for b in 0..ell_total {
                if addr_bits[b][t] == K::ONE {
                    addr_t |= 1u64 << b;
                }
            }

            let w_t = eq_cycle_table[t] * gate;
            *weights.entry(addr_t).or_insert(K::ZERO) += w_t;
        }

        // Claimed sum = Σ_a Table(a) * weight(a), summed over sparse support.
        let mut claimed_sum = K::ZERO;
        for (&addr, &w) in weights.iter() {
            let (rs1, rs2) = uninterleave_bits(addr as u128);
            let out = compute_op(opcode, rs1, rs2, xlen);
            claimed_sum += K::from_u64(out) * w;
        }

        Ok((
            Self {
                opcode,
                xlen,
                ell_total,
                rounds_remaining: ell_total,
                bound_prefix: Vec::with_capacity(ell_total),
                weights,
            },
            claimed_sum,
        ))
    }

    fn table_endpoint(&self, cur_bit: u64, rest_bits: u64) -> K {
        debug_assert!(self.rounds_remaining > 0);
        let bound_len = self.bound_prefix.len();
        let rest_len = self.rounds_remaining - 1;
        debug_assert_eq!(bound_len + self.rounds_remaining, self.ell_total);

        let mut r: Vec<K> = Vec::with_capacity(self.ell_total);
        r.extend_from_slice(&self.bound_prefix);
        r.push(if cur_bit == 1 { K::ONE } else { K::ZERO });
        for j in 0..rest_len {
            let bit = (rest_bits >> j) & 1;
            r.push(if bit == 1 { K::ONE } else { K::ZERO });
        }
        debug_assert_eq!(r.len(), self.ell_total);

        evaluate_opcode_mle(self.opcode, &r, self.xlen)
    }
}

impl RoundOracle for RiscvAddressLookupOracleSparse {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.rounds_remaining == 0 {
            let w = self.weights.get(&0).copied().unwrap_or(K::ZERO);
            let t = evaluate_opcode_mle(self.opcode, &self.bound_prefix, self.xlen);
            let v = t * w;
            return vec![v; points.len()];
        }

        // Group weights by (rest_bits = idx >> 1) into (w0, w1).
        let mut pair_weights: HashMap<u64, (K, K)> = HashMap::new();
        for (&idx, &w) in self.weights.iter() {
            let pair = idx >> 1;
            let entry = pair_weights.entry(pair).or_insert((K::ZERO, K::ZERO));
            if (idx & 1) == 0 {
                entry.0 = w;
            } else {
                entry.1 = w;
            }
        }

        // Accumulate quadratic coefficients over all active pairs.
        let mut c0 = K::ZERO;
        let mut c1 = K::ZERO;
        let mut c2 = K::ZERO;

        for (pair, (w0, w1)) in pair_weights.into_iter() {
            if w0 == K::ZERO && w1 == K::ZERO {
                continue;
            }
            let t0 = self.table_endpoint(0, pair);
            let t1 = self.table_endpoint(1, pair);
            let dw = w1 - w0;
            let dt = t1 - t0;

            c0 += t0 * w0;
            c1 += t0 * dw + w0 * dt;
            c2 += dt * dw;
        }

        points
            .iter()
            .map(|&x| {
                let x2 = x * x;
                c0 + c1 * x + c2 * x2
            })
            .collect()
    }

    fn num_rounds(&self) -> usize {
        self.rounds_remaining
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn fold(&mut self, r: K) {
        if self.rounds_remaining == 0 {
            return;
        }

        let one_minus_r = K::ONE - r;
        let mut next: HashMap<u64, K> = HashMap::with_capacity(self.weights.len());
        for (&idx, &w) in self.weights.iter() {
            let pair = idx >> 1;
            let contrib = if (idx & 1) == 0 { w * one_minus_r } else { w * r };
            if contrib == K::ZERO {
                continue;
            }
            *next.entry(pair).or_insert(K::ZERO) += contrib;
        }

        self.weights = next;
        self.bound_prefix.push(r);
        self.rounds_remaining -= 1;
    }
}
