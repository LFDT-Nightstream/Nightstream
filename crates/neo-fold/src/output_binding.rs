//! Output Binding Integration for Shard Proofs.
//!
//! This module provides helpers to add output binding proofs to shard proofs,
//! ensuring that claimed program outputs are cryptographically bound to the
//! proven execution trace.
//!
//! ## Usage
//!
//! Prefer the shard/session wrappers which wire Route-A `r_time` into output binding automatically:
//!
//! ```ignore
//! let proof = neo_fold::shard::fold_shard_prove_with_output_binding(..., &ob_cfg, &final_memory_state)?;
//! neo_fold::shard::fold_shard_verify_with_output_binding(..., &proof, ..., &ob_cfg)?;
//! ```

use neo_math::{from_complex, F, K};
use neo_memory::bit_ops::eq_bit_affine;
use neo_memory::output_check::{OutputCheckError, ProgramIO};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

/// Configuration for output binding.
#[derive(Clone, Debug)]
pub struct OutputBindingConfig {
    /// Number of address bits for memory.
    pub num_bits: usize,
    /// The claimed program I/O (inputs and outputs with their addresses).
    pub program_io: ProgramIO<F>,
    /// Which mem instance to bind outputs against (default: 0).
    ///
    /// This is an index into `StepWitnessBundle.mem_instances` (i.e. “which Twist instance” in the
    /// per-step instance list). In shared-CPU-bus mode, the builder orders Twist instances by
    /// sorted `twist_id`, and `FoldingSession` uses `ShardWitnessAux.mem_ids` to map indices back
    /// to concrete `twist_id`s when deriving terminal memory state.
    pub mem_idx: usize,
}

/// Label for the optional Route-A batched time claim that binds output sumcheck to Twist increments.
pub const OB_INC_TOTAL_LABEL: &'static [u8] = b"output_binding/inc_total";
/// Label for the optional Route-A batched time claim that binds synthetic exact RV64
/// register-output writes back to the CPU writeback columns.
pub const OB_REG_EXACT_LINKAGE_LABEL: &'static [u8] = b"output_binding/reg_exact_linkage";

impl OutputBindingConfig {
    /// Create a new output binding config with just the I/O claims.
    pub fn new(num_bits: usize, program_io: ProgramIO<F>) -> Self {
        Self {
            num_bits,
            program_io,
            mem_idx: 0,
        }
    }

    pub fn with_mem_idx(mut self, mem_idx: usize) -> Self {
        self.mem_idx = mem_idx;
        self
    }
}

pub(crate) fn addr_bits_as_k(addr: u64, num_bits: usize) -> Vec<K> {
    (0..num_bits)
        .map(|bit| if ((addr >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
        .collect()
}

pub(crate) fn sample_output_lincomb_weights(
    tr: &mut Poseidon2Transcript,
    program_io: &ProgramIO<F>,
) -> Vec<(u64, F, K)> {
    program_io.absorb_into_transcript(tr);
    program_io
        .claims()
        .map(|(addr, value)| {
            tr.append_message(b"output_binding/lincomb/addr", &addr.to_le_bytes());
            let alpha = from_complex(
                tr.challenge_field(b"output_binding/lincomb/alpha/re"),
                tr.challenge_field(b"output_binding/lincomb/alpha/im"),
            );
            (addr, value, alpha)
        })
        .collect()
}

pub(crate) fn val_init_from_mem_init(
    init: &neo_memory::MemInit<F>,
    k: usize,
    r_prime: &[K],
) -> Result<K, OutputCheckError> {
    neo_memory::mem_init::eval_init_at_r_addr::<F, K>(init, k, r_prime)
        .map_err(|e| OutputCheckError::External(format!("MemInit eval failed: {e}")))
}

pub(crate) fn inc_terminal_from_time_openings(
    open: &crate::memory_sidecar::memory::TwistTimeLaneOpenings,
    r_prime: &[K],
) -> Result<K, OutputCheckError> {
    let mut total = K::ZERO;
    for lane in open.lanes.iter() {
        if lane.wa_bits.len() != r_prime.len() {
            return Err(OutputCheckError::DimensionMismatch {
                expected: r_prime.len(),
                got: lane.wa_bits.len(),
            });
        }

        let mut eq = K::ONE;
        for (bit, &u) in lane.wa_bits.iter().zip(r_prime.iter()) {
            eq *= eq_bit_affine(*bit, u);
        }

        total += lane.has_write * lane.inc_at_write_addr * eq;
    }
    Ok(total)
}

pub(crate) fn weighted_inc_terminal_from_time_openings(
    open: &crate::memory_sidecar::memory::TwistTimeLaneOpenings,
    addr_weights: &[(Vec<K>, K)],
) -> Result<K, OutputCheckError> {
    let mut total = K::ZERO;
    for (r_addr, weight) in addr_weights {
        if *weight == K::ZERO {
            continue;
        }
        let mut per_addr_total = K::ZERO;
        for lane in open.lanes.iter() {
            if lane.wa_bits.len() != r_addr.len() {
                return Err(OutputCheckError::DimensionMismatch {
                    expected: r_addr.len(),
                    got: lane.wa_bits.len(),
                });
            }
            let mut eq = K::ONE;
            for (bit, &u) in lane.wa_bits.iter().zip(r_addr.iter()) {
                eq *= eq_bit_affine(*bit, u);
            }
            per_addr_total += lane.has_write * lane.inc_at_write_addr * eq;
        }
        total += *weight * per_addr_total;
    }
    Ok(total)
}

/// Check if a shard proof has output binding attached.
pub fn has_output_binding(proof: &crate::shard_proof_types::ShardProof) -> bool {
    proof.output_proof.is_some()
}

/// Create a simple output binding config for testing.
///
/// This creates a ProgramIO with a single output at the specified address.
pub fn simple_output_config(num_bits: usize, output_addr: u64, expected_output: F) -> OutputBindingConfig {
    let program_io = ProgramIO::new().with_output(output_addr, expected_output);
    OutputBindingConfig::new(num_bits, program_io)
}
