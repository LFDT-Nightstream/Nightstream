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

use neo_math::{F, K};
use neo_memory::bit_ops::eq_bit_affine;
use neo_memory::output_check::{OutputCheckError, ProgramIO};
use p3_field::PrimeCharacteristicRing;

/// Configuration for output binding.
#[derive(Clone, Debug)]
pub struct OutputBindingConfig {
    /// Number of address bits for memory.
    pub num_bits: usize,
    /// The claimed program I/O (inputs and outputs with their addresses).
    pub program_io: ProgramIO<F>,
    /// Which mem instance to bind outputs against (default: 0).
    pub mem_idx: usize,
}

/// Label for the optional Route-A batched time claim that binds output sumcheck to Twist increments.
pub const OB_INC_TOTAL_LABEL: &'static [u8] = b"output_binding/inc_total";

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
    if open.wa_bits.len() != r_prime.len() {
        return Err(OutputCheckError::DimensionMismatch {
            expected: r_prime.len(),
            got: open.wa_bits.len(),
        });
    }

    let mut eq = K::ONE;
    for (bit, &u) in open.wa_bits.iter().zip(r_prime.iter()) {
        eq *= eq_bit_affine(*bit, u);
    }

    Ok(open.has_write * open.inc_at_write_addr * eq)
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
