//! Public statement for a Spartan-compressed Neo shard/FoldRun proof.
//!
//! This statement binds the verifier context (params/CCS/step metadata) and accumulator endpoints
//! for a Spartan-compressed Neo shard/FoldRun proof.

use crate::CircuitF;
use serde::{Deserialize, Serialize};

/// Current public-statement version supported by this crate.
pub const STATEMENT_VERSION: u32 = 5;

// Statement public-IO layout (indices into `SpartanShardStatement::public_io()`).
pub const STATEMENT_IO_VERSION: usize = 0;
pub const STATEMENT_IO_PARAMS_DIGEST_OFFSET: usize = 1;
pub const STATEMENT_IO_CCS_DIGEST_OFFSET: usize = STATEMENT_IO_PARAMS_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_VM_DIGEST_OFFSET: usize = STATEMENT_IO_CCS_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_STEPS_DIGEST_OFFSET: usize = STATEMENT_IO_VM_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_PROGRAM_IO_DIGEST_OFFSET: usize = STATEMENT_IO_STEPS_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_STEP_LINKING_DIGEST_OFFSET: usize = STATEMENT_IO_PROGRAM_IO_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_ACC_INIT_DIGEST_OFFSET: usize = STATEMENT_IO_STEP_LINKING_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_ACC_FINAL_MAIN_DIGEST_OFFSET: usize = STATEMENT_IO_ACC_INIT_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_ACC_FINAL_VAL_DIGEST_OFFSET: usize = STATEMENT_IO_ACC_FINAL_MAIN_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_STEP_COUNT_OFFSET: usize = STATEMENT_IO_ACC_FINAL_VAL_DIGEST_OFFSET + 4;
pub const STATEMENT_IO_MEM_ENABLED_OFFSET: usize = STATEMENT_IO_STEP_COUNT_OFFSET + 1;
pub const STATEMENT_IO_OUTPUT_BINDING_ENABLED_OFFSET: usize = STATEMENT_IO_MEM_ENABLED_OFFSET + 1;
pub const STATEMENT_IO_LEN: usize = STATEMENT_IO_OUTPUT_BINDING_ENABLED_OFFSET + 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpartanShardStatement {
    /// Version tag for domain separation / upgrades.
    pub version: u32,

    /// Bind to system/circuit context.
    pub params_digest: [u8; 32],
    pub ccs_digest: [u8; 32],

    /// Bind to VM/program context (digest of raw ROM/ELF bytes + VM configuration).
    pub vm_digest: [u8; 32],

    /// Bind to step-level public metadata (MCS instances + memory/table config).
    /// In Phase 1 this includes `absorb_step_memory(step)` + MCS instance fields.
    pub steps_digest: [u8; 32],

    /// Bind to claimed program outputs if output binding is enabled (else zero).
    pub program_io_digest: [u8; 32],

    /// Bind to the step-linking policy (pairs of x-coordinate equalities enforced at boundaries).
    pub step_linking_digest: [u8; 32],

    /// Bind to the initial main-lane accumulator (digest, not full public IO).
    pub acc_init_digest: [u8; 32],

    /// Bind to the final main-lane accumulator (digest, not full public IO).
    pub acc_final_main_digest: [u8; 32],

    /// Bind to the final val-lane obligations accumulator (digest).
    pub acc_final_val_digest: [u8; 32],

    /// Bind to execution shape.
    pub step_count: u32,

    /// Feature flags (pinned into the public statement).
    pub mem_enabled: bool,
    pub output_binding_enabled: bool,
}

fn push_u64(out: &mut Vec<CircuitF>, x: u64) {
    out.push(CircuitF::from(x));
}

fn push_digest(out: &mut Vec<CircuitF>, digest: &[u8; 32]) {
    for chunk in digest.chunks(8) {
        let mut limb_bytes = [0u8; 8];
        limb_bytes.copy_from_slice(chunk);
        push_u64(out, u64::from_le_bytes(limb_bytes));
    }
}

impl SpartanShardStatement {
    pub fn new(
        params_digest: [u8; 32],
        ccs_digest: [u8; 32],
        vm_digest: [u8; 32],
        steps_digest: [u8; 32],
        program_io_digest: [u8; 32],
        step_linking_digest: [u8; 32],
        acc_init_digest: [u8; 32],
        acc_final_main_digest: [u8; 32],
        acc_final_val_digest: [u8; 32],
        step_count: u32,
        mem_enabled: bool,
        output_binding_enabled: bool,
    ) -> Self {
        Self {
            version: STATEMENT_VERSION,
            params_digest,
            ccs_digest,
            vm_digest,
            steps_digest,
            program_io_digest,
            step_linking_digest,
            acc_init_digest,
            acc_final_main_digest,
            acc_final_val_digest,
            step_count,
            mem_enabled,
            output_binding_enabled,
        }
    }

    /// Encode the statement as Spartan public IO field elements.
    ///
    /// Canonical encoding:
    /// - `version` as one u64 limb
    /// - each digest as 4x u64 limbs (little-endian)
    /// - `step_count` as one u64 limb
    /// - booleans as 0/1 u64 limbs
    pub fn public_io(&self) -> Vec<CircuitF> {
        let mut out = Vec::with_capacity(STATEMENT_IO_LEN);
        push_u64(&mut out, self.version as u64);
        push_digest(&mut out, &self.params_digest);
        push_digest(&mut out, &self.ccs_digest);
        push_digest(&mut out, &self.vm_digest);
        push_digest(&mut out, &self.steps_digest);
        push_digest(&mut out, &self.program_io_digest);
        push_digest(&mut out, &self.step_linking_digest);
        push_digest(&mut out, &self.acc_init_digest);
        push_digest(&mut out, &self.acc_final_main_digest);
        push_digest(&mut out, &self.acc_final_val_digest);
        push_u64(&mut out, self.step_count as u64);
        push_u64(&mut out, self.mem_enabled as u64);
        push_u64(&mut out, self.output_binding_enabled as u64);
        out
    }
}
