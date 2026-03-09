use crate::riscv::elf_loader::LoadedProgram;
use crate::riscv::lookups::{RiscvInstruction, RiscvMemOp, RiscvOpcode};
use crate::riscv::memory_layout::ProofAddressRemapKind;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiscvTraceProfileKind {
    Rv64Im,
}

/// Minimal verifier-visible profile configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiscvProofProfileConfig {
    pub xlen: usize,
    pub compressed: bool,
    pub atomics: bool,
    pub poseidon_precompile: bool,
    pub lowering_version: u32,
    pub memory_layout_kind: ProofAddressRemapKind,
    pub trace_profile: RiscvTraceProfileKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiscvProofProfile {
    config: RiscvProofProfileConfig,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RiscvProofProfileError {
    #[error("unsupported xlen {0}; expected 64 for the supported RV64IM profile")]
    UnsupportedXlen(usize),
    #[error("RV64IM profile does not support compressed instructions")]
    CompressedNotSupported,
    #[error("RV64IM profile does not support atomics")]
    AtomicsNotSupported,
    #[error("invalid lowering_version {got}; expected {expected}")]
    InvalidLoweringVersion { got: u32, expected: u32 },
    #[error("profile xlen={profile_xlen} does not match loaded program xlen={program_xlen}")]
    XlenMismatch {
        profile_xlen: usize,
        program_xlen: usize,
    },
    #[error(
        "compressed instructions are present in executable ELF segments; the current RV64IM profile does not support C"
    )]
    CompressedProgramNotSupported,
    #[error("unsupported instruction for the current RV64IM profile: {0}")]
    UnsupportedInstruction(String),
}

impl RiscvProofProfile {
    pub const RV64IM_LOWERING_VERSION: u32 = 1;

    pub fn new(config: RiscvProofProfileConfig) -> Result<Self, RiscvProofProfileError> {
        match config.trace_profile {
            RiscvTraceProfileKind::Rv64Im => {
                if config.xlen != 64 {
                    return Err(RiscvProofProfileError::UnsupportedXlen(config.xlen));
                }
                if config.compressed {
                    return Err(RiscvProofProfileError::CompressedNotSupported);
                }
                if config.atomics {
                    return Err(RiscvProofProfileError::AtomicsNotSupported);
                }
                if config.lowering_version != Self::RV64IM_LOWERING_VERSION {
                    return Err(RiscvProofProfileError::InvalidLoweringVersion {
                        got: config.lowering_version,
                        expected: Self::RV64IM_LOWERING_VERSION,
                    });
                }
            }
        }
        Ok(Self { config })
    }

    pub fn rv64im() -> Self {
        Self::new(RiscvProofProfileConfig {
            xlen: 64,
            compressed: false,
            atomics: false,
            poseidon_precompile: cfg!(feature = "poseidon-precompile"),
            lowering_version: Self::RV64IM_LOWERING_VERSION,
            memory_layout_kind: ProofAddressRemapKind::SegmentedWordAddress,
            trace_profile: RiscvTraceProfileKind::Rv64Im,
        })
        .expect("RV64IM profile is valid")
    }

    pub fn config(&self) -> &RiscvProofProfileConfig {
        &self.config
    }

    pub fn xlen(&self) -> usize {
        self.config.xlen
    }

    pub fn validate_loaded_program(&self, loaded: &LoadedProgram) -> Result<(), RiscvProofProfileError> {
        let program_xlen = if loaded.is_64bit { 64 } else { 32 };
        if self.config.xlen != program_xlen {
            return Err(RiscvProofProfileError::XlenMismatch {
                profile_xlen: self.config.xlen,
                program_xlen,
            });
        }
        if !self.config.compressed && loaded.contains_compressed_executable_code() {
            return Err(RiscvProofProfileError::CompressedProgramNotSupported);
        }
        for (_, instruction) in &loaded.instructions {
            self.validate_instruction(instruction)?;
        }
        Ok(())
    }

    pub fn validate_instruction(&self, instruction: &RiscvInstruction) -> Result<(), RiscvProofProfileError> {
        if self.supports_instruction(instruction) {
            Ok(())
        } else {
            Err(RiscvProofProfileError::UnsupportedInstruction(format!(
                "{instruction:?}"
            )))
        }
    }

    pub fn supports_instruction(&self, instruction: &RiscvInstruction) -> bool {
        match self.config.trace_profile {
            RiscvTraceProfileKind::Rv64Im => supports_rv64im_instruction(instruction, self.config.poseidon_precompile),
        }
    }
}

fn supports_rv64im_instruction(instruction: &RiscvInstruction, poseidon_precompile: bool) -> bool {
    match instruction {
        RiscvInstruction::RAlu { op, .. } | RiscvInstruction::IAlu { op, .. } => supports_rv64im_opcode(*op),
        RiscvInstruction::Load { op, .. } | RiscvInstruction::Store { op, .. } => supports_rv64im_mem_op(*op),
        RiscvInstruction::Branch { .. }
        | RiscvInstruction::Jal { .. }
        | RiscvInstruction::Jalr { .. }
        | RiscvInstruction::Lui { .. }
        | RiscvInstruction::Auipc { .. }
        | RiscvInstruction::Fence { .. }
        | RiscvInstruction::Halt
        | RiscvInstruction::Nop
        | RiscvInstruction::Ecall => true,
        RiscvInstruction::RAluw { op, .. } | RiscvInstruction::IAluw { op, .. } => {
            matches!(
                op,
                RiscvOpcode::Addw
                    | RiscvOpcode::Subw
                    | RiscvOpcode::Sllw
                    | RiscvOpcode::Srlw
                    | RiscvOpcode::Sraw
                    | RiscvOpcode::Mulw
                    | RiscvOpcode::Divw
                    | RiscvOpcode::Divuw
                    | RiscvOpcode::Remw
                    | RiscvOpcode::Remuw
            )
        }
        RiscvInstruction::Poseidon2AbsorbElem { .. }
        | RiscvInstruction::Poseidon2Finalize
        | RiscvInstruction::Poseidon2SqueezeWord { .. } => poseidon_precompile,
        RiscvInstruction::LoadReserved { .. }
        | RiscvInstruction::StoreConditional { .. }
        | RiscvInstruction::Amo { .. }
        | RiscvInstruction::Ebreak
        | RiscvInstruction::FenceI => false,
    }
}

fn supports_rv64im_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::And
            | RiscvOpcode::Xor
            | RiscvOpcode::Or
            | RiscvOpcode::Sub
            | RiscvOpcode::Add
            | RiscvOpcode::Mul
            | RiscvOpcode::Mulh
            | RiscvOpcode::Mulhu
            | RiscvOpcode::Mulhsu
            | RiscvOpcode::Div
            | RiscvOpcode::Divu
            | RiscvOpcode::Rem
            | RiscvOpcode::Remu
            | RiscvOpcode::Sltu
            | RiscvOpcode::Slt
            | RiscvOpcode::Eq
            | RiscvOpcode::Neq
            | RiscvOpcode::Sll
            | RiscvOpcode::Srl
            | RiscvOpcode::Sra
            | RiscvOpcode::Addw
            | RiscvOpcode::Subw
            | RiscvOpcode::Sllw
            | RiscvOpcode::Srlw
            | RiscvOpcode::Sraw
            | RiscvOpcode::Mulw
            | RiscvOpcode::Divw
            | RiscvOpcode::Divuw
            | RiscvOpcode::Remw
            | RiscvOpcode::Remuw
    )
}

fn supports_rv64im_mem_op(op: RiscvMemOp) -> bool {
    matches!(
        op,
        RiscvMemOp::Lb
            | RiscvMemOp::Lbu
            | RiscvMemOp::Lh
            | RiscvMemOp::Lhu
            | RiscvMemOp::Lw
            | RiscvMemOp::Lwu
            | RiscvMemOp::Ld
            | RiscvMemOp::Sb
            | RiscvMemOp::Sh
            | RiscvMemOp::Sw
            | RiscvMemOp::Sd
    )
}
