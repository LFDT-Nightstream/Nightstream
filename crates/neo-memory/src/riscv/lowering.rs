use crate::riscv::elf_loader::LoadedProgram;
use crate::riscv::lookups::RiscvInstruction;
use crate::riscv::profile::{RiscvProofProfile, RiscvProofProfileError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoweringProfile {
    IdentityV0,
    Rv64NoteCircuitsV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoweredInstruction {
    Passthrough(RiscvInstruction),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweredProgram {
    pub entry: u64,
    pub is_64bit: bool,
    pub lowering_profile: LoweringProfile,
    pub instructions: Vec<(u64, LoweredInstruction)>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LoweringError {
    #[error(transparent)]
    Profile(#[from] RiscvProofProfileError),
}

impl LoweredProgram {
    pub fn get_instructions(&self) -> Vec<RiscvInstruction> {
        self.instructions
            .iter()
            .map(|(_, lowered)| match lowered {
                LoweredInstruction::Passthrough(instruction) => instruction.clone(),
            })
            .collect()
    }
}

pub fn lower_loaded_program(
    loaded: &LoadedProgram,
    profile: &RiscvProofProfile,
) -> Result<LoweredProgram, LoweringError> {
    profile.validate_loaded_program(loaded)?;
    let lowering_profile = if profile.xlen() == 64 {
        LoweringProfile::Rv64NoteCircuitsV1
    } else {
        LoweringProfile::IdentityV0
    };
    let instructions = loaded
        .instructions
        .iter()
        .cloned()
        .map(|(addr, instruction)| (addr, LoweredInstruction::Passthrough(instruction)))
        .collect();
    Ok(LoweredProgram {
        entry: loaded.entry,
        is_64bit: loaded.is_64bit,
        lowering_profile,
        instructions,
    })
}
