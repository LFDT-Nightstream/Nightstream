//! Owns frontend-produced step-build records and per-step extension data.
//!
//! It owns:
//! - frontend step packaging (`StepBuild`)
//! - bytecode/register/RAM extension records
//!
//! It does not own:
//! - generic session proof objects
//! - VM execution semantics
//! - kernel proof construction

use serde::{Deserialize, Serialize};

use crate::proof::{PublicStep, StepInput};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BytecodeFetchRecord {
    pub pc: u16,
    pub opcode: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RegisterBank {
    V,
    I,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisterAccessRecord {
    pub bank: RegisterBank,
    pub index: u8,
    pub value: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RamAccessRecord {
    pub addr: u16,
    pub value: u8,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepExtensionData {
    pub bytecode_fetch: Option<BytecodeFetchRecord>,
    pub register_reads: Vec<RegisterAccessRecord>,
    pub register_writes: Vec<RegisterAccessRecord>,
    pub ram_reads: Vec<RamAccessRecord>,
    pub ram_writes: Vec<RamAccessRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepBuild {
    pub prepared: StepInput,
    pub public_step: PublicStep,
    pub extension_data: StepExtensionData,
}
