//! Owns static VM architecture contracts.

pub mod decode;
pub mod opcode_classes;
pub mod r1cs_builder;
pub mod state;

use neo_ccs::CcsStructure;
use neo_math::F;

pub use decode::{DecodeField, DecodeSpec};
pub use opcode_classes::OpcodeClassSpec;
pub use state::{RegisterSpec, StateSpec};

#[derive(Clone, Debug)]
pub struct CoreCcsSpec {
    pub structure: CcsStructure<F>,
    pub m_in: usize,
    pub witness_width: usize,
    pub const_one_col: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShoutTableSpec {
    pub name: &'static str,
    pub slots: usize,
    pub width_bits: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TwistTableSpec {
    pub name: &'static str,
    pub slots: usize,
    pub width_bits: u16,
}

pub trait VmSpec {
    type OpcodeId: Copy + Eq + core::hash::Hash;

    fn name(&self) -> &'static str;
    fn state_spec(&self) -> StateSpec;
    fn shout_tables(&self) -> Vec<ShoutTableSpec>;
    fn twist_tables(&self) -> Vec<TwistTableSpec>;
    fn opcode_classes(&self) -> Vec<OpcodeClassSpec<Self::OpcodeId>>;
    fn decode_spec(&self) -> DecodeSpec<Self::OpcodeId>;
    fn core_ccs_spec(&self) -> &CoreCcsSpec;
}

pub trait VmTraceBuilder<V: VmSpec> {
    type Program;
    type MachineState;
    type StepTrace;
    type Error;

    fn build_step(
        &self,
        vm: &V,
        program: &Self::Program,
        prev: &Self::MachineState,
        next: &Self::MachineState,
        trace: &Self::StepTrace,
    ) -> Result<crate::step_build::StepBuild, Self::Error>;
}
