//! Owns shared Construction-2 terminal committed-step data shapes.

use bellpepper_core::num::AllocatedNum;
use neo_math::F;

use crate::spartan_backend::SpartanF;

pub(crate) const U32_BIT_WIDTH: usize = 32;
pub(crate) const U64_BIT_WIDTH: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TerminalPrivateColumnEncoding {
    UnusedPadding,
    Bit,
    U32,
    U64,
}

impl TerminalPrivateColumnEncoding {
    pub(crate) fn limb_count(self) -> usize {
        match self {
            Self::UnusedPadding => 0,
            Self::Bit => 1,
            Self::U32 => U32_BIT_WIDTH,
            Self::U64 => U64_BIT_WIDTH,
        }
    }

    pub(crate) fn limb_label(self, limb_idx: usize) -> String {
        match self {
            Self::UnusedPadding => "padding".to_string(),
            Self::Bit => "bit".to_string(),
            Self::U32 | Self::U64 => format!("bit{limb_idx}"),
        }
    }
}

pub(crate) struct Construction2TerminalBoundaryView<'a> {
    pub(crate) fresh_instance_digest: [u8; 32],
    pub(crate) commitment_digest: [u8; 32],
    pub(crate) commitment_d: u64,
    pub(crate) commitment_kappa: u64,
    pub(crate) commitment_data: &'a [F],
    pub(crate) x_i_bytes: [u8; 32],
}

pub(crate) struct Construction2TerminalBoundaryInputs {
    pub(crate) fresh_instance_digest: [AllocatedNum<SpartanF>; 4],
    pub(crate) commitment_digest: [AllocatedNum<SpartanF>; 4],
    pub(crate) commitment_d: AllocatedNum<SpartanF>,
    pub(crate) commitment_kappa: AllocatedNum<SpartanF>,
    pub(crate) commitment_data: Vec<AllocatedNum<SpartanF>>,
    pub(crate) x_i: [AllocatedNum<SpartanF>; 4],
}
