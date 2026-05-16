//! Lane layout for low-norm direct R1CS lowering.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectLowNormLaneKind {
    Bit,
    U32,
    Field,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectR1csLowNormLayout {
    pub(super) kinds: Vec<DirectLowNormLaneKind>,
    pub(super) public_input_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LaneMap {
    pub(super) bits_start_col: usize,
    pub(super) bit_len: usize,
    pub(super) canonical_aux_start_col: Option<usize>,
}

impl DirectLowNormLaneKind {
    pub(super) fn bit_len(self) -> usize {
        match self {
            Self::Bit => 1,
            Self::U32 => U32_BITS,
            Self::Field => FIELD_BITS,
        }
    }

    pub(super) fn needs_canonical_field_check(self) -> bool {
        matches!(self, Self::Field)
    }
}

impl DirectR1csLowNormLayout {
    pub fn new(public_input_len: usize, kinds: Vec<DirectLowNormLaneKind>) -> Result<Self, DirectCcsFPrimeSnarkError> {
        if public_input_len > kinds.len() {
            return Err(DirectCcsFPrimeSnarkError::Input(format!(
                "direct low-norm R1CS layout public input len {public_input_len} exceeds variable count {}",
                kinds.len()
            )));
        }
        Ok(Self {
            kinds,
            public_input_len,
        })
    }

    pub fn conservative_for_export(export: &DirectSparseR1csExport) -> Self {
        let mut kinds = vec![DirectLowNormLaneKind::Field; export.variable_count];
        if !kinds.is_empty() && export.witness.first().copied() == Some(F::ONE) {
            kinds[0] = DirectLowNormLaneKind::Bit;
        }
        Self {
            kinds,
            public_input_len: export.public_input_len,
        }
    }

    pub fn kinds(&self) -> &[DirectLowNormLaneKind] {
        &self.kinds
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }
}

impl LaneMap {
    pub(super) fn empty() -> Self {
        Self {
            bits_start_col: 0,
            bit_len: 0,
            canonical_aux_start_col: None,
        }
    }
}
