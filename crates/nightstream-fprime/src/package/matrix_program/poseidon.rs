//! Invocation-major Poseidon2 rows from the Lean formula library.

use std::ops::ControlFlow;

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::poseidon_input::Program as InputProgram;
use super::{
    checked_add, checked_mul, exact_array, owned_row, template, usize_atom, Form, PackageError, RetainedBlock,
    RetainedKind, RowForms, RowView,
};

const ROWS_PER_INVOCATION: usize = 94;
const SBOX_ROWS_PER_INVOCATION: usize = 86;
const WIDTH: usize = 8;

#[derive(Clone, Debug)]
pub(super) struct Block {
    invocation_count: usize,
    one_column: usize,
    retained: RetainedBlock,
    input: InputProgram,
}

impl Block {
    pub(super) fn decode(value: &Value) -> Result<Self, PackageError> {
        let fields = exact_array(value, 4, "Poseidon2 matrix block")?;
        Ok(Self {
            invocation_count: usize_atom(&fields[0], "Poseidon2 matrix invocation count")?,
            one_column: usize_atom(&fields[1], "Poseidon2 matrix one column")?,
            retained: RetainedBlock::decode(&fields[2])?,
            input: InputProgram::decode(&fields[3])?,
        })
    }

    pub(super) fn row_count(&self) -> Result<usize, PackageError> {
        checked_mul(self.invocation_count, ROWS_PER_INVOCATION, "Poseidon2 matrix row count")
    }

    fn validate(&self, logical_width: usize) -> Result<(), PackageError> {
        if self.one_column >= logical_width {
            return Err(PackageError::Invalid("Poseidon2 matrix one column"));
        }
        if self.retained.kind() != RetainedKind::Field {
            return Err(PackageError::Invalid("Poseidon2 retained kind"));
        }
        let expected_slots = checked_mul(
            self.invocation_count,
            SBOX_ROWS_PER_INVOCATION,
            "Poseidon2 retained slot count",
        )?;
        if self.retained.slot_count() != expected_slots || !self.retained.fits(logical_width)? {
            return Err(PackageError::Invalid("Poseidon2 retained geometry"));
        }
        Ok(())
    }

    pub(super) fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms, PackageError> {
        if ordinal >= self.row_count()? {
            return Err(PackageError::Invalid("Poseidon2 matrix row ordinal"));
        }
        let mut result = None;
        let _ = self.visit_rows_until(logical_width, ordinal, ordinal + 1, |row| {
            result = Some(owned_row(row));
            Ok(ControlFlow::Break(()))
        })?;
        result.ok_or(PackageError::Invalid("Poseidon2 template row count"))
    }

    pub(super) fn visit_rows_until(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        mut visit: impl FnMut(RowView<'_>) -> Result<ControlFlow<()>, PackageError>,
    ) -> Result<ControlFlow<()>, PackageError> {
        if start > end || end > self.row_count()? {
            return Err(PackageError::Invalid("Poseidon2 matrix row range"));
        }
        if start == end {
            return Ok(ControlFlow::Continue(()));
        }
        self.validate(logical_width)?;
        let mut scratch = template::RowScratch::default();
        let first_invocation = start / ROWS_PER_INVOCATION;
        let last_invocation = (end - 1) / ROWS_PER_INVOCATION;
        for invocation in first_invocation..=last_invocation {
            let invocation_start = checked_mul(invocation, ROWS_PER_INVOCATION, "Poseidon2 matrix row")?;
            let local_start = start
                .saturating_sub(invocation_start)
                .min(ROWS_PER_INVOCATION);
            let local_end = end
                .saturating_sub(invocation_start)
                .min(ROWS_PER_INVOCATION);
            let inputs = self.invocation_inputs(logical_width, invocation)?;
            if scratch
                .visit_rows_until(
                    "poseidon2-permutation-v1",
                    0,
                    &inputs,
                    logical_width,
                    local_start..local_end,
                    &mut visit,
                )?
                .is_break()
            {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    fn invocation_inputs(&self, logical_width: usize, invocation: usize) -> Result<Vec<Form>, PackageError> {
        let mut inputs = Vec::with_capacity(1 + WIDTH + SBOX_ROWS_PER_INVOCATION);
        inputs.push(Form::singleton(self.one_column, Goldilocks::ONE));
        inputs.extend(
            self.input
                .state(logical_width, self.one_column, invocation)?,
        );
        let slot_base = checked_mul(invocation, SBOX_ROWS_PER_INVOCATION, "Poseidon2 retained slot")?;
        for slot in 0..SBOX_ROWS_PER_INVOCATION {
            inputs.push(
                self.retained
                    .form(logical_width, checked_add(slot_base, slot, "Poseidon2 retained slot")?)?,
            );
        }
        Ok(inputs)
    }
}
