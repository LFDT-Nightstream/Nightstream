//! Invocation-major 94-row Poseidon2 matrix blocks.

use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::{round_constants, Poseidon2RoundConstants};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::poseidon_input::Program as InputProgram;
use super::{
    checked_add, checked_mul, empty_row, exact_array, external_layer, usize_atom, Form, PackageError, RetainedBlock,
    RetainedKind, RowForms,
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

    pub(super) fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms, PackageError> {
        if ordinal >= self.row_count()? {
            return Err(PackageError::Invalid("Poseidon2 matrix row ordinal"));
        }
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

        let invocation = ordinal / ROWS_PER_INVOCATION;
        let local_row = ordinal % ROWS_PER_INVOCATION;
        if local_row < SBOX_ROWS_PER_INVOCATION {
            self.sbox_row(logical_width, invocation, local_row)
        } else {
            self.pin_row(logical_width, invocation, local_row - SBOX_ROWS_PER_INVOCATION)
        }
    }

    fn sbox_row(&self, logical_width: usize, invocation: usize, target: usize) -> Result<RowForms, PackageError> {
        let constants = constants();
        let input = self
            .input
            .state(logical_width, self.one_column, invocation)?;
        let mut state = to_state(external_layer(&input)?);
        let slot_base = checked_mul(invocation, SBOX_ROWS_PER_INVOCATION, "Poseidon2 retained slot")?;
        let mut next_sbox = 0usize;

        for round in &constants.initial {
            let mut outputs = empty_state();
            for lane in 0..WIDTH {
                let row_input = add_constant(state[lane].clone(), self.one_column, Goldilocks::from_u64(round[lane]));
                let output = self.retained.form(
                    logical_width,
                    checked_add(slot_base, next_sbox, "Poseidon2 retained slot")?,
                )?;
                if next_sbox == target {
                    return Ok(sbox_ports(self.one_column, row_input, output));
                }
                outputs[lane] = output;
                next_sbox += 1;
            }
            state = to_state(external_layer(&outputs)?);
        }

        for constant in &constants.internal {
            let row_input = add_constant(state[0].clone(), self.one_column, Goldilocks::from_u64(*constant));
            let output = self.retained.form(
                logical_width,
                checked_add(slot_base, next_sbox, "Poseidon2 retained slot")?,
            )?;
            if next_sbox == target {
                return Ok(sbox_ports(self.one_column, row_input, output));
            }
            state[0] = output;
            state = internal_layer(&state, &constants.diag);
            next_sbox += 1;
        }

        for round in &constants.terminal {
            let mut outputs = empty_state();
            for lane in 0..WIDTH {
                let row_input = add_constant(state[lane].clone(), self.one_column, Goldilocks::from_u64(round[lane]));
                let output = self.retained.form(
                    logical_width,
                    checked_add(slot_base, next_sbox, "Poseidon2 retained slot")?,
                )?;
                if next_sbox == target {
                    return Ok(sbox_ports(self.one_column, row_input, output));
                }
                outputs[lane] = output;
                next_sbox += 1;
            }
            state = to_state(external_layer(&outputs)?);
        }

        Err(PackageError::Invalid("Poseidon2 S-box row ordinal"))
    }

    fn pin_row(&self, logical_width: usize, invocation: usize, lane: usize) -> Result<RowForms, PackageError> {
        if lane >= WIDTH {
            return Err(PackageError::Invalid("Poseidon2 pin row ordinal"));
        }
        let invocation_base = checked_mul(invocation, SBOX_ROWS_PER_INVOCATION, "Poseidon2 final retained slot")?;
        let final_base = checked_add(invocation_base, 78, "Poseidon2 final retained slot")?;
        let final_state = (0..WIDTH)
            .map(|selected| {
                self.retained.form(
                    logical_width,
                    checked_add(final_base, selected, "Poseidon2 final retained slot")?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = external_layer(&final_state)?[lane].clone();
        let difference = output.clone().append(output.scaled(-Goldilocks::ONE));
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Goldilocks::ONE);
        row[4] = difference;
        Ok(row)
    }
}

fn constants() -> &'static Poseidon2RoundConstants {
    static CONSTANTS: OnceLock<Poseidon2RoundConstants> = OnceLock::new();
    CONSTANTS.get_or_init(round_constants)
}

fn empty_state() -> [Form; WIDTH] {
    std::array::from_fn(|_| Form::default())
}

fn to_state(forms: Vec<Form>) -> [Form; WIDTH] {
    forms
        .try_into()
        .expect("external_layer always returns eight forms")
}

fn add_constant(form: Form, one_column: usize, coefficient: Goldilocks) -> Form {
    form.append(Form::singleton(one_column, coefficient))
}

fn internal_layer(state: &[Form; WIDTH], diagonal: &[u64; WIDTH]) -> [Form; WIDTH] {
    let sum = state.iter().cloned().fold(Form::default(), Form::append);
    std::array::from_fn(|lane| {
        state[lane]
            .scaled(Goldilocks::from_u64(diagonal[lane]))
            .append(sum.clone())
    })
}

fn sbox_ports(one_column: usize, input: Form, output: Form) -> RowForms {
    let mut row = empty_row();
    row[1] = Form::singleton(one_column, Goldilocks::ONE);
    row[4] = output;
    row[5] = input;
    row
}
