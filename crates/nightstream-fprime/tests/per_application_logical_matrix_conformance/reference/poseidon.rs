//! Independent Poseidon2 matrix-block interpreter.

use serde_json::Value;

use super::poseidon_input::Program as InputProgram;
use super::{
    checked_add, checked_mul, empty_row, exact_array, external_layer, word, Field, Form, Result, RetainedBlock,
    RetainedKind, RowForms,
};

const ROWS_PER_INVOCATION: usize = 94;
const SBOX_ROWS_PER_INVOCATION: usize = 86;
const WIDTH: usize = 8;

const INITIAL: [[u64; WIDTH]; 4] = [
    [
        15504881536434223753,
        2212164856944708396,
        1885257220781225929,
        17531637481572944510,
        16769640728293682348,
        445908668462176974,
        1308472042479836079,
        17465001500823438575,
    ],
    [
        1922033642430128704,
        2657514617275794404,
        17238706657248448792,
        7348277157222259646,
        10777112892842897939,
        1771261721914735482,
        9409693344407549465,
        16619731096074499912,
    ],
    [
        1922036059108268922,
        2681686362645798986,
        12432722052283819565,
        2826979200512189741,
        5080805286413226676,
        16827966425431695029,
        9196241087337510154,
        2350771591198563053,
    ],
    [
        2989012136977041732,
        4359939046747977080,
        16089932437481530267,
        6601984573273403484,
        13005272261058756234,
        17128237926164276121,
        8240789415616872849,
        8676316357341090631,
    ],
];

const INTERNAL: [u64; 22] = [
    7482194551502142718,
    3471957803411196592,
    8846669050136897522,
    4431017908497072775,
    14382646627736292998,
    15636596632746594248,
    14521990061611210983,
    4351091752509404379,
    14119848206371842921,
    528205008764728916,
    15379406877060454284,
    13572057177474709483,
    780214424511389757,
    10591233664360718633,
    1849508423779478786,
    7345390174439848870,
    14580881241235634775,
    8777273265976228774,
    1758781345554053863,
    9701442189086298420,
    15685565327448534444,
    5672331717709479627,
];

const TERMINAL: [[u64; WIDTH]; 4] = [
    [
        16452552554259143025,
        17874550554210084887,
        3031715677034868367,
        18215520516675091549,
        18186005068527139405,
        11138995707668647102,
        15098195648006184282,
        2025927025270509469,
    ],
    [
        9957669227203243937,
        11554336633716867616,
        9729067570563846225,
        4239770196713589268,
        4390607796152185292,
        17647511975646925721,
        7671337049037340193,
        4209452938403606590,
    ],
    [
        6593973666654839090,
        8390781086037206386,
        7324343054784993307,
        17780748563735894140,
        15974082699116886783,
        13213371256836887512,
        7312926934405385057,
        10393853239698468203,
    ],
    [
        2710107888698774842,
        2801523468128575786,
        15894340394120906162,
        13510783799941644149,
        7917164295139071913,
        13839801071899888959,
        6672989303670154677,
        4519956214037211385,
    ],
];

const DIAGONAL: [u64; WIDTH] = [
    0xffff_fffe_ffff_ffff,
    1,
    2,
    0x7fff_ffff_8000_0001,
    3,
    0x7fff_ffff_8000_0000,
    0xffff_fffe_ffff_fffe,
    0xffff_fffe_ffff_fffd,
];

#[derive(Clone, Debug)]
pub struct Block {
    invocation_count: usize,
    one_column: usize,
    retained: RetainedBlock,
    input: InputProgram,
}

impl Block {
    pub fn decode(value: &Value, logical_width: usize) -> Result<Self> {
        let fields = exact_array(value, 4, "Poseidon2 matrix block")?;
        let block = Self {
            invocation_count: word(&fields[0], "Poseidon2 invocation count")?,
            one_column: word(&fields[1], "Poseidon2 one column")?,
            retained: RetainedBlock::decode(&fields[2])?,
            input: InputProgram::decode(&fields[3], logical_width)?,
        };
        if block.one_column != 0 || block.one_column >= logical_width {
            return Err("Poseidon2 one column is not logical column zero".into());
        }
        block.retained.validate(logical_width)?;
        Ok(block)
    }

    pub fn row_count(&self) -> Result<usize> {
        checked_mul(self.invocation_count, ROWS_PER_INVOCATION, "Poseidon2 rows")
    }

    pub fn row(&self, logical_width: usize, ordinal: usize) -> Result<RowForms> {
        if ordinal >= self.row_count()? || self.one_column >= logical_width {
            return Err("Poseidon2 matrix row is out of range".into());
        }
        if self.retained.kind != RetainedKind::Field
            || self.retained.slot_count
                != checked_mul(
                    self.invocation_count,
                    SBOX_ROWS_PER_INVOCATION,
                    "Poseidon2 retained slots",
                )?
            || !self.retained.fits(logical_width)?
        {
            return Err("invalid Poseidon2 retained geometry".into());
        }
        let invocation = ordinal / ROWS_PER_INVOCATION;
        let local_row = ordinal % ROWS_PER_INVOCATION;
        if local_row < SBOX_ROWS_PER_INVOCATION {
            self.sbox_row(logical_width, invocation, local_row)
        } else {
            self.pin_row(logical_width, invocation, local_row - SBOX_ROWS_PER_INVOCATION)
        }
    }

    pub fn visit_rows(
        &self,
        logical_width: usize,
        start: usize,
        end: usize,
        mut visit: impl FnMut(usize, RowForms) -> Result<()>,
    ) -> Result<()> {
        if start > end || end > self.row_count()? {
            return Err("Poseidon2 matrix row range is out of bounds".into());
        }
        if start == end {
            return Ok(());
        }
        if self.one_column >= logical_width
            || self.retained.kind != RetainedKind::Field
            || self.retained.slot_count
                != checked_mul(
                    self.invocation_count,
                    SBOX_ROWS_PER_INVOCATION,
                    "Poseidon2 retained slots",
                )?
            || !self.retained.fits(logical_width)?
        {
            return Err("invalid Poseidon2 retained geometry".into());
        }

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
            self.visit_invocation_rows(
                logical_width,
                invocation,
                local_start,
                local_end,
                &mut |local_row, row| visit(invocation_start + local_row, row),
            )?;
        }
        Ok(())
    }

    fn visit_invocation_rows(
        &self,
        logical_width: usize,
        invocation: usize,
        local_start: usize,
        local_end: usize,
        visit: &mut impl FnMut(usize, RowForms) -> Result<()>,
    ) -> Result<()> {
        let input = self
            .input
            .state(logical_width, self.one_column, invocation)?;
        let mut state = to_state(external_layer(&input)?);
        let slot_base = checked_mul(invocation, SBOX_ROWS_PER_INVOCATION, "Poseidon2 retained slot")?;
        let mut next = 0usize;

        for round in INITIAL {
            let mut outputs = empty_state();
            for lane in 0..WIDTH {
                let output = self
                    .retained
                    .form(logical_width, checked_add(slot_base, next, "Poseidon2 retained slot")?)?;
                if (local_start..local_end).contains(&next) {
                    let row_input = add_constant(state[lane].clone(), self.one_column, round[lane])?;
                    visit(next, sbox_ports(self.one_column, row_input, output.clone()))?;
                }
                outputs[lane] = output;
                next += 1;
            }
            state = to_state(external_layer(&outputs)?);
        }

        for constant in INTERNAL {
            let output = self
                .retained
                .form(logical_width, checked_add(slot_base, next, "Poseidon2 retained slot")?)?;
            if (local_start..local_end).contains(&next) {
                let row_input = add_constant(state[0].clone(), self.one_column, constant)?;
                visit(next, sbox_ports(self.one_column, row_input, output.clone()))?;
            }
            state[0] = output;
            state = internal_layer(&state)?;
            next += 1;
        }

        for round in TERMINAL {
            let mut outputs = empty_state();
            for lane in 0..WIDTH {
                let output = self
                    .retained
                    .form(logical_width, checked_add(slot_base, next, "Poseidon2 retained slot")?)?;
                if (local_start..local_end).contains(&next) {
                    let row_input = add_constant(state[lane].clone(), self.one_column, round[lane])?;
                    visit(next, sbox_ports(self.one_column, row_input, output.clone()))?;
                }
                outputs[lane] = output;
                next += 1;
            }
            state = to_state(external_layer(&outputs)?);
        }

        for lane in 0..WIDTH {
            let local_row = SBOX_ROWS_PER_INVOCATION + lane;
            if (local_start..local_end).contains(&local_row) {
                visit(local_row, pin_ports(self.one_column, state[lane].clone()))?;
            }
        }
        Ok(())
    }

    fn sbox_row(&self, logical_width: usize, invocation: usize, target: usize) -> Result<RowForms> {
        let input = self
            .input
            .state(logical_width, self.one_column, invocation)?;
        let mut state = to_state(external_layer(&input)?);
        let slot_base = checked_mul(invocation, SBOX_ROWS_PER_INVOCATION, "Poseidon2 retained slot")?;
        let mut next = 0usize;

        for round in INITIAL {
            let mut outputs = empty_state();
            for lane in 0..WIDTH {
                let row_input = add_constant(state[lane].clone(), self.one_column, round[lane])?;
                let output = self
                    .retained
                    .form(logical_width, checked_add(slot_base, next, "Poseidon2 retained slot")?)?;
                if next == target {
                    return Ok(sbox_ports(self.one_column, row_input, output));
                }
                outputs[lane] = output;
                next += 1;
            }
            state = to_state(external_layer(&outputs)?);
        }

        for constant in INTERNAL {
            let row_input = add_constant(state[0].clone(), self.one_column, constant)?;
            let output = self
                .retained
                .form(logical_width, checked_add(slot_base, next, "Poseidon2 retained slot")?)?;
            if next == target {
                return Ok(sbox_ports(self.one_column, row_input, output));
            }
            state[0] = output;
            state = internal_layer(&state)?;
            next += 1;
        }

        for round in TERMINAL {
            let mut outputs = empty_state();
            for lane in 0..WIDTH {
                let row_input = add_constant(state[lane].clone(), self.one_column, round[lane])?;
                let output = self
                    .retained
                    .form(logical_width, checked_add(slot_base, next, "Poseidon2 retained slot")?)?;
                if next == target {
                    return Ok(sbox_ports(self.one_column, row_input, output));
                }
                outputs[lane] = output;
                next += 1;
            }
            state = to_state(external_layer(&outputs)?);
        }
        Err("Poseidon2 S-box row is out of range".into())
    }

    fn pin_row(&self, logical_width: usize, invocation: usize, lane: usize) -> Result<RowForms> {
        if lane >= WIDTH {
            return Err("Poseidon2 pin lane is out of range".into());
        }
        let final_base = checked_add(
            checked_mul(invocation, SBOX_ROWS_PER_INVOCATION, "Poseidon2 final state")?,
            78,
            "Poseidon2 final state",
        )?;
        let final_state = (0..WIDTH)
            .map(|selected| {
                self.retained.form(
                    logical_width,
                    checked_add(final_base, selected, "Poseidon2 final state")?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let output = external_layer(&final_state)?[lane].clone();
        let difference = output.clone().append(output.scaled(-Field::ONE));
        let mut row = empty_row();
        row[1] = Form::singleton(self.one_column, Field::ONE);
        row[4] = difference;
        Ok(row)
    }
}

fn empty_state() -> [Form; WIDTH] {
    std::array::from_fn(|_| Form::default())
}

fn to_state(forms: Vec<Form>) -> [Form; WIDTH] {
    forms
        .try_into()
        .expect("external layer returns eight forms")
}

fn add_constant(form: Form, one_column: usize, raw: u64) -> Result<Form> {
    Ok(form.append(Form::singleton(
        one_column,
        Field::checked(raw, "Poseidon2 round constant")?,
    )))
}

fn internal_layer(state: &[Form; WIDTH]) -> Result<[Form; WIDTH]> {
    let sum = state.iter().cloned().fold(Form::default(), Form::append);
    let forms = (0..WIDTH)
        .map(|lane| {
            Ok(state[lane]
                .clone()
                .scaled(Field::checked(DIAGONAL[lane], "Poseidon2 diagonal")?)
                .append(sum.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(forms
        .try_into()
        .expect("internal layer returns eight forms"))
}

fn sbox_ports(one_column: usize, input: Form, output: Form) -> RowForms {
    let mut row = empty_row();
    row[1] = Form::singleton(one_column, Field::ONE);
    row[4] = output;
    row[5] = input;
    row
}

fn pin_ports(one_column: usize, output: Form) -> RowForms {
    let difference = output.clone().append(output.scaled(-Field::ONE));
    let mut row = empty_row();
    row[1] = Form::singleton(one_column, Field::ONE);
    row[4] = difference;
    row
}
