//! Exact Poseidon2 replay rows for one phased PiRLC family.
//!
//! Owns the two cursor-parity shapes, the fixed production source-column
//! placement, and both variable-state Poseidon2 slice replays. The input
//! replay reads the same 918 columns as the PiRLC algebra. The output replay
//! reads the same 54 algebra output columns.
//!
//! Does not own the PiRLC arithmetic, Ajtai residual update, challenge carry,
//! family cursor, selective lowering, or recursive lifecycle.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::{builder::Poseidon2PermutationAudit, R1csBuilder, TranscriptGadget, Var};

pub(crate) const SOURCE_COUNT: usize = 1 + crate::config::K_RHO as usize;
pub(crate) const LANE_COUNT: usize = 54;
pub(crate) const FAMILY_INPUT_FIELDS: usize = SOURCE_COUNT * LANE_COUNT;

pub(crate) const ALGEBRA_CHALLENGE_START: usize = 1;
pub(crate) const ALGEBRA_INPUT_START: usize = ALGEBRA_CHALLENGE_START + FAMILY_INPUT_FIELDS;
pub(crate) const ALGEBRA_OUTPUT_START: usize = ALGEBRA_INPUT_START + FAMILY_INPUT_FIELDS;
pub(crate) const ALGEBRA_PRODUCT_START: usize = ALGEBRA_OUTPUT_START + LANE_COUNT;
pub(crate) const ALGEBRA_PRODUCT_COLUMNS: usize = SOURCE_COUNT * LANE_COUNT * LANE_COUNT;
pub(crate) const ALGEBRA_COLUMNS: usize = ALGEBRA_PRODUCT_START + ALGEBRA_PRODUCT_COLUMNS;

pub(crate) const ZERO_DIGIT_START: usize = ALGEBRA_COLUMNS;
pub(crate) const DIGIT_COUNT: usize = 41;
pub(crate) const OPENING_COLUMNS: usize = 122;
pub(crate) const ACTIVE_DIGIT_START: usize = ZERO_DIGIT_START + DIGIT_COUNT;
pub(crate) const SHAPE_D_COLUMN: usize = ACTIVE_DIGIT_START + FAMILY_INPUT_FIELDS * OPENING_COLUMNS;
pub(crate) const SHAPE_KAPPA_COLUMN: usize = SHAPE_D_COLUMN + 1;
pub(crate) const COMMITMENT_OUTPUT_START: usize = SHAPE_KAPPA_COLUMN + 1;
pub(crate) const COMMITMENT_OUTPUT_FIELDS: usize = 108;
pub(crate) const BEFORE_RESIDUAL_START: usize = COMMITMENT_OUTPUT_START + COMMITMENT_OUTPUT_FIELDS;
pub(crate) const AFTER_RESIDUAL_START: usize = BEFORE_RESIDUAL_START + COMMITMENT_OUTPUT_FIELDS;
pub(crate) const BEFORE_CHALLENGE_START: usize = AFTER_RESIDUAL_START + COMMITMENT_OUTPUT_FIELDS;
pub(crate) const AFTER_CHALLENGE_START: usize = BEFORE_CHALLENGE_START + FAMILY_INPUT_FIELDS;
pub(crate) const BEFORE_CURSOR_COLUMN: usize = AFTER_CHALLENGE_START + FAMILY_INPUT_FIELDS;
pub(crate) const AFTER_CURSOR_COLUMN: usize = BEFORE_CURSOR_COLUMN + 1;
pub(crate) const SOURCE_COLUMNS: usize = AFTER_CURSOR_COLUMN + 1;

pub(crate) const INPUT_REPLAY_BEFORE_START: usize = SOURCE_COLUMNS;
pub(crate) const OUTPUT_REPLAY_BEFORE_START: usize = INPUT_REPLAY_BEFORE_START + 8;
pub(crate) const REPLAY_AUXILIARY_START: usize = OUTPUT_REPLAY_BEFORE_START + 8;

const POSEIDON2_ROWS_PER_CALL: usize = 600;
const POSEIDON2_RATE: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimePiRlcFamilyReplayArmKind {
    Even,
    Odd,
}

impl NebulaFPrimePiRlcFamilyReplayArmKind {
    pub const fn before_absorbed(self) -> usize {
        match self {
            Self::Even => 0,
            Self::Odd => 2,
        }
    }

    pub const fn after_absorbed(self) -> usize {
        match self {
            Self::Even => 2,
            Self::Odd => 0,
        }
    }

    pub const fn input_poseidon2_calls(self) -> usize {
        (self.before_absorbed() + FAMILY_INPUT_FIELDS) / 4
    }

    pub const fn output_poseidon2_calls(self) -> usize {
        (self.before_absorbed() + LANE_COUNT) / 4
    }

    pub const fn poseidon2_calls(self) -> usize {
        self.input_poseidon2_calls() + self.output_poseidon2_calls()
    }

    pub const fn rows(self) -> usize {
        self.poseidon2_calls() * POSEIDON2_ROWS_PER_CALL
    }

    pub const fn columns(self) -> usize {
        REPLAY_AUXILIARY_START + self.rows()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyReplayShapeAudit {
    pub kind: NebulaFPrimePiRlcFamilyReplayArmKind,
    pub rows: usize,
    pub columns: usize,
    pub source_columns: usize,
    pub input_poseidon2_calls: usize,
    pub output_poseidon2_calls: usize,
    pub before_absorbed: usize,
    pub after_absorbed: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimePiRlcFamilyReplayScope {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimePiRlcFamilyReplayCallClass {
    Direct,
    PartialStart,
    Chained,
}

/// Exact source-side semantic owner of one PiRLC replay permutation.
///
/// The record is created beside the transcript append that emits the rows.
/// Selective projection may rename its columns, but it may not infer its
/// scope, call class, state input, or absorbed source slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyReplayCallAudit {
    scope: NebulaFPrimePiRlcFamilyReplayScope,
    index: usize,
    class: NebulaFPrimePiRlcFamilyReplayCallClass,
    row_start: usize,
    row_end: usize,
    state_before_columns: [usize; 8],
    absorbed_columns: Vec<usize>,
    permutation_input_columns: [usize; 8],
    first_allocated_column: usize,
    allocated_column_count: usize,
    output_columns: [usize; 8],
}

impl NebulaFPrimePiRlcFamilyReplayCallAudit {
    pub const fn scope(&self) -> NebulaFPrimePiRlcFamilyReplayScope {
        self.scope
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn class(&self) -> NebulaFPrimePiRlcFamilyReplayCallClass {
        self.class
    }

    pub const fn row_start(&self) -> usize {
        self.row_start
    }

    pub const fn row_end(&self) -> usize {
        self.row_end
    }

    pub const fn state_before_columns(&self) -> [usize; 8] {
        self.state_before_columns
    }

    pub fn absorbed_columns(&self) -> &[usize] {
        &self.absorbed_columns
    }

    pub const fn permutation_input_columns(&self) -> [usize; 8] {
        self.permutation_input_columns
    }

    pub const fn first_allocated_column(&self) -> usize {
        self.first_allocated_column
    }

    pub const fn allocated_column_count(&self) -> usize {
        self.allocated_column_count
    }

    pub const fn output_columns(&self) -> [usize; 8] {
        self.output_columns
    }
}

pub struct NebulaFPrimePiRlcFamilyReplaySynthesis {
    kind: NebulaFPrimePiRlcFamilyReplayArmKind,
    builder: R1csBuilder,
    input_columns: Vec<usize>,
    output_columns: Vec<usize>,
    input_before_columns: [usize; 8],
    input_after_columns: [usize; 8],
    output_before_columns: [usize; 8],
    output_after_columns: [usize; 8],
    call_audits: Vec<NebulaFPrimePiRlcFamilyReplayCallAudit>,
}

pub(crate) struct AppendedPiRlcFamilyReplay {
    pub(crate) input_after_columns: [usize; 8],
    pub(crate) output_after_columns: [usize; 8],
    pub(crate) call_audits: Vec<NebulaFPrimePiRlcFamilyReplayCallAudit>,
}

pub(crate) fn append_pi_rlc_family_replay(
    builder: &mut R1csBuilder,
    kind: NebulaFPrimePiRlcFamilyReplayArmKind,
    input_columns: &[usize],
    output_columns: &[usize],
    input_before_columns: [usize; 8],
    output_before_columns: [usize; 8],
) -> AppendedPiRlcFamilyReplay {
    assert_eq!(input_columns.len(), FAMILY_INPUT_FIELDS);
    assert_eq!(output_columns.len(), LANE_COUNT);
    let row_start = builder.rows();
    let column_start = builder.cols();
    let audit_start = builder.poseidon2_permutation_audits().len();

    builder.begin_encoding_stage("nebula.streaming.pi_rlc.input_replay");
    let mut input_replay = TranscriptGadget::from_variable_state(
        input_before_columns.map(Var::from_column_for_trace),
        kind.before_absorbed(),
    );
    let input_variables = input_columns
        .iter()
        .copied()
        .map(Var::from_column_for_trace)
        .collect::<Vec<_>>();
    input_replay.append_fields_unframed_vars(builder, &input_variables);
    let input_after_columns = input_replay.variable_state().map(Var::col);
    let input_audit_end = builder.poseidon2_permutation_audits().len();
    assert_eq!(input_replay.absorbed(), kind.after_absorbed());
    assert_eq!(
        builder.poseidon2_permutation_audits().len() - audit_start,
        kind.input_poseidon2_calls(),
    );

    builder.begin_encoding_stage("nebula.streaming.pi_rlc.output_replay");
    let mut output_replay = TranscriptGadget::from_variable_state(
        output_before_columns.map(Var::from_column_for_trace),
        kind.before_absorbed(),
    );
    let output_variables = output_columns
        .iter()
        .copied()
        .map(Var::from_column_for_trace)
        .collect::<Vec<_>>();
    output_replay.append_fields_unframed_vars(builder, &output_variables);
    let output_after_columns = output_replay.variable_state().map(Var::col);
    assert_eq!(output_replay.absorbed(), kind.after_absorbed());

    assert_eq!(
        builder.poseidon2_permutation_audits().len() - audit_start,
        kind.poseidon2_calls(),
    );
    assert_eq!(builder.rows() - row_start, kind.rows());
    assert_eq!(builder.cols() - column_start, kind.rows());
    let permutation_audits = builder.poseidon2_permutation_audits();
    let mut call_audits = replay_call_audits(
        kind,
        NebulaFPrimePiRlcFamilyReplayScope::Input,
        input_columns,
        input_before_columns,
        &permutation_audits[audit_start..input_audit_end],
    );
    call_audits.extend(replay_call_audits(
        kind,
        NebulaFPrimePiRlcFamilyReplayScope::Output,
        output_columns,
        output_before_columns,
        &permutation_audits[input_audit_end..],
    ));
    AppendedPiRlcFamilyReplay {
        input_after_columns,
        output_after_columns,
        call_audits,
    }
}

fn replay_call_audits(
    kind: NebulaFPrimePiRlcFamilyReplayArmKind,
    scope: NebulaFPrimePiRlcFamilyReplayScope,
    source_columns: &[usize],
    state_before_columns: [usize; 8],
    permutation_audits: &[Poseidon2PermutationAudit],
) -> Vec<NebulaFPrimePiRlcFamilyReplayCallAudit> {
    let expected_calls = match scope {
        NebulaFPrimePiRlcFamilyReplayScope::Input => kind.input_poseidon2_calls(),
        NebulaFPrimePiRlcFamilyReplayScope::Output => kind.output_poseidon2_calls(),
    };
    assert_eq!(permutation_audits.len(), expected_calls);

    let mut calls = Vec::with_capacity(expected_calls);
    let mut state_before = state_before_columns;
    let mut absorbed = kind.before_absorbed();
    let mut source_cursor = 0;
    for (index, permutation) in permutation_audits.iter().enumerate() {
        let absorbed_count = POSEIDON2_RATE - absorbed;
        let absorbed_columns = source_columns[source_cursor..source_cursor + absorbed_count].to_vec();
        let mut permutation_input = state_before;
        permutation_input[absorbed..POSEIDON2_RATE].copy_from_slice(&absorbed_columns);
        assert_eq!(permutation.input_cols, permutation_input);
        assert_eq!(permutation.row_end - permutation.row_start, POSEIDON2_ROWS_PER_CALL);
        assert_eq!(permutation.allocated_col_count, POSEIDON2_ROWS_PER_CALL);

        let class = if index != 0 {
            NebulaFPrimePiRlcFamilyReplayCallClass::Chained
        } else if absorbed == 0 {
            NebulaFPrimePiRlcFamilyReplayCallClass::Direct
        } else {
            NebulaFPrimePiRlcFamilyReplayCallClass::PartialStart
        };
        calls.push(NebulaFPrimePiRlcFamilyReplayCallAudit {
            scope,
            index,
            class,
            row_start: permutation.row_start,
            row_end: permutation.row_end,
            state_before_columns: state_before,
            absorbed_columns,
            permutation_input_columns: permutation.input_cols,
            first_allocated_column: permutation.first_allocated_col,
            allocated_column_count: permutation.allocated_col_count,
            output_columns: permutation.output_cols,
        });

        source_cursor += absorbed_count;
        absorbed = 0;
        state_before = permutation.output_cols;
    }
    assert_eq!(source_columns.len() - source_cursor, kind.after_absorbed());
    calls
}

impl NebulaFPrimePiRlcFamilyReplaySynthesis {
    pub fn production(kind: NebulaFPrimePiRlcFamilyReplayArmKind) -> Self {
        let mut builder = R1csBuilder::new();
        let mut source_values = vec![F::ZERO; REPLAY_AUXILIARY_START - 1];

        for source in 0..SOURCE_COUNT {
            for lane in 0..LANE_COUNT {
                let offset = source * LANE_COUNT + lane;
                set_column(
                    &mut source_values,
                    ALGEBRA_INPUT_START + offset,
                    F::from_usize(1 + source * 131 + lane * 17),
                );
            }
        }
        for lane in 0..LANE_COUNT {
            set_column(
                &mut source_values,
                ALGEBRA_OUTPUT_START + lane,
                F::from_usize(10_000 + lane * 19),
            );
        }
        for lane in 0..8 {
            set_column(
                &mut source_values,
                INPUT_REPLAY_BEFORE_START + lane,
                F::from_usize(20_000 + lane * 23),
            );
            set_column(
                &mut source_values,
                OUTPUT_REPLAY_BEFORE_START + lane,
                F::from_usize(30_000 + lane * 29),
            );
        }
        builder.alloc_vec(&source_values);
        assert_eq!(builder.cols(), REPLAY_AUXILIARY_START);
        assert_eq!(builder.rows(), 0);

        let input_columns = (0..FAMILY_INPUT_FIELDS)
            .map(|offset| ALGEBRA_INPUT_START + offset)
            .collect::<Vec<_>>();
        let output_columns = (0..LANE_COUNT)
            .map(|lane| ALGEBRA_OUTPUT_START + lane)
            .collect::<Vec<_>>();
        let input_before_columns = std::array::from_fn(|lane| INPUT_REPLAY_BEFORE_START + lane);
        let output_before_columns = std::array::from_fn(|lane| OUTPUT_REPLAY_BEFORE_START + lane);

        let replay = append_pi_rlc_family_replay(
            &mut builder,
            kind,
            &input_columns,
            &output_columns,
            input_before_columns,
            output_before_columns,
        );
        let input_after_columns = replay.input_after_columns;
        let output_after_columns = replay.output_after_columns;
        let call_audits = replay.call_audits;

        assert_eq!(builder.poseidon2_permutation_audits().len(), kind.poseidon2_calls());
        assert_eq!(builder.rows(), kind.rows());
        assert_eq!(builder.cols(), kind.columns());
        assert_eq!(builder.first_unsatisfied_row(), None);

        Self {
            kind,
            builder,
            input_columns,
            output_columns,
            input_before_columns,
            input_after_columns,
            output_before_columns,
            output_after_columns,
            call_audits,
        }
    }

    pub const fn kind(&self) -> NebulaFPrimePiRlcFamilyReplayArmKind {
        self.kind
    }

    pub fn builder(&self) -> &R1csBuilder {
        &self.builder
    }

    pub fn input_columns(&self) -> &[usize] {
        &self.input_columns
    }

    pub fn output_columns(&self) -> &[usize] {
        &self.output_columns
    }

    pub const fn input_before_columns(&self) -> [usize; 8] {
        self.input_before_columns
    }

    pub const fn input_after_columns(&self) -> [usize; 8] {
        self.input_after_columns
    }

    pub const fn output_before_columns(&self) -> [usize; 8] {
        self.output_before_columns
    }

    pub const fn output_after_columns(&self) -> [usize; 8] {
        self.output_after_columns
    }

    pub fn call_audits(&self) -> &[NebulaFPrimePiRlcFamilyReplayCallAudit] {
        &self.call_audits
    }

    pub const fn shape_audit(&self) -> NebulaFPrimePiRlcFamilyReplayShapeAudit {
        NebulaFPrimePiRlcFamilyReplayShapeAudit {
            kind: self.kind,
            rows: self.kind.rows(),
            columns: self.kind.columns(),
            source_columns: SOURCE_COLUMNS,
            input_poseidon2_calls: self.kind.input_poseidon2_calls(),
            output_poseidon2_calls: self.kind.output_poseidon2_calls(),
            before_absorbed: self.kind.before_absorbed(),
            after_absorbed: self.kind.after_absorbed(),
        }
    }
}

fn set_column(values_without_one: &mut [F], column: usize, value: F) {
    assert!(column > 0);
    values_without_one[column - 1] = value;
}
