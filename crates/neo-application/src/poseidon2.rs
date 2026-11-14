//! Native and R1CS forms of the width-12 Poseidon2 permutation of
//! `decisions/poseidon2-p64-instance.md`. Event compression and
//! application-specific round scheduling belong to their respective consumers.

use std::collections::BTreeMap;
use std::{ops::Range, sync::OnceLock};

use neo_ccs::crypto::poseidon2::{HALF_FULL_ROUNDS, INSTANCE_12, PARTIAL_ROUNDS};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use p3_poseidon2::{matmul_internal, mds_light_permutation, MDSMat4};

use crate::gadgets::push_pow3_expression;
use crate::{GadgetDescriptor, TaggedR1csBuilder};

pub const POSEIDON2_WIDTH: usize = 12;
pub const POSEIDON2_HALF_FULL_ROUNDS: usize = HALF_FULL_ROUNDS;
pub const POSEIDON2_FULL_ROUNDS: usize = 2 * POSEIDON2_HALF_FULL_ROUNDS;
pub const POSEIDON2_PARTIAL_ROUNDS: usize = PARTIAL_ROUNDS;
/// The partial rounds that one output row set covers: the permutation
/// groups the 43 rounds into 21 pairs and one last single round.
pub const POSEIDON2_PARTIAL_GROUP: usize = 2;
pub const POSEIDON2_PARTIAL_GROUPS: usize = POSEIDON2_PARTIAL_ROUNDS.div_ceil(POSEIDON2_PARTIAL_GROUP);
pub const POSEIDON2_GROUPED_ROUNDS: usize = POSEIDON2_FULL_ROUNDS + POSEIDON2_PARTIAL_GROUPS;

/// Apply the width-12 Poseidon2 permutation.
pub fn permute(state: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
    INSTANCE_12.permute(state)
}

/// The Poseidon2 S-box `x -> x^3`.
pub fn sbox3(x: F) -> F {
    x * x * x
}

/// Round constants for full round `0..8` in permutation order.
pub fn full_round_constants(round: usize) -> &'static [F; POSEIDON2_WIDTH] {
    assert!(round < POSEIDON2_FULL_ROUNDS, "full-round index out of range");
    if round < POSEIDON2_HALF_FULL_ROUNDS {
        &INSTANCE_12.initial[round]
    } else {
        &INSTANCE_12.terminal[round - POSEIDON2_HALF_FULL_ROUNDS]
    }
}

/// The constant of partial round `0..43`.
pub fn partial_round_constant(round: usize) -> F {
    assert!(round < POSEIDON2_PARTIAL_ROUNDS, "partial-round index out of range");
    INSTANCE_12.internal[round]
}

/// The first partial round and the round count of partial group `0..22`.
pub fn partial_group(group: usize) -> Range<usize> {
    assert!(group < POSEIDON2_PARTIAL_GROUPS, "partial-group index out of range");
    let first = group * POSEIDON2_PARTIAL_GROUP;
    first..(first + POSEIDON2_PARTIAL_GROUP).min(POSEIDON2_PARTIAL_ROUNDS)
}

pub fn external_linear(state: &mut [F; POSEIDON2_WIDTH]) {
    mds_light_permutation(state, &MDSMat4);
}

pub fn internal_linear(state: &mut [F; POSEIDON2_WIDTH]) {
    matmul_internal(state, INSTANCE_12.diagonal);
}

pub fn external_matrix() -> &'static [[F; POSEIDON2_WIDTH]; POSEIDON2_WIDTH] {
    static MATRIX: OnceLock<[[F; POSEIDON2_WIDTH]; POSEIDON2_WIDTH]> = OnceLock::new();
    MATRIX.get_or_init(|| matrix_of(external_linear))
}

pub fn internal_matrix() -> &'static [[F; POSEIDON2_WIDTH]; POSEIDON2_WIDTH] {
    static MATRIX: OnceLock<[[F; POSEIDON2_WIDTH]; POSEIDON2_WIDTH]> = OnceLock::new();
    MATRIX.get_or_init(|| matrix_of(internal_linear))
}

fn matrix_of(apply: fn(&mut [F; POSEIDON2_WIDTH])) -> [[F; POSEIDON2_WIDTH]; POSEIDON2_WIDTH] {
    let mut matrix = [[F::ZERO; POSEIDON2_WIDTH]; POSEIDON2_WIDTH];
    for column in 0..POSEIDON2_WIDTH {
        let mut basis = [F::ZERO; POSEIDON2_WIDTH];
        basis[column] = F::ONE;
        apply(&mut basis);
        for (row, value) in basis.into_iter().enumerate() {
            matrix[row][column] = value;
        }
    }
    matrix
}

/// Apply the initial external layer prescribed before round constants.
pub fn apply_initial_linear(state: &mut [F; POSEIDON2_WIDTH]) {
    external_linear(state);
}

/// Apply one full round to a state that has already received the initial layer.
pub fn apply_full_round(round: usize, state: &mut [F; POSEIDON2_WIDTH]) {
    for (lane, &constant) in state.iter_mut().zip(full_round_constants(round)) {
        *lane = sbox3(*lane + constant);
    }
    external_linear(state);
}

/// Apply partial round `round`.
pub fn apply_partial_round(round: usize, state: &mut [F; POSEIDON2_WIDTH]) {
    state[0] = sbox3(state[0] + partial_round_constant(round));
    internal_linear(state);
}

const FULL_ROUND_POWER_COLUMNS: usize = 2 * POSEIDON2_WIDTH;
const INTERMEDIATE_STATES: usize = POSEIDON2_GROUPED_ROUNDS - 1;

/// Auxiliary columns used by one fully unrolled width-12 permutation.
pub const POSEIDON2_PERMUTATION_AUX_COLUMNS: usize = POSEIDON2_WIDTH
    + POSEIDON2_FULL_ROUNDS * FULL_ROUND_POWER_COLUMNS
    + 2 * POSEIDON2_PARTIAL_ROUNDS
    + INTERMEDIATE_STATES * POSEIDON2_WIDTH;

fn columns<const N: usize>(start: usize) -> [usize; N] {
    core::array::from_fn(|offset| start + offset)
}

fn assign_powers(value: F, powers: [usize; 2], assignment: &mut [F]) {
    let [x2, x3] = powers;
    assignment[x2] = value * value;
    assignment[x3] = assignment[x2] * value;
}

fn push_at_most_one_selectors<Owner: Clone, const N: usize>(
    builder: &mut TaggedR1csBuilder<'_, Owner>,
    selectors: [usize; N],
) {
    for selector in selectors {
        builder.push_boolean(selector);
    }
    let one = builder.const_one_column();
    builder.push_row(
        selectors.map(|selector| (selector, F::ONE)),
        selectors
            .map(|selector| (selector, F::ONE))
            .into_iter()
            .chain([(one, -F::ONE)]),
        [],
    );
}

fn assign_full_round_auxiliaries(
    constants: [F; POSEIDON2_WIDTH],
    state_before: [usize; POSEIDON2_WIDTH],
    powers: [[usize; 2]; POSEIDON2_WIDTH],
    assignment: &mut [F],
) {
    for lane in 0..POSEIDON2_WIDTH {
        assign_powers(
            assignment[state_before[lane]] + constants[lane],
            powers[lane],
            assignment,
        );
    }
}

/// One selector and round index in a selectable full round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2FullRoundChoice {
    pub selector: usize,
    pub round: usize,
}

impl Poseidon2FullRoundChoice {
    pub fn for_round(selector: usize, round: usize) -> Self {
        assert!(round < POSEIDON2_FULL_ROUNDS, "full-round index out of range");
        Self { selector, round }
    }
}

/// A width-12 full round whose round constants are selected by application
/// columns. When every selector is zero, the output is left unconstrained but
/// the power columns retain a canonical assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2FullRound12<const N: usize = 1> {
    pub choices: [Poseidon2FullRoundChoice; N],
    pub state_before: [usize; POSEIDON2_WIDTH],
    pub state_after: [usize; POSEIDON2_WIDTH],
    pub powers: [[usize; 2]; POSEIDON2_WIDTH],
}

impl<const N: usize> Poseidon2FullRound12<N> {
    /// Emit a self-contained selectable round, including Booleanity and
    /// at-most-one enforcement for its selectors.
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        push_at_most_one_selectors(builder, self.choices.map(|choice| choice.selector));
        self.emit_constraints(builder);
        builder.record_gadget(self.descriptor(), first_row);
    }

    /// Emit only the round equations when the surrounding application already
    /// constrains every selector to be Boolean and at most one to be active.
    /// Without that guarantee, blended round constants can remain satisfiable.
    pub fn push_constraints_assuming_preconstrained_selectors<Owner: Clone>(
        &self,
        builder: &mut TaggedR1csBuilder<'_, Owner>,
    ) {
        let first_row = builder.next_row_index();
        self.emit_constraints(builder);
        builder.record_gadget(self.descriptor(), first_row);
    }

    /// Fill only the S-box power columns. `state_after` remains application
    /// owned and is checked by the emitted output constraints.
    pub fn assign_auxiliaries(&self, assignment: &mut [F]) {
        let constants = core::array::from_fn(|lane| {
            self.choices.iter().fold(F::ZERO, |constant, choice| {
                constant + assignment[choice.selector] * full_round_constants(choice.round)[lane]
            })
        });
        assign_full_round_auxiliaries(constants, self.state_before, self.powers, assignment);
    }

    fn emit_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        for lane in 0..POSEIDON2_WIDTH {
            let expression: Vec<_> = core::iter::once((self.state_before[lane], F::ONE))
                .chain(
                    self.choices
                        .iter()
                        .map(|choice| (choice.selector, full_round_constants(choice.round)[lane])),
                )
                .collect();
            push_pow3_expression(builder, &expression, self.powers[lane]);
        }

        let gate: Vec<_> = self
            .choices
            .iter()
            .map(|choice| (choice.selector, F::ONE))
            .collect();
        let matrix = external_matrix();
        for lane in 0..POSEIDON2_WIDTH {
            let terms = core::iter::once((self.state_after[lane], F::ONE)).chain(
                matrix[lane]
                    .iter()
                    .enumerate()
                    .map(|(input, &coefficient)| (self.powers[input][1], -coefficient)),
            );
            builder.push_row(gate.iter().copied(), terms, []);
        }
    }

    fn descriptor(&self) -> GadgetDescriptor {
        GadgetDescriptor::Poseidon2FullRound12 {
            choices: self
                .choices
                .iter()
                .map(|choice| (choice.selector, choice.round))
                .collect(),
            state_before: self.state_before,
            state_after: self.state_after,
            powers: self.powers.to_vec(),
        }
    }
}

fn assign_fixed_full_round(
    round: usize,
    state_before: [usize; POSEIDON2_WIDTH],
    powers: [[usize; 2]; POSEIDON2_WIDTH],
    assignment: &mut [F],
) -> [F; POSEIDON2_WIDTH] {
    assign_full_round_auxiliaries(*full_round_constants(round), state_before, powers, assignment);
    let mut state = core::array::from_fn(|lane| assignment[powers[lane][1]]);
    external_linear(&mut state);
    state
}

/// One selector and the first round of a selectable group of partial rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2PartialRoundsChoice {
    pub selector: usize,
    pub first: usize,
}

impl Poseidon2PartialRoundsChoice {
    pub fn for_first_round(selector: usize, first: usize) -> Self {
        assert!(first < POSEIDON2_PARTIAL_ROUNDS, "partial-round index out of range");
        Self { selector, first }
    }
}

/// Consecutive width-12 partial rounds, one `x^3` power pair per round, that
/// share one set of output rows. Each S-box input and each output lane is a
/// linear form in `state_before`, the earlier S-box outputs and the selected
/// round constants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poseidon2PartialRounds12<const N: usize = 1> {
    pub choices: [Poseidon2PartialRoundsChoice; N],
    pub state_before: [usize; POSEIDON2_WIDTH],
    pub state_after: [usize; POSEIDON2_WIDTH],
    pub powers: Vec<[usize; 2]>,
}

/// A linear form over columns, with merged terms in column order.
type Form = BTreeMap<usize, F>;

fn add_scaled(sum: &mut Form, form: &Form, scale: F) {
    for (&column, &coefficient) in form {
        *sum.entry(column).or_insert(F::ZERO) += scale * coefficient;
    }
}

impl<const N: usize> Poseidon2PartialRounds12<N> {
    /// Emit self-contained selectable rounds, including Booleanity and
    /// at-most-one enforcement for its selectors.
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        push_at_most_one_selectors(builder, self.choices.map(|choice| choice.selector));
        self.emit_constraints(builder);
        builder.record_gadget(self.descriptor(), first_row);
    }

    /// Emit only the round equations when the surrounding application already
    /// constrains every selector to be Boolean and at most one to be active.
    /// Without that guarantee, blended round constants can remain satisfiable.
    pub fn push_constraints_assuming_preconstrained_selectors<Owner: Clone>(
        &self,
        builder: &mut TaggedR1csBuilder<'_, Owner>,
    ) {
        let first_row = builder.next_row_index();
        self.emit_constraints(builder);
        builder.record_gadget(self.descriptor(), first_row);
    }

    /// Fill only the S-box power columns. `state_after` remains application
    /// owned and is checked by the emitted output constraints.
    pub fn assign_auxiliaries(&self, assignment: &mut [F]) {
        let constants = self.selected_constants(assignment);
        let _ = self.assign_rounds(&constants, assignment);
    }

    fn selected_constants(&self, assignment: &[F]) -> Vec<F> {
        (0..self.powers.len())
            .map(|offset| {
                self.choices.iter().fold(F::ZERO, |constant, choice| {
                    constant + assignment[choice.selector] * partial_round_constant(choice.first + offset)
                })
            })
            .collect()
    }

    fn assign_rounds(&self, constants: &[F], assignment: &mut [F]) -> [F; POSEIDON2_WIDTH] {
        let mut state = self.state_before.map(|column| assignment[column]);
        for (&powers, &constant) in self.powers.iter().zip(constants) {
            assign_powers(state[0] + constant, powers, assignment);
            state[0] = assignment[powers[1]];
            internal_linear(&mut state);
        }
        state
    }

    fn emit_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let matrix = internal_matrix();
        let mut lanes: [Form; POSEIDON2_WIDTH] = self
            .state_before
            .map(|column| Form::from([(column, F::ONE)]));
        for (offset, &powers) in self.powers.iter().enumerate() {
            let mut input = lanes[0].clone();
            for choice in &self.choices {
                add_scaled(
                    &mut input,
                    &Form::from([(choice.selector, F::ONE)]),
                    partial_round_constant(choice.first + offset),
                );
            }
            let input: Vec<_> = input.into_iter().collect();
            push_pow3_expression(builder, &input, powers);
            lanes[0] = Form::from([(powers[1], F::ONE)]);
            lanes = core::array::from_fn(|lane| {
                let mut sum = Form::new();
                for (source, &coefficient) in matrix[lane].iter().enumerate() {
                    add_scaled(&mut sum, &lanes[source], coefficient);
                }
                sum
            });
        }

        let gate: Vec<_> = self
            .choices
            .iter()
            .map(|choice| (choice.selector, F::ONE))
            .collect();
        for (lane, form) in lanes.iter().enumerate() {
            let terms = core::iter::once((self.state_after[lane], F::ONE)).chain(
                form.iter()
                    .map(|(&column, &coefficient)| (column, -coefficient)),
            );
            builder.push_row(gate.iter().copied(), terms, []);
        }
    }

    fn descriptor(&self) -> GadgetDescriptor {
        GadgetDescriptor::Poseidon2PartialRounds12 {
            choices: self
                .choices
                .iter()
                .map(|choice| (choice.selector, choice.first))
                .collect(),
            state_before: self.state_before,
            state_after: self.state_after,
            powers: self.powers.clone(),
        }
    }
}

#[derive(Clone)]
enum RoundKind {
    Full(usize),
    Partial(Range<usize>),
}

#[derive(Clone)]
struct RoundLayout {
    kind: RoundKind,
    state_before: [usize; POSEIDON2_WIDTH],
    state_after: [usize; POSEIDON2_WIDTH],
    powers_start: usize,
}

struct PermutationLayout {
    premix: [usize; POSEIDON2_WIDTH],
    rounds: Vec<RoundLayout>,
}

fn permutation_layout(auxiliary_start: usize, output: [usize; POSEIDON2_WIDTH]) -> PermutationLayout {
    let premix = columns(auxiliary_start);
    let mut cursor = auxiliary_start + POSEIDON2_WIDTH;
    let mut state_before = premix;
    let mut rounds = Vec::with_capacity(POSEIDON2_GROUPED_ROUNDS);

    for position in 0..POSEIDON2_GROUPED_ROUNDS {
        let last = position + 1 == POSEIDON2_GROUPED_ROUNDS;
        let kind = if position < POSEIDON2_HALF_FULL_ROUNDS {
            RoundKind::Full(position)
        } else if position >= POSEIDON2_HALF_FULL_ROUNDS + POSEIDON2_PARTIAL_GROUPS {
            RoundKind::Full(position - POSEIDON2_PARTIAL_GROUPS)
        } else {
            RoundKind::Partial(partial_group(position - POSEIDON2_HALF_FULL_ROUNDS))
        };
        let power_columns = match &kind {
            RoundKind::Full(_) => FULL_ROUND_POWER_COLUMNS,
            RoundKind::Partial(rounds) => 2 * rounds.len(),
        };
        let powers_start = cursor;
        cursor += power_columns;
        let state_after = if last {
            output
        } else {
            let state = columns(cursor);
            cursor += POSEIDON2_WIDTH;
            state
        };
        rounds.push(RoundLayout {
            kind,
            state_before,
            state_after,
            powers_start,
        });
        state_before = state_after;
    }
    debug_assert_eq!(cursor, auxiliary_start + POSEIDON2_PERMUTATION_AUX_COLUMNS);
    PermutationLayout { premix, rounds }
}

fn partial_rounds(round: &RoundLayout, rounds: &Range<usize>, selector: usize) -> Poseidon2PartialRounds12 {
    Poseidon2PartialRounds12 {
        choices: [Poseidon2PartialRoundsChoice::for_first_round(selector, rounds.start)],
        state_before: round.state_before,
        state_after: round.state_after,
        powers: (0..rounds.len())
            .map(|offset| columns(round.powers_start + 2 * offset))
            .collect(),
    }
}

/// Fully unrolled width-12 Poseidon2 permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2Permutation12 {
    pub input: [usize; POSEIDON2_WIDTH],
    pub output: [usize; POSEIDON2_WIDTH],
    pub auxiliary_start: usize,
}

impl Poseidon2Permutation12 {
    pub const fn auxiliary_range(&self) -> Range<usize> {
        self.auxiliary_start..self.auxiliary_start + POSEIDON2_PERMUTATION_AUX_COLUMNS
    }

    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        self.emit_constraints(builder);
        builder.record_gadget(
            GadgetDescriptor::Poseidon2Permutation12 {
                input: self.input,
                output: self.output,
                auxiliary_start: self.auxiliary_start,
                auxiliary_len: POSEIDON2_PERMUTATION_AUX_COLUMNS,
            },
            first_row,
        );
    }

    /// Fill the premix, S-box powers, and intermediate states without
    /// overwriting the application-owned permutation output.
    pub fn assign_auxiliaries(&self, assignment: &mut [F]) {
        self.assign_columns(assignment, false);
    }

    pub(crate) fn emit_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let layout = permutation_layout(self.auxiliary_start, self.output);
        let matrix = external_matrix();
        for lane in 0..POSEIDON2_WIDTH {
            let terms = core::iter::once((layout.premix[lane], F::ONE)).chain(
                matrix[lane]
                    .iter()
                    .enumerate()
                    .map(|(input, &coefficient)| (self.input[input], -coefficient)),
            );
            builder.push_linear_zero(terms);
        }

        let selector = builder.const_one_column();
        for round in &layout.rounds {
            match &round.kind {
                RoundKind::Full(round_index) => Poseidon2FullRound12 {
                    choices: [Poseidon2FullRoundChoice::for_round(selector, *round_index)],
                    state_before: round.state_before,
                    state_after: round.state_after,
                    powers: core::array::from_fn(|lane| columns(round.powers_start + 2 * lane)),
                }
                .emit_constraints(builder),
                RoundKind::Partial(rounds) => partial_rounds(round, rounds, selector).emit_constraints(builder),
            }
        }
    }

    pub(crate) fn assign_columns(&self, assignment: &mut [F], assign_output: bool) {
        let layout = permutation_layout(self.auxiliary_start, self.output);
        let mut state = self.input.map(|column| assignment[column]);
        apply_initial_linear(&mut state);
        for lane in 0..POSEIDON2_WIDTH {
            assignment[layout.premix[lane]] = state[lane];
        }

        let round_count = layout.rounds.len();
        for (position, round) in layout.rounds.iter().enumerate() {
            let state_after = match &round.kind {
                RoundKind::Full(round_index) => assign_fixed_full_round(
                    *round_index,
                    round.state_before,
                    core::array::from_fn(|lane| columns(round.powers_start + 2 * lane)),
                    assignment,
                ),
                RoundKind::Partial(rounds) => {
                    let constants: Vec<_> = rounds.clone().map(partial_round_constant).collect();
                    partial_rounds(round, rounds, 0).assign_rounds(&constants, assignment)
                }
            };
            if position + 1 < round_count || assign_output {
                for lane in 0..POSEIDON2_WIDTH {
                    assignment[round.state_after[lane]] = state_after[lane];
                }
            }
        }
    }
}
