//! Native and R1CS forms of the exact width-12 Goldilocks Poseidon2
//! permutation. Event compression and application-specific round scheduling
//! belong to their respective consumers.

use std::{ops::Range, sync::OnceLock};

use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::{
    default_goldilocks_poseidon2_12, Poseidon2Goldilocks, GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL,
    GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL, GOLDILOCKS_POSEIDON2_RC_12_INTERNAL, MATRIX_DIAG_12_GOLDILOCKS,
};
use p3_poseidon2::{matmul_internal, mds_light_permutation, MDSMat4};
use p3_symmetric::Permutation;

use crate::gadgets::push_pow7_expression;
use crate::{GadgetDescriptor, TaggedR1csBuilder};

pub const POSEIDON2_WIDTH: usize = 12;
pub const POSEIDON2_HALF_FULL_ROUNDS: usize = 4;
pub const POSEIDON2_FULL_ROUNDS: usize = 2 * POSEIDON2_HALF_FULL_ROUNDS;
pub const POSEIDON2_PARTIAL_ROUNDS: usize = 22;
const _: () = assert!(POSEIDON2_PARTIAL_ROUNDS.is_multiple_of(2));
pub const POSEIDON2_PARTIAL_PAIRS: usize = POSEIDON2_PARTIAL_ROUNDS / 2;
pub const POSEIDON2_GROUPED_ROUNDS: usize = POSEIDON2_FULL_ROUNDS + POSEIDON2_PARTIAL_PAIRS;

fn permutation() -> &'static Poseidon2Goldilocks<POSEIDON2_WIDTH> {
    static PERMUTATION: OnceLock<Poseidon2Goldilocks<POSEIDON2_WIDTH>> = OnceLock::new();
    PERMUTATION.get_or_init(default_goldilocks_poseidon2_12)
}

/// Apply the exact width-12 Goldilocks Poseidon2 permutation.
pub fn permute(state: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
    permutation().permute(state)
}

/// Goldilocks Poseidon2 S-box `x -> x^7`.
pub fn sbox7(x: F) -> F {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x2 * x
}

/// Round constants for full round `0..8` in permutation order.
pub fn full_round_constants(round: usize) -> &'static [F; POSEIDON2_WIDTH] {
    assert!(round < POSEIDON2_FULL_ROUNDS, "full-round index out of range");
    if round < POSEIDON2_HALF_FULL_ROUNDS {
        &GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL[round]
    } else {
        &GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL[round - POSEIDON2_HALF_FULL_ROUNDS]
    }
}

/// Constants for partial-round pair `0..11` in permutation order.
pub fn partial_pair_constants(pair: usize) -> (F, F) {
    assert!(pair < POSEIDON2_PARTIAL_PAIRS, "partial-pair index out of range");
    (
        GOLDILOCKS_POSEIDON2_RC_12_INTERNAL[2 * pair],
        GOLDILOCKS_POSEIDON2_RC_12_INTERNAL[2 * pair + 1],
    )
}

pub fn external_linear(state: &mut [F; POSEIDON2_WIDTH]) {
    mds_light_permutation(state, &MDSMat4);
}

pub fn internal_linear(state: &mut [F; POSEIDON2_WIDTH]) {
    matmul_internal(state, MATRIX_DIAG_12_GOLDILOCKS);
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
        *lane = sbox7(*lane + constant);
    }
    external_linear(state);
}

/// Apply two consecutive partial rounds.
pub fn apply_partial_pair(pair: usize, state: &mut [F; POSEIDON2_WIDTH]) {
    let (first, second) = partial_pair_constants(pair);
    for constant in [first, second] {
        state[0] = sbox7(state[0] + constant);
        internal_linear(state);
    }
}

const FULL_ROUND_POWER_COLUMNS: usize = 4 * POSEIDON2_WIDTH;
const PARTIAL_PAIR_POWER_COLUMNS: usize = 8;
const INTERMEDIATE_STATES: usize = POSEIDON2_GROUPED_ROUNDS - 1;

/// Auxiliary columns used by one fully unrolled width-12 permutation.
pub const POSEIDON2_PERMUTATION_AUX_COLUMNS: usize = POSEIDON2_WIDTH
    + POSEIDON2_FULL_ROUNDS * FULL_ROUND_POWER_COLUMNS
    + POSEIDON2_PARTIAL_PAIRS * PARTIAL_PAIR_POWER_COLUMNS
    + INTERMEDIATE_STATES * POSEIDON2_WIDTH;

fn columns<const N: usize>(start: usize) -> [usize; N] {
    core::array::from_fn(|offset| start + offset)
}

fn assign_powers(value: F, powers: [usize; 4], assignment: &mut [F]) {
    let [x2, x4, x6, x7] = powers;
    assignment[x2] = value * value;
    assignment[x4] = assignment[x2] * assignment[x2];
    assignment[x6] = assignment[x2] * assignment[x4];
    assignment[x7] = assignment[x6] * value;
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
    powers: [[usize; 4]; POSEIDON2_WIDTH],
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
    pub powers: [[usize; 4]; POSEIDON2_WIDTH],
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
            push_pow7_expression(builder, &expression, self.powers[lane]);
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
                    .map(|(input, &coefficient)| (self.powers[input][3], -coefficient)),
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
    powers: [[usize; 4]; POSEIDON2_WIDTH],
    assignment: &mut [F],
) -> [F; POSEIDON2_WIDTH] {
    assign_full_round_auxiliaries(*full_round_constants(round), state_before, powers, assignment);
    let mut state = core::array::from_fn(|lane| assignment[powers[lane][3]]);
    external_linear(&mut state);
    state
}

fn assign_partial_pair_auxiliaries(
    constants: (F, F),
    state_before: [usize; POSEIDON2_WIDTH],
    powers: [usize; PARTIAL_PAIR_POWER_COLUMNS],
    assignment: &mut [F],
) -> [F; POSEIDON2_WIDTH] {
    let (first_constant, second_constant) = constants;
    let mut state = state_before.map(|column| assignment[column]);
    assign_powers(state[0] + first_constant, powers[..4].try_into().unwrap(), assignment);
    state[0] = assignment[powers[3]];
    internal_linear(&mut state);
    assign_powers(state[0] + second_constant, powers[4..].try_into().unwrap(), assignment);
    state
}

/// One selector and pair index in a selectable pair of partial rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2PartialPairChoice {
    pub selector: usize,
    pub pair: usize,
}

impl Poseidon2PartialPairChoice {
    pub fn for_pair(selector: usize, pair: usize) -> Self {
        assert!(pair < POSEIDON2_PARTIAL_PAIRS, "partial-pair index out of range");
        Self { selector, pair }
    }
}

/// Two width-12 partial rounds sharing eight `x^7` power columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poseidon2PartialPair12<const N: usize = 1> {
    pub choices: [Poseidon2PartialPairChoice; N],
    pub state_before: [usize; POSEIDON2_WIDTH],
    pub state_after: [usize; POSEIDON2_WIDTH],
    pub powers: [usize; PARTIAL_PAIR_POWER_COLUMNS],
}

impl<const N: usize> Poseidon2PartialPair12<N> {
    /// Emit a self-contained selectable pair, including Booleanity and
    /// at-most-one enforcement for its selectors.
    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        push_at_most_one_selectors(builder, self.choices.map(|choice| choice.selector));
        self.emit_constraints(builder);
        builder.record_gadget(self.descriptor(), first_row);
    }

    /// Emit only the pair equations when the surrounding application already
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
        let constants = self
            .choices
            .iter()
            .fold((F::ZERO, F::ZERO), |(first, second), choice| {
                let (choice_first, choice_second) = partial_pair_constants(choice.pair);
                let selector = assignment[choice.selector];
                (first + selector * choice_first, second + selector * choice_second)
            });
        let _ = assign_partial_pair_auxiliaries(constants, self.state_before, self.powers, assignment);
    }

    fn emit_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_input: Vec<_> = core::iter::once((self.state_before[0], F::ONE))
            .chain(
                self.choices
                    .iter()
                    .map(|choice| (choice.selector, partial_pair_constants(choice.pair).0)),
            )
            .collect();
        push_pow7_expression(builder, &first_input, self.powers[..4].try_into().unwrap());

        let matrix = internal_matrix();
        let first_output = self.powers[3];
        let mut second_input = vec![(first_output, matrix[0][0])];
        second_input.extend(
            matrix[0]
                .iter()
                .enumerate()
                .skip(1)
                .map(|(lane, &coefficient)| (self.state_before[lane], coefficient)),
        );
        second_input.extend(
            self.choices
                .iter()
                .map(|choice| (choice.selector, partial_pair_constants(choice.pair).1)),
        );
        push_pow7_expression(builder, &second_input, self.powers[4..].try_into().unwrap());

        let gate: Vec<_> = self
            .choices
            .iter()
            .map(|choice| (choice.selector, F::ONE))
            .collect();
        let second_output = self.powers[7];
        for lane in 0..POSEIDON2_WIDTH {
            let mut coefficient_first_output = F::ZERO;
            let mut coefficient_state = [F::ZERO; POSEIDON2_WIDTH];
            for intermediate in 1..POSEIDON2_WIDTH {
                coefficient_first_output += matrix[lane][intermediate] * matrix[intermediate][0];
                for input in 1..POSEIDON2_WIDTH {
                    coefficient_state[input] += matrix[lane][intermediate] * matrix[intermediate][input];
                }
            }
            let terms = core::iter::once((self.state_after[lane], F::ONE))
                .chain([
                    (second_output, -matrix[lane][0]),
                    (first_output, -coefficient_first_output),
                ])
                .chain(
                    coefficient_state
                        .into_iter()
                        .enumerate()
                        .skip(1)
                        .map(|(input, coefficient)| (self.state_before[input], -coefficient)),
                );
            builder.push_row(gate.iter().copied(), terms, []);
        }
    }

    fn descriptor(&self) -> GadgetDescriptor {
        GadgetDescriptor::Poseidon2PartialPair12 {
            choices: self
                .choices
                .iter()
                .map(|choice| (choice.selector, choice.pair))
                .collect(),
            state_before: self.state_before,
            state_after: self.state_after,
            powers: self.powers,
        }
    }
}

fn assign_fixed_partial_pair(
    pair: usize,
    state_before: [usize; POSEIDON2_WIDTH],
    powers: [usize; PARTIAL_PAIR_POWER_COLUMNS],
    assignment: &mut [F],
) -> [F; POSEIDON2_WIDTH] {
    let mut state = assign_partial_pair_auxiliaries(partial_pair_constants(pair), state_before, powers, assignment);
    state[0] = assignment[powers[7]];
    internal_linear(&mut state);
    state
}

#[derive(Clone, Copy)]
enum RoundKind {
    Full(usize),
    PartialPair(usize),
}

#[derive(Clone, Copy)]
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
        let full_round = if position < POSEIDON2_HALF_FULL_ROUNDS {
            Some(position)
        } else if position >= POSEIDON2_HALF_FULL_ROUNDS + POSEIDON2_PARTIAL_PAIRS {
            Some(position - POSEIDON2_PARTIAL_PAIRS)
        } else {
            None
        };
        let power_columns = if full_round.is_some() {
            FULL_ROUND_POWER_COLUMNS
        } else {
            PARTIAL_PAIR_POWER_COLUMNS
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
        let kind = if let Some(round) = full_round {
            RoundKind::Full(round)
        } else {
            RoundKind::PartialPair(position - POSEIDON2_HALF_FULL_ROUNDS)
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
        for round in layout.rounds {
            match round.kind {
                RoundKind::Full(round_index) => Poseidon2FullRound12 {
                    choices: [Poseidon2FullRoundChoice::for_round(selector, round_index)],
                    state_before: round.state_before,
                    state_after: round.state_after,
                    powers: core::array::from_fn(|lane| columns(round.powers_start + 4 * lane)),
                }
                .emit_constraints(builder),
                RoundKind::PartialPair(pair) => Poseidon2PartialPair12 {
                    choices: [Poseidon2PartialPairChoice::for_pair(selector, pair)],
                    state_before: round.state_before,
                    state_after: round.state_after,
                    powers: columns(round.powers_start),
                }
                .emit_constraints(builder),
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
        for (position, round) in layout.rounds.into_iter().enumerate() {
            let state_after = match round.kind {
                RoundKind::Full(round_index) => assign_fixed_full_round(
                    round_index,
                    round.state_before,
                    core::array::from_fn(|lane| columns(round.powers_start + 4 * lane)),
                    assignment,
                ),
                RoundKind::PartialPair(pair) => {
                    assign_fixed_partial_pair(pair, round.state_before, columns(round.powers_start), assignment)
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
