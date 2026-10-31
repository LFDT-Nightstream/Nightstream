//! Native and R1CS forms of the shared eight-word event commitment.
//!
//! This module owns the four-lane chaining state, block compression, and
//! feed-forward relation. The underlying permutation belongs to
//! [`crate::poseidon2`]; event encoding and trace scheduling belong to the
//! consuming application.

use std::ops::Range;

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::poseidon2::{permute, Poseidon2Permutation12, POSEIDON2_PERMUTATION_AUX_COLUMNS, POSEIDON2_WIDTH};
use crate::{GadgetDescriptor, TaggedR1csBuilder};

pub const EVENT_COMMITMENT_STATE_WIDTH: usize = 4;
pub const EVENT_COMMITMENT_BLOCK_WIDTH: usize = 8;

/// The event compression retains the full permutation output as auxiliary
/// data; only its first four lanes are exposed after feed-forward.
pub const EVENT_COMMITMENT_AUX_COLUMNS: usize = POSEIDON2_PERMUTATION_AUX_COLUMNS + POSEIDON2_WIDTH;

/// Four field elements carried between committed event blocks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventCommitmentState([F; EVENT_COMMITMENT_STATE_WIDTH]);

impl EventCommitmentState {
    pub const fn new(lanes: [F; EVENT_COMMITMENT_STATE_WIDTH]) -> Self {
        Self(lanes)
    }

    pub const fn into_lanes(self) -> [F; EVENT_COMMITMENT_STATE_WIDTH] {
        self.0
    }

    pub fn canonical_u64(self) -> [u64; EVENT_COMMITMENT_STATE_WIDTH] {
        self.0.map(|lane| lane.as_canonical_u64())
    }
}

/// Commit one opaque eight-word block.
pub fn commit_block(
    previous: [F; EVENT_COMMITMENT_STATE_WIDTH],
    block: [F; EVENT_COMMITMENT_BLOCK_WIDTH],
) -> [F; EVENT_COMMITMENT_STATE_WIDTH] {
    let mut state = [F::ZERO; POSEIDON2_WIDTH];
    state[..EVENT_COMMITMENT_STATE_WIDTH].copy_from_slice(&previous);
    state[EVENT_COMMITMENT_STATE_WIDTH..].copy_from_slice(&block);
    let permuted = permute(state);
    core::array::from_fn(|lane| permuted[lane] + previous[lane])
}

/// Fold an ordered block stream from an explicit initial state.
pub fn fold_blocks(
    initial: EventCommitmentState,
    blocks: &[[F; EVENT_COMMITMENT_BLOCK_WIDTH]],
) -> EventCommitmentState {
    let mut state = initial.into_lanes();
    for &block in blocks {
        state = commit_block(state, block);
    }
    EventCommitmentState::new(state)
}

fn columns<const N: usize>(start: usize) -> [usize; N] {
    core::array::from_fn(|offset| start + offset)
}

/// One event-chain transition, including the four-lane feed-forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventCommitment {
    pub previous: [usize; EVENT_COMMITMENT_STATE_WIDTH],
    pub block: [usize; EVENT_COMMITMENT_BLOCK_WIDTH],
    pub output: [usize; EVENT_COMMITMENT_STATE_WIDTH],
    pub auxiliary_start: usize,
}

impl EventCommitment {
    pub const fn auxiliary_range(&self) -> Range<usize> {
        self.auxiliary_start..self.auxiliary_start + EVENT_COMMITMENT_AUX_COLUMNS
    }

    fn permutation(&self) -> Poseidon2Permutation12 {
        Poseidon2Permutation12 {
            input: core::array::from_fn(|lane| {
                if lane < EVENT_COMMITMENT_STATE_WIDTH {
                    self.previous[lane]
                } else {
                    self.block[lane - EVENT_COMMITMENT_STATE_WIDTH]
                }
            }),
            output: columns(self.auxiliary_start + POSEIDON2_PERMUTATION_AUX_COLUMNS),
            auxiliary_start: self.auxiliary_start,
        }
    }

    pub fn push_constraints<Owner: Clone>(&self, builder: &mut TaggedR1csBuilder<'_, Owner>) {
        let first_row = builder.next_row_index();
        let permutation = self.permutation();
        permutation.emit_constraints(builder);
        for lane in 0..EVENT_COMMITMENT_STATE_WIDTH {
            builder.push_linear_zero([
                (self.output[lane], F::ONE),
                (permutation.output[lane], -F::ONE),
                (self.previous[lane], -F::ONE),
            ]);
        }
        builder.record_gadget(
            GadgetDescriptor::EventCommitment {
                previous: self.previous,
                block: self.block,
                output: self.output,
                auxiliary_start: self.auxiliary_start,
                auxiliary_len: EVENT_COMMITMENT_AUX_COLUMNS,
            },
            first_row,
        );
    }

    /// Fill only the permutation auxiliaries. The four-lane commitment output
    /// remains application owned and is checked by the feed-forward rows.
    pub fn assign_auxiliaries(&self, assignment: &mut [F]) {
        let permutation = self.permutation();
        permutation.assign_columns(assignment, true);
    }
}
