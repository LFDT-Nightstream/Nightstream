//! Poseidon2 sponge ownership inside the stabilized Π_CCS output-digest span.
//!
//! Owns: selection of the unique retained hash call; its 64-field, 16-absorb
//! plus one-pad schedule; and one-to-one pairing with retained permutation
//! traces.
//!
//! Does not own: the meaning of the 64 inputs, raw permutation equations,
//! native parity, transcript placement, collision resistance, or row removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: this validates physical schedule continuity only. The
//! hash becomes protocol binding only after its input columns are refined to
//! the canonical SIS envelope and its outputs to the next transcript event.
//!
//! | Leaf | Physical obligation |
//! |---|---|
//! | `absorb[0..16]` | four ordered input columns, four defining rows, one retained permutation |
//! | `pad` | one defining row followed by one retained permutation |
//! | `digest` | first four lanes of the final permutation |

use crate::engine::r1cs_circuit::builder::{Poseidon2HashAudit, Poseidon2HashRoundAuditKind};
use crate::frontends::r1cs_f_prime::SparseR1cs;

use super::invalid;
use crate::frontends::r1cs_f_prime::ivc::R1csIvcError;

const RATE: usize = 4;
const INPUT_FIELDS: usize = 64;
const ABSORB_ROUNDS: usize = INPUT_FIELDS / RATE;
const TOTAL_ROUNDS: usize = ABSORB_ROUNDS + 1;

pub(super) fn recover(
    arm: &SparseR1cs,
    sis_start: usize,
    claim_start: usize,
) -> Result<Poseidon2HashAudit, R1csIvcError> {
    let hashes = arm
        .poseidon2_hash_audits()
        .iter()
        .filter(|hash| sis_start <= hash.row_start && hash.row_end <= claim_start)
        .collect::<Vec<_>>();
    let [hash] = hashes.as_slice() else {
        return Err(invalid(format!(
            "expected one Poseidon2 hash inside PiCCS output SIS rows {sis_start}..{claim_start}, found {}",
            hashes.len()
        )));
    };
    validate_hash(arm, hash)?;
    Ok((*hash).clone())
}

fn validate_hash(arm: &SparseR1cs, hash: &Poseidon2HashAudit) -> Result<(), R1csIvcError> {
    if hash.input_cols.len() != INPUT_FIELDS {
        return Err(invalid(format!(
            "PiCCS output Poseidon2 hash has {} inputs, expected {INPUT_FIELDS}",
            hash.input_cols.len()
        )));
    }
    if hash.rounds.len() != TOTAL_ROUNDS {
        return Err(invalid(format!(
            "PiCCS output Poseidon2 hash has {} rounds, expected {TOTAL_ROUNDS}",
            hash.rounds.len()
        )));
    }
    if hash.row_start != hash.zero_row || hash.row_start >= hash.row_end {
        return Err(invalid(format!(
            "PiCCS output Poseidon2 zero row {} does not open hash span {}..{}",
            hash.zero_row, hash.row_start, hash.row_end
        )));
    }

    let permutations = arm
        .poseidon2_traces()
        .iter()
        .filter(|trace| hash.row_start <= trace.row_start && trace.row_end <= hash.row_end)
        .collect::<Vec<_>>();
    if permutations.len() != TOTAL_ROUNDS {
        return Err(invalid(format!(
            "PiCCS output Poseidon2 hash owns {} retained permutations, expected {TOTAL_ROUNDS}",
            permutations.len()
        )));
    }

    let mut state = [hash.zero_col; 8];
    let mut next_row = hash.zero_row + 1;
    for (index, (round, permutation)) in hash.rounds.iter().zip(permutations).enumerate() {
        if round.state_before_cols != state {
            return Err(invalid(format!(
                "PiCCS output Poseidon2 round {index} does not continue the preceding state"
            )));
        }
        let defining_count = match &round.kind {
            Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
                if index >= ABSORB_ROUNDS {
                    return Err(invalid("PiCCS output Poseidon2 places an absorb after its fixed input"));
                }
                let expected = &hash.input_cols[index * RATE..(index + 1) * RATE];
                if chunk_cols.as_slice() != expected {
                    return Err(invalid(format!(
                        "PiCCS output Poseidon2 absorb {index} does not consume its ordered input slice"
                    )));
                }
                if round.permutation_input_cols[RATE..] != state[RATE..] {
                    return Err(invalid(format!(
                        "PiCCS output Poseidon2 absorb {index} changes an unabsorbed capacity lane"
                    )));
                }
                RATE
            }
            Poseidon2HashRoundAuditKind::Pad => {
                if index != ABSORB_ROUNDS {
                    return Err(invalid(format!(
                        "PiCCS output Poseidon2 padding occurs at round {index}, expected {ABSORB_ROUNDS}"
                    )));
                }
                if round.permutation_input_cols[1..] != state[1..] {
                    return Err(invalid(
                        "PiCCS output Poseidon2 padding changes a nonzero-index state lane",
                    ));
                }
                1
            }
        };
        let expected_rows = (next_row..next_row + defining_count).collect::<Vec<_>>();
        if round.defining_rows != expected_rows {
            return Err(invalid(format!(
                "PiCCS output Poseidon2 round {index} defining rows are not the exact contiguous prefix"
            )));
        }
        if permutation.row_start != next_row + defining_count
            || permutation.row_end <= permutation.row_start
            || permutation.input_cols != round.permutation_input_cols
            || permutation.output_cols != round.permutation_output_cols
        {
            return Err(invalid(format!(
                "PiCCS output Poseidon2 round {index} does not match one retained permutation"
            )));
        }
        next_row = permutation.row_end;
        state = round.permutation_output_cols;
    }

    if next_row != hash.row_end || hash.output_cols != state[..4] {
        return Err(invalid(
            "PiCCS output Poseidon2 digest does not close at the first four final-state lanes",
        ));
    }
    Ok(())
}
