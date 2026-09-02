//! Shared execution of the Lean-owned Stage 1 source-column permutation.

use super::PackageError;

const PILOT_SOURCE_COLUMN_COUNT: usize = 14_722_512;
const PILOT_PRIVATE_COLUMN_COUNT: usize = 14_722_238;
const PILOT_INPUT_PRIVATE_COLUMN_COUNT: usize = 98_786;
const PROOF_INPUT_COLUMN_COUNT: usize = 29_288;
const PROOF_INPUT_SOURCE_START: usize = 14_722_516;
const PI_CCS_PHASE_OFFSET: usize = 14_751_804;
const PI_CCS_LOCAL_START: usize = 14_751_526;
const PRIVATE_COLUMN_COUNT: usize = 29_336_446;
const EXPECTED_CONTEXT_PUBLIC_START: usize = 29_336_721;

const PILOT_PRIOR_PUBLIC_START: usize = 49_393;
const PILOT_OUTPUT_PREIMAGE_START: usize = 49_663;
const PILOT_OUTPUT_DIGEST_START: usize = 99_056;
const PILOT_WITNESS_START: usize = 99_060;
const PILOT_SECOND_PRIVATE_START: usize = 49_393;
const PILOT_WITNESS_PRIVATE_START: usize = 98_786;
const PILOT_FIRST_PUBLIC_START: usize = 14_722_239;
const PILOT_SECOND_PUBLIC_START: usize = 14_722_509;

pub(super) fn source_to_spartan(column: usize) -> Result<usize, PackageError> {
    if column < PILOT_SOURCE_COLUMN_COUNT {
        return lift_pilot_column(pilot_source_to_spartan(column)?);
    }
    if column < PROOF_INPUT_SOURCE_START {
        return add(
            EXPECTED_CONTEXT_PUBLIC_START,
            column - PILOT_SOURCE_COLUMN_COUNT,
            "verifier context column",
        );
    }
    if column < PI_CCS_PHASE_OFFSET {
        return add(
            PILOT_INPUT_PRIVATE_COLUMN_COUNT,
            column - PROOF_INPUT_SOURCE_START,
            "proof input column",
        );
    }
    add(PI_CCS_LOCAL_START, column - PI_CCS_PHASE_OFFSET, "local source column")
}

fn pilot_source_to_spartan(column: usize) -> Result<usize, PackageError> {
    if column < PILOT_PRIOR_PUBLIC_START {
        return Ok(column);
    }
    if column < PILOT_OUTPUT_PREIMAGE_START {
        return add(
            PILOT_FIRST_PUBLIC_START,
            column - PILOT_PRIOR_PUBLIC_START,
            "pilot prior public column",
        );
    }
    if column < PILOT_OUTPUT_DIGEST_START {
        return add(
            PILOT_SECOND_PRIVATE_START,
            column - PILOT_OUTPUT_PREIMAGE_START,
            "pilot output private column",
        );
    }
    if column < PILOT_WITNESS_START {
        return add(
            PILOT_SECOND_PUBLIC_START,
            column - PILOT_OUTPUT_DIGEST_START,
            "pilot output public column",
        );
    }
    add(
        PILOT_WITNESS_PRIVATE_START,
        column - PILOT_WITNESS_START,
        "pilot witness column",
    )
}

fn lift_pilot_column(column: usize) -> Result<usize, PackageError> {
    if column < PILOT_INPUT_PRIVATE_COLUMN_COUNT {
        return Ok(column);
    }
    if column < PILOT_PRIVATE_COLUMN_COUNT {
        return add(column, PROOF_INPUT_COLUMN_COUNT, "lifted pilot column");
    }
    add(
        PRIVATE_COLUMN_COUNT,
        column - PILOT_PRIVATE_COLUMN_COUNT,
        "lifted pilot public column",
    )
}

fn add(left: usize, right: usize, location: &'static str) -> Result<usize, PackageError> {
    left.checked_add(right)
        .ok_or(PackageError::Invalid(location))
}
