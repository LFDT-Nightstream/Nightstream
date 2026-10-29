//! The compression terminal artifact (`formal/compression-lean`): the F′
//! Stage 1 rows up to PiRLC and the columns that layer 1 reads.
//!
//! Owns decoding and the structural checks of the artifact. Does not own the
//! rows or their meaning (Lean does), nor the shrink circuit that uses them.

use serde_json::Value;

use super::v1_2::{
    PI_CCS_V1_2_COEFFICIENT_COUNT, PI_CCS_V1_2_FRESH_COMMITMENT_WORDS, PI_CCS_V1_2_MATRIX_COUNT,
    PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_2_ROUND_COUNT,
};
use super::validate::{decode_components, owned_witness_intervals, witness_segments};
use super::{IdentityKind, LoadedPackage, PackageError};
use crate::identity::relation_identifier;

const ARTIFACT_SCHEMA: u64 = 1;
const STATE_WORDS: usize = 16;

/// The package columns of the PiRLC parent and of the sponge state after the
/// PiRLC sampler, in the order layer 1 reads its claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalOutputs {
    /// 22 rows of 54 lanes, row-major.
    pub commitment: Vec<usize>,
    pub public_input: Vec<usize>,
    /// Two words per coordinate.
    pub point: Vec<usize>,
    pub eval_k: Vec<usize>,
    /// Matrix-major, two words per coefficient.
    pub eval_a: Vec<usize>,
    pub state: Vec<usize>,
}

/// Load the terminal artifact `[1, package, outputs]`. Its package is a
/// subset of a full package, so its rows and witness columns have gaps.
/// Every other check of a full package applies, and no two producers write
/// one column. `expected_identifier` covers the whole artifact.
pub fn load_terminal_package(
    bytes: &[u8],
    expected_identifier: [u64; 4],
) -> Result<(LoadedPackage, TerminalOutputs), PackageError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let mut canonical = serde_json::to_vec(&value)?;
    canonical.push(b'\n');
    if bytes != canonical {
        return Err(PackageError::NonCanonicalBytes);
    }
    let computed = relation_identifier(&value)?;
    if computed != expected_identifier {
        return Err(PackageError::IdentityMismatch {
            kind: IdentityKind::Relation,
            expected: expected_identifier,
            computed,
        });
    }
    let Value::Array(mut parts) = value else {
        return Err(PackageError::Malformed("terminal artifact"));
    };
    if parts.len() != 3 || parts[0].as_u64() != Some(ARTIFACT_SCHEMA) {
        return Err(PackageError::Malformed("terminal artifact schema"));
    }
    let raw_outputs = parts.pop().expect("three parts");
    let (package, terminal) = decode_components(serde_json::from_value(parts.pop().expect("three parts"))?, None)?;
    if terminal != [Value::from(0u64)] {
        return Err(PackageError::Malformed("terminal package option"));
    }
    // Only the prior-state chain: the terminal package has no output state.
    if package.hash_chains.len() != 1 || package.hash_chains[0].phase != 1 {
        return Err(PackageError::Malformed("terminal package hash chains"));
    }
    let mut intervals = owned_witness_intervals(&package, &witness_segments(&package.layout))?;
    intervals.sort_unstable();
    if intervals.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return Err(PackageError::Malformed("witness interval overlap"));
    }
    let outputs = decode_outputs(&raw_outputs, &package)?;
    Ok((package, outputs))
}

fn decode_outputs(value: &Value, package: &LoadedPackage) -> Result<TerminalOutputs, PackageError> {
    let lists = value
        .as_array()
        .filter(|lists| lists.len() == 6)
        .ok_or(PackageError::Malformed("terminal outputs"))?;
    let columns = |index: usize, length: usize| -> Result<Vec<usize>, PackageError> {
        let list = lists[index]
            .as_array()
            .filter(|list| list.len() == length)
            .ok_or(PackageError::Malformed("terminal output length"))?;
        list.iter()
            .map(|expression| match expression.as_array().map(Vec::as_slice) {
                Some([tag, column]) if tag.as_u64() == Some(0) => column
                    .as_u64()
                    .and_then(|column| usize::try_from(column).ok())
                    .filter(|&column| {
                        column < package.layout.total_column_count && column != package.layout.constant_column
                    })
                    .ok_or(PackageError::Malformed("terminal output column")),
                _ => Err(PackageError::Malformed("terminal output expression")),
            })
            .collect()
    };
    Ok(TerminalOutputs {
        commitment: columns(0, PI_CCS_V1_2_FRESH_COMMITMENT_WORDS)?,
        public_input: columns(1, PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS)?,
        point: columns(2, 2 * PI_CCS_V1_2_ROUND_COUNT)?,
        eval_k: columns(3, 2 * PI_CCS_V1_2_COEFFICIENT_COUNT)?,
        eval_a: columns(4, 2 * PI_CCS_V1_2_MATRIX_COUNT * PI_CCS_V1_2_COEFFICIENT_COUNT)?,
        state: columns(5, STATE_WORDS)?,
    })
}
