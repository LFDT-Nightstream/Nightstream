//! Exact PiCCS-prefix assignment gate for the sealed production package.
//!
//! This test reuses the package witness executor and stops before PiRLC. It
//! does not construct PiDEC or application inputs.

use std::{fs, path::PathBuf};

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde_json::Value;

use super::super::{
    checked_end, compact, scheduled_assignments, v1_1, LoadedPackage, ScheduledAssignment, ScheduledInvocation,
};
use crate::sparse::eval_sparse_combination;
use crate::witness::execute_witness_batch;
use crate::{
    load_poseidon2_hash_chain_v1_package, LoadedPerApplicationPackage, PackageError, PiCcsV1_1OutputEvaluations,
    PiCcsV1_1PackageInputs, Stage1VerifierBinding,
};

#[allow(dead_code, unused_imports)]
#[path = "../../src/bin/check_package_conformance/support.rs"]
mod conformance_support;

#[path = "../per_application_logical_matrix_conformance/reference/mod.rs"]
mod logical_reference;

const PI_CCS_CALLER_INPUT_COUNT: usize = 128_074;
const PI_CCS_ROW_START: usize = 14_623_730;
const PI_CCS_ROW_END: usize = 19_936_967;
// Stage1.sourceToSpartan maps the source boundary 20_064_823 here.
const PI_CCS_PRIVATE_END: usize = 20_064_545;
const PI_CCS_FIRST_GENERATED_COLUMN: usize = 14_751_526;
const PI_CCS_LOGICAL_ROW_END: usize = 3_864_823;
const PI_CCS_LOGICAL_BLOCK_ENDS: [usize; 9] = [
    1_160_900,
    2_321_800,
    3_036_576,
    3_051_784,
    3_863_453,
    3_864_783,
    3_864_791,
    PI_CCS_LOGICAL_ROW_END,
    3_879_205,
];
const PI_CCS_LOGICAL_BLOCK_OPCODES: [usize; 9] = [2, 2, 2, 1, 0, 0, 1, 1, 2];
const PUBLIC_INPUT_COUNT: usize = 278;
const PI_RLC_PHASE: u64 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundarySide {
    Prefix,
    Suffix,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PrefixExecutionCounts {
    permutations: usize,
    compact_invocations: usize,
    witness_batches: usize,
    generic_instructions: usize,
}

struct RawPiCcsAssignment {
    private_values: Vec<u64>,
    public_values: Vec<u64>,
    counts: PrefixExecutionCounts,
}

fn artifact_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

fn canonical_words(value: &Value) -> Vec<u64> {
    fn append(value: &Value, output: &mut Vec<u64>) {
        if let Some(word) = value.as_u64() {
            assert!(word < super::super::GOLDILOCKS_MODULUS, "canonical PiCCS fixture word");
            output.push(word);
            return;
        }
        for child in value.as_array().expect("PiCCS fixture word array") {
            append(child, output);
        }
    }

    let mut output = Vec::new();
    append(value, &mut output);
    output
}

fn extension_values(value: &Value) -> Vec<[u64; 2]> {
    value
        .as_array()
        .expect("PiCCS extension-value array")
        .iter()
        .map(|value| {
            canonical_words(value)
                .try_into()
                .expect("two-word PiCCS extension value")
        })
        .collect()
}

fn pi_ccs_inputs(bytes: &[u8], binding: &Stage1VerifierBinding) -> PiCcsV1_1PackageInputs {
    let parity: Value = serde_json::from_slice(bytes).expect("current PiCCS parity JSON");
    let parity = parity.as_array().expect("current PiCCS parity tuple");
    assert_eq!(parity.len(), 3, "current PiCCS parity tuple length");
    assert_eq!(parity[0].as_u64(), Some(8), "current PiCCS parity schema");
    let input = parity[1].as_array().expect("current PiCCS parity input");
    let output = parity[2].as_array().expect("current PiCCS parity output");
    assert_eq!(input.len(), 11, "current PiCCS input field count");
    assert_eq!(output.len(), 16, "current PiCCS output field count");
    assert_eq!(output[0].as_u64(), Some(1), "current PiCCS acceptance result");

    let fixture_verifier_context: [u64; 4] = canonical_words(&input[4])
        .try_into()
        .expect("four-word PiCCS verifier-context digest");
    assert_eq!(
        fixture_verifier_context,
        binding.verifier_context().digest(),
        "PiCCS fixture verifier-context binding",
    );

    let round_messages = input[6]
        .as_array()
        .expect("PiCCS round-message array")
        .iter()
        .map(extension_values)
        .collect();
    let eval_k = input[7]
        .as_array()
        .expect("PiCCS Eval_K source array")
        .iter()
        .map(extension_values)
        .collect();
    let eval_a = input[8]
        .as_array()
        .expect("PiCCS Eval_A source array")
        .iter()
        .map(|source| {
            source
                .as_array()
                .expect("PiCCS Eval_A matrix array")
                .iter()
                .map(extension_values)
                .collect()
        })
        .collect();
    let output_evaluations = PiCcsV1_1OutputEvaluations::new(eval_k, eval_a).expect("current PiCCS output evaluations");

    PiCcsV1_1PackageInputs::new(
        canonical_words(&input[0]),
        canonical_words(&input[1]),
        canonical_words(&input[5]),
        round_messages,
        output_evaluations,
        canonical_words(&input[2]),
        canonical_words(&input[3])
            .try_into()
            .expect("four-word PiCCS output digest"),
        binding.verifier_context().clone(),
    )
    .expect("current typed PiCCS input")
}

fn range_side(start: usize, length: usize, cutoff: usize) -> Result<BoundarySide, PackageError> {
    if length == 0 {
        return Err(PackageError::Invalid("PiCCS prefix empty range"));
    }
    let end = checked_end(start, length)?;
    if end <= cutoff {
        Ok(BoundarySide::Prefix)
    } else if start >= cutoff {
        Ok(BoundarySide::Suffix)
    } else {
        Err(PackageError::Invalid("PiCCS prefix range crossing"))
    }
}

fn require_same_side(left: BoundarySide, right: BoundarySide) -> Result<BoundarySide, PackageError> {
    if left == right {
        Ok(left)
    } else {
        Err(PackageError::Invalid("PiCCS prefix event crossing"))
    }
}

fn assignment_side(circuit: &LoadedPackage, assignment: ScheduledAssignment<'_>) -> Result<BoundarySide, PackageError> {
    let (write_side, row_side) = match assignment {
        ScheduledAssignment::Permutation(invocation) => (
            range_side(
                invocation.witness_start(),
                circuit.permutation.local_column_count,
                PI_CCS_PRIVATE_END,
            )?,
            Some(range_side(
                invocation.row_start(),
                circuit.permutation.rows.len(),
                PI_CCS_ROW_END,
            )?),
        ),
        ScheduledAssignment::Compact(invocation) => {
            let output_side = range_side(invocation.output_column, 1, PI_CCS_PRIVATE_END)?;
            let local_count = invocation.local_column_count(&circuit.compact_templates);
            let write_side = if local_count == 0 {
                output_side
            } else {
                let local_side = range_side(invocation.local_start, local_count, PI_CCS_PRIVATE_END)?;
                require_same_side(output_side, local_side)?
            };
            (
                write_side,
                Some(range_side(
                    invocation.row_start,
                    invocation.row_count(&circuit.compact_templates),
                    PI_CCS_ROW_END,
                )?),
            )
        }
        ScheduledAssignment::Batch(batch) => (
            range_side(batch.start, batch.end() - batch.start, PI_CCS_PRIVATE_END)?,
            None,
        ),
        ScheduledAssignment::Generic(instruction) => (
            range_side(instruction.target, 1, PI_CCS_PRIVATE_END)?,
            Some(range_side(instruction.row_index, 1, PI_CCS_ROW_END)?),
        ),
    };
    if let Some(row_side) = row_side {
        require_same_side(write_side, row_side)
    } else {
        Ok(write_side)
    }
}

fn is_first_pi_rlc_assignment(assignment: ScheduledAssignment<'_>) -> bool {
    matches!(
        assignment,
        ScheduledAssignment::Permutation(ScheduledInvocation::Explicit(invocation))
            if invocation.phase == PI_RLC_PHASE
                && invocation.row_start == PI_CCS_ROW_END
                && invocation.witness_start == PI_CCS_PRIVATE_END
    )
}

fn seed_assignment(
    circuit: &LoadedPackage,
    pi_ccs_private: &[u64],
    public_values: &[u64],
) -> Result<Vec<Goldilocks>, PackageError> {
    if pi_ccs_private.len() != PI_CCS_CALLER_INPUT_COUNT {
        return Err(PackageError::Invalid("PiCCS prefix caller-input length"));
    }
    if public_values.len() != circuit.layout.public_column_count {
        return Err(PackageError::Invalid("PiCCS prefix public-input length"));
    }

    let mut padded_private = vec![0; circuit.private_input_count()];
    padded_private
        .get_mut(..pi_ccs_private.len())
        .ok_or(PackageError::Invalid("PiCCS prefix padded private inputs"))?
        .copy_from_slice(pi_ccs_private);
    let mut assignment = vec![Goldilocks::ZERO; circuit.layout.total_column_count];
    let mut input_cursor = 0;
    for segment in &circuit.layout.private_segments {
        if v1_1::is_witness_role(segment.role) {
            continue;
        }
        let input_end = checked_end(input_cursor, segment.length)?;
        let segment_end = checked_end(segment.start, segment.length)?;
        let inputs = padded_private
            .get(input_cursor..input_end)
            .ok_or(PackageError::Invalid("PiCCS prefix padded private inputs"))?;
        let targets = assignment
            .get_mut(segment.start..segment_end)
            .ok_or(PackageError::Invalid("PiCCS prefix private segment"))?;
        for (target, value) in targets.iter_mut().zip(inputs) {
            *target = Goldilocks::from_u64(*value);
        }
        input_cursor = input_end;
    }
    if input_cursor != padded_private.len() {
        return Err(PackageError::Invalid("PiCCS prefix private-input coverage"));
    }

    assignment[circuit.layout.constant_column] = Goldilocks::ONE;
    for (target, value) in assignment[circuit.layout.constant_column + 1..]
        .iter_mut()
        .zip(public_values)
    {
        *target = Goldilocks::from_u64(*value);
    }
    Ok(assignment)
}

fn execute_pi_ccs_prefix(
    package: &LoadedPerApplicationPackage,
    inputs: &PiCcsV1_1PackageInputs,
) -> Result<RawPiCcsAssignment, PackageError> {
    let encoded = package.encode_pi_ccs_v1_1_inputs(inputs)?;
    let circuit = &package.circuit;
    let mut assignment = seed_assignment(circuit, encoded.private_values(), encoded.public_values())?;
    let mut counts = PrefixExecutionCounts::default();
    let mut found_pi_rlc_start = false;

    for scheduled in scheduled_assignments(circuit)? {
        match assignment_side(circuit, scheduled)? {
            BoundarySide::Prefix => {
                if found_pi_rlc_start {
                    return Err(PackageError::Invalid("PiCCS prefix assignment order"));
                }
                match scheduled {
                    ScheduledAssignment::Permutation(invocation) => {
                        circuit.execute_invocation(invocation, &mut assignment)?;
                        counts.permutations += 1;
                    }
                    ScheduledAssignment::Compact(invocation) => {
                        compact::execute_invocation(invocation, &circuit.compact_templates, &mut assignment)?;
                        counts.compact_invocations += 1;
                    }
                    ScheduledAssignment::Batch(batch) => {
                        execute_witness_batch(batch, &mut assignment);
                        counts.witness_batches += 1;
                    }
                    ScheduledAssignment::Generic(instruction) => {
                        let left = eval_sparse_combination(&instruction.a, &assignment);
                        let right = eval_sparse_combination(&instruction.b, &assignment);
                        assignment[instruction.target] = left * right;
                        counts.generic_instructions += 1;
                    }
                }
            }
            BoundarySide::Suffix => {
                if !found_pi_rlc_start {
                    if !is_first_pi_rlc_assignment(scheduled) {
                        return Err(PackageError::Invalid("first PiRLC assignment boundary"));
                    }
                    found_pi_rlc_start = true;
                }
            }
        }
    }
    if !found_pi_rlc_start {
        return Err(PackageError::Invalid("missing PiRLC assignment boundary"));
    }

    Ok(RawPiCcsAssignment {
        private_values: assignment[..PI_CCS_PRIVATE_END]
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect(),
        public_values: encoded.public_values().to_vec(),
        counts,
    })
}

fn inner_raw_package_bytes(sealed_bytes: &[u8]) -> Vec<u8> {
    let sealed: Value = serde_json::from_slice(sealed_bytes).expect("sealed Stage 1 package JSON");
    let raw = sealed
        .as_array()
        .and_then(|fields| fields.get(1))
        .expect("sealed Stage 1 inner package");
    serde_json::to_vec(raw).expect("sealed Stage 1 inner package JSON")
}

#[test]
#[ignore = "exact 19,936,967-row PiCCS prefix gate; run this target explicitly under the 300-second cap"]
fn sealed_package_generates_and_checks_the_current_pi_ccs_prefix() {
    let sealed_bytes = fs::read(artifact_path("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"))
        .expect("sealed canonical Stage 1 package");
    let pi_ccs_bytes = fs::read(artifact_path("nightstream-fprime-stage1-piccs-parity-v1.json"))
        .expect("current PiCCS parity artifact");
    let ownership_bytes = fs::read(artifact_path("nightstream-fprime-stage1-piccs-ownership-v1.json"))
        .expect("Lean PiCCS ownership audit artifact");
    let package = load_poseidon2_hash_chain_v1_package(&sealed_bytes).expect("verifier-owned production package");
    let binding = package
        .production_verifier_binding()
        .expect("fixed production verifier binding");
    let inputs = pi_ccs_inputs(&pi_ccs_bytes, &binding);
    let assignment = execute_pi_ccs_prefix(&package, &inputs).expect("production PiCCS prefix assignment");
    let expected_structural_identity = package.structural_identifier();
    drop(inputs);
    drop(binding);
    drop(package);

    assert_eq!(assignment.private_values.len(), PI_CCS_PRIVATE_END);
    assert_eq!(assignment.public_values.len(), PUBLIC_INPUT_COUNT);
    assert_eq!(
        assignment.counts,
        PrefixExecutionCounts {
            permutations: 32_304,
            compact_invocations: 0,
            witness_batches: 22,
            generic_instructions: 732_393,
        },
    );

    let raw_package = inner_raw_package_bytes(&sealed_bytes);
    let checked_rows = conformance_support::evaluate_pi_ccs_prefix_assignment(
        &raw_package,
        &assignment.private_values,
        &assignment.public_values,
    )
    .expect("independent PiCCS prefix row evaluation");
    assert_eq!(checked_rows, PI_CCS_ROW_END);

    let mut changed = assignment.private_values.clone();
    changed[PI_CCS_FIRST_GENERATED_COLUMN] =
        (changed[PI_CCS_FIRST_GENERATED_COLUMN] + 1) % super::super::GOLDILOCKS_MODULUS;
    assert!(
        conformance_support::evaluate_pi_ccs_prefix_assignment(&raw_package, &changed, &assignment.public_values,)
            .is_err()
    );
    drop(changed);
    drop(raw_package);

    let report = conformance_support::check_piccs_owner_mutations(
        &sealed_bytes,
        &ownership_bytes,
        expected_structural_identity,
        PI_CCS_ROW_START,
        checked_rows,
        &assignment.private_values,
        &assignment.public_values,
    );
    assert_eq!(report.row_families, 12);
    assert_eq!(report.row_mutations, 12);
    assert_eq!(report.column_families, 14);
    assert_eq!(report.zero_column_families, 2);
    assert_eq!(report.column_mutations, 12);
    assert_eq!(report.public_segments, 3);
    assert_eq!(report.public_mutations, 3);

    let relation =
        logical_reference::relation::Relation::decode(&sealed_bytes).expect("independent final relation decoder");
    let artifact =
        logical_reference::source::SourcePackage::decode(&sealed_bytes).expect("independent sealed source decoder");
    assert_eq!(artifact.sealed_schema, 6);
    assert_eq!(artifact.cube_variables, 28);
    assert_eq!(artifact.logical_public_inputs, 270);
    let program = logical_reference::matrix::MatrixProgram::decode(
        &artifact.matrix_program,
        &artifact.sources,
        artifact.logical_columns,
        artifact.logical_rows,
    )
    .expect("independent final matrix-program decoder");
    assert_eq!(
        program.block_ends().take(9).collect::<Vec<_>>(),
        PI_CCS_LOGICAL_BLOCK_ENDS,
        "Lean-authored PiCCS block boundaries and first sampler marker",
    );
    assert_eq!(
        program.block_opcodes().take(9).collect::<Vec<_>>(),
        PI_CCS_LOGICAL_BLOCK_OPCODES,
        "Lean-authored PiCCS block kinds and first sampler marker",
    );
    let logical_assignment = logical_reference::assignment::PartialLogicalAssignment::decode(
        &sealed_bytes,
        &assignment.private_values,
        &assignment.public_values,
    )
    .expect("independent partial logical assignment decoder");
    assert_eq!(logical_assignment.len(), artifact.logical_columns);
    let logical_rows = logical_reference::evaluation::verify_satisfaction_range_with(
        &program,
        &artifact.sources,
        &relation,
        0,
        PI_CCS_LOGICAL_ROW_END,
        |column| logical_assignment.value(column),
    )
    .expect("raw Rust PiCCS assignment satisfies every final Lean logical prefix row");
    assert_eq!(logical_rows, PI_CCS_LOGICAL_ROW_END);
}
