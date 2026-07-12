use neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2HashAudit;

use super::*;

const ARTIFACT_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryRecursiveOutputArtifact.lean";
const SHARD_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveOutputInstructions";
const HASHES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursiveOutputPoseidonHashes.lean";
const SHARD_SIZE: usize = 1_200;

fn output_range<'a>(
    builder: &'a R1csBuilder,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> &'a RowFamilyRange {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| {
            range.name == "fprime.recursive.output"
                && audit.row_start <= range.row_start
                && range.row_end <= audit.row_end
        })
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one recursive output owner");
    matches[0]
}

fn output_hash<'a>(builder: &'a R1csBuilder, range: &RowFamilyRange) -> Poseidon2HashAudit {
    let matches = builder
        .poseidon2_hash_audits()
        .into_iter()
        .filter(|hash| range.row_start <= hash.row_start && hash.row_end <= range.row_end)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one recursive output state-x_out sponge");
    matches[0].clone()
}

fn render_shard(index: usize, instructions: &[Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated recursive-output instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutput.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutput.Generated\n",
        lean_instructions(instructions),
    )
}

fn render_artifact(
    program: &NormalizedProgram,
    range: &RowFamilyRange,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    let shard_count = program.instructions.len().div_ceil(SHARD_SIZE);
    let imports = (0..shard_count)
        .map(|index| {
            format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveOutputInstructions{index}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let instructions = (0..shard_count)
        .map(|index| format!("Generated.instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{imports}\n\n\
         /-! Exact checked program for the recursive output owner in the two-step full-history profile. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutput\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         def inputColumns : List Nat := {}\n\
         def stateOutColumns : List Nat := {}\n\
         def xOutColumns : List Nat := {}\n\
         def xOutBitColumns : List Nat := {}\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\n\
         def instructions : List Instruction :=\n    {instructions}\n\n\
         def rows : List Row := CheckedProgram.rows instructions\n\n\
         theorem instructions_length : instructions.length = rowCount := by native_decide\n\
         theorem rows_length : rows.length = rowCount := by native_decide\n\
         theorem definitions_length : (definitions instructions).length = definitionCount := by native_decide\n\
         theorem checks_length : (checks instructions).length = checkCount := by native_decide\n\
         theorem definitions_canonical :\n\
             ∀ definition ∈ definitions instructions, definition.Canonical := by native_decide\n\
         theorem definitions_wellFormed :\n\
             WellFormed inputColumns (definitions instructions) := by native_decide\n\
         theorem checks_reference :\n\
             ChecksReference (knownAfter inputColumns (definitions instructions)) instructions := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutput\n",
        lean_nat_list(program.input_columns.iter().copied()),
        lean_nat_list(audit.state_out_columns.iter().copied()),
        lean_nat_list(audit.x_out_columns),
        lean_nat_list(audit.x_out_bit_columns.iter().copied()),
        range.row_start,
        range.row_end,
        range.row_end - range.row_start,
        program.definition_count,
        program.check_count,
    )
}

fn render_hashes(builder: &R1csBuilder, range: &RowFamilyRange, hash: &Poseidon2HashAudit) -> String {
    let calls = builder.poseidon2_permutation_audits();
    let rounds = hash
        .rounds
        .iter()
        .map(|round| {
            let call = calls
                .iter()
                .find(|call| {
                    call.input_cols == round.permutation_input_cols && call.output_cols == round.permutation_output_cols
                })
                .expect("recursive output sponge permutation call");
            let kind = match &round.kind {
                Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
                    format!(".absorb {}", lean_nat_list(chunk_cols.iter().copied()))
                }
                Poseidon2HashRoundAuditKind::Pad => ".pad".to_string(),
            };
            format!(
                "{{ kind := {kind}, stateBeforeColumns := {}, permutationInputColumns := {}, \
                 permutationOutputColumns := {}, definingRows := {}, call := {{ rowStart := {}, \
                 rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }} }}",
                lean_nat_list(round.state_before_cols),
                lean_nat_list(round.permutation_input_cols),
                lean_nat_list(round.permutation_output_cols),
                lean_nat_list(round.defining_rows.iter().map(|row| row - range.row_start)),
                call.row_start - range.row_start,
                call.row_end - range.row_start,
                lean_nat_list(call.input_cols),
                call.first_allocated_col,
            )
        })
        .collect::<Vec<_>>()
        .join("\n    , ");
    let trace = format!(
        "{{ inputColumns := {}, zeroColumn := {}, zeroRow := {}, rounds := [\n      {}\n    ], \
         outputColumns := {} }}",
        lean_nat_list(hash.input_cols.iter().copied()),
        hash.zero_col,
        hash.zero_row - range.row_start,
        rounds,
        lean_nat_list(hash.output_cols),
    );
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRecursiveOutputArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated sponge certificate for the exact recursive output owner. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputPoseidonHashes\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 524288\n\n\
         def xOutTrace : Trace :=\n  {trace}\n\n\
         theorem xOutTrace_valid :\n\
             xOutTrace.Valid FPrimeFullHistoryRecursiveOutput.rows := by native_decide\n\n\
         theorem xOutTrace_output :\n\
             xOutTrace.outputColumns = FPrimeFullHistoryRecursiveOutput.xOutColumns := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputPoseidonHashes\n"
    )
}

pub fn compare_recursive_output_artifacts(
    builder: &R1csBuilder,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) {
    let range = output_range(builder, audit);
    let hash = output_hash(builder, range);
    let first_allocated_column = hash.input_cols[0];
    let program = normalize_range(builder, range.row_start, range.row_end, first_allocated_column);
    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write recursive-output artifact");
            drifted.push(expected);
        }
    };
    compare(root.join(ARTIFACT_PATH), render_artifact(&program, range, audit));
    for (index, shard) in program.instructions.chunks(SHARD_SIZE).enumerate() {
        compare(
            root.join(format!("{SHARD_PREFIX}{index}.lean")),
            render_shard(index, shard),
        );
    }
    compare(root.join(HASHES_PATH), render_hashes(builder, range, &hash));
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history recursive-output artifacts drifted: {drifted:?}"
    );
}
