#[path = "../gadgets/checked_program_artifact_support.rs"]
#[allow(dead_code)]
mod checked_program_artifact_support;
#[path = "full_history_affine_artifact_support.rs"]
mod full_history_affine_artifact_support;
#[path = "full_history_counter_artifact_support.rs"]
mod full_history_counter_artifact_support;
#[path = "full_history_current_terminal_diagnostic_support.rs"]
mod full_history_current_terminal_diagnostic_support;
#[path = "full_history_encoding_artifact_support.rs"]
mod full_history_encoding_artifact_support;
#[path = "full_history_equality_artifact_support.rs"]
mod full_history_equality_artifact_support;
#[path = "full_history_manifest_identity_support.rs"]
mod full_history_manifest_identity_support;
#[path = "full_history_nested_manifest_support.rs"]
mod full_history_nested_manifest_support;
#[path = "full_history_pi_dec_artifact_support.rs"]
mod full_history_pi_dec_artifact_support;
#[path = "full_history_projection_role_support.rs"]
mod full_history_projection_role_support;
#[path = "full_history_public_pins_artifact_support.rs"]
mod full_history_public_pins_artifact_support;
#[path = "full_history_recursive_output_artifact_support.rs"]
mod full_history_recursive_output_artifact_support;
#[path = "full_history_seeded_phi81_artifact_support.rs"]
mod full_history_seeded_phi81_artifact_support;
#[path = "full_history_source_hash_support.rs"]
mod full_history_source_hash_support;
#[path = "full_history_terminal_accumulator_artifact_support.rs"]
mod full_history_terminal_accumulator_artifact_support;
#[path = "full_history_terminal_ce_artifact_support.rs"]
mod full_history_terminal_ce_artifact_support;
#[path = "full_history_terminal_continuity_aggregate_support.rs"]
mod full_history_terminal_continuity_aggregate_support;
#[path = "full_history_transcript_artifact_support.rs"]
mod full_history_transcript_artifact_support;
#[path = "../gadgets/lean_artifact_support.rs"]
#[allow(dead_code)]
mod lean_artifact_support;

use checked_program_artifact_support::{
    canonicalize_program, lean_instructions, normalize_prefix, normalize_range, relabel_instructions,
    CanonicalizedProgram, Instruction, NormalizedProgram, Rhs,
};
use full_history_affine_artifact_support::{compare_affine_artifacts, compare_current_terminal_link_artifact};
use full_history_counter_artifact_support::compare_counter_artifact;
use full_history_current_terminal_diagnostic_support::compare_current_terminal_diagnostic;
use full_history_encoding_artifact_support::render_output_encoding_artifact;
use full_history_equality_artifact_support::render_equality_artifact;
use full_history_manifest_identity_support::{
    assert_partition, range_hash, range_hash as full_history_range_hash, range_json,
    range_json as full_history_range_json, top_ranges,
};
use full_history_nested_manifest_support::{nested_full_history_manifest, render_nested_lean_definitions};
use full_history_pi_dec_artifact_support::compare_pi_dec_artifacts;
use full_history_projection_role_support::compare_projection_role_artifact;
use full_history_public_pins_artifact_support::compare_public_pins_artifacts;
use full_history_recursive_output_artifact_support::compare_recursive_output_artifacts;
use full_history_seeded_phi81_artifact_support::compare_seeded_phi81_artifact;
use full_history_source_hash_support::full_history_source_hashes;
use full_history_terminal_accumulator_artifact_support::compare_terminal_accumulator_artifacts;
use full_history_terminal_ce_artifact_support::compare_terminal_ce_profiles;
use full_history_terminal_continuity_aggregate_support::compare_terminal_continuity_artifacts;
use full_history_transcript_artifact_support::compare_transcript_artifacts;
use lean_artifact_support::lean_nat_list;
use neo_fold_clean::engine::r1cs_circuit::builder::{
    Poseidon2HashRoundAuditKind, Poseidon2PermutationAudit, ProjectionIdentityAudit, ProjectionLadderAudit,
};
use neo_fold_clean::paper::f_prime::r1cs::F_PRIME_ENC_INST_BITS;

use super::*;

const BASE_PROGRAM_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryBaseArtifact.lean";
const BASE_SHARD_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryBaseInstructions";
const BASE_SHARD_SIZE: usize = 1_200;
const BASE_HASHES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryBasePoseidonHashes.lean";
const PRIOR_LINK_PROGRAM_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryPriorLinkArtifact.lean";
const PRIOR_LINK_SHARD_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPriorLinkInstructions";
const PRIOR_LINK_HASHES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryPriorLinkPoseidonHashes.lean";
const FULL_HISTORY_STATE_LINK_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStateLinkArtifact.lean";
const FULL_HISTORY_TERMINAL_LINK_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalLinkArtifact.lean";
const FULL_HISTORY_RECURSIVE_POINT_BINDING_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRecursivePointBindingArtifact.lean";
const FULL_HISTORY_TERMINAL_POINT_BINDING_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryTerminalPointBindingArtifact.lean";
const FULL_HISTORY_OUTPUT_ENCODING_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryOutputEncodingArtifact.lean";
const FULL_HISTORY_PROJECTION_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryProjectionArtifact.lean";
const FULL_HISTORY_PROJECTION_TEMPLATES_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeFullHistory/FPrimeFullHistoryProjectionTemplates.lean";
const FULL_HISTORY_PROJECTION_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryProjectionInstructions";
const FULL_HISTORY_PROJECTION_CENSUS_PREFIX: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryProjectionCensus";
const PROJECTION_TEMPLATE_SHARD_SIZE: usize = 1_200;
const PROJECTION_CENSUS_SHARD_SIZE: usize = 4;
const STAGE_ALL_ARTIFACTS: bool = false;

fn formal_repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn full_history_manifest(builder: &R1csBuilder) -> Value {
    let top = top_ranges(builder, FULL_HISTORY_TOP_LEVEL);
    let nested = nested_full_history_manifest(builder);
    assert_partition(builder, &top);
    let total = builder
        .row_family_ranges()
        .iter()
        .find(|range| range.name == "decider.full_history")
        .expect("full-history total marker");
    assert_eq!((total.row_start, total.row_end), (0, builder.rows()));
    json!({
        "schema": 2,
        "artifact_kind": "r1cs/f-prime-full-history-program-manifest",
        "profile": {
            "layout": "plain",
            "semantic_mode": "stateless",
            "carrier_relation": "minimal-supported-bit-carrier",
            "batch_schedule": [1, 1],
            "recursive_steps": 1,
            "terminal_fold": true,
            "terminal_ce": "direct",
        },
        "source_hashes": full_history_source_hashes(),
        "total": range_json(builder, total),
        "top_level_families": top
            .iter()
            .map(|range| range_json(builder, range))
            .collect::<Vec<_>>(),
        "recursive_families": nested["recursive_families"].clone(),
        "recursive_nifs_families": nested["recursive_nifs_families"].clone(),
        "recursive_pi_ccs_families": nested["recursive_pi_ccs_families"].clone(),
        "recursive_pi_rlc_families": nested["recursive_pi_rlc_families"].clone(),
        "terminal_families": nested["terminal_families"].clone(),
        "terminal_nifs_families": nested["terminal_nifs_families"].clone(),
        "terminal_pi_ccs_families": nested["terminal_pi_ccs_families"].clone(),
        "terminal_pi_rlc_families": nested["terminal_pi_rlc_families"].clone(),
        "full_builder_rows": builder.rows(),
        "full_builder_columns": builder.cols(),
    })
}

fn full_history_lean_string(value: &str) -> String {
    serde_json::to_string(value).expect("Lean-compatible string")
}

fn render_full_history_lean(manifest: &Value) -> String {
    let total = &manifest["total"];
    let ranges = manifest["top_level_families"]
        .as_array()
        .expect("top ranges");
    let mut rendered = String::new();
    rendered
        .push_str("import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema\n\n");
    rendered.push_str("/-! Generated by `system_decider_r1cs`; do not hand-edit. -/\n\n");
    rendered.push_str("namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest\n\n");
    writeln!(rendered, "def schemaVersion : Nat := {}", manifest["schema"]).expect("render");
    writeln!(
        rendered,
        "def totalRows : Nat := {}\ndef totalColumns : Nat := {}",
        manifest["full_builder_rows"], manifest["full_builder_columns"]
    )
    .expect("render");
    writeln!(
        rendered,
        "def totalSha256 : String := {}",
        full_history_lean_string(total["sha256"].as_str().expect("total hash"))
    )
    .expect("render");
    rendered.push_str("def topLevelFamilies : List FPrimeRecursiveManifest.RowRange :=\n");
    for (index, range) in ranges.iter().enumerate() {
        let prefix = if index == 0 { "  [" } else { "  ," };
        writeln!(
            rendered,
            "{prefix} {{ name := {}, rowStart := {}, rowEnd := {}, nonzeroEntries := {}, sha256 := {} }}",
            full_history_lean_string(range["name"].as_str().expect("range name")),
            range["row_start"],
            range["row_end"],
            range["nonzero_entries"],
            full_history_lean_string(range["sha256"].as_str().expect("range hash")),
        )
        .expect("render");
    }
    rendered.push_str("  ]\n\n");
    rendered.push_str(&render_nested_lean_definitions(manifest));
    rendered.push_str("\nend Nightstream.Implementation.R1CS.FPrimeFullHistoryManifest\n");
    rendered
}

fn render_base_shard(index: usize, instructions: &[checked_program_artifact_support::Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated full-history base-owner instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBase.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryBase.Generated\n",
        lean_instructions(instructions),
    )
}

fn render_prior_link_shard(index: usize, instructions: &[checked_program_artifact_support::Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated full-history recursive prior-link instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLink.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLink.Generated\n",
        lean_instructions(instructions),
    )
}

fn prior_encoding_column_map(
    builder: &R1csBuilder,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> Vec<usize> {
    let digest = audit
        .prior_link_digest_columns
        .expect("recursive step prior-link digest columns");
    assert_eq!(audit.prior_link_bit_columns.len(), F_PRIME_ENC_INST_BITS);
    let decompositions = builder.canonical_u64_audits();
    let mut map = vec![0; 525];
    map[0] = 0;
    for lane in 0..4 {
        map[1 + lane] = digest[lane];
        let public_start = 5 + lane * 64;
        map[public_start..public_start + 64].copy_from_slice(&audit.prior_link_bit_columns[lane * 64..(lane + 1) * 64]);
        let decomposition = decompositions
            .iter()
            .find(|candidate| candidate.field_col == digest[lane])
            .unwrap_or_else(|| panic!("canonical-u64 audit for prior-link digest lane {lane}"));
        let canonical_start = 261 + lane * 66;
        map[canonical_start..canonical_start + 64].copy_from_slice(&decomposition.bit_cols);
        map[canonical_start + 64] = decomposition.bit_cols[63] + 1;
        map[canonical_start + 65] = decomposition.bit_cols[63] + 2;
    }
    map
}

fn render_prior_link_program(
    builder: &R1csBuilder,
    program: &NormalizedProgram,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    let shard_count = program.instructions.len().div_ceil(BASE_SHARD_SIZE);
    let imports = (0..shard_count)
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPriorLinkInstructions{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let instructions = (0..shard_count)
        .map(|index| format!("Generated.instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    let (row_start, row_end) = audit
        .prior_link_row_range
        .expect("recursive step prior-link row range");
    let digest = audit
        .prior_link_digest_columns
        .expect("recursive step prior-link digest columns");
    assert_eq!(audit.prior_fresh_public_columns.len(), 1);
    let fresh = &audit.prior_fresh_public_columns[0];
    assert_eq!(fresh.len(), 1 + F_PRIME_ENC_INST_BITS);
    let fresh_bit_pairs = fresh[1..]
        .iter()
        .zip(&audit.prior_link_bit_columns)
        .map(|(&left, &right)| format!("({left}, {right})"))
        .collect::<Vec<_>>()
        .join(", ");
    let constant_pins = program
        .instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::Define(definition) => match &definition.rhs {
                Rhs::Linear(terms) if terms.iter().all(|term| term.0 == 0) => {
                    let value = terms.iter().fold(0_u128, |sum, term| {
                        (sum + u128::from(term.1)) % u128::from(F::ORDER_U64)
                    });
                    Some(format!("({}, {value})", definition.output))
                }
                _ => None,
            },
            Instruction::Check(_) => None,
        })
        .collect::<Vec<_>>()
        .join(", ");
    let selected_u64_fields = [
        audit.state_in_columns[8],
        audit.state_in_columns[9],
        audit.state_in_columns[18],
    ];
    let canonical_u64_maps = selected_u64_fields
        .iter()
        .map(|field| {
            let decomposition = builder
                .canonical_u64_audits()
                .into_iter()
                .find(|decomposition| decomposition.field_col == *field)
                .unwrap_or_else(|| panic!("canonical-u64 decomposition for prior state field column {field}"));
            let first_aux = decomposition.bit_cols[63] + 1;
            lean_nat_list(
                std::iter::once(0)
                    .chain(std::iter::once(decomposition.field_col))
                    .chain(decomposition.bit_cols)
                    .chain([first_aux, first_aux + 1]),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let prior_x_out_hash = builder
        .poseidon2_hash_audits()
        .into_iter()
        .find(|hash| row_start <= hash.row_start && hash.row_end <= row_end)
        .expect("prior-link state-x_out hash audit");
    let half_outputs = &prior_x_out_hash.input_cols[9..15];
    let canonical_u64_half_definitions = selected_u64_fields
        .iter()
        .zip(half_outputs.chunks_exact(2))
        .flat_map(|(field, outputs)| {
            let decomposition = builder
                .canonical_u64_audits()
                .into_iter()
                .find(|decomposition| decomposition.field_col == *field)
                .unwrap_or_else(|| panic!("canonical-u64 decomposition for prior state field column {field}"));
            outputs.iter().enumerate().map(move |(half, output)| {
                let terms = decomposition.bit_cols[half * 32..(half + 1) * 32]
                    .iter()
                    .enumerate()
                    .map(|(bit, column)| format!("({column}, {})", 1_u64 << bit))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("⟨{output}, .linear [{terms}]⟩")
            })
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{imports}\n\n\
         /-! Exact checked program for the recursive consumer of the base step's delayed public link. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLink\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         def inputColumns : List Nat := {}\n\
         def stateInColumns : List Nat := {}\n\
         def digestColumns : List Nat := {}\n\
         def encodedBitColumns : List Nat := {}\n\
         def freshPublicColumns : List Nat := {}\n\
         def encodingColumnMap : List Nat := {}\n\
         def constantPins : List (Nat × Nat) := [{constant_pins}]\n\
         def canonicalU64Maps : List (List Nat) := [{canonical_u64_maps}]\n\
         def canonicalU64HalfDefinitions : List Definition := [{canonical_u64_half_definitions}]\n\
         def freshOnePin : Nat × Nat := ({}, 1)\n\
         def freshBitPairs : List (Nat × Nat) := [{fresh_bit_pairs}]\n\
         def rowStart : Nat := {row_start}\n\
         def rowEnd : Nat := {row_end}\n\
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
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLink\n",
        lean_nat_list(program.input_columns.iter().copied()),
        lean_nat_list(audit.state_in_columns.iter().copied()),
        lean_nat_list(digest),
        lean_nat_list(audit.prior_link_bit_columns.iter().copied()),
        lean_nat_list(fresh.iter().copied()),
        lean_nat_list(prior_encoding_column_map(builder, audit)),
        fresh[0],
        row_end - row_start,
        program.definition_count,
        program.check_count,
    )
}

fn render_full_history_state_link(
    builder: &R1csBuilder,
    base: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
    recursive: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    assert_eq!(base.state_out_columns.len(), recursive.state_in_columns.len());
    let pairs = base
        .state_out_columns
        .iter()
        .zip(&recursive.state_in_columns)
        .map(|(&left, &right)| (left, right))
        .collect::<Vec<_>>();
    let range = builder
        .row_family_ranges()
        .iter()
        .find(|range| range.name == "decider.state_link")
        .expect("one full-history state-link owner");
    assert_eq!(range.row_end - range.row_start, pairs.len());
    let normalized = normalize_range(builder, range.row_start, range.row_end, builder.cols());
    let expected = pairs
        .iter()
        .map(|&(left, right)| {
            Instruction::Check(checked_program_artifact_support::Row {
                a: vec![(left, 1), (right, F::ORDER_U64 - 1)],
                b: vec![(0, 1)],
                c: Vec::new(),
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        normalized.instructions, expected,
        "full-history state-link rows differ from exact equality rows"
    );
    let lean_pairs = pairs
        .iter()
        .map(|&(left, right)| format!("({left}, {right})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "import Nightstream.Implementation.R1CS.Core.EqualityPins\n\n\
         /-! Generated exact adjacent-state equality rows for the two-step full-history profile. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLink\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def pairs : List (Nat × Nat) := [{lean_pairs}]\n\
         def rows : List Row := EqualityPins.rows pairs\n\
         def rowStart : Nat := {}\n\
         def rowEnd : Nat := {}\n\
         def rowCount : Nat := {}\n\n\
         theorem rows_length : rows.length = rowCount := by decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLink\n",
        range.row_start,
        range.row_end,
        pairs.len(),
    )
}

struct ProjectionGroup<'a> {
    name: &'static str,
    ladder: &'a ProjectionLadderAudit,
    identities: Vec<&'a ProjectionIdentityAudit>,
    shared: CanonicalizedProgram,
    identity_template: Vec<Instruction>,
    identity_maps: Vec<Vec<usize>>,
}

fn full_history_projection_groups(builder: &R1csBuilder) -> Vec<ProjectionGroup<'_>> {
    let mut ladders = builder
        .projection_ladder_audits()
        .iter()
        .collect::<Vec<_>>();
    ladders.sort_by_key(|audit| audit.row_start);
    assert_eq!(ladders.len(), 2, "recursive and terminal projection ladders");
    let names = ["Recursive", "Terminal"];
    ladders
        .into_iter()
        .zip(names)
        .map(|(ladder, name)| {
            let mut identities = builder
                .projection_identity_audits()
                .iter()
                .filter(|identity| identity.power_columns == ladder.power_columns)
                .collect::<Vec<_>>();
            identities.sort_by_key(|identity| identity.row_start);
            assert_eq!(identities.len(), 31, "{name} projection identity census");
            let shared_end = identities[0].row_start;
            let shared_program = normalize_range(builder, ladder.row_start, shared_end, ladder.power_columns[0][0]);
            let shared = canonicalize_program(&shared_program);
            assert_eq!(
                relabel_instructions(&shared.instructions, &shared.column_map),
                shared_program.instructions,
                "{name} shared projection canonicalization must be lossless"
            );
            let mut identity_template = None;
            let mut identity_maps = Vec::with_capacity(identities.len());
            for identity in &identities {
                let input_width = identity.input_columns[0].len();
                let first_allocated_column = identity.input_evaluation_outputs[0][0] - 2 * (input_width - 1);
                let program = normalize_range(builder, identity.row_start, identity.row_end, first_allocated_column);
                let canonical = canonicalize_program(&program);
                assert_eq!(
                    relabel_instructions(&canonical.instructions, &canonical.column_map),
                    program.instructions,
                    "{name} identity canonicalization must be lossless"
                );
                match &identity_template {
                    None => identity_template = Some(canonical.instructions.clone()),
                    Some(template) => assert_eq!(
                        &canonical.instructions, template,
                        "{name} identities must share one exact normalized row shape"
                    ),
                }
                identity_maps.push(canonical.column_map);
            }
            ProjectionGroup {
                name,
                ladder,
                identities,
                shared,
                identity_template: identity_template.expect("nonempty projection identities"),
                identity_maps,
            }
        })
        .collect()
}

fn lean_k_columns(columns: [usize; 2]) -> String {
    format!("⟨{}, {}⟩", columns[0], columns[1])
}

fn lean_k_columns_list(columns: &[[usize; 2]]) -> String {
    format!(
        "[{}]",
        columns
            .iter()
            .copied()
            .map(lean_k_columns)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_eval_trace(columns: &[usize], powers: &[[usize; 2]], output: [usize; 2]) -> String {
    format!(
        "EvalTrace.ofColumns {} {} {}",
        lean_nat_list(columns.iter().copied()),
        lean_k_columns_list(powers),
        lean_k_columns(output),
    )
}

fn lean_projection_trace(group: &ProjectionGroup<'_>, identity: &ProjectionIdentityAudit) -> String {
    let pairs = identity
        .rho_columns
        .iter()
        .zip(&identity.rho_evaluation_outputs)
        .zip(&identity.input_columns)
        .zip(&identity.input_evaluation_outputs)
        .zip(&identity.pair_product_outputs)
        .map(
            |((((rho_columns, &rho_output), input_columns), &input_output), &product_output)| {
                format!(
                    "PairTrace.ofColumns {} {} {} {} {} {}",
                    lean_k_columns_list(&identity.power_columns),
                    lean_nat_list(rho_columns.iter().copied()),
                    lean_nat_list(input_columns.iter().copied()),
                    lean_k_columns(rho_output),
                    lean_k_columns(input_output),
                    lean_k_columns(product_output),
                )
            },
        )
        .collect::<Vec<_>>()
        .join(",\n       ");
    format!(
        "⟨{}Ladder,\n     [{}],\n     {}, {},\n     {},\n     {},\n     KMulTrace.quotientPhi {} {} {},\n     106⟩",
        group.name.to_lowercase(),
        pairs,
        lean_nat_list(identity.output_columns.iter().copied()),
        lean_nat_list(identity.quotient_columns.iter().copied()),
        lean_eval_trace(
            &identity.output_columns,
            &identity.power_columns,
            identity.output_evaluation,
        ),
        lean_eval_trace(
            &identity.quotient_columns,
            &identity.power_columns,
            identity.quotient_evaluation,
        ),
        lean_k_columns(identity.quotient_evaluation),
        lean_k_columns_list(&identity.power_columns),
        lean_k_columns(identity.quotient_phi_product),
    )
}

fn projection_template_parts<'a, 'b>(groups: &'a [ProjectionGroup<'b>]) -> Vec<(&'static str, &'a [Instruction])> {
    groups
        .iter()
        .flat_map(|group| {
            [
                (
                    if group.name == "Recursive" {
                        "RecursiveShared"
                    } else {
                        "TerminalShared"
                    },
                    group.shared.instructions.as_slice(),
                ),
                (
                    if group.name == "Recursive" {
                        "RecursiveIdentity"
                    } else {
                        "TerminalIdentity"
                    },
                    group.identity_template.as_slice(),
                ),
            ]
        })
        .collect()
}

fn render_projection_instruction_shard(kind: &str, index: usize, instructions: &[Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated full-history {kind} projection instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.Generated{kind}\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.Generated{kind}\n",
        lean_instructions(instructions),
    )
}

fn render_projection_templates(groups: &[ProjectionGroup<'_>]) -> String {
    let parts = projection_template_parts(groups);
    let imports = parts
        .iter()
        .flat_map(|(kind, instructions)| {
            (0..instructions.len().div_ceil(PROJECTION_TEMPLATE_SHARD_SIZE)).map(move |index| {
                format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryProjectionInstructions{kind}{index}")
            })
        })
        .collect::<Vec<_>>()
        .join("\n");
    let template = |kind: &str, instructions: &[Instruction]| {
        let shards = (0..instructions.len().div_ceil(PROJECTION_TEMPLATE_SHARD_SIZE))
            .map(|index| format!("Generated{kind}.instructions{index}"))
            .collect::<Vec<_>>()
            .join(" ++\n    ");
        format!("def {kind}Instructions : List Instruction :=\n    {shards}\n")
    };
    let ladder = |group: &ProjectionGroup<'_>| {
        format!(
            "def {}Ladder : LadderTrace :=\n  LadderTrace.ofColumns {} {}\n",
            group.name.to_lowercase(),
            lean_k_columns(group.ladder.beta_columns),
            lean_k_columns_list(&group.ladder.power_columns),
        )
    };
    let shared_map = |group: &ProjectionGroup<'_>| {
        format!(
            "def {}SharedMap : List Nat := {}\n",
            group.name.to_lowercase(),
            lean_nat_list(group.shared.column_map.iter().copied()),
        )
    };
    format!(
        "{imports}\n\n\
         import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionBatchSound\n\
         import Nightstream.Implementation.R1CS.Core.Relabel\n\n\
         /-! Shared exact templates for the full-history PiRLC projection census. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\
         open Nightstream.Implementation.R1CS.ProjectionProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         {}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\
         def traceRows (trace : ProjectionTrace) : List Row :=\n\
           trace.definitions.map Program.Definition.builderRow ++ trace.checks\n\n\
         def mappedRows (instructions : List Instruction) (columnMap : List Nat) : List Row :=\n\
           (CheckedProgram.rows instructions).map (Relabel.row columnMap)\n\n\
         def expectedRows (shared identity : List Instruction) (sharedMap : List Nat)\n\
             (identityMaps : List (List Nat)) : List (List Row) :=\n\
           identityMaps.map fun identityMap =>\n\
             mappedRows shared sharedMap ++ mappedRows identity identityMap\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection\n",
        template("RecursiveShared", &groups[0].shared.instructions),
        template("RecursiveIdentity", &groups[0].identity_template),
        template("TerminalShared", &groups[1].shared.instructions),
        template("TerminalIdentity", &groups[1].identity_template),
        ladder(&groups[0]),
        ladder(&groups[1]),
        shared_map(&groups[0]),
        shared_map(&groups[1]),
    )
}

fn render_projection_census_shard(
    group: &ProjectionGroup<'_>,
    index: usize,
    identities: &[&ProjectionIdentityAudit],
    maps: &[Vec<usize>],
) -> String {
    let lower = group.name.to_lowercase();
    let rendered_maps = maps
        .iter()
        .map(|map| lean_nat_list(map.iter().copied()))
        .collect::<Vec<_>>()
        .join(",\n   ");
    let rendered_traces = identities
        .iter()
        .map(|identity| lean_projection_trace(group, identity))
        .collect::<Vec<_>>()
        .join(",\n   ");
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryProjectionTemplates\n\n\
         /-! Generated {lower} projection census shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.Generated{}Census\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.ProjectionProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         def identityMaps{index} : List (List Nat) :=\n  [{rendered_maps}]\n\n\
         def traces{index} : List ProjectionTrace :=\n  [{rendered_traces}]\n\n\
         def expectedRows{index} : List (List Row) :=\n\
           expectedRows {}SharedInstructions {}IdentityInstructions\n\
             {}SharedMap identityMaps{index}\n\n\
         theorem trace_count{index} : traces{index}.length = {} := by native_decide\n\n\
         theorem trace_layouts{index} :\n\
             ∀ trace ∈ traces{index}, trace.LayoutValid := by native_decide\n\n\
         theorem trace_pairs_nonempty{index} :\n\
             ∀ trace ∈ traces{index}, trace.pairs ≠ [] := by native_decide\n\n\
         theorem trace_pair_widths{index} :\n\
             ∀ trace ∈ traces{index}, ∀ pair ∈ trace.pairs,\n\
               pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by native_decide\n\n\
         theorem definitions_canonical{index} :\n\
             ∀ trace ∈ traces{index}, ∀ definition ∈ trace.definitions,\n\
               definition.Canonical := by native_decide\n\n\
         theorem rows_exact{index} :\n\
             traces{index}.map traceRows = expectedRows{index} := by native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.Generated{}Census\n",
        group.name,
        group.name,
        group.name,
        lower,
        identities.len(),
        group.name,
    )
}

fn projection_census_shard_count(group: &ProjectionGroup<'_>) -> usize {
    group
        .identities
        .len()
        .div_ceil(projection_census_shard_size(group))
}

fn projection_census_shard_size(group: &ProjectionGroup<'_>) -> usize {
    if group.name == "Terminal" {
        1
    } else {
        PROJECTION_CENSUS_SHARD_SIZE
    }
}

fn render_full_history_projection(groups: &[ProjectionGroup<'_>]) -> String {
    let imports = groups
        .iter()
        .flat_map(|group| {
            (0..projection_census_shard_count(group)).map(move |index| {
                format!(
                    "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryProjectionCensus{}{index}",
                    group.name,
                )
            })
        })
        .collect::<Vec<_>>()
        .join("\n");
    let group_traces = |group: &ProjectionGroup<'_>| {
        let refs = (0..projection_census_shard_count(group))
            .map(|index| format!("Generated{}Census.traces{index}", group.name))
            .collect::<Vec<_>>()
            .join(" ++\n    ");
        format!(
            "def {}Traces : List ProjectionTrace :=\n    {refs}\n",
            group.name.to_lowercase()
        )
    };
    let shards = groups
        .iter()
        .flat_map(|group| (0..projection_census_shard_count(group)).map(move |index| (group.name, index)))
        .collect::<Vec<_>>();
    let group_proof = |group: &ProjectionGroup<'_>, theorem: &str| {
        let mut proofs = (0..projection_census_shard_count(group))
            .map(|index| format!("Generated{}Census.{theorem}{index}", group.name));
        let first = proofs.next().expect("nonempty projection census");
        proofs.fold(first, |left, right| {
            format!("forall_append\n      ({left})\n      ({right})")
        })
    };
    let count_rewrites = shards
        .iter()
        .map(|(name, index)| format!("Generated{name}Census.trace_count{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{imports}\n\
         import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryProjectionRoles\n\n\
         /-! Complete exact PiRLC projection census for the two-step full-history profile. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.ProjectionProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         {}\n{}\n\
         def traces : List ProjectionTrace := recursiveTraces ++ terminalTraces\n\n\
         def Holds (assignment : Nat → Nat) : Prop :=\n\
           ∀ trace ∈ traces, Satisfies (traceRows trace) assignment\n\n\
         private theorem forall_append {{α : Type}} {{P : α → Prop}}\n\
             {{left right : List α}}\n\
             (leftProof : ∀ value ∈ left, P value)\n\
             (rightProof : ∀ value ∈ right, P value) :\n\
             ∀ value ∈ left ++ right, P value := by\n\
           intro value member\n\
           rcases List.mem_append.mp member with member | member\n\
           · exact leftProof value member\n\
           · exact rightProof value member\n\n\
         theorem trace_count : traces.length = 62 := by\n\
           simp only [traces, recursiveTraces, terminalTraces, List.length_append, {count_rewrites}]\n\n\
         theorem trace_layouts : ∀ trace ∈ traces, trace.LayoutValid := by\n\
           unfold traces recursiveTraces terminalTraces\n\
           exact forall_append ({}) ({})\n\n\
         theorem trace_pairs_nonempty : ∀ trace ∈ traces, trace.pairs ≠ [] := by\n\
           unfold traces recursiveTraces terminalTraces\n\
           exact forall_append ({}) ({})\n\n\
         theorem trace_pair_widths : ∀ trace ∈ traces, ∀ pair ∈ trace.pairs,\n\
             pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by\n\
           unfold traces recursiveTraces terminalTraces\n\
           exact forall_append ({}) ({})\n\n\
         theorem definitions_canonical : ∀ trace ∈ traces,\n\
             ∀ definition ∈ trace.definitions, definition.Canonical := by\n\
           unfold traces recursiveTraces terminalTraces\n\
           exact forall_append ({}) ({})\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection\n",
        group_traces(&groups[0]),
        group_traces(&groups[1]),
        group_proof(&groups[0], "trace_layouts"),
        group_proof(&groups[1], "trace_layouts"),
        group_proof(&groups[0], "trace_pairs_nonempty"),
        group_proof(&groups[1], "trace_pairs_nonempty"),
        group_proof(&groups[0], "trace_pair_widths"),
        group_proof(&groups[1], "trace_pair_widths"),
        group_proof(&groups[0], "definitions_canonical"),
        group_proof(&groups[1], "definitions_canonical"),
    )
}

fn render_base_program(
    builder: &R1csBuilder,
    program: &NormalizedProgram,
    audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit,
) -> String {
    let shard_count = program.instructions.len().div_ceil(BASE_SHARD_SIZE);
    let imports = (0..shard_count)
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryBaseInstructions{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let instructions = (0..shard_count)
        .map(|index| format!("Generated.instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    let state_in_pins = audit
        .state_in_columns
        .iter()
        .map(|&column| {
            let value = builder.witness()[column].as_canonical_u64();
            format!("({column}, {value})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let constant_pins = program
        .instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::Define(definition) => match &definition.rhs {
                Rhs::Linear(terms) if terms.iter().all(|term| term.0 == 0) => {
                    let value = terms.iter().fold(0_u128, |sum, term| {
                        (sum + u128::from(term.1)) % u128::from(F::ORDER_U64)
                    });
                    Some(format!("({}, {value})", definition.output))
                }
                _ => None,
            },
            Instruction::Check(_) => None,
        })
        .collect::<Vec<_>>()
        .join(", ");
    let selected_u64_fields = [
        audit.state_out_columns[8],
        audit.state_out_columns[9],
        audit.state_out_columns[18],
    ];
    let canonical_u64_maps = selected_u64_fields
        .iter()
        .map(|field| {
            let decomposition = builder
                .canonical_u64_audits()
                .into_iter()
                .find(|decomposition| decomposition.field_col == *field)
                .unwrap_or_else(|| panic!("canonical-u64 decomposition for field column {field}"));
            let first_aux = decomposition.bit_cols[63] + 1;
            let map = std::iter::once(0)
                .chain(std::iter::once(decomposition.field_col))
                .chain(decomposition.bit_cols)
                .chain([first_aux, first_aux + 1]);
            lean_nat_list(map)
        })
        .collect::<Vec<_>>()
        .join(", ");
    let hash_audits = builder.poseidon2_hash_audits();
    let x_out_hash = &hash_audits[2];
    let half_outputs = &x_out_hash.input_cols[9..15];
    let canonical_u64_half_definitions = selected_u64_fields
        .iter()
        .zip(half_outputs.chunks_exact(2))
        .flat_map(|(field, outputs)| {
            let decomposition = builder
                .canonical_u64_audits()
                .into_iter()
                .find(|decomposition| decomposition.field_col == *field)
                .unwrap_or_else(|| panic!("canonical-u64 decomposition for field column {field}"));
            outputs.iter().enumerate().map(move |(half, output)| {
                let terms = decomposition.bit_cols[half * 32..(half + 1) * 32]
                    .iter()
                    .enumerate()
                    .map(|(bit, column)| format!("({column}, {})", 1_u64 << bit))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("⟨{output}, .linear [{terms}]⟩")
            })
        })
        .collect::<Vec<_>>()
        .join(", ");
    let chunk_digest_pairs = audit.state_out_columns[14..18]
        .iter()
        .zip(hash_audits[1].output_cols)
        .map(|(&output, source)| format!("({output}, {source})"))
        .collect::<Vec<_>>()
        .join(", ");
    let semantic_accumulator_pairs = audit.state_out_columns[19..23]
        .iter()
        .zip(&audit.state_out_columns[23..27])
        .map(|(&semantic, &accumulator)| format!("({semantic}, {accumulator})"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{imports}\n\n\
         /-! Exact checked program for the base owner in the composed two-step full-history profile. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBase\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         def inputColumns : List Nat := {}\n\
         def stateInColumns : List Nat := {}\n\
         def stateInValues : List Nat := {}\n\
         def stateInPins : List (Nat × Nat) := [{state_in_pins}]\n\
         def stateOutColumns : List Nat := {}\n\
         def xOutColumns : List Nat := {}\n\
         def xOutBitColumns : List Nat := {}\n\
         def constantPins : List (Nat × Nat) := [{constant_pins}]\n\
         def canonicalU64Maps : List (List Nat) := [{canonical_u64_maps}]\n\
         def canonicalU64HalfDefinitions : List Definition := [{canonical_u64_half_definitions}]\n\
         def chunkDigestPairs : List (Nat × Nat) := [{chunk_digest_pairs}]\n\
         def semanticAccumulatorPairs : List (Nat × Nat) := [{semantic_accumulator_pairs}]\n\
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
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryBase\n",
        lean_nat_list(program.input_columns.iter().copied()),
        lean_nat_list(audit.state_in_columns.iter().copied()),
        lean_nat_list(audit.state_in_columns.iter().map(|&column| {
            usize::try_from(builder.witness()[column].as_canonical_u64()).expect("field value fits usize")
        })),
        lean_nat_list(audit.state_out_columns.iter().copied()),
        lean_nat_list(audit.x_out_columns),
        lean_nat_list(audit.x_out_bit_columns.iter().copied()),
        audit.row_end - audit.row_start,
        program.definition_count,
        program.check_count,
    )
}

fn lean_poseidon_call(call: &Poseidon2PermutationAudit) -> String {
    format!(
        "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
        call.row_start,
        call.row_end,
        lean_nat_list(call.input_cols),
        call.first_allocated_col,
    )
}

fn render_base_hashes(builder: &R1csBuilder, row_end: usize) -> String {
    let calls = builder.poseidon2_permutation_audits();
    let hashes = builder
        .poseidon2_hash_audits()
        .into_iter()
        .filter(|hash| hash.row_start < row_end && hash.row_end <= row_end)
        .collect::<Vec<_>>();
    assert_eq!(hashes.len(), 3, "composed base owner has three sponge calls");
    let traces = hashes
        .iter()
        .map(|hash| {
            let rounds = hash
                .rounds
                .iter()
                .map(|round| {
                    let call = calls
                        .iter()
                        .find(|call| {
                            call.input_cols == round.permutation_input_cols
                                && call.output_cols == round.permutation_output_cols
                        })
                        .expect("sponge permutation call");
                    let kind = match &round.kind {
                        Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
                            format!(".absorb {}", lean_nat_list(chunk_cols.iter().copied()))
                        }
                        Poseidon2HashRoundAuditKind::Pad => ".pad".to_string(),
                    };
                    format!(
                        "{{ kind := {kind}, stateBeforeColumns := {}, permutationInputColumns := {}, \
                         permutationOutputColumns := {}, definingRows := {}, call := {} }}",
                        lean_nat_list(round.state_before_cols),
                        lean_nat_list(round.permutation_input_cols),
                        lean_nat_list(round.permutation_output_cols),
                        lean_nat_list(round.defining_rows.iter().copied()),
                        lean_poseidon_call(call),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n    , ");
            format!(
                "{{ inputColumns := {}, zeroColumn := {}, zeroRow := {}, rounds := [\n      {}\n    ], \
                 outputColumns := {} }}",
                lean_nat_list(hash.input_cols.iter().copied()),
                hash.zero_col,
                hash.zero_row,
                rounds,
                lean_nat_list(hash.output_cols),
            )
        })
        .collect::<Vec<_>>()
        .join("\n  , ");
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryBaseArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated sponge certificates for the exact composed base owner. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryBasePoseidonHashes\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 524288\n\n\
         def traces : List Trace :=\n[\n  {traces}\n]\n\n\
         theorem traces_accepted :\n\
             traces.all (fun trace => decide (trace.Valid FPrimeFullHistoryBase.rows)) = true := by\n\
           native_decide\n\n\
         theorem traces_valid :\n\
             ∀ trace ∈ traces, trace.Valid FPrimeFullHistoryBase.rows := by\n\
           intro trace member\n\
           exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)\n\n\
         def xOutTrace : Trace := traces[2]!\n\n\
         theorem xOutTrace_output : xOutTrace.outputColumns = FPrimeFullHistoryBase.xOutColumns := by\n\
           native_decide\n\n\
         theorem xOutTrace_valid : xOutTrace.Valid FPrimeFullHistoryBase.rows := by\n\
           native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryBasePoseidonHashes\n"
    )
}

fn render_prior_link_hashes(builder: &R1csBuilder, row_start: usize, row_end: usize) -> String {
    let calls = builder.poseidon2_permutation_audits();
    let hashes = builder
        .poseidon2_hash_audits()
        .into_iter()
        .filter(|hash| row_start <= hash.row_start && hash.row_end <= row_end)
        .collect::<Vec<_>>();
    assert_eq!(hashes.len(), 1, "recursive prior link has one state-x_out sponge call");
    let traces = hashes
        .iter()
        .map(|hash| {
            let rounds = hash
                .rounds
                .iter()
                .map(|round| {
                    let call = calls
                        .iter()
                        .find(|call| {
                            call.input_cols == round.permutation_input_cols
                                && call.output_cols == round.permutation_output_cols
                        })
                        .expect("sponge permutation call");
                    let kind = match &round.kind {
                        Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
                            format!(".absorb {}", lean_nat_list(chunk_cols.iter().copied()))
                        }
                        Poseidon2HashRoundAuditKind::Pad => ".pad".to_string(),
                    };
                    format!(
                        "{{ kind := {kind}, stateBeforeColumns := {}, permutationInputColumns := {}, \
                         permutationOutputColumns := {}, definingRows := {}, call := {} }}",
                        lean_nat_list(round.state_before_cols),
                        lean_nat_list(round.permutation_input_cols),
                        lean_nat_list(round.permutation_output_cols),
                        lean_nat_list(round.defining_rows.iter().map(|row| row - row_start)),
                        format!(
                            "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
                            call.row_start - row_start,
                            call.row_end - row_start,
                            lean_nat_list(call.input_cols),
                            call.first_allocated_col,
                        ),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n    , ");
            format!(
                "{{ inputColumns := {}, zeroColumn := {}, zeroRow := {}, rounds := [\n      {}\n    ], \
                 outputColumns := {} }}",
                lean_nat_list(hash.input_cols.iter().copied()),
                hash.zero_col,
                hash.zero_row - row_start,
                rounds,
                lean_nat_list(hash.output_cols),
            )
        })
        .collect::<Vec<_>>()
        .join("\n  , ");
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryPriorLinkArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated sponge certificate for the exact recursive prior-link owner. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkPoseidonHashes\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 524288\n\n\
         def traces : List Trace :=\n[\n  {traces}\n]\n\n\
         theorem traces_accepted :\n\
             traces.all (fun trace => decide (trace.Valid FPrimeFullHistoryPriorLink.rows)) = true := by\n\
           native_decide\n\n\
         theorem traces_valid :\n\
             ∀ trace ∈ traces, trace.Valid FPrimeFullHistoryPriorLink.rows := by\n\
           intro trace member\n\
           exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)\n\n\
         def priorXOutTrace : Trace := traces[0]!\n\n\
         theorem priorXOutTrace_output :\n\
             priorXOutTrace.outputColumns = FPrimeFullHistoryPriorLink.digestColumns := by\n\
           native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeFullHistoryPriorLinkPoseidonHashes\n"
    )
}

fn compare_base_program_artifacts(builder: &R1csBuilder, audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit) {
    assert!(audit.is_base && audit.row_start == 0, "first owner must be base prefix");
    let program = normalize_prefix(builder, audit.row_end);
    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write reviewed base-owner artifact");
            drifted.push(expected);
        }
    };
    compare(
        root.join(BASE_PROGRAM_PATH),
        render_base_program(builder, &program, audit),
    );
    for (index, shard) in program.instructions.chunks(BASE_SHARD_SIZE).enumerate() {
        compare(
            root.join(format!("{BASE_SHARD_PREFIX}{index}.lean")),
            render_base_shard(index, shard),
        );
    }
    compare(root.join(BASE_HASHES_PATH), render_base_hashes(builder, audit.row_end));
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history base-owner artifacts drifted: {drifted:?}"
    );
}

fn compare_prior_link_artifacts(builder: &R1csBuilder, audit: &neo_fold_clean::engine::decider::FPrimeStepWireAudit) {
    assert!(!audit.is_base, "prior-link owner must be recursive");
    let (row_start, row_end) = audit
        .prior_link_row_range
        .expect("recursive prior-link row range");
    let first_column = audit
        .prior_link_first_allocated_column
        .expect("recursive prior-link first allocated column");
    let program = normalize_range(builder, row_start, row_end, first_column);
    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write reviewed prior-link artifact");
            drifted.push(expected);
        }
    };
    compare(
        root.join(PRIOR_LINK_PROGRAM_PATH),
        render_prior_link_program(builder, &program, audit),
    );
    for (index, shard) in program.instructions.chunks(BASE_SHARD_SIZE).enumerate() {
        compare(
            root.join(format!("{PRIOR_LINK_SHARD_PREFIX}{index}.lean")),
            render_prior_link_shard(index, shard),
        );
    }
    compare(
        root.join(PRIOR_LINK_HASHES_PATH),
        render_prior_link_hashes(builder, row_start, row_end),
    );
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history prior-link artifacts drifted: {drifted:?}"
    );
}

fn compare_projection_artifacts(groups: &[ProjectionGroup<'_>]) {
    let root = formal_repo_root();
    let mut drifted = Vec::new();
    let mut compare = |path: PathBuf, rendered: String| {
        if fs::read_to_string(&path).ok().as_deref() != Some(&rendered) {
            let expected = path.with_extension("lean.expected");
            fs::write(&expected, rendered).expect("write reviewed projection census artifact");
            drifted.push(expected);
        }
    };
    for (kind, instructions) in projection_template_parts(groups) {
        for (index, shard) in instructions
            .chunks(PROJECTION_TEMPLATE_SHARD_SIZE)
            .enumerate()
        {
            compare(
                root.join(format!("{FULL_HISTORY_PROJECTION_PREFIX}{kind}{index}.lean")),
                render_projection_instruction_shard(kind, index, shard),
            );
        }
    }
    compare(
        root.join(FULL_HISTORY_PROJECTION_TEMPLATES_PATH),
        render_projection_templates(groups),
    );
    for group in groups {
        let shard_size = projection_census_shard_size(group);
        for (index, (identities, maps)) in group
            .identities
            .chunks(shard_size)
            .zip(group.identity_maps.chunks(shard_size))
            .enumerate()
        {
            compare(
                root.join(format!(
                    "{FULL_HISTORY_PROJECTION_CENSUS_PREFIX}{}{index}.lean",
                    group.name
                )),
                render_projection_census_shard(group, index, identities, maps),
            );
        }
    }
    compare(
        root.join(FULL_HISTORY_PROJECTION_PATH),
        render_full_history_projection(groups),
    );
    assert!(
        drifted.is_empty() || STAGE_ALL_ARTIFACTS,
        "full-history projection artifacts drifted: {drifted:?}"
    );
}

fn compare_full_history_artifact(path: &Path, rendered: &str, expected_extension: &str) {
    let committed = fs::read_to_string(path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension(expected_extension);
        fs::write(&expected, rendered).expect("write reviewed full-history artifact");
    }
    assert!(
        committed == rendered || STAGE_ALL_ARTIFACTS,
        "full-history M4 artifact drifted: {}",
        path.display()
    );
}

#[test]
fn terminal_parent_and_accumulator_artifacts_match_exact_rows() {
    let (prep, finished) = build_honest_finished_proof(2);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize full history");
    assert!(synth.builder.is_satisfied(), "honest full-history rows");
    compare_terminal_accumulator_artifacts(&synth.builder);
    let base = synth
        .step_wire_audits
        .iter()
        .find(|audit| audit.is_base)
        .expect("base wire audit");
    let recursive = synth
        .step_wire_audits
        .iter()
        .find(|audit| !audit.is_base)
        .expect("recursive wire audit");
    compare_transcript_artifacts(&synth.builder, base, recursive);
}

#[test]
fn current_terminal_link_full_history_placement_matches_exact_rows() {
    let (prep, finished) = build_honest_finished_proof(2);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize full history");
    assert!(synth.builder.is_satisfied(), "honest full-history rows");
    compare_current_terminal_diagnostic(&synth.builder, prep.structure().t());
    compare_current_terminal_link_artifact(&synth.builder);
}

#[test]
fn full_history_m4_manifest_matches_exact_composed_rows() {
    let (prep, finished) = build_honest_finished_proof(2);
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize full history");
    assert!(synth.builder.is_satisfied(), "honest full-history rows");
    assert!(
        synth.is_self_sufficient_relation(),
        "all direct audit owners must be present"
    );
    assert_eq!(
        synth.builder.projection_ladder_audits().len(),
        2,
        "recursive plus terminal NIFS must each expose one projection ladder"
    );
    assert_eq!(
        synth.builder.projection_identity_audits().len(),
        62,
        "recursive plus terminal NIFS must each expose all 31 projection identities"
    );
    let identity_row_counts = synth
        .builder
        .projection_identity_audits()
        .iter()
        .map(|audit| audit.row_end - audit.row_start)
        .collect::<Vec<_>>();
    assert_eq!(
        identity_row_counts,
        std::iter::repeat_n(334, 31)
            .chain(std::iter::repeat_n(1_916, 31))
            .collect::<Vec<_>>(),
        "recursive one-input and terminal fifteen-input projection shapes"
    );
    compare_projection_role_artifact(&synth.builder);
    compare_seeded_phi81_artifact(&synth.builder);

    let manifest = full_history_manifest(&synth.builder);
    let json_rendered = format!("{}\n", serde_json::to_string_pretty(&manifest).expect("render JSON"));
    compare_full_history_artifact(
        &formal_repo_root().join(FULL_HISTORY_MANIFEST_PATH),
        &json_rendered,
        "json.expected",
    );
    compare_full_history_artifact(
        &formal_repo_root().join(FULL_HISTORY_LEAN_PATH),
        &render_full_history_lean(&manifest),
        "lean.expected",
    );
    compare_affine_artifacts(&synth.builder);
    compare_public_pins_artifacts(&synth.builder);
    let base_audit = synth
        .step_wire_audits
        .first()
        .expect("base step wire audit");
    compare_base_program_artifacts(&synth.builder, base_audit);
    let recursive_audit = synth
        .step_wire_audits
        .iter()
        .find(|audit| !audit.is_base)
        .expect("recursive step wire audit");
    compare_full_history_artifact(
        &formal_repo_root().join(FULL_HISTORY_STATE_LINK_PATH),
        &render_full_history_state_link(&synth.builder, base_audit, recursive_audit),
        "lean.expected",
    );
    compare_prior_link_artifacts(&synth.builder, recursive_audit);
    compare_counter_artifact(&synth.builder, recursive_audit);
    compare_full_history_artifact(
        &formal_repo_root().join(FULL_HISTORY_OUTPUT_ENCODING_PATH),
        &render_output_encoding_artifact(&synth.builder, recursive_audit),
        "lean.expected",
    );
    compare_recursive_output_artifacts(&synth.builder, recursive_audit);
    let projection_groups = full_history_projection_groups(&synth.builder);
    compare_projection_artifacts(&projection_groups);
    compare_pi_dec_artifacts(&synth.builder);
    compare_terminal_ce_profiles(&synth.builder);
    let mut point_bindings = synth
        .builder
        .program_range_audits()
        .iter()
        .filter(|range| range.name == "nifs.point_binding")
        .collect::<Vec<_>>();
    point_bindings.sort_by_key(|range| range.row_start);
    assert_eq!(point_bindings.len(), 2, "recursive and terminal point bindings");
    for (path, namespace, title, audit) in [
        (
            FULL_HISTORY_RECURSIVE_POINT_BINDING_PATH,
            "FPrimeFullHistoryRecursivePointBinding",
            "recursive NIFS PiCCS/PiDEC point binding",
            point_bindings[0],
        ),
        (
            FULL_HISTORY_TERMINAL_POINT_BINDING_PATH,
            "FPrimeFullHistoryTerminalPointBinding",
            "terminal NIFS PiCCS/PiDEC point binding",
            point_bindings[1],
        ),
    ] {
        let range = RowFamilyRange {
            name: "nifs.point_binding",
            row_start: audit.row_start,
            row_end: audit.row_end,
        };
        compare_full_history_artifact(
            &formal_repo_root().join(path),
            &render_equality_artifact(
                &synth.builder,
                &range,
                namespace,
                title,
                &range_hash(&synth.builder, &range),
            ),
            "lean.expected",
        );
    }
    let terminal_link = synth
        .builder
        .row_family_ranges()
        .iter()
        .find(|range| range.name == "terminal.latest_link")
        .expect("terminal latest-link owner");
    compare_full_history_artifact(
        &formal_repo_root().join(FULL_HISTORY_TERMINAL_LINK_PATH),
        &render_equality_artifact(
            &synth.builder,
            terminal_link,
            "FPrimeFullHistoryTerminalLink",
            "terminal delayed public-link rows",
            &range_hash(&synth.builder, terminal_link),
        ),
        "lean.expected",
    );
    compare_terminal_continuity_artifacts(&synth.builder);
}
