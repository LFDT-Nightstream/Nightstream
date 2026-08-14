//! Exact checked-program artifact for the production Nebula base binding.

#![allow(non_snake_case)]

#[path = "checked_program_artifact_support.rs"]
#[allow(dead_code)]
mod checked_program_artifact_support;
#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use checked_program_artifact_support::{lean_instructions, normalize_with_inputs, Instruction, NormalizedProgram, Row};
use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_fold_clean::engine::r1cs_circuit::builder::{Poseidon2HashRoundAuditKind, Poseidon2PermutationAudit};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::paper::construction2::{NebulaConfig, NebulaLane, StackShape};
use neo_fold_clean::paper::digest::NEBULA_PROGRAM_BINDING_TAG;
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::{
    alloc_nebula_lane_wires, enforce_nebula_lane_base_circuit, NebulaBaseBindingWires, NebulaLaneWires,
};
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const SHARD_SIZE: usize = 1_000;
const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/Nebula/NebulaProgramBindingArtifact.lean";
const SHARD_REL_PREFIX: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/NebulaProgramBinding/Generated/NebulaProgramBindingInstructions";
const POSEIDON_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/NebulaProgramBinding/Generated/NebulaProgramBindingPoseidon.lean";

struct BuiltBinding {
    builder: R1csBuilder,
    program: NormalizedProgram,
    lane: NebulaLaneWires,
    semanticState: [Var; 4],
    binding: NebulaBaseBindingWires,
    laneColumns: Vec<usize>,
    tagColumns: Vec<usize>,
    bindingLinkRowStart: usize,
    semanticLinkRowStart: usize,
    memoryLinkRowStart: usize,
}

fn config() -> NebulaConfig {
    let scheme = LaneScheme::from_seeds(
        2,
        LaneRanges {
            ops: 1..2,
            is: 2..3,
            fs: 3..4,
        },
        [1; 32],
        [2; 32],
    )
    .expect("artifact lane scheme");
    NebulaConfig {
        scheme,
        steps_per_segment: 2,
        seg_max: 1,
        stacks: StackShape::NONE,
        initial_semantic_state_digest: [F::from_u64(6); 4],
        plan_digest: [F::from_u64(7); 4],
        d_init: [F::from_u64(41); 4],
    }
}

fn lane_columns(lane: &NebulaLaneWires) -> Vec<usize> {
    let mut columns = Vec::new();
    columns.extend(lane.program_binding_digest.map(Var::col));
    columns.extend([lane.open, lane.seg_idx, lane.idx, lane.ts].map(Var::col));
    for value in lane.gamma.iter().chain(lane.h.iter()) {
        columns.extend([value.c0.col(), value.c1.col()]);
    }
    columns.extend(lane.sp.map(Var::col));
    columns.extend(lane.d_pre.iter().flatten().map(|wire| wire.col()));
    columns.extend(lane.d_seen.iter().flatten().map(|wire| wire.col()));
    columns.extend(lane.d_mem.map(Var::col));
    columns
}

fn binding_input_columns(binding: &NebulaBaseBindingWires) -> Vec<usize> {
    binding
        .initial_semantic_state_digest
        .iter()
        .chain(binding.plan_digest.iter())
        .chain(binding.d_init.iter())
        .map(|wire| wire.col())
        .collect()
}

fn equality_instruction(output: usize, input: usize) -> Instruction {
    Instruction::Check(Row {
        a: vec![(output, 1), (input, F::ORDER_U64 - 1)],
        b: vec![(0, 1)],
        c: Vec::new(),
    })
}

fn equality_run_start(program: &NormalizedProgram, outputs: &[usize], inputs: &[usize]) -> usize {
    assert_eq!(outputs.len(), inputs.len(), "equality run width");
    let expected = outputs
        .iter()
        .zip(inputs)
        .map(|(&output, &input)| equality_instruction(output, input))
        .collect::<Vec<_>>();
    let starts = program
        .instructions
        .windows(expected.len())
        .enumerate()
        .filter_map(|(start, window)| (window == expected).then_some(start))
        .collect::<Vec<_>>();
    let [start] = starts.as_slice() else {
        panic!("named equality run must occur exactly once, found {starts:?}")
    };
    *start
}

fn build() -> BuiltBinding {
    let cfg = config();
    let mut builder = R1csBuilder::new();
    let lane = alloc_nebula_lane_wires(&mut builder, &NebulaLane::base(&cfg));
    let semanticState = cfg
        .initial_semantic_state_digest
        .map(|value| builder.alloc(value));
    let binding = enforce_nebula_lane_base_circuit(&mut builder, &lane, &cfg, &semanticState);
    let laneColumns = lane_columns(&lane);
    let mut declaredInputs = vec![0];
    declaredInputs.extend(laneColumns.iter().copied());
    declaredInputs.extend(semanticState.map(Var::col));
    declaredInputs.extend(binding_input_columns(&binding));
    let program = normalize_with_inputs(&builder, &declaredInputs);
    declaredInputs.sort_unstable();
    declaredInputs.dedup();
    assert_eq!(program.input_columns, declaredInputs, "artifact input ownership");

    let hashes = builder.poseidon2_hash_audits();
    let [hash] = hashes.as_slice() else {
        panic!("Nebula program binding must own one Poseidon2 sponge call")
    };
    let tagCount = 1 + NEBULA_PROGRAM_BINDING_TAG.len().div_ceil(7);
    assert_eq!(hash.input_cols.len(), tagCount + 12, "binding preimage width");
    assert_eq!(
        &hash.input_cols[tagCount..],
        binding_input_columns(&binding),
        "program values must follow the domain tag",
    );
    assert_eq!(
        hash.output_cols,
        binding.computed_program_binding_digest.map(Var::col),
        "binding output columns",
    );
    let computedBindingColumns = binding.computed_program_binding_digest.map(Var::col);
    let carriedBindingColumns = lane.program_binding_digest.map(Var::col);
    let semanticStateColumns = semanticState.map(Var::col);
    let initialSemanticStateColumns = binding.initial_semantic_state_digest.map(Var::col);
    let carriedMemoryColumns = lane.d_mem.map(Var::col);
    let initialMemoryDigestColumns = binding.d_init.map(Var::col);
    let bindingLinkRowStart = equality_run_start(&program, &computedBindingColumns, &carriedBindingColumns);
    let semanticLinkRowStart = equality_run_start(&program, &semanticStateColumns, &initialSemanticStateColumns);
    let memoryLinkRowStart = equality_run_start(&program, &carriedMemoryColumns, &initialMemoryDigestColumns);

    BuiltBinding {
        tagColumns: hash.input_cols[..tagCount].to_vec(),
        builder,
        program,
        lane,
        semanticState,
        binding,
        laneColumns,
        bindingLinkRowStart,
        semanticLinkRowStart,
        memoryLinkRowStart,
    }
}

fn artifact_hashes(built: &BuiltBinding) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/nebula-program-binding\n\
         source=enforce_nebula_lane_base_circuit\ntag={}\ninputs={}\nlane={}\nsemantic={}\n\
         initial_semantic={}\nplan={}\nd_init={}\ncomputed_binding={}\n\
         binding_link_row_start={}\nsemantic_link_row_start={}\nmemory_link_row_start={}\n\
         rows={}\ncols={}\n{}",
        String::from_utf8_lossy(NEBULA_PROGRAM_BINDING_TAG),
        lean_nat_list(built.program.input_columns.iter().copied()),
        lean_nat_list(built.laneColumns.iter().copied()),
        lean_nat_list(built.semanticState.map(Var::col)),
        lean_nat_list(built.binding.initial_semantic_state_digest.map(Var::col)),
        lean_nat_list(built.binding.plan_digest.map(Var::col)),
        lean_nat_list(built.binding.d_init.map(Var::col)),
        lean_nat_list(built.binding.computed_program_binding_digest.map(Var::col)),
        built.bindingLinkRowStart,
        built.semanticLinkRowStart,
        built.memoryLinkRowStart,
        built.builder.rows(),
        built.builder.cols(),
        lean_rows(&built.builder),
    );
    let witness_payload = lean_witness("honestWitness", built.builder.witness());
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

fn render_shard(index: usize, instructions: &[checked_program_artifact_support::Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated Nebula program-binding instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.NebulaProgramBinding.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.NebulaProgramBinding.Generated\n",
        lean_instructions(instructions),
    )
}

fn render_main(built: &BuiltBinding, rowHash: &str, witnessHash: &str) -> String {
    let shardCount = built.program.instructions.len().div_ceil(SHARD_SIZE);
    let imports = (0..shardCount)
        .map(|index| {
            format!(
                "import Nightstream.Implementation.R1CS.Artifacts.NebulaProgramBinding.Generated.NebulaProgramBindingInstructions{index}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let instructions = (0..shardCount)
        .map(|index| format!("Generated.instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{imports}\n\n\
         /-! Exact checked-program artifact for the production Nebula base binding. -/\n\n\
         namespace Nightstream.Implementation.R1CS.NebulaProgramBinding\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\
         set_option maxHeartbeats 5000000\n\n\
         def schemaVersion : Nat := {SCHEMA_VERSION}\n\
         def artifactKind : String := \"r1cs/nebula-program-binding\"\n\
         def sourceAnchor : String := \"enforce_nebula_lane_base_circuit\"\n\
         def domainTag : String := \"{}\"\n\
         def artifactSha256 : String := \"{rowHash}\"\n\
         def witnessSha256 : String := \"{witnessHash}\"\n\n\
         def inputColumns : List Nat := {}\n\
         def laneColumns : List Nat := {}\n\
         def semanticStateColumns : List Nat := {}\n\
         def initialSemanticStateColumns : List Nat := {}\n\
         def planDigestColumns : List Nat := {}\n\
         def initialMemoryDigestColumns : List Nat := {}\n\
         def tagColumns : List Nat := {}\n\
         def computedBindingColumns : List Nat := {}\n\
         def carriedBindingColumns : List Nat := {}\n\
         def carriedMemoryColumns : List Nat := {}\n\
         def bindingLinkRowStart : Nat := {}\n\
         def semanticLinkRowStart : Nat := {}\n\
         def memoryLinkRowStart : Nat := {}\n\
         def rowCount : Nat := {}\n\
         def colCount : Nat := {}\n\
         def definitionCount : Nat := {}\n\
         def checkCount : Nat := {}\n\n\
         def instructions : List Instruction :=\n    {instructions}\n\n\
         def rows : List Row := CheckedProgram.rows instructions\n\n\
         def bindingLinkRows : List Row :=\n\
           (List.range 4).map fun lane =>\n\
             builderLinearRow (computedBindingColumns.getD lane 0)\n\
               [(carriedBindingColumns.getD lane 0, 1)]\n\n\
         def semanticLinkRows : List Row :=\n\
           (List.range 4).map fun lane =>\n\
             builderLinearRow (semanticStateColumns.getD lane 0)\n\
               [(initialSemanticStateColumns.getD lane 0, 1)]\n\n\
         def memoryLinkRows : List Row :=\n\
           (List.range 4).map fun lane =>\n\
             builderLinearRow (carriedMemoryColumns.getD lane 0)\n\
               [(initialMemoryDigestColumns.getD lane 0, 1)]\n\n\
         theorem digest_widths :\n\
             semanticStateColumns.length = 4 ∧\n\
             initialSemanticStateColumns.length = 4 ∧\n\
             planDigestColumns.length = 4 ∧\n\
             initialMemoryDigestColumns.length = 4 ∧\n\
             computedBindingColumns.length = 4 ∧\n\
             carriedBindingColumns.length = 4 ∧\n\
             carriedMemoryColumns.length = 4 := by decide\n\
         theorem binding_link_rows_exact :\n\
             (rows.drop bindingLinkRowStart).take 4 = bindingLinkRows := by decide\n\
         theorem semantic_link_rows_exact :\n\
             (rows.drop semanticLinkRowStart).take 4 = semanticLinkRows := by decide\n\
         theorem memory_link_rows_exact :\n\
             (rows.drop memoryLinkRowStart).take 4 = memoryLinkRows := by decide\n\n\
         end Nightstream.Implementation.R1CS.NebulaProgramBinding\n",
        String::from_utf8_lossy(NEBULA_PROGRAM_BINDING_TAG),
        lean_nat_list(built.program.input_columns.iter().copied()),
        lean_nat_list(built.laneColumns.iter().copied()),
        lean_nat_list(built.semanticState.map(Var::col)),
        lean_nat_list(built.binding.initial_semantic_state_digest.map(Var::col)),
        lean_nat_list(built.binding.plan_digest.map(Var::col)),
        lean_nat_list(built.binding.d_init.map(Var::col)),
        lean_nat_list(built.tagColumns.iter().copied()),
        lean_nat_list(built.binding.computed_program_binding_digest.map(Var::col)),
        lean_nat_list(built.lane.program_binding_digest.map(Var::col)),
        lean_nat_list(built.lane.d_mem.map(Var::col)),
        built.bindingLinkRowStart,
        built.semanticLinkRowStart,
        built.memoryLinkRowStart,
        built.builder.rows(),
        built.builder.cols(),
        built.program.definition_count,
        built.program.check_count,
    )
}

fn lean_poseidon_call(call: &Poseidon2PermutationAudit, rowOffset: usize) -> String {
    format!(
        "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
        call.row_start - rowOffset,
        call.row_end - rowOffset,
        lean_nat_list(call.input_cols),
        call.first_allocated_col,
    )
}

fn render_poseidon(built: &BuiltBinding) -> String {
    let calls = built.builder.poseidon2_permutation_audits();
    let hashes = built.builder.poseidon2_hash_audits();
    let [hash] = hashes.as_slice() else {
        panic!("Nebula program binding must own one Poseidon2 sponge call")
    };
    let inputFieldCount = hash.input_cols.len();
    let emissionCost = inputFieldCount + 2 + 600 * hash.rounds.len();
    assert_eq!(hash.zero_row, hash.row_start, "sponge zero row starts the trace");
    assert_eq!(hash.row_end - hash.row_start, emissionCost, "sponge emission cost");
    for call in &calls {
        assert_eq!(call.row_end - call.row_start, 600, "Poseidon2 row count");
        assert_eq!(call.allocated_col_count, 600, "Poseidon2 fresh-column count");
        let mappedOutputs: [usize; 8] = std::array::from_fn(|lane| call.first_allocated_col + 592 + lane);
        assert_eq!(call.output_cols, mappedOutputs, "Poseidon2 output renaming");
    }
    let roundLiterals = hash
        .rounds
        .iter()
        .map(|round| {
            let call = calls
                .iter()
                .find(|call| {
                    call.input_cols == round.permutation_input_cols && call.output_cols == round.permutation_output_cols
                })
                .expect("every sponge round must own one exact permutation call");
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
                lean_nat_list(round.defining_rows.iter().map(|row| row - hash.row_start)),
                lean_poseidon_call(call, hash.row_start),
            )
        })
        .collect::<Vec<_>>()
        .join("\n    , ");
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.Nebula.NebulaProgramBindingArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated exact Poseidon2 sponge trace for the Nebula program binding. -/\n\n\
         namespace Nightstream.Implementation.R1CS.NebulaProgramBindingPoseidon\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 524288\n\
         set_option maxHeartbeats 5000000\n\n\
         def trace : Trace :=\n\
         {{ inputColumns := {}, zeroColumn := {}, zeroRow := 0, rounds := [\n      {}\n    ], outputColumns := {} }}\n\n\
         def inputFieldCount : Nat := {}\n\
         def rowStart : Nat := {}\n\
         def traceRowCount : Nat := {}\n\n\
         theorem trace_valid :\n\
             trace.Valid trace.rows := by constructor <;> decide\n\
         theorem trace_rows_exact :\n\
             (NebulaProgramBinding.rows.drop rowStart).take traceRowCount =\n\
               trace.rows := by decide\n\
         theorem trace_input_layout :\n\
             trace.inputColumns = NebulaProgramBinding.tagColumns ++\n\
               NebulaProgramBinding.initialSemanticStateColumns ++\n\
               NebulaProgramBinding.planDigestColumns ++\n\
               NebulaProgramBinding.initialMemoryDigestColumns := by decide\n\
         theorem trace_output_layout :\n\
             trace.outputColumns = NebulaProgramBinding.computedBindingColumns := by decide\n\n\
         end Nightstream.Implementation.R1CS.NebulaProgramBindingPoseidon\n",
        lean_nat_list(hash.input_cols.iter().copied()),
        hash.zero_col,
        roundLiterals,
        lean_nat_list(hash.output_cols),
        inputFieldCount,
        hash.row_start,
        hash.row_end - hash.row_start,
    )
}

fn rendered_artifacts(built: &BuiltBinding) -> Vec<(std::path::PathBuf, String)> {
    let manifestDir = env!("CARGO_MANIFEST_DIR");
    let (rowHash, witnessHash) = artifact_hashes(built);
    let mut artifacts = vec![
        (
            std::path::PathBuf::from(format!("{manifestDir}{ARTIFACT_REL_PATH}")),
            render_main(built, &rowHash, &witnessHash),
        ),
        (
            std::path::PathBuf::from(format!("{manifestDir}{POSEIDON_REL_PATH}")),
            render_poseidon(built),
        ),
    ];
    artifacts.extend(
        built
            .program
            .instructions
            .chunks(SHARD_SIZE)
            .enumerate()
            .map(|(index, shard)| {
                (
                    std::path::PathBuf::from(format!("{manifestDir}{SHARD_REL_PREFIX}{index}.lean")),
                    render_shard(index, shard),
                )
            }),
    );
    artifacts
}

fn compare_or_write_expected(path: &std::path::Path, rendered: &str, drifted: &mut Vec<String>) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(rendered) {
        std::fs::create_dir_all(path.parent().expect("artifact parent")).expect("create artifact directory");
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected Lean artifact");
        drifted.push(expected.display().to_string());
    }
}

#[test]
fn program_binding_accepts_honest_witness() {
    let built = build();
    assert!(
        built.builder.is_satisfied(),
        "honest binding failed at {:?}",
        built.builder.first_unsatisfied_row(),
    );
    assert!(built.builder.unconstrained_columns().is_empty());
    assert_eq!(built.program.instructions.len(), built.builder.rows());
    assert_eq!(
        built.program.definition_count + built.program.check_count,
        built.builder.rows(),
    );
}

#[test]
fn program_binding_rejects_each_authoritative_input_mutation() {
    let probes: [(&str, fn(&BuiltBinding) -> usize); 6] = [
        ("initial semantic", |built| {
            built.binding.initial_semantic_state_digest[0].col()
        }),
        ("plan", |built| built.binding.plan_digest[0].col()),
        ("initial memory", |built| built.binding.d_init[0].col()),
        ("carried binding", |built| built.lane.program_binding_digest[0].col()),
        ("carried semantic", |built| built.semanticState[0].col()),
        ("carried memory", |built| built.lane.d_mem[0].col()),
    ];
    for (name, select) in probes {
        let mut built = build();
        let column = select(&built);
        built
            .builder
            .tamper_witness(column, built.builder.witness()[column] + F::ONE);
        assert!(
            !built.builder.is_satisfied(),
            "{name} mutation must fail at column {column}",
        );
    }
}

#[test]
fn lean_program_binding_artifacts_match_committed_files() {
    let built = build();
    assert!(built.builder.is_satisfied());
    let mut drifted = Vec::new();
    for (path, rendered) in rendered_artifacts(&built) {
        compare_or_write_expected(&path, &rendered, &mut drifted);
    }
    assert!(
        drifted.is_empty(),
        "generated Lean Nebula program-binding artifacts drifted: {drifted:?}",
    );
}

#[test]
#[ignore = "writes reviewed generated Lean artifacts"]
fn regenerate_lean_program_binding_artifacts() {
    let built = build();
    assert!(built.builder.is_satisfied());
    for (path, rendered) in rendered_artifacts(&built) {
        std::fs::create_dir_all(path.parent().expect("artifact parent")).expect("create artifact directory");
        std::fs::write(&path, rendered).expect("write generated Lean artifact");
        let expected = path.with_extension("lean.expected");
        if expected.exists() {
            std::fs::remove_file(expected).expect("remove reviewed expected artifact");
        }
    }
}
