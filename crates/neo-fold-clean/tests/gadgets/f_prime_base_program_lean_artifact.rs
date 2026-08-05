//! Exact mixed-program artifact for the production plain F' base step.

#![allow(non_snake_case)]

#[path = "checked_program_artifact_support.rs"]
// Shared gadget-test support: each test binary uses a different subset.
#[allow(dead_code)]
mod checked_program_artifact_support;
#[path = "lean_artifact_support.rs"]
mod lean_artifact_support;

use checked_program_artifact_support::{lean_instructions, normalize, NormalizedProgram};
use lean_artifact_support::{lean_nat_list, lean_rows, lean_witness, sha256_hex, SCHEMA_VERSION};
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::builder::{Poseidon2HashRoundAuditKind, Poseidon2PermutationAudit};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{LaneCommitmentMode, RunningInstance};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest, state_x_out_digest_with_mode,
    AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{
    enforce_f_prime_base_step_circuit, FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeStateIn, FPrimeStateWires,
    FPrimeStepConfig,
};
use neo_fold_clean::paper::f_prime::source_image::{BitRange, FPrimeSourceImage, Word64Image};
use neo_fold_clean::paper::nifs::circuit::NifsVCircuitConfig;
use neo_fold_clean::paper::reductions::pi_ccs_circuit::PiCcsVerifierConfig;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const TRANSCRIPT_LABEL: &[u8] = b"neo.test.f_prime/base-program-artifact/v1";
const ROWS_IN_CHUNK: u64 = 3;
const SHARD_SIZE: usize = 1_200;
const ARTIFACT_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrimeBase/FPrimeBaseProgramArtifact.lean";
const SHARD_REL_PREFIX: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeBase/Generated/FPrimeBaseProgramInstructions";
const POSEIDON_CALLS_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeBase/Generated/FPrimeBasePoseidonCalls.lean";
const POSEIDON_HASHES_REL_PATH: &str =
    "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeBase/Generated/FPrimeBasePoseidonHashes.lean";

struct SourceFixture {
    image: FPrimeSourceImage,
    chunkCount: Word64Image,
    stepCount: Word64Image,
    pc: Word64Image,
    publicXOut: BitRange,
}

struct BuiltBase {
    builder: R1csBuilder,
    program: NormalizedProgram,
    stateInColumns: Vec<usize>,
    stateOutColumns: Vec<usize>,
    xOutColumns: Vec<usize>,
    xOutBitColumns: Vec<usize>,
}

fn bit_carrier_r1cs() -> R1cs {
    let layout = FPrimePublicInputLayout::plain();
    let mut a = Mat::zero(layout.carrier_padding_len(), layout.total_len(), F::ZERO);
    let mut b = Mat::zero(layout.carrier_padding_len(), layout.total_len(), F::ZERO);
    for row in 0..layout.carrier_padding_len() {
        a[(row, layout.carrier_padding_offset() + row)] = F::ONE;
        b[(row, 0)] = F::ONE;
    }
    R1cs {
        a,
        b,
        c: Mat::zero(layout.carrier_padding_len(), layout.total_len(), F::ZERO),
        m_in: layout.total_len(),
    }
}

fn pi_ccs_config(prep: &neo_fold_clean::Preprocessing) -> PiCcsVerifierConfig<'_> {
    PiCcsVerifierConfig {
        params: &prep.params,
        structure: prep.structure().into(),
        matrix_digest: prep.pi_ccs_header_bundle(),
    }
}

fn step_config(prep: &neo_fold_clean::Preprocessing) -> FPrimeStepConfig<'_> {
    FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: pi_ccs_config(prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::plain(),
        nebula: None,
        state_x_out_digest_mode: StateXOutDigestMode::Stateless,
    }
}

fn rand_digest(seed: u64) -> [F; 4] {
    std::array::from_fn(|lane| F::from_u64(seed.wrapping_mul(31).wrapping_add(lane as u64 + 1)))
}

fn base_state(chunk_count: u64) -> FPrimeStateIn {
    let empty_acc = AccumulatorHandle::empty().digest_fields();
    let z0 = rand_digest(0x100);
    FPrimeStateIn {
        vk_fs_digest: rand_digest(0x10),
        pi_ccs_header_bundle: rand_digest(0x20),
        chunk_count_in: chunk_count,
        step_count_in: 0,
        z_0: z0,
        z_i_in: z0,
        pc: 1,
        semantic_state_digest_in: empty_acc,
        acc_digest_in: empty_acc,
        public_trace_in: rand_digest(0x40),
        nebula: None,
    }
}

fn canonical_base_acc_digest(prep: &neo_fold_clean::Preprocessing) -> [F; 4] {
    let m_in = prep
        .public_input_len
        .expect("artifact fixture pins public input width");
    let running = RunningInstance::canonical_zero(&prep.params, prep.structure(), m_in, LaneCommitmentMode::Plain)
        .expect("construct canonical base accumulator");
    AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest_fields()
}

fn native_x_out(state: &FPrimeStateIn, chunk_digest: [F; 4], base_acc: [F; 4]) -> [F; 4] {
    let boundary = digest_fields_as_digest32(chunk_digest);
    digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        digest_fields_as_digest32(state.vk_fs_digest),
        state.pi_ccs_header_bundle,
        &state.pi_ccs_header_bundle,
        1,
        ROWS_IN_CHUNK,
        digest_fields_as_digest32(state.z_0),
        boundary,
        state.pc,
        digest_fields_as_digest32(base_acc),
        digest_fields_as_digest32(base_acc),
        boundary,
        None,
    ))
}

fn source_fixture(state: &FPrimeStateIn, x_out: [F; 4]) -> SourceFixture {
    let mut image = FPrimeSourceImage::new();
    let chunkCount = image.push_u64_le(state.chunk_count_in);
    let stepCount = image.push_u64_le(state.step_count_in);
    let pc = image.push_u64_le(state.pc);
    let publicXOut = image.push_enc_inst(x_out);
    SourceFixture {
        image,
        chunkCount,
        stepCount,
        pc,
        publicXOut,
    }
}

fn state_columns(state: &FPrimeStateWires) -> Vec<usize> {
    let mut columns = Vec::new();
    columns.extend(state.vk_fs_digest.map(Var::col));
    columns.extend(state.pi_ccs_header_bundle.map(Var::col));
    columns.push(state.chunk_count.col());
    columns.push(state.step_count.col());
    columns.extend(state.z_0.map(Var::col));
    columns.extend(state.z_i.map(Var::col));
    columns.push(state.pc.col());
    columns.extend(state.semantic_state_digest.map(Var::col));
    columns.extend(state.acc_digest.map(Var::col));
    columns.extend(state.public_trace.map(Var::col));
    assert!(state.nebula.is_none(), "plain artifact must not contain Nebula wires");
    columns
}

fn build(chunk_count: u64, mutate_chunk_digest: bool) -> BuiltBase {
    let r1cs = bit_carrier_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 42).expect("preprocess");
    let config = step_config(&prep);
    let zero_assignment = vec![F::ZERO; prep.structure().m];
    let template = direct_ccs::build_instance(&prep, &r1cs, &zero_assignment).expect("shape claim");
    let claims = vec![template.claim; ROWS_IN_CHUNK as usize];
    let state = base_state(chunk_count);
    let mut chunk_digest = f_prime_chunk_public_digest(state.step_count_in, &claims);
    if mutate_chunk_digest {
        chunk_digest[0] += F::ONE;
    }
    let base_acc = canonical_base_acc_digest(&prep);
    let expected_x_out = native_x_out(&state, chunk_digest, base_acc);
    let source = source_fixture(&state, expected_x_out);
    let inputs = FPrimeBaseInputs {
        state,
        chunk_digest,
        semantic_state_digest_out: base_acc,
        rows_in_chunk: ROWS_IN_CHUNK,
        source_image: &source.image,
        chunk_count_in_word: source.chunkCount,
        step_count_in_word: source.stepCount,
        pc_word: source.pc,
        public_x_out_bits: source.publicXOut,
    };
    let mut builder = R1csBuilder::new();
    let output = enforce_f_prime_base_step_circuit(&mut builder, &config, &inputs).expect("emit base F' program");
    let program = normalize(&builder);
    BuiltBase {
        stateInColumns: state_columns(&output.state_in),
        stateOutColumns: state_columns(&output.state_out),
        xOutColumns: output.x_out.map(Var::col).to_vec(),
        xOutBitColumns: output.x_out_bits.iter().map(|wire| wire.col()).collect(),
        builder,
        program,
    }
}

fn artifact_hashes(base: &BuiltBase) -> (String, String) {
    let row_payload = format!(
        "schema={SCHEMA_VERSION}\nkind=r1cs/f-prime-base-program\nsource=enforce_f_prime_base_step_circuit\n\
         inputs={}\nstate_in={}\nstate_out={}\nx_out={}\nx_out_bits={}\nrows={}\ncols={}\n{}",
        lean_nat_list(base.program.input_columns.iter().copied()),
        lean_nat_list(base.stateInColumns.iter().copied()),
        lean_nat_list(base.stateOutColumns.iter().copied()),
        lean_nat_list(base.xOutColumns.iter().copied()),
        lean_nat_list(base.xOutBitColumns.iter().copied()),
        base.builder.rows(),
        base.builder.cols(),
        lean_rows(&base.builder),
    );
    let witness_payload = lean_witness("honestWitness", base.builder.witness());
    (sha256_hex(&row_payload), sha256_hex(&witness_payload))
}

fn render_shard(index: usize, instructions: &[checked_program_artifact_support::Instruction]) -> String {
    format!(
        "import Nightstream.Implementation.R1CS.Core.CheckedProgram\n\n\
         /-! Generated production F' base-program instruction shard {index}. Do not hand-edit. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeBaseProgram.Generated\n\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 262144\n\n\
         def instructions{index} : List Instruction :=\n  [{}]\n\n\
         end Nightstream.Implementation.R1CS.FPrimeBaseProgram.Generated\n",
        lean_instructions(instructions),
    )
}

fn render_main(base: &BuiltBase, row_hash: &str, witness_hash: &str) -> String {
    let shard_count = base.program.instructions.len().div_ceil(SHARD_SIZE);
    let imports = (0..shard_count)
        .map(|index| format!("import Nightstream.Implementation.R1CS.Artifacts.FPrimeBase.Generated.FPrimeBaseProgramInstructions{index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let instructions = (0..shard_count)
        .map(|index| format!("Generated.instructions{index}"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{imports}\n\n\
         /-! Exact checked-program artifact for the production plain F' base step. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeBaseProgram\n\n\
         open Nightstream.Implementation.R1CS\n\
         open Nightstream.Implementation.R1CS.Program\n\
         open Nightstream.Implementation.R1CS.CheckedProgram\n\n\
         set_option maxRecDepth 524288\n\n\
         def schemaVersion : Nat := {SCHEMA_VERSION}\n\
         def artifactKind : String := \"r1cs/f-prime-base-program\"\n\
         def sourceAnchor : String := \"enforce_f_prime_base_step_circuit\"\n\
         def artifactSha256 : String := \"{row_hash}\"\n\
         def witnessSha256 : String := \"{witness_hash}\"\n\n\
         def inputColumns : List Nat := {}\n\
         def stateInColumns : List Nat := {}\n\
         def stateOutColumns : List Nat := {}\n\
         def xOutColumns : List Nat := {}\n\
         def xOutBitColumns : List Nat := {}\n\
         def rowCount : Nat := {}\n\
         def colCount : Nat := {}\n\
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
         end Nightstream.Implementation.R1CS.FPrimeBaseProgram\n",
        lean_nat_list(base.program.input_columns.iter().copied()),
        lean_nat_list(base.stateInColumns.iter().copied()),
        lean_nat_list(base.stateOutColumns.iter().copied()),
        lean_nat_list(base.xOutColumns.iter().copied()),
        lean_nat_list(base.xOutBitColumns.iter().copied()),
        base.builder.rows(),
        base.builder.cols(),
        base.program.definition_count,
        base.program.check_count,
    )
}

fn render_poseidon_calls(base: &BuiltBase) -> String {
    let calls = base.builder.poseidon2_permutation_audits();
    assert!(!calls.is_empty(), "base F' must contain Poseidon2 calls");
    for call in &calls {
        assert_eq!(call.row_end - call.row_start, 600, "Poseidon2 row count");
        assert_eq!(call.allocated_col_count, 600, "Poseidon2 fresh-column count");
        let mapped_outputs: [usize; 8] = std::array::from_fn(|lane| call.first_allocated_col + (601 + lane - 9));
        assert_eq!(call.output_cols, mapped_outputs, "Poseidon2 output renaming");
    }
    let call_literals = calls
        .iter()
        .map(|call| {
            format!(
                "{{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }}",
                call.row_start,
                call.row_end,
                lean_nat_list(call.input_cols),
                call.first_allocated_col,
            )
        })
        .collect::<Vec<_>>()
        .join("\n, ");
    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeBase.FPrimeBaseProgramArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Call\n\n\
         /-! Generated exact Poseidon2 call-site certificates for the production plain F' base step. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeBasePoseidonCalls\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Call\n\n\
         set_option maxRecDepth 524288\n\n\
         def calls : List Call :=\n[\n  {call_literals}\n]\n\n\
         theorem calls_match_exact_ranges :\n\
             ∀ call ∈ calls, call.Matches FPrimeBaseProgram.rows := by\n\
           native_decide\n\n\
         end Nightstream.Implementation.R1CS.FPrimeBasePoseidonCalls\n"
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

fn render_poseidon_hashes(base: &BuiltBase) -> String {
    let calls = base.builder.poseidon2_permutation_audits();
    let hashes = base.builder.poseidon2_hash_audits();
    assert_eq!(hashes.len(), 3, "plain base F' owns three Poseidon2 sponge calls");

    let trace_literals = hashes
        .iter()
        .map(|hash| {
            let round_literals = hash
                .rounds
                .iter()
                .map(|round| {
                    let call = calls
                        .iter()
                        .find(|call| {
                            call.input_cols == round.permutation_input_cols
                                && call.output_cols == round.permutation_output_cols
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
                round_literals,
                lean_nat_list(hash.output_cols),
            )
        })
        .collect::<Vec<_>>()
        .join("\n  , ");

    format!(
        "import Nightstream.Implementation.R1CS.Ownership.FPrimeBase.FPrimeBaseProgramArtifact\n\
         import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge\n\n\
         /-! Generated exact Poseidon2 sponge certificates for the production plain F' base step. -/\n\n\
         namespace Nightstream.Implementation.R1CS.FPrimeBasePoseidonHashes\n\n\
         open Nightstream.Implementation.R1CS.Poseidon2Sponge\n\n\
         set_option maxRecDepth 524288\n\n\
         def traces : List Trace :=\n[\n  {trace_literals}\n]\n\n\
         theorem traces_accepted :\n\
             traces.all (fun trace => decide (trace.Valid FPrimeBaseProgram.rows)) = true := by\n\
           native_decide\n\n\
         theorem traces_valid :\n\
             ∀ trace ∈ traces, trace.Valid FPrimeBaseProgram.rows := by\n\
           intro trace member\n\
           exact of_decide_eq_true ((List.all_eq_true.mp traces_accepted) trace member)\n\n\
         end Nightstream.Implementation.R1CS.FPrimeBasePoseidonHashes\n"
    )
}

fn compare_or_write_expected(path: &std::path::Path, rendered: &str, drifted: &mut Vec<String>) {
    if std::fs::read_to_string(path).ok().as_deref() != Some(rendered) {
        let expected = path.with_extension("lean.expected");
        std::fs::write(&expected, rendered).expect("write expected Lean artifact");
        drifted.push(expected.display().to_string());
    }
}

#[test]
fn base_program_accepts_honest_witness() {
    let base = build(0, false);
    assert!(
        base.builder.is_satisfied(),
        "honest full base program failed at {:?}",
        base.builder.first_unsatisfied_row()
    );
    assert_eq!(base.program.instructions.len(), base.builder.rows());
    assert_eq!(
        base.program.definition_count + base.program.check_count,
        base.builder.rows()
    );
}

#[test]
fn base_program_rejects_noninitial_counter() {
    let forged = build(1, false);
    assert!(!forged.builder.is_satisfied());
}

#[test]
fn base_program_rejects_forged_chunk_digest() {
    let forged = build(0, true);
    assert!(!forged.builder.is_satisfied());
}

#[test]
fn lean_base_program_artifact_matches_committed_files() {
    let base = build(0, false);
    assert!(base.builder.is_satisfied());
    let (row_hash, witness_hash) = artifact_hashes(&base);
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut drifted = Vec::new();
    let main_path = std::path::PathBuf::from(format!("{manifest_dir}{ARTIFACT_REL_PATH}"));
    compare_or_write_expected(&main_path, &render_main(&base, &row_hash, &witness_hash), &mut drifted);
    for (index, shard) in base.program.instructions.chunks(SHARD_SIZE).enumerate() {
        let path = std::path::PathBuf::from(format!("{manifest_dir}{SHARD_REL_PREFIX}{index}.lean"));
        compare_or_write_expected(&path, &render_shard(index, shard), &mut drifted);
    }
    let poseidon_path = std::path::PathBuf::from(format!("{manifest_dir}{POSEIDON_CALLS_REL_PATH}"));
    compare_or_write_expected(&poseidon_path, &render_poseidon_calls(&base), &mut drifted);
    let poseidon_hashes_path = std::path::PathBuf::from(format!("{manifest_dir}{POSEIDON_HASHES_REL_PATH}"));
    compare_or_write_expected(&poseidon_hashes_path, &render_poseidon_hashes(&base), &mut drifted);
    assert!(
        drifted.is_empty(),
        "generated Lean base-program artifacts drifted: {drifted:?}"
    );
}
