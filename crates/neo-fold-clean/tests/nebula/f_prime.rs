//! Authoritative `S_mem + F'` composition tests.

use std::sync::{Mutex, MutexGuard};

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::config;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::nebula::circuit::StepData;
use neo_fold_clean::frontends::nebula::f_prime::{
    enforce_nebula_f_prime_base_step, NebulaFPrimeBranch, NebulaFPrimeChainBuilder, NebulaFPrimePreprocessing,
    NebulaFPrimeRelation,
};
use neo_fold_clean::frontends::nebula::fingerprint::Gammas;
use neo_fold_clean::frontends::nebula::layout::{encode_delayed_f_prime_suffix, NebulaParams};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::frontends::r1cs_f_prime::lower_field_r1cs;
use neo_fold_clean::lifecycle::{preprocess, Preprocessing};
use neo_fold_clean::paper::construction2::{LaneCommitmentMode, NebulaLane, RunningInstance};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest_for_uniform_shape,
    state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_x_out_public_bits, FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeStateIn, FPrimeStepConfig,
};
use neo_fold_clean::paper::f_prime::source_image::FPrimeSourceImage;
use neo_fold_clean::paper::nifs::circuit::NifsVCircuitConfig;
use neo_fold_clean::paper::reductions::pi_ccs_circuit::PiCcsVerifierConfig;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

const TRANSCRIPT_LABEL: &[u8] = b"nebula/f-prime/composed-test";

static RELATION_AUDIT_LOCK: Mutex<()> = Mutex::new(());

fn relation_audit_guard() -> MutexGuard<'static, ()> {
    RELATION_AUDIT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn fields(seed: u64) -> [F; 4] {
    std::array::from_fn(|index| F::from_u64(seed + index as u64))
}

fn shape_test_params() -> neo_fold_clean::Params {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 25,
        2,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        2,
        20,
    )
    .expect("reduced fixed-point shape parameters");
    neo_fold_clean::Params::test_only_from_neo_params(inner)
}

fn preprocessing(circuit: &neo_fold_clean::frontends::nebula::circuit::SMemCircuit) -> Preprocessing {
    let structure = circuit.structure().clone();
    let params =
        config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree()).expect("S_mem params");
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(circuit.m_in())).expect("S_mem preprocessing")
}

fn pi_ccs_config(prep: &Preprocessing) -> PiCcsVerifierConfig<'_> {
    PiCcsVerifierConfig {
        params: &prep.params,
        structure: prep.structure().into(),
        matrix_digest: prep.pi_ccs_header_bundle(),
    }
}

#[test]
fn base_step_composes_current_s_mem_and_exports_one_relation() {
    let _guard = relation_audit_guard();
    let params = NebulaParams::test_profile();
    let circuit = neo_fold_clean::frontends::nebula::circuit::SMemCircuit::new(params);
    let prep = preprocessing(&circuit);
    let rom: Vec<u32> = (0..params.rom_cells())
        .map(|index| index as u32 + 10)
        .collect();
    let fixed_params = support::r1cs_compiler_fixtures::tiny_params();
    let plan = NebulaPlan::new(params, rom.clone(), [0x31; 32], fixed_params.kappa() as usize).expect("Nebula plan");
    let current_d_pre = [fields(0x100), fields(0x200), fields(0x300)];
    let nebula = plan.config();
    let public_input_len =
        FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(params.stack_shape())).total_len();
    let zero_running = RunningInstance::canonical_zero(
        &prep.params,
        prep.structure(),
        public_input_len,
        LaneCommitmentMode::Nebula,
    )
    .expect("canonical zero running accumulator");
    let output_acc =
        AccumulatorHandle::from_running_parts(&zero_running.claims, zero_running.parent_authority.as_ref())
            .digest_fields();

    let empty_acc = AccumulatorHandle::empty().digest_fields();
    let state = FPrimeStateIn {
        vk_fs_digest: digest32_as_fields(prep.vk.digest()),
        pi_ccs_header_bundle: prep.pi_ccs_header_bundle(),
        chunk_count_in: 0,
        step_count_in: 0,
        z_0: fields(0x600),
        z_i_in: fields(0x600),
        pc: 1,
        semantic_state_digest_in: empty_acc,
        acc_digest_in: empty_acc,
        public_trace_in: fields(0x700),
        nebula: Some(NebulaLane::base(&nebula)),
    };
    let mut opened = state.nebula.clone().expect("base lane");
    opened
        .open_segment(
            &nebula,
            digest_fields_as_digest32(state.vk_fs_digest),
            digest_fields_as_digest32(state.z_i_in),
            digest_fields_as_digest32(state.acc_digest_in),
            current_d_pre,
        )
        .expect("derive current gamma");
    let gamma = opened.gamma.expect("open gamma");

    let mut memory = Memory::new(params, &rom).expect("memory");
    let mut segment = memory.begin_segment().expect("segment");
    segment.write(true, 0, 9).expect("RAM write");
    let trace = segment.finish().expect("trace");
    let data = StepData {
        seg_idx: 0,
        idx: 0,
        ts_in: trace.ts_in,
        h_in: [K::ONE; 4],
        sp_in: [0; 2],
        ops: trace.step_ops(0),
        is_cells: &trace.is_cells[..params.b_scan],
        fs_cells: &trace.fs_cells[..params.b_scan],
    };
    let (s_mem_assignment, step_x) = circuit
        .witness(
            &Gammas {
                gamma1: gamma[0],
                gamma2: gamma[1],
            },
            &data,
        )
        .expect("S_mem witness");

    let chunk_digest =
        f_prime_chunk_public_digest_for_uniform_shape(0, 1, D, prep.params.kappa() as usize, public_input_len);
    let expected_x_out = digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        digest_fields_as_digest32(state.vk_fs_digest),
        state.pi_ccs_header_bundle,
        prep.structure_digest(),
        1,
        1,
        digest_fields_as_digest32(state.z_0),
        digest_fields_as_digest32(chunk_digest),
        state.pc,
        digest_fields_as_digest32(output_acc),
        digest_fields_as_digest32(output_acc),
        digest_fields_as_digest32(chunk_digest),
        Some(state.nebula.as_ref().expect("lane").digest()),
    ));
    let mut source = FPrimeSourceImage::new();
    let chunk_count_in_word = source.push_u64_le(0);
    let step_count_in_word = source.push_u64_le(0);
    let pc_word = source.push_u64_le(1);
    let public_x_out_bits = source.push_enc_inst(expected_x_out);
    let cfg = FPrimeStepConfig {
        nifs: NifsVCircuitConfig {
            pi_ccs: pi_ccs_config(&prep),
        },
        b: prep.params.b(),
        transcript_label: TRANSCRIPT_LABEL,
        public_input_layout: FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(
            params.stack_shape(),
        )),
        nebula: Some(&nebula),
        state_x_out_digest_mode: StateXOutDigestMode::Stateless,
    };
    let inputs = FPrimeBaseInputs {
        state,
        chunk_digest,
        semantic_state_digest_out: output_acc,
        rows_in_chunk: 1,
        source_image: &source,
        chunk_count_in_word,
        step_count_in_word,
        pc_word,
        public_x_out_bits,
    };

    let mut builder = R1csBuilder::new();
    let output = enforce_nebula_f_prime_base_step(
        &mut builder,
        &circuit,
        &s_mem_assignment,
        Some(current_d_pre),
        &cfg,
        &inputs,
    )
    .expect("composed base F'");
    let actual_x_out = output
        .f_prime
        .x_out
        .map(|wire| builder.witness()[wire.col()]);
    assert_eq!(
        actual_x_out, expected_x_out,
        "native and circuit base-step x_out digests must match"
    );
    let first_unsatisfied = builder.first_unsatisfied_row();
    let failed_owners = first_unsatisfied
        .into_iter()
        .flat_map(|row| {
            builder
                .row_family_ranges()
                .iter()
                .filter(move |family| (family.row_start..family.row_end).contains(&row))
                .map(|family| family.name)
        })
        .collect::<Vec<_>>();
    assert!(
        first_unsatisfied.is_none(),
        "composed field relation must satisfy; first failed row: {first_unsatisfied:?}; owners: {failed_owners:?}"
    );

    let mut witness_only = R1csBuilder::new_witness_only();
    let witness_only_output = enforce_nebula_f_prime_base_step(
        &mut witness_only,
        &circuit,
        &s_mem_assignment,
        Some(current_d_pre),
        &cfg,
        &inputs,
    )
    .expect("witness-only composed base F'");
    assert_eq!(witness_only.rows(), builder.rows(), "witness-only row schedule drifted");
    assert_eq!(
        witness_only.cols(),
        builder.cols(),
        "witness-only column schedule drifted"
    );
    assert_eq!(
        witness_only.witness(),
        builder.witness(),
        "witness-only assignment drifted"
    );
    assert_eq!(
        witness_only_output
            .public_outputs()
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>(),
        output
            .public_outputs()
            .iter()
            .map(|wire| wire.col())
            .collect::<Vec<_>>(),
        "witness-only public wire schedule drifted"
    );
    let witness = builder.witness();
    let public_outputs = output.public_outputs();
    let public_values: Vec<F> = public_outputs
        .iter()
        .map(|wire| witness[wire.col()])
        .collect();
    let expected_suffix =
        encode_delayed_f_prime_suffix(&step_x, params.stack_shape(), Some(current_d_pre)).expect("suffix encoding");
    assert_eq!(
        &public_values[..encode_x_out_public_bits(expected_x_out).len()],
        encode_x_out_public_bits(expected_x_out)
    );
    assert_eq!(
        &public_values[encode_x_out_public_bits(expected_x_out).len()..],
        expected_suffix
    );

    let pad_wire = output.s_mem[circuit.op_slot_column(0)];
    let original = builder.witness()[pad_wire.col()];
    builder.tamper_witness(pad_wire.col(), F::ONE - original);
    assert!(!builder.is_satisfied(), "current S_mem semantics must be load-bearing");
    builder.tamper_witness(pad_wire.col(), original);
    assert!(builder.is_satisfied());

    let lowered = lower_field_r1cs(builder, &public_outputs).expect("export composed relation");
    assert_eq!(lowered.shape().m_in, 1 + public_outputs.len());
    lowered
        .shape()
        .is_satisfied_by(lowered.assignment())
        .expect("one exported field relation must preserve composition");

    let (shape, field_assignment) = lowered.into_parts();
    let fixed = NebulaFPrimeRelation::compile(&shape, &shape, &shape, &plan).expect("fixed-shape Nebula F'");
    support::install_ajtai_module(&fixed_params, fixed.structure());
    let fixed_prep = preprocess(fixed_params, fixed.structure().clone(), Some(fixed.public_input_len()))
        .expect("fixed relation preprocessing")
        .with_nebula(fixed.nebula_config().clone());
    let base_instance = fixed
        .build_instance(&fixed_prep, NebulaFPrimeBranch::Base, &field_assignment)
        .expect("base fixed instance");
    let expected_adv = plan
        .scheme()
        .commit_bits(
            &params.encode_ops_lane(trace.step_ops(0)).expect("ops lane"),
            &params
                .encode_scan_lane(&trace.is_cells[..params.b_scan])
                .expect("initial-state lane"),
            &params
                .encode_scan_lane(&trace.fs_cells[..params.b_scan])
                .expect("final-state lane"),
        )
        .expect("precommitted lanes");
    assert_eq!(
        base_instance.claim.adv,
        Some(expected_adv),
        "the remapped fixed assignment must preserve the precommitted S_mem lanes"
    );
    let recursive_instance = fixed
        .build_instance(&fixed_prep, NebulaFPrimeBranch::Recursive, &field_assignment)
        .expect("recursive fixed instance");
    assert_eq!(
        base_instance.claim.adv, recursive_instance.claim.adv,
        "base and recursive branches must commit the same shared S_mem lanes"
    );
    assert!(
        fixed
            .nebula_config()
            .scheme
            .open_matches(base_instance.claim.adv.as_ref().expect("adv"), &base_instance.witness.Z)
            .expect("lane opening"),
        "the remapped lane scheme must open against the fixed assignment"
    );
}

#[test]
fn fixed_shape_accepts_the_ring_padded_fresh_carrier() {
    let _guard = relation_audit_guard();
    let nebula_params = NebulaParams::new(0, 0, 1, 2, 8).expect("one-step segment params");
    let params = shape_test_params();
    let plan =
        NebulaPlan::new(nebula_params, vec![7], [0xD8; 32], params.kappa() as usize).expect("tiny fixed-shape plan");
    let layout = FPrimePublicInputLayout::with_suffix(delayed_nebula_public_suffix_len(plan.config().stacks));

    assert_ne!(layout.logical_len() % D, 0, "fixture must exercise carrier padding");
    assert_eq!(layout.total_len() % D, 0, "SuperNeo carrier must be ring-aligned");
    NebulaFPrimeRelation::audit_field_shapes(&params, plan.circuit().structure(), &plan)
        .expect("shape synthesis must use the complete ring-padded fresh carrier");
}

#[test]
fn reduced_profile_fixed_point_stabilizes() {
    let _guard = relation_audit_guard();
    let nebula_params = NebulaParams::new(0, 0, 1, 2, 8).expect("one-step segment params");
    let params = shape_test_params();
    let plan =
        NebulaPlan::new(nebula_params, vec![7], [0xD8; 32], params.kappa() as usize).expect("tiny fixed-shape plan");

    let relation = NebulaFPrimeRelation::compile_fixed_point(&params, &plan)
        .expect("the verifier requires one stabilized, selectively lowered authoritative relation");
    eprintln!(
        "fixed-shape F' relation: {} coordinates, {} rows, {} matrices, degree {}",
        relation.structure().m,
        relation.structure().n,
        relation.structure().t(),
        relation.structure().max_degree(),
    );
    assert_ne!(
        relation.structure().n,
        relation.structure().m,
        "the padded one-joint fixed point must remain rectangular"
    );
    assert_eq!(
        relation.structure().t(),
        13,
        "the application relation must not duplicate the virtual PiCCS identity matrix"
    );
    assert!(
        !relation.structure().matrices[0].is_identity(),
        "the first application matrix must not duplicate the virtual PiCCS identity matrix"
    );
    assert_eq!(
        (relation.structure().n, relation.structure().m),
        (8_252_423, 15_319_206),
        "reduced-profile rectangular verifier fixed point drifted"
    );
}

fn minimal_chain_params() -> neo_fold_clean::Params {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        2,
        1,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("minimal chain parameters satisfy the exact RLC guard");
    neo_fold_clean::Params::test_only_from_neo_params(inner)
}

#[test]
fn canonical_encoder_appends_one_memory_segment() {
    run_single_segment_append();
}

fn run_single_segment_append() {
    let memory_params = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step segment");
    assert_eq!(memory_params.steps_per_segment(), 1);
    let params = minimal_chain_params();
    let rom = [10];
    let plan = NebulaPlan::new(memory_params, rom.to_vec(), [0xD3; 32], params.kappa() as usize).expect("plan");
    let prep = NebulaFPrimePreprocessing::new_seeded(params, plan, 0xD3D3_0001).expect("fixed preprocessing");
    assert!(prep.prep.enforces_terminal_induction());

    let mut memory = Memory::new(memory_params, &rom).expect("memory");
    let trace0 = {
        let mut segment = memory.begin_segment().expect("segment 0");
        segment.write(true, 0, 5).expect("RAM write");
        segment.finish().expect("trace 0")
    };

    let mut chain = NebulaFPrimeChainBuilder::new(&prep);
    chain.append_segment(&trace0).expect("base arm");
    let audit = chain.into_audit().expect("one appended segment");

    assert_eq!(audit.proof.state.chunk_count, 1);
    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert!(lane.is_closed());
    assert_eq!(lane.seg_idx, 0, "the first segment remains delayed until the next fold");
    assert_eq!(lane.sp, [0, 0]);
}
