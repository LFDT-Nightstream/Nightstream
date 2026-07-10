//! Authoritative `S_mem + F'` composition tests.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::config;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::nebula::circuit::StepData;
use neo_fold_clean::frontends::nebula::f_prime::{
    enforce_nebula_f_prime_base_step, NebulaFPrimeBranch, NebulaFPrimeRelation, NebulaFPrimeRelationError,
    ROAD_A_COMMITTED_BIT_BUDGET,
};
use neo_fold_clean::frontends::nebula::fingerprint::Gammas;
use neo_fold_clean::frontends::nebula::layout::{encode_delayed_f_prime_suffix, NebulaParams};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::frontends::r1cs_f_prime::lower_field_r1cs;
use neo_fold_clean::lifecycle::{preprocess, Preprocessing};
use neo_fold_clean::paper::construction2::NebulaLane;
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use neo_fold_clean::paper::f_prime::r1cs::{
    encode_x_out_public_bits, FPrimeBaseInputs, FPrimePublicInputLayout, FPrimeStateIn, FPrimeStepConfig,
};
use neo_fold_clean::paper::f_prime::source_image::FPrimeSourceImage;
use neo_fold_clean::paper::nifs::circuit::NifsVCircuitConfig;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsVConfig;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

const TRANSCRIPT_LABEL: &[u8] = b"nebula/f-prime/composed-test";

fn fields(seed: u64) -> [F; 4] {
    std::array::from_fn(|index| F::from_u64(seed + index as u64))
}

fn shape_test_params() -> neo_fold_clean::Params {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 14,
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
    let params = config::r1cs_params(structure.n, structure.m).expect("S_mem params");
    support::install_ajtai_module(&params, &structure);
    preprocess(params, structure, Some(circuit.m_in())).expect("S_mem preprocessing")
}

fn split_nc_config(prep: &Preprocessing) -> SplitNcPiCcsVConfig<'_> {
    let raw_params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs_with(
        prep.structure().n.max(prep.structure().m),
        config::MIN_EFFECTIVE_LAMBDA,
        config::EXTENSION_SAFETY_MARGIN_BITS,
    )
    .expect("raw params");
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&raw_params, prep.structure())
        .expect("engine dimensions");
    let matrix_digest = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(prep.structure(), None);
    let header_bundle = neo_reductions::engines::utils::pi_ccs_header_bundle_digest_fields(
        &raw_params,
        prep.structure(),
        dims,
        &matrix_digest,
    )
    .expect("header bundle");
    SplitNcPiCcsVConfig {
        params: &prep.params,
        structure: prep.structure(),
        header_bundle,
        ell_d: dims.ell_d,
        ell_n: dims.ell_n,
        ell_m: dims.ell_m,
        d_sc: dims.d_sc,
    }
}

#[test]
fn base_step_composes_current_s_mem_and_exports_one_relation() {
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

    let empty_acc = AccumulatorHandle::empty().digest_fields();
    let state = FPrimeStateIn {
        vk_fs_digest: fields(0x500),
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

    let chunk_digest = fields(0x800);
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
        digest_fields_as_digest32(empty_acc),
        digest_fields_as_digest32(empty_acc),
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
            pi_ccs: split_nc_config(&prep),
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
        semantic_state_digest_out: empty_acc,
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
    assert!(
        builder.is_satisfied(),
        "composed field relation must satisfy: {:?}",
        builder.first_unsatisfied_row()
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
#[ignore = "milestone: direct Road A arms must fit the low-norm compilation budget"]
fn road_a_field_arms_must_fit_the_projection_budget() {
    let nebula_params = NebulaParams::new(0, 0, 1, 2, 8).expect("one-step segment params");
    let params = shape_test_params();
    let plan = NebulaPlan::new(nebula_params, vec![7], [0xD8; 32], params.kappa() as usize).expect("tiny Road A plan");
    let audit = NebulaFPrimeRelation::audit_field_shapes(&params, plan.circuit().structure(), &plan)
        .expect("Road A field-shape audit");
    let shared_private_fields = plan.circuit().cols() - plan.circuit().m_in();
    let minimum_committed_bits = audit
        .minimum_low_norm_committed_bits(shared_private_fields)
        .expect("compatible three-arm public and shared prefixes");

    assert!(
        minimum_committed_bits <= ROAD_A_COMMITTED_BIT_BUDGET,
        "after one-hot branch overlay, the field-native relation still requires at least {minimum_committed_bits} committed bits under the optimistic one-bit-per-field bound: {audit:?}"
    );
}

#[test]
fn road_a_fixed_point_compile_fails_closed_before_oversized_lowering() {
    let nebula_params = NebulaParams::new(0, 0, 1, 2, 8).expect("one-step segment params");
    let params = shape_test_params();
    let plan = NebulaPlan::new(nebula_params, vec![7], [0xD8; 32], params.kappa() as usize).expect("tiny Road A plan");

    match NebulaFPrimeRelation::compile_fixed_point(&params, &plan) {
        Err(NebulaFPrimeRelationError::CompileBudgetExceeded {
            minimum_bits,
            budget_bits,
        }) => {
            assert_eq!(minimum_bits, 30_083_645);
            assert_eq!(budget_bits, ROAD_A_COMMITTED_BIT_BUDGET);
        }
        Err(other) => panic!("expected budget guard, got {other}"),
        Ok(_) => panic!("oversized relation reached low-norm compilation"),
    }
}
