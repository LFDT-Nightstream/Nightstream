//! Authoritative `S_mem + F'` composition tests.

use std::collections::BTreeMap;
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

fn print_width_owners(relation: &NebulaFPrimeRelation) {
    let audit = relation
        .low_norm_width_audit()
        .expect("live relation has a width audit");
    let snapshot = relation
        .selective_snapshot()
        .expect("live relation has a selective compiler snapshot");
    eprintln!(
        "FPRIME_CANONICAL_OPENINGS counts={:?}",
        snapshot
            .compiler_audit()
            .canonical_openings()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>()
    );
    eprintln!(
        "FPRIME_WIDTH total={} constant={} public={} selectors={} alignment={} shared_private={} branch_start={}",
        audit.total_coordinates,
        audit.constant_coordinate,
        audit.public_coordinates,
        audit.selector_coordinates,
        audit.alignment_padding,
        audit.shared_private_coordinates,
        audit.branch_start,
    );
    for (arm_index, arm) in audit.arms.iter().enumerate() {
        let mut by_path = BTreeMap::<&str, (usize, usize, usize, usize, usize, usize)>::new();
        let mut by_component = BTreeMap::<&str, usize>::new();
        for stage in &arm.physical_stages {
            let entry = by_path.entry(stage.path).or_default();
            entry.0 += stage.allocated_coordinates;
            entry.1 += stage.source_column_count;
            entry.2 += stage.eliminated_columns;
            entry.3 += stage.unit_columns;
            entry.4 += stage.balanced_columns;
            entry.5 += stage.binary_columns;
            let component = if stage.path.starts_with("nifs.pi_ccs") {
                "pi_ccs"
            } else if stage.path.starts_with("nifs.pi_rlc") {
                "pi_rlc"
            } else if stage.path.contains("pi_dec") {
                "pi_dec"
            } else if stage.path.starts_with("fprime.recursive.step.accumulator") {
                "accumulator"
            } else if stage.path.starts_with("fprime.recursive.step.nebula") {
                "nebula"
            } else {
                "other"
            };
            *by_component.entry(component).or_default() += stage.allocated_coordinates;
        }
        let mut owners = by_path.into_iter().collect::<Vec<_>>();
        owners.sort_unstable_by(|left, right| right.1 .0.cmp(&left.1 .0).then_with(|| left.0.cmp(right.0)));
        eprintln!(
            "FPRIME_WIDTH_ARM arm={} source_columns={} eliminated_columns={} unit_columns={} balanced_columns={} binary_columns={} retained_before_aliases={} decomposition_aliases={} equality_aliases={} branch_coordinates={} derived_product_sums={} derived_coordinates={} total_branch_coordinates={}",
            arm_index,
            arm.branch_source_columns,
            arm.eliminated_columns,
            arm.unit_columns,
            arm.balanced_columns,
            arm.binary_columns,
            arm.retained_coordinates_before_aliases,
            arm.decomposition_aliases,
            arm.equality_aliases,
            arm.branch_coordinates,
            arm.derived_product_sums,
            arm.derived_coordinates,
            arm.total_branch_coordinates,
        );
        for (component, coordinates) in by_component {
            eprintln!(
                "FPRIME_WIDTH_COMPONENT arm={} coordinates={} component={}",
                arm_index, coordinates, component,
            );
        }
        eprintln!(
            "FPRIME_WIDTH_TRACES arm={} poseidon2_permutations={} poseidon2_columns={} poseidon2_coordinates={} polynomial_columns={} polynomial_coordinates={} product_sum_columns={} product_sum_coordinates={} product_sum_internal_columns={} product_sum_internal_coordinates={}",
            arm_index,
            arm.traces.poseidon2_permutations,
            arm.traces.poseidon2_columns,
            arm.traces.poseidon2_coordinates,
            arm.traces.polynomial_evaluation_columns,
            arm.traces.polynomial_evaluation_coordinates,
            arm.traces.product_sum_columns,
            arm.traces.product_sum_coordinates,
            arm.traces.product_sum_internal_columns,
            arm.traces.product_sum_internal_coordinates,
        );
        for (path, (coordinates, source_columns, eliminated_columns, unit_columns, balanced_columns, binary_columns)) in
            owners.into_iter().take(32)
        {
            eprintln!(
                "FPRIME_WIDTH_OWNER arm={} coordinates={} source_columns={} eliminated_columns={} unit_columns={} balanced_columns={} binary_columns={} path={}",
                arm_index,
                coordinates,
                source_columns,
                eliminated_columns,
                unit_columns,
                balanced_columns,
                binary_columns,
                path,
            );
        }
    }
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

fn shape_test_radix_four_params() -> neo_fold_clean::Params {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        1 << 25,
        4,
        7,
        neo_params::goldilocks_paper_b2::T,
        2,
        20,
    )
    .expect("reduced radix-four fixed-point shape parameters");
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
        params: prep.params(),
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
        prep.params(),
        prep.structure(),
        public_input_len,
        LaneCommitmentMode::Nebula,
    )
    .expect("canonical zero running accumulator");
    let output_acc =
        AccumulatorHandle::from_running_parts(2, &zero_running.claims, zero_running.parent_authority.as_ref())
            .digest_fields();

    let empty_acc = AccumulatorHandle::empty().digest_fields();
    let state = FPrimeStateIn {
        vk_fs_digest: digest32_as_fields(prep.verifier_key().digest()),
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
        f_prime_chunk_public_digest_for_uniform_shape(0, 1, D, prep.params().kappa() as usize, public_input_len);
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
        b: prep.params().b(),
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
    let fixed = NebulaFPrimeRelation::compile(&shape, &shape, &plan).expect("fixed-shape Nebula F'");
    assert_eq!(
        fixed
            .low_norm_width_audit()
            .expect("live relation has a width audit")
            .selector_coordinates,
        2,
        "bootstrap and steady recursion must share one physical selector arm"
    );
    assert_eq!(
        fixed
            .encode(NebulaFPrimeBranch::BootstrapRecursive, &field_assignment)
            .expect("bootstrap encoding"),
        fixed
            .encode(NebulaFPrimeBranch::Recursive, &field_assignment)
            .expect("recursive encoding"),
        "bootstrap and steady recursion must encode through the same relation arm"
    );
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
    print_width_owners(&relation);
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
        relation
            .low_norm_width_audit()
            .expect("live relation has a width audit")
            .selector_coordinates,
        2,
        "bootstrap and steady recursion must share one physical selector arm"
    );
    assert_eq!(
        relation
            .low_norm_width_audit()
            .expect("live relation has a width audit")
            .arms
            .len(),
        2,
        "the fixed relation must contain only its two distinct R1CS arms"
    );
    let recursive_width = &relation
        .low_norm_width_audit()
        .expect("live relation has a width audit")
        .arms[1];
    assert!(
        !recursive_width.physical_stages.is_empty(),
        "the composed recursive relation must retain complete physical-stage provenance"
    );
    assert_eq!(
        recursive_width
            .physical_stages
            .iter()
            .map(|stage| stage.allocated_coordinates)
            .sum::<usize>(),
        recursive_width.branch_coordinates,
        "exclusive recursive stages must own every branch coordinate exactly once"
    );
    assert_eq!(
        (relation.structure().n, relation.structure().m),
        (4_113_183, 14_543_442),
        "reduced-profile rectangular verifier fixed point drifted"
    );
}

#[test]
fn reduced_radix_four_profile_fixed_point_stabilizes() {
    let _guard = relation_audit_guard();
    let nebula_params = NebulaParams::new(0, 0, 1, 2, 8).expect("one-step segment params");
    let params = shape_test_radix_four_params();
    let plan =
        NebulaPlan::new(nebula_params, vec![7], [0xD8; 32], params.kappa() as usize).expect("tiny fixed-shape plan");

    let relation = NebulaFPrimeRelation::compile_fixed_point(&params, &plan)
        .expect("the radix-four verifier requires one stabilized authoritative relation");
    print_width_owners(&relation);
    eprintln!(
        "radix-four fixed-shape F' relation: {} coordinates, {} rows, {} matrices, degree {}",
        relation.structure().m,
        relation.structure().n,
        relation.structure().t(),
        relation.structure().max_degree(),
    );
    assert_eq!((params.b(), params.k_rho(), params.big_b()), (4, 7, 16_384));
    assert_eq!(
        (
            relation.structure().n,
            relation.structure().m,
            relation.structure().t(),
            relation.structure().max_degree(),
        ),
        (4_818_211, 7_048_026, 13, 8),
        "reduced radix-four fixed point drifted"
    );
    assert!(
        relation.structure().n <= 1 << 24,
        "radix-four rows exceed the target joint domain"
    );
    assert!(
        relation.structure().m <= 1 << 24,
        "radix-four columns exceed the target joint domain"
    );
}

#[test]
fn reduced_profile_constraint_source_audit_is_exact() {
    let _guard = relation_audit_guard();
    let memory_params = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step segment");
    let params = minimal_chain_params();
    let plan = NebulaPlan::new(memory_params, vec![7], [0xD9; 32], params.kappa() as usize).expect("plan");
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover exact Nebula source arms");

    assert!(audit.fixed_point_rounds() > 0);
    assert_eq!(audit.compiler_audit().rows().total_rows(), audit.verifier_rows());
    assert_eq!(
        audit.compiler_audit().layout().total_columns(),
        audit.verifier_columns()
    );
    assert_eq!(audit.compiler_audit().rows().arms().len(), 2);
    assert_eq!(audit.compiler_audit().source_arm_physical_stages().len(), 2);
    assert!(!audit
        .arm(NebulaFPrimeBranch::Base)
        .physical_stage_ranges()
        .is_empty());
    assert!(!audit
        .arm(NebulaFPrimeBranch::Recursive)
        .physical_stage_ranges()
        .is_empty());
    assert!(std::ptr::eq(
        audit.arm(NebulaFPrimeBranch::BootstrapRecursive),
        audit.arm(NebulaFPrimeBranch::Recursive),
    ));

    for (arm, mapping) in audit
        .physical_arms()
        .iter()
        .zip(audit.compiler_audit().rows().arms())
    {
        assert_eq!(
            mapping
                .source_runs()
                .iter()
                .map(|run| run.source_rows().len())
                .sum::<usize>(),
            arm.n,
        );
    }

    let first_row = audit.compiler_audit().rows().prefix_rows().start;
    let projected = audit
        .audit_selective_rows(&[first_row])
        .expect("project one exact final row");
    assert_eq!(projected.row_artifacts()[0].emitted_row(), first_row);
    assert_eq!(projected.compiler_audit(), audit.compiler_audit());
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
    assert!(prep.preprocessing().enforces_terminal_induction());
    let relation_artifact = prep
        .relation_artifact_json()
        .expect("exact recursive relation artifact");
    let relation_receipt = prep
        .validate_relation_artifact_json(&relation_artifact)
        .expect("validate exact recursive relation artifact");
    assert_eq!(relation_receipt.logical_rows(), prep.relation().structure().n as u64);
    assert_eq!(
        relation_receipt.assignment_fields(),
        prep.relation().structure().m as u64
    );
    assert_eq!(
        relation_receipt.public_field_width(),
        Some(prep.relation().public_input_len() as u64)
    );

    let mut memory = Memory::new(memory_params, &rom).expect("memory");
    let trace0 = {
        let mut segment = memory.begin_segment().expect("segment 0");
        segment.write(true, 0, 5).expect("RAM write");
        segment.finish().expect("trace 0")
    };

    let mut chain = NebulaFPrimeChainBuilder::new(&prep);
    let witnesses = chain
        .append_segment_with_constraint_witness_audit(&trace0)
        .expect("base arm with source audit");
    assert_eq!(witnesses.len(), 1);
    assert_eq!(witnesses[0].branch(), NebulaFPrimeBranch::Base);
    assert_eq!(witnesses[0].source_assignment().first(), Some(&F::ONE));
    assert_eq!(
        witnesses[0].source_assignment().len(),
        prep.relation().field_arm_shapes()[0].columns,
    );
    let audit = chain.into_audit().expect("one appended segment");

    assert_eq!(audit.proof.state.chunk_count, 1);
    let lane = audit.proof.state.nebula.as_ref().expect("lane");
    assert!(lane.is_closed());
    assert_eq!(lane.seg_idx, 0, "the first segment remains delayed until the next fold");
    assert_eq!(lane.sp, [0, 0]);
}
