//! Full `x_out` envelope checks for phased Nebula F-prime circuits.

use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_fold_clean::frontends::nebula::circuit::StepData;
use neo_fold_clean::frontends::nebula::f_prime::{
    enforce_streaming_phase_semantic_digest, enforce_streaming_state_x_out_bits,
    prepare_streaming_lifecycle_preprocessing, production_phase_envelope_link_profile,
    production_pi_rlc_family_body_source_arms, streaming_phase_semantic_digest,
    synthesize_streaming_lifecycle_source_arms, synthesize_streaming_lifecycle_source_arms_with_recursive_assignment,
    NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingLifecycleArm, NebulaFPrimeStreamingProgramAudit,
    NebulaFPrimeStreamingPublicLayout, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY,
};
use neo_fold_clean::frontends::nebula::fingerprint::Gammas;
use neo_fold_clean::frontends::nebula::layout::{encode_delayed_f_prime_suffix, NebulaParams};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::frontends::r1cs_f_prime::{lower_field_r1cs, SparseR1cs};
use neo_fold_clean::paper::construction2::{LaneCommitmentMode, NebulaLane, RunningInstance};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, digest_fields_as_digest32, f_prime_chunk_public_digest_for_uniform_shape,
    initial_boundary_digest, nebula_lane_chains, state_x_out_digest_with_mode, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::digest_circuit::StateXOutDigestInputs;
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use neo_fold_clean::paper::f_prime::r1cs::encode_x_out_public_bits;
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::CcsInstance;
use neo_math::{D, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn fields(seed: u64) -> [F; 4] {
    std::array::from_fn(|lane| F::from_u64(seed + lane as u64 * 17))
}

fn alloc_fields(builder: &mut R1csBuilder, values: [F; 4]) -> [Var; 4] {
    values.map(|value| builder.alloc(value))
}

fn expected_bits(lanes: [F; 4]) -> [u64; 256] {
    std::array::from_fn(|index| {
        let lane = index / 64;
        let bit = index % 64;
        (lanes[lane].as_canonical_u64() >> bit) & 1
    })
}

fn phase_envelope_source(payload_len: usize, omitted_family: Option<&'static str>) -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    let public = (0..640)
        .map(|_| {
            let bit = builder.alloc(F::ZERO);
            enforce_bit(&mut builder, bit);
            bit
        })
        .collect::<Vec<_>>();
    for (name, width) in [
        (STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY, 4),
        (STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY, payload_len),
        (STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, 4),
        (STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY, payload_len),
    ] {
        if omitted_family == Some(name) {
            continue;
        }
        let start = builder.cols();
        let fields = builder.alloc_vec(&vec![F::ZERO; width]);
        builder.record_column_family(name, start);
        for field in fields {
            builder.enforce_zero(&Lc::from_var(field));
        }
    }
    lower_field_r1cs(builder, &public)
        .expect("lower phase-envelope source fixture")
        .into_parts()
        .0
}

#[test]
fn streaming_state_envelope_matches_canonical_state_x_out() {
    let vk_fs = fields(101);
    let header = fields(211);
    let structure = fields(307);
    let initial_boundary = fields(401);
    let current_boundary = fields(503);
    let family_state_digest = fields(601);
    let construction2_acc = fields(701);
    let public_trace = current_boundary;
    let nebula_lane = fields(809);
    let chunk_count = 13;
    let step_count = 29;
    let pc = 1;

    let expected = digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateful,
        digest_fields_as_digest32(vk_fs),
        header,
        &structure,
        chunk_count,
        step_count,
        digest_fields_as_digest32(initial_boundary),
        digest_fields_as_digest32(current_boundary),
        pc,
        digest_fields_as_digest32(family_state_digest),
        digest_fields_as_digest32(construction2_acc),
        digest_fields_as_digest32(public_trace),
        Some(nebula_lane),
    ));

    let mut builder = R1csBuilder::new();
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_fields(&mut builder, vk_fs),
        pi_ccs_header_bundle: alloc_fields(&mut builder, header),
        structure_digest: alloc_fields(&mut builder, structure),
        chunk_count: builder.alloc(F::from_u64(chunk_count)),
        step_count: builder.alloc(F::from_u64(step_count)),
        initial_boundary: alloc_fields(&mut builder, initial_boundary),
        current_boundary: alloc_fields(&mut builder, current_boundary),
        pc: builder.alloc(F::from_u64(pc)),
        semantic_acc: alloc_fields(&mut builder, family_state_digest),
        construction2_acc: alloc_fields(&mut builder, construction2_acc),
        public_trace: alloc_fields(&mut builder, public_trace),
    };
    let lane = alloc_fields(&mut builder, nebula_lane);
    let bits = enforce_streaming_state_x_out_bits(&mut builder, &inputs, lane);

    assert!(builder.is_satisfied(), "full state envelope must satisfy");
    let actual = bits.map(|wire| builder.witness()[wire.col()].as_canonical_u64());
    assert_eq!(actual, expected_bits(expected));
}

#[test]
fn streaming_state_envelope_rejects_a_changed_public_bit() {
    let mut builder = R1csBuilder::new();
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_fields(&mut builder, fields(101)),
        pi_ccs_header_bundle: alloc_fields(&mut builder, fields(211)),
        structure_digest: alloc_fields(&mut builder, fields(307)),
        chunk_count: builder.alloc(F::from_u64(13)),
        step_count: builder.alloc(F::from_u64(29)),
        initial_boundary: alloc_fields(&mut builder, fields(401)),
        current_boundary: alloc_fields(&mut builder, fields(503)),
        pc: builder.alloc(F::ONE),
        semantic_acc: alloc_fields(&mut builder, fields(601)),
        construction2_acc: alloc_fields(&mut builder, fields(701)),
        public_trace: alloc_fields(&mut builder, fields(503)),
    };
    let lane = alloc_fields(&mut builder, fields(809));
    let bits = enforce_streaming_state_x_out_bits(&mut builder, &inputs, lane);
    assert!(builder.is_satisfied());

    let first = bits[0];
    let changed = if builder.witness()[first.col()] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    builder.tamper_witness(first.col(), changed);
    assert!(!builder.is_satisfied(), "a changed public x_out bit must fail");
}

#[test]
fn streaming_phase_semantic_envelope_replays_exact_payload_bits() {
    let local = fields(901);
    let payload = [F::ZERO, F::ONE, F::ONE, F::ZERO, F::ONE];
    let expected = streaming_phase_semantic_digest(local, &payload);
    let mut builder = R1csBuilder::new();
    let local_vars = alloc_fields(&mut builder, local);
    let payload_vars = builder.alloc_vec(&payload);
    let actual = enforce_streaming_phase_semantic_digest(&mut builder, local_vars, &payload_vars, true);

    assert!(builder.is_satisfied(), "phase semantic envelope must satisfy");
    assert_eq!(actual.map(|wire| builder.witness()[wire.col()]), expected);

    builder.tamper_witness(payload_vars[1].col(), F::ZERO);
    assert!(
        !builder.is_satisfied(),
        "changing an authenticated delayed payload bit must fail"
    );
}

#[test]
fn streaming_lifecycle_source_arms_own_complete_context_and_fold_stages() {
    let reference_params = Params::production();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xD9; 32], reference_params.kappa() as usize).expect("Nebula plan");
    let params = Params::for_ccs_shape(
        plan.circuit().structure().n,
        plan.circuit().structure().m,
        plan.circuit().structure().t(),
        plan.circuit().structure().max_degree(),
    )
    .expect("shape-specific Appendix B.2 parameters");
    let log = neo_fold_clean::frontends::direct_ccs::ajtai::setup_seeded(
        &params,
        plan.circuit().structure(),
        0x5354_5245_414d,
    );
    let preprocessing = neo_fold_clean::lifecycle::preprocess_with_test_log(
        params,
        plan.circuit().structure().clone(),
        log,
        Some(NebulaFPrimeStreamingPublicLayout::production().columns()),
    )
    .expect("verifier-owned lifecycle preprocessing");
    let preprocessing =
        prepare_streaming_lifecycle_preprocessing(preprocessing, &plan).expect("fixed streaming lifecycle policy");
    let arms = synthesize_streaming_lifecycle_source_arms(&preprocessing, &plan)
        .expect("synthesize exact streaming lifecycle source rows");
    let public = NebulaFPrimeStreamingPublicLayout::production();

    assert_eq!(public.logical_columns(), 641);
    assert_eq!(public.columns(), 648);
    for arm_kind in [
        NebulaFPrimeStreamingLifecycleArm::Base,
        NebulaFPrimeStreamingLifecycleArm::Recursive,
    ] {
        let arm = arms.arm(arm_kind);
        assert_eq!(arm.m_in, public.logical_columns());
        assert_eq!(
            arm.physical_stage_ranges()
                .iter()
                .map(|stage| stage.rows().len())
                .sum::<usize>(),
            arm.n,
            "exclusive physical stages must own every source row"
        );
        let x_out_columns = arms.x_out_preimage_columns(arm_kind);
        assert_eq!(x_out_columns.before().len(), 32);
        assert_eq!(x_out_columns.after().len(), 32);
        for &column in x_out_columns.before().iter().chain(x_out_columns.after()) {
            assert!(column >= arm.m_in, "x_out authority must remain private");
            assert!(column < arm.m, "x_out authority column must exist in the source arm");
        }
        let lane_columns = arms.after_nebula_lane_columns(arm_kind).all();
        assert_eq!(lane_columns.len(), 50);
        for column in lane_columns {
            assert!(
                column >= arm.m_in,
                "post-step Nebula lane authority must remain private"
            );
            assert!(
                column < arm.m,
                "post-step Nebula lane column must exist in the source arm"
            );
        }
    }

    let base = arms.arm(NebulaFPrimeStreamingLifecycleArm::Base);
    assert_eq!(arms.base_assignment().len(), base.m);
    base.is_satisfied_by(arms.base_assignment())
        .expect("exact normalized base assignment must satisfy every source row");

    let base_phase = arms.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Base);
    assert_eq!(base_phase.before_local_state_digest().len(), 4);
    assert_eq!(base_phase.after_local_state_digest().len(), 4);
    assert_eq!(
        base_phase.before_delayed_payload().len(),
        delayed_nebula_public_suffix_len(plan.config().stacks)
    );
    assert_eq!(
        base_phase.after_delayed_payload().len(),
        delayed_nebula_public_suffix_len(plan.config().stacks)
    );
    for range in [
        base_phase.before_local_state_digest(),
        base_phase.before_delayed_payload(),
        base_phase.after_local_state_digest(),
        base_phase.after_delayed_payload(),
    ] {
        assert!(range.start >= base.m_in);
    }
    let base_stages = base
        .physical_stage_ranges()
        .iter()
        .map(|stage| stage.path())
        .collect::<Vec<_>>();
    assert!(base_stages.contains(&fprime_stage::BASE_VERIFIER_KEY));
    assert!(base_stages.contains(&fprime_stage::BASE_CONTEXT_LINK));

    let recursive = arms.arm(NebulaFPrimeStreamingLifecycleArm::Recursive);
    let delayed_fields = arms.recursive_delayed_input_fields();
    let recursive_phase = arms.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Recursive);
    assert_eq!(recursive_phase.before_delayed_payload(), delayed_fields);
    assert_eq!(recursive_phase.before_local_state_digest().len(), 4);
    assert_eq!(recursive_phase.after_local_state_digest().len(), 4);
    assert_eq!(recursive_phase.after_delayed_payload().len(), delayed_fields.len());
    for range in [
        recursive_phase.before_local_state_digest(),
        recursive_phase.before_delayed_payload(),
        recursive_phase.after_local_state_digest(),
        recursive_phase.after_delayed_payload(),
    ] {
        assert!(range.start >= recursive.m_in);
    }
    assert_eq!(
        delayed_fields.len(),
        delayed_nebula_public_suffix_len(plan.config().stacks)
    );
    assert!(delayed_fields.start >= recursive.m_in);
    assert_eq!(
        recursive
            .column_family_ranges()
            .iter()
            .filter(|family| family.name == "fprime.recursive.nebula.private_delayed_input.raw_bits")
            .map(|family| family.column_start..family.column_end)
            .collect::<Vec<_>>(),
        vec![delayed_fields]
    );
    let recursive_stages = recursive
        .physical_stage_ranges()
        .iter()
        .map(|stage| stage.path())
        .collect::<Vec<_>>();
    for required in [
        fprime_stage::RECURSIVE_TRANSCRIPT,
        fprime_stage::RECURSIVE_NIFS,
        fprime_stage::RECURSIVE_PRIOR_LINK,
        fprime_stage::RECURSIVE_NEBULA_PRIVATE_INPUT,
        fprime_stage::RECURSIVE_ACCUMULATOR,
        fprime_stage::RECURSIVE_VERIFIER_KEY,
        fprime_stage::RECURSIVE_CONTEXT_LINK,
    ] {
        assert!(recursive_stages.contains(&required), "missing stage {required}");
    }
    let recursive_families = recursive
        .row_family_ranges()
        .iter()
        .map(|family| family.name)
        .collect::<Vec<_>>();
    for required in [
        "fprime.recursive.nifs",
        "fprime.recursive.nebula.private_delayed_input",
        "fprime.streaming.recursive.phase.semantic_envelope",
        "fprime.streaming.recursive.verifier_advice",
        "fprime.streaming.recursive.public_envelope",
    ] {
        assert!(recursive_families.contains(&required), "missing family {required}");
    }

    let program = NebulaFPrimeStreamingProgramAudit::production();
    let payload_len = recursive_phase.before_delayed_payload().len();
    let mut phase_sources = (0..program.circuit_kind_count())
        .map(|_| phase_envelope_source(payload_len, None))
        .collect::<Vec<_>>();
    let [even_pi_rlc, odd_pi_rlc]: [SparseR1cs; 2] = production_pi_rlc_family_body_source_arms()
        .expect("synthesize exact PiRLC parity source rows")
        .try_into()
        .expect("two PiRLC parity arms");
    phase_sources[NebulaFPrimeStreamingCircuitKind::PiRlcFamilyEven.code() as usize] = even_pi_rlc;
    phase_sources[NebulaFPrimeStreamingCircuitKind::PiRlcFamilyOdd.code() as usize] = odd_pi_rlc;
    let link_profile = production_phase_envelope_link_profile(&arms, &phase_sources)
        .expect("derive all production common-to-phase envelope links");
    assert_eq!(link_profile.phase_kind_count(), 23);
    assert_eq!(link_profile.fields_per_kind(), 4_346);
    assert_eq!(link_profile.total_links(), 99_958);

    phase_sources[0] = phase_envelope_source(payload_len, Some(STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY));
    let error = production_phase_envelope_link_profile(&arms, &phase_sources)
        .expect_err("a production phase kind cannot omit one envelope source family");
    assert!(error
        .to_string()
        .contains(STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY));
}

#[test]
#[ignore = "expensive real NIFS proof and full recursive source-row replay"]
fn streaming_recursive_source_assignment_uses_a_real_nifs_proof() {
    let started = std::time::Instant::now();
    let reference_params = Params::production();
    let memory_params = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan =
        NebulaPlan::new(memory_params, vec![7], [0xD9; 32], reference_params.kappa() as usize).expect("Nebula plan");
    let seed_structure = CcsStructure::new(
        vec![Mat::zero(1, plan.circuit().cols(), F::ZERO)],
        SparsePoly::new(1, Vec::new()),
    )
    .expect("zero seed relation with the S_mem lane geometry");
    let params = Params::for_ccs_shape(
        seed_structure.n,
        seed_structure.m,
        seed_structure.t(),
        seed_structure.max_degree(),
    )
    .expect("shape-specific Appendix B.2 parameters");
    assert!(params.has_production_core());
    let log = neo_fold_clean::frontends::direct_ccs::ajtai::setup_seeded(&params, &seed_structure, 0x5354_5245_414d);
    let preprocessing = neo_fold_clean::lifecycle::preprocess_with_test_log(
        params,
        seed_structure,
        log,
        Some(NebulaFPrimeStreamingPublicLayout::production().columns()),
    )
    .expect("verifier-owned lifecycle preprocessing");
    let preprocessing =
        prepare_streaming_lifecycle_preprocessing(preprocessing, &plan).expect("fixed streaming lifecycle policy");

    let mut memory = Memory::new(memory_params, &[7]).expect("initial memory");
    let trace = memory
        .begin_segment()
        .expect("open one-step segment")
        .finish()
        .expect("close one-step segment");
    let step = StepData {
        seg_idx: trace.seg_idx,
        idx: 0,
        ts_in: trace.ts_in,
        h_in: [K::ONE; 4],
        sp_in: [0; 2],
        ops: trace.step_ops(0),
        is_cells: &trace.is_cells,
        fs_cells: &trace.fs_cells,
    };

    let (provisional_z, _) = plan
        .circuit()
        .witness(
            &Gammas {
                gamma1: K::ONE,
                gamma2: K::ONE,
            },
            &step,
        )
        .expect("provisional S_mem witness");
    let public_columns = NebulaFPrimeStreamingPublicLayout::production().columns();
    let mut provisional = CcsInstance::from_low_norm_assignment(
        &preprocessing.params,
        &preprocessing.log,
        preprocessing.structure(),
        &provisional_z,
        public_columns,
    )
    .expect("provisional fresh instance");
    let committed_lanes = plan
        .scheme()
        .commit(&provisional.witness.Z)
        .expect("commit provisional S_mem lanes");
    provisional.claim.adv = Some(committed_lanes.clone());
    let d_pre = nebula_lane_chains(std::iter::once(&committed_lanes));
    assert_eq!(
        d_pre[1],
        plan.config().d_init,
        "initial-memory chain must match the plan"
    );
    assert_eq!(d_pre[1], d_pre[2], "an empty segment must preserve its memory snapshot");

    let running = RunningInstance::canonical_zero(
        &preprocessing.params,
        preprocessing.structure(),
        public_columns,
        LaneCommitmentMode::Nebula,
    )
    .expect("canonical recursive input accumulator");
    let accumulator = AccumulatorHandle::from_running_parts(
        preprocessing.params.b(),
        &running.claims,
        running.parent_authority.as_ref(),
    )
    .digest_fields();
    let prior_boundary =
        f_prime_chunk_public_digest_for_uniform_shape(0, 1, D, preprocessing.params.kappa() as usize, public_columns);
    let config = preprocessing
        .nebula()
        .expect("prepared lifecycle carries the verifier-owned Nebula config")
        .clone();
    let mut opened = NebulaLane::base(&config);
    opened
        .open_segment(
            &config,
            preprocessing.vk.digest(),
            digest_fields_as_digest32(prior_boundary),
            digest_fields_as_digest32(accumulator),
            d_pre,
        )
        .expect("derive the committed segment challenge");
    let gamma = opened.gamma.expect("opened segment challenge");

    let (s_mem_z, step_x) = plan
        .circuit()
        .witness(
            &Gammas {
                gamma1: gamma[0],
                gamma2: gamma[1],
            },
            &step,
        )
        .expect("challenge-bound S_mem witness");
    let private_delayed = encode_delayed_f_prime_suffix(&step_x, config.stacks, Some(d_pre))
        .expect("encode the verifier-replayed delayed Nebula input");
    let semantic_state = streaming_phase_semantic_digest([F::ZERO; 4], &private_delayed);
    let initial_boundary = initial_boundary_digest(preprocessing.structure_digest(), Some(public_columns));
    let lane_in = NebulaLane::base(&config);
    let prior_x_out = digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateful,
        preprocessing.vk.digest(),
        preprocessing.pi_ccs_header_bundle(),
        &preprocessing.pi_ccs_header_bundle(),
        1,
        1,
        initial_boundary,
        digest_fields_as_digest32(prior_boundary),
        1,
        digest_fields_as_digest32(semantic_state),
        digest_fields_as_digest32(accumulator),
        digest_fields_as_digest32(prior_boundary),
        Some(lane_in.digest()),
    ));
    let mut fresh_z = vec![F::ZERO; preprocessing.structure().m];
    fresh_z[0] = F::ONE;
    fresh_z[1..1 + 256].copy_from_slice(&encode_x_out_public_bits(prior_x_out));
    let lane_ranges = plan.scheme().lane_ranges();
    for range in [&lane_ranges.ops, &lane_ranges.is, &lane_ranges.fs] {
        assert!(
            range.start * D >= public_columns,
            "the seed relation lane slices must not overlap the F-prime public prefix"
        );
        for ring_column in range.clone() {
            for coefficient in 0..D {
                let field_column = ring_column * D + coefficient;
                if field_column < fresh_z.len() {
                    fresh_z[field_column] = s_mem_z[field_column];
                }
            }
        }
    }
    let mut fresh = CcsInstance::from_low_norm_assignment(
        &preprocessing.params,
        &preprocessing.log,
        preprocessing.structure(),
        &fresh_z,
        public_columns,
    )
    .expect("challenge-bound fresh instance");
    let rebuilt_lanes = plan
        .scheme()
        .commit(&fresh.witness.Z)
        .expect("recommit challenge-bound S_mem lanes");
    assert_eq!(
        rebuilt_lanes, committed_lanes,
        "the commit-then-challenge order must not change the committed lanes"
    );
    fresh.claim.adv = Some(rebuilt_lanes);

    eprintln!("recursive assignment fixture ready after {:?}", started.elapsed());
    let synthesis_started = std::time::Instant::now();
    let arms = synthesize_streaming_lifecycle_source_arms_with_recursive_assignment(
        &preprocessing,
        &plan,
        &fresh,
        &private_delayed,
    )
    .expect("synthesize a proof-backed recursive lifecycle source arm");
    eprintln!(
        "proof-backed lifecycle source synthesized after {:?}",
        synthesis_started.elapsed()
    );
    let recursive = arms.arm(NebulaFPrimeStreamingLifecycleArm::Recursive);
    let assignment = arms
        .recursive_assignment()
        .expect("proof-backed synthesis must retain the recursive assignment");
    assert_eq!(assignment.len(), recursive.m);
    assert_eq!(recursive.m_in, 641);
    eprintln!(
        "Appendix B.2 recursive seed source: rows={} columns={} public={}",
        recursive.n, recursive.m, recursive.m_in
    );
    let replay_started = std::time::Instant::now();
    if let Err(error) = recursive.is_satisfied_by(assignment) {
        let row = match error {
            neo_fold_clean::frontends::direct_ccs::FrontendError::Unsatisfied { row } => row,
            other => panic!("recursive source replay failed before row evaluation: {other}"),
        };
        let families = recursive
            .row_family_ranges()
            .iter()
            .filter(|family| family.row_start <= row && row < family.row_end)
            .map(|family| (family.name, family.row_start..family.row_end))
            .collect::<Vec<_>>();
        let stages = recursive
            .physical_stage_ranges()
            .iter()
            .filter(|stage| stage.contains_row(row))
            .map(|stage| (stage.path(), stage.rows()))
            .collect::<Vec<_>>();
        panic!("recursive source row {row} failed; families={families:?}; stages={stages:?}");
    }
    eprintln!("recursive source replay passed after {:?}", replay_started.elapsed());

    let base = arms.arm(NebulaFPrimeStreamingLifecycleArm::Base);
    base.is_satisfied_by(arms.base_assignment())
        .expect("proof-backed synthesis must preserve the satisfying base assignment");
}
