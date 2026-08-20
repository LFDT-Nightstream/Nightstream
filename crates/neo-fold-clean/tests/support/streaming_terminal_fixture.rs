//! Shared exact source-to-final fixture for streaming terminal audits.

use std::collections::BTreeSet;
use std::ops::Range;

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_fold_clean::engine::r1cs_circuit::boolean::enforce_bit;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder};
use neo_fold_clean::frontends::nebula::f_prime::{
    prepare_streaming_lifecycle_preprocessing, production_streaming_terminal_profile, streaming_phase_semantic_digest,
    synthesize_streaming_lifecycle_source_arms, NebulaFPrimeStreamingCircuitKind, NebulaFPrimeStreamingLifecycleArm,
    NebulaFPrimeStreamingLifecycleSourceArms, NebulaFPrimeStreamingProgramAudit, NebulaFPrimeStreamingPublicLayout,
    NebulaFPrimeStreamingTerminalFieldBinding, NebulaFPrimeStreamingTerminalFieldDomain,
    NebulaFPrimeStreamingTerminalProfile, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY,
    STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY, STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS,
};
use neo_fold_clean::frontends::nebula::layout::{encode_delayed_f_prime_suffix, NebulaParams, StepPublicInput};
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    enforce_streaming_terminal_lifecycle, StreamingTerminalPublicWires,
};
use neo_fold_clean::frontends::r1cs_f_prime::{
    build_multi_branch_selective_low_norm_r1cs_with_alignment,
    build_scheduled_linked_overlay_low_norm_r1cs_with_phase_field_links, lower_field_r1cs, OverlayKindLinks,
    ScheduledCommonPhaseFieldLink, ScheduledCursorBits, ScheduledLinkedOverlayLowNormR1cs, ScheduledPhaseKindLinks,
    SparseR1cs,
};
use neo_fold_clean::paper::construction2::{NebulaLane, StackShape};
use neo_fold_clean::paper::digest::{self, StateXOutDigestMode, StateXOutPreimageInstruction};
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::NebulaLaneWires;
use neo_fold_clean::paper::f_prime::stage as fprime_stage;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::product_commitment_circuit::{adv_commitment_data_wires, alloc_adv};
use neo_math::field::KExtensions;
use neo_math::{D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

fn public_cursor_source() -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    let public = (0..640)
        .map(|_| {
            let bit = builder.alloc(F::ZERO);
            enforce_bit(&mut builder, bit);
            bit
        })
        .collect::<Vec<_>>();
    lower_field_r1cs(builder, &public)
        .expect("lower terminal phase public fixture")
        .into_parts()
        .0
}

fn semantic_links_source(payload_len: usize) -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    let public = (0..640)
        .map(|_| {
            let bit = builder.alloc(F::ZERO);
            enforce_bit(&mut builder, bit);
            bit
        })
        .collect::<Vec<_>>();

    let before_local_start = builder.cols();
    let before_local = builder.alloc_vec(&[F::ZERO; 4]);
    builder.record_column_family(STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY, before_local_start);

    let before_payload_start = builder.cols();
    let before_payload = builder.alloc_vec(&vec![F::ZERO; payload_len]);
    builder.record_column_family(STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY, before_payload_start);
    for &bit in &before_payload {
        enforce_bit(&mut builder, bit);
    }

    let after_local_start = builder.cols();
    let after_local = builder.alloc_vec(&[F::ZERO; 4]);
    builder.record_column_family(STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY, after_local_start);

    let after_payload_start = builder.cols();
    let after_payload = builder.alloc_vec(&vec![F::ZERO; payload_len]);
    builder.record_column_family(STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY, after_payload_start);
    for &bit in &after_payload {
        enforce_bit(&mut builder, bit);
    }

    let helper_start = builder.cols();
    builder.record_column_family("test.streaming_terminal.digest_squares", helper_start);
    for field in before_local.into_iter().chain(after_local) {
        let square = builder.alloc(F::ZERO);
        builder.enforce(&Lc::from_var(field), &Lc::from_var(field), &Lc::from_var(square));
    }
    lower_field_r1cs(builder, &public)
        .expect("lower terminal SemanticLinks fixture")
        .into_parts()
        .0
}

fn overlay_source() -> SparseR1cs {
    let mut builder = R1csBuilder::new();
    let bit = builder.alloc(F::ZERO);
    enforce_bit(&mut builder, bit);
    lower_field_r1cs(builder, &[])
        .expect("lower terminal overlay fixture")
        .into_parts()
        .0
}

fn exact_family(source: &SparseR1cs, name: &'static str) -> Range<usize> {
    let matches = source
        .column_family_ranges()
        .iter()
        .filter(|family| family.name == name)
        .map(|family| family.column_start..family.column_end)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "source family {name}");
    matches[0].clone()
}

fn append_links(links: &mut Vec<ScheduledCommonPhaseFieldLink>, common: Range<usize>, phase: Range<usize>) {
    assert_eq!(common.len(), phase.len());
    links.extend(
        common
            .zip(phase)
            .map(|(common_field, phase_field)| ScheduledCommonPhaseFieldLink {
                common_field,
                phase_field,
            }),
    );
}

fn decoded_field(binding: &NebulaFPrimeStreamingTerminalFieldBinding, assignment: &[F]) -> F {
    binding.decoder_terms().iter().fold(F::ZERO, |sum, term| {
        sum + assignment[term.final_column()] * term.coefficient()
    })
}

fn solve_decoded_fields(
    label: &'static str,
    bindings: &[NebulaFPrimeStreamingTerminalFieldBinding],
    values: &[F],
    assignment: &mut [F],
    locked: &mut BTreeSet<usize>,
) {
    assert_eq!(bindings.len(), values.len());
    for (binding, &expected) in bindings.iter().zip(values) {
        let actual = decoded_field(binding, assignment);
        if actual != expected {
            let pivot = binding
                .decoder_terms()
                .iter()
                .find(|term| {
                    term.final_column() != 0 && term.coefficient() != F::ZERO && !locked.contains(&term.final_column())
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{label} decoder for source field {} has no free pivot: actual {actual:?}, expected {expected:?}",
                        binding.source_field()
                    )
                });
            let column = pivot.final_column();
            assignment[column] += (expected - actual) * pivot.coefficient().inverse();
        }
        assert_eq!(decoded_field(binding, assignment), expected);
        locked.extend(
            binding
                .decoder_terms()
                .iter()
                .map(|term| term.final_column()),
        );
    }
}

fn lane_fields(lane: &NebulaLane) -> Vec<F> {
    let mut fields = Vec::with_capacity(50);
    fields.extend(lane.program_binding_digest);
    fields.extend([
        if lane.gamma.is_some() { F::ONE } else { F::ZERO },
        F::from_u64(lane.seg_idx),
        F::from_u64(lane.idx),
        F::from_u64(lane.ts),
    ]);
    for value in lane.gamma.unwrap_or([K::ONE; 2]) {
        let (c0, c1) = value.to_limbs_u64();
        fields.extend([F::from_u64(c0), F::from_u64(c1)]);
    }
    for value in lane.h {
        let (c0, c1) = value.to_limbs_u64();
        fields.extend([F::from_u64(c0), F::from_u64(c1)]);
    }
    fields.extend(lane.sp.map(F::from_u64));
    fields.extend(lane.d_pre.into_iter().flatten());
    fields.extend(lane.d_seen.into_iter().flatten());
    fields.extend(lane.d_mem);
    assert_eq!(fields.len(), 50);
    fields
}

fn lane_wire_columns(lane: &NebulaLaneWires) -> [usize; 50] {
    let mut fields = Vec::with_capacity(50);
    fields.extend(lane.program_binding_digest.map(|wire| wire.col()));
    fields.extend([lane.open, lane.seg_idx, lane.idx, lane.ts].map(|wire| wire.col()));
    for value in lane.gamma.iter().chain(&lane.h) {
        fields.extend([value.c0.col(), value.c1.col()]);
    }
    fields.extend(lane.sp.map(|wire| wire.col()));
    fields.extend(lane.d_pre.iter().flatten().map(|wire| wire.col()));
    fields.extend(lane.d_seen.iter().flatten().map(|wire| wire.col()));
    fields.extend(lane.d_mem.map(|wire| wire.col()));
    fields.try_into().expect("50 Nebula lane fields")
}

fn terminal_x_out_preimage(
    vk_fs: [F; 4],
    pi_ccs_header: [F; 4],
    boundary: [F; 4],
    semantic: [F; 4],
    accumulator: [F; 4],
    lane_digest: [F; 4],
) -> Vec<F> {
    let mut fields = Vec::with_capacity(32);
    for instruction in digest::state_x_out_preimage_program(StateXOutDigestMode::Stateful, true) {
        match instruction {
            StateXOutPreimageInstruction::Domain { value }
            | StateXOutPreimageInstruction::NebulaPresentMarker { value } => fields.push(F::from_u64(value)),
            StateXOutPreimageInstruction::VerifierDigest => fields.extend(vk_fs),
            StateXOutPreimageInstruction::PiCcsHeader => fields.extend(pi_ccs_header),
            StateXOutPreimageInstruction::ChunkCountHalves | StateXOutPreimageInstruction::StepCountHalves => {
                fields.extend([F::from_u64(STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS as u64), F::ZERO]);
            }
            StateXOutPreimageInstruction::PcHalves => fields.extend([F::ONE, F::ZERO]),
            StateXOutPreimageInstruction::CurrentBoundary => fields.extend(boundary),
            StateXOutPreimageInstruction::SemanticState => fields.extend(semantic),
            StateXOutPreimageInstruction::Construction2Accumulator => fields.extend(accumulator),
            StateXOutPreimageInstruction::NebulaDigest => fields.extend(lane_digest),
        }
    }
    assert_eq!(fields.len(), 32);
    fields
}

pub struct StreamingTerminalAuditFixture {
    pub lifecycle: NebulaFPrimeStreamingLifecycleSourceArms,
    pub relation: ScheduledLinkedOverlayLowNormR1cs,
    pub profile: NebulaFPrimeStreamingTerminalProfile,
    pub terminal: R1csBuilder,
    pub final_witness_column_start: usize,
    pub source_binding_decoded_column_start: usize,
    pub schedule_selector_column: usize,
    pub lifecycle_selector_column: usize,
    pub phase_selector_column: usize,
    pub verifier_key_column: usize,
    pub vk_fs_columns: [usize; 4],
    pub pi_ccs_header_columns: [usize; 4],
    pub boundary_columns: [usize; 4],
    pub accumulator_columns: [usize; 4],
    pub program_binding_column: usize,
    pub delayed_payload_column: usize,
    pub fresh_adv_column: usize,
    pub final_closed_lane_column: usize,
    pub fresh_adv_shape_columns: [[usize; 2]; 3],
    pub fresh_adv_data_columns: LaneCommitments<Vec<usize>>,
    pub fresh_adv_d: usize,
    pub fresh_adv_kappa: usize,
    pub delayed_payload_columns: Vec<usize>,
    pub final_lane_columns: [usize; 50],
    pub steps_per_segment: u64,
    pub seg_max: u64,
    pub stacks: StackShape,
}

pub fn build_streaming_terminal_audit_fixture() -> StreamingTerminalAuditFixture {
    let reference_params = Params::production();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xD9; 32], reference_params.kappa() as usize).expect("Nebula plan");
    let params = Params::for_ccs_shape(
        plan.circuit().structure().n,
        plan.circuit().structure().m,
        plan.circuit().structure().t(),
        plan.circuit().structure().max_degree(),
    )
    .expect("shape-specific Nightstream Goldilocks k_rho=16 parameters");
    assert!(params.has_production_core());
    let log = neo_fold_clean::frontends::direct_ccs::ajtai::setup_seeded(
        &params,
        plan.circuit().structure(),
        0x5354_5245_414d,
    );
    let preprocessing = neo_fold_clean::lifecycle::preprocess_with_test_log(
        params.clone(),
        plan.circuit().structure().clone(),
        log,
        Some(648),
    )
    .expect("verifier-owned lifecycle preprocessing");
    let preprocessing =
        prepare_streaming_lifecycle_preprocessing(preprocessing, &plan).expect("fixed streaming lifecycle policy");
    let lifecycle = synthesize_streaming_lifecycle_source_arms(&preprocessing, &plan)
        .expect("synthesize exact streaming lifecycle source rows");

    let common_arms = [
        lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .clone(),
        lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Recursive)
            .clone(),
    ];
    let common = build_multi_branch_selective_low_norm_r1cs_with_alignment(&common_arms, 0, D, 0)
        .expect("compile exact lifecycle source arms");

    let recursive_fields = lifecycle.phase_envelope_fields(NebulaFPrimeStreamingLifecycleArm::Recursive);
    let payload_len = recursive_fields.after_delayed_payload().len();
    let semantic_links_source = semantic_links_source(payload_len);
    let mut phase_arms = vec![public_cursor_source(); 23];
    let semantic_kind = NebulaFPrimeStreamingCircuitKind::SemanticLinks.code() as usize;
    phase_arms[semantic_kind] = semantic_links_source.clone();
    let phase_kinds = build_multi_branch_selective_low_norm_r1cs_with_alignment(&phase_arms, 0, D, 0)
        .expect("compile terminal phase fixtures");

    let mut fields = Vec::with_capacity(2 * 4 + 2 * payload_len);
    append_links(
        &mut fields,
        recursive_fields.before_local_state_digest(),
        exact_family(&semantic_links_source, STREAMING_PHASE_BEFORE_LOCAL_STATE_FAMILY),
    );
    append_links(
        &mut fields,
        recursive_fields.before_delayed_payload(),
        exact_family(&semantic_links_source, STREAMING_PHASE_BEFORE_DELAYED_PAYLOAD_FAMILY),
    );
    append_links(
        &mut fields,
        recursive_fields.after_local_state_digest(),
        exact_family(&semantic_links_source, STREAMING_PHASE_AFTER_LOCAL_STATE_FAMILY),
    );
    append_links(
        &mut fields,
        recursive_fields.after_delayed_payload(),
        exact_family(&semantic_links_source, STREAMING_PHASE_AFTER_DELAYED_PAYLOAD_FAMILY),
    );

    let overlay_source = overlay_source();
    let overlay =
        build_multi_branch_selective_low_norm_r1cs_with_alignment(&[overlay_source.clone(), overlay_source], 0, 1, 0)
            .expect("compile terminal overlay fixture");
    let program = NebulaFPrimeStreamingProgramAudit::production();
    let public = NebulaFPrimeStreamingPublicLayout::production();
    let mut overlay_kinds = vec![0; STREAMING_TERMINAL_ACCEPTED_WORK_ITEMS];
    overlay_kinds[0] = 1;
    let relation = build_scheduled_linked_overlay_low_norm_r1cs_with_phase_field_links(
        common,
        phase_kinds,
        overlay,
        program.lifecycle_group_map(),
        program.circuit_kind_map(),
        overlay_kinds,
        ScheduledCursorBits::new(public.before_cursor_bits(), public.after_cursor_bits()),
        vec![ScheduledPhaseKindLinks {
            lifecycle_group: 1,
            phase_kind: semantic_kind,
            fields,
        }],
        Vec::<OverlayKindLinks>::new(),
    )
    .expect("compose the exact terminal schedule scope");

    let profile = production_streaming_terminal_profile(&lifecycle, &relation)
        .expect("derive exact terminal source-to-final profile");
    assert_eq!(profile.accepted_work_items(), 436);
    assert_eq!(profile.terminal_arm(), 435);
    assert_eq!(profile.lifecycle_group(), 1);
    assert_eq!(profile.phase_kind(), semantic_kind);
    assert_eq!(
        profile.profile_id(),
        "nightstream/goldilocks/streaming-terminal-slice/v1"
    );
    assert_eq!(profile.lifecycle_scope(), "recursive-terminal-arm-435");
    assert_eq!(
        profile.source_artifact_identity(),
        "rust:nightstream/streaming-lifecycle-recursive/source-rows/v1"
    );
    assert_eq!(
        profile.final_artifact_identity(),
        "rust:nightstream/streaming-selective-ccs/final-rows/v1"
    );
    assert_eq!(profile.source_stage_path(), fprime_stage::RECURSIVE_SEMANTIC_LINKS);
    assert!(!profile.source_stage_rows().is_empty());
    assert!(!profile.final_stage_row_runs().is_empty());
    let x_out = profile.after_x_out();
    assert_eq!(x_out.source_fields().len(), 32);
    assert_eq!(x_out.verifier_key_digest().len(), 4);
    assert_eq!(x_out.pi_ccs_header().len(), 4);
    assert_eq!(x_out.chunk_count_halves().len(), 2);
    assert_eq!(x_out.step_count_halves().len(), 2);
    assert_eq!(x_out.program_counter_halves().len(), 2);
    assert_eq!(x_out.boundary().len(), 4);
    assert_eq!(x_out.semantic_state_digest().len(), 4);
    assert_eq!(x_out.construction2_accumulator_digest().len(), 4);
    assert_eq!(x_out.nebula_state_digest().len(), 4);
    assert!(!x_out.domain_tag().decoder_terms().is_empty());
    assert!(!x_out.nebula_presence_marker().decoder_terms().is_empty());

    let columns = profile.column_layout();
    assert_eq!(columns.public(), relation.layout().public_columns());
    assert_eq!(
        columns.lifecycle_private(),
        relation
            .scheduled_relation()
            .layout()
            .common_private_columns()
    );
    assert_eq!(
        columns.phase_private(),
        relation
            .scheduled_relation()
            .layout()
            .phase_private_columns()
    );
    assert_eq!(
        columns.schedule_selectors(),
        relation
            .scheduled_relation()
            .layout()
            .schedule_selector_columns()
    );
    assert_eq!(
        columns.scheduled_ring_padding(),
        relation
            .scheduled_relation()
            .layout()
            .ring_padding_columns()
    );
    assert_eq!(columns.overlay_private(), relation.layout().overlay_private_columns());
    assert_eq!(
        columns.overlay_selectors(),
        relation.layout().overlay_selector_columns()
    );
    assert_eq!(columns.final_ring_padding(), relation.layout().ring_padding_columns());

    let lane = profile.after_nebula_lane();
    assert_eq!(lane.source_fields().len(), 50);
    assert_eq!(lane.fields().len(), 50);
    assert_eq!(lane.program_binding_digest().len(), 4);
    assert_eq!(lane.gamma().len(), 4);
    assert_eq!(lane.running_products().len(), 8);
    assert_eq!(lane.stack_pointers().len(), 2);
    assert_eq!(lane.pre_chains().len(), 12);
    assert_eq!(lane.seen_chains().len(), 12);
    assert_eq!(lane.memory_digest().len(), 4);
    assert!(!lane.open().decoder_terms().is_empty());
    assert!(!lane.segment_index().decoder_terms().is_empty());
    assert!(!lane.step_index().decoder_terms().is_empty());
    assert!(!lane.timestamp().decoder_terms().is_empty());

    let local = profile.after_local_state_digest();
    assert_eq!(local.source_fields(), recursive_fields.after_local_state_digest());
    assert_eq!(
        local.source_domain(),
        NebulaFPrimeStreamingTerminalFieldDomain::Goldilocks
    );
    assert_eq!(local.fields().len(), 4);
    assert_eq!(local.final_common_phase_link_rows().len(), 4);
    for field in local.fields() {
        assert!(!field.decoder_terms().is_empty());
        assert_eq!(
            relation
                .common_field_decoding_terms(1, field.source_field())
                .expect("decode terminal local field")
                .len(),
            field.decoder_terms().len()
        );
    }

    let payload = profile.after_delayed_payload();
    assert_eq!(payload.source_fields(), recursive_fields.after_delayed_payload());
    assert_eq!(
        payload.source_domain(),
        NebulaFPrimeStreamingTerminalFieldDomain::Boolean
    );
    assert_eq!(payload.fields().len(), payload_len);
    assert_eq!(payload.final_common_phase_link_rows().len(), payload_len);
    assert!(payload
        .fields()
        .iter()
        .all(|field| !field.decoder_terms().is_empty()));

    let expected_owned_fields = x_out
        .source_fields()
        .iter()
        .copied()
        .chain(lane.source_fields().iter().copied())
        .chain(local.source_fields())
        .chain(payload.source_fields())
        .collect::<BTreeSet<_>>();
    let owned_fields = profile
        .source_stage_bindings()
        .iter()
        .flat_map(|binding| binding.source_fields().iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(
        owned_fields.iter().copied().collect::<BTreeSet<_>>(),
        expected_owned_fields
    );
    assert_eq!(owned_fields.len(), expected_owned_fields.len());
    assert!(profile.source_stage_bindings().iter().all(|binding| {
        !binding.source_rows().is_empty()
            && !binding.source_stage_path().is_empty()
            && !binding.final_row_runs().is_empty()
            && binding.final_row_runs().iter().all(|run| {
                !run.rows().is_empty() && run.rows().end <= relation.scheduled_relation().layout().common_rows().end
            })
    }));

    let mut config = plan.config();
    config.initial_semantic_state_digest = streaming_phase_semantic_digest([F::ZERO; 4], &vec![F::ZERO; payload_len]);
    let zero_commitment = Commitment::zeros(D, params.kappa() as usize);
    let fresh_adv = LaneCommitments {
        ops: zero_commitment.clone(),
        is: zero_commitment.clone(),
        fs: zero_commitment,
    };
    let d_pre = digest::nebula_lane_chains(std::iter::once(&fresh_adv));
    let vk_fs = [F::from_u64(11); 4];
    let pi_ccs_header = [F::from_u64(13); 4];
    let boundary = [F::from_u64(17); 4];

    // This fixture isolates the appended terminal rows. The complete final
    // selective CCS relation is the other conjunct in the terminal compiler.
    // Solve only the checked affine decoder equations needed by this helper.
    let mut final_assignment = vec![F::ZERO; profile.final_columns()];
    final_assignment[0] = F::ONE;
    let mut locked = BTreeSet::from([
        0,
        profile.schedule_selector_column(),
        profile.lifecycle_selector_column(),
        profile.phase_selector_column(),
    ]);
    final_assignment[profile.schedule_selector_column()] = F::ONE;
    final_assignment[profile.lifecycle_selector_column()] = F::ONE;
    final_assignment[profile.phase_selector_column()] = F::ONE;

    let x_out_template =
        terminal_x_out_preimage(vk_fs, pi_ccs_header, boundary, [F::ZERO; 4], [F::ZERO; 4], [F::ZERO; 4]);
    solve_decoded_fields(
        "XOut context",
        &profile.after_x_out().fields()[..19],
        &x_out_template[..19],
        &mut final_assignment,
        &mut locked,
    );
    solve_decoded_fields(
        "XOut Nebula marker",
        &profile.after_x_out().fields()[27..28],
        &x_out_template[27..28],
        &mut final_assignment,
        &mut locked,
    );
    let accumulator =
        std::array::from_fn(|index| decoded_field(&profile.after_x_out().fields()[23 + index], &final_assignment));
    solve_decoded_fields(
        "XOut accumulator",
        &profile.after_x_out().fields()[23..27],
        &accumulator,
        &mut final_assignment,
        &mut locked,
    );

    let mut post_phase_lane = NebulaLane::base(&config);
    post_phase_lane.d_mem = d_pre[1];
    let mut opened = post_phase_lane.clone();
    opened
        .open_segment(
            &config,
            digest::digest_fields_as_digest32(vk_fs),
            digest::digest_fields_as_digest32(boundary),
            digest::digest_fields_as_digest32(accumulator),
            d_pre,
        )
        .expect("open one-step terminal segment");
    let step = StepPublicInput {
        seg_idx: opened.seg_idx,
        idx: opened.idx,
        ts_in: opened.ts,
        ts_out: opened.ts + 1,
        gamma: opened.gamma.expect("opened terminal gamma"),
        h_in: opened.h,
        h_out: [K::ONE; 4],
        sp_in: [0; 2],
        sp_out: [0; 2],
    };
    let delayed_payload = encode_delayed_f_prime_suffix(&step, config.stacks, Some(d_pre))
        .expect("encode one-step terminal delayed payload");
    assert_eq!(delayed_payload.len(), payload_len);
    let local_state = [F::from_u64(23); 4];
    solve_decoded_fields(
        "local state",
        profile.after_local_state_digest().fields(),
        &local_state,
        &mut final_assignment,
        &mut locked,
    );
    solve_decoded_fields(
        "delayed payload",
        profile.after_delayed_payload().fields(),
        &delayed_payload,
        &mut final_assignment,
        &mut locked,
    );
    solve_decoded_fields(
        "lane",
        profile.after_nebula_lane().fields(),
        &lane_fields(&post_phase_lane),
        &mut final_assignment,
        &mut locked,
    );
    let semantic = streaming_phase_semantic_digest(local_state, &delayed_payload);
    let x_out_preimage = terminal_x_out_preimage(
        vk_fs,
        pi_ccs_header,
        boundary,
        semantic,
        accumulator,
        post_phase_lane.digest(),
    );
    solve_decoded_fields(
        "XOut semantic state",
        &profile.after_x_out().fields()[19..23],
        &x_out_preimage[19..23],
        &mut final_assignment,
        &mut locked,
    );
    solve_decoded_fields(
        "XOut Nebula digest",
        &profile.after_x_out().fields()[28..32],
        &x_out_preimage[28..32],
        &mut final_assignment,
        &mut locked,
    );
    for (binding, expected) in profile.after_x_out().fields().iter().zip(&x_out_preimage) {
        assert_eq!(decoded_field(binding, &final_assignment), *expected);
    }

    let mut terminal = R1csBuilder::new();
    let final_witness = terminal.alloc_vec(&final_assignment);
    let family_start = terminal.rows();
    let adv_wires = alloc_adv(&mut terminal, Some(&fresh_adv)).expect("terminal fresh adv wires");
    terminal.record_row_family(
        neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::STREAMING_TERMINAL_R1CS_FAMILY_NAMES[6],
        family_start,
    );
    let adv_data_wires = adv_commitment_data_wires(&adv_wires);
    let public = StreamingTerminalPublicWires {
        vk_fs_digest: vk_fs.map(|value| terminal.alloc(value)),
        pi_ccs_header: pi_ccs_header.map(|value| terminal.alloc(value)),
        current_boundary: boundary.map(|value| terminal.alloc(value)),
        accumulator_digest: accumulator.map(|value| terminal.alloc(value)),
    };
    let source_binding_decoded_column_start = terminal.cols();
    let terminal_output = enforce_streaming_terminal_lifecycle(
        &mut terminal,
        &profile,
        &final_witness,
        &adv_data_wires,
        &config,
        public,
    )
    .expect("enforce exact streaming terminal lifecycle");
    assert!(
        terminal.is_satisfied(),
        "honest streaming terminal rows must satisfy; first row {:?}; families {:?}",
        terminal.first_unsatisfied_row(),
        terminal.row_family_ranges(),
    );
    assert_eq!(terminal.witness()[terminal_output.final_lane.open.col()], F::ZERO);
    assert_eq!(terminal.witness()[terminal_output.final_lane.idx.col()], F::ZERO);
    assert_eq!(terminal_output.delayed_payload.len(), payload_len);

    let fresh_adv_shape_columns = [
        [adv_wires.ops.d_var.col(), adv_wires.ops.kappa_var.col()],
        [adv_wires.is.d_var.col(), adv_wires.is.kappa_var.col()],
        [adv_wires.fs.d_var.col(), adv_wires.fs.kappa_var.col()],
    ];
    let fresh_adv_data_columns = LaneCommitments {
        ops: adv_data_wires
            .ops
            .data
            .iter()
            .map(|wire| wire.col())
            .collect(),
        is: adv_data_wires
            .is
            .data
            .iter()
            .map(|wire| wire.col())
            .collect(),
        fs: adv_data_wires
            .fs
            .data
            .iter()
            .map(|wire| wire.col())
            .collect(),
    };
    let delayed_payload_columns = terminal_output
        .delayed_payload
        .iter()
        .map(|wire| wire.col())
        .collect();
    let final_lane_columns = lane_wire_columns(&terminal_output.final_lane);

    StreamingTerminalAuditFixture {
        lifecycle,
        relation,
        final_witness_column_start: final_witness[0].col(),
        source_binding_decoded_column_start,
        schedule_selector_column: final_witness[profile.schedule_selector_column()].col(),
        lifecycle_selector_column: final_witness[profile.lifecycle_selector_column()].col(),
        phase_selector_column: final_witness[profile.phase_selector_column()].col(),
        verifier_key_column: public.vk_fs_digest[0].col(),
        vk_fs_columns: public.vk_fs_digest.map(|wire| wire.col()),
        pi_ccs_header_columns: public.pi_ccs_header.map(|wire| wire.col()),
        boundary_columns: public.current_boundary.map(|wire| wire.col()),
        accumulator_columns: public.accumulator_digest.map(|wire| wire.col()),
        program_binding_column: terminal_output.post_phase_lane.program_binding_digest[0].col(),
        delayed_payload_column: terminal_output.delayed_payload[0].col(),
        fresh_adv_column: adv_wires.ops.data[0].col(),
        final_closed_lane_column: terminal_output.final_lane.h[0].c0.col(),
        fresh_adv_shape_columns,
        fresh_adv_data_columns,
        fresh_adv_d: adv_data_wires.ops.d,
        fresh_adv_kappa: adv_data_wires.ops.kappa,
        delayed_payload_columns,
        final_lane_columns,
        steps_per_segment: config.steps_per_segment,
        seg_max: config.seg_max,
        stacks: config.stacks,
        terminal,
        profile,
    }
}
