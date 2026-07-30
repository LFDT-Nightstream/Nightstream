//! Generic HyperNova terminal-verification acceptance gate.

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::builder::{BlockLaneNcBoundaryAudit, SumcheckRoundAudit};
use neo_fold_clean::frontends::direct_ccs::R1cs;
use neo_fold_clean::frontends::f_prime::recursive_plan::build_semantic_state_preimage_fields;
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    R1csIvc, R1csIvcBranch, R1csIvcGeneratedKSlot, R1csIvcPostPiDecExecutionAudit, R1csIvcPreprocessing,
    R1csIvcRawAssignmentAuthority, R1csIvcRawOldBlockFieldDecoding, R1csIvcRawOldBlockProfile,
};
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::paper::digest::{digest32_as_fields, digest_fields_as_digest32};
use neo_fold_clean::paper::f_prime::poseidon_trace::encode_poseidon_trace;
use neo_math::{F, K};
use neo_reductions::optimized_engine::PiCcsProofVariant;
use p3_field::PrimeCharacteristicRing;

use support::r1cs_compiler_fixtures::{
    assignment_one_product, make_tiny_lifecycle_plan, make_tiny_stateful_lifecycle_plan_with_anchor, one_product_r1cs,
    tiny_params,
};

#[test]
fn generic_ivc_verifies_running_accumulator_and_latest_f_prime() {
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let prep = R1csIvcPreprocessing::new_seeded(tiny_params(), &app, plan, 0x1F15_C007)
        .expect("compile authoritative generic R1CS IVC relation");
    assert!(prep.prep.enforces_terminal_induction());

    let mut chain = R1csIvc::new(&prep);
    for (step, (a, b)) in [(3, 7), (4, 9), (5, 11)].into_iter().enumerate() {
        chain
            .extend(assignment_one_product(a, b))
            .unwrap_or_else(|error| panic!("append satisfying app step {}: {error}", step + 1));
    }
    let execution = chain
        .post_pi_dec_execution_audit()
        .expect("third step executes the active post-PiDEC recursive arm");
    let compilation = prep.relation().compilation_audit();
    let selector_columns = compilation.layout().selector_columns();
    let fixed_point = compilation
        .rounds()
        .last()
        .expect("fixed-point compilation records its final round");
    let source_shape = (fixed_point.arms[2].rows, fixed_point.arms[2].columns);
    let committed_shape = (fixed_point.output.rows, fixed_point.output.columns);
    let static_boundary = compilation.block_lane_nc_boundary();
    let static_rounds = compilation.block_lane_nc_rounds();
    let snapshot = PostPiDecSnapshot::from_audit(execution);
    if let Err(error) = snapshot.validate(
        selector_columns,
        source_shape,
        committed_shape,
        static_boundary,
        static_rounds,
    ) {
        panic!("active execution audit is exact: {error}");
    }

    let mut changed_source_resolution = snapshot.clone();
    changed_source_resolution.raw_children[0].normalized_source_column += 1;
    assert!(
        !changed_source_resolution.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed normalized source resolution must fail closed"
    );

    let mut changed_child_order = snapshot.clone();
    changed_child_order.raw_children.swap(0, 270);
    assert!(
        !changed_child_order.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed raw-child ordering must fail closed"
    );

    let mut changed_challenge_order = snapshot.clone();
    changed_challenge_order.rounds.swap(0, 1);
    assert!(
        !changed_challenge_order.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed challenge ordering must fail closed"
    );

    let mut changed_old_block_child_order = snapshot.clone();
    changed_old_block_child_order
        .raw_old_block_children
        .swap(0, 1);
    assert!(
        !changed_old_block_child_order.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed raw old-block child order must fail closed"
    );

    let mut changed_old_block_lane = snapshot.clone();
    changed_old_block_lane.raw_old_block_children[0].active[0] += K::ONE;
    assert!(
        !changed_old_block_lane.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed raw old-block active lane must fail closed"
    );

    let mut changed_old_block_padding = snapshot.clone();
    changed_old_block_padding.raw_old_block_children[0].padding[0] = K::ONE;
    assert!(
        !changed_old_block_padding.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed raw old-block padding must fail closed"
    );

    let mut changed_old_block_point = snapshot.clone();
    changed_old_block_point.raw_old_block_point[0] += K::ONE;
    assert!(
        !changed_old_block_point.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed raw old-block point association must fail closed"
    );

    let mut changed_old_block_parent = snapshot.clone();
    changed_old_block_parent.raw_old_block_parent[0] += K::ONE;
    assert!(
        !changed_old_block_parent.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed raw old-block parent recomposition must fail closed"
    );

    let mut changed_selector = snapshot.clone();
    changed_selector.selectors[2].value = F::ZERO;
    assert!(
        !changed_selector.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed recursive selector must fail closed"
    );

    let mut changed_constant_one = snapshot.clone();
    changed_constant_one.constant_one_binding.2 = F::ZERO;
    assert!(
        !changed_constant_one.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed constant-one binding must fail closed"
    );

    let mut changed_output_partition = snapshot.clone();
    changed_output_partition.fresh_output_count += 1;
    changed_output_partition.running_output_count -= 1;
    assert!(
        !changed_output_partition.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed fresh/running output partition must fail closed"
    );

    let mut changed_output_padding = snapshot.clone();
    changed_output_padding.output_y_zcol_zero_padding[0][0] = K::ONE;
    assert!(
        !changed_output_padding.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed output-y_zcol zero padding must fail closed"
    );

    let mut changed_generated_source = snapshot.clone();
    changed_generated_source.generated_k_bindings[0].builder_columns[0] += 1;
    assert!(
        !changed_generated_source.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed generated builder column must fail closed"
    );

    let mut changed_generated_value = snapshot.clone();
    changed_generated_value.generated_k_bindings[0].value += K::ONE;
    assert!(
        !changed_generated_value.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed generated semantic value must fail closed"
    );

    let mut changed_generated_slot = snapshot.clone();
    changed_generated_slot.generated_k_bindings[0].slot = R1csIvcGeneratedKSlot::BetaLane(0);
    assert!(
        !changed_generated_slot.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed generated semantic slot must fail closed"
    );

    let mut changed_terminal = snapshot.clone();
    changed_terminal.terminal_rhs += K::ONE;
    assert!(
        !changed_terminal.valid(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds
        ),
        "changed terminal mapping must fail closed"
    );
    let proof = chain.finish().expect("finish compact HyperNova proof");

    assert!(
        proof.final_fold.is_none(),
        "plain HyperNova finalization must keep running and latest separate"
    );
    let ProofState::Active { running, latest } = &proof.state.proof else {
        panic!("two-step IVC proof must be active");
    };
    let running = running
        .materialize()
        .expect("CPU fixture has materialized running state");
    assert!(!running.claims.is_empty(), "running accumulator covers prior steps");
    assert_eq!(latest.instances.len(), 1, "latest relation remains a fresh F' instance");
    neo_fold_clean::verify_uncompressed(&prep.prep, &proof)
        .expect("terminal verifier accepts running accumulator plus latest F'");

    let mut bad_latest = proof.clone();
    let ProofState::Active { latest, .. } = &mut bad_latest.state.proof else {
        unreachable!()
    };
    let instance = &mut latest.instances[0];
    let global_column = instance.claim.m_in;
    let packed_coordinate = (global_column % neo_math::D, global_column / neo_math::D);
    instance.witness.Z[packed_coordinate] = if instance.witness.Z[packed_coordinate] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    instance.claim.c = prep.prep.log.commit(&instance.witness.Z);
    neo_fold_clean::verify_uncompressed(&prep.prep, &bad_latest)
        .expect_err("a consistently recommitted latest witness must still fail the F' relation");

    let mut bad_history = proof.clone();
    let ProofState::Active { running, .. } = &mut bad_history.state.proof else {
        unreachable!()
    };
    let running = running
        .as_materialized_mut()
        .expect("CPU fixture has materialized running state");
    running.claims[0].c.data[0] += F::ONE;
    neo_fold_clean::verify_uncompressed(&prep.prep, &bad_history)
        .expect_err("a changed accumulated proof must be rejected");

    drop(bad_history);
    drop(bad_latest);
    drop(proof);
    drop(prep);
}

#[test]
fn stateful_ivc_threads_the_authoritative_application_state() {
    let app = increment_r1cs();
    let initial = semantic_digest(1);
    let plan = make_tiny_stateful_lifecycle_plan_with_anchor(
        app.m(),
        app.m_in,
        vec![1],
        vec![2],
        Some(digest_fields_as_digest32(initial)),
    );
    let prep = R1csIvcPreprocessing::new_seeded(tiny_params(), &app, plan, 0x1F15_C008)
        .expect("compile authoritative stateful R1CS IVC relation");
    let mut chain = R1csIvc::new(&prep);
    chain
        .extend(increment_assignment(1))
        .expect("1 -> 2 base step");
    chain
        .extend(increment_assignment(9))
        .expect_err("a recursive app input disconnected from the carried state must reject");
    chain
        .extend(increment_assignment(2))
        .expect("2 -> 3 recursive step");
    let proof = chain.finish().expect("finish stateful HyperNova proof");
    assert_eq!(
        digest32_as_fields(proof.state.semantic_state_digest),
        semantic_digest(3),
        "terminal public state is H(3)"
    );
    neo_fold_clean::verify_uncompressed(&prep.prep, &proof)
        .expect("stateful running accumulator plus latest F' verifies");
}

#[derive(Clone)]
struct RawChildSnapshot {
    child: usize,
    logical_column: usize,
    witness_coordinate: (usize, usize),
    builder_column: usize,
    normalized_source_column: usize,
    authority: R1csIvcRawAssignmentAuthority,
}

#[derive(Clone)]
struct RawOldBlockChildSnapshot {
    child: usize,
    authority: R1csIvcRawAssignmentAuthority,
    active: [K; 54],
    padding: [K; 10],
}

#[derive(Clone)]
struct SelectorSnapshot {
    arm: R1csIvcBranch,
    logical_column: usize,
    packed_coordinate: (usize, usize),
    value: F,
}

#[derive(Clone)]
struct RoundSnapshot {
    index: usize,
    coefficients: Vec<K>,
    challenge: K,
    claim_in: K,
    claim_out: K,
}

#[derive(Clone)]
struct GeneratedKBindingSnapshot {
    slot: R1csIvcGeneratedKSlot,
    builder_columns: [usize; 2],
    normalized_columns: [usize; 2],
    value: K,
}

#[derive(Clone)]
struct PostPiDecSnapshot {
    branch: R1csIvcBranch,
    proof_variant: PiCcsProofVariant,
    output_profile: (usize, usize, usize),
    source_builder_rows: usize,
    source_builder_columns: usize,
    committed_rows: usize,
    committed_columns: usize,
    public_output_builder_columns: Vec<usize>,
    constant_one_source_builder_column: usize,
    constant_one_binding: (usize, (usize, usize), F),
    public_writes: Vec<(usize, (usize, usize))>,
    selectors: Vec<SelectorSnapshot>,
    full_z_children: Vec<(usize, usize, (usize, usize), std::ops::Range<usize>)>,
    raw_children: Vec<RawChildSnapshot>,
    raw_old_block_profile: R1csIvcRawOldBlockProfile,
    raw_old_block_field_decoding: R1csIvcRawOldBlockFieldDecoding,
    raw_old_block_logical_columns: usize,
    raw_old_block_packed_shape: (usize, usize),
    raw_old_block_point: Vec<K>,
    raw_old_block_children: Vec<RawOldBlockChildSnapshot>,
    raw_old_block_radix: K,
    raw_old_block_parent: Vec<K>,
    pi_ccs_output_count: usize,
    combined_parent_m_in: usize,
    pi_dec_child_count: usize,
    fresh_output_count: usize,
    running_output_count: usize,
    output_y_zcol_padded_lanes: usize,
    output_y_zcol_zero_padding_lanes: usize,
    gamma: K,
    output_y_zcol_active: Vec<[K; neo_math::D]>,
    output_y_zcol_zero_padding: Vec<[K; 10]>,
    producer_beta: K,
    batch_weight: K,
    pending_old_block: Vec<K>,
    pending_parent_y_zcol: Vec<K>,
    beta_block: Vec<K>,
    beta_lane: Vec<K>,
    block_point: Vec<K>,
    lane_point: Vec<K>,
    rounds: Vec<RoundSnapshot>,
    terminal_initial: K,
    terminal_final: K,
    terminal_rhs: K,
    generated_k_bindings: Vec<GeneratedKBindingSnapshot>,
}

impl PostPiDecSnapshot {
    fn from_audit(audit: &R1csIvcPostPiDecExecutionAudit) -> Self {
        let combined = audit.combined_nc();
        let output_profile = combined.output_profile();
        let constant_one = audit.constant_one_binding();
        let terminal = combined.terminal();
        let raw_old_block = audit.raw_old_block();
        Self {
            branch: audit.branch(),
            proof_variant: combined.proof_variant(),
            output_profile: (
                output_profile.source_count(),
                output_profile.matrix_count(),
                output_profile.lane_count(),
            ),
            source_builder_rows: audit.source_builder_rows(),
            source_builder_columns: audit.source_builder_columns(),
            committed_rows: audit.committed_rows(),
            committed_columns: audit.committed_columns(),
            public_output_builder_columns: audit.public_output_builder_columns().to_vec(),
            constant_one_source_builder_column: audit.constant_one_source_builder_column(),
            constant_one_binding: (
                constant_one.logical_column(),
                constant_one.packed_coordinate(),
                constant_one.value(),
            ),
            public_writes: audit
                .public_writes()
                .iter()
                .map(|write| (write.logical_column(), write.packed_coordinate()))
                .collect(),
            selectors: audit
                .selector_writes()
                .iter()
                .map(|write| SelectorSnapshot {
                    arm: write.arm(),
                    logical_column: write.logical_column(),
                    packed_coordinate: write.packed_coordinate(),
                    value: write.value(),
                })
                .collect(),
            full_z_children: audit
                .full_z_children()
                .iter()
                .map(|child| {
                    (
                        child.child(),
                        child.logical_columns(),
                        child.packed_shape(),
                        child.captured_public_coordinates(),
                    )
                })
                .collect(),
            raw_children: audit
                .raw_child_assignments()
                .iter()
                .map(|record| RawChildSnapshot {
                    child: record.child(),
                    logical_column: record.logical_column(),
                    witness_coordinate: record.witness_coordinate(),
                    builder_column: record.builder_column(),
                    normalized_source_column: record.normalized_source_column(),
                    authority: record.authority(),
                })
                .collect(),
            raw_old_block_profile: raw_old_block.profile(),
            raw_old_block_field_decoding: raw_old_block.field_decoding(),
            raw_old_block_logical_columns: raw_old_block.logical_columns(),
            raw_old_block_packed_shape: raw_old_block.packed_shape(),
            raw_old_block_point: raw_old_block.old_block().to_vec(),
            raw_old_block_children: raw_old_block
                .children()
                .iter()
                .map(|child| RawOldBlockChildSnapshot {
                    child: child.child(),
                    authority: child.authority(),
                    active: *child.active_lanes(),
                    padding: *child.zero_padding(),
                })
                .collect(),
            raw_old_block_radix: raw_old_block.radix(),
            raw_old_block_parent: raw_old_block.recomposed_parent_y_zcol().to_vec(),
            pi_ccs_output_count: audit.pi_ccs_output_count(),
            combined_parent_m_in: audit.combined_parent_m_in(),
            pi_dec_child_count: audit.pi_dec_child_count(),
            fresh_output_count: combined.fresh_output_count(),
            running_output_count: combined.running_output_count(),
            output_y_zcol_padded_lanes: combined.output_y_zcol_padded_lanes(),
            output_y_zcol_zero_padding_lanes: combined.output_y_zcol_zero_padding_lanes(),
            gamma: combined.gamma(),
            output_y_zcol_active: combined.output_y_zcol_active().to_vec(),
            output_y_zcol_zero_padding: combined.output_y_zcol_zero_padding().to_vec(),
            producer_beta: combined.producer_beta(),
            batch_weight: combined.batch_weight(),
            pending_old_block: combined.pending_old_block().to_vec(),
            pending_parent_y_zcol: combined.pending_parent_y_zcol().to_vec(),
            beta_block: combined.beta_block().to_vec(),
            beta_lane: combined.beta_lane().to_vec(),
            block_point: combined.block_point().to_vec(),
            lane_point: combined.lane_point().to_vec(),
            rounds: combined
                .rounds()
                .iter()
                .map(|round| RoundSnapshot {
                    index: round.index(),
                    coefficients: round.coefficients().to_vec(),
                    challenge: round.challenge(),
                    claim_in: round.claim_in(),
                    claim_out: round.claim_out(),
                })
                .collect(),
            terminal_initial: terminal.claimed_initial(),
            terminal_final: terminal.final_sum(),
            terminal_rhs: terminal.rhs(),
            generated_k_bindings: audit
                .generated_k_bindings()
                .iter()
                .map(|binding| GeneratedKBindingSnapshot {
                    slot: binding.slot(),
                    builder_columns: binding.builder_columns(),
                    normalized_columns: binding.normalized_columns(),
                    value: binding.value(),
                })
                .collect(),
        }
    }

    fn valid(
        &self,
        selector_columns: &[usize],
        source_shape: (usize, usize),
        committed_shape: (usize, usize),
        static_boundary: &BlockLaneNcBoundaryAudit,
        static_rounds: &[SumcheckRoundAudit],
    ) -> bool {
        self.validate(
            selector_columns,
            source_shape,
            committed_shape,
            static_boundary,
            static_rounds,
        )
        .is_ok()
    }

    fn validate(
        &self,
        selector_columns: &[usize],
        source_shape: (usize, usize),
        committed_shape: (usize, usize),
        static_boundary: &BlockLaneNcBoundaryAudit,
        static_rounds: &[SumcheckRoundAudit],
    ) -> Result<(), String> {
        const PUBLIC_COORDINATES: usize = 270;
        const BUILDER_PUBLIC_OUTPUTS: usize = 256;
        const CHILDREN: usize = 14;
        const OUTPUTS: usize = CHILDREN + 1;
        const MATRICES: usize = 13;
        const ACTIVE_LANES: usize = 54;
        const PADDED_LANES: usize = 64;
        const ZERO_PADDING_LANES: usize = PADDED_LANES - ACTIVE_LANES;
        const BLOCK_ROUNDS: usize = 19;
        const LANE_ROUNDS: usize = 6;
        const ROUND_COEFFICIENTS: usize = 5;
        const GENERATED_K_BINDINGS: usize = 1
            + LANE_ROUNDS
            + BLOCK_ROUNDS
            + 2
            + BLOCK_ROUNDS
            + ACTIVE_LANES
            + OUTPUTS * PADDED_LANES
            + BLOCK_ROUNDS
            + LANE_ROUNDS
            + 3
            + (BLOCK_ROUNDS + LANE_ROUNDS) * (ROUND_COEFFICIENTS + 3);

        if self.branch != R1csIvcBranch::Recursive
            || self.proof_variant != PiCcsProofVariant::BlockLaneNcDelayedV1
            || self.output_profile != (OUTPUTS, MATRICES, ACTIVE_LANES)
            || (self.source_builder_rows, self.source_builder_columns) != source_shape
            || (self.committed_rows, self.committed_columns) != committed_shape
            || self.public_output_builder_columns.len() != BUILDER_PUBLIC_OUTPUTS
            || self.public_writes.len() != PUBLIC_COORDINATES
            || self.pi_ccs_output_count != OUTPUTS
            || self.combined_parent_m_in != PUBLIC_COORDINATES
            || self.pi_dec_child_count != CHILDREN
            || self.full_z_children.len() != CHILDREN
            || self.raw_children.len() != CHILDREN * PUBLIC_COORDINATES
            || self.raw_old_block_profile != R1csIvcRawOldBlockProfile::ActiveFPrimeCombinedNcDelayedV1
            || self.raw_old_block_field_decoding != R1csIvcRawOldBlockFieldDecoding::BaseFieldEmbedding
            || self.raw_old_block_logical_columns != self.committed_columns
            || self.raw_old_block_packed_shape != (ACTIVE_LANES, self.committed_columns.div_ceil(ACTIVE_LANES))
            || self.raw_old_block_point != self.pending_old_block
            || self.raw_old_block_children.len() != CHILDREN
            || self.raw_old_block_radix != K::from(F::from_u64(2))
            || self.raw_old_block_parent != self.pending_parent_y_zcol
            || self.fresh_output_count != 1
            || self.running_output_count != CHILDREN
            || self.output_y_zcol_padded_lanes != PADDED_LANES
            || self.output_y_zcol_zero_padding_lanes != ZERO_PADDING_LANES
            || self.output_y_zcol_active.len() != OUTPUTS
            || self.output_y_zcol_zero_padding.len() != OUTPUTS
            || self.pending_old_block.len() != BLOCK_ROUNDS
            || self.pending_parent_y_zcol.len() != ACTIVE_LANES
            || self.beta_block.len() != BLOCK_ROUNDS
            || self.beta_lane.len() != LANE_ROUNDS
            || self.block_point.len() != BLOCK_ROUNDS
            || self.lane_point.len() != LANE_ROUNDS
            || self.rounds.len() != BLOCK_ROUNDS + LANE_ROUNDS
            || self.generated_k_bindings.len() != GENERATED_K_BINDINGS
        {
            return Err(format!(
                "header mismatch: branch={:?} variant={:?} profile={:?} source={}x{} committed={}x{} output_map={} public_writes={} pi_ccs_outputs={} partition={}/{} parent_m_in={} pi_dec_children={} full_z={} raw_children={} output_tables={}/{} output_lanes={}/{} pending={}/{} beta={}/{} point={}/{} rounds={} generated_bindings={}",
                self.branch,
                self.proof_variant,
                self.output_profile,
                self.source_builder_rows,
                self.source_builder_columns,
                self.committed_rows,
                self.committed_columns,
                self.public_output_builder_columns.len(),
                self.public_writes.len(),
                self.pi_ccs_output_count,
                self.fresh_output_count,
                self.running_output_count,
                self.combined_parent_m_in,
                self.pi_dec_child_count,
                self.full_z_children.len(),
                self.raw_children.len(),
                self.output_y_zcol_active.len(),
                self.output_y_zcol_zero_padding.len(),
                self.output_y_zcol_padded_lanes,
                self.output_y_zcol_zero_padding_lanes,
                self.pending_old_block.len(),
                self.pending_parent_y_zcol.len(),
                self.beta_block.len(),
                self.beta_lane.len(),
                self.block_point.len(),
                self.lane_point.len(),
                self.rounds.len(),
                self.generated_k_bindings.len(),
            ));
        }

        if self.constant_one_source_builder_column != 0 || self.constant_one_binding != (0, (0, 0), F::ONE) {
            return Err(format!(
                "constant-one binding mismatch: source={} binding={:?}",
                self.constant_one_source_builder_column, self.constant_one_binding,
            ));
        }

        if let Some((source, lane)) = self
            .output_y_zcol_zero_padding
            .iter()
            .enumerate()
            .find_map(|(source, table)| {
                table
                    .iter()
                    .position(|&value| value != K::ZERO)
                    .map(|lane| (source, lane))
            })
        {
            return Err(format!(
                "output y_zcol source {source} has nonzero padding lane {}",
                ACTIVE_LANES + lane
            ));
        }

        let mut recomposed = [K::ZERO; ACTIVE_LANES];
        let mut radix_power = K::ONE;
        for (child, record) in self.raw_old_block_children.iter().enumerate() {
            if record.child != child
                || record.authority != R1csIvcRawAssignmentAuthority::RunningWitnessMat
                || record.padding.iter().any(|&value| value != K::ZERO)
            {
                return Err(format!(
                    "raw old-block child {child} order, authority, or padding drift"
                ));
            }
            for lane in 0..ACTIVE_LANES {
                recomposed[lane] += record.active[lane] * radix_power;
            }
            radix_power *= self.raw_old_block_radix;
        }
        if recomposed.as_slice() != self.raw_old_block_parent {
            return Err("raw old-block child lanes do not recompose to the pending parent".into());
        }

        if self.public_output_builder_columns.contains(&0)
            || self
                .public_output_builder_columns
                .iter()
                .any(|&column| column >= self.source_builder_columns)
            || self
                .public_output_builder_columns
                .windows(2)
                .any(|columns| columns[0] >= columns[1])
        {
            return Err("builder public-output normalization map is not strictly ordered and in range".into());
        }

        let expected_generated_schedule = expected_generated_k_schedule(self);
        if expected_generated_schedule.len() != GENERATED_K_BINDINGS {
            return Err("generated K-binding semantic schedule has the wrong length".into());
        }
        let expected_generated_columns = expected_generated_k_columns(static_boundary, static_rounds)?;
        if expected_generated_columns.len() != GENERATED_K_BINDINGS {
            return Err("compiled generated K-column schedule has the wrong length".into());
        }
        if let Some((index, binding)) = self
            .generated_k_bindings
            .iter()
            .enumerate()
            .find(|(index, binding)| {
                binding.builder_columns[0] == binding.builder_columns[1]
                    || binding.normalized_columns[0] == binding.normalized_columns[1]
                    || binding.builder_columns != expected_generated_columns[*index]
                    || (binding.slot, binding.value) != expected_generated_schedule[*index]
                    || (0..2).any(|limb| {
                        normalized_target_column(
                            self.source_builder_columns,
                            &self.public_output_builder_columns,
                            binding.builder_columns[limb],
                        ) != Some(binding.normalized_columns[limb])
                    })
            })
        {
            return Err(format!(
                "generated K binding {index} mismatch: builder={:?} normalized={:?}",
                binding.builder_columns, binding.normalized_columns,
            ));
        }

        if self
            .public_writes
            .iter()
            .enumerate()
            .any(|(logical, record)| *record != (logical, (logical % neo_math::D, logical / neo_math::D)))
        {
            return Err("public-write ordering mismatch".into());
        }

        let arms = [
            R1csIvcBranch::Base,
            R1csIvcBranch::BootstrapRecursive,
            R1csIvcBranch::Recursive,
        ];
        if selector_columns.len() != arms.len()
            || self.selectors.iter().enumerate().any(|(index, selector)| {
                selector.arm != arms[index]
                    || selector.logical_column != selector_columns[index]
                    || selector.packed_coordinate
                        != (
                            selector.logical_column % neo_math::D,
                            selector.logical_column / neo_math::D,
                        )
                    || selector.value != if index == 2 { F::ONE } else { F::ZERO }
            })
        {
            return Err(format!(
                "selector mismatch: selector_columns={selector_columns:?}, selectors={}",
                self.selectors.len()
            ));
        }

        let packed_columns = self.committed_columns.div_ceil(neo_math::D);
        if self
            .full_z_children
            .iter()
            .enumerate()
            .any(|(child, record)| {
                *record
                    != (
                        child,
                        self.committed_columns,
                        (neo_math::D, packed_columns),
                        child * PUBLIC_COORDINATES..(child + 1) * PUBLIC_COORDINATES,
                    )
            })
        {
            return Err("full-Z child geometry mismatch".into());
        }

        if let Some((index, record)) = self
            .raw_children
            .iter()
            .enumerate()
            .find(|(index, record)| {
                let child = index / PUBLIC_COORDINATES;
                let logical = index % PUBLIC_COORDINATES;
                record.child != child
                    || record.logical_column != logical
                    || record.witness_coordinate != (logical % neo_math::D, logical / neo_math::D)
                    || record.authority != R1csIvcRawAssignmentAuthority::RunningWitnessMat
                    || normalized_target_column(
                        self.source_builder_columns,
                        &self.public_output_builder_columns,
                        record.builder_column,
                    ) != Some(record.normalized_source_column)
            })
        {
            return Err(format!(
                "raw-child mapping mismatch at record {index}: child={} logical={} witness={:?} builder={} normalized={} authority={:?}",
                record.child,
                record.logical_column,
                record.witness_coordinate,
                record.builder_column,
                record.normalized_source_column,
                record.authority,
            ));
        }

        let expected_initial = self
            .pending_parent_y_zcol
            .iter()
            .rev()
            .fold(K::ZERO, |value, coefficient| value * self.producer_beta + *coefficient)
            * self.batch_weight;
        if self.terminal_initial != expected_initial {
            return Err("combined-NC claimed-initial mapping mismatch".into());
        }
        let expected_rhs = replay_terminal_rhs(self)?;

        let mut claim = self.terminal_initial;
        for (index, round) in self.rounds.iter().enumerate() {
            if round.index != index
                || round.coefficients.len() != ROUND_COEFFICIENTS
                || round.claim_in != claim
                || round.coefficients[0] + round.coefficients.iter().copied().sum::<K>() != claim
                || polynomial_evaluation(&round.coefficients, round.challenge) != round.claim_out
                || round.challenge
                    != if index < BLOCK_ROUNDS {
                        self.block_point[index]
                    } else {
                        self.lane_point[index - BLOCK_ROUNDS]
                    }
            {
                return Err(format!(
                    "sumcheck round {index} mismatch: stored_index={} coeffs={} claim_chain={} sum_identity={} evaluation={} challenge={}",
                    round.index,
                    round.coefficients.len(),
                    round.claim_in == claim,
                    round.coefficients[0] + round.coefficients.iter().copied().sum::<K>() == claim,
                    polynomial_evaluation(&round.coefficients, round.challenge) == round.claim_out,
                    round.challenge
                        == if index < BLOCK_ROUNDS {
                            self.block_point[index]
                        } else {
                            self.lane_point[index - BLOCK_ROUNDS]
                        },
                ));
            }
            claim = round.claim_out;
        }
        if claim != self.terminal_final || self.terminal_final != self.terminal_rhs || self.terminal_rhs != expected_rhs
        {
            return Err(format!(
                "terminal mismatch: claim_chain={} terminal_identity={} output_mapping={}",
                claim == self.terminal_final,
                self.terminal_final == self.terminal_rhs,
                self.terminal_rhs == expected_rhs,
            ));
        }
        Ok(())
    }
}

fn expected_generated_k_schedule(snapshot: &PostPiDecSnapshot) -> Vec<(R1csIvcGeneratedKSlot, K)> {
    let mut schedule = Vec::new();
    schedule.push((R1csIvcGeneratedKSlot::Gamma, snapshot.gamma));
    schedule.extend(
        snapshot
            .beta_lane
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::BetaLane(index), value)),
    );
    schedule.extend(
        snapshot
            .beta_block
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::BetaBlock(index), value)),
    );
    schedule.push((R1csIvcGeneratedKSlot::ProducerBeta, snapshot.producer_beta));
    schedule.push((R1csIvcGeneratedKSlot::BatchWeight, snapshot.batch_weight));
    schedule.extend(
        snapshot
            .pending_old_block
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::PendingOldBlock(index), value)),
    );
    schedule.extend(
        snapshot
            .pending_parent_y_zcol
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::PendingParentYZcol(index), value)),
    );
    for (source, (active, padding)) in snapshot
        .output_y_zcol_active
        .iter()
        .zip(&snapshot.output_y_zcol_zero_padding)
        .enumerate()
    {
        schedule.extend(
            active
                .iter()
                .chain(padding)
                .copied()
                .enumerate()
                .map(|(lane, value)| (R1csIvcGeneratedKSlot::OutputYZcol { source, lane }, value)),
        );
    }
    schedule.extend(
        snapshot
            .block_point
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::BlockPoint(index), value)),
    );
    schedule.extend(
        snapshot
            .lane_point
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::LanePoint(index), value)),
    );
    schedule.push((R1csIvcGeneratedKSlot::ClaimedInitial, snapshot.terminal_initial));
    schedule.push((R1csIvcGeneratedKSlot::FinalSum, snapshot.terminal_final));
    schedule.push((R1csIvcGeneratedKSlot::TerminalRhs, snapshot.terminal_rhs));
    for (round_index, round) in snapshot.rounds.iter().enumerate() {
        schedule.extend(
            round
                .coefficients
                .iter()
                .copied()
                .enumerate()
                .map(|(coefficient, value)| {
                    (
                        R1csIvcGeneratedKSlot::RoundCoefficient {
                            round: round_index,
                            coefficient,
                        },
                        value,
                    )
                }),
        );
        schedule.push((R1csIvcGeneratedKSlot::RoundChallenge(round_index), round.challenge));
        schedule.push((R1csIvcGeneratedKSlot::RoundClaimIn(round_index), round.claim_in));
        schedule.push((R1csIvcGeneratedKSlot::RoundClaimOut(round_index), round.claim_out));
    }
    schedule
}

fn expected_generated_k_columns(
    boundary: &BlockLaneNcBoundaryAudit,
    rounds: &[SumcheckRoundAudit],
) -> Result<Vec<[usize; 2]>, String> {
    let pending_old_block = boundary
        .pending_old_block_cols
        .as_deref()
        .ok_or_else(|| "compiled delayed-NC boundary omits pending old-block columns".to_string())?;
    let pending_parent_y_zcol = boundary
        .pending_parent_y_zcol_cols
        .as_deref()
        .ok_or_else(|| "compiled delayed-NC boundary omits pending parent-y_zcol columns".to_string())?;

    let mut columns = Vec::new();
    columns.push(boundary.gamma_cols);
    columns.extend(boundary.beta_lane_cols.iter().copied());
    columns.extend(boundary.beta_block_cols.iter().copied());
    columns.push(boundary.producer_beta_cols);
    columns.push(boundary.batch_weight_cols);
    columns.extend(pending_old_block.iter().copied());
    columns.extend(pending_parent_y_zcol.iter().copied());
    columns.extend(boundary.output_y_zcol_cols.iter().flatten().copied());
    columns.extend(boundary.block_point_cols.iter().copied());
    columns.extend(boundary.lane_point_cols.iter().copied());
    columns.push(boundary.claimed_initial_cols);
    columns.push(boundary.final_sum_cols);
    columns.push(boundary.terminal_rhs_cols);
    for round in rounds {
        columns.extend(round.coefficient_cols.iter().copied());
        columns.push(round.challenge_cols);
        columns.push(round.claim_in_cols);
        columns.push(round.claim_out_cols);
    }
    Ok(columns)
}

fn replay_terminal_rhs(snapshot: &PostPiDecSnapshot) -> Result<K, String> {
    if snapshot.output_y_zcol_active.len() != snapshot.output_y_zcol_zero_padding.len()
        || snapshot.fresh_output_count > snapshot.output_y_zcol_active.len()
    {
        return Err("output-y_zcol table partition mismatch".into());
    }

    let mut ordinary = K::ZERO;
    let mut gamma_power = K::ONE;
    let mut running_evaluation = K::ZERO;
    let mut radix_power = K::ONE;
    let radix = K::from(F::from_u64(2));
    for (source, (active, padding)) in snapshot
        .output_y_zcol_active
        .iter()
        .zip(&snapshot.output_y_zcol_zero_padding)
        .enumerate()
    {
        let value = output_y_zcol_evaluation(active, padding, &snapshot.lane_point)?;
        ordinary += gamma_power * (value * value * value - value);
        gamma_power *= snapshot.gamma;
        if source >= snapshot.fresh_output_count {
            running_evaluation += radix_power * value;
            radix_power *= radix;
        }
    }
    ordinary *= equality_evaluation(&snapshot.block_point, &snapshot.beta_block)?
        * equality_evaluation(&snapshot.lane_point, &snapshot.beta_lane)?;
    let delayed = snapshot.batch_weight
        * equality_evaluation(&snapshot.block_point, &snapshot.pending_old_block)?
        * beta_power_selector(snapshot.producer_beta, &snapshot.lane_point)
        * running_evaluation;
    Ok(ordinary + delayed)
}

fn output_y_zcol_evaluation(active: &[K; neo_math::D], padding: &[K; 10], point: &[K]) -> Result<K, String> {
    let domain = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| "output-y_zcol lane domain overflow".to_string())?;
    if domain != active.len() + padding.len() {
        return Err(format!(
            "output-y_zcol table has {} lanes, expected {domain}",
            active.len() + padding.len()
        ));
    }
    let mut result = K::ZERO;
    for index in 0..domain {
        let value = if index < active.len() {
            active[index]
        } else {
            padding[index - active.len()]
        };
        let mut weight = K::ONE;
        for (bit, &coordinate) in point.iter().enumerate() {
            weight *= if (index >> bit) & 1 == 1 {
                coordinate
            } else {
                K::ONE - coordinate
            };
        }
        result += value * weight;
    }
    Ok(result)
}

fn equality_evaluation(left: &[K], right: &[K]) -> Result<K, String> {
    if left.len() != right.len() {
        return Err("combined-NC equality-point dimension mismatch".into());
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(K::ONE, |value, (&left, &right)| {
            value * ((K::ONE - left) * (K::ONE - right) + left * right)
        }))
}

fn beta_power_selector(producer_beta: K, point: &[K]) -> K {
    let mut beta_power = producer_beta;
    let mut result = K::ONE;
    for &coordinate in point {
        result *= (K::ONE - coordinate) + coordinate * beta_power;
        beta_power *= beta_power;
    }
    result
}

fn normalized_target_column(source_columns: usize, public_outputs: &[usize], source: usize) -> Option<usize> {
    if source >= source_columns {
        return None;
    }
    if source == 0 {
        return Some(0);
    }
    if let Some(public_index) = public_outputs.iter().position(|&output| output == source) {
        return Some(public_index + 1);
    }
    let public_before = public_outputs
        .iter()
        .filter(|&&output| output < source)
        .count();
    Some(1 + public_outputs.len() + (source - 1 - public_before))
}

fn polynomial_evaluation(coefficients: &[K], point: K) -> K {
    coefficients
        .iter()
        .rev()
        .fold(K::ZERO, |value, coefficient| value * point + *coefficient)
}

fn increment_r1cs() -> R1cs {
    let mut a = Mat::zero(1, neo_math::D, F::ZERO);
    let mut b = Mat::zero(1, neo_math::D, F::ZERO);
    let mut c = Mat::zero(1, neo_math::D, F::ZERO);
    a[(0, 0)] = F::ONE;
    a[(0, 1)] = F::ONE;
    b[(0, 0)] = F::ONE;
    c[(0, 2)] = F::ONE;
    R1cs { a, b, c, m_in: 3 }
}

fn increment_assignment(input: u64) -> Vec<F> {
    let mut assignment = vec![F::ZERO; neo_math::D];
    assignment[0] = F::ONE;
    assignment[1] = F::from_u64(input);
    assignment[2] = F::from_u64(input + 1);
    assignment
}

fn semantic_digest(value: u64) -> [F; 4] {
    encode_poseidon_trace(&build_semantic_state_preimage_fields(&[F::from_u64(value)])).digest_native
}
