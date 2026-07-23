//! Runtime old-block, public-write, transcript, and terminal artifact rendering.
//!
//! Owns the compact proof-free certificate extracted after the active
//! post-PiDEC recursive arm, its fail-closed validation against the static
//! production row schedule, and bounded generated Lean shards.
//!
//! Does not own commitment binding, child sidecar values, security events,
//! Lean field decoding, or terminal-decider absolute columns. Its generated
//! pending columns are post-PiDEC source/normalized pins in a different
//! builder coordinate space.
use std::fmt::Write as _;

use neo_fold_clean::engine::r1cs_circuit::{
    RawOldBlockProjectionPlan, RawOldBlockProjectionProgram, RawOldBlockProjectionRowOwner,
};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    R1csIvcBlockLaneNcSelectiveRowsAudit, R1csIvcBranch, R1csIvcGeneratedKSlot, R1csIvcPostPiDecExecutionAudit,
    R1csIvcPublicWriteSource, R1csIvcRawAssignmentAuthority, R1csIvcRawOldBlockFieldDecoding,
    R1csIvcRawOldBlockProfile,
};
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::PiCcsProofVariant;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::execution_format::{
    lean_k, lean_k_list, lean_nat_pairs, lean_nat_values, lean_option_nat, slot_tag, source_tag,
};
use super::execution_projection::{
    execution_root, final_scale_program_fragment, final_scale_row_dispatch_fragment, list_root,
};
use super::render::source_shape;
use super::{GeneratedLeanFile, GENERATED_ROOT, IMPORT_ROOT, NAMESPACE_ROOT};

const SCHEMA_VERSION: usize = 1;
const BRANCH_RECURSIVE: usize = 2;
const PROOF_VARIANT_DELAYED_V1: usize = 1;
const RAW_AUTHORITY_RUNNING_WITNESS_MAT: usize = 0;
const FIELD_DECODING_BASE_EMBEDDING: usize = 0;
const CHILD_COUNT: usize = 14;
const ACTIVE_LANES: usize = 54;
const PADDED_LANES: usize = 64;
const PADDING_LANES: usize = PADDED_LANES - ACTIVE_LANES;
const LOGICAL_COLUMNS: usize = 11_437_038;
const PACKED_COLUMNS: usize = 211_797;
const PUBLIC_WRITES: usize = 270;
const BUILDER_PUBLIC_WRITES: usize = 256;
const OUTPUTS: usize = 15;
const MATRICES: usize = 13;
const BLOCK_ROUNDS: usize = 19;
const TENSOR_ROUNDS: usize = 18;
const LANE_ROUNDS: usize = 6;
const ROUNDS: usize = BLOCK_ROUNDS + LANE_ROUNDS;
const RAW_PROJECTION_TENSOR_MULTIPLICATIONS: usize = 262_143;
const RAW_PROJECTION_FINAL_SCALE_MULTIPLICATIONS: usize = 54;
const RAW_PROJECTION_DERIVED_COLUMNS: usize = 24_185_061;
const RAW_PROJECTION_ROWS: usize = 24_185_169;
const RAW_PROJECTION_PENDING_JOIN_ID: usize = 1;
const ROUND_COEFFICIENTS: usize = 5;
const PUBLIC_WRITE_CHUNK: usize = 135;
const LANE_CHUNK: usize = 224;
const BINDING_CHUNK: usize = 224;
const GENERATED_K_BINDINGS: usize = 1_289;

use super::execution_projection::generated_header as execution_generated_header;

#[derive(Clone, Copy)]
struct PublicWriteRecord {
    logical_column: usize,
    packed_row: usize,
    packed_column: usize,
    source: R1csIvcPublicWriteSource,
    builder_column: Option<usize>,
    normalized_source_column: Option<usize>,
    normalized_column: usize,
    width: usize,
    centered: bool,
    alias_source: Option<usize>,
    value: F,
}

#[derive(Clone, Copy)]
struct LaneRecord {
    child: usize,
    lane: usize,
    padding: bool,
    value: K,
}

#[derive(Clone)]
struct RoundRecord {
    index: usize,
    coefficients: Vec<K>,
    challenge: K,
    claim_in: K,
    claim_out: K,
}

#[derive(Clone, Copy)]
struct BindingRecord {
    slot: R1csIvcGeneratedKSlot,
    builder_columns: [usize; 2],
    normalized_columns: [usize; 2],
    value: K,
}

#[derive(Clone)]
pub(crate) struct ExecutionCertificate {
    branch: R1csIvcBranch,
    proof_variant: PiCcsProofVariant,
    output_profile: (usize, usize, usize),
    fresh_count: usize,
    running_count: usize,
    logical_columns: usize,
    packed_rows: usize,
    packed_columns: usize,
    source_rows: usize,
    source_columns: usize,
    final_rows: usize,
    final_columns: usize,
    public_output_builder_columns: Vec<usize>,
    public_writes: Vec<PublicWriteRecord>,
    selector_columns: Vec<usize>,
    selector_values: Vec<F>,
    gamma: K,
    output_y_zcol: Vec<Vec<K>>,
    raw_profile: R1csIvcRawOldBlockProfile,
    field_decoding: R1csIvcRawOldBlockFieldDecoding,
    old_block: Vec<K>,
    parent_y_zcol: Vec<K>,
    radix: K,
    raw_authority: R1csIvcRawAssignmentAuthority,
    raw_projection_plan: RawOldBlockProjectionPlan,
    lanes: Vec<LaneRecord>,
    producer_beta: K,
    batch_weight: K,
    beta_block: Vec<K>,
    beta_lane: Vec<K>,
    block_point: Vec<K>,
    lane_point: Vec<K>,
    rounds: Vec<RoundRecord>,
    terminal_initial: K,
    terminal_final: K,
    terminal_rhs: K,
    bindings: Vec<BindingRecord>,
}

impl ExecutionCertificate {
    pub(crate) fn capture(audit: &R1csIvcPostPiDecExecutionAudit) -> Self {
        let combined = audit.combined_nc();
        let profile = combined.output_profile();
        let raw = audit.raw_old_block();
        let terminal = combined.terminal();
        let selector_columns = audit
            .selector_writes()
            .iter()
            .map(|write| write.logical_column())
            .collect();
        let selector_values = audit
            .selector_writes()
            .iter()
            .map(|write| write.value())
            .collect();
        let mut lanes = Vec::with_capacity(CHILD_COUNT * PADDED_LANES);
        for child in raw.children() {
            lanes.extend(
                child
                    .active_lanes()
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(lane, value)| LaneRecord {
                        child: child.child(),
                        lane,
                        padding: false,
                        value,
                    }),
            );
            lanes.extend(
                child
                    .zero_padding()
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(offset, value)| LaneRecord {
                        child: child.child(),
                        lane: ACTIVE_LANES + offset,
                        padding: true,
                        value,
                    }),
            );
        }
        Self {
            branch: audit.branch(),
            proof_variant: combined.proof_variant(),
            output_profile: (profile.source_count(), profile.matrix_count(), profile.lane_count()),
            fresh_count: combined.fresh_output_count(),
            running_count: combined.running_output_count(),
            logical_columns: raw.logical_columns(),
            packed_rows: raw.packed_shape().0,
            packed_columns: raw.packed_shape().1,
            source_rows: audit.source_builder_rows(),
            source_columns: audit.source_builder_columns(),
            final_rows: audit.committed_rows(),
            final_columns: audit.committed_columns(),
            public_output_builder_columns: audit.public_output_builder_columns().to_vec(),
            public_writes: audit
                .public_writes()
                .iter()
                .map(|write| PublicWriteRecord {
                    logical_column: write.logical_column(),
                    packed_row: write.packed_coordinate().0,
                    packed_column: write.packed_coordinate().1,
                    source: write.source(),
                    builder_column: write.builder_column(),
                    normalized_source_column: write.normalized_source_column(),
                    normalized_column: write.normalized_column(),
                    width: write.width(),
                    centered: write.centered(),
                    alias_source: write.alias_source(),
                    value: write.value(),
                })
                .collect(),
            selector_columns,
            selector_values,
            gamma: combined.gamma(),
            output_y_zcol: combined
                .output_y_zcol_active()
                .iter()
                .zip(combined.output_y_zcol_zero_padding())
                .map(|(active, padding)| active.iter().chain(padding).copied().collect())
                .collect(),
            raw_profile: raw.profile(),
            field_decoding: raw.field_decoding(),
            old_block: raw.old_block().to_vec(),
            parent_y_zcol: raw.recomposed_parent_y_zcol().to_vec(),
            radix: raw.radix(),
            raw_authority: raw
                .children()
                .first()
                .expect("fixed raw child family")
                .authority(),
            raw_projection_plan: raw.projection_plan(),
            lanes,
            producer_beta: combined.producer_beta(),
            batch_weight: combined.batch_weight(),
            beta_block: combined.beta_block().to_vec(),
            beta_lane: combined.beta_lane().to_vec(),
            block_point: combined.block_point().to_vec(),
            lane_point: combined.lane_point().to_vec(),
            rounds: combined
                .rounds()
                .iter()
                .map(|round| RoundRecord {
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
            bindings: audit
                .generated_k_bindings()
                .iter()
                .map(|binding| BindingRecord {
                    slot: binding.slot(),
                    builder_columns: binding.builder_columns(),
                    normalized_columns: binding.normalized_columns(),
                    value: binding.value(),
                })
                .collect(),
        }
    }

    fn validate(&self, static_audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> Result<(), String> {
        let (expected_source_rows, expected_source_columns) = source_shape(static_audit);
        let projected = static_audit.projected_rows();
        let row_program = RawOldBlockProjectionProgram::new(self.raw_projection_plan, 2).map_err(str::to_owned)?;
        let plan = self.raw_projection_plan;
        let product_first = plan.tensor_rows();
        let final_scale_first = product_first + plan.projection_product_rows();
        let terminal_first = final_scale_first + plan.final_scale_rows();
        if self.branch != R1csIvcBranch::Recursive
            || self.proof_variant != PiCcsProofVariant::BlockLaneNcDelayedV1
            || self.output_profile != (OUTPUTS, MATRICES, ACTIVE_LANES)
            || PADDING_LANES != 10
            || self.fresh_count != 1
            || self.running_count != CHILD_COUNT
            || self.logical_columns != LOGICAL_COLUMNS
            || self.packed_rows != D
            || self.packed_columns != PACKED_COLUMNS
            || self.source_rows != expected_source_rows
            || self.source_columns != expected_source_columns
            || self.final_rows != projected.rows()
            || self.final_columns != projected.columns()
            || self.final_columns != LOGICAL_COLUMNS
            || self.raw_profile != R1csIvcRawOldBlockProfile::ActiveFPrimeCombinedNcDelayedV1
            || self.field_decoding != R1csIvcRawOldBlockFieldDecoding::BaseFieldEmbedding
            || self.raw_authority != R1csIvcRawAssignmentAuthority::RunningWitnessMat
            || self.raw_projection_plan
                != RawOldBlockProjectionPlan::new(LOGICAL_COLUMNS, CHILD_COUNT)
                    .expect("fixed production raw projection plan")
            || self.raw_projection_plan.witness_flat_index(1, 0) != Some(PACKED_COLUMNS)
            || self.raw_projection_plan.witness_flat_index(0, 1) != Some(1)
            || !self.raw_projection_plan.factor_final_round()
            || self.raw_projection_plan.tensor_variables() != TENSOR_ROUNDS
            || self.raw_projection_plan.factored_variable() != Some(TENSOR_ROUNDS)
            || self.raw_projection_plan.tensor_mul_count() != RAW_PROJECTION_TENSOR_MULTIPLICATIONS
            || self.raw_projection_plan.final_scale_mul_count() != RAW_PROJECTION_FINAL_SCALE_MULTIPLICATIONS
            || self.raw_projection_plan.tensor_rows()
                + self.raw_projection_plan.projection_product_rows()
                + self.raw_projection_plan.final_scale_rows()
                != RAW_PROJECTION_DERIVED_COLUMNS
            || self.raw_projection_plan.total_rows() != RAW_PROJECTION_ROWS
            || row_program.row_count() != RAW_PROJECTION_ROWS
            || row_program.owner(0)
                != Some(RawOldBlockProjectionRowOwner::Tensor {
                    round: 0,
                    parent: 0,
                    k_row: 0,
                })
            || row_program.owner(product_first - 1)
                != Some(RawOldBlockProjectionRowOwner::Tensor {
                    round: TENSOR_ROUNDS - 1,
                    parent: (1 << (TENSOR_ROUNDS - 1)) - 1,
                    k_row: 4,
                })
            || row_program.owner(product_first)
                != Some(RawOldBlockProjectionRowOwner::Product {
                    lane: 0,
                    block: 0,
                    limb: 0,
                })
            || row_program.owner(terminal_first - 1)
                != Some(RawOldBlockProjectionRowOwner::FinalScale {
                    lane: ACTIVE_LANES - 1,
                    k_row: 4,
                })
            || row_program.owner(final_scale_first)
                != Some(RawOldBlockProjectionRowOwner::FinalScale { lane: 0, k_row: 0 })
            || row_program.owner(final_scale_first - 1)
                != Some(RawOldBlockProjectionRowOwner::Product {
                    lane: ACTIVE_LANES - 1,
                    block: PACKED_COLUMNS - 1,
                    limb: 1,
                })
            || row_program.owner(terminal_first) != Some(RawOldBlockProjectionRowOwner::Terminal { lane: 0, limb: 0 })
            || row_program.owner(RAW_PROJECTION_ROWS - 1)
                != Some(RawOldBlockProjectionRowOwner::Terminal {
                    lane: ACTIVE_LANES - 1,
                    limb: 1,
                })
            || row_program.owner(RAW_PROJECTION_ROWS).is_some()
            || [
                0,
                product_first - 1,
                product_first,
                final_scale_first - 1,
                final_scale_first,
                terminal_first - 1,
                terminal_first,
                RAW_PROJECTION_ROWS - 1,
            ]
            .into_iter()
            .any(|row| row_program.row_at(row).is_none())
            || self.output_y_zcol.len() != OUTPUTS
            || self.output_y_zcol.iter().any(|output| {
                output.len() != PADDED_LANES || output[ACTIVE_LANES..].iter().any(|&value| value != K::ZERO)
            })
            || self.old_block.len() != BLOCK_ROUNDS
            || self.parent_y_zcol.len() != ACTIVE_LANES
            || self.beta_block.len() != BLOCK_ROUNDS
            || self.beta_lane.len() != LANE_ROUNDS
            || self.block_point.len() != BLOCK_ROUNDS
            || self.lane_point.len() != LANE_ROUNDS
            || self.rounds.len() != ROUNDS
        {
            return Err("execution header does not match the fixed production profile".into());
        }
        self.validate_public_writes(static_audit)?;
        self.validate_lanes()?;
        self.validate_rounds()?;
        self.validate_bindings(static_audit)?;
        Ok(())
    }

    fn validate_public_writes(&self, static_audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> Result<(), String> {
        if self.selector_columns != static_audit.projected_rows().selector_columns()
            || self.selector_values != [F::ZERO, F::ZERO, F::ONE]
        {
            return Err("recursive-selector profile drift".into());
        }
        self.validate_public_write_profile()
    }

    fn validate_public_write_profile(&self) -> Result<(), String> {
        if self.branch != R1csIvcBranch::Recursive
            || self.logical_columns != LOGICAL_COLUMNS
            || self.packed_rows != D
            || self.packed_columns != PACKED_COLUMNS
            || self.final_columns != LOGICAL_COLUMNS
        {
            return Err("active public-write execution profile drift".into());
        }
        self.validate_public_write_records()
    }

    fn validate_public_write_records(&self) -> Result<(), String> {
        if self.public_output_builder_columns.len() != BUILDER_PUBLIC_WRITES
            || self.public_output_builder_columns.contains(&0)
            || self
                .public_output_builder_columns
                .iter()
                .any(|&column| column >= self.source_columns)
            || self
                .public_output_builder_columns
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.public_writes.len() != PUBLIC_WRITES
        {
            return Err("public-write or recursive-selector profile drift".into());
        }
        for (logical, record) in self.public_writes.iter().copied().enumerate() {
            let (source, builder, normalized_source, width) = if logical == 0 {
                (R1csIvcPublicWriteSource::ConstantOne, Some(0), None, 1)
            } else if logical <= BUILDER_PUBLIC_WRITES {
                (
                    R1csIvcPublicWriteSource::BuilderColumn,
                    Some(self.public_output_builder_columns[logical - 1]),
                    Some(logical),
                    1,
                )
            } else {
                (R1csIvcPublicWriteSource::FixedZero, None, None, 0)
            };
            if record.logical_column != logical
                || (record.packed_row, record.packed_column) != (logical % D, logical / D)
                || record.source != source
                || record.builder_column != builder
                || record.normalized_source_column != normalized_source
                || record.normalized_column != logical
                || record.width != width
                || record.centered
                || record.alias_source.is_some()
                || (source == R1csIvcPublicWriteSource::ConstantOne && record.value != F::ONE)
                || (source == R1csIvcPublicWriteSource::FixedZero && record.value != F::ZERO)
            {
                return Err(format!("public-write record {logical} is not exact"));
            }
        }
        Ok(())
    }

    fn validate_lanes(&self) -> Result<(), String> {
        if self.lanes.len() != CHILD_COUNT * PADDED_LANES || self.radix != K::from(F::from_u64(2)) {
            return Err("raw old-block lane cardinality or radix drift".into());
        }
        let mut recomposed = [K::ZERO; ACTIVE_LANES];
        let mut power = K::ONE;
        for child in 0..CHILD_COUNT {
            let start = child * PADDED_LANES;
            for lane in 0..PADDED_LANES {
                let record = self.lanes[start + lane];
                if record.child != child
                    || record.lane != lane
                    || record.padding != (lane >= ACTIVE_LANES)
                    || (record.padding && record.value != K::ZERO)
                {
                    return Err(format!("raw old-block lane ({child}, {lane}) order or padding drift"));
                }
                if lane < ACTIVE_LANES {
                    recomposed[lane] += record.value * power;
                }
            }
            power *= self.radix;
        }
        if recomposed.as_slice() != self.parent_y_zcol {
            return Err("raw old-block lanes do not radix-recompose to the pending parent".into());
        }
        Ok(())
    }

    fn validate_rounds(&self) -> Result<(), String> {
        let mut claim = self.terminal_initial;
        for (index, round) in self.rounds.iter().enumerate() {
            if round.index != index
                || round.coefficients.len() != ROUND_COEFFICIENTS
                || round.claim_in != claim
                || polynomial_evaluation(&round.coefficients, round.challenge) != round.claim_out
            {
                return Err(format!("combined-NC round {index} mapping drift"));
            }
            claim = round.claim_out;
        }
        if claim != self.terminal_final || self.terminal_final != self.terminal_rhs {
            return Err("combined-NC terminal chain or equality drift".into());
        }
        Ok(())
    }

    fn validate_bindings(&self, static_audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> Result<(), String> {
        let schedule = expected_binding_schedule(self);
        let columns = expected_binding_columns(static_audit)?;
        if self.bindings.len() != GENERATED_K_BINDINGS
            || self.bindings.len() != schedule.len()
            || self.bindings.len() != columns.len()
        {
            return Err(format!(
                "generated K binding cardinality drift: actual={} schedule={} columns={}",
                self.bindings.len(),
                schedule.len(),
                columns.len()
            ));
        }
        for (index, ((binding, (slot, value)), builder_columns)) in
            self.bindings.iter().zip(schedule).zip(columns).enumerate()
        {
            let normalized_columns = [
                normalized_target_column(
                    self.source_columns,
                    &self.public_output_builder_columns,
                    builder_columns[0],
                )
                .ok_or_else(|| format!("binding {index} source c0 escapes"))?,
                normalized_target_column(
                    self.source_columns,
                    &self.public_output_builder_columns,
                    builder_columns[1],
                )
                .ok_or_else(|| format!("binding {index} source c1 escapes"))?,
            ];
            if binding.slot != slot
                || binding.value != value
                || binding.builder_columns != builder_columns
                || binding.normalized_columns != normalized_columns
                || binding.builder_columns[0] == binding.builder_columns[1]
                || binding.normalized_columns[0] == binding.normalized_columns[1]
                || binding
                    .normalized_columns
                    .iter()
                    .any(|&column| column >= self.source_columns)
            {
                return Err(format!("generated K binding {index} mapping drift"));
            }
        }
        Ok(())
    }
}

fn expected_binding_schedule(certificate: &ExecutionCertificate) -> Vec<(R1csIvcGeneratedKSlot, K)> {
    let mut schedule = Vec::new();
    schedule.push((R1csIvcGeneratedKSlot::Gamma, certificate.gamma));
    schedule.extend(
        certificate
            .beta_lane
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::BetaLane(index), value)),
    );
    schedule.extend(
        certificate
            .beta_block
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::BetaBlock(index), value)),
    );
    schedule.push((R1csIvcGeneratedKSlot::ProducerBeta, certificate.producer_beta));
    schedule.push((R1csIvcGeneratedKSlot::BatchWeight, certificate.batch_weight));
    schedule.extend(
        certificate
            .old_block
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::PendingOldBlock(index), value)),
    );
    schedule.extend(
        certificate
            .parent_y_zcol
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::PendingParentYZcol(index), value)),
    );
    for (source, output) in certificate.output_y_zcol.iter().enumerate() {
        schedule.extend(
            output
                .iter()
                .copied()
                .enumerate()
                .map(|(lane, value)| (R1csIvcGeneratedKSlot::OutputYZcol { source, lane }, value)),
        );
    }
    schedule.extend(
        certificate
            .block_point
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::BlockPoint(index), value)),
    );
    schedule.extend(
        certificate
            .lane_point
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (R1csIvcGeneratedKSlot::LanePoint(index), value)),
    );
    schedule.push((R1csIvcGeneratedKSlot::ClaimedInitial, certificate.terminal_initial));
    schedule.push((R1csIvcGeneratedKSlot::FinalSum, certificate.terminal_final));
    schedule.push((R1csIvcGeneratedKSlot::TerminalRhs, certificate.terminal_rhs));
    for round in &certificate.rounds {
        schedule.extend(
            round
                .coefficients
                .iter()
                .copied()
                .enumerate()
                .map(|(coefficient, value)| {
                    (
                        R1csIvcGeneratedKSlot::RoundCoefficient {
                            round: round.index,
                            coefficient,
                        },
                        value,
                    )
                }),
        );
        schedule.push((R1csIvcGeneratedKSlot::RoundChallenge(round.index), round.challenge));
        schedule.push((R1csIvcGeneratedKSlot::RoundClaimIn(round.index), round.claim_in));
        schedule.push((R1csIvcGeneratedKSlot::RoundClaimOut(round.index), round.claim_out));
    }
    schedule
}

fn expected_binding_columns(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> Result<Vec<[usize; 2]>, String> {
    let boundary = audit.boundary();
    let old_block = boundary
        .pending_old_block_cols
        .as_deref()
        .ok_or_else(|| "static boundary omits pending old-block columns".to_string())?;
    let parent = boundary
        .pending_parent_y_zcol_cols
        .as_deref()
        .ok_or_else(|| "static boundary omits pending parent columns".to_string())?;
    let mut columns = vec![boundary.gamma_cols];
    columns.extend(boundary.beta_lane_cols.iter().copied());
    columns.extend(boundary.beta_block_cols.iter().copied());
    columns.push(boundary.producer_beta_cols);
    columns.push(boundary.batch_weight_cols);
    columns.extend(old_block.iter().copied());
    columns.extend(parent.iter().copied());
    columns.extend(boundary.output_y_zcol_cols.iter().flatten().copied());
    columns.extend(boundary.block_point_cols.iter().copied());
    columns.extend(boundary.lane_point_cols.iter().copied());
    columns.push(boundary.claimed_initial_cols);
    columns.push(boundary.final_sum_cols);
    columns.push(boundary.terminal_rhs_cols);
    for round in audit.rounds() {
        columns.extend(round.coefficient_cols.iter().copied());
        columns.push(round.challenge_cols);
        columns.push(round.claim_in_cols);
        columns.push(round.claim_out_cols);
    }
    Ok(columns)
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

fn header(certificate: &ExecutionCertificate) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Execution.Header");
    let mut contents =
        execution_generated_header("the fixed execution profile, pending state, challenges, and terminal values");
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("execution header import");
    writeln!(contents, "namespace {namespace}\n").expect("execution header namespace");
    writeln!(contents, "def value : RawExecutionHeader :=").expect("execution header def");
    writeln!(contents, "  {{ schemaVersion := {SCHEMA_VERSION}").expect("schema");
    writeln!(contents, "    branch := {BRANCH_RECURSIVE}").expect("branch");
    writeln!(contents, "    proofVariant := {PROOF_VARIANT_DELAYED_V1}").expect("variant");
    writeln!(contents, "    outputSources := {}", certificate.output_profile.0).expect("sources");
    writeln!(contents, "    outputMatrices := {}", certificate.output_profile.1).expect("matrices");
    writeln!(contents, "    outputActiveLanes := {}", certificate.output_profile.2).expect("lanes");
    writeln!(contents, "    freshCount := {}", certificate.fresh_count).expect("fresh");
    writeln!(contents, "    runningCount := {}", certificate.running_count).expect("running");
    writeln!(contents, "    logicalColumns := {}", certificate.logical_columns).expect("logical");
    writeln!(contents, "    packedRows := {}", certificate.packed_rows).expect("rows");
    writeln!(contents, "    packedColumns := {}", certificate.packed_columns).expect("columns");
    writeln!(contents, "    sourceRows := {}", certificate.source_rows).expect("source rows");
    writeln!(contents, "    sourceColumns := {}", certificate.source_columns).expect("source cols");
    writeln!(contents, "    finalRows := {}", certificate.final_rows).expect("final rows");
    writeln!(contents, "    finalColumns := {}", certificate.final_columns).expect("final cols");
    writeln!(contents, "    publicWriteCount := {}", certificate.public_writes.len()).expect("writes");
    writeln!(contents, "    selectorColumns := {:?}", certificate.selector_columns).expect("selectors");
    writeln!(
        contents,
        "    selectorValues := {}",
        lean_nat_values(&certificate.selector_values)
    )
    .expect("selector values");
    writeln!(contents, "    oldBlock := {}", lean_k_list(&certificate.old_block)).expect("old block");
    writeln!(
        contents,
        "    parentYZcol := {}",
        lean_k_list(&certificate.parent_y_zcol)
    )
    .expect("parent");
    writeln!(contents, "    radix := {}", lean_k(certificate.radix)).expect("radix");
    writeln!(contents, "    producerBeta := {}", lean_k(certificate.producer_beta)).expect("producer beta");
    writeln!(contents, "    batchWeight := {}", lean_k(certificate.batch_weight)).expect("batch weight");
    writeln!(contents, "    betaBlock := {}", lean_k_list(&certificate.beta_block)).expect("beta block");
    writeln!(contents, "    betaLane := {}", lean_k_list(&certificate.beta_lane)).expect("beta lane");
    writeln!(contents, "    blockPoint := {}", lean_k_list(&certificate.block_point)).expect("block point");
    writeln!(contents, "    lanePoint := {}", lean_k_list(&certificate.lane_point)).expect("lane point");
    writeln!(
        contents,
        "    terminalInitial := {}",
        lean_k(certificate.terminal_initial)
    )
    .expect("initial");
    writeln!(contents, "    terminalFinal := {}", lean_k(certificate.terminal_final)).expect("final");
    writeln!(contents, "    terminalRhs := {}", lean_k(certificate.terminal_rhs)).expect("rhs");
    writeln!(contents, "    rawAuthorityTag := {RAW_AUTHORITY_RUNNING_WITNESS_MAT}").expect("authority");
    writeln!(contents, "    fieldDecodingTag := {FIELD_DECODING_BASE_EMBEDDING} }}").expect("field decode");
    writeln!(contents, "\nend {namespace}").expect("execution header end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/Header.lean"),
        contents,
    }
}

fn raw_old_block_projection_plan(certificate: &ExecutionCertificate) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Execution.RawOldBlockProjectionPlan");
    let plan = certificate.raw_projection_plan;
    let round_mul_counts = (0..plan.tensor_variables())
        .map(|round| {
            plan.tensor_round_mul_count(round)
                .expect("fixed tensor round")
        })
        .collect::<Vec<_>>();
    let round_high_counts = (0..plan.tensor_variables())
        .map(|round| {
            plan.tensor_round_high_count(round)
                .expect("fixed tensor round")
        })
        .collect::<Vec<_>>();
    let mut next_row = 0;
    let round_row_starts = round_mul_counts
        .iter()
        .map(|count| {
            let start = next_row;
            next_row += 5 * count;
            start
        })
        .collect::<Vec<_>>();
    assert_eq!(next_row, plan.tensor_rows(), "tensor round partition");

    let binding_columns = |slot_for_index: fn(usize) -> R1csIvcGeneratedKSlot, count: usize| {
        (0..count)
            .map(|index| {
                certificate
                    .bindings
                    .iter()
                    .find(|binding| binding.slot == slot_for_index(index))
                    .unwrap_or_else(|| panic!("missing generated K binding at index {index}"))
                    .normalized_columns
            })
            .collect::<Vec<_>>()
    };
    let old_block_columns = binding_columns(R1csIvcGeneratedKSlot::PendingOldBlock, BLOCK_ROUNDS);
    let parent_columns = binding_columns(R1csIvcGeneratedKSlot::PendingParentYZcol, ACTIVE_LANES);
    let witness_entries_per_child = plan.active_lanes() * plan.packed_columns();
    let witness_family_entries = plan.child_count() * witness_entries_per_child;
    let child_witness_relative_bases = (0..plan.child_count())
        .map(|child| child * witness_entries_per_child)
        .collect::<Vec<_>>();

    let mut contents = execution_generated_header(
        "the compact direct-terminal raw-WitnessMat projection program and active post-PiDEC source-normalized pins",
    );
    writeln!(contents, "import {IMPORT_ROOT}\n\nnamespace {namespace}\n").expect("plan import");
    for (name, value) in [
        ("schemaVersion", SCHEMA_VERSION),
        ("profileTag", PROOF_VARIANT_DELAYED_V1),
        ("pendingProjectionProfileTag", PROOF_VARIANT_DELAYED_V1),
        ("pendingProjectionJoinId", RAW_PROJECTION_PENDING_JOIN_ID),
        ("recursiveSelectorTag", BRANCH_RECURSIVE),
        ("radixBase", 2),
        ("logicalWidth", plan.logical_columns()),
        ("packedRows", plan.packed_rows()),
        ("blockCount", plan.packed_columns()),
        ("blockVariables", plan.block_variables()),
        ("tensorVariables", plan.tensor_variables()),
        ("blockDomainSize", plan.block_domain_size()),
        ("childCount", plan.child_count()),
        ("activeLanes", plan.active_lanes()),
        ("paddedLanes", plan.padded_lanes()),
        ("virtualZeroLanes", plan.virtual_zero_lanes()),
        ("witnessEntriesPerChild", witness_entries_per_child),
        ("witnessFamilyEntries", witness_family_entries),
        ("tensorMultiplications", plan.tensor_mul_count()),
        ("tensorRows", plan.tensor_rows()),
        ("projectionProductRows", plan.projection_product_rows()),
        ("finalScaleMultiplications", plan.final_scale_mul_count()),
        ("finalScaleRows", plan.final_scale_rows()),
        (
            "derivedColumns",
            plan.tensor_rows() + plan.projection_product_rows() + plan.final_scale_rows(),
        ),
        ("terminalRows", plan.terminal_rows()),
        ("totalRows", plan.total_rows()),
        ("tensorRelativeRowStart", 0),
        ("projectionProductRelativeRowStart", plan.tensor_rows()),
        (
            "terminalRelativeRowStart",
            plan.tensor_rows() + plan.projection_product_rows() + plan.final_scale_rows(),
        ),
        ("tensorFirstColumnAfterWitnessFamily", witness_family_entries),
        (
            "projectionProductFirstColumnAfterWitnessFamily",
            witness_family_entries + plan.tensor_rows(),
        ),
        (
            "finalScaleFirstColumnAfterWitnessFamily",
            witness_family_entries + plan.tensor_rows() + plan.projection_product_rows(),
        ),
    ] {
        writeln!(contents, "def {name} : Nat := {value}").expect("plan scalar");
    }
    writeln!(contents, "def factorFinalRound : Bool := {}", plan.factor_final_round()).expect("factor mode");
    writeln!(
        contents,
        "def factoredVariable : Option Nat := some {}",
        plan.factored_variable()
            .expect("production factor variable")
    )
    .expect("factor variable");
    contents.push_str(
        "\n/-- Row-major `FinalWitnessWires` offset: lane first, then packed block. -/\n\
         def witnessOffset (lane block : Nat) : Nat := lane * blockCount + block\n\
         def childWitnessRelativeColumn (child lane block : Nat) : Nat :=\n\
           child * witnessEntriesPerChild + witnessOffset lane block\n\
         def tensorRoundMulCount (round : Nat) : Nat := Nat.min blockCount (2 ^ round)\n\
         def tensorMulOrdinal (round parent : Nat) : Nat :=\n\
           (List.range round).foldl (fun count prior => count + tensorRoundMulCount prior) 0 + parent\n\
         def tensorMulFirstColumnAfterWitnessFamily (round parent : Nat) : Nat :=\n\
           tensorFirstColumnAfterWitnessFamily + 5 * tensorMulOrdinal round parent\n\
         def tensorMulOutputColumnsAfterWitnessFamily (round parent : Nat) : Nat × Nat :=\n\
           (tensorMulFirstColumnAfterWitnessFamily round parent + 3,\n\
            tensorMulFirstColumnAfterWitnessFamily round parent + 4)\n\
         def projectionProductRelativeRow (lane block limb : Nat) : Nat :=\n\
           projectionProductRelativeRowStart + 2 * (lane * blockCount + block) + limb\n\
         def projectionProductColumnAfterWitnessFamily (lane block limb : Nat) : Nat :=\n\
           projectionProductFirstColumnAfterWitnessFamily + 2 * (lane * blockCount + block) + limb\n\
         def finalScaleRelativeRow (lane definition : Nat) : Nat :=\n\
           projectionProductRelativeRowStart + projectionProductRows + 5 * lane + definition\n\
         def terminalRelativeRow (lane limb : Nat) : Nat :=\n\
           terminalRelativeRowStart + 2 * lane + limb\n\n",
    );
    writeln!(
        contents,
        "def tensorRoundMulCounts : List Nat := {:?}",
        round_mul_counts
    )
    .expect("round counts");
    writeln!(
        contents,
        "def tensorRoundHighCounts : List Nat := {:?}",
        round_high_counts
    )
    .expect("high counts");
    writeln!(
        contents,
        "def tensorRoundRelativeRowStarts : List Nat := {:?}",
        round_row_starts
    )
    .expect("round starts");
    writeln!(
        contents,
        "def childWitnessRelativeBases : List Nat := {:?}",
        child_witness_relative_bases
    )
    .expect("child bases");
    writeln!(
        contents,
        "/-- Post-PiDEC source-normalized pins; not terminal absolute columns. -/\ndef pendingOldBlockSourceNormalizedColumns : List (Nat × Nat) := {}",
        lean_nat_pairs(&old_block_columns)
    )
    .expect("old-block columns");
    writeln!(
        contents,
        "/-- Post-PiDEC source-normalized pins; not terminal absolute columns. -/\ndef pendingParentYZcolSourceNormalizedColumns : List (Nat × Nat) := {}",
        lean_nat_pairs(&parent_columns)
    )
    .expect("parent columns");
    writeln!(
        contents,
        "def recursiveSelectorColumns : List Nat := {:?}",
        certificate.selector_columns
    )
    .expect("selector columns");
    writeln!(
        contents,
        "def recursiveSelectorValues : List Nat := {}",
        lean_nat_values(&certificate.selector_values)
    )
    .expect("selector values");
    writeln!(contents, "\nend {namespace}").expect("plan end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/RawOldBlockProjectionPlan.lean"),
        contents,
    }
}

fn raw_old_block_projection_row_at(plan: RawOldBlockProjectionPlan) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Execution.RawOldBlockProjectionRowAt");
    let plan_namespace = format!("{NAMESPACE_ROOT}.Execution.RawOldBlockProjectionPlan");
    let program = RawOldBlockProjectionProgram::new(plan, 2).expect("fixed active raw-old-block row program");
    let layout = program.layout();
    let round_counts = (0..plan.tensor_variables())
        .map(|round| {
            plan.tensor_round_mul_count(round)
                .expect("fixed tensor round")
        })
        .collect::<Vec<_>>();
    let round_high_counts = (0..plan.tensor_variables())
        .map(|round| {
            plan.tensor_round_high_count(round)
                .expect("fixed tensor round")
        })
        .collect::<Vec<_>>();
    let mut multiplication_start = 0;
    let round_starts = round_counts
        .iter()
        .map(|count| {
            let start = multiplication_start;
            multiplication_start += count;
            start
        })
        .collect::<Vec<_>>();
    assert_eq!(multiplication_start, plan.tensor_mul_count());
    let mut tensor_owner = format!(
        "({}, ordinal - {})",
        plan.tensor_variables() - 1,
        round_starts[plan.tensor_variables() - 1]
    );
    for round in (0..plan.tensor_variables() - 1).rev() {
        let stop = round_starts[round] + round_counts[round];
        tensor_owner = format!(
            "if ordinal < {stop} then ({round}, ordinal - {}) else ({tensor_owner})",
            round_starts[round]
        );
    }

    let mut contents = execution_generated_header(
        "the Rust-owned active raw-old-block owner, inverse offsets, canonical columns, and exact indexed A/B/C row formula",
    );
    writeln!(
        contents,
        "import Nightstream.Implementation.R1CS.Core.Projection.Trace\nimport {plan_namespace}\n\nnamespace {namespace}\n\nopen Nightstream.Implementation.R1CS\nopen Nightstream.Implementation.R1CS.Program\nopen Nightstream.Implementation.R1CS.ProjectionProgram\nopen {plan_namespace}\n"
    )
    .expect("row-at imports");
    for (name, value) in [
        ("constantOneColumn", 0),
        ("oldBlockFirstColumn", layout.old_block_first()),
        ("parentFirstColumn", layout.parent_first()),
        ("witnessFamilyFirstColumn", layout.witness_family_first()),
        ("tensorFirstColumn", layout.tensor_first()),
        ("productFirstColumn", layout.product_first()),
        ("finalScaleFirstColumn", layout.final_scale_first()),
        ("canonicalColumnCount", layout.column_count()),
        ("tensorVariables", plan.tensor_variables()),
        (
            "factoredVariable",
            plan.factored_variable()
                .expect("production factored variable"),
        ),
        ("tensorRows", plan.tensor_rows()),
        ("productRows", plan.projection_product_rows()),
        ("finalScaleRows", plan.final_scale_rows()),
        ("terminalRows", plan.terminal_rows()),
        ("totalRows", plan.total_rows()),
        ("productRowFirst", plan.tensor_rows()),
        (
            "finalScaleRowFirst",
            plan.tensor_rows() + plan.projection_product_rows(),
        ),
        (
            "terminalRowFirst",
            plan.tensor_rows() + plan.projection_product_rows() + plan.final_scale_rows(),
        ),
    ] {
        writeln!(contents, "def {name} : Nat := {value}").expect("row-at scalar");
    }
    writeln!(contents, "def tensorRoundMulCounts : List Nat := {:?}", round_counts).expect("row-at counts");
    writeln!(
        contents,
        "def tensorRoundHighCounts : List Nat := {:?}",
        round_high_counts
    )
    .expect("row-at high counts");
    writeln!(contents, "def tensorRoundMulStarts : List Nat := {:?}", round_starts).expect("row-at starts");
    contents.push_str(
        r#"
def kColumnsAt (first : Nat) : KColumns := { c0 := first, c1 := first + 1 }
def oldBlockColumnsNat (round : Nat) : KColumns :=
  kColumnsAt (oldBlockFirstColumn + 2 * round)
def oldBlockColumns (round : Fin blockVariables) : KColumns :=
  oldBlockColumnsNat round.val
def parentColumnsNat (lane : Nat) : KColumns :=
  kColumnsAt (parentFirstColumn + 2 * lane)
def parentColumns (lane : Fin activeLanes) : KColumns :=
  parentColumnsNat lane.val
def witnessEntriesPerChild : Nat := activeLanes * blockCount
def childWitnessFirstNat (child : Nat) : Nat :=
  witnessFamilyFirstColumn + child * witnessEntriesPerChild
def childWitnessFirst (child : Fin childCount) : Nat :=
  childWitnessFirstNat child.val
def witnessOffset (lane block : Nat) : Nat := lane * blockCount + block
def childWitnessColumn (child lane block : Nat) : Nat :=
  childWitnessFirstNat child + witnessOffset lane block
def tensorRoundMulCount (round : Nat) : Nat := Nat.min blockCount (2 ^ round)
def tensorRoundHighCount (round : Nat) : Nat :=
  Nat.min (blockCount - 2 ^ round) (2 ^ round)
def tensorRoundMulStart (round : Nat) : Nat :=
  (List.range round).foldl (fun count prior => count + tensorRoundMulCount prior) 0
def tensorMulOrdinal (round parent : Nat) : Nat :=
  tensorRoundMulStart round + parent
def tensorMulFirstColumn (round parent : Nat) : Nat :=
  tensorFirstColumn + 5 * tensorMulOrdinal round parent
def tensorOutputColumns (round parent : Nat) : KColumns :=
  kColumnsAt (tensorMulFirstColumn round parent + 3)
def productColumn (lane block limb : Nat) : Nat :=
  productFirstColumn + 2 * witnessOffset lane block + limb
def finalScaleOutput (lane : Nat) : KColumns :=
  kColumnsAt (finalScaleFirstColumn + 5 * lane + 3)
def tensorPhysicalRow (round parent definition : Nat) : Nat :=
  5 * tensorMulOrdinal round parent + definition
def productPhysicalRow (lane block limb : Nat) : Nat :=
  productRowFirst + 2 * witnessOffset lane block + limb
def finalScalePhysicalRow (lane definition : Nat) : Nat :=
  finalScaleRowFirst + 5 * lane + definition
def terminalPhysicalRow (lane limb : Nat) : Nat :=
  terminalRowFirst + 2 * lane + limb
def emptyRow : Row := { a := [], b := [], c := [] }
def emptyKTerms : KTerms := { c0 := [], c1 := [] }
def tensorRoot : KTerms := { c0 := [(constantOneColumn, 1)], c1 := [] }
def pointTerms (round : Nat) : KTerms := KTerms.ofColumns (oldBlockColumnsNat round)
def oneMinusPointTerms (round : Nat) : KTerms :=
  let point := oldBlockColumnsNat round
  { c0 := [(constantOneColumn, 1), (point.c0, goldilocksP - 1)]
    c1 := [(point.c1, goldilocksP - 1)] }
def subtractOutput (terms : KTerms) (output : KColumns) : KTerms :=
  { c0 := terms.c0 ++ [(output.c0, goldilocksP - 1)]
    c1 := terms.c1 ++ [(output.c1, goldilocksP - 1)] }
def tensorTermsAt : Nat -> Nat -> KTerms
  | 0, index => if index = 0 then tensorRoot else emptyKTerms
  | round + 1, index =>
      let count := tensorRoundMulCount round
      let high := tensorRoundHighCount round
      if index < count then
        if index < high then
          subtractOutput (tensorTermsAt round index)
            (tensorOutputColumns round index)
        else KTerms.ofColumns (tensorOutputColumns round index)
      else
        let parent := index - count
        if parent < high then
          KTerms.ofColumns (tensorOutputColumns round parent)
        else emptyKTerms
def tensorTrace (round parent : Nat) : KMulTrace :=
  let left := tensorTermsAt round parent
  let right := if parent < tensorRoundHighCount round then
    pointTerms round else oneMinusPointTerms round
  let first := tensorMulFirstColumn round parent
  { left := left
    right := right
    sumLeft := left.c0 ++ left.c1
    sumRight := right.c0 ++ right.c1
    productC0 := first
    productC1 := first + 1
    productSum := first + 2
    output := kColumnsAt (first + 3) }
def rawTerms (lane block : Nat) : List (Nat × Nat) :=
  (List.range childCount).map fun child =>
    (childWitnessColumn child lane block, radixBase ^ child % goldilocksP)
def chiTerms (block : Nat) : KTerms :=
  tensorTermsAt tensorVariables block

"#,
    );
    contents.push_str(final_scale_program_fragment());
    writeln!(
        contents,
        "def tensorOwner (ordinal : Nat) : Nat × Nat :=\n  {tensor_owner}"
    )
    .expect("row-at tensor owner");
    contents.push_str(final_scale_row_dispatch_fragment());
    contents.push_str(
        r#"/-- Serialization of the crate-private `RawOldBlockProjectionColumnMap`
stored in `TerminalPendingProjectionAudit`. Production constructs it
inside `enforce_raw_old_block_projection`; it is not a prover or
theorem-caller authority input. -/
structure EmitterLayout where
  rowFirst : Nat
  rowStop : Nat
  oldBlock : List KColumns
  parent : List KColumns
  finalWitnessFirst : List Nat
  tensorFirst : Nat
  productFirst : Nat
  finalScaleFirst : Nat
deriving DecidableEq, Repr
structure ColumnInterval where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr
def allDistinct : List Nat -> Bool
  | [] => true
  | head :: tail => !(tail.contains head) && allDistinct tail
def emitterScalarColumns (emitter : EmitterLayout) : List Nat :=
  [constantOneColumn] ++
  emitter.oldBlock.flatMap (fun columns => [columns.c0, columns.c1]) ++
  emitter.parent.flatMap (fun columns => [columns.c0, columns.c1])
def childIntervals (emitter : EmitterLayout) : List ColumnInterval :=
  emitter.finalWitnessFirst.map fun first =>
    { start := first, stop := first + witnessEntriesPerChild }
def emitterIntervals (emitter : EmitterLayout) : List ColumnInterval :=
  childIntervals emitter ++
  [ { start := emitter.tensorFirst, stop := emitter.productFirst },
    { start := emitter.productFirst, stop := emitter.finalScaleFirst },
    { start := emitter.finalScaleFirst, stop := emitter.finalScaleFirst + finalScaleRows } ]
def intervalsDisjoint (left right : ColumnInterval) : Bool :=
  decide (left.stop <= right.start) || decide (right.stop <= left.start)
def intervalsPairwiseDisjoint : List ColumnInterval -> Bool
  | [] => true
  | head :: tail =>
      tail.all (intervalsDisjoint head) && intervalsPairwiseDisjoint tail
def intervalContains (interval : ColumnInterval) (column : Nat) : Bool :=
  decide (interval.start <= column) && decide (column < interval.stop)
def scalarsOutsideIntervals (emitter : EmitterLayout) : Bool :=
  (emitterScalarColumns emitter).all fun column =>
    !((emitterIntervals emitter).any fun interval => intervalContains interval column)
def selectLimb (columns : KColumns) (limb : Nat) : Nat :=
  if limb = 0 then columns.c0 else columns.c1
def emitterColumnMap (emitter : EmitterLayout) (column : Nat) : Nat :=
  if column = constantOneColumn then constantOneColumn
  else if column < parentFirstColumn then
    let offset := column - oldBlockFirstColumn
    selectLimb (emitter.oldBlock.getD (offset / 2) default) (offset % 2)
  else if column < witnessFamilyFirstColumn then
    let offset := column - parentFirstColumn
    selectLimb (emitter.parent.getD (offset / 2) default) (offset % 2)
  else if column < tensorFirstColumn then
    let offset := column - witnessFamilyFirstColumn
    emitter.finalWitnessFirst.getD (offset / witnessEntriesPerChild) 0 +
      offset % witnessEntriesPerChild
  else if column < productFirstColumn then
    emitter.tensorFirst + column - tensorFirstColumn
  else if column < finalScaleFirstColumn then
    emitter.productFirst + column - productFirstColumn
  else emitter.finalScaleFirst + column - finalScaleFirstColumn
def findKColumnAux : List KColumns -> Nat -> Nat -> Option (Nat × Nat)
  | [], _, _ => none
  | head :: tail, index, column =>
      if column = head.c0 then some (index, 0)
      else if column = head.c1 then some (index, 1)
      else findKColumnAux tail (index + 1) column
def findWitnessIntervalAux : List Nat -> Nat -> Nat -> Option (Nat × Nat)
  | [], _, _ => none
  | first :: tail, child, column =>
      if first ≤ column ∧ column < first + witnessEntriesPerChild then
        some (child, column - first)
      else findWitnessIntervalAux tail (child + 1) column
def emitterColumnInverse (emitter : EmitterLayout) (column : Nat) : Option Nat :=
  if column = constantOneColumn then some constantOneColumn
  else if emitter.tensorFirst ≤ column ∧ column < emitter.productFirst then
    some (tensorFirstColumn + (column - emitter.tensorFirst))
  else if emitter.productFirst ≤ column ∧ column < emitter.productFirst + productRows then
    some (productFirstColumn + (column - emitter.productFirst))
  else if emitter.finalScaleFirst ≤ column ∧ column < emitter.finalScaleFirst + finalScaleRows then
    some (finalScaleFirstColumn + (column - emitter.finalScaleFirst))
  else
    match findKColumnAux emitter.oldBlock 0 column with
    | some (round, limb) => some (oldBlockFirstColumn + 2 * round + limb)
    | none =>
        match findKColumnAux emitter.parent 0 column with
        | some (lane, limb) => some (parentFirstColumn + 2 * lane + limb)
        | none =>
            match findWitnessIntervalAux emitter.finalWitnessFirst 0 column with
            | some (child, offset) =>
                some (witnessFamilyFirstColumn + child * witnessEntriesPerChild + offset)
            | none => none
def physicalRow (emitter : EmitterLayout) (row : Fin totalRows) : Nat :=
  emitter.rowFirst + row.val
def expectedRowStop (emitter : EmitterLayout) : Nat := emitter.rowFirst + totalRows
def emitterShapePinned (emitter : EmitterLayout) : Bool :=
  emitter.rowStop == expectedRowStop emitter &&
  emitter.oldBlock.length == blockVariables &&
  emitter.parent.length == activeLanes &&
  emitter.finalWitnessFirst.length == childCount &&
  emitter.productFirst == emitter.tensorFirst + tensorRows &&
  emitter.finalScaleFirst == emitter.productFirst + productRows
def emitterColumnMapValid (emitter : EmitterLayout) : Bool :=
  emitterShapePinned emitter &&
  allDistinct (emitterScalarColumns emitter) &&
  intervalsPairwiseDisjoint (emitterIntervals emitter) &&
  scalarsOutsideIntervals emitter
def actualRow (emitter : EmitterLayout) (row : Fin totalRows) : Row :=
  renameRow (emitterColumnMap emitter) (artifactRow row)
def projectionChildWitnessFirst (emitter : EmitterLayout) (child : Nat) : Nat :=
  emitter.finalWitnessFirst.getD child 0
def ajtaiChildWitnessFirst (emitter : EmitterLayout) (child : Nat) : Nat :=
  emitter.finalWitnessFirst.getD child 0
"#,
    );
    writeln!(contents, "\nend {namespace}").expect("row-at end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/RawOldBlockProjectionRowAt.lean"),
        contents,
    }
}

fn public_write_shards(certificate: &ExecutionCertificate) -> Vec<GeneratedLeanFile> {
    let chunks = certificate
        .public_writes
        .chunks(PUBLIC_WRITE_CHUNK)
        .collect::<Vec<_>>();
    assert_eq!(
        chunks.iter().map(|chunk| chunk.len()).collect::<Vec<_>>(),
        vec![135, 135]
    );
    chunks.into_iter().enumerate().map(|(chunk_index, chunk)| {
        let namespace = format!("{NAMESPACE_ROOT}.Execution.PublicWrites.Chunk{chunk_index}");
        let mut contents = execution_generated_header("exactly 135 active runtime public-write records");
        writeln!(contents, "import {IMPORT_ROOT}\n\nnamespace {namespace}\n").expect("public import");
        contents.push_str("def values : List RawPublicWrite := [\n");
        for (index, record) in chunk.iter().enumerate() {
            if index != 0 { contents.push_str(",\n"); }
            write!(contents, "  {{ schemaVersion := {SCHEMA_VERSION}, logicalColumn := {}, packedRow := {}, packedColumn := {}, sourceKind := {}, builderColumn := {}, normalizedSourceColumn := {}, normalizedColumn := {}, width := {}, centered := {}, aliasSource := {}, value := {} }}", record.logical_column, record.packed_row, record.packed_column, source_tag(record.source), lean_option_nat(record.builder_column), lean_option_nat(record.normalized_source_column), record.normalized_column, record.width, record.centered, lean_option_nat(record.alias_source), record.value.as_canonical_u64()).expect("public record");
        }
        writeln!(contents, "\n]\n\nend {namespace}").expect("public end");
        GeneratedLeanFile { relative_path: format!("{GENERATED_ROOT}/Execution/PublicWrites/Chunk{chunk_index}.lean"), contents }
    }).collect()
}

pub(super) fn focused_public_write_files(certificate: &ExecutionCertificate) -> Vec<GeneratedLeanFile> {
    certificate
        .validate_public_write_profile()
        .expect("exact active public-write execution certificate");
    public_write_shards(certificate)
}

fn lane_shards(certificate: &ExecutionCertificate) -> Vec<GeneratedLeanFile> {
    let chunks = certificate.lanes.chunks(LANE_CHUNK).collect::<Vec<_>>();
    assert_eq!(chunks.iter().map(|chunk| chunk.len()).collect::<Vec<_>>(), vec![224; 4]);
    chunks
        .into_iter()
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let namespace = format!("{NAMESPACE_ROOT}.Execution.RawOldBlockLanes.Chunk{chunk_index}");
            let mut contents = execution_generated_header("exactly 224 child-major raw old-block lane records");
            writeln!(contents, "import {IMPORT_ROOT}\n\nnamespace {namespace}\n").expect("lane import");
            contents.push_str("def values : List RawOldBlockLane := [\n");
            for (index, record) in chunk.iter().enumerate() {
                if index != 0 {
                    contents.push_str(",\n");
                }
                write!(
                    contents,
                    "  {{ schemaVersion := {SCHEMA_VERSION}, child := {}, lane := {}, padding := {}, value := {} }}",
                    record.child,
                    record.lane,
                    record.padding,
                    lean_k(record.value)
                )
                .expect("lane record");
            }
            writeln!(contents, "\n]\n\nend {namespace}").expect("lane end");
            GeneratedLeanFile {
                relative_path: format!("{GENERATED_ROOT}/Execution/RawOldBlockLanes/Chunk{chunk_index}.lean"),
                contents,
            }
        })
        .collect()
}

fn binding_shards(certificate: &ExecutionCertificate) -> Vec<GeneratedLeanFile> {
    let chunks = certificate
        .bindings
        .chunks(BINDING_CHUNK)
        .collect::<Vec<_>>();
    assert_eq!(
        chunks.iter().map(|chunk| chunk.len()).collect::<Vec<_>>(),
        vec![224, 224, 224, 224, 224, 169],
    );
    assert_eq!(
        chunks.iter().map(|chunk| chunk.len()).sum::<usize>(),
        certificate.bindings.len()
    );
    chunks.into_iter().enumerate().map(|(chunk_index, chunk)| {
        let namespace = format!("{NAMESPACE_ROOT}.Execution.GeneratedKBindings.Chunk{chunk_index}");
        let mut contents = execution_generated_header(
            "at most 224 exact semantic-slot/source/normalized K-column joins",
        );
        writeln!(contents, "import {IMPORT_ROOT}\n\nnamespace {namespace}\n").expect("binding import");
        contents.push_str("def values : List RawGeneratedKBinding := [\n");
        for (index, record) in chunk.iter().enumerate() {
            if index != 0 { contents.push_str(",\n"); }
            let (kind, index0, index1) = slot_tag(record.slot);
            write!(contents, "  {{ schemaVersion := {SCHEMA_VERSION}, slotKind := {kind}, slotIndex0 := {index0}, slotIndex1 := {index1}, builderC0 := {}, builderC1 := {}, normalizedC0 := {}, normalizedC1 := {}, value := {} }}", record.builder_columns[0], record.builder_columns[1], record.normalized_columns[0], record.normalized_columns[1], lean_k(record.value)).expect("binding record");
        }
        writeln!(contents, "\n]\n\nend {namespace}").expect("binding end");
        GeneratedLeanFile { relative_path: format!("{GENERATED_ROOT}/Execution/GeneratedKBindings/Chunk{chunk_index}.lean"), contents }
    }).collect()
}

fn rounds(certificate: &ExecutionCertificate) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Execution.Rounds");
    let mut contents =
        execution_generated_header("the exact 25 replayed five-coefficient combined-NC messages and claims");
    writeln!(contents, "import {IMPORT_ROOT}\n\nnamespace {namespace}\n").expect("round import");
    contents.push_str("def values : List RawCombinedNcRound := [\n");
    for (index, round) in certificate.rounds.iter().enumerate() {
        if index != 0 {
            contents.push_str(",\n");
        }
        write!(contents, "  {{ schemaVersion := {SCHEMA_VERSION}, index := {}, coefficients := {}, challenge := {}, claimIn := {}, claimOut := {} }}", round.index, lean_k_list(&round.coefficients), lean_k(round.challenge), lean_k(round.claim_in), lean_k(round.claim_out)).expect("round record");
    }
    writeln!(contents, "\n]\n\nend {namespace}").expect("round end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/Rounds.lean"),
        contents,
    }
}

/// Render only the compact indexed production-row contract, its exact plan,
/// and their facade.
///
/// This path deliberately avoids synthesizing the full post-PiDEC fixture.
/// The ordinary drift test still reconciles these three files against the
/// `ExecutionCertificate` extracted from that fixture before final validation.
pub(super) fn focused_raw_old_block_projection_contract(certificate: &ExecutionCertificate) -> Vec<GeneratedLeanFile> {
    let plan = certificate.raw_projection_plan;
    assert_eq!(
        plan,
        RawOldBlockProjectionPlan::new(LOGICAL_COLUMNS, CHILD_COUNT)
            .expect("fixed production raw-old-block projection profile")
    );
    assert_eq!(plan.packed_columns(), PACKED_COLUMNS);
    assert!(plan.factor_final_round());
    assert_eq!(plan.tensor_variables(), TENSOR_ROUNDS);
    assert_eq!(plan.factored_variable(), Some(TENSOR_ROUNDS));
    assert_eq!(plan.tensor_mul_count(), RAW_PROJECTION_TENSOR_MULTIPLICATIONS);
    assert_eq!(plan.final_scale_mul_count(), RAW_PROJECTION_FINAL_SCALE_MULTIPLICATIONS);
    assert_eq!(
        plan.tensor_rows() + plan.projection_product_rows() + plan.final_scale_rows(),
        RAW_PROJECTION_DERIVED_COLUMNS
    );
    assert_eq!(plan.total_rows(), RAW_PROJECTION_ROWS);
    vec![
        raw_old_block_projection_plan(certificate),
        raw_old_block_projection_row_at(plan),
        execution_root(),
    ]
}

pub(super) fn render(
    certificate: &ExecutionCertificate,
    static_audit: &R1csIvcBlockLaneNcSelectiveRowsAudit,
) -> Vec<GeneratedLeanFile> {
    certificate
        .validate(static_audit)
        .expect("exact post-PiDEC execution certificate");
    let mut files = vec![
        header(certificate),
        raw_old_block_projection_plan(certificate),
        raw_old_block_projection_row_at(certificate.raw_projection_plan),
        rounds(certificate),
    ];
    let public = public_write_shards(certificate);
    files.extend(public);
    files.push(list_root("PublicWrites", 2, "RawPublicWrite"));
    let lanes = lane_shards(certificate);
    files.extend(lanes);
    files.push(list_root("RawOldBlockLanes", 4, "RawOldBlockLane"));
    let bindings = binding_shards(certificate);
    files.extend(bindings);
    files.push(list_root("GeneratedKBindings", 6, "RawGeneratedKBinding"));
    files.push(execution_root());
    files
}

pub(super) fn assert_mutations_fail(
    certificate: &ExecutionCertificate,
    static_audit: &R1csIvcBlockLaneNcSelectiveRowsAudit,
) {
    let reject = |mutated: &ExecutionCertificate, label: &str| {
        assert!(mutated.validate(static_audit).is_err(), "{label} must fail closed");
    };
    let mut changed = certificate.clone();
    changed.lanes.swap(0, PADDED_LANES);
    reject(&changed, "child order mutation");
    let mut changed = certificate.clone();
    changed.lanes[0].value += K::ONE;
    reject(&changed, "active lane mutation");
    let mut changed = certificate.clone();
    changed.lanes[ACTIVE_LANES].value = K::ONE;
    reject(&changed, "padding mutation");
    let mut changed = certificate.clone();
    changed.old_block[0] += K::ONE;
    reject(&changed, "old-block coordinate mutation");
    let mut changed = certificate.clone();
    changed.parent_y_zcol[0] += K::ONE;
    reject(&changed, "pending parent mutation");
    let mut changed = certificate.clone();
    changed.radix += K::ONE;
    reject(&changed, "raw-child radix mutation");
    let mut changed = certificate.clone();
    changed.producer_beta += K::ONE;
    reject(&changed, "producer-beta mutation");
    let mut changed = certificate.clone();
    changed.batch_weight += K::ONE;
    reject(&changed, "batch-weight mutation");
    let mut changed = certificate.clone();
    changed.selector_values[2] = F::ZERO;
    reject(&changed, "selector mutation");
    assert_public_write_mutations_fail(certificate);
    let mut changed = certificate.clone();
    changed.rounds.swap(0, 1);
    reject(&changed, "challenge/message order mutation");
    let mut changed = certificate.clone();
    changed.block_point.swap(0, 1);
    reject(&changed, "transcript block-point order mutation");
    let mut changed = certificate.clone();
    changed.terminal_rhs += K::ONE;
    reject(&changed, "terminal mapping mutation");
    let mut changed = certificate.clone();
    changed.bindings[0].normalized_columns[0] += 1;
    reject(&changed, "generated-column mapping mutation");
}

pub(super) fn assert_public_write_mutations_fail(certificate: &ExecutionCertificate) {
    let reject = |mutated: &ExecutionCertificate, label: &str| {
        assert!(
            mutated.validate_public_write_profile().is_err(),
            "{label} must fail closed"
        );
    };
    let mut changed = certificate.clone();
    changed.logical_columns -= 1;
    reject(&changed, "public dimension mutation");
    let mut changed = certificate.clone();
    changed.branch = R1csIvcBranch::Base;
    reject(&changed, "public profile arm mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].builder_column = Some(0);
    reject(&changed, "public source mapping mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].source = R1csIvcPublicWriteSource::FixedZero;
    reject(&changed, "public source-kind mutation");
    let mut changed = certificate.clone();
    changed.public_writes.swap(1, 2);
    reject(&changed, "public write order mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].normalized_source_column = Some(2);
    reject(&changed, "public normalized-source mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].normalized_column += 1;
    reject(&changed, "public normalized-target mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].packed_row += 1;
    reject(&changed, "public packed-address mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].width = 2;
    reject(&changed, "public width mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].centered = true;
    reject(&changed, "public centeredness mutation");
    let mut changed = certificate.clone();
    changed.public_writes[1].alias_source = Some(1);
    reject(&changed, "public alias mutation");
    let mut changed = certificate.clone();
    changed.public_writes[PUBLIC_WRITES - 1].value = F::ONE;
    reject(&changed, "public fixed-zero value mutation");
}
