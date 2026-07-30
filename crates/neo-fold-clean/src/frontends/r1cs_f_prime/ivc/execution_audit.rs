//! Proof-free execution evidence for the active recursive R1CS-IVC arm.
//!
//! This module records values already present at the synthesis boundary. It
//! neither emits constraints nor treats a digest or a CE `y_zcol` sidecar as
//! authority for an incoming child's raw public assignment.
//!
//! Owns: proof-free joins from the active recursive builder, raw carried
//! witnesses, delayed combined-NC transcript, and normalized assignment.
//!
//! Does not own: semantic authority from digests or child `y_zcol` sidecars,
//! commitment binding, Lean decoding, or row-removal permission.
//!
//! Emits constraints: no; this module only records and replays live values.
//!
//! | Stable stage path | Obligation | Authority class |
//! |---|---|---|
//! | `f_prime.post_pi_dec.raw_children` | Join carried raw witness values to live builder and normalized columns | direct dataflow |
//! | `f_prime.post_pi_dec.raw_old_block` | Recompute delayed parent projection from fourteen full witness matrices | checked |
//! | `f_prime.post_pi_dec.paper_shape` | Join the 4,590 strict public-X rows to native parent/ordered-child values | checked |
//! | `f_prime.post_pi_dec.combined_nc` | Replay challenges, 25 messages, and terminal mapping | checked |

use core::ops::Range;

use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::PiCcsProofVariant;
use p3_field::PrimeCharacteristicRing;

use super::{R1csIvcPreprocessing, StateCoordinates};
use crate::engine::r1cs_circuit::{R1csBuilder, Var};
use crate::engine::transcript::Transcript;
use crate::frontends::r1cs_f_prime::ivc::relation::R1csIvcBranch;
use crate::frontends::r1cs_f_prime::lowering::normalized_source_column;
use crate::paper::construction2::{PendingProjectionState, PENDING_PROJECTION_OLD_BLOCK_LEN};
use crate::paper::digest::{
    pending_accumulator_family_digest, pi_ccs_instance_digest_parent_authority, PendingAccumulatorFamilyState,
};
use crate::paper::f_prime::r1cs::{FPrimeStepOutput, F_PRIME_PUBLIC_INPUT_LEN, F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN};
use crate::paper::nifs::NifsProof;
use crate::paper::reductions::pi_ccs_output_message::Profile as PiCcsOutputProfile;
use crate::paper::relations::{CcsClaim, CcsInstance, CeClaim, WitnessMat};

const ACTIVE_RECURSIVE_PUBLIC_WRITES: usize = F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN;
pub(super) const ACTIVE_RECURSIVE_FRESH_OUTPUTS: usize = 1;
pub(super) const OUTPUT_Y_ZCOL_PADDED_LANES: usize = 64;
pub(super) const OUTPUT_Y_ZCOL_ZERO_PADDING_LANES: usize = OUTPUT_Y_ZCOL_PADDED_LANES - D;

/// The prover-side source of every raw incoming-child assignment value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcRawAssignmentAuthority {
    /// The value was read from the actual carried `RunningInstance.witnesses` matrix.
    RunningWitnessMat,
}

#[path = "execution_audit/combined_nc.rs"]
mod combined_nc;
#[path = "execution_audit/pi_dec_paper_shape.rs"]
mod pi_dec_paper_shape;
#[path = "execution_audit/raw_old_block.rs"]
mod raw_old_block;

use combined_nc::{append_sumcheck_prolog, capture_output_y_zcol_tables, combined_nc_terminal_rhs};
use pi_dec_paper_shape::capture_and_validate as capture_pi_dec_paper_shape;
pub use pi_dec_paper_shape::{
    R1csIvcPiDecCanonicalXCoordinateAudit, R1csIvcPiDecPaperShapeExecutionAudit, R1csIvcPiDecPaperShapeProfile,
    R1csIvcPiDecPaperTraceColumnAudit, R1csIvcPiDecPaperXOwner, R1csIvcPiDecPaperXPinAudit,
    PI_DEC_PAPER_ACTIVE_X_COLUMNS, PI_DEC_PAPER_CANONICALITY_ROWS_PER_COORDINATE, PI_DEC_PAPER_CHILD_COUNT,
    PI_DEC_PAPER_EVALUATION_ARITY, PI_DEC_PAPER_PUBLIC_COORDINATES,
};
use raw_old_block::capture_and_validate_raw_old_block_execution;
pub use raw_old_block::{
    validate_raw_old_block_execution, R1csIvcRawOldBlockChildAudit, R1csIvcRawOldBlockExecutionAudit,
    R1csIvcRawOldBlockFieldDecoding, R1csIvcRawOldBlockProfile, RAW_OLD_BLOCK_ACTIVE_LANES, RAW_OLD_BLOCK_CHILD_COUNT,
    RAW_OLD_BLOCK_PADDED_LANES, RAW_OLD_BLOCK_ZERO_PADDING_LANES,
};

/// Compact identity and coordinate rule for one complete carried witness `Z`.
///
/// Logical column `j` is read from `Z[j % packed_rows, j / packed_rows]`.
/// The audit retains only the public-prefix coordinates actually written into
/// the recursive verifier, not a second copy of the potentially huge matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcFullZChildAudit {
    child: usize,
    logical_columns: usize,
    packed_rows: usize,
    packed_columns: usize,
    captured_public_coordinates: Range<usize>,
}

impl R1csIvcFullZChildAudit {
    pub fn child(&self) -> usize {
        self.child
    }

    pub fn logical_columns(&self) -> usize {
        self.logical_columns
    }

    pub fn packed_shape(&self) -> (usize, usize) {
        (self.packed_rows, self.packed_columns)
    }

    pub fn captured_public_coordinates(&self) -> Range<usize> {
        self.captured_public_coordinates.clone()
    }

    pub fn authority(&self) -> R1csIvcRawAssignmentAuthority {
        R1csIvcRawAssignmentAuthority::RunningWitnessMat
    }
}

/// One raw child-public value, joined across full `Z`, the live verifier wire,
/// and the exact normalized source-field assignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcRawChildAssignmentAudit {
    child: usize,
    logical_column: usize,
    witness_row: usize,
    witness_column: usize,
    builder_column: usize,
    normalized_source_column: usize,
    value: F,
}

impl R1csIvcRawChildAssignmentAudit {
    pub fn child(&self) -> usize {
        self.child
    }

    pub fn logical_column(&self) -> usize {
        self.logical_column
    }

    pub fn witness_coordinate(&self) -> (usize, usize) {
        (self.witness_row, self.witness_column)
    }

    pub fn builder_column(&self) -> usize {
        self.builder_column
    }

    pub fn normalized_source_column(&self) -> usize {
        self.normalized_source_column
    }

    /// Join key for the fixed-point raw-running assignment audit.
    pub fn source_column(&self) -> usize {
        self.normalized_source_column
    }

    pub fn value(&self) -> F {
        self.value
    }

    pub fn authority(&self) -> R1csIvcRawAssignmentAuthority {
        R1csIvcRawAssignmentAuthority::RunningWitnessMat
    }
}

/// One exact public-coordinate write in the committed recursive-arm witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcPublicWriteSource {
    ConstantOne,
    BuilderColumn,
    FixedZero,
}

/// One exact public-coordinate write in the committed recursive-arm witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcPublicWriteAudit {
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

impl R1csIvcPublicWriteAudit {
    pub fn logical_column(&self) -> usize {
        self.logical_column
    }

    pub fn packed_coordinate(&self) -> (usize, usize) {
        (self.packed_row, self.packed_column)
    }

    pub fn source(&self) -> R1csIvcPublicWriteSource {
        self.source
    }

    pub fn builder_column(&self) -> Option<usize> {
        self.builder_column
    }

    pub fn normalized_source_column(&self) -> Option<usize> {
        self.normalized_source_column
    }

    pub fn normalized_column(&self) -> usize {
        self.normalized_column
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn centered(&self) -> bool {
        self.centered
    }

    pub fn alias_source(&self) -> Option<usize> {
        self.alias_source
    }

    pub fn value(&self) -> F {
        self.value
    }
}

/// Semantic identity of one K-valued generated column pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R1csIvcGeneratedKSlot {
    Gamma,
    BetaLane(usize),
    BetaBlock(usize),
    ProducerBeta,
    BatchWeight,
    PendingOldBlock(usize),
    PendingParentYZcol(usize),
    OutputYZcol { source: usize, lane: usize },
    BlockPoint(usize),
    LanePoint(usize),
    ClaimedInitial,
    FinalSum,
    TerminalRhs,
    RoundCoefficient { round: usize, coefficient: usize },
    RoundChallenge(usize),
    RoundClaimIn(usize),
    RoundClaimOut(usize),
}

/// Exact source-to-normalized join for one generated K value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcGeneratedKBindingAudit {
    slot: R1csIvcGeneratedKSlot,
    builder_columns: [usize; 2],
    normalized_columns: [usize; 2],
    value: K,
}

impl R1csIvcGeneratedKBindingAudit {
    pub fn slot(&self) -> R1csIvcGeneratedKSlot {
        self.slot
    }

    pub fn builder_columns(&self) -> [usize; 2] {
        self.builder_columns
    }

    pub fn normalized_columns(&self) -> [usize; 2] {
        self.normalized_columns
    }

    pub fn value(&self) -> K {
        self.value
    }
}

/// One committed one-hot branch-selector write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcSelectorWriteAudit {
    arm: R1csIvcBranch,
    logical_column: usize,
    packed_row: usize,
    packed_column: usize,
    value: F,
}

impl R1csIvcSelectorWriteAudit {
    pub fn arm(&self) -> R1csIvcBranch {
        self.arm
    }

    pub fn logical_column(&self) -> usize {
        self.logical_column
    }

    pub fn packed_coordinate(&self) -> (usize, usize) {
        (self.packed_row, self.packed_column)
    }

    pub fn value(&self) -> F {
        self.value
    }
}

/// One quartic combined-NC SumCheck message and its verifier-derived state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcCombinedNcRoundAudit {
    index: usize,
    coefficients: Vec<K>,
    challenge: K,
    claim_in: K,
    claim_out: K,
}

impl R1csIvcCombinedNcRoundAudit {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn coefficients(&self) -> &[K] {
        &self.coefficients
    }

    pub fn challenge(&self) -> K {
        self.challenge
    }

    pub fn claim_in(&self) -> K {
        self.claim_in
    }

    pub fn claim_out(&self) -> K {
        self.claim_out
    }
}

/// The terminal equality checked after the 25 combined-NC rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1csIvcCombinedNcTerminalAudit {
    claimed_initial: K,
    final_sum: K,
    rhs: K,
}

impl R1csIvcCombinedNcTerminalAudit {
    pub fn claimed_initial(&self) -> K {
        self.claimed_initial
    }

    pub fn final_sum(&self) -> K {
        self.final_sum
    }

    pub fn rhs(&self) -> K {
        self.rhs
    }
}

/// Exact delayed block/lane values consumed by the active recursive arm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcCombinedNcExecutionAudit {
    proof_variant: PiCcsProofVariant,
    output_profile: PiCcsOutputProfile,
    fresh_output_count: usize,
    running_output_count: usize,
    gamma: K,
    output_y_zcol_active: Vec<[K; D]>,
    output_y_zcol_zero_padding: Vec<[K; OUTPUT_Y_ZCOL_ZERO_PADDING_LANES]>,
    producer_beta: K,
    batch_weight: K,
    pending_old_block: [K; PENDING_PROJECTION_OLD_BLOCK_LEN],
    pending_parent_y_zcol: [K; D],
    beta_block: Vec<K>,
    beta_lane: Vec<K>,
    block_point: Vec<K>,
    lane_point: Vec<K>,
    rounds: Vec<R1csIvcCombinedNcRoundAudit>,
    terminal: R1csIvcCombinedNcTerminalAudit,
}

impl R1csIvcCombinedNcExecutionAudit {
    pub fn proof_variant(&self) -> PiCcsProofVariant {
        self.proof_variant
    }

    pub fn output_profile(&self) -> PiCcsOutputProfile {
        self.output_profile
    }

    pub fn fresh_output_count(&self) -> usize {
        self.fresh_output_count
    }

    pub fn running_output_count(&self) -> usize {
        self.running_output_count
    }

    pub fn output_y_zcol_padded_lanes(&self) -> usize {
        OUTPUT_Y_ZCOL_PADDED_LANES
    }

    pub fn output_y_zcol_zero_padding_lanes(&self) -> usize {
        OUTPUT_Y_ZCOL_ZERO_PADDING_LANES
    }

    pub fn gamma(&self) -> K {
        self.gamma
    }

    /// PiCCS-produced output tables in protocol order: fresh, then running.
    ///
    /// These values are terminal-check evidence. They are never an authority
    /// for an incoming child's raw public assignment.
    pub fn output_y_zcol_active(&self) -> &[[K; D]] {
        &self.output_y_zcol_active
    }

    /// The ten checked-zero implementation lanes paired with each active
    /// output table, in the same fresh-then-running order.
    pub fn output_y_zcol_zero_padding(&self) -> &[[K; OUTPUT_Y_ZCOL_ZERO_PADDING_LANES]] {
        &self.output_y_zcol_zero_padding
    }

    pub fn producer_beta(&self) -> K {
        self.producer_beta
    }

    pub fn batch_weight(&self) -> K {
        self.batch_weight
    }

    pub fn pending_old_block(&self) -> &[K; PENDING_PROJECTION_OLD_BLOCK_LEN] {
        &self.pending_old_block
    }

    pub fn pending_parent_y_zcol(&self) -> &[K; D] {
        &self.pending_parent_y_zcol
    }

    pub fn beta_block(&self) -> &[K] {
        &self.beta_block
    }

    pub fn beta_lane(&self) -> &[K] {
        &self.beta_lane
    }

    pub fn block_point(&self) -> &[K] {
        &self.block_point
    }

    pub fn lane_point(&self) -> &[K] {
        &self.lane_point
    }

    pub fn rounds(&self) -> &[R1csIvcCombinedNcRoundAudit] {
        &self.rounds
    }

    pub fn terminal(&self) -> &R1csIvcCombinedNcTerminalAudit {
        &self.terminal
    }
}

/// Read-only evidence captured after active recursive synthesis and PiDEC.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csIvcPostPiDecExecutionAudit {
    branch: R1csIvcBranch,
    source_builder_rows: usize,
    source_builder_columns: usize,
    committed_rows: usize,
    committed_columns: usize,
    public_output_builder_columns: Vec<usize>,
    constant_one_source_builder_column: usize,
    constant_one_binding: R1csIvcPublicWriteAudit,
    public_writes: Vec<R1csIvcPublicWriteAudit>,
    selector_writes: Vec<R1csIvcSelectorWriteAudit>,
    full_z_children: Vec<R1csIvcFullZChildAudit>,
    raw_child_assignments: Vec<R1csIvcRawChildAssignmentAudit>,
    raw_old_block: R1csIvcRawOldBlockExecutionAudit,
    pi_dec_paper_shape: R1csIvcPiDecPaperShapeExecutionAudit,
    pi_ccs_output_count: usize,
    combined_parent_m_in: usize,
    pi_dec_child_count: usize,
    combined_nc: R1csIvcCombinedNcExecutionAudit,
    generated_k_bindings: Vec<R1csIvcGeneratedKBindingAudit>,
}

impl R1csIvcPostPiDecExecutionAudit {
    pub fn branch(&self) -> R1csIvcBranch {
        self.branch
    }

    pub fn source_builder_columns(&self) -> usize {
        self.source_builder_columns
    }

    pub fn source_builder_rows(&self) -> usize {
        self.source_builder_rows
    }

    pub fn committed_rows(&self) -> usize {
        self.committed_rows
    }

    pub fn committed_columns(&self) -> usize {
        self.committed_columns
    }

    /// Builder columns moved to the normalized public prefix, in normalized
    /// public-coordinate order. Together with `source_builder_columns`, this
    /// makes every raw-child source-resolution record independently replayable.
    pub fn public_output_builder_columns(&self) -> &[usize] {
        &self.public_output_builder_columns
    }

    pub fn constant_one_source_builder_column(&self) -> usize {
        self.constant_one_source_builder_column
    }

    /// Typed committed-public binding for the builder's distinguished `ONE`
    /// variable. The source and normalized logical column are both zero.
    pub fn constant_one_binding(&self) -> &R1csIvcPublicWriteAudit {
        &self.constant_one_binding
    }

    pub fn public_writes(&self) -> &[R1csIvcPublicWriteAudit] {
        &self.public_writes
    }

    pub fn selector_writes(&self) -> &[R1csIvcSelectorWriteAudit] {
        &self.selector_writes
    }

    pub fn full_z_children(&self) -> &[R1csIvcFullZChildAudit] {
        &self.full_z_children
    }

    pub fn raw_child_assignments(&self) -> &[R1csIvcRawChildAssignmentAudit] {
        &self.raw_child_assignments
    }

    /// Native values recomputed from the exact ordered full witness family.
    pub fn raw_old_block(&self) -> &R1csIvcRawOldBlockExecutionAudit {
        &self.raw_old_block
    }

    pub fn pi_dec_paper_shape(&self) -> &R1csIvcPiDecPaperShapeExecutionAudit {
        &self.pi_dec_paper_shape
    }

    pub fn pi_ccs_output_count(&self) -> usize {
        self.pi_ccs_output_count
    }

    pub fn combined_parent_m_in(&self) -> usize {
        self.combined_parent_m_in
    }

    pub fn pi_dec_child_count(&self) -> usize {
        self.pi_dec_child_count
    }

    pub fn combined_nc(&self) -> &R1csIvcCombinedNcExecutionAudit {
        &self.combined_nc
    }

    /// Live builder-to-normalized joins in canonical semantic-slot order.
    pub fn generated_k_bindings(&self) -> &[R1csIvcGeneratedKBindingAudit] {
        &self.generated_k_bindings
    }
}

#[derive(Clone)]
pub(super) struct RawRunningWitnessCapture {
    children: Vec<R1csIvcFullZChildAudit>,
    coordinates: Vec<RawRunningWitnessCoordinate>,
    raw_old_block: Option<R1csIvcRawOldBlockExecutionAudit>,
}

#[derive(Clone, Copy)]
struct RawRunningWitnessCoordinate {
    child: usize,
    logical_column: usize,
    witness_row: usize,
    witness_column: usize,
    value: F,
}

pub(super) fn capture_running_witnesses(
    witnesses: &[WitnessMat],
    logical_columns: usize,
    pending: Option<&PendingProjectionState>,
    radix: K,
) -> Result<RawRunningWitnessCapture, String> {
    if logical_columns < ACTIVE_RECURSIVE_PUBLIC_WRITES {
        return Err(format!(
            "full running witness width {logical_columns} is smaller than the {ACTIVE_RECURSIVE_PUBLIC_WRITES}-coordinate public carrier"
        ));
    }
    let packed_columns = logical_columns.div_ceil(D);
    let mut children = Vec::with_capacity(witnesses.len());
    let mut coordinates = Vec::with_capacity(witnesses.len() * ACTIVE_RECURSIVE_PUBLIC_WRITES);
    for (child, witness) in witnesses.iter().enumerate() {
        if witness.rows() != D || witness.cols() != packed_columns {
            return Err(format!(
                "post-PiDEC audit running.witnesses[{child}].Z has shape {}x{}, expected {D}x{packed_columns} for full relation width {logical_columns}",
                witness.rows(),
                witness.cols(),
            ));
        }
        let start = coordinates.len();
        for logical_column in 0..ACTIVE_RECURSIVE_PUBLIC_WRITES {
            let witness_row = logical_column % D;
            let witness_column = logical_column / D;
            coordinates.push(RawRunningWitnessCoordinate {
                child,
                logical_column,
                witness_row,
                witness_column,
                value: witness[(witness_row, witness_column)],
            });
        }
        children.push(R1csIvcFullZChildAudit {
            child,
            logical_columns,
            packed_rows: D,
            packed_columns,
            captured_public_coordinates: start..coordinates.len(),
        });
    }
    let raw_old_block = pending
        .map(|pending| {
            capture_and_validate_raw_old_block_execution(
                witnesses,
                logical_columns,
                pending.old_block(),
                pending.parent_y_zcol(),
                radix,
            )
        })
        .transpose()?;
    Ok(RawRunningWitnessCapture {
        children,
        coordinates,
        raw_old_block,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn capture_post_pi_dec_execution(
    prep: &R1csIvcPreprocessing,
    branch: R1csIvcBranch,
    pre: &StateCoordinates,
    chunk_digest: [F; 4],
    builder: &R1csBuilder,
    output: &FPrimeStepOutput,
    public_outputs: &[Var],
    normalized_field_assignment: &[F],
    instance: &CcsInstance,
    raw_witnesses: &RawRunningWitnessCapture,
    fresh: &[CcsClaim],
    running: &[CeClaim],
    running_parent_authority: Option<&CeClaim>,
    running_pending_projection: Option<&PendingProjectionState>,
    nifs: &NifsProof,
) -> Result<R1csIvcPostPiDecExecutionAudit, String> {
    if branch != R1csIvcBranch::Recursive {
        return Err("post-PiDEC execution audit requires the active recursive arm".into());
    }
    let public_writes = capture_public_writes(prep, builder, public_outputs, instance)?;
    let (constant_one_source_builder_column, constant_one_binding) =
        capture_constant_one_binding(builder, normalized_field_assignment, &public_writes)?;
    let selector_writes = capture_selector_writes(prep, branch, instance)?;
    let raw_child_assignments = join_raw_running_assignments(
        builder,
        output,
        public_outputs,
        normalized_field_assignment,
        raw_witnesses,
        running,
    )?;
    let raw_old_block = raw_witnesses
        .raw_old_block
        .clone()
        .ok_or_else(|| "active recursive execution audit is missing the raw old-block projection".to_string())?;
    let pi_dec_paper_shape = capture_pi_dec_paper_shape(
        builder,
        public_outputs,
        normalized_field_assignment,
        &selector_writes,
        output.pi_dec_canonical_x_receipt.as_ref(),
        nifs,
    )?;
    let combined_nc = replay_combined_nc(
        prep,
        pre,
        chunk_digest,
        fresh,
        running,
        running_parent_authority,
        running_pending_projection,
        &nifs.pi_ccs.sumcheck,
        &nifs.pi_ccs.outputs,
    )?;
    if raw_old_block.old_block() != combined_nc.pending_old_block()
        || raw_old_block.recomposed_parent_y_zcol() != combined_nc.pending_parent_y_zcol()
    {
        return Err("raw old-block witness projection is not associated with the combined-NC pending input".into());
    }
    let generated_k_bindings =
        capture_generated_k_bindings(builder, public_outputs, normalized_field_assignment, &combined_nc)?;

    Ok(R1csIvcPostPiDecExecutionAudit {
        branch,
        source_builder_rows: builder.rows(),
        source_builder_columns: builder.cols(),
        committed_rows: prep.prep.structure().n,
        committed_columns: prep.prep.structure().m,
        public_output_builder_columns: public_outputs.iter().map(|output| output.col()).collect(),
        constant_one_source_builder_column,
        constant_one_binding,
        public_writes,
        selector_writes,
        full_z_children: raw_witnesses.children.clone(),
        raw_child_assignments,
        raw_old_block,
        pi_dec_paper_shape,
        pi_ccs_output_count: nifs.pi_ccs.outputs.len(),
        combined_parent_m_in: nifs.pi_rlc.combined.m_in,
        pi_dec_child_count: nifs.pi_dec.children.len(),
        combined_nc,
        generated_k_bindings,
    })
}

fn capture_constant_one_binding(
    builder: &R1csBuilder,
    normalized_assignment: &[F],
    public_writes: &[R1csIvcPublicWriteAudit],
) -> Result<(usize, R1csIvcPublicWriteAudit), String> {
    let source_builder_column = Var::ONE.col();
    let builder_value = builder
        .witness()
        .get(source_builder_column)
        .copied()
        .ok_or_else(|| "constant-one source column escapes the live builder".to_string())?;
    let normalized_value = normalized_assignment
        .first()
        .copied()
        .ok_or_else(|| "normalized assignment omits the constant-one column".to_string())?;
    let binding = public_writes
        .first()
        .copied()
        .ok_or_else(|| "committed public assignment omits the constant-one column".to_string())?;
    if source_builder_column != 0
        || binding.logical_column != 0
        || (binding.packed_row, binding.packed_column) != (0, 0)
        || binding.source != R1csIvcPublicWriteSource::ConstantOne
        || binding.builder_column != Some(0)
        || binding.normalized_column != 0
        || builder_value != F::ONE
        || normalized_value != F::ONE
        || binding.value != F::ONE
    {
        return Err("constant-one builder, normalized, and committed-public bindings disagree".into());
    }
    Ok((source_builder_column, binding))
}

fn capture_public_writes(
    prep: &R1csIvcPreprocessing,
    builder: &R1csBuilder,
    public_outputs: &[Var],
    instance: &CcsInstance,
) -> Result<Vec<R1csIvcPublicWriteAudit>, String> {
    let layout = prep.relation().compilation_audit().layout();
    if layout.logical_public_input_len() != F_PRIME_PUBLIC_INPUT_LEN
        || layout.public_input_len() != ACTIVE_RECURSIVE_PUBLIC_WRITES
        || layout
            .public_padding_columns()
            .iter()
            .copied()
            .ne(F_PRIME_PUBLIC_INPUT_LEN..ACTIVE_RECURSIVE_PUBLIC_WRITES)
    {
        return Err("active recursive public writes disagree with the emitted public-coordinate profile".into());
    }
    if instance.claim.m_in != ACTIVE_RECURSIVE_PUBLIC_WRITES || instance.claim.x.len() != ACTIVE_RECURSIVE_PUBLIC_WRITES
    {
        return Err(format!(
            "active recursive public write count is m_in={} x.len()={}, expected {ACTIVE_RECURSIVE_PUBLIC_WRITES}",
            instance.claim.m_in,
            instance.claim.x.len(),
        ));
    }
    let mut writes = Vec::with_capacity(ACTIVE_RECURSIVE_PUBLIC_WRITES);
    for (logical_column, &value) in instance.claim.x.iter().enumerate() {
        let packed_row = logical_column % D;
        let packed_column = logical_column / D;
        let (source, builder_column, normalized_source_column, width) = if logical_column == 0 {
            (R1csIvcPublicWriteSource::ConstantOne, Some(Var::ONE.col()), None, 1)
        } else if logical_column <= public_outputs.len() {
            (
                R1csIvcPublicWriteSource::BuilderColumn,
                Some(public_outputs[logical_column - 1].col()),
                Some(logical_column),
                1,
            )
        } else {
            (R1csIvcPublicWriteSource::FixedZero, None, None, 0)
        };
        let z_value = packed_value(&instance.witness.Z, logical_column, "committed public write")?;
        let builder_matches = builder_column.is_none_or(|column| builder.witness().get(column).copied() == Some(value));
        if z_value != value || !builder_matches || (source == R1csIvcPublicWriteSource::FixedZero && value != F::ZERO) {
            return Err(format!(
                "public write {logical_column} disagrees across builder source, normalized public column, and packed instance witness"
            ));
        }
        writes.push(R1csIvcPublicWriteAudit {
            logical_column,
            packed_row,
            packed_column,
            source,
            builder_column,
            normalized_source_column,
            normalized_column: logical_column,
            width,
            centered: false,
            alias_source: None,
            value,
        });
    }
    Ok(writes)
}

fn capture_selector_writes(
    prep: &R1csIvcPreprocessing,
    branch: R1csIvcBranch,
    instance: &CcsInstance,
) -> Result<Vec<R1csIvcSelectorWriteAudit>, String> {
    let selector_columns = prep
        .relation()
        .compilation_audit()
        .layout()
        .selector_columns();
    if selector_columns.len() != 3 {
        return Err(format!(
            "active recursive relation has {} selector columns, expected 3",
            selector_columns.len()
        ));
    }
    let arms = [
        R1csIvcBranch::Base,
        R1csIvcBranch::BootstrapRecursive,
        R1csIvcBranch::Recursive,
    ];
    let mut writes = Vec::with_capacity(arms.len());
    for (index, (&logical_column, arm)) in selector_columns.iter().zip(arms).enumerate() {
        let value = packed_value(&instance.witness.Z, logical_column, "branch selector")?;
        let expected = if index == branch.index() { F::ONE } else { F::ZERO };
        if value != expected {
            return Err(format!(
                "selector for {arm:?} at committed column {logical_column} is {value:?}, expected {expected:?}"
            ));
        }
        writes.push(R1csIvcSelectorWriteAudit {
            arm,
            logical_column,
            packed_row: logical_column % D,
            packed_column: logical_column / D,
            value,
        });
    }
    Ok(writes)
}

fn join_raw_running_assignments(
    builder: &R1csBuilder,
    output: &FPrimeStepOutput,
    public_outputs: &[Var],
    normalized_assignment: &[F],
    raw_witnesses: &RawRunningWitnessCapture,
    running: &[CeClaim],
) -> Result<Vec<R1csIvcRawChildAssignmentAudit>, String> {
    let wires = output
        .nifs_running
        .as_deref()
        .ok_or_else(|| "active recursive output omitted running PiCCS wires".to_string())?;
    if wires.len() != running.len() || raw_witnesses.children.len() != running.len() {
        return Err(format!(
            "raw child count drift: wires={} claims={} witnesses={}",
            wires.len(),
            running.len(),
            raw_witnesses.children.len(),
        ));
    }
    if normalized_assignment.len() != builder.cols() {
        return Err(format!(
            "normalized assignment has {} columns, live builder has {}",
            normalized_assignment.len(),
            builder.cols(),
        ));
    }

    let mut joined = Vec::with_capacity(raw_witnesses.coordinates.len());
    for raw in &raw_witnesses.coordinates {
        let claim = &running[raw.child];
        let child_wires = &wires[raw.child];
        if claim.m_in != ACTIVE_RECURSIVE_PUBLIC_WRITES
            || claim.X.rows() != D
            || claim.X.cols() < ACTIVE_RECURSIVE_PUBLIC_WRITES.div_ceil(D)
            || child_wires.m_in != ACTIVE_RECURSIVE_PUBLIC_WRITES
            || child_wires.x_rows != D
            || child_wires.x_cols == 0
            || child_wires.m_in > child_wires.x_rows * child_wires.x_cols
            || child_wires.x.len() != child_wires.x_rows * child_wires.x_cols
        {
            return Err(format!("running child {} has malformed raw public geometry", raw.child));
        }
        let claim_value = claim.X[(raw.witness_row, raw.witness_column)];
        let wire_index = raw.witness_row * child_wires.x_cols + raw.witness_column;
        let builder_column = child_wires.x[wire_index].col();
        let normalized_column =
            normalized_target_column(builder.cols(), public_outputs, builder_column).ok_or_else(|| {
                format!(
                    "running child {} column {} escapes the builder",
                    raw.child, builder_column
                )
            })?;
        if normalized_source_column(builder.cols(), public_outputs, normalized_column) != Some(builder_column) {
            return Err(format!(
                "running child {} logical column {} failed normalization round trip",
                raw.child, raw.logical_column
            ));
        }
        let builder_value = builder.witness()[builder_column];
        let normalized_value = normalized_assignment[normalized_column];
        if claim_value != raw.value || builder_value != raw.value || normalized_value != raw.value {
            return Err(format!(
                "running child {} logical column {} is not sourced from running.witnesses[{}].Z[{}, {}]",
                raw.child, raw.logical_column, raw.child, raw.witness_row, raw.witness_column,
            ));
        }
        joined.push(R1csIvcRawChildAssignmentAudit {
            child: raw.child,
            logical_column: raw.logical_column,
            witness_row: raw.witness_row,
            witness_column: raw.witness_column,
            builder_column,
            normalized_source_column: normalized_column,
            value: raw.value,
        });
    }
    Ok(joined)
}

fn capture_generated_k_bindings(
    builder: &R1csBuilder,
    public_outputs: &[Var],
    normalized_assignment: &[F],
    semantic: &R1csIvcCombinedNcExecutionAudit,
) -> Result<Vec<R1csIvcGeneratedKBindingAudit>, String> {
    const BLOCKS: usize = 19;
    const LANES: usize = 6;
    const ROUNDS: usize = BLOCKS + LANES;
    const COEFFICIENTS: usize = 5;
    const OUTPUTS: usize = 15;

    if normalized_assignment.len() != builder.cols() {
        return Err(format!(
            "generated-column join assignment width {} differs from builder width {}",
            normalized_assignment.len(),
            builder.cols()
        ));
    }
    let mut normalized_public = public_outputs
        .iter()
        .map(|output| output.col())
        .collect::<Vec<_>>();
    normalized_public.sort_unstable();
    if normalized_public.contains(&Var::ONE.col())
        || normalized_public
            .iter()
            .any(|&column| column >= builder.cols())
        || normalized_public.windows(2).any(|pair| pair[0] == pair[1])
    {
        return Err("generated-column join received an invalid public-prefix relocation".into());
    }

    let [boundary] = builder.block_lane_nc_boundary_audits() else {
        return Err(format!(
            "active builder has {} block/lane NC boundaries, expected one",
            builder.block_lane_nc_boundary_audits().len()
        ));
    };
    if boundary.round_audit_indices.start > boundary.round_audit_indices.end
        || boundary.round_audit_indices.end > builder.sumcheck_round_audits().len()
    {
        return Err("block/lane NC round-audit range escapes the live builder".into());
    }
    let rounds = &builder.sumcheck_round_audits()[boundary.round_audit_indices.clone()];
    let pending_old_block = boundary
        .pending_old_block_cols
        .as_deref()
        .ok_or_else(|| "block/lane NC boundary omits pending old-block columns".to_string())?;
    let pending_parent_y_zcol = boundary
        .pending_parent_y_zcol_cols
        .as_deref()
        .ok_or_else(|| "block/lane NC boundary omits pending parent-y_zcol columns".to_string())?;
    if rounds.len() != ROUNDS
        || boundary.beta_lane_cols.len() != LANES
        || boundary.beta_block_cols.len() != BLOCKS
        || pending_old_block.len() != BLOCKS
        || pending_parent_y_zcol.len() != D
        || boundary.output_y_zcol_cols.len() != OUTPUTS
        || boundary
            .output_y_zcol_cols
            .iter()
            .any(|output| output.len() != OUTPUT_Y_ZCOL_PADDED_LANES)
        || boundary.block_point_cols.len() != BLOCKS
        || boundary.lane_point_cols.len() != LANES
        || semantic.beta_lane.len() != LANES
        || semantic.beta_block.len() != BLOCKS
        || semantic.block_point.len() != BLOCKS
        || semantic.lane_point.len() != LANES
        || semantic.rounds.len() != ROUNDS
        || semantic.output_y_zcol_active.len() != OUTPUTS
        || semantic.output_y_zcol_zero_padding.len() != OUTPUTS
        || rounds
            .iter()
            .any(|round| round.coefficient_cols.len() != COEFFICIENTS)
    {
        return Err("block/lane NC generated-column shape drift".into());
    }

    let valid_rows = |range: &Range<usize>| range.start < range.end && range.end <= builder.rows();
    if !valid_rows(&boundary.claimed_initial_rows)
        || !valid_rows(&boundary.terminal_identity_rows)
        || !valid_rows(&boundary.terminal_final_equality_rows)
        || boundary.terminal_final_equality_rows.len() != 2
        || boundary.claimed_initial_rows.end > rounds[0].row_start
        || rounds[ROUNDS - 1].row_end > boundary.terminal_identity_rows.start
        || boundary.terminal_identity_rows.end > boundary.terminal_final_equality_rows.start
        || rounds.iter().any(|round| {
            round.row_start >= round.row_end
                || round.row_end > builder.rows()
                || round.first_allocated_column > builder.cols()
                || round
                    .allocated_cols
                    .iter()
                    .any(|&column| column >= builder.cols())
                || round
                    .allocated_cols
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
        })
        || rounds
            .windows(2)
            .any(|pair| pair[0].row_end > pair[1].row_start)
    {
        return Err("block/lane NC generated-column row or allocation range drift".into());
    }
    if rounds[0].claim_in_cols != boundary.claimed_initial_cols
        || rounds[ROUNDS - 1].claim_out_cols != boundary.final_sum_cols
        || rounds
            .windows(2)
            .any(|pair| pair[0].claim_out_cols != pair[1].claim_in_cols)
        || rounds.iter().enumerate().any(|(index, round)| {
            let expected = if index < BLOCKS {
                boundary.block_point_cols[index]
            } else {
                boundary.lane_point_cols[index - BLOCKS]
            };
            round.challenge_cols != expected
        })
    {
        return Err("block/lane NC generated-column claimed chain or challenge order drift".into());
    }

    let mut bindings = Vec::new();
    let mut push = |slot, columns, value| -> Result<(), String> {
        bindings.push(capture_generated_k_binding(
            builder,
            public_outputs,
            normalized_assignment,
            slot,
            columns,
            value,
        )?);
        Ok(())
    };
    push(R1csIvcGeneratedKSlot::Gamma, boundary.gamma_cols, semantic.gamma)?;
    for (index, (&columns, &value)) in boundary
        .beta_lane_cols
        .iter()
        .zip(&semantic.beta_lane)
        .enumerate()
    {
        push(R1csIvcGeneratedKSlot::BetaLane(index), columns, value)?;
    }
    for (index, (&columns, &value)) in boundary
        .beta_block_cols
        .iter()
        .zip(&semantic.beta_block)
        .enumerate()
    {
        push(R1csIvcGeneratedKSlot::BetaBlock(index), columns, value)?;
    }
    push(
        R1csIvcGeneratedKSlot::ProducerBeta,
        boundary.producer_beta_cols,
        semantic.producer_beta,
    )?;
    push(
        R1csIvcGeneratedKSlot::BatchWeight,
        boundary.batch_weight_cols,
        semantic.batch_weight,
    )?;
    for (index, (&columns, &value)) in pending_old_block
        .iter()
        .zip(&semantic.pending_old_block)
        .enumerate()
    {
        push(R1csIvcGeneratedKSlot::PendingOldBlock(index), columns, value)?;
    }
    for (index, (&columns, &value)) in pending_parent_y_zcol
        .iter()
        .zip(&semantic.pending_parent_y_zcol)
        .enumerate()
    {
        push(R1csIvcGeneratedKSlot::PendingParentYZcol(index), columns, value)?;
    }
    for (source, columns) in boundary.output_y_zcol_cols.iter().enumerate() {
        for (lane, &column_pair) in columns.iter().enumerate() {
            let value = if lane < D {
                semantic.output_y_zcol_active[source][lane]
            } else {
                semantic.output_y_zcol_zero_padding[source][lane - D]
            };
            push(R1csIvcGeneratedKSlot::OutputYZcol { source, lane }, column_pair, value)?;
        }
    }
    for (index, (&columns, &value)) in boundary
        .block_point_cols
        .iter()
        .zip(&semantic.block_point)
        .enumerate()
    {
        push(R1csIvcGeneratedKSlot::BlockPoint(index), columns, value)?;
    }
    for (index, (&columns, &value)) in boundary
        .lane_point_cols
        .iter()
        .zip(&semantic.lane_point)
        .enumerate()
    {
        push(R1csIvcGeneratedKSlot::LanePoint(index), columns, value)?;
    }
    push(
        R1csIvcGeneratedKSlot::ClaimedInitial,
        boundary.claimed_initial_cols,
        semantic.terminal.claimed_initial,
    )?;
    push(
        R1csIvcGeneratedKSlot::FinalSum,
        boundary.final_sum_cols,
        semantic.terminal.final_sum,
    )?;
    push(
        R1csIvcGeneratedKSlot::TerminalRhs,
        boundary.terminal_rhs_cols,
        semantic.terminal.rhs,
    )?;
    for (round_index, (columns, values)) in rounds.iter().zip(&semantic.rounds).enumerate() {
        for (coefficient, (&column_pair, &value)) in columns
            .coefficient_cols
            .iter()
            .zip(&values.coefficients)
            .enumerate()
        {
            push(
                R1csIvcGeneratedKSlot::RoundCoefficient {
                    round: round_index,
                    coefficient,
                },
                column_pair,
                value,
            )?;
        }
        push(
            R1csIvcGeneratedKSlot::RoundChallenge(round_index),
            columns.challenge_cols,
            values.challenge,
        )?;
        push(
            R1csIvcGeneratedKSlot::RoundClaimIn(round_index),
            columns.claim_in_cols,
            values.claim_in,
        )?;
        push(
            R1csIvcGeneratedKSlot::RoundClaimOut(round_index),
            columns.claim_out_cols,
            values.claim_out,
        )?;
    }
    Ok(bindings)
}

fn capture_generated_k_binding(
    builder: &R1csBuilder,
    public_outputs: &[Var],
    normalized_assignment: &[F],
    slot: R1csIvcGeneratedKSlot,
    builder_columns: [usize; 2],
    expected: K,
) -> Result<R1csIvcGeneratedKBindingAudit, String> {
    if builder_columns[0] == builder_columns[1] {
        return Err(format!("generated K slot {slot:?} aliases its two builder limbs"));
    }
    let mut normalized_columns = [0usize; 2];
    let mut builder_coefficients = [F::ZERO; 2];
    let mut normalized_coefficients = [F::ZERO; 2];
    for limb in 0..2 {
        let source = builder_columns[limb];
        let normalized = normalized_target_column(builder.cols(), public_outputs, source)
            .ok_or_else(|| format!("generated K slot {slot:?} builder limb {limb} is out of range"))?;
        if normalized_source_column(builder.cols(), public_outputs, normalized) != Some(source) {
            return Err(format!(
                "generated K slot {slot:?} limb {limb} fails normalization round trip"
            ));
        }
        normalized_columns[limb] = normalized;
        builder_coefficients[limb] = builder.witness()[source];
        normalized_coefficients[limb] = normalized_assignment[normalized];
    }
    if normalized_columns[0] == normalized_columns[1]
        || builder_coefficients != normalized_coefficients
        || K::from_coeffs(builder_coefficients) != expected
        || K::from_coeffs(normalized_coefficients) != expected
    {
        return Err(format!(
            "generated K slot {slot:?} does not join builder, normalized assignment, and semantic replay"
        ));
    }
    Ok(R1csIvcGeneratedKBindingAudit {
        slot,
        builder_columns,
        normalized_columns,
        value: expected,
    })
}

#[allow(clippy::too_many_arguments)]
fn replay_combined_nc(
    prep: &R1csIvcPreprocessing,
    pre: &StateCoordinates,
    chunk_digest: [F; 4],
    fresh: &[CcsClaim],
    running: &[CeClaim],
    running_parent_authority: Option<&CeClaim>,
    pending: Option<&PendingProjectionState>,
    proof: &neo_reductions::optimized_engine::PiCcsProof,
    outputs: &[CeClaim],
) -> Result<R1csIvcCombinedNcExecutionAudit, String> {
    use neo_reductions::engines::utils::{
        self, PiCcsTranscriptVariant, PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG, PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
        PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
    };
    use neo_reductions::optimized_engine::oracle::{
        BLOCK_LANE_NC_BLOCK_VARIABLES, BLOCK_LANE_NC_LANE_VARIABLES, BLOCK_LANE_NC_ROUND_COEFFICIENTS,
    };
    use neo_reductions::optimized_engine::{
        claimed_initial_sum_from_inputs_with_k_mcs, delayed_beta_power_selector, eq_points,
    };
    use neo_reductions::sumcheck::{
        poly_eval_k, verify_sumcheck_rounds_poseidon_v3, SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
    };

    if proof.variant != PiCcsProofVariant::BlockLaneNcDelayedV1 {
        return Err(format!(
            "active recursive audit requires BlockLaneNcDelayedV1, got {:?}",
            proof.variant
        ));
    }
    let pending = pending.ok_or_else(|| "active recursive combined-NC audit is missing pending state".to_string())?;
    let parent = running_parent_authority
        .ok_or_else(|| "active recursive combined-NC audit is missing the PiRLC parent cache".to_string())?;
    if running.is_empty() {
        return Err("active recursive combined-NC audit received no running children".into());
    }
    if prep.prep.nebula().is_some() {
        return Err("generic R1CS-IVC combined-NC audit does not accept a Nebula transcript suffix".into());
    }
    let output_profile = PiCcsOutputProfile::active_f_prime();
    let (output_y_zcol_active, output_y_zcol_zero_padding) =
        capture_output_y_zcol_tables(outputs, fresh.len(), running.len(), output_profile)?;

    let mut transcript = Transcript::with_label(crate::paper::f_prime::native::F_PRIME_STEP_TRANSCRIPT_LABEL);
    transcript.append_fields(
        b"f_prime/vk_fs",
        &crate::paper::digest::digest32_as_fields(prep.prep.vk.digest()),
    );
    transcript.append_fields(b"f_prime/pi_ccs_header", &prep.prep.vk.pi_ccs_header_bundle());
    transcript.append_fields(b"f_prime/chunk_count_in", &[F::from_u64(pre.chunk_count)]);
    transcript.append_fields(b"f_prime/step_count_in", &[F::from_u64(pre.step_count)]);
    transcript.append_fields(b"f_prime/z_0", &pre.z_0);
    transcript.append_fields(b"f_prime/z_i_in", &pre.z_i);
    transcript.append_fields(b"f_prime/pc", &[F::from_u64(pre.pc)]);
    transcript.append_fields(b"f_prime/semantic_state_in", &pre.semantic_state_digest);
    transcript.append_fields(b"f_prime/acc_digest_in", &pre.acc_digest);
    transcript.append_fields(b"f_prime/public_trace_in", &pre.public_trace);
    transcript.append_fields(b"f_prime/chunk_digest", &chunk_digest);

    let structure = prep.prep.structure();
    let dims = utils::build_dims_and_policy(prep.prep.params.inner(), structure).map_err(|error| error.to_string())?;
    if dims.ell_d != BLOCK_LANE_NC_LANE_VARIABLES {
        return Err(format!(
            "combined-NC lane dimension is {}, expected {BLOCK_LANE_NC_LANE_VARIABLES}",
            dims.ell_d
        ));
    }
    let instance_digest = pi_ccs_instance_digest_parent_authority(fresh, running.len(), Some(parent));
    utils::bind_header_and_instance_digest_with_digest_for_variant(
        transcript.inner_mut(),
        prep.prep.params.inner(),
        structure,
        dims,
        prep.prep.optimized_cache().mat_digest(),
        &instance_digest,
        PiCcsTranscriptVariant::BlockLaneNcDelayedV1,
    )
    .map_err(|error| error.to_string())?;
    let handle = pending_accumulator_family_digest(
        running,
        running[0].c.kappa,
        Some(PendingAccumulatorFamilyState {
            old_block: pending.old_block(),
            parent_y_zcol: pending.parent_y_zcol(),
        }),
    )
    .map_err(|error| error.to_string())?;
    if handle != pre.acc_digest {
        return Err("running pending-family handle disagrees with the active F-prime state".into());
    }
    utils::bind_me_inputs_accumulator_handle(transcript.inner_mut(), running.len(), &handle)
        .map_err(|error| error.to_string())?;

    let mut public_challenges =
        utils::sample_challenges(transcript.inner_mut(), dims.ell_d, dims.ell).map_err(|error| error.to_string())?;
    let beta_block = utils::sample_beta_block(transcript.inner_mut(), BLOCK_LANE_NC_BLOCK_VARIABLES)
        .map_err(|error| error.to_string())?;
    let (producer_beta, batch_weight) =
        utils::sample_delayed_projection_challenges(transcript.inner_mut()).map_err(|error| error.to_string())?;
    public_challenges.beta_m = beta_block.clone();

    let claimed_fe = claimed_initial_sum_from_inputs_with_k_mcs(structure, &public_challenges, fresh.len(), running);
    append_sumcheck_prolog(
        transcript.inner_mut(),
        PI_CCS_SUMCHECK_FE_RAW_DOMAIN_TAG,
        PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
        SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
        claimed_fe,
    );
    let (_, _, fe_ok) =
        verify_sumcheck_rounds_poseidon_v3(transcript.inner_mut(), dims.d_sc, claimed_fe, &proof.sumcheck_rounds);
    if !fe_ok {
        return Err("generated PiCCS FE SumCheck messages fail transcript replay".into());
    }

    let claimed_nc = pending
        .parent_y_zcol()
        .iter()
        .rev()
        .fold(K::ZERO, |value, coefficient| value * producer_beta + *coefficient)
        * batch_weight;
    append_sumcheck_prolog(
        transcript.inner_mut(),
        PI_CCS_SUMCHECK_NC_RAW_DOMAIN_TAG,
        PI_CCS_SUMCHECK_INITIAL_RAW_TAG,
        SUMCHECK_TRANSCRIPT_V3_RAW_DOMAIN_TAG,
        claimed_nc,
    );
    let nc_degree = BLOCK_LANE_NC_ROUND_COEFFICIENTS - 1;
    let (nc_challenges, nc_final, nc_ok) =
        verify_sumcheck_rounds_poseidon_v3(transcript.inner_mut(), nc_degree, claimed_nc, &proof.sumcheck_rounds_nc);
    let expected_rounds = BLOCK_LANE_NC_BLOCK_VARIABLES + BLOCK_LANE_NC_LANE_VARIABLES;
    if !nc_ok
        || proof.sumcheck_rounds_nc.len() != expected_rounds
        || nc_challenges.len() != expected_rounds
        || proof
            .sumcheck_rounds_nc
            .iter()
            .any(|round| round.len() != BLOCK_LANE_NC_ROUND_COEFFICIENTS)
    {
        return Err("generated 25-round combined-NC transcript disagrees with replay".into());
    }
    let (block_point, lane_point) = nc_challenges.split_at(BLOCK_LANE_NC_BLOCK_VARIABLES);
    let rhs = combined_nc_terminal_rhs(
        outputs,
        fresh.len(),
        public_challenges.gamma,
        &public_challenges.beta_a,
        &beta_block,
        producer_beta,
        batch_weight,
        pending.old_block(),
        block_point,
        lane_point,
        eq_points,
        delayed_beta_power_selector,
    )?;
    if nc_final != rhs {
        return Err("generated combined-NC final sum disagrees with the terminal identity".into());
    }

    let mut claim = claimed_nc;
    let mut rounds = Vec::with_capacity(expected_rounds);
    for (index, (coefficients, &challenge)) in proof
        .sumcheck_rounds_nc
        .iter()
        .zip(&nc_challenges)
        .enumerate()
    {
        let claim_out = poly_eval_k(coefficients, challenge);
        rounds.push(R1csIvcCombinedNcRoundAudit {
            index,
            coefficients: coefficients.clone(),
            challenge,
            claim_in: claim,
            claim_out,
        });
        claim = claim_out;
    }
    if claim != nc_final {
        return Err("combined-NC round audit did not terminate at the replayed final sum".into());
    }

    Ok(R1csIvcCombinedNcExecutionAudit {
        proof_variant: proof.variant,
        output_profile,
        fresh_output_count: fresh.len(),
        running_output_count: running.len(),
        gamma: public_challenges.gamma,
        output_y_zcol_active,
        output_y_zcol_zero_padding,
        producer_beta,
        batch_weight,
        pending_old_block: *pending.old_block(),
        pending_parent_y_zcol: *pending.parent_y_zcol(),
        beta_block,
        beta_lane: public_challenges.beta_a,
        block_point: block_point.to_vec(),
        lane_point: lane_point.to_vec(),
        rounds,
        terminal: R1csIvcCombinedNcTerminalAudit {
            claimed_initial: claimed_nc,
            final_sum: nc_final,
            rhs,
        },
    })
}

fn packed_value(matrix: &WitnessMat, logical_column: usize, label: &str) -> Result<F, String> {
    let row = logical_column % D;
    let column = logical_column / D;
    if matrix.rows() != D || column >= matrix.cols() {
        return Err(format!(
            "{label} logical column {logical_column} escapes packed matrix {}x{}",
            matrix.rows(),
            matrix.cols(),
        ));
    }
    Ok(matrix[(row, column)])
}

fn normalized_target_column(source_columns: usize, public_outputs: &[Var], source: usize) -> Option<usize> {
    if source >= source_columns {
        return None;
    }
    if source == Var::ONE.col() {
        return Some(0);
    }
    if let Some(public_index) = public_outputs
        .iter()
        .position(|output| output.col() == source)
    {
        return Some(public_index + 1);
    }
    let public_before = public_outputs
        .iter()
        .filter(|output| output.col() < source)
        .count();
    Some(1 + public_outputs.len() + (source - 1 - public_before))
}
