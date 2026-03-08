use crate::pi_ccs::RotRho;
use crate::PiCcsProof;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{matrix::Mat, CeClaim};
use neo_math::{F, K};
use neo_memory::{output_check::OutputBindingProof, RiscvGuestMemoryLayout, RiscvProofProfileConfig};

pub type TwistProofK = neo_memory::twist::TwistProof<K>;
pub type ShoutProofK = neo_memory::shout::ShoutProof<K>;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CpuTimeSumcheckProof {
    pub claimed_sum: K,
    pub round_polys: Vec<Vec<K>>,
    pub r_time: Vec<K>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ShiftTimeSumcheckProof {
    pub claimed_sum: K,
    pub round_polys: Vec<Vec<K>>,
    pub r_time: Vec<K>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TimeOpeningSource {
    /// Invalid placeholder; prover must never emit this in canonical proofs.
    #[default]
    Unknown,
    /// Opening value comes from a verified committed-column opening proof.
    CommittedOpening,
    /// Opening value comes from a verified virtual-to-committed reduction chain.
    VirtualReducedOpening,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TimePointOpening {
    pub point: Vec<K>,
    pub col_ids: Vec<usize>,
    pub evals: Vec<K>,
    pub source: TimeOpeningSource,
}

/// Proof that a batch of named openings at one point is bound to the committed time columns.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TimeOpeningProof {
    pub point: Vec<K>,
    pub col_ids: Vec<usize>,
    pub evals: Vec<K>,
    /// Per-column vector-partial evaluations (length D each) at `point`.
    ///
    /// `digit_evals[i][rho]` corresponds to the rho-th digit-row evaluation for
    /// `col_ids[i]`. Scalar `evals[i]` must equal base-`b` recomposition of this row.
    pub digit_evals: Vec<Vec<K>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum OpeningDomain {
    #[default]
    Cpu,
    Mem,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpeningClaimEntry {
    pub point: Vec<K>,
    pub col_ids: Vec<usize>,
    pub source: TimeOpeningSource,
    pub domain: OpeningDomain,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpeningClaimManifest {
    pub entries: Vec<OpeningClaimEntry>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpeningReductionGroup {
    pub point: Vec<K>,
    pub domain: OpeningDomain,
    pub claim_indices: Vec<usize>,
    /// Canonical digest of `(domain, point, claim_indices)` under Stage-8 v1 encoding.
    pub group_digest: [u8; 32],
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpeningReductionProof {
    pub groups: Vec<OpeningReductionGroup>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpeningUnificationProof {
    /// Claimed total sum over the deterministic reduction-group table.
    pub claimed_sum: K,
    /// Sumcheck rounds for the group-selector unification reduction.
    pub round_polys: Vec<Vec<K>>,
    /// Transcript-derived unified selector point.
    pub r_unify: Vec<K>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JointOpeningGroupProof {
    pub point: Vec<K>,
    pub domain: OpeningDomain,
    pub claim_indices: Vec<usize>,
    pub group_digest: [u8; 32],
    /// Vector-partial joint claim (ME-native) at `point`.
    pub joint_claim_digits: Vec<K>,
    /// Scalar recomposition of `joint_claim_digits` under base `b`.
    pub joint_claim: K,
    pub joint_commitment: Cmt,
    /// Optional Π_CCS proof that this joint commitment opens at `point` to
    /// `joint_claim_digits` / `joint_claim`.
    ///
    /// Present for per-group opening claims, absent for transcript-mixed
    /// synthetic aggregates such as `unified_fold`.
    pub opening_ccs_proof: Option<crate::PiCcsProof>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum JointClaimKind {
    #[default]
    VectorPartial,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct JointOpeningLaneProof {
    pub claim_kind: JointClaimKind,
    pub groups: Vec<JointOpeningGroupProof>,
    /// Optional unified Stage-8 fold claim derived from `groups` under transcript-bound mixers.
    pub unified_fold: Option<JointOpeningGroupProof>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FoldingLanes {
    pub main_children: usize,
    pub val_children: usize,
    pub wb_children: usize,
    pub wp_children: usize,
    pub stage8_children: usize,
}

/// Route A Shout address pre-time proof metadata, grouped by `ell_addr`.
///
/// Shout addr-pre is an address-domain sumcheck, and the number of rounds equals
/// `ell_addr = d * ell` (the number of address-bit columns per lane under bit-addressing).
///
/// For performance, we batch multiple Shout lanes together using shared challenges
/// (batched sumcheck). The batched sumcheck requires *all claims in the batch to have the same
/// number of rounds*, so we group lanes by `ell_addr` and run one batched sumcheck per group.
///
/// Within each group, when a Shout lane is provably inactive for a step (no lookups), we can
/// skip its address-domain sumcheck entirely. We still bind all `claimed_sums` to the transcript,
/// but we include sumcheck rounds only for the active subset.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShoutAddrPreProof<KK> {
    /// Claimed sums per Shout lane.
    ///
    /// Lanes are flattened in `(lut_idx, lane_idx)` order, where `lut_idx` is the
    /// index in `step.lut_instances`, and `lane_idx` ranges over `inst.lanes.max(1)`.
    pub claimed_sums: Vec<KK>,
    /// Per-`ell_addr` batched sumcheck proofs.
    ///
    /// Groups must be sorted by `ell_addr` and contain at most one entry per `ell_addr`.
    pub groups: Vec<ShoutAddrPreGroupProof<KK>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShoutAddrPreGroupProof<KK> {
    /// Address-bit width (sumcheck round count) for this group.
    pub ell_addr: u32,
    /// Indices of active lanes (into `claimed_sums`) that include address sumcheck rounds.
    ///
    /// This list must be strictly increasing.
    pub active_lanes: Vec<u32>,
    /// Sumcheck rounds for active lanes only, in `active_lanes` order.
    ///
    /// `round_polys[active_idx][round] = coeffs`, and each inner `round` vector has length `ell_addr`.
    pub round_polys: Vec<Vec<Vec<KK>>>,
    /// Shared terminal address point for this group (length = `ell_addr`).
    pub r_addr: Vec<KK>,
}

impl<KK> Default for ShoutAddrPreProof<KK> {
    fn default() -> Self {
        Self {
            claimed_sums: Vec::new(),
            groups: Vec::new(),
        }
    }
}

/// One fold step’s artifacts (Π_CCS → Π_RLC → Π_DEC).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TimeFoldStep {
    /// Π_CCS outputs (k ME(b,L) instances)
    pub ccs_out: Vec<CeClaim<Cmt, F, K>>,
    /// Π_CCS proof (engine-agnostic re-export)
    pub ccs_proof: PiCcsProof,
    /// RLC mixing matrices ρ_i ∈ S ⊆ F^{D×D}
    pub rlc_rhos: Vec<RotRho>,
    /// The combined parent after RLC: ME(B,L) with B=b^k
    pub rlc_parent: CeClaim<Cmt, F, K>,
    /// DEC children: k ME(b,L) after decomposition of the parent
    pub dec_children: Vec<CeClaim<Cmt, F, K>>,
    /// Time-domain CPU outer sumcheck metadata (new path).
    pub cpu_sumcheck: CpuTimeSumcheckProof,
    /// Time-domain shift/linkage sumcheck metadata (new path).
    pub shift_sumcheck: ShiftTimeSumcheckProof,
    /// Ajtai commitments to canonical per-step CPU time columns (`cols[col][j]` over time).
    pub time_cpu_commitments: Vec<Cmt>,
    /// Ajtai commitments to canonical per-step MEM/bus time columns (`cols[col][j]` over time).
    pub time_mem_commitments: Vec<Cmt>,
    /// Canonical time-domain length for the proof-carried time columns.
    pub time_t: usize,
    /// Declared active-row count for this shard step (must satisfy 0 <= len <= time_t).
    pub time_declared_len: usize,
    /// Logical column ids in the same order as `time_cpu_commitments || time_mem_commitments`.
    pub time_col_ids: Vec<usize>,
    /// Memory time-proof labels for this step (new path).
    pub memory_time_proofs: Vec<Vec<u8>>,
    /// Named column openings grouped by evaluation point (new path).
    pub openings: Vec<TimePointOpening>,
    /// Commitment-bound opening proofs for named time openings.
    pub opening_proofs: Vec<TimeOpeningProof>,
    /// Canonical claim manifest for time openings (strictly ordered, transcript-bound).
    pub opening_manifest: OpeningClaimManifest,
    /// Deterministic grouping of opening claims for Stage-8 reduction.
    pub opening_reduction: OpeningReductionProof,
    /// Sumcheck proof that binds point/domain reduction groups into one transcript-derived selector point.
    pub opening_unification: OpeningUnificationProof,
    /// Stage-8 Neo-native joint opening lane proof.
    pub joint_opening_lane: JointOpeningLaneProof,
    /// Folding-lane summary for this step (new path).
    pub folding_lanes: FoldingLanes,
}

pub type FoldStep = TimeFoldStep;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize, FF: serde::Serialize, KK: serde::Serialize",
    deserialize = "C: serde::de::DeserializeOwned, FF: serde::de::DeserializeOwned, KK: serde::de::DeserializeOwned + Default"
))]
#[must_use]
pub struct ShardObligations<C, FF, KK> {
    pub main: Vec<CeClaim<C, FF, KK>>,
    pub val: Vec<CeClaim<C, FF, KK>>,
}

impl<C, FF, KK> ShardObligations<C, FF, KK> {
    pub fn all_len(&self) -> usize {
        self.main.len() + self.val.len()
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &CeClaim<C, FF, KK>> {
        self.main.iter().chain(self.val.iter())
    }

    pub fn require_all_finalized(
        &self,
        did_finalize_main: bool,
        did_finalize_val: bool,
    ) -> Result<(), crate::PiCcsError> {
        if !self.main.is_empty() && !did_finalize_main {
            return Err(crate::PiCcsError::ProtocolError(
                "finalizer did not process main obligations".into(),
            ));
        }
        if !self.val.is_empty() && !did_finalize_val {
            return Err(crate::PiCcsError::ProtocolError(
                "finalizer did not process val-lane obligations".into(),
            ));
        }
        Ok(())
    }

    pub fn split(self) -> (Vec<CeClaim<C, FF, KK>>, Vec<CeClaim<C, FF, KK>>) {
        (self.main, self.val)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize, FF: serde::Serialize, KK: serde::Serialize",
    deserialize = "C: serde::de::DeserializeOwned, FF: serde::de::DeserializeOwned, KK: serde::de::DeserializeOwned + Default"
))]
#[must_use]
pub struct ShardFoldOutputs<C, FF, KK> {
    pub obligations: ShardObligations<C, FF, KK>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShardFoldWitnesses<FF> {
    /// Witnesses for `ShardFoldOutputs::obligations.main` (one per ME instance).
    pub final_main_wits: Vec<Mat<FF>>,
    /// Witnesses for `ShardFoldOutputs::obligations.val` (one per ME instance).
    pub val_lane_wits: Vec<Mat<FF>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MemOrLutProof {
    Twist(TwistProofK),
    Shout(ShoutProofK),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize, FF: serde::Serialize, KK: serde::Serialize",
    deserialize = "C: serde::de::DeserializeOwned, FF: serde::de::DeserializeOwned, KK: serde::de::DeserializeOwned + Default"
))]
pub struct MemSidecarProof<C, FF, KK> {
    /// ME claims evaluated at `r_val` (Twist val-eval terminal point).
    ///
    /// Shared-bus mode only: these are CPU ME openings at `r_val` that include appended bus openings.
    pub val_me_claims: Vec<CeClaim<C, FF, KK>>,
    /// CPU ME openings at `r_time` used to bind WB booleanity terminals to committed trace columns.
    pub wb_me_claims: Vec<CeClaim<C, FF, KK>>,
    /// CPU ME openings at `r_time` used to bind WP quiescence terminals to committed trace columns.
    pub wp_me_claims: Vec<CeClaim<C, FF, KK>>,
    /// Poseidon cycle-lane ME openings at `r_time` used to bind Poseidon Route-A cycle terminals.
    pub poseidon_cycle_me_claims: Vec<CeClaim<C, FF, KK>>,
    /// Poseidon local-lane ME openings used to bind local round/transition terminals.
    pub poseidon_local_me_claims: Vec<CeClaim<C, FF, KK>>,
    /// Route A Shout address pre-time proofs batched across all Shout instances in the step.
    pub shout_addr_pre: ShoutAddrPreProof<KK>,
    pub proofs: Vec<MemOrLutProof>,
}

/// Proof for the Route A shared-challenge batched sum-check (time/row rounds).
///
/// This batches CCS (row/time rounds) with Twist/Shout time-domain oracles so all
/// protocols share the same transcript-derived `r` (enabling Π_RLC folding).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchedTimeProof {
    /// Claimed sums per participating oracle (in the same order as `round_polys`).
    pub claimed_sums: Vec<K>,
    /// Degree bounds per participating oracle.
    pub degree_bounds: Vec<usize>,
    /// Domain-separation labels per participating oracle.
    pub labels: Vec<Vec<u8>>,
    /// Per-claim sum-check messages: `round_polys[claim][round] = coeffs`.
    pub round_polys: Vec<Vec<Vec<K>>>,
}

/// Proof data for a standalone Π_RLC → Π_DEC lane (no Π_CCS).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RlcDecProof {
    /// RLC mixing matrices ρ_i ∈ S ⊆ F^{D×D}
    pub rlc_rhos: Vec<RotRho>,
    /// The combined parent after RLC: ME(B,L) with B=b^k
    pub rlc_parent: CeClaim<Cmt, F, K>,
    /// DEC children: k ME(b,L) after decomposition of the parent
    pub dec_children: Vec<CeClaim<Cmt, F, K>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StepProof {
    pub fold: FoldStep,
    pub mem: MemSidecarProof<Cmt, F, K>,
    pub batched_time: BatchedTimeProof,
    /// Optional Poseidon local-domain batched-time proof (ell = 6 for subperm+row).
    pub poseidon_local_time: Option<BatchedTimeProof>,
    /// Folding lane(s) for Poseidon cycle-lane ME claims evaluated at `r_time`.
    pub poseidon_cycle_fold: Vec<RlcDecProof>,
    /// Folding lane(s) for Poseidon local-lane ME claims.
    pub poseidon_local_fold: Vec<RlcDecProof>,
    /// Optional folding lane(s) for ME claims evaluated at `r_val`.
    ///
    /// Each proof is an independent Π_RLC→Π_DEC lane (k=1 in current usage).
    pub val_fold: Vec<RlcDecProof>,
    /// Reserved WB folding lane(s) for staged booleanity claims.
    pub wb_fold: Vec<RlcDecProof>,
    /// Reserved WP folding lane(s) for staged quiescence claims.
    pub wp_fold: Vec<RlcDecProof>,
    /// Optional nested per-step proofs used by compressed segment encodings.
    ///
    /// When present, this `StepProof` acts as a container and verification should
    /// expand and check the nested steps against the corresponding public segment.
    pub compressed_substeps: Option<Vec<StepProof>>,
    /// Stage-8 joint-opening folding lane(s) (canonical mode uses exactly one unified lane).
    pub stage8_fold: Vec<RlcDecProof>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ShardSegmentKind {
    CcsOnly,
    RouteA,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ShardSegmentMeta {
    /// Segment kind derived from step content.
    pub kind: ShardSegmentKind,
    /// Number of public steps in the segment.
    pub public_steps: usize,
    /// Number of proof steps consumed from `ShardProof::steps`.
    ///
    /// For CCS-only batched segments this is typically `< public_steps`.
    /// Route-A uses compressed chunk containers (typically `1` per chunk).
    pub proof_steps: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShardProof {
    pub steps: Vec<StepProof>,
    /// Optional output binding proof (proves final memory matches claimed outputs).
    /// Twist linkage is proven as an extra Route-A batched-time claim on the final step.
    pub output_proof: Option<OutputBindingProof>,
    /// Optional verifier-visible RISC-V proof profile metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub riscv_profile: Option<RiscvProofProfileConfig>,
    /// Optional deterministic guest-memory layout bound to the proof.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub riscv_memory_layout: Option<RiscvGuestMemoryLayout>,
    /// Segment metadata for mixed CCS-only/Route-A proving.
    ///
    /// Mixed verification requires this metadata and uses it to partition
    /// `steps` against contiguous public segments.
    ///
    /// `None` is reserved for non-mixed helper paths that do not use mixed-segment verification.
    pub segment_meta: Option<Vec<ShardSegmentMeta>>,
}

impl ShardProof {
    fn step_final_main(step: &StepProof) -> Vec<CeClaim<Cmt, F, K>> {
        // For compressed Route-A chunks, the container is the terminal step.
        step.fold.dec_children.clone()
    }

    fn extend_val_from_step(step: &StepProof, out: &mut Vec<CeClaim<Cmt, F, K>>) {
        if let Some(sub) = step.compressed_substeps.as_ref() {
            for inner in sub {
                Self::extend_val_from_step(inner, out);
            }
        }
        for p in &step.poseidon_cycle_fold {
            out.extend_from_slice(&p.dec_children);
        }
        for p in &step.poseidon_local_fold {
            out.extend_from_slice(&p.dec_children);
        }
        for p in &step.val_fold {
            out.extend_from_slice(&p.dec_children);
        }
        for p in &step.wb_fold {
            out.extend_from_slice(&p.dec_children);
        }
        for p in &step.wp_fold {
            out.extend_from_slice(&p.dec_children);
        }
        for p in &step.stage8_fold {
            out.extend_from_slice(&p.dec_children);
        }
    }

    pub fn compute_final_obligations(&self, acc_init: &[CeClaim<Cmt, F, K>]) -> ShardObligations<Cmt, F, K> {
        self.compute_fold_outputs(acc_init).obligations
    }

    /// Returns the final main accumulator only (does not include Twist `r_val` obligations).
    pub fn compute_final_main_children(&self, acc_init: &[CeClaim<Cmt, F, K>]) -> Vec<CeClaim<Cmt, F, K>> {
        self.compute_fold_outputs(acc_init).obligations.main
    }

    pub fn compute_fold_outputs(&self, acc_init: &[CeClaim<Cmt, F, K>]) -> ShardFoldOutputs<Cmt, F, K> {
        let main = if self.steps.is_empty() {
            acc_init.to_vec()
        } else {
            Self::step_final_main(self.steps.last().expect("non-empty"))
        };

        let mut val = Vec::new();
        for step in &self.steps {
            Self::extend_val_from_step(step, &mut val);
        }

        ShardFoldOutputs {
            obligations: ShardObligations { main, val },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn batched_time_proof_bincode_roundtrip_keeps_labels() {
        let proof = BatchedTimeProof {
            claimed_sums: vec![K::ONE, K::ZERO],
            degree_bounds: vec![3, 5],
            labels: vec![b"shout/value".to_vec(), b"control/next_pc_linear".to_vec()],
            round_polys: vec![vec![vec![K::ONE]], vec![vec![K::ZERO, K::ONE]]],
        };

        let bytes = bincode::serialize(&proof).expect("serialize BatchedTimeProof");
        let decoded: BatchedTimeProof = bincode::deserialize(&bytes).expect("deserialize BatchedTimeProof");

        assert_eq!(decoded.claimed_sums, proof.claimed_sums);
        assert_eq!(decoded.degree_bounds, proof.degree_bounds);
        assert_eq!(decoded.labels, proof.labels);
        assert_eq!(decoded.round_polys, proof.round_polys);
    }
}
