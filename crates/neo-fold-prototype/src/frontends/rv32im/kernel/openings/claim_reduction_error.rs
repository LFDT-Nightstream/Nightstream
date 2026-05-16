//! Owns Phase 1 claim-reduction error reporting.

use neo_math::K;
use thiserror::Error;

use crate::opening::OpeningDomain;

use super::opening_eval_claims::{EvalClaimError, FamilyEvalClaimId, FamilyEvalSchemaId};

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ClaimReductionError {
    #[error("phase1 claim-reduction bucket cannot be empty")]
    EmptyBucket,
    #[error("phase1 eval-claim bundle digest mismatch: expected {expected:?}, got {actual:?}")]
    EvalClaimBundleDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 could not build a canonical claim bundle from witnesses: {0}")]
    WitnessClaimBundleBuildFailed(EvalClaimError),
    #[error("phase1 bucket contains invalid phase0 claim at index {index}: {source}")]
    InvalidPhase0Claim {
        index: usize,
        source: EvalClaimError,
    },
    #[error("phase1 witness set contains duplicate claim id {claim_id:?}")]
    DuplicateWitnessClaimId { claim_id: FamilyEvalClaimId },
    #[error("phase1 witness set is missing claim witness for id {claim_id:?}")]
    MissingWitnessForClaimId { claim_id: FamilyEvalClaimId },
    #[error("phase1 witness claim for id {claim_id:?} does not match the bucket claim")]
    WitnessClaimMismatch { claim_id: FamilyEvalClaimId },
    #[error("phase1 binding surface target count mismatch: expected {expected}, got {actual}")]
    BindingSurfaceTargetCountMismatch { expected: usize, actual: usize },
    #[error("phase1 binding surface schema mismatch at index {index}: expected {expected:?}, got {actual:?}")]
    BindingSurfaceSchemaMismatch {
        index: usize,
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
    },
    #[error("phase1 binding surface is missing target for {schema:?}")]
    BindingSurfaceTargetMissing { schema: FamilyEvalSchemaId },
    #[error("phase1 binding surface target digest mismatch at index {index}: expected {expected:?}, got {actual:?}")]
    BindingSurfaceTargetDigestMismatch {
        index: usize,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 binding surface digest mismatch: expected {expected:?}, got {actual:?}")]
    BindingSurfaceDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 witness claim {claim_id:?} binding digest mismatch: expected {expected:?}, got {actual:?}")]
    WitnessClaimBindingDigestMismatch {
        claim_id: FamilyEvalClaimId,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 witness claim {claim_id:?} point does not match its canonical binding-derived point")]
    WitnessClaimPointBindingMismatch { claim_id: FamilyEvalClaimId },
    #[error("phase1 bucket mixes commitment contexts at claim {index}")]
    MixedCommitmentContext { index: usize },
    #[error("phase1 bucket mixes schemas: expected {expected:?}, got {actual:?} at claim {index}")]
    MixedSchema {
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
        index: usize,
    },
    #[error("phase1 bucket mixes point arities: expected {expected}, got {actual} at claim {index}")]
    MixedPointArity {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("phase1 bucket mixes payload widths: expected {expected}, got {actual} at claim {index}")]
    MixedPayloadWidth {
        expected: usize,
        actual: usize,
        index: usize,
    },
    #[error("phase1 bucket mixes opening domains: expected {expected:?}, got {actual:?} at claim {index}")]
    MixedOpeningDomain {
        expected: OpeningDomain,
        actual: OpeningDomain,
        index: usize,
    },
    #[error("phase1 proof bucket digest mismatch: expected {expected:?}, got {actual:?}")]
    BucketDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 proof for payload width {payload_width} requires gamma")]
    MissingGamma { payload_width: usize },
    #[error("phase1 proof for payload width {payload_width} must not carry gamma")]
    UnexpectedGamma { payload_width: usize },
    #[error("phase1 scalar eval count mismatch: expected {expected}, got {actual}")]
    ScalarEvalCountMismatch { expected: usize, actual: usize },
    #[error("phase1 proof digest mismatch: expected {expected:?}, got {actual:?}")]
    ProofDigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("phase1 eta mismatch: expected {expected:?}, got {actual:?}")]
    EtaMismatch { expected: K, actual: K },
    #[error("phase1 gamma mismatch: expected {expected:?}, got {actual:?}")]
    GammaMismatch {
        expected: Option<K>,
        actual: Option<K>,
    },
    #[error("phase1 rho mismatch: expected {expected:?}, got {actual:?}")]
    RhoMismatch { expected: K, actual: K },
    #[error("phase1 result unified-claim count mismatch: expected {expected}, got {actual}")]
    UnifiedClaimCountMismatch { expected: usize, actual: usize },
    #[error("phase1 result unified-point arity mismatch: expected {expected}, got {actual}")]
    UnifiedPointArityMismatch { expected: usize, actual: usize },
    #[error("phase1 result round count mismatch: expected {expected}, got {actual}")]
    RoundCountMismatch { expected: usize, actual: usize },
    #[error("phase1 result contains invalid unified claim at index {index}: {source}")]
    InvalidUnifiedClaim {
        index: usize,
        source: EvalClaimError,
    },
    #[error("phase1 result unified claim {index} does not carry the result unified point")]
    UnifiedPointMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed opened object")]
    UnifiedOpenedObjectMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed claim id")]
    UnifiedClaimIdMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed commitment context")]
    UnifiedCommitmentContextMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed binding digest")]
    UnifiedBindingDigestMismatch { index: usize },
    #[error("phase1 result unified claim {index} changed schema: expected {expected:?}, got {actual:?}")]
    UnifiedSchemaMismatch {
        expected: FamilyEvalSchemaId,
        actual: FamilyEvalSchemaId,
        index: usize,
    },
    #[error(
        "phase1 result carries different payloads for opened object {opened_object_digest:?} at one unified point"
    )]
    SameObjectPayloadMismatch { opened_object_digest: [u8; 32] },
    #[error("phase1 sumcheck round {round} invariant failed: expected {expected:?}, got {actual:?}")]
    SumcheckInvariantFailed {
        round: usize,
        expected: K,
        actual: K,
    },
    #[error("phase1 result unified point does not match the transcript-derived r*")]
    UnifiedPointTranscriptMismatch,
    #[error("phase1 witness payload evaluation failed for unified claim {index}: {source}")]
    WitnessPayloadEvaluationFailed {
        index: usize,
        source: EvalClaimError,
    },
    #[error("phase1 unified claim {index} payload does not match its real witness at r*")]
    UnifiedPayloadDoesNotMatchWitness { index: usize },
    #[error("phase1 scalar eval mismatch for unified claim {index}: expected {expected:?}, got {actual:?}")]
    ScalarEvalMismatch {
        index: usize,
        expected: K,
        actual: K,
    },
    #[error("phase1 combined scalar check failed: expected {expected:?}, got {actual:?}")]
    CombinedScalarMismatch { expected: K, actual: K },
    #[error("phase1 result count mismatch: expected {expected}, got {actual}")]
    ResultCountMismatch { expected: usize, actual: usize },
    #[error("phase1 result at index {index} carries an unexpected bucket")]
    UnexpectedBucketAtIndex { index: usize },
}
