import Nightstream.Protocol.Nebula.ApplicationRowRun
import Nightstream.Protocol.Nebula.IdealFingerprint
import Nightstream.Protocol.Nebula.IdealSequence
import Nightstream.Protocol.Nebula.FullClaim
import Nightstream.Protocol.Nebula.GlobalFPrime
import Nightstream.Protocol.Nebula.ProductState
import Nightstream.Protocol.Nebula.Soundness
import Nightstream.Protocol.Nebula.Transcript

/-!
Contract: independent ideal acceptance semantics for Nebula V2.

Assurance tier: model-level and security-reduction boundary.

Owns raw per-segment ideal checks, exact F-prime runs, root-linked segment
chains, and the direct reduction from ideal acceptance to one completed
application and memory execution or an explicit algebraic/root failure.

The acceptance structure does not contain `Balanced`, `ValidSegment`,
`ValidChain`, `CompletedExecution`, `ExecutionWitness`, `HasSoundExecution`,
or `AcceptanceReduction`. Its application side is an explicit row-by-row run;
the completed execution is reconstructed by theorem.

Does not own generated rows, Rust refinement, application-port coverage,
Poseidon2 or Fiat-Shamir security, commitment binding, NIFS extraction, or the
deployed terminal verifier.

The model requires every verified claim to update the two-by-four product
state from its exact typed record chunk. Exact chunk coverage then proves that
the closing state is the product of the complete semantic multisets. Generated
rows and Rust still need a refinement theorem to this independent update.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.IdealAcceptance

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationRowRun
open Nightstream.Protocol.Nebula.ApplicationTrace
open Nightstream.Protocol.Nebula.Completion
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.FullClaim
open Nightstream.Protocol.Nebula.GlobalFPrime
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.Protocol.Nebula.IdealSequence
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.Memory
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.Protocol.Nebula.SequenceBinding
open Nightstream.Protocol.Nebula.Soundness

abbrev FullVerifier
    (schema : FullClaim.Schema) (Digest ChallengeField : Type) :=
  schema.NifsProof →
    FullClaim.Claim schema Digest (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField) → Prop

abbrev VerifiedFullClaim
    (schema : FullClaim.Schema) (Digest ChallengeField : Type)
    (verify : FullVerifier schema Digest ChallengeField) :=
  FullClaim.Verified schema Digest (ProductState.Challenges ChallengeField)
    (ProductState.State ChallengeField) verify

/-- Independent arithmetic relation between one exact verified full claim and
the typed records consumed by that checked step. -/
def ClaimProductUpdate
    {ChallengeField Digest : Type} [Field ChallengeField]
    {schema : FullClaim.Schema}
    {verify : FullVerifier schema Digest ChallengeField}
    (encode : Nat → ChallengeField)
    (challenges : ProductState.Challenges ChallengeField)
    (claim : VerifiedFullClaim schema Digest ChallengeField verify)
    (chunk : ProductState.Chunk) : Prop :=
  claim.claim.memory.productsAfter =
    ProductState.update encode challenges
      claim.claim.memory.productsBefore chunk

/-- Verifier-owned functions and values used by the ideal model. -/
structure Config
    (ChallengeField Profile Plan Commitment Digest : Type)
    [Field ChallengeField] where
  encode : Nat → ChallengeField
  encodeInjective : InjectiveBelowGoldilocks encode
  snapshotRoot : Snapshot → Digest
  chainRoot : FramedSequence Profile Plan Commitment → Digest
  profile : Profile
  plan : Plan
  operationsLane : List Access → FramedSequence Profile Plan Commitment
  snapshotLane : Snapshot → FramedSequence Profile Plan Commitment
  chainHeader : Profile → Plan → LaneDomain → Digest
  priorStateDigest : Nat → Digest
  runningAccumulatorDigest : Nat → Digest
  challengeOracle : Transcript.Oracle Digest ChallengeField
  statementIdentity : StatementIdentity Digest

def Config.challengeFrame
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (segmentIndex timestampIn activeAccessCount : Nat)
    (roots : Roots Digest) : Transcript.Frame Digest :=
  { profile := config.statementIdentity.profile
    verifierKeyDigest := config.statementIdentity.verifierKey.digest
    applicationRelationDigest :=
      config.statementIdentity.applicationRelationDigest
    programDigest := config.statementIdentity.programDigest
    memoryPlanDigest := config.statementIdentity.memoryPlanDigest
    laneLayoutDigest :=
      config.statementIdentity.verifierKey.laneLayoutDigest
    priorStateDigest := config.priorStateDigest segmentIndex
    runningAccumulatorDigest := config.runningAccumulatorDigest segmentIndex
    segmentIndex := segmentIndex
    segmentStartTimestamp := timestampIn
    activeAccessCount := activeAccessCount
    segmentEndTimestamp := timestampIn + activeAccessCount
    roots := roots }

def Config.deriveChallenge
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (segmentIndex timestampIn activeAccessCount : Nat)
    (roots : Roots Digest) : ProductState.Challenges ChallengeField :=
  Transcript.derive config.challengeOracle
    (config.challengeFrame segmentIndex timestampIn activeAccessCount roots)

def Config.canonicalHeaders
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest) :
    Roots Digest :=
  { operations := config.chainHeader config.profile config.plan .operations
    initialSnapshot := config.chainHeader config.profile config.plan .memory
    finalSnapshot := config.chainHeader config.profile config.plan .memory }

def Config.authoritativeRoots
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (initial : Snapshot) (accesses : List Access) (final : Snapshot) :
    Roots Digest :=
  { operations := config.chainRoot (config.operationsLane accesses)
    initialSnapshot := config.chainRoot (config.snapshotLane initial)
    finalSnapshot := config.chainRoot (config.snapshotLane final) }

def Config.authoritativeSequence
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (initial : Snapshot) (accesses : List Access) (final : Snapshot) :
    Role → FramedSequence Profile Plan Commitment
  | .operations => config.operationsLane accesses
  | .initialSnapshot => config.snapshotLane initial
  | .finalSnapshot => config.snapshotLane final

def SnapshotRootCollision
    {Digest : Type} (snapshotRoot : Snapshot → Digest) : Prop :=
  ∃ left right,
    left ≠ right ∧ snapshotRoot left = snapshotRoot right

theorem snapshot_eq_or_collision
    {Digest : Type} {snapshotRoot : Snapshot → Digest}
    {left right : Snapshot}
    (equalRoot : snapshotRoot left = snapshotRoot right) :
    left = right ∨ SnapshotRootCollision snapshotRoot := by
  by_cases equal : left = right
  · exact Or.inl equal
  · exact Or.inr ⟨left, right, equal, equalRoot⟩

/-- Exact F-prime evidence for one segment. Fingerprint acceptance is not a
field: it must be exposed by the closing transition of `run`. -/
structure FPrimeEvidence
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (schema : FullClaim.Schema)
    (verify : FullVerifier schema Digest ChallengeField)
    (segmentIndex timestampIn timestampOut : Nat)
    {initial final : Snapshot} {accesses : List Access}
    (fingerprint :
      Check config.encode initial accesses final) where
  active : ActiveCarry Digest (ProductState.Challenges ChallengeField)
    (ProductState.State ChallengeField)
  closed : ClosedCarry Digest
  claims : List (VerifiedFullClaim schema Digest ChallengeField verify)
  productChunks : List ProductState.Chunk
  productUpdates :
    List.Forall₂
      (ClaimProductUpdate config.encode fingerprint.challenges)
      claims productChunks
  productCoverage :
    ProductState.Covers initial accesses final productChunks
  stepZero : active.stepIndex.val = 0
  activeSegment : active.segmentIndex = segmentIndex
  activeTimestamp : active.globalTimestamp = timestampIn
  segmentStartTimestamp : active.segmentStartTimestamp = timestampIn
  activeAccessCount : active.segmentActiveAccessCount = accesses.length
  segmentEndTimestamp : active.segmentEndTimestamp = timestampOut
  activeMemoryRoot : active.memoryRoot = config.snapshotRoot initial
  openingProducts : active.products = ProductState.one
  openingHeaders : active.dSeen = config.canonicalHeaders
  closedSegment : closed.segmentIndex = segmentIndex + 1
  closedTimestamp : closed.globalTimestamp = timestampOut
  closedMemoryRoot : closed.memoryRoot = config.snapshotRoot final
  run :
    FullClaim.VerifiedRun verify
      ProductState.Balanced
      (.active active) claims (.closed closed)

namespace FPrimeEvidence

private theorem accumulated_products_are_balanced_aux
    {ChallengeField Digest : Type} [Field ChallengeField]
    {schema : FullClaim.Schema}
    {verify : FullVerifier schema Digest ChallengeField}
    {encode : Nat → ChallengeField}
    {challenges : ProductState.Challenges ChallengeField}
    {before after : Carry Digest
      (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {claims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
    {chunks : List ProductState.Chunk}
    (run : FullClaim.VerifiedRun verify ProductState.Balanced
      before claims after)
    (updates : List.Forall₂ (ClaimProductUpdate encode challenges)
      claims chunks)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∀ active, before = .active active →
      ProductState.Balanced
        (ProductState.accumulate encode challenges active.products chunks) := by
  induction run generalizing chunks with
  | nil =>
      intro active beforeEqual
      rcases afterClosed with ⟨closed, afterEqual⟩
      rw [beforeEqual] at afterEqual
      cases afterEqual
  | cons step rest inductionHypothesis =>
      cases updates with
      | cons updateHead updatesTail =>
          intro requestedActive beforeEqual
          unfold ClaimProductUpdate at updateHead
          cases step.consumes with
          | interior agreement notLast =>
              cases beforeEqual
              have tailBalanced := inductionHypothesis updatesTail
                ⟨_, rfl⟩ afterClosed _ rfl
              simpa [ProductState.accumulate, interiorCarry, updateHead,
                agreement.products] using tailBalanced
          | close agreement last checks =>
              cases beforeEqual
              have tailEmpty := rest.from_closed_is_empty
              rw [tailEmpty.1] at updatesTail
              cases updatesTail
              simpa [ProductState.accumulate, updateHead,
                agreement.products] using
                checks.productsBalanced

private theorem accumulated_products_are_balanced
    {ChallengeField Digest : Type} [Field ChallengeField]
    {schema : FullClaim.Schema}
    {verify : FullVerifier schema Digest ChallengeField}
    {encode : Nat → ChallengeField}
    {challenges : ProductState.Challenges ChallengeField}
    {active : ActiveCarry Digest
      (ProductState.Challenges ChallengeField)
      (ProductState.State ChallengeField)}
    {closed : ClosedCarry Digest}
    {claims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
    {chunks : List ProductState.Chunk}
    (run : FullClaim.VerifiedRun verify ProductState.Balanced
      (.active active) claims (.closed closed))
    (updates : List.Forall₂ (ClaimProductUpdate encode challenges)
      claims chunks) :
    ProductState.Balanced
      (ProductState.accumulate encode challenges active.products chunks) :=
  accumulated_products_are_balanced_aux run updates
    ⟨active, rfl⟩ ⟨closed, rfl⟩ active rfl

theorem fingerprintAccepted
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config :
      Config ChallengeField Profile Plan Commitment Digest}
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    {fingerprint : Check config.encode initial accesses final}
    {verify : FullVerifier schema Digest ChallengeField}
    (evidence : FPrimeEvidence config schema verify segmentIndex timestampIn
      timestampOut fingerprint) :
    fingerprint.Accepts := by
  have accumulatedBalanced := accumulated_products_are_balanced
    evidence.run evidence.productUpdates
  have productsExact :
      ProductState.accumulate config.encode fingerprint.challenges
          ProductState.one evidence.productChunks =
        ProductState.expected fingerprint :=
    ProductState.accumulate_one_eq_expected fingerprint
      evidence.productCoverage
  rw [evidence.openingProducts, productsExact] at accumulatedBalanced
  exact (ProductState.accepts_iff_expected_balanced fingerprint).mpr
    accumulatedBalanced

theorem claimCount
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config :
      Config ChallengeField Profile Plan Commitment Digest}
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    {fingerprint : Check config.encode initial accesses final}
    {verify : FullVerifier schema Digest ChallengeField}
    (evidence : FPrimeEvidence config schema verify segmentIndex timestampIn
      timestampOut fingerprint) :
    evidence.claims.length = claimsPerSegment :=
  FullClaim.VerifiedRun.full_segment_has_exact_claim_count
    evidence.stepZero evidence.run

def claimAt
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    {fingerprint : Check config.encode initial accesses final}
    {verify : FullVerifier schema Digest ChallengeField}
    (evidence : FPrimeEvidence config schema verify segmentIndex timestampIn
      timestampOut fingerprint)
    (step : Fin claimsPerSegment) :
    VerifiedFullClaim schema Digest ChallengeField verify :=
  evidence.claims.get
    ⟨step.val, by rw [evidence.claimCount]; exact step.isLt⟩

end FPrimeEvidence

/-- Both the prechallenge and replay sequences are the exact lane components
of the mandatory bundles in the verified full claims. -/
structure BundleAuthority
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    {fingerprint : Check config.encode initial accesses final}
    (bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment)
    (evidence : FPrimeEvidence config schema verify segmentIndex timestampIn
      timestampOut fingerprint)
    (sequences : Checks config.chainRoot config.profile config.plan) : Prop where
  precommit : ∀ role,
    (sequences.lane role).precommit.sequence.commitments =
      fun step =>
        bundleComponent
          (evidence.claimAt step).claim.commitmentBundle role.component
  replay : ∀ role,
    (sequences.lane role).replay.sequence.commitments =
      fun step =>
        bundleComponent
          (evidence.claimAt step).claim.commitmentBundle role.component

/-- Raw ideal checks for one segment. Exact balance and final-snapshot validity
are deliberately absent and will be derived. -/
structure SegmentCheck
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (schema : FullClaim.Schema)
    (bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment)
    (verify : FullVerifier schema Digest ChallengeField)
    (segmentIndex : Nat)
    (initial : Snapshot) (timestampIn : Nat)
    (accesses : List Access)
    (final : Snapshot) (timestampOut : Nat) where
  accessBound : accesses.length ≤ 63 * 1088
  initialValid : initial.ValidAt timestampIn
  ordered : Ordered timestampIn accesses timestampOut
  fingerprint : Check config.encode initial accesses final
  sequences : Checks config.chainRoot config.profile config.plan
  fprime : FPrimeEvidence config schema verify segmentIndex timestampIn
    timestampOut fingerprint
  operationsAuthority :
    sequences.operations.precommit.sequence = config.operationsLane accesses
  initialSnapshotAuthority :
    sequences.initialSnapshot.precommit.sequence = config.snapshotLane initial
  finalSnapshotAuthority :
    sequences.finalSnapshot.precommit.sequence = config.snapshotLane final
  dPreMatches : fprime.active.dPre = sequences.committedRoots
  challengeDerived :
    fprime.active.challenge =
      config.deriveChallenge segmentIndex timestampIn accesses.length
        sequences.committedRoots
  fingerprintChallenges :
    fingerprint.challenges = fprime.active.challenge
  bundleAuthority :
    BundleAuthority bundleComponent fprime sequences

/-- Failures that the ideal deterministic reduction exposes instead of hiding
them in an assumed final witness. -/
inductive Failure
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest) :
    Prop where
  | fingerprint
      {schema : FullClaim.Schema}
      {bundleComponent :
        schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
      {verify : FullVerifier schema Digest ChallengeField}
      {segmentIndex timestampIn timestampOut : Nat}
      {initial final : Snapshot} {accesses : List Access}
      (segment : SegmentCheck config schema bundleComponent verify segmentIndex initial
        timestampIn accesses final timestampOut)
      (failure : EvaluationFailure segment.fingerprint)
      (sequencesExact : segment.sequences.Exact) : Failure config
  | sequence
      (collision : RootCollision config.chainRoot) : Failure config
  | snapshotRoot
      (collision : SnapshotRootCollision config.snapshotRoot) : Failure config

namespace SegmentCheck

theorem precommit_sequence_is_authoritative
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex
      initial timestampIn accesses final timestampOut)
    (role : Role) :
    (segment.sequences.lane role).precommit.sequence =
      config.authoritativeSequence initial accesses final role := by
  cases role with
  | operations => exact segment.operationsAuthority
  | initialSnapshot => exact segment.initialSnapshotAuthority
  | finalSnapshot => exact segment.finalSnapshotAuthority

/-- Every canonical prechallenge commitment is the selected component of the
mandatory bundle in the exact verified full claim at that step. -/
theorem claim_bundles_bind_authoritative_sequence
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex
      initial timestampIn accesses final timestampOut)
    (role : Role) :
    (config.authoritativeSequence initial accesses final role).commitments =
      fun step =>
        bundleComponent
          (segment.fprime.claimAt step).claim.commitmentBundle role.component := by
  rw [← segment.precommit_sequence_is_authoritative role]
  exact segment.bundleAuthority.precommit role

/-- Replay uses those same verified bundles, not a post-challenge sidecar. -/
theorem replay_uses_exact_verified_claim_bundles
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex
      initial timestampIn accesses final timestampOut)
    (role : Role) :
    (segment.sequences.lane role).replay.sequence.commitments =
      fun step =>
        bundleComponent
          (segment.fprime.claimAt step).claim.commitmentBundle role.component :=
  segment.bundleAuthority.replay role

/-- The carried prechallenge roots are the roots of the canonical lanes of the
actual records checked by this segment. -/
theorem dPre_binds_authoritative_lanes
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {config :
      Config ChallengeField Profile Plan Commitment Digest}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex initial
      timestampIn accesses final timestampOut) :
    segment.fprime.active.dPre =
      config.authoritativeRoots initial accesses final := by
  rw [segment.dPreMatches]
  apply Roots.ext
  · exact segment.sequences.operations.precommit.rootCorrect.symm.trans
      (congrArg config.chainRoot segment.operationsAuthority)
  · exact
      segment.sequences.initialSnapshot.precommit.rootCorrect.symm.trans
        (congrArg config.chainRoot segment.initialSnapshotAuthority)
  · exact segment.sequences.finalSnapshot.precommit.rootCorrect.symm.trans
      (congrArg config.chainRoot segment.finalSnapshotAuthority)

/-- The polynomial evaluation points come from the challenge that was derived
from the roots of those same canonical lanes. -/
theorem fingerprint_challenges_bind_authoritative_lanes
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {config :
      Config ChallengeField Profile Plan Commitment Digest}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex initial
      timestampIn accesses final timestampOut) :
    segment.fingerprint.challenges =
      config.deriveChallenge segmentIndex timestampIn accesses.length
        (config.authoritativeRoots initial accesses final) := by
  have roots :
      segment.sequences.committedRoots =
        config.authoritativeRoots initial accesses final :=
    segment.dPreMatches.symm.trans segment.dPre_binds_authoritative_lanes
  rw [segment.fingerprintChallenges, segment.challengeDerived, roots]

/-- Expanded form: all four challenge coordinates use the one complete typed
frame, including profile, key, application, program, plan, lane layout, prior
state, running accumulator, counters, dimensions, and authoritative roots. -/
theorem fingerprint_challenges_use_complete_frame
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex
      initial timestampIn accesses final timestampOut) :
    segment.fingerprint.challenges =
      Transcript.derive config.challengeOracle
        (config.challengeFrame segmentIndex timestampIn accesses.length
          (config.authoritativeRoots initial accesses final)) := by
  exact segment.fingerprint_challenges_bind_authoritative_lanes

/-- One raw segment yields the semantic segment relation or an exact named
fingerprint/sequence failure. -/
theorem valid_or_failure
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {config :
      Config ChallengeField Profile Plan Commitment Digest}
    {schema : FullClaim.Schema}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {accesses : List Access}
    (segment : SegmentCheck config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut) :
    Failure config ∨
      ValidSegment initial final timestampIn accesses timestampOut := by
  rcases segment.sequences.exact_or_collision with
    _exactSequences | collision
  · have accepted := segment.fprime.fingerprintAccepted
    rcases balance_or_evaluationFailure config.encodeInjective
        segment.fingerprint accepted with balance | fingerprintFailure
    · exact Or.inr
        { initialValid := segment.initialValid
          finalValid := ValidSegment.finalValid_of_balance
            segment.initialValid segment.ordered balance
          ordered := segment.ordered
          balanced := balance }
    · exact Or.inl (.fingerprint segment fingerprintFailure _exactSequences)
  · exact Or.inl (.sequence collision)

end SegmentCheck

/-- Segment checks are linked by the carried snapshot root and timestamp. The
global claim list is the exact concatenation of the per-segment F-prime runs. -/
inductive CheckedChain
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (schema : FullClaim.Schema)
    (bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment)
    (verify : FullVerifier schema Digest ChallengeField) :
    Nat → Snapshot → Nat → List (List Access) →
      List (VerifiedFullClaim schema Digest ChallengeField verify) → Snapshot → Nat → Prop
  | nil (segmentIndex : Nat) (snapshot : Snapshot) (timestamp : Nat) :
      CheckedChain config schema bundleComponent verify segmentIndex snapshot timestamp
        [] [] snapshot timestamp
  | cons
      {segmentIndex : Nat}
      {initial headFinal tailInitial final : Snapshot}
      {timestampIn timestampMiddle timestampOut : Nat}
      {accesses : List Access} {rest : List (List Access)}
      {tailClaims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
      (head : SegmentCheck config schema bundleComponent verify segmentIndex initial
        timestampIn accesses headFinal timestampMiddle)
      (boundaryRoot :
        config.snapshotRoot headFinal = config.snapshotRoot tailInitial)
      (tail : CheckedChain config schema bundleComponent verify (segmentIndex + 1) tailInitial
        timestampMiddle rest tailClaims final timestampOut) :
      CheckedChain config schema bundleComponent verify segmentIndex initial timestampIn
        (accesses :: rest) (head.fprime.claims ++ tailClaims)
        final timestampOut

namespace CheckedChain

theorem valid_or_failure
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {config :
      Config ChallengeField Profile Plan Commitment Digest}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {segments : List (List Access)}
    {verify : FullVerifier schema Digest ChallengeField}
    {claims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
    (chain : CheckedChain config schema bundleComponent verify segmentIndex initial timestampIn
      segments claims final timestampOut) :
    Failure config ∨
      ValidChain initial timestampIn segments final timestampOut := by
  induction chain with
  | nil => exact Or.inr (.nil _ _)
  | cons head boundaryRoot _ inductionHypothesis =>
      rcases head.valid_or_failure with failure | validHead
      · exact Or.inl failure
      rcases inductionHypothesis with failure | validTail
      · exact Or.inl failure
      rcases snapshot_eq_or_collision boundaryRoot with
        boundaryExact | collision
      · cases boundaryExact
        exact Or.inr (.cons validHead validTail)
      · exact Or.inl (.snapshotRoot collision)

/-- Exact F-prime accounting is derived from each checked run, not copied from
the profile table. -/
theorem globalClaimCount
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {config :
      Config ChallengeField Profile Plan Commitment Digest}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {segments : List (List Access)}
    {verify : FullVerifier schema Digest ChallengeField}
    {claims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
    (chain : CheckedChain config schema bundleComponent verify segmentIndex initial timestampIn
      segments claims final timestampOut) :
    claims.length = segments.length * claimsPerSegment := by
  induction chain with
  | nil => simp
  | cons head _ _ inductionHypothesis =>
      rw [List.length_append, head.fprime.claimCount, inductionHypothesis]
      simp only [List.length_cons]
      unfold claimsPerSegment
      omega

end CheckedChain

/-- Acceptance by the independent ideal verifier. The application relation is
an explicit local row run plus public bounds and ordered port equality. This
structure contains neither a completed application execution nor a semantic
memory-execution conclusion. -/
structure IdealAcceptV2
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (schema : FullClaim.Schema)
    (bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment)
    (verify : FullVerifier schema Digest ChallengeField)
    {Program ApplicationState : Type}
    (applicationSemantics :
      ApplicationTrace.Semantics Program ApplicationState)
    (statement : PublicStatement Program ApplicationState Digest) where
  initialSnapshot : Snapshot
  finalSnapshot : Snapshot
  segmentAccesses : List (List Access)
  globalClaims : List (VerifiedFullClaim schema Digest ChallengeField verify)
  statementIdentityCheck : statement.identity = config.statementIdentity
  chain :
    CheckedChain config schema bundleComponent verify 0 initialSnapshot 0 segmentAccesses
      globalClaims finalSnapshot statement.finalGlobalTimestamp
  globalFPrime :
    GlobalFPrime.Chain schema Digest verify
      { segmentIndex := 0
        globalTimestamp := 0
        memoryRoot := config.snapshotRoot initialSnapshot }
      globalClaims
      { segmentIndex := statement.segmentCount
        globalTimestamp := statement.finalGlobalTimestamp
        memoryRoot := config.snapshotRoot finalSnapshot }
      statement.segmentCount
  initialRootCheck :
    config.snapshotRoot initialSnapshot =
      config.snapshotRoot (Snapshot.ofImage statement.initialImage)
  applicationRows :
    ApplicationRowRun.CheckedCompletedRows applicationSemantics statement.program
      statement.initialApplicationState statement.expectedResult
      statement.segmentCount
  applicationMemoryCoverage :
    segmentAccesses =
      ApplicationRowRun.segmentAccessesOfRows statement.segmentCount
        statement.expectedResult.realApplicationRowCount applicationRows.rows
  finalMemoryAuthority :
    statement.expectedResult.finalMemoryRoot =
      config.snapshotRoot finalSnapshot

/-- Full model-level conclusion, including the lifecycle counts that the ideal
chain actually contains. -/
structure CertifiedExecution
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {Program ApplicationState : Type}
    {applicationSemantics :
      ApplicationTrace.Semantics Program ApplicationState}
    {statement : PublicStatement Program ApplicationState Digest}
    (acceptance :
      IdealAcceptV2 config schema bundleComponent verify applicationSemantics
        statement) : Prop where
  execution :
    HasSoundExecution applicationSemantics statement config.snapshotRoot
  freshClaimCount :
    acceptance.globalClaims.length = totalClaims statement.segmentCount
  augmentedInvocationCount :
    acceptance.globalClaims.length + 1 =
      totalClaims statement.segmentCount + 1
  lifecycle : CompleteSchedule acceptance.globalClaims.length

/-- Main ideal soundness theorem. It reconstructs the completed application
execution from local row transitions and reconstructs sequential memory from
the independent segment checks. It does not take a completed execution,
`AcceptanceReduction`, or `ExecutionWitness`. -/
theorem ideal_acceptance_implies_execution_or_failure
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {Program ApplicationState : Type}
    {applicationSemantics :
      ApplicationTrace.Semantics Program ApplicationState}
    {statement : PublicStatement Program ApplicationState Digest}
    {verify : FullVerifier schema Digest ChallengeField}
    (acceptance :
      IdealAcceptV2 config schema bundleComponent verify applicationSemantics
        statement) :
    Failure config ∨ CertifiedExecution acceptance := by
  rcases acceptance.chain.valid_or_failure with failure | validChain
  · exact Or.inl failure
  rcases snapshot_eq_or_collision acceptance.initialRootCheck with
    initialExact | collision
  · right
    rcases acceptance.applicationRows.completedExecution with
      ⟨applicationExecution, applicationRowsExact⟩
    have segmentCountExact :
        acceptance.segmentAccesses.length = statement.segmentCount := by
      calc
        acceptance.segmentAccesses.length =
            (ApplicationRowRun.segmentAccessesOfRows statement.segmentCount
              statement.expectedResult.realApplicationRowCount
              acceptance.applicationRows.rows).length :=
          congrArg List.length acceptance.applicationMemoryCoverage
        _ = statement.segmentCount :=
          ApplicationRowRun.segmentAccessesOfRows_length _ _ _
    have applicationMemoryCoverage :
        applicationExecution.CoversMemory acceptance.segmentAccesses := by
      unfold ApplicationTrace.CompletedExecution.CoversMemory
      calc
        acceptance.segmentAccesses =
            ApplicationRowRun.segmentAccessesOfRows statement.segmentCount
              statement.expectedResult.realApplicationRowCount
              acceptance.applicationRows.rows :=
          acceptance.applicationMemoryCoverage
        _ = ApplicationRowRun.segmentAccessesOfRows statement.segmentCount
              statement.expectedResult.realApplicationRowCount
              applicationExecution.rows := by
          rw [applicationRowsExact]
        _ = applicationExecution.segmentAccesses :=
          ApplicationRowRun.CheckedCompletedRows.segmentAccessesOfRows_execution
            applicationExecution
    have execution :
        HasSoundExecution applicationSemantics statement
          config.snapshotRoot := by
      refine
        ⟨acceptance.finalSnapshot, acceptance.segmentAccesses,
          applicationExecution, ?_, segmentCountExact,
          applicationMemoryCoverage,
          acceptance.finalMemoryAuthority⟩
      rw [← initialExact]
      exact validChain.executes
    have claimCountFromChain := acceptance.chain.globalClaimCount
    have freshClaimCount :
        acceptance.globalClaims.length =
          totalClaims statement.segmentCount := by
      calc
        acceptance.globalClaims.length =
            acceptance.segmentAccesses.length * claimsPerSegment :=
          claimCountFromChain
        _ = statement.segmentCount * claimsPerSegment := by
          rw [segmentCountExact]
        _ = totalClaims statement.segmentCount := rfl
    exact
      { execution := execution
        freshClaimCount := freshClaimCount
        augmentedInvocationCount := congrArg (fun count => count + 1)
          freshClaimCount
        lifecycle := acceptance.globalFPrime.completeDelayedSchedule
          acceptance.applicationRows.segmentCountPositive }
  · exact Or.inl (.snapshotRoot collision)

end Nightstream.Protocol.Nebula.IdealAcceptance
