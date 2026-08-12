import Nightstream.Protocol.NebulaV2.IdealAcceptance

/-!
Contract: constructive completeness boundary for the Nebula V2 ideal verifier.

Assurance tier: model-level.

Owns conversion of valid bounded semantic segments plus honest sequence and
F-prime primitive artifacts into `IdealAcceptV2`. Exact memory balance proves
the fingerprint close predicate; it is not accepted as a prover claim.

Does not prove completeness of NIFS, commitments, Poseidon2, generated rows,
Rust witness generation, the sampler, or the deployed terminal backend. Those
primitive witnesses are explicit inputs to this model-level construction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.IdealCompleteness

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationRowRun
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.IdealAcceptance
open Nightstream.Protocol.NebulaV2.IdealFingerprint
open Nightstream.Protocol.NebulaV2.IdealSequence
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.Memory
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.Soundness

/-- Honest F-prime primitive data before the semantic balance theorem is used.
The close predicate is `True`; `toEvidence` strengthens it to the concrete
fingerprint acceptance predicate proved from exact balance. -/
structure HonestFPrimeArtifacts
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (schema : FullClaim.Schema)
    (verify : FullVerifier schema Digest ChallengeField)
    (segmentIndex timestampIn timestampOut : Nat)
    {initial final : Snapshot} {accesses : List Access}
    (fingerprint : Check config.encode initial accesses final) where
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
      (fun products => products = ProductState.expected fingerprint)
      (.active active) claims (.closed closed)

namespace HonestFPrimeArtifacts

def toEvidence
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex timestampIn timestampOut : Nat}
    {initial final : Snapshot} {accesses : List Access}
    (fingerprint : Check config.encode initial accesses final)
    (artifacts : HonestFPrimeArtifacts config schema verify segmentIndex timestampIn
      timestampOut (initial := initial) (final := final)
      (accesses := accesses) fingerprint)
    (balanced : ProductState.Balanced (ProductState.expected fingerprint)) :
    FPrimeEvidence config schema verify segmentIndex timestampIn timestampOut
      fingerprint where
  active := artifacts.active
  closed := artifacts.closed
  claims := artifacts.claims
  productChunks := artifacts.productChunks
  productUpdates := artifacts.productUpdates
  productCoverage := artifacts.productCoverage
  stepZero := artifacts.stepZero
  activeSegment := artifacts.activeSegment
  activeTimestamp := artifacts.activeTimestamp
  segmentStartTimestamp := artifacts.segmentStartTimestamp
  activeAccessCount := artifacts.activeAccessCount
  segmentEndTimestamp := artifacts.segmentEndTimestamp
  activeMemoryRoot := artifacts.activeMemoryRoot
  openingProducts := artifacts.openingProducts
  openingHeaders := artifacts.openingHeaders
  closedSegment := artifacts.closedSegment
  closedTimestamp := artifacts.closedTimestamp
  closedMemoryRoot := artifacts.closedMemoryRoot
  run := artifacts.run.mono (by
    intro products productsExact
    simpa [productsExact] using balanced)

end HonestFPrimeArtifacts

/-- One semantically valid segment together with honest primitive artifacts.
This is a completeness input, not an ideal-verifier acceptance predicate. -/
structure HonestSegment
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
  valid : ValidSegment initial final timestampIn accesses timestampOut
  fingerprint : Check config.encode initial accesses final
  sequences : Checks config.chainRoot config.profile config.plan
  fprime : HonestFPrimeArtifacts config schema verify segmentIndex timestampIn
    timestampOut (initial := initial) (final := final) (accesses := accesses)
      fingerprint
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
    BundleAuthority bundleComponent
      (fprime.toEvidence fingerprint
        (ProductState.balanced_expected_of_memory_balance fingerprint
          valid.balanced))
      sequences

namespace HonestSegment

/-- Canonical closed carry at one semantic segment boundary. Both the local
honest segment and the global F-prime chain use this single definition. -/
def boundaryCarry
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest)
    (segmentIndex : Nat) (snapshot : Snapshot) (timestamp : Nat) :
    ClosedCarry Digest :=
  { segmentIndex := segmentIndex
    globalTimestamp := timestamp
    memoryRoot := config.snapshotRoot snapshot }

/-- Canonical two-lane chain headers whose root view is the verifier-owned
three-root header in `Config`. -/
def chainHeaders
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest) :
    ChainHeaders Digest :=
  { operations := config.chainHeader config.profile config.plan .operations
    memory := config.chainHeader config.profile config.plan .memory }

@[simp] theorem chainHeaders_roots
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest) :
    (chainHeaders config).roots = config.canonicalHeaders := rfl

/-- Exact challenge function used by the global F-prime opening. It reads the
segment index and timestamp from the authoritative closed carry. -/
def derive
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    (config : Config ChallengeField Profile Plan Commitment Digest) :
    ClosedCarry Digest → Roots Digest → Nat →
      ProductState.Challenges ChallengeField :=
  fun closed roots activeAccessCount =>
    config.deriveChallenge closed.segmentIndex closed.globalTimestamp
      activeAccessCount roots

def toSegmentCheck
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut) :
    SegmentCheck config schema bundleComponent verify segmentIndex initial timestampIn accesses
      final timestampOut := by
  have productsBalanced :=
    ProductState.balanced_expected_of_memory_balance honest.fingerprint
      honest.valid.balanced
  exact
    { accessBound := honest.accessBound
      initialValid := honest.valid.initialValid
      ordered := honest.valid.ordered
      fingerprint := honest.fingerprint
      sequences := honest.sequences
      fprime := honest.fprime.toEvidence honest.fingerprint productsBalanced
      operationsAuthority := honest.operationsAuthority
      initialSnapshotAuthority := honest.initialSnapshotAuthority
      finalSnapshotAuthority := honest.finalSnapshotAuthority
      dPreMatches := honest.dPreMatches
      challengeDerived := honest.challengeDerived
      fingerprintChallenges := honest.fingerprintChallenges
      bundleAuthority := honest.bundleAuthority }

/-- The verified local F-prime run already proves that its active opening is
well formed. Completeness does not need a second range assumption. -/
theorem activeWellFormed
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut) :
    honest.fprime.active.WellFormed :=
  honest.fprime.run.initialActiveWellFormed

/-- The local honest close value is exactly the canonical next segment
boundary. -/
theorem closedExact
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut) :
    honest.fprime.closed =
      boundaryCarry config (segmentIndex + 1) final timestampOut := by
  apply ClosedCarry.ext
  · exact honest.fprime.closedSegment
  · exact honest.fprime.closedTimestamp
  · exact honest.fprime.closedMemoryRoot

/-- The segment access bound is strictly below the 17-bit active-count limit. -/
theorem activeCountInRange
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut) :
    accesses.length < operationCountLimit := by
  have capacityBelowLimit : 63 * 1088 < operationCountLimit := by
    norm_num [operationCountLimit, operationCountBits]
  exact lt_of_le_of_lt honest.accessBound capacityBelowLimit

/-- The end timestamp bound is recovered from the verified active carry and
the semantic ordered-access schedule. -/
theorem endTimestampInRange
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut) :
    timestampIn + accesses.length < timestampLimit := by
  rcases honest.activeWellFormed with
    ⟨_indexBound, _countBound, _endExact, endBound, _startLe, _globalLe⟩
  calc
    timestampIn + accesses.length = timestampOut :=
      honest.valid.timestampOut_eq.symm
    _ = honest.fprime.active.segmentEndTimestamp :=
      honest.fprime.segmentEndTimestamp.symm
    _ < timestampLimit := endBound

/-- A bounded segment index and the verified active carry give the exact
closed-state opening precondition. -/
theorem canOpen
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut)
    (segmentIndexBound : segmentIndex < maximumSegments) :
    (boundaryCarry config segmentIndex initial timestampIn).CanOpen := by
  constructor
  · exact segmentIndexBound
  · exact Nat.lt_of_le_of_lt (Nat.le_add_right timestampIn accesses.length)
      honest.endTimestampInRange

/-- The honest local fields determine the exact canonical open transition.
No global F-prime chain is an input to this theorem. -/
theorem opened
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut)
    (segmentIndexBound : segmentIndex < maximumSegments) :
    openSegment (derive config) (chainHeaders config)
        honest.sequences.committedRoots accesses.length
        (boundaryCarry config segmentIndex initial timestampIn)
        (honest.canOpen segmentIndexBound) honest.activeCountInRange
        honest.endTimestampInRange =
      .active honest.fprime.active := by
  apply congrArg Carry.active
  apply ActiveCarry.ext
  · exact honest.fprime.activeSegment.symm
  · apply Fin.ext
    exact honest.fprime.stepZero.symm
  · exact honest.fprime.activeTimestamp.symm
  · exact honest.fprime.segmentStartTimestamp.symm
  · exact honest.fprime.activeAccessCount.symm
  · exact honest.valid.timestampOut_eq.symm.trans
      honest.fprime.segmentEndTimestamp.symm
  · exact honest.challengeDerived.symm
  · exact honest.fprime.openingProducts.symm
  · exact honest.dPreMatches.symm
  · exact honest.fprime.openingHeaders.symm
  · exact honest.fprime.activeMemoryRoot.symm

/-- One honest semantic segment constructs the exact global F-prime segment
run. Product balance is derived from exact memory balance. -/
noncomputable def toGlobalRun
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
    (honest : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
      accesses final timestampOut)
    (segmentIndexBound : segmentIndex < maximumSegments) :
    GlobalFPrime.SegmentRun schema Digest verify
      (boundaryCarry config segmentIndex initial timestampIn) where
  derive := derive config
  headers := chainHeaders config
  precommit := honest.sequences.committedRoots
  activeAccessCount := accesses.length
  canOpen := honest.canOpen segmentIndexBound
  activeCountInRange := honest.activeCountInRange
  endTimestampInRange := honest.endTimestampInRange
  active := honest.fprime.active
  after := boundaryCarry config (segmentIndex + 1) final timestampOut
  claims := honest.fprime.claims
  opened := honest.opened segmentIndexBound
  consumed := by
    rw [← honest.closedExact]
    exact (honest.fprime.toEvidence honest.fingerprint
      (ProductState.balanced_expected_of_memory_balance honest.fingerprint
        honest.valid.balanced)).run

end HonestSegment

/-- Valid segments share the exact boundary snapshot in the completeness
direction. The claim list is still assembled from the supplied honest F-prime
runs, so claim accounting remains checked. -/
inductive HonestChain
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
      HonestChain config schema bundleComponent verify segmentIndex snapshot timestamp
        [] [] snapshot timestamp
  | cons
      {segmentIndex : Nat}
      {initial middle final : Snapshot}
      {timestampIn timestampMiddle timestampOut : Nat}
      {accesses : List Access} {rest : List (List Access)}
      {tailClaims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
      (head : HonestSegment config schema bundleComponent verify segmentIndex initial timestampIn
        accesses middle timestampMiddle)
      (tail : HonestChain config schema bundleComponent verify (segmentIndex + 1) middle
        timestampMiddle rest tailClaims final timestampOut) :
      HonestChain config schema bundleComponent verify segmentIndex initial timestampIn
        (accesses :: rest) (head.fprime.claims ++ tailClaims)
        final timestampOut

namespace HonestChain

def toCheckedChain
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {segments : List (List Access)}
    {claims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
    (honest : HonestChain config schema bundleComponent verify segmentIndex initial timestampIn
      segments claims final timestampOut) :
    CheckedChain config schema bundleComponent verify segmentIndex initial timestampIn segments
      claims final timestampOut := by
  induction honest with
  | nil => exact .nil _ _ _
  | cons head _ inductionHypothesis =>
      exact .cons head.toSegmentCheck rfl inductionHypothesis

/-- The ordered honest segment chain constructs the complete global F-prime
chain. The only extra premise is the public lifetime segment bound. The
complete global chain is therefore not a completeness assumption. -/
noncomputable def toGlobalFPrime
    {ChallengeField Profile Plan Commitment Digest : Type}
    [Field ChallengeField]
    {schema : FullClaim.Schema}
    {config : Config ChallengeField Profile Plan Commitment Digest}
    {bundleComponent :
      schema.CommitmentBundle → CommitmentBundle.Component → Commitment}
    {verify : FullVerifier schema Digest ChallengeField}
    {segmentIndex : Nat}
    {initial final : Snapshot} {timestampIn timestampOut : Nat}
    {segments : List (List Access)}
    {claims : List (VerifiedFullClaim schema Digest ChallengeField verify)}
    (honest : HonestChain config schema bundleComponent verify segmentIndex initial timestampIn
      segments claims final timestampOut)
    (withinLifetime : segmentIndex + segments.length ≤ maximumSegments) :
    GlobalFPrime.Chain schema Digest verify
      (HonestSegment.boundaryCarry config segmentIndex initial timestampIn)
      claims
      (HonestSegment.boundaryCarry config (segmentIndex + segments.length)
        final timestampOut)
      segments.length := by
  induction honest with
  | nil =>
      exact .nil _
  | @cons segmentIndex initial middle final timestampIn timestampMiddle
      timestampOut accesses rest tailClaims head tail inductionHypothesis =>
      have headBound : segmentIndex < maximumSegments := by
        simp only [List.length_cons] at withinLifetime
        omega
      have tailBound : segmentIndex + 1 + rest.length ≤ maximumSegments := by
        simp only [List.length_cons] at withinLifetime
        omega
      have chained := GlobalFPrime.Chain.cons
        (head.toGlobalRun headBound)
        (inductionHypothesis tailBound)
      simpa only [List.length_cons, Nat.add_assoc, Nat.add_comm,
        Nat.add_left_comm] using chained

end HonestChain

/-- A valid bounded execution and explicit honest primitive artifacts. This is
the exact premise of model-level ideal completeness. -/
structure CompletenessInput
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
  statementIdentityAuthority :
    statement.identity = config.statementIdentity
  chain :
    HonestChain config schema bundleComponent verify 0 initialSnapshot 0 segmentAccesses
      globalClaims finalSnapshot statement.finalGlobalTimestamp
  initialAuthority :
    initialSnapshot = Snapshot.ofImage statement.initialImage
  segmentCountExact : segmentAccesses.length = statement.segmentCount
  applicationExecution :
    ApplicationTrace.CompletedExecution applicationSemantics statement.program
      statement.initialApplicationState statement.expectedResult
      statement.segmentCount
  applicationMemoryCoverage :
    applicationExecution.CoversMemory segmentAccesses
  finalMemoryAuthority :
    statement.expectedResult.finalMemoryRoot =
      config.snapshotRoot finalSnapshot

namespace CompletenessInput

/-- The full lifetime F-prime chain is derived from the ordered honest
segments and the segment bound in the completed application execution. It is
not a field of `CompletenessInput`. -/
noncomputable def globalFPrime
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
    (input : CompletenessInput config schema bundleComponent verify
      applicationSemantics statement) :
    GlobalFPrime.Chain schema Digest verify
      { segmentIndex := 0
        globalTimestamp := 0
        memoryRoot := config.snapshotRoot input.initialSnapshot }
      input.globalClaims
      { segmentIndex := statement.segmentCount
        globalTimestamp := statement.finalGlobalTimestamp
        memoryRoot := config.snapshotRoot input.finalSnapshot }
      statement.segmentCount := by
  have withinLifetime :
      0 + input.segmentAccesses.length ≤ maximumSegments := by
    rw [Nat.zero_add, input.segmentCountExact]
    exact input.applicationExecution.segmentCountBound
  simpa [HonestSegment.boundaryCarry, input.segmentCountExact] using
    input.chain.toGlobalFPrime withinLifetime

end CompletenessInput

/-- Constructive ideal completeness. It consumes valid semantic segments and
honest primitive artifacts and produces the exact raw ideal acceptance type. -/
def valid_execution_with_honest_artifacts_is_accepted
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
    (input : CompletenessInput config schema bundleComponent verify
      applicationSemantics statement) :
    IdealAcceptV2 config schema bundleComponent verify applicationSemantics
      statement :=
  { initialSnapshot := input.initialSnapshot
    finalSnapshot := input.finalSnapshot
    segmentAccesses := input.segmentAccesses
    globalClaims := input.globalClaims
    statementIdentityCheck := input.statementIdentityAuthority
    chain := input.chain.toCheckedChain
    globalFPrime := input.globalFPrime
    initialRootCheck := congrArg config.snapshotRoot input.initialAuthority
    applicationRows :=
      ApplicationRowRun.CheckedCompletedRows.ofCompletedExecution
        input.applicationExecution
    applicationMemoryCoverage := by
      rw [input.applicationMemoryCoverage]
      simpa [ApplicationRowRun.CheckedCompletedRows.ofCompletedExecution] using
        (ApplicationRowRun.CheckedCompletedRows.segmentAccessesOfRows_execution
          input.applicationExecution).symm
    finalMemoryAuthority := input.finalMemoryAuthority }

end Nightstream.Protocol.NebulaV2.IdealCompleteness
