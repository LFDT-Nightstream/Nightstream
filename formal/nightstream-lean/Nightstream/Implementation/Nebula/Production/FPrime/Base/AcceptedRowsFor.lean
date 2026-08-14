import Nightstream.Implementation.Nebula.FPrime.Manifest.BaseStateAuthority
import Nightstream.Implementation.Nebula.Production.FPrime.Base.ChallengeAuthoritySoundFor
import Nightstream.Implementation.Nebula.Production.FPrime.Base.CurrentMemoryRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Base.InvocationFor

/-!
Contract: row-derived segment opening for the production paper base branch.

The package contains one base-manifest artifact, one assignment, the typed
placements for that assignment, and satisfaction of the complete manifest.
It derives the canonical initial carry, the exact segment-open transition,
and the semantic outgoing active carry. It does not accept a semantic opening
or an exact base invocation as an input.

Application lowering, fresh-claim construction, accumulator arithmetic, and
control refinement remain opaque row families. The base challenge-authority
family has an exact contained row program and a semantic soundness theorem.

Assurance tier: base memory/state row extraction.

Emits constraints: no new rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.BaseManifestSchema
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationBatch
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The row-facing base package. No semantic transition is a field. -/
structure Accepted (widths : FullClaimEnvelope.CompilerWidths) where
  artifact : BaseManifestSchema.Artifact widths
  assignment : Nat -> Nat
  call : BaseManifestStateAuthority.Call artifact assignment
  satisfies : Satisfies artifact.programRows assignment

namespace Accepted

/-- The satisfying base manifest fixes the actual F-prime input iteration to
zero. This fact is derived from a row, not supplied by the invocation. -/
theorem inputIterationZero
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) :
    accepted.assignment
        accepted.artifact.layouts.baseIteration.iterationColumn = 0 :=
  FPrimeIterationInputRows.sound accepted.call.canonicalAssignment
    accepted.call.one
    (accepted.artifact.baseIteration_satisfied accepted.satisfies)

def initialCanonical
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) :
    accepted.call.initialValue.Canonical accepted.call.headers :=
  (accepted.call.initialCarryColumnsMatch accepted.satisfies).parserCanonical

def outgoingCanonical
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) :
    accepted.call.outgoingValue.Canonical accepted.call.headers :=
  (accepted.call.outgoingCarryColumnsMatch accepted.satisfies).parserCanonical

/-- The initial wire is the unique canonical verifier-authoritative base
carry. -/
theorem initialValueExact
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) :
    accepted.call.initialValue =
      InitialMemoryCarryRows.expectedValue accepted.call.headers
        accepted.call.initialMemoryRoot :=
  (accepted.call.initialExact accepted.satisfies).value_eq_expected
    accepted.initialCanonical

/-- The semantic closed state used by the paper base branch is the state
decoded by the base rows. -/
theorem initialClosedExact
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) :
    MemoryOpenSegmentSound.closedOfWire accepted.call.initialValue =
      ProductionPaperBaseInvocationFor.initialClosed
        accepted.call.initialMemoryRoot := by
  rw [accepted.initialValueExact]
  rfl

/-- Named result of the local segment-open rows. All fields below are
conclusions of `BaseManifestStateAuthority.Call.opensExactInitialCarry`. -/
structure Opened
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) : Prop where
  canOpen :
    (MemoryOpenSegmentSound.closedOfWire
      accepted.call.initialValue).CanOpen
  activeCountInRange :
    accepted.call.outgoingValue.segmentActiveAccessCount < operationCountLimit
  endTimestampInRange :
    (MemoryOpenSegmentSound.closedOfWire
        accepted.call.initialValue).globalTimestamp +
      accepted.call.outgoingValue.segmentActiveAccessCount < timestampLimit
  stepBound :
    accepted.call.outgoingValue.stepIndex < Lifecycle.claimsPerSegment
  beforeClosed : accepted.call.initialValue.phase = .closed
  afterActive : accepted.call.outgoingValue.phase = .active
  transition :
    Carry.active
        (MemoryOpenSegmentSound.activeOfWire accepted.call.outgoingValue
          stepBound) =
      MemoryOpenSegment.openCarryFor accepted.artifact.profile
        accepted.call.openingAuthority
        accepted.call.headers accepted.call.outgoingValue.dPre
        accepted.call.outgoingValue.segmentActiveAccessCount
        (MemoryOpenSegmentSound.closedOfWire accepted.call.initialValue)
        canOpen activeCountInRange endTimestampInRange

theorem openedExists
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) : Nonempty (Opened accepted) := by
  rcases accepted.call.opensExactInitialCarry accepted.satisfies with
    ⟨canOpen, activeCount, endTimestamp, stepBound, beforeClosed,
      afterActive, transition⟩
  exact ⟨
    { canOpen := canOpen
      activeCountInRange := activeCount
      endTimestampInRange := endTimestamp
      stepBound := stepBound
      beforeClosed := beforeClosed
      afterActive := afterActive
      transition := transition }⟩

/-- Deterministic proof projection from the row-derived existential. -/
noncomputable def opened
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) : Opened accepted :=
  Classical.choice accepted.openedExists

/-- The exact paper-base opening reconstructed from the row values. -/
noncomputable def opening
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths) : ProductionPaperBaseInvocationFor.Opening :=
  { initialMemoryRoot := accepted.call.initialMemoryRoot
    authority := accepted.call.openingAuthority
    precommit := accepted.call.outgoingValue.dPre
    activeAccessCount :=
      accepted.call.outgoingValue.segmentActiveAccessCount
    activeCountInRange := accepted.opened.activeCountInRange
    endTimestampInRange := by
      have bound := accepted.opened.endTimestampInRange
      rw [accepted.initialClosedExact] at bound
      simpa [ProductionPaperBaseInvocationFor.initialClosed] using bound }

/-- `openCarry` is independent of the proof terms used to establish its range
conditions. This lemma transports an opening across equality of the closed
carry without eliminating an equality through the dependent `CanOpen` field. -/
private theorem openCarry_congr_closed
    {profile : Profile.Identity}
    {headers : ChainHeaders Digest.Value}
    {authority : MemoryOpenSegment.Authority}
    {precommit : Roots Digest.Value}
    {activeAccessCount : Nat}
    {left right : ClosedCarry Digest.Value}
    (same : left = right)
    (leftCanOpen : left.CanOpen)
    (rightCanOpen : right.CanOpen)
    (leftActive : activeAccessCount < operationCountLimit)
    (rightActive : activeAccessCount < operationCountLimit)
    (leftEnd : left.globalTimestamp + activeAccessCount < timestampLimit)
    (rightEnd : right.globalTimestamp + activeAccessCount < timestampLimit) :
    MemoryOpenSegment.openCarryFor profile authority headers precommit
        activeAccessCount
        left leftCanOpen leftActive leftEnd =
      MemoryOpenSegment.openCarryFor profile authority headers precommit
        activeAccessCount
        right rightCanOpen rightActive rightEnd := by
  cases same
  rfl

/-- The row-decoded outgoing active carry is exactly the paper-base opening,
including the post-precommit challenge and all-one product state. -/
theorem activeOfWireExact
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths)
    (candidate : ProductionProfileCandidates.Id)
    (profileExact : accepted.artifact.profile =
      ProductionProfileCandidates.identity candidate) :
    MemoryOpenSegmentSound.activeOfWire accepted.call.outgoingValue
        accepted.opened.stepBound =
      (accepted.opening.activeFor candidate accepted.call.headers) := by
  have rowTransition := accepted.opened.transition
  rw [profileExact] at rowTransition
  have carryTransport :=
    openCarry_congr_closed
      (profile := ProductionProfileCandidates.identity candidate)
      (headers := accepted.call.headers)
      (authority := accepted.call.openingAuthority)
      (precommit := accepted.call.outgoingValue.dPre)
      accepted.initialClosedExact
      accepted.opened.canOpen
      (ProductionPaperBaseInvocationFor.initialClosed_canOpen
        accepted.call.initialMemoryRoot)
      accepted.opened.activeCountInRange
      accepted.opening.activeCountInRange
      accepted.opened.endTimestampInRange
      accepted.opening.initialEndTimestampInRange
  have semanticTransition :=
    accepted.opening.open_exact_for candidate accepted.call.headers
  have activeEquality :
      Carry.active
          (MemoryOpenSegmentSound.activeOfWire accepted.call.outgoingValue
            accepted.opened.stepBound) =
        Carry.active (accepted.opening.activeFor candidate
          accepted.call.headers) :=
    rowTransition.trans (carryTransport.trans semanticTransition)
  exact Carry.active.inj activeEquality

/-- The public carry parser and the segment-open rows select the same semantic
active carry used by the paper F-prime base node. -/
theorem outgoingSemanticExact
    {widths : FullClaimEnvelope.CompilerWidths}
    (accepted : Accepted widths)
    (candidate : ProductionProfileCandidates.Id)
    (profileExact : accepted.artifact.profile =
      ProductionProfileCandidates.identity candidate) :
    MemoryCarryParser.semanticCarry accepted.call.outgoingValue
        accepted.outgoingCanonical.stepIndex =
      .active (accepted.opening.activeFor candidate
        accepted.call.headers) := by
  rw [MemoryCarryParser.semanticCarry, accepted.opened.afterActive]
  exact congrArg Carry.active
    (accepted.activeOfWireExact candidate profileExact)

end Accepted

/-! ## Complete base invocation from shared rows -/

/-- The non-row inputs that remain after the base outgoing carry and fixed
first checked-memory batch use one assignment. Application lowering and
fresh-claim construction remain named compiler boundaries because their
base-manifest row families are still opaque. -/
structure Supplement
    {Program : Type} (candidate : ProductionProfileCandidates.Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {widths : FullClaimEnvelope.CompilerWidths}
    (statementId : ProductPoseidon2.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact)
    (headers : ChainHeaders Digest.Value)
    (statement : WasmStatement.ProductionStatement Program)
    (base : Accepted widths)
    (machine : WasmState.Machine Program)
    (after : WasmState.AppStateVector)
    (batch : ApplicationBatch.Batch candidate machine statement.base.program
      statement.base.initialApplicationState after)
    (memoryAuthority : ProductionBaseCurrentMemoryRowsFor.Authority candidate
      base.artifact)
    (challengeProgram :
      ProductionBaseChallengeAuthorityRowsFor.Program candidate rowVariables)
    (assignment : ProductionPaperBaseInvocationFor.FreshAssignment rowVariables
      logicalWidth publicFits) : Prop where
  headersExact : base.call.headers = headers
  statementCanonical :
    (WasmPublicStatementEncoding.PublicImage.ofStatement statement).DecodesFor
      (ProductionProfileCandidates.identity candidate) statement
  /-- Generated-program containment and typed source placement. These facts
  do not assume a digest, challenge, authority equality, or F-prime result. -/
  challengeRowsMatched : challengeProgram.MatchesArtifact base.artifact
  challengeStatementIdExact : challengeProgram.statementId = statementId
  challengeStatementIdentityExact :
    challengeProgram.statementIdentity = statement.base.identity
  initialStatePlaced : ProductionSuccessorStateBindingRowsFor.Placed
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) challengeProgram.initialLayout base.assignment
    (ProductionPaperBaseInvocationFor.initialState
      (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement
      base.opening.initialMemoryRoot)
  successorStatePlaced : ProductionSuccessorStateBindingRowsFor.Placed
    (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
      publicFits) challengeProgram.successorLayout base.assignment
    (ProductionPaperBaseInvocationFor.state
      (rowVariables := rowVariables) (logicalWidth := logicalWidth)
      (publicFits := publicFits) candidate headers statement base.opening batch)
  applicationMatched :
    ProductionApplicationBatchBridge.Matches
      (memoryAuthority.result base.call headers headersExact base.satisfies)
      batch
  freshRelation : ProductionFreshClaimProducerFor.FreshRelationWitnessForRows
    statementId config artifact relationAuthority
    (ProductionPaperBaseInvocationFor.state (rowVariables := rowVariables)
      (logicalWidth := logicalWidth) (publicFits := publicFits) candidate headers
      statement base.opening batch)
    (memoryAuthority.result base.call headers headersExact
      base.satisfies).suffixBatch assignment base.assignment

namespace Supplement

/-- The current memory result is a projection of the fixed base rows. It is
not a field of the supplement. -/
@[irreducible] noncomputable def memoryResult
    {Program : Type} {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {widths : FullClaimEnvelope.CompilerWidths}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {headers : ChainHeaders Digest.Value}
    {statement : WasmStatement.ProductionStatement Program}
    {base : Accepted widths}
    {machine : WasmState.Machine Program}
    {after : WasmState.AppStateVector}
    {batch : ApplicationBatch.Batch candidate machine statement.base.program
      statement.base.initialApplicationState after}
    {memoryAuthority : ProductionBaseCurrentMemoryRowsFor.Authority candidate
      base.artifact}
    {challengeProgram :
      ProductionBaseChallengeAuthorityRowsFor.Program candidate rowVariables}
    {assignment : ProductionPaperBaseInvocationFor.FreshAssignment rowVariables
      logicalWidth publicFits}
    (supplement : Supplement candidate statementId config artifact
      relationAuthority headers statement base machine after batch
      memoryAuthority challengeProgram assignment) :
    ProductionMemoryCheckedBatchRows.Result memoryAuthority.layout
      base.assignment headers :=
  memoryAuthority.result base.call headers supplement.headersExact
    base.satisfies

private theorem semanticCarry_congr_value
    {left right : MemoryCarryCodec.Value}
    (same : left = right)
    (leftBound : left.stepIndex < Lifecycle.claimsPerSegment)
    (rightBound : right.stepIndex < Lifecycle.claimsPerSegment) :
    MemoryCarryParser.semanticCarry left leftBound =
      MemoryCarryParser.semanticCarry right rightBound := by
  cases same
  rfl

/-- Shared assignment and carry columns force the base output value to equal
the first checked-batch boundary. -/
theorem outgoingValue_eq_firstBoundary
    {Program : Type} {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {widths : FullClaimEnvelope.CompilerWidths}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {headers : ChainHeaders Digest.Value}
    {statement : WasmStatement.ProductionStatement Program}
    {base : Accepted widths}
    {machine : WasmState.Machine Program}
    {after : WasmState.AppStateVector}
    {batch : ApplicationBatch.Batch candidate machine statement.base.program
      statement.base.initialApplicationState after}
    {memoryAuthority : ProductionBaseCurrentMemoryRowsFor.Authority candidate
      base.artifact}
    {challengeProgram :
      ProductionBaseChallengeAuthorityRowsFor.Program candidate rowVariables}
    {assignment : ProductionPaperBaseInvocationFor.FreshAssignment rowVariables
      logicalWidth publicFits}
    (supplement : Supplement candidate statementId config artifact
      relationAuthority headers
      statement base machine after batch memoryAuthority challengeProgram
      assignment) :
    base.call.outgoingValue = supplement.memoryResult.boundary 0 := by
  unfold memoryResult
  exact memoryAuthority.outgoingValue_eq_firstBoundary base.call headers
    supplement.headersExact base.satisfies

/-- The first checked-batch semantic state is derived from the satisfying base
opening rows and the shared physical boundary. -/
theorem memoryStartsAt
    {Program : Type} {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {widths : FullClaimEnvelope.CompilerWidths}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {headers : ChainHeaders Digest.Value}
    {statement : WasmStatement.ProductionStatement Program}
    {base : Accepted widths}
    {machine : WasmState.Machine Program}
    {after : WasmState.AppStateVector}
    {batch : ApplicationBatch.Batch candidate machine statement.base.program
      statement.base.initialApplicationState after}
    {memoryAuthority : ProductionBaseCurrentMemoryRowsFor.Authority candidate
      base.artifact}
    {challengeProgram :
      ProductionBaseChallengeAuthorityRowsFor.Program candidate rowVariables}
    {assignment : ProductionPaperBaseInvocationFor.FreshAssignment rowVariables
      logicalWidth publicFits}
    (supplement : Supplement candidate statementId config artifact
      relationAuthority headers
      statement base machine after batch memoryAuthority challengeProgram
      assignment)
    (baseArtifactProfileExact :
      base.artifact.profile = ProductionProfileCandidates.identity candidate) :
    supplement.memoryResult.semantic 0 =
      .active (base.opening.activeFor candidate headers) := by
  calc
    supplement.memoryResult.semantic 0 =
        MemoryCarryParser.semanticCarry (supplement.memoryResult.boundary 0)
          (supplement.memoryResult.boundaryParsed 0).parserCanonical.stepIndex :=
      supplement.memoryResult.semanticExact 0
    _ = MemoryCarryParser.semanticCarry base.call.outgoingValue
        base.outgoingCanonical.stepIndex :=
      semanticCarry_congr_value
        supplement.outgoingValue_eq_firstBoundary.symm _ _
    _ = .active (base.opening.activeFor candidate base.call.headers) :=
      base.outgoingSemanticExact candidate baseArtifactProfileExact
    _ = .active (base.opening.activeFor candidate headers) := by
      rw [supplement.headersExact]

/-- The base challenge authority is a conclusion of the selected rows and
the two typed source placements. The desired authority equality is not a
field of `Supplement`. -/
theorem challengeAuthorityExact
    {Program : Type} {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {widths : FullClaimEnvelope.CompilerWidths}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {headers : ChainHeaders Digest.Value}
    {statement : WasmStatement.ProductionStatement Program}
    {base : Accepted widths}
    {machine : WasmState.Machine Program}
    {after : WasmState.AppStateVector}
    {batch : ApplicationBatch.Batch candidate machine statement.base.program
      statement.base.initialApplicationState after}
    {memoryAuthority : ProductionBaseCurrentMemoryRowsFor.Authority candidate
      base.artifact}
    {challengeProgram :
      ProductionBaseChallengeAuthorityRowsFor.Program candidate rowVariables}
    {assignment : ProductionPaperBaseInvocationFor.FreshAssignment rowVariables
      logicalWidth publicFits}
    (supplement : Supplement candidate statementId config artifact
      relationAuthority headers statement base machine after batch
      memoryAuthority challengeProgram assignment) :
    base.opening.authority =
      ProductionPaperBaseInvocationFor.challengeAuthority
        (rowVariables := rowVariables) (logicalWidth := logicalWidth)
        (publicFits := publicFits) candidate statementId headers statement
        base.opening batch := by
  have challengeRows : Satisfies challengeProgram.rows base.assignment :=
    challengeProgram.satisfies_of_matchesArtifact
      supplement.challengeRowsMatched
      base.satisfies
  have derived :=
    challengeProgram.rows_imply_openingAuthorityPlaced
      supplement.initialStatePlaced supplement.successorStatePlaced
      base.call.canonicalAssignment base.call.one challengeRows
  have claimed : MemoryOpenSegmentSound.AuthorityPlaced
      challengeProgram.openingLayout base.assignment
      base.opening.authority := by
    rw [supplement.challengeRowsMatched.openingExact]
    exact base.call.openingAuthorityPlaced
  have authorityEqual : base.opening.authority =
      challengeProgram.openingAuthority
        (ProductionPaperBaseInvocationFor.initialState
          (rowVariables := rowVariables) (logicalWidth := logicalWidth)
          (publicFits := publicFits) candidate headers statement
          base.opening.initialMemoryRoot)
        (ProductionPaperBaseInvocationFor.state
          (rowVariables := rowVariables) (logicalWidth := logicalWidth)
          (publicFits := publicFits) candidate headers statement base.opening
          batch) :=
    MemoryOpenSegmentSound.AuthorityPlaced.unique claimed derived
  rw [authorityEqual]
  simp only [ProductionBaseChallengeAuthorityRowsFor.Program.openingAuthority,
    supplement.challengeStatementIdExact,
    supplement.challengeStatementIdentityExact,
    ProductionBaseChallengeAuthorityRowsFor.canonicalDigestValue,
    ProductionPaperBaseInvocationFor.challengeAuthority,
    ProductionPaperBaseInvocationFor.digestValue]

/-- Complete local base evidence. Its memory-start fact is reconstructed from
shared satisfying rows and is not supplied as a semantic premise. -/
noncomputable def evidence
    {Program : Type} {candidate : ProductionProfileCandidates.Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {widths : FullClaimEnvelope.CompilerWidths}
    {statementId : ProductPoseidon2.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
      publicFits artifact}
    {headers : ChainHeaders Digest.Value}
    {statement : WasmStatement.ProductionStatement Program}
    {base : Accepted widths}
    {machine : WasmState.Machine Program}
    {after : WasmState.AppStateVector}
    {batch : ApplicationBatch.Batch candidate machine statement.base.program
      statement.base.initialApplicationState after}
    {memoryAuthority : ProductionBaseCurrentMemoryRowsFor.Authority candidate
      base.artifact}
    {challengeProgram :
      ProductionBaseChallengeAuthorityRowsFor.Program candidate rowVariables}
    {assignment : ProductionPaperBaseInvocationFor.FreshAssignment rowVariables
      logicalWidth publicFits}
    (supplement : Supplement candidate statementId config artifact
      relationAuthority headers
      statement base machine after batch memoryAuthority challengeProgram
      assignment)
    (baseArtifactProfileExact :
      base.artifact.profile = ProductionProfileCandidates.identity candidate) :
    ProductionPaperBaseInvocationFor.Evidence candidate statementId config
      artifact relationAuthority headers statement base.opening machine after batch
      supplement.memoryResult assignment where
  statementCanonical := supplement.statementCanonical
  challengeAuthorityExact := supplement.challengeAuthorityExact
  memoryStartsAt := supplement.memoryStartsAt baseArtifactProfileExact
  applicationMatched := by
    simpa [memoryResult] using supplement.applicationMatched
  freshRelation := by
    simpa [memoryResult] using supplement.freshRelation

end Supplement

end Nightstream.Implementation.Nebula.ProductionPaperBaseAcceptedRowsFor
