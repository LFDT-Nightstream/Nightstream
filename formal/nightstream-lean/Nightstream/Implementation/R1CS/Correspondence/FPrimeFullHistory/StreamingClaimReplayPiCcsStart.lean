import Nightstream.Implementation.Nebula.Production.NIFS.PiCCS.TypedBridgeFor
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCSAuthority
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplaySequence

/-!
Contract: exact handoff from the 86 claim-replay phases to PiCCS start.

Owns equality between the replay target and the selected paper-NIFS public
state, the exact production PiCCS start continuation, and the reduction from
an accepted claim sequence to the canonical claim frame and that start state.

Does not own PiCCS-start rows, the 26 round rows, PiCCS finish, selector rows,
the other F-prime phases, collision resistance, or recursive integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsStart

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.Nebula.ProductionProductPiCcsTypedBridgeFor
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsAuthority
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev ProductionFullShape (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape 26 logicalWidth publicFits

def productionFullShapeContract
    (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    ProductNifsCodec.FullShapeContractFor 26
      (ProductionFullShape logicalWidth publicFits) :=
  ProductPaperAlgebraFor.fullShapeContract 26 logicalWidth publicFits

/-- The state checked at the end of claim replay is exactly the public-input
state used by the selected paper-NIFS key. -/
theorem selected_public_state_eq_bindingState
    (candidate : Id)
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config 26 logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact 26 logicalWidth
      publicFits)
    (value : Value candidate (ProductionFullShape logicalWidth publicFits)) :
    (paperKey candidate statementId config artifact).publicInputState
        value.recursiveState
        (freshOfValue (productionFullShapeContract logicalWidth publicFits)
          value) =
      bindingState statementId 9 value := by
  rw [paperKey_publicInputState]
  unfold ProductionProductConcreteNifsKey.publicAbsorber bindingState
    ProductionProductNifsPublicTranscript.publicState
  rw [ProductionProductNifsPublicTranscript.publicNifsFields_of_value
    (productionFullShapeContract logicalWidth publicFits)]

/-- The exact continuation immediately before the first prover SumCheck
message. Alpha and gamma are derived after the complete selected-key public
state and statement have been absorbed. -/
noncomputable def piCcsStartContinuation
    (candidate : Id)
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config 26 logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact 26 logicalWidth
      publicFits)
    (value : Value candidate (ProductionFullShape logicalWidth publicFits)) :
    Continuation K ProductPoseidon2.State :=
  let contract := productionFullShapeContract logicalWidth publicFits
  let key := paperKey candidate statementId config artifact
  let fresh := freshOfValue contract value
  let input := exactVerifierInput candidate statementId config artifact
    value.recursiveState fresh
  let statement : ProtocolVerifier.Statement K ProductPoseidon2.State
      (ProductNifsCodec.shapeFor 26) :=
    { priorState := key.publicInputState value.recursiveState fresh
      input }
  let pre := FiatShamir.derivePreSumcheck key.oracle.transcript statement
  {
    transcriptState := pre.state
    current := input.initial key.extensionOps pre.gamma
    point := []
    cursor := 0
  }

/-- Semantic contract of the PiCCS-start phase. The prior replay state must
cover the complete canonical claim frame and equal the selected-key public
state. The successor is computed, not supplied as authority. -/
def PiCcsStartRelation
    (candidate : Id)
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config 26 logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact 26 logicalWidth
      publicFits)
    (value : Value candidate (ProductionFullShape logicalWidth publicFits))
    (before : ReplayState)
    (after : Continuation K ProductPoseidon2.State) : Prop :=
  before.cursor = (authoritativeFrame statementId 9 value).length /\
    before.transcript =
      (paperKey candidate statementId config artifact).publicInputState
        value.recursiveState
        (freshOfValue (productionFullShapeContract logicalWidth publicFits)
          value) /\
    after = piCcsStartContinuation candidate statementId config artifact value

/-- With no adjacent state-digest collision, the final accepted replay state
is ready for the exact selected-key PiCCS start. -/
noncomputable def acceptedRunReadyForSelectedKey
    (candidate : Id)
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config 26 logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact 26 logicalWidth
      publicFits)
    (value : Value candidate (ProductionFullShape logicalWidth publicFits))
    {phases : List AcceptedPhase}
    (run : AcceptedRunFrom 0 phases)
    (initialExpected :
      run.first.before.expected = bindingState statementId 9 value)
    (noStateCollision : ¬ StateDigestCollision) :
    Ready
      ((paperKey candidate statementId config artifact).publicInputState
        value.recursiveState
        (freshOfValue (productionFullShapeContract logicalWidth publicFits)
          value))
      (authoritativeFrame statementId 9 value).length := by
  let selectedState :=
    (paperKey candidate statementId config artifact).publicInputState
      value.recursiveState
      (freshOfValue (productionFullShapeContract logicalWidth publicFits) value)
  refine {
    runtime := run.last.after.runtime
    cursorExact := ?_
    transcriptExact := ?_
  }
  · have finalReady := run.final_runtime_ready
    have frameLength := authoritativeFrame_lengthFor
      (productionFullShapeContract logicalWidth publicFits) statementId 9 value
    change run.last.after.runtime.cursor =
      (authoritativeFrame statementId 9 value).length
    rw [frameLength]
    simpa [ProductNifsCodec.runningFieldCountFor] using finalReady.2
  · have finalReady := run.final_runtime_ready
    have expectedCarry := run.expected_carry_of_no_collision noStateCollision
    change run.last.after.runtime.transcript = selectedState
    calc
      run.last.after.runtime.transcript = run.last.after.expected :=
        finalReady.1
      _ = run.first.before.expected := expectedCarry
      _ = bindingState statementId 9 value := initialExpected
      _ = selectedState :=
        (selected_public_state_eq_bindingState candidate statementId config
          artifact value).symm

/-- The final replay-ready value gives the exact semantic PiCCS-start phase. -/
theorem acceptedRun_implies_piCcsStartRelation_of_no_state_collision
    (candidate : Id)
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config 26 logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact 26 logicalWidth
      publicFits)
    (value : Value candidate (ProductionFullShape logicalWidth publicFits))
    {phases : List AcceptedPhase}
    (run : AcceptedRunFrom 0 phases)
    (initialExpected :
      run.first.before.expected = bindingState statementId 9 value)
    (noStateCollision : ¬ StateDigestCollision) :
    PiCcsStartRelation candidate statementId config artifact value
      run.last.after.runtime
      (piCcsStartContinuation candidate statementId config artifact value) := by
  let ready := acceptedRunReadyForSelectedKey candidate statementId config
    artifact value run initialExpected noStateCollision
  refine ⟨?_, ?_, rfl⟩
  · simpa [acceptedRunReadyForSelectedKey] using ready.cursorExact
  · simpa [acceptedRunReadyForSelectedKey] using ready.transcriptExact

/-- An accepted 86-phase replay supplies the canonical claim frame to the
exact PiCCS start, or exposes one of the two named Poseidon2 failures. -/
theorem acceptedRun_initializes_piCcs_or_named_collision
    (candidate : Id)
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config 26 logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact 26 logicalWidth
      publicFits)
    (value : Value candidate (ProductionFullShape logicalWidth publicFits))
    {phases : List AcceptedPhase}
    (run : AcceptedRunFrom 0 phases)
    (initialExpected :
      run.first.before.expected = bindingState statementId 9 value)
    (initialRuntime : run.first.before.runtime = initialReplayState) :
    ((AcceptedRunFrom.activeChunks phases).flatten =
        authoritativeFrame statementId 9 value /\
      PiCcsStartRelation candidate statementId config artifact value
        run.last.after.runtime
        (piCcsStartContinuation candidate statementId config artifact value)) \/
      StateDigestCollision \/
        FrameReplayCollision statementId 9 value := by
  by_cases stateCollision : StateDigestCollision
  · exact Or.inr (Or.inl stateCollision)
  · have startRelation :=
      acceptedRun_implies_piCcsStartRelation_of_no_state_collision candidate
        statementId config artifact value run initialExpected stateCollision
    rcases accepted_run_recovers_frame_or_named_collision
        (productionFullShapeContract logicalWidth publicFits) statementId 9 value
        run initialExpected initialRuntime with
      exactFrame | collision
    · exact Or.inl ⟨exactFrame, startRelation⟩
    · rcases collision with adjacent | replay
      · exact False.elim (stateCollision adjacent)
      · exact Or.inr (Or.inr replay)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsStart
