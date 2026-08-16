import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsStatementChunkSelection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplaySequence

/-!
Contract: connect accepted claim-replay phases to the production PiCCS
variable-field binding.

Assurance tier: model-level phase-composition and cryptographic-reduction
boundary.

Owns the implication from the 86 accepted claim chunks to the exact prior
point and carried evaluations used by PiCCS. A mismatch is classified as one
of the existing adjacent-state or full-frame failures, or as the named
variable-field Poseidon2 replay collision.

Does not own generated selector rows, PiCCS rows, collision resistance, Rust
refinement, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsBinding

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBindingState
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementChunkSelection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The selected fields from all accepted claim phases are authoritative, or
the earlier claim replay exposes one of its two named failures. -/
theorem acceptedRun_selects_authoritativeFields_or_named_collision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape)
    {phases : List AcceptedPhase}
    (run : AcceptedRunFrom 0 phases)
    (initialExpected :
      run.first.before.expected =
        ProductionFullClaimStateBinding.bindingState statementId degreeBound
          value)
    (initialRuntime : run.first.before.runtime = initialReplayState) :
    selectedFieldsFromChunks (AcceptedRunFrom.activeChunks phases) =
        selectedAuthoritativeFields statementId degreeBound value \/
      StateDigestCollision \/
        ProductionFullClaimStreaming.FrameReplayCollision statementId
          degreeBound value := by
  rcases accepted_run_recovers_frame_or_named_collision contract statementId
      degreeBound value run initialExpected initialRuntime with
    exactFrame | stateCollision | frameCollision
  · exact Or.inl
      (selectedFieldsFromChunks_eq_authoritative contract statementId
        degreeBound value _ exactFrame)
  · exact Or.inr (Or.inl stateCollision)
  · exact Or.inr (Or.inr frameCollision)

/-- The state computed from accepted claim chunks is the direct
authoritative PiCCS variable-field state, or claim replay exposes a named
failure. -/
theorem acceptedRun_selectedState_eq_authoritative_or_named_collision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape)
    {phases : List AcceptedPhase}
    (run : AcceptedRunFrom 0 phases)
    (initialExpected :
      run.first.before.expected =
        ProductionFullClaimStateBinding.bindingState statementId degreeBound
          value)
    (initialRuntime : run.first.before.runtime = initialReplayState) :
    selectedStateFromChunks candidate fullShape statementId degreeBound
          (AcceptedRunFrom.activeChunks phases) =
        authoritativeState statementId degreeBound value \/
      StateDigestCollision \/
        ProductionFullClaimStreaming.FrameReplayCollision statementId
          degreeBound value := by
  rcases acceptedRun_selects_authoritativeFields_or_named_collision contract
      statementId degreeBound value run initialExpected initialRuntime with
    exactFields | stateCollision | frameCollision
  · left
    unfold selectedStateFromChunks authoritativeState
    rw [exactFields]
  · exact Or.inr (Or.inl stateCollision)
  · exact Or.inr (Or.inr frameCollision)

/-- A PiCCS-start assignment checked against the state computed from the
accepted claim chunks uses the exact direct verifier fields. Otherwise it
exposes one of the three named replay failures. -/
theorem acceptedRun_matches_exactPiCcsFields_or_named_collision
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (productionContract : ProductNifsCodec.FullShapeContractFor 26
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (statementId : ProductConcreteNifsFor.StatementId)
    (degreeBound : Nat)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (value : Value candidate
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    {phases : List AcceptedPhase}
    (run : AcceptedRunFrom 0 phases)
    (initialExpected :
      run.first.before.expected =
        ProductionFullClaimStateBinding.bindingState statementId degreeBound
          value)
    (initialRuntime : run.first.before.runtime = initialReplayState)
    (supplied : List Nat)
    (checked :
      stateForFields candidate
          (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
            publicFits)
          statementId degreeBound supplied =
        selectedStateFromChunks candidate
          (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
            publicFits)
          statementId degreeBound
          (AcceptedRunFrom.activeChunks phases)) :
    supplied =
        frameOrderVariableFields
          (ProductionProductPiCcsTypedBridgeFor.exactVerifierInput candidate
            statementId config artifact value.recursiveState fresh) \/
      StateDigestCollision \/
        ProductionFullClaimStreaming.FrameReplayCollision statementId
          degreeBound value \/
          VariableReplayCollision statementId degreeBound value := by
  rcases acceptedRun_selectedState_eq_authoritative_or_named_collision
      productionContract statementId degreeBound value run initialExpected
        initialRuntime with stateExact | stateCollision | frameCollision
  · have suppliedExact :
        stateForFields candidate
            (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth
              publicFits)
            statementId degreeBound supplied =
          authoritativeState statementId degreeBound value :=
      checked.trans stateExact
    rcases accepted_fields_match_exactVerifierInput_or_collision candidate
        statementId degreeBound config artifact value fresh supplied
        suppliedExact with exactFields | variableCollision
    · exact Or.inl exactFields
    · exact Or.inr (Or.inr (Or.inr variableCollision))
  · exact Or.inr (Or.inl stateCollision)
  · exact Or.inr (Or.inr (Or.inl frameCollision))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsBinding
