import Nightstream.Implementation.NebulaV2.Production.NIFS.PiCCS.TypedBridgeFor
import Nightstream.Implementation.NebulaV2.NIFS.PiCCS.TranscriptCursorFor
import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.TranscriptSemantics

/-!
Contract: exponent-indexed production state handoff from PiCCS to PiRLC.

The generated-relation exponent is one explicit parameter. Satisfied PiCCS
rows fix the complete paper PiCCS output state, and satisfied sampler rows fix
every PiRLC candidate derived from that same state. No output state or
candidate is a placement premise.

Does not own candidate classification, PiRLC algebra, PiDEC, generated row
placement, cryptographic security, Rust refinement, or terminal verification.

Assurance tier: exponent-indexed row-to-paper refinement.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxHeartbeats 1200000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductionProductPiRlcPostPiCcsBridgeFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- History-free sampler input for one symbolic builder. -/
private def samplerInputFromBuilder
    (builder : SymbolicDuplex.Builder) (samplerBase : Nat) :
    ProductPiRlcTranscriptRows.Input where
  postPiCcsLanes := builder.lanes
  transcriptBase := samplerBase

/-- History-free sampler input after the complete PiCCS output at one relation
exponent. -/
def samplerInput
    {rowVariables : Nat}
    (piCcsInput : ProductPiCcsTranscriptRowsFor.Input rowVariables)
    (samplerBase : Nat) : ProductPiRlcTranscriptRows.Input :=
  samplerInputFromBuilder
    (ProductPiCcsTranscriptRowsFor.afterFullOutput piCcsInput) samplerBase

private theorem valueStart_eq_builder
    (assignment : Nat -> Nat) (builder : SymbolicDuplex.Builder)
    (samplerBase : Nat) (absorbed : builder.absorbed = 4) :
    ProductPiRlcTranscriptSemantics.valueStart assignment
        (samplerInputFromBuilder builder samplerBase) =
      SymbolicDuplexSemantics.decodedBuilder assignment builder := by
  apply SymbolicDuplexSemantics.decodedBuilder_eq_of_lanes_absorbed
  · rfl
  · exact absorbed.symm

/-- Dropping old builder entries preserves the complete output value state. -/
theorem valueStart_eq_afterFullOutput
    {rowVariables : Nat}
    (assignment : Nat -> Nat)
    (piCcsInput : ProductPiCcsTranscriptRowsFor.Input rowVariables)
    (samplerBase : Nat) :
    ProductPiRlcTranscriptSemantics.valueStart assignment
        (samplerInput piCcsInput samplerBase) =
      SymbolicDuplexSemantics.decodedBuilder assignment
        (ProductPiCcsTranscriptRowsFor.afterFullOutput piCcsInput) := by
  simpa only [samplerInput] using
    valueStart_eq_builder assignment
      (ProductPiCcsTranscriptRowsFor.afterFullOutput piCcsInput) samplerBase
      (ProductPiCcsTranscriptCursorFor.afterFullOutput_absorbed piCcsInput)

/-- The sampler starts at the selected exponent-indexed paper execution
state. -/
theorem valueStart_eq_executionOutgoing
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (assignment : Nat -> Nat) (samplerBase : Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (satisfied : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment) :
    ProductPiRlcTranscriptSemantics.valueStart assignment
        (samplerInput
          (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
            config artifact running fresh wires) samplerBase) =
      ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
        config artifact).piCcsExecution running fresh proof).outgoingState := by
  exact (valueStart_eq_afterFullOutput assignment _ samplerBase).trans
    (ProductionProductPiCcsTypedBridgeFor.rows_imply_outgoingState candidate
      statementId config artifact running fresh proof wires assignment residues
      one placement satisfied)

/-- Combined rows fix every indexed full-field sampler candidate. -/
theorem rows_imply_candidate_exact
    (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (running : ProductNifsCodec.RunningFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (fresh : ProductNifsCodec.FreshFor rowVariables
      (ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (wires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (assignment : Nat -> Nat) (samplerBase : Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : ProductionProductPiCcsTypedBridgeFor.Placement candidate
      statementId config artifact running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRowsFor.rows
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires)) assignment)
    (piRlcRows : ProductPiRlcTranscriptRows.RowsHold
      (samplerInput
        (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
          config artifact running fresh wires) samplerBase) assignment)
    (index : ProductPiRlcTranscriptRows.CandidateIndex) :
    lcEval assignment
        (ProductPiRlcTranscriptRows.candidate
          (samplerInput
            (ProductionProductPiCcsTypedBridgeFor.rowInput candidate statementId
              config artifact running fresh wires) samplerBase) index) =
      (ProductPoseidon2.candidateValue
        ((ProductionProductPiCcsTypedBridgeFor.paperKey candidate statementId
          config artifact).piCcsExecution running fresh proof).outgoingState
        (Fin.cast ProductPiRlcTranscriptRows.scalarCount_profile index.source)
        (Fin.cast ProductPiRlcTranscriptRows.coefficientCount_profile
          index.coefficient)
        (Fin.cast ProductPiRlcTranscriptRows.attemptCount_profile
          index.attempt)).val := by
  let piInput := ProductionProductPiCcsTypedBridgeFor.rowInput candidate
    statementId config artifact running fresh wires
  let sampleInput := samplerInput piInput samplerBase
  have candidateEq :=
    ProductPiRlcTranscriptSemantics.candidate_rows_sound sampleInput assignment
      residues one piRlcRows index
  have stateEq := valueStart_eq_executionOutgoing candidate statementId config
    artifact running fresh proof wires assignment samplerBase residues one
    placement piCcsRows
  rw [stateEq] at candidateEq
  exact candidateEq

end Nightstream.Implementation.NebulaV2.ProductionProductPiRlcPostPiCcsBridgeFor
