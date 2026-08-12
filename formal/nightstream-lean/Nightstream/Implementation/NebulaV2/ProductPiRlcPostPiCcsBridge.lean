import Nightstream.Implementation.NebulaV2.ProductPiCcsTypedBridge
import Nightstream.Implementation.NebulaV2.ProductPiCcsTranscriptCursor
import Nightstream.Implementation.NebulaV2.ProductPiRlcTranscriptSemantics

/-!
Contract: exact state handoff from selected V2 PiCCS to PiRLC sampling.

Owns the proof that the indexed PiRLC candidate family starts from the lanes
and cursor produced by the complete PiCCS output absorption. Combined PiCCS
and PiRLC row satisfaction therefore derives every candidate from the exact
paper execution's outgoing state.

Does not own candidate classification, response construction, PiRLC algebra,
column-window placement, PiDEC, cryptographic security, Rust, or complete
NIFS acceptance.
-/

set_option autoImplicit false
set_option maxHeartbeats 1200000
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcPostPiCcsBridge

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The PiRLC candidate input is a history-free view of the exact complete
PiCCS output lanes. The candidate row family owns its new disjoint base. -/
def samplerInput
    (piCcsInput : ProductPiCcsTranscriptRows.Input)
    (samplerBase : Nat) : ProductPiRlcTranscriptRows.Input where
  postPiCcsLanes := (ProductPiCcsTranscriptRows.afterFullOutput piCcsInput).lanes
  transcriptBase := samplerBase

/-- Forgetting the old builder entries does not change the carried value
state. The exact full-output length supplies cursor four. -/
theorem valueStart_eq_afterFullOutput
    (assignment : Nat -> Nat)
    (piCcsInput : ProductPiCcsTranscriptRows.Input)
    (samplerBase : Nat) :
    ProductPiRlcTranscriptSemantics.valueStart assignment
        (samplerInput piCcsInput samplerBase) =
      SymbolicDuplexSemantics.decodedBuilder assignment
        (ProductPiCcsTranscriptRows.afterFullOutput piCcsInput) := by
  apply SymbolicDuplexSemantics.decodedBuilder_eq_of_lanes_absorbed
  · rfl
  · exact
      (ProductPiCcsTranscriptCursor.afterFullOutput_absorbed piCcsInput).symm

/-- The history-free sampler view denotes the exact selected paper PiCCS
outgoing state. -/
theorem valueStart_eq_executionOutgoing
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : ProductPiCcsTypedBridge.ExactProof)
    (wires : ProductPiCcsTypedBridge.Wires)
    (assignment : Nat -> Nat) (samplerBase : Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : ProductPiCcsTypedBridge.Placement statementId config artifact
      running fresh proof wires assignment)
    (satisfied : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (ProductPiCcsTypedBridge.rowInput statementId config artifact
          running fresh wires)) assignment) :
    ProductPiRlcTranscriptSemantics.valueStart assignment
        (samplerInput
          (ProductPiCcsTypedBridge.rowInput statementId config artifact
            running fresh wires) samplerBase) =
      ((ProductConcreteNifs.key statementId config artifact
        ).piCcsExecution running fresh proof).outgoingState := by
  exact (valueStart_eq_afterFullOutput assignment _ samplerBase).trans
    (ProductPiCcsTypedBridge.rows_imply_outgoingState statementId config artifact
      running fresh proof wires assignment residues one placement satisfied)

/-- Combined row satisfaction fixes every indexed candidate to the selected
paper execution. No outgoing state or candidate value is a premise. -/
theorem rows_imply_candidate_exact
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof : ProductPiCcsTypedBridge.ExactProof)
    (wires : ProductPiCcsTypedBridge.Wires)
    (assignment : Nat -> Nat) (samplerBase : Nat)
    (residues : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placement : ProductPiCcsTypedBridge.Placement statementId config artifact
      running fresh proof wires assignment)
    (piCcsRows : Satisfies
      (ProductPiCcsTranscriptRows.rows
        (ProductPiCcsTypedBridge.rowInput statementId config artifact
          running fresh wires)) assignment)
    (piRlcRows : ProductPiRlcTranscriptRows.RowsHold
      (samplerInput
        (ProductPiCcsTypedBridge.rowInput statementId config artifact
          running fresh wires) samplerBase) assignment)
    (index : ProductPiRlcTranscriptRows.CandidateIndex) :
    lcEval assignment
        (ProductPiRlcTranscriptRows.candidate
          (samplerInput
            (ProductPiCcsTypedBridge.rowInput statementId config artifact
              running fresh wires) samplerBase) index) =
      (ProductPoseidon2.candidateValue
        ((ProductConcreteNifs.key statementId config artifact
          ).piCcsExecution running fresh proof).outgoingState
        (Fin.cast ProductPiRlcTranscriptRows.scalarCount_profile index.source)
        (Fin.cast ProductPiRlcTranscriptRows.coefficientCount_profile
          index.coefficient)
        (Fin.cast ProductPiRlcTranscriptRows.attemptCount_profile
          index.attempt)).val := by
  let piInput := ProductPiCcsTypedBridge.rowInput statementId config artifact
    running fresh wires
  let sampleInput := samplerInput piInput samplerBase
  have candidateEq :=
    ProductPiRlcTranscriptSemantics.candidate_rows_sound sampleInput assignment
      residues one piRlcRows index
  have stateEq := valueStart_eq_executionOutgoing statementId config artifact
    running fresh proof wires assignment samplerBase residues one placement
    piCcsRows
  rw [stateEq] at candidateEq
  exact candidateEq

end Nightstream.Implementation.NebulaV2.ProductPiRlcPostPiCcsBridge
