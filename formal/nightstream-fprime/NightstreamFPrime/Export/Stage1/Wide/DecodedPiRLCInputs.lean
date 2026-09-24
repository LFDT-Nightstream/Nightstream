import NightstreamFPrime.Export.Stage1.Wide.DecodedPrefix
import NightstreamFPrime.Layout.Stage1.Wide.PiRLCInputs

/-! The direct PiRLC reads the same values and final transcript state as the
decoded PiCCS verifier, for any accepted assignment. -/

namespace NightstreamFPrime.Export.Stage1.Wide.DecodedPiRLCInputs

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ProductionRelation

theorem value (program : RetainedLayout.Program)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (ring : PiRLCGeometry.RingIndex) (lane : Fin ringDegree) :
    PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) assignment ring lane =
      DecodedPrefix.piCcsEnv program assignment
        ((PiRLCProductSchedule.descriptor (PiRLCProductRingSchedule.laneInvocation ring lane)).valueColumn lane) := by
  dsimp only [PiRLCWitness.inputValues, Phi81ProductPlan.evalState, Stage1Plan.piRlcInterface, Stage1Plan.value]
  rw [AssignmentPullback.form_eval]
  rw [PiRLCValueWiring.form_eval_eq_decodedEnv]
  have descriptorLane :
      (PiRLCProductSchedule.descriptor (PiRLCProductRingSchedule.laneInvocation ring lane)).lane = lane := by
    rw [PiRLCProductRingSchedule.descriptor_laneInvocation]
    rfl
  rw [descriptorLane]
  rfl

theorem initialState {width : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (program : RetainedLayout.Program) (relation : ProductionKey.LogicalRelation width publicFits)
    (assignment : Assignment F (RetainedLayout.logicalWidth program))
    (one : assignment (Stage1Plan.piRlcInterface program).oneColumn = 1)
    (rows : (Stage1Plan.prefixPlan program relation).RowsZero assignment) :
    PiRLCWitness.initial (Stage1Plan.piRlcInterface program) assignment =
      fun lane => (Layout.Stage1.Wide.PiRLCInputs.piCcsOutputState
        (logicalWidth := width) (publicFits := publicFits) lane).eval (DecodedPrefix.piCcsEnv program assignment) := by
  have parts := (AssignmentPullback.rowsZero_iff program assignment _ _).mp rows
  simp only [DirectPiDECPrefixPlan.piCcsCompletePlan, Plan.append_rowsZero_iff] at parts
  have endpoint := PiCCSDecodedEndpoints.rowsZero_implies_endpointStates
    (Stage1Plan.piCcsGeometry program) (Stage1Plan.poseidonGeometry program)
    (AssignmentPullback.assignment program assignment)
    (DecodedPrefix.reference_one program assignment one) parts.2
    PiCCSTranscriptEndpointPlan.outputFamily
  funext lane
  dsimp only [PiRLCWitness.initial, SparseLayer.evalState, Stage1Plan.piRlcInterface, Stage1Plan.initialState]
  rw [AssignmentPullback.form_eval]
  have stateColumn : (Layout.Stage1.Wide.PiRLCInputs.piCcsOutputState
      (logicalWidth := width) (publicFits := publicFits) lane) =
      .var (PiCCSTranscriptEndpointPlan.endpointColumn PiCCSTranscriptEndpointPlan.outputFamily lane) :=
    PiCCSTranscriptEndpointPlan.outputFinalState_endpoint_of_shape width publicFits lane
  rw [stateColumn]
  exact congrFun (List.ofFn_injective endpoint) lane

end NightstreamFPrime.Export.Stage1.Wide.DecodedPiRLCInputs
