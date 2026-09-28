import NightstreamFPrime.Export.Stage1.ApplicationAssignmentSoundness
import NightstreamFPrime.Export.Stage1.ActualPreimageFraming
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint

/-!
Owns the application equation on the typed states decoded from the actual
pilot preimages. The witness and both states are read from their existing
owned forms. The selected-row theorem supplies every application row.
-/

namespace NightstreamFPrime.Export.Stage1.ActualApplicationStep

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

variable {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}

def witness (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) : AppWitness :=
  List.ofFn fun index : Fin application.witnessWordCount =>
    ((ApplicationDirectPlan.Location.witness index).form geometry).eval assignment

private theorem input_eq_forms
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    StateDecoder.currentState (ActualPreimageFraming.priorState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment) =
      List.ofFn (fun index : Lifecycle.Stage1.Application.StateIndex =>
        ((PiRLCPoseidonGeometry.priorInputBlock application).form
          (PiRLCPoseidonGeometry.priorInputStart application)
          (PiRLCPoseidonGeometry.priorInputFits
            (ApplicationRetainedGeometry.pilotGeometry geometry))
          (ApplicationDirectPlan.Location.preimageWord index)).eval assignment) := by
  unfold StateDecoder.currentState StateDecoder.slice
  apply congrArg List.ofFn
  funext index
  have bounded : RunningTransitionInputs.currentStateWordStart + index.val <
      PilotProduction.stateHashWords := by
    have bound := index.isLt
    rw [PilotProduction.stateHashWords_eq]
    norm_num [RunningTransitionInputs.currentStateWordStart,
      Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
    omega
  rw [ActualPreimageFraming.priorState, dif_pos bounded]
  rfl

private theorem output_eq_forms
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) :
    StateDecoder.currentState (ActualPreimageFraming.outputState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment) =
      List.ofFn (fun index : Lifecycle.Stage1.Application.StateIndex =>
        ((PiRLCPoseidonGeometry.outputInputBlock application).form
          (PiRLCPoseidonGeometry.outputInputStart application)
          (PiRLCPoseidonGeometry.outputInputFits
            (ApplicationRetainedGeometry.pilotGeometry geometry))
          (ApplicationDirectPlan.Location.preimageWord index)).eval assignment) := by
  unfold StateDecoder.currentState StateDecoder.slice
  apply congrArg List.ofFn
  funext index
  have bounded : RunningTransitionInputs.currentStateWordStart + index.val <
      PilotProduction.stateHashWords := by
    have bound := index.isLt
    rw [PilotProduction.stateHashWords_eq]
    norm_num [RunningTransitionInputs.currentStateWordStart,
      Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
    omega
  rw [ActualPreimageFraming.outputState, dif_pos bounded]
  rfl

/-- Arbitrary accepted application rows force the decoded next current state
to be the selected application step on the decoded prior state and witness. -/
theorem rowsZero_implies_decodedStep
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ApplicationRetainedGeometry.oneColumn geometry) = 1)
    (rows : (ApplicationDirectPlan.plan fits geometry).RowsZero assignment) :
    StateDecoder.currentState (ActualPreimageFraming.outputState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment) =
      application.step
        (StateDecoder.currentState (ActualPreimageFraming.priorState
          (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry geometry) assignment))
        (witness geometry assignment) := by
  rw [input_eq_forms geometry assignment, output_eq_forms geometry assignment]
  exact ApplicationAssignmentSoundness.rowsZero_implies_step
    fits geometry assignment one rows

/-- The selected complete Stage 1 rows imply the same typed application
equation. This theorem needs no canonical raw packet or representation. -/
theorem selectedRowsZero_implies_decodedStep
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (one : assignment (ApplicationRetainedGeometry.oneColumn
      (PerApplicationFixedPoint.geometry application)) = 1)
    (accepted : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
      assignment) :
    StateDecoder.currentState (ActualPreimageFraming.outputState
        (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
          (PerApplicationFixedPoint.geometry application)) assignment) =
      application.step
        (StateDecoder.currentState (ActualPreimageFraming.priorState
          (DirectApplicationPrefixPlan.piCcsOrdinaryGeometry
            (PerApplicationFixedPoint.geometry application)) assignment))
        (witness (PerApplicationFixedPoint.geometry application) assignment) := by
  have selected : (DirectApplicationPrefixPlan.plan
      (PerApplicationFixedPoint.relation application fits) fits.package
      (PerApplicationFixedPoint.geometry application)).RowsZero assignment := by
    rw [PerApplicationFixedPoint.plan_fixedPoint]
    exact accepted
  have children := (DirectApplicationPrefixPlan.rowsZero_iff
    (PerApplicationFixedPoint.relation application fits) fits.package
    (PerApplicationFixedPoint.geometry application) assignment).mp selected
  exact rowsZero_implies_decodedStep fits.package
    (PerApplicationFixedPoint.geometry application) assignment one children.1.1.2

end NightstreamFPrime.Export.Stage1.ActualApplicationStep
