import NightstreamFPrime.Export.Stage1.ApplicationDirectPlan
import NightstreamFPrime.Export.Stage1.ApplicationOrdinarySoundness
import Mathlib.Data.List.OfFn

/-! Application rows bind the actual pilot states to the declared step. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationAssignmentSoundness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ApplicationDirectPlan ApplicationOrdinaryGeometry

theorem rowsZero_implies_step {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application columns) (assignment : Assignment F columns)
    (one : assignment (oneColumn geometry) = 1)
    (rows : (plan fits geometry).RowsZero assignment) :
    (List.ofFn fun lane : Stage1.Application.StateIndex =>
      ((PiRLCPoseidonGeometry.outputInputBlock application).form
        (PiRLCPoseidonGeometry.outputInputStart application)
        (PiRLCPoseidonGeometry.outputInputFits (pilotGeometry geometry))
        (Location.preimageWord lane)).eval assignment) =
      application.step
        (List.ofFn fun lane : Stage1.Application.StateIndex =>
          ((PiRLCPoseidonGeometry.priorInputBlock application).form
            (PiRLCPoseidonGeometry.priorInputStart application)
            (PiRLCPoseidonGeometry.priorInputFits (pilotGeometry geometry))
            (Location.preimageWord lane)).eval assignment)
        (List.ofFn fun lane : Fin application.witnessWordCount =>
          (witnessForm geometry lane).eval assignment) := by
  exact ApplicationOrdinarySoundness.rowsZero_implies_step fits geometry assignment one rows

theorem rowsZero_implies_encodedHolds {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application columns) (assignment : Assignment F columns)
    (source : Fin (ApplicationRetainedBlocks.sourceWidth application) → F)
    (encodes : Encodes geometry assignment source)
    (one : assignment (oneColumn geometry) = 1)
    (rows : (plan fits geometry).RowsZero assignment) :
    Stage1.Application.Holds application.step (Layout.Stage1.ApplicationInputs.interface application)
      (Layout.Stage1.ApplicationInputs.localStart application) (sourceEnv source) := by
  have step := rowsZero_implies_step fits geometry assignment one rows
  simp_rw [← inputForm_eq_pilot geometry, ← outputForm_eq_pilot geometry] at step
  have inputValues (lane : Stage1.Application.StateIndex) :
      (inputForm geometry lane).eval assignment =
        sourceEnv source (Layout.Stage1.ApplicationInputs.inputColumn lane) := by
    rw [inputForm, LowNormBlock.Block.form_eval _ _ _ _ _ encodes.input]
    unfold sourceEnv ApplicationOrdinaryPlan.sourceEnv
    rw [dif_pos (show Layout.Stage1.ApplicationInputs.inputColumn lane <
      ApplicationRetainedBlocks.sourceWidth application from
        ((ApplicationRetainedBlocks.inputBlock application).source lane).isLt)]
    rfl
  have outputValues (lane : Stage1.Application.StateIndex) :
      (outputForm geometry lane).eval assignment =
        sourceEnv source (Layout.Stage1.ApplicationInputs.outputColumn lane) := by
    rw [outputForm, LowNormBlock.Block.form_eval _ _ _ _ _ encodes.output]
    unfold sourceEnv ApplicationOrdinaryPlan.sourceEnv
    rw [dif_pos (show Layout.Stage1.ApplicationInputs.outputColumn lane <
      ApplicationRetainedBlocks.sourceWidth application from
        ((ApplicationRetainedBlocks.outputBlock application).source lane).isLt)]
    rfl
  have witnessValues (lane : Fin application.witnessWordCount) :
      (witnessForm geometry lane).eval assignment =
        sourceEnv source (Layout.Stage1.ApplicationInputs.witnessColumn lane) := by
    rw [witnessForm, LowNormBlock.Block.form_eval _ _ _ _ _ encodes.witness]
    unfold sourceEnv ApplicationOrdinaryPlan.sourceEnv
    rw [dif_pos (show Layout.Stage1.ApplicationInputs.witnessColumn lane <
      ApplicationRetainedBlocks.sourceWidth application from
        ((ApplicationRetainedBlocks.witnessBlock application).source lane).isLt)]
    rfl
  simp only [inputValues, outputValues, witnessValues] at step
  exact step

end NightstreamFPrime.Export.Stage1.ApplicationAssignmentSoundness
