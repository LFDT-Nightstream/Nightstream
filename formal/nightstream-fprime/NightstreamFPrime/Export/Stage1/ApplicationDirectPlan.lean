import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryGeometry
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryPlan

/-! The application plan uses every row of its proved physical circuit. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationDirectPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open ApplicationOrdinaryGeometry

abbrev sourceEnv := @ApplicationOrdinaryPlan.sourceEnv
abbrev Location.preimageWord := ApplicationOrdinaryPlan.Location.preimageWord

def inputForm {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) (lane : Stage1.Application.StateIndex) : SparseForm columns :=
  (ApplicationRetainedBlocks.inputBlock application).form (inputStart application) (inputFits geometry) lane

def witnessForm {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) (lane : Fin application.witnessWordCount) : SparseForm columns :=
  (ApplicationRetainedBlocks.witnessBlock application).form
    (witnessStart application) (witnessFits geometry) lane

def outputForm {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) (lane : Stage1.Application.StateIndex) : SparseForm columns :=
  (ApplicationRetainedBlocks.outputBlock application).form (outputStart application) (outputFits geometry) lane

def rowCount (application : Stage1.Application.Program) : Nat :=
  (PerApplicationPackage.applicationPlan application).rowCount

def plan {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application columns) : ProductionRelation.Plan columns :=
  ApplicationOrdinaryPlan.plan fits geometry

@[simp] theorem plan_rowCount {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application) (geometry : Geometry application columns) :
    (plan fits geometry).rowCount = rowCount application :=
  ApplicationOrdinaryPlan.plan_rowCount fits geometry

theorem inputForm_eq_pilot {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) (lane : Stage1.Application.StateIndex) :
    inputForm geometry lane =
      (PiRLCPoseidonGeometry.priorInputBlock application).form
        (PiRLCPoseidonGeometry.priorInputStart application)
        (PiRLCPoseidonGeometry.priorInputFits (pilotGeometry geometry))
        (Location.preimageWord lane) := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCPoseidonGeometry.priorInputStart application + 35 * 41 + lane.val * 41 =
      PiRLCPoseidonGeometry.priorInputStart application + (35 + lane.val) * 41
    omega

theorem outputForm_eq_pilot {application : Stage1.Application.Program} {columns : Nat}
    (geometry : Geometry application columns) (lane : Stage1.Application.StateIndex) :
    outputForm geometry lane =
      (PiRLCPoseidonGeometry.outputInputBlock application).form
        (PiRLCPoseidonGeometry.outputInputStart application)
        (PiRLCPoseidonGeometry.outputInputFits (pilotGeometry geometry))
        (Location.preimageWord lane) := by
  apply LowNormBlock.Block.form_eq_of_coordinates
  · rfl
  · change PiRLCPoseidonGeometry.outputInputStart application + 35 * 41 + lane.val * 41 =
      PiRLCPoseidonGeometry.outputInputStart application + (35 + lane.val) * 41
    omega

end NightstreamFPrime.Export.Stage1.ApplicationDirectPlan
