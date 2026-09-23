import NightstreamFPrime.Export.Stage1.ApplicationRetainedGeometry
import NightstreamFPrime.Export.Stage1.ApplicationOrdinaryPlan

/-! Select the application's proved ordinary or compact Poseidon2 relation. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationDirectPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open ApplicationRetainedGeometry

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
  match application.compactHashChain with
  | none => (PerApplicationPackage.applicationPlan application).rowCount
  | some _ => 262

def plan {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : Geometry application columns) : ProductionRelation.Plan columns :=
  match selected : application.compactHashChain with
  | none => ApplicationOrdinaryPlan.plan fits (ordinaryGeometry geometry selected)
  | some certificate => ApplicationPoseidonRetainedGeometry.plan (poseidonGeometry geometry certificate selected)

theorem plan_none {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application) (geometry : Geometry application columns)
    (selected : application.compactHashChain = none) :
    plan fits geometry = ApplicationOrdinaryPlan.plan fits (ordinaryGeometry geometry selected) := by
  unfold plan
  split
  · rfl
  · rename_i certificate found
    rw [selected] at found
    cases found

theorem plan_some {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application) (geometry : Geometry application columns)
    (certificate : ApplicationPoseidonRetainedBlock.Certificate application)
    (selected : application.compactHashChain = some certificate) :
    plan fits geometry = ApplicationPoseidonRetainedGeometry.plan
      (poseidonGeometry geometry certificate selected) := by
  unfold plan
  split
  · rename_i found
    rw [selected] at found
    cases found
  · rename_i candidate found
    have same : certificate = candidate := Option.some.inj (selected.symm.trans found)
    subst candidate
    rfl

@[simp] theorem plan_rowCount {application : Stage1.Application.Program} {columns : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application) (geometry : Geometry application columns) :
    (plan fits geometry).rowCount = rowCount application := by
  cases selected : application.compactHashChain with
  | none => rw [plan_none fits geometry selected, ApplicationOrdinaryPlan.plan_rowCount, rowCount, selected]
  | some certificate =>
    rw [plan_some fits geometry certificate selected,
      ApplicationPoseidonRetainedGeometry.plan_rowCount, rowCount, selected]

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
