import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources

/-!
Focused model-level regressions for the F' Split-NC source adapter.

| Protocol | Phase | Family | Regression |
|---|---|---|---|
| `Pi_CCS` | source shape | aligned / complete width | Split-NC and typed F' carriers are identical |
| `Pi_CCS` | matrix source | shifted private column | the completed source retains the legacy matrix value |
| `Pi_CCS` | fresh source | fixed padding / completion | Split-NC uses the exact F' fresh assignment constructor |
| `Pi_CCS` | running source | full carrier | a post-logical running coordinate passes through unchanged |
| FE | fresh CCS | empty explicit polynomial | adapter truth is obtained exactly from legacy fresh truth |
-/

namespace tests.FPrimeCarrier270PiCcsSources

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsSources

#check semanticShape_carrierWidth
#check Inputs.data_matrixSource_matrix
#check Inputs.data_freshAssignment_eq
#check Inputs.data_runningAssignment_eq
#check Inputs.data_assignment_runningIndex_eq
#check Inputs.freshMatrixImagesAt_eq
#check Inputs.freshResidualAt_eq
#check Inputs.freshTruth_iff_legacy

def dimensions : Dimensions where
  rowVariables := 1
  legacyLogicalWidth := 258
  matrixCount := 1
  legacyPublicFits := by decide

def freshCount : Nat := 2
def runningCount : Nat := 1

def emptyPolynomial : CCSResidualTable.ConstraintPolynomial F 1 where
  degreeBound := 1
  terms := []
  termsBelowDegree := by simp

def legacyStructure :
    LegacyBatchStructure dimensions freshCount runningCount where
  matrices := fun _ _ column => if column.val = 257 then 9 else 0
  constraintPolynomial := emptyPolynomial

def legacyFreshAssignments :
    Fin freshCount -> LegacyAssignment dimensions :=
  fun source column =>
    if source.val = 0 && column.val = 257 then 7 else 0

/-- Coordinate 300 lies after the aligned logical width 271 but inside its
324-coordinate completed carrier. Running CE sources retain it. -/
def runningTailColumn : Fin dimensions.shape.carrierWidth := ⟨300, by decide⟩

def runningAssignments : Fin runningCount -> Assignment dimensions.shape :=
  fun _ column => if column = runningTailColumn then 11 else 0

def inputs : Inputs dimensions freshCount runningCount where
  legacyStructure := legacyStructure
  freshAssignments := legacyFreshAssignments
  runningAssignments := runningAssignments
  priorPoint := { coordinates := [K.zero], dimension := rfl }
  claimedCoefficient := fun _ => K.zero

def matrixZero : Fin dimensions.matrixCount := ⟨0, by decide⟩
def freshZero : Fin freshCount := ⟨0, by decide⟩
def runningZero : Fin runningCount := ⟨0, by decide⟩
def legacyPrivate : Fin dimensions.legacyLogicalWidth := ⟨257, by decide⟩

example :
    (semanticShape dimensions freshCount runningCount).carrierWidth =
      dimensions.shape.carrierWidth := by
  exact semanticShape_carrierWidth dimensions freshCount runningCount

example :
    inputs.data.matrixSource.matrices matrixZero
        (.cons false .nil)
        (alignedCarrierIndex dimensions legacyPrivate) = 9 := by
  rw [Inputs.data_matrixSource_matrix,
    ColumnMap.carrierMatrix_at_alignedCarrierIndex]
  simp [inputs, legacyStructure, matrixZero, legacyPrivate]

example :
    inputs.data.freshAssignment freshZero =
      assignment dimensions (legacyFreshAssignments freshZero) := by
  exact Inputs.data_freshAssignment_eq inputs freshZero

example :
    inputs.data.assignment (Data.runningIndex runningZero)
        runningTailColumn = 11 := by
  rw [Inputs.data_assignment_runningIndex_eq]
  simp [inputs, runningAssignments, runningTailColumn]

example :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe.FreshTruth
      inputs.data := by
  apply (Inputs.freshTruth_iff_legacy inputs).2
  intro source vertex
  rfl

end tests.FPrimeCarrier270PiCcsSources
