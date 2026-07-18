import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement

/-!
Focused model-level regressions for the five-ring F' CCS carrier refinement.

| Protocol | Phase | Family | Regression |
|---|---|---|---|
| F' / CCS | aligned columns | public / padding / private | a nontrivial legacy matrix image survives insertion of the thirteen fixed zeros |
| F' / CCS | carrier completion | logical prefix / completed suffix | the same image survives completion to whole Phi81 blocks |
| F' / CCS | numeric row | little-endian row two | the decoded row selects the same legacy and completed image |
| F' / CCS | lifted structure | explicit polynomial | the exact polynomial object and every residual are preserved |
| F' / CCS | relation membership | Boolean zero set | legacy and lifted `ConstraintSatisfied` are equivalent |
-/

namespace tests.FPrimeCarrier270CcsRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.ColumnMap
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.CcsRefinement

#check alignedMatrixVectorAt_eq
#check carrierMatrixVectorAt_eq
#check carrierMatrixVectorAt_numericRow_eq
#check carrierMatrixVectorAt_rowIndex_eq
#check liftStructure_constraintPolynomial
#check matrixImagesAt_eq
#check residualAt_eq
#check constraintSatisfied_iff

/-- Two row variables and one private legacy coordinate make this fixture
exercise public preservation, middle insertion, and private relocation. -/
def dimensions : Dimensions where
  rowVariables := 2
  legacyLogicalWidth := 258
  matrixCount := 1
  legacyPublicFits := by decide

def legacyAssignment : LegacyAssignment dimensions :=
  fun column =>
    if column.val = 0 then 3
    else if column.val = 257 then 7
    else 0

def legacyMatrices : Fin dimensions.matrixCount ->
    BooleanMatrix F dimensions.rowVariables dimensions.legacyLogicalWidth :=
  fun _ vertex column =>
    if rowIndex vertex = 2 then
      if column.val = 0 then 4
      else if column.val = 257 then 9
      else 0
    else 0

/-- Identity in the sole matrix-image variable, so the residual depends on
the preserved matrix-vector value rather than being a constant fixture. -/
def identityMonomial : CCSResidualTable.Monomial F 1 where
  coefficient := 1
  exponents := fun _ => 1

def identityPolynomial : CCSResidualTable.ConstraintPolynomial F 1 where
  degreeBound := 2
  terms := [identityMonomial]
  termsBelowDegree := by
    intro term member
    simp only [List.mem_singleton] at member
    subst term
    decide

def legacyStructure : LegacyStructure dimensions where
  matrices := legacyMatrices
  constraintPolynomial := identityPolynomial

def numericRowTwo : Fin (2 ^ dimensions.rowVariables) := ⟨2, by decide⟩
def numericRowZero : Fin (2 ^ dimensions.rowVariables) := ⟨0, by decide⟩
def matrixZero : Fin dimensions.matrixCount := ⟨0, by decide⟩
def columnZero : Fin dimensions.legacyLogicalWidth := ⟨0, by decide⟩

/-- The fixture is genuinely row-sensitive: it exercises little-endian row
decoding instead of making the numeric-row regression true for every row. -/
example :
    legacyMatrices matrixZero
        (rowVertex dimensions.rowVariables numericRowTwo) columnZero = 4 := by
  decide

example :
    legacyMatrices matrixZero
        (rowVertex dimensions.rowVariables numericRowZero) columnZero = 0 := by
  decide

example :
    matrixVectorAt ConcreteCarrier.baseOps
        (alignedMatrix dimensions (legacyMatrices matrixZero))
        (alignedLogicalAssignment dimensions legacyAssignment)
        (rowVertex dimensions.rowVariables numericRowTwo) =
      matrixVectorAt ConcreteCarrier.baseOps (legacyMatrices matrixZero)
        legacyAssignment (rowVertex dimensions.rowVariables numericRowTwo) := by
  exact alignedMatrixVectorAt_eq dimensions (legacyMatrices matrixZero)
    legacyAssignment _

example :
    matrixVectorAt ConcreteCarrier.baseOps
        (carrierMatrix dimensions (legacyMatrices matrixZero))
        (assignment dimensions legacyAssignment)
        (rowVertex dimensions.rowVariables numericRowTwo) =
      matrixVectorAt ConcreteCarrier.baseOps (legacyMatrices matrixZero)
        legacyAssignment (rowVertex dimensions.rowVariables numericRowTwo) := by
  exact carrierMatrixVectorAt_numericRow_eq dimensions
    (legacyMatrices matrixZero)
    legacyAssignment numericRowTwo

example :
    (liftStructure dimensions legacyStructure).constraintPolynomial =
      identityPolynomial := by
  rfl

example (vertex : BooleanVertex dimensions.rowVariables) :
    CCSResidualTable.residualAt ConcreteCarrier.baseOps
        (liftStructure dimensions legacyStructure).matrixSource.system
        (assignment dimensions legacyAssignment) vertex =
      CCSResidualTable.residualAt ConcreteCarrier.baseOps legacyStructure
        legacyAssignment vertex := by
  exact residualAt_eq dimensions legacyStructure legacyAssignment vertex

example :
    CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
        (liftStructure dimensions legacyStructure).matrixSource.system
        (assignment dimensions legacyAssignment) <->
      CCSResidualTable.ConstraintSatisfied ConcreteCarrier.baseOps
        legacyStructure legacyAssignment := by
  exact constraintSatisfied_iff dimensions legacyStructure legacyAssignment

end tests.FPrimeCarrier270CcsRefinement
