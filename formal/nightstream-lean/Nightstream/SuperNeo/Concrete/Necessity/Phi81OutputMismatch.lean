import Nightstream.SuperNeo.Concrete.Relation
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics

/-!
Model-level separation witness for two current CE-output interpretations.

Owns: one row with `M[0] = 1`, one complete Phi81 carrier with `z[1] = 1`,
and the proof that current `Concrete.Relation.matrixEvaluations` returns zero
at lane one while `OutputClaims.canonicalYRing` returns one.

Does not own: production Rust behavior, serialization, transcript authority,
R1CS lowering, or a repair of either relation model.

Emits constraints: no.

Authority boundary: both evaluations below use the same completed matrix row
and assignment. The theorem only proves that the two Lean definitions are not
equivalent on this input; it makes no claim about which one production refines.

| Interpretation | Construction | Lane-one result |
|---|---|---|
| current concrete CE | scalar `M * z`, then pack output rows into 54 lanes | `0` |
| paper Phi81 CE | derive `bar(M)` coefficient matrices, then evaluate each lane | `1` |
-/

namespace Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-- One Boolean row, one original matrix column, and one running source. -/
def modelShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 1
  freshCount := 0
  runningCount := 1
  matrixCount := 1

/-- Empty relation polynomial; this witness isolates CE output evaluation. -/
def emptyConstraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F modelShape.matrixCount where
  degreeBound := 1
  terms := []
  termsBelowDegree := by simp

/-- The sole original matrix entry `M[0]` is one. -/
def originalMatrices :
    Fin modelShape.matrixCount ->
      BooleanMatrix F modelShape.rowVariables modelShape.logicalWidth :=
  fun _ _ _ => 1

/-- Carrier column one, which is outside the original width but inside the
completed 54-coordinate carrier. -/
def carrierColumnOne : Fin modelShape.carrierWidth :=
  ⟨1, by decide⟩

/-- The complete running carrier is supported only at `z[1]`. -/
def runningAssignment : PaperLinearAlgebra.Assignment F modelShape.carrierWidth :=
  oneHotAssignment
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps
    carrierColumnOne 1

/-- Independent source data consumed by the paper Phi81 output semantics. -/
def sourceData : Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data modelShape where
  matrices := originalMatrices
  constraintPolynomial := emptyConstraintPolynomial
  freshAssignments := fun source => Fin.elim0 source
  runningAssignments := fun _ => runningAssignment
  priorPoint := { coordinates := [], dimension := rfl }
  claimedCoefficient := fun _ => K.zero

/-- Only the row point is relevant here; the column point is inert. -/
def verifierPoints : VerifierPoints modelShape { columnVariables := 0, laneVariables := 0 } where
  rPrime := { coordinates := [], dimension := rfl }
  sPrime := { coordinates := [], dimension := rfl }

def source : Fin modelShape.sourceCount := ⟨0, by decide⟩
def matrix : Fin modelShape.matrixCount := ⟨0, by decide⟩
def laneOne : Fin ringDegree := ⟨1, by decide⟩

/-- The explicit completed matrix row consumed by the current concrete CE
model: one at column zero and zero in the remaining 53 columns. -/
def concreteMatrixRow : List F := 1 :: List.replicate 53 0

/-- The explicit completed assignment: zero at column zero, one at column one,
and zero in the remaining 52 columns. -/
def concreteAssignment : Concrete.Assignment := 0 :: 1 :: List.replicate 52 0

/-- Current concrete relation instance on the completed row/carrier. -/
def concreteSystem : Concrete.Structure where
  matrices := [[concreteMatrixRow]]
  polynomial := []
  rows := 1
  columns := ringDegree
  pointDimension := 0

def concretePoint : Concrete.Point := []

/-- The list assignment is exactly the canonical materialization of the typed
source assignment used by the paper-side construction. -/
theorem concreteAssignment_eq_orderedSource :
    concreteAssignment = sourceData.orderedAssignment source := by
  decide

/-- The list matrix row is exactly the canonical materialization of the same
completed matrix source used by the paper-side construction. -/
theorem concreteMatrixRow_eq_completedSource :
    concreteMatrixRow =
      (canonicalFinIndices modelShape.carrierWidth).map fun column =>
        sourceData.matrixSource.matrices matrix (.nil) column := by
  decide

/-- Current `matrixEvaluations` first forms the scalar dot product, which is
zero because the unit matrix and unit assignment occupy different columns. -/
theorem concrete_laneOne_eq_zero :
    ((Concrete.matrixEvaluations concreteSystem concreteAssignment concretePoint).getD
      0 Concrete.ringKZero) laneOne = K.zero := by
  decide

/-- The independent Phi81 coefficient kernel maps `bar(e_0) * e_1` to one in
output lane one. -/
theorem canonicalYRing_laneOne_eq_one :
    canonicalYRing sourceData verifierPoints source matrix laneOne = K.one := by
  decide

/-- The current concrete CE output interpretation and the paper Phi81 output
interpretation are not equivalent, even on the same matrix and assignment. -/
theorem currentConcrete_ne_canonicalPhi81 :
    ((Concrete.matrixEvaluations concreteSystem concreteAssignment concretePoint).getD
      0 Concrete.ringKZero) laneOne ≠
      canonicalYRing sourceData verifierPoints source matrix laneOne := by
  rw [concrete_laneOne_eq_zero, canonicalYRing_laneOne_eq_one]
  decide

end Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch
