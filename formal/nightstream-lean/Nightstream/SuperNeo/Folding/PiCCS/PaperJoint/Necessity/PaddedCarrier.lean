import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource

/-!
Necessity witness for retaining the completed Phi81 carrier after folding.

Protocol: SuperNeo coefficient embedding (Section 5) and carried evaluations
inside `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: completed assignment carrier to a nonconstant matrix-image coefficient.
Constraint family: original-width projection / carried tail lane / output
coefficient one.

Owns: a kernel-checked one-block witness in which two carried assignments have
the same original CCS-width projection but different derived coefficient
images. The sole matrix is zero beyond its one original column, yet assignment
carrier lane one contributes one to output coefficient one through the
independently defined Phi81 kernel.

Does not own: a claim that fresh CCS padding may be nonzero, production Rust
refinement, transcript, SumCheck, R1CS lowering, row removal, or constraint
counts.

Emits constraints: no.

Authority boundary: both coefficient matrices are derived from the same sole
original matrix. The witness changes only a carried assignment coordinate that
lies outside the original width but inside the paper-required complete
carrier. It therefore proves that projecting a carried CE back to the original
CCS width loses semantically live information.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| CCS structure | original matrix | width-one column | the sole original entry is one |
| CCS structure | carrier completion | tail column one | the completed matrix entry is zero |
| carried CE | assignment carrier | tail column one | compare zero with one while preserving the logical projection |
| coefficient embedding | Phi81 ring action | output coefficient one | `bar(e_0) * e_1` contributes one |
| carried evaluation | matrix image | full 54-column dot product | tail assignment changes the derived output from zero to one |
| assurance | necessity | carrier retention | original-width projection cannot determine the carried coefficient image |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier

open Nightstream.SuperNeo.Concrete
open PaperLinearAlgebra
open MatrixCoefficientSource
open Phi81MatrixSource

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-- Empty constraint polynomial; the witness isolates coefficient embedding
rather than CCS satisfiability. -/
def emptyConstraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F 1 where
  degreeBound := 1
  terms := []
  termsBelowDegree := by simp

/-- The unique original matrix entry is one. -/
def originalMatrices : Fin 1 -> BooleanMatrix F 0 1 :=
  fun _ _ _ => 1

/-- Width-one original matrix completed to one full 54-lane block. -/
def matrixSource :=
  Phi81MatrixSource.source 0 0 1 1 1 originalMatrices
    emptyConstraintPolynomial

/-- The unique matrix and Boolean row. -/
def matrix : Fin 1 := ⟨0, by decide⟩

def vertex : BooleanVertex 0 := .nil

/-- First completed assignment coordinate outside the original width. -/
def tailColumn : Fin (Phi81CarrierLayout.carrierWidth 1) :=
  ⟨1, by decide⟩

/-- Nonconstant output coefficient reached by the tail assignment lane. -/
def outputOne : Fin ringDegree := ⟨1, by decide⟩

/-- A carried assignment with no active carrier coordinate. -/
def zeroAssignment : Assignment F (Phi81CarrierLayout.carrierWidth 1) :=
  fun _ => 0

/-- A carried assignment supported only at the first completed tail lane. -/
def tailAssignment : Assignment F (Phi81CarrierLayout.carrierWidth 1) :=
  fun column => if column = tailColumn then 1 else 0

/-- Project a completed carried assignment back to the original CCS width.
This is precisely the lossy operation under audit. -/
def logicalProjection
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth 1)) :
    Assignment F 1 :=
  fun column => assignment (Phi81CarrierLayout.embedLogical column)

/-- Full derived coefficient image, using all completed carrier coordinates. -/
def coefficientImage
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth 1)) : F :=
  matrixVectorAt ConcreteCarrier.baseOps
    (matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix outputOne)
    assignment vertex

/-- The sole matrix remains zero at the completed tail coordinate. -/
theorem completed_matrix_tail_zero :
    matrixSource.matrices matrix vertex tailColumn = 0 := by
  exact Phi81MatrixSource.source_matrix_tail_zero 0 0 1 1 1
    originalMatrices emptyConstraintPolynomial matrix vertex tailColumn
    (by decide)

/-- Nevertheless, the coefficient-expanded matrix has a unit entry at output
coefficient one and assignment carrier lane one. -/
theorem tail_coefficient_entry_eq_one :
    matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix outputOne
        vertex tailColumn = 1 := by
  decide

/-- The two carried assignments are indistinguishable after truncation to the
original CCS width. -/
theorem logicalProjections_eq :
    logicalProjection zeroAssignment = logicalProjection tailAssignment := by
  funext column
  have columnZero : column = (0 : Fin 1) := by
    apply Fin.ext
    omega
  rw [columnZero]
  decide

/-- They are genuinely different on the complete carrier. -/
theorem carrierAssignments_ne : zeroAssignment ≠ tailAssignment := by
  intro equal
  have atTail := congrFun equal tailColumn
  exact (by decide : (0 : F) ≠ 1) (by simpa [zeroAssignment, tailAssignment]
    using atTail)

/-- The all-zero carried assignment has a zero output-one coefficient image. -/
theorem zeroAssignment_coefficientImage_eq_zero :
    coefficientImage zeroAssignment = 0 := by
  have assignmentEq :
      zeroAssignment =
        oneHotAssignment ConcreteCarrier.baseOps tailColumn 0 := by
    funext column
    simp [zeroAssignment, oneHotAssignment, ConcreteCarrier.baseOps]
  unfold coefficientImage
  rw [assignmentEq, matrixVectorAt_oneHot ConcreteCarrier.baseOps
    ConcreteCarrier.baseLaws]
  exact ConcreteCarrier.baseLaws.mul_zero _

/-- Activating only the carried tail lane changes that full derived image to
one, even though the original-width projection is unchanged. -/
theorem tailAssignment_coefficientImage_eq_one :
    coefficientImage tailAssignment = 1 := by
  unfold coefficientImage
  change
    matrixVectorAt ConcreteCarrier.baseOps
        (matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix outputOne)
        (oneHotAssignment ConcreteCarrier.baseOps tailColumn 1) vertex = 1
  rw [matrixVectorAt_oneHot ConcreteCarrier.baseOps ConcreteCarrier.baseLaws]
  rw [tail_coefficient_entry_eq_one]
  exact ConcreteCarrier.baseLaws.one_mul 1

/-- Inclusion-necessity witness: retaining only the original CCS-width
projection cannot determine the carried coefficient image. -/
theorem omitting_completed_carrier_changes_coefficient_image :
    exists left right :
        Assignment F (Phi81CarrierLayout.carrierWidth 1),
      logicalProjection left = logicalProjection right /\
      coefficientImage left ≠ coefficientImage right := by
  refine ⟨zeroAssignment, tailAssignment, logicalProjections_eq, ?_⟩
  rw [zeroAssignment_coefficientImage_eq_zero,
    tailAssignment_coefficientImage_eq_one]
  decide

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier
