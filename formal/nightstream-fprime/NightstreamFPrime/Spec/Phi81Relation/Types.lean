import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81Evaluation

/-! SuperNeo v1.2 Definitions 19–21. -/

namespace NightstreamFPrime.Spec.Phi81Relation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

structure Shape where
  rowVariables : Nat
  logicalWidth : Nat
  matrixCount : Nat
  publicRingColumns : Nat
  publicFits :
    ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth

namespace Shape

@[implicit_reducible]
def carrierWidth (shape : Shape) : Nat :=
  Phi81CarrierLayout.carrierWidth shape.logicalWidth

def publicWidth (shape : Shape) : Nat :=
  ringDegree * shape.publicRingColumns

@[implicit_reducible]
def sourceShape (shape : Shape) :
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Shape :=
  Phi81MatrixSource.phi81Shape shape.rowVariables 0 0 shape.matrixCount

theorem publicWidth_aligned (shape : Shape) :
    ringDegree ∣ shape.publicWidth := by
  exact ⟨shape.publicRingColumns, rfl⟩

theorem publicWidth_ne_257 (shape : Shape) :
    shape.publicWidth ≠ 257 := by
  simp only [publicWidth, ringDegree]
  omega

def publicColumn (shape : Shape)
    (column : Fin shape.publicWidth) : Fin shape.carrierWidth :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt shape.publicFits⟩

@[simp] theorem publicColumn_val (shape : Shape)
    (column : Fin shape.publicWidth) :
    (shape.publicColumn column).val = column.val := by
  rfl

end Shape

abbrev Assignment (shape : Shape) :=
  PaperLinearAlgebra.Assignment F shape.carrierWidth

abbrev PublicInput (shape : Shape) :=
  PaperLinearAlgebra.Assignment F shape.publicWidth

abbrev Point (shape : Shape) := CubePoint K shape.rowVariables

abbrev Evaluation := RingK

def projectPublicInput {shape : Shape}
    (assignment : Assignment shape) : PublicInput shape :=
  fun column => assignment (shape.publicColumn column)

structure Structure (shape : Shape) where
  matrices : Fin shape.matrixCount ->
    BooleanMatrix F shape.rowVariables shape.logicalWidth
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount

namespace Structure

def matrixSource {shape : Shape}
    (system : Structure shape) :
    MatrixCoefficientSource.MatrixSource F shape.sourceShape shape.carrierWidth
      (Phi81ColumnLayout.blockCount shape.carrierWidth) :=
  Phi81MatrixSource.source shape.rowVariables 0 0 shape.matrixCount
    shape.logicalWidth system.matrices system.constraintPolynomial

@[simp] theorem matrixSource_kernel_eq {shape : Shape}
    (system : Structure shape) :
    system.matrixSource.kernel = Phi81CoefficientKernel.phi81Kernel := by
  rfl

end Structure

end NightstreamFPrime.Spec.Phi81Relation
