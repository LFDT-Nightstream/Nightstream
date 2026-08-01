import Nightstream.Implementation.Lowering.Nebula.Layout

/-!
Physical sparse-row vocabulary for the Lean-owned Nebula CCS relation.

Assurance tier: model-level.

Owns: sparse linear combinations over numeric logical columns, fifteen named
matrix images per row, stable positional row owners, exact row satisfaction,
and the five primitive row constructors used by the stackless compiler.

Does not own: a concrete row schedule, witness construction, application
ports, transcript binding, a proof-free manifest, Rust, or costs.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Nebula.Rows

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.StepPolynomial

structure Term where
  column : Nat
  coefficient : F
deriving DecidableEq, Repr

abbrev LinearCombination := List Term

namespace LinearCombination

def eval (assignment : Nat -> F) : LinearCombination -> F
  | [] => 0
  | term :: rest =>
      term.coefficient * assignment term.column + eval assignment rest

def zero : LinearCombination := []

def constant (coefficient : F) : LinearCombination :=
  [{ column := 0, coefficient := coefficient }]

def bit (column : Nat) : LinearCombination :=
  [{ column := column, coefficient := 1 }]

def fieldTwoPower : Nat -> F
  | 0 => 1
  | exponent + 1 => fieldTwoPower exponent * 2

def wordScaled (start width : Nat) (scale : F) : LinearCombination :=
  (List.range width).map fun bitIndex =>
    { column := start + bitIndex
      coefficient := scale * fieldTwoPower bitIndex }

def word (start width : Nat) : LinearCombination :=
  wordScaled start width 1

def add (left right : LinearCombination) : LinearCombination :=
  left ++ right

def scale (coefficient : F)
    (combination : LinearCombination) : LinearCombination :=
  combination.map fun term =>
    { term with coefficient := coefficient * term.coefficient }

def neg (combination : LinearCombination) : LinearCombination :=
  scale (-1) combination

def sub (left right : LinearCombination) : LinearCombination :=
  add left (neg right)

@[simp] theorem eval_zero (assignment : Nat -> F) :
    eval assignment zero = 0 := rfl

@[simp] theorem eval_constant (assignment : Nat -> F) (coefficient : F) :
    eval assignment (constant coefficient) =
      coefficient * assignment 0 := by
  simp [constant, eval]

@[simp] theorem eval_bit (assignment : Nat -> F) (column : Nat) :
    eval assignment (bit column) = assignment column := by
  simp [bit, eval, Fin.one_mul]

theorem eval_add (assignment : Nat -> F) (left right : LinearCombination) :
    eval assignment (add left right) =
      eval assignment left + eval assignment right := by
  induction left with
  | nil => simp [add, eval]
  | cons head tail inductionHypothesis =>
      unfold add at inductionHypothesis ⊢
      simp only [List.cons_append, eval]
      rw [inductionHypothesis, Lean.Grind.Fin.add_assoc]

private theorem mul_assoc (left middle right : F) :
    (left * middle) * right = left * (middle * right) :=
  Fin.mul_assoc _ _ _

private theorem mul_add (left middle right : F) :
    left * (middle + right) = left * middle + left * right :=
  Lean.Grind.Fin.left_distrib _ _ _

theorem eval_scale (assignment : Nat -> F) (coefficient : F)
    (combination : LinearCombination) :
    eval assignment (scale coefficient combination) =
      coefficient * eval assignment combination := by
  induction combination with
  | nil => simp [scale, eval, Fin.mul_zero]
  | cons head tail inductionHypothesis =>
      change
        (coefficient * head.coefficient) * assignment head.column +
            eval assignment (scale coefficient tail) =
          coefficient *
            (head.coefficient * assignment head.column +
              eval assignment tail)
      rw [inductionHypothesis]
      rw [mul_add, mul_assoc]

theorem eval_neg (assignment : Nat -> F)
    (combination : LinearCombination) :
    eval assignment (neg combination) = -eval assignment combination := by
  rw [show neg combination = scale (-1) combination from rfl,
    eval_scale]
  calc
    (-1 : F) * eval assignment combination =
        -(1 * eval assignment combination) :=
      Lean.Grind.Fin.neg_mul _ _
    _ = -eval assignment combination := by rw [Fin.one_mul]

theorem eval_sub (assignment : Nat -> F)
    (left right : LinearCombination) :
    eval assignment (sub left right) =
      eval assignment left + -eval assignment right := by
  rw [show sub left right = add left (neg right) from rfl,
    eval_add, eval_neg]

def columns (combination : LinearCombination) : List Nat :=
  combination.map Term.column

end LinearCombination

/-- Exact fifteen matrix images for one CCS row. -/
structure Images where
  bit : LinearCombination := []
  productLeft : LinearCombination := []
  productRight : LinearCombination := []
  linearLeft : LinearCombination := []
  linearRight : LinearCombination := []
  output : LinearCombination := []
  extensionA : LinearCombination := []
  extensionB : LinearCombination := []
  pad : LinearCombination := []
  active : LinearCombination := []
  fingerprintA : LinearCombination := []
  fingerprintB : LinearCombination := []
  valueA : LinearCombination := []
  valueB : LinearCombination := []
  value : LinearCombination := []
deriving DecidableEq, Repr

@[simp] def Images.at (images : Images) : Role -> LinearCombination
  | .bit => images.bit
  | .productLeft => images.productLeft
  | .productRight => images.productRight
  | .linearLeft => images.linearLeft
  | .linearRight => images.linearRight
  | .output => images.output
  | .extensionA => images.extensionA
  | .extensionB => images.extensionB
  | .pad => images.pad
  | .active => images.active
  | .fingerprintA => images.fingerprintA
  | .fingerprintB => images.fingerprintB
  | .valueA => images.valueA
  | .valueB => images.valueB
  | .value => images.value

def Images.columns (images : Images) : List Nat :=
  [ images.bit, images.productLeft, images.productRight,
    images.linearLeft, images.linearRight, images.output,
    images.extensionA, images.extensionB, images.pad, images.active,
    images.fingerprintA, images.fingerprintB, images.valueA,
    images.valueB, images.value ].flatMap LinearCombination.columns

/-- Physical equation family. The family and coordinates form the stable row
occurrence identity; equal equations at different positions stay distinct. -/
inductive Family where
  | filler
  | operationBit
  | operationCount
  | readWrite
  | timestampOrder
  | romWrite
  | romRange
  | padding
  | readProduct
  | writeProduct
  | initialScanBit
  | finalScanBit
  | initialScanProduct
  | finalScanProduct
  | boundaryTimestamp
  | boundaryProduct
deriving DecidableEq, Repr

structure RowId where
  family : Family
  slot : Nat
  component : Nat
  ordinal : Nat
  /-- Global emitted-row position. Primitive row constructors leave this at
  zero; the complete compiler assigns the unique physical position. -/
  position : Nat := 0
deriving DecidableEq, Repr

structure Row where
  id : RowId
  images : Images
deriving DecidableEq, Repr

namespace Row

def withPosition (position : Nat) (row : Row) : Row :=
  { row with id := { row.id with position := position } }

@[simp] theorem withPosition_id_position (position : Nat) (row : Row) :
    (row.withPosition position).id.position = position :=
  rfl

@[simp] theorem withPosition_images (position : Nat) (row : Row) :
    (row.withPosition position).images = row.images :=
  rfl

def point (row : Row) (assignment : Nat -> F) : Fin matrixCount -> F :=
  fun matrix => LinearCombination.eval assignment
    (row.images.at (Role.ofIndex matrix))

def Holds (row : Row) (assignment : Nat -> F) : Prop :=
  StepPolynomial.evaluate (row.point assignment) = 0

instance (row : Row) (assignment : Nat -> F) :
    Decidable (row.Holds assignment) := by
  unfold Holds
  infer_instance

def columns (row : Row) : List Nat := row.images.columns

@[simp] theorem withPosition_holds_iff
    (position : Nat) (row : Row) (assignment : Nat -> F) :
    (row.withPosition position).Holds assignment ↔ row.Holds assignment :=
  Iff.rfl

end Row

def Satisfies (rows : List Row) (assignment : Nat -> F) : Prop :=
  ∀ row, row ∈ rows -> row.Holds assignment

def bitRow (id : RowId) (column : Nat) : Row where
  id := id
  images := { bit := LinearCombination.bit column }

def productRow (id : RowId)
    (left right : LinearCombination) : Row where
  id := id
  images := { productLeft := left, productRight := right }

def productEqualityRow (id : RowId)
    (left right output : LinearCombination) : Row where
  id := id
  images := {
    productLeft := left
    productRight := right
    linearRight := output
  }

def linearRow (id : RowId)
    (left right : LinearCombination) : Row where
  id := id
  images := { linearLeft := left, linearRight := right }

def extensionUpdateRow (id : RowId)
    (output a b pad active fingerprintA fingerprintB valueA valueB value :
      LinearCombination) : Row where
  id := id
  images := {
    output := output
    extensionA := a
    extensionB := b
    pad := pad
    active := active
    fingerprintA := fingerprintA
    fingerprintB := fingerprintB
    valueA := valueA
    valueB := valueB
    value := value
  }

theorem bitRow_holds_iff (id : RowId) (column : Nat)
    (assignment : Nat -> F) :
    (bitRow id column).Holds assignment ↔
      assignment column * assignment column + -assignment column = 0 := by
  unfold Row.Holds Row.point bitRow
  rw [StepPolynomial.evaluate_eq_residual]
  simp [StepPolynomial.residual, Images.at, Role.ofIndex, matrixCount,
    LinearCombination.eval, LinearCombination.bit, Fin.mul_zero,
    Fin.one_mul, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem productRow_holds_iff (id : RowId)
    (left right : LinearCombination) (assignment : Nat -> F) :
    (productRow id left right).Holds assignment ↔
      LinearCombination.eval assignment left *
        LinearCombination.eval assignment right = 0 := by
  unfold Row.Holds Row.point productRow
  rw [StepPolynomial.evaluate_eq_residual]
  simp [StepPolynomial.residual, Images.at, Role.ofIndex, matrixCount,
    LinearCombination.eval, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem productEqualityRow_holds_iff (id : RowId)
    (left right output : LinearCombination) (assignment : Nat -> F) :
    (productEqualityRow id left right output).Holds assignment ↔
      LinearCombination.eval assignment left *
          LinearCombination.eval assignment right +
        -LinearCombination.eval assignment output = 0 := by
  unfold Row.Holds Row.point productEqualityRow
  rw [StepPolynomial.evaluate_eq_residual]
  simp [StepPolynomial.residual, Images.at, Role.ofIndex, matrixCount,
    LinearCombination.eval, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem linearRow_holds_iff (id : RowId)
    (left right : LinearCombination) (assignment : Nat -> F) :
    (linearRow id left right).Holds assignment ↔
      LinearCombination.eval assignment left +
        -LinearCombination.eval assignment right = 0 := by
  unfold Row.Holds Row.point linearRow
  rw [StepPolynomial.evaluate_eq_residual]
  simp [StepPolynomial.residual, Images.at, Role.ofIndex, matrixCount,
    LinearCombination.eval, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

theorem extensionUpdateRow_holds_iff (id : RowId)
    (output a b pad active fingerprintA fingerprintB valueA valueB value :
      LinearCombination)
    (assignment : Nat -> F) :
    (extensionUpdateRow id output a b pad active fingerprintA fingerprintB
      valueA valueB value).Holds assignment ↔
      -LinearCombination.eval assignment output +
        LinearCombination.eval assignment a *
          LinearCombination.eval assignment pad +
        LinearCombination.eval assignment a *
          LinearCombination.eval assignment active *
          LinearCombination.eval assignment fingerprintA +
        -(LinearCombination.eval assignment a *
          LinearCombination.eval assignment active *
          LinearCombination.eval assignment valueA *
          LinearCombination.eval assignment value) +
        LinearCombination.eval assignment b *
          LinearCombination.eval assignment active *
          LinearCombination.eval assignment fingerprintB +
        -(LinearCombination.eval assignment b *
          LinearCombination.eval assignment active *
          LinearCombination.eval assignment valueB *
          LinearCombination.eval assignment value) = 0 := by
  unfold Row.Holds Row.point extensionUpdateRow
  rw [StepPolynomial.evaluate_eq_residual]
  simp [StepPolynomial.residual, Images.at, Role.ofIndex, matrixCount,
    LinearCombination.eval, Fin.mul_zero, Fin.zero_add, Fin.add_zero,
    Lean.Grind.AddCommGroup.neg_zero]

end Nightstream.Implementation.Lowering.Nebula.Rows
