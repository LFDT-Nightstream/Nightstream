import Mathlib.Tactic.Ring
import Nightstream.Implementation.R1CS.Correspondence.TerminalR1cs.Atoms
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: one quadratic-extension multiplication over structural Goldilocks
R1CS rows.

Assurance tier: model-level.

Owns: the three Karatsuba rows, three auxiliary columns, exact carried output,
structural ownership, cost, soundness, and completeness for a prefilled honest
witness.

Does not own: frame placement, multiplication chains, matrix evaluation,
terminal assembly, Rust, or extension-field nonresidue security.

Emits constraints: three product rows and three auxiliary columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.TerminalR1cs.Extension

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS.TerminalR1cs
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- One carried quadratic-extension value. -/
structure Value where
  low : LinearCombination
  high : LinearCombination

/-- Three Karatsuba product columns. -/
structure Frame where
  lowLow : ColumnId
  highHigh : ColumnId
  cross : ColumnId

/-- Coordinatewise sum used by the cross product. -/
def sumValue (value : Value) : LinearCombination :=
  value.low ++ value.high

/-- Exact three-row multiplication program. -/
def rows
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (left right : Value)
    (frame : Frame) : List OwnedRow :=
  [ ⟨⟨owner, firstOrdinal⟩,
      ⟨left.low, right.low,
        Nightstream.Implementation.Lowering.Goldilocks.singleton
          frame.lowLow 1⟩⟩,
    ⟨⟨owner, firstOrdinal + 1⟩,
      ⟨left.high, right.high,
        Nightstream.Implementation.Lowering.Goldilocks.singleton
          frame.highHigh 1⟩⟩,
    ⟨⟨owner, firstOrdinal + 2⟩,
      ⟨sumValue left, sumValue right,
        Nightstream.Implementation.Lowering.Goldilocks.singleton
          frame.cross 1⟩⟩ ]

/-- Exact allocated column family. -/
def columns (frame : Frame) : List OwnedColumn :=
  [ ⟨frame.lowLow, .auxiliaryColumn⟩,
    ⟨frame.highHigh, .auxiliaryColumn⟩,
    ⟨frame.cross, .auxiliaryColumn⟩ ]

@[simp] theorem rows_length
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (left right : Value)
    (frame : Frame) :
    (rows owner firstOrdinal left right frame).length = 3 :=
  rfl

@[simp] theorem columns_length (frame : Frame) :
    (columns frame).length = 3 :=
  rfl

/-- A frame is physical only when all three allocated columns differ. -/
structure Distinct (frame : Frame) : Prop where
  lowNeHigh : frame.lowLow ≠ frame.highHigh
  lowNeCross : frame.lowLow ≠ frame.cross
  highNeCross : frame.highHigh ≠ frame.cross

theorem columnIds_nodup (frame : Frame)
    (distinct : Distinct frame) :
    ((columns frame).map fun column => column.id).Nodup := by
  simp [columns, distinct.lowNeHigh, distinct.lowNeCross,
    distinct.highNeCross]

theorem rowIds_nodup
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (left right : Value)
    (frame : Frame) :
    ((rows owner firstOrdinal left right frame).map
      fun owned => owned.id).Nodup := by
  simp [rows]

theorem rows_owned
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (left right : Value)
    (frame : Frame)
    (owned : OwnedRow)
    (member : owned ∈ rows owner firstOrdinal left right frame) :
    owned.id.owner = owner := by
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl <;> rfl

private theorem eval_append
    (assignment : ColumnId → F)
    (left right : LinearCombination) :
    LinearCombination.eval assignment (left ++ right) =
      LinearCombination.eval assignment left +
        LinearCombination.eval assignment right := by
  induction left with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, LinearCombination.eval_cons,
        inductionHypothesis]
      exact (Lean.Grind.Fin.add_assoc _ _ _).symm

/-- Low carried coordinate of the product. -/
def outLow (frame : Frame) : LinearCombination :=
  [ ⟨frame.lowLow, 1⟩, ⟨frame.highHigh, 7⟩ ]

/-- High carried coordinate of the product. -/
def outHigh (frame : Frame) : LinearCombination :=
  [ ⟨frame.cross, 1⟩, ⟨frame.lowLow, -1⟩,
    ⟨frame.highHigh, -1⟩ ]

/-- Carried product value. -/
def output (frame : Frame) : Value :=
  ⟨outLow frame, outHigh frame⟩

private theorem singleton_eval
    (assignment : ColumnId → F)
    (column : ColumnId) :
    LinearCombination.eval assignment
        (Nightstream.Implementation.Lowering.Goldilocks.singleton column 1) =
      assignment column := by
  simp [Nightstream.Implementation.Lowering.Goldilocks.singleton,
    LinearCombination.eval, Fin.one_mul]

private theorem frame_products
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (left right : Value)
    (frame : Frame)
    (assignment : ColumnId → F)
    (satisfied :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (rows owner firstOrdinal left right frame) assignment) :
    assignment frame.lowLow =
        left.low.eval assignment * right.low.eval assignment ∧
      assignment frame.highHigh =
        left.high.eval assignment * right.high.eval assignment ∧
      assignment frame.cross =
        (sumValue left).eval assignment *
          (sumValue right).eval assignment := by
  change
    (left.low.eval assignment * right.low.eval assignment =
        (Nightstream.Implementation.Lowering.Goldilocks.singleton
          frame.lowLow 1).eval assignment) ∧
      (left.high.eval assignment * right.high.eval assignment =
        (Nightstream.Implementation.Lowering.Goldilocks.singleton
          frame.highHigh 1).eval assignment) ∧
      ((sumValue left).eval assignment *
        (sumValue right).eval assignment =
          (Nightstream.Implementation.Lowering.Goldilocks.singleton
            frame.cross 1).eval assignment) ∧
      True at satisfied
  rw [singleton_eval, singleton_eval, singleton_eval] at satisfied
  exact ⟨satisfied.1.symm, satisfied.2.1.symm, satisfied.2.2.1.symm⟩

/-- The three physical rows compute exact multiplication in
`K = F[X]/(X² - 7)`. -/
theorem rows_sound
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (left right : Value)
    (frame : Frame)
    (assignment : ColumnId → F)
    (satisfied :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (rows owner firstOrdinal left right frame) assignment) :
    let leftValue : K :=
      ⟨left.low.eval assignment, left.high.eval assignment⟩
    let rightValue : K :=
      ⟨right.low.eval assignment, right.high.eval assignment⟩
    let outputValue : K :=
      ⟨(outLow frame).eval assignment,
        (outHigh frame).eval assignment⟩
    outputValue = K.mul leftValue rightValue := by
  dsimp
  rcases frame_products owner firstOrdinal left right frame assignment
    satisfied with ⟨lowLow, highHigh, cross⟩
  have leftSum :
      (sumValue left).eval assignment =
        left.low.eval assignment + left.high.eval assignment :=
    eval_append assignment left.low left.high
  have rightSum :
      (sumValue right).eval assignment =
        right.low.eval assignment + right.high.eval assignment :=
    eval_append assignment right.low right.high
  change K.mk _ _ = K.mk _ _
  congr 1
  · simp only [outLow, LinearCombination.eval_cons,
      LinearCombination.eval_nil, Fin.add_zero, Fin.one_mul]
    rw [lowLow, highHigh]
    exact congrArg
      (fun value : F =>
        left.low.eval assignment * right.low.eval assignment + value)
      (ConcreteCarrier.baseLaws.mul_assoc 7
        (left.high.eval assignment)
        (right.high.eval assignment)).symm
  · simp only [outHigh, LinearCombination.eval_cons,
      LinearCombination.eval_nil, Fin.add_zero, Fin.one_mul,
      Lean.Grind.Fin.neg_mul]
    rw [cross, lowLow, highHigh, leftSum, rightSum]
    let a := left.low.eval assignment
    let b := left.high.eval assignment
    let c := right.low.eval assignment
    let d := right.high.eval assignment
    change
      (a + b) * (c + d) + (-(a * c) + -(b * d)) =
        a * d + b * c
    have expanded :
        (a + b) * (c + d) =
          (a * c + a * d) + (b * c + b * d) := by
      calc
        (a + b) * (c + d) =
            (c + d) * (a + b) := Fin.mul_comm _ _
        _ = (c + d) * a + (c + d) * b :=
          Lean.Grind.Fin.left_distrib _ _ _
        _ = a * (c + d) + b * (c + d) := by
          rw [Fin.mul_comm (c + d) a, Fin.mul_comm (c + d) b]
        _ = (a * c + a * d) + (b * c + b * d) := by
          rw [Lean.Grind.Fin.left_distrib,
            Lean.Grind.Fin.left_distrib]
    have cancelAC : a * c + -(a * c) = 0 := by
      calc
        a * c + -(a * c) = -(a * c) + a * c :=
          Lean.Grind.Fin.add_comm _ _
        _ = 0 := Lean.Grind.Fin.neg_add_cancel _
    have cancelBD : b * d + -(b * d) = 0 := by
      calc
        b * d + -(b * d) = -(b * d) + b * d :=
          Lean.Grind.Fin.add_comm _ _
        _ = 0 := Lean.Grind.Fin.neg_add_cancel _
    letI : Std.Associative (fun (left right : F) => left + right) :=
      ⟨ConcreteCarrier.baseLaws.add_assoc⟩
    letI : Std.Commutative (fun (left right : F) => left + right) :=
      ⟨ConcreteCarrier.baseLaws.add_comm⟩
    rw [expanded]
    calc
      ((a * c + a * d) + (b * c + b * d)) +
          (-(a * c) + -(b * d)) =
          (a * c + -(a * c)) + (b * d + -(b * d)) +
            (a * d + b * c) := by
        ac_rfl
      _ = (0 : F) + 0 + (a * d + b * c) := by
        rw [cancelAC, cancelBD]
      _ = a * d + b * c := by
        rw [Fin.zero_add, Fin.zero_add]

/-- If the three frame products are prefilled honestly, all rows hold. -/
theorem rows_honest
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (left right : Value)
    (frame : Frame)
    (assignment : ColumnId → F)
    (lowLow :
      assignment frame.lowLow =
        left.low.eval assignment * right.low.eval assignment)
    (highHigh :
      assignment frame.highHigh =
        left.high.eval assignment * right.high.eval assignment)
    (cross :
      assignment frame.cross =
        (sumValue left).eval assignment *
          (sumValue right).eval assignment) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rows owner firstOrdinal left right frame) assignment := by
  change
    (left.low.eval assignment * right.low.eval assignment =
        (Nightstream.Implementation.Lowering.Goldilocks.singleton
          frame.lowLow 1).eval assignment) ∧
      (left.high.eval assignment * right.high.eval assignment =
        (Nightstream.Implementation.Lowering.Goldilocks.singleton
          frame.highHigh 1).eval assignment) ∧
      ((sumValue left).eval assignment *
        (sumValue right).eval assignment =
          (Nightstream.Implementation.Lowering.Goldilocks.singleton
            frame.cross 1).eval assignment) ∧
      True
  simp only [singleton_eval]
  exact ⟨lowLow.symm, highHigh.symm, cross.symm, trivial⟩

/-- Exact local resource receipt. -/
def cost : Cost :=
  ⟨3, 0, 0, 3⟩

@[simp] theorem cost_rows :
    cost.recurringRows = 3 :=
  rfl

@[simp] theorem cost_auxiliary :
    cost.auxiliaryColumns = 3 :=
  rfl

end Nightstream.Implementation.R1CS.TerminalR1cs.Extension
