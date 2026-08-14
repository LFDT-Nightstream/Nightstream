import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictSound
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4

/-!
Model-level row refinement for one strict radix-four `PiDEC` public-X
coordinate.

Assurance tier: model-level.

Owns: the exact recomposition, shared-sign, two-limb, and digit-reconstruction
row formulas used by the radix-four candidate; the proof that their
satisfaction forces the verifier-computed canonical seven-child split.

Does not own: generated Rust row equality, the complete strict-`PiDEC`
program, the complete recursive relation, or cryptographic soundness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.PiDecRadix4Candidate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.PiDecStrictSound
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.TerminalCeCompiler
open Nightstream.Implementation.R1CS.TerminalCeSound
open Nightstream.SuperNeo.Concrete

abbrev ChildIndex :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.ChildIndex

abbrev BaseField := Nightstream.SuperNeo.Concrete.F

abbrev candidateParams :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.params

abbrev radix4FieldOfNat :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.fieldOfNat

abbrev radix4RecomposeScalar :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recomposeScalar

abbrev radix4SignedDigit :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.signedDigit

abbrev radix4SplitScalar :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.splitScalar

private def child0 : ChildIndex := ⟨0, by decide⟩
private def child1 : ChildIndex := ⟨1, by decide⟩
private def child2 : ChildIndex := ⟨2, by decide⟩
private def child3 : ChildIndex := ⟨3, by decide⟩
private def child4 : ChildIndex := ⟨4, by decide⟩
private def child5 : ChildIndex := ⟨5, by decide⟩
private def child6 : ChildIndex := ⟨6, by decide⟩

private theorem radix4FieldOfNat_four :
    radix4FieldOfNat 4 = (4 : BaseField) := by
  rfl

private theorem radix4FieldOfNat_zero :
    radix4FieldOfNat 0 = (0 : BaseField) := by
  decide

private theorem radix4FieldOfNat_one :
    radix4FieldOfNat 1 = (1 : BaseField) := by
  decide

private theorem radix4FieldOfNat_two :
    radix4FieldOfNat 2 = (2 : BaseField) := by
  decide

private theorem fourSquared (value : BaseField) :
    4 * (4 * value) = 16 * value := by
  calc
    4 * (4 * value) = ((4 : BaseField) * 4) * value :=
      (Fin.mul_assoc _ _ _).symm
    _ = 16 * value := by
      have coefficient : (4 : BaseField) * 4 = 16 := by decide
      rw [coefficient]

private theorem fourCubed (value : BaseField) :
    4 * (4 * (4 * value)) = 64 * value := by
  calc
    4 * (4 * (4 * value)) = 4 * (16 * value) :=
      congrArg (fun current : BaseField => 4 * current) (fourSquared value)
    _ = ((4 : BaseField) * 16) * value := (Fin.mul_assoc _ _ _).symm
    _ = 64 * value := by
      have coefficient : (4 : BaseField) * 16 = 64 := by decide
      rw [coefficient]

private theorem fourFourth (value : BaseField) :
    4 * (4 * (4 * (4 * value))) = 256 * value := by
  calc
    4 * (4 * (4 * (4 * value))) = 4 * (64 * value) :=
      congrArg (fun current : BaseField => 4 * current) (fourCubed value)
    _ = ((4 : BaseField) * 64) * value := (Fin.mul_assoc _ _ _).symm
    _ = 256 * value := by
      have coefficient : (4 : BaseField) * 64 = 256 := by decide
      rw [coefficient]

private theorem fourFifth (value : BaseField) :
    4 * (4 * (4 * (4 * (4 * value)))) = 1024 * value := by
  calc
    4 * (4 * (4 * (4 * (4 * value)))) = 4 * (256 * value) := by
      exact congrArg (fun current : BaseField => 4 * current) (fourFourth value)
    _ = ((4 : BaseField) * 256) * value := (Fin.mul_assoc _ _ _).symm
    _ = 1024 * value := by
      have coefficient : (4 : BaseField) * 256 = 1024 := by decide
      rw [coefficient]

private theorem fourSixth (value : BaseField) :
    4 * (4 * (4 * (4 * (4 * (4 * value))))) = 4096 * value := by
  calc
    4 * (4 * (4 * (4 * (4 * (4 * value))))) = 4 * (1024 * value) := by
      exact congrArg (fun current : BaseField => 4 * current) (fourFifth value)
    _ = ((4 : BaseField) * 1024) * value := (Fin.mul_assoc _ _ _).symm
    _ = 4096 * value := by
      have coefficient : (4 : BaseField) * 1024 = 4096 := by decide
      rw [coefficient]

/-- Exact columns for one public-X coordinate. -/
structure CoordinateColumns where
  parent : Nat
  children : ChildIndex → Nat
  sign : Nat
  product : Nat
  low : ChildIndex → Nat
  high : ChildIndex → Nat

/-- Verifier-owned powers `1, 4, ..., 4^6`. -/
def powers : List Nat :=
  (List.range candidateParams.k).map (fun exponent => 4 ^ exponent)

theorem powers_exact : powers = [1, 4, 16, 64, 256, 1024, 4096] := by
  decide

def childColumns (columns : CoordinateColumns) : List Nat :=
  [columns.children child0, columns.children child1,
   columns.children child2, columns.children child3,
   columns.children child4, columns.children child5,
   columns.children child6]

def recompositionRow (columns : CoordinateColumns) : Row :=
  (recompositionCheck columns.parent (childColumns columns) powers).row

def signRows (columns : CoordinateColumns) : List Row :=
  [⟨[(0, 1), (columns.sign, 1)],
      [(columns.sign, 1)], [(columns.product, 1)]⟩,
   ⟨[(columns.product, 1)],
      [(0, goldilocksP - 1), (columns.sign, 1)], []⟩]

def limbRow (columns : CoordinateColumns) (child : ChildIndex)
    (high : Bool) : Row :=
  let limb := if high then columns.high child else columns.low child
  ⟨[(limb, 1)],
    [(columns.sign, goldilocksP - 1), (limb, 1)], []⟩

def reconstructionRow (columns : CoordinateColumns)
    (child : ChildIndex) : Row :=
  (recompositionCheck (columns.children child)
    [columns.low child, columns.high child] [1, 2]).row

def childRows (columns : CoordinateColumns) (child : ChildIndex) : List Row :=
  [limbRow columns child false,
   limbRow columns child true,
   reconstructionRow columns child]

/-- Exact local row schedule for one radix-four public-X coordinate. -/
def rows (columns : CoordinateColumns) : List Row :=
  recompositionRow columns ::
    signRows columns ++ (List.ofFn (childRows columns)).flatten

/-- Canonical Goldilocks field view of one R1CS assignment value. -/
def fieldAt (assignment : Nat → Nat) (column : Nat) :
    Nightstream.SuperNeo.Concrete.F :=
  ⟨assignment column % goldilocksModulus,
    Nat.mod_lt _ (by decide : 0 < goldilocksModulus)⟩

/-- One Boolean limb as a natural digit bit. -/
def bitNat (value : Bool) : Nat :=
  if value then 1 else 0

/-- Whether one canonical limb assignment is nonzero. -/
def limbBit (assignment : Nat → Nat) (column : Nat) : Bool :=
  decide (assignment column ≠ 0)

/-- The radix-four magnitude digit represented by two Boolean limbs. -/
def digitMagnitude (assignment : Nat → Nat)
    (columns : CoordinateColumns) (child : ChildIndex) : Nat :=
  bitNat (limbBit assignment (columns.low child)) +
    2 * bitNat (limbBit assignment (columns.high child))

theorem digitMagnitude_lt_four (assignment : Nat → Nat)
    (columns : CoordinateColumns) (child : ChildIndex) :
    digitMagnitude assignment columns child < 4 := by
  unfold digitMagnitude
  cases low : limbBit assignment (columns.low child) <;>
    cases high : limbBit assignment (columns.high child) <;>
      decide

private theorem fieldAt_val_of_canonical
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) :
    (fieldAt assignment column).val = assignment column := by
  have valueLt : assignment column < goldilocksModulus := by
    simpa [goldilocksP, goldilocksModulus] using canonical column
  simp [fieldAt, Nat.mod_eq_of_lt valueLt]

private theorem fieldAt_eq_zero
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {column : Nat} (value : assignment column = 0) :
    fieldAt assignment column = 0 := by
  apply Fin.ext
  simp [fieldAt_val_of_canonical canonical, value]

private theorem fieldAt_eq_one
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {column : Nat} (value : assignment column = 1) :
    fieldAt assignment column = 1 := by
  apply Fin.ext
  rw [fieldAt_val_of_canonical canonical column, value]
  decide

private theorem fieldAt_eq_neg_one
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    {column : Nat} (value : assignment column = goldilocksP - 1) :
    fieldAt assignment column = -(1 : BaseField) := by
  apply Fin.ext
  rw [fieldAt_val_of_canonical canonical column, value]
  decide

private theorem signRows_satisfy
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows columns) assignment) :
    Satisfies (signRows columns) assignment := by
  intro row member
  apply satisfies row
  simp [rows, member]

private theorem childRows_satisfy
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows columns) assignment)
    (child : ChildIndex) :
    Satisfies (childRows columns child) assignment := by
  intro row member
  apply satisfies row
  unfold rows
  exact List.mem_cons.mpr <| Or.inr <| List.mem_append.mpr <| Or.inr <|
    List.mem_flatten.mpr
      ⟨childRows columns child, List.mem_ofFn.mpr ⟨child, rfl⟩, member⟩

private theorem recompositionRow_holds
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows columns) assignment) :
    RowHolds assignment (recompositionRow columns) := by
  apply satisfies
  simp [rows]

private theorem lcEval_swap_two (assignment : Nat → Nat)
    (left right : Nat × Nat) :
    lcEval assignment [left, right] = lcEval assignment [right, left] := by
  rw [Program.lcEval_eq_raw_mod, Program.lcEval_eq_raw_mod]
  simp [Program.rawLcEval, Nat.add_comm]

/-- The two shared-sign rows force exactly `0`, `1`, or `-1`. -/
theorem sign_centered
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment) :
    TerminalCeCompiler.CenteredUnit (assignment columns.sign) := by
  apply centeredUnitInstructions_sound
    Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
    canonical one columns.sign columns.product
  intro row member
  simp only [TerminalCeCompiler.centeredUnitInstructions,
    CheckedProgram.rows, List.map_cons, Instruction.row, List.map_nil,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · have exactRow := signRows_satisfy satisfies
      ⟨[(0, 1), (columns.sign, 1)],
        [(columns.sign, 1)], [(columns.product, 1)]⟩ (by simp [signRows])
    change
      lcEval assignment [(columns.sign, 1), (0, 1)] *
          lcEval assignment [(columns.sign, 1)] % goldilocksP =
        lcEval assignment [(columns.product, 1)]
    change
      lcEval assignment [(0, 1), (columns.sign, 1)] *
          lcEval assignment [(columns.sign, 1)] % goldilocksP =
        lcEval assignment [(columns.product, 1)] at exactRow
    rw [lcEval_swap_two]
    exact exactRow
  · have exactRow := signRows_satisfy satisfies
      ⟨[(columns.product, 1)],
        [(0, goldilocksP - 1), (columns.sign, 1)], []⟩
      (by simp [signRows])
    change
      lcEval assignment [(columns.product, 1)] *
          lcEval assignment
            [(columns.sign, 1), (0, goldilocksP - 1)] % goldilocksP =
        lcEval assignment []
    change
      lcEval assignment [(columns.product, 1)] *
          lcEval assignment
            [(0, goldilocksP - 1), (columns.sign, 1)] % goldilocksP =
        lcEval assignment [] at exactRow
    rw [lcEval_swap_two]
    exact exactRow

private theorem limb_choice
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (child : ChildIndex) (high : Bool) :
    let limb := if high then columns.high child else columns.low child
    assignment limb = 0 ∨ assignment limb = assignment columns.sign := by
  let limb := if high then columns.high child else columns.low child
  have holds : RowHolds assignment (limbRow columns child high) := by
    apply childRows_satisfy satisfies child
    cases high <;> simp [childRows]
  have productZero :
      lcEval assignment [(limb, 1)] *
          lcEval assignment
            [(columns.sign, goldilocksP - 1), (limb, 1)] %
        goldilocksP = 0 := by
    simpa [limbRow, limb, RowHolds] using holds
  rcases Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
      _ _ productZero with limbZero | differenceZero
  · left
    simpa [lcEval, Nat.mod_eq_of_lt (canonical limb)] using limbZero
  · right
    have differenceExact :
        lcEval assignment
          [(columns.sign, goldilocksP - 1), (limb, 1)] = 0 := by
      have valueLt : lcEval assignment
          [(columns.sign, goldilocksP - 1), (limb, 1)] < goldilocksP :=
        Nat.mod_lt _ (by decide)
      simpa [Nat.mod_eq_of_lt valueLt] using differenceZero
    have differenceCanonical :
        lcEval assignment
          [(limb, 1), (columns.sign, goldilocksP - 1)] = 0 := by
      simpa [lcEval, Nat.add_comm] using differenceExact
    apply equalityCheck_sound canonical one limb columns.sign
    show lcEval assignment
        [(limb, 1), (columns.sign, goldilocksP - 1)] *
          lcEval assignment [(0, 1)] % goldilocksP =
        lcEval assignment []
    rw [differenceCanonical]
    simp [lcEval, one]

private theorem signedDigit_bits_recompose
    (negative low high : Bool) :
    radix4SignedDigit negative (bitNat low) +
        radix4FieldOfNat 2 * radix4SignedDigit negative (bitNat high) =
      radix4SignedDigit negative (bitNat low + 2 * bitNat high) := by
  cases negative <;> cases low <;> cases high <;> decide

private theorem limb_field_of_sign_zero
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (signZero : assignment columns.sign = 0)
    (child : ChildIndex) (high : Bool) :
    let limb := if high then columns.high child else columns.low child
    fieldAt assignment limb =
      radix4SignedDigit false (bitNat (limbBit assignment limb)) := by
  let limb := if high then columns.high child else columns.low child
  change fieldAt assignment limb =
    radix4SignedDigit false (bitNat (limbBit assignment limb))
  have choice := limb_choice canonical one satisfies child high
  change assignment limb = 0 ∨ assignment limb = assignment columns.sign at choice
  have limbZero : assignment limb = 0 :=
    choice.elim id (fun value => value.trans signZero)
  have fieldZero := fieldAt_eq_zero canonical limbZero
  simpa [limbBit, limbZero, bitNat, radix4SignedDigit,
    radix4FieldOfNat_zero] using fieldZero

private theorem limb_field_of_sign_one
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (signOne : assignment columns.sign = 1)
    (child : ChildIndex) (high : Bool) :
    let limb := if high then columns.high child else columns.low child
    fieldAt assignment limb =
      radix4SignedDigit false (bitNat (limbBit assignment limb)) := by
  let limb := if high then columns.high child else columns.low child
  change fieldAt assignment limb =
    radix4SignedDigit false (bitNat (limbBit assignment limb))
  have choice := limb_choice canonical one satisfies child high
  change assignment limb = 0 ∨ assignment limb = assignment columns.sign at choice
  rcases choice with limbZero | limbSign
  · have fieldZero := fieldAt_eq_zero canonical limbZero
    simpa [limbBit, limbZero, bitNat, radix4SignedDigit,
      radix4FieldOfNat_zero] using fieldZero
  · have limbOne : assignment limb = 1 := limbSign.trans signOne
    have fieldOne := fieldAt_eq_one canonical limbOne
    simpa [limbBit, limbOne, bitNat, radix4SignedDigit,
      radix4FieldOfNat_one] using fieldOne

private theorem limb_field_of_sign_neg_one
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (signNegOne : assignment columns.sign = goldilocksP - 1)
    (child : ChildIndex) (high : Bool) :
    let limb := if high then columns.high child else columns.low child
    fieldAt assignment limb =
      radix4SignedDigit true (bitNat (limbBit assignment limb)) := by
  let limb := if high then columns.high child else columns.low child
  change fieldAt assignment limb =
    radix4SignedDigit true (bitNat (limbBit assignment limb))
  have choice := limb_choice canonical one satisfies child high
  change assignment limb = 0 ∨ assignment limb = assignment columns.sign at choice
  rcases choice with limbZero | limbSign
  · have fieldZero := fieldAt_eq_zero canonical limbZero
    simpa [limbBit, limbZero, bitNat, radix4SignedDigit,
      radix4FieldOfNat_zero] using fieldZero
  · have limbNegOne : assignment limb = goldilocksP - 1 :=
      limbSign.trans signNegOne
    have fieldNegOne := fieldAt_eq_neg_one canonical limbNegOne
    simpa [limbBit, limbNegOne, bitNat, radix4SignedDigit,
      radix4FieldOfNat_one] using fieldNegOne

private theorem reconstruction_sound
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (child : ChildIndex) :
    fieldAt assignment (columns.children child) =
      fieldAt assignment (columns.low child) +
        radix4FieldOfNat 2 * fieldAt assignment (columns.high child) := by
  have holds : RowHolds assignment (reconstructionRow columns child) := by
    apply childRows_satisfy satisfies child
    simp [childRows]
  have equation := recompositionCheck_sound canonical one
    (columns.children child)
    [columns.low child, columns.high child] [1, 2]
    (by simp [goldilocksP]) holds
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [fieldAt_val_of_canonical canonical (columns.children child),
    fieldAt_val_of_canonical canonical (columns.low child),
    fieldAt_val_of_canonical canonical (columns.high child)]
  simpa [Recomposes, lcEval, goldilocksP, goldilocksModulus,
    radix4FieldOfNat,
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.fieldOfNat,
    Fin.val_add, Fin.val_mul, Nat.add_mod, Nat.mul_mod] using equation

private theorem child_field_of_sign_zero
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (signZero : assignment columns.sign = 0)
    (child : ChildIndex) :
    fieldAt assignment (columns.children child) =
      radix4SignedDigit false (digitMagnitude assignment columns child) := by
  rw [reconstruction_sound canonical one satisfies child]
  have lowExact : fieldAt assignment (columns.low child) =
      radix4SignedDigit false
        (bitNat (limbBit assignment (columns.low child))) := by
    simpa using limb_field_of_sign_zero canonical one satisfies signZero
      child false
  have highExact : fieldAt assignment (columns.high child) =
      radix4SignedDigit false
        (bitNat (limbBit assignment (columns.high child))) := by
    simpa using limb_field_of_sign_zero canonical one satisfies signZero
      child true
  rw [lowExact, highExact]
  change _ = radix4SignedDigit false
    (bitNat (limbBit assignment (columns.low child)) +
      2 * bitNat (limbBit assignment (columns.high child)))
  exact signedDigit_bits_recompose false
    (limbBit assignment (columns.low child))
    (limbBit assignment (columns.high child))

private theorem child_field_of_sign_one
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (signOne : assignment columns.sign = 1)
    (child : ChildIndex) :
    fieldAt assignment (columns.children child) =
      radix4SignedDigit false (digitMagnitude assignment columns child) := by
  rw [reconstruction_sound canonical one satisfies child]
  have lowExact : fieldAt assignment (columns.low child) =
      radix4SignedDigit false
        (bitNat (limbBit assignment (columns.low child))) := by
    simpa using limb_field_of_sign_one canonical one satisfies signOne
      child false
  have highExact : fieldAt assignment (columns.high child) =
      radix4SignedDigit false
        (bitNat (limbBit assignment (columns.high child))) := by
    simpa using limb_field_of_sign_one canonical one satisfies signOne
      child true
  rw [lowExact, highExact]
  change _ = radix4SignedDigit false
    (bitNat (limbBit assignment (columns.low child)) +
      2 * bitNat (limbBit assignment (columns.high child)))
  exact signedDigit_bits_recompose false
    (limbBit assignment (columns.low child))
    (limbBit assignment (columns.high child))

private theorem child_field_of_sign_neg_one
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (signNegOne : assignment columns.sign = goldilocksP - 1)
    (child : ChildIndex) :
    fieldAt assignment (columns.children child) =
      radix4SignedDigit true (digitMagnitude assignment columns child) := by
  rw [reconstruction_sound canonical one satisfies child]
  have lowExact : fieldAt assignment (columns.low child) =
      radix4SignedDigit true
        (bitNat (limbBit assignment (columns.low child))) := by
    simpa using limb_field_of_sign_neg_one canonical one satisfies signNegOne
      child false
  have highExact : fieldAt assignment (columns.high child) =
      radix4SignedDigit true
        (bitNat (limbBit assignment (columns.high child))) := by
    simpa using limb_field_of_sign_neg_one canonical one satisfies signNegOne
      child true
  rw [lowExact, highExact]
  change _ = radix4SignedDigit true
    (bitNat (limbBit assignment (columns.low child)) +
      2 * bitNat (limbBit assignment (columns.high child)))
  exact signedDigit_bits_recompose true
    (limbBit assignment (columns.low child))
    (limbBit assignment (columns.high child))

private theorem recomposition_sound
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment) :
    fieldAt assignment columns.parent =
      radix4RecomposeScalar
        (fun child => fieldAt assignment (columns.children child)) := by
  have equation := recompositionCheck_sound canonical one
    columns.parent (childColumns columns) powers
    (by
      intro coefficient member
      rw [powers_exact] at member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        decide)
    (recompositionRow_holds satisfies)
  let childValue : ChildIndex → BaseField :=
    fun child => fieldAt assignment (columns.children child)
  have weighted : fieldAt assignment columns.parent =
      childValue child0 + 4 * childValue child1 + 16 * childValue child2 +
        64 * childValue child3 + 256 * childValue child4 +
          1024 * childValue child5 + 4096 * childValue child6 := by
    apply Fin.ext
    simp only [Fin.val_add, Fin.val_mul]
    simp only [childValue, fieldAt_val_of_canonical canonical]
    simpa [Recomposes, childColumns, powers_exact, lcEval, goldilocksP,
      goldilocksModulus, Nat.add_mod, Nat.mul_mod] using equation
  rw [weighted]
  change _ =
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recomposeScalar
      (fun child => fieldAt assignment (columns.children child))
  rw [Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recomposeScalar_seven]
  change
    childValue child0 + 4 * childValue child1 + 16 * childValue child2 +
          64 * childValue child3 + 256 * childValue child4 +
        1024 * childValue child5 + 4096 * childValue child6 =
      childValue child0 + radix4FieldOfNat 4 *
        (childValue child1 + radix4FieldOfNat 4 *
          (childValue child2 + radix4FieldOfNat 4 *
            (childValue child3 + radix4FieldOfNat 4 *
              (childValue child4 + radix4FieldOfNat 4 *
                (childValue child5 + radix4FieldOfNat 4 * childValue child6)))))
  rw [radix4FieldOfNat_four]
  simp only [Lean.Grind.Fin.left_distrib]
  rw [fourSixth, fourFifth, fourFourth, fourCubed, fourSquared]
  simp only [Lean.Grind.Fin.add_assoc]

private theorem canonical_split_of_child_fields
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment)
    (negative : Bool)
    (childExact : ∀ child,
      fieldAt assignment (columns.children child) =
        radix4SignedDigit negative
          (digitMagnitude assignment columns child)) :
    ∀ child,
      fieldAt assignment (columns.children child) =
        radix4SplitScalar (fieldAt assignment columns.parent) child := by
  let digits : ChildIndex → Nat :=
    fun child => digitMagnitude assignment columns child
  have recomposes :
      radix4RecomposeScalar
          (fun child => radix4SignedDigit negative (digits child)) =
        fieldAt assignment columns.parent := by
    calc
      radix4RecomposeScalar
          (fun child => radix4SignedDigit negative (digits child)) =
          radix4RecomposeScalar
            (fun child => fieldAt assignment (columns.children child)) := by
              apply congrArg radix4RecomposeScalar
              funext child
              exact (childExact child).symm
      _ = fieldAt assignment columns.parent :=
        (recomposition_sound canonical one satisfies).symm
  have canonicalChildren :=
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.splitScalar_eq_signed_of_recompose
      (fieldAt assignment columns.parent) negative digits
      (digitMagnitude_lt_four assignment columns) recomposes
  intro child
  exact (childExact child).trans (canonicalChildren child).symm

/-- The exact 24 local rows force the unique canonical seven-child
radix-four decomposition of the parent coordinate. -/
theorem rows_force_canonical_split
    {columns : CoordinateColumns} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows columns) assignment) :
    ∀ child,
      fieldAt assignment (columns.children child) =
        radix4SplitScalar (fieldAt assignment columns.parent) child := by
  rcases sign_centered canonical one satisfies with
    signZero | signOne | signNegOne
  · exact canonical_split_of_child_fields canonical one satisfies false
      (child_field_of_sign_zero canonical one satisfies signZero)
  · exact canonical_split_of_child_fields canonical one satisfies false
      (child_field_of_sign_one canonical one satisfies signOne)
  · exact canonical_split_of_child_fields canonical one satisfies true
      (child_field_of_sign_neg_one canonical one satisfies signNegOne)

end Nightstream.Implementation.R1CS.PiDecRadix4Candidate
