import Nightstream.Implementation.Lowering.Goldilocks.Rows
import Nightstream.Implementation.R1CS.Canonical.KLowNorm
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: small structural Goldilocks R1CS atoms used by the Lean-owned
terminal relation compiler.

Assurance tier: model-level.

Owns: one linear-equality row, the two-row strict `b = 2` norm check, their
structural owners, exact local costs, soundness, and honest completion.

Does not own: terminal layout, Ajtai commitments, matrix evaluation, complete
CCS satisfaction, a terminal call recipe, Rust, or generated artifacts.

Emits constraints:
- one row for a linear equality;
- two rows and one auxiliary square column for one strict `b = 2` norm check.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Atoms

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-! ## Linear equality -/

/-- Enforce equality of two sparse linear combinations. -/
def linearCheckRow
    (one : ColumnId)
    (left right : LinearCombination) : Row :=
  ⟨left, singleton one 1, right⟩

/-- Give one linear check its exact physical occurrence identity. -/
def linearCheckOwnedRow
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (one : ColumnId)
    (left right : LinearCombination) : OwnedRow :=
  ⟨⟨owner, ordinal⟩, linearCheckRow one left right⟩

/-- One linear check is exactly one emitted row. -/
def linearCheckRows
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (one : ColumnId)
    (left right : LinearCombination) : List OwnedRow :=
  [linearCheckOwnedRow owner ordinal one left right]

@[simp] theorem linearCheckRows_length
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (one : ColumnId)
    (left right : LinearCombination) :
    (linearCheckRows owner ordinal one left right).length = 1 :=
  rfl

/-- A verifier-owned constant-one column makes the row exactly a linear
equality. -/
theorem linearCheckRow_iff
    (assignment : ColumnId → F)
    (one : ColumnId)
    (left right : LinearCombination)
    (constantOne : assignment one = 1) :
    (linearCheckRow one left right).Holds assignment ↔
      left.eval assignment = right.eval assignment := by
  simp only [linearCheckRow, Row.Holds,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    LinearCombination.eval, constantOne, Fin.mul_one,
    Fin.add_zero]

/-- A linear check allocates no column. -/
def linearCheckCost : Cost :=
  ⟨1, 0, 0, 0⟩

@[simp] theorem linearCheckCost_rows :
    linearCheckCost.recurringRows = 1 :=
  rfl

@[simp] theorem linearCheckCost_auxiliary :
    linearCheckCost.auxiliaryColumns = 0 :=
  rfl

/-! ## Strict `b = 2` norm -/

/-- The two raw equations `x*x = square` and `square*x = x`. -/
def normRawRows (value square : ColumnId) : List Row :=
  [ (CanonicalRow.product square value value).row,
    (CanonicalRow.product value square value).row ]

/-- The same equations with exact structural occurrence identities. -/
def normRows
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square : ColumnId) : List OwnedRow :=
  [ ⟨⟨owner, firstOrdinal⟩,
      (CanonicalRow.product square value value).row⟩,
    ⟨⟨owner, firstOrdinal + 1⟩,
      (CanonicalRow.product value square value).row⟩ ]

/-- The one auxiliary column allocated by a norm check. -/
def normColumns (square : ColumnId) : List OwnedColumn :=
  [⟨square, .auxiliaryColumn⟩]

@[simp] theorem normRows_length
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square : ColumnId) :
    (normRows owner firstOrdinal value square).length = 2 :=
  rfl

@[simp] theorem normColumns_length (square : ColumnId) :
    (normColumns square).length = 1 :=
  rfl

theorem normColumnIds_nodup (square : ColumnId) :
    ((normColumns square).map fun column => column.id).Nodup := by
  simp [normColumns]

theorem normRowIds_nodup
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square : ColumnId) :
    ((normRows owner firstOrdinal value square).map
      fun row => row.id).Nodup := by
  simp [normRows]

/-- Every norm row occurrence belongs to the declared structural owner. -/
theorem normRows_owned
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square : ColumnId)
    (row : OwnedRow)
    (member : row ∈ normRows owner firstOrdinal value square) :
    row.id.owner = owner := by
  simp only [normRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> rfl

/-- The norm atom can mention only the checked coordinate and its square. -/
theorem normRows_supported
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square column : ColumnId)
    (row : OwnedRow)
    (member : row ∈ normRows owner firstOrdinal value square)
    (mentioned : column ∈ row.columnIds) :
    column = value ∨ column = square := by
  simp only [normRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · simpa [OwnedRow.columnIds, Row.columnIds, CanonicalRow.row,
      Nightstream.Implementation.Lowering.Goldilocks.singleton]
      using mentioned
  · simpa [OwnedRow.columnIds, Row.columnIds, CanonicalRow.row, or_comm,
      Nightstream.Implementation.Lowering.Goldilocks.singleton]
      using mentioned

/-- Satisfaction of the two rows forces `x³ = x` in the Goldilocks carrier. -/
theorem normRows_cube
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square : ColumnId)
    (assignment : ColumnId → F)
    (satisfied :
      Satisfies (normRows owner firstOrdinal value square) assignment) :
    assignment value * assignment value * assignment value =
      assignment value := by
  change
    (CanonicalRow.product square value value).row.Holds assignment ∧
      (CanonicalRow.product value square value).row.Holds assignment ∧
        True at satisfied
  have squareEquation :=
    (CanonicalRow.product_iff assignment square value value).mp satisfied.1
  have cubeEquation :=
    (CanonicalRow.product_iff assignment value square value).mp
      satisfied.2.1
  calc
    assignment value * assignment value * assignment value =
        assignment square * assignment value := by
      exact congrArg (fun product => product * assignment value)
        squareEquation
    _ = assignment value := cubeEquation

/-- The physical rows imply the verifier-authoritative strict centered
`b = 2` norm. The only algebraic premise is the same no-zero-divisor boundary
used by the semantic relation. -/
theorem normRows_strictNormTwo
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square : ColumnId)
    (assignment : ColumnId → F)
    (satisfied :
      Satisfies (normRows owner firstOrdinal value square) assignment) :
    centeredMagnitude (assignment value) < 2 := by
  have cube := normRows_cube owner firstOrdinal value square assignment
    satisfied
  have cubeValues := congrArg Fin.val cube
  have cubeNat :
      (assignment value).val * (assignment value).val %
          goldilocksModulus *
        (assignment value).val % goldilocksModulus =
          (assignment value).val := by
    simpa [Fin.val_mul] using cubeValues
  have residual :=
    Nightstream.Implementation.R1CS.Canonical.KLowNorm.cubicResidual_eq_zero_of_cube
      (assignment value).val (assignment value).isLt cubeNat
  exact
    (NormRange.cubicResidual_eq_zero_iff_strictNormTwo
      noZeroDivisors (assignment value)).mp residual

/-- Honest completion writes the square and preserves every other column. -/
def normWitness
    (assignment : ColumnId → F)
    (value square : ColumnId) :
    ColumnId → F :=
  fun column =>
    if column = square then
      assignment value * assignment value
    else
      assignment column

@[simp] theorem normWitness_square
    (assignment : ColumnId → F)
    (value square : ColumnId) :
    normWitness assignment value square square =
      assignment value * assignment value := by
  simp [normWitness]

theorem normWitness_off_square
    (assignment : ColumnId → F)
    (value square column : ColumnId)
    (distinct : column ≠ square) :
    normWitness assignment value square column = assignment column := by
  simp [normWitness, distinct]

theorem normWitness_value
    (assignment : ColumnId → F)
    (value square : ColumnId)
    (fresh : value ≠ square) :
    normWitness assignment value square value = assignment value :=
  normWitness_off_square assignment value square value fresh

/-- Every value in the strict window has an honest satisfying completion.
Freshness is explicit because the auxiliary square must not alias the checked
coordinate. -/
theorem normRows_honest
    (owner : PhysicalOwner)
    (firstOrdinal : Nat)
    (value square : ColumnId)
    (assignment : ColumnId → F)
    (fresh : value ≠ square)
    (bounded : centeredMagnitude (assignment value) < 2) :
    Satisfies (normRows owner firstOrdinal value square)
      (normWitness assignment value square) := by
  have cube :
      assignment value * assignment value * assignment value =
        assignment value := by
    rcases
        (NormRange.strictNormTwo_iff_representedRoot
          (assignment value)).mp bounded with
      negative | zero | one
    · rw [negative]
      decide
    · rw [zero]
      rfl
    · rw [one]
      rfl
  change
    (CanonicalRow.product square value value).row.Holds
        (normWitness assignment value square) ∧
      (CanonicalRow.product value square value).row.Holds
          (normWitness assignment value square) ∧
        True
  constructor
  · apply
      (CanonicalRow.product_iff
        (normWitness assignment value square) square value value).mpr
    rw [normWitness_value assignment value square fresh,
      normWitness_square]
  · constructor
    · apply
        (CanonicalRow.product_iff
          (normWitness assignment value square) value square value).mpr
      rw [normWitness_value assignment value square fresh,
        normWitness_square, cube]
    · trivial

/-- Exact local resource receipt for one norm coordinate. -/
def normCost : Cost :=
  ⟨2, 0, 0, 1⟩

@[simp] theorem normCost_rows :
    normCost.recurringRows = 2 :=
  rfl

@[simp] theorem normCost_auxiliary :
    normCost.auxiliaryColumns = 1 :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Atoms
