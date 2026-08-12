import Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge

/-!
Contract: semantics-preserving lowering from stable typed Goldilocks rows to
the repository's numeric Goldilocks R1CS rows.

One verifier-owned column allocation maps every typed column to a numeric
column. Numeric satisfaction is equivalent to typed satisfaction on the
canonical field view of the same numeric assignment. If the allocation is
injective, every typed satisfying assignment has a canonical numeric lift.

Assurance tier: model-level row-lowering equivalence.

This module does not select a concrete allocation, prove finite column bounds,
place the rows in a generated artifact, or prove Rust/backend refinement.

Emits constraints: no. It translates each supplied row occurrence exactly
once and preserves source order.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.SuperNeo.Concrete

abbrev NumericRow := Nightstream.Implementation.R1CS.Row

/-- Interpret one numeric assignment through a fixed typed-column allocation.
Every numeric value is reduced to its canonical Goldilocks field value. -/
def typedAssignment
    (columnIndex : ColumnId -> Nat)
    (assignment : Nat -> Nat) :
    ColumnId -> F :=
  fun column => residue (assignment (columnIndex column))

/-- Lower one typed sparse term without changing its coefficient value. -/
def numericTerm
    (columnIndex : ColumnId -> Nat)
    (source : Term) : Nat × Nat :=
  (columnIndex source.column, source.coefficient.val)

/-- Lower one typed linear combination in source order. -/
def numericTerms
    (columnIndex : ColumnId -> Nat)
    (source : LinearCombination) : List (Nat × Nat) :=
  source.map (numericTerm columnIndex)

private theorem residue_rawLcEval
    (columnIndex : ColumnId -> Nat)
    (assignment : Nat -> Nat) :
    forall source : LinearCombination,
      residue
          (Nightstream.Implementation.R1CS.Program.rawLcEval assignment
            (numericTerms columnIndex source)) =
        source.eval (typedAssignment columnIndex assignment) := by
  intro source
  induction source with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [numericTerms, List.map_cons,
        Nightstream.Implementation.R1CS.Program.rawLcEval,
        LinearCombination.eval, numericTerm, typedAssignment]
      rw [residue_add, residue_mul]
      have tailEqual := inductionHypothesis
      simp only [numericTerms] at tailEqual
      rw [tailEqual]
      simp only [residue_field_val]

/-- Numeric evaluation and typed evaluation agree exactly in Goldilocks. -/
theorem residue_lcEval
    (columnIndex : ColumnId -> Nat)
    (assignment : Nat -> Nat)
    (source : LinearCombination) :
    residue
        (Nightstream.Implementation.R1CS.lcEval assignment
          (numericTerms columnIndex source)) =
      source.eval (typedAssignment columnIndex assignment) := by
  rw [Nightstream.Implementation.R1CS.Program.lcEval_eq_raw_mod]
  change
    residue
        (Nightstream.Implementation.R1CS.Program.rawLcEval assignment
          (numericTerms columnIndex source) % Numeric.modulus) =
      source.eval (typedAssignment columnIndex assignment)
  rw [residue_mod]
  exact residue_rawLcEval columnIndex assignment source

/-- Lower one typed R1CS equation through the fixed allocation. -/
def numericRow
    (columnIndex : ColumnId -> Nat)
    (source : Row) : NumericRow where
  a := numericTerms columnIndex source.a
  b := numericTerms columnIndex source.b
  c := numericTerms columnIndex source.c

/-- One lowered numeric equation holds exactly when its typed source equation
holds on the canonical view of the same numeric assignment. -/
theorem numericRow_holds_iff
    (columnIndex : ColumnId -> Nat)
    (assignment : Nat -> Nat)
    (source : Row) :
    Nightstream.Implementation.R1CS.RowHolds assignment
        (numericRow columnIndex source) ↔
      source.Holds (typedAssignment columnIndex assignment) := by
  let left := Nightstream.Implementation.R1CS.lcEval assignment
    (numericTerms columnIndex source.a)
  let right := Nightstream.Implementation.R1CS.lcEval assignment
    (numericTerms columnIndex source.b)
  let output := Nightstream.Implementation.R1CS.lcEval assignment
    (numericTerms columnIndex source.c)
  have outputLt : output < Numeric.modulus := by
    unfold output Numeric.modulus Nightstream.Implementation.R1CS.lcEval
    exact Nat.mod_lt _ (by decide)
  have productLt : left * right % Numeric.modulus < Numeric.modulus :=
    Nat.mod_lt _ (by decide)
  change
    left * right % Numeric.modulus = output ↔
      source.a.eval (typedAssignment columnIndex assignment) *
          source.b.eval (typedAssignment columnIndex assignment) =
        source.c.eval (typedAssignment columnIndex assignment)
  rw [← residue_lcEval columnIndex assignment source.a,
    ← residue_lcEval columnIndex assignment source.b,
    ← residue_lcEval columnIndex assignment source.c]
  constructor
  · intro equal
    calc
      residue left * residue right = residue (left * right) :=
        (residue_mul left right).symm
      _ = residue (left * right % Numeric.modulus) :=
        (residue_mod (left * right)).symm
      _ = residue output := congrArg residue equal
  · intro equal
    apply residue_injective_of_lt productLt outputLt
    calc
      residue (left * right % Numeric.modulus) = residue (left * right) :=
        residue_mod (left * right)
      _ = residue left * residue right := residue_mul left right
      _ = residue output := equal

/-- Lower every owned typed row exactly once. Numeric rows do not carry the
typed ownership identifier, but list order and multiplicity remain exact. -/
def rows
    (columnIndex : ColumnId -> Nat)
    (source : List OwnedRow) : List NumericRow :=
  source.map fun owned => numericRow columnIndex owned.row

@[simp] theorem rows_length
    (columnIndex : ColumnId -> Nat)
    (source : List OwnedRow) :
    (rows columnIndex source).length = source.length := by
  simp [rows]

/-- Whole-list numeric satisfaction is equivalent to typed satisfaction on
the canonical view of the same numeric assignment. -/
theorem rows_satisfied_iff
    (columnIndex : ColumnId -> Nat)
    (source : List OwnedRow)
    (assignment : Nat -> Nat) :
    Nightstream.Implementation.R1CS.Satisfies
        (rows columnIndex source) assignment ↔
      Satisfies source (typedAssignment columnIndex assignment) := by
  induction source with
  | nil =>
      simp [rows, Nightstream.Implementation.R1CS.Satisfies]
  | cons head tail inductionHypothesis =>
      change
        Nightstream.Implementation.R1CS.Satisfies
            (numericRow columnIndex head.row :: rows columnIndex tail)
            assignment ↔
          head.row.Holds (typedAssignment columnIndex assignment) /\
            Satisfies tail (typedAssignment columnIndex assignment)
      constructor
      · intro numericSatisfied
        constructor
        · apply (numericRow_holds_iff columnIndex assignment head.row).1
          exact numericSatisfied _ (by simp)
        · apply inductionHypothesis.mp
          intro row member
          exact numericSatisfied row (by simp [member])
      · rintro ⟨headHolds, tailSatisfied⟩
        intro row member
        rcases List.mem_cons.mp member with rowExact | tailMember
        · subst row
          exact (numericRow_holds_iff columnIndex assignment head.row).2
            headHolds
        · exact (inductionHypothesis.mpr tailSatisfied) row tailMember

private def fallbackColumn : ColumnId where
  owner := .prelude
  bundleIndex := 0
  coordinateIndex := 0

noncomputable def selectedColumn
    (columnIndex : ColumnId -> Nat)
    (column : Nat) : ColumnId := by
  classical
  exact
    if present : exists source, columnIndex source = column then
      Classical.choose present
    else
      fallbackColumn

theorem selectedColumn_at
    {columnIndex : ColumnId -> Nat}
    (injective : Function.Injective columnIndex)
    (column : ColumnId) :
    selectedColumn columnIndex (columnIndex column) = column := by
  classical
  unfold selectedColumn
  split
  next present =>
    exact injective (Classical.choose_spec present)
  next absent =>
    exact False.elim (absent ⟨column, rfl⟩)

/-- Canonical numeric lift of a typed assignment through an injective column
allocation. Coordinates outside the allocation image are irrelevant. -/
noncomputable def liftAssignment
    (columnIndex : ColumnId -> Nat)
    (assignment : ColumnId -> F) : Nat -> Nat :=
  fun column => (assignment (selectedColumn columnIndex column)).val

theorem liftAssignment_at
    {columnIndex : ColumnId -> Nat}
    (injective : Function.Injective columnIndex)
    (assignment : ColumnId -> F)
    (column : ColumnId) :
    liftAssignment columnIndex assignment (columnIndex column) =
      (assignment column).val := by
  simp [liftAssignment, selectedColumn_at injective]

/-- Pulling an injective numeric lift back to the typed columns recovers the
original typed assignment exactly. -/
theorem typedAssignment_lift
    {columnIndex : ColumnId -> Nat}
    (injective : Function.Injective columnIndex)
    (assignment : ColumnId -> F) :
    typedAssignment columnIndex (liftAssignment columnIndex assignment) =
      assignment := by
  funext column
  simp [typedAssignment, liftAssignment_at injective, residue_field_val]

/-- Completeness of the lowering: an injective allocation gives every typed
satisfying assignment a canonical numeric satisfying assignment. -/
theorem exists_numeric_assignment_of_satisfies
    {columnIndex : ColumnId -> Nat}
    (injective : Function.Injective columnIndex)
    {source : List OwnedRow}
    {assignment : ColumnId -> F}
    (satisfied : Satisfies source assignment) :
    exists numericAssignment : Nat -> Nat,
      Nightstream.Implementation.R1CS.Satisfies
          (rows columnIndex source) numericAssignment /\
        typedAssignment columnIndex numericAssignment = assignment := by
  refine ⟨liftAssignment columnIndex assignment, ?_,
    typedAssignment_lift injective assignment⟩
  apply (rows_satisfied_iff columnIndex source
    (liftAssignment columnIndex assignment)).2
  simpa [typedAssignment_lift injective assignment] using satisfied

/-- One exact ordered embedding of a typed prefix and suffix into a generated
numeric row list. The split lets large consumers recover each proof without
normalizing the complete dependent source program. -/
structure SplitEmbedding
    (columnIndex : ColumnId -> Nat)
    (firstRows secondRows : List OwnedRow)
    (target : List NumericRow) : Prop where
  injective : Function.Injective columnIndex
  included : (rows columnIndex (firstRows ++ secondRows)).Sublist target

namespace SplitEmbedding

/-- Generated numeric satisfaction implies satisfaction of both exact typed
parts on one canonical typed assignment. -/
theorem satisfies
    {columnIndex : ColumnId -> Nat}
    {firstRows secondRows : List OwnedRow}
    {target : List NumericRow}
    (embedding : SplitEmbedding columnIndex firstRows secondRows target)
    {assignment : Nat -> Nat}
    (targetSatisfied :
      Nightstream.Implementation.R1CS.Satisfies target assignment) :
    Satisfies firstRows (typedAssignment columnIndex assignment) /\
      Satisfies secondRows (typedAssignment columnIndex assignment) := by
  apply (satisfies_append_iff firstRows secondRows
    (typedAssignment columnIndex assignment)).1
  apply (rows_satisfied_iff columnIndex (firstRows ++ secondRows) assignment).1
  intro row member
  exact targetSatisfied row (embedding.included.subset member)

/-- Every pair of satisfying typed parts has one canonical numeric assignment
that satisfies their exact ordered lowering. -/
theorem complete
    {columnIndex : ColumnId -> Nat}
    {firstRows secondRows : List OwnedRow}
    {target : List NumericRow}
    (embedding : SplitEmbedding columnIndex firstRows secondRows target)
    {assignment : ColumnId -> F}
    (firstSatisfied : Satisfies firstRows assignment)
    (secondSatisfied : Satisfies secondRows assignment) :
    exists numericAssignment : Nat -> Nat,
      Nightstream.Implementation.R1CS.Satisfies
          (rows columnIndex (firstRows ++ secondRows)) numericAssignment /\
        typedAssignment columnIndex numericAssignment = assignment := by
  apply exists_numeric_assignment_of_satisfies embedding.injective
  exact (satisfies_append_iff firstRows secondRows assignment).2
    ⟨firstSatisfied, secondSatisfied⟩

end SplitEmbedding

end Nightstream.Implementation.Lowering.Goldilocks.TypedNumericRowBridge
