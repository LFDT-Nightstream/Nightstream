import Nightstream.Implementation.Lowering.Goldilocks.Rows
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: native CCS selection for canonical Goldilocks rows.

Assurance tier: model-level.

Owns:
- the selected row equation `S · (A · B - C) = 0`;
- the exact four-matrix polynomial used by the Rust CCS backend;
- active soundness, inactive satisfaction, honest completeness, support, and
  positional ownership for selected row lists.

Does not own: branch selection, a protocol call, a proof-free manifest, Rust
matrix emission, or a generated artifact.

Emits constraints: one CCS row for each source row and no auxiliary columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

private abbrev Field := Nightstream.SuperNeo.Concrete.F

/-- The fixed matrix arity of the native selected-R1CS CCS relation. -/
def matrixCount : Nat := 4

/-- The fixed total degree of `S · (A · B - C)`. -/
def polynomialDegree : Nat := 3

/-- The selected CCS polynomial over the four row evaluations
`[A z, B z, C z, S z]`. -/
def polynomial (a b c selector : Field) : Field :=
  selector * (a * b - c)

/-- Exponents of the positive term `A * B * S`. -/
def positiveExponents (index : Fin matrixCount) : Nat :=
  if index.val = 0 then 1
  else if index.val = 1 then 1
  else if index.val = 3 then 1
  else 0

/-- Exponents of the negative term `C * S`. -/
def negativeExponents (index : Fin matrixCount) : Nat :=
  if index.val = 2 then 1
  else if index.val = 3 then 1
  else 0

def positiveMonomial : Monomial Field matrixCount where
  coefficient := 1
  exponents := positiveExponents

def negativeMonomial : Monomial Field matrixCount where
  coefficient := -1
  exponents := negativeExponents

/-- SuperNeo Definition 11 syntax for the native selector. The strict
degree bound is four because the largest explicit term has degree three. -/
def constraintPolynomial : ConstraintPolynomial Field matrixCount where
  degreeBound := 4
  terms := [positiveMonomial, negativeMonomial]
  termsBelowDegree := by
    intro term member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> decide

theorem constraintPolynomial_terms_exact :
    constraintPolynomial.terms =
      [positiveMonomial, negativeMonomial] :=
  rfl

theorem equalityGatedDegree_exact :
    constraintPolynomial.canonicalEqualityGatedDegreeBound = 4 := by
  rfl

/-- Direct SuperNeo sparse-polynomial evaluation of the native selector. -/
def evaluate (point : Fin matrixCount → Field) : Field :=
  evaluatePolynomial baseOps constraintPolynomial point

private theorem canonicalFinIndices_exact :
    canonicalFinIndices matrixCount =
      [⟨0, by decide⟩, ⟨1, by decide⟩,
        ⟨2, by decide⟩, ⟨3, by decide⟩] := by
  rfl

private theorem mul_neg (left right : Field) :
    left * -right = -(left * right) := by
  calc
    left * -right = -right * left := Fin.mul_comm _ _
    _ = -(right * left) := Lean.Grind.Fin.neg_mul _ _
    _ = -(left * right) := by rw [Fin.mul_comm right left]

theorem evaluate_exact (point : Fin matrixCount → Field) :
    evaluate point =
      polynomial
        (point ⟨0, by decide⟩)
        (point ⟨1, by decide⟩)
        (point ⟨2, by decide⟩)
        (point ⟨3, by decide⟩) := by
  simp [evaluate, evaluatePolynomial, evaluateMonomial,
    constraintPolynomial, positiveMonomial, negativeMonomial,
    positiveExponents, negativeExponents, canonicalFinIndices_exact, pow,
    baseOps, polynomial, Fin.one_mul, Fin.mul_one]
  rw [Fin.sub_eq_add_neg, Lean.Grind.Fin.left_distrib]
  simp only [Fin.mul_assoc, Fin.mul_comm, Lean.Grind.Fin.neg_mul, mul_neg]
  ac_rfl

/-- One ordinary sparse row paired with its CCS selector column.  The selector
is a matrix input.  It is not an allocated residual or a second constraint. -/
structure SelectedRow where
  source : OwnedRow
  selector : ColumnId
deriving DecidableEq, Repr

namespace SelectedRow

def Holds (row : SelectedRow) (assignment : ColumnId → Field) : Prop :=
  polynomial
    (row.source.row.a.eval assignment)
    (row.source.row.b.eval assignment)
    (row.source.row.c.eval assignment)
    (assignment row.selector) = 0

def columnIds (row : SelectedRow) : List ColumnId :=
  row.selector :: row.source.columnIds

@[simp] theorem source_id (row : SelectedRow) :
    row.source.id = row.source.id :=
  rfl

theorem holds_of_source
    (row : SelectedRow)
    (assignment : ColumnId → Field)
    (sourceHolds : row.source.row.Holds assignment) :
    row.Holds assignment := by
  unfold Holds polynomial
  change
    row.source.row.a.eval assignment *
        row.source.row.b.eval assignment =
      row.source.row.c.eval assignment at sourceHolds
  rw [sourceHolds]
  simp only [Lean.Grind.AddCommGroup.sub_self, Fin.mul_zero]

theorem source_holds_of_selector_one
    (row : SelectedRow)
    (assignment : ColumnId → Field)
    (selectorOne : assignment row.selector = 1)
    (holds : row.Holds assignment) :
    row.source.row.Holds assignment := by
  unfold Holds polynomial at holds
  rw [selectorOne, Fin.one_mul] at holds
  change
    row.source.row.a.eval assignment *
        row.source.row.b.eval assignment =
      row.source.row.c.eval assignment
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp holds

theorem holds_of_selector_zero
    (row : SelectedRow)
    (assignment : ColumnId → Field)
    (selectorZero : assignment row.selector = 0) :
    row.Holds assignment := by
  unfold Holds polynomial
  rw [selectorZero, Fin.zero_mul]

end SelectedRow

/-- Attach one selector column to every source row without changing row order,
row identity, or source equation. -/
def select (selector : ColumnId) (rows : List OwnedRow) : List SelectedRow :=
  rows.map fun row => ⟨row, selector⟩

/-- Exact occurrence-preserving satisfaction of a selected CCS row list. -/
def Satisfies : List SelectedRow → (ColumnId → Field) → Prop
  | [], _ => True
  | row :: rows, assignment =>
      row.Holds assignment ∧ Satisfies rows assignment

@[simp] theorem satisfies_nil (assignment : ColumnId → Field) :
    Satisfies [] assignment :=
  True.intro

@[simp] theorem satisfies_cons
    (row : SelectedRow)
    (rows : List SelectedRow)
    (assignment : ColumnId → Field) :
    Satisfies (row :: rows) assignment ↔
      row.Holds assignment ∧ Satisfies rows assignment :=
  Iff.rfl

theorem satisfies_append_iff
    (left right : List SelectedRow)
    (assignment : ColumnId → Field) :
    Satisfies (left ++ right) assignment ↔
      Satisfies left assignment ∧ Satisfies right assignment := by
  induction left with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, satisfies_cons, inductionHypothesis]
      constructor
      · rintro ⟨headHolds, tailHolds, rightHolds⟩
        exact ⟨⟨headHolds, tailHolds⟩, rightHolds⟩
      · rintro ⟨⟨headHolds, tailHolds⟩, rightHolds⟩
        exact ⟨headHolds, tailHolds, rightHolds⟩

theorem select_length (selector : ColumnId) (rows : List OwnedRow) :
    (select selector rows).length = rows.length := by
  simp [select]

theorem select_source (selector : ColumnId) (rows : List OwnedRow) :
    (select selector rows).map SelectedRow.source = rows := by
  simp [select, Function.comp_def]

theorem select_row_ids
    (selector : ColumnId)
    (rows : List OwnedRow) :
    (select selector rows).map (fun row => row.source.id) =
      rows.map (fun row => row.id) := by
  simp [select, Function.comp_def]

theorem select_row_ids_nodup
    (selector : ColumnId)
    (rows : List OwnedRow)
    (nodup : (rows.map fun row => row.id).Nodup) :
    ((select selector rows).map fun row => row.source.id).Nodup := by
  rw [select_row_ids]
  exact nodup

theorem select_rows_owned
    (owner : PhysicalOwner)
    (selector : ColumnId)
    (rows : List OwnedRow)
    (owned :
      ∀ row, row ∈ rows → row.id.owner = owner)
    (row : SelectedRow)
    (member : row ∈ select selector rows) :
    row.source.id.owner = owner := by
  rcases List.mem_map.1 member with ⟨source, sourceMember, rfl⟩
  exact owned source sourceMember

theorem select_supported
    (selector : ColumnId)
    (rows : List OwnedRow)
    (allowed : List ColumnId)
    (sourceSupported :
      ∀ row, row ∈ rows →
        ∀ column, column ∈ row.columnIds → column ∈ allowed)
    (row : SelectedRow)
    (member : row ∈ select selector rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ selector :: allowed := by
  rcases List.mem_map.1 member with ⟨source, sourceMember, rfl⟩
  simp only [SelectedRow.columnIds, List.mem_cons] at columnMember
  rcases columnMember with rfl | sourceColumn
  · exact List.mem_cons_self
  · exact List.mem_cons_of_mem selector
      (sourceSupported source sourceMember column sourceColumn)

/-- Active native selection is exactly the original R1CS program. -/
theorem active_sound
    (selector : ColumnId)
    (rows : List OwnedRow)
    (assignment : ColumnId → Field)
    (selectorOne : assignment selector = 1)
    (satisfied : Satisfies (select selector rows) assignment) :
    Goldilocks.Satisfies rows assignment := by
  induction rows with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      exact ⟨
        SelectedRow.source_holds_of_selector_one
          ⟨head, selector⟩ assignment selectorOne satisfied.1,
        inductionHypothesis satisfied.2
      ⟩

/-- A source-satisfying assignment is a selected-CCS satisfying assignment for
every selector value. -/
theorem complete
    (selector : ColumnId)
    (rows : List OwnedRow)
    (assignment : ColumnId → Field)
    (satisfied : Goldilocks.Satisfies rows assignment) :
    Satisfies (select selector rows) assignment := by
  induction rows with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      exact ⟨
        SelectedRow.holds_of_source
          ⟨head, selector⟩ assignment satisfied.1,
        inductionHypothesis satisfied.2
      ⟩

/-- An inactive selector disables every source row without a residual witness
or a completion write. -/
theorem inactive_satisfies
    (selector : ColumnId)
    (rows : List OwnedRow)
    (assignment : ColumnId → Field)
    (selectorZero : assignment selector = 0) :
    Satisfies (select selector rows) assignment := by
  induction rows with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      exact ⟨
        SelectedRow.holds_of_selector_zero
          ⟨head, selector⟩ assignment selectorZero,
        inductionHypothesis
      ⟩

theorem matrixCount_exact : matrixCount = 4 :=
  rfl

theorem polynomialDegree_exact : polynomialDegree = 3 :=
  rfl

end Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
