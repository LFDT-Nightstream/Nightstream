import Nightstream.Implementation.Lowering.Typed.Receipt
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: independent sparse R1CS row semantics over the paper Goldilocks
carrier.

Owns:
- stable physical column and row identities;
- sparse Goldilocks linear combinations and R1CS satisfaction;
- the selected one-row pin, affine, product, bit, gated-assertion,
  activation, and mux recipes;
- local soundness and completeness of those recipes.

Does not own: Rust column numbers, generated rows, protocol call recipes,
whole-program compilation, or a proof of the Goldilocks prime-field law.

Emits constraints: one row for each constructor in `CanonicalRow`.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete

/-- Additional physical owners introduced by the canonical encoding.  The
constant-one prelude and branch activation wires are explicit, never compiler
side effects. -/
inductive PhysicalOwner where
  | prelude
  | typed (owner : Typed.Owner)
  | branchActivation (path : OwnerPath) (selected : Bool)
deriving DecidableEq, Repr

/-- Stable physical identity of one allocated field coordinate. -/
structure ColumnId where
  owner : PhysicalOwner
  bundleIndex : Nat
  coordinateIndex : Nat
deriving DecidableEq, Repr

/-- Stable physical identity of one emitted row occurrence. -/
structure RowId where
  owner : PhysicalOwner
  ordinal : Nat
deriving DecidableEq, Repr

/-- A physical column and its accounting class. -/
structure OwnedColumn where
  id : ColumnId
  ownership : Ownership
deriving DecidableEq, Repr

/-- One sparse coefficient-column pair. -/
structure Term where
  column : ColumnId
  coefficient : F
deriving DecidableEq, Repr

abbrev LinearCombination := List Term

namespace LinearCombination

/-- Evaluate one sparse linear combination directly in the paper field. -/
def eval (assignment : ColumnId -> F) : LinearCombination -> F
  | [] => 0
  | term :: tail =>
      term.coefficient * assignment term.column + eval assignment tail

@[simp] theorem eval_nil (assignment : ColumnId -> F) :
    eval assignment [] = 0 :=
  rfl

@[simp] theorem eval_cons (assignment : ColumnId -> F)
    (term : Term) (tail : LinearCombination) :
    eval assignment (term :: tail) =
      term.coefficient * assignment term.column + eval assignment tail :=
  rfl

end LinearCombination

/-- One physical R1CS equation `(A z) * (B z) = C z`. -/
structure Row where
  a : LinearCombination
  b : LinearCombination
  c : LinearCombination
deriving DecidableEq, Repr

namespace Row

def Holds (assignment : ColumnId -> F) (row : Row) : Prop :=
  row.a.eval assignment * row.b.eval assignment = row.c.eval assignment

/-- Exact ordered physical support of a sparse row.  This is shared by call
recipes and the whole-program scoping proof so neither layer reconstructs
dependencies from emitter metadata. -/
def columnIds (row : Row) : List ColumnId :=
  (row.a ++ row.b ++ row.c).map (fun term => term.column)

instance (assignment : ColumnId -> F) (row : Row) :
    Decidable (row.Holds assignment) := by
  unfold Holds
  infer_instance

end Row

/-- A row occurrence with exactly one structural owner. -/
structure OwnedRow where
  id : RowId
  row : Row
deriving DecidableEq, Repr

namespace OwnedRow

def columnIds (row : OwnedRow) : List ColumnId :=
  row.row.columnIds

end OwnedRow

/-- Every physical occurrence has one owner by construction.  This is about
occurrences, not syntactic row equality: equal equations at different
positions remain separately owned physical rows. -/
theorem rowOccurrence_has_unique_owner (row : OwnedRow) :
    ∃ owner, row.id.owner = owner ∧
      ∀ candidate, row.id.owner = candidate -> candidate = owner := by
  refine ⟨row.id.owner, rfl, ?_⟩
  intro candidate equal
  exact equal.symm

theorem columnOccurrence_has_unique_owner (column : OwnedColumn) :
    ∃ owner, column.id.owner = owner ∧
      ∀ candidate, column.id.owner = candidate -> candidate = owner := by
  refine ⟨column.id.owner, rfl, ?_⟩
  intro candidate equal
  exact equal.symm

/-- Exact satisfaction traverses the physical occurrence list.  Equal
equations with different `RowId`s remain separate occurrences and therefore
remain separately counted and owned. -/
def Satisfies : List OwnedRow -> (ColumnId -> F) -> Prop
  | [], _ => True
  | row :: tail, assignment =>
      row.row.Holds assignment ∧ Satisfies tail assignment

@[simp] theorem satisfies_nil (assignment : ColumnId -> F) :
    Satisfies [] assignment :=
  True.intro

@[simp] theorem satisfies_cons (row : OwnedRow) (tail : List OwnedRow)
    (assignment : ColumnId -> F) :
    Satisfies (row :: tail) assignment ↔
      row.row.Holds assignment ∧ Satisfies tail assignment :=
  Iff.rfl

theorem satisfies_append_iff (left right : List OwnedRow)
    (assignment : ColumnId -> F) :
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

/-- Named paper-field boundary needed only when a zero product is split into
its factors.  This is a field law, not a Rust or artifact assumption. -/
structure FieldLaws where
  noZeroDivisors : forall left right : F,
    left * right = 0 -> left = 0 ∨ right = 0

def singleton (column : ColumnId) (coefficient : F) :
    LinearCombination :=
  [{ column := column, coefficient := coefficient }]

def difference (left right : ColumnId) : LinearCombination :=
  [{ column := left, coefficient := 1 },
    { column := right, coefficient := -1 }]

def oneMinus (one value : ColumnId) : LinearCombination :=
  [{ column := one, coefficient := 1 },
    { column := value, coefficient := -1 }]

/-- The selected finite local row vocabulary. -/
inductive CanonicalRow where
  | pin (one output : ColumnId) (value : F)
  | affine (one output : ColumnId) (terms : LinearCombination)
  | product (output left right : ColumnId)
  | bit (one value : ColumnId)
  | gatedAssert (one active condition : ColumnId)
  | activateTrue (output active selector : ColumnId)
  | activateFalse (one output active selector : ColumnId)
  | mux (joined selector onTrue onFalse : ColumnId)
deriving DecidableEq, Repr

namespace CanonicalRow

/-- Canonical sparse equation emitted by each vocabulary member. -/
def row : CanonicalRow -> Row
  | .pin one output value =>
      ⟨singleton output 1, singleton one 1, singleton one value⟩
  | .affine one output terms =>
      ⟨terms, singleton one 1, singleton output 1⟩
  | .product output left right =>
      ⟨singleton left 1, singleton right 1, singleton output 1⟩
  | .bit one value =>
      ⟨singleton value 1, difference value one, []⟩
  | .gatedAssert one active condition =>
      ⟨singleton active 1, oneMinus one condition, []⟩
  | .activateTrue output active selector =>
      ⟨singleton active 1, singleton selector 1, singleton output 1⟩
  | .activateFalse one output active selector =>
      ⟨singleton active 1, oneMinus one selector, singleton output 1⟩
  | .mux joined selector onTrue onFalse =>
      ⟨singleton selector 1, difference onTrue onFalse,
        difference joined onFalse⟩

theorem pin_iff
    (assignment : ColumnId -> F) (one output : ColumnId) (value : F)
    (constantOne : assignment one = 1) :
    (pin one output value).row.Holds assignment ↔
      assignment output = value := by
  simp only [row, Row.Holds, singleton, LinearCombination.eval, constantOne,
    Fin.one_mul, Fin.mul_one, Fin.add_zero]

theorem affine_iff
    (assignment : ColumnId -> F) (one output : ColumnId)
    (terms : LinearCombination) (constantOne : assignment one = 1) :
    (affine one output terms).row.Holds assignment ↔
      terms.eval assignment = assignment output := by
  simp only [row, Row.Holds, singleton, LinearCombination.eval, constantOne,
    Fin.one_mul, Fin.mul_one, Fin.add_zero]

theorem product_iff
    (assignment : ColumnId -> F) (output left right : ColumnId) :
    (product output left right).row.Holds assignment ↔
      assignment left * assignment right = assignment output := by
  simp only [row, Row.Holds, singleton, LinearCombination.eval,
    Fin.one_mul, Fin.mul_one, Fin.add_zero]

theorem bit_sound
    (laws : FieldLaws) (assignment : ColumnId -> F)
    (one value : ColumnId) (constantOne : assignment one = 1)
    (holds : (bit one value).row.Holds assignment) :
    assignment value = 0 ∨ assignment value = 1 := by
  simp only [row, Row.Holds, singleton, difference,
    LinearCombination.eval, constantOne, Fin.mul_one, Fin.one_mul,
    Fin.add_zero, Fin.zero_add, Lean.Grind.Fin.neg_mul,
    Fin.sub_eq_add_neg] at holds
  rcases laws.noZeroDivisors _ _ holds with valueZero | differenceZero
  · exact Or.inl valueZero
  · right
    apply Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp
    simpa only [Fin.sub_eq_add_neg] using differenceZero

theorem bit_complete
    (assignment : ColumnId -> F) (one value : ColumnId)
    (constantOne : assignment one = 1)
    (boolean : assignment value = 0 ∨ assignment value = 1) :
    (bit one value).row.Holds assignment := by
  rcases boolean with valueZero | valueOne
  · simp only [row, Row.Holds, singleton, difference,
      LinearCombination.eval, constantOne, valueZero, Fin.one_mul,
      Fin.zero_mul, Fin.mul_zero, Fin.add_zero, Fin.zero_add,
      Lean.Grind.Fin.neg_mul, Lean.Grind.AddCommGroup.neg_zero]
  · simp only [row, Row.Holds, singleton, difference,
      LinearCombination.eval, constantOne, valueOne, Fin.one_mul,
      Fin.mul_one, Fin.add_zero, Lean.Grind.Fin.neg_mul]
    rw [Lean.Grind.Fin.add_comm 1 (-1), Lean.Grind.Fin.neg_add_cancel]

theorem gatedAssert_iff_of_active
    (laws : FieldLaws) (assignment : ColumnId -> F)
    (one active condition : ColumnId)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1) :
    (gatedAssert one active condition).row.Holds assignment ↔
      assignment condition = 1 := by
  simp only [row, Row.Holds, singleton, oneMinus,
    LinearCombination.eval, constantOne, activeOne, Fin.mul_one, Fin.one_mul,
    Fin.add_zero, Fin.zero_add, Lean.Grind.Fin.neg_mul,
    Fin.sub_eq_add_neg]
  constructor
  · intro differenceZero
    have subZero : (1 : F) - assignment condition = 0 := by
      simpa only [Fin.sub_eq_add_neg] using differenceZero
    exact (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp subZero).symm
  · intro conditionOne
    rw [conditionOne]
    have selfSub : (1 : F) - 1 = 0 :=
      Lean.Grind.AddCommGroup.sub_eq_zero_iff.mpr rfl
    simpa only [Fin.sub_eq_add_neg] using selfSub

theorem gatedAssert_complete_of_inactive
    (assignment : ColumnId -> F) (one active condition : ColumnId)
    (inactive : assignment active = 0) :
    (gatedAssert one active condition).row.Holds assignment := by
  simp only [row, Row.Holds, singleton, oneMinus,
    LinearCombination.eval, inactive, Fin.one_mul, Fin.zero_mul,
    Fin.mul_zero, Fin.add_zero]

theorem activateTrue_iff
    (assignment : ColumnId -> F) (output active selector : ColumnId) :
    (activateTrue output active selector).row.Holds assignment ↔
      assignment active * assignment selector = assignment output := by
  simp only [row, Row.Holds, singleton, LinearCombination.eval,
    Fin.one_mul, Fin.add_zero]

theorem activateFalse_iff
    (assignment : ColumnId -> F) (one output active selector : ColumnId)
    (constantOne : assignment one = 1) :
    (activateFalse one output active selector).row.Holds assignment ↔
      assignment active * (1 - assignment selector) = assignment output := by
  simp only [row, Row.Holds, singleton, oneMinus,
    LinearCombination.eval, constantOne, Fin.one_mul, Fin.add_zero,
    Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg]

theorem mux_iff
    (assignment : ColumnId -> F)
    (joined selector onTrue onFalse : ColumnId) :
    (mux joined selector onTrue onFalse).row.Holds assignment ↔
      assignment selector *
          (assignment onTrue - assignment onFalse) =
        assignment joined - assignment onFalse := by
  simp only [row, Row.Holds, singleton, difference, LinearCombination.eval,
    Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
    Fin.sub_eq_add_neg]

private theorem field_add_right_cancel {left right suffix : F}
    (equal : left + suffix = right + suffix) : left = right := by
  calc
    left = (left + suffix) + -suffix := by
      rw [Lean.Grind.Fin.add_assoc,
        Lean.Grind.Fin.add_comm suffix (-suffix),
        Lean.Grind.Fin.neg_add_cancel, Fin.add_zero]
    _ = (right + suffix) + -suffix :=
      congrArg (fun value => value + -suffix) equal
    _ = right := by
      rw [Lean.Grind.Fin.add_assoc,
        Lean.Grind.Fin.add_comm suffix (-suffix),
        Lean.Grind.Fin.neg_add_cancel, Fin.add_zero]

theorem mux_selects_true
    (assignment : ColumnId -> F)
    (joined selector onTrue onFalse : ColumnId)
    (selectorOne : assignment selector = 1)
    (holds : (mux joined selector onTrue onFalse).row.Holds assignment) :
    assignment joined = assignment onTrue := by
  have equation :=
    (mux_iff assignment joined selector onTrue onFalse).mp holds
  rw [selectorOne, Fin.one_mul] at equation
  have cancelled : assignment onTrue = assignment joined := by
    apply field_add_right_cancel
    simpa only [Fin.sub_eq_add_neg] using equation
  exact cancelled.symm

theorem mux_selects_false
    (assignment : ColumnId -> F)
    (joined selector onTrue onFalse : ColumnId)
    (selectorZero : assignment selector = 0)
    (holds : (mux joined selector onTrue onFalse).row.Holds assignment) :
    assignment joined = assignment onFalse := by
  have equation :=
    (mux_iff assignment joined selector onTrue onFalse).mp holds
  rw [selectorZero, Fin.zero_mul] at equation
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp equation.symm

end CanonicalRow

end Nightstream.Implementation.Lowering.Goldilocks
