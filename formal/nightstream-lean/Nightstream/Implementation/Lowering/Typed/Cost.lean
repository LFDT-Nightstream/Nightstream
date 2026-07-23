/-!
Contract: four-way resource accounting for an independently selected lowering
vocabulary.

Owns: the cost order and the additive accounting law used by typed lowering
programs.  The fields are resources, not measurements from a Rust circuit.

Does not own: R1CS rows, column numbers, an encoding choice, or a claim that a
particular production circuit realizes a cost.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Typed

/-- The only column ownership classes counted by the selected lowering.
The constant-one coordinate, when materialized, is an ordinary public source
port rather than an unclassified exception. -/
inductive Ownership where
  | committedColumn
  | publicColumn
  | auxiliaryColumn
deriving DecidableEq, Repr

/-- Exact symbolic resource use of one typed lowering recipe.

The optimization order is lexicographic: recurring rows first, then committed,
public, and auxiliary columns.  This is deliberately narrower than global
arithmetization minimality. -/
structure Cost where
  recurringRows : Nat
  committedColumns : Nat
  publicColumns : Nat
  auxiliaryColumns : Nat
deriving DecidableEq, Repr

namespace Cost

def zero : Cost := ⟨0, 0, 0, 0⟩

def add (left right : Cost) : Cost :=
  ⟨left.recurringRows + right.recurringRows,
    left.committedColumns + right.committedColumns,
    left.publicColumns + right.publicColumns,
    left.auxiliaryColumns + right.auxiliaryColumns⟩

instance : Add Cost where
  add := Cost.add

@[simp] theorem add_recurringRows (left right : Cost) :
    (left + right).recurringRows =
      left.recurringRows + right.recurringRows :=
  rfl

@[simp] theorem add_committedColumns (left right : Cost) :
    (left + right).committedColumns =
      left.committedColumns + right.committedColumns :=
  rfl

@[simp] theorem add_publicColumns (left right : Cost) :
    (left + right).publicColumns =
      left.publicColumns + right.publicColumns :=
  rfl

@[simp] theorem add_auxiliaryColumns (left right : Cost) :
    (left + right).auxiliaryColumns =
      left.auxiliaryColumns + right.auxiliaryColumns :=
  rfl

/-- One allocated column with an explicit ownership class. -/
def oneColumn : Ownership -> Cost
  | .committedColumn => ⟨0, 1, 0, 0⟩
  | .publicColumn => ⟨0, 0, 1, 0⟩
  | .auxiliaryColumn => ⟨0, 0, 0, 1⟩

/-- One recurring verifier equation.  Its columns are accounted separately. -/
def oneRow : Cost := ⟨1, 0, 0, 0⟩

/-- Definitional fold used by a program and by every receipt summary. -/
def sum : List Cost -> Cost
  | [] => zero
  | head :: tail => head + sum tail

/-- The fixed optimization comparison for this project vocabulary. -/
def LexLe (left right : Cost) : Prop :=
  left.recurringRows < right.recurringRows ∨
  (left.recurringRows = right.recurringRows ∧
    (left.committedColumns < right.committedColumns ∨
      (left.committedColumns = right.committedColumns ∧
        (left.publicColumns < right.publicColumns ∨
          (left.publicColumns = right.publicColumns ∧
            left.auxiliaryColumns ≤ right.auxiliaryColumns)))))

theorem add_assoc (first second third : Cost) :
    (first + second) + third = first + (second + third) := by
  change Cost.add (Cost.add first second) third =
    Cost.add first (Cost.add second third)
  cases first
  cases second
  cases third
  simp [Cost.add, Nat.add_assoc]

theorem zero_add (cost : Cost) : zero + cost = cost := by
  change Cost.add zero cost = cost
  cases cost
  simp [zero, Cost.add]

theorem sum_append (left right : List Cost) :
    sum (left ++ right) = sum left + sum right := by
  induction left with
  | nil =>
      simp only [List.nil_append, sum]
      exact (zero_add (sum right)).symm
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, sum, inductionHypothesis]
      exact (add_assoc head (sum tail) (sum right)).symm

end Cost

end Nightstream.Implementation.Lowering.Typed
