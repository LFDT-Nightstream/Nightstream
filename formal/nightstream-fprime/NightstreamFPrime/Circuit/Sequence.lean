import NightstreamFPrime.Circuit.StraightLine

/-!
Owns the proof-only sequencing rule for opaque `FormalCircuit` children.
It does not define a second circuit representation. A completed prefix stores
the environment, exact operation order, flat-row satisfaction, row scope, and
external-variable agreement needed to append the next child safely.
-/

namespace NightstreamFPrime.Circuit.Sequence

open NightstreamFPrime.Circuit

/-- Evidence for one honestly completed prefix of adjacent child calls. -/
structure Prefix (initial : Env) (base : Nat) where
  current : Env
  operations : List Op
  agrees : AgreesOutside initial current base (localLength operations)
  scope : ∀ expression ∈ flatConstraints operations,
    expression.VarsBelow (base + localLength operations)
  rows : holdsFlat current operations

/-- A later prefix preserves every value allocated or supplied before the
end of an earlier prefix. This is logical sequencing evidence, not a row
proof. -/
structure PreservesPrefix {initial : Env} {base : Nat}
    (before after : Prefix initial base) : Prop where
  length_le : localLength before.operations ≤ localLength after.operations
  values : ∀ index, index < base + localLength before.operations →
    after.current index = before.current index

theorem PreservesPrefix.trans
    {initial : Env} {base : Nat}
    {first second third : Prefix initial base}
    (left : PreservesPrefix first second)
    (right : PreservesPrefix second third) :
    PreservesPrefix first third := by
  constructor
  · exact Nat.le_trans left.length_le right.length_le
  · intro index below
    rw [right.values index (lt_of_lt_of_le below
      (Nat.add_le_add_left left.length_le base))]
    exact left.values index below

/-- The empty prefix changes no variable and has no row. -/
def empty (env : Env) (base : Nat) : Prefix env base where
  current := env
  operations := []
  agrees := by
    intro _ _
    rfl
  scope := by
    intro expression member
    change expression ∈ ([] : List Expr) at member
    cases member
  rows := by
    intro expression member
    change expression ∈ ([] : List Expr) at member
    cases member

@[simp] theorem localLength_append (left right : List Op) :
    localLength (left ++ right) = localLength left + localLength right := by
  unfold localLength
  rw [List.map_append, List.sum_append]

@[simp] theorem localLength_singleton (operation : Op) :
    localLength [operation] = operation.localLength := by
  simp [localLength]

/-- Splice an already-built opaque child assignment into a completed prefix.
The caller supplies the child's row proof; this theorem owns only sequencing. -/
theorem appendBuilt
    {initial : Env} {base : Nat}
    (completedPrefix : Prefix initial base)
    (child : FormalCircuit)
    (parentOp : Op)
    (parentLength : parentOp.localLength = localLength
      (Circuit.ops child.main
        (base + localLength completedPrefix.operations)))
    (parentConstraints : parentOp.flatConstraints = flatConstraints
      (Circuit.ops child.main
        (base + localLength completedPrefix.operations)))
    (childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main
        (base + localLength completedPrefix.operations)),
      expression.VarsBelow
        (base + localLength completedPrefix.operations +
          localLength (Circuit.ops child.main
            (base + localLength completedPrefix.operations))))
    (after : Env)
    (childAgrees : AgreesOutside completedPrefix.current after
      (base + localLength completedPrefix.operations)
      (localLength (Circuit.ops child.main
        (base + localLength completedPrefix.operations))))
    (childRows : holdsFlat after
      (Circuit.ops child.main
        (base + localLength completedPrefix.operations))) :
    ∃ completed : Prefix initial base,
      completed.operations = completedPrefix.operations ++ [parentOp] ∧
      PreservesPrefix completedPrefix completed ∧
      holdsFlat completed.current
        (Circuit.ops child.main
          (base + localLength completedPrefix.operations)) := by
  let childLength := localLength
    (Circuit.ops child.main (base + localLength completedPrefix.operations))
  have priorRows : ConstraintsHold after
      (flatConstraints completedPrefix.operations) := by
    apply constraintsHold_of_agree_below completedPrefix.current after
      (flatConstraints completedPrefix.operations)
      (base + localLength completedPrefix.operations)
    · exact completedPrefix.scope
    · intro index below
      exact childAgrees index (Or.inl below)
    · exact completedPrefix.rows
  have parentRows : ConstraintsHold after parentOp.flatConstraints := by
    rw [parentConstraints]
    exact childRows
  let completed : Prefix initial base := {
    current := after
    operations := completedPrefix.operations ++ [parentOp]
    agrees := by
      have combined := completedPrefix.agrees.append childAgrees
      rw [localLength_append, localLength_singleton, parentLength]
      simpa [childLength] using combined
    scope := by
      intro expression member
      rw [flatConstraints_append, flatConstraints_singleton] at member
      rcases List.mem_append.mp member with previous | added
      · apply Expr.VarsBelow.mono expression
          (completedPrefix.scope expression previous)
        rw [localLength_append, localLength_singleton]
        omega
      · rw [parentConstraints] at added
        apply Expr.VarsBelow.mono expression (childScope expression added)
        rw [localLength_append, localLength_singleton, parentLength]
        omega
    rows := by
      change ConstraintsHold after
        (flatConstraints (completedPrefix.operations ++ [parentOp]))
      rw [flatConstraints_append, flatConstraints_singleton]
      exact (constraintsHold_append after _ _).mpr ⟨priorRows, parentRows⟩
  }
  refine ⟨completed, rfl, ?_, childRows⟩
  constructor
  · simp [completed, localLength_append]
  · intro index below
    exact childAgrees index (Or.inl below)

/-- Append one opaque child at the exact end of a completed prefix. -/
theorem append
    {initial : Env} {base : Nat}
    (completedPrefix : Prefix initial base)
    (child : FormalCircuit)
    (parentOp : Op)
    (parentLength : parentOp.localLength = localLength
      (Circuit.ops child.main
        (base + localLength completedPrefix.operations)))
    (parentConstraints : parentOp.flatConstraints = flatConstraints
      (Circuit.ops child.main
        (base + localLength completedPrefix.operations)))
    (childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops child.main
        (base + localLength completedPrefix.operations)),
      expression.VarsBelow
        (base + localLength completedPrefix.operations +
          localLength (Circuit.ops child.main
            (base + localLength completedPrefix.operations))))
    (assumptions : child.assumptions
      (base + localLength completedPrefix.operations) completedPrefix.current)
    (specification : child.spec
      (base + localLength completedPrefix.operations) completedPrefix.current) :
    ∃ completed : Prefix initial base,
      completed.operations = completedPrefix.operations ++ [parentOp] ∧
      PreservesPrefix completedPrefix completed ∧
      holdsFlat completed.current
        (Circuit.ops child.main
          (base + localLength completedPrefix.operations)) := by
  rcases child.completeness completedPrefix.current
      (base + localLength completedPrefix.operations)
      assumptions specification with ⟨after, childAgrees, childRows⟩
  exact appendBuilt completedPrefix child parentOp parentLength
    parentConstraints childScope after childAgrees childRows

end NightstreamFPrime.Circuit.Sequence
