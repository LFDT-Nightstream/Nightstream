import Nightstream.Implementation.R1CS.Core.CheckedProgram

/-!
Shared semantics for constant-definition prefixes used by compact accumulator
owners.

| Predicate / theorem | Mathematical obligation | Emits constraints? | Concrete owner |
|---|---|---|---|
| `definition` | `z[c] = 0` or `z[c] = v · z[0]` | no | terminal / recursive prefix |
| `definitions` | Zip concrete constant columns with verifier-owned values | no | concrete owner |
| `values_of_assignmentHolds` | Recover the exact value list from checked prefix definitions | no | concrete owner |

Owns: the reusable meaning of zero and nonzero constant rows.
Does not own: concrete columns, artifact membership, accumulator serialization,
or Poseidon2 rows.
Emits constraints: no.
Authority boundary: callers must prove that every instantiated definition is
present in their exact checked prefix; this module never supplies that fact.
-/

namespace Nightstream.Implementation.R1CS.AccumulatorConstantPrefix

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

/-- Rust normalizes zero to an empty linear combination and every nonzero
constant to one multiplication by the constant-one column. -/
def definition (column value : Nat) : Definition :=
  ⟨column, .linear (if value = 0 then [] else [(0, value)])⟩

def definitions (columns values : List Nat) : List Definition :=
  (columns.zip values).map fun pair => definition pair.1 pair.2

private theorem definition_value
    {assignment : Nat → Nat} {column value : Nat}
    (one : assignment 0 = 1) (valueLt : value < goldilocksP)
    (holds : (definition column value).Holds assignment) :
    assignment column = value := by
  by_cases zero : value = 0
  · subst value
    simpa [definition, Definition.Holds, Rhs.eval, lcEval] using holds
  · simpa [definition, zero, Definition.Holds, Rhs.eval, lcEval,
      one, Nat.mod_eq_of_lt valueLt] using holds

private theorem values_of_definitions
    {assignment : Nat → Nat} (one : assignment 0 = 1) :
    ∀ {columns values : List Nat},
      columns.length = values.length →
      (∀ value ∈ values, value < goldilocksP) →
      (∀ pair ∈ columns.zip values,
        (definition pair.1 pair.2).Holds assignment) →
      columns.map assignment = values := by
  intro columns
  induction columns with
  | nil =>
      intro values sameLength _ _
      cases values with
      | nil => rfl
      | cons _ _ => simp at sameLength
  | cons column columns inductionHypothesis =>
      intro values sameLength canonical holds
      cases values with
      | nil => simp at sameLength
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          have headEq : assignment column = value :=
            definition_value one (canonical value (by simp))
              (holds (column, value) (by simp))
          have tailEq := inductionHypothesis sameLength
            (fun candidate member => canonical candidate (by simp [member]))
            (fun pair member => holds pair (by simp [member]))
          simp only [List.map_cons, List.cons.injEq]
          exact ⟨headEq, tailEq⟩

/-- Exact checked-prefix membership is sufficient to recover every requested
constant value.  Concrete artifact membership remains a caller obligation. -/
theorem values_of_assignmentHolds
    {assignment : Nat → Nat} {columns values : List Nat}
    {instructions : List Instruction}
    (one : assignment 0 = 1)
    (sameLength : columns.length = values.length)
    (canonical : ∀ value ∈ values, value < goldilocksP)
    (included : ∀ current ∈ definitions columns values,
      current ∈ CheckedProgram.definitions instructions)
    (program : AssignmentHolds instructions assignment) :
    columns.map assignment = values := by
  apply values_of_definitions one sameLength canonical
  intro pair member
  apply program.definitions (definition pair.1 pair.2)
  apply included
  exact List.mem_map.mpr ⟨pair, member, rfl⟩

end Nightstream.Implementation.R1CS.AccumulatorConstantPrefix
