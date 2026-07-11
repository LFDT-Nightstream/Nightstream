import Nightstream.Implementation.R1CS.Semantics

/-!
Contract: semantics for generated rows whose left and output combinations are
both identically zero. Such rows are exact compiler artifacts, but they impose
no witness condition regardless of their right combination.
-/

namespace Nightstream.Implementation.R1CS.TrivialRows

open Nightstream.Implementation.R1CS

def Valid (rows : List Row) : Prop :=
  ∀ row ∈ rows, row.a = [] ∧ row.c = []

instance (rows : List Row) : Decidable (Valid rows) := by
  unfold Valid
  infer_instance

theorem satisfy
    {rows : List Row} (valid : Valid rows) (assignment : Nat → Nat) :
    Satisfies rows assignment := by
  intro row member
  have shape := valid row member
  simp [RowHolds, shape.1, shape.2, lcEval]

end Nightstream.Implementation.R1CS.TrivialRows
