import Nightstream.Implementation.R1CS.Core.Projection.Trace

/-!
Shared exact-row matching for emitted three-matrix diagnostic PiRLC projection artifacts.

Owns: index-preserving comparison between normalized source rows and
reconstructed SSA definitions, modulo sparse-term ordering.

Does not own: generated data, semantic interpretation, row satisfaction,
whole-matrix embedding, transcript authority, or row removal.

Emits constraints: no.

| Relation | Mathematical obligation | Authority |
|---|---|---|
| `RowsPermutationEquivalent` | preserve each A/B/C linear combination modulo term order | exact source row |
| `IndexedRowMatchesDefinition` | preserve absolute row index and reconstructed equation | exact source row plus definition |
| `indexedRowsMatch` | compare complete schedules in lockstep and fail on length mismatch | executable kernel check |
| `indexedRowsMatchRows` | compare indexed source rows to reconstructed assertion rows | executable kernel check |
-/

namespace Nightstream.Implementation.R1CS.ActiveIndexedRows

open Nightstream.Implementation.R1CS

def RowsPermutationEquivalent (source reconstructed : Row) : Prop :=
  source.a.Perm reconstructed.a ∧
  source.b.Perm reconstructed.b ∧
  source.c.Perm reconstructed.c

instance (source reconstructed : Row) :
    Decidable (RowsPermutationEquivalent source reconstructed) := by
  unfold RowsPermutationEquivalent
  infer_instance

def IndexedRowMatchesDefinition
    (source : Nat × Row) (reconstructed : Nat × Program.Definition) : Prop :=
  source.1 = reconstructed.1 ∧
  RowsPermutationEquivalent source.2 reconstructed.2.builderRow

instance (source : Nat × Row) (reconstructed : Nat × Program.Definition) :
    Decidable (IndexedRowMatchesDefinition source reconstructed) := by
  unfold IndexedRowMatchesDefinition
  infer_instance

def IndexedRowMatchesRow
    (source reconstructed : Nat × Row) : Prop :=
  source.1 = reconstructed.1 ∧
  RowsPermutationEquivalent source.2 reconstructed.2

instance (source reconstructed : Nat × Row) :
    Decidable (IndexedRowMatchesRow source reconstructed) := by
  unfold IndexedRowMatchesRow
  infer_instance

/-- Lockstep comparison fails closed on either length mismatch. -/
def indexedRowsMatch :
    List (Nat × Row) → List (Nat × Program.Definition) → Bool
  | [], [] => true
  | source :: sources, reconstructed :: reconstructions =>
      decide (IndexedRowMatchesDefinition source reconstructed) &&
        indexedRowsMatch sources reconstructions
  | _, _ => false

/-- Lockstep assertion-row comparison, also fail-closed on length mismatch. -/
def indexedRowsMatchRows :
    List (Nat × Row) → List (Nat × Row) → Bool
  | [], [] => true
  | source :: sources, reconstructed :: reconstructions =>
      decide (IndexedRowMatchesRow source reconstructed) &&
        indexedRowsMatchRows sources reconstructions
  | _, _ => false

end Nightstream.Implementation.R1CS.ActiveIndexedRows
