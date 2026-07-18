import Nightstream.Implementation.R1CS.Core.Program

/-!
Profile-neutral schema for comparing physical sparse R1CS rows with a
reconstructed projection program.

Owns: exact A/B/C equality modulo sparse-term order, absolute row-index
matching, fail-closed lockstep checks, and the proposition that selected rows
occur at their advertised positions in a larger program.

Does not own: any generated data, row satisfaction, semantic interpretation,
profile selection, costs, or permission to remove constraints.

Emits constraints: no.

| Relation | Mathematical obligation | Authority class |
|---|---|---|
| `RowsPermutationEquivalent` | preserve all three sparse linear combinations | exact physical row |
| `indexedRowsMatch` | match a source row to one reconstructed SSA definition at the same index | checked artifact structure |
| `indexedRowsMatchRows` | match a source assertion row to one reconstructed assertion | checked artifact structure |
| `SourceRowsEmbedded` | selected source rows occur exactly at their advertised full-program indices | checked refinement boundary |
-/

namespace Nightstream.Implementation.R1CS.ProjectionIndexedRows

open Nightstream.Implementation.R1CS

def RowsPermutationEquivalent (source reconstructed : Row) : Prop :=
  source.a.Perm reconstructed.a ∧
    source.b.Perm reconstructed.b ∧
    source.c.Perm reconstructed.c

instance (source reconstructed : Row) :
    Decidable (RowsPermutationEquivalent source reconstructed) := by
  unfold RowsPermutationEquivalent
  letI : Decidable (source.a.Perm reconstructed.a) :=
    List.decidablePerm source.a reconstructed.a
  letI : Decidable (source.b.Perm reconstructed.b) :=
    List.decidablePerm source.b reconstructed.b
  letI : Decidable (source.c.Perm reconstructed.c) :=
    List.decidablePerm source.c reconstructed.c
  infer_instance

def IndexedRowMatchesDefinition
    (source : Nat × Row)
    (reconstructed : Nat × Program.Definition) : Prop :=
  source.1 = reconstructed.1 ∧
    RowsPermutationEquivalent source.2 reconstructed.2.builderRow

instance (source : Nat × Row)
    (reconstructed : Nat × Program.Definition) :
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
    List (Nat × Row) -> List (Nat × Program.Definition) -> Bool
  | [], [] => true
  | source :: sources, reconstructed :: reconstructions =>
      decide (IndexedRowMatchesDefinition source reconstructed) &&
        indexedRowsMatch sources reconstructions
  | _, _ => false

/-- Assertion-row counterpart of `indexedRowsMatch`. -/
def indexedRowsMatchRows :
    List (Nat × Row) -> List (Nat × Row) -> Bool
  | [], [] => true
  | source :: sources, reconstructed :: reconstructions =>
      decide (IndexedRowMatchesRow source reconstructed) &&
        indexedRowsMatchRows sources reconstructions
  | _, _ => false

/-- Exact shard matches compose without weakening either shard's length or
index checks. This is the scaling path for kernel-checking bounded generated
blocks instead of one monolithic artifact. -/
theorem indexedRowsMatch_append
    {leftSources rightSources : List (Nat × Row)}
    {leftDefinitions rightDefinitions :
      List (Nat × Program.Definition)}
    (left : indexedRowsMatch leftSources leftDefinitions = true)
    (right : indexedRowsMatch rightSources rightDefinitions = true) :
    indexedRowsMatch (leftSources ++ rightSources)
      (leftDefinitions ++ rightDefinitions) = true := by
  induction leftSources generalizing leftDefinitions with
  | nil =>
      cases leftDefinitions with
      | nil => simpa [indexedRowsMatch] using right
      | cons head tail => simp [indexedRowsMatch] at left
  | cons source sources inductionHypothesis =>
      cases leftDefinitions with
      | nil => simp [indexedRowsMatch] at left
      | cons definition definitions =>
          simp only [indexedRowsMatch, Bool.and_eq_true] at left
          rcases left with ⟨headMatches, tailMatches⟩
          simp only [List.cons_append, indexedRowsMatch,
            Bool.and_eq_true, headMatches, true_and]
          exact inductionHypothesis tailMatches

/-- Assertion-row counterpart of `indexedRowsMatch_append`. -/
theorem indexedRowsMatchRows_append
    {leftSources rightSources leftRows rightRows : List (Nat × Row)}
    (left : indexedRowsMatchRows leftSources leftRows = true)
    (right : indexedRowsMatchRows rightSources rightRows = true) :
    indexedRowsMatchRows (leftSources ++ rightSources)
      (leftRows ++ rightRows) = true := by
  induction leftSources generalizing leftRows with
  | nil =>
      cases leftRows with
      | nil => simpa [indexedRowsMatchRows] using right
      | cons head tail => simp [indexedRowsMatchRows] at left
  | cons source sources inductionHypothesis =>
      cases leftRows with
      | nil => simp [indexedRowsMatchRows] at left
      | cons row rows =>
          simp only [indexedRowsMatchRows, Bool.and_eq_true] at left
          rcases left with ⟨headMatches, tailMatches⟩
          simp only [List.cons_append, indexedRowsMatchRows,
            Bool.and_eq_true, headMatches, true_and]
          exact inductionHypothesis tailMatches

/-- Lockstep sparse-row equivalence without absolute indices. -/
def RowsPermutationEquivalentList : List Row -> List Row -> Prop
  | [], [] => True
  | source :: sources, reconstructed :: reconstructions =>
      RowsPermutationEquivalent source reconstructed ∧
        RowsPermutationEquivalentList sources reconstructions
  | _, _ => False

/-- Every selected row occurs at its exact absolute index. -/
def SourceRowsEmbedded
    (sourceRows : List (Nat × Row)) (fullRows : List Row) : Prop :=
  forall entry, entry ∈ sourceRows -> fullRows[entry.1]? = some entry.2

/-- A contiguous full-program slice matches a reconstructed row schedule. -/
def RowsEmbeddedAt
    (fullRows : List Row) (start : Nat) (reconstructed : List Row) : Prop :=
  RowsPermutationEquivalentList
    ((fullRows.drop start).take reconstructed.length) reconstructed

end Nightstream.Implementation.R1CS.ProjectionIndexedRows
