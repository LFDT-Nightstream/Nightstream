import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.IndexedRows

/-!
Shared satisfaction transport for emitted three-matrix diagnostic PiRLC projection rows.

Owns: transport from exact normalized source-row satisfaction to matched
builder equations, plus exact indexed embedding into a larger row list.

Does not own: artifact generation, semantic interpretation, whole-program
embedding facts, assignment canonicality, transcript authority, or row
removal.

Emits constraints: no.

| Theorem | Premise | Result |
|---|---|---|
| `builderRows_satisfied_of_indexedRowsMatch` | exact lockstep row match and source satisfaction | reconstructed builder rows satisfy |
| `rows_satisfied_of_indexedRowsMatchRows` | exact lockstep assertion-row match and source satisfaction | reconstructed assertion rows satisfy |
| `sourceRows_satisfied_of_embedded` | absolute-index embedding and full-row satisfaction | exact source rows satisfy |
| `rows_satisfied_of_embeddedAt` | lockstep slice embedding modulo sparse-term order | reconstructed leaf rows satisfy |
-/

namespace Nightstream.Implementation.R1CS.ActiveIndexedRows

open Nightstream.Implementation.R1CS

theorem rowHolds_of_permutationEquivalent
    {assignment : Nat → Nat} {source reconstructed : Row}
    (equivalent : RowsPermutationEquivalent source reconstructed)
    (sourceHolds : RowHolds assignment source) :
    RowHolds assignment reconstructed := by
  rcases equivalent with ⟨aPerm, bPerm, cPerm⟩
  unfold RowHolds at sourceHolds ⊢
  rw [← Program.lcEval_eq_of_perm assignment aPerm,
    ← Program.lcEval_eq_of_perm assignment bPerm,
    ← Program.lcEval_eq_of_perm assignment cPerm]
  exact sourceHolds

theorem builderRows_satisfied_of_indexedRowsMatch
    {assignment : Nat → Nat}
    (sources : List (Nat × Row))
    (definitions : List (Nat × Program.Definition))
    (matchCheck : indexedRowsMatch sources definitions = true)
    (sourceSatisfies : Satisfies (sources.map Prod.snd) assignment) :
    Satisfies
      (definitions.map fun entry => entry.2.builderRow) assignment := by
  induction sources generalizing definitions with
  | nil =>
      cases definitions with
      | nil => simp [Satisfies]
      | cons head tail => simp [indexedRowsMatch] at matchCheck
  | cons source sources inductionHypothesis =>
      cases definitions with
      | nil => simp [indexedRowsMatch] at matchCheck
      | cons definition definitions =>
          simp only [indexedRowsMatch, Bool.and_eq_true] at matchCheck
          rcases matchCheck with ⟨headMatches, tailMatches⟩
          have headRelation :
              IndexedRowMatchesDefinition source definition :=
            of_decide_eq_true headMatches
          have tailSatisfies :
              Satisfies (sources.map Prod.snd) assignment := by
            intro row member
            exact sourceSatisfies row (by simp [member])
          have tailBuilderRows := inductionHypothesis definitions
            tailMatches tailSatisfies
          intro row member
          simp only [List.map_cons, List.mem_cons] at member
          rcases member with isHead | inTail
          · subst row
            apply rowHolds_of_permutationEquivalent headRelation.2
            exact sourceSatisfies source.2 (by simp)
          · exact tailBuilderRows row inTail

theorem rows_satisfied_of_indexedRowsMatchRows
    {assignment : Nat → Nat}
    (sources reconstructed : List (Nat × Row))
    (matchCheck : indexedRowsMatchRows sources reconstructed = true)
    (sourceSatisfies : Satisfies (sources.map Prod.snd) assignment) :
    Satisfies (reconstructed.map Prod.snd) assignment := by
  induction sources generalizing reconstructed with
  | nil =>
      cases reconstructed with
      | nil => simp [Satisfies]
      | cons head tail => simp [indexedRowsMatchRows] at matchCheck
  | cons source sources inductionHypothesis =>
      cases reconstructed with
      | nil => simp [indexedRowsMatchRows] at matchCheck
      | cons row rows =>
          simp only [indexedRowsMatchRows, Bool.and_eq_true] at matchCheck
          rcases matchCheck with ⟨headMatches, tailMatches⟩
          have headRelation : IndexedRowMatchesRow source row :=
            of_decide_eq_true headMatches
          have tailSatisfies :
              Satisfies (sources.map Prod.snd) assignment := by
            intro current member
            exact sourceSatisfies current (by simp [member])
          have tailRows := inductionHypothesis rows tailMatches tailSatisfies
          intro current member
          simp only [List.map_cons, List.mem_cons] at member
          rcases member with isHead | inTail
          · subst current
            apply rowHolds_of_permutationEquivalent headRelation.2
            exact sourceSatisfies source.2 (by simp)
          · exact tailRows current inTail

/-- Exact absolute-index embedding of a generated source schedule. -/
def SourceRowsEmbedded
    (sourceRows : List (Nat × Row)) (fullRows : List Row) : Prop :=
  ∀ entry ∈ sourceRows, fullRows[entry.1]? = some entry.2

private theorem get?_eq_some_mem
    {rows : List Row} {index : Nat} {row : Row}
    (found : rows[index]? = some row) : row ∈ rows := by
  rcases List.getElem_of_getElem? found with ⟨bound, equality⟩
  rw [← equality]
  exact List.get_mem rows ⟨index, bound⟩

theorem sourceRows_satisfied_of_embedded
    {sourceRows : List (Nat × Row)}
    {fullRows : List Row} {assignment : Nat → Nat}
    (embedded : SourceRowsEmbedded sourceRows fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    Satisfies (sourceRows.map Prod.snd) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨entry, entryMember, rowEq⟩
  subst row
  exact fullSatisfies entry.2
    (get?_eq_some_mem (embedded entry entryMember))

/-- Lockstep sparse-row equivalence. The relation is local because this Lean
workspace intentionally has no Mathlib dependency. -/
def RowsPermutationEquivalentList : List Row → List Row → Prop
  | [], [] => True
  | source :: sources, reconstructed :: reconstructions =>
      RowsPermutationEquivalent source reconstructed ∧
        RowsPermutationEquivalentList sources reconstructions
  | _, _ => False

/-- One contiguous reconstructed leaf matches the normalized source slice at
`start`, preserving row order and every A/B/C linear combination while
allowing sparse terms to be sorted differently. Lockstep construction also
makes a short or long source slice fail closed. -/
def RowsEmbeddedAt
    (fullRows : List Row) (start : Nat) (reconstructed : List Row) : Prop :=
  RowsPermutationEquivalentList
    ((fullRows.drop start).take reconstructed.length) reconstructed

private theorem satisfies_of_permutationEquivalent
    {source reconstructed : List Row} {assignment : Nat → Nat}
    (equivalent : RowsPermutationEquivalentList source reconstructed)
    (sourceSatisfies : Satisfies source assignment) :
    Satisfies reconstructed assignment := by
  induction source generalizing reconstructed with
  | nil =>
      cases reconstructed with
      | nil =>
          intro row member
          simp at member
      | cons head tail => simp [RowsPermutationEquivalentList] at equivalent
  | cons sourceRow sourceRows inductionHypothesis =>
      cases reconstructed with
      | nil => simp [RowsPermutationEquivalentList] at equivalent
      | cons reconstructedRow reconstructedRows =>
          rcases equivalent with ⟨headEquivalent, tailEquivalent⟩
          intro row member
          simp only [List.mem_cons] at member
          rcases member with rfl | inTail
          · exact rowHolds_of_permutationEquivalent headEquivalent
              (sourceSatisfies sourceRow (by simp))
          · apply inductionHypothesis tailEquivalent
            · intro current currentMember
              exact sourceSatisfies current (by simp [currentMember])
            · exact inTail

/-- Satisfaction of the complete normalized source program transports to one
exact reconstructed leaf embedded at an absolute row index. -/
theorem rows_satisfied_of_embeddedAt
    {fullRows reconstructed : List Row} {start : Nat}
    {assignment : Nat → Nat}
    (embedded : RowsEmbeddedAt fullRows start reconstructed)
    (fullSatisfies : Satisfies fullRows assignment) :
    Satisfies reconstructed assignment := by
  have sourceSatisfies :
      Satisfies ((fullRows.drop start).take reconstructed.length)
        assignment := by
    intro row member
    exact fullSatisfies row
      (List.mem_of_mem_drop (List.mem_of_mem_take member))
  exact satisfies_of_permutationEquivalent embedded sourceSatisfies

end Nightstream.Implementation.R1CS.ActiveIndexedRows
