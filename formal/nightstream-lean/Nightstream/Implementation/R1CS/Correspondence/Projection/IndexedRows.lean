import Nightstream.Implementation.R1CS.Artifacts.Projection.IndexedRows

/-!
Profile-neutral satisfaction transport for exact indexed projection rows.

Owns: transport from a physical row through sparse-term permutation, from
matched definition/assertion schedules to reconstructed rows, and from exact
full-program embedding to selected-row satisfaction.

Does not own: generated rows, assignment canonicality, projection semantics,
profile selection, whole-program embedding facts, costs, or row removal.

Emits constraints: no.

| Theorem family | Premise | Result |
|---|---|---|
| matched definition rows | exact indexed match plus source satisfaction | reconstructed builder rows satisfy |
| matched assertion rows | exact indexed match plus source satisfaction | reconstructed checks satisfy |
| embedded rows | exact absolute or contiguous embedding plus full satisfaction | selected rows satisfy |
| reverse permutation | exact lockstep row equivalence plus reconstructed satisfaction | source rows satisfy |
-/

namespace Nightstream.Implementation.R1CS.ProjectionIndexedRows

open Nightstream.Implementation.R1CS

theorem rowHolds_of_permutationEquivalent
    {assignment : Nat -> Nat} {source reconstructed : Row}
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
    {assignment : Nat -> Nat}
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
    {assignment : Nat -> Nat}
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
          have tailRows := inductionHypothesis rows
            tailMatches tailSatisfies
          intro current member
          simp only [List.map_cons, List.mem_cons] at member
          rcases member with isHead | inTail
          · subst current
            apply rowHolds_of_permutationEquivalent headRelation.2
            exact sourceSatisfies source.2 (by simp)
          · exact tailRows current inTail

private theorem get?_eq_some_mem
    {rows : List Row} {index : Nat} {row : Row}
    (found : rows[index]? = some row) : row ∈ rows := by
  rcases List.getElem_of_getElem? found with ⟨bound, equality⟩
  rw [← equality]
  exact List.get_mem rows ⟨index, bound⟩

theorem sourceRows_satisfied_of_embedded
    {sourceRows : List (Nat × Row)}
    {fullRows : List Row} {assignment : Nat -> Nat}
    (embedded : SourceRowsEmbedded sourceRows fullRows)
    (fullSatisfies : Satisfies fullRows assignment) :
    Satisfies (sourceRows.map Prod.snd) assignment := by
  intro row member
  rcases List.mem_map.mp member with
    ⟨entry, entryMember, rowEquality⟩
  subst row
  exact fullSatisfies entry.2
    (get?_eq_some_mem (embedded entry entryMember))

private theorem satisfies_of_permutationEquivalent
    {source reconstructed : List Row} {assignment : Nat -> Nat}
    (equivalent : RowsPermutationEquivalentList source reconstructed)
    (sourceSatisfies : Satisfies source assignment) :
    Satisfies reconstructed assignment := by
  induction source generalizing reconstructed with
  | nil =>
      cases reconstructed with
      | nil =>
          intro row member
          simp at member
      | cons head tail =>
          simp [RowsPermutationEquivalentList] at equivalent
  | cons sourceRow sourceRows inductionHypothesis =>
      cases reconstructed with
      | nil =>
          simp [RowsPermutationEquivalentList] at equivalent
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

/-- Satisfaction transports back from reconstructed rows to source rows under
the same exact lockstep A/B/C permutation relation. This is the reverse
direction needed when emitted rows are the checked side and source equations
are the semantic side. -/
theorem sourceRows_satisfied_of_permutationEquivalent
    {source reconstructed : List Row} {assignment : Nat -> Nat}
    (equivalent : RowsPermutationEquivalentList source reconstructed)
    (reconstructedSatisfies : Satisfies reconstructed assignment) :
    Satisfies source assignment := by
  have reverseEquivalent :
      RowsPermutationEquivalentList reconstructed source := by
    clear reconstructedSatisfies
    induction source generalizing reconstructed with
    | nil =>
        cases reconstructed with
        | nil => trivial
        | cons head tail =>
            simp [RowsPermutationEquivalentList] at equivalent
    | cons sourceRow sourceRows inductionHypothesis =>
        cases reconstructed with
        | nil =>
            simp [RowsPermutationEquivalentList] at equivalent
        | cons reconstructedRow reconstructedRows =>
            rcases equivalent with ⟨headEquivalent, tailEquivalent⟩
            exact ⟨⟨headEquivalent.1.symm, headEquivalent.2.1.symm,
              headEquivalent.2.2.symm⟩,
              inductionHypothesis tailEquivalent⟩
  exact satisfies_of_permutationEquivalent reverseEquivalent
    reconstructedSatisfies

theorem rows_satisfied_of_embeddedAt
    {fullRows reconstructed : List Row} {start : Nat}
    {assignment : Nat -> Nat}
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

end Nightstream.Implementation.R1CS.ProjectionIndexedRows
