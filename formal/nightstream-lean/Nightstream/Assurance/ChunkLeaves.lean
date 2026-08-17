import Nightstream.Assurance.CompactSourceArtifact

/-!
Single-evaluation leaf predicates. A generated leaf module proves one
`chunkFacts`/`classFacts` boolean per chunk by `native_decide`; the
row chunk is expanded exactly once per proof because it enters as a
strict function argument. The split lemmas hand the assembly the same
propositional legs the composition theorems already consume. This
module owns no expansion semantics; those live in
CompactSourceArtifact.
-/

namespace Nightstream.Assurance.CompactSourceArtifact

open Nightstream.Assurance.ConstraintMinimization

/-- Census, well-formedness, plan coverage, and family presence of one
row chunk, checked in one pass over an already-expanded chunk. -/
def chunkFacts (rows : List IndexedRow) (start length bound cols : Nat)
    (families present : List String) : Bool :=
  (rows.map (fun row => row.sourceIndex) == List.range' start length) &&
    (rows.all (rowWellFormedAt bound cols) &&
      (rows.all (fun row => decide (row.family ∈ families)) &&
        present.all (fun family =>
          rows.any (fun row => decide (row.family = family)))))

theorem chunkFacts_split {rows : List IndexedRow}
    {start length bound cols : Nat} {families present : List String}
    (facts : chunkFacts rows start length bound cols families present = true) :
    (rows.map (fun row => row.sourceIndex) = List.range' start length) ∧
      ((rows.all (rowWellFormedAt bound cols) = true) ∧
        ((rows.all (fun row => decide (row.family ∈ families)) = true) ∧
          (present.all (fun family =>
            rows.any (fun row => decide (row.family = family))) = true))) := by
  unfold chunkFacts at facts
  simp only [Bool.and_eq_true, beq_iff_eq] at facts
  exact facts

theorem presence_of_chunkFacts {rows : List IndexedRow}
    {start length bound cols : Nat} {families present : List String}
    (facts : chunkFacts rows start length bound cols families present = true)
    {family : String} (member : family ∈ present) :
    rows.any (fun row => decide (row.family = family)) = true := by
  have all := (chunkFacts_split facts).2.2.2
  rw [List.all_eq_true] at all
  exact all family member

/-- Background satisfaction and override guarding of one row chunk,
checked in one pass. -/
def classFacts (values : Array Nat) (pairs : List (Nat × String))
    (rows : List IndexedRow) : Bool :=
  rows.all (fun row =>
      decide (Algebraic.Holds (backgroundFn values) row.row)) &&
    chunkGuardsOverrides pairs rows

theorem classFacts_split {values : Array Nat}
    {pairs : List (Nat × String)} {rows : List IndexedRow}
    (facts : classFacts values pairs rows = true) :
    (rows.all (fun row =>
        decide (Algebraic.Holds (backgroundFn values) row.row)) = true) ∧
      chunkGuardsOverrides pairs rows = true := by
  unfold classFacts at facts
  simp only [Bool.and_eq_true] at facts
  exact facts

end Nightstream.Assurance.CompactSourceArtifact
