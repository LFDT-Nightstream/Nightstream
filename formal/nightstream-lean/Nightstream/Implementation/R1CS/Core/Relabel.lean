import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Contract: transport an exact R1CS gadget theorem through a checked column map.

The generated program remains authoritative.  A call-site certificate supplies
only a source-column to program-column renaming and proves that every renamed
gadget row occurs in the enclosing program.  No satisfaction or protocol
conclusion is stored in the certificate.
-/

namespace Nightstream.Implementation.R1CS.Relabel

open Nightstream.Implementation.R1CS

def column (columnMap : List Nat) (source : Nat) : Nat :=
  columnMap.getD source 0

def terms (columnMap : List Nat) (source : List (Nat × Nat)) :
    List (Nat × Nat) :=
  source.map fun term => (column columnMap term.1, term.2)

def row (columnMap : List Nat) (source : Row) : Row where
  a := terms columnMap source.a
  b := terms columnMap source.b
  c := terms columnMap source.c

def assignment (columnMap : List Nat) (programAssignment : Nat → Nat) :
    Nat → Nat :=
  fun source => programAssignment (column columnMap source)

private theorem foldl_terms (columnMap : List Nat)
    (programAssignment : Nat → Nat) (source : List (Nat × Nat)) (initial : Nat) :
    (terms columnMap source).foldl
        (fun acc term => acc + term.2 * programAssignment term.1) initial =
      source.foldl
        (fun acc term =>
          acc + term.2 * assignment columnMap programAssignment term.1) initial := by
  induction source generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [terms, List.map_cons, List.foldl, assignment, column]
      exact inductionHypothesis _

theorem lcEval_terms (columnMap : List Nat) (programAssignment : Nat → Nat)
    (source : List (Nat × Nat)) :
    lcEval programAssignment (terms columnMap source) =
      lcEval (assignment columnMap programAssignment) source := by
  unfold lcEval
  rw [foldl_terms]

theorem rowHolds_iff (columnMap : List Nat) (programAssignment : Nat → Nat)
    (source : Row) :
    RowHolds programAssignment (row columnMap source) ↔
      RowHolds (assignment columnMap programAssignment) source := by
  simp only [RowHolds, row, lcEval_terms]

theorem satisfies_mapped_iff
    (sourceRows : List Row) (columnMap : List Nat)
    (programAssignment : Nat → Nat) :
    Satisfies (sourceRows.map (row columnMap)) programAssignment ↔
      Satisfies sourceRows (assignment columnMap programAssignment) := by
  constructor
  · intro satisfies source sourceMember
    apply (rowHolds_iff columnMap programAssignment source).mp
    apply satisfies
    exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩
  · intro satisfies mapped mappedMember
    rcases List.mem_map.mp mappedMember with ⟨source, sourceMember, rfl⟩
    exact (rowHolds_iff columnMap programAssignment source).mpr
      (satisfies source sourceMember)

theorem satisfies_of_included
    {sourceRows programRows : List Row} {columnMap : List Nat}
    {programAssignment : Nat → Nat}
    (included : rowsIncluded (sourceRows.map (row columnMap)) programRows = true)
    (satisfies : Satisfies programRows programAssignment) :
    Satisfies sourceRows (assignment columnMap programAssignment) := by
  intro source sourceMember
  apply (rowHolds_iff columnMap programAssignment source).mp
  apply satisfies
  apply rowsIncluded_sound included
  exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩

theorem canonical
    {columnMap : List Nat} {programAssignment : Nat → Nat}
    (programCanonical : ∀ column, programAssignment column < goldilocksP) :
    ∀ source, assignment columnMap programAssignment source < goldilocksP := by
  intro source
  exact programCanonical _

theorem constantOne
    {columnMap : List Nat} {programAssignment : Nat → Nat}
    (mapsOne : column columnMap 0 = 0)
    (programOne : programAssignment 0 = 1) :
    assignment columnMap programAssignment 0 = 1 := by
  simpa [assignment, mapsOne]

end Nightstream.Implementation.R1CS.Relabel
