import Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-!
Contract: the converse half of row-program column conservation.

`RowsCover rows columns` says every declared column occurs in at least one
emitted row operand.  It rejects an allocation whose length and placement are
correct but whose columns are never constrained.  It deliberately does not say
which operand writes a witness: an R1CS row has no syntactic write position.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.AllocationCoverage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

def RowsCover (rows : List Row) (columns : List Nat) : Prop :=
  ∀ column ∈ columns,
    ∃ row ∈ rows,
      Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column

theorem append
    (leftRows rightRows : List Row)
    (leftColumns rightColumns : List Nat)
    (left : RowsCover leftRows leftColumns)
    (right : RowsCover rightRows rightColumns) :
    RowsCover (leftRows ++ rightRows) (leftColumns ++ rightColumns) := by
  intro column member
  rcases List.mem_append.1 member with inLeft | inRight
  · rcases left column inLeft with ⟨row, rowMember, mentioned⟩
    exact ⟨row, List.mem_append_left _ rowMember, mentioned⟩
  · rcases right column inRight with ⟨row, rowMember, mentioned⟩
    exact ⟨row, List.mem_append_right _ rowMember, mentioned⟩

theorem flatMap
    {α : Type}
    (parts : List α)
    (rows : α → List Row)
    (columns : α → List Nat)
    (covered : ∀ part ∈ parts, RowsCover (rows part) (columns part)) :
    RowsCover (parts.flatMap rows) (parts.flatMap columns) := by
  intro column member
  rcases List.mem_flatMap.1 member with
    ⟨part, partMember, columnMember⟩
  rcases covered part partMember column columnMember with
    ⟨row, rowMember, mentioned⟩
  exact
    ⟨row, List.mem_flatMap.2 ⟨part, partMember, rowMember⟩, mentioned⟩

/-- Adding a declared column absent from every row is rejected. -/
theorem not_append_unused
    (rows : List Row) (columns : List Nat) (extra : Nat)
    (unused :
      ¬ ∃ row ∈ rows,
        Mentions row.a extra ∨ Mentions row.b extra ∨ Mentions row.c extra) :
    ¬ RowsCover rows (columns ++ [extra]) := by
  intro covered
  exact unused (covered extra (List.mem_append_right _ (by simp)))

end Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
