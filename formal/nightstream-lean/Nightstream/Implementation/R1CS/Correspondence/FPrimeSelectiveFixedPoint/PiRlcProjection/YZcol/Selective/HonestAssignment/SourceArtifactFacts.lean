import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode

/-!
Exports retained-source artifact facts used by the bounded honest assignment.

The lookup theorem proves only what its consumer needs: a first-match result
whose decoded slot has the queried source column. It does not normalize the
eleven-thousand-slot payload or prove global uniqueness.

Owns: constant-one provenance and existence/data correctness of retained-slot
lookup for every declared source slot.

Does not own: source-program semantics, rewrite execution, selected-row
satisfaction, projection authority, or security reduction.

Emits constraints: no.

| Source-artifact leaf | Mathematical obligation | Authority class |
|---|---|---|
| constant one | distinguished source column is column zero | checked |
| retained lookup | every declared source column has a decoded first-match slot | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

theorem constantOneColumnZero :
    Materialized.Checked.constantOneColumn = 0 := by
  exact Materialized.Checked.distinguishedColumns.1

private theorem findSlotByColumn_of_member
    (slots : List SourceDecode.DecodedSourceSlot)
    (slot : SourceDecode.DecodedSourceSlot)
    (member : slot ∈ slots) :
    ∃ found,
      slots.find? (fun candidate =>
          decide (candidate.column = slot.column)) = some found ∧
        found ∈ slots ∧ found.column = slot.column := by
  induction slots with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
    rcases List.mem_cons.mp member with headEqual | tailMember
    · subst slot
      exact ⟨head, by simp [List.find?], List.mem_cons_self, rfl⟩
    · by_cases equal : head.column = slot.column
      · exact ⟨head, by simp [List.find?, equal],
          List.mem_cons_self, equal⟩
      · rcases inductionHypothesis tailMember with
          ⟨found, lookup, foundMember, foundColumn⟩
        refine ⟨found, ?_, List.mem_cons_of_mem head foundMember, foundColumn⟩
        simp [List.find?, equal, lookup]

theorem retainedSlotFast_exists
    (slot : SourceDecode.DecodedSourceSlot)
    (member : slot ∈ SourceDecode.decoded.slots) :
    ∃ found,
      SourceDecode.retainedSlotFast? slot.column = some found ∧
        found ∈ SourceDecode.decoded.slots ∧
          found.column = slot.column := by
  rw [SourceDecode.retainedSlotFast_eq]
  exact findSlotByColumn_of_member SourceDecode.decoded.slots slot member

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
