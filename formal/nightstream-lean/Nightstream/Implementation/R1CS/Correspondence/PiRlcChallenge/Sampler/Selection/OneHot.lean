import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.LinearEquality
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Selection.Acceptance

/-!
Semantic refinement of the Boolean one-hot selector family for one `Pi_RLC`
output position.

Owns: the eleven selector-bit leaves, exact one-hot sum, and existence of a
selected offset inside the production window.

Does not own: selector products, accepted/prefix/symbol bindings,
first-accepted ordering, production placement, coefficient assembly, Rust
conformance, row removal, or costs.

Emits constraints: no.

Authority boundary: a selector is only a routing witness. This file proves it
is one-hot; later binding files must still prove that its selected candidate is
accepted at the required prefix and that the output equals that candidate's
verifier-derived symbol.

| Protocol | Phase | Constraint family | Multiplicity | Lean guarantee |
|---|---|---|---:|---|
| `Pi_RLC` | sampler/selection position | selector Boolean leaves | 11 | every selector is zero or one |
| `Pi_RLC` | sampler/selection position | one-hot equality | 1 | selector sum is exactly one over the integers |
| `Pi_RLC` | sampler/selection position | selected offset | 1 witness | some offset in `[0, 11)` is selected |
| `Pi_RLC` | sampler/selection position | selector uniqueness | 10 remaining leaves | every non-selected offset is zero |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.OneHot

open Nightstream.Implementation.R1CS

private theorem range11 :
    List.range SelectionRows.selectionWindow =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] := by
  decide

def selectorTerms (position : Nat) : List (Nat × Nat) :=
  (List.range SelectionRows.selectionWindow).map
    (fun offset => (SelectionRows.selectorCol position offset, 1))

def selectorSum (assignment : Nat -> Nat) (position : Nat) : Nat :=
  (List.range SelectionRows.selectionWindow).foldl
    (fun sum offset =>
      sum + assignment (SelectionRows.selectorCol position offset)) 0

private theorem satisfies_selectionRows
    {assignment : Nat -> Nat} {position : Nat}
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    Satisfies (SelectionRows.selectionRows position) assignment := by
  intro row member
  apply satisfies row
  rw [SelectionRows.rows]
  apply List.mem_append_right
  exact List.mem_flatMap.mpr
    ⟨position, List.mem_range.mpr positionLt, member⟩

private theorem satisfies_oneHotRows
    {assignment : Nat -> Nat} {position : Nat}
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    Satisfies (SelectionRows.oneHotRows position) assignment := by
  intro row member
  apply satisfies_selectionRows positionLt satisfies row
  simp [SelectionRows.selectionRows, member]

theorem selectorBitsBoolean
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    ∀ offset, offset < SelectionRows.selectionWindow ->
      assignment (SelectionRows.selectorCol position offset) <= 1 := by
  intro offset offsetLt
  apply bitRow_le_one prime (canonical _) one
  apply satisfies_oneHotRows positionLt satisfies
  rw [SelectionRows.oneHotRows]
  exact List.mem_append_left _
    (List.mem_map.mpr ⟨offset, List.mem_range.mpr offsetLt, rfl⟩)

theorem selectorSum_le_eleven
    {assignment : Nat -> Nat} {position : Nat}
    (bits : ∀ offset, offset < SelectionRows.selectionWindow ->
      assignment (SelectionRows.selectorCol position offset) <= 1) :
    selectorSum assignment position <= 11 := by
  have bit0 := bits 0 (by decide)
  have bit1 := bits 1 (by decide)
  have bit2 := bits 2 (by decide)
  have bit3 := bits 3 (by decide)
  have bit4 := bits 4 (by decide)
  have bit5 := bits 5 (by decide)
  have bit6 := bits 6 (by decide)
  have bit7 := bits 7 (by decide)
  have bit8 := bits 8 (by decide)
  have bit9 := bits 9 (by decide)
  have bit10 := bits 10 (by decide)
  simp [selectorSum, range11]
  omega

private theorem constantOneTerms_canonical :
    Program.CanonicalTerms [(0, 1)] := by
  intro term member
  simp at member
  rcases member with rfl
  decide

theorem selectorSum_eq_one
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    selectorSum assignment position = 1 := by
  have bits := selectorBitsBoolean prime canonical one positionLt satisfies
  have sumLe := selectorSum_le_eleven bits
  have sumLtGoldilocks : selectorSum assignment position < goldilocksP := by
    have bound : 11 < goldilocksP := by decide
    omega
  have rowHolds : RowHolds assignment
      (SelectionRows.zeroEqualityRow
        (selectorTerms position ++ [(0, goldilocksP - 1)])) := by
    apply satisfies_oneHotRows positionLt satisfies
    simp [SelectionRows.oneHotRows, selectorTerms]
  have subtractionHolds : RowHolds assignment
      ⟨selectorTerms position ++ Program.negateTerms [(0, 1)],
        [(0, 1)], []⟩ := by
    simpa [SelectionRows.zeroEqualityRow, Program.negateTerms,
      Program.negCoeff] using rowHolds
  have decoded := LinearEquality.sound one
    (selectorTerms position) [(0, 1)]
    constantOneTerms_canonical subtractionHolds
  have decodedSum : selectorSum assignment position % goldilocksP = 1 := by
    simpa [selectorTerms, selectorSum, lcEval, one, range11,
      goldilocksP] using decoded
  rw [Nat.mod_eq_of_lt sumLtGoldilocks] at decodedSum
  exact decodedSum

/-- One selected routing offset exists in the exact eleven-candidate window. -/
theorem exists_selectedOffset
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    ∃ offset, offset < SelectionRows.selectionWindow ∧
      assignment (SelectionRows.selectorCol position offset) = 1 := by
  have bits := selectorBitsBoolean prime canonical one positionLt satisfies
  have sumEq := selectorSum_eq_one prime canonical one positionLt satisfies
  have bit0 := bits 0 (by decide)
  have bit1 := bits 1 (by decide)
  have bit2 := bits 2 (by decide)
  have bit3 := bits 3 (by decide)
  have bit4 := bits 4 (by decide)
  have bit5 := bits 5 (by decide)
  have bit6 := bits 6 (by decide)
  have bit7 := bits 7 (by decide)
  have bit8 := bits 8 (by decide)
  have bit9 := bits 9 (by decide)
  have bit10 := bits 10 (by decide)
  simp [selectorSum, range11] at sumEq
  have selected :
      assignment (SelectionRows.selectorCol position 0) = 1 ∨
      assignment (SelectionRows.selectorCol position 1) = 1 ∨
      assignment (SelectionRows.selectorCol position 2) = 1 ∨
      assignment (SelectionRows.selectorCol position 3) = 1 ∨
      assignment (SelectionRows.selectorCol position 4) = 1 ∨
      assignment (SelectionRows.selectorCol position 5) = 1 ∨
      assignment (SelectionRows.selectorCol position 6) = 1 ∨
      assignment (SelectionRows.selectorCol position 7) = 1 ∨
      assignment (SelectionRows.selectorCol position 8) = 1 ∨
      assignment (SelectionRows.selectorCol position 9) = 1 ∨
      assignment (SelectionRows.selectorCol position 10) = 1 := by
    omega
  rcases selected with h | h | h | h | h | h | h | h | h | h | h
  · exact ⟨0, by decide, h⟩
  · exact ⟨1, by decide, h⟩
  · exact ⟨2, by decide, h⟩
  · exact ⟨3, by decide, h⟩
  · exact ⟨4, by decide, h⟩
  · exact ⟨5, by decide, h⟩
  · exact ⟨6, by decide, h⟩
  · exact ⟨7, by decide, h⟩
  · exact ⟨8, by decide, h⟩
  · exact ⟨9, by decide, h⟩
  · exact ⟨10, by decide, h⟩

theorem selector_eq_zero_of_ne
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat} {position selected offset : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (positionLt : position < SelectionRows.outputCount)
    (selectedLt : selected < SelectionRows.selectionWindow)
    (offsetLt : offset < SelectionRows.selectionWindow)
    (selectedOne :
      assignment (SelectionRows.selectorCol position selected) = 1)
    (different : offset ≠ selected)
    (satisfies : Satisfies SelectionRows.rows assignment) :
    assignment (SelectionRows.selectorCol position offset) = 0 := by
  have bits := selectorBitsBoolean prime canonical one positionLt satisfies
  have sumEq := selectorSum_eq_one prime canonical one positionLt satisfies
  have bit0 := bits 0 (by decide)
  have bit1 := bits 1 (by decide)
  have bit2 := bits 2 (by decide)
  have bit3 := bits 3 (by decide)
  have bit4 := bits 4 (by decide)
  have bit5 := bits 5 (by decide)
  have bit6 := bits 6 (by decide)
  have bit7 := bits 7 (by decide)
  have bit8 := bits 8 (by decide)
  have bit9 := bits 9 (by decide)
  have bit10 := bits 10 (by decide)
  simp [selectorSum, range11] at sumEq
  have selectedCases : selected = 0 ∨ selected = 1 ∨ selected = 2 ∨
      selected = 3 ∨ selected = 4 ∨ selected = 5 ∨ selected = 6 ∨
      selected = 7 ∨ selected = 8 ∨ selected = 9 ∨ selected = 10 := by
    simp only [SelectionRows.selectionWindow, SelectionRows.candidateCount,
      SelectionRows.outputCount] at selectedLt
    omega
  have offsetCases : offset = 0 ∨ offset = 1 ∨ offset = 2 ∨
      offset = 3 ∨ offset = 4 ∨ offset = 5 ∨ offset = 6 ∨
      offset = 7 ∨ offset = 8 ∨ offset = 9 ∨ offset = 10 := by
    simp only [SelectionRows.selectionWindow, SelectionRows.candidateCount,
      SelectionRows.outputCount] at offsetLt
    omega
  rcases selectedCases with rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl <;>
    rcases offsetCases with rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl <;>
    simp at different <;> omega

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Selection.OneHot
