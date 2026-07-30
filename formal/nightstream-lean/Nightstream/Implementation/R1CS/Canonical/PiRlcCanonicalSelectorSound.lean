import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector

/-!
Contract: semantic soundness of the Lean-owned 54-of-64 `Pi_RLC` selector.

The proof treats selector, product, slack, and output columns solely as
non-authoritative witnesses.  Row satisfaction first derives exact Boolean
one-hot routing, then binds the selected route to the candidate recipe's
accept, prefix, and verifier-owned residue columns.

This file initially proves the physical selector result.  Composition with the
independent `FirstAccepted` list is stated after the candidate-prefix bridge.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector

private theorem finRange11 :
    List.finRange selectionWindow =
      [⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩,
        ⟨3, by decide⟩, ⟨4, by decide⟩, ⟨5, by decide⟩,
        ⟨6, by decide⟩, ⟨7, by decide⟩, ⟨8, by decide⟩,
        ⟨9, by decide⟩, ⟨10, by decide⟩] := by
  decide

private theorem range4 :
    List.range slackBitCount = [0, 1, 2, 3] := by
  decide

private theorem singleton_eval
    (assignment : Nat → Nat) (column : Nat)
    (canonical : assignment column < goldilocksP) :
    lcEval assignment [(column, 1)] = assignment column := by
  simp [lcEval, Nat.mod_eq_of_lt canonical]

private theorem one_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1) :
    lcEval assignment [(0, 1)] = 1 := by
  simp [lcEval, constantWire, goldilocksP]

def slackValue (selectorBase : Nat) {count : Nat}
    (coordinate : Fin count) (assignment : Nat → Nat) : Nat :=
  assignment (slackBitColumn selectorBase coordinate 0) +
    2 * assignment (slackBitColumn selectorBase coordinate 1) +
    4 * assignment (slackBitColumn selectorBase coordinate 2) +
    8 * assignment (slackBitColumn selectorBase coordinate 3)

theorem slackBits_le_one
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    ∀ offset, offset < slackBitCount →
      assignment (slackBitColumn selectorBase coordinate offset) ≤ 1 := by
  intro offset offsetLt
  apply bitRow_le_one prime (canonical _) constantWire
  apply satisfies_acceptanceBoundRows duplexBase u64Base candidateBase
    selectorBase count initial assignment satisfied coordinate
  unfold acceptanceBoundRows
  apply List.mem_append_left
  exact List.mem_map.mpr ⟨offset, List.mem_range.mpr offsetLt, rfl⟩

theorem slackValue_le_fifteen
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (bits : ∀ offset, offset < slackBitCount →
      assignment (slackBitColumn selectorBase coordinate offset) ≤ 1) :
    slackValue selectorBase coordinate assignment ≤ 15 := by
  have bit0 := bits 0 (by decide)
  have bit1 := bits 1 (by decide)
  have bit2 := bits 2 (by decide)
  have bit3 := bits 3 (by decide)
  unfold slackValue
  omega

private theorem slackTerms_eval
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (valueBound : slackValue selectorBase coordinate assignment ≤ 15) :
    lcEval assignment (slackTerms selectorBase coordinate) =
      slackValue selectorBase coordinate assignment := by
  have valueLt : slackValue selectorBase coordinate assignment <
      goldilocksP := by
    have modulus : 15 < goldilocksP := by decide
    omega
  unfold lcEval
  have raw :
      (slackTerms selectorBase coordinate).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
        slackValue selectorBase coordinate assignment := by
    simp [slackTerms, range4, slackValue]
  rw [raw, Nat.mod_eq_of_lt valueLt]

theorem slack_eq_value
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    assignment (slackColumn selectorBase coordinate) =
      slackValue selectorBase coordinate assignment := by
  have bits := slackBits_le_one prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
  have valueBound := slackValue_le_fifteen selectorBase coordinate assignment
    bits
  have holds :=
    satisfies_acceptanceBoundRows duplexBase u64Base candidateBase selectorBase
      count initial assignment satisfied coordinate
      ⟨[(slackColumn selectorBase coordinate, 1)], [(0, 1)],
        slackTerms selectorBase coordinate⟩
      (by simp [acceptanceBoundRows])
  have left :=
    singleton_eval assignment (slackColumn selectorBase coordinate)
      (canonical _)
  have one := one_eval assignment constantWire
  have right :=
    slackTerms_eval selectorBase coordinate assignment valueBound
  simpa [RowHolds, left, one, right,
    Nat.mod_eq_of_lt (canonical _)] using holds

/-- The six acceptance-bound rows force the final accepted count to be at
least 54 over the integers. -/
theorem finalCount_eq_outputCount_add_slack
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    assignment
        (finalCountSource duplexBase u64Base candidateBase initial coordinate) =
      outputCount + assignment (slackColumn selectorBase coordinate) := by
  have slackEq := slack_eq_value prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
  have bits := slackBits_le_one prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
  have valueBound := slackValue_le_fifteen selectorBase coordinate assignment
    bits
  have sumLt :
      assignment (slackColumn selectorBase coordinate) + outputCount <
        goldilocksP := by
    rw [slackEq]
    simp only [outputCount]
    have modulus : 69 < goldilocksP := by decide
    omega
  have holds :=
    satisfies_acceptanceBoundRows duplexBase u64Base candidateBase selectorBase
      count initial assignment satisfied coordinate
      ⟨[(finalCountSource duplexBase u64Base candidateBase initial coordinate,
          1)],
        [(0, 1)],
        [(slackColumn selectorBase coordinate, 1), (0, outputCount)]⟩
      (by simp [acceptanceBoundRows])
  have left :=
    singleton_eval assignment
      (finalCountSource duplexBase u64Base candidateBase initial coordinate)
      (canonical _)
  have one := one_eval assignment constantWire
  have right :
      lcEval assignment
          [(slackColumn selectorBase coordinate, 1), (0, outputCount)] =
        assignment (slackColumn selectorBase coordinate) + outputCount := by
    unfold lcEval
    simp only [List.foldl, Nat.one_mul, Nat.mul_one, Nat.zero_add,
      constantWire]
    rw [Nat.mod_eq_of_lt sumLt]
  have exactSum :
      assignment
          (finalCountSource duplexBase u64Base candidateBase initial
            coordinate) =
        assignment (slackColumn selectorBase coordinate) + outputCount := by
    simpa [RowHolds, left, one, right,
      Nat.mod_eq_of_lt (canonical _)] using holds
  simpa [Nat.add_comm] using exactSum

theorem enoughAccepted
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) :
    outputCount ≤
      assignment
        (finalCountSource duplexBase u64Base candidateBase initial coordinate) := by
  rw [finalCount_eq_outputCount_add_slack prime duplexBase u64Base
    candidateBase selectorBase count initial canonical constantWire satisfied
    coordinate]
  omega

def selectorSum
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (assignment : Nat → Nat) : Nat :=
  (List.finRange selectionWindow).foldl
    (fun total offset =>
      total + assignment
        (selectorColumn selectorBase coordinate position offset))
    0

theorem selectorBits_le_one
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    ∀ offset : Fin selectionWindow,
      assignment (selectorColumn selectorBase coordinate position offset) ≤
        1 := by
  intro offset
  apply bitRow_le_one prime (canonical _) constantWire
  apply satisfies_oneHotRows duplexBase u64Base candidateBase selectorBase
    count initial assignment satisfied coordinate position
  unfold oneHotRows
  apply List.mem_append_left
  exact List.mem_map.mpr ⟨offset, List.mem_finRange offset, rfl⟩

theorem selectorSum_le_eleven
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (assignment : Nat → Nat)
    (bits : ∀ offset : Fin selectionWindow,
      assignment (selectorColumn selectorBase coordinate position offset) ≤
        1) :
    selectorSum selectorBase coordinate position assignment ≤ 11 := by
  have bit0 := bits ⟨0, by decide⟩
  have bit1 := bits ⟨1, by decide⟩
  have bit2 := bits ⟨2, by decide⟩
  have bit3 := bits ⟨3, by decide⟩
  have bit4 := bits ⟨4, by decide⟩
  have bit5 := bits ⟨5, by decide⟩
  have bit6 := bits ⟨6, by decide⟩
  have bit7 := bits ⟨7, by decide⟩
  have bit8 := bits ⟨8, by decide⟩
  have bit9 := bits ⟨9, by decide⟩
  have bit10 := bits ⟨10, by decide⟩
  simp [selectorSum, finRange11]
  omega

private theorem selectorTerms_eval
    (selectorBase : Nat) {count : Nat} (coordinate : Fin count)
    (position : Fin outputCount) (assignment : Nat → Nat)
    (sumBound :
      selectorSum selectorBase coordinate position assignment ≤ 11) :
    lcEval assignment (selectorTerms selectorBase coordinate position) =
      selectorSum selectorBase coordinate position assignment := by
  have sumLt :
      selectorSum selectorBase coordinate position assignment <
        goldilocksP := by
    have modulus : 11 < goldilocksP := by decide
    omega
  unfold lcEval
  have raw :
      (selectorTerms selectorBase coordinate position).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
        selectorSum selectorBase coordinate position assignment := by
    simp [selectorTerms, selectorSum, finRange11]
  rw [raw, Nat.mod_eq_of_lt sumLt]

theorem selectorSum_eq_one
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    selectorSum selectorBase coordinate position assignment = 1 := by
  have bits := selectorBits_le_one prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
    position
  have sumBound := selectorSum_le_eleven selectorBase coordinate position
    assignment bits
  have holds :=
    satisfies_oneHotRows duplexBase u64Base candidateBase selectorBase count
      initial assignment satisfied coordinate position
      ⟨selectorTerms selectorBase coordinate position, [(0, 1)], [(0, 1)]⟩
      (by simp [oneHotRows])
  have sumEval :=
    selectorTerms_eval selectorBase coordinate position assignment sumBound
  have one := one_eval assignment constantWire
  have congruence :
      selectorSum selectorBase coordinate position assignment %
          goldilocksP =
        1 := by
    simpa [RowHolds, sumEval, one, goldilocksP] using holds
  have sumLt :
      selectorSum selectorBase coordinate position assignment <
        goldilocksP := by
    have modulus : 11 < goldilocksP := by decide
    omega
  rw [Nat.mod_eq_of_lt sumLt] at congruence
  exact congruence

/-- Every output position has a selected offset in the exact eleven-candidate
window. -/
theorem exists_selected
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    ∃ offset : Fin selectionWindow,
      assignment (selectorColumn selectorBase coordinate position offset) =
        1 := by
  have bits := selectorBits_le_one prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
    position
  have sumEq := selectorSum_eq_one prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
    position
  have bit0 := bits ⟨0, by decide⟩
  have bit1 := bits ⟨1, by decide⟩
  have bit2 := bits ⟨2, by decide⟩
  have bit3 := bits ⟨3, by decide⟩
  have bit4 := bits ⟨4, by decide⟩
  have bit5 := bits ⟨5, by decide⟩
  have bit6 := bits ⟨6, by decide⟩
  have bit7 := bits ⟨7, by decide⟩
  have bit8 := bits ⟨8, by decide⟩
  have bit9 := bits ⟨9, by decide⟩
  have bit10 := bits ⟨10, by decide⟩
  simp [selectorSum, finRange11] at sumEq
  have selected :
      assignment
          (selectorColumn selectorBase coordinate position ⟨0, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨1, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨2, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨3, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨4, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨5, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨6, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨7, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨8, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨9, by decide⟩) =
        1 ∨
      assignment
          (selectorColumn selectorBase coordinate position ⟨10, by decide⟩) =
        1 := by
    omega
  rcases selected with h | h | h | h | h | h | h | h | h | h | h
  · exact ⟨⟨0, by decide⟩, h⟩
  · exact ⟨⟨1, by decide⟩, h⟩
  · exact ⟨⟨2, by decide⟩, h⟩
  · exact ⟨⟨3, by decide⟩, h⟩
  · exact ⟨⟨4, by decide⟩, h⟩
  · exact ⟨⟨5, by decide⟩, h⟩
  · exact ⟨⟨6, by decide⟩, h⟩
  · exact ⟨⟨7, by decide⟩, h⟩
  · exact ⟨⟨8, by decide⟩, h⟩
  · exact ⟨⟨9, by decide⟩, h⟩
  · exact ⟨⟨10, by decide⟩, h⟩

theorem selector_eq_zero_of_ne
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount)
    (selected offset : Fin selectionWindow)
    (selectedOne :
      assignment (selectorColumn selectorBase coordinate position selected) =
        1)
    (different : offset ≠ selected) :
    assignment (selectorColumn selectorBase coordinate position offset) = 0 := by
  have bits := selectorBits_le_one prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
    position
  have sumEq := selectorSum_eq_one prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
    position
  have bit0 := bits ⟨0, by decide⟩
  have bit1 := bits ⟨1, by decide⟩
  have bit2 := bits ⟨2, by decide⟩
  have bit3 := bits ⟨3, by decide⟩
  have bit4 := bits ⟨4, by decide⟩
  have bit5 := bits ⟨5, by decide⟩
  have bit6 := bits ⟨6, by decide⟩
  have bit7 := bits ⟨7, by decide⟩
  have bit8 := bits ⟨8, by decide⟩
  have bit9 := bits ⟨9, by decide⟩
  have bit10 := bits ⟨10, by decide⟩
  simp [selectorSum, finRange11] at sumEq
  rcases selected with ⟨selected, selectedLt⟩
  rcases offset with ⟨offset, offsetLt⟩
  have selectedCases : selected = 0 ∨ selected = 1 ∨
      selected = 2 ∨ selected = 3 ∨ selected = 4 ∨
      selected = 5 ∨ selected = 6 ∨ selected = 7 ∨
      selected = 8 ∨ selected = 9 ∨ selected = 10 := by
    simp only [selectionWindow] at selectedLt
    omega
  have offsetCases : offset = 0 ∨ offset = 1 ∨
      offset = 2 ∨ offset = 3 ∨ offset = 4 ∨
      offset = 5 ∨ offset = 6 ∨ offset = 7 ∨
      offset = 8 ∨ offset = 9 ∨ offset = 10 := by
    simp only [selectionWindow] at offsetLt
    omega
  rcases selectedCases with rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl <;>
    rcases offsetCases with rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl <;>
    simp at different <;> omega

private theorem product_eq_source_of_selector_one
    {assignment : Nat → Nat} {selector product : Nat} {source : LinComb}
    (canonical : ∀ column, assignment column < goldilocksP)
    (selectorOne : assignment selector = 1)
    (holds : RowHolds assignment
      ⟨[(selector, 1)], source, [(product, 1)]⟩) :
    assignment product = lcEval assignment source := by
  have selectorEval := singleton_eval assignment selector (canonical _)
  have productEval := singleton_eval assignment product (canonical _)
  have sourceLt : lcEval assignment source < goldilocksP :=
    Nat.mod_lt _ (by decide)
  unfold RowHolds at holds
  rw [selectorEval, productEval, selectorOne, Nat.one_mul,
    Nat.mod_eq_of_lt sourceLt] at holds
  exact holds.symm

private theorem product_eq_zero_of_selector_zero
    {assignment : Nat → Nat} {selector product : Nat} {source : LinComb}
    (canonical : ∀ column, assignment column < goldilocksP)
    (selectorZero : assignment selector = 0)
    (holds : RowHolds assignment
      ⟨[(selector, 1)], source, [(product, 1)]⟩) :
    assignment product = 0 := by
  have selectorEval := singleton_eval assignment selector (canonical _)
  have productEval := singleton_eval assignment product (canonical _)
  unfold RowHolds at holds
  rw [selectorEval, productEval, selectorZero] at holds
  simp only [Nat.zero_mul, Nat.zero_mod] at holds
  exact holds.symm

private theorem selectedProduct_eq
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount)
    (selected offset : Fin selectionWindow)
    (selectedOne :
      assignment (selectorColumn selectorBase coordinate position selected) =
        1)
    (source : Fin selectionWindow → LinComb)
    (product : Fin selectionWindow → Nat)
    (rowAt : ∀ offset,
      RowHolds assignment
        ⟨[(selectorColumn selectorBase coordinate position offset, 1)],
          source offset, [(product offset, 1)]⟩) :
    assignment (product offset) =
      if offset = selected then lcEval assignment (source offset) else 0 := by
  by_cases same : offset = selected
  · subst offset
    simp only [↓reduceIte]
    exact product_eq_source_of_selector_one canonical selectedOne
      (rowAt selected)
  · rw [if_neg same]
    have selectorZero := selector_eq_zero_of_ne prime duplexBase u64Base
      candidateBase selectorBase count initial canonical constantWire satisfied
      coordinate position selected offset selectedOne same
    exact product_eq_zero_of_selector_zero canonical selectorZero
      (rowAt offset)

private theorem lcEval_selectedProducts
    {assignment : Nat → Nat}
    (product : Fin selectionWindow → Nat)
    (source : Fin selectionWindow → LinComb)
    (selected : Fin selectionWindow)
    (productEq : ∀ offset,
      assignment (product offset) =
        if offset = selected then lcEval assignment (source offset) else 0) :
    lcEval assignment
        ((List.finRange selectionWindow).map
          (fun offset => (product offset, 1))) =
      lcEval assignment (source selected) := by
  have product0 := productEq ⟨0, by decide⟩
  have product1 := productEq ⟨1, by decide⟩
  have product2 := productEq ⟨2, by decide⟩
  have product3 := productEq ⟨3, by decide⟩
  have product4 := productEq ⟨4, by decide⟩
  have product5 := productEq ⟨5, by decide⟩
  have product6 := productEq ⟨6, by decide⟩
  have product7 := productEq ⟨7, by decide⟩
  have product8 := productEq ⟨8, by decide⟩
  have product9 := productEq ⟨9, by decide⟩
  have product10 := productEq ⟨10, by decide⟩
  rcases selected with ⟨selected, selectedLt⟩
  have selectedCases : selected = 0 ∨ selected = 1 ∨
      selected = 2 ∨ selected = 3 ∨ selected = 4 ∨
      selected = 5 ∨ selected = 6 ∨ selected = 7 ∨
      selected = 8 ∨ selected = 9 ∨ selected = 10 := by
    simp only [selectionWindow] at selectedLt
    omega
  rcases selectedCases with rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl
  all_goals
    simp at product0 product1 product2 product3 product4 product5 product6 product7 product8 product9 product10
    rw [finRange11]
    simp [lcEval, product0, product1, product2, product3, product4,
      product5, product6, product7, product8, product9, product10]

private theorem symbolProduct_eq
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount)
    (selected offset : Fin selectionWindow)
    (selectedOne :
      assignment (selectorColumn selectorBase coordinate position selected) =
        1) :
    assignment
        (symbolProductColumn selectorBase coordinate position offset) =
      if offset = selected then
        assignment
          (symbolSource duplexBase u64Base candidateBase initial coordinate
            (candidateAt position offset))
      else 0 := by
  let source : Fin selectionWindow → LinComb := fun current =>
    [(symbolSource duplexBase u64Base candidateBase initial coordinate
      (candidateAt position current), 1)]
  have rowAt : ∀ current,
      RowHolds assignment
        ⟨[(selectorColumn selectorBase coordinate position current, 1)],
          source current,
          [(symbolProductColumn selectorBase coordinate position current,
            1)]⟩ := by
    intro current
    apply satisfies_productRowsAt duplexBase u64Base candidateBase selectorBase
      count initial assignment satisfied coordinate position current
    simp [productRowsAt, source]
  have productEq := selectedProduct_eq prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
    position selected offset selectedOne source
    (symbolProductColumn selectorBase coordinate position) rowAt
  rw [productEq]
  by_cases same : offset = selected
  · simp only [same, ↓reduceIte]
    exact singleton_eval assignment _ (canonical _)
  · simp [same]

private theorem acceptProduct_eq
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount)
    (selected offset : Fin selectionWindow)
    (selectedOne :
      assignment (selectorColumn selectorBase coordinate position selected) =
        1) :
    assignment
        (acceptProductColumn selectorBase coordinate position offset) =
      if offset = selected then
        assignment
          (acceptSource duplexBase u64Base candidateBase initial coordinate
            (candidateAt position offset))
      else 0 := by
  let source : Fin selectionWindow → LinComb := fun current =>
    [(acceptSource duplexBase u64Base candidateBase initial coordinate
      (candidateAt position current), 1)]
  have rowAt : ∀ current,
      RowHolds assignment
        ⟨[(selectorColumn selectorBase coordinate position current, 1)],
          source current,
          [(acceptProductColumn selectorBase coordinate position current,
            1)]⟩ := by
    intro current
    apply satisfies_productRowsAt duplexBase u64Base candidateBase selectorBase
      count initial assignment satisfied coordinate position current
    simp [productRowsAt, source]
  have productEq := selectedProduct_eq prime duplexBase u64Base candidateBase
    selectorBase count initial canonical constantWire satisfied coordinate
    position selected offset selectedOne source
    (acceptProductColumn selectorBase coordinate position) rowAt
  rw [productEq]
  by_cases same : offset = selected
  · simp only [same, ↓reduceIte]
    exact singleton_eval assignment _ (canonical _)
  · simp [same]

private theorem prefixProduct_eq
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount)
    (selected offset : Fin selectionWindow)
    (selectedOne :
      assignment (selectorColumn selectorBase coordinate position selected) =
        1) :
    assignment
        (prefixProductColumn selectorBase coordinate position offset) =
      if offset = selected then
        lcEval assignment
          (prefixSource duplexBase u64Base candidateBase initial coordinate
            (candidateAt position offset))
      else 0 := by
  let source : Fin selectionWindow → LinComb := fun current =>
    prefixSource duplexBase u64Base candidateBase initial coordinate
      (candidateAt position current)
  have rowAt : ∀ current,
      RowHolds assignment
        ⟨[(selectorColumn selectorBase coordinate position current, 1)],
          source current,
          [(prefixProductColumn selectorBase coordinate position current,
            1)]⟩ := by
    intro current
    apply satisfies_productRowsAt duplexBase u64Base candidateBase selectorBase
      count initial assignment satisfied coordinate position current
    simp [productRowsAt, source]
  exact selectedProduct_eq prime duplexBase u64Base candidateBase selectorBase
    count initial canonical constantWire satisfied coordinate position selected
    offset selectedOne source
    (prefixProductColumn selectorBase coordinate position) rowAt

private theorem positionTerms_eval
    {assignment : Nat → Nat}
    (constantWire : assignment 0 = 1)
    (position : Fin outputCount) :
    lcEval assignment (positionTerms position) = position.val := by
  by_cases zero : position.val = 0
  · simp [positionTerms, zero, lcEval]
  · have positionLt : position.val < goldilocksP := by
      have bounded := position.isLt
      simp only [outputCount] at bounded
      have modulus : 54 < goldilocksP := by decide
      omega
    simp [positionTerms, zero, lcEval, constantWire,
      Nat.mod_eq_of_lt positionLt]

structure PositionRefines
    (assignment : Nat → Nat)
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initial : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) (position : Fin outputCount)
    (selected : Fin selectionWindow) : Prop where
  selectorOne :
    assignment (selectorColumn selectorBase coordinate position selected) = 1
  accepted :
    assignment
        (acceptSource duplexBase u64Base candidateBase initial coordinate
          (candidateAt position selected)) =
      1
  priorCount :
    lcEval assignment
        (prefixSource duplexBase u64Base candidateBase initial coordinate
          (candidateAt position selected)) =
      position.val
  output :
    assignment (outputColumn selectorBase coordinate position) =
      (assignment
          (symbolSource duplexBase u64Base candidateBase initial coordinate
            (candidateAt position selected)) +
        (goldilocksP - 2)) % goldilocksP

/-- One complete position family selects an accepted source with exactly the
requested number of prior accepts and copies its verifier-owned residue. -/
theorem position_refines
    (prime : EuclidPrime goldilocksP)
    (duplexBase u64Base candidateBase selectorBase count : Nat)
    (initial : SymbolicDuplex.Builder)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies
        (rows duplexBase u64Base candidateBase selectorBase count initial)
        assignment)
    (coordinate : Fin count) (position : Fin outputCount) :
    ∃ selected : Fin selectionWindow,
      PositionRefines assignment duplexBase u64Base candidateBase selectorBase
        initial coordinate position selected := by
  obtain ⟨selected, selectedOne⟩ :=
    exists_selected prime duplexBase u64Base candidateBase selectorBase count
      initial canonical constantWire satisfied coordinate position
  have acceptEq : ∀ offset,
      assignment
          (acceptProductColumn selectorBase coordinate position offset) =
        if offset = selected then
          assignment
            (acceptSource duplexBase u64Base candidateBase initial coordinate
              (candidateAt position offset))
        else 0 := by
    intro offset
    exact acceptProduct_eq prime duplexBase u64Base candidateBase selectorBase
      count initial canonical constantWire satisfied coordinate position
      selected offset selectedOne
  have acceptSelected := lcEval_selectedProducts (assignment := assignment)
    (acceptProductColumn selectorBase coordinate position)
    (fun offset =>
      [(acceptSource duplexBase u64Base candidateBase initial coordinate
        (candidateAt position offset), 1)])
    selected (fun offset => by
      rw [acceptEq offset]
      by_cases same : offset = selected
      · simp only [same, ↓reduceIte]
        exact congrArg (fun value => value)
          (singleton_eval assignment _ (canonical _)).symm
      · simp [same])
  have acceptHolds :=
    satisfies_bindingRows duplexBase u64Base candidateBase selectorBase count
      initial assignment satisfied coordinate position
      ⟨acceptProductTerms selectorBase coordinate position, [(0, 1)],
        [(0, 1)]⟩
      (by simp [bindingRows])
  have one := one_eval assignment constantWire
  have accepted :
      assignment
          (acceptSource duplexBase u64Base candidateBase initial coordinate
            (candidateAt position selected)) =
        1 := by
    have congruenceMod :
        lcEval assignment
            (acceptProductTerms selectorBase coordinate position) %
            goldilocksP =
          1 := by
      simpa [RowHolds, one] using acceptHolds
    have productLt :
        lcEval assignment
            (acceptProductTerms selectorBase coordinate position) <
          goldilocksP :=
      Nat.mod_lt _ (by decide)
    rw [Nat.mod_eq_of_lt productLt] at congruenceMod
    have congruence :
        lcEval assignment
            (acceptProductTerms selectorBase coordinate position) =
          1 := congruenceMod
    have selectedEval :
        lcEval assignment
            (acceptProductTerms selectorBase coordinate position) =
          lcEval assignment
            [(acceptSource duplexBase u64Base candidateBase initial coordinate
              (candidateAt position selected), 1)] := by
      simpa [acceptProductTerms] using acceptSelected
    rw [selectedEval] at congruence
    simpa [singleton_eval assignment _ (canonical _)] using congruence

  have prefixEq : ∀ offset,
      assignment
          (prefixProductColumn selectorBase coordinate position offset) =
        if offset = selected then
          lcEval assignment
            (prefixSource duplexBase u64Base candidateBase initial coordinate
              (candidateAt position offset))
        else 0 := by
    intro offset
    exact prefixProduct_eq prime duplexBase u64Base candidateBase selectorBase
      count initial canonical constantWire satisfied coordinate position
      selected offset selectedOne
  have prefixSelected := lcEval_selectedProducts (assignment := assignment)
    (prefixProductColumn selectorBase coordinate position)
    (fun offset =>
      prefixSource duplexBase u64Base candidateBase initial coordinate
        (candidateAt position offset))
    selected prefixEq
  have prefixHolds :=
    satisfies_bindingRows duplexBase u64Base candidateBase selectorBase count
      initial assignment satisfied coordinate position
      ⟨prefixProductTerms selectorBase coordinate position, [(0, 1)],
        positionTerms position⟩
      (by simp [bindingRows])
  have positionEval := positionTerms_eval constantWire position
  have priorCount :
      lcEval assignment
          (prefixSource duplexBase u64Base candidateBase initial coordinate
            (candidateAt position selected)) =
        position.val := by
    have congruenceMod :
        lcEval assignment
            (prefixProductTerms selectorBase coordinate position) %
            goldilocksP =
          position.val := by
      simpa [RowHolds, one, positionEval] using prefixHolds
    have productLt :
        lcEval assignment
            (prefixProductTerms selectorBase coordinate position) <
          goldilocksP :=
      Nat.mod_lt _ (by decide)
    rw [Nat.mod_eq_of_lt productLt] at congruenceMod
    have congruence :
        lcEval assignment
            (prefixProductTerms selectorBase coordinate position) =
          position.val := congruenceMod
    have selectedEval :
        lcEval assignment
            (prefixProductTerms selectorBase coordinate position) =
          lcEval assignment
            (prefixSource duplexBase u64Base candidateBase initial coordinate
              (candidateAt position selected)) := by
      simpa [prefixProductTerms] using prefixSelected
    rw [selectedEval] at congruence
    exact congruence

  have symbolEq : ∀ offset,
      assignment
          (symbolProductColumn selectorBase coordinate position offset) =
        if offset = selected then
          assignment
            (symbolSource duplexBase u64Base candidateBase initial coordinate
              (candidateAt position offset))
        else 0 := by
    intro offset
    exact symbolProduct_eq prime duplexBase u64Base candidateBase selectorBase
      count initial canonical constantWire satisfied coordinate position
      selected offset selectedOne
  have symbolSelected := lcEval_selectedProducts (assignment := assignment)
    (symbolProductColumn selectorBase coordinate position)
    (fun offset =>
      [(symbolSource duplexBase u64Base candidateBase initial coordinate
        (candidateAt position offset), 1)])
    selected (fun offset => by
      rw [symbolEq offset]
      by_cases same : offset = selected
      · simp only [same, ↓reduceIte]
        exact congrArg (fun value => value)
          (singleton_eval assignment _ (canonical _)).symm
      · simp [same])
  have symbolHolds :=
    satisfies_bindingRows duplexBase u64Base candidateBase selectorBase count
      initial assignment satisfied coordinate position
      ⟨[(outputColumn selectorBase coordinate position, 1)], [(0, 1)],
        centeredSymbolTerms selectorBase coordinate position⟩
      (by simp [bindingRows])
  have outputEval :=
    singleton_eval assignment (outputColumn selectorBase coordinate position)
      (canonical _)
  have output :
      assignment (outputColumn selectorBase coordinate position) =
        (assignment
            (symbolSource duplexBase u64Base candidateBase initial coordinate
              (candidateAt position selected)) +
          (goldilocksP - 2)) % goldilocksP := by
    have equation :
        assignment (outputColumn selectorBase coordinate position) =
          lcEval assignment
            (centeredSymbolTerms selectorBase coordinate position) := by
      simpa [RowHolds, outputEval, one,
        Nat.mod_eq_of_lt (canonical _)] using symbolHolds
    have selectedEval :
        lcEval assignment
            (symbolProductTerms selectorBase coordinate position) =
          lcEval assignment
            [(symbolSource duplexBase u64Base candidateBase initial coordinate
              (candidateAt position selected), 1)] := by
      simpa [symbolProductTerms] using symbolSelected
    have centeredEval :
        lcEval assignment
            (centeredSymbolTerms selectorBase coordinate position) =
          (lcEval assignment
              (symbolProductTerms selectorBase coordinate position) +
            (goldilocksP - 2)) % goldilocksP := by
      rw [centeredSymbolTerms, KHorner.lcEval_append]
      simp [lcEval, constantWire, goldilocksP]
    rw [centeredEval, selectedEval] at equation
    simpa [singleton_eval assignment _ (canonical _)] using equation
  exact ⟨selected, {
    selectorOne := selectedOne
    accepted := accepted
    priorCount := priorCount
    output := output
  }⟩

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorSound
