import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

/-!
Contract: honest completeness for a normalized Poseidon2 permutation entered
on an arbitrary carried sparse state.

Owns: the assignment-indexed forward argument that turns authoritative entry
values plus honest S-box/output coordinates into satisfaction of the exact
352-row normalized program.

Does not own: allocation of those coordinates, sponge chunking, or any
cryptographic property.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

section

variable
  (layout : Layout)
  (entry : State)
  (constants : Constants)
  (input : Values)
  (z : Nat → Nat)
  (residues : ∀ column, z column < goldilocksP)
  (constantWire : z 0 = 1)
  (entryAgrees : ∀ lane : Fin width, lcEval z (entry lane) = input lane)
  (sboxAgrees : ∀ (index : Fin sboxCount) (slot : Fin columnsPerSbox),
    z (sboxColumn layout index slot)
      = chainSlot (sboxInputValue constants input index.val) slot.val)
  (outputAgrees : ∀ lane : Fin width,
    z (layout.outputPort lane) = referencePermutation constants input lane)

include residues entryAgrees sboxAgrees in
theorem initialStateFrom_eval
    (round : Nat) (roundLe : round ≤ halfFullRounds) (lane : Fin width) :
    lcEval z (initialStateFrom layout entry round lane)
      = refInitial constants input round lane := by
  cases round with
  | zero =>
      simp only [initialStateFrom, refInitial]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      exact entryAgrees source
  | succ previous =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢
        omega
      simp only [initialStateFrom, refInitial, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton _ _ (residues _)]
      let index : Fin sboxCount :=
        ⟨initialSboxIndex previous source.val, by
          have sourceLt := source.isLt
          simp only [initialSboxIndex, sboxCount, externalRounds,
            width, partialRounds, halfFullRounds] at previousLt sourceLt ⊢
          omega⟩
      change z (sboxOutput layout index.val) = _
      rw [sboxOutput_eq_sboxColumn layout index,
        sboxAgrees index ⟨3, by decide⟩,
        sboxInputValue_initial constants input previous source previousLt]
      rfl

include residues entryAgrees sboxAgrees in
theorem partialState_eval
    (round : Nat) (roundLe : round ≤ partialRounds) (lane : Fin width) :
    lcEval z (partialState layout round lane)
      = refPartial constants input round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [partialState, refPartial]
      rw [← initialStateFrom_halfFull_eq layout entry]
      exact initialStateFrom_eval layout entry constants input z residues
        entryAgrees sboxAgrees halfFullRounds (Nat.le_refl _) lane
  | succ previous hypothesis =>
      have previousLt : previous < partialRounds := by
        simp only [partialRounds] at roundLe ⊢
        omega
      have previousLe : previous ≤ partialRounds := Nat.le_of_lt previousLt
      simp only [partialState, refPartial, partialRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      by_cases isLaneZero : source.val = 0
      · rw [if_pos isLaneZero, if_pos isLaneZero,
          lcEval_singleton _ _ (residues _)]
        let index : Fin sboxCount :=
          ⟨partialSboxIndex previous, by
            simp only [partialSboxIndex, sboxCount, externalRounds,
              width, partialRounds, halfFullRounds] at previousLt ⊢
            omega⟩
        change z (sboxOutput layout index.val) = _
        rw [sboxOutput_eq_sboxColumn layout index,
          sboxAgrees index ⟨3, by decide⟩,
          sboxInputValue_partial constants input previous previousLt]
        rfl
      · rw [if_neg isLaneZero, if_neg isLaneZero]
        exact hypothesis previousLe source

include residues entryAgrees sboxAgrees in
theorem terminalState_eval
    (round : Nat) (roundLe : round ≤ halfFullRounds) (lane : Fin width) :
    lcEval z (terminalState layout round lane)
      = refTerminal constants input round lane := by
  cases round with
  | zero =>
      simp only [terminalState, refTerminal]
      exact partialState_eval layout entry constants input z residues
        entryAgrees sboxAgrees partialRounds (Nat.le_refl _) lane
  | succ previous =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢
        omega
      simp only [terminalState, refTerminal, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton _ _ (residues _)]
      let index : Fin sboxCount :=
        ⟨terminalSboxIndex previous source.val, by
          have sourceLt := source.isLt
          simp only [terminalSboxIndex, sboxCount, externalRounds,
            width, partialRounds, halfFullRounds] at previousLt sourceLt ⊢
          omega⟩
      change z (sboxOutput layout index.val) = _
      rw [sboxOutput_eq_sboxColumn layout index,
        sboxAgrees index ⟨3, by decide⟩,
        sboxInputValue_terminal constants input previous source previousLt]
      rfl

include residues constantWire entryAgrees sboxAgrees in
theorem scheduleOfFrom_eval (index : Fin sboxCount) :
    lcEval z (scheduleOfFrom layout entry constants index)
      = sboxInputValue constants input index.val := by
  have indexLt : index.val < sboxCount := index.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  by_cases isInitial : index.val < 32
  · have laneLt : index.val % 8 < width := by
      simp only [width]
      omega
    have roundLt : index.val / 8 < halfFullRounds := by
      simp only [halfFullRounds]
      omega
    have isIdx : index.val
        = initialSboxIndex (index.val / 8)
          (⟨index.val % 8, laneLt⟩ : Fin width).val := by
      simp only [initialSboxIndex, width]
      omega
    rw [scheduleOfFrom_initial layout entry constants index (index.val / 8)
        ⟨index.val % 8, laneLt⟩ isIdx roundLt,
      lcEval_addConstant _ _ _ constantWire,
      initialStateFrom_eval layout entry constants input z residues entryAgrees
        sboxAgrees _ (Nat.le_of_lt roundLt),
      ← sboxInputValue_initial constants input (index.val / 8)
        ⟨index.val % 8, laneLt⟩ roundLt,
      ← isIdx]
  · have notInitial :
        ¬ (index.val < halfFullRounds * width) := by
      simpa only [halfFullRounds, width] using isInitial
    rw [scheduleOfFrom_nonInitial layout entry constants index notInitial]
    by_cases isPartial : index.val < 54
    · have roundLt : index.val - 32 < partialRounds := by
        simp only [partialRounds]
        omega
      have isIdx : index.val = 32 + (index.val - 32) := by omega
      rw [Poseidon2Support.scheduleOf_partial layout constants index
          (index.val - 32) isIdx roundLt,
        Poseidon2Support.partialSboxInput,
        lcEval_addConstant _ _ _ constantWire,
        partialState_eval layout entry constants input z residues entryAgrees
          sboxAgrees _ (Nat.le_of_lt roundLt),
        ← sboxInputValue_partial constants input (index.val - 32) roundLt]
      congr 2
      simp only [partialSboxIndex, halfFullRounds, width]
      omega
    · have laneLt : (index.val - 54) % 8 < width := by
        simp only [width]
        omega
      have roundLt : (index.val - 54) / 8 < halfFullRounds := by
        simp only [halfFullRounds]
        omega
      have isIdx : index.val = terminalSboxIndex ((index.val - 54) / 8)
          (⟨(index.val - 54) % 8, laneLt⟩ : Fin width).val := by
        simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]
        omega
      rw [scheduleOf_terminal layout constants index
          ((index.val - 54) / 8) ⟨(index.val - 54) % 8, laneLt⟩ isIdx roundLt,
        terminalSboxInput,
        lcEval_addConstant _ _ _ constantWire,
        terminalState_eval layout entry constants input z residues entryAgrees
          sboxAgrees _ (Nat.le_of_lt roundLt),
        ← sboxInputValue_terminal constants input ((index.val - 54) / 8)
          ⟨(index.val - 54) % 8, laneLt⟩ roundLt,
        ← isIdx]

include residues constantWire entryAgrees sboxAgrees outputAgrees in
/-- **A reference execution satisfies the exact normalized carried-entry
program.** -/
theorem honest_satisfies_normalizedFrom :
    Satisfies (normalizedCanonicalProgramFrom layout entry constants) z := by
  apply (satisfies_normalizeProgram _ z).2
  intro row member
  rcases List.mem_append.1 member with inSbox | inBinding
  · rcases List.mem_flatMap.1 inSbox with ⟨index, _, rowMember⟩
    have value := scheduleOfFrom_eval layout entry constants input z residues
      constantWire entryAgrees sboxAgrees index
    simp only [sboxRows, List.mem_cons, List.not_mem_nil, or_false] at rowMember
    rcases rowMember with rfl | rfl | rfl | rfl
    · simp only [RowHolds, rowSquare, frameAt]
      rw [value, lcEval_singleton _ _ (residues _),
        sboxAgrees index ⟨0, by decide⟩]
      rfl
    · simp only [RowHolds, rowFourth, frameAt]
      rw [lcEval_singleton _ _ (residues _),
        lcEval_singleton _ _ (residues _),
        sboxAgrees index ⟨0, by decide⟩,
        sboxAgrees index ⟨1, by decide⟩]
      rfl
    · simp only [RowHolds, rowSixth, frameAt]
      rw [lcEval_singleton _ _ (residues _),
        lcEval_singleton _ _ (residues _),
        lcEval_singleton _ _ (residues _),
        sboxAgrees index ⟨0, by decide⟩,
        sboxAgrees index ⟨1, by decide⟩,
        sboxAgrees index ⟨2, by decide⟩]
      rfl
    · simp only [RowHolds, rowSeventh, frameAt]
      rw [value, lcEval_singleton _ _ (residues _),
        lcEval_singleton _ _ (residues _),
        sboxAgrees index ⟨2, by decide⟩,
        sboxAgrees index ⟨3, by decide⟩]
      rfl
  · rcases List.mem_map.1 inBinding with ⟨lane, _, rfl⟩
    simp only [RowHolds, bindRow]
    rw [lcEval_singleton _ 0 (residues 0),
      lcEval_singleton _ _ (residues _), constantWire, Nat.mul_one,
      outputAgrees lane]
    unfold finalState referencePermutation
    rw [terminalState_eval layout entry constants input z residues entryAgrees
      sboxAgrees halfFullRounds (Nat.le_refl _) lane]
    exact Nat.mod_eq_of_lt (refTerminal_lt constants input _ lane)

end

end Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
