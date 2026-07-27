import Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Support

/-!
Contract: semantic conformance of the canonical encoding to the reference
permutation.

Owns: the round induction — that every carried combination evaluates to the
corresponding reference state — and the end-to-end consequence that satisfying
the canonical program forces the output ports to the reference image of the
input ports.

Does not own: the reference itself (`Poseidon2Reference`), the schedule
(`Poseidon2Schedule`), or row counts.

## What this closes

Until now every claim about the canonical encoding was *structural*: row
counts, ownership, support.  The schedule was transcribed from the Rust round
loop by inspection, and nothing proved the encoding computed that permutation.
`canonicalProgram_computes_reference` closes that gap.

The proof is three inductions in the Rust phase order, each composing two
facts: `lcEval_applyMatrix` (a linear layer computes the matrix-vector product,
so emitting no row loses nothing) and `sboxRows_chain` (the four emitted rows
force the `1 → 2 → 4 → 6 → 7` addition chain).  The partial phase is the only one whose induction
step needs the previous state, mirroring the fact that lanes 1..7 are not
S-boxed.

## Direction

This is **soundness**: satisfaction implies the reference relation.  Honest
completeness — that a reference execution yields a satisfying assignment — is a
separate obligation and is not proved here.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference

/-! ## Helpers -/

/-- A unit-coefficient singleton reads back its column. -/
theorem lcEval_singleton (z : Nat → Nat) (column : Nat)
    (residue : z column < goldilocksP) :
    lcEval z [(column, 1)] = z column := by
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul]
  exact Nat.mod_eq_of_lt residue

theorem applyMatrixValues_congr
    (matrix : Fin width → Fin width → Nat) (f g : Fin width → Nat)
    (target : Fin width) (agree : ∀ source, f source = g source) :
    applyMatrixValues matrix f target = applyMatrixValues matrix g target := by
  rw [funext agree]

/-! ## The one consequence of satisfaction the inductions need

Each round induction consults the row program in exactly one way: every S-box
output column carries `x⁷` of its scheduled input.  `SboxChain` names that, so
the inductions depend on the row program only through it.

It does **not** let the two directions share these inductions.  Soundness needs
`z output = sbox7 (lcEval z (scheduleOf i))`; an honest witness can only supply
`z output = sbox7 (sboxInputValue i)`, and the equality of those two is what is
being proved.  `Poseidon2Honest` therefore carries its own forward argument —
cheaper, because a full round's next state depends only on that round's fresh
outputs, so only its partial phase needs an induction hypothesis. -/

def SboxChain (layout : Layout) (constants : Constants) (z : Nat → Nat) : Prop :=
  ∀ index : Fin sboxCount,
    z (sboxOutput layout index.val)
      = sbox7 (lcEval z (scheduleOf layout constants index))

/-- **The four emitted rows force `x⁷` on the scheduled combination.** -/
theorem satisfies_sboxChain
    (layout : Layout) (constants : Constants) (z : Nat → Nat)
    (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies (canonicalProgram layout constants) z) :
    SboxChain layout constants z := by
  intro index
  obtain ⟨square, fourth, sixth, output⟩ :=
    canonicalProgram_sbox_chains layout constants z residues satisfied index
  simp only [frameAt] at square fourth sixth output
  rw [sboxOutput_eq_sboxColumn]
  unfold sbox7
  rw [output, sixth, fourth, square]

/-! ## The schedule at each family

`scheduleOf_partial` lives in `Poseidon2Support`; these are its two siblings. -/

/-- The combination the initial full-round S-box for `(round, lane)` consumes. -/
def initialSboxInput (layout : Layout) (constants : Constants)
    (round : Nat) (lane : Fin width) : Poseidon2Core.LinComb :=
  addConstant (constants.initial round lane) (initialState layout round lane)

/-- The combination the terminal full-round S-box for `(round, lane)` consumes. -/
def terminalSboxInput (layout : Layout) (constants : Constants)
    (round : Nat) (lane : Fin width) : Poseidon2Core.LinComb :=
  addConstant (constants.terminal round lane) (terminalState layout round lane)

theorem scheduleOf_initial
    (layout : Layout) (constants : Constants) (index : Fin sboxCount)
    (round : Nat) (lane : Fin width)
    (isIndex : index.val = initialSboxIndex round lane.val)
    (roundLt : round < halfFullRounds) :
    scheduleOf layout constants index
      = initialSboxInput layout constants round lane := by
  have laneLt : lane.val < width := lane.isLt
  simp only [initialSboxIndex, width, halfFullRounds] at isIndex roundLt laneLt
  have isInitial : index.val < halfFullRounds * width := by
    simp only [halfFullRounds, width]; omega
  have divEq : index.val / width = round := by simp only [width]; omega
  have modEq : index.val % width = lane.val := by simp only [width]; omega
  unfold scheduleOf initialSboxInput
  rw [if_pos isInitial]
  congr 1 <;> · rw [divEq]; congr 1; exact Fin.ext modEq

theorem scheduleOf_terminal
    (layout : Layout) (constants : Constants) (index : Fin sboxCount)
    (round : Nat) (lane : Fin width)
    (isIndex : index.val = terminalSboxIndex round lane.val)
    (roundLt : round < halfFullRounds) :
    scheduleOf layout constants index
      = terminalSboxInput layout constants round lane := by
  have laneLt : lane.val < width := lane.isLt
  simp only [terminalSboxIndex, width, halfFullRounds, partialRounds]
    at isIndex roundLt laneLt
  have notInitial : ¬ (index.val < halfFullRounds * width) := by
    simp only [halfFullRounds, width]; omega
  have notPartial : ¬ (index.val < halfFullRounds * width + partialRounds) := by
    simp only [halfFullRounds, width, partialRounds]; omega
  have divEq :
      (index.val - (halfFullRounds * width + partialRounds)) / width = round := by
    simp only [halfFullRounds, width, partialRounds]; omega
  have modEq :
      (index.val - (halfFullRounds * width + partialRounds)) % width = lane.val := by
    simp only [halfFullRounds, width, partialRounds]; omega
  unfold scheduleOf terminalSboxInput
  rw [if_neg notInitial, if_neg notPartial]
  congr 1 <;> · rw [divEq]; congr 1; exact Fin.ext modEq

/-! ## Phase inductions

Each says: the combination the encoding carries evaluates to the reference
state at the same point in the round order. -/

section Conformance

variable (layout : Layout) (constants : Constants) (z : Nat → Nat)
  (residues : ∀ column, z column < goldilocksP)
  (constantWire : z 0 = 1)
  (chain : SboxChain layout constants z)

/-- The input values the encoding is applied to. -/
def inputValues : Values := fun lane => z (layout.inputPort lane)

include residues constantWire chain in
/-- **Initial full rounds.** -/
theorem initialState_eval (round : Nat) (roundLe : round ≤ halfFullRounds)
    (lane : Fin width) :
    lcEval z (initialState layout round lane)
      = refInitial constants (inputValues layout z) round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [initialState, refInitial]
      rw [lcEval_applyMatrix]
      exact applyMatrixValues_congr _ _ _ _
        (fun source => lcEval_singleton z _ (residues _))
  | succ previous hypothesis =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      have previousLe : previous ≤ halfFullRounds := Nat.le_of_lt previousLt
      simp only [initialState, refInitial, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton z _ (residues _)]
      have indexLt : initialSboxIndex previous source.val < sboxCount := by
        have := source.isLt
        simp only [initialSboxIndex, sboxCount, externalRounds, width,
          partialRounds, halfFullRounds] at *
        omega
      have schedule :=
        scheduleOf_initial layout constants
          ⟨initialSboxIndex previous source.val, indexLt⟩ previous source rfl
          previousLt
      rw [chain ⟨initialSboxIndex previous source.val, indexLt⟩, schedule,
        initialSboxInput, lcEval_addConstant z _ _ constantWire,
        hypothesis previousLe]

include residues constantWire chain in
/-- **Partial rounds.**  The only phase whose step consults the previous state,
because lanes 1..7 are not S-boxed. -/
theorem partialState_eval (round : Nat) (roundLe : round ≤ partialRounds)
    (lane : Fin width) :
    lcEval z (partialState layout round lane)
      = refPartial constants (inputValues layout z) round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [partialState, refPartial]
      exact initialState_eval layout constants z residues constantWire chain
        halfFullRounds (Nat.le_refl _) lane
  | succ previous hypothesis =>
      have previousLt : previous < partialRounds := by
        simp only [partialRounds] at roundLe ⊢; omega
      have previousLe : previous ≤ partialRounds := Nat.le_of_lt previousLt
      simp only [partialState, refPartial, partialRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      by_cases isLaneZero : source.val = 0
      · rw [if_pos isLaneZero, if_pos isLaneZero,
          lcEval_singleton z _ (residues _)]
        have indexLt : partialSboxIndex previous < sboxCount := by
          simp only [partialSboxIndex, sboxCount, externalRounds, width,
            partialRounds, halfFullRounds] at *
          omega
        have schedule :=
          Poseidon2Support.scheduleOf_partial layout constants
            ⟨partialSboxIndex previous, indexLt⟩ previous
            (by simp only [partialSboxIndex, halfFullRounds, width]) previousLt
        rw [chain ⟨partialSboxIndex previous, indexLt⟩, schedule,
          Poseidon2Support.partialSboxInput,
          lcEval_addConstant z _ _ constantWire,
          hypothesis previousLe ⟨0, by decide⟩]
      · rw [if_neg isLaneZero, if_neg isLaneZero]
        exact hypothesis previousLe source

include residues constantWire chain in
/-- **Terminal full rounds.** -/
theorem terminalState_eval (round : Nat) (roundLe : round ≤ halfFullRounds)
    (lane : Fin width) :
    lcEval z (terminalState layout round lane)
      = refTerminal constants (inputValues layout z) round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [terminalState, refTerminal]
      exact partialState_eval layout constants z residues constantWire chain
        partialRounds (Nat.le_refl _) lane
  | succ previous hypothesis =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      have previousLe : previous ≤ halfFullRounds := Nat.le_of_lt previousLt
      simp only [terminalState, refTerminal, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton z _ (residues _)]
      have indexLt : terminalSboxIndex previous source.val < sboxCount := by
        have := source.isLt
        simp only [terminalSboxIndex, sboxCount, externalRounds, width,
          partialRounds, halfFullRounds] at *
        omega
      have schedule :=
        scheduleOf_terminal layout constants
          ⟨terminalSboxIndex previous source.val, indexLt⟩ previous source rfl
          previousLt
      rw [chain ⟨terminalSboxIndex previous source.val, indexLt⟩, schedule,
        terminalSboxInput, lcEval_addConstant z _ _ constantWire,
        hypothesis previousLe]


/-! ## End to end -/

include residues constantWire in
/-- **The canonical encoding computes the reference permutation.**  Satisfying
the 352-row program forces every output port to the reference image of the
input ports.

This is the obligation `Poseidon2Core` and `Poseidon2Schedule` both deferred as
`POSEIDON2-ROUND-INDUCTION`.  Soundness direction only. -/
theorem canonicalProgram_computes_reference
    (satisfied : Satisfies (canonicalProgram layout constants) z)
    (lane : Fin width) :
    z (layout.outputPort lane)
      = referencePermutation constants (inputValues layout z) lane := by
  have bindingRow : RowHolds z (bindRow (finalState layout lane)
      (layout.outputPort lane)) := by
    refine satisfied _ (List.mem_append.2 (Or.inr ?_))
    exact List.mem_map.2 ⟨lane, List.mem_finRange lane, rfl⟩
  simp only [RowHolds, bindRow] at bindingRow
  rw [lcEval_singleton z 0 (residues 0),
    lcEval_singleton z _ (residues _), constantWire, Nat.mul_one] at bindingRow
  have lcBound : lcEval z (finalState layout lane) < goldilocksP :=
    Nat.mod_lt _ (by decide)
  rw [Nat.mod_eq_of_lt lcBound] at bindingRow
  rw [← bindingRow]
  exact terminalState_eval layout constants z residues constantWire
    (satisfies_sboxChain layout constants z residues satisfied)
    halfFullRounds (Nat.le_refl _) lane

end Conformance

/-! ## The initial phase on a carried entry

Per `POSEIDON2-SPONGE-SOUNDNESS-SHAPE`, only this phase has new content: the
base case reads the entry's values instead of the input ports, and the successor
case is unchanged because a full round has already replaced every lane. -/

theorem scheduleOfFrom_initial
    (layout : Layout) (entry : State) (constants : Constants)
    (index : Fin sboxCount) (round : Nat) (lane : Fin width)
    (isIndex : index.val = initialSboxIndex round lane.val)
    (roundLt : round < halfFullRounds) :
    scheduleOfFrom layout entry constants index
      = addConstant (constants.initial round lane)
          (initialStateFrom layout entry round lane) := by
  have laneLt : lane.val < width := lane.isLt
  simp only [initialSboxIndex, width, halfFullRounds] at isIndex roundLt laneLt
  have isInitial : index.val < halfFullRounds * width := by
    simp only [halfFullRounds, width]; omega
  have divEq : index.val / width = round := by simp only [width]; omega
  have modEq : index.val % width = lane.val := by simp only [width]; omega
  unfold scheduleOfFrom
  rw [if_pos isInitial]
  congr 1 <;> · rw [divEq]; congr 1; exact Fin.ext modEq


theorem scheduleOfFrom_nonInitial
    (layout : Layout) (entry : State) (constants : Constants)
    (index : Fin sboxCount)
    (notInitial : ¬ (index.val < halfFullRounds * width)) :
    scheduleOfFrom layout entry constants index
      = scheduleOf layout constants index := by
  unfold scheduleOfFrom; rw [if_neg notInitial]

section CarriedEntry

variable (layout : Layout) (constants : Constants) (z : Nat → Nat)
  (entry : State) (entryValues : Values)
  (residues : ∀ column, z column < goldilocksP)
  (constantWire : z 0 = 1)
  (entryAgrees : ∀ source : Fin width, lcEval z (entry source) = entryValues source)
  (chainFrom : ∀ index : Fin sboxCount,
    z (sboxOutput layout index.val)
      = sbox7 (lcEval z (scheduleOfFrom layout entry constants index)))

include residues constantWire entryAgrees chainFrom in
/-- **Initial full rounds on a carried entry.**  The reference sits at the
entry's values, not the port values — that difference is the whole reason this
theorem exists rather than reusing `initialState_eval`. -/
theorem initialStateFrom_eval (round : Nat) (roundLe : round ≤ halfFullRounds)
    (lane : Fin width) :
    lcEval z (initialStateFrom layout entry round lane)
      = refInitial constants entryValues round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [initialStateFrom, refInitial]
      rw [lcEval_applyMatrix]
      exact applyMatrixValues_congr _ _ _ _ (fun source => entryAgrees source)
  | succ previous hypothesis =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      have previousLe : previous ≤ halfFullRounds := Nat.le_of_lt previousLt
      simp only [initialStateFrom, refInitial, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton z _ (residues _)]
      have indexLt : initialSboxIndex previous source.val < sboxCount := by
        have := source.isLt
        simp only [initialSboxIndex, sboxCount, externalRounds, width,
          partialRounds, halfFullRounds] at *
        omega
      rw [chainFrom ⟨initialSboxIndex previous source.val, indexLt⟩,
        scheduleOfFrom_initial layout entry constants
          ⟨initialSboxIndex previous source.val, indexLt⟩ previous source rfl
          previousLt,
        lcEval_addConstant z _ _ constantWire, hypothesis previousLe]


include residues constantWire entryAgrees chainFrom in
/-- **Partial rounds on a carried entry.**  The *state* is `partialState`
unchanged — cycle 196 showed it is entry-independent — but the *reference* sits
at the entry values, which is the distinction `POSEIDON2-SPONGE-SOUNDNESS-SHAPE`
identified. -/
theorem partialStateFrom_eval (round : Nat) (roundLe : round ≤ partialRounds)
    (lane : Fin width) :
    lcEval z (partialState layout round lane)
      = refPartial constants entryValues round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [partialState, refPartial]
      rw [← initialStateFrom_halfFull_eq layout entry]
      exact initialStateFrom_eval layout constants z entry entryValues residues
        constantWire entryAgrees chainFrom halfFullRounds (Nat.le_refl _) lane
  | succ previous hypothesis =>
      have previousLt : previous < partialRounds := by
        simp only [partialRounds] at roundLe ⊢; omega
      have previousLe : previous ≤ partialRounds := Nat.le_of_lt previousLt
      simp only [partialState, refPartial, partialRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      by_cases isLaneZero : source.val = 0
      · rw [if_pos isLaneZero, if_pos isLaneZero,
          lcEval_singleton z _ (residues _)]
        have indexLt : partialSboxIndex previous < sboxCount := by
          simp only [partialSboxIndex, sboxCount, externalRounds, width,
            partialRounds, halfFullRounds] at *
          omega
        have notInitial : ¬ ((⟨partialSboxIndex previous, indexLt⟩ :
            Fin sboxCount).val < halfFullRounds * width) := by
          simp only [partialSboxIndex, halfFullRounds, width]; omega
        rw [chainFrom ⟨partialSboxIndex previous, indexLt⟩,
          scheduleOfFrom_nonInitial layout entry constants _ notInitial,
          Poseidon2Support.scheduleOf_partial layout constants
            ⟨partialSboxIndex previous, indexLt⟩ previous
            (by simp only [partialSboxIndex, halfFullRounds, width]) previousLt,
          Poseidon2Support.partialSboxInput,
          lcEval_addConstant z _ _ constantWire,
          hypothesis previousLe ⟨0, by decide⟩]
      · rw [if_neg isLaneZero, if_neg isLaneZero]
        exact hypothesis previousLe source

include residues constantWire entryAgrees chainFrom in
/-- **Terminal full rounds on a carried entry.** -/
theorem terminalStateFrom_eval (round : Nat) (roundLe : round ≤ halfFullRounds)
    (lane : Fin width) :
    lcEval z (terminalState layout round lane)
      = refTerminal constants entryValues round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [terminalState, refTerminal]
      exact partialStateFrom_eval layout constants z entry entryValues residues
        constantWire entryAgrees chainFrom partialRounds (Nat.le_refl _) lane
  | succ previous hypothesis =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      simp only [terminalState, refTerminal, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton z _ (residues _)]
      have indexLt : terminalSboxIndex previous source.val < sboxCount := by
        have := source.isLt
        simp only [terminalSboxIndex, sboxCount, externalRounds, width,
          partialRounds, halfFullRounds] at *
        omega
      have notInitial : ¬ ((⟨terminalSboxIndex previous source.val, indexLt⟩ :
          Fin sboxCount).val < halfFullRounds * width) := by
        have := source.isLt
        simp only [terminalSboxIndex, halfFullRounds, width, partialRounds] at *
        omega
      rw [chainFrom ⟨terminalSboxIndex previous source.val, indexLt⟩,
        scheduleOfFrom_nonInitial layout entry constants _ notInitial,
        scheduleOf_terminal layout constants
          ⟨terminalSboxIndex previous source.val, indexLt⟩ previous source rfl
          previousLt,
        terminalSboxInput, lcEval_addConstant z _ _ constantWire,
        hypothesis (Nat.le_of_lt previousLt) source]

end CarriedEntry



/-- **A carried-entry program forces the reference at the entry's values.**
This is the sponge's per-call soundness step: satisfying one call's 352 rows
puts the reference image of that call's entry on its output ports. -/
theorem canonicalProgramFrom_computes_reference
    (layout : Layout) (constants : Constants) (z : Nat → Nat)
    (entry : State) (entryValues : Values)
    (residues : ∀ column, z column < goldilocksP)
    (constantWire : z 0 = 1)
    (entryAgrees : ∀ source : Fin width,
      lcEval z (entry source) = entryValues source)
    (satisfied : Satisfies (canonicalProgramFrom layout entry constants) z)
    (lane : Fin width) :
    z (layout.outputPort lane)
      = referencePermutation constants entryValues lane := by
  have chainFrom : ∀ index : Fin sboxCount,
      z (sboxOutput layout index.val)
        = sbox7 (lcEval z (scheduleOfFrom layout entry constants index)) := by
    intro index
    obtain ⟨square, fourth, sixth, output⟩ :=
      permutationProgram_sbox_chains layout (scheduleOfFrom layout entry constants)
        (finalState layout) z residues satisfied index
    simp only [frameAt] at square fourth sixth output
    rw [sboxOutput_eq_sboxColumn]
    unfold sbox7
    rw [output, sixth, fourth, square]
  have bindingRow : RowHolds z (bindRow (finalState layout lane)
      (layout.outputPort lane)) := by
    refine satisfied _ (List.mem_append.2 (Or.inr ?_))
    exact List.mem_map.2 ⟨lane, List.mem_finRange lane, rfl⟩
  simp only [RowHolds, bindRow] at bindingRow
  rw [lcEval_singleton z 0 (residues 0),
    lcEval_singleton z _ (residues _), constantWire, Nat.mul_one] at bindingRow
  have lcBound : lcEval z (finalState layout lane) < goldilocksP :=
    Nat.mod_lt _ (by decide)
  rw [Nat.mod_eq_of_lt lcBound] at bindingRow
  rw [← bindingRow]
  exact terminalStateFrom_eval layout constants z entry entryValues residues
    constantWire entryAgrees chainFrom halfFullRounds (Nat.le_refl _) lane

end Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
