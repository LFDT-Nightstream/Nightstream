import Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes

/-!
Contract: column conservation for the emitted permutation program.

Owns: the classification of every column any emitted row can reference, and the
consequence that the program stays inside its declared column space.

Does not own: counts, ownership, or semantics.

## What conservation means here

`Poseidon2Layout` proved the *allocation* is coherent — ports are distinct from
each other and from the auxiliary block.  That says nothing about whether the
rows stay inside it.  Conservation is the other half: no emitted row touches a
column outside the allocation plus the declared shared read (the constant
wire).

Over `canonicalLayout` the column space is concrete — wire `0`, inputs `1..8`,
outputs `9..16`, auxiliaries `17..360` — so conservation is exactly
`column < canonicalColumnTotal`.  Stating it that way keeps the proof about
arithmetic on real column indices rather than about an abstract allocation
predicate.

The state classification underneath is reusable and slightly stronger: every
column a carried state can reference is either a declared input port or an
S-box output with a bounded index.  Only the post-pre-layer state references
ports at all; every later state references outputs only.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-! ## What a carried state can reference -/

theorem initialState_columns
    (layout : Layout) (round : Nat) (roundLe : round ≤ halfFullRounds)
    (lane : Fin width) (column : Nat)
    (mentioned : Mentions (initialState layout round lane) column) :
    (∃ port : Fin width, column = layout.inputPort port)
      ∨ (∃ index, index < sboxCount ∧ column = sboxOutput layout index) := by
  cases round with
  | zero =>
      simp only [initialState] at mentioned
      rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
      simp only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] at member
      exact Or.inl ⟨source, member⟩
  | succ previous =>
      simp only [initialState] at mentioned
      rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
      simp only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] at member
      have laneLt := source.isLt
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      refine Or.inr ⟨initialSboxIndex previous source.val, ?_, member⟩
      simp only [initialSboxIndex, sboxCount, externalRounds, width,
        partialRounds, halfFullRounds] at *
      omega

theorem partialState_columns
    (layout : Layout) (round : Nat) (roundLe : round ≤ partialRounds)
    (lane : Fin width) (column : Nat)
    (mentioned : Mentions (partialState layout round lane) column) :
    ∃ index, index < sboxCount ∧ column = sboxOutput layout index := by
  rcases partialSupportList_index layout round column
    (partialState_mentions_subset layout round lane column mentioned) with
    ⟨index, bound, image⟩
  refine ⟨index, ?_, image⟩
  simp only [sboxCount, externalRounds, width, partialRounds,
    halfFullRounds] at *
  omega

theorem terminalState_columns
    (layout : Layout) (round : Nat) (roundLe : round ≤ halfFullRounds)
    (lane : Fin width) (column : Nat)
    (mentioned : Mentions (terminalState layout round lane) column) :
    ∃ index, index < sboxCount ∧ column = sboxOutput layout index := by
  cases round with
  | zero =>
      exact partialState_columns layout partialRounds (Nat.le_refl _) lane column
        mentioned
  | succ previous =>
      simp only [terminalState] at mentioned
      rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
      simp only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] at member
      have laneLt := source.isLt
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      refine ⟨terminalSboxIndex previous source.val, ?_, member⟩
      simp only [terminalSboxIndex, sboxCount, externalRounds, width,
        partialRounds, halfFullRounds] at *
      omega

/-! ## What a scheduled input can reference -/

theorem scheduleOf_columns
    (layout : Layout) (constants : Constants) (index : Fin sboxCount)
    (column : Nat)
    (mentioned : Mentions (scheduleOf layout constants index) column) :
    column = 0
      ∨ (∃ port : Fin width, column = layout.inputPort port)
      ∨ (∃ other, other < sboxCount ∧ column = sboxOutput layout other) := by
  have indexLt : index.val < sboxCount := index.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  unfold scheduleOf at mentioned
  split at mentioned
  · rcases (mentions_addConstant _ _ _).1 mentioned with wire | inState
    · exact Or.inl wire
    · rcases initialState_columns layout _ (by
        simp only [halfFullRounds, width] at *; omega) _ column inState with
        port | output
      · exact Or.inr (Or.inl port)
      · exact Or.inr (Or.inr output)
  · split at mentioned
    · rcases (mentions_addConstant _ _ _).1 mentioned with wire | inState
      · exact Or.inl wire
      · exact Or.inr (Or.inr (partialState_columns layout _ (by
          simp only [halfFullRounds, width, partialRounds] at *; omega) _ column
          inState))
    · rcases (mentions_addConstant _ _ _).1 mentioned with wire | inState
      · exact Or.inl wire
      · exact Or.inr (Or.inr (terminalState_columns layout _ (by
          simp only [halfFullRounds, width, partialRounds] at *; omega) _ column
          inState))

/-! ## Conservation over the canonical layout

Every referenced column lies inside the declared space: constant wire, eight
inputs, eight outputs, 344 auxiliaries. -/

theorem canonicalLayout_inputPort_lt (lane : Fin width) :
    canonicalLayout.inputPort lane < canonicalColumnTotal := by
  have := lane.isLt
  simp only [width] at this
  show 1 + lane.val < canonicalColumnTotal
  simp only [canonicalColumnTotal, width, sboxCount, externalRounds,
    partialRounds, columnsPerSbox]
  omega

theorem canonicalLayout_outputPort_lt (lane : Fin width) :
    canonicalLayout.outputPort lane < canonicalColumnTotal := by
  have := lane.isLt
  simp only [width] at this
  show 9 + lane.val < canonicalColumnTotal
  simp only [canonicalColumnTotal, width, sboxCount, externalRounds,
    partialRounds, columnsPerSbox]
  omega

theorem canonicalLayout_sboxOutput_lt (index : Nat) (bound : index < sboxCount) :
    sboxOutput canonicalLayout index < canonicalColumnTotal := by
  simp only [sboxCount, externalRounds, width, partialRounds] at bound
  show 17 + columnsPerSbox * index + 3 < canonicalColumnTotal
  simp only [canonicalColumnTotal, width, sboxCount, externalRounds,
    partialRounds, columnsPerSbox]
  omega

theorem canonicalLayout_sboxColumn_lt
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    sboxColumn canonicalLayout index slot < canonicalColumnTotal := by
  have indexLt := index.isLt
  have slotLt := slot.isLt
  simp only [sboxCount, externalRounds, width, partialRounds,
    columnsPerSbox] at indexLt slotLt
  show 17 + columnsPerSbox * index.val + slot.val < canonicalColumnTotal
  simp only [canonicalColumnTotal, width, sboxCount, externalRounds,
    partialRounds, columnsPerSbox]
  omega

/-- **A scheduled input stays inside the declared column space.** -/
theorem scheduleOf_conservation
    (constants : Constants) (index : Fin sboxCount) (column : Nat)
    (mentioned : Mentions (scheduleOf canonicalLayout constants index) column) :
    column < canonicalColumnTotal := by
  rcases scheduleOf_columns canonicalLayout constants index column mentioned with
    wire | port | output
  · rw [wire]; simp only [canonicalColumnTotal]; decide
  · rcases port with ⟨lane, rfl⟩
    exact canonicalLayout_inputPort_lt lane
  · rcases output with ⟨other, bound, rfl⟩
    exact canonicalLayout_sboxOutput_lt other bound

/-- **The final state stays inside the declared column space.** -/
theorem finalState_conservation
    (lane : Fin width) (column : Nat)
    (mentioned : Mentions (finalState canonicalLayout lane) column) :
    column < canonicalColumnTotal := by
  rcases terminalState_columns canonicalLayout halfFullRounds (Nat.le_refl _)
    lane column mentioned with ⟨index, bound, rfl⟩
  exact canonicalLayout_sboxOutput_lt index bound


/-! ## Carried-entry columns

The sponge enters each call on a carried state rather than declared ports, so
its conservation argument needs `scheduleOfFrom`'s classification.  The state
lemmas above are already layout-generic; only the initial family changes, and
only at round 0, where the entry's own columns appear instead of input ports. -/

theorem initialStateFrom_columns
    (layout : Layout) (entry : State) (round : Nat)
    (roundLe : round ≤ halfFullRounds) (lane : Fin width) (column : Nat)
    (mentioned : Mentions (initialStateFrom layout entry round lane) column) :
    (∃ source : Fin width, Mentions (entry source) column)
      ∨ (∃ index, index < sboxCount ∧ column = sboxOutput layout index) := by
  cases round with
  | zero =>
      exact Or.inl ((initialStateFrom_zero_mentions layout entry lane column).1
        mentioned)
  | succ previous =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      rcases initialStateFrom_succ_mentions layout entry previous lane column
        mentioned with ⟨source, image⟩
      have laneLt := source.isLt
      refine Or.inr ⟨initialSboxIndex previous source.val, ?_, image⟩
      simp only [initialSboxIndex, sboxCount, externalRounds, width,
        partialRounds, halfFullRounds] at *
      omega

/-- **A carried-entry scheduled input references only the constant wire, the
entry's own columns, or S-box outputs.**  This is what a sponge-level
conservation argument composes per call. -/
theorem scheduleOfFrom_columns
    (layout : Layout) (entry : State) (constants : Constants)
    (index : Fin sboxCount) (column : Nat)
    (mentioned : Mentions (scheduleOfFrom layout entry constants index) column) :
    column = 0
      ∨ (∃ source : Fin width, Mentions (entry source) column)
      ∨ (∃ other, other < sboxCount ∧ column = sboxOutput layout other) := by
  have indexLt : index.val < sboxCount := index.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  by_cases isInitial : index.val < halfFullRounds * width
  · rw [scheduleOfFrom, if_pos isInitial] at mentioned
    rcases (mentions_addConstant _ _ _).1 mentioned with wire | inState
    · exact Or.inl wire
    · rcases initialStateFrom_columns layout entry _ (by
        simp only [halfFullRounds, width] at *; omega) _ column inState with
        fromEntry | output
      · exact Or.inr (Or.inl fromEntry)
      · exact Or.inr (Or.inr output)
  · rw [scheduleOfFrom_nonInitial layout entry constants index isInitial,
      scheduleOf] at mentioned
    rw [if_neg isInitial] at mentioned
    split at mentioned
    · rcases (mentions_addConstant _ _ _).1 mentioned with wire | inState
      · exact Or.inl wire
      · exact Or.inr (Or.inr (partialState_columns layout _ (by
          simp only [halfFullRounds, width, partialRounds] at *; omega) _ column
          inState))
    · rcases (mentions_addConstant _ _ _).1 mentioned with wire | inState
      · exact Or.inl wire
      · exact Or.inr (Or.inr (terminalState_columns layout _ (by
          simp only [halfFullRounds, width, partialRounds] at *; omega) _ column
          inState))

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Conservation
