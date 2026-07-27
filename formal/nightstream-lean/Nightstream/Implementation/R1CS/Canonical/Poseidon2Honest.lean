import Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout

/-!
Contract: honest completeness for the canonical Poseidon2 permutation.

Owns: the value each S-box consumes expressed through the reference, the
witness assignment built from a reference execution, and the proof that it
satisfies the 352-row program.

Does not own: soundness (`Poseidon2RoundInduction`), the reference
(`Poseidon2Reference`), or costs.

## Why the argument is not circular

The witness cannot be defined through `lcEval` of the encoding's own carried
combinations — that is what the completeness proof is trying to establish.
`sboxInputValue` is therefore defined through the *reference* states, mirroring
`scheduleOf`'s index decomposition but never mentioning a `LinComb`.  The
correspondence `honest_scheduleOf` is then a theorem rather than a definition.

This is also why the soundness inductions could not be reused.  Soundness needs
`z output = sbox7 (lcEval z (scheduleOf i))`; the witness can only supply
`z output = sbox7 (sboxInputValue i)`.  The equality of those two is exactly
what is being proved, so assuming it would beg the question.

## Why the forward direction is cheaper

A full round S-boxes every lane, so the next state depends only on that round's
eight fresh output columns and not at all on the previous state.  With the
witness pinning those columns to reference-derived values, the initial and
terminal phases need only a case split on the round — no induction hypothesis.
Only `honest_partialState` is a genuine induction, because lanes 1..7 are not
S-boxed and flow onward unchanged.  The same asymmetry drives the support
recurrence.

## Scope

Stated over `canonicalLayout` rather than an arbitrary well-formed layout.  A
concrete layout lets the witness be defined by arithmetic on the column index,
so no choice principle is needed to invert the port maps.  `WellFormed` is what
makes the general soundness results and this construction describe the same
column discipline.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout

/-! ## Reference values are canonical residues -/

theorem applyMatrixValues_lt
    (matrix : Fin width → Fin width → Nat) (values : Fin width → Nat)
    (target : Fin width) : applyMatrixValues matrix values target < goldilocksP :=
  Nat.mod_lt _ (by decide)

theorem refInitial_lt (constants : Constants) (input : Values)
    (round : Nat) (lane : Fin width) :
    refInitial constants input round lane < goldilocksP := by
  cases round <;> exact applyMatrixValues_lt _ _ _

theorem refPartial_lt (constants : Constants) (input : Values)
    (round : Nat) (lane : Fin width) :
    refPartial constants input round lane < goldilocksP := by
  cases round with
  | zero => exact refInitial_lt _ _ _ _
  | succ _ => exact applyMatrixValues_lt _ _ _

theorem refTerminal_lt (constants : Constants) (input : Values)
    (round : Nat) (lane : Fin width) :
    refTerminal constants input round lane < goldilocksP := by
  cases round with
  | zero => exact refPartial_lt _ _ _ _
  | succ _ => exact applyMatrixValues_lt _ _ _

theorem sbox7_lt (x : Nat) : sbox7 x < goldilocksP := Nat.mod_lt _ (by decide)

/-! ## The addition chain on values -/

def sboxSquare (x : Nat) : Nat := x * x % goldilocksP
def sboxFourth (x : Nat) : Nat := sboxSquare x * sboxSquare x % goldilocksP
def sboxSixth (x : Nat) : Nat := sboxSquare x * sboxFourth x % goldilocksP

theorem sbox7_eq_sixth (x : Nat) : sbox7 x = x * sboxSixth x % goldilocksP := rfl

/-- The value the `slot`-th column of an S-box frame carries. -/
def chainSlot (x : Nat) (slot : Nat) : Nat :=
  match slot with
  | 0 => sboxSquare x
  | 1 => sboxFourth x
  | 2 => sboxSixth x
  | _ => sbox7 x

theorem chainSlot_lt (x : Nat) (slot : Nat) : chainSlot x slot < goldilocksP := by
  unfold chainSlot
  split
  · exact Nat.mod_lt _ (by decide)
  · exact Nat.mod_lt _ (by decide)
  · exact Nat.mod_lt _ (by decide)
  · exact sbox7_lt x

/-! ## The value each S-box consumes

Mirrors `scheduleOf`'s index decomposition, but through the reference states.
No `LinComb` appears, which is what keeps the completeness argument free of the
statement it is proving. -/

def sboxInputValue (constants : Constants) (input : Values) (index : Nat) : Nat :=
  if index < halfFullRounds * width then
    (constants.initial (index / width) ⟨index % width, Nat.mod_lt _ (by decide)⟩
      + refInitial constants input (index / width)
          ⟨index % width, Nat.mod_lt _ (by decide)⟩) % goldilocksP
  else if index < halfFullRounds * width + partialRounds then
    (constants.internal (index - halfFullRounds * width)
      + refPartial constants input (index - halfFullRounds * width)
          ⟨0, by decide⟩) % goldilocksP
  else
    (constants.terminal ((index - (halfFullRounds * width + partialRounds)) / width)
        ⟨(index - (halfFullRounds * width + partialRounds)) % width,
          Nat.mod_lt _ (by decide)⟩
      + refTerminal constants input
          ((index - (halfFullRounds * width + partialRounds)) / width)
          ⟨(index - (halfFullRounds * width + partialRounds)) % width,
            Nat.mod_lt _ (by decide)⟩) % goldilocksP

theorem sboxInputValue_initial
    (constants : Constants) (input : Values) (round : Nat) (lane : Fin width)
    (roundLt : round < halfFullRounds) :
    sboxInputValue constants input (initialSboxIndex round lane.val)
      = (constants.initial round lane
          + refInitial constants input round lane) % goldilocksP := by
  have laneLt : lane.val < width := lane.isLt
  simp only [initialSboxIndex, width, halfFullRounds] at roundLt laneLt ⊢
  have isInitial : round * 8 + lane.val < halfFullRounds * width := by
    simp only [halfFullRounds, width]; omega
  have divEq : (round * 8 + lane.val) / width = round := by
    simp only [width]; omega
  have modEq : (round * 8 + lane.val) % width = lane.val := by
    simp only [width]; omega
  unfold sboxInputValue
  rw [if_pos isInitial, divEq]
  congr 2 <;> first
    | rfl
    | exact Fin.ext modEq
    | (congr 1; exact Fin.ext modEq)

theorem sboxInputValue_partial
    (constants : Constants) (input : Values) (round : Nat)
    (roundLt : round < partialRounds) :
    sboxInputValue constants input (partialSboxIndex round)
      = (constants.internal round
          + refPartial constants input round ⟨0, by decide⟩) % goldilocksP := by
  simp only [partialRounds] at roundLt
  have notInitial : ¬ (partialSboxIndex round < halfFullRounds * width) := by
    simp only [partialSboxIndex, halfFullRounds, width]; omega
  have isPartial :
      partialSboxIndex round < halfFullRounds * width + partialRounds := by
    simp only [partialSboxIndex, halfFullRounds, width, partialRounds]; omega
  have reduce : partialSboxIndex round - halfFullRounds * width = round := by
    simp only [partialSboxIndex, halfFullRounds, width]; omega
  unfold sboxInputValue
  rw [if_neg notInitial, if_pos isPartial, reduce]

theorem sboxInputValue_terminal
    (constants : Constants) (input : Values) (round : Nat) (lane : Fin width)
    (roundLt : round < halfFullRounds) :
    sboxInputValue constants input (terminalSboxIndex round lane.val)
      = (constants.terminal round lane
          + refTerminal constants input round lane) % goldilocksP := by
  have laneLt : lane.val < width := lane.isLt
  simp only [halfFullRounds, width] at roundLt laneLt
  have notInitial :
      ¬ (terminalSboxIndex round lane.val < halfFullRounds * width) := by
    simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]; omega
  have notPartial :
      ¬ (terminalSboxIndex round lane.val
          < halfFullRounds * width + partialRounds) := by
    simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]; omega
  have divEq :
      (terminalSboxIndex round lane.val
        - (halfFullRounds * width + partialRounds)) / width = round := by
    simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]; omega
  have modEq :
      (terminalSboxIndex round lane.val
        - (halfFullRounds * width + partialRounds)) % width = lane.val := by
    simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]; omega
  unfold sboxInputValue
  rw [if_neg notInitial, if_neg notPartial, divEq]
  congr 2 <;> first
    | rfl
    | exact Fin.ext modEq
    | (congr 1; exact Fin.ext modEq)

/-! ## The witness

Columns are laid out as constant wire, inputs, outputs, auxiliary block, so the
assignment is a case split on the column index. -/

def honestAssignment (constants : Constants) (input : Values) : Nat → Nat :=
  fun column =>
    if column = 0 then 1
    else if isInput : column < 9 then
      input ⟨column - 1, by simp only [width]; omega⟩
    else if isOutput : column < 17 then
      referencePermutation constants input
        ⟨column - 9, by simp only [width]; omega⟩
    else
      chainSlot (sboxInputValue constants input ((column - 17) / 4))
        ((column - 17) % 4)

variable (constants : Constants) (input : Values)

theorem honest_constantWire : honestAssignment constants input 0 = 1 := rfl

theorem honest_inputPort (lane : Fin width) :
    honestAssignment constants input (canonicalLayout.inputPort lane)
      = input lane := by
  have laneLt : lane.val < width := lane.isLt
  simp only [width] at laneLt
  show honestAssignment constants input (1 + lane.val) = input lane
  unfold honestAssignment
  rw [if_neg (by omega : ¬ (1 + lane.val = 0)),
    dif_pos (by omega : 1 + lane.val < 9)]
  congr 1
  exact Fin.ext (show 1 + lane.val - 1 = lane.val by omega)

theorem honest_outputPort (lane : Fin width) :
    honestAssignment constants input (canonicalLayout.outputPort lane)
      = referencePermutation constants input lane := by
  have laneLt : lane.val < width := lane.isLt
  simp only [width] at laneLt
  show honestAssignment constants input (9 + lane.val) = _
  unfold honestAssignment
  rw [if_neg (by omega : ¬ (9 + lane.val = 0)),
    dif_neg (by omega : ¬ (9 + lane.val < 9)),
    dif_pos (by omega : 9 + lane.val < 17)]
  congr 1
  exact Fin.ext (show 9 + lane.val - 9 = lane.val by omega)

theorem honest_sboxColumn (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    honestAssignment constants input (sboxColumn canonicalLayout index slot)
      = chainSlot (sboxInputValue constants input index.val) slot.val := by
  have slotLt : slot.val < columnsPerSbox := slot.isLt
  simp only [columnsPerSbox] at slotLt
  show honestAssignment constants input (17 + 4 * index.val + slot.val) = _
  unfold honestAssignment
  rw [if_neg (by omega : ¬ (17 + 4 * index.val + slot.val = 0)),
    dif_neg (by omega : ¬ (17 + 4 * index.val + slot.val < 9)),
    dif_neg (by omega : ¬ (17 + 4 * index.val + slot.val < 17))]
  have divEq : (17 + 4 * index.val + slot.val - 17) / 4 = index.val := by omega
  have modEq : (17 + 4 * index.val + slot.val - 17) % 4 = slot.val := by omega
  rw [divEq, modEq]

theorem honest_sboxOutput (index : Nat) :
    honestAssignment constants input (sboxOutput canonicalLayout index)
      = sbox7 (sboxInputValue constants input index) := by
  show honestAssignment constants input (17 + 4 * index + 3) = _
  unfold honestAssignment
  rw [if_neg (by omega : ¬ (17 + 4 * index + 3 = 0)),
    dif_neg (by omega : ¬ (17 + 4 * index + 3 < 9)),
    dif_neg (by omega : ¬ (17 + 4 * index + 3 < 17))]
  have divEq : (17 + 4 * index + 3 - 17) / 4 = index := by omega
  have modEq : (17 + 4 * index + 3 - 17) % 4 = 3 := by omega
  rw [divEq, modEq]
  rfl

/-! ## Residues -/

theorem honest_residues (inputResidues : ∀ lane, input lane < goldilocksP) :
    ∀ column, honestAssignment constants input column < goldilocksP := by
  intro column
  unfold honestAssignment
  split
  · decide
  · split
    · exact inputResidues _
    · split
      · exact refTerminal_lt _ _ _ _
      · exact chainSlot_lt _ _


/-! ## Forward state evaluation

Each state the encoding carries evaluates to the reference state at the same
point.  Note the asymmetry: the two full-round phases are `cases`, not
`induction` — a full round S-boxes every lane, so its state depends only on
that round's fresh output columns, which the witness pins directly.  Only the
partial phase needs an induction hypothesis. -/

theorem honest_initialState
    (inputResidues : ∀ lane, input lane < goldilocksP)
    (round : Nat) (roundLe : round ≤ halfFullRounds) (lane : Fin width) :
    lcEval (honestAssignment constants input)
        (initialState canonicalLayout round lane)
      = refInitial constants input round lane := by
  cases round with
  | zero =>
      simp only [initialState, refInitial]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton _ _ (honest_residues constants input inputResidues _)]
      exact honest_inputPort constants input source
  | succ previous =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      simp only [initialState, refInitial, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton _ _ (honest_residues constants input inputResidues _),
        honest_sboxOutput,
        sboxInputValue_initial constants input previous source previousLt]

theorem honest_partialState
    (inputResidues : ∀ lane, input lane < goldilocksP)
    (round : Nat) (roundLe : round ≤ partialRounds) (lane : Fin width) :
    lcEval (honestAssignment constants input)
        (partialState canonicalLayout round lane)
      = refPartial constants input round lane := by
  induction round generalizing lane with
  | zero =>
      simp only [partialState, refPartial]
      exact honest_initialState constants input inputResidues halfFullRounds
        (Nat.le_refl _) lane
  | succ previous hypothesis =>
      have previousLt : previous < partialRounds := by
        simp only [partialRounds] at roundLe ⊢; omega
      have previousLe : previous ≤ partialRounds := Nat.le_of_lt previousLt
      simp only [partialState, refPartial, partialRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      by_cases isLaneZero : source.val = 0
      · rw [if_pos isLaneZero, if_pos isLaneZero,
          lcEval_singleton _ _ (honest_residues constants input inputResidues _),
          honest_sboxOutput,
          sboxInputValue_partial constants input previous previousLt]
      · rw [if_neg isLaneZero, if_neg isLaneZero]
        exact hypothesis previousLe source

theorem honest_terminalState
    (inputResidues : ∀ lane, input lane < goldilocksP)
    (round : Nat) (roundLe : round ≤ halfFullRounds) (lane : Fin width) :
    lcEval (honestAssignment constants input)
        (terminalState canonicalLayout round lane)
      = refTerminal constants input round lane := by
  cases round with
  | zero =>
      simp only [terminalState, refTerminal]
      exact honest_partialState constants input inputResidues partialRounds
        (Nat.le_refl _) lane
  | succ previous =>
      have previousLt : previous < halfFullRounds := by
        simp only [halfFullRounds] at roundLe ⊢; omega
      simp only [terminalState, refTerminal, fullRoundValues]
      rw [lcEval_applyMatrix]
      refine applyMatrixValues_congr _ _ _ _ (fun source => ?_)
      rw [lcEval_singleton _ _ (honest_residues constants input inputResidues _),
        honest_sboxOutput,
        sboxInputValue_terminal constants input previous source previousLt]

/-! ## The scheduled combination evaluates to the consumed value

This is the theorem the whole construction exists to reach: the encoding's
carried combination and the reference-derived value agree at every S-box. -/

theorem honest_scheduleOf
    (inputResidues : ∀ lane, input lane < goldilocksP) (index : Fin sboxCount) :
    lcEval (honestAssignment constants input)
        (scheduleOf canonicalLayout constants index)
      = sboxInputValue constants input index.val := by
  have indexLt : index.val < sboxCount := index.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  by_cases isInitial : index.val < 32
  · have laneLt : index.val % 8 < width := by simp only [width]; omega
    have roundLt : index.val / 8 < halfFullRounds := by
      simp only [halfFullRounds]; omega
    have isIdx : index.val
        = initialSboxIndex (index.val / 8) (⟨index.val % 8, laneLt⟩ : Fin width).val := by
      simp only [initialSboxIndex, width]; omega
    rw [scheduleOf_initial canonicalLayout constants index (index.val / 8)
        ⟨index.val % 8, laneLt⟩ isIdx roundLt,
      initialSboxInput,
      lcEval_addConstant _ _ _ (honest_constantWire constants input),
      honest_initialState constants input inputResidues _
        (Nat.le_of_lt roundLt),
      ← sboxInputValue_initial constants input (index.val / 8)
        ⟨index.val % 8, laneLt⟩ roundLt, ← isIdx]
  · by_cases isPartial : index.val < 54
    · have roundLt : index.val - 32 < partialRounds := by
        simp only [partialRounds]; omega
      have isIdx : index.val = 32 + (index.val - 32) := by omega
      rw [Poseidon2Support.scheduleOf_partial canonicalLayout constants index
          (index.val - 32) isIdx roundLt,
        Poseidon2Support.partialSboxInput,
        lcEval_addConstant _ _ _ (honest_constantWire constants input),
        honest_partialState constants input inputResidues _
          (Nat.le_of_lt roundLt),
        ← sboxInputValue_partial constants input (index.val - 32) roundLt]
      congr 2
      simp only [partialSboxIndex, halfFullRounds, width]; omega
    · have laneLt : (index.val - 54) % 8 < width := by simp only [width]; omega
      have roundLt : (index.val - 54) / 8 < halfFullRounds := by
        simp only [halfFullRounds]; omega
      have isIdx : index.val = terminalSboxIndex ((index.val - 54) / 8)
          (⟨(index.val - 54) % 8, laneLt⟩ : Fin width).val := by
        simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]
        omega
      rw [scheduleOf_terminal canonicalLayout constants index
          ((index.val - 54) / 8) ⟨(index.val - 54) % 8, laneLt⟩ isIdx roundLt,
        terminalSboxInput,
        lcEval_addConstant _ _ _ (honest_constantWire constants input),
        honest_terminalState constants input inputResidues _
          (Nat.le_of_lt roundLt),
        ← sboxInputValue_terminal constants input ((index.val - 54) / 8)
          ⟨(index.val - 54) % 8, laneLt⟩ roundLt, ← isIdx]

/-! ## Honest completeness -/

/-- **A reference execution yields a satisfying assignment.**  Every row of the
352-row program holds under the witness built from the reference, so the
encoding accepts honest executions and the soundness direction is not vacuous. -/
theorem honest_satisfies
    (inputResidues : ∀ lane, input lane < goldilocksP) :
    Satisfies (canonicalProgram canonicalLayout constants)
      (honestAssignment constants input) := by
  intro row member
  rcases List.mem_append.1 member with inSbox | inBinding
  · rcases List.mem_flatMap.1 inSbox with ⟨index, _, rowMember⟩
    have value := honest_scheduleOf constants input inputResidues index
    have residues := honest_residues constants input inputResidues
    simp only [sboxRows, List.mem_cons, List.not_mem_nil, or_false] at rowMember
    rcases rowMember with rfl | rfl | rfl | rfl
    · simp only [RowHolds, rowSquare, frameAt]
      rw [value, lcEval_singleton _ _ (residues _),
        honest_sboxColumn constants input index ⟨0, by decide⟩]
      rfl
    · simp only [RowHolds, rowFourth, frameAt]
      rw [lcEval_singleton _ _ (residues _), lcEval_singleton _ _ (residues _),
        honest_sboxColumn constants input index ⟨0, by decide⟩,
        honest_sboxColumn constants input index ⟨1, by decide⟩]
      rfl
    · simp only [RowHolds, rowSixth, frameAt]
      rw [lcEval_singleton _ _ (residues _), lcEval_singleton _ _ (residues _),
        lcEval_singleton _ _ (residues _),
        honest_sboxColumn constants input index ⟨0, by decide⟩,
        honest_sboxColumn constants input index ⟨1, by decide⟩,
        honest_sboxColumn constants input index ⟨2, by decide⟩]
      rfl
    · simp only [RowHolds, rowSeventh, frameAt]
      rw [value, lcEval_singleton _ _ (residues _),
        lcEval_singleton _ _ (residues _),
        honest_sboxColumn constants input index ⟨2, by decide⟩,
        honest_sboxColumn constants input index ⟨3, by decide⟩]
      rfl
  · rcases List.mem_map.1 inBinding with ⟨lane, _, rfl⟩
    have residues := honest_residues constants input inputResidues
    simp only [RowHolds, bindRow]
    rw [lcEval_singleton _ 0 (residues 0), lcEval_singleton _ _ (residues _),
      honest_constantWire, Nat.mul_one, honest_outputPort]
    show lcEval (honestAssignment constants input)
        (terminalState canonicalLayout halfFullRounds lane) % goldilocksP = _
    rw [honest_terminalState constants input inputResidues halfFullRounds
      (Nat.le_refl _) lane]
    exact Nat.mod_eq_of_lt (refTerminal_lt _ _ _ _)


/-- **The two directions compose without contradiction.**  Deriving the output
port through soundness applied to the honest witness gives the same value the
witness was built from.

This is a coherence check, nothing stronger.  It is NOT witness uniqueness and
not an independent verification of the witness: `honest_outputPort` already
defines the output ports from `referencePermutation`, so the statement is
provable by construction too.  What it adds is that `honest_satisfies` really
typechecks as soundness's hypothesis and the two routes do not disagree.  The
substantive completeness result is `honest_satisfies`. -/
theorem honest_directions_agree
    (inputResidues : ∀ lane, input lane < goldilocksP) (lane : Fin width) :
    honestAssignment constants input (canonicalLayout.outputPort lane)
      = referencePermutation constants input lane := by
  rw [canonicalProgram_computes_reference canonicalLayout constants
    (honestAssignment constants input)
    (honest_residues constants input inputResidues)
    (honest_constantWire constants input)
    (honest_satisfies constants input inputResidues) lane]
  congr 1
  funext other
  exact honest_inputPort constants input other

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
