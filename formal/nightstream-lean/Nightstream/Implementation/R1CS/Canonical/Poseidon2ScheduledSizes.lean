import Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
import Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

/-!
Contract: the exact normalized size of every scheduled S-box input.

Owns: the per-family state sizes, the constant-wire increment, and the sum over
all 86 S-boxes.

Does not own: row counts, the support recurrence, or cancellation.

## Shape

Each S-box consumes its lane's state plus a round constant on column 0.  The
state size is eight for every full-round state — a full round S-boxes all lanes,
so its state references exactly that round's eight fresh outputs — and
`8 + round` inside the partial block, where lanes 1..7 carry support forward.
The constant wire adds one, and it is always fresh because every state column
is either an S-box output (at least `auxBase + 3`) or a declared input port
(nonzero by `WellFormed`).

    initial full   [0, 32)   9
    partial        [32, 54)  9 + (index - 32)
    terminal r=0   [54, 62)  31        (its state IS partialState 22)
    terminal r≥1   [62, 86)  9

Terminal round 0 is the maximum and the only place 31 occurs.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Coefficients
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction

/-! ## The constant wire is always a fresh column -/

theorem normalize_addConstant_length
    (constant : Nat) (comb : Poseidon2Core.LinComb)
    (fresh : ¬ Mentions comb 0) :
    (normalize (addConstant constant comb)).length
      = (normalize comb).length + 1 := by
  unfold addConstant normalize
  simp only [List.foldr_cons]
  exact insertTerm_length_of_fresh (0, constant) _
    (fun mentioned => fresh ((mentions_normalize comb 0).1 mentioned))

theorem applyMatrix_singletons_not_zero
    (matrix : Fin width → Fin width → Nat) (f : Fin width → Nat)
    (nonzero : ∀ source, f source ≠ 0) (target : Fin width) :
    ¬ Mentions (applyMatrix matrix (fun lane => [(f lane, 1)]) target) 0 := by
  intro mentioned
  rcases (mentions_applyMatrix _ _ _ _).1 mentioned with ⟨source, member⟩
  simp only [Mentions, List.map_cons, List.map_nil, List.mem_singleton] at member
  exact nonzero source member.symm

theorem sboxOutput_ne_zero (layout : Layout) (index : Nat) :
    sboxOutput layout index ≠ 0 := by
  simp only [sboxOutput, columnsPerSbox]; omega

theorem partialState_not_mentions_zero
    (layout : Layout) (round : Nat) (lane : Fin width) :
    ¬ Mentions (partialState layout round lane) 0 := by
  intro mentioned
  rcases partialSupportList_index layout round 0
    (partialState_mentions_subset layout round lane 0 mentioned) with
    ⟨index, _, image⟩
  exact sboxOutput_ne_zero layout index image.symm

/-! ## Full-round states reference exactly eight columns -/

theorem initialState_normalize_length
    (layout : Layout) (wellFormed : WellFormed layout)
    (round : Nat) (lane : Fin width) :
    (normalize (initialState layout round lane)).length = width := by
  cases round with
  | zero =>
      exact normalize_length_applyMatrix_singletons _ _
        (fun a b image => wellFormed.inputInjective a b image) lane
  | succ previous =>
      refine normalize_length_applyMatrix_singletons _ _ (fun a b image => ?_) lane
      have := sboxOutput_injective layout _ _ image
      simp only [initialSboxIndex] at this
      exact Fin.ext (by omega)

theorem initialState_not_mentions_zero
    (layout : Layout) (wellFormed : WellFormed layout)
    (round : Nat) (lane : Fin width) :
    ¬ Mentions (initialState layout round lane) 0 := by
  cases round with
  | zero =>
      exact applyMatrix_singletons_not_zero _ _
        (fun source => wellFormed.inputNotConstantWire source) lane
  | succ previous =>
      exact applyMatrix_singletons_not_zero _ _
        (fun source => sboxOutput_ne_zero layout _) lane

theorem terminalState_succ_normalize_length
    (layout : Layout) (round : Nat) (lane : Fin width) :
    (normalize (terminalState layout (round + 1) lane)).length = width := by
  refine normalize_length_applyMatrix_singletons _ _ (fun a b image => ?_) lane
  have := sboxOutput_injective layout _ _ image
  simp only [terminalSboxIndex] at this
  exact Fin.ext (by omega)

theorem terminalState_succ_not_mentions_zero
    (layout : Layout) (round : Nat) (lane : Fin width) :
    ¬ Mentions (terminalState layout (round + 1) lane) 0 :=
  applyMatrix_singletons_not_zero _ _
    (fun source => sboxOutput_ne_zero layout _) lane

/-! ## The pointwise size -/

/-- The normalized size of the combination S-box `index` consumes. -/
def scheduledSize (index : Nat) : Nat :=
  if index < 32 then 9
  else if index < 54 then 9 + (index - 32)
  else if index < 62 then 31
  else 9

theorem scheduledSize_sum : ((List.finRange sboxCount).map
    (fun index => scheduledSize index.val)).sum = 1181 := by decide


/-- **Every scheduled input's normalized size.**  Each case is the state size
plus one for the constant wire. -/
theorem scheduledSizes_pointwise
    (layout : Layout) (wellFormed : WellFormed layout) (constants : Constants)
    (index : Fin sboxCount) :
    (normalize (scheduleOf layout constants index)).length
      = scheduledSize index.val := by
  have indexLt : index.val < sboxCount := index.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  unfold scheduledSize
  by_cases isInitial : index.val < 32
  · have laneLt : index.val % 8 < width := by simp only [width]; omega
    have roundLt : index.val / 8 < halfFullRounds := by
      simp only [halfFullRounds]; omega
    have isIdx : index.val
        = initialSboxIndex (index.val / 8) (⟨index.val % 8, laneLt⟩ : Fin width).val := by
      simp only [initialSboxIndex, width]; omega
    rw [if_pos isInitial,
      scheduleOf_initial layout constants index _ _ isIdx roundLt,
      initialSboxInput,
      normalize_addConstant_length _ _
        (initialState_not_mentions_zero layout wellFormed _ _),
      initialState_normalize_length layout wellFormed]
    decide
  · by_cases isPartial : index.val < 54
    · have roundLt : index.val - 32 < partialRounds := by
        simp only [partialRounds]; omega
      have isIdx : index.val = 32 + (index.val - 32) := by omega
      rw [if_neg isInitial, if_pos isPartial,
        Poseidon2Support.scheduleOf_partial layout constants index _ isIdx roundLt,
        Poseidon2Support.partialSboxInput,
        normalize_addConstant_length _ _
          (partialState_not_mentions_zero layout _ _),
        partialState_normalize_length]
      simp only [width]; omega
    · have laneLt : (index.val - 54) % 8 < width := by simp only [width]; omega
      have roundLt : (index.val - 54) / 8 < halfFullRounds := by
        simp only [halfFullRounds]; omega
      have isIdx : index.val = terminalSboxIndex ((index.val - 54) / 8)
          (⟨(index.val - 54) % 8, laneLt⟩ : Fin width).val := by
        simp only [terminalSboxIndex, halfFullRounds, width, partialRounds]
        omega
      rw [if_neg isInitial, if_neg isPartial,
        scheduleOf_terminal layout constants index _ _ isIdx roundLt,
        terminalSboxInput]
      by_cases isRoundZero : index.val < 62
      · have zeroRound : (index.val - 54) / 8 = 0 := by omega
        rw [if_pos isRoundZero, zeroRound]
        show (normalize (addConstant _ (terminalState layout 0 _))).length = 31
        simp only [terminalState]
        rw [normalize_addConstant_length _ _
            (partialState_not_mentions_zero layout _ _),
          partialState_normalize_length]
        simp only [width, partialRounds]
      · have positiveRound : 1 ≤ (index.val - 54) / 8 := by omega
        rw [if_neg isRoundZero]
        obtain ⟨previous, isSucc⟩ : ∃ previous, (index.val - 54) / 8 = previous + 1 :=
          ⟨(index.val - 54) / 8 - 1, by omega⟩
        rw [isSucc, normalize_addConstant_length _ _
            (terminalState_succ_not_mentions_zero layout _ _),
          terminalState_succ_normalize_length]
        decide


/-! ## The totals

Both summands of `canonicalProgram_termCount` are now theorems, so the
structural term total is a numeral. -/

theorem scheduledSizes_sum
    (layout : Layout) (wellFormed : WellFormed layout) (constants : Constants) :
    (scheduledSizes layout constants).sum = 1181 := by
  unfold scheduledSizes
  rw [show (fun index => (normalize (scheduleOf layout constants index)).length)
        = (fun index : Fin sboxCount => scheduledSize index.val) from
      funext (fun index => scheduledSizes_pointwise layout wellFormed constants index)]
  exact scheduledSize_sum

/-- **The structural term total of the canonical permutation: 4397.**  Derived
from the emitted rows through a receipt fold — `3·1181 + 9·86` for the S-box
families, `64 + 16` for the terminal binding. Nothing is declared, measured, or
read from an artifact. -/
theorem canonicalProgram_termCount_eq
    (layout : Layout) (wellFormed : WellFormed layout) (constants : Constants) :
    programTermCount (canonicalProgram layout constants) = 4397 := by
  rw [canonicalProgram_termCount, scheduledSizes_sum layout wellFormed constants,
    finalSizes_sum layout]
  decide

/-- **The emitted program carries at most 4397 nonzero coefficients.**  Equality
holds exactly when no coefficient cancels modulo the prime, which is
`POSEIDON2-NO-CANCELLATION` and depends on the selected round constants rather
than on the round structure. -/
theorem normalizedCanonicalProgram_termCount_bound
    (layout : Layout) (wellFormed : WellFormed layout) (constants : Constants) :
    Poseidon2Normalized.rawProgramTermCount
        (Poseidon2Normalized.normalizedCanonicalProgram layout constants) ≤ 4397 := by
  have bound := Poseidon2Normalized.rawProgramTermCount_normalizeProgram_le
    (canonicalProgram layout constants)
  rw [canonicalProgram_termCount_eq layout wellFormed constants] at bound
  exact bound


/-! ## No cancellation in the full-round states

The same argument as `finalState_fieldNormalize_length`, instantiated at the 56
full-round S-box states.  Coefficients are external-matrix entries and the
matrix is dense, so nothing is dropped and no round constant is consulted. -/

theorem initialState_fieldNormalize_length
    (layout : Layout) (wellFormed : WellFormed layout)
    (round : Nat) (lane : Fin width) :
    (fieldNormalize (initialState layout round lane)).length = width := by
  cases round with
  | zero =>
      exact fieldNormalize_length_applyMatrix_singletons _ _
        (fun a b image => wellFormed.inputInjective a b image)
        (fun a b => externalMatrix_nonzero a b)
        (fun a b => externalMatrix_lt a b) lane
  | succ previous =>
      refine fieldNormalize_length_applyMatrix_singletons _ _ (fun a b image => ?_)
        (fun a b => externalMatrix_nonzero a b)
        (fun a b => externalMatrix_lt a b) lane
      have := sboxOutput_injective layout _ _ image
      simp only [initialSboxIndex] at this
      exact Fin.ext (by omega)

theorem terminalState_succ_fieldNormalize_length
    (layout : Layout) (round : Nat) (lane : Fin width) :
    (fieldNormalize (terminalState layout (round + 1) lane)).length = width := by
  refine fieldNormalize_length_applyMatrix_singletons _ _ (fun a b image => ?_)
    (fun a b => externalMatrix_nonzero a b)
    (fun a b => externalMatrix_lt a b) lane
  have := sboxOutput_injective layout _ _ image
  simp only [terminalSboxIndex] at this
  exact Fin.ext (by omega)

end Nightstream.Implementation.R1CS.Canonical.Poseidon2ScheduledSizes
