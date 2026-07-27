import Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23

/-!
Contract: honest completeness for the exact fixed-23 canonical sponge core.

Owns: one explicit global assignment over the seven disjoint permutation
spaces and authoritative input block, plus the proof that it satisfies all
2,464 normalized rows.

Does not own: activation/output-copy rows or typed call serialization.
-/

set_option autoImplicit false
set_option maxRecDepth 4096

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
open Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23

def entryValues (constants : Constants) (input : Preimage) (call : Nat) :
    Values :=
  chainValues constants (chunkAt input) call

/-- Values carried inside one 361-column permutation space. -/
def blockValue
    (constants : Constants) (input : Preimage) (call offset : Nat) : Nat :=
  if belowOutputs : offset < 9 then 0
  else if output : offset < 17 then
    referencePermutation constants (entryValues constants input call)
      ⟨offset - 9, by
        simp only [width]
        omega⟩
  else
    chainSlot
      (sboxInputValue constants (entryValues constants input call)
        ((offset - 17) / columnsPerSbox))
      ((offset - 17) % columnsPerSbox)

/-- Global reference assignment.  Columns below `inputBase` are seven
consecutive 361-column permutation spaces.  The 23 authoritative inputs follow
those spaces.  Unowned input-port/gap columns carry zero. -/
def assignment (constants : Constants) (input : Preimage) : Nat → Nat :=
  fun column =>
    if column = 0 then 1
    else if beforeInputs : column < inputBase then
      blockValue constants input
        (column / callStride) (column % callStride)
    else if isInput : column < inputBase + sponge23Fields then
      input ⟨column - inputBase, by omega⟩
    else 0

theorem assignment_constantWire (constants : Constants) (input : Preimage) :
    assignment constants input 0 = 1 := rfl

private theorem call_column_before_inputs
    (call : Nat) (callLt : call < calls) (offset : Nat)
    (offsetLt : offset < callStride) :
    call * callStride + offset < inputBase := by
  rw [callStride_eq] at offsetLt ⊢
  rw [inputBase_eq]
  simp only [calls] at callLt
  omega

private theorem call_column_div
    (call offset : Nat) (offsetLt : offset < callStride) :
    (call * callStride + offset) / callStride = call := by
  have positive : 0 < callStride := by decide
  rw [Nat.mul_comm call callStride, Nat.mul_add_div positive,
    Nat.div_eq_of_lt offsetLt, Nat.add_zero]

private theorem call_column_mod
    (call offset : Nat) (offsetLt : offset < callStride) :
    (call * callStride + offset) % callStride = offset := by
  exact Nat.mul_add_mod_of_lt offsetLt

theorem assignment_input
    (constants : Constants) (input : Preimage)
    (index : Fin sponge23Fields) :
    assignment constants input (inputColumn index.val) = input index := by
  have indexLt := index.isLt
  simp only [sponge23Fields] at indexLt
  change assignment constants input (2527 + index.val) = _
  unfold assignment
  rw [if_neg (by omega), dif_neg (by rw [inputBase_eq]; omega),
    dif_pos (by
      rw [inputBase_eq]
      simp only [sponge23Fields]
      omega)]
  apply congrArg input
  apply Fin.ext
  change 2527 + index.val - inputBase = index.val
  have base := inputBase_eq
  omega

theorem assignment_output
    (constants : Constants) (input : Preimage)
    (call : Nat) (callLt : call < calls) (lane : Fin width) :
    assignment constants input ((layout.call call).outputPort lane)
      = referencePermutation constants (entryValues constants input call) lane := by
  have laneLt := lane.isLt
  simp only [width] at laneLt
  change assignment constants input
    (call * callStride + 9 + lane.val) = _
  rw [show call * callStride + 9 + lane.val
    = call * callStride + (9 + lane.val) by omega]
  have localLt : 9 + lane.val < callStride := by
    rw [callStride_eq]
    omega
  unfold assignment
  rw [if_neg (by
      rw [callStride_eq]
      omega),
    dif_pos (call_column_before_inputs call callLt _ localLt),
    call_column_div call _ localLt,
    call_column_mod call _ localLt,
    blockValue, dif_neg (by omega),
    dif_pos (by omega)]
  apply congrArg (referencePermutation constants
    (entryValues constants input call))
  apply Fin.ext
  change 9 + lane.val - 9 = lane.val
  omega

theorem assignment_sbox
    (constants : Constants) (input : Preimage)
    (call : Nat) (callLt : call < calls)
    (index : Fin sboxCount) (slot : Fin columnsPerSbox) :
    assignment constants input
        (sboxColumn (layout.call call) index slot)
      = chainSlot
          (sboxInputValue constants (entryValues constants input call) index.val)
          slot.val := by
  have indexLt := index.isLt
  have slotLt := slot.isLt
  simp only [sboxCount, externalRounds, width, partialRounds] at indexLt
  simp only [columnsPerSbox] at slotLt
  change assignment constants input
    ((call * callStride + 17) + 4 * index.val + slot.val) = _
  rw [show (call * callStride + 17) + 4 * index.val + slot.val
    = call * callStride + (17 + 4 * index.val + slot.val) by omega]
  have localLt : 17 + 4 * index.val + slot.val < callStride := by
    rw [callStride_eq]
    omega
  unfold assignment
  rw [if_neg (by
      rw [callStride_eq]
      omega),
    dif_pos (call_column_before_inputs call callLt _ localLt),
    call_column_div call _ localLt,
    call_column_mod call _ localLt,
    blockValue, dif_neg (by omega),
    dif_neg (by omega)]
  have divEq :
      (17 + 4 * index.val + slot.val - 17) / columnsPerSbox
        = index.val := by
    simp only [columnsPerSbox]
    omega
  have modEq :
      (17 + 4 * index.val + slot.val - 17) % columnsPerSbox
        = slot.val := by
    simp only [columnsPerSbox]
    omega
  rw [divEq, modEq]

theorem assignment_inputsAgree
    (constants : Constants) (input : Preimage) :
    InputsAgree (assignment constants input) input :=
  assignment_input constants input

theorem assignment_residues
    (constants : Constants) (input : Preimage)
    (inputResidues : ∀ index, input index < goldilocksP) :
    ∀ column, assignment constants input column < goldilocksP := by
  have blockResidues : ∀ call offset,
      blockValue constants input call offset < goldilocksP := by
    intro call offset
    unfold blockValue
    split
    · decide
    · split
      · exact refTerminal_lt _ _ _ _
      · exact chainSlot_lt _ _
  intro column
  unfold assignment
  split
  · decide
  · split
    · exact blockResidues _ _
    · split
      · exact inputResidues _
      · decide

theorem entry_agrees
    (constants : Constants) (input : Preimage)
    (inputResidues : ∀ index, input index < goldilocksP)
    (call : Nat) (callLt : call < calls) (lane : Fin width) :
    lcEval (assignment constants input)
        (entryOf layout chunkLength call lane)
      = entryValues constants input call lane := by
  cases call with
  | zero =>
      exact entryOf_zero_eval_is_absorbChunk layout chunkLength
        (assignment constants input) (chunkAt input)
        (chunkAt_length input 0).symm
        (chunkAgrees (assignment constants input) input
          (assignment_constantWire constants input)
          (assignment_inputsAgree constants input) 0)
        lane
  | succ previous =>
      apply entryOf_eval_is_absorbChunk layout chunkLength
        (assignment constants input) previous (chunkAt input (previous + 1))
        (referencePermutation constants
          (entryValues constants input previous))
        (assignment_residues constants input inputResidues)
        (chunkAt_length input (previous + 1)).symm
        (assignment_output constants input previous (by omega))
        (chunkAgrees (assignment constants input) input
          (assignment_constantWire constants input)
          (assignment_inputsAgree constants input) (previous + 1))
        lane

/-- Every one of the seven normalized carried-entry programs is satisfied by
the corresponding slice of the global reference assignment. -/
theorem call_satisfies
    (constants : Constants) (input : Preimage)
    (inputResidues : ∀ index, input index < goldilocksP)
    (call : Nat) (callLt : call < calls) :
    Satisfies
      (normalizedCanonicalProgramFrom (layout.call call)
        (entryOf layout chunkLength call) constants)
      (assignment constants input) :=
  honest_satisfies_normalizedFrom
    (layout.call call)
    (entryOf layout chunkLength call)
    constants
    (entryValues constants input call)
    (assignment constants input)
    (assignment_residues constants input inputResidues)
    (assignment_constantWire constants input)
    (entry_agrees constants input inputResidues call callLt)
    (assignment_sbox constants input call callLt)
    (assignment_output constants input call callLt)

/-- **Fixed-23 honest completeness.** -/
theorem honest_satisfies
    (constants : Constants) (input : Preimage)
    (inputResidues : ∀ index, input index < goldilocksP) :
    Satisfies (program constants) (assignment constants input) := by
  intro row member
  unfold program spongeProgram at member
  rcases List.mem_flatMap.1 member with ⟨call, callMember, rowMember⟩
  exact call_satisfies constants input inputResidues call
    (by simpa [calls] using List.mem_range.1 callMember)
    row rowMember

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23Honest
