import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafFinalSlice

/-!
Contract: structural row decoding for one Rust-emitted PiRLC Poseidon2 run.

Assurance tier: artifact-checked row-position transport for the Nightstream
b2/k16 profile.

Owns: conversion of an emitted final-row position to its call index and local
86-row offset, and recovery of the exact compact port combination at that
position.

Does not own: a shared finite relation, row satisfaction, phase semantics,
lifecycle semantics, or permission to remove rows.

Emits constraints: no.
-/

set_option autoImplicit false
set_option compiler.extract_closed false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunFinalSlice

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafFinalSlice

/-- The exact contiguous final-row interval owned by one emitted call run. -/
def RowOwned (run : Run) (row : Nat) : Prop :=
  run.raw.emittedRowStart <= row /\
    row < run.raw.emittedRowStart + run.raw.callCount * 86

local instance rowOwnedDecidable (run : Run) (row : Nat) :
    Decidable (RowOwned run row) := by
  unfold RowOwned
  infer_instance

/-- Total row decoder for one run. Rows outside the run, or an impossible
local offset, return `none`. The column count stays generic so compilation
cannot construct the production column domain. -/
noncomputable def rowCombinationAt {columns : Nat}
    (run : Run) (role : Role) (row : Nat) : Option (Combination columns) :=
  if owned : RowOwned run row then
    let relative := row - run.raw.emittedRowStart
    let index := relative / 86
    let offset := relative % 86
    match (rowsFor (run.leafClassAt index))[offset]? with
    | some decodedRow =>
        some (portCombination (run.callSiteAt index)
          (decodedRow.port role.index))
    | none => none
  else
    none

theorem emitted_block_row_owned
    (run : Run) (index : Fin run.raw.callCount)
    (offset : Fin (run.emittedBlockAt index).rows.length) :
    RowOwned run
      ((run.emittedBlockAt index).finalRowStart + offset.val) := by
  have offsetLt : offset.val < 86 := by
    have bounded := offset.isLt
    simpa [Run.emittedBlockAt, rowsFor_length] using bounded
  change
    run.raw.emittedRowStart <=
        run.raw.emittedRowStart + index.val *
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows +
          offset.val /\
      run.raw.emittedRowStart + index.val *
          Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows +
          offset.val <
        run.raw.emittedRowStart + run.raw.callCount * 86
  rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows_exact]
  have indexLt := index.isLt
  omega

/-- Every indexed block row decodes to the exact compact combination used by
that Rust-emitted block. This is structural in the run and does not inspect a
closed call schedule or row array. -/
theorem rowCombinationAt_emitted_block
    {columns : Nat} (run : Run) (index : Fin run.raw.callCount)
    (offset : Fin (run.emittedBlockAt index).rows.length) (role : Role) :
    rowCombinationAt (columns := columns) run role
        ((run.emittedBlockAt index).finalRowStart + offset.val) =
      some (portCombination (run.emittedBlockAt index).site
        (((run.emittedBlockAt index).rows.get offset).port role.index)) := by
  have offsetLt : offset.val < 86 := by
    have bounded := offset.isLt
    simpa [Run.emittedBlockAt, rowsFor_length] using bounded
  have relativeExact :
      (run.emittedBlockAt index).finalRowStart + offset.val -
          run.raw.emittedRowStart =
        index.val * 86 + offset.val := by
    change
      run.raw.emittedRowStart + index.val *
            Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows +
          offset.val - run.raw.emittedRowStart =
        index.val * 86 + offset.val
    rw [Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallLayout.emittedCallRows_exact]
    omega
  have indexExact :
      ((run.emittedBlockAt index).finalRowStart + offset.val -
          run.raw.emittedRowStart) / 86 = index.val := by
    rw [relativeExact]
    omega
  have offsetExact :
      ((run.emittedBlockAt index).finalRowStart + offset.val -
          run.raw.emittedRowStart) % 86 = offset.val := by
    rw [relativeExact]
    omega
  have rowBound :
      offset.val < (rowsFor (run.leafClassAt index.val)).length := by
    simpa [Run.emittedBlockAt] using offset.isLt
  unfold rowCombinationAt
  rw [dif_pos (emitted_block_row_owned run index offset)]
  simp only [indexExact, offsetExact]
  rw [List.getElem?_eq_getElem rowBound]
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonRunFinalSlice
