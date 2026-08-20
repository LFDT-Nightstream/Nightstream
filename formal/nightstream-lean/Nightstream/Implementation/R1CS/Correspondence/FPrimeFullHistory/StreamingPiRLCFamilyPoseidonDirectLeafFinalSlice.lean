import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafFinalSlice

/-!
Contract: exact Rust-generated first direct PiRLC Poseidon2 matrix slice.

Assurance tier: artifact-checked matrix-action leaf certificate.

Owns: selection of the first production even-input call, its generated
86-row direct leaf, final row start 74375, and its `FinalRowSliceExact` value.

Does not own: the other 485 call blocks, complete relation satisfaction,
lifecycle semantics, or permission to remove rows.

Emits constraints: no new rows. It selects one existing emitted block.
-/

set_option autoImplicit false
set_option compiler.extract_closed false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonDirectLeafFinalSlice

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonFinalRowBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafFinalSlice

abbrev firstDirectIndex : Fin evenInputRun.raw.callCount :=
  ⟨0, evenInputRun_valid.callCountPositive⟩

abbrev firstDirectBlock : EmittedBlock :=
  evenInputRun.emittedBlockAt firstDirectIndex

theorem firstDirectBlock_rows :
    firstDirectBlock.rows =
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedRows := by
  rfl

theorem firstDirectBlock_finalRowStart :
    firstDirectBlock.finalRowStart = 74375 := by
  rfl

theorem firstDirectBlock_rows_length :
    firstDirectBlock.rows.length = 86 := by
  rw [firstDirectBlock_rows]
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decoded_rows_length

/-- Exact generated-row matrix action for the first production direct block,
on every assignment. No digest or evaluated complete artifact is used. -/
theorem firstDirectBlock_exact
    (assignment : Fin productionFinalColumns -> F) :
    FinalRowSliceExact firstDirectBlock
      (blockRelation firstDirectBlock) assignment :=
  blockRelation_exact firstDirectBlock assignment

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonDirectLeafFinalSlice
