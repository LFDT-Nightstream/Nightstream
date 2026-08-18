import Batteries.Data.List.Basic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeaf
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafCertificate

/-!
Contract: bounded structural decoding certificate for one generated chained
production PiRLC Poseidon2 leaf.

Assurance tier: artifact-checked leaf certificate.

Owns exact decoding of all 86 chained rows in two 43-entry leaves, exact row
count, and index-preserving pairing with the shared typed S-box steps.

Does not own source-slot reconstruction, row satisfaction, replay-batch
coverage, absolute Rust placement, or lifecycle semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeaf
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

private abbrev sharedSteps :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps

def rawRowHead : List RawRow := rawRows.take 43
def rawRowTail : List RawRow := rawRows.drop 43

def decodedRowHead : List Wire.Row :=
  (rawRowHead.mapM Wire.decodeRow).getD []

def decodedRowTail : List Wire.Row :=
  (rawRowTail.mapM Wire.decodeRow).getD []

def decodedRows : List Wire.Row := decodedRowHead ++ decodedRowTail

theorem raw_rows_partition : rawRows = rawRowHead ++ rawRowTail := by
  simp [rawRowHead, rawRowTail]

/-- Each reduction is bounded by one generated 43-entry row shard. -/
theorem raw_row_head_decodes :
    rawRowHead.mapM Wire.decodeRow = some decodedRowHead := by
  rfl

theorem raw_row_tail_decodes :
    rawRowTail.mapM Wire.decodeRow = some decodedRowTail := by
  rfl

theorem raw_rows_decode :
    rawRows.mapM Wire.decodeRow = some decodedRows := by
  rw [raw_rows_partition, List.mapM_append, raw_row_head_decodes,
    raw_row_tail_decodes]
  rfl

theorem decoded_rows_length : decodedRows.length = 86 := by
  rfl

private theorem head_offsets_exact :
    (sharedSteps.take 43).map (fun step => step.rowOffset) =
      decodedRowHead.map (fun row => row.rowOffset) := by
  rfl

private theorem tail_offsets_exact :
    (sharedSteps.drop 43).map (fun step => step.rowOffset) =
      decodedRowTail.map (fun row => row.rowOffset) := by
  rfl

private theorem shared_steps_partition :
    sharedSteps = sharedSteps.take 43 ++ sharedSteps.drop 43 := by
  simp

theorem decoded_offsets_exact :
    sharedSteps.map (fun step => step.rowOffset) =
      decodedRows.map (fun row => row.rowOffset) := by
  rw [shared_steps_partition]
  simp only [decodedRows, List.map_append]
  rw [head_offsets_exact, tail_offsets_exact]

private theorem paired_of_offsets :
    ∀ (steps : List Wire.Step) (rows : List Wire.Row),
      steps.map (fun step => step.rowOffset) =
          rows.map (fun row => row.rowOffset) →
        List.Forall₂ (fun step row => step.rowOffset = row.rowOffset)
          steps rows
  | [], [], _ => .nil
  | [], _ :: _, offsets => by simp at offsets
  | _ :: _, [], offsets => by simp at offsets
  | step :: steps, row :: rows, offsets => by
      simp only [List.map_cons, List.cons.injEq] at offsets
      exact .cons offsets.1 (paired_of_offsets steps rows offsets.2)

/-- Every chained row is paired with the shared source step at the same list
index and row offset. -/
theorem decoded_rows_pair_with_shared_steps :
    List.Forall₂ (fun step row => step.rowOffset = row.rowOffset)
      sharedSteps decodedRows :=
  paired_of_offsets sharedSteps decodedRows decoded_offsets_exact

theorem decoded_row_head_shapes_checked :
    decodedRowHead.all sboxShapeCheck = true := by
  rfl

theorem decoded_row_tail_shapes_checked :
    decodedRowTail.all sboxShapeCheck = true := by
  rfl

private theorem shapes_of_all_checked :
    ∀ (rows : List Wire.Row), rows.all sboxShapeCheck = true →
      ∀ row ∈ rows, IsSboxShape row
  | [], _ => by simp
  | head :: tail, checked => by
      simp only [List.all_cons, Bool.and_eq_true] at checked
      intro row member
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact sboxShapeCheck_sound _ checked.1
      · exact shapes_of_all_checked tail checked.2 row tailMember

theorem decoded_rows_have_sbox_shape :
    ∀ row ∈ decodedRows, IsSboxShape row := by
  intro row member
  rw [decodedRows, List.mem_append] at member
  rcases member with headMember | tailMember
  · exact shapes_of_all_checked decodedRowHead
      decoded_row_head_shapes_checked row headMember
  · exact shapes_of_all_checked decodedRowTail
      decoded_row_tail_shapes_checked row tailMember

/-- On one same assignment, all active chained rows and exact source-port
realization imply the 86 shared typed S-box equations. -/
theorem decoded_rows_imply_shared_step_sboxes
    (source : SourceAssignment) (final : FinalAssignment)
    (selectorOne : final.explicit .selector = 1)
    (realized : List.Forall₂ (StepRealized source final)
      sharedSteps decodedRows)
    (holds : ∀ row ∈ decodedRows, residual row final = 0) :
    ∀ step ∈ sharedSteps, StepSboxHolds source step :=
  realized_rows_imply_step_sboxes source final selectorOne realized
    decoded_rows_have_sbox_shape holds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafCertificate
