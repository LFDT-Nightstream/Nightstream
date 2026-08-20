import Batteries.Data.List.Basic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeaf
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafCertificate

/-!
Contract: bounded structural certificate for the first recursive-terminal
XOut Poseidon2 leaf.

Assurance tier: artifact-checked leaf certificate.

Owns: exact decoding of 86 source steps and 86 final rows in two 43-entry
leaves, exact order, row-step pairing, and S-box row shape.

Does not own: source-definition reconstruction, row satisfaction, complete
hash replay, lifecycle semantics, or permission to remove constraints.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeaf
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel

def rawStepHead : List RawStep := rawSteps.take 43
def rawStepTail : List RawStep := rawSteps.drop 43

def decodedStepHead : List Wire.Step :=
  (rawStepHead.mapM Wire.decodeStep).getD []

def decodedStepTail : List Wire.Step :=
  (rawStepTail.mapM Wire.decodeStep).getD []

def decodedRowHead : List Wire.Row :=
  (rawRowHead.mapM Wire.decodeRow).getD []

def decodedRowTail : List Wire.Row :=
  (rawRowTail.mapM Wire.decodeRow).getD []

def decodedSteps : List Wire.Step := decodedStepHead ++ decodedStepTail
def decodedRows : List Wire.Row := decodedRowHead ++ decodedRowTail

theorem raw_steps_partition : rawSteps = rawStepHead ++ rawStepTail := by
  simp [rawStepHead, rawStepTail]

theorem raw_rows_partition : rawRows = rawRowHead ++ rawRowTail := by
  rfl

theorem raw_step_head_decodes :
    rawStepHead.mapM Wire.decodeStep = some decodedStepHead := by
  rfl

theorem raw_step_tail_decodes :
    rawStepTail.mapM Wire.decodeStep = some decodedStepTail := by
  rfl

theorem raw_row_head_decodes :
    rawRowHead.mapM Wire.decodeRow = some decodedRowHead := by
  rfl

theorem raw_row_tail_decodes :
    rawRowTail.mapM Wire.decodeRow = some decodedRowTail := by
  rfl

theorem raw_steps_decode :
    rawSteps.mapM Wire.decodeStep = some decodedSteps := by
  rw [raw_steps_partition, List.mapM_append, raw_step_head_decodes,
    raw_step_tail_decodes]
  rfl

theorem raw_rows_decode :
    rawRows.mapM Wire.decodeRow = some decodedRows := by
  rw [raw_rows_partition, List.mapM_append, raw_row_head_decodes,
    raw_row_tail_decodes]
  rfl

theorem decoded_steps_length : decodedSteps.length = 86 := by
  rfl

theorem decoded_rows_length : decodedRows.length = 86 := by
  rfl

private theorem head_offsets_exact :
    decodedStepHead.map (fun step => step.rowOffset) =
      decodedRowHead.map (fun row => row.rowOffset) := by
  rfl

private theorem tail_offsets_exact :
    decodedStepTail.map (fun step => step.rowOffset) =
      decodedRowTail.map (fun row => row.rowOffset) := by
  rfl

theorem decoded_offsets_exact :
    decodedSteps.map (fun step => step.rowOffset) =
      decodedRows.map (fun row => row.rowOffset) := by
  simp only [decodedSteps, decodedRows, List.map_append]
  rw [head_offsets_exact, tail_offsets_exact]

theorem decoded_rows_pair_with_steps :
    List.Forall₂ (fun step row => step.rowOffset = row.rowOffset)
      decodedSteps decodedRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.paired_of_offsets
    decodedSteps decodedRows decoded_offsets_exact

theorem decoded_row_head_shapes_checked :
    decodedRowHead.all sboxShapeCheck = true := by
  rfl

theorem decoded_row_tail_shapes_checked :
    decodedRowTail.all sboxShapeCheck = true := by
  rfl

theorem decoded_rows_have_sbox_shape :
    ∀ row ∈ decodedRows, IsSboxShape row := by
  intro row member
  rw [decodedRows, List.mem_append] at member
  rcases member with headMember | tailMember
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.shapes_of_all_checked
        decodedRowHead
      decoded_row_head_shapes_checked row headMember
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.shapes_of_all_checked
        decodedRowTail
      decoded_row_tail_shapes_checked row tailMember

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafCertificate
