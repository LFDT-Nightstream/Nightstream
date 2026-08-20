import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash

/-!
Contract: structural validity of the exact recursive-terminal XOut public-hash
artifact.

Owns the compact artifact geometry and the self-owned nine-round Poseidon2
trace certificate. It reuses the isolated permutation row-length certificate
and does not evaluate any renamed 600-row call.

Does not own public-word row soundness, final selective-row transport, or
collision resistance.

Assurance tier: artifact-checked for the Nightstream b2/k16 terminal profile.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash

export Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
  (callPlacements outputCopies outputImages publicWords rawArtifact rounds trace
    xOutImages)

theorem firstLeafPlacement_valid :
    rawArtifact.firstLeafPlacement.Valid := by
  change
    ({ rewriteId := 9585
       sourceRows := { start := 30658245, stop := 30658845 }
       finalRows := { start := 5093743, stop := 5093829 }
       finalColumns := 28033344
       selectorColumn := 649
       externalSlotStarts := [766, 807, 848]
       localSlotStart := 22023158
       slotWidth := 41
       localSlotCount := 86 } : FirstLeafPlacement).Valid
  unfold FirstLeafPlacement.Valid
  refine ⟨?_, ?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_, ?_⟩
  · norm_num [Range.Valid]
  · norm_num [Range.Valid]
  · norm_num
  · norm_num
  · norm_num
  · intro start member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl
    all_goals norm_num
  · norm_num

theorem callPlacement0_valid : callPlacement0.Valid := by
  norm_num [callPlacement0, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement1_valid : callPlacement1.Valid := by
  norm_num [callPlacement1, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement2_valid : callPlacement2.Valid := by
  norm_num [callPlacement2, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement3_valid : callPlacement3.Valid := by
  norm_num [callPlacement3, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement4_valid : callPlacement4.Valid := by
  norm_num [callPlacement4, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement5_valid : callPlacement5.Valid := by
  norm_num [callPlacement5, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement6_valid : callPlacement6.Valid := by
  norm_num [callPlacement6, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement7_valid : callPlacement7.Valid := by
  norm_num [callPlacement7, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacement8_valid : callPlacement8.Valid := by
  norm_num [callPlacement8, PoseidonCallPlacement.Valid, Range.Valid]

theorem callPlacements_valid :
    ∀ placement ∈ callPlacements, placement.Valid := by
  intro placement member
  simp only [callPlacements, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact callPlacement0_valid
  · exact callPlacement1_valid
  · exact callPlacement2_valid
  · exact callPlacement3_valid
  · exact callPlacement4_valid
  · exact callPlacement5_valid
  · exact callPlacement6_valid
  · exact callPlacement7_valid
  · exact callPlacement8_valid

private theorem generated_outputCopies_paired :
    List.Forall₂
      (fun lane placement =>
        placement.lane = lane ∧
          placement.outputSourceColumn = trace.outputColumns.getD lane 0 ∧
          placement.Valid)
      [0, 1, 2, 3] outputCopies := by
  norm_num [outputCopies, trace, OutputCopyPlacement.Valid, Range.Valid]

theorem rawArtifact_valid : rawArtifact.Valid := by
  unfold RawArtifact.Valid
  refine
    ⟨rfl, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, firstLeafPlacement_valid,
      rfl, callPlacements_valid, generated_outputCopies_paired, rfl, rfl, rfl,
      rfl, rfl, rfl, ?_⟩
  · norm_num [Range.Valid, rawArtifact]
  · intro range member
    simp only [rawArtifact, List.mem_singleton] at member
    subst range
    norm_num [Range.Valid]
  · intro range member
    simp only [rawArtifact, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    all_goals norm_num [Range.Valid]
  · decide

theorem outputCopies_paired :
    List.Forall₂
      (fun lane placement =>
        placement.lane = lane ∧
          placement.outputSourceColumn = trace.outputColumns.getD lane 0 ∧
          placement.Valid)
      [0, 1, 2, 3] outputCopies := by
  exact generated_outputCopies_paired

theorem publicWords_paired :
    List.Forall₂
      (fun lane word =>
        word.fieldColumn = trace.outputColumns.getD lane 0 ∧ word.Valid)
      (List.range 4) publicWords := by
  have valid := rawArtifact_valid
  unfold RawArtifact.Valid at valid
  rcases valid with
    ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, paired⟩
  exact paired

theorem xOutImages_sourceColumns :
    rawArtifact.xOutImages.map SourceImage.sourceColumn =
      trace.inputColumns := by
  rfl

theorem outputImages_sourceColumns :
    rawArtifact.outputImages.map SourceImage.sourceColumn =
      trace.outputColumns := by
  rfl

private theorem call_rows_length (call : Call) :
    call.rows.length = Poseidon2Permutation.rowCount := by
  rw [Call.rows, List.length_map, Poseidon2Permutation.rows_length]

private theorem expectedDefinitionRows_length (round : Round) :
    round.expectedDefinitionRows.length =
      match round.kind with
      | .absorb chunkColumns => chunkColumns.length
      | .pad => 1 := by
  cases kind : round.kind <;>
    simp [Round.expectedDefinitionRows, kind]

private theorem trace_roundsAccepted :
    rounds.all (fun round => decide (round.Valid round.rows)) = true := by
  apply List.all_eq_true.mpr
  intro round member
  simp only [rounds, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with h | h | h | h | h | h | h | h | h
  all_goals subst round
  all_goals apply decide_eq_true
  all_goals
    apply Round.selfValid (by
      unfold Round.metadataValid
      decide)
    · norm_num [expectedDefinitionRows_length]
    · norm_num [call_rows_length, expectedDefinitionRows_length,
        Poseidon2Permutation.rowCount]

theorem trace_owned_valid : trace.OwnedValid := by
  refine {
    roundsAccepted := trace_roundsAccepted
    linked := by decide
    inputsOwned := by decide
    finalOutput := by decide
    outputLength := rfl
    terminalPad := rfl }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
