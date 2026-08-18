import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPriorStateReplaySource

/-!
Facade for the compact exact Rust prior-state replay source artifacts.

Owns the handwritten import boundary for the full and final arms. Generated
data stays opaque until a narrow leaf certificate exposes an exact slice.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact

def fullArtifact : RawArm :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplaySource.fullArtifact

def finalArtifact : RawArm :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplaySource.finalArtifact

def fullCallsPart0 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart0

def fullCallsPart1 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart1

def fullCallsPart2 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart2

def fullCallsPart3 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart3

def fullCallsPart4 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart4

def finalCallsPart0 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.callsPart0

def finalCallsPart1 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.callsPart1

def finalCallsPart2 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.callsPart2

def fullResidualRows0Part0 : List IndexedRow :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFullResidualRows0.rowsPart0

def finalResidualRows0Part0 : List IndexedRow :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rowsPart0

def finalResidualRows0Part1 : List IndexedRow :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rowsPart1

def finalResidualRows0Part2 : List IndexedRow :=
  Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rowsPart2

theorem fullCallsPart0_subset :
    ∀ call ∈ fullCallsPart0, call ∈ fullArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart0
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  simp only [List.mem_append, member, true_or]

theorem fullCallsPart1_subset :
    ∀ call ∈ fullCallsPart1, call ∈ fullArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart1
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  simp only [List.mem_append, member, true_or, or_true]

theorem fullCallsPart2_subset :
    ∀ call ∈ fullCallsPart2, call ∈ fullArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart2
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  simp only [List.mem_append, member, true_or, or_true]

theorem fullCallsPart3_subset :
    ∀ call ∈ fullCallsPart3, call ∈ fullArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart3
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  simp only [List.mem_append, member, true_or, or_true]

theorem fullCallsPart4_subset :
    ∀ call ∈ fullCallsPart4, call ∈ fullArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.callsPart4
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFullPoseidonCalls.calls
  simp only [List.mem_append, member, true_or, or_true]

theorem finalCallsPart0_subset :
    ∀ call ∈ finalCallsPart0, call ∈ finalArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.callsPart0
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.calls
  simp only [List.mem_append, member, true_or]

theorem finalCallsPart1_subset :
    ∀ call ∈ finalCallsPart1, call ∈ finalArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.callsPart1
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.calls
  simp only [List.mem_append, member, true_or, or_true]

theorem finalCallsPart2_subset :
    ∀ call ∈ finalCallsPart2, call ∈ finalArtifact.poseidon2Calls := by
  intro call member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.callsPart2
    at member
  change call ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalPoseidonCalls.calls
  simp only [List.mem_append, member, true_or, or_true]

private theorem fullArtifact_residualRows :
    fullArtifact.residualRows =
      Generated.FPrimeFullHistoryStreamingPriorStateReplaySource.fullResidualRows := by
  rfl

private theorem fullResidualRows0_subset :
    ∀ indexed ∈
      Generated.FPrimeFullHistoryStreamingPriorStateReplayFullResidualRows0.rows,
      indexed ∈ fullArtifact.residualRows := by
  intro indexed member
  rw [fullArtifact_residualRows]
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplaySource.fullResidualRows
  simp only [List.mem_append, member, true_or]

theorem fullResidualRows0Part0_subset :
    ∀ indexed ∈ fullResidualRows0Part0,
      indexed ∈ fullArtifact.residualRows := by
  intro indexed member
  apply fullResidualRows0_subset indexed
  change indexed ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFullResidualRows0.rowsPart0
    at member
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFullResidualRows0.rows
  simp only [List.mem_append, member, true_or]

private theorem finalArtifact_residualRows :
    finalArtifact.residualRows =
      Generated.FPrimeFullHistoryStreamingPriorStateReplaySource.finalResidualRows := by
  rfl

private theorem finalResidualRows0_subset :
    ∀ indexed ∈
      Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rows,
      indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  rw [finalArtifact_residualRows]
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplaySource.finalResidualRows
  simp only [List.mem_append, member, true_or]

theorem finalResidualRows0Part0_subset :
    ∀ indexed ∈ finalResidualRows0Part0,
      indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  apply finalResidualRows0_subset indexed
  change indexed ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rowsPart0
    at member
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rows
  simp only [List.mem_append, member, true_or]

theorem finalResidualRows0Part1_subset :
    ∀ indexed ∈ finalResidualRows0Part1,
      indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  apply finalResidualRows0_subset indexed
  change indexed ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rowsPart1
    at member
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rows
  simp only [List.mem_append, member, true_or, or_true]

theorem finalResidualRows0Part2_subset :
    ∀ indexed ∈ finalResidualRows0Part2,
      indexed ∈ finalArtifact.residualRows := by
  intro indexed member
  apply finalResidualRows0_subset indexed
  change indexed ∈
    Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rowsPart2
    at member
  unfold Generated.FPrimeFullHistoryStreamingPriorStateReplayFinalResidualRows0.rows
  simp only [List.mem_append, member, true_or, or_true]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
