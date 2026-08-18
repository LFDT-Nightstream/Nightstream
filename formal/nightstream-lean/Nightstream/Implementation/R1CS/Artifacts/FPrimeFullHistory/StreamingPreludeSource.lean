import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPreludeSource

/-!
Facade for the compact exact Rust Prelude source artifact.

Owns the handwritten import boundary only. Generated data remains opaque to
later correspondence proofs unless a leaf certificate exposes it.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Artifact

def artifact : RawArtifact :=
  Generated.FPrimeFullHistoryStreamingPreludeSource.artifact

def poseidonCallsPart0 : List Poseidon2Call.Call :=
  Generated.FPrimeFullHistoryStreamingPreludePoseidonCalls.callsPart0

def residualRows0Part0 : List IndexedRow :=
  Generated.FPrimeFullHistoryStreamingPreludeResidualRows0.rowsPart0

def residualRows1Part3 : List IndexedRow :=
  Generated.FPrimeFullHistoryStreamingPreludeResidualRows1.rowsPart3

theorem poseidonCallsPart0_subset :
    ∀ call ∈ poseidonCallsPart0, call ∈ artifact.poseidon2Calls := by
  intro call member
  change call ∈ Generated.FPrimeFullHistoryStreamingPreludePoseidonCalls.calls
  unfold Generated.FPrimeFullHistoryStreamingPreludePoseidonCalls.calls
  simp only [List.append_assoc]
  exact List.mem_append_left _ member

theorem residualRows0Part0_subset :
    ∀ indexed ∈ residualRows0Part0, indexed ∈ artifact.residualRows := by
  intro indexed member
  have inShard :
      indexed ∈ Generated.FPrimeFullHistoryStreamingPreludeResidualRows0.rows := by
    unfold Generated.FPrimeFullHistoryStreamingPreludeResidualRows0.rows
    simp only [List.append_assoc]
    exact List.mem_append_left _ member
  change indexed ∈ Generated.FPrimeFullHistoryStreamingPreludeSource.residualRows
  unfold Generated.FPrimeFullHistoryStreamingPreludeSource.residualRows
  simp only [List.append_assoc]
  exact List.mem_append_left _ inShard

theorem residualRows1Part3_subset :
    ∀ indexed ∈ residualRows1Part3, indexed ∈ artifact.residualRows := by
  intro indexed member
  have inShard :
      indexed ∈ Generated.FPrimeFullHistoryStreamingPreludeResidualRows1.rows := by
    unfold Generated.FPrimeFullHistoryStreamingPreludeResidualRows1.rows
    simp only [List.append_assoc]
    apply List.mem_append_right
    apply List.mem_append_right
    apply List.mem_append_right
    exact List.mem_append_left _ member
  change indexed ∈ Generated.FPrimeFullHistoryStreamingPreludeSource.residualRows
  unfold Generated.FPrimeFullHistoryStreamingPreludeSource.residualRows
  simp only [List.append_assoc]
  apply List.mem_append_right
  exact List.mem_append_left _ inShard

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource
