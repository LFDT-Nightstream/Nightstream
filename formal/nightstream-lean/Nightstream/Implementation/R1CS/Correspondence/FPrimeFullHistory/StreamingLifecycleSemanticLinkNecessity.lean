import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLink

/-!
Contract: exact omission counterexample for the lifecycle semantic-link family.

The selected family owns both Poseidon2 traces and the eight outer semantic
links. When it is absent, the retained payload-domain rows accept this
canonical assignment, but the independent `SemanticLink` target rejects it.

This file does not claim that any row is redundant. It proves that the complete
semantic-link family must be retained.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLinkNecessity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink

def baselineAssignment (column : Nat) : Nat :=
  if column = 0 then 1 else 0

def beforeLane : Fin 4 := ⟨0, by omega⟩

def baselineDigest : Nat :=
  phaseEnvelopeDigest rawArtifact .before baselineAssignment beforeLane

def differentFrom (value : Nat) : Nat :=
  if value = 0 then 1 else 0

def omissionAssignment (column : Nat) : Nat :=
  if column = 0 then 1
  else if column = 1 then differentFrom baselineDigest
  else 0

private theorem differentFrom_ne (value : Nat) : differentFrom value ≠ value := by
  by_cases zero : value = 0
  · simp [differentFrom, zero]
  · intro equal
    exact zero (by simpa [differentFrom, zero] using equal.symm)

theorem omissionAssignment_one : omissionAssignment 0 = 1 := by
  simp [omissionAssignment]

theorem omissionAssignment_canonical (column : Nat) :
    omissionAssignment column < goldilocksP := by
  by_cases zero : column = 0
  · subst column
    norm_num [omissionAssignment, goldilocksP]
  by_cases semantic : column = 1
  · subst column
    simp only [omissionAssignment, if_false (by omega), if_true]
    by_cases digestZero : baselineDigest = 0 <;>
      norm_num [differentFrom, digestZero, goldilocksP]
  · simp [omissionAssignment, zero, semantic, goldilocksP]

private theorem inputAssignment_eq
    {column : Nat} (notSemantic : column ≠ 1) :
    omissionAssignment column = baselineAssignment column := by
  by_cases zero : column = 0
  · subst column
    simp [omissionAssignment, baselineAssignment]
  · simp [omissionAssignment, baselineAssignment, zero, notSemantic]

private theorem beforeLocal_notSemantic
    {column : Nat} (member : column ∈ rawArtifact.localColumns .before) :
    column ≠ 1 := by
  change column ∈ [9, 10, 11, 12] at member
  simp at member
  omega

private theorem payloadColumn_ge
    (side : StateSide) {column : Nat}
    (member : column ∈ rawArtifact.payloadColumns side) :
    17 ≤ column := by
  cases side with
  | before =>
      change column ∈ List.range' 17 payloadFields at member
      rcases List.mem_range'.mp member with ⟨index, indexLt, rfl⟩
      omega
  | after =>
      change column ∈ List.range' (17 + payloadFields) payloadFields at member
      rcases List.mem_range'.mp member with ⟨index, indexLt, rfl⟩
      omega

private theorem beforePhasePreimage_eq :
    phasePreimage rawArtifact .before omissionAssignment =
      phasePreimage rawArtifact .before baselineAssignment := by
  have localEqual :
      (rawArtifact.localColumns .before).map omissionAssignment =
        (rawArtifact.localColumns .before).map baselineAssignment := by
    apply List.map_congr_left
    intro column member
    exact inputAssignment_eq (beforeLocal_notSemantic member)
  have payloadEqual :
      (rawArtifact.payloadColumns .before).map omissionAssignment =
        (rawArtifact.payloadColumns .before).map baselineAssignment := by
    apply List.map_congr_left
    intro column member
    exact inputAssignment_eq (by
      have lower := payloadColumn_ge .before member
      omega)
  simp only [phasePreimage]
  rw [localEqual, payloadEqual]

private theorem beforeDigest_eq :
    phaseEnvelopeDigest rawArtifact .before omissionAssignment beforeLane =
      baselineDigest := by
  rw [baselineDigest]
  unfold phaseEnvelopeDigest phaseChunks
  rw [beforePhasePreimage_eq]

theorem retainedPayloadRows_hold :
    Satisfies rawArtifact.payloadRows omissionAssignment := by
  intro row member
  rw [RawArtifact.payloadRows] at member
  rcases List.mem_map.mp member with ⟨column, columnMember, rfl⟩
  rw [List.mem_append] at columnMember
  have lower : 17 ≤ column := by
    rcases columnMember with beforeMember | afterMember
    · exact payloadColumn_ge .before beforeMember
    · exact payloadColumn_ge .after afterMember
  have valueZero : omissionAssignment column = 0 := by
    simp [omissionAssignment, show column ≠ 0 by omega,
      show column ≠ 1 by omega]
  simp [bitRow, RowHolds, lcEval, valueZero]

theorem semanticLink_fails :
    ¬ SemanticLink rawArtifact omissionAssignment := by
  intro accepted
  have exact := accepted.semanticExact .before beforeLane
  change omissionAssignment 1 =
    phaseEnvelopeDigest rawArtifact .before omissionAssignment beforeLane at exact
  rw [beforeDigest_eq] at exact
  exact differentFrom_ne baselineDigest (by
    simpa [omissionAssignment] using exact)

/-- Lean-checked removal counterexample for the exact Rust semantic-link
artifact. The selected semantic-link family is absent; all retained payload
rows hold, but the complete typed target fails. -/
theorem exact_removal_counterexample :
    omissionAssignment 0 = 1 ∧
      (∀ column, omissionAssignment column < goldilocksP) ∧
      Satisfies rawArtifact.payloadRows omissionAssignment ∧
      ¬ SemanticLink rawArtifact omissionAssignment :=
  ⟨omissionAssignment_one, omissionAssignment_canonical,
    retainedPayloadRows_hold, semanticLink_fails⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLinkNecessity
