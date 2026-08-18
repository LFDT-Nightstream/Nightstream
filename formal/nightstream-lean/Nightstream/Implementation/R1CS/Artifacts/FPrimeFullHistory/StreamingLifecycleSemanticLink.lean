import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink

/-! Public facade and structural certificate for the exact Rust lifecycle
semantic-link artifact. -/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLink

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink

theorem rawArtifact_valid :
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.Valid
      rawArtifact := by
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.Valid
  refine ⟨rfl, rfl, rfl, by decide, ?_⟩
  norm_num [rawArtifact, phaseConstantValues,
    expectedConstantValues, totalRows, payloadRowCount, equalityRowCount,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.payloadFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.hashTotalRows,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.hashConstantFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.hashTraceRows,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.hashInputFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.domainFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.digestFields,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.absorbRounds,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.absorbRoundRows,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.permutationRows,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.HashRecipe.callOutputColumns,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.HashRecipe.callFirstAllocatedColumn,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.HashRecipe.roundColumnStart,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.HashRecipe.zeroColumn,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.HashRecipe.constantColumns,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact.HashRecipe.definitionCount,
    List.range',
    RawArtifact.hashRecipe, RawArtifact.hashConstantStartColumn,
    RawArtifact.hashOutputColumns]

theorem input_length (side : StateSide) :
    ((rawArtifact.hashRecipe side).inputColumns).length = hashInputFields := by
  cases side <;>
    norm_num [HashRecipe.inputColumns, HashRecipe.constantColumns,
      RawArtifact.hashRecipe, RawArtifact.hashConstantStartColumn,
      RawArtifact.localColumns, RawArtifact.payloadColumns,
      RawArtifact.payloadStartColumn, rawArtifact, phaseConstantValues,
      hashInputFields, hashConstantFields, domainFields, payloadFields,
      digestFields]

theorem output_exact (side : StateSide) :
    (rawArtifact.hashRecipe side).outputColumns =
      ((rawArtifact.hashRecipe side).callOutputColumns absorbRounds).take 4 := by
  cases side <;> rfl

theorem constantValues_canonical :
    ∀ value ∈ rawArtifact.constantValues,
      0 < value ∧ value < goldilocksP := by
  norm_num [rawArtifact, phaseConstantValues, goldilocksP]

theorem semanticColumns_length (side : StateSide) :
    (rawArtifact.semanticColumns side).length = digestFields := by
  cases side <;>
    norm_num [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.semanticColumns,
      rawArtifact, digestFields]

theorem hashOutputColumns_length (side : StateSide) :
    (rawArtifact.hashOutputColumns side).length = digestFields := by
  cases side <;>
    norm_num [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.hashOutputColumns,
      rawArtifact, digestFields]

theorem semanticColumns_between
    (side : StateSide) {column : Nat}
    (member : column ∈ rawArtifact.semanticColumns side) :
    1 ≤ column ∧ column ≤ 8 := by
  cases side <;>
    simp [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.semanticColumns,
      rawArtifact] at member <;> omega

private theorem digestLaneRange_exact :
    List.range digestFields = [0, 1, 2, 3] := by
  rfl

theorem equalitySemanticColumns_nodup :
    ((List.range digestFields).flatMap fun lane =>
      [(rawArtifact.semanticColumns .before).getD lane 0,
       (rawArtifact.semanticColumns .after).getD lane 0]).Nodup := by
  rw [digestLaneRange_exact]
  norm_num [rawArtifact,
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact.semanticColumns]

theorem trace_ownedValid (side : StateSide) :
    (rawArtifact.hashRecipe side).trace.OwnedValid :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.hashRecipe_trace_ownedValid
    _ (input_length side) (output_exact side)

theorem valueSchedules_exact (side : StateSide) :
    Nightstream.Implementation.R1CS.Poseidon2Sponge.valueSchedules
        (rawArtifact.hashRecipe side).trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.hashRecipe_valueSchedules_exact
    _ (input_length side)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLink
