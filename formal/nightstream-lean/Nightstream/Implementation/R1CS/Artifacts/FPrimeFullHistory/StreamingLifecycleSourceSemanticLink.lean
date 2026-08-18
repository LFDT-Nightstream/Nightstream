import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink

/-! Public facade and structural certificates for the exact base and recursive
lifecycle semantic-link source artifacts. -/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSourceSemanticLink

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink

theorem baseArtifact_valid : baseArtifact.Valid := by
  unfold SourceArtifact.Valid
  refine ⟨rfl, rfl, rfl, by decide, ?_⟩
  norm_num [baseArtifact, phaseConstantValues, expectedConstantValues,
    SourceScope.rowCount, SourceScope.beforePayloadRowCount,
    payloadFields, hashTotalRows, hashConstantFields, hashTraceRows,
    hashInputFields, domainFields, digestFields, absorbRounds,
    absorbRoundRows, permutationRows,
    HashRecipe.callOutputColumns, HashRecipe.callFirstAllocatedColumn,
    HashRecipe.roundColumnStart, HashRecipe.zeroColumn,
    HashRecipe.constantColumns, HashRecipe.definitionCount,
    List.range', SourceArtifact.hashRecipe,
    SourceArtifact.hashConstantStartColumn,
    SourceArtifact.hashOutputColumns]

theorem recursiveArtifact_valid : recursiveArtifact.Valid := by
  unfold SourceArtifact.Valid
  refine ⟨rfl, rfl, rfl, by decide, ?_⟩
  norm_num [recursiveArtifact, phaseConstantValues, expectedConstantValues,
    SourceScope.rowCount, SourceScope.beforePayloadRowCount,
    payloadFields, hashTotalRows, hashConstantFields, hashTraceRows,
    hashInputFields, domainFields, digestFields, absorbRounds,
    absorbRoundRows, permutationRows,
    HashRecipe.callOutputColumns, HashRecipe.callFirstAllocatedColumn,
    HashRecipe.roundColumnStart, HashRecipe.zeroColumn,
    HashRecipe.constantColumns, HashRecipe.definitionCount,
    List.range', SourceArtifact.hashRecipe,
    SourceArtifact.hashConstantStartColumn,
    SourceArtifact.hashOutputColumns]

private theorem base_input_length (side : StateSide) :
    ((baseArtifact.hashRecipe side).inputColumns).length = hashInputFields := by
  cases side <;>
    norm_num [HashRecipe.inputColumns, HashRecipe.constantColumns,
      SourceArtifact.hashRecipe, SourceArtifact.hashConstantStartColumn,
      SourceArtifact.localColumns, SourceArtifact.payloadColumns,
      SourceArtifact.payloadStartColumn, baseArtifact, phaseConstantValues,
      hashInputFields, hashConstantFields, domainFields, payloadFields,
      digestFields]

private theorem base_output_exact (side : StateSide) :
    (baseArtifact.hashRecipe side).outputColumns =
      ((baseArtifact.hashRecipe side).callOutputColumns absorbRounds).take 4 := by
  cases side <;> rfl

private theorem recursive_hashRecipe_eq (side : StateSide) :
    recursiveArtifact.hashRecipe side = baseArtifact.hashRecipe side := by
  cases side <;> rfl

theorem base_trace_ownedValid (side : StateSide) :
    (baseArtifact.hashRecipe side).trace.OwnedValid :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.hashRecipe_trace_ownedValid
    _ (base_input_length side) (base_output_exact side)

theorem recursive_trace_ownedValid (side : StateSide) :
    (recursiveArtifact.hashRecipe side).trace.OwnedValid := by
  rw [recursive_hashRecipe_eq]
  exact base_trace_ownedValid side

theorem base_valueSchedules_exact (side : StateSide) :
    Nightstream.Implementation.R1CS.Poseidon2Sponge.valueSchedules
        (baseArtifact.hashRecipe side).trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.hashRecipe_valueSchedules_exact
    _ (base_input_length side)

theorem recursive_valueSchedules_exact (side : StateSide) :
    Nightstream.Implementation.R1CS.Poseidon2Sponge.valueSchedules
        (recursiveArtifact.hashRecipe side).trace.rounds =
      List.replicate absorbRounds (.absorb 4) ++ [.pad] := by
  rw [recursive_hashRecipe_eq]
  exact base_valueSchedules_exact side

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSourceSemanticLink
