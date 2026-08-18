import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-!
Contract: structural validity certificate for the 19-field terminal
phase-semantic VariableHashRecipe.

Owns five absorb rounds with chunk sizes `4, 4, 4, 4, 3`, the terminal pad
round, and the exact four-lane hash output. It does not evaluate the expanded
row program.

Assurance tier: artifact-checked.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic

theorem input_length :
    rawArtifact.hashRecipe.inputColumns.length = hashInputFields := by
  norm_num [VariableHashRecipe.inputColumns, VariableHashRecipe.constantColumns,
    RawArtifact.hashRecipe, rawArtifact, phaseConstantValues,
    hashInputFields]

theorem absorbRounds_exact :
    rawArtifact.hashRecipe.absorbRounds = absorbRounds := by
  norm_num [VariableHashRecipe.absorbRounds, input_length, rate,
    absorbRounds, hashInputFields]

theorem output_exact :
    rawArtifact.hashRecipe.outputColumns =
      (rawArtifact.hashRecipe.callOutputColumns
        rawArtifact.hashRecipe.absorbRounds).take 4 := by
  rw [absorbRounds_exact]
  rfl

theorem trace_ownedValid : rawArtifact.hashRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.hashRecipe (by
      rw [absorbRounds_exact]
      norm_num [absorbRounds]) (by
      rw [input_length, absorbRounds_exact]
      norm_num [hashInputFields, absorbRounds]) output_exact

theorem valueSchedules_exact :
    valueSchedules rawArtifact.hashRecipe.trace.rounds =
      [.absorb 4, .absorb 4, .absorb 4, .absorb 4, .absorb 3, .pad] := by
  rw [VariableHashRecipe.trace, VariableHashRecipe.rounds,
    absorbRounds_exact, valueSchedules, List.map_append]
  simp only [List.map_singleton, VariableHashRecipe.padRound,
    Round.valueSchedule]
  norm_num [List.range_succ, Function.comp_apply,
    rawArtifact, RawArtifact.hashRecipe, phaseConstantValues,
    VariableHashRecipe.absorbRounds, VariableHashRecipe.inputColumns,
    VariableHashRecipe.constantColumns, VariableHashRecipe.absorbRound,
    VariableHashRecipe.chunkColumns, Round.valueSchedule,
    rate, absorbRounds, hashInputFields]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic
