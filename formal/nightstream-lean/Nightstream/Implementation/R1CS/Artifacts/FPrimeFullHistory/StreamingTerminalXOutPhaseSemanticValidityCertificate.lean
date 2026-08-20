import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-!
Contract: structural validity certificate for the production terminal
phase-semantic VariableHashRecipe.

Owns complete input absorption, the terminal pad round, and the exact
four-lane hash output. It does not evaluate or enumerate the expanded row
program.

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
    hashInputFields, constantFields, digestFields, payloadFields]

theorem absorbRounds_exact :
    rawArtifact.hashRecipe.absorbRounds = absorbRounds := by
  norm_num [VariableHashRecipe.absorbRounds, input_length, rate,
    absorbRounds, hashInputFields]

theorem output_exact :
    rawArtifact.hashRecipe.outputColumns =
      (rawArtifact.hashRecipe.callOutputColumns
        rawArtifact.hashRecipe.absorbRounds).take 4 := by
  have full :
      rawArtifact.hashRecipe.inputColumns.length =
        rate * rawArtifact.hashRecipe.absorbRounds := by
    rw [input_length, absorbRounds_exact]
    norm_num [rate, hashInputFields, absorbRounds,
      constantFields, digestFields, payloadFields]
  rw [finalCallOutputColumns_eq_of_fullAbsorbRounds
    rawArtifact.hashRecipe full]
  rw [absorbRounds_exact]
  norm_num [rawArtifact, RawArtifact.hashRecipe, phaseConstantValues,
    VariableHashRecipe.zeroColumn, absorbRounds,
    hashInputFields, constantFields, digestFields, payloadFields,
    rate, permutationRows]
  rfl

theorem trace_ownedValid : rawArtifact.hashRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.hashRecipe (by
      rw [absorbRounds_exact]
      norm_num [absorbRounds, hashInputFields,
        constantFields, digestFields, payloadFields, rate]) (by
      rw [input_length, absorbRounds_exact]
      norm_num [hashInputFields, absorbRounds,
        constantFields, digestFields, payloadFields, rate]) output_exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic
