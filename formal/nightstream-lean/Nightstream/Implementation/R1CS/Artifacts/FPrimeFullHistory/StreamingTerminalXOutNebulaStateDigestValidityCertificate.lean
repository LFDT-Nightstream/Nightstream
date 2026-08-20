import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLink
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-!
Contract: structural validity certificates for both exact terminal Nebula
lane-digest branches.

Owns the 15 absorb rounds, terminal pad round, and four-lane output of each
compact recipe. It does not evaluate the expanded 19,353-row program.

Assurance tier: artifact-checked.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigest

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink

theorem absent_input_length :
    rawArtifact.absentRecipe.inputColumns.length = absentInputFields := by
  norm_num [VariableHashRecipe.inputColumns, RawArtifact.absentRecipe,
    rawArtifact, absentInputFields]

theorem absent_absorbRounds_exact :
    rawArtifact.absentRecipe.absorbRounds = 15 := by
  norm_num [VariableHashRecipe.absorbRounds, absent_input_length, rate,
    absentInputFields]

theorem absent_output_exact :
    rawArtifact.absentRecipe.outputColumns =
      (rawArtifact.absentRecipe.callOutputColumns
        rawArtifact.absentRecipe.absorbRounds).take 4 := by
  rw [absent_absorbRounds_exact]
  rfl

theorem absent_trace_ownedValid :
    rawArtifact.absentRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.absentRecipe (by
      rw [absent_absorbRounds_exact]
      norm_num) (by
      rw [absent_input_length, absent_absorbRounds_exact]
      norm_num [absentInputFields]) absent_output_exact

theorem present_input_length :
    rawArtifact.presentRecipe.inputColumns.length = presentInputFields := by
  norm_num [VariableHashRecipe.inputColumns, RawArtifact.presentRecipe,
    rawArtifact, presentInputFields]

theorem present_absorbRounds_exact :
    rawArtifact.presentRecipe.absorbRounds = 15 := by
  norm_num [VariableHashRecipe.absorbRounds, present_input_length, rate,
    presentInputFields]

theorem present_output_exact :
    rawArtifact.presentRecipe.outputColumns =
      (rawArtifact.presentRecipe.callOutputColumns
        rawArtifact.presentRecipe.absorbRounds).take 4 := by
  rw [present_absorbRounds_exact]
  rfl

theorem present_trace_ownedValid :
    rawArtifact.presentRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.presentRecipe (by
      rw [present_absorbRounds_exact]
      norm_num) (by
      rw [present_input_length, present_absorbRounds_exact]
      norm_num [presentInputFields]) present_output_exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigest
