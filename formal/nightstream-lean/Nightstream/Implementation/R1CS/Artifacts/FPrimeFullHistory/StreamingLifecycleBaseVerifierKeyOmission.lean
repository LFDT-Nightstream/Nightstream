import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission

/-! Public facade for the exact base verifier-key omission projection. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmission

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission

theorem rawArtifact_valid : rawArtifact.Valid := by
  refine {
    schemaVersion := rfl
    profileId := rfl
    lifecycleScope := rfl
    family := rfl
    stagePath := rfl
    sourceRowsOrdered := by decide
    sourceColumnsOrdered := by decide
    rowPartition := by decide
    constantOne := rfl
    changedColumnInBounds := by decide
    baselineCanonical := by decide
    candidateCanonical := by decide
    targetFails := by decide
    sourceRunsCover := sourceRuns_cover
    finalRunsInside := finalRuns_inside
    retainedRowsIgnore := occurrences_owned
  }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmission
