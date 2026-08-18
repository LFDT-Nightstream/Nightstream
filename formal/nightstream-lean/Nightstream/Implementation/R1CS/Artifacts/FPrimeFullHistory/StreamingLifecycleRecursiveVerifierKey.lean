import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey

/-!
Facade for the exact recursive lifecycle verifier-key source-stage artifact.

Owns the compact artifact identity and transports the generated structural
certificates without unfolding either run list.

Does not prove row semantics, Rust assignment conformance, or that this
monolithic-reference placement is the final phased profile.

Assurance tier: artifact-checked for
`FPRIME-STREAMING-LIFECYCLE-RECURSIVE-VERIFIER-KEY-PROVENANCE-V1`,
Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKey

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey

/-- A four-field digest preimage slice owned by one source-stage column range. -/
def FourFieldSliceInside (outer inner : Range) : Prop :=
  inner.stop = inner.start + 4 ∧
  outer.start ≤ inner.start ∧
  inner.stop ≤ outer.stop

/-- Exact identity and structural coverage exposed to handwritten consumers. -/
structure Valid (artifact : RawArtifact) : Prop where
  schemaVersion : artifact.schemaVersion = 2
  profileId : artifact.profileId =
    "nightstream/goldilocks/streaming-lifecycle-selective/v1"
  sourceArtifactIdentity : artifact.sourceArtifactIdentity =
    "rust:nightstream/streaming-lifecycle-recursive/source-rows/v1"
  finalArtifactIdentity : artifact.finalArtifactIdentity =
    "rust:nightstream/streaming-lifecycle-selective/final-rows/v1"
  stagePath : artifact.stagePath = "fprime.recursive.verifier_key"
  sourceColumns : artifact.sourceColumns =
    { start := 30388263, stop := 30400381 }
  structureDigestColumns : artifact.structureDigestColumns =
    { start := 30388263, stop := 30388267 }
  ajtaiPpDigestColumns : artifact.ajtaiPpDigestColumns =
    { start := 30388267, stop := 30388271 }
  initialSemanticStateDigestColumns : artifact.initialSemanticStateDigestColumns =
    { start := 30388271, stop := 30388275 }
  structureDigestInside : FourFieldSliceInside artifact.sourceColumns
    artifact.structureDigestColumns
  ajtaiPpDigestInside : FourFieldSliceInside artifact.sourceColumns
    artifact.ajtaiPpDigestColumns
  initialSemanticStateDigestInside : FourFieldSliceInside artifact.sourceColumns
    artifact.initialSemanticStateDigestColumns
  sourceRunCoverage : SourceRunChain artifact.sourceRows.start
    artifact.sourceRuns artifact.sourceRows.stop
  finalRunBounds : FinalRunsWithin artifact.finalRowCount artifact.finalRuns

theorem rawArtifact_valid : Valid rawArtifact where
  schemaVersion := rfl
  profileId := rfl
  sourceArtifactIdentity := rfl
  finalArtifactIdentity := rfl
  stagePath := rfl
  sourceColumns := rfl
  structureDigestColumns := rfl
  ajtaiPpDigestColumns := rfl
  initialSemanticStateDigestColumns := rfl
  structureDigestInside := by
    change 30388267 = 30388263 + 4 ∧
      30388263 ≤ 30388263 ∧ 30388267 ≤ 30400381
    exact ⟨by decide, by decide, by decide⟩
  ajtaiPpDigestInside := by
    change 30388271 = 30388267 + 4 ∧
      30388263 ≤ 30388267 ∧ 30388271 ≤ 30400381
    exact ⟨by decide, by decide, by decide⟩
  initialSemanticStateDigestInside := by
    change 30388275 = 30388271 + 4 ∧
      30388263 ≤ 30388271 ∧ 30388275 ≤ 30400381
    exact ⟨by decide, by decide, by decide⟩
  sourceRunCoverage := by
    change SourceRunChain 30664206 sourceRuns 30676324
    exact sourceRuns_cover
  finalRunBounds := by
    change FinalRunsWithin 10306243 finalRuns
    exact finalRuns_inside

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKey
