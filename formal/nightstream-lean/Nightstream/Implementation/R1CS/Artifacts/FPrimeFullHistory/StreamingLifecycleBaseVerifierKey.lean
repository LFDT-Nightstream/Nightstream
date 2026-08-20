import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey

/-!
Facade for the exact base lifecycle verifier-key core artifact.

Owns the compact artifact identity and transports the generated structural
certificates without unfolding either run list.

Does not prove row semantics, Rust assignment conformance, or the additional
base-stage public-trace and Nebula initialization rows.

Assurance tier: artifact-checked for
`FPRIME-STREAMING-LIFECYCLE-BASE-VERIFIER-KEY-PROVENANCE-V1`,
Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKey

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey

/-- A four-field digest preimage slice owned by one source-stage column range. -/
def FourFieldSliceInside (outer inner : Range) : Prop :=
  inner.stop = inner.start + 4 ∧
  outer.start ≤ inner.start ∧
  inner.stop ≤ outer.stop

/-- Exact identity and structural coverage exposed to handwritten consumers. -/
structure Valid (artifact : RawArtifact) : Prop where
  schemaVersion : artifact.schemaVersion = 1
  profileId : artifact.profileId =
    "nightstream/goldilocks/streaming-lifecycle-selective/v1"
  sourceArtifactIdentity : artifact.sourceArtifactIdentity =
    "rust:nightstream/streaming-lifecycle-base/source-rows/v1"
  finalArtifactIdentity : artifact.finalArtifactIdentity =
    "rust:nightstream/streaming-lifecycle-selective/final-rows/v1"
  stagePath : artifact.stagePath = "fprime.base.verifier_key"
  sourceColumns : artifact.sourceColumns =
    { start := 36007, stop := 50543 }
  structureDigestColumns : artifact.structureDigestColumns =
    { start := 36007, stop := 36011 }
  ajtaiPpDigestColumns : artifact.ajtaiPpDigestColumns =
    { start := 36011, stop := 36015 }
  initialSemanticStateDigestColumns : artifact.initialSemanticStateDigestColumns =
    { start := 36015, stop := 36019 }
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
    change 36011 = 36007 + 4 ∧
      36007 ≤ 36007 ∧ 36011 ≤ 50543
    exact ⟨by decide, by decide, by decide⟩
  ajtaiPpDigestInside := by
    change 36015 = 36011 + 4 ∧
      36007 ≤ 36011 ∧ 36015 ≤ 50543
    exact ⟨by decide, by decide, by decide⟩
  initialSemanticStateDigestInside := by
    change 36019 = 36015 + 4 ∧
      36007 ≤ 36015 ∧ 36019 ≤ 50543
    exact ⟨by decide, by decide, by decide⟩
  sourceRunCoverage := by
    change SourceRunChain 36133 sourceRuns 50723
    exact sourceRuns_cover
  finalRunBounds := by
    change FinalRunsWithin 10306243 finalRuns
    exact finalRuns_inside

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKey
