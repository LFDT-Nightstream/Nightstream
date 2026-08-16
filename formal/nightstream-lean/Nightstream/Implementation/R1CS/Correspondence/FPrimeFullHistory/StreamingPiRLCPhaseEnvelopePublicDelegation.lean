import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope

/-!
Contract: exact row-range bridge from the public PiRLC family artifact to the
separate phase-envelope artifact.

Owns only the four delegated row boundaries. It does not own either artifact's
validity, phase semantics, public-family semantics, or selective lowering.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope

/-- The public suffix delegates exactly the same two source intervals that
the phase-envelope artifact owns. -/
theorem public_delegation_exact :
    Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic.rawArtifact.even.phaseEnvelopeRowStart =
        evenArm.phaseRowStart ∧
      Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic.rawArtifact.even.phaseEnvelopeRowEnd =
        evenArm.phaseRowEnd ∧
      Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic.rawArtifact.odd.phaseEnvelopeRowStart =
        oddArm.phaseRowStart ∧
      Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic.rawArtifact.odd.phaseEnvelopeRowEnd =
        oddArm.phaseRowEnd := by
  constructor
  · rfl
  constructor
  · rfl
  constructor <;> rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact
