import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound

/-!
Contract: all exact full-layout terminal Nebula-state rows compute both
Poseidon2 lane-digest branches, select exactly one with the Boolean presence
wire, and bind XOut lanes 28 through 31 to the selected digest.

This leaf does not claim that the decoded lane fields are lifecycle-authoritative.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigestRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound

private abbrev fullArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest.rawArtifact

abbrev Sound := SoundFor fullArtifact

/-- Full-layout Rust satisfaction implies the selected Nebula-state digest
relation. Rust checked that this recipe is the exact relocation of the
authoritative small audit family. -/
theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    Sound assignment := by
  exact rows_sound_for fullArtifact assignment canonical one satisfied
    absent_trace_ownedValid present_trace_ownedValid
    rawArtifact_valid.absentConstantsCanonical
    rawArtifact_valid.presentConstantsCanonical

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigestRowSound
