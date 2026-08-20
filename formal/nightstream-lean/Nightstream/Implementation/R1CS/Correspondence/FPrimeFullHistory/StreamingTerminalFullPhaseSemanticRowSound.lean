import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullPhaseSemantic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound

/-!
Contract: all exact full-layout terminal phase-semantic rows recompute
Poseidon2 from the phase-local state and delayed payload and bind XOut lanes
19 through 22 to that result.

This leaf does not claim that either preimage slice is lifecycle-authoritative.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullPhaseSemanticRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullPhaseSemantic
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound

private abbrev fullArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic.rawArtifact

abbrev Sound := SoundFor fullArtifact

/-- Full-layout Rust satisfaction implies the named phase-semantic relation.
Rust checked that this recipe is the exact relocation of the authoritative
small audit family. -/
theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    Sound assignment := by
  exact rows_sound_for fullArtifact assignment canonical one satisfied
    trace_ownedValid rawArtifact_valid.constantsCanonical

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullPhaseSemanticRowSound
