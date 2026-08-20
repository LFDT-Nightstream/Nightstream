import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound

/-!
Contract: concrete exact row soundness for the terminal `is` and `fs` leaves.

Both leaves instantiate the shared leaf theorem with Rust-owned source order,
row geometry, and verifier-owned schedules. Their four-field digests are
derived from checked canonical openings, seeded maps, and Poseidon2 rows.

It does not own sampler no-rejection liveness, collision resistance,
Module-SIS security, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerIsFsLeafRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound

abbrev IsLeafSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfied rawArtifact.isLeaf assignment

abbrev IsLeafSound (assignment : Nat → Nat) : Prop :=
  Sound rawArtifact.isLeaf rawArtifact.isColumns
    rawArtifact.opsLeaf.digestRowStop rawArtifact.fsLeaf.prefixPinRowStart
    assignment

theorem isLeaf_rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : IsLeafSatisfied assignment) :
    IsLeafSound assignment := by
  exact rows_sound isLeaf_valid assignment canonical one satisfied

abbrev FsLeafSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfied rawArtifact.fsLeaf assignment

abbrev FsLeafSound (assignment : Nat → Nat) : Prop :=
  Sound rawArtifact.fsLeaf rawArtifact.fsColumns
    rawArtifact.isLeaf.digestRowStop
    (rawArtifact.coreRowStart + rawArtifact.leavesRowStop) assignment

theorem fsLeaf_rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : FsLeafSatisfied assignment) :
    FsLeafSound assignment := by
  exact rows_sound fsLeaf_valid assignment canonical one satisfied

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerIsFsLeafRowSound
