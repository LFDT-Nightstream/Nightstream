import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the Lean-owned PiRLC transcript machine.
-/

namespace NightstreamTests.Axioms.CanonicalPiRlcCanonicalMachine

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine.digestChunks_lane_part' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalMachine.digestChunks_lane_part

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine.digestBlock_absorbed_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalMachine.digestBlock_absorbed_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine.fixedSchedule_successorState' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalMachine.fixedSchedule_successorState

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine.sampledChallenge_valid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalMachine.sampledChallenge_valid

end NightstreamTests.Axioms.CanonicalPiRlcCanonicalMachine
