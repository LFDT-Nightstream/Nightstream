import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the Lean-owned symbolic PiRLC sampler schedule.
-/

namespace NightstreamTests.Axioms.CanonicalPiRlcCanonicalSymbolicMachine

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.decoded_appendRawPair' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachine.decoded_appendRawPair

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.decoded_digestBlock' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachine.decoded_digestBlock

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.digestLanes_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachine.digestLanes_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.decoded_stateBeforeBlock' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachine.decoded_stateBeforeBlock

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.decoded_scalarBuilder' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachine.decoded_scalarBuilder

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.fixedActive_entries_length_of_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  PiRlcCanonicalSymbolicMachine.fixedActive_entries_length_of_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.fixedActive_rows_length_of_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms
  PiRlcCanonicalSymbolicMachine.fixedActive_rows_length_of_zero

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.decoded_stateAt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachine.decoded_stateAt

/-- info: 'Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSymbolicMachine.fixedActive_challengeCount' does not depend on any axioms -/
#guard_msgs in
#audit_axioms PiRlcCanonicalSymbolicMachine.fixedActive_challengeCount

end NightstreamTests.Axioms.CanonicalPiRlcCanonicalSymbolicMachine
