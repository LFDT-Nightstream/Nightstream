import Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
import tests.Axioms.Support

/-!
Fail-closed dependency gate for Poseidon2 honest completeness.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Honest

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_directions_agree' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_directions_agree

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_scheduleOf' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_scheduleOf

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_initialState' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_initialState

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_partialState' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_partialState

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_terminalState' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_terminalState

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_residues' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_residues

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_inputPort' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_inputPort

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_outputPort' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_outputPort

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_sboxColumn' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_sboxColumn

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.honest_sboxOutput' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.honest_sboxOutput

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.sboxInputValue_initial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.sboxInputValue_initial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.sboxInputValue_partial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.sboxInputValue_partial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest.sboxInputValue_terminal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Honest.sboxInputValue_terminal

end NightstreamTests.Axioms.CanonicalPoseidon2Honest
