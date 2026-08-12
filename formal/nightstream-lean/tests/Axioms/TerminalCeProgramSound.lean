import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeProgramSound
import tests.Axioms.Support

open Nightstream.Implementation.R1CS.TerminalCeProgramSound

/-- info: 'Nightstream.Implementation.R1CS.TerminalCeProgramSound.decodedEvaluations_eq_expected_of_fields' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodedEvaluations_eq_expected_of_fields

/-- info: 'Nightstream.Implementation.R1CS.TerminalCeProgramSound.decodedNc_eq_expected_of_fields' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms decodedNc_eq_expected_of_fields

/-- info: 'Nightstream.Implementation.R1CS.TerminalCeProgramSound.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
