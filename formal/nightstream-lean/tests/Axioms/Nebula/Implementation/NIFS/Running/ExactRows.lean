import Nightstream.Implementation.Nebula.NIFS.Running.ExactRows
import tests.Axioms.Support

/-! Axiom gates for the exact accepted-running-input row bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ExactNifsRunningRows.bitsPlaced_of_fullClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ExactNifsRunningRows.bitsPlaced_of_fullClaim

/-- info: 'Nightstream.Implementation.Nebula.ExactNifsRunningRows.input_matches_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ExactNifsRunningRows.input_matches_rows

/-- info: 'Nightstream.Implementation.Nebula.ExactNifsRunningRows.selected_input_matches_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ExactNifsRunningRows.selected_input_matches_rows
