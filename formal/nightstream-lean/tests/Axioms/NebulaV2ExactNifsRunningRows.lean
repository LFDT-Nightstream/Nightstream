import Nightstream.Implementation.NebulaV2.ExactNifsRunningRows
import tests.Axioms.Support

/-! Axiom gates for the exact accepted-running-input row bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ExactNifsRunningRows.bitsPlaced_of_fullClaim' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ExactNifsRunningRows.bitsPlaced_of_fullClaim

/-- info: 'Nightstream.Implementation.NebulaV2.ExactNifsRunningRows.input_matches_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ExactNifsRunningRows.input_matches_rows

/-- info: 'Nightstream.Implementation.NebulaV2.ExactNifsRunningRows.selected_input_matches_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ExactNifsRunningRows.selected_input_matches_rows
