import Nightstream.Implementation.R1CS.Correspondence.Gadgets.Phi81SharedEvaluationRows
import tests.Axioms.Support

open Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows

/-- info: 'Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.rowCombination_eval' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms rowCombination_eval

/-- info: 'Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.rowActive_false_coefficients_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rowActive_false_coefficients_zero

/-- info: 'Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
