import Nightstream.Implementation.R1CS.Correspondence.Gadgets.Phi81DynamicEvaluationRows
import tests.Axioms.Support

open Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows

/-- info: 'Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows.combineFields_eq_matrixVectorAt' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms combineFields_eq_matrixVectorAt

/-- info: 'Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows.decodeCarried_rowCarried' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeCarried_rowCarried

/-- info: 'Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows.decodeTable_eq_phi81Table' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodeTable_eq_phi81Table

/-- info: 'Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows.decodePoint_eq_decodedPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms decodePoint_eq_decodedPoint

/-- info: 'Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound
