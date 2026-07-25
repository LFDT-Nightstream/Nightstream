import tests.NumericRowBridge
import tests.Axioms.Support

/-!
Fail-closed guards for the semantics-preserving numeric-to-typed Goldilocks
row translation.
-/

namespace NightstreamTests.Axioms.NumericRowBridge

open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_mod' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms residue_mod

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_add' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms residue_add

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.residue_mul' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms residue_mul

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.numericAssignment_canonical' does not depend on any axioms -/
#guard_msgs in
#audit_axioms numericAssignment_canonical

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.terms_eval_eq_residue_lcEval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms terms_eval_eq_residue_lcEval

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.row_columnIds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms row_columnIds

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.row_holds_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms row_holds_iff

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ownedRowsFrom_length

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_rows' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ownedRowsFrom_rows

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_ids_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ownedRowsFrom_ids_exact

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_owned' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ownedRowsFrom_owned

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_ids_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ownedRowsFrom_ids_nodup

/-- info: 'Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge.ownedRowsFrom_satisfies_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ownedRowsFrom_satisfies_iff

end NightstreamTests.Axioms.NumericRowBridge
