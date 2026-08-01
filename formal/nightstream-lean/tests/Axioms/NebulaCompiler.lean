import Nightstream.Implementation.Lowering.Nebula.Compiler
import tests.Axioms.Support

/-! Fail-closed dependency guards for the Lean-owned Nebula compiler. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Layout.wasm42x6_valid' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Layout.wasm42x6_valid

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Layout.wasm42x6_columnCount' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Layout.wasm42x6_columnCount

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Compiler.operationRows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Compiler.operationRows_length

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Compiler.scanRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Compiler.scanRows_length

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Compiler.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Compiler.rows_length

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Compiler.wasm42x6_rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Compiler.wasm42x6_rows_length
