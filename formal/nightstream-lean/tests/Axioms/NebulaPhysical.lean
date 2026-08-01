import Nightstream.Implementation.Lowering.Nebula.Physical
import tests.Axioms.Support

/-! Fail-closed dependency guards for the Lean-owned Nebula physical layout. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Compiler.rows_ids_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Compiler.rows_ids_nodup

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_rows_bounded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_rows_bounded

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Physical.allocatedColumns_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Physical.allocatedColumns_nodup

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_every_term_allocated' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_every_term_allocated

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_allocatedColumns_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_allocatedColumns_length

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_publicColumnCount' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_publicColumnCount

/-- info: 'Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_witnessColumnCount' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.Physical.wasm42x6_witnessColumnCount
