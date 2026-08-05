import Nightstream.Implementation.Lowering.Nebula.TerminalR1csCost
import Nightstream.Implementation.Lowering.Nebula.TerminalR1csHonest
import tests.Axioms.Support

/-! Fail-closed dependency guards for the Lean-owned Nebula terminal R1CS. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1cs.lowerRow_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1cs.lowerRow_sound

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1cs.lowerRow_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1cs.lowerRow_complete

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram.sound

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram.wasm42x6_rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram.wasm42x6_rows_length

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram.wasm42x6_columns_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1csProgram.wasm42x6_columns_length

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1csCompilerShape.compilerRows_wellShaped' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1csCompilerShape.compilerRows_wellShaped

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1csHonest.completeCompiler_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1csHonest.completeCompiler_satisfies

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1csCost.relationRows_eq_emitted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1csCost.relationRows_eq_emitted

/-- info: 'Nightstream.Implementation.Lowering.Nebula.TerminalR1csCost.auxiliaryColumns_eq_emitted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.TerminalR1csCost.auxiliaryColumns_eq_emitted
