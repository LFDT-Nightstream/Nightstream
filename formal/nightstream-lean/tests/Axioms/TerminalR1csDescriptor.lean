import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest
import tests.Axioms.Support

/-! Fail-closed dependency guard for the direct terminal-R1CS descriptor. -/

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Descriptor.cost_ofProgram' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Descriptor.cost_ofProgram

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest.terminalR1cs_logicalWidth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest.terminalR1cs_logicalWidth

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest.terminalR1cs_recursiveRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsManifest.terminalR1cs_recursiveRows
