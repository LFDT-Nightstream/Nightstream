import Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
import tests.Axioms.Support

/-!
Fail-closed dependency gate for Poseidon2 semantic conformance.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2RoundInduction

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.lcEval_singleton' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.lcEval_singleton

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.applyMatrixValues_congr' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.applyMatrixValues_congr

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.satisfies_sboxChain' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.satisfies_sboxChain

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.scheduleOf_initial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.scheduleOf_initial

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.scheduleOf_terminal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.scheduleOf_terminal

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.initialState_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.initialState_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.partialState_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.partialState_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.terminalState_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.terminalState_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.canonicalProgram_computes_reference' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.canonicalProgram_computes_reference

/-! The initial phase on a carried entry. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.initialStateFrom_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.initialStateFrom_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.scheduleOfFrom_initial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.scheduleOfFrom_initial

/-! Per-call soundness on a carried entry. -/

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.canonicalProgramFrom_computes_reference' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.canonicalProgramFrom_computes_reference

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.partialStateFrom_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.partialStateFrom_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.terminalStateFrom_eval' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.terminalStateFrom_eval

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction.scheduleOfFrom_nonInitial' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2RoundInduction.scheduleOfFrom_nonInitial

end NightstreamTests.Axioms.CanonicalPoseidon2RoundInduction
