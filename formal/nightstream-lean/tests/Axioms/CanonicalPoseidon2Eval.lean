import Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
import tests.Axioms.Support

/-!
Fail-closed dependency gate for combination evaluation algebra.
No theorem may acquire `Lean.trustCompiler`.
-/

namespace NightstreamTests.Axioms.CanonicalPoseidon2Eval

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval.mul_mod_shift' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Eval.mul_mod_shift

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval.mul_mod_right_reduce' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Eval.mul_mod_right_reduce

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval.rawSum_scale_mod' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Eval.rawSum_scale_mod

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval.lcEval_scale' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Eval.lcEval_scale

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval.sum_mod_congr' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Poseidon2Eval.sum_mod_congr

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval.lcEval_applyMatrix' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Eval.lcEval_applyMatrix

/-- info: 'Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval.lcEval_addConstant' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Poseidon2Eval.lcEval_addConstant

end NightstreamTests.Axioms.CanonicalPoseidon2Eval
