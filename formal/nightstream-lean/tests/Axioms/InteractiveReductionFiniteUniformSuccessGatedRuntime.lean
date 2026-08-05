import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the success-gated extractor runtime theorem.
-/

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.gatedRetryExpectedWork_le_oneRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms gatedRetryExpectedWork_le_oneRun

/-- info: 'Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.SuccessGatedRuntime.extractorExpectedPolynomialTime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms extractorExpectedPolynomialTime
