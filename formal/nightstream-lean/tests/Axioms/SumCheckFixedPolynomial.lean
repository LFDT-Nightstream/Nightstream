import Nightstream.SuperNeo.SumCheck.FixedPolynomial
import tests.Axioms.Support

/-!
Fail-closed dependency gate for typed fixed-width SumCheck polynomials.

| Audited theorem | Guarantee |
|---|---|
| `toMessage_degreeUpperBound` | exact typed width becomes the exact verifier-derived degree |
| `evaluate_eq_message_evaluate` | no second polynomial evaluator is trusted |
| `evaluate_widen` | fixed wider messages append only semantically inert high zeros |
| `evaluate_mul` | exact-width convolution has product semantics |
| `evaluate_power` | natural powers preserve multiplied static degree and evaluation |
| `evaluate_sum` | explicit finite summation preserves evaluation |
-/

open Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.toMessage_degreeUpperBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms toMessage_degreeUpperBound

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.evaluate_eq_message_evaluate' does not depend on any axioms -/
#guard_msgs in
#audit_axioms evaluate_eq_message_evaluate

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.evaluate_widen' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms evaluate_widen

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.evaluate_mul' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms evaluate_mul

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.evaluate_power' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms evaluate_power

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.evaluate_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluate_sum
