import Nightstream.Implementation.Lowering.Nebula.StepPolynomial
import tests.Axioms.Support

/-! Fail-closed dependency guards for the Lean-owned Nebula step polynomial. -/

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.Role.index_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.Role.index_injective

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.canonicalEqualityGatedDegreeBound_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.canonicalEqualityGatedDegreeBound_exact

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_eq_residual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_eq_residual

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_bitPoint' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_bitPoint

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_productPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_productPoint

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_productEqualityPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_productEqualityPoint

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_linearPoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_linearPoint

/-- info: 'Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_extensionUpdatePoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Lowering.Nebula.StepPolynomial.evaluate_extensionUpdatePoint
