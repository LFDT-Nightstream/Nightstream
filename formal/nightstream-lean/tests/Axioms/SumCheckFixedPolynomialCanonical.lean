import Nightstream.SuperNeo.SumCheck.FixedPolynomialCanonical
import tests.Axioms.Support

/-! Fail-closed dependency gate for the artifact-independent fixed-polynomial
canonicalizer. -/

open Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.canonicalMessage_coefficients_eq_prefix_zero_padding' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms canonicalMessage_coefficients_eq_prefix_zero_padding

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.canonicalMessage_evaluate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms canonicalMessage_evaluate

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.canonicalMessage_canonical' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms canonicalMessage_canonical

/-- info: 'Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial.canonicalMessage_degreeUpperBound_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms canonicalMessage_degreeUpperBound_le
