import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits
import tests.Axioms.Support

/-!
Fail-closed dependency gate for the model-level common-sign radix family.

These theorems may use Lean's ordinary proposition and quotient soundness.
They must not acquire compiler trust or project-added protocol assumptions.
-/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.honest_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.honest_complete

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.accepted_digits_eq_splitScalar' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.accepted_digits_eq_splitScalar

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.exists_accepted_iff_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.exists_accepted_iff_exact

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.Accepted.parentBounded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.Accepted.parentBounded

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.Accepted.digits_eq_splitScalar' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits.Accepted.digits_eq_splitScalar
