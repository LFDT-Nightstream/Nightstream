import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4
import tests.Axioms.Support

/-!
Fail-closed axiom guards for the model-level radix-four width candidate.

These facts cover parameter arithmetic only. They do not certify a radix-four
PiDEC circuit or select this candidate for production.
-/

namespace NightstreamTests.Axioms.SuperNeoRadix4Candidate

open NightstreamTests.Axioms
open Nightstream.SuperNeo.Concrete.Radix4Candidate

/-- info: 'Nightstream.SuperNeo.Concrete.Radix4Candidate.parameter_values' does not depend on any axioms -/
#guard_msgs in
#audit_axioms parameter_values

/-- info: 'Nightstream.SuperNeo.Concrete.Radix4Candidate.maxFresh_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms maxFresh_exact

/-- info: 'Nightstream.SuperNeo.Concrete.Radix4Candidate.oneFresh_rlc_bound' does not depend on any axioms -/
#guard_msgs in
#audit_axioms oneFresh_rlc_bound

/-- info: 'Nightstream.SuperNeo.Concrete.Radix4Candidate.msisNormBound_eq_production' does not depend on any axioms -/
#guard_msgs in
#audit_axioms msisNormBound_eq_production

/-- info: 'Nightstream.SuperNeo.Concrete.Radix4Candidate.degreeEight_verifierDegree_eq_production' does not depend on any axioms -/
#guard_msgs in
#audit_axioms degreeEight_verifierDegree_eq_production

/-- info: 'Nightstream.SuperNeo.Concrete.Radix4Candidate.runningSourceCount_halved' does not depend on any axioms -/
#guard_msgs in
#audit_axioms runningSourceCount_halved

/-! The candidate split and norm laws use only Lean's standard proposition,
choice, and quotient soundness axioms. They remain model-level until a circuit
refinement theorem consumes them. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.split_recompose' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.split_recompose

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recomposeAssignment_eq_weighted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recomposeAssignment_eq_weighted

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recomposeScalar_seven' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recomposeScalar_seven

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.splitScalar_eq_signed_of_recompose' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.splitScalar_eq_signed_of_recompose

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.split_norm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.split_norm

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recompose_norm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.recompose_norm

end NightstreamTests.Axioms.SuperNeoRadix4Candidate
