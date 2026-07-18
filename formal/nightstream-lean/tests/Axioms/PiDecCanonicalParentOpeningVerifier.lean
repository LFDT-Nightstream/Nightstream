import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality
import tests.Axioms.Support

/-! Fail-closed dependency gate for the canonical-parent opening verifier. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier.accepted_iff_parentHolds' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier.accepted_iff_parentHolds

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier.canonicalChildren' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier.canonicalChildren

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.parentHolds_iff_commitment_and_norm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.parentHolds_iff_commitment_and_norm

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.canonicalChildren_of_commitment_and_norm' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.canonicalChildren_of_commitment_and_norm

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.plan_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.plan_exact

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.commitment_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.commitment_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.combinedNorm_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.combinedNorm_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.plan_inclusionMinimalSound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.Minimality.plan_inclusionMinimalSound
