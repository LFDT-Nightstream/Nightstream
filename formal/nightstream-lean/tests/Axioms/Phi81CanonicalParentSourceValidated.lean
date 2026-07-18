import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated
import tests.Axioms.Support

/-! Fail-closed dependency gate for source-derived canonical-parent checks. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated.combinedNorm_of_childHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated.combinedNorm_of_childHolds

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated.piDecAccepted_and_childHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated.piDecAccepted_and_childHolds
