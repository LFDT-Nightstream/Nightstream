import Nightstream.Implementation.NebulaV2.ProductionSemanticAuthority
import tests.Axioms.Support

/-! Dependency audit for the verifier-owned semantic authority. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSemanticAuthority.MatchesStatement.identityDigestsExact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSemanticAuthority.MatchesStatement.identityDigestsExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSemanticAuthority.equal_identifiers_do_not_bind_machine' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSemanticAuthority.equal_identifiers_do_not_bind_machine
