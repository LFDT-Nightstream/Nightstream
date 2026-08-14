import Nightstream.Implementation.Nebula.Production.Artifact.SemanticAuthority

/-! Regression surface for the verifier-owned semantic authority. -/

set_option autoImplicit false

namespace tests.NebulaProductionSemanticAuthority

open Nightstream.Implementation.Nebula.ProductionSemanticAuthority

#check Artifact
#check MatchesStatement
#check MatchesStatement.identityDigestsExact
#check RejectMachine
#check ReturnMachine
#check equal_identifiers_do_not_bind_machine

end tests.NebulaProductionSemanticAuthority
