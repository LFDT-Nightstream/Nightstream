import Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalNifsRecipeShape

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape.badRoot_at_production_ops' does not depend on any axioms -/
#guard_msgs in
#audit_axioms NifsRecipeShape.badRoot_at_production_ops

/-- info: 'Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape.collidingIdentity_accepted' does not depend on any axioms -/
#guard_msgs in
#audit_axioms NifsRecipeShape.collidingIdentity_accepted

/-- info: 'Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape.separatedIdentity_not_accepted' does not depend on any axioms -/
#guard_msgs in
#audit_axioms NifsRecipeShape.separatedIdentity_not_accepted

/-- info: 'Nightstream.Implementation.R1CS.Canonical.NifsRecipeShape.unbound_event_is_inhabited' does not depend on any axioms -/
#guard_msgs in
#audit_axioms NifsRecipeShape.unbound_event_is_inhabited

end NightstreamTests.Axioms.CanonicalNifsRecipeShape
