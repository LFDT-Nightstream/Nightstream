import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler.CanonicalX
import tests.Axioms.Support

/-! Fail-closed axiom guard for the selected PiDEC canonical-X profile. -/

namespace NightstreamTests.Axioms.PiDecCanonicalXRustConformance

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX.generated_profile_matches_nightstream' does not depend on any axioms -/
#guard_msgs in
#audit_axioms generated_profile_matches_nightstream

end NightstreamTests.Axioms.PiDecCanonicalXRustConformance
