import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
import tests.Axioms.Support

/-! Fail-closed axiom guard for the bounded PiDEC source profile. -/

namespace NightstreamTests.Axioms.PiDecSourceProfile

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec

/-- info: 'Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.profile_matches_nightstream' does not depend on any axioms -/
#guard_msgs in
#audit_axioms profile_matches_nightstream

end NightstreamTests.Axioms.PiDecSourceProfile
