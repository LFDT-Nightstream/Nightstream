import Nightstream.Implementation.R1CS.Canonical.KHornerSupport
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKHornerSupport

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerSupport.satisfies_extend' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerSupport.satisfies_extend

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerSupport.hornerCarried_mentions' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerSupport.hornerCarried_mentions

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerSupport.hornerRows_mentions' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerSupport.hornerRows_mentions

end NightstreamTests.Axioms.CanonicalKHornerSupport
