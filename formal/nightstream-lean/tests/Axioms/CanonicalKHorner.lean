import Nightstream.Implementation.R1CS.Canonical.KHorner
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKHorner

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHorner.hornerRows_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHorner.hornerRows_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHorner.hornerRows_length_of_degree' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHorner.hornerRows_length_of_degree

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHorner.hornerRows_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHorner.hornerRows_sound

end NightstreamTests.Axioms.CanonicalKHorner
