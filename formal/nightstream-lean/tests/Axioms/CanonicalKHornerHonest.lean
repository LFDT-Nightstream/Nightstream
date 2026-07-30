import Nightstream.Implementation.R1CS.Canonical.KHornerHonest
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKHornerHonest

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerHonest.fresh_of_belowBase' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerHonest.fresh_of_belowBase

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerHonest.later_frame_fresh' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerHonest.later_frame_fresh

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerHonest.suffix_fresh' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerHonest.suffix_fresh

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerHonest.hornerWitness_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerHonest.hornerWitness_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KHornerHonest.hornerWitness_off_block' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KHornerHonest.hornerWitness_off_block

end NightstreamTests.Axioms.CanonicalKHornerHonest
