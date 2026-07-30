import Nightstream.Implementation.R1CS.Canonical.KPairLaws
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKPairLaws

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.mulPair_comm' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPairLaws.mulPair_comm

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.mulPair_zero_right' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPairLaws.mulPair_zero_right

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.mulPair_addPair_distrib_right' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.mulPair_addPair_distrib_right

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.mulPair_assoc' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.mulPair_assoc

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.mulPair_one_left' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPairLaws.mulPair_one_left

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.powPair_add' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.powPair_add

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.mulPair_add_self' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.mulPair_add_self

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.reduction_single_fold' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.reduction_single_fold

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.root_shifted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.root_shifted

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.powPair_eightyOne' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.powPair_eightyOne

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.subPair_canonical' does not depend on any axioms -/
#guard_msgs in
#audit_axioms KPairLaws.subPair_canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.complement_add' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.complement_add

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.addPair_subPair' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KPairLaws.addPair_subPair

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KPairLaws.subPair_zero_right' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KPairLaws.subPair_zero_right

end NightstreamTests.Axioms.CanonicalKPairLaws
