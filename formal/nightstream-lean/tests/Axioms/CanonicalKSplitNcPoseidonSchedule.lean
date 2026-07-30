import Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKSplitNcPoseidonSchedule

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.Tag.code_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.Tag.code_injective

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.ncMessageFields_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.ncMessageFields_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.squeezeManyK_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.squeezeManyK_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.schedule_bindStatement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.schedule_bindStatement

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.schedule_deriveCore' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.schedule_deriveCore

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.schedule_absorbFeRound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.schedule_absorbFeRound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.schedule_absorbNcRound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.schedule_absorbNcRound

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KSplitNcPoseidonSchedule.schedule_absorbOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KSplitNcPoseidonSchedule.schedule_absorbOutput

end NightstreamTests.Axioms.CanonicalKSplitNcPoseidonSchedule
