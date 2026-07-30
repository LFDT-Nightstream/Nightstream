import Nightstream.Implementation.R1CS.Canonical.KFrames
import tests.Axioms.Support

namespace NightstreamTests.Axioms.CanonicalKFrames

open NightstreamTests.Axioms
open Nightstream.Implementation.R1CS.Canonical

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFrames.frameColumn_step_disjoint' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KFrames.frameColumn_step_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFrames.frameColumn_slot_disjoint' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms KFrames.frameColumn_slot_disjoint

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFrames.frameColumns_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms KFrames.frameColumns_length

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFrames.frameColumns_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFrames.frameColumns_nodup

/-- info: 'Nightstream.Implementation.R1CS.Canonical.KFrames.frameColumns_mem_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms KFrames.frameColumns_mem_iff

end NightstreamTests.Axioms.CanonicalKFrames
