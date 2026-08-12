import Nightstream.Implementation.NebulaV2.Production.Memory.SegmentSoundness
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionMemorySegmentSoundness

open Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness.SegmentRun.covers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.covers

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness.SegmentRun.fingerprintAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.fingerprintAccepted

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness.SegmentRun.balanceOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.balanceOrEvaluationFailure

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness.SegmentRun.executesOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.executesOrEvaluationFailure

end tests.Axioms.NebulaV2ProductionMemorySegmentSoundness
