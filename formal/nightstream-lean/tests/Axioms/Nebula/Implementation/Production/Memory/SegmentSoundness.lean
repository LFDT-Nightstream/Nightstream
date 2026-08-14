import Nightstream.Implementation.Nebula.Production.Memory.SegmentSoundness
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionMemorySegmentSoundness

open Nightstream.Implementation.Nebula.ProductionMemorySegmentSoundness

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySegmentSoundness.SegmentRun.covers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.covers

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySegmentSoundness.SegmentRun.fingerprintAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.fingerprintAccepted

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySegmentSoundness.SegmentRun.balanceOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.balanceOrEvaluationFailure

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySegmentSoundness.SegmentRun.executesOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.executesOrEvaluationFailure

end tests.Axioms.NebulaProductionMemorySegmentSoundness
