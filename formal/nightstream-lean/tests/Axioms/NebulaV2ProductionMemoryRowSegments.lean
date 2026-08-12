import Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionMemoryRowSegments

open Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.BatchRun.toStepRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BatchRun.toStepRun

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.BatchRun.claimsExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BatchRun.claimsExact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.SegmentRun.exactStepCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.exactStepCount

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.SegmentRun.stepIndexAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.stepIndexAt

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.SegmentRun.segmentBoundsAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.segmentBoundsAt

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.delayedRun_to_rowSegmentChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms delayedRun_to_rowSegmentChain

end tests.Axioms.NebulaV2ProductionMemoryRowSegments
