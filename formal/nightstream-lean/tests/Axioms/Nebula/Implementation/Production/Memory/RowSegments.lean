import Nightstream.Implementation.Nebula.Production.Memory.RowSegments
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionMemoryRowSegments

open Nightstream.Implementation.Nebula.ProductionMemoryRowSegments

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryRowSegments.BatchRun.toStepRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BatchRun.toStepRun

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryRowSegments.BatchRun.claimsExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BatchRun.claimsExact

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryRowSegments.SegmentRun.exactStepCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.exactStepCount

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryRowSegments.SegmentRun.stepIndexAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.stepIndexAt

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryRowSegments.SegmentRun.segmentBoundsAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.segmentBoundsAt

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryRowSegments.delayedRun_to_rowSegmentChain' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms delayedRun_to_rowSegmentChain

end tests.Axioms.NebulaProductionMemoryRowSegments
