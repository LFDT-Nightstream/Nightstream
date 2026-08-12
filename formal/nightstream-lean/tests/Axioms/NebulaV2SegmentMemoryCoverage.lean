import Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage.writesCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.writesCover

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage.readsCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.readsCover

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage.ordered' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.ordered

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage.covers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.covers

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemoryCoverage.orderedActiveToClosed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.orderedActiveToClosed
