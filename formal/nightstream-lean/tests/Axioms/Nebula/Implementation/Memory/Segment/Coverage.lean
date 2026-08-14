import Nightstream.Implementation.Nebula.Memory.Segment.Coverage
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemoryCoverage.writesCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.writesCover

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemoryCoverage.readsCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.readsCover

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemoryCoverage.ordered' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.ordered

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemoryCoverage.covers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.covers

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemoryCoverage.orderedActiveToClosed' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemoryCoverage.orderedActiveToClosed
