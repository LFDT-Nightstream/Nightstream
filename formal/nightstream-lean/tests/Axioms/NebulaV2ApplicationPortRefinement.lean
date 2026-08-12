import Nightstream.Implementation.NebulaV2.ApplicationPortRefinement
import Nightstream.Implementation.NebulaV2.SegmentCheckedRows
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.ApplicationPortRefinement.checkedStep_physicalPort' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.checkedStep_physicalPort

/-- info: 'Nightstream.Implementation.NebulaV2.ApplicationPortRefinement.accesses_length_eq_claimActiveCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.accesses_length_eq_claimActiveCount

/-- info: 'Nightstream.Implementation.NebulaV2.ApplicationPortRefinement.ordered' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.ordered

/-- info: 'Nightstream.Implementation.NebulaV2.ApplicationPortRefinement.readRecordMultiset_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.readRecordMultiset_eq

/-- info: 'Nightstream.Implementation.NebulaV2.ApplicationPortRefinement.writeRecordMultiset_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.writeRecordMultiset_eq

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Invocation.applicationAccessesOrdered' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.applicationAccessesOrdered

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Invocation.chunk_reads_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.chunk_reads_eq

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Invocation.chunk_writes_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.chunk_writes_eq
