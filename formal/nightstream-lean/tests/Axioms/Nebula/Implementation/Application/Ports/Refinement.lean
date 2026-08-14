import Nightstream.Implementation.Nebula.Application.Ports.Refinement
import Nightstream.Implementation.Nebula.Memory.Segment.CheckedRows
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.ApplicationPortRefinement.checkedStep_physicalPort' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.checkedStep_physicalPort

/-- info: 'Nightstream.Implementation.Nebula.ApplicationPortRefinement.accesses_length_eq_claimActiveCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.accesses_length_eq_claimActiveCount

/-- info: 'Nightstream.Implementation.Nebula.ApplicationPortRefinement.ordered' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.ordered

/-- info: 'Nightstream.Implementation.Nebula.ApplicationPortRefinement.readRecordMultiset_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.readRecordMultiset_eq

/-- info: 'Nightstream.Implementation.Nebula.ApplicationPortRefinement.writeRecordMultiset_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ApplicationPortRefinement.writeRecordMultiset_eq

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Invocation.applicationAccessesOrdered' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.applicationAccessesOrdered

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Invocation.chunk_reads_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.chunk_reads_eq

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Invocation.chunk_writes_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.chunk_writes_eq
