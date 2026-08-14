import Nightstream.Implementation.Nebula.Memory.Segment.CheckedRows
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.MemoryProductBalanceBridge.concreteBalanced_iff_mapped' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductBalanceBridge.concreteBalanced_iff_mapped

/-- info: 'Nightstream.Implementation.Nebula.MemoryProductBalanceBridge.mapState_oneProductsK' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductBalanceBridge.mapState_oneProductsK

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Invocation.productUpdate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.productUpdate

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Run.toVerifiedRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.toVerifiedRun

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Run.accumulatedProductsBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.accumulatedProductsBalanced

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Run.accumulatedFromOneBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.accumulatedFromOneBalanced

/-- info: 'Nightstream.Implementation.Nebula.SegmentCheckedRows.Run.exactClaimCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.exactClaimCount
