import Nightstream.Implementation.NebulaV2.SegmentCheckedRows
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge.concreteBalanced_iff_mapped' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductBalanceBridge.concreteBalanced_iff_mapped

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge.mapState_oneProductsK' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductBalanceBridge.mapState_oneProductsK

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Invocation.productUpdate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Invocation.productUpdate

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Run.toVerifiedRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.toVerifiedRun

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Run.accumulatedProductsBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.accumulatedProductsBalanced

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Run.accumulatedFromOneBalanced' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.accumulatedFromOneBalanced

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentCheckedRows.Run.exactClaimCount' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheckedRows.Run.exactClaimCount
