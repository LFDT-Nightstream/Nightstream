import Nightstream.Implementation.NebulaV2.SegmentMemorySoundness
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemorySoundness.fingerprintCheck' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.fingerprintCheck

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemorySoundness.fingerprintAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.fingerprintAccepted

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemorySoundness.balanceOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.balanceOrEvaluationFailure

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemorySoundness.executesOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.executesOrEvaluationFailure

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentMemorySoundness.globallyOpenedExecutesOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.globallyOpenedExecutesOrEvaluationFailure
