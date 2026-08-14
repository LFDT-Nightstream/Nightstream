import Nightstream.Implementation.Nebula.Memory.Segment.Soundness
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemorySoundness.fingerprintCheck' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.fingerprintCheck

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemorySoundness.fingerprintAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.fingerprintAccepted

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemorySoundness.balanceOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.balanceOrEvaluationFailure

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemorySoundness.executesOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.executesOrEvaluationFailure

/-- info: 'Nightstream.Implementation.Nebula.SegmentMemorySoundness.globallyOpenedExecutesOrEvaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentMemorySoundness.globallyOpenedExecutesOrEvaluationFailure
