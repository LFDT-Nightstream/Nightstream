import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegmentBlockRows
import Nightstream.Implementation.NebulaV2.Production.Memory.TranscriptHashFrame
import tests.Axioms.Support

/-! Dependency audit for profile-indexed segment opening. -/

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryOpenSegment.open_exact_for' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryOpenSegment.open_exact_for

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryOpenSegmentSound.rows_sound_for' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryOpenSegmentSound.rows_sound_for

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryOpenSegmentBlockRows.ProfileIndexed.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryOpenSegmentBlockRows.ProfileIndexed.sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryOpenSegmentBlockRows.ProfileIndexed.complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryOpenSegmentBlockRows.ProfileIndexed.complete

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryTranscriptHashFrame.encode_ne_v2' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProductionMemoryTranscriptHashFrame.encode_ne_v2
