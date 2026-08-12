import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegmentBlockRows
import Nightstream.Implementation.NebulaV2.Production.Memory.TranscriptHashFrame

/-!
Regression surface for verifier-profile binding in the segment-open
challenge path.
-/

set_option autoImplicit false

namespace NightstreamTests.NebulaV2ProfileIndexedMemoryOpening

open Nightstream.Implementation.NebulaV2

#check MemoryOpenSegment.open_exact_for
#check MemoryOpenSegmentSound.rows_sound_for
#check MemoryOpenSegmentBlockRows.ProfileIndexed.sound
#check MemoryOpenSegmentBlockRows.ProfileIndexed.complete
#check ProductionMemoryTranscriptHashFrame.encode_ne_v2

end NightstreamTests.NebulaV2ProfileIndexedMemoryOpening
