import Nightstream.Implementation.Nebula.Memory.Transition.OpenSegmentBlockRows
import Nightstream.Implementation.Nebula.Production.Memory.TranscriptHashFrame

/-!
Regression surface for verifier-profile binding in the segment-open
challenge path.
-/

set_option autoImplicit false

namespace NightstreamTests.NebulaProfileIndexedMemoryOpening

open Nightstream.Implementation.Nebula

#check MemoryOpenSegment.open_exact_for
#check MemoryOpenSegmentSound.rows_sound_for
#check MemoryOpenSegmentBlockRows.ProfileIndexed.sound
#check MemoryOpenSegmentBlockRows.ProfileIndexed.complete
#check ProductionMemoryTranscriptHashFrame.encode_ne_v2

end NightstreamTests.NebulaProfileIndexedMemoryOpening
