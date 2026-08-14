import Nightstream.Implementation.Nebula.Memory.Transition.OpenSegmentSound

/-! Regression surface for lossless challenge-authority placement. -/

namespace tests.NebulaMemoryAuthorityInjectivity

open Nightstream.Implementation.Nebula

#check MemoryOpenSegment.Authority.digestFields_injective
#check MemoryOpenSegmentSound.AuthorityPlaced.unique

end tests.NebulaMemoryAuthorityInjectivity
