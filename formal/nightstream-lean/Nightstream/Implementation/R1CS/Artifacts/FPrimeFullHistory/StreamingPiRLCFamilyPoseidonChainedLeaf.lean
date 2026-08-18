import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafRows0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafRows1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafImages

/-!
Artifact facade for one chained production PiRLC Poseidon2 selective-row leaf.

Owns the ordered composition of the two exact Rust-emitted row shards. The
source steps are the shared direct-leaf steps after the Rust-checked input-role
renaming.

Does not own semantic soundness, replay-batch coverage, or recursive
orchestration.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeaf

open StreamingPiRLCFamilyPoseidonLeafSchema

def rawRows : List RawRow :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf.rawRows0 ++
    Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf.rawRows1

def rawImages : List RawSourceImage :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf.rawImages

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeaf
