import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafRows0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafRows1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafSteps

/-!
Artifact facade for one relative production PiRLC Poseidon2 selective-row leaf.

Owns the ordered composition of the generated step and row shards. It does
not own semantic soundness, replay-batch coverage, or recursive orchestration.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeaf

open StreamingPiRLCFamilyPoseidonLeafSchema

def schemaVersion : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.schemaVersion
def sourceWidth : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.sourceWidth
def slotWidth : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.slotWidth
def externalLaneCount : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.externalLaneCount
def rowCount : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.rowCount
def rawSteps : List RawStep :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.rawSteps
def rawRows : List RawRow :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.rawRows0 ++
    Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeaf.rawRows1

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeaf
