import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafRows0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafRows1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafSteps

/-!
Artifact facade for one partial-start production PiRLC Poseidon2 leaf.

Owns the ordered composition of the exact generated step and row shards. It
does not own operand-order equivalence, semantic soundness, replay coverage,
or recursive orchestration.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonPartialLeaf

open StreamingPiRLCFamilyPoseidonLeafSchema

def schemaVersion : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.schemaVersion
def sourceWidth : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.sourceWidth
def slotWidth : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.slotWidth
def externalLaneCount : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.externalLaneCount
def rowCount : Nat :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.rowCount
def rawSteps : List RawStep :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.rawSteps
def rawRows : List RawRow :=
  Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.rawRows0 ++
    Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeaf.rawRows1

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonPartialLeaf
