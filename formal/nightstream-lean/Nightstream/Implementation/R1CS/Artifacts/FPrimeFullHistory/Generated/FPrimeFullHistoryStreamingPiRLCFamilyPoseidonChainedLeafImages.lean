import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

/-! Generated file: exact final low-norm images of the four prior-output
lanes consumed by one chained production PiRLC Poseidon2 leaf.

Does not own: source authority, row satisfaction, replay-batch coverage,
recursive orchestration, or permission to remove constraints.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

def rawImage0 : RawSourceImage where
  lane := 0
  port := { explicit := [], geometric := [{ slot := .previousLocal 85, initial := 2, ratio := 3 }, { slot := .previousLocal 84, initial := 2, ratio := 3 }, { slot := .previousLocal 83, initial := 6, ratio := 3 }, { slot := .previousLocal 82, initial := 4, ratio := 3 }, { slot := .previousLocal 81, initial := 1, ratio := 3 }, { slot := .previousLocal 80, initial := 1, ratio := 3 }, { slot := .previousLocal 79, initial := 3, ratio := 3 }, { slot := .previousLocal 78, initial := 2, ratio := 3 }] }

def rawImage1 : RawSourceImage where
  lane := 1
  port := { explicit := [], geometric := [{ slot := .previousLocal 85, initial := 2, ratio := 3 }, { slot := .previousLocal 84, initial := 6, ratio := 3 }, { slot := .previousLocal 83, initial := 4, ratio := 3 }, { slot := .previousLocal 82, initial := 2, ratio := 3 }, { slot := .previousLocal 81, initial := 1, ratio := 3 }, { slot := .previousLocal 80, initial := 3, ratio := 3 }, { slot := .previousLocal 79, initial := 2, ratio := 3 }, { slot := .previousLocal 78, initial := 1, ratio := 3 }] }

def rawImage2 : RawSourceImage where
  lane := 2
  port := { explicit := [], geometric := [{ slot := .previousLocal 85, initial := 6, ratio := 3 }, { slot := .previousLocal 84, initial := 4, ratio := 3 }, { slot := .previousLocal 83, initial := 2, ratio := 3 }, { slot := .previousLocal 82, initial := 2, ratio := 3 }, { slot := .previousLocal 81, initial := 3, ratio := 3 }, { slot := .previousLocal 80, initial := 2, ratio := 3 }, { slot := .previousLocal 79, initial := 1, ratio := 3 }, { slot := .previousLocal 78, initial := 1, ratio := 3 }] }

def rawImage3 : RawSourceImage where
  lane := 3
  port := { explicit := [], geometric := [{ slot := .previousLocal 85, initial := 4, ratio := 3 }, { slot := .previousLocal 84, initial := 2, ratio := 3 }, { slot := .previousLocal 83, initial := 2, ratio := 3 }, { slot := .previousLocal 82, initial := 6, ratio := 3 }, { slot := .previousLocal 81, initial := 2, ratio := 3 }, { slot := .previousLocal 80, initial := 1, ratio := 3 }, { slot := .previousLocal 79, initial := 1, ratio := 3 }, { slot := .previousLocal 78, initial := 3, ratio := 3 }] }

def rawImages : List RawSourceImage := [rawImage0, rawImage1, rawImage2, rawImage3]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeaf
