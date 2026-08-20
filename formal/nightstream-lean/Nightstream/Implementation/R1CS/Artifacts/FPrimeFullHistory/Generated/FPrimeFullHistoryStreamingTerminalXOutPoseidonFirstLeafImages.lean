import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

/-! Generated exact external source images for the first terminal XOut Poseidon2 leaf. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema

def rawImage0 : RawSourceImage where
  lane := 0
  port := { explicit := [], geometric := [] }

def rawImage1 : RawSourceImage where
  lane := 1
  port := { explicit := [{ column := .one, coefficient := 1313210370 }], geometric := [] }

def rawImage2 : RawSourceImage where
  lane := 2
  port := { explicit := [], geometric := [{ slot := .externalA 0, initial := 1, ratio := 3 }] }

def rawImage3 : RawSourceImage where
  lane := 3
  port := { explicit := [], geometric := [{ slot := .externalA 1, initial := 1, ratio := 3 }] }

def rawImage4 : RawSourceImage where
  lane := 4
  port := { explicit := [], geometric := [{ slot := .externalA 2, initial := 1, ratio := 3 }] }

def rawImages : List RawSourceImage := [
  rawImage0
, rawImage1
, rawImage2
, rawImage3
, rawImage4
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf
