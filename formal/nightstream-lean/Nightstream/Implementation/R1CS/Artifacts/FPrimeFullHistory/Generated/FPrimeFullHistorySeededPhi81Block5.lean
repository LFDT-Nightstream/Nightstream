import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Schema

/-! Generated production SeededPhi81 block 5. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS.SeededPhi81

set_option maxRecDepth 1048576
set_option maxHeartbeats 0

def block5 : SeededPhi81.Block :=
  { rowStart := 823784
    wordStarts := (List.range 108).map fun index => 826948 + index * 122
    wordWidth := 41
    kappa := 1
    messageCols := 82
    outputColumns := (List.range 54).map fun index => 826894 + index * 1
    superneoTransformedColumns := false
    schedule :=
      { chunkSize := 1024
        seedsByOutput := [[[222, 97, 160, 75, 169, 146, 205, 28, 66, 7, 37, 46, 38, 226, 240, 160, 130, 181, 109, 118, 6, 248, 19, 168, 202, 255, 83, 20, 122, 228, 97, 38]]]
        rejectionFuel := rejectionFuel } }

theorem block5_certified :
    block5.Valid ∧ MetadataValid block5 ∧ RowsMapped block5 ∧
      block5.superneoTransformedColumns = false := by native_decide

def certifiedBlock5 : CertifiedBlock := ⟨block5, block5_certified⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
