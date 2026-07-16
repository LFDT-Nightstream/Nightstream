import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Schema

/-! Generated production SeededPhi81 block 11. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS.SeededPhi81

set_option maxRecDepth 1048576
set_option maxHeartbeats 0

def block11 : SeededPhi81.Block :=
  { rowStart := 3316399
    wordStarts := (List.range 108).map fun index => 3143329 + index * 122
    wordWidth := 41
    kappa := 1
    messageCols := 82
    outputColumns := (List.range 54).map fun index => 3143275 + index * 1
    superneoTransformedColumns := false
    schedule :=
      { chunkSize := 1024
        seedsByOutput := [[[222, 97, 160, 75, 169, 146, 205, 28, 66, 7, 37, 46, 38, 226, 240, 160, 130, 181, 109, 118, 6, 248, 19, 168, 202, 255, 83, 20, 122, 228, 97, 38]]]
        rejectionFuel := rejectionFuel } }

theorem block11_certified :
    block11.Valid ∧ MetadataValid block11 ∧ RowsMapped block11 ∧
      block11.superneoTransformedColumns = false := by native_decide

def certifiedBlock11 : CertifiedBlock := ⟨block11, block11_certified⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
