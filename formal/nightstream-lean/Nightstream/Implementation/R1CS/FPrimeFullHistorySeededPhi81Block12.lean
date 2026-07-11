import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Schema

/-! Generated production SeededPhi81 block 12. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS.SeededPhi81

set_option maxRecDepth 1048576
set_option maxHeartbeats 0

def block12 : SeededPhi81.Block :=
  { rowStart := 2633248
    wordStarts := (List.range 7883).map fun index => 1486491 + index * 122
    wordWidth := 41
    kappa := 2
    messageCols := 5986
    outputColumns := (List.range 108).map fun index => 1486383 + index * 1
    superneoTransformedColumns := false
    schedule :=
      { chunkSize := 5986
        seedsByOutput := [[[216, 10, 232, 51, 138, 24, 205, 89, 240, 118, 170, 228, 175, 184, 181, 191, 147, 71, 39, 190, 69, 239, 98, 77, 67, 15, 47, 136, 189, 79, 134, 243]], [[227, 116, 17, 69, 75, 42, 154, 78, 131, 122, 13, 46, 159, 245, 178, 93, 250, 165, 184, 133, 218, 223, 237, 244, 211, 177, 48, 249, 175, 40, 115, 61]]]
        rejectionFuel := rejectionFuel } }

theorem block12_certified :
    block12.Valid ∧ MetadataValid block12 ∧ RowsMapped block12 ∧
      block12.superneoTransformedColumns = false := by native_decide

def certifiedBlock12 : CertifiedBlock := ⟨block12, block12_certified⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
