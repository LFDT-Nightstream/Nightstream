import Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81Schema

/-! Generated production SeededPhi81 block 17. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS.SeededPhi81

set_option maxRecDepth 1048576
set_option maxHeartbeats 0

def block17 : SeededPhi81.Block :=
  { rowStart := 3541829
    wordStarts := (List.range 108).map fun index => 3181507 + index * 122
    wordWidth := 41
    kappa := 1
    messageCols := 82
    outputColumns := (List.range 54).map fun index => 3181453 + index * 1
    superneoTransformedColumns := false
    schedule :=
      { chunkSize := 1024
        seedsByOutput := [[[222, 97, 160, 75, 169, 146, 205, 28, 66, 7, 37, 46, 38, 226, 240, 160, 130, 181, 109, 118, 6, 248, 19, 168, 202, 255, 83, 20, 122, 228, 97, 38]]]
        rejectionFuel := rejectionFuel } }

theorem block17_certified :
    block17.Valid ∧ MetadataValid block17 ∧ RowsMapped block17 ∧
      block17.superneoTransformedColumns = false := by native_decide

def certifiedBlock17 : CertifiedBlock := ⟨block17, block17_certified⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
