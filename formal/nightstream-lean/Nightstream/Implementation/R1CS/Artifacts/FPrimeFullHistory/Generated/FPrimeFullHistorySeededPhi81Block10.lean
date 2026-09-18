import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Schema

/-! Generated production SeededPhi81 block 10. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS.SeededPhi81

set_option maxRecDepth 1048576
set_option maxHeartbeats 0

def block10 : SeededPhi81.Block :=
  { rowStart := 1324042
    wordStarts := (List.range 1650).map fun index => 1163100 + index * 122
    wordWidth := 41
    kappa := 2
    messageCols := 1253
    outputColumns := (List.range 108).map fun index => 1162992 + index * 1
    superneoTransformedColumns := false
    schedule :=
      { chunkSize := 1253
        seedsByOutput := [[[86, 42, 148, 207, 25, 79, 16, 47, 196, 116, 40, 120, 217, 88, 154, 122, 30, 23, 140, 56, 187, 143, 93, 162, 220, 95, 239, 73, 250, 113, 225, 105]], [[141, 244, 128, 208, 147, 79, 116, 194, 51, 31, 25, 137, 234, 142, 110, 234, 203, 26, 211, 3, 150, 158, 227, 214, 198, 230, 165, 130, 138, 231, 94, 41]]]
        rejectionFuel := rejectionFuel } }

theorem block10_certified :
    block10.Valid ∧ MetadataValid block10 ∧ RowsMapped block10 ∧
      block10.superneoTransformedColumns = false := by native_decide

def certifiedBlock10 : CertifiedBlock := ⟨block10, block10_certified⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
