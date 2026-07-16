import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Schema

/-! Generated production SeededPhi81 block 6. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS.SeededPhi81

set_option maxRecDepth 1048576
set_option maxHeartbeats 0

def block6 : SeededPhi81.Block :=
  { rowStart := 1289948
    wordStarts := (List.range 1240).map fun index => 1189349 + index * 122
    wordWidth := 41
    kappa := 2
    messageCols := 942
    outputColumns := (List.range 108).map fun index => 1189241 + index * 1
    superneoTransformedColumns := false
    schedule :=
      { chunkSize := 1024
        seedsByOutput := [[[38, 211, 109, 159, 59, 171, 145, 212, 224, 9, 137, 114, 0, 33, 135, 195, 145, 140, 110, 98, 162, 105, 221, 232, 106, 51, 11, 162, 137, 91, 166, 122]], [[237, 109, 60, 42, 136, 179, 125, 54, 28, 238, 133, 6, 178, 187, 50, 176, 0, 116, 114, 207, 182, 169, 56, 229, 185, 237, 46, 221, 253, 2, 74, 50]]]
        rejectionFuel := rejectionFuel } }

theorem block6_certified :
    block6.Valid ∧ MetadataValid block6 ∧ RowsMapped block6 ∧
      block6.superneoTransformedColumns = false := by native_decide

def certifiedBlock6 : CertifiedBlock := ⟨block6, block6_certified⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
