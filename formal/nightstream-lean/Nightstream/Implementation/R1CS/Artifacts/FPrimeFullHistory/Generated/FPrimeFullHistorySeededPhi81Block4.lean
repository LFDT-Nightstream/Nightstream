import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySeededPhi81Schema

/-! Generated production SeededPhi81 block 4. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81

open Nightstream.Implementation.R1CS.SeededPhi81

set_option maxRecDepth 1048576
set_option maxHeartbeats 0

def block4 : SeededPhi81.Block :=
  { rowStart := 833813
    wordStarts := (List.range 3616).map fun index => 400783 + index * 122
    wordWidth := 41
    kappa := 2
    messageCols := 2746
    outputColumns := (List.range 108).map fun index => 400675 + index * 1
    superneoTransformedColumns := false
    schedule :=
      { chunkSize := 2746
        seedsByOutput := [[[155, 172, 141, 237, 241, 37, 62, 48, 231, 186, 174, 92, 253, 182, 136, 23, 144, 88, 194, 62, 72, 86, 244, 81, 95, 5, 239, 244, 206, 198, 155, 18]], [[40, 94, 19, 184, 73, 118, 112, 98, 82, 191, 229, 41, 224, 250, 187, 145, 78, 35, 9, 84, 198, 101, 115, 56, 83, 48, 154, 186, 218, 107, 125, 19]]]
        rejectionFuel := rejectionFuel } }

theorem block4_certified :
    block4.Valid ∧ MetadataValid block4 ∧ RowsMapped block4 ∧
      block4.superneoTransformedColumns = false := by native_decide

def certifiedBlock4 : CertifiedBlock := ⟨block4, block4_certified⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySeededPhi81
