import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema

/-! Generated file: compact receipt for the exhaustive normalized production
PiRLC family-overlay retained-row scan.

Owns: dimensions, selector and retained-row affine geometry, source and final
slot starts, low-norm widths and radices, the exact verifier seed chunks, and
compact-block and explicit nonzero censuses observed by the Rust scan.

Does not own: semantic truth, matrix authority in Lean, assignment values,
body-to-overlay links, selector authority, recursive orchestration, or
lifecycle soundness. Lean recomputes the arithmetic properties of this inert
receipt.

Emits constraints: no. Rust checks all 110 source blocks, all 110 final
blocks, and every retained explicit final row before it renders this data.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetainedSchema

def audit : RawAudit where
  schemaVersion := 1
  familyCount := 110
  sourceRows := 108
  sourceColumns := 37788
  finalRows := 12001
  finalColumns := 42228
  selectorStart := 1
  selectorCount := 110
  retainedStart := 111
  retainedStride := 108
  sourceStarts := [1, 42, 37680]
  finalStarts := [111, 152, 37790]
  widths := [1, 41]
  radices := [2, 3]
  chunkSize := 32768
  chunkSeedsByRow := [[[63, 30, 174, 162, 0, 236, 164, 15, 112, 184, 219, 193, 87, 199, 177, 192, 164, 187, 240, 253, 61, 134, 245, 41, 164, 242, 123, 75, 30, 91, 95, 22], [170, 30, 21, 88, 158, 179, 136, 195, 74, 178, 126, 216, 115, 81, 206, 108, 53, 169, 80, 22, 36, 74, 24, 88, 4, 121, 41, 241, 174, 49, 67, 143], [92, 129, 108, 73, 233, 97, 159, 86, 136, 140, 248, 19, 39, 162, 210, 251, 113, 68, 174, 135, 211, 51, 125, 118, 184, 143, 8, 187, 129, 45, 42, 79]], [[197, 12, 44, 8, 53, 23, 209, 249, 57, 95, 97, 47, 228, 178, 54, 33, 145, 237, 90, 110, 180, 37, 85, 68, 117, 62, 131, 251, 117, 55, 252, 109], [216, 235, 55, 41, 102, 238, 74, 133, 240, 135, 253, 12, 109, 162, 110, 108, 39, 99, 165, 9, 133, 213, 34, 164, 22, 107, 28, 32, 203, 31, 248, 32], [245, 70, 118, 102, 57, 150, 196, 122, 158, 115, 60, 90, 255, 202, 21, 46, 190, 177, 86, 202, 42, 68, 7, 150, 50, 108, 114, 224, 14, 51, 218, 237]]]
  sourceExplicitNnz := [0, 11880, 11880]
  finalBlockCounts := [0, 0, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  finalExplicitPortNnz := [0, 11880, 0, 11880, 487080, 0, 0, 0, 0, 0, 0, 0, 0]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyOverlayRetained
