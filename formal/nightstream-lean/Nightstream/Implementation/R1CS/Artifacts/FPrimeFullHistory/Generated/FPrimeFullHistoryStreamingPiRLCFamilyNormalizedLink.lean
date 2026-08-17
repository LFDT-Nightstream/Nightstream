import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLinkSchema

/-! Generated file: compact receipt for the normalized production PiRLC
body-overlay link audit.

Owns: the public-prefix shift, both final column bounds, parity kind codes,
the three source-field runs, and the exact final low-norm slots and radices.

Does not own: semantic truth, selector authority, shifted-ternary
canonicality, row satisfaction, recursive orchestration, or lifecycle
soundness. Lean checks the arithmetic properties of this inert receipt.

Emits constraints: no. Rust checks both parity body maps against the prepared
production layout. The separate overlay receipt checks all 110 overlay maps.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLinkSchema

def runs : List RawRun :=
  [
    { bodySourceStart := 52103, overlaySourceStart := 1, outerCount := 1, bodySourceStride := 41, overlaySourceStride := 41, fieldCount := 41, bodyFinalStart := 2110644, overlayFinalStart := 111, finalOuterStride := 41, finalFieldStride := 1, width := 1, radix := 2 }
  , { bodySourceStart := 52144, overlaySourceStart := 42, outerCount := 918, bodySourceStride := 122, overlaySourceStride := 41, fieldCount := 41, bodyFinalStart := 38340, overlayFinalStart := 152, finalOuterStride := 41, finalFieldStride := 1, width := 1, radix := 2 }
  , { bodySourceStart := 164142, overlaySourceStart := 37680, outerCount := 1, bodySourceStride := 108, overlaySourceStride := 108, fieldCount := 108, bodyFinalStart := 2129127, overlayFinalStart := 37790, finalOuterStride := 4428, finalFieldStride := 41, width := 41, radix := 3 }
  ]

def audit : RawAudit where
schemaVersion := 1
familyCount := 110
parityCount := 2
publicOutputCount := 640
bodyFinalColumns := 8858862
overlayFinalColumns := 42228
linkCountPerFamily := 37787
totalLinkCount := 4156570
phaseKinds := [10, 11]
runs := runs

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink
