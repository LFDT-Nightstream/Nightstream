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
    { bodySourceStart := 46055, overlaySourceStart := 1, outerCount := 1, bodySourceStride := 41, overlaySourceStride := 41, fieldCount := 41, bodyFinalStart := 1059804, overlayFinalStart := 111, finalOuterStride := 41, finalFieldStride := 1, width := 1, radix := 2 }
  , { bodySourceStart := 46096, overlaySourceStart := 42, outerCount := 810, bodySourceStride := 122, overlaySourceStride := 41, fieldCount := 41, bodyFinalStart := 19332, overlayFinalStart := 152, finalOuterStride := 41, finalFieldStride := 1, width := 1, radix := 2 }
  , { bodySourceStart := 144918, overlaySourceStart := 33252, outerCount := 1, bodySourceStride := 108, overlaySourceStride := 108, fieldCount := 108, bodyFinalStart := 1076091, overlayFinalStart := 33362, finalOuterStride := 2484, finalFieldStride := 23, width := 23, radix := 7 }
  ]

def audit : RawAudit where
schemaVersion := 1
familyCount := 110
parityCount := 2
publicOutputCount := 640
bodyFinalColumns := 2521314
overlayFinalColumns := 35856
linkCountPerFamily := 33359
totalLinkCount := 3669490
phaseKinds := [10, 11]
runs := runs

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink
