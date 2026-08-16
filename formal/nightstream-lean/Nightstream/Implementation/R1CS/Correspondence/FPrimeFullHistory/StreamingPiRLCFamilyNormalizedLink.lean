import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLink
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetained

/-!
Contract: independent validation of the compact normalized production PiRLC
body-overlay link receipt.

Assurance tier: Rust-conformant for property
`FPRIME-PIRLC-FAMILY-NORMALIZED-LINK-SLOT-IMAGE`.

Owns the 640-column public-prefix shift, both parity kind codes, all three
source-field runs, the body and overlay final slots read by the link compiler,
and the exact 33,359-per-family and 3,669,490-total censuses.

Does not own assignment values, equality-row acceptance, selector authority,
shifted-ternary canonicality, recursive orchestration, lifecycle soundness,
or commitment hardness. The Rust drift test checks the two prepared parity
maps. The separate overlay receipt checks all 110 overlay maps.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLinkSchema

abbrev audit :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLink.audit

def expectedRuns : List RawRun :=
  [
    { bodySourceStart := 46055
      overlaySourceStart := 1
      outerCount := 1
      bodySourceStride := 41
      overlaySourceStride := 41
      fieldCount := 41
      bodyFinalStart := 1059804
      overlayFinalStart := 111
      finalOuterStride := 41
      finalFieldStride := 1
      width := 1
      radix := 2 }
  , { bodySourceStart := 46096
      overlaySourceStart := 42
      outerCount := 810
      bodySourceStride := 122
      overlaySourceStride := 41
      fieldCount := 41
      bodyFinalStart := 19332
      overlayFinalStart := 152
      finalOuterStride := 41
      finalFieldStride := 1
      width := 1
      radix := 2 }
  , { bodySourceStart := 144918
      overlaySourceStart := 33252
      outerCount := 1
      bodySourceStride := 108
      overlaySourceStride := 108
      fieldCount := 108
      bodyFinalStart := 1076091
      overlayFinalStart := 33362
      finalOuterStride := 2484
      finalFieldStride := 23
      width := 23
      radix := 7 }
  ]

def RawRun.linkCount (run : RawRun) : Nat :=
  run.outerCount * run.fieldCount

def RawRun.bodyFinalEnd (run : RawRun) : Nat :=
  run.bodyFinalStart + (run.outerCount - 1) * run.finalOuterStride +
    (run.fieldCount - 1) * run.finalFieldStride + run.width

def RawRun.overlayFinalEnd (run : RawRun) : Nat :=
  run.overlayFinalStart + (run.outerCount - 1) * run.finalOuterStride +
    (run.fieldCount - 1) * run.finalFieldStride + run.width

def exactShape : Prop :=
  audit.schemaVersion = supportedSchemaVersion /\
    audit.familyCount = 110 /\
    audit.parityCount = 2 /\
    audit.publicOutputCount = 640 /\
    audit.bodyFinalColumns = 2484972 /\
    audit.overlayFinalColumns = 35856 /\
    audit.phaseKinds = [10, 11] /\
    audit.runs = expectedRuns

def sourceGeometryCoherent : Prop :=
  audit.runs.map RawRun.bodySourceStart =
      [45415 + audit.publicOutputCount,
       45456 + audit.publicOutputCount,
       144278 + audit.publicOutputCount] /\
    audit.runs.map RawRun.overlaySourceStart = [1, 42, 33252] /\
    audit.runs.map RawRun.bodySourceStride = [41, 122, 108] /\
    audit.runs.map RawRun.overlaySourceStride = [41, 41, 108] /\
    audit.runs.map RawRun.outerCount = [1, 810, 1] /\
    audit.runs.map RawRun.fieldCount = [41, 41, 108]

def finalGeometryCoherent : Prop :=
  audit.runs.map RawRun.bodyFinalStart = [1059804, 19332, 1076091] /\
    audit.runs.map RawRun.overlayFinalStart = [111, 152, 33362] /\
    audit.runs.map RawRun.finalOuterStride = [41, 41, 2484] /\
    audit.runs.map RawRun.finalFieldStride = [1, 1, 23] /\
    audit.runs.map RawRun.width = [1, 1, 23] /\
    audit.runs.map RawRun.radix = [2, 2, 7] /\
    (audit.runs.all fun run =>
      decide (RawRun.bodyFinalEnd run <= audit.bodyFinalColumns)) = true /\
    (audit.runs.all fun run =>
      decide (RawRun.overlayFinalEnd run <= audit.overlayFinalColumns)) = true

def censusCoherent : Prop :=
  audit.runs.map RawRun.linkCount = [41, 33210, 108] /\
    (audit.runs.map RawRun.linkCount).sum = audit.linkCountPerFamily /\
    audit.linkCountPerFamily = 33359 /\
    audit.totalLinkCount = audit.familyCount * audit.linkCountPerFamily /\
    audit.totalLinkCount = 3669490

def crossReceiptCoherent : Prop :=
  let overlay :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyOverlayRetained.audit
  let even :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm
  let odd :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm
  audit.overlayFinalColumns = overlay.finalColumns /\
    audit.bodyFinalColumns = even.finalColumns /\
    audit.bodyFinalColumns = odd.finalColumns /\
    audit.runs.map RawRun.overlaySourceStart = overlay.sourceStarts /\
    audit.runs.map RawRun.overlayFinalStart = overlay.finalStarts /\
    audit.runs.map RawRun.width = [overlay.widths.getD 0 0,
      overlay.widths.getD 0 0, overlay.widths.getD 1 0] /\
    audit.runs.map RawRun.radix = [overlay.radices.getD 0 0,
      overlay.radices.getD 0 0, overlay.radices.getD 1 0]

def AuditValid : Prop :=
  exactShape /\ sourceGeometryCoherent /\ finalGeometryCoherent /\
    censusCoherent /\ crossReceiptCoherent

/-- The normalized source fields equal the physical private fields after the
exact 640-column public prefix is inserted. -/
theorem source_geometry_exact : sourceGeometryCoherent := by
  unfold sourceGeometryCoherent audit
  native_decide

/-- Every decoded body and overlay slot uses the exact affine start, stride,
width, and radix read by the generic link compiler. -/
theorem final_geometry_exact : finalGeometryCoherent := by
  unfold finalGeometryCoherent RawRun.bodyFinalEnd RawRun.overlayFinalEnd
    audit
  native_decide

/-- The three runs contain exactly 33,359 links per family and 3,669,490
links across all 110 family positions. -/
theorem link_census_exact : censusCoherent := by
  unfold censusCoherent RawRun.linkCount audit
  native_decide

/-- The link receipt and the independently checked body-decoder and overlay
receipts use the same final column bounds and overlay slot image. -/
theorem cross_receipts_exact : crossReceiptCoherent := by
  unfold crossReceiptCoherent audit
  native_decide

/-- The generated receipt has the exact source geometry, final slot image,
cross-receipt agreement, and link censuses checked by Rust. -/
theorem audit_valid : AuditValid := by
  unfold AuditValid exactShape sourceGeometryCoherent finalGeometryCoherent
    censusCoherent crossReceiptCoherent RawRun.bodyFinalEnd
    RawRun.overlayFinalEnd RawRun.linkCount audit expectedRuns
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink
