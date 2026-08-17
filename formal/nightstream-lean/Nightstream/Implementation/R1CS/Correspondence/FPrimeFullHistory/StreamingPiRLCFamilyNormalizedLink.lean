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
and the exact 37,787-per-family and 4,156,570-total censuses.

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
    { bodySourceStart := 52103
      overlaySourceStart := 1
      outerCount := 1
      bodySourceStride := 41
      overlaySourceStride := 41
      fieldCount := 41
      bodyFinalStart := 2110644
      overlayFinalStart := 111
      finalOuterStride := 41
      finalFieldStride := 1
      width := 1
      radix := 2 }
  , { bodySourceStart := 52144
      overlaySourceStart := 42
      outerCount := 918
      bodySourceStride := 122
      overlaySourceStride := 41
      fieldCount := 41
      bodyFinalStart := 38340
      overlayFinalStart := 152
      finalOuterStride := 41
      finalFieldStride := 1
      width := 1
      radix := 2 }
  , { bodySourceStart := 164142
      overlaySourceStart := 37680
      outerCount := 1
      bodySourceStride := 108
      overlaySourceStride := 108
      fieldCount := 108
      bodyFinalStart := 2129127
      overlayFinalStart := 37790
      finalOuterStride := 4428
      finalFieldStride := 41
      width := 41
      radix := 3 }
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
    audit.bodyFinalColumns = 8858862 /\
    audit.overlayFinalColumns = 42228 /\
    audit.phaseKinds = [10, 11] /\
    audit.runs = expectedRuns

def sourceGeometryCoherent : Prop :=
  audit.runs.map RawRun.bodySourceStart =
      [51463 + audit.publicOutputCount,
       51504 + audit.publicOutputCount,
       163502 + audit.publicOutputCount] /\
    audit.runs.map RawRun.overlaySourceStart = [1, 42, 37680] /\
    audit.runs.map RawRun.bodySourceStride = [41, 122, 108] /\
    audit.runs.map RawRun.overlaySourceStride = [41, 41, 108] /\
    audit.runs.map RawRun.outerCount = [1, 918, 1] /\
    audit.runs.map RawRun.fieldCount = [41, 41, 108]

def finalGeometryCoherent : Prop :=
  audit.runs.map RawRun.bodyFinalStart = [2110644, 38340, 2129127] /\
    audit.runs.map RawRun.overlayFinalStart = [111, 152, 37790] /\
    audit.runs.map RawRun.finalOuterStride = [41, 41, 4428] /\
    audit.runs.map RawRun.finalFieldStride = [1, 1, 41] /\
    audit.runs.map RawRun.width = [1, 1, 41] /\
    audit.runs.map RawRun.radix = [2, 2, 3] /\
    (audit.runs.all fun run =>
      decide (RawRun.bodyFinalEnd run <= audit.bodyFinalColumns)) = true /\
    (audit.runs.all fun run =>
      decide (RawRun.overlayFinalEnd run <= audit.overlayFinalColumns)) = true

def censusCoherent : Prop :=
  audit.runs.map RawRun.linkCount = [41, 37638, 108] /\
    (audit.runs.map RawRun.linkCount).sum = audit.linkCountPerFamily /\
    audit.linkCountPerFamily = 37787 /\
    audit.totalLinkCount = audit.familyCount * audit.linkCountPerFamily /\
    audit.totalLinkCount = 4156570

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
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Every decoded body and overlay slot uses the exact affine start, stride,
width, and radix read by the generic link compiler. -/
theorem final_geometry_exact : finalGeometryCoherent := by
  unfold finalGeometryCoherent RawRun.bodyFinalEnd RawRun.overlayFinalEnd
    audit
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The three runs contain exactly 37,787 links per family and 4,156,570
links across all 110 family positions. -/
theorem link_census_exact : censusCoherent := by
  unfold censusCoherent RawRun.linkCount audit
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- The link receipt and the independently checked body-decoder and overlay
receipts use the same final column bounds and overlay slot image. -/
theorem cross_receipts_exact : crossReceiptCoherent := by
  unfold crossReceiptCoherent audit
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

private theorem exact_shape_exact : exactShape := by
  unfold exactShape audit expectedRuns
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The generated receipt has the exact source geometry, final slot image,
cross-receipt agreement, and link censuses checked by Rust. -/
theorem audit_valid : AuditValid := by
  refine ⟨exact_shape_exact, source_geometry_exact, final_geometry_exact,
    link_census_exact, cross_receipts_exact⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyNormalizedLink
