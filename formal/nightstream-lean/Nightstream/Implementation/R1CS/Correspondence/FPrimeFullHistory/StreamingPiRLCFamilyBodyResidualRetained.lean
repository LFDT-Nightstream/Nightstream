import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetained
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: independent validation of the compact production PiRLC residual
retained-row scan receipt.

Assurance tier: Rust-conformant for property
`FPRIME-PIRLC-FAMILY-BODY-RESIDUAL-RETAINED-PORT-IMAGE`.

Owns agreement with the 108-row residual interval in both parity arms, the
direct radix-seven decoder slots, and independent source and final nonzero
censuses.

Does not own matrix authority in Lean, assignment values, row satisfaction,
selector authority, the local commitment output, recursive orchestration, or
lifecycle soundness. The Rust drift test compares every source and final
matrix row with the recipe represented by this receipt.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage

abbrev audit :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetained.audit

abbrev rowLedger :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger

abbrev evenDecoder :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm

abbrev oddDecoder :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm

def directResidualRun : RawStridedRun :=
  { sourceStart := 144916
    count := 1964
    sourceStride := 1
    resolution := .direct 1076045 23 23 false }

def sourceNnzExpected : List Nat :=
  [108 * 3, 108, 0]

def finalPortNnzExpected : List Nat :=
  let arms := 2
  let width := 23
  let rows := 108
  [0, arms * rows, arms * rows * (3 * width), arms * rows,
    0, 0, 0, 0, 0, 0, 0, 0, 0]

def retainedIntervalsExpected : List RawRetainedRun :=
  [ { arm := 0, sourceStart := 144277, length := 108,
      emittedStart := 78005 }
  , { arm := 1, sourceStart := 144277, length := 108,
      emittedStart := 200363 }
  ]

def residualRetainedIntervals :=
  rowLedger.retainedRuns.filter fun run =>
    run.sourceStart == audit.sourceRowStart &&
      run.length == audit.sourceRows

def exactShape : Prop :=
  audit.schemaVersion =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema.supportedSchemaVersion /\
    audit.sourceRowStart = 144277 /\
    audit.sourceRows = 108 /\
    audit.localColumns = 146224 /\
    audit.sourceColumnShift = 640 /\
    audit.finalRows = 279089 /\
    audit.finalColumns = 2484972 /\
    audit.selectorColumns = [648, 649] /\
    audit.emittedStarts = [78005, 200363] /\
    audit.sourceStarts = [144918, 145026, 145134] /\
    audit.finalStarts = [1076091, 1078575, 1081059] /\
    audit.widths = [23, 23, 23] /\
    audit.radices = audit.widths.map (fun width => (slotRadix width).val) /\
    audit.sourceNnz = sourceNnzExpected /\
    audit.finalPortNnz = finalPortNnzExpected

def decoderCoherent : Prop :=
  (evenDecoder.residualRuns.drop 5).head? = some directResidualRun /\
    (oddDecoder.residualRuns.drop 5).head? = some directResidualRun /\
    audit.finalStarts =
      [1076045 + (144918 - 144916) * 23,
        1076045 + (145026 - 144916) * 23,
        1076045 + (145134 - 144916) * 23] /\
    1076045 + 1964 * 23 <= audit.finalColumns

def rowLedgerCoherent : Prop :=
  residualRetainedIntervals = retainedIntervalsExpected /\
    audit.emittedStarts.map (fun start => start + audit.sourceRows) =
      [78113, 200471] /\
    (audit.emittedStarts.all fun start =>
      decide (start + audit.sourceRows <= audit.finalRows)) = true

def AuditValid : Prop :=
  exactShape /\ decoderCoherent /\ rowLedgerCoherent

/-- The source and final nonzero counts follow from the exact additive
residual row shape and the three 23-coordinate source images. -/
theorem nonzero_census_exact :
    audit.sourceNnz = sourceNnzExpected /\
      audit.finalPortNnz = finalPortNnzExpected := by
  native_decide

/-- Both parity decoders use the same direct radix-seven run for every
residual field referenced by the retained block. -/
theorem decoder_run_exact : decoderCoherent := by
  unfold decoderCoherent
  native_decide

/-- The row ledger maps the same 108 source rows to the two exact emitted
intervals scanned by Rust. -/
theorem retained_intervals_exact : rowLedgerCoherent := by
  unfold rowLedgerCoherent
  native_decide

/-- The generated receipt agrees with the decoder, row ledger, and
independently recomputed nonzero census. -/
theorem audit_valid : AuditValid := by
  unfold AuditValid exactShape decoderCoherent rowLedgerCoherent
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained
