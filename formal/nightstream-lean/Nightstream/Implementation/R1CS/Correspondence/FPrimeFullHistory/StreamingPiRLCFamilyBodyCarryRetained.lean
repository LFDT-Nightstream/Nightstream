import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetained
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: independent validation of the compact production PiRLC carry
retained-row scan receipt.

Assurance tier: Rust-conformant for property
`FPRIME-PIRLC-FAMILY-BODY-CARRY-RETAINED-PORT-IMAGE`.

Owns agreement with the 1,621-row carry interval in both parity arms, the
direct radix-seven decoder slots, and independent source and final nonzero
censuses.

Does not own matrix authority in Lean, assignment values, row satisfaction,
selector authority, challenge range, recursive orchestration, or lifecycle
soundness. The Rust drift test compares every source and final matrix row with
the recipe represented by this receipt.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetainedSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage

abbrev audit :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetained.audit

abbrev rowLedger :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger

abbrev evenDecoder :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm

abbrev oddDecoder :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm

def directCarryRun : RawStridedRun :=
  { sourceStart := 144916
    count := 1964
    sourceStride := 1
    resolution := .direct 1076045 23 23 false }

def sourceNnzExpected : List Nat :=
  [810 * 3 + 810 * 2 + 3, 1621, 0]

def finalPortNnzExpected : List Nat :=
  let arms := 2
  let width := 23
  let rows := 1621
  [0,
    arms * rows,
    arms * (810 * (2 * width + 1) + 810 * (2 * width) +
      (2 * width + 1)),
    arms * rows,
    0, 0, 0, 0, 0, 0, 0, 0, 0]

def retainedIntervalsExpected : List RawRetainedRun :=
  [ { arm := 0, sourceStart := 144385, length := 1621,
      emittedStart := 78113 }
  , { arm := 1, sourceStart := 144385, length := 1621,
      emittedStart := 200471 }
  ]

def carryRetainedIntervals :=
  rowLedger.retainedRuns.filter fun run =>
    run.sourceStart == audit.sourceRowStart &&
      run.length == audit.sourceRows

def exactShape : Prop :=
  audit.schemaVersion =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetainedSchema.supportedSchemaVersion /\
    audit.sourceRowStart = 144385 /\
    audit.sourceRows = 1621 /\
    audit.localColumns = 146224 /\
    audit.sourceColumnShift = 640 /\
    audit.finalRows = 279089 /\
    audit.finalColumns = 2484972 /\
    audit.selectorColumns = [648, 649] /\
    audit.emittedStarts = [78113, 200471] /\
    audit.sourceStarts = [641, 145242, 146052, 146862, 146863] /\
    audit.finalStarts = [702, 1083543, 1102173, 1120803, 1120826] /\
    audit.widths = [23, 23, 23, 23, 23] /\
    audit.radices = audit.widths.map (fun width => (slotRadix width).val) /\
    audit.sourceNnz = sourceNnzExpected /\
    audit.finalPortNnz = finalPortNnzExpected

def decoderCoherent : Prop :=
  (evenDecoder.residualRuns.drop 5).head? = some directCarryRun /\
    (oddDecoder.residualRuns.drop 5).head? = some directCarryRun /\
    audit.finalStarts =
      [702,
        1076045 + (145242 - 144916) * 23,
        1076045 + (146052 - 144916) * 23,
        1076045 + (146862 - 144916) * 23,
        1076045 + (146863 - 144916) * 23] /\
    1076045 + 1964 * 23 <= audit.finalColumns

def rowLedgerCoherent : Prop :=
  carryRetainedIntervals = retainedIntervalsExpected /\
    audit.emittedStarts.map (fun start => start + audit.sourceRows) =
      [79734, 202092] /\
    (audit.emittedStarts.all fun start =>
      decide (start + audit.sourceRows <= audit.finalRows)) = true

def AuditValid : Prop :=
  exactShape /\ decoderCoherent /\ rowLedgerCoherent

/-- The source and final nonzero counts follow from the three exact equality
row shapes and the 23-coordinate source images. -/
theorem nonzero_census_exact :
    audit.sourceNnz = sourceNnzExpected /\
      audit.finalPortNnz = finalPortNnzExpected := by
  native_decide

/-- Both parity decoders use the same direct radix-seven run for every carry
field referenced by the retained block. -/
theorem decoder_run_exact : decoderCoherent := by
  unfold decoderCoherent
  native_decide

/-- The row ledger maps the same 1,621 source rows to the two exact emitted
intervals scanned by Rust. -/
theorem retained_intervals_exact : rowLedgerCoherent := by
  unfold rowLedgerCoherent
  native_decide

/-- The generated receipt agrees with the decoder, row ledger, and
independently recomputed nonzero census. -/
theorem audit_valid : AuditValid := by
  unfold AuditValid exactShape decoderCoherent rowLedgerCoherent
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained
