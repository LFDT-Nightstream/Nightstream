import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetained
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceImage

/-!
Contract: independent validation of the compact production PiRLC residual
retained-row scan receipt.

Assurance tier: Rust-conformant for property
`FPRIME-PIRLC-FAMILY-BODY-RESIDUAL-RETAINED-PORT-IMAGE` under the supported
Goldilocks `b = 2`, `k_rho = 16` profile.

Owns agreement with the 108-row residual interval in both parity arms, the
direct radix-three decoder slots, and independent source and final nonzero
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

def directResidualBatch : RawResidualBatch :=
  { sourceStart := 164140
    instanceCount := 1
    instanceStride := 0
    width := 2180
    resolution := .direct 2129045 41 41 false }

def sourceNnzExpected : List Nat :=
  [108 * 3, 108, 0]

def finalPortNnzExpected : List Nat :=
  let arms := 2
  let width := 41
  let rows := 108
  [0, arms * rows, arms * rows * (3 * width), arms * rows,
    0, 0, 0, 0, 0, 0, 0, 0, 0]

def retainedIntervalsExpected : List RawRetainedRun :=
  [ { arm := 0, sourceStart := 163501, length := 108,
      emittedStart := 69499 }
  , { arm := 1, sourceStart := 163501, length := 108,
      emittedStart := 305010 }
  ]

def residualRetainedIntervals :=
  rowLedger.retainedRuns.filter fun run =>
    run.sourceStart == audit.sourceRowStart &&
      run.length == audit.sourceRows

def exactShape : Prop :=
  audit.schemaVersion =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyResidualRetainedSchema.supportedSchemaVersion /\
    audit.sourceRowStart = 163501 /\
    audit.sourceRows = 108 /\
    audit.localColumns = 165664 /\
    audit.sourceColumnShift = 640 /\
    audit.finalRows = 491046 /\
    audit.finalColumns = 8858862 /\
    audit.selectorColumns = [648, 649] /\
    audit.emittedStarts = [69499, 305010] /\
    audit.sourceStarts = [164142, 164250, 164358] /\
    audit.finalStarts = [2129127, 2133555, 2137983] /\
    audit.widths = [41, 41, 41] /\
    audit.radices = audit.widths.map (fun width => (slotRadix width).val) /\
    audit.sourceNnz = sourceNnzExpected /\
    audit.finalPortNnz = finalPortNnzExpected

def decoderCoherent : Prop :=
  (evenDecoder.residualBatches.drop 3).head? = some directResidualBatch /\
    (oddDecoder.residualBatches.drop 3).head? = some directResidualBatch /\
    audit.finalStarts =
      [2129045 + (164142 - 164140) * 41,
        2129045 + (164250 - 164140) * 41,
        2129045 + (164358 - 164140) * 41] /\
    2129045 + 2180 * 41 <= audit.finalColumns

def rowLedgerCoherent : Prop :=
  residualRetainedIntervals = retainedIntervalsExpected /\
    audit.emittedStarts.map (fun start => start + audit.sourceRows) =
      [69607, 305118] /\
    (audit.emittedStarts.all fun start =>
      decide (start + audit.sourceRows <= audit.finalRows)) = true

def AuditValid : Prop :=
  exactShape /\ decoderCoherent /\ rowLedgerCoherent

/-- The source and final nonzero counts follow from the exact additive
residual row shape and the three 23-coordinate source images. -/
theorem nonzero_census_exact :
    audit.sourceNnz = sourceNnzExpected /\
      audit.finalPortNnz = finalPortNnzExpected := by
  exact ⟨rfl, rfl⟩

/-- Both parity decoders use the same direct radix-three batch for every
residual field referenced by the retained block. -/
theorem decoder_run_exact : decoderCoherent := by
  unfold decoderCoherent directResidualBatch
  exact ⟨rfl, rfl, rfl, by decide⟩

/-- The row ledger maps the same 108 source rows to the two exact emitted
intervals scanned by Rust. -/
theorem retained_intervals_exact : rowLedgerCoherent := by
  unfold rowLedgerCoherent residualRetainedIntervals retainedIntervalsExpected
  exact ⟨rfl, rfl, rfl⟩

private theorem shape_exact : exactShape := by
  unfold exactShape
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    by decide, nonzero_census_exact.1, nonzero_census_exact.2⟩

/-- The generated receipt agrees with the decoder, row ledger, and
independently recomputed nonzero census. -/
theorem audit_valid : AuditValid := by
  unfold AuditValid
  exact ⟨shape_exact, decoder_run_exact, retained_intervals_exact⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyResidualRetained
