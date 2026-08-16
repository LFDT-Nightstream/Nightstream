import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows
import Nightstream.Implementation.R1CS.Artifacts.ShiftedTernary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger

/-!
Contract: independent arithmetic validation of the production PiRLC
opening-row scan receipt.

Assurance tier: Rust-conformant for the exact source traces, final matrix port
images, and compact geometry checked by the Rust producer.

Owns the receipt shape and its joins to the complete row ledger, source
decoder, and generic 21-row shifted-ternary artifact.

Does not own assignment values, outer norm authority, row semantics,
canonical-word soundness, recursive orchestration, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema

abbrev audit :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.audit

private abbrev ledger :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger

def auditedRowCount : Nat :=
  audit.centeredRowCount +
    audit.armCount * (audit.digitCount + audit.openingCount * audit.chunkCount)

def ExactReceipt : Prop :=
  audit.schemaVersion =
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema.supportedSchemaVersion /\
  audit.armCount = 2 /\
  audit.openingCount = 810 /\
  audit.digitCount = 41 /\
  audit.borrowCount = 20 /\
  audit.chunkCount = 21 /\
  audit.sourceZeroRowStart = 43794 /\
  audit.sourceZeroDigitStart = 46055 /\
  audit.sourceFieldStart = 1451 /\
  audit.sourceDigitStart = 46096 /\
  audit.sourceDigitStride = 122 /\
  audit.sourceCanonicalRowStart = 43835 /\
  audit.sourceCanonicalRowStride = 124 /\
  audit.centeredRowStart = 2 /\
  audit.centeredRowCount = 16605 /\
  audit.zeroEmittedStarts = [77962, 200320] /\
  audit.canonicalEmittedStarts = [139516, 262046] /\
  audit.selectorColumns = [648, 649] /\
  audit.finalDigitStart = 19332 /\
  audit.finalDigitStride = 41 /\
  audit.finalZeroStart = 1059804 /\
  audit.finalBorrowStart = 1059845 /\
  audit.finalBorrowStride = 20 /\
  audit.finalRows = 279089 /\
  audit.finalColumns = 2484972 /\
  audit.normalizedChunkBounds =
    [3, 0, 3, 3, 3, 0, 1, 3, 1, 2, 4, 3, 2, 1, 3, 0, 0, 0, 3, 4, 1] /\
  audit.complementedChunks =
    [false, false, false, true, true, false, true, false, false, false,
      false, false, true, true, true, true, false, true, true, false, false] /\
  audit.sourceZeroNnz = [41, 41, 0] /\
  audit.finalPortNnz =
    [46980, 50707, 50707, 82, 46980, 46980, 50625, 16605, 8100,
      6480, 3240, 12960, 3240]

theorem exact_receipt : ExactReceipt := by
  simp [ExactReceipt, audit,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.audit,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRowsSchema.supportedSchemaVersion]

theorem audited_row_count_exact : auditedRowCount = 50707 := by
  native_decide

/-- The receipt accounts for the same 1,620 rewrites and 34,020 emitted rows
as the complete compiler row ledger. -/
theorem row_ledger_canonical_census_join :
    audit.armCount * audit.openingCount =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger.rewriteInstanceCount
          ledger .shiftedTernaryCanonical /\
      audit.armCount * audit.openingCount * audit.chunkCount =
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyRowLedger.rewriteEmittedRowCount
          ledger .shiftedTernaryCanonical /\
      audit.finalRows = ledger.rows /\
      audit.finalColumns = ledger.columns := by
  native_decide

/-- Both decoder arms use the exact source-digit, final-borrow, and shared
final-digit affine template recorded by this receipt. -/
theorem decoder_opening_template_join :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.evenTemplateInstances00.map
        (fun entry =>
          (entry.sourceStart, entry.count, entry.sourceStride,
            entry.finalStart, entry.finalStride,
            entry.referenceFinalStart, entry.referenceFinalStride)) =
      [(audit.sourceDigitStart, audit.openingCount, audit.sourceDigitStride,
        audit.finalBorrowStart, audit.finalBorrowStride,
        audit.finalDigitStart, audit.finalDigitStride)] /\
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyBodyDecoder.oddTemplateInstances00.map
        (fun entry =>
          (entry.sourceStart, entry.count, entry.sourceStride,
            entry.finalStart, entry.finalStride,
            entry.referenceFinalStart, entry.referenceFinalStride)) =
      [(audit.sourceDigitStart, audit.openingCount, audit.sourceDigitStride,
        audit.finalBorrowStart, audit.finalBorrowStride,
        audit.finalDigitStart, audit.finalDigitStride)] := by
  constructor <;> rfl

/-- The production repeated-row width is the exact width of the separately
generated and semantically interpreted one-opening artifact. -/
theorem generic_artifact_join :
    Nightstream.Implementation.R1CS.ShiftedTernarySelectiveArtifact.rowPorts.length =
        audit.chunkCount /\
      Nightstream.Implementation.R1CS.ShiftedTernarySelectiveArtifact.digitCoordinates.length =
        audit.digitCount /\
      Nightstream.Implementation.R1CS.ShiftedTernarySelectiveArtifact.borrowCoordinates.length =
        audit.borrowCount /\
      Nightstream.Implementation.R1CS.ShiftedTernarySelectiveArtifact.polynomialTerms.length = 74 := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows
