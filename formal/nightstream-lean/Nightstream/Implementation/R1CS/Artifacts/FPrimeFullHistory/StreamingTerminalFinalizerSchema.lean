import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema
import Nightstream.Implementation.R1CS.Core.SeededPhi81
import Nightstream.Implementation.R1CS.Core.TranscriptCertificate

/-!
Contract: compact geometry schema for the terminal Nebula finalizer family.

It records the two Rust-owned source ranges, verifier-owned input columns,
ordered core phases, and final outputs. It does not claim that the rows imply
the typed Nebula close relation.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact

def laneFields : Nat := 50
def delayedPayloadFields : Nat := 2169
def commitmentDataFields : Nat := 972
def digestFields : Nat := 4
def shapeFields : Nat := 6
def stepBitFields : Nat := 1400
def dPreBitFields : Nat := 768
def openBitIndex : Nat := stepBitFields
def dPreBitStart : Nat := openBitIndex + 1
def stepWordWidths : List Nat :=
  [16, 16, 44, 44] ++ List.replicate 20 64
def stepWordCount : Nat := stepWordWidths.length
def dPreWordCount : Nat := 12
def segmentIndexBits : Nat := 16
def highSegmentIndexBits : Nat := segmentIndexBits - 1
def openAlgebraRowCount : Nat := 53
def stagedDigestConstantFields : Nat := 13
def stagedDigestInputFields : Nat := 58
def balancedTernaryDigits : Nat := 41
def canonicalOpeningRows : Nat := 3 * balancedTernaryDigits + 1
def canonicalOpeningColumns : Nat := 3 * balancedTernaryDigits - 1
def leafPrefixFields : Nat := 9
def leafPrimaryFields : Nat := leafPrefixFields + commitmentDataFields
def leafPrimaryOutputs : Nat := 2 * 54
def leafCompressionFields : Nat := leafPrimaryOutputs
def leafCompressionOutputs : Nat := 54
def leafEnvelopeConstantFields : Nat := 10

structure SeededBindingArtifact where
  sourceColumns : List Nat
  metadataPinRowStart : Nat
  metadataValues : List Nat
  metadataStartColumn : Nat
  openingRowStart : Nat
  block : SeededPhi81.Block
deriving DecidableEq, Repr

structure LeafHashArtifact where
  prefixPinRowStart : Nat
  prefixConstantValues : List Nat
  prefixConstantStartColumn : Nat
  primary : SeededBindingArtifact
  compression : SeededBindingArtifact
  envelopeConstantValues : List Nat
  envelopeConstantStartColumn : Nat
  digestInputColumns : List Nat
  digestOutputColumns : List Nat
  digestRowStart : Nat
  digestRowStop : Nat
deriving DecidableEq, Repr

/-- One compact Poseidon2 hash with exact constant-row and trace ownership. -/
structure PoseidonHashArtifact where
  constantRowStart : Nat
  traceRowStart : Nat
  traceRowStop : Nat
  recipe : FPrimeFullHistoryStreamingVariableHashRecipe.Artifact.VariableHashRecipe
deriving DecidableEq, Repr

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceIdentity : String
  sourceRowsSha256 : String
  columnCount : Nat
  shapeRowStart : Nat
  shapeRowStop : Nat
  shapeColumns : List Nat
  dimension : Nat
  kappa : Nat
  stepsPerSegment : Nat
  segmentMaximum : Nat
  stackCount : Nat
  stackPointerBits : Nat
  coreRowStart : Nat
  coreRowStop : Nat
  internalColumnStart : Nat
  laneColumns : List Nat
  payloadColumns : List Nat
  opsColumns : List Nat
  isColumns : List Nat
  fsColumns : List Nat
  vkFsColumns : List Nat
  boundaryColumns : List Nat
  accumulatorColumns : List Nat
  decodeRowStop : Nat
  openRowStop : Nat
  leavesRowStop : Nat
  openAlgebraRowStop : Nat
  openInternalColumnStart : Nat
  stagedDigestConstantValues : List Nat
  stagedDigestConstantStartColumn : Nat
  stagedDigestInputColumns : List Nat
  stagedDigestOutputColumns : List Nat
  stagedDigestRowStart : Nat
  stagedDigestRowStop : Nat
  gammaTranscriptRowStart : Nat
  gammaTranscriptRowStop : Nat
  gammaTranscriptPinRows : List Nat
  gammaTranscriptPinColumns : List Nat
  gammaTranscriptPinValues : List Nat
  gammaTranscriptInitialAbsorbed : Nat
  gammaTranscriptCalls : List Poseidon2Call.Call
  gamma1Columns : List Nat
  gamma2Columns : List Nat
  gammaMuxSelectorColumn : Nat
  gammaMuxOpenedColumns : List Nat
  gammaMuxCarriedColumns : List Nat
  gammaMuxOutputColumns : List Nat
  opsLeaf : LeafHashArtifact
  isLeaf : LeafHashArtifact
  fsLeaf : LeafHashArtifact
  advanceChainLinks : List PoseidonHashArtifact
  openedLaneColumns : List Nat
  advancedLaneColumns : List Nat
  advanceAlgebraRows : List (Nat × Row)
  closeRows : List (Nat × Row)
  terminalClosedRow : Nat × Row
  advanceRowStop : Nat
  closeRowStop : Nat
  coreRowCount : Nat
  finalLaneColumns : List Nat
  closedColumn : Nat
deriving DecidableEq, Repr

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

def PoseidonHashArtifact.rows (artifact : PoseidonHashArtifact) : List Row :=
  constantRows artifact.recipe ++ artifact.recipe.trace.rows

def PoseidonHashArtifact.Satisfied
    (artifact : PoseidonHashArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.rows assignment

instance (artifact : PoseidonHashArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.Satisfied assignment) := by
  unfold PoseidonHashArtifact.Satisfied
  infer_instance

def indexedRowValues (rows : List (Nat × Row)) : List Row :=
  rows.map Prod.snd

def RawArtifact.AdvanceAlgebraSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies (indexedRowValues artifact.advanceAlgebraRows) assignment

def RawArtifact.CloseSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies (indexedRowValues artifact.closeRows) assignment

def RawArtifact.TerminalClosedSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  RowHolds assignment artifact.terminalClosedRow.2

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.AdvanceAlgebraSatisfied assignment) := by
  unfold RawArtifact.AdvanceAlgebraSatisfied
  infer_instance

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.CloseSatisfied assignment) := by
  unfold RawArtifact.CloseSatisfied
  infer_instance

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.TerminalClosedSatisfied assignment) := by
  unfold RawArtifact.TerminalClosedSatisfied
  infer_instance

def LeafHashArtifact.envelopeRecipe (artifact : LeafHashArtifact) :
    VariableHashRecipe where
  constantValues := artifact.envelopeConstantValues
  constantStartColumn := artifact.envelopeConstantStartColumn
  localColumns := []
  payloadColumns := []
  orderedInputColumns := artifact.digestInputColumns
  outputColumns := artifact.digestOutputColumns

def SeededBindingArtifact.wordStart
    (artifact : SeededBindingArtifact) (index : Nat) : Nat :=
  artifact.block.wordStarts.getD index 0

def SeededBindingArtifact.openingRow
    (artifact : SeededBindingArtifact) (index : Nat) : Nat :=
  artifact.openingRowStart + canonicalOpeningRows * index

def SeededBindingArtifact.digitColumns
    (artifact : SeededBindingArtifact) (index : Nat) : List Nat :=
  List.range' (artifact.wordStart index) balancedTernaryDigits

def SeededBindingArtifact.negativeColumns
    (artifact : SeededBindingArtifact) (index : Nat) : List Nat :=
  List.range' (artifact.wordStart index + balancedTernaryDigits)
    balancedTernaryDigits

def SeededBindingArtifact.borrowColumns
    (artifact : SeededBindingArtifact) (index : Nat) : List Nat :=
  List.range' (artifact.wordStart index + 2 * balancedTernaryDigits)
    (balancedTernaryDigits - 1)

/-- Column-normalized form of Rust's Boolean row. -/
def normalizedBitRow (column : Nat) : Row :=
  ⟨[(column, 1)], [(0, negCoeff 1), (column, 1)], []⟩

/-- Column-normalized form of a linear definition whose fresh output follows
all referenced input columns. -/
def outputLastLinearRow
    (output : Nat) (terms : List (Nat × Nat)) : Row :=
  ⟨negateTerms terms ++ [(output, 1)], [(0, 1)], []⟩

def RawArtifact.internalColumn (artifact : RawArtifact) (offset : Nat) : Nat :=
  artifact.internalColumnStart + offset

def RawArtifact.payloadColumn (artifact : RawArtifact) (index : Nat) : Nat :=
  artifact.payloadColumns.getD index 0

def RawArtifact.stepBitColumns (artifact : RawArtifact) : List Nat :=
  artifact.payloadColumns.take stepBitFields

def RawArtifact.openColumn (artifact : RawArtifact) : Nat :=
  artifact.payloadColumn openBitIndex

def RawArtifact.dPreBitColumns (artifact : RawArtifact) : List Nat :=
  (artifact.payloadColumns.drop dPreBitStart).take dPreBitFields

def wordStart (widths : List Nat) (index : Nat) : Nat :=
  (widths.take index).sum

def RawArtifact.wordTerms
    (artifact : RawArtifact) (start width : Nat) : List (Nat × Nat) :=
  (List.range width).map fun bit =>
    (artifact.payloadColumn (start + bit), 2 ^ bit)

def RawArtifact.stepWordTerms
    (artifact : RawArtifact) (index : Nat) : List (Nat × Nat) :=
  artifact.wordTerms (wordStart stepWordWidths index)
    (stepWordWidths.getD index 0)

def RawArtifact.dPreWordTerms
    (artifact : RawArtifact) (index : Nat) : List (Nat × Nat) :=
  artifact.wordTerms (dPreBitStart + 64 * index) 64

def RawArtifact.stepWordColumn
    (artifact : RawArtifact) (index : Nat) : Nat :=
  artifact.internalColumn index

def RawArtifact.constantZeroColumn (artifact : RawArtifact) : Nat :=
  artifact.internalColumn stepWordCount

def RawArtifact.dPreWordColumn
    (artifact : RawArtifact) (index : Nat) : Nat :=
  artifact.internalColumn (stepWordCount + 1 + index)

def RawArtifact.stepBitRows (artifact : RawArtifact) : List Row :=
  artifact.stepBitColumns.map normalizedBitRow

def RawArtifact.stepWordRow (artifact : RawArtifact) (index : Nat) : Row :=
  outputLastLinearRow (artifact.stepWordColumn index)
    (artifact.stepWordTerms index)

def RawArtifact.stepWordRows (artifact : RawArtifact) : List Row :=
  (List.range stepWordCount).map artifact.stepWordRow

def RawArtifact.constantZeroRow (artifact : RawArtifact) : Row :=
  outputLastLinearRow artifact.constantZeroColumn []

def RawArtifact.openBitRows (artifact : RawArtifact) : List Row :=
  [normalizedBitRow artifact.openColumn]

def RawArtifact.dPreBitRows (artifact : RawArtifact) : List Row :=
  artifact.dPreBitColumns.map normalizedBitRow

def RawArtifact.inactiveDPreRow
    (artifact : RawArtifact) (bitColumn : Nat) : Row :=
  ⟨[(artifact.openColumn, goldilocksP - 1), (0, 1)],
    [(bitColumn, 1)], []⟩

def RawArtifact.inactiveDPreRows (artifact : RawArtifact) : List Row :=
  artifact.dPreBitColumns.map artifact.inactiveDPreRow

def RawArtifact.dPreWordRow (artifact : RawArtifact) (index : Nat) : Row :=
  outputLastLinearRow (artifact.dPreWordColumn index)
    (artifact.dPreWordTerms index)

def RawArtifact.dPreWordRows (artifact : RawArtifact) : List Row :=
  (List.range dPreWordCount).map artifact.dPreWordRow

def RawArtifact.decodePieces (artifact : RawArtifact) : List (List Row) :=
  [artifact.stepBitRows,
    artifact.stepWordRows,
    [artifact.constantZeroRow],
    artifact.openBitRows,
    artifact.dPreBitRows,
    artifact.inactiveDPreRows,
    artifact.dPreWordRows]

def RawArtifact.decodeRows (artifact : RawArtifact) : List Row :=
  artifact.decodePieces.flatten

def RawArtifact.DecodeSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.decodeRows assignment

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.DecodeSatisfied assignment) := by
  unfold RawArtifact.DecodeSatisfied
  infer_instance

def RawArtifact.laneColumn (artifact : RawArtifact) (index : Nat) : Nat :=
  artifact.laneColumns.getD index 0

def RawArtifact.laneOpenColumn (artifact : RawArtifact) : Nat :=
  artifact.laneColumn 4

def RawArtifact.laneSegmentIndexColumn (artifact : RawArtifact) : Nat :=
  artifact.laneColumn 5

def RawArtifact.laneStepIndexColumn (artifact : RawArtifact) : Nat :=
  artifact.laneColumn 6

def RawArtifact.segmentBitColumn
    (artifact : RawArtifact) (index : Nat) : Nat :=
  artifact.openInternalColumnStart + index

def RawArtifact.segmentBitColumns (artifact : RawArtifact) : List Nat :=
  (List.range segmentIndexBits).map artifact.segmentBitColumn

def RawArtifact.segmentBitRows (artifact : RawArtifact) : List Row :=
  artifact.segmentBitColumns.map normalizedBitRow

def RawArtifact.segmentRecompositionTerms
    (artifact : RawArtifact) : List (Nat × Nat) :=
  (List.range segmentIndexBits).map fun index =>
    (artifact.segmentBitColumn index, 2 ^ index)

def RawArtifact.segmentRecompositionRow (artifact : RawArtifact) : Row :=
  builderLinearRow artifact.laneSegmentIndexColumn
    artifact.segmentRecompositionTerms

def RawArtifact.comparisonOutputColumn
    (artifact : RawArtifact) (iteration : Nat) : Nat :=
  artifact.openInternalColumnStart + segmentIndexBits + iteration

def RawArtifact.comparisonInputColumn
    (artifact : RawArtifact) (iteration : Nat) : Nat :=
  if iteration = 0 then 0 else artifact.comparisonOutputColumn (iteration - 1)

def highSegmentIndex (iteration : Nat) : Nat :=
  segmentIndexBits - 1 - iteration

def RawArtifact.highSegmentForbidRow
    (artifact : RawArtifact) (iteration : Nat) : Row :=
  ⟨[(artifact.comparisonInputColumn iteration, 1)],
    [(artifact.segmentBitColumn (highSegmentIndex iteration), 1)], []⟩

def RawArtifact.highSegmentEqualDefinition
    (artifact : RawArtifact) (iteration : Nat) : Definition :=
  { output := artifact.comparisonOutputColumn iteration
    rhs := .product
      [(artifact.comparisonInputColumn iteration, 1)]
      [(0, 1),
        (artifact.segmentBitColumn (highSegmentIndex iteration),
          goldilocksP - 1)] }

def RawArtifact.highSegmentComparisonPieces
    (artifact : RawArtifact) : List (List Row) :=
  (List.range highSegmentIndexBits).map fun iteration =>
    [artifact.highSegmentForbidRow iteration,
      (artifact.highSegmentEqualDefinition iteration).builderRow]

def RawArtifact.highSegmentComparisonRows
    (artifact : RawArtifact) : List Row :=
  artifact.highSegmentComparisonPieces.flatten

def RawArtifact.lowSegmentEqualDefinition (artifact : RawArtifact) : Definition :=
  { output := artifact.comparisonOutputColumn highSegmentIndexBits
    rhs := .product
      [(artifact.comparisonInputColumn highSegmentIndexBits, 1)]
      [(artifact.segmentBitColumn 0, 1)] }

def RawArtifact.finalSegmentEqualZeroRow (artifact : RawArtifact) : Row :=
  outputLastLinearRow
    (artifact.comparisonOutputColumn highSegmentIndexBits) []

def RawArtifact.openFlagRows (artifact : RawArtifact) : List Row :=
  [normalizedBitRow artifact.laneOpenColumn,
    normalizedBitRow artifact.openColumn]

def RawArtifact.exclusiveOpenRow (artifact : RawArtifact) : Row :=
  ⟨[(0, goldilocksP - 1),
      (artifact.laneOpenColumn, 1), (artifact.openColumn, 1)],
    [(0, 1)], []⟩

def RawArtifact.newOpenZeroIndexRow (artifact : RawArtifact) : Row :=
  ⟨[(artifact.openColumn, 1)],
    [(artifact.laneStepIndexColumn, 1)], []⟩

def RawArtifact.openAlgebraPieces (artifact : RawArtifact) : List (List Row) :=
  [artifact.segmentBitRows,
    [artifact.segmentRecompositionRow],
    artifact.highSegmentComparisonRows,
    [(artifact.lowSegmentEqualDefinition).builderRow],
    [artifact.finalSegmentEqualZeroRow],
    artifact.openFlagRows,
    [artifact.exclusiveOpenRow],
    [artifact.newOpenZeroIndexRow]]

def RawArtifact.openAlgebraRows (artifact : RawArtifact) : List Row :=
  artifact.openAlgebraPieces.flatten

def RawArtifact.OpenAlgebraSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.openAlgebraRows assignment

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.OpenAlgebraSatisfied assignment) := by
  unfold RawArtifact.OpenAlgebraSatisfied
  infer_instance

def RawArtifact.stagedDigestRecipe (artifact : RawArtifact) :
    VariableHashRecipe where
  constantValues := artifact.stagedDigestConstantValues
  constantStartColumn := artifact.stagedDigestConstantStartColumn
  localColumns := []
  payloadColumns := []
  orderedInputColumns := artifact.stagedDigestInputColumns
  outputColumns := artifact.stagedDigestOutputColumns

def RawArtifact.stagedDigestExpectedInputs (artifact : RawArtifact) : List Nat :=
  (artifact.stagedDigestRecipe.constantColumns.take 7) ++
    (artifact.laneColumns.take 4) ++
    [artifact.laneSegmentIndexColumn, artifact.laneStepIndexColumn,
      artifact.laneColumn 7, artifact.laneColumn 20, artifact.laneColumn 21] ++
    (artifact.stagedDigestRecipe.constantColumns.drop 7) ++
    ((artifact.laneColumns.drop 12).take 8) ++
    (List.range dPreWordCount).map artifact.dPreWordColumn ++
    ((artifact.laneColumns.drop 34).take 16)

def RawArtifact.stagedDigestPieces (artifact : RawArtifact) : List (List Row) :=
  [constantRows artifact.stagedDigestRecipe,
    artifact.stagedDigestRecipe.trace.rows]

def RawArtifact.stagedDigestRows (artifact : RawArtifact) : List Row :=
  artifact.stagedDigestPieces.flatten

def RawArtifact.StagedDigestSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.stagedDigestRows assignment

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.StagedDigestSatisfied assignment) := by
  unfold RawArtifact.StagedDigestSatisfied
  infer_instance

def RawArtifact.gammaTranscriptPins
    (artifact : RawArtifact) : List (Nat × Nat) :=
  artifact.gammaTranscriptPinColumns.zip artifact.gammaTranscriptPinValues

def RawArtifact.gammaTranscriptTrace
    (artifact : RawArtifact) : TranscriptCertificate.Trace where
  pins := artifact.gammaTranscriptPins
  calls := artifact.gammaTranscriptCalls

def RawArtifact.GammaTranscriptSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies (ConstantPins.rows artifact.gammaTranscriptPins) assignment ∧
    ∀ call ∈ artifact.gammaTranscriptCalls,
      Satisfies call.rows assignment

def orderedDifferenceTerms (positive negative : Nat) : List (Nat × Nat) :=
  if positive = negative then []
  else if positive < negative then
    [(positive, 1), (negative, goldilocksP - 1)]
  else
    [(negative, goldilocksP - 1), (positive, 1)]

def RawArtifact.gammaMuxExpectedOpenedColumns
    (artifact : RawArtifact) : List Nat :=
  artifact.gamma1Columns ++ artifact.gamma2Columns ++
    (List.range dPreWordCount).map artifact.dPreWordColumn

def RawArtifact.gammaMuxExpectedCarriedColumns
    (artifact : RawArtifact) : List Nat :=
  (List.range 4).map (fun index => artifact.laneColumn (8 + index)) ++
    (List.range dPreWordCount).map
      (fun index => artifact.laneColumn (22 + index))

def RawArtifact.gammaMuxRow (artifact : RawArtifact) (index : Nat) : Row :=
  ⟨[(artifact.gammaMuxSelectorColumn, 1)],
    orderedDifferenceTerms
      (artifact.gammaMuxOpenedColumns.getD index 0)
      (artifact.gammaMuxCarriedColumns.getD index 0),
    orderedDifferenceTerms
      (artifact.gammaMuxOutputColumns.getD index 0)
      (artifact.gammaMuxCarriedColumns.getD index 0)⟩

def RawArtifact.gammaMuxRows (artifact : RawArtifact) : List Row :=
  (List.range 16).map artifact.gammaMuxRow

def RawArtifact.GammaMuxSatisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.gammaMuxRows assignment

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.GammaMuxSatisfied assignment) := by
  unfold RawArtifact.GammaMuxSatisfied
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
