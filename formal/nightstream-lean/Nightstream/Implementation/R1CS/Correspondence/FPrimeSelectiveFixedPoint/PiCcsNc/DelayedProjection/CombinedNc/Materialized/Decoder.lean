import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Decoder

/-!
Fail-closed decoder for the production delayed combined-NC row artifact.

Owns: canonical Goldilocks decoding; bounded source and emitted coordinates;
exact thirteen-port, 25-round, 54/64-lane profile checks; and consistency of
every named SumCheck column with its complete 43-column affine map.

Does not own: generated-data truth, row satisfaction, compiler rewrite
soundness, transcript order, raw-child authority, commitment binding, costs,
or row removal.

Emits constraints: none.

The decoder never repairs malformed data: field aliases, zero sparse terms,
empty or escaping ranges, missing pending columns, wrong profile lengths, and
self-inconsistent round maps all return `none`.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.decoder` | Decode proof-free generated row and provenance records into typed R1CS structures. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc

abbrev decodeField :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField

structure DecodedTerm (columns : Nat) where
  column : Fin columns
  coefficient : F
  coefficientNonzero : coefficient ≠ 0

structure DecodedGeometricRun (columns : Nat) where
  columnStart : Nat
  length : Nat
  lengthPositive : 0 < length
  endBound : columnStart + length ≤ columns
  initial : F
  ratio : F
  initialNonzero : initial ≠ 0
  ratioNonzero : ratio ≠ 0

def DecodedGeometricRun.column {columns : Nat}
    (run : DecodedGeometricRun columns) (offset : Fin run.length) :
    Fin columns :=
  ⟨run.columnStart + offset.val,
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left offset.isLt run.columnStart)
      run.endBound⟩

structure DecodedPort (columns : Nat) where
  explicit : List (DecodedTerm columns)
  geometric : List (DecodedGeometricRun columns)

structure DecodedEmittedRow where
  rows : Nat
  columns : Nat
  rowsPositive : 0 < rows
  columnsPositive : 0 < columns
  emittedRow : Fin rows
  runIndex : Nat
  family : RawFamily
  arm : Option Nat
  ports : Fin selectivePortCount → DecodedPort columns

def DecodedEmittedRow.port (row : DecodedEmittedRow)
    (port : Fin selectivePortCount) : DecodedPort row.columns :=
  row.ports port

structure DecodedSourceRow where
  rows : Nat
  columns : Nat
  rowsPositive : 0 < rows
  columnsPositive : 0 < columns
  sourceRow : Fin rows
  a : List (DecodedTerm columns)
  b : List (DecodedTerm columns)
  c : List (DecodedTerm columns)

structure DecodedLinearCombination (columns : Nat) where
  constant : F
  terms : List (DecodedTerm columns)

structure DecodedProductFactor (columns : Nat) where
  left : DecodedLinearCombination columns
  right : DecodedLinearCombination columns
  coefficient : F

inductive DecodedRewriteOutput (columns : Nat) where
  | source (value : DecodedLinearCombination columns)
  | derivedProductSum (compilerIndex : Nat)

def decodeTerm (columns : Nat) (raw : RawTerm) :
    Option (DecodedTerm columns) :=
  if columnInRange : raw.column < columns then do
    let coefficient ← decodeField raw.coefficient
    if coefficientNonzero : coefficient ≠ 0 then
      pure
        { column := ⟨raw.column, columnInRange⟩
          coefficient
          coefficientNonzero }
    else
      none
  else
    none

def canonicalDecodedTerm (columns column coefficient : Nat)
    (columnInRange : column < columns)
    (coefficientCanonical : coefficient < goldilocksModulus)
    (coefficientNonzero :
      (⟨coefficient, coefficientCanonical⟩ : F) ≠ 0) :
    DecodedTerm columns :=
  { column := ⟨column, columnInRange⟩
    coefficient := ⟨coefficient, coefficientCanonical⟩
    coefficientNonzero }

/-- A canonical nonzero raw term decodes to the corresponding typed term.
This kernel theorem keeps downstream certificates proof-free. -/
theorem decodeTerm_canonical (columns column coefficient : Nat)
    (columnInRange : column < columns)
    (coefficientCanonical : coefficient < goldilocksModulus)
    (coefficientNonzero :
      (⟨coefficient, coefficientCanonical⟩ : F) ≠ 0) :
    decodeTerm columns { column, coefficient } =
      some (canonicalDecodedTerm columns column coefficient columnInRange
        coefficientCanonical coefficientNonzero) := by
  unfold decodeTerm
  simp only
  rw [dif_pos columnInRange]
  unfold decodeField
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
    dif_pos coefficientCanonical]
  change
    (if h : (⟨coefficient, coefficientCanonical⟩ : F) ≠ 0 then
        some (canonicalDecodedTerm columns column coefficient columnInRange
          coefficientCanonical h)
      else none) =
      some (canonicalDecodedTerm columns column coefficient columnInRange
        coefficientCanonical coefficientNonzero)
  rw [dif_pos coefficientNonzero]

def decodeTerms (columns : Nat) (raw : List RawTerm) :
    Option (List (DecodedTerm columns)) :=
  raw.mapM (decodeTerm columns)

def decodeGeometricRun (columns : Nat) (raw : RawGeometricRun) :
    Option (DecodedGeometricRun columns) :=
  if lengthPositive : 0 < raw.length then
    if endBound : raw.columnStart + raw.length ≤ columns then do
      let initial ← decodeField raw.initial
      let ratio ← decodeField raw.ratio
      if initialNonzero : initial ≠ 0 then
        if ratioNonzero : ratio ≠ 0 then
          pure
            { columnStart := raw.columnStart
              length := raw.length
              lengthPositive
              endBound
              initial
              ratio
              initialNonzero
              ratioNonzero }
        else
          none
      else
        none
    else
      none
  else
    none

def decodePort (columns : Nat) (raw : RawPort) :
    Option (DecodedPort columns) := do
  let explicit ← decodeTerms columns raw.explicit
  let geometric ← raw.geometric.mapM (decodeGeometricRun columns)
  pure { explicit, geometric }

/-- Decode one compact final row without dropping or merging contributions. -/
def decodeEmittedRow (raw : RawEmittedRow) : Option DecodedEmittedRow :=
  if version : raw.schemaVersion = supportedSchemaVersion then
    if rowsPositive : 0 < raw.rows then
      if columnsPositive : 0 < raw.columns then
        if rowInRange : raw.emittedRow < raw.rows then do
          let ports ← raw.ports.mapM (decodePort raw.columns)
          if portCount : ports.length = selectivePortCount then
            pure
              { rows := raw.rows
                columns := raw.columns
                rowsPositive
                columnsPositive
                emittedRow := ⟨raw.emittedRow, rowInRange⟩
                runIndex := raw.runIndex
                family := raw.family
                arm := raw.arm
                ports := fun port => ports.get ⟨port.val, by
                  rw [portCount]
                  exact port.isLt⟩ }
          else
            none
        else
          none
      else
        none
    else
      none
  else
    none

/-- Decode one source A/B/C row into bounded columns. -/
def decodeSourceRow (raw : RawSourceRow) : Option DecodedSourceRow :=
  if raw.schemaVersion = supportedSchemaVersion then
    if rowsPositive : 0 < raw.rows then
      if columnsPositive : 0 < raw.columns then
        if rowInRange : raw.sourceRow < raw.rows then do
          let a ← decodeTerms raw.columns raw.a
          let b ← decodeTerms raw.columns raw.b
          let c ← decodeTerms raw.columns raw.c
          pure
            { rows := raw.rows
              columns := raw.columns
              rowsPositive
              columnsPositive
              sourceRow := ⟨raw.sourceRow, rowInRange⟩
              a
              b
              c }
        else
          none
      else
        none
    else
      none
  else
    none

def decodeLinearCombination (columns : Nat) (raw : RawLinearCombination) :
    Option (DecodedLinearCombination columns) := do
  let constant ← decodeField raw.constant
  let terms ← decodeTerms columns raw.terms
  pure { constant, terms }

def decodeProductFactor (columns : Nat) (raw : RawProductFactor) :
    Option (DecodedProductFactor columns) := do
  let left ← decodeLinearCombination columns raw.left
  let right ← decodeLinearCombination columns raw.right
  let coefficient ← decodeField raw.coefficient
  pure { left, right, coefficient }

def decodeRewriteOutput (columns : Nat) :
    RawRewriteOutput → Option (DecodedRewriteOutput columns)
  | .source value => do
      let decoded ← decodeLinearCombination columns value
      pure (.source decoded)
  | .derivedProductSum compilerIndex =>
      some (.derivedProductSum compilerIndex)

def rowRangeValid (limit : Nat) (range : RawRowRange) : Prop :=
  range.start < range.stop ∧ range.stop ≤ limit

instance (limit : Nat) (range : RawRowRange) :
    Decidable (rowRangeValid limit range) :=
  inferInstanceAs (Decidable (range.start < range.stop ∧ range.stop ≤ limit))

def rawKColumnsInRange (columns : Nat) (pair : RawKColumns) : Prop :=
  pair.c0 < columns ∧ pair.c1 < columns

instance (columns : Nat) (pair : RawKColumns) :
    Decidable (rawKColumnsInRange columns pair) :=
  inferInstanceAs (Decidable (pair.c0 < columns ∧ pair.c1 < columns))

def ColumnsInRange (columns : Nat) (values : List Nat) : Prop :=
  ∀ value ∈ values, value < columns

instance (columns : Nat) (values : List Nat) :
    Decidable (ColumnsInRange columns values) := by
  unfold ColumnsInRange
  infer_instance

def KColumnsInRange (columns : Nat) (values : List RawKColumns) : Prop :=
  ∀ value ∈ values, rawKColumnsInRange columns value

instance (columns : Nat) (values : List RawKColumns) :
    Decidable (KColumnsInRange columns values) := by
  unfold KColumnsInRange
  infer_instance

def NestedKColumnsInRange (columns : Nat)
    (values : List (List RawKColumns)) : Prop :=
  ∀ output ∈ values, KColumnsInRange columns output

instance (columns : Nat) (values : List (List RawKColumns)) :
    Decidable (NestedKColumnsInRange columns values) := by
  unfold NestedKColumnsInRange
  infer_instance

structure DecodedSourceSlot where
  raw : RawSourceSlot
  sourceColumns : Nat
  finalColumns : Nat
  sourceBound : raw.column < sourceColumns
  widthPositive : 0 < raw.width
  finalBound : raw.start + raw.width ≤ finalColumns

def decodeSourceSlot (sourceColumns finalColumns : Nat)
    (raw : RawSourceSlot) : Option DecodedSourceSlot :=
  if sourceBound : raw.column < sourceColumns then
    if widthPositive : 0 < raw.width then
      if finalBound : raw.start + raw.width ≤ finalColumns then
        some
          { raw
            sourceColumns
            finalColumns
            sourceBound
            widthPositive
            finalBound }
      else none
    else none
  else none

structure DecodedSourceDefinition (columns : Nat) where
  target : Fin columns
  value : DecodedLinearCombination columns

def decodeSourceDefinition (columns : Nat) (raw : RawSourceDefinition) :
    Option (DecodedSourceDefinition columns) :=
  if targetBound : raw.target < columns then do
    let value ← decodeLinearCombination columns raw.value
    pure { target := ⟨raw.target, targetBound⟩, value }
  else
    none

structure DecodedDerivedProductSum (sourceColumns finalColumns : Nat) where
  compilerIndex : Nat
  start : Nat
  width : Nat
  widthPositive : 0 < width
  finalBound : start + width ≤ finalColumns
  factors : List (DecodedProductFactor sourceColumns)
  previous : Option Nat

def decodeDerivedProductSum (sourceColumns finalColumns : Nat)
    (raw : RawDerivedProductSum) :
    Option (DecodedDerivedProductSum sourceColumns finalColumns) :=
  if widthPositive : 0 < raw.width then
    if finalBound : raw.start + raw.width ≤ finalColumns then do
      let factors ← raw.factors.mapM (decodeProductFactor sourceColumns)
      pure
        { compilerIndex := raw.compilerIndex
          start := raw.start
          width := raw.width
          widthPositive
          finalBound
          factors
          previous := raw.previous }
    else none
  else none

structure DecodedRewriteStep (sourceColumns : Nat) where
  sourceRowCount : Nat
  finalRowCount : Nat
  emittedRow : Nat
  emittedBound : emittedRow < finalRowCount
  rewriteId : Nat
  kind : RawRewriteKind
  sourceRows : List RawRowRange
  sourceRanges : ∀ range ∈ sourceRows, rowRangeValid sourceRowCount range
  output : DecodedRewriteOutput sourceColumns
  base : DecodedLinearCombination sourceColumns
  previous : Option Nat
  factors : List (DecodedProductFactor sourceColumns)

def decodeRewriteStep (sourceRowCount sourceColumns finalRowCount : Nat)
    (raw : RawRewriteStep) : Option (DecodedRewriteStep sourceColumns) :=
  if emittedBound : raw.emittedRow < finalRowCount then
    if sourceRanges : ∀ range ∈ raw.sourceRows,
        rowRangeValid sourceRowCount range then do
      let output ← decodeRewriteOutput sourceColumns raw.output
      let base ← decodeLinearCombination sourceColumns raw.base
      let factors ← raw.factors.mapM (decodeProductFactor sourceColumns)
      pure
        { sourceRowCount
          finalRowCount
          emittedRow := raw.emittedRow
          emittedBound
          rewriteId := raw.rewriteId
          kind := raw.kind
          sourceRows := raw.sourceRows
          sourceRanges
          output
          base
          previous := raw.previous
          factors }
    else none
  else none

structure DecodedRetainedStep (sourceColumns : Nat) where
  sourceRowCount : Nat
  finalRowCount : Nat
  emittedRow : Nat
  emittedBound : emittedRow < finalRowCount
  sourceRow : Nat
  sourceBound : sourceRow < sourceRowCount
  a : DecodedLinearCombination sourceColumns
  b : DecodedLinearCombination sourceColumns
  c : DecodedLinearCombination sourceColumns

def decodeRetainedStep (sourceRowCount sourceColumns finalRowCount : Nat)
    (raw : RawRetainedStep) : Option (DecodedRetainedStep sourceColumns) :=
  if emittedBound : raw.emittedRow < finalRowCount then
    if sourceBound : raw.sourceRow < sourceRowCount then do
      let a ← decodeLinearCombination sourceColumns raw.a
      let b ← decodeLinearCombination sourceColumns raw.b
      let c ← decodeLinearCombination sourceColumns raw.c
      pure
        { sourceRowCount
          finalRowCount
          emittedRow := raw.emittedRow
          emittedBound
          sourceRow := raw.sourceRow
          sourceBound
          a
          b
          c }
    else none
  else none

def sourceResolutionValid (sourceColumns finalColumns : Nat) :
    RawSourceResolution → Prop
  | .constantOne => True
  | .direct start width _ =>
      0 < width ∧ start + width ≤ finalColumns
  | .decompositionAlias source _ start _ =>
      source < sourceColumns ∧ start < finalColumns
  | .equalityAlias source start width _ =>
      source < sourceColumns ∧ 0 < width ∧ start + width ≤ finalColumns
  | .linearDefinition => True
  | .traceEliminated => True

instance (sourceColumns finalColumns : Nat)
    (resolution : RawSourceResolution) :
    Decidable (sourceResolutionValid sourceColumns finalColumns resolution) := by
  cases resolution <;> simp only [sourceResolutionValid] <;> infer_instance

def sourceDecoderValid (sourceColumns finalColumns : Nat)
    (raw : RawSourceDecoder) : Prop :=
  raw.column < sourceColumns ∧
  sourceResolutionValid sourceColumns finalColumns raw.resolution ∧
  (raw.column = 0 ↔ raw.resolution = .constantOne)

instance (sourceColumns finalColumns : Nat) (raw : RawSourceDecoder) :
    Decidable (sourceDecoderValid sourceColumns finalColumns raw) := by
  unfold sourceDecoderValid
  infer_instance

structure DecodedSourceDecoder where
  raw : RawSourceDecoder
  sourceColumns : Nat
  finalColumns : Nat
  valid : sourceDecoderValid sourceColumns finalColumns raw

def decodeSourceDecoder (sourceColumns finalColumns : Nat)
    (raw : RawSourceDecoder) : Option DecodedSourceDecoder :=
  if valid : sourceDecoderValid sourceColumns finalColumns raw then
    some { raw, sourceColumns, finalColumns, valid }
  else
    none

def mappedKColumns (columnMap : List Nat) (localColumns : Nat × Nat) :
    RawKColumns where
  c0 := columnMap.getD localColumns.1 0
  c1 := columnMap.getD localColumns.2 0

def expectedCoefficientColumns (columnMap : List Nat) : List RawKColumns :=
  (List.range roundCoefficientCount).map fun index =>
    mappedKColumns columnMap
      (2 + index, roundCoefficientCount + 3 + index)

def expectedAllocatedColumns (columnMap : List Nat) : List Nat :=
  columnMap.getD (2 * roundCoefficientCount + 4) 0 ::
    (List.range (isolatedRoundAllocatedCount - 1)).map fun offset =>
      columnMap.getD (2 * roundCoefficientCount + 6 + offset) 0

def expectedChallengeColumns (columnMap : List Nat) : RawKColumns :=
  mappedKColumns columnMap
    (2 * roundCoefficientCount + 3, 2 * roundCoefficientCount + 5)

def expectedClaimInColumns (columnMap : List Nat) : RawKColumns :=
  mappedKColumns columnMap (1, roundCoefficientCount + 2)

def expectedClaimOutColumns (columnMap : List Nat) : RawKColumns :=
  mappedKColumns columnMap
    (isolatedRoundColumnCount - 2, isolatedRoundColumnCount - 1)

def roundMapValid (raw : RawRoundMap) : Prop :=
  raw.schemaVersion = supportedSchemaVersion ∧
  0 < raw.sourceRows ∧
  0 < raw.sourceColumns ∧
  raw.columnMap.getD 0 0 = 0 ∧
  raw.roundIndex < sumcheckRoundCount ∧
  rowRangeValid raw.sourceRows raw.rowRange ∧
  raw.rowRange.stop - raw.rowRange.start = isolatedRoundRowCount ∧
  raw.firstAllocatedColumn < raw.sourceColumns ∧
  raw.allocatedColumns.length = isolatedRoundAllocatedCount ∧
  raw.allocatedColumns.Nodup ∧
  ColumnsInRange raw.sourceColumns raw.allocatedColumns ∧
  raw.firstAllocatedColumn ∈ raw.allocatedColumns ∧
  raw.coefficientColumns.length = roundCoefficientCount ∧
  KColumnsInRange raw.sourceColumns raw.coefficientColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.challengeColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.claimInColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.claimOutColumns ∧
  raw.columnMap.length = isolatedRoundColumnCount ∧
  raw.columnMap.Nodup ∧
  ColumnsInRange raw.sourceColumns raw.columnMap ∧
  raw.allocatedColumns = expectedAllocatedColumns raw.columnMap ∧
  raw.coefficientColumns = expectedCoefficientColumns raw.columnMap ∧
  raw.challengeColumns = expectedChallengeColumns raw.columnMap ∧
  raw.claimInColumns = expectedClaimInColumns raw.columnMap ∧
  raw.claimOutColumns = expectedClaimOutColumns raw.columnMap

instance (raw : RawRoundMap) : Decidable (roundMapValid raw) := by
  unfold roundMapValid rowRangeValid rawKColumnsInRange
    ColumnsInRange KColumnsInRange expectedAllocatedColumns
    expectedCoefficientColumns expectedChallengeColumns
    expectedClaimInColumns expectedClaimOutColumns mappedKColumns
  infer_instance

structure DecodedRoundMap where
  raw : RawRoundMap
  valid : roundMapValid raw

def decodeRoundMap (raw : RawRoundMap) : Option DecodedRoundMap :=
  if valid : roundMapValid raw then some ⟨raw, valid⟩ else none

def OrderedDisjointRanges (ranges : List RawRowRange) : Prop :=
  ranges.Pairwise fun left right => left.stop ≤ right.start

instance (ranges : List RawRowRange) : Decidable (OrderedDisjointRanges ranges) := by
  unfold OrderedDisjointRanges
  infer_instance

def boundaryMapValid (raw : RawBoundaryMap) : Prop :=
  raw.schemaVersion = supportedSchemaVersion ∧
  0 < raw.sourceRows ∧
  0 < raw.sourceColumns ∧
  raw.constantOneColumn = 0 ∧
  rowRangeValid raw.sourceRows raw.claimedInitialRows ∧
  rowRangeValid raw.sourceRows raw.terminalIdentityRows ∧
  rowRangeValid raw.sourceRows raw.terminalFinalEqualityRows ∧
  raw.claimedInitialRows.stop ≤ raw.terminalIdentityRows.start ∧
  raw.terminalIdentityRows.stop ≤ raw.terminalFinalEqualityRows.start ∧
  raw.terminalFinalEqualityRows.stop - raw.terminalFinalEqualityRows.start = 2 ∧
  raw.outputYZcolPaddingRows.length = outputCount ∧
  (∀ range ∈ raw.outputYZcolPaddingRows,
    rowRangeValid raw.sourceRows range ∧
      range.stop - range.start = outputPaddingRowsPerOutput) ∧
  OrderedDisjointRanges raw.outputYZcolPaddingRows ∧
  rawKColumnsInRange raw.sourceColumns raw.gammaColumns ∧
  raw.betaLaneColumns.length = laneBitCount ∧
  KColumnsInRange raw.sourceColumns raw.betaLaneColumns ∧
  raw.betaBlockColumns.length = blockBitCount ∧
  KColumnsInRange raw.sourceColumns raw.betaBlockColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.producerBetaColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.batchWeightColumns ∧
  raw.pendingOldBlockColumns.length = blockBitCount ∧
  KColumnsInRange raw.sourceColumns raw.pendingOldBlockColumns ∧
  raw.pendingParentYZcolColumns.length = activeLaneCount ∧
  KColumnsInRange raw.sourceColumns raw.pendingParentYZcolColumns ∧
  raw.outputYZcolColumns.length = outputCount ∧
  (∀ output ∈ raw.outputYZcolColumns,
    output.length = paddedLaneCount) ∧
  NestedKColumnsInRange raw.sourceColumns raw.outputYZcolColumns ∧
  raw.blockPointColumns.length = blockBitCount ∧
  KColumnsInRange raw.sourceColumns raw.blockPointColumns ∧
  raw.lanePointColumns.length = laneBitCount ∧
  KColumnsInRange raw.sourceColumns raw.lanePointColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.claimedInitialColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.finalSumColumns ∧
  rawKColumnsInRange raw.sourceColumns raw.terminalRhsColumns

instance (raw : RawBoundaryMap) : Decidable (boundaryMapValid raw) := by
  unfold boundaryMapValid rowRangeValid rawKColumnsInRange
    KColumnsInRange NestedKColumnsInRange OrderedDisjointRanges
  infer_instance

structure DecodedBoundaryMap where
  raw : RawBoundaryMap
  valid : boundaryMapValid raw

def decodeBoundaryMap (raw : RawBoundaryMap) : Option DecodedBoundaryMap :=
  if valid : boundaryMapValid raw then some ⟨raw, valid⟩ else none

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.Decoder
