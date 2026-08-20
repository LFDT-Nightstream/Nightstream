import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafScheduleModel
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-! Structural validation of the exact full-layout terminal Nebula finalizer artifact. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFullFinalizerLeafScheduleCertificate
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

private theorem range_getD_eq
    (start count index fallback : Nat) (bounded : index < count) :
    (List.range' start count).getD index fallback = start + index := by
  have inRange : index < (List.range' start count).length := by
    simpa using bounded
  rw [← List.getElem_eq_getD fallback]
  exact List.getElem_range'_1 index inRange

private theorem map_range_getD_eq
    (f : Nat → Nat) (count index fallback : Nat)
    (bounded : index < count) :
    ((List.range count).map f).getD index fallback = f index := by
  have mappedBound : index < ((List.range count).map f).length := by
    simpa using bounded
  rw [← List.getElem_eq_getD fallback]
  rw [List.getElem_map, List.getElem_range]
  exact mappedBound

/-- Exact geometry and verifier-owned constants for one terminal commitment
leaf. The carried digest is an output of the checked Poseidon2 recipe. -/
structure LeafValid
    (leaf : LeafHashArtifact) (authoritativeColumns : List Nat)
    (phaseStart phaseStop : Nat) : Prop where
  starts : leaf.prefixPinRowStart = phaseStart
  prefixCount : leaf.prefixConstantValues.length = leafPrefixFields
  prefixCanonical :
    ∀ value ∈ leaf.prefixConstantValues, value < goldilocksP
  primarySources :
    leaf.primary.sourceColumns =
      List.range' leaf.prefixConstantStartColumn leafPrefixFields ++
        authoritativeColumns
  primaryMetadataRows :
    leaf.primary.metadataPinRowStart =
      leaf.prefixPinRowStart + leafPrefixFields
  primaryMetadata : leaf.primary.metadataValues = [54, 2]
  primaryMetadataCanonical :
    ∀ value ∈ leaf.primary.metadataValues, value < goldilocksP
  primaryMetadataColumns :
    leaf.primary.metadataStartColumn =
      leaf.prefixConstantStartColumn + leafPrefixFields
  primaryOpeningRows :
    leaf.primary.openingRowStart = leaf.primary.metadataPinRowStart + 2
  primaryRowStart :
    leaf.primary.block.rowStart =
      leaf.primary.openingRowStart +
        canonicalOpeningRows * leafPrimaryFields
  primaryWordStartCount :
    leaf.primary.block.wordStarts.length = leafPrimaryFields
  primaryWordStart :
    ∀ index, index < leafPrimaryFields →
      leaf.primary.wordStart index =
        leaf.primary.metadataStartColumn + 2 + leafPrimaryOutputs +
          canonicalOpeningColumns * index
  primaryShape :
    leaf.primary.block.wordWidth = balancedTernaryDigits ∧
      leaf.primary.block.kappa = 2 ∧
      leaf.primary.block.messageCols = 745 ∧
      leaf.primary.block.outputColumns =
        List.range' (leaf.primary.metadataStartColumn + 2)
          leafPrimaryOutputs
  primarySchedule :
    leaf.primary.block.schedule = expectedPrimarySchedule
  compressionSources :
    leaf.compression.sourceColumns = leaf.primary.block.outputColumns
  compressionMetadataRows :
    leaf.compression.metadataPinRowStart =
      leaf.primary.block.rowStart + leafPrimaryOutputs
  compressionMetadata : leaf.compression.metadataValues = [54, 1]
  compressionMetadataCanonical :
    ∀ value ∈ leaf.compression.metadataValues, value < goldilocksP
  compressionMetadataColumns :
    leaf.compression.metadataStartColumn =
      leaf.primary.wordStart (leafPrimaryFields - 1) +
        canonicalOpeningColumns
  compressionOpeningRows :
    leaf.compression.openingRowStart =
      leaf.compression.metadataPinRowStart + 2
  compressionRowStart :
    leaf.compression.block.rowStart =
      leaf.compression.openingRowStart +
        canonicalOpeningRows * leafCompressionFields
  compressionWordStartCount :
    leaf.compression.block.wordStarts.length = leafCompressionFields
  compressionWordStart :
    ∀ index, index < leafCompressionFields →
      leaf.compression.wordStart index =
        leaf.compression.metadataStartColumn + 2 +
          leafCompressionOutputs + canonicalOpeningColumns * index
  compressionShape :
    leaf.compression.block.wordWidth = balancedTernaryDigits ∧
      leaf.compression.block.kappa = 1 ∧
      leaf.compression.block.messageCols = 82 ∧
      leaf.compression.block.outputColumns =
        List.range' (leaf.compression.metadataStartColumn + 2)
          leafCompressionOutputs
  compressionSchedule :
    leaf.compression.block.schedule = expectedCompressionSchedule
  envelopeCount :
    leaf.envelopeConstantValues.length = leafEnvelopeConstantFields
  envelopeCanonical :
    ∀ value ∈ leaf.envelopeConstantValues, value < goldilocksP
  envelopeRows :
    leaf.digestRowStart = leaf.compression.block.rowStart +
      leafCompressionOutputs + leafEnvelopeConstantFields
  digestInputs :
    leaf.digestInputColumns =
      List.range' leaf.envelopeConstantStartColumn
        leafEnvelopeConstantFields ++
      leaf.compression.block.outputColumns
  digestOutputs : leaf.digestOutputColumns.length = digestFields
  digestOutputExact :
    leaf.envelopeRecipe.outputColumns =
      (leaf.envelopeRecipe.callOutputColumns
        leaf.envelopeRecipe.absorbRounds).take digestFields
  stops : leaf.digestRowStop = phaseStop

structure FullValid : Prop where
  schemaVersion : rawArtifact.schemaVersion = 3
  profileId : rawArtifact.profileId =
    "nightstream/goldilocks/streaming-terminal-full-finalizer/v1"
  sourceIdentity : rawArtifact.sourceIdentity =
    "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1"
  legacyShaNonAuthoritative : rawArtifact.sourceRowsSha256 = ""
  lifecycleScope : lifecycleScope = "recursive-terminal-arm-435"
  columnCount : rawArtifact.columnCount = 28863843
  shapeRows : rawArtifact.shapeRowStop - rawArtifact.shapeRowStart = shapeFields
  shapeColumns : rawArtifact.shapeColumns =
    [28038961, 28038962, 28039935, 28039936, 28040909, 28040910]
  commitmentDimension : rawArtifact.dimension = 54
  commitmentKappa : rawArtifact.kappa = 18
  stepsPerSegment : rawArtifact.stepsPerSegment = 1
  segmentMaximum : rawArtifact.segmentMaximum = 1
  stackCount : rawArtifact.stackCount = 0
  stackPointerBits : rawArtifact.stackPointerBits = 0
  coreRows : rawArtifact.coreRowStop - rawArtifact.coreRowStart = rawArtifact.coreRowCount
  coreRowCount : rawArtifact.coreRowCount = 475837
  internalColumnStart : rawArtifact.internalColumnStart = 28397539
  laneColumns : rawArtifact.laneColumns = List.range' 28041931 laneFields
  payloadColumns :
    rawArtifact.payloadColumns = List.range' 28041985 delayedPayloadFields
  opsColumns :
    rawArtifact.opsColumns = List.range' 28038963 commitmentDataFields
  isColumns :
    rawArtifact.isColumns = List.range' 28039937 commitmentDataFields
  fsColumns :
    rawArtifact.fsColumns = List.range' 28040911 commitmentDataFields
  vkFsCount : rawArtifact.vkFsColumns.length = digestFields
  boundaryCount : rawArtifact.boundaryColumns.length = digestFields
  accumulatorCount : rawArtifact.accumulatorColumns.length = digestFields
  decodeRows : rawArtifact.decodeRowStop = 2974
  openRows : rawArtifact.openRowStop = 30200
  leafRows : rawArtifact.leavesRowStop = 466661
  advanceRows : rawArtifact.advanceRowStop = 475746
  closeRows : rawArtifact.closeRowStop = 475836
  terminalClosedRow : rawArtifact.coreRowCount = rawArtifact.closeRowStop + 1
  decodeProgramRows :
    rawArtifact.decodeRows.length = rawArtifact.decodeRowStop
  openAlgebraStart :
    rawArtifact.openAlgebraRowStop =
      rawArtifact.decodeRowStop + openAlgebraRowCount
  openInternalStart :
    rawArtifact.openInternalColumnStart =
      rawArtifact.internalColumnStart + stepWordCount + 1 + dPreWordCount
  openAlgebraProgramRows :
    rawArtifact.openAlgebraRows.length = openAlgebraRowCount
  stagedDigestConstantCount :
    rawArtifact.stagedDigestConstantValues.length = stagedDigestConstantFields
  stagedDigestConstantsCanonical :
    ∀ value ∈ rawArtifact.stagedDigestConstantValues, value < goldilocksP
  stagedDigestInputCount :
    rawArtifact.stagedDigestInputColumns.length = stagedDigestInputFields
  stagedDigestOutputCount :
    rawArtifact.stagedDigestOutputColumns.length = digestFields
  stagedDigestInputOrder :
    rawArtifact.stagedDigestInputColumns = rawArtifact.stagedDigestExpectedInputs
  stagedDigestConstantRowsStart :
    rawArtifact.coreRowStart + rawArtifact.openAlgebraRowStop =
      rawArtifact.stagedDigestRowStart - stagedDigestConstantFields
  stagedDigestTraceRows :
    rawArtifact.stagedDigestRowStop - rawArtifact.stagedDigestRowStart = 9660
  gammaTranscriptStartsAfterDigest :
    rawArtifact.gammaTranscriptRowStart = rawArtifact.stagedDigestRowStop
  gammaTranscriptStopsBeforeMuxes :
    rawArtifact.gammaTranscriptRowStop =
      rawArtifact.coreRowStart + rawArtifact.openRowStop - 16
  gammaTranscriptPinCount :
    rawArtifact.gammaTranscriptPinRows.length = 84
  gammaTranscriptPinColumnCount :
    rawArtifact.gammaTranscriptPinColumns.length = 84
  gammaTranscriptPinValueCount :
    rawArtifact.gammaTranscriptPinValues.length = 84
  gammaTranscriptCallCount :
    rawArtifact.gammaTranscriptCalls.length = 29
  gammaTranscriptRowPartition :
    rawArtifact.gammaTranscriptRowStop -
        rawArtifact.gammaTranscriptRowStart =
      rawArtifact.gammaTranscriptPinRows.length +
        600 * rawArtifact.gammaTranscriptCalls.length
  gammaTranscriptInitialAbsorbed :
    rawArtifact.gammaTranscriptInitialAbsorbed = 3
  gamma1Count : rawArtifact.gamma1Columns.length = 2
  gamma2Count : rawArtifact.gamma2Columns.length = 2
  gammaMuxSelector :
    rawArtifact.gammaMuxSelectorColumn = rawArtifact.openColumn
  gammaMuxOpenedOrder :
    rawArtifact.gammaMuxOpenedColumns =
      rawArtifact.gammaMuxExpectedOpenedColumns
  gammaMuxCarriedOrder :
    rawArtifact.gammaMuxCarriedColumns =
      rawArtifact.gammaMuxExpectedCarriedColumns
  gammaMuxOutputCount : rawArtifact.gammaMuxOutputColumns.length = 16
  opsLeafStart : rawArtifact.opsLeaf.prefixPinRowStart =
    rawArtifact.coreRowStart + rawArtifact.openRowStop
  opsLeafPrefixCount :
    rawArtifact.opsLeaf.prefixConstantValues.length = leafPrefixFields
  opsLeafPrimarySources :
    rawArtifact.opsLeaf.primary.sourceColumns =
      List.range' rawArtifact.opsLeaf.prefixConstantStartColumn
        leafPrefixFields ++ rawArtifact.opsColumns
  opsLeafPrimaryMetadataRows :
    rawArtifact.opsLeaf.primary.metadataPinRowStart =
      rawArtifact.opsLeaf.prefixPinRowStart + leafPrefixFields
  opsLeafPrimaryMetadata :
    rawArtifact.opsLeaf.primary.metadataValues = [54, 2]
  opsLeafPrimaryMetadataColumns :
    rawArtifact.opsLeaf.primary.metadataStartColumn =
      rawArtifact.opsLeaf.prefixConstantStartColumn + leafPrefixFields
  opsLeafPrimaryOpeningRows :
    rawArtifact.opsLeaf.primary.openingRowStart =
      rawArtifact.opsLeaf.primary.metadataPinRowStart + 2
  opsLeafPrimaryRowStart :
    rawArtifact.opsLeaf.primary.block.rowStart =
      rawArtifact.opsLeaf.primary.openingRowStart +
        canonicalOpeningRows * leafPrimaryFields
  opsLeafPrimaryWordStartCount :
    rawArtifact.opsLeaf.primary.block.wordStarts.length = leafPrimaryFields
  opsLeafPrimaryWordStart :
    ∀ index, index < leafPrimaryFields →
      rawArtifact.opsLeaf.primary.wordStart index =
        rawArtifact.opsLeaf.primary.metadataStartColumn + 2 +
          leafPrimaryOutputs + canonicalOpeningColumns * index
  opsLeafPrimaryShape :
    rawArtifact.opsLeaf.primary.block.wordWidth = balancedTernaryDigits ∧
      rawArtifact.opsLeaf.primary.block.kappa = 2 ∧
      rawArtifact.opsLeaf.primary.block.messageCols = 745 ∧
      rawArtifact.opsLeaf.primary.block.outputColumns =
        List.range' (rawArtifact.opsLeaf.primary.metadataStartColumn + 2)
          leafPrimaryOutputs
  opsLeafPrimarySchedule :
    rawArtifact.opsLeaf.primary.block.schedule = expectedPrimarySchedule
  opsLeafCompressionSources :
    rawArtifact.opsLeaf.compression.sourceColumns =
      rawArtifact.opsLeaf.primary.block.outputColumns
  opsLeafCompressionMetadataRows :
    rawArtifact.opsLeaf.compression.metadataPinRowStart =
      rawArtifact.opsLeaf.primary.block.rowStart + leafPrimaryOutputs
  opsLeafCompressionMetadata :
    rawArtifact.opsLeaf.compression.metadataValues = [54, 1]
  opsLeafCompressionMetadataColumns :
    rawArtifact.opsLeaf.compression.metadataStartColumn =
      rawArtifact.opsLeaf.primary.wordStart (leafPrimaryFields - 1) +
        canonicalOpeningColumns
  opsLeafCompressionOpeningRows :
    rawArtifact.opsLeaf.compression.openingRowStart =
      rawArtifact.opsLeaf.compression.metadataPinRowStart + 2
  opsLeafCompressionRowStart :
    rawArtifact.opsLeaf.compression.block.rowStart =
      rawArtifact.opsLeaf.compression.openingRowStart +
        canonicalOpeningRows * leafCompressionFields
  opsLeafCompressionWordStartCount :
    rawArtifact.opsLeaf.compression.block.wordStarts.length =
      leafCompressionFields
  opsLeafCompressionWordStart :
    ∀ index, index < leafCompressionFields →
      rawArtifact.opsLeaf.compression.wordStart index =
        rawArtifact.opsLeaf.compression.metadataStartColumn + 2 +
          leafCompressionOutputs + canonicalOpeningColumns * index
  opsLeafCompressionShape :
    rawArtifact.opsLeaf.compression.block.wordWidth = balancedTernaryDigits ∧
      rawArtifact.opsLeaf.compression.block.kappa = 1 ∧
      rawArtifact.opsLeaf.compression.block.messageCols = 82 ∧
      rawArtifact.opsLeaf.compression.block.outputColumns =
        List.range' (rawArtifact.opsLeaf.compression.metadataStartColumn + 2)
          leafCompressionOutputs
  opsLeafCompressionSchedule :
    rawArtifact.opsLeaf.compression.block.schedule =
      expectedCompressionSchedule
  opsLeafEnvelopeCount :
    rawArtifact.opsLeaf.envelopeConstantValues.length =
      leafEnvelopeConstantFields
  opsLeafEnvelopeRows :
    rawArtifact.opsLeaf.digestRowStart =
      rawArtifact.opsLeaf.compression.block.rowStart +
        leafCompressionOutputs + leafEnvelopeConstantFields
  opsLeafDigestInputs :
    rawArtifact.opsLeaf.digestInputColumns =
      List.range' rawArtifact.opsLeaf.envelopeConstantStartColumn
        leafEnvelopeConstantFields ++
      rawArtifact.opsLeaf.compression.block.outputColumns
  opsLeafDigestOutputs :
    rawArtifact.opsLeaf.digestOutputColumns.length = digestFields
  opsLeafStop : rawArtifact.opsLeaf.digestRowStop =
    rawArtifact.isLeaf.prefixPinRowStart
  openedLaneCount : rawArtifact.openedLaneColumns.length = laneFields
  advancedLaneCount : rawArtifact.advancedLaneColumns.length = laneFields
  advanceAlgebraCount : rawArtifact.advanceAlgebraRows.length = 19
  closeCount : rawArtifact.closeRows.length = 90
  terminalClosedIndex : rawArtifact.terminalClosedRow.1 =
    rawArtifact.coreRowStart + rawArtifact.closeRowStop
  finalLaneCount : rawArtifact.finalLaneColumns.length = laneFields
  programBindingPreserved :
    rawArtifact.finalLaneColumns.take digestFields =
      rawArtifact.laneColumns.take digestFields
  finalColumn : rawArtifact.finalLaneColumns.getD (laneFields - 1) 0 + 1 =
    rawArtifact.columnCount
  closedColumnBound : rawArtifact.closedColumn < rawArtifact.columnCount

theorem rawArtifact_valid : FullValid := by
  refine {
    schemaVersion := rfl
    profileId := rfl
    sourceIdentity := rfl
    legacyShaNonAuthoritative := rfl
    lifecycleScope := rfl
    columnCount := rfl
    shapeRows := by decide
    shapeColumns := rfl
    commitmentDimension := rfl
    commitmentKappa := rfl
    stepsPerSegment := rfl
    segmentMaximum := rfl
    stackCount := rfl
    stackPointerBits := rfl
    coreRows := by decide
    coreRowCount := rfl
    internalColumnStart := rfl
    laneColumns := rfl
    payloadColumns := rfl
    opsColumns := rfl
    isColumns := rfl
    fsColumns := rfl
    vkFsCount := rfl
    boundaryCount := rfl
    accumulatorCount := rfl
    decodeRows := rfl
    openRows := rfl
    leafRows := rfl
    advanceRows := rfl
    closeRows := rfl
    terminalClosedRow := by decide
    decodeProgramRows := by
      norm_num [RawArtifact.decodeRows, RawArtifact.decodePieces,
        RawArtifact.stepBitRows, RawArtifact.stepBitColumns,
        RawArtifact.stepWordRows, RawArtifact.constantZeroRow,
        RawArtifact.openBitRows, RawArtifact.dPreBitRows,
        RawArtifact.dPreBitColumns, RawArtifact.inactiveDPreRows,
        RawArtifact.dPreWordRows, stepWordCount, stepWordWidths,
        dPreWordCount, stepBitFields, openBitIndex, dPreBitStart, dPreBitFields,
        rawArtifact]
    openAlgebraStart := by decide
    openInternalStart := by decide
    openAlgebraProgramRows := by decide
    stagedDigestConstantCount := rfl
    stagedDigestConstantsCanonical := by
      norm_num [rawArtifact, goldilocksP]
    stagedDigestInputCount := rfl
    stagedDigestOutputCount := rfl
    stagedDigestInputOrder := rfl
    stagedDigestConstantRowsStart := by decide
    stagedDigestTraceRows := by decide
    gammaTranscriptStartsAfterDigest := rfl
    gammaTranscriptStopsBeforeMuxes := by decide
    gammaTranscriptPinCount := rfl
    gammaTranscriptPinColumnCount := rfl
    gammaTranscriptPinValueCount := rfl
    gammaTranscriptCallCount := rfl
    gammaTranscriptRowPartition := by decide
    gammaTranscriptInitialAbsorbed := rfl
    gamma1Count := rfl
    gamma2Count := rfl
    gammaMuxSelector := by
      change 28043385 = (List.range' 28041985 2169).getD 1400 0
      rw [range_getD_eq 28041985 2169 1400 0 (by decide)]
    gammaMuxOpenedOrder := rfl
    gammaMuxCarriedOrder := rfl
    gammaMuxOutputCount := rfl
    opsLeafStart := by decide
    opsLeafPrefixCount := rfl
    opsLeafPrimarySources := rfl
    opsLeafPrimaryMetadataRows := by decide
    opsLeafPrimaryMetadata := rfl
    opsLeafPrimaryMetadataColumns := by decide
    opsLeafPrimaryOpeningRows := by decide
    opsLeafPrimaryRowStart := by decide
    opsLeafPrimaryWordStartCount := by
      simp only [rawArtifact, opsLeaf, List.length_map, List.length_range]
      decide
    opsLeafPrimaryWordStart := by
      intro index bounded
      have bounded' : index < 981 := by
        simpa [leafPrimaryFields, leafPrefixFields, commitmentDataFields]
          using bounded
      simp only [SeededBindingArtifact.wordStart, rawArtifact, opsLeaf]
      rw [map_range_getD_eq _ _ _ _ bounded']
      norm_num [leafPrimaryOutputs, canonicalOpeningColumns,
        balancedTernaryDigits]
    opsLeafPrimaryShape := by
      refine ⟨rfl, rfl, rfl, ?_⟩
      rfl
    opsLeafPrimarySchedule := primary_schedule_exact
    opsLeafCompressionSources := rfl
    opsLeafCompressionMetadataRows := by decide
    opsLeafCompressionMetadata := rfl
    opsLeafCompressionMetadataColumns := by
      simp only [SeededBindingArtifact.wordStart, rawArtifact, opsLeaf]
      rw [map_range_getD_eq _ _ _ _ (by decide)]
      norm_num [leafPrimaryFields, leafPrefixFields, commitmentDataFields,
        canonicalOpeningColumns, balancedTernaryDigits]
    opsLeafCompressionOpeningRows := by decide
    opsLeafCompressionRowStart := by decide
    opsLeafCompressionWordStartCount := by
      simp only [rawArtifact, opsLeaf, List.length_map, List.length_range]
      decide
    opsLeafCompressionWordStart := by
      intro index bounded
      have bounded' : index < 108 := by
        simpa [leafCompressionFields, leafPrimaryOutputs] using bounded
      simp only [SeededBindingArtifact.wordStart, rawArtifact, opsLeaf]
      rw [map_range_getD_eq _ _ _ _ bounded']
      norm_num [leafCompressionOutputs, canonicalOpeningColumns,
        balancedTernaryDigits]
    opsLeafCompressionShape := by
      refine ⟨rfl, rfl, rfl, ?_⟩
      rfl
    opsLeafCompressionSchedule := compression_schedule_exact
    opsLeafEnvelopeCount := rfl
    opsLeafEnvelopeRows := by decide
    opsLeafDigestInputs := rfl
    opsLeafDigestOutputs := rfl
    opsLeafStop := rfl
    openedLaneCount := rfl
    advancedLaneCount := rfl
    advanceAlgebraCount := rfl
    closeCount := rfl
    terminalClosedIndex := rfl
    finalLaneCount := rfl
    programBindingPreserved := rfl
    finalColumn := by decide
    closedColumnBound := by decide }

private theorem leaf_absorbRounds_exact
    (leaf : LeafHashArtifact)
    (inputLength : leaf.digestInputColumns.length = 64) :
    leaf.envelopeRecipe.absorbRounds = 16 := by
  change (leaf.digestInputColumns.length + (rate - 1)) / rate = 16
  rw [inputLength]
  decide

theorem isLeaf_valid :
    LeafValid rawArtifact.isLeaf rawArtifact.isColumns
      rawArtifact.opsLeaf.digestRowStop rawArtifact.fsLeaf.prefixPinRowStart := by
  have absorbRounds : rawArtifact.isLeaf.envelopeRecipe.absorbRounds = 16 :=
    leaf_absorbRounds_exact rawArtifact.isLeaf (by rfl)
  refine {
    starts := rfl
    prefixCount := rfl
    prefixCanonical := by
      norm_num [rawArtifact, isLeaf, goldilocksP]
    primarySources := rfl
    primaryMetadataRows := by decide
    primaryMetadata := rfl
    primaryMetadataCanonical := by
      norm_num [rawArtifact, isLeaf, goldilocksP]
    primaryMetadataColumns := by decide
    primaryOpeningRows := by decide
    primaryRowStart := by decide
    primaryWordStartCount := by
      simp only [rawArtifact, isLeaf, List.length_map, List.length_range]
      decide
    primaryWordStart := ?_
    primaryShape := by
      refine ⟨rfl, rfl, rfl, ?_⟩
      rfl
    primarySchedule := primary_schedule_exact
    compressionSources := rfl
    compressionMetadataRows := by decide
    compressionMetadata := rfl
    compressionMetadataCanonical := by
      norm_num [rawArtifact, isLeaf, goldilocksP]
    compressionMetadataColumns := ?_
    compressionOpeningRows := by decide
    compressionRowStart := by decide
    compressionWordStartCount := by
      simp only [rawArtifact, isLeaf, List.length_map, List.length_range]
      decide
    compressionWordStart := ?_
    compressionShape := by
      refine ⟨rfl, rfl, rfl, ?_⟩
      rfl
    compressionSchedule := compression_schedule_exact
    envelopeCount := rfl
    envelopeCanonical := by
      norm_num [rawArtifact, isLeaf, goldilocksP]
    envelopeRows := by decide
    digestInputs := rfl
    digestOutputs := rfl
    digestOutputExact := ?_
    stops := rfl }
  · intro index bounded
    have bounded' : index < 981 := by
      simpa [leafPrimaryFields, leafPrefixFields, commitmentDataFields]
        using bounded
    simp only [SeededBindingArtifact.wordStart, rawArtifact, isLeaf]
    rw [map_range_getD_eq _ _ _ _ bounded']
    norm_num [leafPrimaryOutputs, canonicalOpeningColumns,
      balancedTernaryDigits]
  · simp only [SeededBindingArtifact.wordStart, rawArtifact, isLeaf]
    rw [map_range_getD_eq _ _ _ _ (by decide)]
    norm_num [leafPrimaryFields, leafPrefixFields, commitmentDataFields,
      canonicalOpeningColumns, balancedTernaryDigits]
  · intro index bounded
    have bounded' : index < 108 := by
      simpa [leafCompressionFields, leafPrimaryOutputs] using bounded
    simp only [SeededBindingArtifact.wordStart, rawArtifact, isLeaf]
    rw [map_range_getD_eq _ _ _ _ bounded']
    norm_num [leafCompressionOutputs, canonicalOpeningColumns,
      balancedTernaryDigits]
  · rw [absorbRounds]
    rfl

theorem fsLeaf_valid :
    LeafValid rawArtifact.fsLeaf rawArtifact.fsColumns
      rawArtifact.isLeaf.digestRowStop
      (rawArtifact.coreRowStart + rawArtifact.leavesRowStop) := by
  have absorbRounds : rawArtifact.fsLeaf.envelopeRecipe.absorbRounds = 16 :=
    leaf_absorbRounds_exact rawArtifact.fsLeaf (by rfl)
  refine {
    starts := rfl
    prefixCount := rfl
    prefixCanonical := by
      norm_num [rawArtifact, fsLeaf, goldilocksP]
    primarySources := rfl
    primaryMetadataRows := by decide
    primaryMetadata := rfl
    primaryMetadataCanonical := by
      norm_num [rawArtifact, fsLeaf, goldilocksP]
    primaryMetadataColumns := by decide
    primaryOpeningRows := by decide
    primaryRowStart := by decide
    primaryWordStartCount := by
      simp only [rawArtifact, fsLeaf, List.length_map, List.length_range]
      decide
    primaryWordStart := ?_
    primaryShape := by
      refine ⟨rfl, rfl, rfl, ?_⟩
      rfl
    primarySchedule := primary_schedule_exact
    compressionSources := rfl
    compressionMetadataRows := by decide
    compressionMetadata := rfl
    compressionMetadataCanonical := by
      norm_num [rawArtifact, fsLeaf, goldilocksP]
    compressionMetadataColumns := ?_
    compressionOpeningRows := by decide
    compressionRowStart := by decide
    compressionWordStartCount := by
      simp only [rawArtifact, fsLeaf, List.length_map, List.length_range]
      decide
    compressionWordStart := ?_
    compressionShape := by
      refine ⟨rfl, rfl, rfl, ?_⟩
      rfl
    compressionSchedule := compression_schedule_exact
    envelopeCount := rfl
    envelopeCanonical := by
      norm_num [rawArtifact, fsLeaf, goldilocksP]
    envelopeRows := by decide
    digestInputs := rfl
    digestOutputs := rfl
    digestOutputExact := ?_
    stops := by decide }
  · intro index bounded
    have bounded' : index < 981 := by
      simpa [leafPrimaryFields, leafPrefixFields, commitmentDataFields]
        using bounded
    simp only [SeededBindingArtifact.wordStart, rawArtifact, fsLeaf]
    rw [map_range_getD_eq _ _ _ _ bounded']
    norm_num [leafPrimaryOutputs, canonicalOpeningColumns,
      balancedTernaryDigits]
  · simp only [SeededBindingArtifact.wordStart, rawArtifact, fsLeaf]
    rw [map_range_getD_eq _ _ _ _ (by decide)]
    norm_num [leafPrimaryFields, leafPrefixFields, commitmentDataFields,
      canonicalOpeningColumns, balancedTernaryDigits]
  · intro index bounded
    have bounded' : index < 108 := by
      simpa [leafCompressionFields, leafPrimaryOutputs] using bounded
    simp only [SeededBindingArtifact.wordStart, rawArtifact, fsLeaf]
    rw [map_range_getD_eq _ _ _ _ bounded']
    norm_num [leafCompressionOutputs, canonicalOpeningColumns,
      balancedTernaryDigits]
  · rw [absorbRounds]
    rfl

def advanceChainLink (lane : Fin 3) : PoseidonHashArtifact :=
  rawArtifact.advanceChainLinks.get
    ⟨lane.val, by
      simpa [rawArtifact, advanceChainLinks] using lane.isLt⟩

theorem advance_chain_constants_canonical (lane : Fin 3) :
    ∀ value ∈ (advanceChainLink lane).recipe.constantValues,
      value < goldilocksP := by
  fin_cases lane <;> decide

theorem advance_chain_trace_ownedValid (lane : Fin 3) :
    (advanceChainLink lane).recipe.trace.OwnedValid := by
  fin_cases lane <;>
    exact ownedValid _ (by decide) (by decide) (by decide)

theorem phase_order :
    rawArtifact.decodeRowStop < rawArtifact.openRowStop ∧
      rawArtifact.openRowStop < rawArtifact.leavesRowStop ∧
      rawArtifact.leavesRowStop < rawArtifact.advanceRowStop ∧
      rawArtifact.advanceRowStop < rawArtifact.closeRowStop ∧
      rawArtifact.closeRowStop < rawArtifact.coreRowCount := by
  norm_num [rawArtifact]

theorem staged_digest_input_length :
    rawArtifact.stagedDigestRecipe.inputColumns.length =
      stagedDigestInputFields := by
  exact rawArtifact_valid.stagedDigestInputCount

theorem staged_digest_absorbRounds_exact :
    rawArtifact.stagedDigestRecipe.absorbRounds = 15 := by
  norm_num [VariableHashRecipe.absorbRounds, staged_digest_input_length,
    stagedDigestInputFields, rate]

theorem staged_digest_output_exact :
    rawArtifact.stagedDigestRecipe.outputColumns =
      (rawArtifact.stagedDigestRecipe.callOutputColumns
        rawArtifact.stagedDigestRecipe.absorbRounds).take digestFields := by
  rw [staged_digest_absorbRounds_exact]
  rfl

theorem staged_digest_trace_ownedValid :
    rawArtifact.stagedDigestRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.stagedDigestRecipe (by
      rw [staged_digest_absorbRounds_exact]
      norm_num) (by
      rw [staged_digest_input_length, staged_digest_absorbRounds_exact]
      norm_num [stagedDigestInputFields]) staged_digest_output_exact

theorem staged_digest_dPre_columns :
    (rawArtifact.stagedDigestRecipe.inputColumns.drop 30).take dPreWordCount =
      (List.range dPreWordCount).map rawArtifact.dPreWordColumn := by
  rfl

/-- Exact Rust-emitted constant-one coordinate of the opened lane. -/
theorem opened_lane_open_column :
    rawArtifact.openedLaneColumns.getD 4 0 = 0 := by
  rfl

/-- Exact Rust-emitted delayed-open input coordinate. -/
theorem open_column_exact : rawArtifact.openColumn = 28043385 := by
  calc
    rawArtifact.openColumn = rawArtifact.gammaMuxSelectorColumn :=
      rawArtifact_valid.gammaMuxSelector.symm
    _ = 28043385 := by rfl

/-- Exact Rust-emitted coordinate of each post-phase lane field. -/
theorem lane_column_exact (index : Nat) (bounded : index < laneFields) :
    rawArtifact.laneColumns.getD index 0 = 28041931 + index := by
  rw [rawArtifact_valid.laneColumns]
  exact range_getD_eq 28041931 laneFields index 0 bounded

/-- Exact Rust-emitted coordinate of each delayed-payload bit. -/
theorem payload_column_exact
    (index : Nat) (bounded : index < delayedPayloadFields) :
    rawArtifact.payloadColumn index = 28041985 + index := by
  unfold RawArtifact.payloadColumn
  rw [rawArtifact_valid.payloadColumns]
  exact range_getD_eq 28041985 delayedPayloadFields index 0 bounded

/-- The twelve gamma-mux opened digest inputs are exactly the decoded
`D_pre` words, in lane-major coordinate order. -/
theorem gamma_mux_opened_dPre_column (index : Fin dPreWordCount) :
    rawArtifact.gammaMuxOpenedColumns.getD (4 + index.val) 0 =
      rawArtifact.dPreWordColumn index.val := by
  rw [rawArtifact_valid.gammaMuxOpenedOrder]
  fin_cases index <;> rfl

/-- The four open-mux gamma outputs occupy the opened lane's gamma slice. -/
theorem opened_lane_gamma_mux_column (index : Fin 4) :
    rawArtifact.openedLaneColumns.getD (8 + index.val) 0 =
      rawArtifact.gammaMuxOutputColumns.getD index.val 0 := by
  fin_cases index <;> rfl

/-- The four carried gamma inputs are the post-phase lane's gamma slice. -/
theorem carried_lane_gamma_mux_column (index : Fin 4) :
    rawArtifact.laneColumns.getD (8 + index.val) 0 =
      rawArtifact.gammaMuxCarriedColumns.getD index.val 0 := by
  fin_cases index <;> rfl

/-- The twelve open-mux digest outputs occupy the opened lane's `dPre` slice. -/
theorem opened_lane_dPre_mux_column (index : Fin 3) (coordinate : Fin 4) :
    rawArtifact.openedLaneColumns.getD
        (22 + 4 * index.val + coordinate.val) 0 =
      rawArtifact.gammaMuxOutputColumns.getD
        (4 + 4 * index.val + coordinate.val) 0 := by
  fin_cases index <;> fin_cases coordinate <;> rfl

/-- The twelve carried digest inputs are the post-phase lane's `dPre` slice. -/
theorem carried_lane_dPre_mux_column (index : Fin 3) (coordinate : Fin 4) :
    rawArtifact.laneColumns.getD
        (22 + 4 * index.val + coordinate.val) 0 =
      rawArtifact.gammaMuxCarriedColumns.getD
        (4 + 4 * index.val + coordinate.val) 0 := by
  fin_cases index <;> fin_cases coordinate <;> rfl

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
