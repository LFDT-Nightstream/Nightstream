import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Canonical.KConcreteFixedPhaseBridge
import Nightstream.Implementation.R1CS.Core.Projection.Interpretation
import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation

/-!
Contract: typed assignment view for the exact terminal Nebula lane layouts.

Owns the structural decoder from the Rust-emitted 50-column field order to
the terminal transition model. This leaf currently certifies the opened-lane
constant-one coordinate. It does not yet claim the complete transition.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation

private abbrev artifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact

def columnAt (columns : List Nat) (index : Nat) : Nat :=
  columns.getD index 0

def fieldValue (assignment : Nat → Nat) (column : Nat) : F :=
  ⟨assignment column % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

def fieldAt
    (assignment : Nat → Nat) (columns : List Nat) (index : Nat) : F :=
  fieldValue assignment (columnAt columns index)

def boolAt
    (assignment : Nat → Nat) (columns : List Nat) (index : Nat) : Bool :=
  assignment (columnAt columns index) == 1

def digestAt
    (assignment : Nat → Nat) (columns : List Nat) (start : Nat) : Digest :=
  fun lane => fieldAt assignment columns (start + lane.val)

def kAt
    (assignment : Nat → Nat) (columns : List Nat) (start : Nat) : K where
  c0 := fieldAt assignment columns start
  c1 := fieldAt assignment columns (start + 1)

/-- Exact 50-field order shared by the Rust lane source and transition
outputs. -/
def laneAt (assignment : Nat → Nat) (columns : List Nat) : Lane where
  programBindingDigest := digestAt assignment columns 0
  isOpen := boolAt assignment columns 4
  segmentIndex := assignment (columnAt columns 5)
  stepIndex := assignment (columnAt columns 6)
  timestamp := assignment (columnAt columns 7)
  gamma := fun index => kAt assignment columns (8 + 2 * index.val)
  products := fun index => kAt assignment columns (12 + 2 * index.val)
  stackPointers := fun index => assignment (columnAt columns (20 + index.val))
  dPre := fun index => digestAt assignment columns (22 + 4 * index.val)
  dSeen := fun index => digestAt assignment columns (34 + 4 * index.val)
  dMem := digestAt assignment columns 46

def openedLane (assignment : Nat → Nat) : Lane :=
  laneAt assignment artifact.openedLaneColumns

def advancedLane (assignment : Nat → Nat) : Lane :=
  laneAt assignment artifact.advancedLaneColumns

def finalLane (assignment : Nat → Nat) : Lane :=
  laneAt assignment artifact.finalLaneColumns

def stepFieldAt (assignment : Nat → Nat) (index : Nat) : F :=
  fieldValue assignment (artifact.stepWordColumn index)

def stepKAt (assignment : Nat → Nat) (start : Nat) : K where
  c0 := stepFieldAt assignment start
  c1 := stepFieldAt assignment (start + 1)

def stepInput (assignment : Nat → Nat) : StepInput where
  segmentIndex := assignment (artifact.stepWordColumn 0)
  stepIndex := assignment (artifact.stepWordColumn 1)
  timestampIn := assignment (artifact.stepWordColumn 2)
  timestampOut := assignment (artifact.stepWordColumn 3)
  gamma := fun index => stepKAt assignment (4 + 2 * index.val)
  productsIn := fun index => stepKAt assignment (8 + 2 * index.val)
  productsOut := fun index => stepKAt assignment (16 + 2 * index.val)
  stackPointersIn := fun _ => assignment artifact.constantZeroColumn
  stackPointersOut := fun _ => assignment artifact.constantZeroColumn

/-- The 17 equality rows at the start of the exact advance-algebra block. -/
def stepMatchPairs : List (Nat × Nat) :=
  [(artifact.stepWordColumn 0, artifact.openedLaneColumns.getD 5 0),
    (artifact.stepWordColumn 1, artifact.openedLaneColumns.getD 6 0),
    (artifact.stepWordColumn 2, artifact.openedLaneColumns.getD 7 0),
    (artifact.stepWordColumn 4, artifact.openedLaneColumns.getD 8 0),
    (artifact.stepWordColumn 5, artifact.openedLaneColumns.getD 9 0),
    (artifact.stepWordColumn 6, artifact.openedLaneColumns.getD 10 0),
    (artifact.stepWordColumn 7, artifact.openedLaneColumns.getD 11 0),
    (artifact.stepWordColumn 8, artifact.openedLaneColumns.getD 12 0),
    (artifact.stepWordColumn 9, artifact.openedLaneColumns.getD 13 0),
    (artifact.stepWordColumn 10, artifact.openedLaneColumns.getD 14 0),
    (artifact.stepWordColumn 11, artifact.openedLaneColumns.getD 15 0),
    (artifact.stepWordColumn 12, artifact.openedLaneColumns.getD 16 0),
    (artifact.stepWordColumn 13, artifact.openedLaneColumns.getD 17 0),
    (artifact.stepWordColumn 14, artifact.openedLaneColumns.getD 18 0),
    (artifact.stepWordColumn 15, artifact.openedLaneColumns.getD 19 0),
    (artifact.constantZeroColumn, artifact.openedLaneColumns.getD 20 0),
    (artifact.constantZeroColumn, artifact.openedLaneColumns.getD 21 0)]

/-- Rust emits the first three and last ten equalities with the output term
last. The four gamma equalities already use builder order. -/
def outputLastRows (pairs : List (Nat × Nat)) : List Row :=
  pairs.map fun pair => outputLastLinearRow pair.1 [(pair.2, 1)]

def builderRows (pairs : List (Nat × Nat)) : List Row :=
  pairs.map fun pair => builderLinearRow pair.1 [(pair.2, 1)]

def stepMatchPrefixPairs : List (Nat × Nat) :=
  stepMatchPairs.take 3

def stepMatchGammaPairs : List (Nat × Nat) :=
  (stepMatchPairs.drop 3).take 4

def stepMatchSuffixPairs : List (Nat × Nat) :=
  stepMatchPairs.drop 7

def emittedStepMatchRows : List Row :=
  outputLastRows stepMatchPrefixPairs ++
    builderRows stepMatchGammaPairs ++
    outputLastRows stepMatchSuffixPairs

/-- Exact mixed-order 17-row slice in the Rust-emitted advance block. -/
theorem emitted_step_match_rows_exact :
    emittedStepMatchRows =
      ((indexedRowValues artifact.advanceAlgebraRows).drop 1).take 17 := by
  norm_num [emittedStepMatchRows, outputLastRows, builderRows,
    stepMatchPrefixPairs, stepMatchGammaPairs, stepMatchSuffixPairs,
    stepMatchPairs, indexedRowValues, outputLastLinearRow, builderLinearRow,
    negateTerms, negCoeff, goldilocksP, artifact,
    RawArtifact.stepWordColumn, RawArtifact.internalColumn,
    RawArtifact.constantZeroColumn, stepWordCount, stepWordWidths,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.advanceAlgebraRows]

/-- Rust represents the opened lane's open flag by the verifier-owned
constant-one column. -/
theorem openedLane_open
    (assignment : Nat → Nat)
    (one : assignment 0 = 1) :
    (openedLane assignment).isOpen = true := by
  change (assignment (artifact.openedLaneColumns.getD 4 0) == 1) = true
  rw [opened_lane_open_column, one]
  rfl

private def zeroRow : Row :=
  ⟨[], [], []⟩

/-- Output-last linear equality gated by an explicit selector column. -/
private def gatedOutputLastRow
    (selector output : Nat) (terms : List (Nat × Nat)) : Row :=
  ⟨negateTerms terms ++ [(output, 1)], [(selector, 1)], []⟩

/-- Exact Rust row that gates the terminal step-index distance by `closed`. -/
private def closeStepIndexRow : Row :=
  (indexedRowValues artifact.closeRows).getD 2 zeroRow

private def closeZeroColumn : Nat :=
  artifact.closedColumn + 13

private def closeSegmentResetColumn : Nat :=
  artifact.closedColumn + 12

private def closeOneColumn : Nat :=
  artifact.closedColumn + 14

private def closeOperationsHeaderColumn (index : Nat) : Nat :=
  artifact.closedColumn + 15 + index

private def closeMemoryHeaderColumn (index : Nat) : Nat :=
  artifact.closedColumn + 19 + index

/-- Exact Goldilocks values emitted by Rust's Poseidon2 operations-chain
header computation. -/
def operationsHeaderValues : List Nat :=
  [17244392684944334319, 15605506905146296732,
    12295989976012019387, 14371939316173953392]

/-- Exact Goldilocks values emitted by Rust's Poseidon2 memory-chain header
computation. -/
def memoryHeaderValues : List Nat :=
  [5556117167390284352, 9149922206073305624,
    6267898749355880157, 2793896763717765528]

def emittedOperationsHeader : Digest := fun index =>
  fieldValue (fun position => operationsHeaderValues.getD position 0) index.val

def emittedMemoryHeader : Digest := fun index =>
  fieldValue (fun position => memoryHeaderValues.getD position 0) index.val

private def closeZeroRow : Row :=
  (indexedRowValues artifact.closeRows).getD 35 zeroRow

private def selectedMuxRow
    (selector reset live output : Nat) : Row :=
  ⟨[(selector, 1)], [(live, negCoeff 1), (reset, 1)],
    [(live, negCoeff 1), (output, 1)]⟩

private def closeMuxFieldIndex (position : Nat) : Nat :=
  if position < 3 then 4 + position else 5 + position

private def closeMuxResetColumn (position : Nat) : Nat :=
  if position = 0 then closeZeroColumn
  else if position = 1 then closeSegmentResetColumn
  else if position = 2 then closeZeroColumn
  else if position < 15 then
    if (position - 3) % 2 = 0 then closeOneColumn else closeZeroColumn
  else if position < 17 then closeZeroColumn
  else if position < 21 then closeOperationsHeaderColumn (position - 17)
  else if position < 29 then closeMemoryHeaderColumn ((position - 21) % 4)
  else if position < 33 then closeOperationsHeaderColumn (position - 29)
  else if position < 41 then closeMemoryHeaderColumn ((position - 33) % 4)
  else artifact.advancedLaneColumns.getD (42 + (position - 41)) 0

private def emittedCloseResetRows : List Row :=
  [outputLastLinearRow closeSegmentResetColumn
      [(0, 1), (artifact.advancedLaneColumns.getD 5 0, 1)],
    outputLastLinearRow closeZeroColumn [],
    outputLastLinearRow closeOneColumn [(0, 1)],
    outputLastLinearRow (closeOperationsHeaderColumn 0)
      [(0, operationsHeaderValues.getD 0 0)],
    outputLastLinearRow (closeOperationsHeaderColumn 1)
      [(0, operationsHeaderValues.getD 1 0)],
    outputLastLinearRow (closeOperationsHeaderColumn 2)
      [(0, operationsHeaderValues.getD 2 0)],
    outputLastLinearRow (closeOperationsHeaderColumn 3)
      [(0, operationsHeaderValues.getD 3 0)],
    outputLastLinearRow (closeMemoryHeaderColumn 0)
      [(0, memoryHeaderValues.getD 0 0)],
    outputLastLinearRow (closeMemoryHeaderColumn 1)
      [(0, memoryHeaderValues.getD 1 0)],
    outputLastLinearRow (closeMemoryHeaderColumn 2)
      [(0, memoryHeaderValues.getD 2 0)],
    outputLastLinearRow (closeMemoryHeaderColumn 3)
      [(0, memoryHeaderValues.getD 3 0)]]

private def closeMuxPositions : List Nat :=
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]

private def emittedCloseMuxRows : List Row :=
  closeMuxPositions.map fun position =>
    selectedMuxRow artifact.closedColumn
      (closeMuxResetColumn position)
      (artifact.advancedLaneColumns.getD (closeMuxFieldIndex position) 0)
      (artifact.finalLaneColumns.getD (closeMuxFieldIndex position) 0)

private def finalOpenMuxRow : Row :=
  (indexedRowValues artifact.closeRows).getD 45 zeroRow

def stackPointerColumns : List Nat :=
  [artifact.advancedLaneColumns.getD 20 0,
    artifact.advancedLaneColumns.getD 21 0]

def seenPrecommitPairs : List (Nat × Nat) :=
  [(artifact.advancedLaneColumns.getD 34 0,
      artifact.advancedLaneColumns.getD 22 0),
    (artifact.advancedLaneColumns.getD 35 0,
      artifact.advancedLaneColumns.getD 23 0),
    (artifact.advancedLaneColumns.getD 36 0,
      artifact.advancedLaneColumns.getD 24 0),
    (artifact.advancedLaneColumns.getD 37 0,
      artifact.advancedLaneColumns.getD 25 0),
    (artifact.advancedLaneColumns.getD 38 0,
      artifact.advancedLaneColumns.getD 26 0),
    (artifact.advancedLaneColumns.getD 39 0,
      artifact.advancedLaneColumns.getD 27 0),
    (artifact.advancedLaneColumns.getD 40 0,
      artifact.advancedLaneColumns.getD 28 0),
    (artifact.advancedLaneColumns.getD 41 0,
      artifact.advancedLaneColumns.getD 29 0),
    (artifact.advancedLaneColumns.getD 42 0,
      artifact.advancedLaneColumns.getD 30 0),
    (artifact.advancedLaneColumns.getD 43 0,
      artifact.advancedLaneColumns.getD 31 0),
    (artifact.advancedLaneColumns.getD 44 0,
      artifact.advancedLaneColumns.getD 32 0),
    (artifact.advancedLaneColumns.getD 45 0,
      artifact.advancedLaneColumns.getD 33 0)]

def initialMemoryPairs : List (Nat × Nat) :=
  [(artifact.advancedLaneColumns.getD 38 0,
      artifact.advancedLaneColumns.getD 46 0),
    (artifact.advancedLaneColumns.getD 39 0,
      artifact.advancedLaneColumns.getD 47 0),
    (artifact.advancedLaneColumns.getD 40 0,
      artifact.advancedLaneColumns.getD 48 0),
    (artifact.advancedLaneColumns.getD 41 0,
      artifact.advancedLaneColumns.getD 49 0)]

def gatedZeroRow (selector value : Nat) : Row :=
  ⟨[(selector, 1)], [(value, 1)], []⟩

def gatedDifferenceRow
    (selector positive negative : Nat) : Row :=
  ⟨[(selector, 1)], [(negative, negCoeff 1), (positive, 1)], []⟩

def emittedStackPointerRows : List Row :=
  stackPointerColumns.map (gatedZeroRow artifact.closedColumn)

def emittedSeenPrecommitRows : List Row :=
  seenPrecommitPairs.map fun pair =>
    gatedDifferenceRow artifact.closedColumn pair.1 pair.2

def emittedInitialMemoryRows : List Row :=
  initialMemoryPairs.map fun pair =>
    gatedDifferenceRow artifact.closedColumn pair.1 pair.2

def leftProductTrace : ProjectionProgram.KMulTrace :=
  ProjectionProgram.KMulTrace.ofColumns
    ⟨artifact.advancedLaneColumns.getD 16 0,
      artifact.advancedLaneColumns.getD 17 0⟩
    ⟨artifact.advancedLaneColumns.getD 14 0,
      artifact.advancedLaneColumns.getD 15 0⟩
    ⟨artifact.closedColumn + 5, artifact.closedColumn + 6⟩

def rightProductTrace : ProjectionProgram.KMulTrace :=
  ProjectionProgram.KMulTrace.ofColumns
    ⟨artifact.advancedLaneColumns.getD 12 0,
      artifact.advancedLaneColumns.getD 13 0⟩
    ⟨artifact.advancedLaneColumns.getD 18 0,
      artifact.advancedLaneColumns.getD 19 0⟩
    ⟨artifact.closedColumn + 10, artifact.closedColumn + 11⟩

private def indexedDefinitions : Nat → List Definition →
    List (Nat × Definition)
  | _, [] => []
  | rowIndex, definition :: definitions =>
      (rowIndex, definition) :: indexedDefinitions (rowIndex + 1) definitions

def positiveFirstGatedDifferenceRow
    (selector positive negative : Nat) : Row :=
  ⟨[(selector, 1)], [(positive, 1), (negative, negCoeff 1)], []⟩

def productOutputPairs : List (Nat × Nat) :=
  [(leftProductTrace.output.c0, rightProductTrace.output.c0),
    (leftProductTrace.output.c1, rightProductTrace.output.c1)]

def emittedProductBalanceRows : List Row :=
  productOutputPairs.map fun pair =>
    positiveFirstGatedDifferenceRow artifact.closedColumn pair.1 pair.2

private theorem close_step_index_row_exact :
    closeStepIndexRow =
      gatedOutputLastRow artifact.closedColumn
        (artifact.advancedLaneColumns.getD 6 0) [(0, 1)] := by
  rfl

private theorem terminal_closed_row_exact :
    artifact.terminalClosedRow.2 =
      outputLastLinearRow artifact.closedColumn [(0, 1)] := by
  rfl

private theorem close_zero_row_exact :
    closeZeroRow = builderLinearRow closeZeroColumn [] := by
  rfl

private theorem final_open_mux_row_exact :
    finalOpenMuxRow =
      selectedMuxRow artifact.closedColumn closeZeroColumn
        (artifact.advancedLaneColumns.getD 4 0)
        (artifact.finalLaneColumns.getD 4 0) := by
  rfl

/-- Exact linear close-check slices emitted by Rust. -/
theorem emitted_linear_close_rows_exact :
    emittedStackPointerRows =
        ((indexedRowValues artifact.closeRows).drop 4).take 2 ∧
      emittedSeenPrecommitRows =
        ((indexedRowValues artifact.closeRows).drop 6).take 12 ∧
      emittedInitialMemoryRows =
        ((indexedRowValues artifact.closeRows).drop 30).take 4 := by
  norm_num [emittedStackPointerRows, emittedSeenPrecommitRows,
    emittedInitialMemoryRows, stackPointerColumns, seenPrecommitPairs,
    initialMemoryPairs, gatedZeroRow, gatedDifferenceRow, indexedRowValues,
    negCoeff, goldilocksP, artifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.closeRows]

/-- Exact Rust ownership of the two five-row Karatsuba blocks and their two
gated output equalities. The row matcher accepts only coefficient-order
permutations. -/
theorem emitted_product_balance_rows_exact :
    indexedRowsMatch
        ((artifact.closeRows.drop 18).take 5)
        (indexedDefinitions 831450 leftProductTrace.definitions) = true ∧
      indexedRowsMatch
        ((artifact.closeRows.drop 23).take 5)
        (indexedDefinitions 831455 rightProductTrace.definitions) = true ∧
      emittedProductBalanceRows =
        ((indexedRowValues artifact.closeRows).drop 28).take 2 := by
  norm_num [indexedDefinitions, leftProductTrace, rightProductTrace,
    ProjectionProgram.KMulTrace.ofColumns,
    ProjectionProgram.KMulTrace.definitions,
    ProjectionProgram.KTerms.ofColumns,
    emittedProductBalanceRows, productOutputPairs,
    positiveFirstGatedDifferenceRow, indexedRowsMatch,
    IndexedRowMatchesDefinition, RowsPermutationEquivalent,
    builderLinearRow, Definition.builderRow, negateTerms, negCoeff,
    indexedRowValues, goldilocksP, artifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.closeRows] <;>
    decide

/-- Exact Rust ownership of the reset constants and the 45 close mux rows. -/
theorem emitted_close_output_rows_exact :
    emittedCloseResetRows =
        ((indexedRowValues artifact.closeRows).drop 34).take 11 ∧
      emittedCloseMuxRows =
        ((indexedRowValues artifact.closeRows).drop 45).take 45 := by
  constructor <;>
    norm_num [emittedCloseResetRows, emittedCloseMuxRows,
      closeMuxFieldIndex, closeMuxResetColumn,
      closeSegmentResetColumn, closeZeroColumn, closeOneColumn,
      closeOperationsHeaderColumn, closeMemoryHeaderColumn,
      operationsHeaderValues, memoryHeaderValues, selectedMuxRow,
      closeMuxPositions, outputLastLinearRow, negateTerms, negCoeff,
      indexedRowValues,
      goldilocksP, artifact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.closeRows]

theorem final_carried_program_column_exact (index : Fin 4) :
    artifact.finalLaneColumns.getD index.val 0 =
      artifact.advancedLaneColumns.getD index.val 0 := by
  fin_cases index <;> rfl

theorem final_carried_timestamp_column_exact :
    artifact.finalLaneColumns.getD 7 0 =
      artifact.advancedLaneColumns.getD 7 0 := by
  rfl

private theorem getD_mem_of_lt {alpha : Type}
    {entries : List alpha} {index : Nat} (fallback : alpha)
    (bounded : index < entries.length) :
    entries.getD index fallback ∈ entries := by
  have member := List.getElem_mem (l := entries) bounded
  rwa [List.getElem_eq_getD fallback] at member

private theorem close_rows_length :
    (indexedRowValues artifact.closeRows).length = 90 := by
  norm_num [indexedRowValues, artifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.closeRows]

private theorem close_row_mem (index : Nat) (bounded : index < 90) :
    (indexedRowValues artifact.closeRows).getD index zeroRow ∈
      indexedRowValues artifact.closeRows := by
  apply getD_mem_of_lt zeroRow
  rw [close_rows_length]
  exact bounded

private theorem close_step_index_row_mem :
    closeStepIndexRow ∈ indexedRowValues artifact.closeRows := by
  exact close_row_mem 2 (by decide)

private theorem close_zero_row_mem :
    closeZeroRow ∈ indexedRowValues artifact.closeRows := by
  exact close_row_mem 35 (by decide)

private theorem final_open_mux_row_mem :
    finalOpenMuxRow ∈ indexedRowValues artifact.closeRows := by
  exact close_row_mem 45 (by decide)

private theorem fieldValue_eq_of_assignment_eq
    {assignment : Nat → Nat} {left right : Nat}
    (equal : assignment left = assignment right) :
    fieldValue assignment left = fieldValue assignment right := by
  apply Fin.ext
  simpa [fieldValue] using
    congrArg (fun value => value % goldilocksModulus) equal

private theorem output_last_equality_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (output source : Nat)
    (holds : RowHolds assignment
      (outputLastLinearRow output [(source, 1)])) :
    assignment output = assignment source := by
  have builderHolds :
      RowHolds assignment (builderLinearRow output [(source, 1)]) := by
    apply rowHolds_of_permutationEquivalent
      (source := outputLastLinearRow output [(source, 1)])
      (reconstructed := builderLinearRow output [(source, 1)])
    · refine ⟨?_, List.Perm.refl _, List.Perm.refl _⟩
      simpa [outputLastLinearRow, builderLinearRow] using
        (List.Perm.append_comm (negateTerms [(source, 1)]) [(output, 1)])
    · exact holds
  have defined := builderLinearRow_sound canonical one output [(source, 1)]
    (by simp [CanonicalTerms, goldilocksP]) builderHolds
  simpa [lcEval, Nat.mod_eq_of_lt (canonical source)] using defined

private theorem output_last_linear_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (output : Nat) (terms : List (Nat × Nat))
    (termsCanonical : CanonicalTerms terms)
    (holds : RowHolds assignment (outputLastLinearRow output terms)) :
    assignment output = lcEval assignment terms := by
  have builderHolds :
      RowHolds assignment (builderLinearRow output terms) := by
    apply rowHolds_of_permutationEquivalent
      (source := outputLastLinearRow output terms)
      (reconstructed := builderLinearRow output terms)
    · refine ⟨?_, List.Perm.refl _, List.Perm.refl _⟩
      simpa [outputLastLinearRow, builderLinearRow] using
        (List.Perm.append_comm (negateTerms terms) [(output, 1)])
    · exact holds
  exact builderLinearRow_sound canonical one output terms
    termsCanonical builderHolds

private theorem builder_equality_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (output source : Nat)
    (holds : RowHolds assignment
      (builderLinearRow output [(source, 1)])) :
    assignment output = assignment source := by
  have defined := builderLinearRow_sound canonical one output [(source, 1)]
    (by simp [CanonicalTerms, goldilocksP]) holds
  simpa [lcEval, Nat.mod_eq_of_lt (canonical source)] using defined

private theorem gated_output_last_sound
    {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (selector output : Nat) (terms : List (Nat × Nat))
    (selectorOne : assignment selector = 1)
    (holds : RowHolds assignment
      (gatedOutputLastRow selector output terms)) :
    RowHolds assignment (outputLastLinearRow output terms) := by
  simpa [RowHolds, gatedOutputLastRow, outputLastLinearRow, lcEval,
    selectorOne, one] using holds

private theorem selected_mux_one_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (selector reset live output : Nat)
    (selectorOne : assignment selector = 1)
    (holds : RowHolds assignment
      (selectedMuxRow selector reset live output)) :
    assignment output = assignment reset := by
  have resetCanonical := canonical reset
  have liveCanonical := canonical live
  have outputCanonical := canonical output
  simp [selectedMuxRow, RowHolds, lcEval, negCoeff, selectorOne,
    goldilocksP] at holds
  simp only [goldilocksP] at resetCanonical liveCanonical outputCanonical
  omega

private theorem gated_zero_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (selector value : Nat)
    (selectorOne : assignment selector = 1)
    (holds : RowHolds assignment (gatedZeroRow selector value)) :
    assignment value = 0 := by
  have valueCanonical := canonical value
  simpa [gatedZeroRow, RowHolds, lcEval, selectorOne,
    Nat.mod_eq_of_lt valueCanonical] using holds

private theorem gated_difference_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selector positive negative : Nat)
    (selectorOne : assignment selector = 1)
    (holds : RowHolds assignment
      (gatedDifferenceRow selector positive negative)) :
    assignment positive = assignment negative := by
  apply output_last_equality_sound canonical one
  simpa [gatedDifferenceRow, outputLastLinearRow, RowHolds, lcEval,
    negateTerms, negCoeff, selectorOne, one] using holds

private theorem positive_first_gated_difference_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (selector positive negative : Nat)
    (selectorOne : assignment selector = 1)
    (holds : RowHolds assignment
      (positiveFirstGatedDifferenceRow selector positive negative)) :
    assignment positive = assignment negative := by
  apply gated_difference_sound canonical one selector positive negative
    selectorOne
  apply rowHolds_of_permutationEquivalent
    (source := positiveFirstGatedDifferenceRow selector positive negative)
    (reconstructed := gatedDifferenceRow selector positive negative)
  · exact ⟨List.Perm.refl _, List.Perm.swap _ _ [], List.Perm.refl _⟩
  · exact holds

private theorem terminal_closed_one
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    assignment artifact.closedColumn = 1 := by
  have closedEqualsOneColumn :
      assignment artifact.closedColumn = assignment 0 := by
    apply output_last_equality_sound canonical one
    rw [← terminal_closed_row_exact]
    exact terminalClosed
  exact closedEqualsOneColumn.trans one

private theorem emitted_stack_pointer_rows_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.CloseSatisfied assignment) :
    Satisfies emittedStackPointerRows assignment := by
  rw [emitted_linear_close_rows_exact.1]
  intro row member
  exact satisfied row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem emitted_seen_precommit_rows_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.CloseSatisfied assignment) :
    Satisfies emittedSeenPrecommitRows assignment := by
  rw [emitted_linear_close_rows_exact.2.1]
  intro row member
  exact satisfied row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem emitted_initial_memory_rows_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.CloseSatisfied assignment) :
    Satisfies emittedInitialMemoryRows assignment := by
  rw [emitted_linear_close_rows_exact.2.2]
  intro row member
  exact satisfied row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem close_indexed_slice_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.CloseSatisfied assignment)
    (start count : Nat) :
    Satisfies (((artifact.closeRows.drop start).take count).map Prod.snd)
      assignment := by
  intro row member
  rcases List.mem_map.mp member with
    ⟨entry, entryMember, rowExact⟩
  subst row
  exact satisfied entry.2
    (List.mem_map.mpr ⟨entry,
      List.mem_of_mem_drop (List.mem_of_mem_take entryMember), rfl⟩)

private theorem emitted_close_reset_rows_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.CloseSatisfied assignment) :
    Satisfies emittedCloseResetRows assignment := by
  rw [emitted_close_output_rows_exact.1]
  intro row member
  exact satisfied row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem emitted_close_mux_rows_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.CloseSatisfied assignment) :
    Satisfies emittedCloseMuxRows assignment := by
  rw [emitted_close_output_rows_exact.2]
  intro row member
  exact satisfied row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem close_segment_reset_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (noWrap :
      assignment (artifact.advancedLaneColumns.getD 5 0) + 1 < goldilocksP) :
    assignment closeSegmentResetColumn =
      assignment (artifact.advancedLaneColumns.getD 5 0) + 1 := by
  have equation := output_last_linear_sound canonical one
    closeSegmentResetColumn
    [(0, 1), (artifact.advancedLaneColumns.getD 5 0, 1)]
    (by simp [CanonicalTerms, goldilocksP])
    (emitted_close_reset_rows_satisfied assignment closeSatisfied _
      (by simp [emittedCloseResetRows]))
  have noWrap' :
      1 + assignment (artifact.advancedLaneColumns.getD 5 0) <
        goldilocksP := by
    omega
  simp [lcEval, one] at equation
  exact equation.trans ((Nat.mod_eq_of_lt noWrap').trans
    (Nat.add_comm 1 _))

private theorem close_zero_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment) :
    assignment closeZeroColumn = 0 := by
  have equation := output_last_linear_sound canonical one closeZeroColumn []
    (by simp [CanonicalTerms])
    (emitted_close_reset_rows_satisfied assignment closeSatisfied _
      (by simp [emittedCloseResetRows]))
  simpa [lcEval] using equation

private theorem close_constant_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (output value : Nat) (valuePositive : 0 < value)
    (valueCanonical : value < goldilocksP)
    (member :
      outputLastLinearRow output [(0, value)] ∈ emittedCloseResetRows) :
    assignment output = value := by
  have equation := output_last_linear_sound canonical one output [(0, value)]
    (by simpa [CanonicalTerms] using And.intro valuePositive valueCanonical)
    (emitted_close_reset_rows_satisfied assignment closeSatisfied _ member)
  simpa [lcEval, one, Nat.mod_eq_of_lt valueCanonical] using equation

private theorem close_one_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment) :
    assignment closeOneColumn = 1 := by
  apply close_constant_exact assignment canonical one closeSatisfied
  · norm_num
  · norm_num [goldilocksP]
  · simp [emittedCloseResetRows]

private theorem close_operations_header_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (index : Fin 4) :
    assignment (closeOperationsHeaderColumn index.val) =
      operationsHeaderValues.getD index.val 0 := by
  fin_cases index <;>
    apply close_constant_exact assignment canonical one closeSatisfied <;>
    norm_num [emittedCloseResetRows, closeOperationsHeaderColumn,
      closeMemoryHeaderColumn, operationsHeaderValues, memoryHeaderValues,
      goldilocksP]

private theorem close_memory_header_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (index : Fin 4) :
    assignment (closeMemoryHeaderColumn index.val) =
      memoryHeaderValues.getD index.val 0 := by
  fin_cases index <;>
    apply close_constant_exact assignment canonical one closeSatisfied <;>
    norm_num [emittedCloseResetRows, closeOperationsHeaderColumn,
      closeMemoryHeaderColumn, operationsHeaderValues, memoryHeaderValues,
      goldilocksP]

private theorem close_mux_output_eq_reset
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment)
    (position : Nat) (bounded : position < 45) :
    assignment
        (artifact.finalLaneColumns.getD (closeMuxFieldIndex position) 0) =
      assignment (closeMuxResetColumn position) := by
  apply selected_mux_one_sound canonical artifact.closedColumn
    (closeMuxResetColumn position)
    (artifact.advancedLaneColumns.getD (closeMuxFieldIndex position) 0)
    (artifact.finalLaneColumns.getD (closeMuxFieldIndex position) 0)
    (terminal_closed_one canonical one terminalClosed)
  exact emitted_close_mux_rows_satisfied assignment closeSatisfied _
    (by
      simp only [emittedCloseMuxRows, List.mem_map]
      refine ⟨position, ?_, rfl⟩
      interval_cases position <;> simp [closeMuxPositions])

private theorem close_mux_field_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment)
    (position : Nat) (bounded : position < 45) :
    fieldAt assignment artifact.finalLaneColumns
        (closeMuxFieldIndex position) =
      fieldValue assignment (closeMuxResetColumn position) := by
  apply fieldValue_eq_of_assignment_eq
  exact close_mux_output_eq_reset assignment canonical one closeSatisfied
    terminalClosed position bounded

private theorem close_gamma_c0_reset_column (index : Fin 2) :
    closeMuxResetColumn (3 + 2 * index.val) = closeOneColumn := by
  fin_cases index <;> rfl

private theorem close_gamma_c1_reset_column (index : Fin 2) :
    closeMuxResetColumn (4 + 2 * index.val) = closeZeroColumn := by
  fin_cases index <;> rfl

private theorem close_product_c0_reset_column (index : Fin 4) :
    closeMuxResetColumn (7 + 2 * index.val) = closeOneColumn := by
  fin_cases index <;> rfl

private theorem close_product_c1_reset_column (index : Fin 4) :
    closeMuxResetColumn (8 + 2 * index.val) = closeZeroColumn := by
  fin_cases index <;> rfl

private theorem close_stack_reset_column (index : Fin 2) :
    closeMuxResetColumn (15 + index.val) = closeZeroColumn := by
  fin_cases index <;> rfl

private theorem close_mux_field_index_later
    (position : Nat) (atLeastThree : 3 ≤ position) :
    closeMuxFieldIndex position = 5 + position := by
  simp [closeMuxFieldIndex, Nat.not_lt.mpr atLeastThree]

private theorem close_dPre_reset_column (lane : Fin 3) (index : Fin 4) :
    closeMuxResetColumn (17 + 4 * lane.val + index.val) =
      if lane.val = 0 then closeOperationsHeaderColumn index.val
      else closeMemoryHeaderColumn index.val := by
  fin_cases lane <;> fin_cases index <;> rfl

private theorem close_dSeen_reset_column (lane : Fin 3) (index : Fin 4) :
    closeMuxResetColumn (29 + 4 * lane.val + index.val) =
      if lane.val = 0 then closeOperationsHeaderColumn index.val
      else closeMemoryHeaderColumn index.val := by
  fin_cases lane <;> fin_cases index <;> rfl

private theorem close_dMem_reset_column (index : Fin 4) :
    closeMuxResetColumn (41 + index.val) =
      artifact.advancedLaneColumns.getD (42 + index.val) 0 := by
  fin_cases index <;> rfl

private theorem definition_mem_indexedDefinitions
    (rowIndex : Nat) {definitions : List Definition}
    {definition : Definition} (member : definition ∈ definitions) :
    ∃ sourceRow, (sourceRow, definition) ∈
      indexedDefinitions rowIndex definitions := by
  induction definitions generalizing rowIndex with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.mem_cons] at member
      rcases member with rfl | inTail
      · exact ⟨rowIndex, by simp [indexedDefinitions]⟩
      · rcases inductionHypothesis (rowIndex := rowIndex + 1) inTail with
          ⟨sourceRow, sourceMember⟩
        exact ⟨sourceRow, by simp [indexedDefinitions, sourceMember]⟩

private theorem kmul_definitions_hold
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceRows : List (Nat × Row))
    (firstRow : Nat) (trace : ProjectionProgram.KMulTrace)
    (matchCheck : indexedRowsMatch sourceRows
      (indexedDefinitions firstRow trace.definitions) = true)
    (sourceSatisfied : Satisfies (sourceRows.map Prod.snd) assignment)
    (definitionsCanonical : ∀ definition ∈ trace.definitions,
      definition.Canonical) :
    ProjectionProgram.DefinitionsHold assignment trace.definitions := by
  have builderSatisfied :=
    builderRows_satisfied_of_indexedRowsMatch sourceRows
      (indexedDefinitions firstRow trace.definitions) matchCheck sourceSatisfied
  intro definition member
  apply builderDefinition_sound canonical one definition
    (definitionsCanonical definition member)
  rcases definition_mem_indexedDefinitions firstRow member with
    ⟨sourceRow, indexedMember⟩
  exact builderSatisfied definition.builderRow
    (List.mem_map.mpr ⟨(sourceRow, definition), indexedMember, rfl⟩)

private theorem emitted_product_balance_rows_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.CloseSatisfied assignment) :
    Satisfies emittedProductBalanceRows assignment := by
  rw [emitted_product_balance_rows_exact.2.2]
  intro row member
  exact satisfied row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

private theorem left_product_definitions_canonical :
    ∀ definition ∈ leftProductTrace.definitions,
      definition.Canonical := by
  intro definition member
  simp only [leftProductTrace, ProjectionProgram.KMulTrace.ofColumns,
    ProjectionProgram.KMulTrace.definitions,
    ProjectionProgram.KTerms.ofColumns, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;>
    norm_num [Definition.Canonical, CanonicalTerms, goldilocksP, artifact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact]

private theorem right_product_definitions_canonical :
    ∀ definition ∈ rightProductTrace.definitions,
      definition.Canonical := by
  intro definition member
  simp only [rightProductTrace, ProjectionProgram.KMulTrace.ofColumns,
    ProjectionProgram.KMulTrace.definitions,
    ProjectionProgram.KTerms.ofColumns, List.mem_cons,
    List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;>
    norm_num [Definition.Canonical, CanonicalTerms, goldilocksP, artifact,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact]

private theorem left_product_layout :
    leftProductTrace.SumLayoutValid := by
  simp [leftProductTrace, ProjectionProgram.KMulTrace.SumLayoutValid,
    ProjectionProgram.KMulTrace.ofColumns,
    ProjectionProgram.KTerms.ofColumns]

private theorem right_product_layout :
    rightProductTrace.SumLayoutValid := by
  simp [rightProductTrace, ProjectionProgram.KMulTrace.SumLayoutValid,
    ProjectionProgram.KMulTrace.ofColumns,
    ProjectionProgram.KTerms.ofColumns]

private theorem emitted_step_match_rows_satisfied
    (assignment : Nat → Nat)
    (satisfied : artifact.AdvanceAlgebraSatisfied assignment) :
    Satisfies emittedStepMatchRows assignment := by
  rw [emitted_step_match_rows_exact]
  intro row member
  exact satisfied row
    (List.mem_of_mem_drop (List.mem_of_mem_take member))

/-- Satisfaction of the exact advance-algebra block proves that the delayed
step is the claim for the same opened lane. -/
theorem rows_imply_stepMatches
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.AdvanceAlgebraSatisfied assignment) :
    StepMatches (openedLane assignment) (stepInput assignment) := by
  have rowsSatisfied :=
    emitted_step_match_rows_satisfied assignment satisfied
  have prefixSatisfied : Satisfies
      (outputLastRows stepMatchPrefixPairs) assignment := by
    intro row member
    exact rowsSatisfied row
      (List.mem_append_left _ (List.mem_append_left _ member))
  have gammaSatisfied : Satisfies
      (builderRows stepMatchGammaPairs) assignment := by
    intro row member
    apply rowsSatisfied row
    apply List.mem_append_left
    exact List.mem_append_right _ member
  have suffixSatisfied : Satisfies
      (outputLastRows stepMatchSuffixPairs) assignment := by
    intro row member
    apply rowsSatisfied row
    exact List.mem_append_right _ member
  have prefixEqualities : ∀ pair ∈ stepMatchPrefixPairs,
      assignment pair.1 = assignment pair.2 := by
    intro pair member
    apply output_last_equality_sound canonical one
    exact prefixSatisfied _ (List.mem_map.mpr ⟨pair, member, rfl⟩)
  have gammaEqualities : ∀ pair ∈ stepMatchGammaPairs,
      assignment pair.1 = assignment pair.2 := by
    intro pair member
    apply builder_equality_sound canonical one
    exact gammaSatisfied _ (List.mem_map.mpr ⟨pair, member, rfl⟩)
  have suffixEqualities : ∀ pair ∈ stepMatchSuffixPairs,
      assignment pair.1 = assignment pair.2 := by
    intro pair member
    apply output_last_equality_sound canonical one
    exact suffixSatisfied _ (List.mem_map.mpr ⟨pair, member, rfl⟩)
  have gammaFieldEqual (stepIndex laneIndex : Nat)
      (member : (artifact.stepWordColumn stepIndex,
        artifact.openedLaneColumns.getD laneIndex 0) ∈
          stepMatchGammaPairs) :
      stepFieldAt assignment stepIndex =
        fieldAt assignment artifact.openedLaneColumns laneIndex := by
    apply fieldValue_eq_of_assignment_eq
    exact gammaEqualities _ member
  have suffixFieldEqual (stepIndex laneIndex : Nat)
      (member : (artifact.stepWordColumn stepIndex,
        artifact.openedLaneColumns.getD laneIndex 0) ∈
          stepMatchSuffixPairs) :
      stepFieldAt assignment stepIndex =
        fieldAt assignment artifact.openedLaneColumns laneIndex := by
    apply fieldValue_eq_of_assignment_eq
    exact suffixEqualities _ member
  refine {
    laneOpen := openedLane_open assignment one
    segmentIndex := ?_
    stepIndex := ?_
    timestampIn := ?_
    gamma := ?_
    productsIn := ?_
    stackPointersIn := ?_ }
  · exact prefixEqualities
      (artifact.stepWordColumn 0, artifact.openedLaneColumns.getD 5 0)
      (by simp [stepMatchPrefixPairs, stepMatchPairs])
  · exact prefixEqualities
      (artifact.stepWordColumn 1, artifact.openedLaneColumns.getD 6 0)
      (by simp [stepMatchPrefixPairs, stepMatchPairs])
  · exact prefixEqualities
      (artifact.stepWordColumn 2, artifact.openedLaneColumns.getD 7 0)
      (by simp [stepMatchPrefixPairs, stepMatchPairs])
  · funext index
    fin_cases index
    · exact congrArg₂ K.mk
        (gammaFieldEqual 4 8
          (by simp [stepMatchGammaPairs, stepMatchPairs]))
        (gammaFieldEqual 5 9
          (by simp [stepMatchGammaPairs, stepMatchPairs]))
    · exact congrArg₂ K.mk
        (gammaFieldEqual 6 10
          (by simp [stepMatchGammaPairs, stepMatchPairs]))
        (gammaFieldEqual 7 11
          (by simp [stepMatchGammaPairs, stepMatchPairs]))
  · funext index
    fin_cases index
    · exact congrArg₂ K.mk
        (suffixFieldEqual 8 12
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
        (suffixFieldEqual 9 13
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
    · exact congrArg₂ K.mk
        (suffixFieldEqual 10 14
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
        (suffixFieldEqual 11 15
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
    · exact congrArg₂ K.mk
        (suffixFieldEqual 12 16
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
        (suffixFieldEqual 13 17
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
    · exact congrArg₂ K.mk
        (suffixFieldEqual 14 18
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
        (suffixFieldEqual 15 19
          (by simp [stepMatchSuffixPairs, stepMatchPairs]))
  · funext index
    fin_cases index
    · exact suffixEqualities
        (artifact.constantZeroColumn, artifact.openedLaneColumns.getD 20 0)
        (by simp [stepMatchSuffixPairs, stepMatchPairs])
    · exact suffixEqualities
        (artifact.constantZeroColumn, artifact.openedLaneColumns.getD 21 0)
        (by simp [stepMatchSuffixPairs, stepMatchPairs])

/-- The exact close-distance row and final `closed = 1` row force the
advanced lane to reach the one-step segment boundary. -/
theorem rows_imply_advanced_stepIndex
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    (advancedLane assignment).stepIndex = 1 := by
  have closedOne := terminal_closed_one canonical one terminalClosed
  have closeRowHolds : RowHolds assignment closeStepIndexRow :=
    closeSatisfied closeStepIndexRow close_step_index_row_mem
  rw [close_step_index_row_exact] at closeRowHolds
  have ungatedHolds : RowHolds assignment
      (outputLastLinearRow
        (artifact.advancedLaneColumns.getD 6 0) [(0, 1)]) :=
    gated_output_last_sound one artifact.closedColumn
      (artifact.advancedLaneColumns.getD 6 0) [(0, 1)] closedOne closeRowHolds
  change assignment (artifact.advancedLaneColumns.getD 6 0) = 1
  exact (output_last_equality_sound canonical one
    (artifact.advancedLaneColumns.getD 6 0) 0 ungatedHolds).trans one

/-- The exact close mux selects the Rust-allocated zero value into the final
lane's open flag when the separately pinned close selector is one. -/
theorem rows_imply_finalLane_open_false
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    (finalLane assignment).isOpen = false := by
  have closedOne := terminal_closed_one canonical one terminalClosed
  have zeroRowHolds : RowHolds assignment closeZeroRow :=
    closeSatisfied closeZeroRow close_zero_row_mem
  rw [close_zero_row_exact] at zeroRowHolds
  have zeroExact : assignment closeZeroColumn = 0 := by
    simpa [lcEval] using
      (builderLinearRow_sound canonical one closeZeroColumn []
        (by simp [CanonicalTerms]) zeroRowHolds)
  have muxRowHolds : RowHolds assignment finalOpenMuxRow :=
    closeSatisfied finalOpenMuxRow final_open_mux_row_mem
  rw [final_open_mux_row_exact] at muxRowHolds
  have outputZero :
      assignment (artifact.finalLaneColumns.getD 4 0) = 0 := by
    exact (selected_mux_one_sound canonical artifact.closedColumn
      closeZeroColumn (artifact.advancedLaneColumns.getD 4 0)
      (artifact.finalLaneColumns.getD 4 0) closedOne muxRowHolds).trans zeroExact
  change (assignment (artifact.finalLaneColumns.getD 4 0) == 1) = false
  rw [outputZero]
  rfl

/-- The two exact close-only rows force both terminal stack pointers to zero. -/
theorem rows_imply_stackPointersZero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    ∀ index, (advancedLane assignment).stackPointers index = 0 := by
  have closedOne := terminal_closed_one canonical one terminalClosed
  have rowsSatisfied :=
    emitted_stack_pointer_rows_satisfied assignment closeSatisfied
  have zeros : ∀ column ∈ stackPointerColumns,
      assignment column = 0 := by
    intro column member
    apply gated_zero_sound canonical artifact.closedColumn column closedOne
    exact rowsSatisfied _
      (List.mem_map.mpr ⟨column, member, rfl⟩)
  intro index
  fin_cases index
  · simpa [advancedLane, laneAt, columnAt] using
      zeros _ (by simp [stackPointerColumns])
  · simpa [advancedLane, laneAt, columnAt] using
      zeros _ (by simp [stackPointerColumns])

/-- The twelve exact close-only rows bind every seen digest lane to its
precommit digest lane. -/
theorem rows_imply_seenEqualsPrecommit
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    (advancedLane assignment).dSeen =
      (advancedLane assignment).dPre := by
  have closedOne := terminal_closed_one canonical one terminalClosed
  have rowsSatisfied :=
    emitted_seen_precommit_rows_satisfied assignment closeSatisfied
  have equalities : ∀ pair ∈ seenPrecommitPairs,
      assignment pair.1 = assignment pair.2 := by
    intro pair member
    apply gated_difference_sound canonical one artifact.closedColumn
      pair.1 pair.2 closedOne
    exact rowsSatisfied _
      (List.mem_map.mpr ⟨pair, member, rfl⟩)
  funext lane digestLane
  apply fieldValue_eq_of_assignment_eq
  have member :
      (artifact.advancedLaneColumns.getD
          (34 + 4 * lane.val + digestLane.val) 0,
        artifact.advancedLaneColumns.getD
          (22 + 4 * lane.val + digestLane.val) 0) ∈
        seenPrecommitPairs := by
    fin_cases lane <;> fin_cases digestLane <;>
      simp [seenPrecommitPairs]
  simpa [columnAt] using equalities _ member

/-- The four exact close-only rows bind the initial-memory seen root to the
carried memory root. -/
theorem rows_imply_initialMemoryExact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    (advancedLane assignment).dSeen ⟨1, by decide⟩ =
      (advancedLane assignment).dMem := by
  have closedOne := terminal_closed_one canonical one terminalClosed
  have rowsSatisfied :=
    emitted_initial_memory_rows_satisfied assignment closeSatisfied
  have equalities : ∀ pair ∈ initialMemoryPairs,
      assignment pair.1 = assignment pair.2 := by
    intro pair member
    apply gated_difference_sound canonical one artifact.closedColumn
      pair.1 pair.2 closedOne
    exact rowsSatisfied _
      (List.mem_map.mpr ⟨pair, member, rfl⟩)
  funext digestLane
  apply fieldValue_eq_of_assignment_eq
  have member :
      (artifact.advancedLaneColumns.getD (38 + digestLane.val) 0,
        artifact.advancedLaneColumns.getD (46 + digestLane.val) 0) ∈
        initialMemoryPairs := by
    fin_cases digestLane <;> simp [initialMemoryPairs]
  simpa [columnAt] using equalities _ member

/-- The exact two Karatsuba blocks and their two close-gated output rows
force the terminal product-balance equation in the concrete extension field. -/
theorem rows_imply_productsBalanced
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    K.mul ((advancedLane assignment).products ⟨2, by decide⟩)
        ((advancedLane assignment).products ⟨1, by decide⟩) =
      K.mul ((advancedLane assignment).products ⟨0, by decide⟩)
        ((advancedLane assignment).products ⟨3, by decide⟩) := by
  have closedOne := terminal_closed_one canonical one terminalClosed
  have leftDefinitions := kmul_definitions_hold assignment canonical one
    ((artifact.closeRows.drop 18).take 5) 831450 leftProductTrace
    emitted_product_balance_rows_exact.1
    (close_indexed_slice_satisfied assignment closeSatisfied 18 5)
    left_product_definitions_canonical
  have rightDefinitions := kmul_definitions_hold assignment canonical one
    ((artifact.closeRows.drop 23).take 5) 831455 rightProductTrace
    emitted_product_balance_rows_exact.2.1
    (close_indexed_slice_satisfied assignment closeSatisfied 23 5)
    right_product_definitions_canonical
  have leftProduct := leftProductTrace.sound assignment
    left_product_layout leftDefinitions
  have rightProduct := rightProductTrace.sound assignment
    right_product_layout rightDefinitions
  have balanceRows :=
    emitted_product_balance_rows_satisfied assignment closeSatisfied
  have outputEqualities : ∀ pair ∈ productOutputPairs,
      assignment pair.1 = assignment pair.2 := by
    intro pair member
    apply positive_first_gated_difference_sound canonical one
      artifact.closedColumn pair.1 pair.2 closedOne
    exact balanceRows _
      (List.mem_map.mpr ⟨pair, member, rfl⟩)
  have outputValuesEqual :
      leftProductTrace.output.value assignment =
        rightProductTrace.output.value assignment := by
    simp only [ProjectionProgram.KColumns.value]
    change ProjectionProgram.K.mk
        (ProjectionProgram.baseAt assignment leftProductTrace.output.c0)
        (ProjectionProgram.baseAt assignment leftProductTrace.output.c1) =
      ProjectionProgram.K.mk
        (ProjectionProgram.baseAt assignment rightProductTrace.output.c0)
        (ProjectionProgram.baseAt assignment rightProductTrace.output.c1)
    apply congrArg₂ ProjectionProgram.K.mk
    · simpa [ProjectionProgram.baseAt] using
        congrArg ProjectionProgram.residue
        (outputEqualities
          (leftProductTrace.output.c0, rightProductTrace.output.c0)
          (by simp [productOutputPairs]))
    · simpa [ProjectionProgram.baseAt] using
        congrArg ProjectionProgram.residue
        (outputEqualities
          (leftProductTrace.output.c1, rightProductTrace.output.c1)
          (by simp [productOutputPairs]))
  have leftProductTyped :
      leftProductTrace.output.value assignment =
        ProjectionProgram.K.mul
          (Canonical.KConcreteFixedPhaseBridge.toProjection
            (kAt assignment artifact.advancedLaneColumns 16))
          (Canonical.KConcreteFixedPhaseBridge.toProjection
            (kAt assignment artifact.advancedLaneColumns 14)) := by
    simpa [leftProductTrace, ProjectionProgram.KMulTrace.ofColumns,
      ProjectionProgram.KTerms.ofColumns_value,
      ProjectionProgram.KColumns.value, ProjectionProgram.baseAt,
      ProjectionProgram.residue,
      Canonical.KConcreteFixedPhaseBridge.toProjection,
      kAt, fieldAt, fieldValue, columnAt]
      using leftProduct
  have rightProductTyped :
      rightProductTrace.output.value assignment =
        ProjectionProgram.K.mul
          (Canonical.KConcreteFixedPhaseBridge.toProjection
            (kAt assignment artifact.advancedLaneColumns 12))
          (Canonical.KConcreteFixedPhaseBridge.toProjection
            (kAt assignment artifact.advancedLaneColumns 18)) := by
    simpa [rightProductTrace, ProjectionProgram.KMulTrace.ofColumns,
      ProjectionProgram.KTerms.ofColumns_value,
      ProjectionProgram.KColumns.value, ProjectionProgram.baseAt,
      ProjectionProgram.residue,
      Canonical.KConcreteFixedPhaseBridge.toProjection,
      kAt, fieldAt, fieldValue, columnAt]
      using rightProduct
  change K.mul
      (kAt assignment artifact.advancedLaneColumns 16)
      (kAt assignment artifact.advancedLaneColumns 14) =
    K.mul
      (kAt assignment artifact.advancedLaneColumns 12)
      (kAt assignment artifact.advancedLaneColumns 18)
  apply Canonical.KConcreteFixedPhaseBridge.toProjection_injective
  rw [Canonical.KConcreteFixedPhaseBridge.toProjection_mul,
    Canonical.KConcreteFixedPhaseBridge.toProjection_mul]
  exact leftProductTyped.symm.trans
    (outputValuesEqual.trans rightProductTyped)

/-- The exact Rust close rows select the canonical reset lane. The only
non-row premise is the natural-number no-wrap fact for the segment counter;
the full terminal parent derives it from decoder and advance rows. -/
theorem rows_imply_outputExact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment)
    (segmentNoWrap :
      assignment (artifact.advancedLaneColumns.getD 5 0) + 1 < goldilocksP)
    (hash : HashSemantics)
    (operationsHeaderExact :
      hash.operationsHeader = emittedOperationsHeader)
    (memoryHeaderExact : hash.memoryHeader = emittedMemoryHeader) :
    finalLane assignment = closeLane hash (advancedLane assignment) := by
  have segmentReset := close_segment_reset_exact assignment canonical one
    closeSatisfied segmentNoWrap
  have zeroExact := close_zero_exact assignment canonical one closeSatisfied
  have oneExact := close_one_exact assignment canonical one closeSatisfied
  have zeroField : fieldValue assignment closeZeroColumn = 0 := by
    apply Fin.ext
    simp [fieldValue, zeroExact]
  have oneField : fieldValue assignment closeOneColumn = 1 := by
    apply Fin.ext
    simp [fieldValue, oneExact]
  have operationsFieldExact (index : Fin 4) :
      fieldValue assignment (closeOperationsHeaderColumn index.val) =
        emittedOperationsHeader index := by
    apply Fin.ext
    change assignment (closeOperationsHeaderColumn index.val) %
        goldilocksModulus =
      (operationsHeaderValues.getD index.val 0) % goldilocksModulus
    rw [close_operations_header_exact assignment canonical one
      closeSatisfied index]
  have memoryFieldExact (index : Fin 4) :
      fieldValue assignment (closeMemoryHeaderColumn index.val) =
        emittedMemoryHeader index := by
    apply Fin.ext
    change assignment (closeMemoryHeaderColumn index.val) %
        goldilocksModulus =
      (memoryHeaderValues.getD index.val 0) % goldilocksModulus
    rw [close_memory_header_exact assignment canonical one
      closeSatisfied index]
  have headerFieldExact (lane : Fin 3) (index : Fin 4) :
      fieldValue assignment
          (if lane.val = 0 then closeOperationsHeaderColumn index.val
          else closeMemoryHeaderColumn index.val) =
        hash.header lane index := by
    by_cases laneZero : lane.val = 0
    · simpa [HashSemantics.header, laneZero, operationsHeaderExact] using
        operationsFieldExact index
    · simpa [HashSemantics.header, laneZero, memoryHeaderExact] using
        memoryFieldExact index
  apply Lane.ext
  · funext index
    simpa [finalLane, closeLane, advancedLane, laneAt, digestAt, fieldAt,
      columnAt] using congrArg (fieldValue assignment)
        (final_carried_program_column_exact index)
  · exact rows_imply_finalLane_open_false assignment canonical one
      closeSatisfied terminalClosed
  · have selected := close_mux_output_eq_reset assignment canonical one
      closeSatisfied terminalClosed 1 (by decide)
    simpa [closeMuxFieldIndex, closeMuxResetColumn] using
      selected.trans segmentReset
  · have selected := close_mux_output_eq_reset assignment canonical one
      closeSatisfied terminalClosed 2 (by decide)
    simpa [closeMuxFieldIndex, closeMuxResetColumn] using
      selected.trans zeroExact
  · change assignment (artifact.finalLaneColumns.getD 7 0) =
      assignment (artifact.advancedLaneColumns.getD 7 0)
    rw [final_carried_timestamp_column_exact]
  · funext index
    change kAt assignment artifact.finalLaneColumns
        (8 + 2 * index.val) = K.one
    apply congrArg₂ K.mk
    · have selected := close_mux_field_exact assignment canonical one
        closeSatisfied terminalClosed (3 + 2 * index.val) (by omega)
      rw [close_gamma_c0_reset_column,
        close_mux_field_index_later _ (by omega)] at selected
      have indexExact :
          5 + (3 + 2 * index.val) = 8 + 2 * index.val := by
        omega
      rw [indexExact] at selected
      exact selected.trans oneField
    · have selected := close_mux_field_exact assignment canonical one
        closeSatisfied terminalClosed (4 + 2 * index.val) (by omega)
      rw [close_gamma_c1_reset_column,
        close_mux_field_index_later _ (by omega)] at selected
      have indexExact :
          5 + (4 + 2 * index.val) = 8 + 2 * index.val + 1 := by
        omega
      rw [indexExact] at selected
      exact selected.trans zeroField
  · funext index
    change kAt assignment artifact.finalLaneColumns
        (12 + 2 * index.val) = K.one
    apply congrArg₂ K.mk
    · have selected := close_mux_field_exact assignment canonical one
        closeSatisfied terminalClosed (7 + 2 * index.val) (by omega)
      rw [close_product_c0_reset_column,
        close_mux_field_index_later _ (by omega)] at selected
      have indexExact :
          5 + (7 + 2 * index.val) = 12 + 2 * index.val := by
        omega
      rw [indexExact] at selected
      exact selected.trans oneField
    · have selected := close_mux_field_exact assignment canonical one
        closeSatisfied terminalClosed (8 + 2 * index.val) (by omega)
      rw [close_product_c1_reset_column,
        close_mux_field_index_later _ (by omega)] at selected
      have indexExact :
          5 + (8 + 2 * index.val) = 12 + 2 * index.val + 1 := by
        omega
      rw [indexExact] at selected
      exact selected.trans zeroField
  · funext index
    have selected := close_mux_output_eq_reset assignment canonical one
      closeSatisfied terminalClosed (15 + index.val) (by omega)
    rw [close_stack_reset_column,
      close_mux_field_index_later _ (by omega)] at selected
    have indexExact : 5 + (15 + index.val) = 20 + index.val := by
      omega
    rw [indexExact] at selected
    change assignment
        (artifact.finalLaneColumns.getD (20 + index.val) 0) = 0
    exact selected.trans zeroExact
  · funext lane index
    have selected := close_mux_field_exact assignment canonical one
      closeSatisfied terminalClosed (17 + 4 * lane.val + index.val)
      (by omega)
    rw [close_dPre_reset_column,
      close_mux_field_index_later _ (by omega)] at selected
    have indexExact :
        5 + (17 + 4 * lane.val + index.val) =
          22 + 4 * lane.val + index.val := by
      omega
    rw [indexExact] at selected
    change fieldAt assignment artifact.finalLaneColumns
        (22 + 4 * lane.val + index.val) = hash.header lane index
    exact selected.trans (headerFieldExact lane index)
  · funext lane index
    have selected := close_mux_field_exact assignment canonical one
      closeSatisfied terminalClosed (29 + 4 * lane.val + index.val)
      (by omega)
    rw [close_dSeen_reset_column,
      close_mux_field_index_later _ (by omega)] at selected
    have indexExact :
        5 + (29 + 4 * lane.val + index.val) =
          34 + 4 * lane.val + index.val := by
      omega
    rw [indexExact] at selected
    change fieldAt assignment artifact.finalLaneColumns
        (34 + 4 * lane.val + index.val) = hash.header lane index
    exact selected.trans (headerFieldExact lane index)
  · funext index
    have selected := close_mux_field_exact assignment canonical one
      closeSatisfied terminalClosed (41 + index.val) (by omega)
    rw [close_dMem_reset_column,
      close_mux_field_index_later _ (by omega)] at selected
    have indexExact : 5 + (41 + index.val) = 46 + index.val := by
      omega
    rw [indexExact] at selected
    change fieldAt assignment artifact.finalLaneColumns (46 + index.val) =
      fieldAt assignment artifact.advancedLaneColumns (42 + index.val)
    exact selected

/-- The exact close block and final selector row imply every close-only field
of the typed one-step terminal relation. -/
theorem rows_imply_closeChecks
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    CloseChecks (advancedLane assignment) := by
  exact {
    closeIndex := rows_imply_advanced_stepIndex assignment canonical one
      closeSatisfied terminalClosed
    stackPointersZero := rows_imply_stackPointersZero assignment canonical one
      closeSatisfied terminalClosed
    seenEqualsPrecommit := rows_imply_seenEqualsPrecommit assignment canonical one
      closeSatisfied terminalClosed
    productsBalanced := rows_imply_productsBalanced assignment canonical one
      closeSatisfied terminalClosed
    initialMemoryExact := rows_imply_initialMemoryExact assignment canonical one
      closeSatisfied terminalClosed }

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound
