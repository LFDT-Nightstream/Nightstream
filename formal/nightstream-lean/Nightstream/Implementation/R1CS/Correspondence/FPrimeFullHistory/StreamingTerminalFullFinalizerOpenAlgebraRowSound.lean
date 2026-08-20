import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows

/-!
Contract: the first 53 rows of the terminal Nebula `maybe_open` phase enforce
the selected segment bound and exclusive open rule.

It does not own the staged lane digest, transcript challenges, or output muxes.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenAlgebraRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

private theorem all_pieces_satisfied
    (assignment : Nat → Nat)
    (satisfied : rawArtifact.OpenAlgebraSatisfied assignment) :
    ∀ piece ∈ rawArtifact.openAlgebraPieces, Satisfies piece assignment := by
  apply (satisfies_flatten_iff rawArtifact.openAlgebraPieces assignment).mp
  simpa [RawArtifact.OpenAlgebraSatisfied, RawArtifact.openAlgebraRows] using satisfied

private theorem normalized_bit_holds
    (assignment : Nat → Nat) (column : Nat)
    (holds : RowHolds assignment (normalizedBitRow column)) :
    RowHolds assignment (bitRow column) := by
  apply rowHolds_of_permutationEquivalent (source := normalizedBitRow column)
  · refine ⟨List.Perm.refl _, ?_, List.Perm.refl _⟩
    simpa [normalizedBitRow, bitRow, negCoeff] using
      (List.Perm.swap (0, goldilocksP - 1) (column, 1) []).symm
  · exact holds

private theorem normalized_bit_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (column : Nat)
    (holds : RowHolds assignment (normalizedBitRow column)) :
    assignment column = 0 ∨ assignment column = 1 := by
  have bounded := bitRow_le_one goldilocks_euclidPrime
    (canonical column) one (normalized_bit_holds assignment column holds)
  omega

private theorem high_piece_satisfied
    (assignment : Nat → Nat)
    (satisfied : Satisfies rawArtifact.highSegmentComparisonRows assignment) :
    ∀ iteration < highSegmentIndexBits,
      Satisfies
        [rawArtifact.highSegmentForbidRow iteration,
          (rawArtifact.highSegmentEqualDefinition iteration).builderRow]
        assignment := by
  have pieces :=
    (satisfies_flatten_iff rawArtifact.highSegmentComparisonPieces assignment).mp
      (by simpa [RawArtifact.highSegmentComparisonRows] using satisfied)
  intro iteration bounded
  apply pieces
  apply List.mem_map.mpr
  exact ⟨iteration, by simp [bounded], rfl⟩

private theorem forbid_bit_zero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (input bit : Nat)
    (inputOne : assignment input = 1)
    (holds : RowHolds assignment
      ⟨[(input, 1)], [(bit, 1)], []⟩) :
    assignment bit = 0 := by
  have bitCanonical := canonical bit
  have equation : assignment bit % goldilocksP = 0 := by
    simpa [RowHolds, lcEval, inputOne, goldilocksP] using holds
  simpa [Nat.mod_eq_of_lt bitCanonical] using equation

private theorem high_output_one
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (iteration : Nat)
    (inputOne : assignment (rawArtifact.comparisonInputColumn iteration) = 1)
    (bitZero :
      assignment (rawArtifact.segmentBitColumn (highSegmentIndex iteration)) = 0)
    (holds : RowHolds assignment
      (rawArtifact.highSegmentEqualDefinition iteration).builderRow) :
    assignment (rawArtifact.comparisonOutputColumn iteration) = 1 := by
  have definitionHolds := builderDefinition_sound canonical one
    (rawArtifact.highSegmentEqualDefinition iteration) trivial holds
  simpa [Definition.Holds, RawArtifact.highSegmentEqualDefinition,
    Rhs.eval, lcEval, inputOne, bitZero, one, goldilocksP] using
    definitionHolds

private theorem high_input_one
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies rawArtifact.highSegmentComparisonRows assignment) :
    ∀ iteration ≤ highSegmentIndexBits,
      assignment (rawArtifact.comparisonInputColumn iteration) = 1 := by
  have pieceSatisfied := high_piece_satisfied assignment satisfied
  intro iteration bounded
  induction iteration with
  | zero => simpa [RawArtifact.comparisonInputColumn] using one
  | succ previous inductionHypothesis =>
      have previousLt : previous < highSegmentIndexBits := by omega
      have inputOne := inductionHypothesis (by omega)
      have rows := pieceSatisfied previous previousLt
      have forbidHolds := rows (rawArtifact.highSegmentForbidRow previous)
        (by simp)
      have bitZero := forbid_bit_zero assignment canonical
        (rawArtifact.comparisonInputColumn previous)
        (rawArtifact.segmentBitColumn (highSegmentIndex previous))
        inputOne forbidHolds
      have equalHolds := rows
        (rawArtifact.highSegmentEqualDefinition previous).builderRow (by simp)
      have outputOne := high_output_one assignment canonical one previous
        inputOne bitZero equalHolds
      simpa [RawArtifact.comparisonInputColumn] using outputOne

private theorem high_bit_zero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies rawArtifact.highSegmentComparisonRows assignment) :
    ∀ iteration < highSegmentIndexBits,
      assignment
        (rawArtifact.segmentBitColumn (highSegmentIndex iteration)) = 0 := by
  have pieceSatisfied := high_piece_satisfied assignment satisfied
  have inputOne := high_input_one assignment canonical one satisfied
  intro iteration bounded
  have rows := pieceSatisfied iteration bounded
  have forbidHolds := rows (rawArtifact.highSegmentForbidRow iteration) (by simp)
  exact forbid_bit_zero assignment canonical
    (rawArtifact.comparisonInputColumn iteration)
    (rawArtifact.segmentBitColumn (highSegmentIndex iteration))
    (inputOne iteration (by omega)) forbidHolds

private theorem lcEval_zero_of_columns_zero
    (assignment : Nat → Nat) (terms : List (Nat × Nat))
    (zero : ∀ term ∈ terms, assignment term.1 = 0) :
    lcEval assignment terms = 0 := by
  unfold lcEval
  have foldZero :
      terms.foldl (fun total term => total + term.2 * assignment term.1) 0 = 0 := by
    induction terms with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [zero head (by simp)]
        simp only [Nat.mul_zero, Nat.add_zero]
        apply inductionHypothesis
        intro term member
        exact zero term (by simp [member])
  rw [foldZero]
  simp

structure Sound (assignment : Nat → Nat) : Prop where
  segmentIndexZero : assignment rawArtifact.laneSegmentIndexColumn = 0
  segmentIndexBound :
    assignment rawArtifact.laneSegmentIndexColumn < rawArtifact.segmentMaximum
  laneOpenExact : assignment rawArtifact.laneOpenColumn = 0 ∨
    assignment rawArtifact.laneOpenColumn = 1
  inputOpenExact : assignment rawArtifact.openColumn = 0 ∨
    assignment rawArtifact.openColumn = 1
  exactlyOneOpen :
    assignment rawArtifact.laneOpenColumn + assignment rawArtifact.openColumn = 1
  newOpenZeroIndex : assignment rawArtifact.openColumn = 1 →
    assignment rawArtifact.laneStepIndexColumn = 0

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.OpenAlgebraSatisfied assignment) :
    Sound assignment := by
  have pieces := all_pieces_satisfied assignment satisfied
  have segmentBitsSatisfied : Satisfies rawArtifact.segmentBitRows assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])
  have recompositionSatisfied :
      Satisfies [rawArtifact.segmentRecompositionRow] assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])
  have highSatisfied :
      Satisfies rawArtifact.highSegmentComparisonRows assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])
  have lowSatisfied :
      Satisfies [(rawArtifact.lowSegmentEqualDefinition).builderRow] assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])
  have finalSatisfied :
      Satisfies [rawArtifact.finalSegmentEqualZeroRow] assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])
  have flagsSatisfied : Satisfies rawArtifact.openFlagRows assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])
  have exclusiveSatisfied : Satisfies [rawArtifact.exclusiveOpenRow] assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])
  have newOpenSatisfied : Satisfies [rawArtifact.newOpenZeroIndexRow] assignment :=
    pieces _ (by simp [RawArtifact.openAlgebraPieces])

  have segmentBitExact : ∀ index < segmentIndexBits,
      assignment (rawArtifact.segmentBitColumn index) = 0 ∨
        assignment (rawArtifact.segmentBitColumn index) = 1 := by
    intro index bounded
    have columnMember : rawArtifact.segmentBitColumn index ∈
        rawArtifact.segmentBitColumns := by
      apply List.mem_map.mpr
      exact ⟨index, by simp [bounded], rfl⟩
    have rowMember : normalizedBitRow (rawArtifact.segmentBitColumn index) ∈
        rawArtifact.segmentBitRows :=
      List.mem_map.mpr ⟨rawArtifact.segmentBitColumn index, columnMember, rfl⟩
    exact normalized_bit_exact assignment canonical one _
      (segmentBitsSatisfied _ rowMember)

  have highZero := high_bit_zero assignment canonical one highSatisfied
  have highInputs := high_input_one assignment canonical one highSatisfied
  have lowDefinitionHolds := builderDefinition_sound canonical one
    rawArtifact.lowSegmentEqualDefinition trivial
    (lowSatisfied _ (by simp))
  have lowOutputValue :
      assignment (rawArtifact.comparisonOutputColumn highSegmentIndexBits) =
        assignment (rawArtifact.segmentBitColumn 0) := by
    have bitCanonical := canonical (rawArtifact.segmentBitColumn 0)
    have equation :
        assignment (rawArtifact.comparisonOutputColumn highSegmentIndexBits) =
          assignment (rawArtifact.segmentBitColumn 0) % goldilocksP := by
      simpa [Definition.Holds, RawArtifact.lowSegmentEqualDefinition,
      Rhs.eval, lcEval, highInputs highSegmentIndexBits (by omega), one,
        goldilocksP] using lowDefinitionHolds
    simpa [Nat.mod_eq_of_lt bitCanonical] using equation
  have finalHolds := finalSatisfied rawArtifact.finalSegmentEqualZeroRow (by simp)
  have finalBuilderHolds : RowHolds assignment
      (builderLinearRow
        (rawArtifact.comparisonOutputColumn highSegmentIndexBits) []) := by
    simpa [RawArtifact.finalSegmentEqualZeroRow, outputLastLinearRow] using
      finalHolds
  have finalOutputZero := builderLinearRow_sound canonical one
    (rawArtifact.comparisonOutputColumn highSegmentIndexBits) []
    (by simp [CanonicalTerms]) finalBuilderHolds
  have lowBitZero : assignment (rawArtifact.segmentBitColumn 0) = 0 := by
    simpa [lcEval] using lowOutputValue.symm.trans finalOutputZero

  have allSegmentBitsZero : ∀ index < segmentIndexBits,
      assignment (rawArtifact.segmentBitColumn index) = 0 := by
    intro index bounded
    by_cases isLow : index = 0
    · simpa [isLow] using lowBitZero
    · let iteration := segmentIndexBits - 1 - index
      have bounded' : index < 16 := by
        simpa [segmentIndexBits] using bounded
      have indexPositive : 0 < index := Nat.pos_of_ne_zero isLow
      have iterationExact : iteration = 15 - index := by
        simp [iteration, segmentIndexBits]
      have iterationLt : iteration < highSegmentIndexBits := by
        rw [iterationExact]
        simp [highSegmentIndexBits, segmentIndexBits]
        omega
      have indexExact : highSegmentIndex iteration = index := by
        rw [iterationExact]
        simp [highSegmentIndex, segmentIndexBits]
        omega
      simpa [indexExact] using highZero iteration iterationLt

  have recompositionHolds := recompositionSatisfied
    rawArtifact.segmentRecompositionRow (by simp)
  have recompositionTermsCanonical :
      CanonicalTerms rawArtifact.segmentRecompositionTerms := by
    intro term member
    rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
    have indexLt : index < segmentIndexBits := by simpa using indexMember
    have indexLt' : index < 16 := by simpa [segmentIndexBits] using indexLt
    constructor
    · positivity
    · interval_cases index <;> norm_num [goldilocksP]
  have segmentValue := builderLinearRow_sound canonical one
    rawArtifact.laneSegmentIndexColumn rawArtifact.segmentRecompositionTerms
    recompositionTermsCanonical recompositionHolds
  have recompositionZero :
      lcEval assignment rawArtifact.segmentRecompositionTerms = 0 := by
    apply lcEval_zero_of_columns_zero assignment
      rawArtifact.segmentRecompositionTerms
    intro term member
    rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
    exact allSegmentBitsZero index (by simpa using indexMember)
  have segmentIndexZero : assignment rawArtifact.laneSegmentIndexColumn = 0 :=
    segmentValue.trans recompositionZero

  have laneBitHolds := flagsSatisfied
    (normalizedBitRow rawArtifact.laneOpenColumn)
    (by simp [RawArtifact.openFlagRows])
  have inputBitHolds := flagsSatisfied
    (normalizedBitRow rawArtifact.openColumn)
    (by simp [RawArtifact.openFlagRows])
  have laneOpenExact := normalized_bit_exact assignment canonical one _ laneBitHolds
  have inputOpenExact := normalized_bit_exact assignment canonical one _ inputBitHolds
  have exclusiveHolds := exclusiveSatisfied rawArtifact.exclusiveOpenRow (by simp)
  have exactlyOneOpen :
      assignment rawArtifact.laneOpenColumn + assignment rawArtifact.openColumn = 1 := by
    rcases laneOpenExact with hLane | hLane <;>
      rcases inputOpenExact with hInput | hInput <;>
      simp [RawArtifact.exclusiveOpenRow, RowHolds, lcEval, one,
        hLane, hInput, goldilocksP] at exclusiveHolds ⊢
  refine {
    segmentIndexZero := segmentIndexZero
    segmentIndexBound := by rw [segmentIndexZero, rawArtifact_valid.segmentMaximum]; decide
    laneOpenExact := laneOpenExact
    inputOpenExact := inputOpenExact
    exactlyOneOpen := exactlyOneOpen
    newOpenZeroIndex := ?_ }
  intro inputOne
  have holds := newOpenSatisfied rawArtifact.newOpenZeroIndexRow (by simp)
  have indexCanonical := canonical rawArtifact.laneStepIndexColumn
  have equation : assignment rawArtifact.laneStepIndexColumn % goldilocksP = 0 := by
    simpa [RawArtifact.newOpenZeroIndexRow, RowHolds, lcEval, inputOne,
      goldilocksP] using holds
  simpa [Nat.mod_eq_of_lt indexCanonical] using equation

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenAlgebraRowSound
