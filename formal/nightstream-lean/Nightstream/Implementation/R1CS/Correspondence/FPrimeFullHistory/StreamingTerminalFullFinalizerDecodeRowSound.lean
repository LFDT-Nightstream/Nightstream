import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows

/-!
Contract: the first 2,974 exact full-layout terminal-finalizer rows decode the
stack-free delayed Nebula suffix.

It proves Boolean input coordinates, little-endian field recomposition, and
the canonical zero rule for inactive `D_pre`. It does not own the later open,
leaf-digest, advance, or close phases.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerDecodeRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

private theorem all_decode_pieces_satisfied
    (assignment : Nat → Nat)
    (satisfied : rawArtifact.DecodeSatisfied assignment) :
    ∀ piece ∈ rawArtifact.decodePieces, Satisfies piece assignment := by
  apply (satisfies_flatten_iff rawArtifact.decodePieces assignment).mp
  simpa [RawArtifact.DecodeSatisfied, RawArtifact.decodeRows] using satisfied

private theorem bit_rows_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (columns : List Nat)
    (satisfied : Satisfies (columns.map normalizedBitRow) assignment) :
    ∀ column ∈ columns,
      assignment column = 0 ∨ assignment column = 1 := by
  intro column member
  have normalizedHolds := satisfied (normalizedBitRow column)
    (List.mem_map.mpr ⟨column, member, rfl⟩)
  have holds : RowHolds assignment (bitRow column) := by
    apply rowHolds_of_permutationEquivalent (source := normalizedBitRow column)
    · refine ⟨List.Perm.refl _, ?_, List.Perm.refl _⟩
      simpa [normalizedBitRow, bitRow, negCoeff] using
        (List.Perm.swap (0, goldilocksP - 1) (column, 1) []).symm
    · exact normalizedHolds
  have bounded := bitRow_le_one goldilocks_euclidPrime
    (canonical column) one holds
  omega

private theorem pow_two_lt_goldilocks
    (index : Nat) (bounded : index < 64) :
    2 ^ index < goldilocksP := by
  interval_cases index <;> norm_num [goldilocksP]

private theorem word_terms_canonical
    (artifact : RawArtifact) (start width : Nat)
    (widthBound : width ≤ 64) :
    CanonicalTerms (artifact.wordTerms start width) := by
  intro term member
  rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
  have indexLt : index < width := by simpa using indexMember
  constructor
  · positivity
  · exact pow_two_lt_goldilocks index (by omega)

private theorem step_word_width_le
    (index : Nat) (bounded : index < stepWordCount) :
    stepWordWidths.getD index 0 ≤ 64 := by
  norm_num [stepWordCount, stepWordWidths] at bounded
  interval_cases index <;> norm_num [stepWordWidths]

private theorem normalized_linear_holds_builder
    (assignment : Nat → Nat) (output : Nat)
    (terms : List (Nat × Nat))
    (holds : RowHolds assignment (outputLastLinearRow output terms)) :
    RowHolds assignment (builderLinearRow output terms) := by
  apply rowHolds_of_permutationEquivalent
    (source := outputLastLinearRow output terms)
    (reconstructed := builderLinearRow output terms)
  · refine ⟨?_, List.Perm.refl _, List.Perm.refl _⟩
    simpa [outputLastLinearRow, builderLinearRow] using
      (List.Perm.append_comm (negateTerms terms) [(output, 1)])
  · exact holds

private theorem inactive_row_zero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (artifact : RawArtifact) (bitColumn : Nat)
    (openZero : assignment artifact.openColumn = 0)
    (holds : RowHolds assignment (artifact.inactiveDPreRow bitColumn)) :
    assignment bitColumn = 0 := by
  have bitCanonical := canonical bitColumn
  have equation : assignment bitColumn % goldilocksP = 0 := by
    simpa [RawArtifact.inactiveDPreRow, RowHolds, lcEval, openZero, one,
      goldilocksP] using holds
  simpa [Nat.mod_eq_of_lt bitCanonical] using equation

def stepWordValue (assignment : Nat → Nat) (index : Nat) : Nat :=
  lcEval assignment (rawArtifact.stepWordTerms index)

def dPreWordValue (assignment : Nat → Nat) (index : Nat) : Nat :=
  lcEval assignment (rawArtifact.dPreWordTerms index)

private def stepWordDigits
    (assignment : Nat → Nat) (index : Nat) : List Nat :=
  (List.range (stepWordWidths.getD index 0)).map fun bit =>
    assignment (rawArtifact.payloadColumn
      (wordStart stepWordWidths index + bit))

private theorem ofDigits_range_map
    (base count : Nat) (digit : Nat → Nat) :
    Nat.ofDigits base ((List.range count).map digit) =
      (List.range count).foldl
        (fun value index => value + base ^ index * digit index) 0 := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      simp [List.range_succ, Nat.ofDigits_append, inductionHypothesis]

private theorem payload_column_mem_step_bits
    (index : Nat) (bounded : index < stepBitFields) :
    rawArtifact.payloadColumn index ∈ rawArtifact.stepBitColumns := by
  rw [RawArtifact.stepBitColumns, List.mem_take_iff_getElem]
  have payloadLength :
      rawArtifact.payloadColumns.length = delayedPayloadFields := by
    rw [rawArtifact_valid.payloadColumns]
    simp
  have indexPayload : index < rawArtifact.payloadColumns.length := by
    rw [payloadLength]
    norm_num [stepBitFields, delayedPayloadFields] at bounded ⊢
    omega
  refine ⟨index, by omega, ?_⟩
  change rawArtifact.payloadColumns[index]'indexPayload =
    rawArtifact.payloadColumns.getD index 0
  rw [List.getD_eq_getElem?_getD,
    getElem?_pos rawArtifact.payloadColumns index indexPayload,
    Option.getD_some]

structure Sound (assignment : Nat → Nat) : Prop where
  stepBitsExact : ∀ column ∈ rawArtifact.stepBitColumns,
    assignment column = 0 ∨ assignment column = 1
  openExact : assignment rawArtifact.openColumn = 0 ∨
    assignment rawArtifact.openColumn = 1
  dPreBitsExact : ∀ column ∈ rawArtifact.dPreBitColumns,
    assignment column = 0 ∨ assignment column = 1
  inactiveDPreZero : assignment rawArtifact.openColumn = 0 →
    ∀ column ∈ rawArtifact.dPreBitColumns, assignment column = 0
  stepWords : ∀ index < stepWordCount,
    assignment (rawArtifact.stepWordColumn index) =
      stepWordValue assignment index
  constantZero : assignment rawArtifact.constantZeroColumn = 0
  dPreWords : ∀ index < dPreWordCount,
    assignment (rawArtifact.dPreWordColumn index) =
      dPreWordValue assignment index

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.DecodeSatisfied assignment) :
    Sound assignment := by
  have pieces := all_decode_pieces_satisfied assignment satisfied
  have stepBitsSatisfied : Satisfies rawArtifact.stepBitRows assignment :=
    pieces _ (by simp [RawArtifact.decodePieces])
  have stepWordsSatisfied : Satisfies rawArtifact.stepWordRows assignment :=
    pieces _ (by simp [RawArtifact.decodePieces])
  have constantZeroSatisfied : Satisfies [rawArtifact.constantZeroRow] assignment :=
    pieces _ (by simp [RawArtifact.decodePieces])
  have openBitSatisfied : Satisfies rawArtifact.openBitRows assignment :=
    pieces _ (by simp [RawArtifact.decodePieces])
  have dPreBitsSatisfied : Satisfies rawArtifact.dPreBitRows assignment :=
    pieces _ (by simp [RawArtifact.decodePieces])
  have inactiveSatisfied : Satisfies rawArtifact.inactiveDPreRows assignment :=
    pieces _ (by simp [RawArtifact.decodePieces])
  have dPreWordsSatisfied : Satisfies rawArtifact.dPreWordRows assignment :=
    pieces _ (by simp [RawArtifact.decodePieces])
  have stepBitsExact := bit_rows_exact assignment canonical one
    rawArtifact.stepBitColumns stepBitsSatisfied
  have openBitsExact := bit_rows_exact assignment canonical one
    [rawArtifact.openColumn] (by
      simpa [RawArtifact.openBitRows] using openBitSatisfied)
  have dPreBitsExact := bit_rows_exact assignment canonical one
    rawArtifact.dPreBitColumns dPreBitsSatisfied
  refine {
    stepBitsExact := stepBitsExact
    openExact := openBitsExact rawArtifact.openColumn (by simp)
    dPreBitsExact := dPreBitsExact
    inactiveDPreZero := ?_
    stepWords := ?_
    constantZero := ?_
    dPreWords := ?_ }
  · intro openZero column member
    have holds := inactiveSatisfied (rawArtifact.inactiveDPreRow column)
      (List.mem_map.mpr ⟨column, member, rfl⟩)
    exact inactive_row_zero assignment canonical one rawArtifact column
      openZero holds
  · intro index bounded
    have indexMember : index ∈ List.range stepWordCount := by
      simp [bounded]
    have normalizedHolds := stepWordsSatisfied (rawArtifact.stepWordRow index)
      (List.mem_map.mpr ⟨index, indexMember, rfl⟩)
    have holds := normalized_linear_holds_builder assignment
      (rawArtifact.stepWordColumn index) (rawArtifact.stepWordTerms index)
      normalizedHolds
    exact builderLinearRow_sound canonical one
      (rawArtifact.stepWordColumn index)
      (rawArtifact.stepWordTerms index)
      (word_terms_canonical rawArtifact
        (wordStart stepWordWidths index)
        (stepWordWidths.getD index 0)
        (step_word_width_le index bounded)) holds
  · have normalizedHolds := constantZeroSatisfied rawArtifact.constantZeroRow (by simp)
    have holds := normalized_linear_holds_builder assignment
      rawArtifact.constantZeroColumn [] normalizedHolds
    have exact := builderLinearRow_sound canonical one
      rawArtifact.constantZeroColumn [] (by simp [CanonicalTerms]) holds
    simpa [lcEval] using exact
  · intro index bounded
    have indexMember : index ∈ List.range dPreWordCount := by
      simp [bounded]
    have normalizedHolds := dPreWordsSatisfied (rawArtifact.dPreWordRow index)
      (List.mem_map.mpr ⟨index, indexMember, rfl⟩)
    have holds := normalized_linear_holds_builder assignment
      (rawArtifact.dPreWordColumn index) (rawArtifact.dPreWordTerms index)
      normalizedHolds
    exact builderLinearRow_sound canonical one
      (rawArtifact.dPreWordColumn index)
      (rawArtifact.dPreWordTerms index)
      (word_terms_canonical rawArtifact
        (dPreBitStart + 64 * index) 64 (by decide)) holds

private theorem step_word_value_lt
    (assignment : Nat → Nat) (sound : Sound assignment)
    (index : Nat)
    (sliceBound :
      wordStart stepWordWidths index + stepWordWidths.getD index 0 ≤
        stepBitFields)
    (fits : 2 ^ (stepWordWidths.getD index 0) ≤ goldilocksP) :
    stepWordValue assignment index <
      2 ^ (stepWordWidths.getD index 0) := by
  let width := stepWordWidths.getD index 0
  let start := wordStart stepWordWidths index
  have digitsBinary :
      ∀ digit ∈ stepWordDigits assignment index, digit < 2 := by
    intro digit member
    rcases List.mem_map.mp member with ⟨bit, bitMember, rfl⟩
    have bitBound : bit < width := by
      simpa [stepWordDigits, width] using bitMember
    have columnMember :
        rawArtifact.payloadColumn (start + bit) ∈
          rawArtifact.stepBitColumns := by
      apply payload_column_mem_step_bits
      dsimp [start, width] at *
      omega
    change assignment (rawArtifact.payloadColumn (start + bit)) < 2
    rcases sound.stepBitsExact _ columnMember with h | h
    · omega
    · omega
  have decodedBound :
      Nat.ofDigits 2 (stepWordDigits assignment index) < 2 ^ width := by
    simpa [stepWordDigits, width] using
      (Nat.ofDigits_lt_base_pow_length (by decide : 1 < 2) digitsBinary)
  have decodedBelowField :
      Nat.ofDigits 2 (stepWordDigits assignment index) < goldilocksP :=
    decodedBound.trans_le (by simpa [width] using fits)
  have valueExact :
      stepWordValue assignment index =
        Nat.ofDigits 2 (stepWordDigits assignment index) := by
    unfold stepWordValue
    rw [lcEval]
    simp only [RawArtifact.stepWordTerms, RawArtifact.wordTerms,
      List.foldl_map]
    rw [← ofDigits_range_map 2 width
      (fun bit => assignment (rawArtifact.payloadColumn (start + bit)))]
    change Nat.ofDigits 2 (stepWordDigits assignment index) %
        goldilocksP = Nat.ofDigits 2 (stepWordDigits assignment index)
    exact Nat.mod_eq_of_lt decodedBelowField
  rw [valueExact]
  simpa [width] using decodedBound

/-- The delayed segment and step counters are exact 16-bit words. -/
theorem counter_words_bound
    (assignment : Nat → Nat) (sound : Sound assignment) :
    assignment (rawArtifact.stepWordColumn 0) < 2 ^ 16 ∧
      assignment (rawArtifact.stepWordColumn 1) < 2 ^ 16 := by
  constructor
  · rw [sound.stepWords 0 (by norm_num [stepWordCount, stepWordWidths])]
    exact step_word_value_lt assignment sound 0
      (by norm_num [wordStart, stepWordWidths, stepBitFields])
      (by norm_num [stepWordWidths, goldilocksP])
  · rw [sound.stepWords 1 (by norm_num [stepWordCount, stepWordWidths])]
    exact step_word_value_lt assignment sound 1
      (by norm_num [wordStart, stepWordWidths, stepBitFields])
      (by norm_num [stepWordWidths, goldilocksP])

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerDecodeRowSound
