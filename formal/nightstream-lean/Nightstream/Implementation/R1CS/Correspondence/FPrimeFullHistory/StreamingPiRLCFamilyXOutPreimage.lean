import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPublicState
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeArtifact

/-!
Contract: exact role map for the 32 physical full-XOut preimage fields of one
PiRLC family arm.

Owns the generated rows that fix the XOut domain, both copies of the global
program-cursor halves, the fixed program-counter halves, the stateful semantic
envelope slots, and the Nebula-present marker. It also gives names to the five
four-field outer coordinates that this phase carries as opaque data.

Does not give authority to the verifier digest, PiCCS header, boundary,
Construction-2 accumulator, or Nebula digest. Lifecycle circuits must derive
those fields from verifier-owned inputs and checked transitions. Common-to-
phase links must bind the delayed payload to the selected physical arm.

Assurance tier: artifact-checked for property
`FPRIME-STREAMING-PIRLC-FAMILY-XOUT-PREIMAGE-V1`.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutPreimage

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyState
open Nightstream.Implementation.R1CS.Program

def xOutDomain : Nat := 0x4e460002

def nebulaPresentMarker : Nat := 0x4e424c41

/-- Phase artifact parity that corresponds to one public-suffix parity. -/
def phaseArtifactKindFor : ArmKind →
    FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.ArmKind
  | .even => .even
  | .odd => .odd

def cursorCallIndex : StateSide -> Nat
  | .before => 0
  | .after => 1

def cursorCall (kind : ArmKind) (side : StateSide) : CanonicalCall :=
  (armFor kind).canonicalCalls.getD (cursorCallIndex side) default

def pcCall (kind : ArmKind) : CanonicalCall :=
  (armFor kind).canonicalCalls.getD 2 default

def artifactLinearRow (output : Nat) (terms : List (Nat × Nat)) : Row :=
  ⟨negateTerms terms ++ [(output, 1)], [(0, 1)], []⟩

def rawStructuralRows (kind : ArmKind) (side : StateSide) : List Row :=
  [artifactLinearRow (xOutPreimageColumn kind side 0) [(0, xOutDomain)],
    artifactLinearRow (pcCall kind).fieldColumn [(0, 1)],
    artifactLinearRow (xOutPreimageColumn kind side 9)
      (lowTerms (cursorCall kind side).layout),
    artifactLinearRow (xOutPreimageColumn kind side 10)
      (highTerms (cursorCall kind side).layout),
    artifactLinearRow (xOutPreimageColumn kind side 11)
      (lowTerms (cursorCall kind side).layout),
    artifactLinearRow (xOutPreimageColumn kind side 12)
      (highTerms (cursorCall kind side).layout),
    artifactLinearRow (xOutPreimageColumn kind side 13)
      (lowTerms (pcCall kind).layout),
    artifactLinearRow (xOutPreimageColumn kind side 14)
      (highTerms (pcCall kind).layout),
    artifactLinearRow (xOutPreimageColumn kind side 27)
      [(0, nebulaPresentMarker)]]

def structuralGlueIndices : StateSide → List Nat
  | .after => [37, 36, 38, 39, 40, 41, 42, 43, 44]
  | .before => [79, 36, 80, 81, 82, 83, 84, 85, 86]

def structuralIndexedRows (kind : ArmKind) (side : StateSide) :
    List IndexedRow :=
  indexedRowsAt (armFor kind).glueRows (structuralGlueIndices side)

private theorem glue_rows_length (kind : ArmKind) :
    (armFor kind).glueRows.length = 121 := by
  cases kind with
  | even =>
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenGlueRowCertificate.evenArm_glueRows_length
  | odd =>
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddGlueRowCertificate.oddArm_glueRows_length

private theorem structural_glue_indices_bounded
    (kind : ArmKind) (side : StateSide) :
    ∀ index ∈ structuralGlueIndices side,
      index < (armFor kind).glueRows.length := by
  intro index member
  rw [glue_rows_length]
  cases side <;> simp [structuralGlueIndices] at member <;> omega

/-- The nine normalized XOut structural rows are the exact named rows in the
Rust artifact. This is a structural certificate over nine fixed positions;
it does not search or decide the generated row set. -/
theorem structural_rows_exact (kind : ArmKind) (side : StateSide) :
    rawStructuralRows kind side =
      (structuralIndexedRows kind side).map IndexedRow.row := by
  cases kind <;> cases side <;> rfl

private theorem raw_structural_satisfies
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (rawStructuralRows kind side) assignment := by
  intro row member
  rw [structural_rows_exact kind side] at member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed
    (indexedRowsAt_subset (structural_glue_indices_bounded kind side)
      indexed indexedMember)

private theorem rowHolds_of_operand_perms
    (assignment : Nat → Nat) {source target : Row}
    (a : source.a.Perm target.a)
    (b : source.b.Perm target.b)
    (c : source.c.Perm target.c)
    (holds : RowHolds assignment source) :
    RowHolds assignment target := by
  unfold RowHolds at holds ⊢
  calc
    lcEval assignment target.a * lcEval assignment target.b % goldilocksP =
        lcEval assignment source.a * lcEval assignment source.b %
          goldilocksP := by
      rw [Program.lcEval_eq_of_perm assignment a,
        Program.lcEval_eq_of_perm assignment b]
    _ = lcEval assignment source.c := holds
    _ = lcEval assignment target.c :=
      Program.lcEval_eq_of_perm assignment c

private theorem artifact_linear_row_a_perm (output : Nat)
    (terms : List (Nat × Nat)) :
    (artifactLinearRow output terms).a.Perm
      (builderLinearRow output terms).a := by
  simpa [artifactLinearRow, builderLinearRow] using
    (List.perm_append_comm : List.Perm
      (negateTerms terms ++ [(output, 1)])
      ([(output, 1)] ++ negateTerms terms))

private theorem artifact_linear_row_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (output : Nat) (terms : List (Nat × Nat))
    (termsCanonical : CanonicalTerms terms)
    (holds : RowHolds assignment (artifactLinearRow output terms)) :
    assignment output = lcEval assignment terms := by
  have builderHolds := rowHolds_of_operand_perms assignment
    (artifact_linear_row_a_perm output terms) (List.Perm.refl _)
    (List.Perm.refl _) holds
  exact builderLinearRow_sound canonical one output terms termsCanonical
    builderHolds

private theorem cursor_call_mem (kind : ArmKind) (side : StateSide) :
    cursorCall kind side ∈ (armFor kind).canonicalCalls := by
  cases kind <;> cases side <;> native_decide

private theorem pc_call_mem (kind : ArmKind) :
    pcCall kind ∈ (armFor kind).canonicalCalls := by
  cases kind <;> native_decide

private theorem cursor_call_field_column
    (kind : ArmKind) (side : StateSide) :
    (cursorCall kind side).fieldColumn = programCursorColumn kind side := by
  cases kind <;> cases side <;> native_decide

private theorem cursor_low_terms_canonical
    (kind : ArmKind) (side : StateSide) :
    CanonicalTerms (lowTerms (cursorCall kind side).layout) := by
  cases kind <;> cases side <;> native_decide

private theorem cursor_high_terms_canonical
    (kind : ArmKind) (side : StateSide) :
    CanonicalTerms (highTerms (cursorCall kind side).layout) := by
  cases kind <;> cases side <;> native_decide

private theorem pc_low_terms_canonical (kind : ArmKind) :
    CanonicalTerms (lowTerms (pcCall kind).layout) := by
  cases kind <;> native_decide

private theorem pc_high_terms_canonical (kind : ArmKind) :
    CanonicalTerms (highTerms (pcCall kind).layout) := by
  cases kind <;> native_decide

private theorem low_terms_eval
    (call : CanonicalCall)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (callSatisfied : Satisfies (CanonicalU64Recipe.rows call.layout) assignment) :
    lcEval assignment (lowTerms call.layout) =
      lowValue assignment call.layout := by
  have raw :
      (lowTerms call.layout).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
      lowValue assignment call.layout := by
    simp [lowTerms, lowValue, bitValue, List.foldl_map]
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt
    (lowValue_le_highMax goldilocks_euclidPrime canonical one callSatisfied)
    (by decide)

private theorem high_terms_eval
    (call : CanonicalCall)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (callSatisfied : Satisfies (CanonicalU64Recipe.rows call.layout) assignment) :
    lcEval assignment (highTerms call.layout) =
      highValue assignment call.layout := by
  have raw :
      (highTerms call.layout).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
      highValue assignment call.layout := by
    simp [highTerms, highValue, bitValue, List.foldl_map]
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt
    (highValue_le_highMax goldilocks_euclidPrime canonical one callSatisfied)
    (by decide)

private theorem cursor_call_value
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    CanonicalU64RecipeSound.bitsValue assignment
        (cursorCall kind side).layout =
      assignment (programCursorColumn kind side) := by
  have refined := canonical_call_refines (armFor kind) assignment canonical
    one satisfied (cursorCall kind side) (cursor_call_mem kind side)
  have input := refined.input_eq.symm
  simpa [CanonicalCall.layout, lcEval,
    cursor_call_field_column kind side,
    Nat.mod_eq_of_lt (canonical (programCursorColumn kind side))] using input

private theorem pc_call_value
    (kind : ArmKind)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    CanonicalU64RecipeSound.bitsValue assignment (pcCall kind).layout = 1 := by
  have rows := raw_structural_satisfies kind .after assignment satisfied
  have constantHolds := rows
    (artifactLinearRow (pcCall kind).fieldColumn [(0, 1)]) (by
      simp [rawStructuralRows])
  have pcExact := artifact_linear_row_sound assignment canonical one
    (pcCall kind).fieldColumn [(0, 1)] (by native_decide) constantHolds
  have refined := canonical_call_refines (armFor kind) assignment canonical
    one satisfied (pcCall kind) (pc_call_mem kind)
  have input := refined.input_eq.symm
  have fieldExact : assignment (pcCall kind).fieldColumn = 1 := by
    simpa [lcEval, one] using pcExact
  simpa [CanonicalCall.layout, lcEval, fieldExact,
    Nat.mod_eq_of_lt (canonical (pcCall kind).fieldColumn)] using input

private theorem small_recipe_halves
    (call : CanonicalCall)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (callSatisfied : Satisfies (CanonicalU64Recipe.rows call.layout) assignment)
    (value : Nat)
    (valueExact :
      CanonicalU64RecipeSound.bitsValue assignment call.layout = value)
    (valueBound : value < 2 ^ 32) :
    lowValue assignment call.layout = value ∧
      highValue assignment call.layout = 0 := by
  have lowBound :=
    lowValue_le_highMax goldilocks_euclidPrime canonical one callSatisfied
  have highBound :=
    highValue_le_highMax goldilocks_euclidPrime canonical one callSatisfied
  have split := CanonicalU64RecipeSound.bitsValue_eq_low_add_high
    assignment call.layout
  rw [valueExact] at split
  unfold highMax at lowBound highBound
  omega

private theorem half_output_exact
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (output : Nat) (terms : List (Nat × Nat))
    (termsCanonical : CanonicalTerms terms)
    (rowMember : artifactLinearRow output terms ∈
      rawStructuralRows kind side) :
    assignment output = lcEval assignment terms := by
  exact artifact_linear_row_sound assignment canonical one output terms
    termsCanonical
    (raw_structural_satisfies kind side assignment satisfied
      (artifactLinearRow output terms) rowMember)

/-- Exact non-hash field facts for one accepted generated XOut preimage.

The five four-field outer values that are not listed here remain opaque
assignment data. This result does not give them lifecycle authority. -/
structure PreimageBinding
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  publicState :
    FPrimeFullHistoryStreamingPiRLCFamilyPublicState.Binding
      kind assignment canonical
  domain : assignment (xOutPreimageColumn kind side 0) = xOutDomain
  chunkCountLow :
    assignment (xOutPreimageColumn kind side 9) =
      assignment (programCursorColumn kind side)
  chunkCountHigh : assignment (xOutPreimageColumn kind side 10) = 0
  stepCountLow :
    assignment (xOutPreimageColumn kind side 11) =
      assignment (programCursorColumn kind side)
  stepCountHigh : assignment (xOutPreimageColumn kind side 12) = 0
  pcLow : assignment (xOutPreimageColumn kind side 13) = 1
  pcHigh : assignment (xOutPreimageColumn kind side 14) = 0
  semanticState : ∀ lane : Fin 4,
    assignment (xOutPreimageColumn kind side (19 + lane.val)) =
      FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.phaseEnvelopeDigest
        (phaseArtifactKindFor kind) (phaseEnvelopeSideFor side) assignment lane
  nebulaPresent :
    assignment (xOutPreimageColumn kind side 27) = nebulaPresentMarker

/-- Accepted generated rows determine every structural XOut preimage field.
All other outer coordinates are named by position but remain non-authoritative
until a lifecycle circuit derives them. -/
theorem x_out_preimage_refines
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat -> Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (phaseSatisfied :
      (FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.armFor
        (phaseArtifactKindFor kind)).Satisfied
          Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope.phaseConstantValues
          assignment)
    (beforeBound :
      (familyStateAt assignment canonical kind .before).familyCursor < 111)
    (afterBound :
      (familyStateAt assignment canonical kind .after).familyCursor < 111) :
    PreimageBinding kind side assignment canonical := by
  have publicState := shared_public_state_refines kind assignment canonical
    one satisfied beforeBound afterBound
  have cursorExact : assignment (programCursorColumn kind side) =
      (familyStateAt assignment canonical kind side).familyCursor +
        firstFamilyProgramCursor := by
    cases side with
    | before =>
        calc
          assignment (programCursorColumn kind .before) =
              publicWordValue assignment kind
                (cursorPublicWordIndex .before) := by
            simpa [programCursorColumn] using
              (public_word_refines kind (cursorPublicWordIndex .before)
                assignment canonical one satisfied).symm
          _ = (familyStateAt assignment canonical kind .before).familyCursor +
                firstFamilyProgramCursor := publicState.beforeCursor
    | after =>
        calc
          assignment (programCursorColumn kind .after) =
              publicWordValue assignment kind
                (cursorPublicWordIndex .after) := by
            simpa [programCursorColumn] using
              (public_word_refines kind (cursorPublicWordIndex .after)
                assignment canonical one satisfied).symm
          _ = (familyStateAt assignment canonical kind .after).familyCursor +
                firstFamilyProgramCursor := publicState.afterCursor
  have cursorBound : assignment (programCursorColumn kind side) < 2 ^ 32 := by
    rw [cursorExact]
    cases side <;> simp_all [firstFamilyProgramCursor] <;> omega
  have cursorCallSatisfied :
      Satisfies (CanonicalU64Recipe.rows (cursorCall kind side).layout)
        assignment :=
    satisfied.1 (cursorCall kind side) (cursor_call_mem kind side)
  have cursorHalves := small_recipe_halves (cursorCall kind side) assignment
    canonical one cursorCallSatisfied
    (assignment (programCursorColumn kind side))
    (cursor_call_value kind side assignment canonical one satisfied)
    cursorBound
  have pcCallSatisfied :
      Satisfies (CanonicalU64Recipe.rows (pcCall kind).layout) assignment :=
    satisfied.1 (pcCall kind) (pc_call_mem kind)
  have pcHalves := small_recipe_halves (pcCall kind) assignment canonical one
    pcCallSatisfied 1 (pc_call_value kind assignment canonical one satisfied)
    (by decide)
  have rawRows := raw_structural_satisfies kind side assignment satisfied
  refine {
    publicState := publicState
    domain := ?_
    chunkCountLow := ?_
    chunkCountHigh := ?_
    stepCountLow := ?_
    stepCountHigh := ?_
    pcLow := ?_
    pcHigh := ?_
    semanticState := ?_
    nebulaPresent := ?_ }
  · have exact := artifact_linear_row_sound assignment canonical one
      (xOutPreimageColumn kind side 0) [(0, xOutDomain)]
      (by unfold xOutDomain; native_decide)
      (rawRows _ (by simp [rawStructuralRows]))
    simpa [lcEval, one] using exact
  · rw [half_output_exact kind side assignment canonical one satisfied
      (xOutPreimageColumn kind side 9)
      (lowTerms (cursorCall kind side).layout)
      (cursor_low_terms_canonical kind side) (by simp [rawStructuralRows]),
      low_terms_eval (cursorCall kind side) assignment canonical one
        cursorCallSatisfied,
      cursorHalves.1]
  · rw [half_output_exact kind side assignment canonical one satisfied
      (xOutPreimageColumn kind side 10)
      (highTerms (cursorCall kind side).layout)
      (cursor_high_terms_canonical kind side) (by simp [rawStructuralRows]),
      high_terms_eval (cursorCall kind side) assignment canonical one
        cursorCallSatisfied,
      cursorHalves.2]
  · rw [half_output_exact kind side assignment canonical one satisfied
      (xOutPreimageColumn kind side 11)
      (lowTerms (cursorCall kind side).layout)
      (cursor_low_terms_canonical kind side) (by simp [rawStructuralRows]),
      low_terms_eval (cursorCall kind side) assignment canonical one
        cursorCallSatisfied,
      cursorHalves.1]
  · rw [half_output_exact kind side assignment canonical one satisfied
      (xOutPreimageColumn kind side 12)
      (highTerms (cursorCall kind side).layout)
      (cursor_high_terms_canonical kind side) (by simp [rawStructuralRows]),
      high_terms_eval (cursorCall kind side) assignment canonical one
        cursorCallSatisfied,
      cursorHalves.2]
  · rw [half_output_exact kind side assignment canonical one satisfied
      (xOutPreimageColumn kind side 13) (lowTerms (pcCall kind).layout)
      (pc_low_terms_canonical kind) (by simp [rawStructuralRows]),
      low_terms_eval (pcCall kind) assignment canonical one pcCallSatisfied,
      pcHalves.1]
  · rw [half_output_exact kind side assignment canonical one satisfied
      (xOutPreimageColumn kind side 14) (highTerms (pcCall kind).layout)
      (pc_high_terms_canonical kind) (by simp [rawStructuralRows]),
      high_terms_eval (pcCall kind) assignment canonical one pcCallSatisfied,
      pcHalves.2]
  · intro lane
    have columnExact :
        xOutPreimageColumn kind side (19 + lane.val) =
          ((FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.armFor
              (phaseArtifactKindFor kind)).xOutSemanticColumns
            (phaseEnvelopeSideFor side)).getD lane.val 0 := by
      cases kind <;> cases side <;> fin_cases lane <;> native_decide
    rw [columnExact]
    exact
      FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact.x_out_semantic_refines_phase_envelope
        (phaseArtifactKindFor kind) (phaseEnvelopeSideFor side) assignment
        canonical one phaseSatisfied lane
  · have exact := artifact_linear_row_sound assignment canonical one
      (xOutPreimageColumn kind side 27) [(0, nebulaPresentMarker)]
      (by unfold nebulaPresentMarker; native_decide)
      (rawRows _ (by simp [rawStructuralRows]))
    simpa [lcEval, one] using exact

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutPreimage
