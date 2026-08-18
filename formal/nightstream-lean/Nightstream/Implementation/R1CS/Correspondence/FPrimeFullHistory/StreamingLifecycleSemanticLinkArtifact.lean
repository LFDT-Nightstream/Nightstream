import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLink
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact Rust semantic-link rows refine an independent Lean relation.

Assurance tier: artifact-checked, conditional on the Rust drift test that
compares every source row with the compact recipe.

Does not own phase-local semantics, the complete lifecycle relation,
selective-CCS lowering, or Poseidon2 collision resistance.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLinkArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLink

private theorem piece_satisfies
    {artifact :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact}
    {assignment : Nat → Nat}
    (satisfied : artifact.Satisfied assignment)
    {piece : List Row} (member : piece ∈ artifact.programPieces) :
    Satisfies piece assignment := by
  intro row rowMember
  exact satisfied row (List.mem_flatten.mpr ⟨piece, member, rowMember⟩)

private theorem payload_piece_satisfies
    (assignment : Nat → Nat) (satisfied : rawArtifact.Satisfied assignment) :
    Satisfies rawArtifact.payloadRows assignment :=
  piece_satisfies satisfied (by simp [RawArtifact.programPieces])

private theorem trace_piece_satisfies
    (side : StateSide) (assignment : Nat → Nat)
    (satisfied : rawArtifact.Satisfied assignment) :
    Satisfies (rawArtifact.hashRecipe side).trace.rows assignment := by
  cases side <;>
    exact piece_satisfies satisfied (by simp [RawArtifact.programPieces])

private theorem constant_piece_satisfies
    (side : StateSide) (assignment : Nat → Nat)
    (satisfied : rawArtifact.Satisfied assignment) :
    Satisfies (constantRows (rawArtifact.hashRecipe side)) assignment := by
  cases side <;>
    exact piece_satisfies satisfied (by simp [RawArtifact.programPieces])

private theorem equality_piece_satisfies
    (assignment : Nat → Nat) (satisfied : rawArtifact.Satisfied assignment) :
    Satisfies rawArtifact.equalityRows assignment :=
  piece_satisfies satisfied (by simp [RawArtifact.programPieces])

private theorem constant_rows_values
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (columns values : List Nat)
    (lengths : columns.length = values.length)
    (valuesCanonical : ∀ value ∈ values, value < goldilocksP)
    (holds : Satisfies
      ((columns.zip values).map fun entry =>
        builderLinearRow entry.1 [(0, entry.2)]) assignment) :
    columns.map assignment = values := by
  induction columns generalizing values with
  | nil =>
      cases values <;> simp_all
  | cons column columns inductionHypothesis =>
      cases values with
      | nil => simp at lengths
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          have rowHolds := holds (builderLinearRow column [(0, value)]) (by simp)
          have valueCanonical : value < goldilocksP :=
            valuesCanonical value (by simp)
          have head : assignment column = value := by
            by_cases zero : value = 0
            · subst value
              simpa [builderLinearRow, RowHolds, lcEval, negateTerms,
                negCoeff, one, Nat.mod_eq_of_lt (canonical column)] using
                rowHolds
            · have defined := builderLinearRow_sound canonical one column
                [(0, value)]
                (by simp [CanonicalTerms, Nat.pos_of_ne_zero zero,
                  valueCanonical]) rowHolds
              simpa [lcEval, one, Nat.mod_eq_of_lt valueCanonical] using defined
          have tailHolds : Satisfies
              ((columns.zip values).map fun entry =>
                builderLinearRow entry.1 [(0, entry.2)]) assignment := by
            intro row member
            exact holds row (by simp [member])
          simp only [List.map_cons, List.cons.injEq]
          exact ⟨head, inductionHypothesis values lengths
            (fun current member => valuesCanonical current
              (List.mem_cons_of_mem value member))
            tailHolds⟩

private theorem constants_canonical :
    ∀ value ∈ phaseConstantValues, value < goldilocksP := by
  norm_num [phaseConstantValues, goldilocksP]

private theorem hash_input_values
    (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    (rawArtifact.hashRecipe side).trace.inputColumns.map assignment =
      phasePreimage rawArtifact side assignment := by
  have constants := constant_rows_values assignment canonical one
    (rawArtifact.hashRecipe side).constantColumns
    (rawArtifact.hashRecipe side).constantValues
    (by cases side <;> rfl) constants_canonical
    (by simpa [constantRows] using
      constant_piece_satisfies side assignment satisfied)
  change (rawArtifact.hashRecipe side).inputColumns.map assignment = _
  rw [HashRecipe.inputColumns, List.map_append, List.map_append, constants]
  rfl

private theorem phase_preimage_length
    (side : StateSide) (assignment : Nat → Nat) :
    (phasePreimage rawArtifact side assignment).length = 4 * absorbRounds := by
  cases side <;>
    norm_num [phasePreimage, RawArtifact.localColumns,
      RawArtifact.payloadColumns, RawArtifact.payloadStartColumn,
      rawArtifact, phaseConstantValues, absorbRounds, hashInputFields,
      hashConstantFields, domainFields, payloadFields, digestFields]

private theorem phase_preimage_canonical
    (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ value ∈ phasePreimage rawArtifact side assignment,
      value < goldilocksP := by
  intro value member
  rw [phasePreimage, List.mem_append] at member
  rcases member with constantsAndLocal | payloadMember
  · rw [List.mem_append] at constantsAndLocal
    rcases constantsAndLocal with constantMember | localMember
    · exact constants_canonical value constantMember
    · rcases List.mem_map.mp localMember with ⟨column, _member, rfl⟩
      exact canonical column
  · rcases List.mem_map.mp payloadMember with ⟨column, _member, rfl⟩
    exact canonical column

private theorem hash_rows_refine
    (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment ((rawArtifact.hashOutputColumns side).getD lane.val 0) =
      phaseEnvelopeDigest rawArtifact side assignment lane := by
  calc
    assignment ((rawArtifact.hashOutputColumns side).getD lane.val 0) =
        runValueRounds (rawArtifact.hashRecipe side).trace.rounds
          ((rawArtifact.hashRecipe side).trace.inputColumns.map assignment)
          (fun _ => 0) lane.val := by
      exact ownedTrace_values_sound (trace_ownedValid side) canonical one
        (trace_piece_satisfies side assignment satisfied) lane.val lane.isLt
    _ = runValueRounds (rawArtifact.hashRecipe side).trace.rounds
          (phasePreimage rawArtifact side assignment) (fun _ => 0) lane.val := by
      rw [hash_input_values side assignment canonical one satisfied]
    _ = runValueRounds
          (Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds
            absorbRounds)
          (phasePreimage rawArtifact side assignment) (fun _ => 0) lane.val := by
      exact congrFun
        (runValueRounds_eq_of_schedules (valueSchedules_exact side)
          (phasePreimage rawArtifact side assignment) (fun _ => 0)) lane.val
    _ = phaseEnvelopeDigest rawArtifact side assignment lane := by
      exact
        Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds_compute_digest
          absorbRounds (phasePreimage rawArtifact side assignment)
          (phase_preimage_length side assignment)
          (phase_preimage_canonical side assignment canonical) lane

private theorem equality_row_mem
    (side : StateSide) (lane : Fin 4) :
    builderLinearRow
        ((rawArtifact.semanticColumns side).getD lane.val 0)
        [((rawArtifact.hashOutputColumns side).getD lane.val 0, 1)] ∈
      rawArtifact.equalityRows := by
  rw [RawArtifact.equalityRows]
  apply List.mem_flatMap.mpr
  refine ⟨lane.val, List.mem_range.mpr ?_, ?_⟩
  · simpa [digestFields] using lane.isLt
  · cases side <;>
      simp [RawArtifact.semanticColumns, RawArtifact.hashOutputColumns]

private theorem semantic_equals_hash_output
    (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment ((rawArtifact.semanticColumns side).getD lane.val 0) =
      assignment ((rawArtifact.hashOutputColumns side).getD lane.val 0) := by
  have rowHolds := equality_piece_satisfies assignment satisfied _
    (equality_row_mem side lane)
  have defined := builderLinearRow_sound canonical one
    ((rawArtifact.semanticColumns side).getD lane.val 0)
    [((rawArtifact.hashOutputColumns side).getD lane.val 0, 1)]
    (by norm_num [CanonicalTerms, goldilocksP]) rowHolds
  have reduced :
      assignment ((rawArtifact.semanticColumns side).getD lane.val 0) =
        assignment ((rawArtifact.hashOutputColumns side).getD lane.val 0) %
          goldilocksP := by
    simpa [lcEval] using defined
  exact reduced.trans (Nat.mod_eq_of_lt
    (canonical ((rawArtifact.hashOutputColumns side).getD lane.val 0)))

/-- Exact Rust rows imply the independent same-wire semantic-link target. -/
theorem rows_refine_semanticLink
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    SemanticLink rawArtifact assignment := by
  constructor
  · intro side column member
    have payloadMember : bitRow column ∈ rawArtifact.payloadRows := by
      rw [RawArtifact.payloadRows]
      apply List.mem_map.mpr
      refine ⟨column, ?_, rfl⟩
      rw [List.mem_append]
      cases side with
      | before => exact Or.inl member
      | after => exact Or.inr member
    apply bitRow_le_one goldilocks_euclidPrime (canonical column) one
    exact payload_piece_satisfies assignment satisfied _ payloadMember
  · intro side lane
    exact (semantic_equals_hash_output side assignment canonical one
      satisfied lane).trans
        (hash_rows_refine side assignment canonical one satisfied lane)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLinkArtifact
