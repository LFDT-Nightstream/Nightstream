import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSourceSemanticLink
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact base and recursive lifecycle semantic-link source rows refine
the independent Lean relation.

Base owns a zero before payload. Recursive consumes the Boolean before-payload
premise owned by its private delayed-input relation. Both own the complete
after-payload domain, Poseidon2 replay, and outer semantic links.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLinkArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSourceSemanticLink

private theorem piece_satisfies
    {artifact : SourceArtifact} {assignment : Nat → Nat}
    (satisfied : artifact.Satisfied assignment)
    {piece : List Row} (member : piece ∈ artifact.programPieces) :
    Satisfies piece assignment := by
  intro row rowMember
  exact satisfied row (List.mem_flatten.mpr ⟨piece, member, rowMember⟩)

private theorem before_payload_piece_satisfies
    (artifact : SourceArtifact) (assignment : Nat → Nat)
    (satisfied : artifact.Satisfied assignment) :
    Satisfies artifact.beforePayloadRows assignment :=
  piece_satisfies satisfied (by simp [SourceArtifact.programPieces])

private theorem after_payload_piece_satisfies
    (artifact : SourceArtifact) (assignment : Nat → Nat)
    (satisfied : artifact.Satisfied assignment) :
    Satisfies artifact.afterPayloadRows assignment :=
  piece_satisfies satisfied (by simp [SourceArtifact.programPieces])

private theorem trace_piece_satisfies
    (artifact : SourceArtifact) (side : StateSide)
    (assignment : Nat → Nat) (satisfied : artifact.Satisfied assignment) :
    Satisfies (artifact.hashRecipe side).trace.rows assignment := by
  cases side <;>
    exact piece_satisfies satisfied (by simp [SourceArtifact.programPieces])

private theorem constant_piece_satisfies
    (artifact : SourceArtifact) (side : StateSide)
    (assignment : Nat → Nat) (satisfied : artifact.Satisfied assignment) :
    Satisfies (constantRows (artifact.hashRecipe side)) assignment := by
  cases side <;>
    exact piece_satisfies satisfied (by simp [SourceArtifact.programPieces])

private theorem equality_piece_satisfies
    (artifact : SourceArtifact) (assignment : Nat → Nat)
    (satisfied : artifact.Satisfied assignment) :
    Satisfies artifact.equalityRows assignment :=
  piece_satisfies satisfied (by simp [SourceArtifact.programPieces])

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
    (artifact : SourceArtifact)
    (constantsExact : artifact.constantValues = phaseConstantValues)
    (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment) :
    (artifact.hashRecipe side).trace.inputColumns.map assignment =
      phasePreimage artifact.semanticGeometry side assignment := by
  have constants := constant_rows_values assignment canonical one
    (artifact.hashRecipe side).constantColumns
    (artifact.hashRecipe side).constantValues
    (by simp [HashRecipe.constantColumns])
    (by
      have valuesExact :
          (artifact.hashRecipe side).constantValues = phaseConstantValues := by
        simpa [SourceArtifact.hashRecipe] using constantsExact
      rw [valuesExact]
      exact constants_canonical)
    (by simpa [constantRows] using
      constant_piece_satisfies artifact side assignment satisfied)
  change (artifact.hashRecipe side).inputColumns.map assignment = _
  rw [HashRecipe.inputColumns, List.map_append, List.map_append, constants]
  rfl

private theorem phase_preimage_canonical
    (artifact : SourceArtifact)
    (constantsExact : artifact.constantValues = phaseConstantValues)
    (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ value ∈ phasePreimage artifact.semanticGeometry side assignment,
      value < goldilocksP := by
  intro value member
  rw [phasePreimage, List.mem_append] at member
  rcases member with constantsAndLocal | payloadMember
  · rw [List.mem_append] at constantsAndLocal
    rcases constantsAndLocal with constantMember | localMember
    · apply constants_canonical value
      simpa [SourceArtifact.semanticGeometry, constantsExact] using constantMember
    · rcases List.mem_map.mp localMember with ⟨column, _member, rfl⟩
      exact canonical column
  · rcases List.mem_map.mp payloadMember with ⟨column, _member, rfl⟩
    exact canonical column

private theorem hash_rows_refine
    (artifact : SourceArtifact)
    (constantsExact : artifact.constantValues = phaseConstantValues)
    (traceValid : ∀ side, (artifact.hashRecipe side).trace.OwnedValid)
    (scheduleExact : ∀ side,
      valueSchedules (artifact.hashRecipe side).trace.rounds =
        List.replicate absorbRounds (.absorb 4) ++ [.pad])
    (preimageLength : ∀ side assignment,
      (phasePreimage artifact.semanticGeometry side assignment).length =
        4 * absorbRounds)
    (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment ((artifact.hashOutputColumns side).getD lane.val 0) =
      phaseEnvelopeDigest artifact.semanticGeometry side assignment lane := by
  calc
    assignment ((artifact.hashOutputColumns side).getD lane.val 0) =
        runValueRounds (artifact.hashRecipe side).trace.rounds
          ((artifact.hashRecipe side).trace.inputColumns.map assignment)
          (fun _ => 0) lane.val := by
      exact ownedTrace_values_sound (traceValid side) canonical one
        (trace_piece_satisfies artifact side assignment satisfied)
        lane.val lane.isLt
    _ = runValueRounds (artifact.hashRecipe side).trace.rounds
          (phasePreimage artifact.semanticGeometry side assignment)
          (fun _ => 0) lane.val := by
      rw [hash_input_values artifact constantsExact side assignment
        canonical one satisfied]
    _ = runValueRounds
          (Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds
            absorbRounds)
          (phasePreimage artifact.semanticGeometry side assignment)
          (fun _ => 0) lane.val := by
      exact congrFun
        (runValueRounds_eq_of_schedules (scheduleExact side)
          (phasePreimage artifact.semanticGeometry side assignment) (fun _ => 0))
        lane.val
    _ = phaseEnvelopeDigest artifact.semanticGeometry side assignment lane := by
      exact
        Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds_compute_digest
          absorbRounds (phasePreimage artifact.semanticGeometry side assignment)
          (preimageLength side assignment)
          (phase_preimage_canonical artifact constantsExact side assignment
            canonical)
          lane

private theorem equality_row_mem
    (artifact : SourceArtifact) (side : StateSide) (lane : Fin 4) :
    builderLinearRow
        ((artifact.semanticColumns side).getD lane.val 0)
        [((artifact.hashOutputColumns side).getD lane.val 0, 1)] ∈
      artifact.equalityRows := by
  rw [SourceArtifact.equalityRows, List.mem_append]
  cases side with
  | before =>
      apply Or.inl
      apply List.mem_map.mpr
      exact ⟨lane.val, List.mem_range.mpr lane.isLt, by
        simp [SourceArtifact.semanticColumns,
          SourceArtifact.hashOutputColumns]⟩
  | after =>
      apply Or.inr
      apply List.mem_map.mpr
      exact ⟨lane.val, List.mem_range.mpr lane.isLt, by
        simp [SourceArtifact.semanticColumns,
          SourceArtifact.hashOutputColumns]⟩

private theorem semantic_equals_hash_output
    (artifact : SourceArtifact) (side : StateSide)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (lane : Fin 4) :
    assignment ((artifact.semanticColumns side).getD lane.val 0) =
      assignment ((artifact.hashOutputColumns side).getD lane.val 0) := by
  have rowHolds := equality_piece_satisfies artifact assignment satisfied _
    (equality_row_mem artifact side lane)
  have defined := builderLinearRow_sound canonical one
    ((artifact.semanticColumns side).getD lane.val 0)
    [((artifact.hashOutputColumns side).getD lane.val 0, 1)]
    (by norm_num [CanonicalTerms, goldilocksP]) rowHolds
  have reduced :
      assignment ((artifact.semanticColumns side).getD lane.val 0) =
        assignment ((artifact.hashOutputColumns side).getD lane.val 0) %
          goldilocksP := by
    simpa [lcEval] using defined
  exact reduced.trans (Nat.mod_eq_of_lt
    (canonical ((artifact.hashOutputColumns side).getD lane.val 0)))

private theorem after_bit_row_mem
    (artifact : SourceArtifact) {column : Nat}
    (member : column ∈ artifact.payloadColumns .after) :
    bitRow column ∈ artifact.afterPayloadRows := by
  rw [SourceArtifact.afterPayloadRows]
  exact List.mem_map.mpr ⟨column, member, rfl⟩

private theorem base_before_zero_row_mem
    {column : Nat} (member : column ∈ baseArtifact.payloadColumns .before) :
    builderLinearRow column [] ∈ baseArtifact.beforePayloadRows := by
  rw [SourceArtifact.beforePayloadRows]
  exact List.mem_map.mpr ⟨column, member, rfl⟩

private theorem base_before_payload_zero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : baseArtifact.Satisfied assignment)
    {column : Nat} (member : column ∈ baseArtifact.payloadColumns .before) :
    assignment column = 0 := by
  have rowHolds := before_payload_piece_satisfies baseArtifact assignment
    satisfied _ (base_before_zero_row_mem member)
  have defined := builderLinearRow_sound canonical one column []
    (by simp [CanonicalTerms]) rowHolds
  simpa [lcEval] using defined

private theorem source_rows_refine
    (artifact : SourceArtifact)
    (constantsExact : artifact.constantValues = phaseConstantValues)
    (traceValid : ∀ side, (artifact.hashRecipe side).trace.OwnedValid)
    (scheduleExact : ∀ side,
      valueSchedules (artifact.hashRecipe side).trace.rounds =
        List.replicate absorbRounds (.absorb 4) ++ [.pad])
    (preimageLength : ∀ side assignment,
      (phasePreimage artifact.semanticGeometry side assignment).length =
        4 * absorbRounds)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (beforeBinary : ∀ column,
      column ∈ artifact.payloadColumns .before → assignment column ≤ 1) :
    artifact.SemanticLink assignment := by
  constructor
  · intro side column member
    cases side with
    | before => exact beforeBinary column member
    | after =>
        apply bitRow_le_one goldilocks_euclidPrime (canonical column) one
        exact after_payload_piece_satisfies artifact assignment satisfied _
          (after_bit_row_mem artifact member)
  · intro side lane
    exact (semantic_equals_hash_output artifact side assignment canonical one
      satisfied lane).trans
        (hash_rows_refine artifact constantsExact traceValid scheduleExact
          preimageLength side assignment canonical one satisfied lane)

private theorem base_preimage_length
    (side : StateSide) (assignment : Nat → Nat) :
    (phasePreimage baseArtifact.semanticGeometry side assignment).length =
      4 * absorbRounds := by
  cases side <;>
    norm_num [phasePreimage, SourceArtifact.semanticGeometry,
      RawArtifact.localColumns, RawArtifact.payloadColumns,
      RawArtifact.payloadStartColumn, baseArtifact, phaseConstantValues,
      absorbRounds, hashInputFields, hashConstantFields, domainFields,
      payloadFields, digestFields]

private theorem recursive_preimage_length
    (side : StateSide) (assignment : Nat → Nat) :
    (phasePreimage recursiveArtifact.semanticGeometry side assignment).length =
      4 * absorbRounds := by
  cases side <;>
    norm_num [phasePreimage, SourceArtifact.semanticGeometry,
      RawArtifact.localColumns, RawArtifact.payloadColumns,
      RawArtifact.payloadStartColumn, recursiveArtifact,
      phaseConstantValues, absorbRounds, hashInputFields,
      hashConstantFields, domainFields, payloadFields, digestFields]

/-- Exact base source rows imply the complete independent semantic link. -/
theorem base_rows_refine_semanticLink
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : baseArtifact.Satisfied assignment) :
    baseArtifact.SemanticLink assignment := by
  apply source_rows_refine baseArtifact rfl base_trace_ownedValid
    base_valueSchedules_exact base_preimage_length assignment canonical one
    satisfied
  intro column member
  rw [base_before_payload_zero assignment canonical one satisfied member]
  omega

/-- Recursive source rows imply the complete independent semantic link once
the separately owned private delayed-input family supplies before bitness. -/
theorem recursive_rows_refine_semanticLink
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : recursiveArtifact.Satisfied assignment)
    (beforeBinary : ∀ column,
      column ∈ recursiveArtifact.payloadColumns .before →
        assignment column ≤ 1) :
    recursiveArtifact.SemanticLink assignment :=
  source_rows_refine recursiveArtifact rfl recursive_trace_ownedValid
    recursive_valueSchedules_exact recursive_preimage_length assignment
    canonical one satisfied beforeBinary

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLinkArtifact
