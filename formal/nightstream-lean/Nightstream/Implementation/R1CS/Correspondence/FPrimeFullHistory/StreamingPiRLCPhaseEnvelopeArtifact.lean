import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.PureSponge

/-!
Contract: exact Rust-to-Lean refinement for the PiRLC carry-phase semantic
envelope.

Owns both exact 662,971-row source ranges, the shared delayed-payload slice,
local-digest aliases, fixed domain and length pins, both complete Poseidon2
sponges, and the direct links from their outputs to XOut semantic fields
19 through 22.

Does not own the local family-state digest theorem, common-to-phase lifecycle
links, selective-CCS lowering, or collision resistance.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 10000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelope

inductive ArmKind where
  | even
  | odd
deriving DecidableEq, Repr

def armFor : ArmKind → RawArm
  | .even => evenArm
  | .odd => oddArm

theorem artifact_valid : rawArtifact.Valid :=
  rawArtifact_valid

theorem exact_source_identity :
    evenArm.sourceRowsSha256 =
        "2ef4f3217310c361be90d53c37e852f9ea362786aeb4e7cd212cf56ea8e4cfce" ∧
      oddArm.sourceRowsSha256 =
        "45612a50dd5521e239f48594315aa86ed28ff53df81a3b73e1bd825d5b3c1f50" := by
  exact ⟨rfl, rfl⟩

theorem exact_source_ranges :
    evenArm.phaseRowStart = 558380 ∧ evenArm.phaseRowEnd = 1221351 ∧
      evenArm.phaseColumnStart = 558608 ∧
      evenArm.phaseColumnEnd = 1221579 ∧
      oddArm.phaseRowStart = 559580 ∧ oddArm.phaseRowEnd = 1222551 ∧
      oddArm.phaseColumnStart = 559808 ∧
      oddArm.phaseColumnEnd = 1222779 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- The two boundary names refer to one exact payload slice. -/
theorem shared_payload_columns (kind : ArmKind) :
    (armFor kind).payloadColumns =
      List.range' (armFor kind).payloadStartColumn 2169 := by
  rfl

/-- Each recomputed phase digest is the exact four-column semantic field of
the later 32-field XOut preimage. -/
theorem semantic_outputs_feed_x_out (kind : ArmKind) (side : StateSide) :
    (armFor kind).semanticDigestColumns side =
      (armFor kind).xOutSemanticColumns side := by
  cases kind <;> cases side <;> rfl

theorem hash_trace_valid (kind : ArmKind) (side : StateSide) :
    ((armFor kind).hashRecipe phaseConstantValues side).trace.OwnedValid := by
  cases kind <;> cases side
  · exact evenBeforeHash_trace_ownedValid
  · exact evenAfterHash_trace_ownedValid
  · exact oddBeforeHash_trace_ownedValid
  · exact oddAfterHash_trace_ownedValid

private theorem phase_piece_satisfies
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied phaseConstantValues assignment)
    (piece : List Row) (member : piece ∈ arm.phasePieces phaseConstantValues) :
    Satisfies piece assignment := by
  exact (satisfies_flatten_iff
    (arm.phasePieces phaseConstantValues) assignment).mp satisfied piece member

private theorem alias_piece_satisfies
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment) :
    Satisfies
      (aliasRows ((armFor kind).localSourceColumns side)
        ((armFor kind).localAliasColumns side)) assignment := by
  apply phase_piece_satisfies (armFor kind) assignment satisfied
  cases side <;> simp [RawArm.phasePieces, RawArm.localSourceColumns,
    RawArm.localAliasColumns]

private theorem payload_piece_satisfies
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment) :
    Satisfies (payloadRows (armFor kind)) assignment := by
  apply phase_piece_satisfies (armFor kind) assignment satisfied
  simp [RawArm.phasePieces]

private theorem constant_piece_satisfies
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment) :
    Satisfies
      (constantRows ((armFor kind).hashRecipe phaseConstantValues side))
      assignment := by
  apply phase_piece_satisfies (armFor kind) assignment satisfied
  cases side <;> simp [RawArm.phasePieces]

private theorem trace_piece_satisfies
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment) :
    Satisfies
      ((armFor kind).hashRecipe phaseConstantValues side).trace.rows
      assignment := by
  apply phase_piece_satisfies (armFor kind) assignment satisfied
  cases side <;> simp [RawArm.phasePieces]

private theorem constant_rows_values
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    ∀ (columns values : List Nat),
      columns.length = values.length →
      (∀ value ∈ values, 0 < value ∧ value < goldilocksP) →
      Satisfies
        ((columns.zip values).map fun entry =>
          builderLinearRow entry.1 [(0, entry.2)]) assignment →
      columns.map assignment = values := by
  intro columns
  induction columns with
  | nil =>
      intro values lengths _canonicalValues _holds
      cases values with
      | nil => rfl
      | cons _ _ => simp at lengths
  | cons column rest inductionHypothesis =>
      intro values lengths canonicalValues holds
      cases values with
      | nil => simp at lengths
      | cons value tail =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          have valueCanonical := canonicalValues value (by simp)
          have rowHolds :
              RowHolds assignment (builderLinearRow column [(0, value)]) :=
            holds _ (by simp)
          have defined := builderLinearRow_sound canonical one column [(0, value)]
            (by simp [CanonicalTerms, valueCanonical.1, valueCanonical.2]) rowHolds
          have head : assignment column = value := by
            simpa [lcEval, one, Nat.mod_eq_of_lt valueCanonical.2] using defined
          have tailCanonical :
              ∀ candidate ∈ tail, 0 < candidate ∧ candidate < goldilocksP := by
            intro candidate member
            exact canonicalValues candidate (by simp [member])
          have tailHolds :
              Satisfies
                ((rest.zip tail).map fun entry =>
                  builderLinearRow entry.1 [(0, entry.2)]) assignment := by
            intro row member
            exact holds row (by simp [member])
          simp only [List.map_cons, List.cons.injEq]
          exact ⟨head, inductionHypothesis tail lengths tailCanonical tailHolds⟩

private theorem alias_rows_values
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) :
    ∀ (sources aliases : List Nat),
      sources.length = aliases.length →
      Satisfies (aliasRows sources aliases) assignment →
      aliases.map assignment = sources.map assignment := by
  intro sources
  induction sources with
  | nil =>
      intro aliases lengths _holds
      cases aliases with
      | nil => rfl
      | cons _ _ => simp at lengths
  | cons source rest inductionHypothesis =>
      intro aliases lengths holds
      cases aliases with
      | nil => simp at lengths
      | cons aliasColumn tail =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          have rowHolds :
              RowHolds assignment (builderLinearRow aliasColumn [(source, 1)]) :=
            holds _ (by simp [aliasRows])
          have defined := builderLinearRow_sound canonical one aliasColumn [(source, 1)]
            (by simp [CanonicalTerms, goldilocksP]) rowHolds
          have head : assignment aliasColumn = assignment source := by
            simpa [lcEval, Nat.mod_eq_of_lt (canonical source)] using defined
          have tailHolds : Satisfies (aliasRows rest tail) assignment := by
            intro row member
            apply holds row
            change row ∈
              builderLinearRow aliasColumn [(source, 1)] :: aliasRows rest tail
            exact List.mem_cons_of_mem _ member
          simp only [List.map_cons, List.cons.injEq]
          exact ⟨head, inductionHypothesis tail lengths tailHolds⟩

private theorem hash_constant_shape_of_valid
    {recipe : HashRecipe} {columnCount : Nat}
    (valid : recipe.Valid columnCount) :
    recipe.constantColumns.length = recipe.constantValues.length ∧
      ∀ value ∈ recipe.constantValues,
        0 < value ∧ value < goldilocksP := by
  exact ⟨valid.2.2.1.1.trans valid.1.symm, valid.2.1⟩

private theorem hash_constant_shape (kind : ArmKind) (side : StateSide) :
    let recipe := (armFor kind).hashRecipe phaseConstantValues side;
    recipe.constantColumns.length = recipe.constantValues.length ∧
      ∀ value ∈ recipe.constantValues,
        0 < value ∧ value < goldilocksP := by
  cases kind <;> cases side
  · exact hash_constant_shape_of_valid evenBeforeHash_valid
  · exact hash_constant_shape_of_valid evenAfterHash_valid
  · exact hash_constant_shape_of_valid oddBeforeHash_valid
  · exact hash_constant_shape_of_valid oddAfterHash_valid

private theorem local_alias_shape (kind : ArmKind) (side : StateSide) :
    ((armFor kind).localSourceColumns side).length =
      ((armFor kind).localAliasColumns side).length := by
  cases kind <;> cases side <;> rfl

/-- The eleven domain and length columns contain their fixed Rust values. -/
theorem hash_constant_values
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment) :
    (((armFor kind).hashRecipe phaseConstantValues side).constantColumns.map
      assignment) = phaseConstantValues := by
  have shape := hash_constant_shape kind side
  have exact := constant_rows_values assignment canonical one
    ((armFor kind).hashRecipe phaseConstantValues side).constantColumns
    ((armFor kind).hashRecipe phaseConstantValues side).constantValues
    shape.1 shape.2
    (constant_piece_satisfies kind side assignment satisfied)
  simpa [RawArm.hashRecipe] using exact

/-- The four hash-local columns are aliases of the exact phase-local digest
source columns. -/
theorem hash_local_alias_values
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment) :
    ((armFor kind).localAliasColumns side).map assignment =
      ((armFor kind).localSourceColumns side).map assignment := by
  exact alias_rows_values assignment canonical one
    ((armFor kind).localSourceColumns side)
    ((armFor kind).localAliasColumns side)
    (local_alias_shape kind side)
    (alias_piece_satisfies kind side assignment satisfied)

/-- Ordered authoritative preimage of one phase-envelope hash. -/
def phasePreimage
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat) : List Nat :=
  phaseConstantValues ++
    ((armFor kind).localSourceColumns side).map assignment ++
    (armFor kind).payloadColumns.map assignment

/-- The sponge receives the fixed protocol domain, then the exact local
digest source, then the one shared delayed-payload source slice. -/
theorem hash_trace_input_values
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment) :
    ((armFor kind).hashRecipe phaseConstantValues side).trace.inputColumns.map
        assignment =
      phasePreimage kind side assignment := by
  change
    ((((armFor kind).hashRecipe phaseConstantValues side).constantColumns ++
      ((armFor kind).hashRecipe phaseConstantValues side).localColumns ++
      ((armFor kind).hashRecipe phaseConstantValues side).payloadColumns).map
        assignment) = phasePreimage kind side assignment
  simp only [List.map_append]
  rw [hash_constant_values kind side assignment canonical one satisfied]
  change phaseConstantValues ++
      ((armFor kind).localAliasColumns side).map assignment ++
      (armFor kind).payloadColumns.map assignment =
    phasePreimage kind side assignment
  rw [hash_local_alias_values kind side assignment canonical one satisfied]
  rfl

private theorem phase_preimage_source_lengths
    (kind : ArmKind) (side : StateSide) :
    ((armFor kind).localSourceColumns side).length = digestFields ∧
      (armFor kind).payloadColumns.length = payloadFields := by
  cases kind <;> cases side <;>
    simp [armFor, RawArm.localSourceColumns, RawArm.payloadColumns,
      evenArm, oddArm, digestFields]

theorem phase_preimage_length
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat) :
    (phasePreimage kind side assignment).length = 4 * absorbRounds := by
  have shape := phase_preimage_source_lengths kind side
  simp [phasePreimage, shape.1, shape.2, phaseConstantValues, digestFields,
    payloadFields, absorbRounds, hashInputFields, hashConstantFields,
    domainFields]

private theorem phase_constants_canonical :
    ∀ value ∈ phaseConstantValues, value < goldilocksP := by
  norm_num [phaseConstantValues, goldilocksP]

theorem phase_preimage_canonical
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ∀ value ∈ phasePreimage kind side assignment,
      value < goldilocksP := by
  intro value member
  rw [phasePreimage, List.mem_append] at member
  rcases member with constantsAndSource | payloadMember
  · rw [List.mem_append] at constantsAndSource
    rcases constantsAndSource with constantMember | sourceMember
    · exact phase_constants_canonical value constantMember
    · rcases List.mem_map.mp sourceMember with ⟨column, _member, rfl⟩
      exact canonical column
  · rcases List.mem_map.mp payloadMember with ⟨column, _member, rfl⟩
    exact canonical column

/-- The generated trace has exactly 546 full-rate absorbs and one padding
call. -/
private theorem fullRateSchedule_exact (count : Nat) :
    Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateSchedule count =
      List.replicate count (.absorb 4) ++ [.pad] := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateSchedule,
        inductionHypothesis, List.replicate_succ]

theorem hash_schedule_exact (kind : ArmKind) (side : StateSide) :
    Nightstream.Implementation.R1CS.Poseidon2Sponge.valueSchedules
        ((armFor kind).hashRecipe phaseConstantValues side).trace.rounds =
      Nightstream.Implementation.R1CS.Poseidon2Sponge.valueSchedules
        (Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds
          absorbRounds) := by
  rw [Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds_schedule,
    fullRateSchedule_exact]
  cases kind <;> cases side
  · exact evenBeforeHash_valueSchedules_exact
  · exact evenAfterHash_valueSchedules_exact
  · exact oddBeforeHash_valueSchedules_exact
  · exact oddAfterHash_valueSchedules_exact

/-- Canonical selected-Poseidon2 chunks for the exact phase preimage. -/
def phaseChunks
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat) :=
  Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateChunks
    (phasePreimage kind side assignment) absorbRounds

/-- Authoritative outer semantic digest recomputed from the exact local
phase digest and the shared delayed payload. -/
def phaseEnvelopeDigest
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (lane : Fin 4) : Nat :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
    Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
    (phaseChunks kind side assignment) lane

/-- Every one of the 2,169 exact delayed-payload source columns is Boolean. -/
theorem payload_column_le_one
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment)
    (column : Nat) (member : column ∈ (armFor kind).payloadColumns) :
    assignment column ≤ 1 := by
  apply bitRow_le_one goldilocks_euclidPrime (canonical column) one
  exact payload_piece_satisfies kind assignment satisfied
    (bitRow column) (List.mem_map.mpr ⟨column, member, rfl⟩)

/-- Exact phase rows compute the production sponge on the ordered input-wire
values. No supplied digest appears in this statement. -/
theorem hash_rows_compute
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment)
    (lane : Fin 4) :
    assignment
        (((armFor kind).semanticDigestColumns side).getD lane.val 0) =
      Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds
        ((armFor kind).hashRecipe phaseConstantValues side).trace.rounds
        (((armFor kind).hashRecipe phaseConstantValues side).trace.inputColumns.map
          assignment)
        (fun _ => 0) lane.val := by
  exact Nightstream.Implementation.R1CS.Poseidon2Sponge.ownedTrace_values_sound
    (hash_trace_valid kind side) canonical one
    (trace_piece_satisfies kind side assignment satisfied)
    lane.val lane.isLt

/-- The exact Rust rows recompute the selected canonical Poseidon2 phase
envelope. The digest is a consequence of the local source digest and delayed
payload columns; it is not an authority input. -/
theorem hash_rows_refine_phase_envelope
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment)
    (lane : Fin 4) :
    assignment
        (((armFor kind).semanticDigestColumns side).getD lane.val 0) =
      phaseEnvelopeDigest kind side assignment lane := by
  calc
    assignment
        (((armFor kind).semanticDigestColumns side).getD lane.val 0) =
        Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds
          ((armFor kind).hashRecipe phaseConstantValues side).trace.rounds
          (((armFor kind).hashRecipe phaseConstantValues side).trace.inputColumns.map
            assignment)
          (fun _ => 0) lane.val :=
      hash_rows_compute kind side assignment canonical one satisfied lane
    _ = Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds
          ((armFor kind).hashRecipe phaseConstantValues side).trace.rounds
          (phasePreimage kind side assignment) (fun _ => 0) lane.val := by
      rw [hash_trace_input_values kind side assignment canonical one satisfied]
    _ = Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds
          (Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds
            absorbRounds)
          (phasePreimage kind side assignment) (fun _ => 0) lane.val := by
      exact congrFun
        (Nightstream.Implementation.R1CS.Poseidon2Sponge.runValueRounds_eq_of_schedules
          (hash_schedule_exact kind side) (phasePreimage kind side assignment)
          (fun _ => 0)) lane.val
    _ = phaseEnvelopeDigest kind side assignment lane := by
      exact
        Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateRounds_compute_digest
          absorbRounds (phasePreimage kind side assignment)
          (phase_preimage_length kind side assignment)
          (phase_preimage_canonical kind side assignment canonical) lane

/-- The four recomputed digest lanes are the exact semantic fields later
consumed by the 32-field XOut hash. -/
theorem x_out_semantic_refines_phase_envelope
    (kind : ArmKind) (side : StateSide) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied phaseConstantValues assignment)
    (lane : Fin 4) :
    assignment ((armFor kind).xOutSemanticColumns side |>.getD lane.val 0) =
      phaseEnvelopeDigest kind side assignment lane := by
  rw [← semantic_outputs_feed_x_out kind side]
  exact hash_rows_refine_phase_envelope kind side assignment canonical one
    satisfied lane

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelopeArtifact
