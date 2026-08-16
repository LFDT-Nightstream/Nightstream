import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayPublic
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachineDuplex
import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.EqualityPins

/-!
Contract: same-assignment phase semantics for one bounded claim-frame chunk.

Owns the exact decoding of the 128 Rust state fields, the fixed 1,024-field
chunk, all state and cursor glue equations, the exact Poseidon2 replay and
coordinate-commitment transitions, final-chunk readiness, and the ten public
digest/cursor words.

Does not own the 86-step sequence, equality of supplied and authoritative
claim frames, claim-local algebra, the next PiCCS phase, selectors, or the
recursive lifecycle. Cursor equations are field equations here. The
verifier-owned schedule must supply the small exact cursor before they become
natural-number replay equations.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPhase

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex
open Nightstream.Implementation.R1CS.Program

def chunkWidth : Nat := 1024

def activeFields : ArmKind → Nat
  | .full => 1024
  | .final => 983

def successorAbsorbed : ArmKind → Nat
  | .full => 0
  | .final => 3

theorem activeFields_artifact_exact (kind : ArmKind) :
    (armFor kind).activeFields = activeFields kind := by
  cases kind <;> native_decide

/-- Exact 128-field state whose field order is used by the state digest. -/
structure Persistent where
  expected : ProductionFullClaimStreaming.State
  runtime : ProductionFullClaimStreaming.ReplayState
  programCursor : Nat
  coordinateCommitment : Fin outputWidth → Nat

private theorem duplexStateExt
    {left right : ProductionFullClaimStreaming.State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

def decodeSponge
    (assignment : Nat → Nat) (kind : ArmKind) (side : StateSide)
    (offset : Nat) : ProductionFullClaimStreaming.State where
  lanes := fun lane =>
    assignment (stateWordColumnFor kind side (offset + lane.val))
  absorbed := assignment (stateWordColumnFor kind side (offset + 8))

def decodePersistent
    (assignment : Nat → Nat) (kind : ArmKind) (side : StateSide) :
    Persistent where
  expected := decodeSponge assignment kind side 0
  runtime := {
    transcript := decodeSponge assignment kind side 9
    cursor := assignment (stateWordColumnFor kind side 18) }
  programCursor := assignment (stateWordColumnFor kind side 19)
  coordinateCommitment := fun output =>
    assignment (commitmentColumn kind side output)

def chunkColumns (kind : ArmKind) : List Nat :=
  (List.range chunkWidth).map fun index => chunkColumn (armFor kind) index

def activeChunkColumns (kind : ArmKind) : List Nat :=
  (chunkColumns kind).take (activeFields kind)

def chunkValues (assignment : Nat → Nat) (kind : ArmKind) : List Nat :=
  (chunkColumns kind).map assignment

@[simp] theorem chunkColumns_length (kind : ArmKind) :
    (chunkColumns kind).length = chunkWidth := by
  simp [chunkColumns]

@[simp] theorem chunkValues_length
    (assignment : Nat → Nat) (kind : ArmKind) :
    (chunkValues assignment kind).length = chunkWidth := by
  simp [chunkValues]

theorem activeChunkColumns_map
    (assignment : Nat → Nat) (kind : ArmKind) :
    (activeChunkColumns kind).map assignment =
      (chunkValues assignment kind).take (activeFields kind) := by
  simp [activeChunkColumns, chunkValues, List.map_take]

/-! ## Exact compact glue programs -/

def glueProgram (kind : ArmKind) : List Row :=
  (armFor kind).glueRows.map IndexedRow.row

private theorem glue_satisfies
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (glueProgram kind) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

private theorem included_normalized_satisfies
    (kind : ArmKind) (rows : List Row) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment)
    (included : rowsIncluded
      (Poseidon2Normalized.normalizeProgram rows) (glueProgram kind) = true) :
    Satisfies rows assignment := by
  apply (Poseidon2Normalized.satisfies_normalizeProgram rows assignment).mp
  intro row member
  exact glue_satisfies kind assignment satisfied row
    (rowsIncluded_sound included row member)

def commonStatePins (kind : ArmKind) : List (Nat × Nat) :=
  [(stateWordColumnFor kind .before 8, 3),
    (stateWordColumnFor kind .after 8, 3),
    (stateWordColumnFor kind .before 17, 0),
    (stateWordColumnFor kind .after 17, successorAbsorbed kind)]

private theorem commonStatePins_canonical (kind : ArmKind) :
    ConstantPins.ValuesCanonical (commonStatePins kind) := by
  cases kind <;> native_decide

private theorem commonStatePins_in_glue (kind : ArmKind) :
    rowsIncluded
      (Poseidon2Normalized.normalizeProgram
        (ConstantPins.rows (commonStatePins kind)))
      (glueProgram kind) = true := by
  cases kind <;> native_decide

theorem common_state_pin_facts
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    ∀ pin ∈ commonStatePins kind, assignment pin.1 = pin.2 := by
  have compactSatisfies := included_normalized_satisfies kind
    (ConstantPins.rows (commonStatePins kind)) assignment satisfied
    (commonStatePins_in_glue kind)
  exact ConstantPins.sound (commonStatePins_canonical kind)
    (by cases kind <;> native_decide) canonical one compactSatisfies

def expectedCarryPairs (kind : ArmKind) : List (Nat × Nat) :=
  (List.range 9).map fun index =>
    (stateWordColumnFor kind .before index,
      stateWordColumnFor kind .after index)

private theorem expectedCarryPairs_in_glue (kind : ArmKind) :
    rowsIncluded (EqualityPins.rows (expectedCarryPairs kind))
      (glueProgram kind) = true := by
  cases kind <;> native_decide

theorem expected_carry_facts
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    ∀ pair ∈ expectedCarryPairs kind,
      assignment pair.1 = assignment pair.2 := by
  have compactSatisfies :
      Satisfies (EqualityPins.rows (expectedCarryPairs kind)) assignment := by
    intro row member
    exact glue_satisfies kind assignment satisfied row
      (rowsIncluded_sound (expectedCarryPairs_in_glue kind) row member)
  exact EqualityPins.rows_sound canonical one compactSatisfies

def finalConstantPins : List (Nat × Nat) :=
  [(stateWordColumnFor .final .before 18, 87040),
    (stateWordColumnFor .final .before 19, 168)] ++
    ((List.range 41).map fun offset =>
      (chunkColumn (armFor .final) (983 + offset), 0)) ++
    [(stateWordColumnFor .final .after 18, 88023)]

private theorem finalConstantPins_canonical :
    ConstantPins.ValuesCanonical finalConstantPins := by
  native_decide

private theorem finalConstantPins_in_glue :
    rowsIncluded
      (Poseidon2Normalized.normalizeProgram
        (ConstantPins.rows finalConstantPins))
      (glueProgram .final) = true := by
  native_decide

theorem final_constant_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor .final).Satisfied assignment) :
    ∀ pin ∈ finalConstantPins, assignment pin.1 = pin.2 := by
  have compactSatisfies := included_normalized_satisfies .final
    (ConstantPins.rows finalConstantPins) assignment satisfied
    finalConstantPins_in_glue
  exact ConstantPins.sound finalConstantPins_canonical
    (by native_decide) canonical one compactSatisfies

def finalReadyPairs : List (Nat × Nat) :=
  (List.range 9).map fun index =>
    (stateWordColumnFor .final .after (9 + index),
      stateWordColumnFor .final .after index)

private theorem finalReadyPairs_in_glue :
    rowsIncluded
      (Poseidon2Normalized.normalizeProgram
        (EqualityPins.rows finalReadyPairs))
      (glueProgram .final) = true := by
  native_decide

theorem final_ready_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor .final).Satisfied assignment) :
    ∀ pair ∈ finalReadyPairs,
      assignment pair.1 = assignment pair.2 := by
  have compactSatisfies := included_normalized_satisfies .final
    (EqualityPins.rows finalReadyPairs) assignment satisfied
    finalReadyPairs_in_glue
  exact EqualityPins.rows_sound canonical one compactSatisfies

/-! ## Field cursor equations -/

def alignmentTerms (kind : ArmKind) : List (Nat × Nat) :=
  [(stateWordColumnFor kind .before 19, 1024),
    (0, goldilocksP - 84992)]

def frameAdvanceTerms (kind : ArmKind) : List (Nat × Nat) :=
  [(stateWordColumnFor kind .before 18, 1),
    (0, activeFields kind)]

def programAdvanceTerms (kind : ArmKind) : List (Nat × Nat) :=
  [(stateWordColumnFor kind .before 19, 1), (0, 1)]

def alignmentRow (kind : ArmKind) : Row :=
  ⟨[(0, 84992),
      (stateWordColumnFor kind .before 18, 1),
      (stateWordColumnFor kind .before 19, goldilocksP - 1024)],
    [(0, 1)], []⟩

def frameAdvanceRow (kind : ArmKind) : Row :=
  ⟨[(0, goldilocksP - activeFields kind),
      (stateWordColumnFor kind .before 18, goldilocksP - 1),
      (stateWordColumnFor kind .after 18, 1)],
    [(0, 1)], []⟩

def programAdvanceRow (kind : ArmKind) : Row :=
  ⟨[(0, goldilocksP - 1),
      (stateWordColumnFor kind .before 19, goldilocksP - 1),
      (stateWordColumnFor kind .after 19, 1)],
    [(0, 1)], []⟩

def cursorRows (kind : ArmKind) : List Row :=
  [alignmentRow kind, frameAdvanceRow kind, programAdvanceRow kind]

private theorem cursorRows_in_glue (kind : ArmKind) :
    rowsIncluded (cursorRows kind) (glueProgram kind) = true := by
  cases kind <;> native_decide

private theorem alignmentTerms_canonical (kind : ArmKind) :
    CanonicalTerms (alignmentTerms kind) := by
  cases kind <;> native_decide

private theorem frameAdvanceTerms_canonical (kind : ArmKind) :
    CanonicalTerms (frameAdvanceTerms kind) := by
  cases kind <;> native_decide

private theorem programAdvanceTerms_canonical (kind : ArmKind) :
    CanonicalTerms (programAdvanceTerms kind) := by
  cases kind <;> native_decide

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

private theorem alignmentRow_perms (kind : ArmKind) :
    (alignmentRow kind).a.Perm
        (builderLinearRow (stateWordColumnFor kind .before 18)
          (alignmentTerms kind)).a ∧
      (alignmentRow kind).b.Perm
        (builderLinearRow (stateWordColumnFor kind .before 18)
          (alignmentTerms kind)).b ∧
      (alignmentRow kind).c.Perm
        (builderLinearRow (stateWordColumnFor kind .before 18)
          (alignmentTerms kind)).c := by
  cases kind <;> native_decide

private theorem frameAdvanceRow_perms (kind : ArmKind) :
    (frameAdvanceRow kind).a.Perm
        (builderLinearRow (stateWordColumnFor kind .after 18)
          (frameAdvanceTerms kind)).a ∧
      (frameAdvanceRow kind).b.Perm
        (builderLinearRow (stateWordColumnFor kind .after 18)
          (frameAdvanceTerms kind)).b ∧
      (frameAdvanceRow kind).c.Perm
        (builderLinearRow (stateWordColumnFor kind .after 18)
          (frameAdvanceTerms kind)).c := by
  cases kind <;> native_decide

private theorem programAdvanceRow_perms (kind : ArmKind) :
    (programAdvanceRow kind).a.Perm
        (builderLinearRow (stateWordColumnFor kind .after 19)
          (programAdvanceTerms kind)).a ∧
      (programAdvanceRow kind).b.Perm
        (builderLinearRow (stateWordColumnFor kind .after 19)
          (programAdvanceTerms kind)).b ∧
      (programAdvanceRow kind).c.Perm
        (builderLinearRow (stateWordColumnFor kind .after 19)
          (programAdvanceTerms kind)).c := by
  cases kind <;> native_decide

theorem cursor_field_equations
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment (stateWordColumnFor kind .before 18) =
        (1024 * assignment (stateWordColumnFor kind .before 19) +
          (goldilocksP - 84992)) % goldilocksP ∧
      assignment (stateWordColumnFor kind .after 18) =
        (assignment (stateWordColumnFor kind .before 18) +
          activeFields kind) % goldilocksP ∧
      assignment (stateWordColumnFor kind .after 19) =
        (assignment (stateWordColumnFor kind .before 19) + 1) %
          goldilocksP := by
  have compactSatisfies : Satisfies (cursorRows kind) assignment := by
    intro row member
    exact glue_satisfies kind assignment satisfied row
      (rowsIncluded_sound (cursorRows_in_glue kind) row member)
  have alignmentHolds := compactSatisfies (alignmentRow kind)
    (by simp [cursorRows])
  have alignmentBuilderHolds := rowHolds_of_operand_perms assignment
    (alignmentRow_perms kind).1 (alignmentRow_perms kind).2.1
    (alignmentRow_perms kind).2.2 alignmentHolds
  have alignment := builderLinearRow_sound canonical one
    (stateWordColumnFor kind .before 18) (alignmentTerms kind)
    (alignmentTerms_canonical kind)
    alignmentBuilderHolds
  have frameAdvanceHolds := compactSatisfies (frameAdvanceRow kind)
    (by simp [cursorRows])
  have frameAdvanceBuilderHolds := rowHolds_of_operand_perms assignment
    (frameAdvanceRow_perms kind).1 (frameAdvanceRow_perms kind).2.1
    (frameAdvanceRow_perms kind).2.2 frameAdvanceHolds
  have frameAdvance := builderLinearRow_sound canonical one
    (stateWordColumnFor kind .after 18) (frameAdvanceTerms kind)
    (frameAdvanceTerms_canonical kind)
    frameAdvanceBuilderHolds
  have programAdvanceHolds := compactSatisfies (programAdvanceRow kind)
    (by simp [cursorRows])
  have programAdvanceBuilderHolds := rowHolds_of_operand_perms assignment
    (programAdvanceRow_perms kind).1 (programAdvanceRow_perms kind).2.1
    (programAdvanceRow_perms kind).2.2 programAdvanceHolds
  have programAdvance := builderLinearRow_sound canonical one
    (stateWordColumnFor kind .after 19) (programAdvanceTerms kind)
    (programAdvanceTerms_canonical kind)
    programAdvanceBuilderHolds
  simpa [alignmentTerms, frameAdvanceTerms, programAdvanceTerms,
    lcEval, one, Nat.add_comm, Nat.mul_comm] using
    And.intro alignment (And.intro frameAdvance programAdvance)

/-! ## Poseidon2 transition on the exact chunk values -/

private theorem decoded_before_runtime_eq_start
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    decodeSponge assignment kind .before 9 =
      toDuplex
        (ColumnReplay.decodeRun assignment canonical
          (startRun (armFor kind))).state := by
  have pins := common_state_pin_facts kind assignment canonical one satisfied
  apply duplexStateExt
  · funext lane
    cases kind <;>
      simp [decodeSponge, toDuplex, ColumnReplay.decodeRun,
        ColumnReplay.decodeCursor, startRun, stateWordColumnFor,
        stateWordColumn, stateWordOffset, armFor]
  · have absorbed := pins
      (stateWordColumnFor kind .before 17, 0) (by
        simp [commonStatePins])
    simpa [decodeSponge] using absorbed

private theorem decoded_after_runtime_eq_declared
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    decodeSponge assignment kind .after 9 =
      toDuplex
        (declaredRuntimeState (armFor kind) assignment canonical
          ⟨successorAbsorbed kind, by cases kind <;> decide⟩) := by
  have pins := common_state_pin_facts kind assignment canonical one satisfied
  apply duplexStateExt
  · funext lane
    have indexEq : 128 + (9 + lane.val) = 137 + lane.val := by omega
    cases kind <;>
      simp [decodeSponge, toDuplex, declaredRuntimeState,
        afterRuntimeColumn, stateWordColumnFor, stateWordColumn,
        stateWordOffset, armFor, indexEq]
  · have absorbed := pins
      (stateWordColumnFor kind .after 17, successorAbsorbed kind) (by
        cases kind <;> simp [commonStatePins, successorAbsorbed])
    simpa [decodeSponge, toDuplex] using absorbed

theorem runtime_transcript_transition
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (decodePersistent assignment kind .after).runtime.transcript =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        ((chunkValues assignment kind).take (activeFields kind))
        (decodePersistent assignment kind .before).runtime.transcript := by
  have rowsRefine :
      (ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical (startRun (armFor kind)))
        (chunkOperations (armFor kind) (activeFields kind))).state =
      declaredRuntimeState (armFor kind) assignment canonical
        ⟨successorAbsorbed kind, by cases kind <;> decide⟩ := by
    cases kind
    · simpa [activeFields, successorAbsorbed] using
        full_rows_refine_declared_runtime assignment canonical one satisfied
    · simpa [activeFields, successorAbsorbed] using
        final_rows_refine_declared_runtime assignment canonical one satisfied
  have converted := congrArg toDuplex rowsRefine
  have operationsExact :
      chunkOperations (armFor kind) (activeFields kind) =
        (activeChunkColumns kind).map ColumnReplay.Operation.external := by
    cases kind <;> native_decide
  have bulk := semanticExecuteSlice_external_toDuplex assignment canonical
    (ColumnReplay.decodeRun assignment canonical (startRun (armFor kind)))
    (activeChunkColumns kind)
  rw [← operationsExact] at bulk
  calc
    (decodePersistent assignment kind .after).runtime.transcript =
        toDuplex
          (declaredRuntimeState (armFor kind) assignment canonical
            ⟨successorAbsorbed kind, by cases kind <;> decide⟩) :=
      decoded_after_runtime_eq_declared kind assignment canonical one satisfied
    _ = toDuplex
          (ColumnReplay.semanticExecuteSlice assignment canonical
            (ColumnReplay.decodeRun assignment canonical
              (startRun (armFor kind)))
            (chunkOperations (armFor kind) (activeFields kind))).state :=
      converted.symm
    _ = Poseidon2Duplex.absorbSlice
          Poseidon2CanonicalConstants.selected
          ((activeChunkColumns kind).map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              (startRun (armFor kind))).state) := bulk
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((chunkValues assignment kind).take (activeFields kind))
          (decodePersistent assignment kind .before).runtime.transcript := by
      rw [activeChunkColumns_map]
      rw [← decoded_before_runtime_eq_start kind assignment canonical one
        satisfied]
      rfl

/-! ## Independent phase relation -/

def CoordinateTransition
    (kind : ArmKind) (before after : Persistent) (chunk : List Nat) : Prop :=
  match kind with
  | .full =>
      before.coordinateCommitment = (fun _ => 0) ∧
        ∀ output : Fin outputWidth,
          after.coordinateCommitment output =
            (before.coordinateCommitment output +
              (partialCommitment firstChunk chunk output).val) % goldilocksP
  | .final =>
      after.coordinateCommitment = before.coordinateCommitment

def FinalChecks
    (kind : ArmKind) (before after : Persistent) (chunk : List Nat) : Prop :=
  match kind with
  | .full => True
  | .final =>
      before.runtime.cursor = 87040 ∧
        before.programCursor = 168 ∧
        chunk.drop 983 = List.replicate 41 0 ∧
        after.runtime.transcript = after.expected ∧
        after.runtime.cursor = 88023

structure PhaseRelation
    (kind : ArmKind) (before after : Persistent) (chunk : List Nat) : Prop where
  chunkLength : chunk.length = chunkWidth
  expectedCarry : after.expected = before.expected
  coordinateTransition : CoordinateTransition kind before after chunk
  expectedAbsorbed : before.expected.absorbed = 3
  runtimeStartAbsorbed : before.runtime.transcript.absorbed = 0
  runtimeAfterAbsorbed :
    after.runtime.transcript.absorbed = successorAbsorbed kind
  transcriptTransition :
    after.runtime.transcript =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (chunk.take (activeFields kind)) before.runtime.transcript
  frameAlignment :
    before.runtime.cursor =
      (1024 * before.programCursor + (goldilocksP - 84992)) % goldilocksP
  frameAdvance :
    after.runtime.cursor =
      (before.runtime.cursor + activeFields kind) % goldilocksP
  programAdvance :
    after.programCursor = (before.programCursor + 1) % goldilocksP
  finalChecks : FinalChecks kind before after chunk

def PublicBinding
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  (∀ lane : Fin 4,
      (stateDigest assignment canonical kind .after lane).val =
        publicWordValue assignment kind
          (digestPublicWordIndex .after lane)) ∧
    (∀ lane : Fin 4,
      (stateDigest assignment canonical kind .before lane).val =
        publicWordValue assignment kind
          (digestPublicWordIndex .before lane)) ∧
    publicWordValue assignment kind (cursorPublicWordIndex .before) =
      (decodePersistent assignment kind .before).programCursor ∧
    publicWordValue assignment kind (cursorPublicWordIndex .after) =
      (decodePersistent assignment kind .after).programCursor

structure RowsRelation
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  phase : PhaseRelation kind
    (decodePersistent assignment kind .before)
    (decodePersistent assignment kind .after)
    (chunkValues assignment kind)
  publicBinding : PublicBinding kind assignment canonical

private theorem expected_carry_state
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (decodePersistent assignment kind .after).expected =
      (decodePersistent assignment kind .before).expected := by
  have facts := expected_carry_facts kind assignment canonical one satisfied
  apply duplexStateExt
  · funext lane
    have laneBound : lane.val < 9 := by
      have laneLt := lane.isLt
      change lane.val < 8 at laneLt
      omega
    simpa [decodePersistent, decodeSponge] using (facts
      (stateWordColumnFor kind .before lane.val,
        stateWordColumnFor kind .after lane.val)
      (List.mem_map.mpr
        ⟨lane.val, List.mem_range.mpr laneBound, rfl⟩)).symm
  · exact (facts
      (stateWordColumnFor kind .before 8,
        stateWordColumnFor kind .after 8)
      (List.mem_map.mpr ⟨8, by decide, rfl⟩)).symm

private theorem coordinate_transition_from_rows
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    CoordinateTransition kind
      (decodePersistent assignment kind .before)
      (decodePersistent assignment kind .after)
      (chunkValues assignment kind) := by
  cases kind with
  | full =>
      constructor
      · funext output
        simpa [decodePersistent] using
          full_before_zero assignment canonical one satisfied output
      · intro output
        have chunksExact :
            fullChunkValues assignment = chunkValues assignment .full := by
          simp [fullChunkValues, chunkValues, chunkColumns,
            Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.claimChunkWidth,
            chunkWidth, armFor, Function.comp_def]
        have transition :=
          full_commitment_transition assignment canonical one satisfied output
        rw [chunksExact] at transition
        simpa [decodePersistent] using transition
  | final =>
      funext output
      simpa [decodePersistent] using
        final_commitment_carry assignment canonical one satisfied output

private theorem final_checks_from_rows
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor .final).Satisfied assignment) :
    FinalChecks .final
      (decodePersistent assignment .final .before)
      (decodePersistent assignment .final .after)
      (chunkValues assignment .final) := by
  have pins := final_constant_facts assignment canonical one satisfied
  have ready := final_ready_facts assignment canonical one satisfied
  constructor
  · exact pins (stateWordColumnFor .final .before 18, 87040)
      (by simp [finalConstantPins])
  constructor
  · exact pins (stateWordColumnFor .final .before 19, 168)
      (by simp [finalConstantPins])
  constructor
  · apply List.ext_getElem
    · simp [chunkValues, chunkColumns, chunkWidth]
    · intro index leftBound rightBound
      have indexLt : index < 41 := by simpa using leftBound
      have pinMember :
          (chunkColumn (armFor .final) (983 + index), 0) ∈
            finalConstantPins := by
        unfold finalConstantPins
        apply List.mem_append.mpr
        left
        apply List.mem_append.mpr
        right
        exact List.mem_map.mpr
          ⟨index, List.mem_range.mpr indexLt, rfl⟩
      have pinFact := pins
        (chunkColumn (armFor .final) (983 + index), 0) pinMember
      rw [List.getElem_replicate rightBound]
      simpa [chunkValues, chunkColumns, chunkWidth, indexLt] using pinFact
  constructor
  · apply duplexStateExt
    · funext lane
      have laneBound : lane.val < 9 := by
        have laneLt := lane.isLt
        change lane.val < 8 at laneLt
        omega
      simpa [decodePersistent, decodeSponge] using ready
        (stateWordColumnFor .final .after (9 + lane.val),
          stateWordColumnFor .final .after lane.val)
        (List.mem_map.mpr ⟨lane.val, List.mem_range.mpr laneBound, rfl⟩)
    · exact ready
        (stateWordColumnFor .final .after 17,
          stateWordColumnFor .final .after 8)
        (List.mem_map.mpr ⟨8, by decide, rfl⟩)
  · exact pins (stateWordColumnFor .final .after 18, 88023)
      (by simp [finalConstantPins])

/-- Every exact satisfying Rust arm implies the complete independent phase
relation and the shared public binding on the same assignment. -/
theorem rows_imply_relation
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    RowsRelation kind assignment canonical := by
  have common := common_state_pin_facts kind assignment canonical one satisfied
  have cursors := cursor_field_equations kind assignment canonical one satisfied
  constructor
  · constructor
    · exact chunkValues_length assignment kind
    · exact expected_carry_state kind assignment canonical one satisfied
    · exact coordinate_transition_from_rows kind assignment canonical one
        satisfied
    · exact common (stateWordColumnFor kind .before 8, 3)
        (by simp [commonStatePins])
    · exact common (stateWordColumnFor kind .before 17, 0)
        (by simp [commonStatePins])
    · exact common
        (stateWordColumnFor kind .after 17, successorAbsorbed kind)
        (by cases kind <;> simp [commonStatePins, successorAbsorbed])
    · exact runtime_transcript_transition kind assignment canonical one
        satisfied
    · exact cursors.1
    · exact cursors.2.1
    · exact cursors.2.2
    · cases kind
      · trivial
      · exact final_checks_from_rows assignment canonical one satisfied
  · simpa [PublicBinding, decodePersistent] using
      shared_public_words_refine kind assignment canonical one satisfied

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPhase
