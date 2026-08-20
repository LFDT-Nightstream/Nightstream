import Nightstream.Implementation.Nebula.Production.Carrier.StreamingClaimReplayTransition
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayStateCursorRowCertificate
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayExpectedCarryArtifact

/-!
Contract: generated-row refinement for the state constants and verifier-owned
cursors of one production claim-replay phase.

Assurance tier: Rust-conformant for the constant and cursor phase fields.

Owns four exact state-pin rows, three exact cursor rows, and their refinement
to the v6 decoded transition. Exact phase cursors also require the selected
chunk and its arm scope from the verifier-owned schedule.

Does not own expected carry, Poseidon2 execution, coordinate accumulation,
complete arm validity, lifecycle selection, or collision resistance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorArtifact

open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState
open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayTransition
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.ConstantPins
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryArtifact
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement

def activeFields : ArmKind → Nat
  | .full => 1024
  | .final => 575

def afterRuntimeAbsorbed : ArmKind → Nat
  | .full => 0
  | .final => 3

def statePins : ArmKind → List (Nat × Nat)
  | .full =>
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.fullStatePins
  | .final =>
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.finalStatePins

def canonicalStatePinRows (kind : ArmKind) : List Row :=
  ConstantPins.rows (statePins kind)

def emittedStatePinRows : ArmKind → List Row
  | .full =>
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.fullStatePinRows
  | .final =>
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.finalStatePinRows

theorem exact_statePinRows (kind : ArmKind) :
    (((armFor kind).glueRows.map IndexedRow.row).drop 9).take 4 =
      emittedStatePinRows kind := by
  cases kind
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.fullArm_statePinRows_exact
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.finalArm_statePinRows_exact

def cursorRows : ArmKind → List Row
  | .full =>
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.fullCursorRows
  | .final =>
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.finalCursorRows

theorem exact_cursorRows (kind : ArmKind) :
    (((armFor kind).glueRows.map IndexedRow.row).drop 13).take 3 =
      cursorRows kind := by
  cases kind
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.fullArm_cursorRows_exact
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorRowCertificate.finalArm_cursorRows_exact

private theorem sliced_rows_satisfy
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment)
    (start count : Nat) :
    Satisfies
      ((((armFor kind).glueRows.map IndexedRow.row).drop start).take count)
      assignment := by
  intro row member
  rcases List.mem_map.mp
      (List.mem_of_mem_drop (List.mem_of_mem_take member)) with
    ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

private theorem emittedStatePinRows_satisfy
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (emittedStatePinRows kind) assignment := by
  rw [← exact_statePinRows kind]
  exact sliced_rows_satisfy kind assignment satisfied 9 4

private theorem canonicalStatePinRows_normalized_in_emitted
    (kind : ArmKind) :
    rowsIncluded
      (Poseidon2Normalized.normalizeProgram (canonicalStatePinRows kind))
      (emittedStatePinRows kind) = true := by
  cases kind <;> decide

private theorem canonicalStatePinRows_satisfy
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (canonicalStatePinRows kind) assignment := by
  apply (Poseidon2Normalized.satisfies_normalizeProgram
    (canonicalStatePinRows kind) assignment).mp
  intro row member
  exact emittedStatePinRows_satisfy kind assignment satisfied row
    (rowsIncluded_sound
      (canonicalStatePinRows_normalized_in_emitted kind) row member)

private theorem state_pin_facts
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    ∀ pin ∈ statePins kind, assignment pin.1 = pin.2 := by
  exact ConstantPins.sound
    (pins := statePins kind)
    (programRows := canonicalStatePinRows kind)
    (assignment := assignment)
    (by cases kind <;> decide)
    (by cases kind <;> decide)
    canonical one
    (canonicalStatePinRows_satisfy kind assignment satisfied)

@[simp] theorem transitionColumn_before_runtimeAbsorbed (kind : ArmKind) :
    transitionColumn kind (transitionIndex .before runtimeAbsorbedIndex) = 18 := by
  rw [transitionColumn_eq_structural]
  rfl

@[simp] theorem transitionColumn_after_runtimeAbsorbed (kind : ArmKind) :
    transitionColumn kind (transitionIndex .after runtimeAbsorbedIndex) = 428 := by
  rw [transitionColumn_eq_structural]
  rfl

@[simp] theorem transitionColumn_before_frameCursor (kind : ArmKind) :
    transitionColumn kind (transitionIndex .before frameCursorIndex) = 19 := by
  rw [transitionColumn_eq_structural]
  rfl

@[simp] theorem transitionColumn_after_frameCursor (kind : ArmKind) :
    transitionColumn kind (transitionIndex .after frameCursorIndex) = 429 := by
  rw [transitionColumn_eq_structural]
  rfl

@[simp] theorem transitionColumn_before_programCursor (kind : ArmKind) :
    transitionColumn kind (transitionIndex .before programCursorIndex) = 344 := by
  rw [transitionColumn_eq_structural]
  rfl

@[simp] theorem transitionColumn_after_programCursor (kind : ArmKind) :
    transitionColumn kind (transitionIndex .after programCursorIndex) = 754 := by
  rw [transitionColumn_eq_structural]
  rfl

@[simp] theorem transitionColumn_before_expectedAbsorbed (kind : ArmKind) :
    transitionColumn kind (transitionIndex .before expectedAbsorbedIndex) = 9 := by
  rw [show expectedAbsorbedIndex = expectedWordIndex ⟨8, by decide⟩ by
    apply Fin.ext
    rfl]
  exact transitionColumn_before_expected kind ⟨8, by decide⟩

@[simp] theorem transitionColumn_after_expectedAbsorbed (kind : ArmKind) :
    transitionColumn kind (transitionIndex .after expectedAbsorbedIndex) = 419 := by
  rw [show expectedAbsorbedIndex = expectedWordIndex ⟨8, by decide⟩ by
    apply Fin.ext
    rfl]
  exact transitionColumn_after_expected kind ⟨8, by decide⟩

/-- The four exact generated pins establish the two phase preconditions and
both declared absorbed counters. -/
theorem generated_rows_imply_state_constants
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (decodedTransition kind assignment).before.expected.absorbed =
        residueNat (claimFrameLength % Poseidon2Sponge.rate) ∧
      (decodedTransition kind assignment).after.expected.absorbed =
        residueNat (claimFrameLength % Poseidon2Sponge.rate) ∧
      (decodedTransition kind assignment).before.runtime.absorbed = 0 ∧
      (decodedTransition kind assignment).after.runtime.absorbed =
        residueNat (activeFields kind % Poseidon2Sponge.rate) := by
  have facts := state_pin_facts kind assignment canonical one satisfied
  have beforeExpected := facts (9, 3) (by cases kind <;> decide)
  have afterExpected := facts (419, 3) (by cases kind <;> decide)
  have beforeRuntime := facts (18, 0) (by cases kind <;> decide)
  have afterRuntime := facts (428, afterRuntimeAbsorbed kind)
    (by cases kind <;> decide)
  constructor
  · change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .before expectedAbsorbedIndex))) = _
    rw [transitionColumn_before_expectedAbsorbed]
    simpa [claimFrameLength, Poseidon2Sponge.rate] using
      congrArg residueNat beforeExpected
  constructor
  · change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .after expectedAbsorbedIndex))) = _
    rw [transitionColumn_after_expectedAbsorbed]
    simpa [claimFrameLength, Poseidon2Sponge.rate] using
      congrArg residueNat afterExpected
  constructor
  · change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .before runtimeAbsorbedIndex))) = 0
    rw [transitionColumn_before_runtimeAbsorbed]
    simpa using congrArg residueNat beforeRuntime
  · change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .after runtimeAbsorbedIndex))) = _
    rw [transitionColumn_after_runtimeAbsorbed]
    cases kind <;>
      simpa [activeFields, afterRuntimeAbsorbed, Poseidon2Sponge.rate] using
        congrArg residueNat afterRuntime

def alignmentTerms : List (Nat × Nat) :=
  [(344, 1024), (0, goldilocksP - 97280)]

def frameAdvanceTerms (kind : ArmKind) : List (Nat × Nat) :=
  [(19, 1), (0, activeFields kind)]

def programAdvanceTerms : List (Nat × Nat) :=
  [(344, 1), (0, 1)]

def alignmentRow : Row :=
  ⟨[(0, 97280), (19, 1), (344, goldilocksP - 1024)],
    [(0, 1)], []⟩

def frameAdvanceRow (kind : ArmKind) : Row :=
  ⟨[(0, goldilocksP - activeFields kind), (19, goldilocksP - 1),
      (429, 1)], [(0, 1)], []⟩

def programAdvanceRow : Row :=
  ⟨[(0, goldilocksP - 1), (344, goldilocksP - 1), (754, 1)],
    [(0, 1)], []⟩

private theorem cursorRows_shape (kind : ArmKind) :
    cursorRows kind =
      [alignmentRow, frameAdvanceRow kind, programAdvanceRow] := by
  cases kind <;> rfl

private theorem cursorRows_satisfy
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (cursorRows kind) assignment := by
  rw [← exact_cursorRows kind]
  exact sliced_rows_satisfy kind assignment satisfied 13 3

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

private theorem alignmentRow_perms :
    alignmentRow.a.Perm (builderLinearRow 19 alignmentTerms).a ∧
      alignmentRow.b.Perm (builderLinearRow 19 alignmentTerms).b ∧
      alignmentRow.c.Perm (builderLinearRow 19 alignmentTerms).c := by
  decide

private theorem frameAdvanceRow_perms (kind : ArmKind) :
    (frameAdvanceRow kind).a.Perm
        (builderLinearRow 429 (frameAdvanceTerms kind)).a ∧
      (frameAdvanceRow kind).b.Perm
        (builderLinearRow 429 (frameAdvanceTerms kind)).b ∧
      (frameAdvanceRow kind).c.Perm
        (builderLinearRow 429 (frameAdvanceTerms kind)).c := by
  cases kind <;> decide

private theorem programAdvanceRow_perms :
    programAdvanceRow.a.Perm (builderLinearRow 754 programAdvanceTerms).a ∧
      programAdvanceRow.b.Perm
        (builderLinearRow 754 programAdvanceTerms).b ∧
      programAdvanceRow.c.Perm
        (builderLinearRow 754 programAdvanceTerms).c := by
  decide

private theorem alignmentTerms_canonical : CanonicalTerms alignmentTerms := by
  decide

private theorem frameAdvanceTerms_canonical (kind : ArmKind) :
    CanonicalTerms (frameAdvanceTerms kind) := by
  cases kind <;> decide

private theorem programAdvanceTerms_canonical :
    CanonicalTerms programAdvanceTerms := by
  decide

/-- Exact modular equations enforced by the three Rust cursor rows. -/
theorem cursor_field_equations
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    assignment 19 =
        (1024 * assignment 344 + (goldilocksP - 97280)) % goldilocksP ∧
      assignment 429 =
        (assignment 19 + activeFields kind) % goldilocksP ∧
      assignment 754 = (assignment 344 + 1) % goldilocksP := by
  have rows := cursorRows_satisfy kind assignment satisfied
  rw [cursorRows_shape] at rows
  have alignmentHolds := rows alignmentRow (by simp)
  have alignmentBuilder := rowHolds_of_operand_perms assignment
    alignmentRow_perms.1 alignmentRow_perms.2.1
    alignmentRow_perms.2.2 alignmentHolds
  have alignment := builderLinearRow_sound canonical one 19 alignmentTerms
    alignmentTerms_canonical alignmentBuilder
  have frameHolds := rows (frameAdvanceRow kind) (by simp)
  have frameBuilder := rowHolds_of_operand_perms assignment
    (frameAdvanceRow_perms kind).1 (frameAdvanceRow_perms kind).2.1
    (frameAdvanceRow_perms kind).2.2 frameHolds
  have frame := builderLinearRow_sound canonical one 429
    (frameAdvanceTerms kind) (frameAdvanceTerms_canonical kind) frameBuilder
  have programHolds := rows programAdvanceRow (by simp)
  have programBuilder := rowHolds_of_operand_perms assignment
    programAdvanceRow_perms.1 programAdvanceRow_perms.2.1
    programAdvanceRow_perms.2.2 programHolds
  have program := builderLinearRow_sound canonical one 754
    programAdvanceTerms programAdvanceTerms_canonical programBuilder
  simpa [alignmentTerms, frameAdvanceTerms, programAdvanceTerms, lcEval,
    one, Nat.add_comm, Nat.mul_comm] using
      And.intro alignment (And.intro frame program)

private theorem residueNat_eq_small_nat
    {value target : Nat}
    (valueCanonical : value < goldilocksP)
    (targetCanonical : target < goldilocksP)
    (equal : residueNat value = residueNat target) : value = target := by
  have values := congrArg Fin.val equal
  simp only [residueNat_val] at values
  rwa [Nat.mod_eq_of_lt valueCanonical,
    Nat.mod_eq_of_lt targetCanonical] at values

/-- Cursor rows plus the verifier-selected chunk imply all four exact cursor
fields required by `PhaseStep`. The rows cannot select their own phase. -/
theorem generated_rows_and_selected_chunk_imply_phase_cursors
    (kind : ArmKind) (chunk : Fin claimChunkCount)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (activeExact : activeFields kind = claimChunkFieldCount chunk)
    (programSelected :
      (decodedTransition kind assignment).before.programCursor =
        residueNat (claimProgramStart + chunk.val)) :
    (decodedTransition kind assignment).before.frameCursor =
        residueNat (chunk.val * claimChunkWidth) ∧
      (decodedTransition kind assignment).after.frameCursor =
        residueNat
          (chunk.val * claimChunkWidth + claimChunkFieldCount chunk) ∧
      (decodedTransition kind assignment).before.programCursor =
        residueNat (claimProgramStart + chunk.val) ∧
      (decodedTransition kind assignment).after.programCursor =
        residueNat (claimProgramStart + chunk.val + 1) := by
  have selectedField :
      residueNat (assignment 344) =
        residueNat (claimProgramStart + chunk.val) := by
    change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .before programCursorIndex))) = _ at programSelected
    rwa [transitionColumn_before_programCursor] at programSelected
  have selectedLt : claimProgramStart + chunk.val < goldilocksP := by
    have chunkBound := chunk.isLt
    have startExact := claimProgramStart_exact
    have modulusLarge : 1000000 < goldilocksP := by decide
    unfold claimChunkCount at chunkBound
    omega
  have selectedNat :
      assignment 344 = claimProgramStart + chunk.val :=
    residueNat_eq_small_nat (canonical 344) selectedLt selectedField
  have equations := cursor_field_equations kind assignment canonical one satisfied
  have frameBeforeSmall : chunk.val * claimChunkWidth < goldilocksP := by
    have chunkBound := chunk.isLt
    have modulusLarge : 1000000 < goldilocksP := by decide
    unfold claimChunkCount claimChunkWidth at *
    omega
  have alignmentValue :
      (1024 * (claimProgramStart + chunk.val) +
          (goldilocksP - 97280)) % goldilocksP =
        chunk.val * claimChunkWidth := by
    have startExact := claimProgramStart_exact
    have widthExact : claimChunkWidth = 1024 := rfl
    have modulusLarge : 97280 < goldilocksP := by decide
    rw [startExact, widthExact]
    have arithmetic :
        1024 * (95 + chunk.val) + (goldilocksP - 97280) =
          goldilocksP + chunk.val * 1024 := by
      omega
    rw [arithmetic]
    rw [Nat.add_mod, Nat.mod_self, zero_add, Nat.mod_mod]
    exact Nat.mod_eq_of_lt (by simpa [claimChunkWidth] using frameBeforeSmall)
  have beforeFrameNat :
      assignment 19 = chunk.val * claimChunkWidth := by
    rw [equations.1, selectedNat, alignmentValue]
  have frameAfterSmall :
      chunk.val * claimChunkWidth + claimChunkFieldCount chunk <
        goldilocksP := by
    have chunkBound := chunk.isLt
    have fieldCountBound := claimChunkFieldCount_le chunk
    have modulusLarge : 1000000 < goldilocksP := by decide
    unfold claimChunkCount at chunkBound
    unfold claimChunkWidth at fieldCountBound ⊢
    omega
  have afterFrameNat :
      assignment 429 =
        chunk.val * claimChunkWidth + claimChunkFieldCount chunk := by
    calc
      assignment 429 =
          (assignment 19 + activeFields kind) % goldilocksP := equations.2.1
      _ = (chunk.val * claimChunkWidth + claimChunkFieldCount chunk) %
          goldilocksP := by rw [beforeFrameNat, activeExact]
      _ = chunk.val * claimChunkWidth + claimChunkFieldCount chunk :=
        Nat.mod_eq_of_lt frameAfterSmall
  have programAfterSmall : claimProgramStart + chunk.val + 1 < goldilocksP := by
    have chunkBound := chunk.isLt
    have startExact := claimProgramStart_exact
    have modulusLarge : 1000000 < goldilocksP := by decide
    unfold claimChunkCount at chunkBound
    omega
  have afterProgramNat :
      assignment 754 = claimProgramStart + chunk.val + 1 := by
    calc
      assignment 754 = (assignment 344 + 1) % goldilocksP := equations.2.2
      _ = (claimProgramStart + chunk.val + 1) % goldilocksP := by
        rw [selectedNat]
      _ = claimProgramStart + chunk.val + 1 :=
        Nat.mod_eq_of_lt programAfterSmall
  constructor
  · change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .before frameCursorIndex))) = _
    rw [transitionColumn_before_frameCursor]
    rw [beforeFrameNat]
  constructor
  · change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .after frameCursorIndex))) = _
    rw [transitionColumn_after_frameCursor]
    rw [afterFrameNat]
  constructor
  · exact programSelected
  · change residueNat
      (assignment (transitionColumn kind
        (transitionIndex .after programCursorIndex))) = _
    rw [transitionColumn_after_programCursor]
    rw [afterProgramNat]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateCursorArtifact
