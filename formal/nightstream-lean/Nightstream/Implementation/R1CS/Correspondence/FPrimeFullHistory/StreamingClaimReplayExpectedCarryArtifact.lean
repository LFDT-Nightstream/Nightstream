import Nightstream.Implementation.Nebula.Production.Carrier.StreamingClaimReplayState
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayExpectedCarryRowCertificate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayStateWordLayoutCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayArtifact

/-!
Contract: exact generated-row refinement for the expected-state carry family
of one production claim-replay phase.

Assurance tier: Rust-conformant for the expected-carry phase field.

Owns the v6 state-column decoder and the nine generated equality rows that
imply `after.expected = before.expected` for either physical arm.

Does not own the remaining phase fields, complete arm validity, Poseidon2
execution, coordinate accumulation, lifecycle selection, or collision
resistance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.EqualityPins
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateWordLayoutCertificate
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.SuperNeo.Concrete

inductive ArmKind where
  | full
  | final
deriving DecidableEq, Repr

def armFor : ArmKind → RawArm
  | .full => fullArm
  | .final => finalArm

theorem arm_stateWordColumns_exact (kind : ArmKind) :
    (armFor kind).stateWordColumns = transitionStateWordColumns := by
  cases kind
  · exact fullArm_stateWordColumns_exact
  · exact finalArm_stateWordColumns_exact

theorem arm_stateWordColumns_length (kind : ArmKind) :
    (armFor kind).stateWordColumns.length = transitionWordCount := by
  rw [arm_stateWordColumns_exact]
  exact transitionStateWordColumns_length.trans exact_word_counts.2.2.symm

/-- Column selected by the exact Rust-emitted state-word list. -/
def transitionColumn
    (kind : ArmKind) (index : Fin transitionWordCount) : Nat :=
  (armFor kind).stateWordColumns[index.val]'(by
    rw [arm_stateWordColumns_length kind]
    exact index.isLt)

def structuralColumn (index : Fin transitionWordCount) : Nat :=
  transitionStateWordColumns[index.val]'(by
    rw [transitionStateWordColumns_length]
    have bound := index.isLt
    have count := exact_word_counts.2.2
    omega)

theorem transitionColumn_eq_structural
    (kind : ArmKind) (index : Fin transitionWordCount) :
    transitionColumn kind index = structuralColumn index := by
  simp only [transitionColumn, structuralColumn,
    arm_stateWordColumns_exact]

def expectedWordIndex (index : Fin 9) : Fin stateWordCount :=
  ⟨index.val, by
    have bound := index.isLt
    have count := exact_word_counts.2.1
    omega⟩

@[simp] theorem transitionColumn_before_expected
    (kind : ArmKind) (index : Fin 9) :
    transitionColumn kind (transitionIndex .before (expectedWordIndex index)) =
      1 + index.val := by
  rw [transitionColumn_eq_structural]
  fin_cases index <;> rfl

@[simp] theorem transitionColumn_after_expected
    (kind : ArmKind) (index : Fin 9) :
    transitionColumn kind (transitionIndex .after (expectedWordIndex index)) =
      411 + index.val := by
  rw [transitionColumn_eq_structural]
  fin_cases index <;> rfl

def transitionWords
    (kind : ArmKind) (assignment : Nat → Nat) :
    Fin transitionWordCount → F :=
  fun index => residueNat (assignment (transitionColumn kind index))

/-- The semantic transition decoded from the arm's exact state-word list. -/
def decodedTransition
    (kind : ArmKind) (assignment : Nat → Nat) : Transition :=
  Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState.decodeTransition
    (transitionWords kind assignment)

def expectedCarryPairs : List (Nat × Nat) :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryRowCertificate.expectedCarryPairs

def expectedCarryRows : List Row :=
  EqualityPins.rows expectedCarryPairs

theorem exact_expectedCarryRows (kind : ArmKind) :
    ((armFor kind).glueRows.map IndexedRow.row).take 9 =
      expectedCarryRows := by
  cases kind
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryRowCertificate.fullArm_expectedCarryRows_exact
  · exact
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryRowCertificate.finalArm_expectedCarryRows_exact

private theorem expectedCarryRows_satisfy
    (kind : ArmKind) (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies expectedCarryRows assignment := by
  intro row member
  have prefixMember :
      row ∈ ((armFor kind).glueRows.map IndexedRow.row).take 9 := by
    rw [exact_expectedCarryRows kind]
    exact member
  rcases List.mem_map.mp (List.mem_of_mem_take prefixMember) with
    ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds (armFor kind) assignment satisfied indexed indexedMember

private theorem expected_column_equal
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (index : Fin 9) :
    assignment (1 + index.val) = assignment (411 + index.val) := by
  have facts := EqualityPins.rows_sound canonical one
    (expectedCarryRows_satisfy kind assignment satisfied)
  apply facts (1 + index.val, 411 + index.val)
  exact List.mem_map.mpr
    ⟨index.val, List.mem_range.mpr index.isLt, rfl⟩

private theorem spongeState_ext
    {left right : SpongeState}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

/-- The exact nine-row generated family proves the `expectedCarry` field of
the authoritative v6 phase relation. -/
theorem generated_rows_imply_expectedCarry
    (kind : ArmKind) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment) :
    (decodedTransition kind assignment).after.expected =
      (decodedTransition kind assignment).before.expected := by
  apply spongeState_ext
  · funext lane
    let index : Fin 9 := ⟨lane.val, by
      have bound := lane.isLt
      unfold spongeWidth at bound
      omega⟩
    have wordIndex : expectedLaneIndex lane = expectedWordIndex index := by
      apply Fin.ext
      simp [expectedLaneIndex, expectedOffset, expectedWordIndex, index]
    have equal := expected_column_equal kind assignment canonical one
      satisfied index
    change residueNat
        (assignment (transitionColumn kind
          (transitionIndex .after (expectedLaneIndex lane)))) =
      residueNat
        (assignment (transitionColumn kind
          (transitionIndex .before (expectedLaneIndex lane))))
    rw [wordIndex, transitionColumn_after_expected,
      transitionColumn_before_expected, equal]
  · let index : Fin 9 := ⟨8, by decide⟩
    have wordIndex : expectedAbsorbedIndex = expectedWordIndex index := by
      apply Fin.ext
      rfl
    have equal := expected_column_equal kind assignment canonical one
      satisfied index
    change residueNat
        (assignment (transitionColumn kind
          (transitionIndex .after expectedAbsorbedIndex))) =
      residueNat
        (assignment (transitionColumn kind
          (transitionIndex .before expectedAbsorbedIndex)))
    rw [wordIndex, transitionColumn_after_expected,
      transitionColumn_before_expected, equal]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryArtifact
