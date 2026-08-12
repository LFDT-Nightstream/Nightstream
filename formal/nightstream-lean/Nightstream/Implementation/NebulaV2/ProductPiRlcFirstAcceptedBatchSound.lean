import Nightstream.Implementation.NebulaV2.ProductPiRlcCandidateClassificationSound
import Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows
import Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedSound

/-!
Contract: exact row-derived V2 PiRLC sampler result for all 15 x 54
coordinates.

For each coordinate, transcript rows fix three full-field candidates,
classification rows fix their accept bits and modulo-five digits, and the
fail-closed selector rows fix the first accepted digit. The headline theorem
derives an actual successful `ProductPoseidon2.sampleCoefficient` result and
the exact matching output wire.

Sampler availability is a conclusion. It is not an assumption.
-/

set_option autoImplicit false
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchRows

def samplerState
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat) :=
  ProductPiRlcTranscriptSemantics.valueStart assignment input

def paperSource (index : CoordinateIndex) :=
  Fin.cast ProductPiRlcTranscriptRows.scalarCount_profile index.source

def paperCoefficient (index : CoordinateIndex) :=
  Fin.cast ProductPiRlcTranscriptRows.coefficientCount_profile index.coefficient

def exactCandidate
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (index : CoordinateIndex) (attempt : Fin attemptCount) :=
  ProductPoseidon2.candidateValue (samplerState input assignment)
    (paperSource index) (paperCoefficient index)
    (Fin.cast ProductPiRlcTranscriptRows.attemptCount_profile attempt)

theorem classifier_refines
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (index : CoordinateIndex) (attempt : Fin attemptCount) :
    ProductPiRlcCandidateClassificationSound.CandidateRefines input assignment
      (candidateIndex index attempt) :=
  ProductPiRlcCandidateClassificationSound.candidate_sound input assignment
    canonical one transcriptRows classificationRows (candidateIndex index attempt)

theorem accept_bits
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (index : CoordinateIndex) :
    ProductPiRlcFirstAcceptedSound.AcceptBits assignment (layout input index) := by
  intro attempt
  have refined := classifier_refines input assignment canonical one
    transcriptRows classificationRows index attempt
  change assignment
    (ProductPiRlcFullFieldCandidateRows.acceptColumn
      (ProductPiRlcCandidateClassificationRows.layout input
        (candidateIndex index attempt))) ≤ 1
  rw [refined.accepted]
  split <;> simp

theorem residues_in_range
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (index : CoordinateIndex) :
    ProductPiRlcFirstAcceptedSound.ResiduesInRange assignment
      (layout input index) := by
  intro attempt
  have refined := classifier_refines input assignment canonical one
    transcriptRows classificationRows index attempt
  change assignment
    (ProductPiRlcFullFieldCandidateRows.residueColumn
      (ProductPiRlcCandidateClassificationRows.layout input
        (candidateIndex index attempt))) < 5
  rw [refined.digit]
  exact (ProductPoseidon2.candidateDigit
    (ProductPiRlcCandidateClassificationSound.exactCandidate input assignment
      (candidateIndex index attempt))).isLt

theorem selector_refines
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (selectorRows : RowsHold input assignment)
    (index : CoordinateIndex) :
    ProductPiRlcFirstAcceptedSound.Refines assignment (layout input index) :=
  ProductPiRlcFirstAcceptedSound.sound canonical one
    (accept_bits input assignment canonical one transcriptRows
      classificationRows index)
    (residues_in_range input assignment canonical one transcriptRows
      classificationRows index)
    (selectorRows index)

private theorem exactCandidate_eq_classifier
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (index : CoordinateIndex) (attempt : Fin attemptCount) :
    exactCandidate input assignment index attempt =
      ProductPiRlcCandidateClassificationSound.exactCandidate input assignment
        (candidateIndex index attempt) := by
  rfl

/-- Every coordinate has a successful exact sampler result whose value is the
physical selector output. This excludes the old silent-default behavior. -/
theorem sampleCoefficient_eq_some_output
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (selectorRows : RowsHold input assignment)
    (index : CoordinateIndex) :
    exists selected : ProductPoseidon2.Coefficient,
      ProductPoseidon2.sampleCoefficient (samplerState input assignment)
          (paperSource index) (paperCoefficient index) = some selected /\
        assignment
            (ProductPiRlcFirstAcceptedRows.outputColumn (layout input index)) =
          selected.val := by
  let firstCandidate := exactCandidate input assignment index
    ProductPiRlcFirstAcceptedRows.first
  let secondCandidate := exactCandidate input assignment index
    ProductPiRlcFirstAcceptedRows.second
  let thirdCandidate := exactCandidate input assignment index
    ProductPiRlcFirstAcceptedRows.third
  have firstRefined := classifier_refines input assignment canonical one
    transcriptRows classificationRows index ProductPiRlcFirstAcceptedRows.first
  have secondRefined := classifier_refines input assignment canonical one
    transcriptRows classificationRows index ProductPiRlcFirstAcceptedRows.second
  have thirdRefined := classifier_refines input assignment canonical one
    transcriptRows classificationRows index ProductPiRlcFirstAcceptedRows.third
  have selector := selector_refines input assignment canonical one transcriptRows
    classificationRows selectorRows index
  have firstCandidateEq := exactCandidate_eq_classifier input assignment index
    ProductPiRlcFirstAcceptedRows.first
  have secondCandidateEq := exactCandidate_eq_classifier input assignment index
    ProductPiRlcFirstAcceptedRows.second
  have thirdCandidateEq := exactCandidate_eq_classifier input assignment index
    ProductPiRlcFirstAcceptedRows.third
  have firstCandidatePaper : firstCandidate =
      ProductPoseidon2.candidateValue (samplerState input assignment)
        (paperSource index) (paperCoefficient index)
        ProductPoseidon2.firstAttempt := by
    rfl
  have secondCandidatePaper : secondCandidate =
      ProductPoseidon2.candidateValue (samplerState input assignment)
        (paperSource index) (paperCoefficient index)
        ProductPoseidon2.secondAttempt := by
    rfl
  have thirdCandidatePaper : thirdCandidate =
      ProductPoseidon2.candidateValue (samplerState input assignment)
        (paperSource index) (paperCoefficient index)
        ProductPoseidon2.thirdAttempt := by
    rfl
  by_cases firstAccepted : ProductPoseidon2.candidateAccepted firstCandidate = true
  · refine ⟨ProductPoseidon2.candidateDigit firstCandidate, ?_, ?_⟩
    · rw [firstCandidatePaper] at firstAccepted ⊢
      exact ProductPoseidon2.sampleCoefficient_of_first _ _ _ firstAccepted
    · have acceptOne :
          assignment ((layout input index).accept
            ProductPiRlcFirstAcceptedRows.first) = 1 := by
        change assignment
          (ProductPiRlcFullFieldCandidateRows.acceptColumn
            (ProductPiRlcCandidateClassificationRows.layout input
              (candidateIndex index ProductPiRlcFirstAcceptedRows.first))) = 1
        rw [firstRefined.accepted, ← firstCandidateEq]
        simp [firstCandidate, firstAccepted]
      have output := selector.output
      rw [if_pos acceptOne] at output
      exact output.trans (by
        change assignment
          (ProductPiRlcFullFieldCandidateRows.residueColumn
            (ProductPiRlcCandidateClassificationRows.layout input
              (candidateIndex index ProductPiRlcFirstAcceptedRows.first))) = _
        rw [firstRefined.digit, ← firstCandidateEq])
  · have firstRejected : ProductPoseidon2.candidateAccepted firstCandidate = false := by
      exact Bool.eq_false_of_not_eq_true firstAccepted
    by_cases secondAccepted :
        ProductPoseidon2.candidateAccepted secondCandidate = true
    · refine ⟨ProductPoseidon2.candidateDigit secondCandidate, ?_, ?_⟩
      · rw [firstCandidatePaper] at firstRejected
        rw [secondCandidatePaper] at secondAccepted ⊢
        exact ProductPoseidon2.sampleCoefficient_of_second _ _ _
          firstRejected secondAccepted
      · have acceptZero :
            assignment ((layout input index).accept
              ProductPiRlcFirstAcceptedRows.first) = 0 := by
          change assignment
            (ProductPiRlcFullFieldCandidateRows.acceptColumn
              (ProductPiRlcCandidateClassificationRows.layout input
                (candidateIndex index ProductPiRlcFirstAcceptedRows.first))) = 0
          rw [firstRefined.accepted, ← firstCandidateEq]
          simp [firstCandidate, firstRejected]
        have acceptOne :
            assignment ((layout input index).accept
              ProductPiRlcFirstAcceptedRows.second) = 1 := by
          change assignment
            (ProductPiRlcFullFieldCandidateRows.acceptColumn
              (ProductPiRlcCandidateClassificationRows.layout input
                (candidateIndex index ProductPiRlcFirstAcceptedRows.second))) = 1
          rw [secondRefined.accepted, ← secondCandidateEq]
          simp [secondCandidate, secondAccepted]
        have output := selector.output
        have acceptNotOne : assignment ((layout input index).accept
            ProductPiRlcFirstAcceptedRows.first) ≠ 1 := by omega
        rw [if_neg acceptNotOne, if_pos acceptOne] at output
        exact output.trans (by
          change assignment
            (ProductPiRlcFullFieldCandidateRows.residueColumn
              (ProductPiRlcCandidateClassificationRows.layout input
                (candidateIndex index ProductPiRlcFirstAcceptedRows.second))) = _
          rw [secondRefined.digit, ← secondCandidateEq])
    · have secondRejected :
          ProductPoseidon2.candidateAccepted secondCandidate = false := by
        exact Bool.eq_false_of_not_eq_true secondAccepted
      by_cases thirdAccepted :
          ProductPoseidon2.candidateAccepted thirdCandidate = true
      · refine ⟨ProductPoseidon2.candidateDigit thirdCandidate, ?_, ?_⟩
        · rw [firstCandidatePaper] at firstRejected
          rw [secondCandidatePaper] at secondRejected
          rw [thirdCandidatePaper] at thirdAccepted ⊢
          exact ProductPoseidon2.sampleCoefficient_of_third _ _ _
            firstRejected secondRejected thirdAccepted
        · have acceptZero0 :
              assignment ((layout input index).accept
                ProductPiRlcFirstAcceptedRows.first) = 0 := by
            change assignment
              (ProductPiRlcFullFieldCandidateRows.acceptColumn
                (ProductPiRlcCandidateClassificationRows.layout input
                  (candidateIndex index ProductPiRlcFirstAcceptedRows.first))) = 0
            rw [firstRefined.accepted, ← firstCandidateEq]
            simp [firstCandidate, firstRejected]
          have acceptZero1 :
              assignment ((layout input index).accept
                ProductPiRlcFirstAcceptedRows.second) = 0 := by
            change assignment
              (ProductPiRlcFullFieldCandidateRows.acceptColumn
                (ProductPiRlcCandidateClassificationRows.layout input
                  (candidateIndex index ProductPiRlcFirstAcceptedRows.second))) = 0
            rw [secondRefined.accepted, ← secondCandidateEq]
            simp [secondCandidate, secondRejected]
          have output := selector.output
          have firstNotOne : assignment ((layout input index).accept
              ProductPiRlcFirstAcceptedRows.first) ≠ 1 := by omega
          have secondNotOne : assignment ((layout input index).accept
              ProductPiRlcFirstAcceptedRows.second) ≠ 1 := by omega
          rw [if_neg firstNotOne, if_neg secondNotOne] at output
          exact output.trans (by
            change assignment
              (ProductPiRlcFullFieldCandidateRows.residueColumn
                (ProductPiRlcCandidateClassificationRows.layout input
                  (candidateIndex index ProductPiRlcFirstAcceptedRows.third))) = _
            rw [thirdRefined.digit, ← thirdCandidateEq])
      · have thirdRejected :
            ProductPoseidon2.candidateAccepted thirdCandidate = false := by
          exact Bool.eq_false_of_not_eq_true thirdAccepted
        have acceptZero0 :
            assignment ((layout input index).accept
              ProductPiRlcFirstAcceptedRows.first) = 0 := by
          change assignment
            (ProductPiRlcFullFieldCandidateRows.acceptColumn
              (ProductPiRlcCandidateClassificationRows.layout input
                (candidateIndex index ProductPiRlcFirstAcceptedRows.first))) = 0
          rw [firstRefined.accepted, ← firstCandidateEq]
          simp [firstCandidate, firstRejected]
        have acceptZero1 :
            assignment ((layout input index).accept
              ProductPiRlcFirstAcceptedRows.second) = 0 := by
          change assignment
            (ProductPiRlcFullFieldCandidateRows.acceptColumn
              (ProductPiRlcCandidateClassificationRows.layout input
                (candidateIndex index ProductPiRlcFirstAcceptedRows.second))) = 0
          rw [secondRefined.accepted, ← secondCandidateEq]
          simp [secondCandidate, secondRejected]
        have acceptZero2 :
            assignment ((layout input index).accept
              ProductPiRlcFirstAcceptedRows.third) = 0 := by
          change assignment
            (ProductPiRlcFullFieldCandidateRows.acceptColumn
              (ProductPiRlcCandidateClassificationRows.layout input
                (candidateIndex index ProductPiRlcFirstAcceptedRows.third))) = 0
          rw [thirdRefined.accepted, ← thirdCandidateEq]
          simp [thirdCandidate, thirdRejected]
        rcases selector.available with firstCase | secondCase | thirdCase
        · omega
        · omega
        · omega

/-- A physical selector occurrence excludes shortfall at its exact paper
coordinate. -/
theorem sampleCoefficient_ne_none
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (selectorRows : RowsHold input assignment)
    (index : CoordinateIndex) :
    ProductPoseidon2.sampleCoefficient (samplerState input assignment)
      (paperSource index) (paperCoefficient index) ≠ none := by
  obtain ⟨selected, sampled, _⟩ := sampleCoefficient_eq_some_output input
    assignment canonical one transcriptRows classificationRows selectorRows index
  rw [sampled]
  simp

end Nightstream.Implementation.NebulaV2.ProductPiRlcFirstAcceptedBatchSound
