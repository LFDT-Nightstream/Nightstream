import Nightstream.Implementation.Nebula.NIFS.PiRLC.CandidateClassificationRows
import Nightstream.Implementation.Nebula.NIFS.PiRLC.FullFieldCandidateSound
import Nightstream.Implementation.Nebula.NIFS.PiRLC.TranscriptSemantics

/-!
Contract: exact semantic refinement for all V2 PiRLC candidate classifiers.

Combined transcript and classification row satisfaction derives, for every
source, coefficient, and attempt:

* the exact full-field Poseidon2 candidate;
* the exact reject-only-`q-1` accept bit; and
* the exact modulo-five digit.

The theorem does not receive any candidate value, accept decision, or digit
as an assumption.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationRows

def exactCandidate
    (input : ProductPiRlcTranscriptRows.Input)
    (assignment : Nat -> Nat)
    (index : CandidateIndex) :=
  ProductPoseidon2.candidateValue
    (ProductPiRlcTranscriptSemantics.valueStart assignment input)
    (Fin.cast ProductPiRlcTranscriptRows.scalarCount_profile index.source)
    (Fin.cast ProductPiRlcTranscriptRows.coefficientCount_profile
      index.coefficient)
    (Fin.cast ProductPiRlcTranscriptRows.attemptCount_profile index.attempt)

theorem classified_value_exact
    (input : ProductPiRlcTranscriptRows.Input)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows : RowsHold input assignment)
    (index : CandidateIndex) :
    ProductPiRlcFullFieldCandidateSound.candidateValue assignment
        (layout input index) =
      (exactCandidate input assignment index).val := by
  have classified :=
    ProductPiRlcFullFieldCandidateSound.input_eq_candidateValue
      canonical one (classificationRows index)
  have transcript :=
    ProductPiRlcTranscriptSemantics.candidate_rows_sound input assignment
      canonical one transcriptRows index
  have linked := classified.symm.trans transcript
  simpa [layout] using linked

/-- One physical classifier refines its exact transcript candidate. -/
structure CandidateRefines
    (input : ProductPiRlcTranscriptRows.Input)
    (assignment : Nat -> Nat)
    (index : CandidateIndex) : Prop where
  value :
    ProductPiRlcFullFieldCandidateSound.candidateValue assignment
        (layout input index) =
      (exactCandidate input assignment index).val
  accepted :
    assignment
        (ProductPiRlcFullFieldCandidateRows.acceptColumn (layout input index)) =
      if ProductPoseidon2.candidateAccepted
          (exactCandidate input assignment index) then 1 else 0
  digit :
    assignment
        (ProductPiRlcFullFieldCandidateRows.residueColumn (layout input index)) =
      (ProductPoseidon2.candidateDigit
        (exactCandidate input assignment index)).val

theorem candidate_sound
    (input : ProductPiRlcTranscriptRows.Input)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows : RowsHold input assignment)
    (index : CandidateIndex) :
    CandidateRefines input assignment index := by
  have localSound := ProductPiRlcFullFieldCandidateSound.sound
    canonical one (classificationRows index)
  have valueEq := classified_value_exact input assignment canonical one
    transcriptRows classificationRows index
  refine {
    value := valueEq
    accepted := ?_
    digit := ?_ }
  · rw [localSound.accepted, valueEq]
    simp [ProductPoseidon2.candidateAccepted, goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus]
  · rw [localSound.residue, valueEq]
    rfl

/-- Every physical classifier refines every exact transcript candidate. -/
theorem all_candidates_sound
    (input : ProductPiRlcTranscriptRows.Input)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows : RowsHold input assignment) :
    forall index, CandidateRefines input assignment index :=
  candidate_sound input assignment canonical one transcriptRows
    classificationRows

end Nightstream.Implementation.Nebula.ProductPiRlcCandidateClassificationSound
