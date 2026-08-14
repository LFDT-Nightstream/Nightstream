import Nightstream.Implementation.Nebula.NIFS.PiRLC.AlgebraSound
import Nightstream.Implementation.Nebula.NIFS.PiRLC.SamplerResponseSound

/-!
Contract: exact selector-to-PiRLC-algebra challenge bridge.

This file binds every shared algebra challenge-symbol column to the matching
first-accepted sampler output. Transcript, classifier, and selector row
satisfaction then derives the five-symbol range and proves that the decoded
challenge rings equal the selected Poseidon2 PiRLC response.

The placement contains column identities only. It does not contain a
challenge value, sampler result, range proof, parent equation, or verifier
acceptance result.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Nebula.ProductPiRlcChallengeBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows
open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraSound
open Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchRows
open Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchSound
open Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound
open Nightstream.SuperNeo.Concrete

/-- Physical column placement of all 810 selected challenge symbols. -/
structure Placement
    (samplerInput : ProductPiRlcTranscriptRows.Input)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout) : Prop where
  challengeSymbol : forall source lane,
    algebraLayout.challengeSymbol source lane =
      ProductPiRlcFirstAcceptedRows.outputColumn
        (ProductPiRlcFirstAcceptedBatchRows.layout samplerInput
          (coordinateIndex source lane))

/-- The exact sampler rows derive the range needed by every ring-algebra
occurrence. No range premise is accepted from the caller. -/
theorem challengeSymbol_range
    (samplerInput : ProductPiRlcTranscriptRows.Input)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold samplerInput assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold samplerInput assignment)
    (selectorRows :
      ProductPiRlcFirstAcceptedBatchRows.RowsHold samplerInput assignment)
    (placement : Placement samplerInput algebraLayout) :
    forall source lane,
      assignment (algebraLayout.challengeSymbol source lane) < 5 := by
  intro source lane
  have output := output_eq_scalarResponse samplerInput assignment canonical one
    transcriptRows classificationRows selectorRows
    (coordinateIndex source lane)
  rw [placement.challengeSymbol source lane, output]
  exact
    (ProductPoseidon2.scalarResponse (samplerState samplerInput assignment)
      source lane).isLt

/-- The decoded algebra challenges are exactly the successful selected
Poseidon2 response at the common post-PiCCS state. -/
theorem decodeChallenges_eq_piRlcResponse
    (samplerInput : ProductPiRlcTranscriptRows.Input)
    (algebraLayout : ProductPiRlcAlgebraRows.Layout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold samplerInput assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold samplerInput assignment)
    (selectorRows :
      ProductPiRlcFirstAcceptedBatchRows.RowsHold samplerInput assignment)
    (placement : Placement samplerInput algebraLayout) :
    decodeChallenges algebraLayout assignment
        (challengeSymbol_range samplerInput algebraLayout assignment canonical one
          transcriptRows classificationRows selectorRows placement) =
      ProductPoseidon2.piRlcResponse
        (samplerState samplerInput assignment) := by
  funext source lane
  let symbolRange :=
    challengeSymbol_range samplerInput algebraLayout assignment canonical one
      transcriptRows classificationRows selectorRows placement
  have output := output_eq_scalarResponse samplerInput assignment canonical one
    transcriptRows classificationRows selectorRows
    (coordinateIndex source lane)
  have valueEq :
      assignment (algebraLayout.challengeSymbol source lane) =
        (ProductPoseidon2.scalarResponse
          (samplerState samplerInput assignment) source lane).val := by
    rw [placement.challengeSymbol source lane]
    simpa only [paperSource_coordinateIndex, paperCoefficient_coordinateIndex]
      using output
  have coefficientEq :
      (⟨assignment (algebraLayout.challengeSymbol source lane),
          symbolRange source lane⟩ : ProductPoseidon2.Coefficient) =
        ProductPoseidon2.scalarResponse
          (samplerState samplerInput assignment) source lane := by
    apply Fin.ext
    exact valueEq
  change
    Phi81StrongSet.embedCoefficient
        ⟨assignment (algebraLayout.challengeSymbol source lane),
          symbolRange source lane⟩ =
      Phi81StrongSet.embedCoefficient
        (ProductPoseidon2.scalarResponse
          (samplerState samplerInput assignment) source lane)
  exact congrArg Phi81StrongSet.embedCoefficient coefficientEq

end Nightstream.Implementation.Nebula.ProductPiRlcChallengeBridge
