import Nightstream.Implementation.Nebula.NIFS.PiRLC.FirstAcceptedBatchSound

/-!
Contract: global row-derived V2 PiRLC sampler availability and response.

The local batch theorem fixes one physical selector output. This file maps
every exact paper coordinate to that local theorem. It then derives the
fail-closed sampler gate and the exact scalar response used by PiRLC.
-/

set_option autoImplicit false
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchRows
open Nightstream.Implementation.Nebula.ProductPiRlcFirstAcceptedBatchSound

/-- Convert an exact paper sampler coordinate to its indexed physical row
occurrence. -/
def coordinateIndex
    (source : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total)
    (coefficient : Fin ProductPoseidon2.samplerCoefficientCount) :
    CoordinateIndex where
  source := Fin.cast ProductPiRlcTranscriptRows.scalarCount_profile.symm source
  coefficient := Fin.cast
    ProductPiRlcTranscriptRows.coefficientCount_profile.symm coefficient

@[simp] theorem paperSource_coordinateIndex
    (source : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total)
    (coefficient : Fin ProductPoseidon2.samplerCoefficientCount) :
    paperSource (coordinateIndex source coefficient) = source := by
  apply Fin.ext
  rfl

@[simp] theorem paperCoefficient_coordinateIndex
    (source : Fin Nightstream.SuperNeo.Folding.Nifs.PaperProfile.arity.total)
    (coefficient : Fin ProductPoseidon2.samplerCoefficientCount) :
    paperCoefficient (coordinateIndex source coefficient) = coefficient := by
  apply Fin.ext
  rfl

/-- Satisfaction of all transcript, classifier, and selector rows excludes a
sampler shortfall at every exact paper coordinate. -/
theorem sampler_available
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (selectorRows : RowsHold input assignment) :
    ProductPoseidon2.SamplerAvailable (samplerState input assignment) := by
  apply ProductPoseidon2.samplerAvailable_of_all
  intro source coefficient
  have succeeds := sampleCoefficient_ne_none input assignment canonical one
    transcriptRows classificationRows selectorRows
    (coordinateIndex source coefficient)
  simpa only [paperSource_coordinateIndex, paperCoefficient_coordinateIndex]
    using succeeds

/-- The selected verifier's executable sampler gate is true by row
satisfaction, not by a caller-provided Boolean. -/
theorem samplerSucceeded_eq_true
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (selectorRows : RowsHold input assignment) :
    ProductPoseidon2.samplerSucceeded (samplerState input assignment) = true :=
  (ProductPoseidon2.samplerSucceeded_eq_true_iff _).2
    (sampler_available input assignment canonical one transcriptRows
      classificationRows selectorRows)

/-- Each physical selector output is the exact coefficient used by the
selected 15-scalar PiRLC response. -/
theorem output_eq_scalarResponse
    (input : ProductPiRlcTranscriptRows.Input) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (transcriptRows : ProductPiRlcTranscriptRows.RowsHold input assignment)
    (classificationRows :
      ProductPiRlcCandidateClassificationRows.RowsHold input assignment)
    (selectorRows : RowsHold input assignment)
    (index : CoordinateIndex) :
    assignment
        (ProductPiRlcFirstAcceptedRows.outputColumn (layout input index)) =
      (ProductPoseidon2.scalarResponse (samplerState input assignment)
        (paperSource index)
        (Fin.cast (by rfl) (paperCoefficient index))).val := by
  obtain ⟨selected, sampled, output⟩ := sampleCoefficient_eq_some_output input
    assignment canonical one transcriptRows classificationRows selectorRows index
  let scalarCoefficient : Fin
      Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.coefficientCount :=
    Fin.cast (by rfl) (paperCoefficient index)
  have sampledAtScalar :
      ProductPoseidon2.sampleCoefficient (samplerState input assignment)
          (paperSource index) (Fin.cast (by rfl) scalarCoefficient) =
        some selected := by
    simpa only [scalarCoefficient] using sampled
  have responseEq := ProductPoseidon2.scalarResponse_eq_of_sampled
    (samplerState input assignment) (paperSource index) scalarCoefficient
    selected sampledAtScalar
  exact output.trans (congrArg Fin.val responseEq.symm)

end Nightstream.Implementation.Nebula.ProductPiRlcSamplerResponseSound
