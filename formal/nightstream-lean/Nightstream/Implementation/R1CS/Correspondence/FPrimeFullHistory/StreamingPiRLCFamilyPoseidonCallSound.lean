import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallRowProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeafReconstruction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafReconstruction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonPartialLeafCertificate

/-!
Contract: same-assignment semantic composition for each production PiRLC
Poseidon2 replay call class.

Assurance tier: artifact-checked call semantics.

Owns: transport from absolute production-column row satisfaction to the 86
typed S-box equations certified for direct, partial-start, and chained leaves.

Does not own: Rust call-class coverage, selector authority, source-value
authority, complete family replay, PiRLC combination, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallSound

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection

private abbrev directRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedRows

private abbrev directSteps :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.decodedSteps

private abbrev partialRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.partialDecodedRows

private abbrev chainedRows :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafCertificate.decodedRows

private abbrev StepSboxHolds :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepSboxHolds

private abbrev directSource :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction.reconstructedSource

private abbrev chainedSource :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.reconstructedSource

theorem direct_absolute_rows_imply_step_sboxes
    (index : Nat) (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment directSelectorColumn = 1)
    (holds : ∀ row ∈ directRows,
      absoluteResidual .direct index assignment row = 0) :
    ∀ step ∈ directSteps,
      StepSboxHolds
        (directSource (projectFinalAssignment .direct index assignment)) step := by
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction.decoded_rows_imply_reconstructed_step_sboxes
  · simpa using one
  · simpa [selectorColumn] using selectorOne
  · exact absolute_rows_imply_projected_rows
      .direct index assignment directRows holds

theorem partial_absolute_rows_imply_step_sboxes
    (index : Nat) (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment partialSelectorColumn = 1)
    (holds : ∀ row ∈ partialRows,
      absoluteResidual .partialStart index assignment row = 0) :
    ∀ step ∈ directSteps,
      StepSboxHolds
        (directSource
          (projectFinalAssignment .partialStart index assignment)) step := by
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.partial_rows_imply_direct_reconstructed_step_sboxes
  · simpa using one
  · simpa [selectorColumn] using selectorOne
  · exact absolute_rows_imply_projected_rows
      .partialStart index assignment partialRows holds

theorem chained_absolute_rows_imply_step_sboxes
    (selector index : Nat) (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment selector = 1)
    (holds : ∀ row ∈ chainedRows,
      absoluteResidual (.chained selector) index assignment row = 0) :
    ∀ step ∈ directSteps,
      StepSboxHolds
        (chainedSource
          (projectFinalAssignment (.chained selector) index assignment)) step := by
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.decoded_rows_imply_reconstructed_step_sboxes
  · simpa using one
  · simpa [selectorColumn] using selectorOne
  · exact absolute_rows_imply_projected_rows
      (.chained selector) index assignment chainedRows holds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallSound
