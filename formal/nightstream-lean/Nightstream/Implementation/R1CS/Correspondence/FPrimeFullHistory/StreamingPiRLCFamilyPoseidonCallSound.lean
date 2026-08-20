import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallRowProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonChainedLeafReconstruction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCompactTrace
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
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCompactTrace
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

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
    (site : CallSite) (kindExact : site.kind = .direct)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment directSelectorColumn = 1)
    (holds : ∀ row ∈ directRows,
      absoluteResidual site assignment row = 0) :
    ∀ step ∈ directSteps,
      StepSboxHolds
        (directSource (projectFinalAssignment site assignment)) step := by
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafReconstruction.decoded_rows_imply_reconstructed_step_sboxes
  · simpa using one
  · simpa [projected_selector, kindExact, selectorColumn] using selectorOne
  · exact absolute_rows_imply_projected_rows
      site assignment directRows holds

theorem partial_absolute_rows_imply_step_sboxes
    (site : CallSite) (kindExact : site.kind = .partialStart)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment partialSelectorColumn = 1)
    (holds : ∀ row ∈ partialRows,
      absoluteResidual site assignment row = 0) :
    ∀ step ∈ directSteps,
      StepSboxHolds
        (directSource
          (projectFinalAssignment site assignment)) step := by
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonPartialLeafCertificate.partial_rows_imply_direct_reconstructed_step_sboxes
  · simpa using one
  · simpa [projected_selector, kindExact, selectorColumn] using selectorOne
  · exact absolute_rows_imply_projected_rows
      site assignment partialRows holds

theorem chained_absolute_rows_imply_step_sboxes
    (site : CallSite) (selector : Nat)
    (kindExact : site.kind = .chained selector)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment selector = 1)
    (holds : ∀ row ∈ chainedRows,
      absoluteResidual site assignment row = 0) :
    ∀ step ∈ directSteps,
      StepSboxHolds
        (chainedSource
          (projectFinalAssignment site assignment)) step := by
  apply
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonChainedLeafReconstruction.decoded_rows_imply_reconstructed_step_sboxes
  · simpa using one
  · simpa [projected_selector, kindExact, selectorColumn] using selectorOne
  · exact absolute_rows_imply_projected_rows
      site assignment chainedRows holds

/-- Retained direct-call rows force the complete independent Lean Poseidon2
reference result on the same projected assignment. -/
theorem direct_absolute_rows_compute_reference
    (site : CallSite) (kindExact : site.kind = .direct)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment directSelectorColumn = 1)
    (holds : ∀ row ∈ directRows,
      absoluteResidual site assignment row = 0)
    (lane : Fin width) :
    lcEval
        (sourcePhysical
          (directSource (projectFinalAssignment site assignment)))
        (traceFinalForm lane) =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane =>
          (sourceInput
            (directSource
              (projectFinalAssignment site assignment))
            inputLane).val)
        lane :=
  step_sboxes_compute_reference _
    (direct_absolute_rows_imply_step_sboxes
      site kindExact assignment one selectorOne holds) lane

/-- Retained partial-start rows force the complete independent Lean Poseidon2
reference result on the same projected assignment. -/
theorem partial_absolute_rows_compute_reference
    (site : CallSite) (kindExact : site.kind = .partialStart)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment partialSelectorColumn = 1)
    (holds : ∀ row ∈ partialRows,
      absoluteResidual site assignment row = 0)
    (lane : Fin width) :
    lcEval
        (sourcePhysical
          (directSource
            (projectFinalAssignment site assignment)))
        (traceFinalForm lane) =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane =>
          (sourceInput
            (directSource
              (projectFinalAssignment site assignment))
            inputLane).val)
        lane :=
  step_sboxes_compute_reference _
    (partial_absolute_rows_imply_step_sboxes
      site kindExact assignment one selectorOne holds) lane

/-- Retained chained-call rows force the complete independent Lean Poseidon2
reference result on the same projected assignment. -/
theorem chained_absolute_rows_compute_reference
    (site : CallSite) (selector : Nat)
    (kindExact : site.kind = .chained selector)
    (assignment : Fin productionFinalColumns → F)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne : absoluteValue assignment selector = 1)
    (holds : ∀ row ∈ chainedRows,
      absoluteResidual site assignment row = 0)
    (lane : Fin width) :
    lcEval
        (sourcePhysical
          (chainedSource
            (projectFinalAssignment site assignment)))
        (traceFinalForm lane) =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane =>
          (sourceInput
            (chainedSource
              (projectFinalAssignment site assignment))
            inputLane).val)
        lane :=
  step_sboxes_compute_reference _
    (chained_absolute_rows_imply_step_sboxes
      site selector kindExact assignment one selectorOne holds) lane

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallSound
