import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepPlan
import Nightstream.Implementation.Lowering.Goldilocks.ReceiptSatisfaction

/-!
Contract: row evidence interface for the canonical fixed-one Step proof.

Assurance tier: model-level.

Owns:
- the common evidence accepted by legacy R1CS and native CCS Step soundness;
- the adapter from the legacy physical R1CS encoding.

Does not own: Step semantic reconstruction, native CCS receipt replacement,
honest assignment construction, or manifests.

Emits constraints: none.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalStepSoundness

def encoding
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    Goldilocks.Encoding
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.program
        parameters) :=
  (CanonicalStepPlan.physical
    parameters profile recipes defaultAdmissible).toEncoding

def recursiveNifsOwner : PhysicalOwner :=
  .typed (.instruction SourceOwners.stepRecursiveNifsPath)

/-- The row evidence used by the Step semantic proof.

The selected NIFS occurrence has a separate semantic field because a native
CCS selector does not emit the legacy activated-R1CS rows. Every other
receipt remains an ordinary R1CS receipt. -/
structure Evidence
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (assignment : ColumnId → Field) : Prop where
  constantOne : assignment oneColumn = 1
  ordinaryRows :
    ∀ receipt,
      receipt ∈
          CanonicalStepPlan.receipts
            parameters profile recipes defaultAdmissible →
        receipt.owner ≠ recursiveNifsOwner →
          Satisfies receipt.rows assignment
  recursiveNifs :
    ∀ (inputs :
        HVec (typeSystem parameters).Value
          ((signature parameters).callInputs Call.nifsVerify)),
      assignment
          (CanonicalStepPlan.recursiveNifsInvokePlan
            parameters profile recipes).frame.one = 1 →
      assignment
          (CanonicalStepPlan.recursiveNifsInvokePlan
            parameters profile recipes).frame.active = 1 →
      (CanonicalStepPlan.recursiveNifsInvokePlan
          parameters profile recipes).frame.operands.Decodes
        (profile.family parameters) assignment inputs →
      ∃ outputs :
          Schema.Values (typeSystem parameters)
            ((signature parameters).callOutputs Call.nifsVerify),
        (signature parameters).callEval Call.nifsVerify inputs =
            some outputs ∧
          (CanonicalStepPlan.recursiveNifsInvokePlan
            parameters profile recipes).frame.outputs.Decodes
              (profile.family parameters) assignment outputs

private theorem recursiveNifsReceiptMember
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    (CanonicalStepPlan.recursiveNifsPlan.{0}
        parameters profile recipes).receipt ∈
      CanonicalStepPlan.receipts
        parameters profile recipes defaultAdmissible := by
  simp [CanonicalStepPlan.receipts, CanonicalStepPlan.bodyReceipts]

/-- Adapt the legacy all-R1CS physical program to the shared Step evidence. -/
def evidenceOfPhysical
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (assignment : ColumnId → Field)
    (physical :
      (encoding parameters profile recipes defaultAdmissible
        ).PhysicalSatisfies assignment) :
    Evidence parameters profile recipes defaultAdmissible assignment where
  constantOne := by
    simpa [encoding, CanonicalStepPlan.physical] using physical.1
  ordinaryRows := by
    intro receipt member _
    exact
      (encoding parameters profile recipes defaultAdmissible
        ).receiptSatisfies assignment physical receipt
          (by simpa [encoding, CanonicalStepPlan.physical] using member)
  recursiveNifs := by
    intro inputs constantOne activeOne decoded
    let invokePlan :=
      CanonicalStepPlan.recursiveNifsInvokePlan
        parameters profile recipes
    have rows :
        Satisfies
          (invokePlan.recipe.rows invokePlan.frame) assignment := by
      exact
        (encoding parameters profile recipes defaultAdmissible
          ).receiptSatisfies assignment physical
            (CanonicalStepPlan.recursiveNifsPlan.{0}
              parameters profile recipes).receipt
            (by
              simpa [encoding, CanonicalStepPlan.physical] using
                recursiveNifsReceiptMember
                  parameters profile recipes defaultAdmissible)
    exact
      invokePlan.recipe.activeSoundness invokePlan.frame assignment inputs
        constantOne activeOne decoded rows

end CanonicalStepSoundness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
