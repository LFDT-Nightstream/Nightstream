import Ajtai.EstimatorModel
import SuperNeo.Commitment.LatticeReductionsDerived

/-!
Bridge from the computed protocol-binding width to SuperNeo's existing
collision-to-MSIS reduction. The concrete hardness premise remains explicit.
-/

namespace Ajtai.SecurityBoundary

open Ajtai.EstimatorModel
open Ajtai.Parameters
open SuperNeo.ProofSystem

def protocolBindingParams : AjtaiParams where
  ringDim := protocolBindingRank
  messageLength := computedMaxRingColumns
  bindingNormBound := 2
  relaxedExpansion := 1

theorem protocolBindingParams_values :
    protocolBindingParams.kappa = 2 ∧
      protocolBindingParams.msgLen = 50_371 ∧
      protocolBindingParams.bindingNormBound = 2 ∧
      protocolBindingParams.relaxedExpansion = 1 := by
  native_decide

theorem protocolBindingParams_sideConditions :
    protocolBindingParams.SideConditions := by
  change 0 < 2 ∧ 0 < computedMaxRingColumns ∧ 0 < 2 ∧ 0 < 1
  rw [computedMaxRingColumns_eq]
  decide

/-- The actual SuperNeo extractor, specialized to the computed width. -/
theorem collision_implies_msis_break
    (laws : LatticeReductionLaws protocolBindingParams) :
    Nonempty (BindingCollision protocolBindingParams) →
      MSISBreakEvent protocolBindingParams :=
  bindingCollisionEvent_implies_msisBreakEvent laws (by decide)

/-- Security-reduced claim: an explicit MSIS boundary and the existing
extractor laws imply Ajtai binding at the computed parameters. -/
theorem binding_of_msis_boundary
    (laws : LatticeReductionLaws protocolBindingParams)
    (hardness : MSISHardnessBoundary protocolBindingParams) :
    AjtaiBindingAssumption protocolBindingParams := by
  let reductions :=
    MSISToAjtaiReductions.ofLawsAndMSISBoundary
      laws (by decide) hardness
  exact ajtaiBinding_of_msis reductions hardness.hardness

end Ajtai.SecurityBoundary
