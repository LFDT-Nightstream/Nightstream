import NightstreamFPrime.Layout.Stage1.Wide.PiDECInputs
import NightstreamFPrime.Layout.Stage1.PiDECInputBounds
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.OutputBindingTransport

/-! All wide PiDEC inputs precede its local allocation. These bounds derive
the gadget assumptions from the fixed layout and add no constraint row. -/

namespace NightstreamFPrime.Layout.Stage1.Wide.PiDECInputs

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint
open Stage1.PiDECInputs (combinationOutput_varsBelow)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

theorem parentBelow (_relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiRLC.v1_1.OutputBinding.InputsBelow (piRlcOutputInterface logicalWidth publicFits)
      PiRLCStarts.outputLogicalStart PiRLCStarts.phaseFreshStart := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro coordinate
    change ((Stage1.RunningTransitionInputs.recursiveRunningExpr logicalWidth publicFits).point coordinate).VarsBelow PiRLCStarts.phaseFreshStart
    rw [Stage1.RunningTransitionInputs.recursivePoint_eq_direct, PiCCSStarts.roundTranscriptWitnessStart_eq]
    have bound : coordinate.val < 28 := coordinate.isLt
    change (15027676 + coordinate.val * 5328 + 4136 < 19620846) ∧
      (15027676 + coordinate.val * 5328 + 4728 < 19620846)
    constructor <;> omega
  · intro row lane
    exact Expr.VarsBelow.mono _ (combinationOutput_varsBelow
      (PiRLC.v1_1.CommitmentCombination.familyInterface
        (PiRLC.Wide.Formal.commitmentInterface (piRlcSharedInterface logicalWidth publicFits)))
      PiRLCStarts.commitmentLogicalStart row lane PiRLC.v1_1.CommitmentCombination.cell) (by decide)
  · intro column
    exact Expr.VarsBelow.mono _ (combinationOutput_varsBelow
      (PiRLC.v1_1.PublicInputCombination.familyInterface
        (PiRLC.Wide.Formal.publicInputInterface (piRlcSharedInterface logicalWidth publicFits)))
      PiRLCStarts.publicInputLogicalStart
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex (FullShape logicalWidth publicFits) column)
      (Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex column) PiRLC.v1_1.PublicInputCombination.cell) (by decide)
  · intro coefficient
    constructor
    · exact Expr.VarsBelow.mono _ (combinationOutput_varsBelow
        (PiRLC.v1_1.RingKCombination.familyInterface
          (PiRLC.v1_1.EvalKCombination.ringInterface (PiRLC.Wide.Formal.evalKInterface (piRlcSharedInterface logicalWidth publicFits))))
        PiRLCStarts.evalKLogicalStart PiRLC.v1_1.EvalKCombination.block
        (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient) PiRLC.v1_1.RingKCombination.c0Cell) (by decide)
    · exact Expr.VarsBelow.mono _ (combinationOutput_varsBelow
        (PiRLC.v1_1.RingKCombination.familyInterface
          (PiRLC.v1_1.EvalKCombination.ringInterface (PiRLC.Wide.Formal.evalKInterface (piRlcSharedInterface logicalWidth publicFits))))
        PiRLCStarts.evalKLogicalStart PiRLC.v1_1.EvalKCombination.block
        (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient) PiRLC.v1_1.RingKCombination.c1Cell) (by decide)
  · intro matrix coefficient
    constructor
    · exact Expr.VarsBelow.mono _ (combinationOutput_varsBelow
        (PiRLC.v1_1.RingKCombination.familyInterface
          (PiRLC.v1_1.EvalACombination.ringInterface (PiRLC.Wide.Formal.evalAInterface (piRlcSharedInterface logicalWidth publicFits))))
        PiRLCStarts.evalALogicalStart matrix
        (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient) PiRLC.v1_1.RingKCombination.c0Cell) (by decide)
    · exact Expr.VarsBelow.mono _ (combinationOutput_varsBelow
        (PiRLC.v1_1.RingKCombination.familyInterface
          (PiRLC.v1_1.EvalACombination.ringInterface (PiRLC.Wide.Formal.evalAInterface (piRlcSharedInterface logicalWidth publicFits))))
        PiRLCStarts.evalALogicalStart matrix
        (Fin.cast PiRLC.v1_1.EvalKCombination.coefficientCount_eq coefficient) PiRLC.v1_1.RingKCombination.c1Cell) (by decide)

theorem parentEnd_le_proofInputStart : PiRLCStarts.phaseFreshStart ≤ proofInputStart := by
  change PiRLCStarts.phaseFreshStart ≤ PiRLCStarts.outputFreshStart
  unfold PiRLCStarts.outputFreshStart PiRLCStarts.evalAFreshStart PiRLCStarts.evalKFreshStart
    PiRLCStarts.publicInputFreshStart PiRLCStarts.commitmentFreshStart PiRLCStarts.samplerFreshStart
  omega

theorem inputsBelow (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    PiDEC.v1_1.Formal.InputsBelow (interface logicalWidth publicFits) phaseOffset := by
  have parent := parentBelow relation
  have expands : PiRLCStarts.phaseFreshStart ≤ phaseOffset := parentEnd_le_proofInputStart.trans (Nat.le_add_right _ _)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro index
    exact Quadratic.KExpr.varsBelow_mono _ (parent.point index) expands
  · intro row lane
    exact Expr.VarsBelow.mono _ (parent.commitment row lane) expands
  · intro column
    exact Expr.VarsBelow.mono _ (parent.publicInput column) expands
  · intro coefficient
    exact Quadratic.KExpr.varsBelow_mono _ (parent.eval_K coefficient) expands
  · intro matrix coefficient
    exact Quadratic.KExpr.varsBelow_mono _ (parent.eval_A matrix coefficient) expands
  · intro child row lane
    have hc : child.val < 16 := child.isLt
    have hr : row.val < 22 := row.isLt
    have hl : lane.val < 54 := lane.isLt
    change 27496062 + child.val * 1188 + row.val * 54 + lane.val < 27545310
    omega
  · intro child coefficient
    have hc : child.val < 16 := child.isLt
    have hi : coefficient.val < 54 := coefficient.isLt
    change (27515070 + child.val * 108 + coefficient.val * 2 < 27545310) ∧
      (27515070 + child.val * 108 + coefficient.val * 2 + 1 < 27545310)
    constructor <;> omega
  · intro child matrix coefficient
    have hc : child.val < 16 := child.isLt
    have hm : matrix.val < 14 := matrix.isLt
    have hi : coefficient.val < 54 := coefficient.isLt
    change (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 < 27545310) ∧
      (27516798 + child.val * 1512 + matrix.val * 108 + coefficient.val * 2 + 1 < 27545310)
    constructor <;> omega
  · intro child coordinate
    have hc : child.val < 16 := child.isLt
    have hi : coordinate.val < 270 := coordinate.isLt
    change 27540990 + child.val * 270 + coordinate.val < 27545310
    omega

theorem assumptions (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (env : Env) :
    PiDEC.v1_1.Formal.Assumptions relation (interface logicalWidth publicFits) phaseOffset env :=
  ⟨inputsBelow relation⟩

end NightstreamFPrime.Layout.Stage1.Wide.PiDECInputs
