import NightstreamFPrime.Lifecycle.OptionB.LightRlcLayout
import NightstreamFPrime.Lifecycle.OptionB.LightCommitment

/-! Input scoping of the light Π_RLC layout (assembler A2b): when the sampler's start state, the
Π_CCS output words, the slot claim's words and the fresh instance's words lie below the parent
offset, `LightRlcLayout.Assumptions` holds (`assumptions`). The combinations read `ρ` from the
sampler, which ends where the commitment combination starts. -/

namespace NightstreamFPrime.Layout.OptionB.LightRlcBounds

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Lifecycle.OptionB.LightRlcLayout

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The light Π_RLC layout's inputs lie below the parent offset. -/
structure InputsBelow (piccs : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
    (interface : LightRlcLayout.Interface logicalWidth publicFits) (offset : Nat) : Prop where
  samplerStart : ∀ lane,
    (OutputBinding.finalState piccs.output piccs.outputOffset lane).VarsBelow offset
  outputPad : ∀ source coefficient,
    ((piccs.output.output piccs.outputOffset).padCoordinate source coefficient).VarsBelow offset
  outputMatrix : ∀ source matrix coefficient,
    ((piccs.output.output piccs.outputOffset).matrixCoordinate source matrix
      coefficient).VarsBelow offset
  runningCommitment : ∀ row lane, (interface.runningCommitment offset row lane).VarsBelow offset
  runningPublicInput : ∀ column, (interface.runningPublicInput offset column).VarsBelow offset
  freshCommitment : ∀ source row lane,
    ((interface.fresh offset).commitment source row lane).VarsBelow offset
  freshPublicInput : ∀ source column,
    ((interface.fresh offset).publicInput source column).VarsBelow offset

variable (piccs : LightPiCCSPhase.Leaves logicalWidth degreeBound publicFits)
  (interface : LightRlcLayout.Interface logicalWidth publicFits) (offset : Nat)

theorem commitmentOffset_eq :
    commitmentOffset piccs offset = offset + Sampler.logicalPrivateCount := by
  unfold commitmentOffset Formal.nextOffset Formal.childLength samplerCircuit
  change offset + localLength (Sampler.opsAt _ 0 offset) = _
  rw [Sampler.localLength_eq]

theorem commitment_le : offset ≤ commitmentOffset piccs offset :=
  Nat.le_add_right _ _

theorem publicInput_le :
    commitmentOffset piccs offset ≤ publicInputOffset piccs interface offset :=
  Nat.le_add_right _ _

theorem evaluation_le :
    publicInputOffset piccs interface offset ≤ evaluationOffset piccs interface offset :=
  Nat.le_add_right _ _

theorem challenge_below (bound : Nat) (le : commitmentOffset piccs offset ≤ bound)
    (position : Fin ringDegree) : (Sampler.outputChallenge offset position).VarsBelow bound :=
  Expr.VarsBelow.mono _ (Sampler.outputChallenge_below offset position)
    (by rw [commitmentOffset_eq] at le; exact le)

variable (below : InputsBelow piccs interface offset) (env : Env)

include below

theorem family_below (bound : Nat) (le : offset ≤ bound) (source : Fin LightShape.sourceCount)
    (block : Fin LightEvaluation.blockCount) (lane : Fin ringDegree)
    (cell : Fin LightEvaluation.cellCount) :
    (RingKCombination.expressionCell cell
      (familyExpr (piccs.output.output piccs.outputOffset) source block lane)).VarsBelow bound := by
  have word : (familyExpr (piccs.output.output piccs.outputOffset) source block lane).VarsBelow
      bound := by
    unfold familyExpr
    split
    · exact KExpr.varsBelow_mono _ (below.outputPad source lane) le
    · exact KExpr.varsBelow_mono _ (below.outputMatrix source _ lane) le
  unfold RingKCombination.expressionCell
  split
  · exact word.1
  · exact word.2

/-- Input scoping of the light Π_RLC layout. -/
theorem assumptions : LightRlcLayout.Assumptions piccs interface offset env := by
  have publicInputLe :=
    Nat.le_trans (commitment_le piccs offset) (publicInput_le piccs interface offset)
  have evaluationLe := Nat.le_trans publicInputLe (evaluation_le piccs interface offset)
  refine ⟨below.samplerStart, ⟨fun lane => ?_, fun row lane _ => ?_, fun row lane _ => ?_⟩,
    ⟨fun lane => ?_, fun block lane _ => ?_, fun block lane _ => ?_⟩,
    ⟨fun lane => ?_, fun block lane cell => ?_, fun block lane cell => ?_⟩⟩
  · exact challenge_below piccs offset _ (Nat.le_refl _) lane
  · exact Expr.VarsBelow.mono _ (below.runningCommitment row lane) (commitment_le piccs offset)
  · exact Expr.VarsBelow.mono _ (below.freshCommitment _ row lane) (commitment_le piccs offset)
  · exact challenge_below piccs offset _ (publicInput_le piccs interface offset) lane
  · exact Expr.VarsBelow.mono _ (below.runningPublicInput _) publicInputLe
  · exact Expr.VarsBelow.mono _ (below.freshPublicInput _ _) publicInputLe
  · exact challenge_below piccs offset _ (Nat.le_trans (publicInput_le piccs interface offset)
      (evaluation_le piccs interface offset)) lane
  · exact family_below piccs interface offset below _ evaluationLe _ _ _ _
  · exact family_below piccs interface offset below _ evaluationLe _ _ _ _

end NightstreamFPrime.Layout.OptionB.LightRlcBounds
