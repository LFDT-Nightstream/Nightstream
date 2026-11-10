import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Algebra
import NightstreamFPrime.Spec.Folding.Light.Rlc
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge

/-! Cyclo (ePrint 2026/359) Theorem 3, item 1, on the production algebra: a light Π_RLC adds
at most `T = 216` to each coordinate of the accumulator. -/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.LightNorm

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding

theorem act_one {logicalWidth : Nat}
    (assignment : CarrierAction.CompleteAssignment logicalWidth) :
    CarrierAction.act ringFOne assignment = assignment := by
  funext column
  unfold CarrierAction.act CarrierAction.assignmentBlock
  simp only [RingFLaws.ringFMul_one_left]
  congr 1
  exact Fin.ext (PiCCS.PaperJoint.Phi81ColumnLayout.flatIndex_decode column)

theorem norm_law {shape : Shape} {verifierRows : Nat}
    (key : Commitment.Key shape verifierRows) :
    Light.Rlc.NormLaw (Algebra.concrete key) ringFOne := by
  constructor
  intro bound challenge fresh running valid freshBounded runningBounded column
  have freshTernary : assignmentNormBounded 2 fresh := by
    simpa [relationSemantics, productionGlobalParams] using freshBounded
  have freshAct := Norm.Finite.act_coordinate_le_expansion challenge fresh valid
    freshTernary column
  have runningColumn : centeredMagnitude (running column) < bound := runningBounded column
  change centeredMagnitude
      (PiRLCFinite.combineAssignments (Light.Rlc.pair challenge ringFOne)
        (Light.Rlc.pair fresh running) column) < bound + 216 * (2 - 1)
  simp only [PiRLCFinite.combineAssignments, Light.Rlc.pair, BaseLinear.assignmentAdd,
    BaseLinear.Raw.assignmentAdd, BaseLinear.assignmentZero, BaseLinear.Raw.assignmentZero,
    Fin.val_zero, Fin.val_succ, Nat.zero_add, Nat.one_ne_zero, ↓reduceIte, act_one,
    Lean.Grind.Fin.add_zero]
  have sum := Norm.Centered.centeredMagnitude_add_le
    (CarrierAction.act challenge fresh column) (running column)
  omega

/-- The production form of `Light.Rlc.ExtractionNormLaw`: the extracted accumulator
`w − ρ · u` has each coordinate within `T = 216` of `w`. -/
theorem sub_act_bound {shape : Shape} (accumulated fresh : Assignment shape)
    (challenge : RingF) (valid : Challenge.challengeValid challenge) (bound : Nat)
    (accumulatedBounded : assignmentNormBounded bound accumulated)
    (freshTernary : assignmentNormBounded 2 fresh) :
    assignmentNormBounded (bound + 216)
      (fun column => accumulated column + -(CarrierAction.act challenge fresh column)) := by
  intro column
  have accumulatedColumn := accumulatedBounded column
  have freshAct := Norm.Finite.act_coordinate_le_expansion challenge fresh valid
    freshTernary column
  have sum := Norm.Centered.centeredMagnitude_add_le (accumulated column)
    (-(CarrierAction.act challenge fresh column))
  rw [Norm.Centered.centeredMagnitude_neg] at sum
  change centeredMagnitude
    (accumulated column + -(CarrierAction.act challenge fresh column)) < bound + 216
  omega

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.LightNorm
