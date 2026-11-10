import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink
import NightstreamFPrime.Lifecycle.RandomOracle.Uniqueness
import NightstreamFPrime.Lifecycle.Relation
import NightstreamFPrime.Spec.Folding.PiRLC.ChainBinding
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.RelaxedBinding

/-! The part of the bounded-depth uniqueness step that the light and refresh folds share. The
extraction target is the walk's `BoundedDepthReverse.Extracted` (`Extracted`): the answer opens
the fold's input claim at `extractedBefore` of the attempt's bound (`ForkLink.SlotHolds`) and
the fresh instance under the circuit key's relation (`FreshHolds`).

A resampled valid run with the base fork index has the base run's statement, so it has the base
run's fresh instance. When it moves the start claim, `priorLink_binds` gives a collision of the
two prior states (`startChance_le_stateChance`). `ajtaiCollision` reads a relaxed binding
collision on the production relation as one on the Ajtai key's relation, which
`Binding.chainCollision_to_shortKernel` takes. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.UniquenessLink

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt Folds)

attribute [local instance low] Classical.propDecidable

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- A fresh instance holds under the relation of `key` (the walk's `FreshHolds`). -/
def FreshHolds (key : CircuitKey logicalWidth publicFits)
    (fresh : StepFresh logicalWidth publicFits)
    (witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Prop :=
  CCS.Holds (semantics ajtai) productionGlobalParams (freshStatement key.relation fresh) witness

/-- The answer opens the fold's input claim at `extractedBefore` of the attempt's bound and its
fresh instance (the walk's `WalkExtracted`). -/
abbrev Extracted (attempt : Attempt logicalWidth publicFits)
    (answer : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits) ×
      PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) : Prop :=
  BoundedDepthReverse.Extracted productionGlobalParams (ForkLink.SlotHolds ajtai) (FreshHolds ajtai)
    attempt answer

/-- A relaxed binding collision on the production relation, read on the Ajtai key's relation. -/
def ajtaiCollision {bound : Nat} {commitment : PaperAlgebra.Commitment}
    (collision : Folding.PiRLC.ChainBinding.CollisionAt (semantics ajtai) productionGlobalParams
      (Phi81Relation.PiRLCAlgebra.Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
        (rows := productionProfile.commitmentWidth)) bound commitment) :
    Folding.PiRLC.ChainBinding.CollisionAt
      (Phi81Relation.relationSemantics (Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai))
      productionGlobalParams Phi81Relation.PiRLCAlgebra.Binding.relaxedOps bound commitment where
  collision :=
    { delta₁ := collision.collision.delta₁
      delta₂ := collision.collision.delta₂
      opening₁ := collision.collision.opening₁
      opening₂ := collision.collision.opening₂
      delta₁Valid := collision.collision.delta₁Valid
      delta₂Valid := collision.collision.delta₂Valid
      firstEquation := collision.collision.firstEquation
      secondEquation := collision.collision.secondEquation
      firstNorm := collision.collision.firstNorm
      secondNorm := collision.collision.secondNorm
      crossDifferent := collision.collision.crossDifferent }
  firstNorm := collision.firstNorm
  secondNorm := collision.secondNorm

/-- An opening at a bound opens at every larger bound. -/
theorem holds_mono {claim : CE.Instance (PaperAlgebra.Structure logicalWidth)
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      PaperAlgebra.Point PaperAlgebra.Evaluation PaperAlgebra.Commitment}
    {witness : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {bound bound' : Nat}
    (holds : Spec.Folding.Light.Rlc.Holds (semantics ajtai) bound claim witness)
    (larger : bound ≤ bound') :
    Spec.Folding.Light.Rlc.Holds (semantics ajtai) bound' claim witness :=
  ⟨⟨holds.1.1, holds.1.2.1, fun column => Nat.lt_of_lt_of_le (holds.1.2.2 column) larger⟩,
    holds.2⟩

section Game

variable {schedule : Schedule} (key : CircuitKey logicalWidth publicFits)
  (folds : (Point schedule 8 → Answer) → Folds logicalWidth publicFits)
  (statement : Attempt logicalWidth publicFits → schedule.Statement)
  (messages : Attempt logicalWidth publicFits → schedule.Messages 8)
  (adversary : OracleComp (Point schedule 8) Answer (Attempt logicalWidth publicFits))

/-- A successful attempt's bound is within the chain bound. -/
theorem bound_le_chain {oracle : Point schedule 8 → Answer}
    (succeeds : Fork.Succeeds (ForkLink.game ajtai key folds statement messages) adversary
      oracle) :
    (adversary.run oracle).bound ≤
      BoundedDepthReverse.chainBound productionGlobalParams
        Folding.Schedule.productionLightFoldCap :=
  succeeds.2.2.2.2.2

/-- The chance that a resampled valid run gives a collision of the base and resampled prior
states. -/
noncomputable def stateChance : ℝ :=
  Uniqueness.reductionChance (ForkLink.game ajtai key folds statement messages) adversary
    fun oracle _ other _ =>
      StateCollision (toState (adversary.run oracle).link.prior)
        (toState (adversary.run other).link.prior)

variable {statement}

/-- A resampled valid run that moves the start claim gives a prior-state collision, when the
statement fixes the fresh instance. -/
theorem startChance_le_stateChance
    (freshOf : ∀ {attempt attempt'}, statement attempt = statement attempt' →
      attempt.link.fold.fresh = attempt'.link.fold.fresh) :
    Uniqueness.startChance (ForkLink.game ajtai key folds statement messages) adversary ≤
      stateChance ajtai key folds statement messages adversary := by
  apply Uniqueness.reductionChance_mono
  intro oracle _ valid sample _ forked otherValid moved
  have sameStatement := Uniqueness.statement_eq_of_fork _ adversary oracle sample forked
  rcases ForkLink.start_eq_or_collision ajtai valid.1.2 otherValid.1.2
      (freshOf sameStatement.symm) with same | collision
  · exact absurd same.symm moved
  · exact collision

end Game

end NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.UniquenessLink
