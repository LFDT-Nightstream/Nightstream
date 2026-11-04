import NightstreamFPrime.Spec.Folding.Light.Binding
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.MinimalAxioms

/-! SuperNeo v1.2 Definition 7 for the running source of the light Π_RLC. The running source has
coefficient 1, so its extracted witness is `r = w_b − ρ_b · u`, with `u` the extracted fresh
witness. Two light forks on one fresh and one running commitment that extract one `u` and
different `r` give a relaxed binding collision on the fresh commitment, with `δ₁ = ρ_b − ρ_f` of
the first fork and `δ₂ = ρ_b − ρ'_b`, the difference of the two base challenges
(`runningCollisionAbove`). Its openings are differences of fork outputs, so they open at twice
the output bound, as in `collisionAbove`, and `Binding.collisionAt_to_shortKernel` gives a
kernel vector below the chain norm. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light.Rlc

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (ExtractionAlgebra StrongSetUnits)
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding (Compatible)
open NightstreamFPrime.Spec.Folding.PiRLC.ChainBinding (CollisionAt)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar uValue

section Group

variable {Scalar : Type uScalar} {Value : Type uValue} {ring : CommutativeRingOps Scalar}
  {module : ModuleOps Scalar Value}

/-- The additive group that `ModuleLaws` describes. -/
abbrev groupOf (laws : ModuleLaws ring module) : AddCommGroup Value :=
  let _ : Add Value := ⟨module.add⟩
  let _ : Zero Value := ⟨module.zero⟩
  let _ : Neg Value := ⟨module.neg⟩
  { AddGroup.ofLeftAxioms laws.add_assoc laws.zero_add
      (fun value => (laws.add_comm _ _).trans (laws.add_neg value)) with
    add_comm := laws.add_comm }

theorem smul_sub_value (laws : ModuleLaws ring module)
    (scalar : Scalar) (left right : Value) :
    module.smul scalar (module.sub left right) =
      module.sub (module.smul scalar left) (module.smul scalar right) := by
  let _ := groupOf laws
  have negated : module.smul scalar (module.neg right) = module.neg (module.smul scalar right) := by
    refine eq_neg_of_add_eq_zero_left ?_
    change module.add _ _ = module.zero
    rw [← laws.smul_add]
    change module.smul scalar (-right + right) = _
    rw [neg_add_cancel]
    exact laws.smul_zero scalar
  unfold ModuleOps.sub
  rw [laws.smul_add, negated]

theorem sub_smul_value (ringLaws : CommutativeRingLaws ring) (laws : ModuleLaws ring module)
    (left right : Scalar) (value : Value) :
    module.smul (ring.sub left right) value =
      module.sub (module.smul left value) (module.smul right value) := by
  let _ := groupOf laws
  have negated : module.smul (ring.neg right) value = module.neg (module.smul right value) := by
    refine eq_neg_of_add_eq_zero_left ?_
    change module.add _ _ = module.zero
    rw [← laws.add_smul, ringLaws.add_comm, ringLaws.add_neg]
    exact laws.zero_smul value
  unfold CommutativeRingOps.sub ModuleOps.sub
  rw [laws.add_smul, negated]

end Group

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams}
  {algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
    semantics params}
  {one : Scalar} {bound : Nat} {system system' : Structure} {point point' : Point}
  {leftFresh leftRunning rightFresh rightRunning :
    CE.Instance Structure PublicInput Point Evaluation Commitment}

/-- Two light forks that extract one fresh witness, and whose base outputs differ by `δ₂` times
the first fork's output difference scaled by `δ₁⁻¹`, extract one running witness. -/
theorem running_eq_of_cross (laws : ExtractionAlgebra semantics params algebra)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (left : Fork algebra one bound system point leftFresh leftRunning)
    (right : Fork algebra one bound system' point' rightFresh rightRunning)
    (sameFresh : left.freshAssignment laws strongSet = right.freshAssignment laws strongSet)
    (cross : laws.assignmentModule.smul (laws.ring.sub left.baseChallenge left.forkChallenge)
        (laws.assignmentModule.sub left.baseAssignment right.baseAssignment) =
      laws.assignmentModule.smul (laws.ring.sub left.baseChallenge right.baseChallenge)
        (laws.assignmentModule.sub left.baseAssignment left.forkAssignment)) :
    left.runningAssignment laws strongSet = right.runningAssignment laws strongSet := by
  let _ := groupOf laws.assignmentLaws
  let module := laws.assignmentModule
  let unit := left.unit laws strongSet
  let fresh := left.freshAssignment laws strongSet
  have scaled : module.smul (laws.ring.sub left.baseChallenge left.forkChallenge) fresh =
      module.sub left.baseAssignment left.forkAssignment := by
    change module.smul _ (module.smul unit.inverse _) = _
    rw [← laws.assignmentLaws.mul_smul, unit.mul_inverse, laws.assignmentLaws.one_smul]
  have outputs : module.sub left.baseAssignment right.baseAssignment =
      module.smul (laws.ring.sub left.baseChallenge right.baseChallenge) fresh := by
    rw [← scaled, ← laws.assignmentLaws.mul_smul, laws.ringLaws.mul_comm,
      laws.assignmentLaws.mul_smul] at cross
    have cancelled := congrArg (module.smul unit.inverse) cross
    rwa [inverseActionCancellation _ _ laws.assignmentLaws _ unit,
      inverseActionCancellation _ _ laws.assignmentLaws _ unit] at cancelled
  rw [sub_smul_value laws.ringLaws laws.assignmentLaws] at outputs
  change module.sub left.baseAssignment (module.smul left.baseChallenge fresh) =
    module.sub right.baseAssignment
      (module.smul right.baseChallenge (right.freshAssignment laws strongSet))
  rw [← sameFresh]
  change left.baseAssignment - module.smul left.baseChallenge fresh =
    right.baseAssignment - module.smul right.baseChallenge fresh
  change left.baseAssignment - right.baseAssignment =
    module.smul left.baseChallenge fresh - module.smul right.baseChallenge fresh at outputs
  exact sub_eq_sub_iff_sub_eq_sub.mpr outputs

theorem base_difference_commits (laws : ExtractionAlgebra semantics params algebra)
    (oneEq : one = laws.ring.one)
    (left : Fork algebra one bound system point leftFresh leftRunning)
    (right : Fork algebra one bound system' point' rightFresh rightRunning)
    (sameFreshCommitment : leftFresh.commitment = rightFresh.commitment)
    (sameRunningCommitment : leftRunning.commitment = rightRunning.commitment) :
    semantics.commit (laws.assignmentModule.sub left.baseAssignment right.baseAssignment) =
      laws.commitmentModule.smul (laws.ring.sub left.baseChallenge right.baseChallenge)
        leftFresh.commitment := by
  subst oneEq
  have rightBase := right.baseHolds.1.1.trans (laws.combineCommitment_eq _ _)
  change semantics.commit right.baseAssignment = linearCombination laws.ring laws.commitmentModule
    (pair right.baseChallenge laws.ring.one)
    (pair rightFresh.commitment rightRunning.commitment) at rightBase
  rw [← sameFreshCommitment, ← sameRunningCommitment] at rightBase
  rw [laws.commitMap.map_sub, left.baseHolds.1.1.trans (laws.combineCommitment_eq _ _), rightBase]
  exact coordinateIsolation laws.ring laws.commitmentModule laws.ringLaws laws.commitmentLaws
    _ _ _ (0 : Fin 2) (agree_pair _ _ _)

/-- Two light forks on one fresh and one running commitment that extract one fresh witness and
different running witnesses give a relaxed binding collision on the fresh commitment, with
openings at twice the output bound. -/
def runningCollisionAbove (laws : ExtractionAlgebra semantics params algebra)
    (ops : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    (compatible : Compatible laws ops)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (oneEq : one = laws.ring.one)
    (differenceBounded : ∀ left right, semantics.normBounded bound left →
      semantics.normBounded bound right →
      semantics.normBounded (2 * bound) (laws.assignmentModule.sub left right))
    (left : Fork algebra one bound system point leftFresh leftRunning)
    (right : Fork algebra one bound system' point' rightFresh rightRunning)
    (sameFreshCommitment : leftFresh.commitment = rightFresh.commitment)
    (sameRunningCommitment : leftRunning.commitment = rightRunning.commitment)
    (sameFresh : left.freshAssignment laws strongSet = right.freshAssignment laws strongSet)
    (different : left.runningAssignment laws strongSet ≠ right.runningAssignment laws strongSet) :
    CollisionAt semantics params ops (2 * bound) leftFresh.commitment where
  collision :=
    { delta₁ := laws.ring.sub left.baseChallenge left.forkChallenge
      delta₂ := laws.ring.sub left.baseChallenge right.baseChallenge
      opening₁ := laws.assignmentModule.sub left.baseAssignment left.forkAssignment
      opening₂ := laws.assignmentModule.sub left.baseAssignment right.baseAssignment
      delta₁Valid := compatible.differenceValid _ _ left.baseValid left.forkValid
      delta₂Valid := compatible.differenceValid _ _ left.baseValid right.baseValid
      firstEquation := by
        rw [compatible.commitmentAction]
        exact (difference_commits laws oneEq left).symm
      secondEquation := by
        rw [compatible.commitmentAction]
        exact (base_difference_commits laws oneEq left right sameFreshCommitment
          sameRunningCommitment).symm
      firstNorm := trivial
      secondNorm := trivial
      crossDifferent := by
        rw [compatible.assignmentAction]
        exact fun cross => different (running_eq_of_cross laws strongSet left right sameFresh cross) }
  firstNorm := differenceBounded _ _ left.baseHolds.1.2.2 left.forkHolds.1.2.2
  secondNorm := differenceBounded _ _ left.baseHolds.1.2.2 right.baseHolds.1.2.2

end NightstreamFPrime.Spec.Folding.Light.Rlc
