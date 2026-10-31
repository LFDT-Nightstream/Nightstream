import NightstreamFPrime.Spec.Folding.Light.Extraction
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding

/-! SuperNeo v1.2 Definition 7 for the light Π_RLC. Two light forks on one fresh commitment
whose extracted fresh witnesses differ give SuperNeo's relaxed binding collision. When both
outputs are bounded by `B`, the openings are bounded by `2B`, so the MSIS norm `8TB` of
SuperNeo v1.2 Section 8 applies unchanged. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light.Rlc

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction (ExtractionAlgebra StrongSetUnits)
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding (Compatible)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams}
  {algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
    semantics params}
  {one : Scalar} {bound : Nat} {system : Structure} {point : Point}

theorem difference_commits
    {fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment}
    (laws : ExtractionAlgebra semantics params algebra) (oneEq : one = laws.ring.one)
    (fork : Fork algebra one bound system point fresh running) :
    semantics.commit (laws.assignmentModule.sub fork.baseAssignment fork.forkAssignment) =
      laws.commitmentModule.smul (laws.ring.sub fork.baseChallenge fork.forkChallenge)
        fresh.commitment := by
  subst oneEq
  rw [laws.commitMap.map_sub, fork.baseHolds.1.1.trans (laws.combineCommitment_eq _ _),
    fork.forkHolds.1.1.trans (laws.combineCommitment_eq _ _)]
  exact coordinateIsolation laws.ring laws.commitmentModule laws.ringLaws laws.commitmentLaws
    _ _ _ (0 : Fin 2) (agree_pair _ _ _)

private theorem inverse_cross_eq (laws : ExtractionAlgebra semantics params algebra)
    (delta₁ delta₂ : Scalar)
    (unit₁ : UnitWitness laws.ring delta₁) (unit₂ : UnitWitness laws.ring delta₂)
    (opening₁ opening₂ : Assignment)
    (cross : laws.assignmentModule.smul delta₁ opening₂ =
      laws.assignmentModule.smul delta₂ opening₁) :
    laws.assignmentModule.smul unit₁.inverse opening₁ =
      laws.assignmentModule.smul unit₂.inverse opening₂ := by
  have swap (first second : Scalar) (value : Assignment) :
      laws.assignmentModule.smul first (laws.assignmentModule.smul second value) =
        laws.assignmentModule.smul second (laws.assignmentModule.smul first value) := by
    rw [← laws.assignmentLaws.mul_smul, laws.ringLaws.mul_comm, laws.assignmentLaws.mul_smul]
  calc
    _ = laws.assignmentModule.smul unit₁.inverse
        (laws.assignmentModule.smul unit₂.inverse
          (laws.assignmentModule.smul delta₂ opening₁)) := by
      rw [inverseActionCancellation laws.ring laws.assignmentModule laws.assignmentLaws
        delta₂ unit₂]
    _ = laws.assignmentModule.smul unit₂.inverse
        (laws.assignmentModule.smul unit₁.inverse
          (laws.assignmentModule.smul delta₂ opening₁)) := swap _ _ _
    _ = laws.assignmentModule.smul unit₂.inverse
        (laws.assignmentModule.smul unit₁.inverse
          (laws.assignmentModule.smul delta₁ opening₂)) := by rw [cross]
    _ = _ := by
      rw [inverseActionCancellation laws.ring laws.assignmentModule laws.assignmentLaws
        delta₁ unit₁]

/-- SuperNeo v1.2 Definition 7: two light forks on one fresh commitment that extract
different fresh witnesses. -/
def collisionAt
    {leftFresh leftRunning rightFresh rightRunning :
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (laws : ExtractionAlgebra semantics params algebra)
    (ops : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    (compatible : Compatible laws ops)
    (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
    (oneEq : one = laws.ring.one)
    (withinB : ∀ assignment, semantics.normBounded bound assignment →
      semantics.normBounded params.bigB assignment)
    (left : Fork algebra one bound system point leftFresh leftRunning)
    (right : Fork algebra one bound system point rightFresh rightRunning)
    (sameFresh : leftFresh.commitment = rightFresh.commitment)
    (different : left.freshAssignment laws strongSet ≠ right.freshAssignment laws strongSet) :
    PiRLC.RelaxedBindingCollision semantics params ops leftFresh.commitment where
  delta₁ := laws.ring.sub left.baseChallenge left.forkChallenge
  delta₂ := laws.ring.sub right.baseChallenge right.forkChallenge
  opening₁ := laws.assignmentModule.sub left.baseAssignment left.forkAssignment
  opening₂ := laws.assignmentModule.sub right.baseAssignment right.forkAssignment
  delta₁Valid := compatible.differenceValid _ _ left.baseValid left.forkValid
  delta₂Valid := compatible.differenceValid _ _ right.baseValid right.forkValid
  firstEquation := by
    rw [compatible.commitmentAction]
    exact (difference_commits laws oneEq left).symm
  secondEquation := by
    rw [compatible.commitmentAction, sameFresh]
    exact (difference_commits laws oneEq right).symm
  firstNorm := compatible.differenceBounded _ _
    (withinB _ left.baseHolds.1.2.2) (withinB _ left.forkHolds.1.2.2)
  secondNorm := compatible.differenceBounded _ _
    (withinB _ right.baseHolds.1.2.2) (withinB _ right.forkHolds.1.2.2)
  crossDifferent := by
    intro cross
    apply different
    rw [compatible.assignmentAction] at cross
    exact inverse_cross_eq laws _ _ (left.unit laws strongSet) (right.unit laws strongSet)
      _ _ cross

end NightstreamFPrime.Spec.Folding.Light.Rlc
