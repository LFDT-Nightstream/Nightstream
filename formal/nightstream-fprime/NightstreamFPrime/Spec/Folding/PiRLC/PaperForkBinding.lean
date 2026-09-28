import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction

/-!
Relaxed-binding collisions from two actual coordinate forks. The openings
are the differences of the B-bounded response assignments, not arbitrary
ambient openings. Their challenge differences and commitment equations are
retained in the constructed collision. Hardness is not a premise here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding

open NightstreamFPrime.Spec
open PaperForkAlgebra PaperForkExtraction

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  {algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params}
  (laws : ExtractionAlgebra semantics params algebra)
  (ops : RelaxedBindingOps Assignment Commitment Scalar)

/-- Only local operation agreement and the subtraction norm bound are needed.
No uniqueness or collision conclusion is a field of this record. -/
structure Compatible : Prop where
  differenceValid : ∀ left right, algebra.challengeValid left → algebra.challengeValid right →
    ops.differenceChallenge (laws.ring.sub left right)
  assignmentAction : ops.scaleAssignment = laws.assignmentModule.smul
  commitmentAction : ops.scaleCommitment = laws.commitmentModule.smul
  differenceBounded : ∀ left right,
    semantics.normBounded params.bigB left → semantics.normBounded params.bigB right →
    semantics.normBounded (2 * params.bigB) (laws.assignmentModule.sub left right)

variable (compatible : Compatible laws ops)
  (strongSet : StrongSetUnits laws.ring algebra.challengeValid)
  (leftBatch rightBatch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)
  (left : CompleteFork semantics params algebra leftBatch)
  (right : CompleteFork semantics params algebra rightBatch)

def coordinateDelta {batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity}
    (fork : CompleteFork semantics params algebra batch) (coordinate : Fin arity.total) : Scalar :=
  laws.ring.sub (fork.base.challenges coordinate) ((fork.forks coordinate).challenges coordinate)

def coordinateDifference {batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity}
    (fork : CompleteFork semantics params algebra batch) (coordinate : Fin arity.total) : Assignment :=
  laws.assignmentModule.sub fork.base.assignment (fork.forks coordinate).assignment

private theorem difference_commits
    {batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity}
    (fork : CompleteFork semantics params algebra batch) (coordinate : Fin arity.total) :
    semantics.commit (coordinateDifference laws fork coordinate) =
      laws.commitmentModule.smul (coordinateDelta laws fork coordinate)
        (batch.inputs coordinate).commitment := by
  have baseCommitment : semantics.commit fork.base.assignment =
      linearCombination laws.ring laws.commitmentModule fork.base.challenges
        (fun index => (batch.inputs index).commitment) :=
    fork.baseSuccess.1.1.trans (laws.combineCommitment_eq _ _)
  have forkCommitment : semantics.commit (fork.forks coordinate).assignment =
      linearCombination laws.ring laws.commitmentModule (fork.forks coordinate).challenges
        (fun index => (batch.inputs index).commitment) :=
    (fork.forkSuccess coordinate).1.1.trans (laws.combineCommitment_eq _ _)
  unfold coordinateDifference coordinateDelta
  rw [laws.commitMap.map_sub, baseCommitment, forkCommitment]
  exact coordinateIsolation laws.ring laws.commitmentModule laws.ringLaws laws.commitmentLaws
    fork.base.challenges (fork.forks coordinate).challenges _ coordinate (fork.agreeExcept coordinate)

private theorem inverse_cross_eq (delta₁ delta₂ : Scalar)
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
        (laws.assignmentModule.smul unit₂.inverse (laws.assignmentModule.smul delta₂ opening₁)) := by
      rw [inverseActionCancellation laws.ring laws.assignmentModule laws.assignmentLaws delta₂ unit₂]
    _ = laws.assignmentModule.smul unit₂.inverse
        (laws.assignmentModule.smul unit₁.inverse (laws.assignmentModule.smul delta₂ opening₁)) :=
      swap _ _ _
    _ = laws.assignmentModule.smul unit₂.inverse
        (laws.assignmentModule.smul unit₁.inverse (laws.assignmentModule.smul delta₁ opening₂)) := by rw [cross]
    _ = _ := by
      rw [inverseActionCancellation laws.ring laws.assignmentModule laws.assignmentLaws delta₁ unit₁]

/-- The collision data are precisely the two observed challenge and response
differences at this coordinate. Unit inverses occur only in its proof. -/
def collisionAt (samePhi : phi leftBatch.inputs = phi rightBatch.inputs)
    (coordinate : Fin arity.total)
    (different : extractedAssignment laws strongSet left coordinate ≠
      extractedAssignment laws strongSet right coordinate) :
    RelaxedBindingCollision semantics params ops (leftBatch.inputs coordinate).commitment where
  delta₁ := coordinateDelta laws left coordinate
  delta₂ := coordinateDelta laws right coordinate
  opening₁ := coordinateDifference laws left coordinate
  opening₂ := coordinateDifference laws right coordinate
  delta₁Valid := compatible.differenceValid _ _
    (left.baseStrong coordinate) (left.forkStrong coordinate coordinate)
  delta₂Valid := compatible.differenceValid _ _
    (right.baseStrong coordinate) (right.forkStrong coordinate coordinate)
  firstEquation := by
    rw [compatible.commitmentAction]
    exact (difference_commits laws left coordinate).symm
  secondEquation := by
    rw [compatible.commitmentAction]
    have same : (leftBatch.inputs coordinate).commitment = (rightBatch.inputs coordinate).commitment :=
      congrFun samePhi coordinate
    rw [same]
    exact (difference_commits laws right coordinate).symm
  firstNorm := compatible.differenceBounded _ _ left.baseSuccess.1.2.2
    (left.forkSuccess coordinate).1.2.2
  secondNorm := compatible.differenceBounded _ _ right.baseSuccess.1.2.2
    (right.forkSuccess coordinate).1.2.2
  crossDifferent := by
    intro cross
    apply different
    rw [compatible.assignmentAction] at cross
    exact inverse_cross_eq laws _ _ (left.coordinateUnit laws strongSet coordinate)
      (right.coordinateUnit laws strongSet coordinate) _ _ cross

include compatible in
/-- Unequal extracted vectors expose a collision at one of their actual
coordinates. The collision constructor uses no intermediate opening premise. -/
theorem two_forks_unique_or_collision (samePhi : phi leftBatch.inputs = phi rightBatch.inputs) :
    extractedAssignment laws strongSet left = extractedAssignment laws strongSet right ∨
      ∃ coordinate, Nonempty (RelaxedBindingCollision semantics params ops
        (leftBatch.inputs coordinate).commitment) := by
  classical
  by_cases equal : extractedAssignment laws strongSet left = extractedAssignment laws strongSet right
  · exact Or.inl equal
  · have different : ∃ coordinate, extractedAssignment laws strongSet left coordinate ≠
        extractedAssignment laws strongSet right coordinate :=
      Classical.byContradiction fun noDifference => equal (funext fun coordinate =>
        Classical.byContradiction fun different => noDifference ⟨coordinate, different⟩)
    obtain ⟨coordinate, different⟩ := different
    exact Or.inr ⟨coordinate, ⟨collisionAt laws ops compatible strongSet leftBatch rightBatch
      left right samePhi coordinate different⟩⟩

end NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding
