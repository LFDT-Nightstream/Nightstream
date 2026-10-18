import NightstreamFPrime.Spec.Relation

/-!
Model-level Π_DEC recomposition reduction and standard parent-opening binding
boundary.

Owns: exact base-`b` recomposition semantics, completeness, knowledge reduction,
and the model-level collision exposed when a separately valid parent opening
differs from the opening reconstructed from valid children.

This module's `Accepted` is intentionally the reusable recomposition core. It
is weaker than the Section-7.5 operational verifier, which computes child
public inputs from the parent; that verifier is owned by `PiDEC.PaperVerifier`.
-/

namespace NightstreamFPrime.Spec.Folding.PiDEC

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

structure Algebra
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams) where
  splitAssignment : Assignment → Fin params.k → Assignment
  recomposeAssignment : (Fin params.k → Assignment) → Assignment
  recomposeCommitment : (Fin params.k → Commitment) → Commitment
  recomposePublicInput : (Fin params.k → PublicInput) → PublicInput
  recomposeEvaluations : (Fin params.k → Array Evaluation) → Array Evaluation
  split_recompose : ∀ assignment,
    recomposeAssignment (splitAssignment assignment) = assignment
  split_norm : ∀ assignment,
    semantics.normBounded params.bigB assignment →
      ∀ i, semantics.normBounded params.b (splitAssignment assignment i)
  recompose_norm : ∀ assignments,
    (∀ i, semantics.normBounded params.b (assignments i)) →
      semantics.normBounded params.bigB (recomposeAssignment assignments)
  commit_hom : ∀ assignments,
    semantics.commit (recomposeAssignment assignments) =
      recomposeCommitment (fun i => semantics.commit (assignments i))
  publicInput_hom : ∀ assignments,
    semantics.projectPublicInput (recomposeAssignment assignments) =
      recomposePublicInput
        (fun i => semantics.projectPublicInput (assignments i))
  evaluations_hom : ∀ (system : Structure) (point : Point) assignments,
    semantics.evaluations system (recomposeAssignment assignments) point =
      recomposeEvaluations
        (fun i => semantics.evaluations system (assignments i) point)

structure Attempt
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams) where
  parent : CE.Instance Structure PublicInput Point Evaluation Commitment
  children : Fin params.k →
    CE.Instance Structure PublicInput Point Evaluation Commitment

/-- Public recomposition equations used by the Π_DEC knowledge reduction.

This is not the exact Section-7.5 operational verifier: it accepts any child
public-input family that recomposes to the parent. -/
structure Accepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params) : Prop where
  parentCombined : attempt.parent.stage = .combined
  childFresh : ∀ i, (attempt.children i).stage = .fresh
  sameStructure : ∀ i,
    (attempt.children i).constraintSystem = attempt.parent.constraintSystem
  samePoint : ∀ i, (attempt.children i).point = attempt.parent.point
  commitmentEquation :
    attempt.parent.commitment =
      algebra.recomposeCommitment (fun i => (attempt.children i).commitment)
  publicInputEquation :
    attempt.parent.publicInput =
      algebra.recomposePublicInput (fun i => (attempt.children i).publicInput)
  evaluationEquation :
    attempt.parent.evaluations =
      algebra.recomposeEvaluations (fun i => (attempt.children i).evaluations)

theorem Accepted.parent_eq_of_children_eq
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params}
    {left right :
      Attempt Structure PublicInput Point Evaluation Commitment params}
    (kPositive : 0 < params.k)
    (leftAccepted : Accepted algebra left)
    (rightAccepted : Accepted algebra right)
    (childrenEq : left.children = right.children) :
    left.parent = right.parent := by
  rcases left with ⟨leftParent, leftChildren⟩
  rcases right with ⟨rightParent, rightChildren⟩
  change leftChildren = rightChildren at childrenEq
  let first : Fin params.k := ⟨0, kPositive⟩
  have structureEq :
      leftParent.constraintSystem = rightParent.constraintSystem := by
    calc
      leftParent.constraintSystem =
          (leftChildren first).constraintSystem :=
        (leftAccepted.sameStructure first).symm
      _ = (rightChildren first).constraintSystem := by
        rw [childrenEq]
      _ = rightParent.constraintSystem :=
        rightAccepted.sameStructure first
  have commitmentEq :
      leftParent.commitment = rightParent.commitment := by
    calc
      leftParent.commitment =
          algebra.recomposeCommitment
            (fun i => (leftChildren i).commitment) :=
        leftAccepted.commitmentEquation
      _ = algebra.recomposeCommitment
            (fun i => (rightChildren i).commitment) := by
        rw [childrenEq]
      _ = rightParent.commitment :=
        rightAccepted.commitmentEquation.symm
  have publicInputEq :
      leftParent.publicInput = rightParent.publicInput := by
    calc
      leftParent.publicInput =
          algebra.recomposePublicInput
            (fun i => (leftChildren i).publicInput) :=
        leftAccepted.publicInputEquation
      _ = algebra.recomposePublicInput
            (fun i => (rightChildren i).publicInput) := by
        rw [childrenEq]
      _ = rightParent.publicInput :=
        rightAccepted.publicInputEquation.symm
  have pointEq : leftParent.point = rightParent.point := by
    calc
      leftParent.point = (leftChildren first).point :=
        (leftAccepted.samePoint first).symm
      _ = (rightChildren first).point := by
        rw [childrenEq]
      _ = rightParent.point := rightAccepted.samePoint first
  have evaluationsEq :
      leftParent.evaluations = rightParent.evaluations := by
    calc
      leftParent.evaluations =
          algebra.recomposeEvaluations
            (fun i => (leftChildren i).evaluations) :=
        leftAccepted.evaluationEquation
      _ = algebra.recomposeEvaluations
            (fun i => (rightChildren i).evaluations) := by
        rw [childrenEq]
      _ = rightParent.evaluations :=
        rightAccepted.evaluationEquation.symm
  have stageEq : leftParent.stage = rightParent.stage :=
    leftAccepted.parentCombined.trans rightAccepted.parentCombined.symm
  rcases leftParent with ⟨_, _, _, _, _, _⟩
  rcases rightParent with ⟨_, _, _, _, _, _⟩
  simp_all

def childrenOf
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) :
    Fin params.k → CE.Instance Structure PublicInput Point Evaluation Commitment :=
  fun i => {
    constraintSystem := parent.constraintSystem
    commitment := semantics.commit (algebra.splitAssignment assignment i)
    publicInput := semantics.projectPublicInput (algebra.splitAssignment assignment i)
    point := parent.point
    evaluations := semantics.evaluations parent.constraintSystem
      (algebra.splitAssignment assignment i) parent.point
    stage := .fresh
  }

theorem childrenOf_holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentCombined : parent.stage = .combined)
    (parentValid : CE.Holds semantics params parent assignment) :
    ∀ i, CE.Holds semantics params (childrenOf algebra parent assignment i)
      (algebra.splitAssignment assignment i) := by
  intro i
  have parentNorm := parentValid.1.2.2
  have combinedNorm : semantics.normBounded params.bigB assignment := by
    simpa [parentCombined] using! parentNorm
  exact ⟨⟨rfl, rfl, algebra.split_norm assignment combinedNorm i⟩,
    parentValid.2.1, rfl⟩

theorem complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentCombined : parent.stage = .combined)
    (parentValid : CE.Holds semantics params parent assignment) :
    let attempt : Attempt Structure PublicInput Point Evaluation Commitment params := {
      parent := parent
      children := childrenOf algebra parent assignment
    }
    Accepted algebra attempt ∧
      ∀ i, CE.Holds semantics params (attempt.children i)
        (algebra.splitAssignment assignment i) := by
  dsimp only
  constructor
  · refine {
      parentCombined := parentCombined
      childFresh := fun _ => rfl
      sameStructure := fun _ => rfl
      samePoint := fun _ => rfl
      commitmentEquation := ?_
      publicInputEquation := ?_
      evaluationEquation := ?_
    }
    · calc
        parent.commitment = semantics.commit assignment := parentValid.1.1.symm
        _ = semantics.commit
            (algebra.recomposeAssignment (algebra.splitAssignment assignment)) := by
              rw [algebra.split_recompose]
        _ = algebra.recomposeCommitment
            (fun i => semantics.commit (algebra.splitAssignment assignment i)) :=
              algebra.commit_hom (algebra.splitAssignment assignment)
    · calc
        parent.publicInput = semantics.projectPublicInput assignment :=
          parentValid.1.2.1.symm
        _ = semantics.projectPublicInput
            (algebra.recomposeAssignment (algebra.splitAssignment assignment)) := by
              rw [algebra.split_recompose]
        _ = algebra.recomposePublicInput
            (fun i => semantics.projectPublicInput
              (algebra.splitAssignment assignment i)) :=
              algebra.publicInput_hom (algebra.splitAssignment assignment)
    · calc
        parent.evaluations =
            semantics.evaluations parent.constraintSystem assignment parent.point :=
              parentValid.2.2.symm
        _ = semantics.evaluations parent.constraintSystem
            (algebra.recomposeAssignment (algebra.splitAssignment assignment))
            parent.point := by rw [algebra.split_recompose]
        _ = algebra.recomposeEvaluations
            (fun i => semantics.evaluations parent.constraintSystem
              (algebra.splitAssignment assignment i) parent.point) :=
              algebra.evaluations_hom parent.constraintSystem parent.point
                (algebra.splitAssignment assignment)
  · exact childrenOf_holds semantics params algebra parent assignment
      parentCombined parentValid

theorem split_recompose_exact
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (assignment : Assignment) :
    algebra.recomposeAssignment (algebra.splitAssignment assignment) = assignment :=
  algebra.split_recompose assignment

theorem reduce_knowledge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (childAssignments : Fin params.k → Assignment)
    (kPositive : 0 < params.k)
    (accepted : Accepted algebra attempt)
    (childrenValid : ∀ i,
      CE.Holds semantics params (attempt.children i) (childAssignments i)) :
    CE.Holds semantics params attempt.parent
      (algebra.recomposeAssignment childAssignments) := by
  have commitmentsAgree :
      (fun i => semantics.commit (childAssignments i)) =
        (fun i => (attempt.children i).commitment) := by
    funext i
    exact (childrenValid i).1.1
  have publicInputsAgree :
      (fun i => semantics.projectPublicInput (childAssignments i)) =
        (fun i => (attempt.children i).publicInput) := by
    funext i
    exact (childrenValid i).1.2.1
  have evaluationsAgree :
      (fun i => semantics.evaluations attempt.parent.constraintSystem
        (childAssignments i) attempt.parent.point) =
      (fun i => (attempt.children i).evaluations) := by
    funext i
    calc
      semantics.evaluations attempt.parent.constraintSystem
          (childAssignments i) attempt.parent.point =
          semantics.evaluations (attempt.children i).constraintSystem
            (childAssignments i) (attempt.children i).point := by
              rw [accepted.sameStructure i, accepted.samePoint i]
      _ = (attempt.children i).evaluations := (childrenValid i).2.2
  have freshNorms : ∀ i, semantics.normBounded params.b (childAssignments i) := by
    intro i
    have childNorm := (childrenValid i).1.2.2
    simpa [accepted.childFresh i] using! childNorm
  let first : Fin params.k := ⟨0, kPositive⟩
  have parentPointValid :
      semantics.evaluationPointValid attempt.parent.constraintSystem
        attempt.parent.point := by
    have childPointValid := (childrenValid first).2.1
    simpa [accepted.sameStructure first, accepted.samePoint first] using childPointValid
  refine ⟨⟨?_, ?_, ?_⟩, parentPointValid, ?_⟩
  · exact (algebra.commit_hom childAssignments).trans
      ((congrArg algebra.recomposeCommitment commitmentsAgree).trans
        accepted.commitmentEquation.symm)
  · exact (algebra.publicInput_hom childAssignments).trans
      ((congrArg algebra.recomposePublicInput publicInputsAgree).trans
        accepted.publicInputEquation.symm)
  · have combinedNorm := algebra.recompose_norm childAssignments freshNorms
    simpa [accepted.parentCombined] using! combinedNorm
  · exact (algebra.evaluations_hom attempt.parent.constraintSystem
      attempt.parent.point childAssignments).trans
      ((congrArg algebra.recomposeEvaluations evaluationsAgree).trans
        accepted.evaluationEquation.symm)

structure ParentOpeningBindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (commitment : Commitment) where
  parentOpening : Assignment
  recomposedOpening : Assignment
  parentCommits : semantics.commit parentOpening = commitment
  recomposedCommits : semantics.commit recomposedOpening = commitment
  parentNorm : semantics.normBounded params.bigB parentOpening
  recomposedNorm : semantics.normBounded params.bigB recomposedOpening
  different : parentOpening ≠ recomposedOpening

theorem accepted_parent_eq_recompose_or_bindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment semantics params)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (parentAssignment : Assignment)
    (childAssignments : Fin params.k → Assignment)
    (accepted : Accepted algebra attempt)
    (parentValid : CE.Holds semantics params attempt.parent parentAssignment)
    (childrenValid : ∀ i,
      CE.Holds semantics params (attempt.children i) (childAssignments i)) :
    parentAssignment = algebra.recomposeAssignment childAssignments ∨
      Nonempty (ParentOpeningBindingCollision semantics params
        attempt.parent.commitment) := by
  have commitmentsAgree :
      (fun i => semantics.commit (childAssignments i)) =
        (fun i => (attempt.children i).commitment) := by
    funext i
    exact (childrenValid i).1.1
  have recomposedCommits :
      semantics.commit (algebra.recomposeAssignment childAssignments) =
        attempt.parent.commitment := by
    exact (algebra.commit_hom childAssignments).trans
      ((congrArg algebra.recomposeCommitment commitmentsAgree).trans
        accepted.commitmentEquation.symm)
  have freshNorms : ∀ i,
      semantics.normBounded params.b (childAssignments i) := by
    intro i
    have childNorm := (childrenValid i).1.2.2
    simpa [accepted.childFresh i] using! childNorm
  have recomposedNorm :
      semantics.normBounded params.bigB
        (algebra.recomposeAssignment childAssignments) :=
    algebra.recompose_norm childAssignments freshNorms
  have parentNorm : semantics.normBounded params.bigB parentAssignment := by
    have validNorm := parentValid.1.2.2
    simpa [accepted.parentCombined] using! validNorm
  by_cases same : parentAssignment = algebra.recomposeAssignment childAssignments
  · exact Or.inl same
  · exact Or.inr ⟨{
      parentOpening := parentAssignment
      recomposedOpening := algebra.recomposeAssignment childAssignments
      parentCommits := parentValid.1.1
      recomposedCommits := recomposedCommits
      parentNorm := parentNorm
      recomposedNorm := recomposedNorm
      different := same
    }⟩

end NightstreamFPrime.Spec.Folding.PiDEC
