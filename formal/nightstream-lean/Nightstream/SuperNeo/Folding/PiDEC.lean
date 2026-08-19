import Nightstream.SuperNeo.Relations

/-!
Model-level Π_DEC reduction (SuperNeo Theorem 7).

The verifier checks exact base-`b` recomposition of commitments, public inputs,
and evaluation claims. Completeness splits one `CE(B)` witness into exactly `k`
fresh `CE(b)` witnesses. Knowledge reduction runs in the reverse direction:
valid child openings and the public recomposition equations construct an actual
opening of the parent statement.
-/

namespace Nightstream.SuperNeo.Folding.PiDEC

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

/-- Digit decomposition and the homomorphism laws checked by Π_DEC. -/
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

/-- Public parent and exactly `k` decomposed child statements. -/
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

/-- Exact verifier equations for Π_DEC. -/
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

/-- Canonical fresh child statements produced from the parent's digit witnesses. -/
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

/-- Each honest digit is a fresh CE opening. -/
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
    simpa [parentCombined] using parentNorm
  exact ⟨⟨rfl, rfl, algebra.split_norm assignment combinedNorm i⟩,
    parentValid.2.1, rfl⟩

/-- Perfect completeness of the exact Π_DEC recomposition checks. -/
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

/-- Exact digit recomposition is exposed as a first-class theorem. -/
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

/--
Π_DEC reduction of knowledge: valid fresh child witnesses and accepted public
recomposition equations construct a valid combined parent witness.
-/
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
    simpa [accepted.childFresh i] using childNorm
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
    simpa [accepted.parentCombined] using combinedNorm
  · exact (algebra.evaluations_hom attempt.parent.constraintSystem
      attempt.parent.point childAssignments).trans
      ((congrArg algebra.recomposeEvaluations evaluationsAgree).trans
        accepted.evaluationEquation.symm)

end Nightstream.SuperNeo.Folding.PiDEC
