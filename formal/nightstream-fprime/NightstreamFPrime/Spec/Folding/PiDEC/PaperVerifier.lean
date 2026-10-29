import NightstreamFPrime.Spec.Folding.PiDEC

/-! SuperNeo Section 7.5. -/

namespace NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment

structure PublicInputSplit
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params) where
  parentBounded : PublicInput → Prop
  parentBounded_decidable : ∀ input, Decidable (parentBounded input)
  parentBounded_project : ∀ assignment,
    semantics.normBounded params.bigB assignment →
      parentBounded (semantics.projectPublicInput assignment)
  split : PublicInput → Fin params.k → PublicInput
  recompose_split : ∀ input,
    algebra.recomposePublicInput (split input) = input
  split_project : ∀ assignment child,
    split (semantics.projectPublicInput assignment) child =
      semantics.projectPublicInput (algebra.splitAssignment assignment child)

namespace PublicInputSplit

/-- SuperNeo Section 7.5. -/
def checked
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (input : PublicInput) : Option (Fin params.k → PublicInput) :=
  letI := publicSplit.parentBounded_decidable input
  if publicSplit.parentBounded input then
    some (publicSplit.split input)
  else
    none

theorem checked_eq_some
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (input : PublicInput)
    (bounded : publicSplit.parentBounded input) :
    publicSplit.checked input = some (publicSplit.split input) := by
  simp [checked, bounded]

theorem checked_eq_none
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (input : PublicInput)
    (unbounded : ¬ publicSplit.parentBounded input) :
    publicSplit.checked input = none := by
  simp [checked, unbounded]

end PublicInputSplit

structure EvaluationArity
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment) where
  count : Structure → Nat
  evaluations_size : ∀ system assignment point,
    (semantics.evaluations system assignment point).size = count system

/-- SuperNeo Section 7.5. -/
structure ChildMessage
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment) where
  commitment : Commitment
  evaluations : Array Evaluation

structure Attempt
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams) where
  parent : CE.Instance Structure PublicInput Point Evaluation Commitment
  messages : Fin params.k → ChildMessage Evaluation Commitment

def children
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params) :
    Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment :=
  fun child => {
    constraintSystem := attempt.parent.constraintSystem
    commitment := (attempt.messages child).commitment
    publicInput := publicSplit.split attempt.parent.publicInput child
    point := attempt.parent.point
    evaluations := (attempt.messages child).evaluations
    stage := .fresh
  }

@[simp] theorem children_publicInput
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (child : Fin params.k) :
    (children publicSplit attempt child).publicInput =
      publicSplit.split attempt.parent.publicInput child := by
  rfl

/-- SuperNeo Section 7.5. -/
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
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment
      params) : Prop where
  parentBounded : publicSplit.parentBounded attempt.parent.publicInput
  parentCombined : attempt.parent.stage = .combined
  parentEvaluationSize :
    attempt.parent.evaluations.size =
      evaluationArity.count attempt.parent.constraintSystem
  messageEvaluationSize : ∀ child,
    (attempt.messages child).evaluations.size =
      evaluationArity.count attempt.parent.constraintSystem
  commitmentEquation :
    attempt.parent.commitment =
      algebra.recomposeCommitment fun child =>
        (attempt.messages child).commitment
  evaluationEquation :
    attempt.parent.evaluations =
      algebra.recomposeEvaluations fun child =>
        (attempt.messages child).evaluations

def acceptedDecision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    [DecidableEq Commitment]
    [DecidableEq Evaluation] :
    Decidable (Accepted algebra publicSplit evaluationArity attempt) := by
  letI := publicSplit.parentBounded_decidable attempt.parent.publicInput
  by_cases parentBounded :
      publicSplit.parentBounded attempt.parent.publicInput
  · by_cases parentCombined : attempt.parent.stage = .combined
    · by_cases parentEvaluationSize :
          attempt.parent.evaluations.size =
            evaluationArity.count attempt.parent.constraintSystem
      · by_cases messageEvaluationSize : ∀ child,
            (attempt.messages child).evaluations.size =
              evaluationArity.count attempt.parent.constraintSystem
        · by_cases commitmentEquation :
              attempt.parent.commitment =
                algebra.recomposeCommitment fun child =>
                  (attempt.messages child).commitment
          · by_cases evaluationEquation :
                attempt.parent.evaluations =
                  algebra.recomposeEvaluations fun child =>
                    (attempt.messages child).evaluations
            · exact isTrue {
                parentBounded
                parentCombined
                parentEvaluationSize
                messageEvaluationSize
                commitmentEquation
                evaluationEquation
              }
            · exact isFalse fun accepted =>
                evaluationEquation accepted.evaluationEquation
          · exact isFalse fun accepted =>
              commitmentEquation accepted.commitmentEquation
        · exact isFalse fun accepted =>
            messageEvaluationSize accepted.messageEvaluationSize
      · exact isFalse fun accepted =>
          parentEvaluationSize accepted.parentEvaluationSize
    · exact isFalse fun accepted =>
        parentCombined accepted.parentCombined
  · exact isFalse fun accepted =>
      parentBounded accepted.parentBounded

def toRecompositionAttempt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params) :
    PiDEC.Attempt Structure PublicInput Point Evaluation Commitment params := {
  parent := attempt.parent
  children := children publicSplit attempt
}

def messagesOf
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    (output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    Fin params.k → ChildMessage Evaluation Commitment :=
  fun child => {
    commitment := (output child).commitment
    evaluations := (output child).evaluations
  }

def attemptForOutput
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment) :
    Attempt Structure PublicInput Point Evaluation Commitment params := {
  parent := parent
  messages := messagesOf output
}

structure OutputAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment) : Prop where
  outputComputed :
    children publicSplit (attemptForOutput parent output) = output
  checks : Accepted algebra publicSplit evaluationArity
    (attemptForOutput parent output)

namespace Accepted

theorem toRecompositionAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {attempt : Attempt Structure PublicInput Point Evaluation Commitment params}
    (accepted : Accepted algebra publicSplit evaluationArity attempt) :
    PiDEC.Accepted algebra (toRecompositionAttempt publicSplit attempt) := {
  parentCombined := accepted.parentCombined
  childFresh := fun _ => rfl
  sameStructure := fun _ => rfl
  samePoint := fun _ => rfl
  commitmentEquation := accepted.commitmentEquation
  publicInputEquation :=
    (publicSplit.recompose_split attempt.parent.publicInput).symm
  evaluationEquation := accepted.evaluationEquation
}

end Accepted

namespace OutputAccepted

theorem childPublicInput_eq
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted : OutputAccepted algebra publicSplit evaluationArity parent output)
    (child : Fin params.k) :
    (output child).publicInput =
      publicSplit.split parent.publicInput child := by
  have childEq := congrFun accepted.outputComputed child
  calc
    (output child).publicInput =
        (children publicSplit (attemptForOutput parent output) child).publicInput :=
      (congrArg CE.Instance.publicInput childEq).symm
    _ = publicSplit.split parent.publicInput child := rfl

theorem publicInputs_eq_of_parentPublicInput_eq
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {leftParent rightParent :
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    {leftOutput rightOutput : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (leftAccepted :
      OutputAccepted algebra publicSplit evaluationArity leftParent leftOutput)
    (rightAccepted :
      OutputAccepted algebra publicSplit evaluationArity rightParent rightOutput)
    (parentPublicInputEq :
      leftParent.publicInput = rightParent.publicInput) :
    (fun child => (leftOutput child).publicInput) =
      (fun child => (rightOutput child).publicInput) := by
  funext child
  rw [leftAccepted.childPublicInput_eq child,
    rightAccepted.childPublicInput_eq child, parentPublicInputEq]

theorem childEvaluations_size
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted :
      OutputAccepted algebra publicSplit evaluationArity parent output)
    (child : Fin params.k) :
    (output child).evaluations.size =
      evaluationArity.count parent.constraintSystem := by
  exact accepted.checks.messageEvaluationSize child

theorem parentEvaluations_size
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted :
      OutputAccepted algebra publicSplit evaluationArity parent output) :
    parent.evaluations.size =
      evaluationArity.count parent.constraintSystem :=
  accepted.checks.parentEvaluationSize

theorem toRecompositionAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    {publicSplit : PublicInputSplit algebra}
    {evaluationArity : EvaluationArity semantics}
    {parent : CE.Instance Structure PublicInput Point Evaluation Commitment}
    {output : Fin params.k →
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (accepted :
      OutputAccepted algebra publicSplit evaluationArity parent output) :
    PiDEC.Accepted algebra { parent := parent, children := output } := by
  have relaxed := accepted.checks.toRecompositionAccepted
  change PiDEC.Accepted algebra {
    parent := parent
    children := children publicSplit (attemptForOutput parent output)
  } at relaxed
  rw [accepted.outputComputed] at relaxed
  exact relaxed

end OutputAccepted

theorem childPublicInputs_eq_of_parentPublicInput_eq
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params}
    (publicSplit : PublicInputSplit algebra)
    (left right : Attempt Structure PublicInput Point Evaluation Commitment
      params)
    (parentPublicInputEq :
      left.parent.publicInput = right.parent.publicInput) :
    (fun child => (children publicSplit left child).publicInput) =
      (fun child => (children publicSplit right child).publicInput) := by
  funext child
  exact congrArg (fun input => publicSplit.split input child)
    parentPublicInputEq

def honestMessages
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) :
    Fin params.k → ChildMessage Evaluation Commitment :=
  fun child => {
    commitment := semantics.commit (algebra.splitAssignment assignment child)
    evaluations := semantics.evaluations parent.constraintSystem
      (algebra.splitAssignment assignment child) parent.point
  }

def honestAttempt
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) :
    Attempt Structure PublicInput Point Evaluation Commitment params := {
  parent := parent
  messages := honestMessages algebra parent assignment
}

theorem honestChildren_eq_childrenOf
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentPublicInput :
      semantics.projectPublicInput assignment = parent.publicInput) :
    children publicSplit (honestAttempt algebra parent assignment) =
      PiDEC.childrenOf algebra parent assignment := by
  funext child
  have publicInputEq :
      publicSplit.split parent.publicInput child =
        semantics.projectPublicInput
          (algebra.splitAssignment assignment child) := by
    calc
      publicSplit.split parent.publicInput child =
          publicSplit.split
            (semantics.projectPublicInput assignment) child := by
        rw [parentPublicInput]
      _ = semantics.projectPublicInput
          (algebra.splitAssignment assignment child) :=
        publicSplit.split_project assignment child
  simp [children, honestAttempt, honestMessages, PiDEC.childrenOf,
    publicInputEq]

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
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentCombined : parent.stage = .combined)
    (parentValid : CE.Holds semantics params parent assignment) :
    let attempt := honestAttempt algebra parent assignment
    Accepted algebra publicSplit evaluationArity attempt ∧
      ∀ child, CE.Holds semantics params (children publicSplit attempt child)
        (algebra.splitAssignment assignment child) := by
  dsimp only
  have recomposition := PiDEC.complete semantics params algebra parent
    assignment parentCombined parentValid
  have childrenEq := honestChildren_eq_childrenOf algebra publicSplit parent
    assignment parentValid.1.2.1
  constructor
  · exact {
      parentBounded := by
        have parentNorm : semantics.normBounded params.bigB assignment := by
          have bounded := parentValid.1.2.2
          rw [parentCombined] at bounded
          exact bounded
        change publicSplit.parentBounded parent.publicInput
        rw [← parentValid.1.2.1]
        exact publicSplit.parentBounded_project assignment parentNorm
      parentCombined := recomposition.1.parentCombined
      parentEvaluationSize := by
        simp only [honestAttempt]
        rw [← parentValid.2.2]
        exact evaluationArity.evaluations_size _ _ _
      messageEvaluationSize := by
        intro child
        simpa only [honestAttempt, honestMessages] using
          evaluationArity.evaluations_size parent.constraintSystem
            (algebra.splitAssignment assignment child) parent.point
      commitmentEquation := recomposition.1.commitmentEquation
      evaluationEquation := recomposition.1.evaluationEquation
    }
  · intro child
    rw [childrenEq]
    exact recomposition.2 child

theorem output_complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (parent : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment)
    (parentCombined : parent.stage = .combined)
    (parentValid : CE.Holds semantics params parent assignment) :
    let output := PiDEC.childrenOf algebra parent assignment
    OutputAccepted algebra publicSplit evaluationArity parent output ∧
      ∀ child, CE.Holds semantics params (output child)
        (algebra.splitAssignment assignment child) := by
  dsimp only
  have operational := complete semantics params algebra publicSplit
    evaluationArity parent assignment parentCombined parentValid
  refine ⟨{
    outputComputed := ?_
    checks := ?_
  }, ?_⟩
  · exact honestChildren_eq_childrenOf algebra publicSplit parent assignment
      parentValid.1.2.1
  · simpa [attemptForOutput, messagesOf, honestAttempt, honestMessages,
      PiDEC.childrenOf] using! operational.1
  · intro child
    have childrenEq := honestChildren_eq_childrenOf algebra publicSplit parent
      assignment parentValid.1.2.1
    rw [← childrenEq]
    exact operational.2 child

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
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (childAssignments : Fin params.k → Assignment)
    (kPositive : 0 < params.k)
    (accepted : Accepted algebra publicSplit evaluationArity attempt)
    (childrenValid : ∀ child,
      CE.Holds semantics params (children publicSplit attempt child)
        (childAssignments child)) :
    CE.Holds semantics params attempt.parent
      (algebra.recomposeAssignment childAssignments) :=
  PiDEC.reduce_knowledge semantics params algebra
    (toRecompositionAttempt publicSplit attempt) childAssignments kPositive
    accepted.toRecompositionAccepted childrenValid

theorem parent_eq_recompose_or_bindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : PiDEC.Algebra Structure Assignment PublicInput Point Evaluation
      Commitment semantics params)
    (publicSplit : PublicInputSplit algebra)
    (evaluationArity : EvaluationArity semantics)
    (attempt : Attempt Structure PublicInput Point Evaluation Commitment params)
    (parentAssignment : Assignment)
    (childAssignments : Fin params.k → Assignment)
    (accepted : Accepted algebra publicSplit evaluationArity attempt)
    (parentValid : CE.Holds semantics params attempt.parent parentAssignment)
    (childrenValid : ∀ child,
      CE.Holds semantics params (children publicSplit attempt child)
        (childAssignments child)) :
    parentAssignment = algebra.recomposeAssignment childAssignments ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision semantics params
        attempt.parent.commitment) :=
  PiDEC.accepted_parent_eq_recompose_or_bindingCollision semantics params
    algebra (toRecompositionAttempt publicSplit attempt) parentAssignment
    childAssignments accepted.toRecompositionAccepted
    parentValid childrenValid

end NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier
