import NightstreamFPrime.Spec.Folding.BatchArity

/-! SuperNeo v1.2 Lemma 8. -/

namespace NightstreamFPrime.Spec.Folding.PiRLC

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar

structure Algebra
    (Structure : Type uStructure)
    (Assignment : Type uAssignment)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams) where
  challengeValid : Scalar → Prop
  combineAssignment : {n : Nat} →
    (Fin n → Scalar) → (Fin n → Assignment) → Assignment
  combineCommitment : {n : Nat} →
    (Fin n → Scalar) → (Fin n → Commitment) → Commitment
  combinePublicInput : {n : Nat} →
    (Fin n → Scalar) → (Fin n → PublicInput) → PublicInput
  combineEvaluations : {n : Nat} →
    (Fin n → Scalar) → (Fin n → Array Evaluation) → Array Evaluation
  commit_hom : ∀ {n} (challenges : Fin n → Scalar)
      (assignments : Fin n → Assignment),
    semantics.commit (combineAssignment challenges assignments) =
      combineCommitment challenges (fun i => semantics.commit (assignments i))
  publicInput_hom : ∀ {n} (challenges : Fin n → Scalar)
      (assignments : Fin n → Assignment),
    semantics.projectPublicInput (combineAssignment challenges assignments) =
      combinePublicInput challenges
        (fun i => semantics.projectPublicInput (assignments i))
  evaluations_hom : ∀ {n} (system : Structure) (point : Point)
      (challenges : Fin n → Scalar) (assignments : Fin n → Assignment),
    semantics.evaluations system (combineAssignment challenges assignments) point =
      combineEvaluations challenges
        (fun i => semantics.evaluations system (assignments i) point)
  /-- SuperNeo v1.2 Definition 22. -/
  norm_growth : ∀ {n : Nat}
      (_totalBound : n ≤ params.maxFresh + params.k)
      (challenges : Fin n → Scalar)
      (assignments : Fin n → Assignment),
    (∀ i, challengeValid (challenges i)) →
    (∀ i, semantics.normBounded params.b (assignments i)) →
    semantics.normBounded params.bigB
      (combineAssignment challenges assignments)

structure Attempt
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (params : GlobalParams)
    (arity : BatchArity params) where
  inputs : Fin arity.total →
    CE.Instance Structure PublicInput Point Evaluation Commitment
  challenges : Fin arity.total → Scalar
  output : CE.Instance Structure PublicInput Point Evaluation Commitment

structure Equations
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity) : Prop where
  inputFresh : ∀ i, (attempt.inputs i).stage = .fresh
  sameStructure : ∀ i,
    (attempt.inputs i).constraintSystem = attempt.output.constraintSystem
  samePoint : ∀ i, (attempt.inputs i).point = attempt.output.point
  outputCombined : attempt.output.stage = .combined
  commitmentEquation :
    attempt.output.commitment =
      algebra.combineCommitment attempt.challenges
        (fun i => (attempt.inputs i).commitment)
  publicInputEquation :
    attempt.output.publicInput =
      algebra.combinePublicInput attempt.challenges
        (fun i => (attempt.inputs i).publicInput)
  evaluationEquation :
    attempt.output.evaluations =
      algebra.combineEvaluations attempt.challenges
        (fun i => (attempt.inputs i).evaluations)

structure Accepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity)
    : Prop extends Equations algebra attempt where
  challengesValid : ∀ i, algebra.challengeValid (attempt.challenges i)

namespace Equations

theorem withChallengesValid
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    {algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params}
    {attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity}
    (equations : Equations algebra attempt)
    (challengesValid : ∀ i, algebra.challengeValid (attempt.challenges i)) :
    Accepted algebra attempt :=
  { equations with challengesValid := challengesValid }

end Equations

def combinedOutput
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    {arity : BatchArity params}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (system : Structure)
    (point : Point)
    (inputs : Fin arity.total →
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (challenges : Fin arity.total → Scalar) :
    CE.Instance Structure PublicInput Point Evaluation Commitment where
  constraintSystem := system
  commitment := algebra.combineCommitment challenges (fun i => (inputs i).commitment)
  publicInput := algebra.combinePublicInput challenges (fun i => (inputs i).publicInput)
  point := point
  evaluations := algebra.combineEvaluations challenges (fun i => (inputs i).evaluations)
  stage := .combined

def combinedWitness
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment}
    {params : GlobalParams}
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    {n : Nat}
    (challenges : Fin n → Scalar)
    (assignments : Fin n → Assignment) : Assignment :=
  algebra.combineAssignment challenges assignments

theorem combinedOutput_holds
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (arity : BatchArity params)
    (system : Structure)
    (point : Point)
    (inputs : Fin arity.total →
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (challenges : Fin arity.total → Scalar)
    (assignments : Fin arity.total → Assignment)
    (inputFresh : ∀ i, (inputs i).stage = .fresh)
    (sameStructure : ∀ i, (inputs i).constraintSystem = system)
    (samePoint : ∀ i, (inputs i).point = point)
    (challengesValid : ∀ i, algebra.challengeValid (challenges i))
    (inputValid : ∀ i, CE.Holds semantics params (inputs i) (assignments i))
    (pointValid : semantics.evaluationPointValid system point) :
    CE.Holds semantics params
      (combinedOutput algebra system point inputs challenges)
      (combinedWitness algebra challenges assignments) := by
  have commitmentsAgree :
      (fun i => semantics.commit (assignments i)) =
        (fun i => (inputs i).commitment) := by
    funext i
    exact (inputValid i).1.1
  have publicInputsAgree :
      (fun i => semantics.projectPublicInput (assignments i)) =
        (fun i => (inputs i).publicInput) := by
    funext i
    exact (inputValid i).1.2.1
  have evaluationsAgree :
      (fun i => semantics.evaluations system (assignments i) point) =
        (fun i => (inputs i).evaluations) := by
    funext i
    calc
      semantics.evaluations system (assignments i) point =
          semantics.evaluations (inputs i).constraintSystem (assignments i) (inputs i).point := by
            rw [sameStructure i, samePoint i]
      _ = (inputs i).evaluations := (inputValid i).2.2
  refine ⟨⟨?_, ?_, ?_⟩, pointValid, ?_⟩
  · exact (algebra.commit_hom challenges assignments).trans
      (congrArg (algebra.combineCommitment challenges) commitmentsAgree)
  · exact (algebra.publicInput_hom challenges assignments).trans
      (congrArg (algebra.combinePublicInput challenges) publicInputsAgree)
  · exact algebra.norm_growth arity.total_le challenges assignments
      challengesValid (fun i => by
        have inputNorm := (inputValid i).1.2.2
        simpa [inputFresh i] using! inputNorm)
  · exact (algebra.evaluations_hom system point challenges assignments).trans
      (congrArg (algebra.combineEvaluations challenges) evaluationsAgree)

theorem complete
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (algebra : Algebra
      Structure Assignment PublicInput Point Evaluation Commitment Scalar semantics params)
    (arity : BatchArity params)
    (system : Structure)
    (point : Point)
    (inputs : Fin arity.total →
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (challenges : Fin arity.total → Scalar)
    (assignments : Fin arity.total → Assignment)
    (inputFresh : ∀ i, (inputs i).stage = .fresh)
    (sameStructure : ∀ i, (inputs i).constraintSystem = system)
    (samePoint : ∀ i, (inputs i).point = point)
    (challengesValid : ∀ i, algebra.challengeValid (challenges i))
    (inputValid : ∀ i, CE.Holds semantics params (inputs i) (assignments i))
    (pointValid : semantics.evaluationPointValid system point) :
    let attempt : Attempt
        Structure PublicInput Point Evaluation Commitment Scalar params arity := {
      inputs := inputs
      challenges := challenges
      output := combinedOutput algebra system point inputs challenges
    }
    Accepted algebra attempt ∧
      CE.Holds semantics params attempt.output
        (combinedWitness algebra challenges assignments) := by
  dsimp only
  constructor
  · exact {
      inputFresh := inputFresh
      sameStructure := sameStructure
      samePoint := samePoint
      challengesValid := challengesValid
      outputCombined := rfl
      commitmentEquation := rfl
      publicInputEquation := rfl
      evaluationEquation := rfl
    }
  · exact combinedOutput_holds semantics params algebra arity system point
      inputs challenges assignments inputFresh sameStructure samePoint
      challengesValid inputValid pointValid

/-- SuperNeo v1.2 Definition 7. -/
structure RelaxedBindingOps
    (Assignment : Type uAssignment)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar) where
  scaleAssignment : Scalar → Assignment → Assignment
  scaleCommitment : Scalar → Commitment → Commitment
  differenceChallenge : Scalar → Prop

/-- SuperNeo v1.2 Definition 7. -/
structure RelaxedBindingCollision
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (commitment : Commitment) where
  delta₁ : Scalar
  delta₂ : Scalar
  opening₁ : Assignment
  opening₂ : Assignment
  delta₁Valid : ops.differenceChallenge delta₁
  delta₂Valid : ops.differenceChallenge delta₂
  firstEquation : ops.scaleCommitment delta₁ commitment = semantics.commit opening₁
  secondEquation : ops.scaleCommitment delta₂ commitment = semantics.commit opening₂
  firstNorm : semantics.normBounded (2 * params.bigB) opening₁
  secondNorm : semantics.normBounded (2 * params.bigB) opening₂
  crossDifferent :
    ops.scaleAssignment delta₁ opening₂ ≠ ops.scaleAssignment delta₂ opening₁

def phi
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {n : Nat}
    (inputs : Fin n → CE.Instance Structure PublicInput Point Evaluation Commitment) :
    Fin n → Commitment :=
  fun i => (inputs i).commitment

end NightstreamFPrime.Spec.Folding.PiRLC
