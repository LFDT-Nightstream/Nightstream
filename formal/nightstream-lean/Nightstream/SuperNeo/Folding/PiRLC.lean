import Nightstream.SuperNeo.Folding.BatchArity

/-!
Model-level Π_RLC reduction (SuperNeo Lemma 4).

This file separates three claims which must not be conflated:

1. linear-combination completeness into `CE(B)`;
2. weak extraction into the ambient input relation for the selected batch; and
3. uniqueness of two extracted input witnesses, except for the paper's
   `(2B, C)` relaxed-binding collision.

There is intentionally no standalone Π_RLC knowledge-soundness theorem.
-/

namespace Nightstream.SuperNeo.Folding.PiRLC

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar

/-- Algebraic laws used by the verifier's single, shared linear combination. -/
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
  /-- Definition 14's verifier-owned arity cap and strong challenge set imply
  that combining fresh witnesses lands strictly below `B = b^k`. -/
  norm_growth : ∀ {n : Nat}
      (_totalBound : n ≤ params.maxFresh + params.k)
      (challenges : Fin n → Scalar)
      (assignments : Fin n → Assignment),
    (∀ i, challengeValid (challenges i)) →
    (∀ i, semantics.normBounded params.b (assignments i)) →
    semantics.normBounded params.bigB
      (combineAssignment challenges assignments)

/-- Public Π_RLC input/output at one production batch arity. -/
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

/-- Exact verifier equations for Π_RLC. Every public component uses the same challenges. -/
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
      Structure PublicInput Point Evaluation Commitment Scalar params arity) : Prop where
  inputFresh : ∀ i, (attempt.inputs i).stage = .fresh
  sameStructure : ∀ i,
    (attempt.inputs i).constraintSystem = attempt.output.constraintSystem
  samePoint : ∀ i, (attempt.inputs i).point = attempt.output.point
  challengesValid : ∀ i, algebra.challengeValid (attempt.challenges i)
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

/-- The verifier-computed combined CE statement. -/
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

/-- The prover uses the same challenges to combine witnesses. -/
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

/-- Honest fresh inputs combine to an actual `CE(B)` opening. -/
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
        simpa [inputFresh i] using inputNorm)
  · exact (algebra.evaluations_hom system point challenges assignments).trans
      (congrArg (algebra.combineEvaluations challenges) evaluationsAgree)

/-- Perfect completeness: verifier equations and the combined relation both hold. -/
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

/-- Interpret an extracted Π_RLC input witness in `CE(q/2)`. -/
def ambientInput
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (statement : CE.Instance Structure PublicInput Point Evaluation Commitment) :
    CE.Instance Structure PublicInput Point Evaluation Commitment :=
  { statement with stage := .ambient }

/-- The weak extractor's successful postcondition, not fresh relation membership. -/
def AmbientOpenings
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    {n : Nat}
    (inputs : Fin n → CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignments : Fin n → Assignment) : Prop :=
  ∀ i, CE.Holds semantics params (ambientInput (inputs i)) (assignments i)

/-- The `(K+k)+1` singular/repeated challenge events in Appendix D.5. -/
inductive SamplingFailure (n : Nat) where
  | baseFork
  | coordinateFork (index : Fin n)
deriving Repr

def samplingErrorNumerator (n : Nat) : Nat := n + 1

/-- A concrete extractor supplies the actual failure predicate for its fixed
sampling schedule. Keeping it as a proposition prevents the named reason type
from making the bad event vacuously inhabited. -/
structure SamplingBoundary (n : Nat) where
  Failure : Prop
  classify : Failure → SamplingFailure n

structure ExtractedAmbient
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    {n : Nat}
    (inputs : Fin n → CE.Instance Structure PublicInput Point Evaluation Commitment) where
  assignments : Fin n → Assignment
  valid : AmbientOpenings semantics params inputs assignments

/-- Honest statement of the Π_RLC extraction boundary. -/
inductive ExtractionOutcome
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    (semantics : RelationSemantics
      Structure Assignment PublicInput Point Evaluation Commitment)
    (params : GlobalParams)
    {n : Nat}
    (inputs : Fin n → CE.Instance Structure PublicInput Point Evaluation Commitment)
    (sampling : SamplingBoundary n) where
  | extracted (result : ExtractedAmbient semantics params inputs)
  | failed (evidence : sampling.Failure)

/-- Exact algebra needed to state Definition 4's relaxed-binding game. -/
structure RelaxedBindingOps
    (Assignment : Type uAssignment)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar) where
  scaleAssignment : Scalar → Assignment → Assignment
  scaleCommitment : Scalar → Commitment → Commitment
  differenceChallenge : Scalar → Prop

/-- A literal `(2B, C)`-relaxed binding collision from Definition 4. -/
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

/-- The Appendix D.5 rewinding/algebra bridge, isolated from deterministic verifier logic. -/
structure UniquenessBridge
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
    {n : Nat} where
  disagreement_to_collision :
    ∀ (leftInputs rightInputs : Fin n →
        CE.Instance Structure PublicInput Point Evaluation Commitment)
      (leftAssignments rightAssignments : Fin n → Assignment),
      (fun i => (leftInputs i).commitment) =
        (fun i => (rightInputs i).commitment) →
      AmbientOpenings semantics params leftInputs leftAssignments →
      AmbientOpenings semantics params rightInputs rightAssignments →
      leftAssignments ≠ rightAssignments →
      ∃ i, Nonempty
        (RelaxedBindingCollision semantics params ops (leftInputs i).commitment)

/-- The weak reduction's `φ`: the uncombined vector of input commitments. -/
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

/-- Two successful weak extractions at the same `φ` are unique or break relaxed binding. -/
theorem same_phi_extractions_unique_or_collision
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
    {n : Nat}
    (bridge : UniquenessBridge semantics params ops (n := n))
    (leftInputs rightInputs : Fin n →
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (leftAssignments rightAssignments : Fin n → Assignment)
    (samePhi : phi leftInputs = phi rightInputs)
    (leftValid : AmbientOpenings semantics params leftInputs leftAssignments)
    (rightValid : AmbientOpenings semantics params rightInputs rightAssignments) :
    leftAssignments = rightAssignments ∨
      ∃ i, Nonempty
        (RelaxedBindingCollision semantics params ops (leftInputs i).commitment) := by
  by_cases sameAssignments : leftAssignments = rightAssignments
  · exact Or.inl sameAssignments
  · exact Or.inr (bridge.disagreement_to_collision leftInputs rightInputs
      leftAssignments rightAssignments samePhi leftValid rightValid sameAssignments)

end Nightstream.SuperNeo.Folding.PiRLC
