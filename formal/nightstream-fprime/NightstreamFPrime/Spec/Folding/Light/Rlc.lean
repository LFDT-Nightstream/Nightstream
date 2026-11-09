import NightstreamFPrime.Spec.Folding.PiRLC

/-! Cyclo (ePrint 2026/359) Figure 3, steps 5 and 6, on SuperNeo v1.2 Section 7.4: the light
Π_RLC combines the fresh source with a sampled `ρ` and the running source with 1, so the
accumulator's norm grows by `T · (b − 1)` per fold. -/

namespace NightstreamFPrime.Spec.Folding.Light.Rlc

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment uScalar uValue uResult

variable {Structure : Type uStructure} {Assignment : Type uAssignment}
  {PublicInput : Type uPublicInput} {Point : Type uPoint} {Evaluation : Type uEvaluation}
  {Commitment : Type uCommitment} {Scalar : Type uScalar}
  {semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams}

/-- SuperNeo v1.2 Definition 21 at an explicit norm bound. -/
def Holds
    (semantics : RelationSemantics Structure Assignment PublicInput Point Evaluation Commitment)
    (bound : Nat)
    (statement : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (assignment : Assignment) : Prop :=
  Opening.Holds semantics bound statement.commitment statement.publicInput assignment ∧
    semantics.evaluationPointValid statement.constraintSystem statement.point ∧
    semantics.evaluations statement.constraintSystem assignment statement.point =
      statement.evaluations

/-- Sources in Π_CCS order, the fresh source first. -/
def pair {Value : Type uValue} (fresh running : Value) : Fin 2 → Value :=
  fun index => if index.val = 0 then fresh else running

structure NormLaw
    (algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (one : Scalar) : Prop where
  add_bound : ∀ (bound : Nat) (challenge : Scalar) (fresh running : Assignment),
    algebra.challengeValid challenge →
    semantics.normBounded params.b fresh →
    semantics.normBounded bound running →
    semantics.normBounded (bound + params.expansionT * (params.b - 1))
      (algebra.combineAssignment (pair challenge one) (pair fresh running))

def output
    (algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (one challenge : Scalar)
    (system : Structure) (point : Point)
    (fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment) :
    CE.Instance Structure PublicInput Point Evaluation Commitment where
  constraintSystem := system
  commitment := algebra.combineCommitment (pair challenge one)
    (pair fresh.commitment running.commitment)
  publicInput := algebra.combinePublicInput (pair challenge one)
    (pair fresh.publicInput running.publicInput)
  point := point
  evaluations := algebra.combineEvaluations (pair challenge one)
    (pair fresh.evaluations running.evaluations)
  stage := .combined

def witness
    (algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (one challenge : Scalar) (fresh running : Assignment) : Assignment :=
  algebra.combineAssignment (pair challenge one) (pair fresh running)

private theorem pair_map {Value : Type uValue} {Result : Type uResult}
    (map : Value → Result) (fresh running : Value) :
    (fun index => map (pair fresh running index)) = pair (map fresh) (map running) := by
  funext index
  by_cases first : index.val = 0 <;> simp [pair, first]

/-- Cyclo Theorem 3, item 1, for one fresh source. -/
theorem complete
    (algebra : PiRLC.Algebra Structure Assignment PublicInput Point Evaluation Commitment Scalar
      semantics params)
    (one : Scalar) (law : NormLaw algebra one)
    (challenge : Scalar) (valid : algebra.challengeValid challenge)
    (system : Structure) (point : Point)
    (fresh running : CE.Instance Structure PublicInput Point Evaluation Commitment)
    (freshWitness runningWitness : Assignment) (bound : Nat)
    (freshSystem : fresh.constraintSystem = system) (freshPoint : fresh.point = point)
    (runningSystem : running.constraintSystem = system) (runningPoint : running.point = point)
    (freshHolds : Holds semantics params.b fresh freshWitness)
    (runningHolds : Holds semantics bound running runningWitness) :
    Holds semantics (bound + params.expansionT * (params.b - 1))
      (output algebra one challenge system point fresh running)
      (witness algebra one challenge freshWitness runningWitness) := by
  rcases freshHolds with ⟨⟨freshCommit, freshPublic, freshNorm⟩, freshValid, freshEvaluations⟩
  rcases runningHolds with
    ⟨⟨runningCommit, runningPublic, runningNorm⟩, _, runningEvaluations⟩
  subst freshSystem freshPoint
  refine ⟨⟨?_, ?_, ?_⟩, freshValid, ?_⟩
  · change semantics.commit (algebra.combineAssignment _ _) = _
    rw [algebra.commit_hom, pair_map semantics.commit, freshCommit, runningCommit]
    rfl
  · change semantics.projectPublicInput (algebra.combineAssignment _ _) = _
    rw [algebra.publicInput_hom, pair_map semantics.projectPublicInput, freshPublic,
      runningPublic]
    rfl
  · exact law.add_bound bound challenge freshWitness runningWitness valid freshNorm runningNorm
  · change semantics.evaluations fresh.constraintSystem (algebra.combineAssignment _ _)
      fresh.point = _
    rw [algebra.evaluations_hom,
      pair_map (fun assignment =>
        semantics.evaluations fresh.constraintSystem assignment fresh.point),
      freshEvaluations, ← runningSystem, ← runningPoint, runningEvaluations]
    rfl

end NightstreamFPrime.Spec.Folding.Light.Rlc
