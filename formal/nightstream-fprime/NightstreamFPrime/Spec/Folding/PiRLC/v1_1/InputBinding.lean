import NightstreamFPrime.Spec.Folding.PiRLC

/-!
Paper authority: SuperNeo v1.1, Section 7.4, PiRLC input and output.
Obligation: All 17 PiRLC inputs are fresh CE claims for one structure and one
evaluation point.

This predicate is exactly the `inputFresh`, `sameStructure`, and `samePoint`
prefix of `PiRLC.Equations`. It does not define another PiRLC relation.
-/

namespace NightstreamFPrime.Spec.Folding.PiRLC.v1_1.InputBinding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar

/-- The three input-binding fields of the exact PiRLC verifier relation. -/
structure Holds
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {n : Nat}
    (inputs : Fin n →
      CE.Instance Structure PublicInput Point Evaluation Commitment)
    (system : Structure)
    (point : Point) : Prop where
  inputFresh : ∀ i, (inputs i).stage = .fresh
  sameStructure : ∀ i, (inputs i).constraintSystem = system
  samePoint : ∀ i, (inputs i).point = point

/-- Mechanical parent coverage: this leaf supplies exactly the first three
fields of `PiRLC.Equations`; later leaves supply the four output equations. -/
def Holds.toEquations
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
      Structure Assignment PublicInput Point Evaluation Commitment Scalar
        semantics params}
    {attempt : Attempt
      Structure PublicInput Point Evaluation Commitment Scalar params arity}
    (binding : Holds attempt.inputs attempt.output.constraintSystem
      attempt.output.point)
    (outputCombined : attempt.output.stage = .combined)
    (commitmentEquation :
      attempt.output.commitment =
        algebra.combineCommitment attempt.challenges
          (fun i => (attempt.inputs i).commitment))
    (publicInputEquation :
      attempt.output.publicInput =
        algebra.combinePublicInput attempt.challenges
          (fun i => (attempt.inputs i).publicInput))
    (evaluationEquation :
      attempt.output.evaluations =
        algebra.combineEvaluations attempt.challenges
          (fun i => (attempt.inputs i).evaluations)) :
    Equations algebra attempt where
  inputFresh := binding.inputFresh
  sameStructure := binding.sameStructure
  samePoint := binding.samePoint
  outputCombined := outputCombined
  commitmentEquation := commitmentEquation
  publicInputEquation := publicInputEquation
  evaluationEquation := evaluationEquation

end NightstreamFPrime.Spec.Folding.PiRLC.v1_1.InputBinding
