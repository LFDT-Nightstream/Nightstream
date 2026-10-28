import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction

/-!
Paper authority: SuperNeo v1.2, Section 7.3, `Pi_CCS` input and output.
Obligation: Bind the prior evaluation point and keep `Eval_K` (Pad) separate
from `Eval_A` (all CCS matrices).

Inputs:
- the existing public `StrongReduction.Statement`;
- the existing executable `ProtocolPolynomial.VerifierInput`.

Outputs:
- a named binding predicate for the prior point, `Eval_K`, and `Eval_A`.

Parent coverage:
- `ProtocolPolynomial.VerifierInput.priorPoint`;
- `ProtocolPolynomial.VerifierInput.claimedPadCoefficient`;
- `ProtocolPolynomial.VerifierInput.claimedMatrixCoefficient`.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.Statement

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction

universe uExtension uCommitment uPublicInput

abbrev Eval_K (Extension : Type uExtension) (shape : Shape) :=
  Fin shape.coefficientCount → Extension

abbrev Eval_A (Extension : Type uExtension) (shape : Shape) :=
  Fin shape.matrixCount → Fin shape.coefficientCount → Extension

abbrev Evaluation (Extension : Type uExtension) (shape : Shape) :=
  EvaluationFamily Extension shape

def eval_K
    {Extension : Type uExtension}
    {shape : Shape}
    (evaluation : Evaluation Extension shape) : Eval_K Extension shape :=
  evaluation.pad

def eval_A
    {Extension : Type uExtension}
    {shape : Shape}
    (evaluation : Evaluation Extension shape) : Eval_A Extension shape :=
  evaluation.matrix

structure Holds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (statement : StrongReduction.Statement Extension Commitment PublicInput
      shape columns blockCount baseOps)
    (input : ProtocolPolynomial.VerifierInput Extension shape) : Prop where
  priorPoint : input.priorPoint = statement.priorPoint
  eval_K : input.claimedPadCoefficient = statement.claimedPadCoefficient
  eval_A : input.claimedMatrixCoefficient = statement.claimedMatrixCoefficient

theorem verifierInput_holds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (lift : F → Extension)
    (statement : StrongReduction.Statement Extension Commitment PublicInput
      shape columns blockCount baseOps) :
    Holds statement (statement.verifierInput lift) := by
  exact ⟨rfl, rfl, rfl⟩

theorem verifierInput_eval_K
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (lift : F → Extension)
    (statement : StrongReduction.Statement Extension Commitment PublicInput
      shape columns blockCount baseOps)
    (coordinate : PadCoordinate shape) :
    (statement.verifierInput lift).claimedPadCoefficient coordinate =
      statement.claimedPadCoefficient coordinate := by
  rfl

theorem verifierInput_eval_A
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {baseOps : InterpolationOps F}
    (lift : F → Extension)
    (statement : StrongReduction.Statement Extension Commitment PublicInput
      shape columns blockCount baseOps)
    (coordinate : MatrixCoordinate shape) :
    (statement.verifierInput lift).claimedMatrixCoefficient coordinate =
      statement.claimedMatrixCoefficient coordinate := by
  rfl

end NightstreamFPrime.Spec.Folding.PiCCS.Statement
