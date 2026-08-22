import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction

/-!
Paper authority: SuperNeo v1.1, Section 7.3, `Pi_CCS` input and output.
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

This module is an audit facade over the canonical v1.1 semantics. It defines
no alternate verifier relation and emits no circuit constraints.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.v1_1.Statement

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction

universe uExtension uCommitment uPublicInput

/-- Paper `Eval_K`: one coefficient-complete Pad evaluation. -/
abbrev Eval_K (Extension : Type uExtension) (shape : Shape) :=
  Fin shape.coefficientCount → Extension

/-- Paper `Eval_A`: one coefficient-complete evaluation for every genuine
CCS matrix. `Pad` is not in this index. -/
abbrev Eval_A (Extension : Type uExtension) (shape : Shape) :=
  Fin shape.matrixCount → Fin shape.coefficientCount → Extension

/-- The existing complete v1.1 evaluation carrier. -/
abbrev Evaluation (Extension : Type uExtension) (shape : Shape) :=
  EvaluationFamily Extension shape

/-- Read only the paper's Pad evaluation family. -/
def eval_K
    {Extension : Type uExtension}
    {shape : Shape}
    (evaluation : Evaluation Extension shape) : Eval_K Extension shape :=
  evaluation.pad

/-- Read only the paper's CCS-matrix evaluation family. -/
def eval_A
    {Extension : Type uExtension}
    {shape : Shape}
    (evaluation : Evaluation Extension shape) : Eval_A Extension shape :=
  evaluation.matrix

/-- Named statement-binding conjuncts for the canonical verifier input. -/
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

/-- The canonical verifier input satisfies all three statement-binding
conjuncts by construction. -/
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

/-- The Pad claim is taken from the distinct `Eval_K` statement field. -/
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

/-- Every genuine matrix claim is taken from the distinct `Eval_A` statement
field. -/
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

end NightstreamFPrime.Spec.Folding.PiCCS.v1_1.Statement
