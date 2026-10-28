import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Algebra
import NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier

/-! SuperNeo v1.2 Section 7.5. -/

namespace NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PaperVerifier

open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Folding

/-- SuperNeo v1.2 Section 7.5. -/
def publicInputSplit {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    PiDEC.PaperVerifier.PublicInputSplit (Algebra.concrete key) where
  parentBounded := PublicInput.parentBounded
  parentBounded_decidable := PublicInput.parentBounded_decidable
  parentBounded_project := PublicInput.parentBounded_project
  split := PublicInput.splitPublicInput
  recompose_split := PublicInput.splitPublicInput_recompose
  split_project := PublicInput.splitPublicInput_project

def evaluationArity {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    PiDEC.PaperVerifier.EvaluationArity
      (Phi81Relation.relationSemantics
        (PiRLCAlgebra.Commitment.commit key)) where
  count := fun _ => shape.matrixCount
  evaluations_size := Phi81Relation.evaluations_size

@[simp] theorem publicInputSplit_split {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    (publicInputSplit key).split = PublicInput.splitPublicInput := by
  rfl

@[simp] theorem evaluationArity_count {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape) :
    (evaluationArity key).count system = shape.matrixCount := by
  rfl

theorem accepted_child_publicInput_eq_boundedDigit
    {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (attempt : PiDEC.PaperVerifier.Attempt
      (Phi81Relation.Structure shape) (Phi81Relation.PublicInput shape)
      (Phi81Relation.Point shape) Phi81Relation.Evaluation
      (PiRLCAlgebra.Commitment.Value verifierRows) productionGlobalParams)
    (accepted : PiDEC.PaperVerifier.Accepted (Algebra.concrete key)
      (publicInputSplit key) (evaluationArity key) attempt)
    (child : Radix.ChildIndex)
    (column : Fin shape.publicWidth) :
    (PiDEC.PaperVerifier.children (publicInputSplit key) attempt child).publicInput
        column =
      Radix.boundedDigit (attempt.parent.publicInput column) child := by
  exact PublicInput.splitPublicInput_eq_boundedDigit
    attempt.parent.publicInput accepted.parentBounded child column

end NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PaperVerifier
