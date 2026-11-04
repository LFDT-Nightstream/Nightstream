import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Commitment
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PublicInput

namespace NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Algebra

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding

def concrete {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    PiDEC.Algebra
      (Structure shape)
      (Assignment shape)
      (PublicInput shape)
      (Point shape)
      Evaluation
      (PiRLCAlgebra.Commitment.Value verifierRows)
      (relationSemantics (PiRLCAlgebra.Commitment.commit key))
      productionGlobalParams where
  splitAssignment := Radix.splitAssignment
  recomposeAssignment := Radix.recomposeAssignment
  recomposeCommitment := Commitment.recomposeCommitment
  recomposePublicInput := PublicInput.recomposePublicInput
  recomposeEvaluations := EvaluationHomomorphism.PiDEC.recomposeEvaluations
  split_recompose := Radix.split_recompose
  split_norm := Radix.split_norm
  recompose_norm := Radix.recompose_norm
  commit_hom := Commitment.relation_commit_hom key
  publicInput_hom := PublicInput.relation_publicInput_hom
    (PiRLCAlgebra.Commitment.commit key)
  evaluations_hom :=
    EvaluationHomomorphism.PiDEC.relation_evaluations_hom
      (PiRLCAlgebra.Commitment.commit key)

@[simp] theorem concrete_splitAssignment
    {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    (concrete key).splitAssignment = Radix.splitAssignment := by
  rfl

@[simp] theorem concrete_recomposeAssignment
    {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    (concrete key).recomposeAssignment = Radix.recomposeAssignment := by
  rfl

end NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Algebra
