import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput

/-!
Complete typed Phi81 algebra used by the semantic `Pi_RLC` verifier.

Protocol: SuperNeo `Pi_RLC` at the production global parameters.

Owns: one concrete `PiRLC.Algebra` assembled from the exact challenge set,
complete-assignment action, typed Ajtai map, whole-ring public projection,
all-matrix evaluation action, and production norm-growth theorem.

Does not own: Poseidon2 transcript replay, sampler-to-challenge refinement,
Ajtai/MSIS binding security, production key serialization, PiCCS or PiDEC,
NIFS composition, Rust/R1CS refinement, row removal, or constraint counts.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Algebra

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding

def concrete {shape : Shape} {verifierRows : Nat}
    (key : Commitment.Key shape verifierRows) :
    PiRLC.Algebra
      (Structure shape)
      (Assignment shape)
      (PublicInput shape)
      (Point shape)
      Evaluation
      (Commitment.Value verifierRows)
      RingF
      (relationSemantics (Commitment.commit key))
      productionGlobalParams where
  challengeValid := Challenge.challengeValid
  combineAssignment := PiRLCFinite.combineAssignments
  combineCommitment := Commitment.combineCommitments
  combinePublicInput := PublicInput.combinePublicInputs
  combineEvaluations := PiRLCFinite.combineEvaluations (shape := shape)
  commit_hom := by
    intro count challenges assignments
    exact Commitment.relation_commit_hom key challenges assignments
  publicInput_hom := by
    intro count challenges assignments
    exact PublicInput.relation_publicInput_hom
      (Commitment.commit key) challenges assignments
  evaluations_hom := by
    intro count system point challenges assignments
    exact PiRLCFinite.relation_evaluations_hom
      (Commitment.commit key) system point challenges assignments
  norm_growth := by
    intro count totalBound challenges assignments challengesValid assignmentsFresh
    exact Norm.relation_norm_growth
      (Commitment.commit key) totalBound challenges assignments
        challengesValid assignmentsFresh

@[simp] theorem concrete_challengeValid
    {shape : Shape} {verifierRows : Nat}
    (key : Commitment.Key shape verifierRows) :
    (concrete key).challengeValid = Challenge.challengeValid := by
  rfl

@[simp] theorem concrete_combineAssignment
    {shape : Shape} {verifierRows count : Nat}
    (key : Commitment.Key shape verifierRows)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    (concrete key).combineAssignment challenges assignments =
      PiRLCFinite.combineAssignments challenges assignments := by
  rfl

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Algebra
