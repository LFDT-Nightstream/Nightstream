import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/PiRLCAlgebra/Algebra.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Complete typed Phi81 algebra used by the semantic `Pi_RLC` verifier.

Protocol: SuperNeo `Pi_RLC` at the production global parameters.
Phase: construction of the independently specified verifier algebra.
Constraint family: semantic operations and laws only; this file emits no rows.

Owns: one concrete `PiRLC.Algebra` assembled from the exact challenge set,
complete-assignment action, typed Ajtai map, whole-ring public projection,
all-matrix evaluation action, and production norm-growth theorem.

Does not own: Poseidon2 transcript replay, sampler-to-challenge refinement,
Ajtai/MSIS binding security, production key serialization, PiCCS or PiDEC,
NIFS composition, Rust/R1CS refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: every algebra operation is executable and every law is a
theorem over that operation. The verifier-owned key remains an explicit typed
input. No caller supplies a homomorphism, expansion law, digest, or circuit
acceptance fact.

| Stage path | Algebra field | Concrete owner | Authority class |
|---|---|---|---|
| `nifs.pi_rlc.verify.challenge` | `challengeValid` | `Challenge.challengeValid` | checked predicate |
| `nifs.pi_rlc.verify.assignment_hom` | `combineAssignment` | `PiRLCFinite.combineAssignments` | computed |
| `nifs.pi_rlc.verify.commitment_hom` | `combineCommitment`, `commit_hom` | `Commitment.combineCommitments`, `Commitment.relation_commit_hom` | computed / derived |
| `nifs.pi_rlc.verify.public_input_hom` | `combinePublicInput`, `publicInput_hom` | `PublicInput.combinePublicInputs`, `PublicInput.relation_publicInput_hom` | computed / derived |
| `nifs.pi_rlc.verify.evaluation_hom` | `combineEvaluations`, `evaluations_hom` | `PiRLCFinite.combineEvaluations`, `PiRLCFinite.relation_evaluations_hom` | computed / derived |
| `nifs.pi_rlc.verify.norm_growth` | `norm_growth` | `Norm.relation_norm_growth` | derived |
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Algebra

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding

/-- The complete independent Phi81 `Pi_RLC` algebra for one exact typed Ajtai
key. The verifier-row count remains generic; a production-profile bridge must
later prove the concrete row count and key decoding. -/
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
