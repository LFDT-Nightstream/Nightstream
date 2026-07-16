import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Commitment
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PublicInput

/-!
Complete typed Phi81 algebra used by the semantic `Pi_DEC` verifier.

Protocol: SuperNeo `Pi_DEC` at production `b = 2`, `k = 14`.
Phase: construction of the independently specified verifier algebra.
Constraint family: semantic operations and laws only; this file emits no rows.

Owns: one concrete `PiDEC.Algebra` assembled from deterministic signed-binary
splitting, exact assignment recomposition, typed Ajtai commitment
recomposition, whole-ring public-input recomposition, exact evaluation
recomposition, and both production norm directions.

Does not own: child CE membership, PiCCS or PiRLC acceptance, Ajtai/MSIS
binding security, key generation or serialization, transcript binding, NIFS
composition, Rust/R1CS refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: every algebra operation is executable and every law is a
theorem over that operation. The verifier-owned typed Ajtai key is an explicit
input. No caller supplies a split, recomposition equation, homomorphism, norm
law, digest, or circuit acceptance fact.

| Stage path | Algebra field | Concrete owner | Authority class |
|---|---|---|---|
| `nifs.pi_dec.verify.radix.split` | `splitAssignment` | `Radix.splitAssignment` | computed |
| `nifs.pi_dec.verify.radix.recompose` | `recomposeAssignment`, `split_recompose` | `Radix.recomposeAssignment`, `Radix.split_recompose` | computed / derived |
| `nifs.pi_dec.verify.radix.norm` | `split_norm`, `recompose_norm` | `Radix.split_norm`, `Radix.recompose_norm` | derived |
| `nifs.pi_dec.verify.commitment_hom` | `recomposeCommitment`, `commit_hom` | `Commitment.recomposeCommitment`, `Commitment.relation_commit_hom` | computed / derived |
| `nifs.pi_dec.verify.public_input_hom` | `recomposePublicInput`, `publicInput_hom` | `PublicInput.recomposePublicInput`, `PublicInput.relation_publicInput_hom` | computed / derived |
| `nifs.pi_dec.verify.evaluation_hom` | `recomposeEvaluations`, `evaluations_hom` | `EvaluationHomomorphism.PiDEC.recomposeEvaluations`, `relation_evaluations_hom` | computed / derived |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding

/-- The complete independent Phi81 `Pi_DEC` algebra for one exact typed Ajtai
key. The verifier-row count remains generic; a production bridge must later
prove the concrete row count, key decoding, and binding assumptions. -/
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

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra
