import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra

/-!
Focused theorem-surface checks for typed Phi81 Π_DEC public recomposition.

| Stage path | Regression |
|---|---|
| `nifs.pi_dec.verify.commitment_hom.scale` | base scaling commutes with the typed Ajtai commitment |
| `nifs.pi_dec.verify.commitment_hom.radix` | fourteen public commitments use the exact production radix weights |
| `nifs.pi_dec.verify.commitment_hom.algebra` | theorem matches the `PiDEC.Algebra.commit_hom` field shape |
| `nifs.pi_dec.verify.public_input_hom.scale` | base scaling commutes with the authoritative public projection |
| `nifs.pi_dec.verify.public_input_hom.radix` | fourteen public inputs use the exact production radix weights |
| `nifs.pi_dec.verify.public_input_hom.algebra` | theorem matches the `PiDEC.Algebra.publicInput_hom` field shape |
| `nifs.pi_dec.verify.algebra` | all independently proved operations and laws assemble into the exact concrete algebra |
-/

namespace tests.Phi81PiDECRecomposition

open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra

#check Commitment.commitmentScale
#check Commitment.combineCommitments
#check Commitment.recomposeCommitment
#check Commitment.commit_scale
#check Commitment.commit_combine
#check Commitment.commit_recompose
#check Commitment.relation_commit_hom

#check PublicInput.publicInputScale
#check PublicInput.combinePublicInputs
#check PublicInput.recomposePublicInput
#check PublicInput.projectPublicInput_scale
#check PublicInput.projectPublicInput_combine
#check PublicInput.projectPublicInput_recompose
#check PublicInput.relation_publicInput_hom
#check EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
#check EvaluationHomomorphism.PiDEC.raw_recomposeAssignment_eq
#check Algebra.concrete
#check Algebra.concrete_splitAssignment
#check Algebra.concrete_recomposeAssignment

/-! These applications fail to elaborate if either exported theorem drifts
from the exact algebra-field quantifier and relation-semantics boundary. -/

example {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (assignments : Radix.ChildIndex -> Assignment shape) :
    (relationSemantics (PiRLCAlgebra.Commitment.commit key)).commit
        (Radix.recomposeAssignment assignments) =
      Commitment.recomposeCommitment fun index =>
        (relationSemantics
          (PiRLCAlgebra.Commitment.commit key)).commit
            (assignments index) :=
  Commitment.relation_commit_hom key assignments

example {shape : Shape} {CommitmentType : Type}
    (commit : Assignment shape -> CommitmentType)
    (assignments : Radix.ChildIndex -> Assignment shape) :
    (relationSemantics commit).projectPublicInput
        (Radix.recomposeAssignment assignments) =
      PublicInput.recomposePublicInput fun index =>
        (relationSemantics commit).projectPublicInput (assignments index) :=
  PublicInput.relation_publicInput_hom commit assignments

example {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    (Algebra.concrete key).recomposeCommitment =
      Commitment.recomposeCommitment := by
  rfl

example {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    (Algebra.concrete key).recomposePublicInput =
      PublicInput.recomposePublicInput := by
  rfl

end tests.Phi81PiDECRecomposition
