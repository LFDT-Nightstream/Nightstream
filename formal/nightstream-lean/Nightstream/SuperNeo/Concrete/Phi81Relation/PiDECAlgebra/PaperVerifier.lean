import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra
import Nightstream.SuperNeo.Folding.PiDEC.PaperVerifier

/-!
Concrete Phi81 public splitter for the paper-exact `Pi_DEC` verifier.

Protocol: SuperNeo Section 7.5 at production `b = 2`, `k = 14`.
Phase: verifier computation of the fourteen child public inputs.
Constraint family: semantic public-input operations only; this file emits no
rows.

Assurance tier: model-level.

Owns: assembly of the coordinatewise public split, unconditional public
recomposition, the projection-commuting law, and the exact structure-owned
evaluation arity used by `PiDEC.PaperVerifier`.

Does not own: prover messages, acceptance, child CE membership, Ajtai binding,
transcript binding, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PaperVerifier

open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding

/-- The exact public split run by the concrete Section-7.5 verifier. -/
def publicInputSplit {shape : Shape} {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows) :
    PiDEC.PaperVerifier.PublicInputSplit (Algebra.concrete key) where
  split := PublicInput.splitPublicInput
  recompose_split := PublicInput.splitPublicInput_recompose
  split_project := PublicInput.splitPublicInput_project

/-- Phi81 carries exactly one evaluation for every matrix in the fixed
relation shape. -/
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

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PaperVerifier
