import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier

/-!
Canonical-parent verification under already-validated NIFS child sources.

Assurance tier: model-level obligation derivation.

Owns: the exact proof that fresh norm validity of every deterministically
split child implies the combined parent norm; construction of the canonical
parent verifier from those existing source facts; and derivation of strict
PiDEC plus every canonical child opening without another parent-norm check.

Does not own: physical child-source validation, raw assignment decoding,
Ajtai/MSIS security, Poseidon2, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: this module does not delete the combined-norm obligation.
It proves that, when the NIFS source relation has already validated all
fourteen computed fresh children, the obligation is owned by those source
norm facts through exact radix recomposition. A verifier that accepts a parent
opening without validating these children must retain the standalone norm
leaf from `CanonicalParentVerifier`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.source_validated_parent.carrier` | parent point is carried; commitment is computed from the opening | computed | `computedCarrier` |
| `nifs.source_validated_parent.children` | all children are the deterministic production split | computed | `computedChildren` |
| `nifs.source_validated_parent.child_norms` | every computed child is fresh-bound | delegated checked fact | `combinedNorm_of_childHolds` |
| `nifs.source_validated_parent.parent_norm` | recomposed opening is combined-bound | derived | `combinedNorm_of_childHolds` |
| `nifs.source_validated_parent.canonical` | parent CE membership, strict PiDEC, and child openings need no duplicate norm check | derived | `accepted_of_childHolds`, `piDecAccepted_and_childHolds` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding

open CanonicalParentVerifier

/-- Compact carrier whose commitment is derived from the same opening that
will materialize its canonical child family. -/
def computedCarrier
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (point : Point shape)
    (assignment : Assignment shape) : Generic.Carrier shape verifierRows where
  point := point
  commitment := PiRLCAlgebra.Commitment.commit key assignment

def computedParent
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (point : Point shape)
    (assignment : Assignment shape) :
    CEStatement shape (PiRLCAlgebra.Commitment.Value verifierRows) :=
  Generic.parent key system (computedCarrier key point assignment) assignment

def computedChildren
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (point : Point shape)
    (assignment : Assignment shape) :
    Fin productionGlobalParams.k ->
      CEStatement shape (PiRLCAlgebra.Commitment.Value verifierRows) :=
  Generic.children key system (computedCarrier key point assignment) assignment

/-- Existing fresh CE membership of every computed child contains all norm
facts needed to derive the combined parent bound. -/
theorem combinedNorm_of_childHolds
    {shape : Shape}
    {verifierRows : Nat}
    {key : PiRLCAlgebra.Commitment.Key shape verifierRows}
    {system : Structure shape}
    {point : Point shape}
    {assignment : Assignment shape}
    (childrenValid : forall child,
      CE.Holds (relationSemantics (PiRLCAlgebra.Commitment.commit key))
        productionGlobalParams
        (computedChildren key system point assignment child)
        ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child)) :
    assignmentNormBounded productionGlobalParams.bigB assignment := by
  have childNorms : forall child,
      assignmentNormBounded productionGlobalParams.b
        ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child) := by
    intro child
    have childNorm := (childrenValid child).1.2.2
    change assignmentNormBounded productionGlobalParams.b
      ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child)
      at childNorm
    exact childNorm
  have recomposed :=
    (PiDECAlgebra.Algebra.concrete key).recompose_norm
      ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment)
      childNorms
  rw [(PiDECAlgebra.Algebra.concrete key).split_recompose assignment]
    at recomposed
  exact recomposed

/-- No separate parent commitment or norm check remains after commitment is
computed and all canonical child sources have already been validated. -/
theorem accepted_of_childHolds
    {shape : Shape}
    {verifierRows : Nat}
    {key : PiRLCAlgebra.Commitment.Key shape verifierRows}
    {system : Structure shape}
    {point : Point shape}
    {assignment : Assignment shape}
    (childrenValid : forall child,
      CE.Holds (relationSemantics (PiRLCAlgebra.Commitment.commit key))
        productionGlobalParams
        (computedChildren key system point assignment child)
        ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child)) :
    Generic.Accepted key system (computedCarrier key point assignment)
      assignment := by
  exact accepted_of_commitment_and_norm rfl
    (combinedNorm_of_childHolds childrenValid)

theorem canonicalChildren_of_childHolds
    {shape : Shape}
    {verifierRows : Nat}
    {key : PiRLCAlgebra.Commitment.Key shape verifierRows}
    {system : Structure shape}
    {point : Point shape}
    {assignment : Assignment shape}
    (childrenValid : forall child,
      CE.Holds (relationSemantics (PiRLCAlgebra.Commitment.commit key))
        productionGlobalParams
        (computedChildren key system point assignment child)
        ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child)) :
    PiDEC.CanonicalChildren.ForOpening
      (PiDECAlgebra.Algebra.concrete key)
      (computedParent key system point assignment) assignment
      (computedChildren key system point assignment) := by
  exact canonicalChildren_of_commitment_and_norm rfl
    (combinedNorm_of_childHolds childrenValid)

/-- Strict public PiDEC and all child openings are consequences of the
already-owned source facts for this computed carrier. -/
theorem piDecAccepted_and_childHolds
    {shape : Shape}
    {verifierRows : Nat}
    {key : PiRLCAlgebra.Commitment.Key shape verifierRows}
    {system : Structure shape}
    {point : Point shape}
    {assignment : Assignment shape}
    (childrenValid : forall child,
      CE.Holds (relationSemantics (PiRLCAlgebra.Commitment.commit key))
        productionGlobalParams
        (computedChildren key system point assignment child)
        ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child)) :
    PiDEC.Accepted (PiDECAlgebra.Algebra.concrete key) {
      parent := computedParent key system point assignment
      children := computedChildren key system point assignment
    } /\
      forall child,
        CE.Holds (relationSemantics (PiRLCAlgebra.Commitment.commit key))
          productionGlobalParams
          (computedChildren key system point assignment child)
          ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child) := by
  exact ⟨(canonicalChildren_of_childHolds childrenValid).complete.1,
    childrenValid⟩

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier.SourceValidated
