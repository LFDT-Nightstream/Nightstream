import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra
import Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier

/-!
Two-check canonical-parent verifier for the typed Phi81 relation.

Assurance tier: model-level obligation exactness.

Owns: specialization of the generic canonical-opening verifier to the
independent Phi81 relation; proof that typed points eliminate the generic
runtime point check; exact equivalence between parent CE membership and the
remaining commitment and combined-norm checks; and derivation of the complete
canonical PiDEC child family.

Does not own: raw point decoding, the Ajtai/MSIS binding reduction, a concrete
key artifact, Poseidon2, Rust/R1CS refinement, materialization costs, or row
removal.

Emits constraints: no.

Authority boundary: structure, shape, parameters, and the commitment key are
verifier-owned. The private assignment is accepted only when it commits to the
carried parent commitment and is `B`-bounded. Point dimension is a type fact.
Public input, evaluations, stage, and children are computed; lowering those
computations may still emit constraints.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.canonical_opening.phi81.commitment` | exact typed Ajtai opening matches the parent commitment | checked | `Accepted.commitment` |
| `nifs.pi_dec.canonical_opening.phi81.norm` | complete typed opening is strictly `B`-bounded (270 coordinates at the production shape) | checked | `Accepted.combinedNorm` |
| `nifs.pi_dec.canonical_opening.phi81.point` | point has exactly `rowVariables` extension coordinates | type-derived | `accepted_of_commitment_and_norm` |
| `nifs.pi_dec.canonical_opening.phi81.parent` | CE membership is exactly the two retained checks | derived | `parentHolds_iff_commitment_and_norm` |
| `nifs.pi_dec.canonical_opening.phi81.children` | accepted opening determines every fresh child | derived | `canonicalChildren_of_commitment_and_norm` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding

namespace Generic

abbrev Carrier (shape : Shape) (verifierRows : Nat) :=
  PiDEC.CanonicalChildren.OpeningVerifier.Input
    (Point shape) (PiRLCAlgebra.Commitment.Value verifierRows)

abbrev Accepted
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :=
  PiDEC.CanonicalChildren.OpeningVerifier.Accepted
    (relationSemantics (PiRLCAlgebra.Commitment.commit key))
    productionGlobalParams system carrier assignment

def parent
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    CEStatement shape (PiRLCAlgebra.Commitment.Value verifierRows) :=
  PiDEC.CanonicalChildren.OpeningVerifier.parent
    (relationSemantics (PiRLCAlgebra.Commitment.commit key))
    system carrier assignment

def children
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    Fin productionGlobalParams.k →
      CEStatement shape (PiRLCAlgebra.Commitment.Value verifierRows) :=
  PiDEC.CanonicalChildren.OpeningVerifier.children
    (PiDECAlgebra.Algebra.concrete key) system carrier assignment

end Generic

open Generic

/-- Typed point validity is intrinsic, so commitment equality and the
combined norm bound construct the complete generic acceptance record. -/
theorem accepted_of_commitment_and_norm
    {shape : Shape}
    {verifierRows : Nat}
    {key : PiRLCAlgebra.Commitment.Key shape verifierRows}
    {system : Structure shape}
    {carrier : Carrier shape verifierRows}
    {assignment : Assignment shape}
    (commitment :
      PiRLCAlgebra.Commitment.commit key assignment = carrier.commitment)
    (combinedNorm :
      assignmentNormBounded productionGlobalParams.bigB assignment) :
    Accepted key system carrier assignment := {
  commitment := commitment
  combinedNorm := combinedNorm
  pointValid := evaluationPointValid_holds system carrier.point
}

/-- The generic three-check record contains no additional Phi81 obligation. -/
theorem accepted_iff_commitment_and_norm
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    Accepted key system carrier assignment ↔
      PiRLCAlgebra.Commitment.commit key assignment = carrier.commitment ∧
      assignmentNormBounded productionGlobalParams.bigB assignment := by
  constructor
  · intro accepted
    exact ⟨accepted.commitment, accepted.combinedNorm⟩
  · rintro ⟨commitment, combinedNorm⟩
    exact accepted_of_commitment_and_norm commitment combinedNorm

/-- Exact Phi81 parent CE membership after all derivable fields are computed. -/
theorem parentHolds_iff_commitment_and_norm
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    CE.Holds
      (relationSemantics (PiRLCAlgebra.Commitment.commit key))
      productionGlobalParams (parent key system carrier assignment) assignment ↔
      PiRLCAlgebra.Commitment.commit key assignment = carrier.commitment ∧
      assignmentNormBounded productionGlobalParams.bigB assignment := by
  rw [← accepted_iff_commitment_and_norm key system carrier assignment]
  exact (PiDEC.CanonicalChildren.OpeningVerifier.accepted_iff_parentHolds
    (relationSemantics (PiRLCAlgebra.Commitment.commit key))
    productionGlobalParams system carrier assignment).symm

/-- The two retained checks derive the exact canonical child family required
for a parent-only accumulator carrier. -/
theorem canonicalChildren_of_commitment_and_norm
    {shape : Shape}
    {verifierRows : Nat}
    {key : PiRLCAlgebra.Commitment.Key shape verifierRows}
    {system : Structure shape}
    {carrier : Carrier shape verifierRows}
    {assignment : Assignment shape}
    (commitment :
      PiRLCAlgebra.Commitment.commit key assignment = carrier.commitment)
    (combinedNorm :
      assignmentNormBounded productionGlobalParams.bigB assignment) :
    PiDEC.CanonicalChildren.ForOpening
      (PiDECAlgebra.Algebra.concrete key)
      (parent key system carrier assignment) assignment
      (children key system carrier assignment) := by
  exact PiDEC.CanonicalChildren.OpeningVerifier.canonicalChildren
    (PiDECAlgebra.Algebra.concrete key) system carrier assignment
    (accepted_of_commitment_and_norm commitment combinedNorm)

/-- Strict public PiDEC and every fresh child opening are consequences of the
same two checks; neither is a third independent verifier family. -/
theorem piDecAccepted_and_childHolds_of_commitment_and_norm
    {shape : Shape}
    {verifierRows : Nat}
    {key : PiRLCAlgebra.Commitment.Key shape verifierRows}
    {system : Structure shape}
    {carrier : Carrier shape verifierRows}
    {assignment : Assignment shape}
    (commitment :
      PiRLCAlgebra.Commitment.commit key assignment = carrier.commitment)
    (combinedNorm :
      assignmentNormBounded productionGlobalParams.bigB assignment) :
    PiDEC.Accepted (PiDECAlgebra.Algebra.concrete key) {
      parent := parent key system carrier assignment
      children := children key system carrier assignment
    } ∧
      ∀ child,
        CE.Holds
          (relationSemantics (PiRLCAlgebra.Commitment.commit key))
          productionGlobalParams
          (children key system carrier assignment child)
          ((PiDECAlgebra.Algebra.concrete key).splitAssignment assignment child) :=
  PiDEC.CanonicalChildren.OpeningVerifier.piDecAccepted_and_childHolds
    (PiDECAlgebra.Algebra.concrete key) system carrier assignment
    (accepted_of_commitment_and_norm commitment combinedNorm)

end Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier
