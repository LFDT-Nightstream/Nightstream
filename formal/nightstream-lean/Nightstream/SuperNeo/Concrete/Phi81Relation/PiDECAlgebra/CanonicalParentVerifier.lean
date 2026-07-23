import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra
import Nightstream.SuperNeo.Folding.PiDEC.CanonicalChildren.OpeningVerifier

/-!
Two-check canonical-parent verifier for the typed Phi81 relation.

Assurance tier: model-level obligation exactness.

Owns: specialization of the generic canonical-opening verifier to the
independent Phi81 relation; proof that typed points eliminate the generic
runtime point check; exact equivalence between parent CE membership and the
remaining commitment and combined-norm checks; an executable finite checker
returning only the computed parent and children; and derivation of the
complete canonical PiDEC child family.

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
| `nifs.pi_dec.canonical_opening.phi81.execute` | finite commitment coefficients and assignment coordinates implement exactly the two checks | computed/derived | `run?_eq_some_iff_accepted` |
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

/-! ## Executable canonical verifier -/

/-- The complete deterministic output of the canonical-parent verifier.
Neither the parent fields nor the ordered children are prover inputs. -/
structure Output
    (shape : Shape) (verifierRows : Nat) where
  parent : CEStatement shape (PiRLCAlgebra.Commitment.Value verifierRows)
  children : Fin productionGlobalParams.k ->
    CEStatement shape (PiRLCAlgebra.Commitment.Value verifierRows)

/-- Materialize the deterministic result independently of the two checks. -/
def output
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) : Output shape verifierRows where
  parent := Generic.parent key system carrier assignment
  children := Generic.children key system carrier assignment

/-- Executable pointwise commitment equality. Function equality is not used
as an execution oracle. -/
def commitmentMatches
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) : Bool :=
  (List.finRange verifierRows).all fun row =>
    (List.finRange ringDegree).all fun coefficient =>
      decide (PiRLCAlgebra.Commitment.commit key assignment row coefficient =
        carrier.commitment row coefficient)

/-- Executable combined-norm check over the complete typed assignment. -/
def combinedNormMatches
    {shape : Shape}
    (assignment : Assignment shape) : Bool :=
  (List.finRange shape.carrierWidth).all fun column =>
    decide (centeredMagnitude (assignment column) <
      productionGlobalParams.bigB)

/-- The canonical semantic verifier has exactly two runtime checks. -/
def verify
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) : Bool :=
  commitmentMatches key carrier assignment &&
    combinedNormMatches assignment

/-- Return the computed parent and ordered child family only after both
semantic checks succeed. -/
def run?
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) : Option (Output shape verifierRows) :=
  if verify key carrier assignment then
    some (output key system carrier assignment)
  else
    none

theorem commitmentMatches_eq_true_iff
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    commitmentMatches key carrier assignment = true <->
      PiRLCAlgebra.Commitment.commit key assignment = carrier.commitment := by
  constructor
  · intro checked
    apply funext
    intro row
    apply funext
    intro coefficient
    exact of_decide_eq_true ((List.all_eq_true.mp
      ((List.all_eq_true.mp checked) row (List.mem_finRange row)))
        coefficient (List.mem_finRange coefficient))
  · intro equal
    apply List.all_eq_true.mpr
    intro row _
    apply List.all_eq_true.mpr
    intro coefficient _
    exact decide_eq_true (congrFun (congrFun equal row) coefficient)

theorem combinedNormMatches_eq_true_iff
    {shape : Shape}
    (assignment : Assignment shape) :
    combinedNormMatches assignment = true <->
      assignmentNormBounded productionGlobalParams.bigB assignment := by
  simp [combinedNormMatches, assignmentNormBounded, List.all_eq_true,
    decide_eq_true_eq]

theorem verify_eq_true_iff
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    verify key carrier assignment = true <->
      PiRLCAlgebra.Commitment.commit key assignment = carrier.commitment /\
      assignmentNormBounded productionGlobalParams.bigB assignment := by
  simp [verify, commitmentMatches_eq_true_iff,
    combinedNormMatches_eq_true_iff]

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

/-- Executable success is exactly the independent canonical-parent
acceptance predicate. The returned parent and children are computed outputs,
not caller-supplied claims. -/
theorem run?_eq_some_iff_accepted
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    run? key system carrier assignment =
        some (output key system carrier assignment) <->
      Accepted key system carrier assignment := by
  rw [accepted_iff_commitment_and_norm]
  simp [run?, verify_eq_true_iff]

/-- Executable success is also exactly CE membership of the deterministically
materialized parent. -/
theorem run?_eq_some_iff_parentHolds
    {shape : Shape}
    {verifierRows : Nat}
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Structure shape)
    (carrier : Carrier shape verifierRows)
    (assignment : Assignment shape) :
    run? key system carrier assignment =
        some (output key system carrier assignment) <->
      CE.Holds
        (relationSemantics (PiRLCAlgebra.Commitment.commit key))
        productionGlobalParams
        (Generic.parent key system carrier assignment) assignment := by
  rw [run?_eq_some_iff_accepted]
  exact PiDEC.CanonicalChildren.OpeningVerifier.accepted_iff_parentHolds
    (relationSemantics (PiRLCAlgebra.Commitment.commit key))
    productionGlobalParams system carrier assignment

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
