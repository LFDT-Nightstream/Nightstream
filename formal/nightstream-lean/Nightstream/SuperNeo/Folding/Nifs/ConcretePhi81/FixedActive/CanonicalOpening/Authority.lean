import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.Context
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority

/-!
Incoming authority for the opening-derived fixed-active context.

Assurance tier: model-level obligation derivation.

Owns: exact reconstruction of an already-canonical parent and child family;
the sole delegated child-source validity premise; derivation of the combined
opening norm, canonical child family, strict public PiDEC, and the existing
active `RunningAuthority.Accepted` record.

Does not own: proof that physical PiCCS rows establish child-source validity,
transcript soundness, an opening-handle serializer, Rust/R1CS refinement,
physical costs, or row removal.

Emits constraints: no.

Authority boundary: strict incoming PiDEC is not assumed. All parent and child
fields are computed from one opening. Once the existing NIFS source relation
validates those fourteen children, their norm facts derive the parent bound
and every public PiDEC equation. Until physical source validation refines to
`ChildSourcesValid`, this theorem cannot authorize deleting production rows.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.opening.reconstruct.parent` | one canonical opening reconstructs every parent field | computed completeness | `OpeningPayload.parent_ofCanonical` |
| `nifs.fixed_active.opening.reconstruct.children` | the same opening reconstructs the ordered child vector | computed completeness | `OpeningPayload.children_ofCanonical` |
| `nifs.fixed_active.opening.result` | every independent active NIFS result has one exact compact representation | semantic completeness | `resultCarrier_complete` |
| `nifs.fixed_active.opening.authority.sources` | every computed running child has its exact split opening | delegated checked fact | `ChildSourcesValid` |
| `nifs.fixed_active.opening.authority.norm` | complete opening is combined-bound | derived | `combinedNorm` |
| `nifs.fixed_active.opening.authority.canonical` | parent and children form the deterministic canonical family | derived | `canonicalChildren` |
| `nifs.fixed_active.opening.authority.pi_dec` | strict incoming PiDEC acceptance | derived/eliminated | `piDecAccepted` |
| `nifs.fixed_active.opening.authority.running` | existing incoming-authority interface is satisfied | derived refinement | `runningAuthority` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.CanonicalParentVerifier
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

private theorem statement_eq_of_fields
    {shape : Phi81Relation.Shape}
    {Commitment : Type}
    (left right : CEStatement shape Commitment)
    (structureEq : left.constraintSystem = right.constraintSystem)
    (commitmentEq : left.commitment = right.commitment)
    (publicInputEq : left.publicInput = right.publicInput)
    (pointEq : left.point = right.point)
    (evaluationsEq : left.evaluations = right.evaluations)
    (stageEq : left.stage = right.stage) :
    left = right := by
  rcases left with
    ⟨leftStructure, leftCommitment, leftPublicInput, leftPoint,
      leftEvaluations, leftStage⟩
  rcases right with
    ⟨rightStructure, rightCommitment, rightPublicInput, rightPoint,
      rightEvaluations, rightStage⟩
  cases structureEq
  cases commitmentEq
  cases publicInputEq
  cases pointEq
  cases evaluationsEq
  cases stageEq
  rfl

namespace OpeningPayload

/-- Canonical compact representation of one already-authorized semantic
parent opening. -/
def ofCanonical
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (parent : CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (assignment : Assignment
      (RelationShape shape publicRingColumns publicFits)) :
    OpeningPayload shape publicRingColumns publicFits where
  point := parent.point
  assignment := assignment

/-- A valid combined opening reconstructs its complete public parent exactly;
no parent field remains an independent caller value. -/
theorem parent_ofCanonical
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {parent : CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)}
    {assignment : Assignment
      (RelationShape shape publicRingColumns publicFits)}
    {children : Fin productionGlobalParams.k ->
      CEStatement (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    (canonical : PiDEC.CanonicalChildren.ForOpening (decAlgebra key)
      parent assignment children) :
    (ofCanonical parent assignment).parent key parent.constraintSystem =
      parent := by
  apply statement_eq_of_fields
  · rfl
  · exact canonical.parentValid.1.1
  · exact canonical.parentValid.1.2.1
  · rfl
  · exact canonical.parentValid.2.2
  · exact canonical.parentCombined.symm

/-- The same compact payload reconstructs the complete canonical child vector
in exact child order. -/
theorem children_ofCanonical
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {parent : CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)}
    {assignment : Assignment
      (RelationShape shape publicRingColumns publicFits)}
    {children : Fin productionGlobalParams.k ->
      CEStatement (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)}
    (canonical : PiDEC.CanonicalChildren.ForOpening (decAlgebra key)
      parent assignment children) :
    (ofCanonical parent assignment).children key parent.constraintSystem =
      children := by
  calc
    (ofCanonical parent assignment).children key parent.constraintSystem =
        PiDEC.childrenOf (decAlgebra key)
          ((ofCanonical parent assignment).parent
            key parent.constraintSystem) assignment := rfl
    _ = PiDEC.childrenOf (decAlgebra key) parent assignment := by
      rw [parent_ofCanonical canonical]
    _ = children := canonical.childrenEq.symm

end OpeningPayload

/-- Every result accepted by the independent fixed-active NIFS relation has
one point-plus-assignment representation that reconstructs both its checked
parent cache and its complete ordered child accumulator exactly.

This is representation completeness only. It does not expose the private
opening publicly, provide hiding, or establish a binding transcript handle. -/
theorem resultCarrier_complete
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      FixedActive.Context shape State publicRingColumns publicFits verifierRows}
    {result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows}
    (accepted : FixedActive.ResultTransition context result) :
    ∃ payload : OpeningPayload shape publicRingColumns publicFits,
      payload.parent context.key result.parent.constraintSystem =
          result.parent ∧
      payload.children context.key result.parent.constraintSystem =
          result.children := by
  rcases FixedActive.ResultTransition.canonicalChildren accepted with
    ⟨assignment, canonical⟩
  exact ⟨OpeningPayload.ofCanonical result.parent assignment,
    OpeningPayload.parent_ofCanonical canonical,
    OpeningPayload.children_ofCanonical canonical⟩

/-- The source-validity facts already required by the selected NIFS relation,
specialized to the computed child family. -/
def ChildSourcesValid
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context : Context shape State publicRingColumns publicFits verifierRows) :
    Prop :=
  forall child,
    CE.Holds (semantics context.key) productionGlobalParams
      (context.input.opening.children context.key context.input.system child)
      ((decAlgebra context.key).splitAssignment
        context.input.opening.assignment child)

theorem combinedNorm
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context : Context shape State publicRingColumns publicFits verifierRows}
    (sources : ChildSourcesValid context) :
    assignmentNormBounded productionGlobalParams.bigB
      context.input.opening.assignment := by
  exact SourceValidated.combinedNorm_of_childHolds sources

theorem canonicalChildren
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context : Context shape State publicRingColumns publicFits verifierRows}
    (sources : ChildSourcesValid context) :
    PiDEC.CanonicalChildren.ForOpening (decAlgebra context.key)
      (context.input.opening.parent context.key context.input.system)
      context.input.opening.assignment
      (context.input.opening.children context.key context.input.system) := by
  exact SourceValidated.canonicalChildren_of_childHolds sources

theorem piDecAccepted
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context : Context shape State publicRingColumns publicFits verifierRows}
    (sources : ChildSourcesValid context) :
    PiDEC.Accepted (decAlgebra context.key) {
      parent := context.input.opening.parent context.key context.input.system
      children := context.input.opening.children context.key context.input.system
    } :=
  (canonicalChildren sources).complete.1

/-- The legacy incoming strict-PiDEC authority interface is a theorem for the
opening-derived carrier; it is not an additional accepted witness. -/
theorem runningAuthority
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context : Context shape State publicRingColumns publicFits verifierRows}
    (sources : ChildSourcesValid context) :
    RunningAuthority.Accepted context.full := by
  apply RunningAuthority.Accepted.active
  exact {
    active := rfl
    parent := context.input.opening.parent context.key context.input.system
    parentBound := context.full_parent
    piDec := by
      simpa [RunningAuthority.attempt, RunningAuthority.children,
        RunningAuthority.activeIndex] using piDecAccepted sources
  }

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening
