import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics
import Nightstream.SuperNeo.Folding.PiCCS
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters

/-!
Typed public products shared by the Split-NC `Pi_CCS` input and output
authority modules.

Protocol: SuperNeo `Pi_CCS`.
Phase: public source product and fresh CE output product.
Constraint family: carrier types only; this file emits no rows.

Owns: the batch-invariant concrete Phi81 relation shape; the production
fresh/running input product type; and the complete CE output product type.

Does not own: source indexing, field authority, assignments, transcripts,
materialization, relation membership, Rust, R1CS, rows, or costs.

Emits constraints: no.

Authority boundary: these aliases fix the exact concrete relation carrier but
assert no equality between public fields and semantic source data. That
binding belongs to the input/output authority modules.

| Stage path | Carrier | Mathematical obligation | Authority class |
|---|---|---|---|
| `nifs.pi_ccs.product.relation` | `RelationShape` | one batch-invariant Phi81 relation shape | computed |
| `nifs.pi_ccs.product.assignment` | `SourceAssignment` | one complete semantic carrier independent of public-width proof terms | authoritative witness |
| `nifs.pi_ccs.product.semantics` | `productSemantics` | concrete relation operations over that same assignment carrier | computed |
| `nifs.pi_ccs.product.input` | `SourceProduct` | fresh CCS followed by optional running CE statements | public input |
| `nifs.pi_ccs.product.output` | `Product` | one fresh CE statement per source | checked/derived output |
| `nifs.pi_ccs.product.membership` | `ProductHolds` | every output is genuine CE membership for its aligned assignment | derived handoff |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uCommitment

/-- One complete semantic source assignment. Keeping this carrier independent
of `publicRingColumns` avoids making witness identity depend on a public-width
proof term. -/
abbrev SourceAssignment (shape : SemanticShape) :=
  PaperLinearAlgebra.Assignment F shape.carrierWidth

/-- Batch-invariant relation shape used by both the source product and its CE
outputs. -/
abbrev RelationShape
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :=
  Phi81Relation.Shape.ofSemantic shape publicRingColumns publicFits

/-- Project a semantic source assignment into the exact public carrier of the
batch-invariant relation shape. -/
def sourcePublicInput
    {shape : SemanticShape}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (assignment : SourceAssignment shape) :
    Phi81Relation.PublicInput
      (RelationShape shape publicRingColumns publicFits) :=
  Phi81Relation.projectPublicInput assignment

/-- Concrete Phi81 relation semantics over the proof-term-independent source
assignment carrier. -/
def productSemantics
    {shape : SemanticShape}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment) :
    RelationSemantics
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (SourceAssignment shape)
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation Commitment :=
  Phi81Relation.relationSemantics
    (shape := RelationShape shape publicRingColumns publicFits) commit

/-- Full public source product at the concrete Phi81 relation type. -/
abbrev SourceProduct
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (arity : BatchArity params) :=
  PiCCS.InputProduct
    (Phi81Relation.Structure
      (RelationShape shape publicRingColumns publicFits))
    (Phi81Relation.PublicInput
      (RelationShape shape publicRingColumns publicFits))
    (Phi81Relation.Point
      (RelationShape shape publicRingColumns publicFits))
    Phi81Relation.Evaluation Commitment params arity

/-- Complete CE output product at the concrete Phi81 relation type. -/
abbrev Product
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (arity : BatchArity params) :=
  Fin arity.total ->
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits) Commitment

/-- Exact product-level relation handed from `Pi_CCS` to `Pi_RLC`. This is a
deep interface: callers see one batch predicate, while the definition remains
precisely one concrete `CE.Holds` obligation per aligned source. -/
def ProductHolds
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (outputs :
      Product shape publicRingColumns publicFits Commitment params arity)
    (assignments : Fin arity.total -> SourceAssignment shape) : Prop :=
  ∀ source,
    CE.Holds
      (productSemantics publicRingColumns publicFits commit) params
      (outputs source) (assignments source)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
