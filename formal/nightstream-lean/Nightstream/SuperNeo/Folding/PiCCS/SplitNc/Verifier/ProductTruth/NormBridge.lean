import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.InputAuthority

/-!
Norm projection for the authoritative Split-NC source product.

Assurance tier: model-level.

Owns: the definitional projection from the concrete product semantics to the
Phi81 norm predicate and the partition-preserving source reindexing.

Does not own: a norm proof, transcript acceptance, commitments, extraction,
probability, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.product_truth.norm` | product-indexed norm predicate equals source-indexed norm predicate | definitional |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.NormBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uCommitment

@[simp] theorem productNormBounded
    {shape : SemanticShape}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (bound : Nat)
    (assignment : SourceAssignment shape) :
    (productSemantics publicRingColumns publicFits commit).normBounded
        bound assignment =
      sourceNormBounded bound assignment := by
  rfl

/-- The norm obligation at a product index is the obligation for the exact
semantic source at the inverse partition-preserving index. -/
theorem atProductIndex
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {Commitment : Type uCommitment}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (commit : SourceAssignment shape -> Commitment)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (norms :
      ∀ source,
        (productSemantics publicRingColumns publicFits commit).normBounded
          params.b (InputAuthority.productAssignments data alignment source))
    (source : Fin shape.sourceCount) :
    sourceNormBounded params.b (data.assignment source) := by
  have bounded := norms (alignment.productIndex source)
  simpa only [productNormBounded, InputAuthority.productAssignments,
    SourceAlignment.semanticIndex_productIndex] using bounded

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth.NormBridge
