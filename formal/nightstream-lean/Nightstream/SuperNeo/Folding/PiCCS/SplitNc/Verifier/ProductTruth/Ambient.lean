import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTypes
import Nightstream.SuperNeo.Folding.PiRLC

/-!
Fresh-to-ambient transport for the concrete Phi81 product semantics.

Assurance tier: model-level.

Owns: monotonicity of the explicit source-carrier norm predicate. It does not
claim the paper's false universal strict `q / 2` bound.

Does not own: output construction, commitments, extraction, probability,
Fiat--Shamir, Rust, R1CS, costs, or rows.

Emits constraints: no.

| Stage path | Owned equation | Authority |
|---|---|---|
| `piccs.split_nc.product_truth.ambient` | fresh CE openings imply ambient openings under the explicit bound order | derived |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uCommitment

/-- A genuinely fresh-valid concrete CE product is also valid at the literal
ambient stage whenever the verifier-owned fresh bound is no larger. -/
theorem ambientOpenings_of_productHolds
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
    (assignments : Fin arity.total -> SourceAssignment shape)
    (outputFresh : ∀ source, (outputs source).stage = .fresh)
    (freshLeAmbient : params.b <= params.q / 2)
    (holds :
      ProductHolds publicRingColumns publicFits commit outputs assignments) :
    PiRLC.AmbientOpenings
      (productSemantics publicRingColumns publicFits commit) params outputs
      assignments := by
  intro source
  have valid := holds source
  rcases valid with ⟨opening, pointValid, evaluations⟩
  refine ⟨?_, pointValid, evaluations⟩
  refine ⟨opening.1, opening.2.1, ?_⟩
  have freshNorm :
      sourceNormBounded params.b (assignments source) := by
    simpa only [productSemantics, sourceNormBounded, outputFresh source,
      NormStage.bound] using opening.2.2
  change sourceNormBounded (params.q / 2) (assignments source)
  intro column
  exact Nat.lt_of_lt_of_le (freshNorm column) freshLeAmbient

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.ProductTruth
