import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum

/-!
Focused regressions for the canonical block×lane NC initial sum.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.initial.bridge` | the actual polynomial cube equals the source-grouped leaf-cubic mix | MLE/cubic confusion or finite-sum drift |
| `nifs.pi_ccs.nc.block_lane.initial.sumcheck.order` | recursive completions equal the block-then-lane cube | flattened coordinate-order drift |
| `nifs.pi_ccs.nc.block_lane.initial.claim` | the initial claim is definitionally zero | prover-controlled initial claim |
| `nifs.pi_ccs.nc.block_lane.initial.complete` | independent NC truth closes the exact cube | hidden output/callback premise |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- Selector linearity groups the same Boolean polynomial cube by source. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    hypercubeSum covers data coins =
      mixedResidualAtBeta covers data coins :=
  hypercubeSum_eq_mixedResidualAtBeta covers data coins

/-- The totalized recursive cube enumerates the same block-then-lane points. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    sumcheckHypercubeSum covers data coins =
      hypercubeSum covers data coins :=
  sumcheckHypercubeSum_eq_hypercubeSum covers data coins

/-- Independent truth zeros each source's MLE of Boolean leaf cubics. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data)
    (source : Fin shape.sourceCount) :
    sourceResidualAtBeta covers data coins source = K.zero :=
  sourceResidualAtBeta_eq_zero_of_truth
    covers data coins truth source

/-- The canonical NC initial claim is not certificate data. -/
example : claimedInitial = K.zero := by
  rfl

/-- Honest full-carrier NC truth closes the exact polynomial cube. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    claimedInitial = hypercubeSum covers data coins :=
  claimedInitial_eq_hypercubeSum_of_truth covers data coins truth

/-- Honest truth also closes the generic recursive SumCheck cube. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data) :
    claimedInitial = sumcheckHypercubeSum covers data coins :=
  claimedInitial_eq_sumcheckHypercubeSum_of_truth
    covers data coins truth

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum.Tests
