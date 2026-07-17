import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum

/-!
Focused regressions for the independent Split-NC norm initial sum.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.initial.sumcheck.bridge` | generic completions equal the typed product cube | coordinate-order drift |
| `nifs.pi_ccs.nc.initial.residual.bridge` | polynomial cube equals the independently grouped source mix | omitted or duplicated source/point |
| `nifs.pi_ccs.nc.initial.claim` | claimed initial is definitionally zero | prover-controlled claim |
| `nifs.pi_ccs.nc.initial.complete` | honest norm truth equals the generic cube sum for every schedule | hidden output or callback premise |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing

/-- The totalized recursive cube and typed product cube are the same object. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    sumcheckHypercubeSum convention covers data coins =
      hypercubeSum convention covers data coins :=
  sumcheckHypercubeSum_eq_hypercubeSum
    convention covers data coins

/-- Finite rearrangement groups the same exact cube by source specialization. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    hypercubeSum convention covers data coins =
      mixedResidualAtBeta convention covers data coins :=
  hypercubeSum_eq_mixedResidualAtBeta
    convention covers data coins

/-- The NC initial claim is not certificate data. -/
example : claimedInitial = K.zero := by
  rfl

/-- Honest full-carrier norm truth closes the exact generic SumCheck initial
sum under every explicitly named gamma convention. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (truth : Semantics.Nc.Truth data) :
    claimedInitial = sumcheckHypercubeSum convention covers data coins :=
  claimedInitial_eq_sumcheckHypercubeSum_of_truth
    convention covers data coins truth

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum.Tests
