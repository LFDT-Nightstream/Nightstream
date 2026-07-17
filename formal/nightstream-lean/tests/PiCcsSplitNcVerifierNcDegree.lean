import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree

/-!
Focused type-level regressions for the independent Split-NC degree contract.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.degree.width` | degree four maps to five constant-first slots | width drift |
| `nifs.pi_ccs.nc.degree.polynomial` | every exact total-polynomial slice is quartic | decoder split drift |
| `nifs.pi_ccs.nc.degree.sumcheck` | every honest expected round has five coefficients | suffix-sum degree drift |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing

example : ncSumcheckDegreeBound = 4 ∧ ncMessageWidth = 5 := by
  decide

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (before after : List K)
    (length : before.length + 1 + after.length =
      domain.columnVariables + domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      InitialSum.sumcheckPolynomial convention covers data coins
        (before ++ point :: after) :=
  sumcheckPolynomial_slice_quartic convention covers data coins
    before after length

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (fixed : List K)
    (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.columnVariables + domain.laneVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = 5 ∧
      message.degreeUpperBound = 4 ∧
      ∀ point,
        message.evaluate
            Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
            point =
          Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
            Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps.toOps
            (InitialSum.sumcheckPolynomial convention covers data coins)
            (fixed ++ [point]) remaining :=
  expectedRound_has_five_coefficients convention covers data coins
    fixed remaining length

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree.Tests
