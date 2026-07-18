import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness

/-!
Focused regressions for canonical block×lane NC mixing soundness.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.soundness.gamma_polynomial` | explicit coefficients evaluate to the paper-relative source mix | hidden gamma schedule or coefficient-order drift |
| `nifs.pi_ccs.nc.block_lane.soundness.lane_to_block` | zero lane specializations force a zero source specialization | collapsed selector-stage reasoning |
| `nifs.pi_ccs.nc.block_lane.soundness.decompose` | zero mix exposes truth or exactly one of three root families | opaque or missing deterministic bad event |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- The source count alone determines the explicit gamma degree bound. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    (gammaPolynomial covers data coins).degreeUpperBound =
      shape.sourceCount - 1 :=
  gammaPolynomial_degreeUpperBound covers data coins

/-- The coefficient object and the independently named semantic mix agree. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    (gammaPolynomial covers data coins).evaluate
        ConcreteCarrier.extensionOps.toOps coins.gamma =
      InitialSum.mixedResidualAtBeta covers data coins :=
  gammaPolynomial_evaluate_eq_mixedResidualAtBeta covers data coins

/-- Lane-selector failure and block-selector failure remain separate proof
stages. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (allZero : ∀ source block,
      InitialSum.laneResidualAtBeta covers data coins source block = K.zero)
    (source : Fin shape.sourceCount) :
    InitialSum.sourceResidualAtBeta covers data coins source = K.zero :=
  sourceResidualAtBeta_eq_zero_of_all_lane_specializations_zero
    covers data coins allZero source

/-- A zero canonical source mix has the exact deterministic event split. -/
example
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain) :
    InitialSum.mixedResidualAtBeta covers data coins = K.zero ↔
      Semantics.Nc.Truth data ∨
        LaneSelectorRoot covers data coins ∨
        BlockSelectorRoot covers data coins ∨
        GammaPolynomialRoot covers data coins :=
  mixedResidualAtBeta_eq_zero_iff_truth_or_laneSelectorRoot_or_blockSelectorRoot_or_gammaPolynomialRoot
    noZeroDivisors covers data coins

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness.Tests
