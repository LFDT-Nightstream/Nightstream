import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness

/-!
Focused regressions for deterministic Split-NC mixing soundness.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.soundness.all_source_zero` | pointwise-zero source specializations zero every convention's mix | convention-specific hidden premise |
| `nifs.pi_ccs.nc.soundness.paper_nc` | zero paper-relative mix iff truth, selector root, or gamma root | unnamed deterministic acceptance case |
| `nifs.pi_ccs.nc.mixing.split_v1.gamma_zero` | Split-V1 remains unconditionally zero when gamma is zero | accidental claim that paper-relative soundness already covers Split-V1 |
| `nifs.pi_ccs.nc.soundness.split_v1` | zero Split-V1 mix iff truth, selector root, paper gamma root, or extra gamma-zero root | hidden production-convention acceptance case |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.InitialSum

/-- Pointwise-zero specializations zero every named gamma schedule. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (convention : GammaConvention)
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (allZero :
      ∀ source, sourceResidualAtBeta covers data coins source = K.zero) :
    mixedResidualAtBeta convention covers data coins = K.zero :=
  mixedResidualAtBeta_eq_zero_of_all_source_specializations_zero
    convention covers data coins allZero

/-- Paper-relative zero acceptance has exactly the two named deterministic
failure events besides semantic truth. -/
example
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    mixedResidualAtBeta .paperNc covers data coins = K.zero ↔
      Semantics.Nc.Truth data ∨
        SelectorRoot covers data coins ∨ GammaRoot covers data coins :=
  paperNc_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot
    noZeroDivisors covers data coins

/-- Split-V1's zero-gamma root remains unconditional. This is deliberately
not a cancellation theorem back to the paper-relative mixture. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain)
    (gammaZero : coins.gamma = K.zero) :
    mixedRangeAt .splitV1 covers data coins point = K.zero :=
  splitV1Mix_eq_zero_of_gamma_zero
    covers data coins point gammaZero

/-- Under the explicit concrete-carrier algebraic premises, Split-V1 has
exactly one additional deterministic root beyond the paper-relative
decomposition. -/
example
    (baseNoZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (sevenNonresidue : ConcreteCarrier.SevenProjectiveNonresidue)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain) :
    mixedResidualAtBeta .splitV1 covers data coins = K.zero ↔
      Semantics.Nc.Truth data ∨
        SelectorRoot covers data coins ∨
          GammaRoot covers data coins ∨
            SplitV1GammaZero covers data coins :=
  splitV1_mixedResidualAtBeta_eq_zero_iff_truth_or_selectorRoot_or_gammaRoot_or_gammaZero
    baseNoZeroDivisors sevenNonresidue covers data coins

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.MixingSoundness.Tests
