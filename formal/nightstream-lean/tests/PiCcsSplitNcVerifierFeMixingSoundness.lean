import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness

/-!
Focused regression for the deterministic Split-NC FE compression boundary.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.fe.soundness.decomposition` | zero FE mix is truth or an explicit nonzero-residual compression root | unnamed false-acceptance case |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.InitialSum

/-- The facade exposes the complete deterministic FE compression split; a
false semantic relation cannot disappear into an unnamed acceptance case. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (profile : SupportedProfile shape domain)
    (data : Data shape)
    (coins : Coins shape domain) :
    mixedResidual profile data coins = K.zero ↔
      Semantics.Fe.Truth data ∨ MixingRoot profile data coins :=
  mixedResidual_eq_zero_iff_truth_or_mixingRoot profile data coins

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.MixingSoundness.Tests
