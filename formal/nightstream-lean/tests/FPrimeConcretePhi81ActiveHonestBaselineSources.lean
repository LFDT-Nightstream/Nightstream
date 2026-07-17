import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources

/-!
Focused compile-time regression for the production-shaped fixed-carrier,
model-level source baseline at the intended fixed-active profile.

| Stage path | Regression |
|---|---|
| `fprime.active.honest_baseline.sources.profile` | intended fixed-active model dimensions reduce to the 270-coordinate carrier |
| `fprime.active.honest_baseline.sources.polynomial` | the two monomial degrees remain two and one |
| `fprime.active.honest_baseline.sources.paper` | the independently defined source product still proves `Semantics.Paper.Holds` |
-/

namespace Nightstream.Tests.FPrimeConcretePhi81ActiveHonestBaselineSources

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline

#check Sources.profile_exact
#check Sources.monomial_degrees_exact
#check Sources.freshCcsHolds
#check Sources.allSourceNormsHold
#check Sources.carriedEvaluationsHold
#check Sources.paperHolds

example : Sources.shape.carrierWidth = 270 := Sources.profile_exact.2.2.2.2.2.2.2

example :
    Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper.Holds
      Sources.data :=
  Sources.paperHolds

end Nightstream.Tests.FPrimeConcretePhi81ActiveHonestBaselineSources
