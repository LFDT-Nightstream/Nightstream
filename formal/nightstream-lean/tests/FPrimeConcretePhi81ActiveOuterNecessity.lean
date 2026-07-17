import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies

/-!
Compile-time surface regression for lifting actual outer countermodels into
the exact six-family ConcretePhi81 active-obligation plan.

| Stage path | Guarded surface | Assurance status |
|---|---|---|
| `fprime.active.necessity.outer.map` | `mappedFamily` | exact family mapping |
| `fprime.active.necessity.outer.candidate` | `candidate` | actual typed carrier |
| `fprime.active.necessity.outer.preservation` | `weakened` | conditional actual-type theorem |
| `fprime.active.necessity.outer.rejection` | `rejected` | conditional actual-type theorem |
| `fprime.active.necessity.outer.necessary` | `necessary` | conditional actual-type necessity |
| `fprime.active.necessity.outer.outcome` | `*_necessary_or_samplerShortfall_of_semanticPremises` | exhaustive model outcome |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies

#check mappedFamily
#check candidate
#check weakened
#check rejected
#check necessary
#check activeIteration_necessary_or_samplerShortfall_of_semanticPremises
#check priorPublicLink_necessary_or_samplerShortfall_of_semanticPremises
#check dispatch_necessary_or_samplerShortfall_of_semanticPremises
