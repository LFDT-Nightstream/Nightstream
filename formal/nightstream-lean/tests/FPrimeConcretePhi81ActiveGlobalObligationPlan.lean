import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global

/-!
Compile-time surface regression for the global six-family ConcretePhi81
active-obligation plan.

| Stage path | Guarded surface | Assurance status |
|---|---|---|
| `fprime.active.obligation_plan.global.case` | `Case`, `Case.toLocal` | actual typed carrier |
| `fprime.active.obligation_plan.global.exact` | `accepts_iff_obligations`, `exact` | exact global plan |
| `fprime.active.obligation_plan.global.lift` | `lift_local_necessary` | exact witness transport |
| `fprime.active.obligation_plan.global.witnesses` | `Witnesses` | explicit closure boundary |
| `fprime.active.obligation_plan.global.minimal` | `inclusionMinimalSound` | conditional inclusion-minimality |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global

#check Case
#check Case.toLocal
#check accepts_iff_obligations
#check exact
#check lift_local_necessary
#check Witnesses
#check inclusionMinimalSound
