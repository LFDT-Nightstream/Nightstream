import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

/-!
Compile-time surface regression for the exact six-family active-obligation
plan.

| Stage path | Property under test |
|---|---|
| `fprime.active.obligation_plan.carrier` | the plan consumes a real typed slot and complete fold result |
| `fprime.active.obligation_plan.families` | all six retained semantic fields have stable names and order |
| `fprime.active.obligation_plan.exact` | plan acceptance is exactly `ActiveSemantics.Obligations` |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

#check ObligationPlan.Family
#check ObligationPlan.checks
#check ObligationPlan.Candidate
#check ObligationPlan.semantics
#check ObligationPlan.target
#check ObligationPlan.accepts_iff_obligations
#check ObligationPlan.exact
