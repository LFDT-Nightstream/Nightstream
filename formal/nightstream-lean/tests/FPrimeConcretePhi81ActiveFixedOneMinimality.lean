import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne.Minimality

/-!
Compile-time surface regression for the closed canonical fixed-one
inclusion-minimality fixture.

| Stage path | Guarded surface | Assurance status |
|---|---|---|
| `fprime.fixed_one.canonical.global` | `Global.exact` | exact global three-family plan |
| `fprime.fixed_one.minimality.context` | `Baseline.contextAt_inputAt` | outer iteration mutations preserve the full NIFS context |
| `fprime.fixed_one.minimality.baseline` | `Baseline.exists_honestNext` | explicit sampler-backed semantic result exists |
| `fprime.fixed_one.minimality.retained` | three removal witnesses | every retained family is inclusion-necessary |
| `fprime.fixed_one.minimality.ledger` | `family_ledger` | exhaustive and disjoint family classification |
| `fprime.fixed_one.minimality.closed` | `inclusionMinimalSound` | closed model-level inclusion minimality |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne.Minimality

#check Global.Case
#check Global.exact
#check Global.lift_local_necessary
#check Baseline.machine
#check Baseline.contextAt_inputAt
#check Baseline.exists_honestNext
#check iteration_necessary
#check priorPublicInput_necessary
#check selectedNifs_necessary
#check family_ledger
#check retained_necessary
#check inclusionMinimalSound
