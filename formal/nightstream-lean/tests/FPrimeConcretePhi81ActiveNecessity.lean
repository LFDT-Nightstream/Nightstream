import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

/-!
Compile-time surface regression for the generic outer-necessity factorization.

These checks do not assert that a production ConcretePhi81 side anchor has
been constructed.

| Stage path | Property under test |
|---|---|
| `fprime.active.necessity.factor` | actual obligations and `Holds` factor through the generic outer boundary |
| `fprime.active.necessity.outer_plan` | the three-family generic plan is exact |
| `fprime.active.necessity.countermodels` | iteration, prior-link, and dispatch each have a closed generic removal witness |
| `fprime.active.necessity.lift` | concrete lifting requires an explicit actual side anchor and realization |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity

#check OuterPlan.View
#check OuterPlan.Boundary
#check OuterPlan.Family
#check OuterPlan.checks
#check OuterPlan.semantics
#check OuterPlan.accepts_iff_boundary
#check OuterPlan.exact
#check OuterPlan.Countermodel
#check OuterPlan.activeIteration_countermodel
#check OuterPlan.priorPublicLink_countermodel
#check OuterPlan.dispatch_countermodel
#check OuterPlan.activeIteration_necessary
#check OuterPlan.priorPublicLink_necessary
#check OuterPlan.dispatch_necessary
#check SideConditions
#check viewOf
#check obligations_iff_side_and_outer
#check holds_iff_exists_side_and_outer
#check WeakAccepts
#check SideAnchor
#check SideAnchor.liftCountermodel
