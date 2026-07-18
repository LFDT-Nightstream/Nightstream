import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne

/-!
Compile-time surface regression for the fixed-one active-obligation profile.

| Stage path | Guarded surface | Assurance status |
|---|---|---|
| `fprime.fixed_one.carrier` | `FixedOneCanonical.Input`, `Obligations` | sole canonical semantic carrier and target |
| `fprime.fixed_one.raw` | `Raw.exact` | exact five-family raw plan |
| `fprime.fixed_one.canonical.eliminated` | `Canonical.eliminated_hold` | prior-slot, structure, and dispatch derived |
| `fprime.fixed_one.canonical` | `Canonical.exact` | exact three-family canonical plan |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne

#check FixedOneCanonical.selected
#check FixedOneCanonical.fin_eq_selected
#check FixedOneCanonical.dispatch_derived
#check FixedOneCanonical.Input
#check FixedOneCanonical.Obligations
#check FixedOneCanonical.obligations_iff_active
#check FixedOneCanonical.holds_projection_iff
#check Raw.checks
#check Raw.eliminated
#check Raw.classified
#check Raw.classification_disjoint
#check Raw.exact
#check Canonical.checks
#check Canonical.eliminated
#check Canonical.classified
#check Canonical.classification_disjoint
#check Canonical.eliminated_hold
#check Canonical.accepts_iff_obligations
#check Canonical.exact
