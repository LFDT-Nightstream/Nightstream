import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies

/-!
Compile-time surface regression for actual-type prior-slot and selected-
structure removal witnesses.

| Stage path | Property under test |
|---|---|
| `fprime.active.necessity.prior_slot.*` | the conditional witness uses the actual selected slot and complete NIFS result |
| `fprime.active.necessity.structure.*` | the conditional witness uses the actual selected structure equation and complete NIFS result |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies

#check priorSlot_necessary_of_transition
#check priorSlot_necessary_or_samplerShortfall_of_semanticPremises
#check priorSlot_necessary_of_honestNifs
#check expectedStructure_necessary_of_transition
#check expectedStructure_necessary_or_samplerShortfall_of_semanticPremises
#check expectedStructure_necessary_of_honestNifs
