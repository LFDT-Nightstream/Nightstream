import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs

/-!
Compile-time surface regression for the actual-type selected-NIFS removal
witness.

| Stage path | Property under test |
|---|---|
| `fprime.active.necessity.selected_nifs.mutation` | the adversarial result changes only its parent norm stage |
| `fprime.active.necessity.selected_nifs.rejection` | unchanged checked children reject the changed parent |
| `fprime.active.necessity.selected_nifs.necessary` | the exact six-family plan becomes unsound when selected NIFS is removed |
| `fprime.active.necessity.selected_nifs.outcome` | independent honest NIFS premises construct the valid baseline or expose exact sampler shortfall |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan

#check SelectedNifs.differentStage_ne
#check SelectedNifs.Realization
#check SelectedNifs.Realization.forgedNext
#check SelectedNifs.Realization.forged_not_transition
#check SelectedNifs.Realization.weakened
#check SelectedNifs.Realization.rejected
#check SelectedNifs.Realization.necessary
#check SelectedNifs.exists_or_samplerShortfall_of_semanticPremises
#check SelectedNifs.necessary_or_samplerShortfall_of_semanticPremises
#check SelectedNifs.exists_of_honestNifs
#check SelectedNifs.necessary_of_honestNifs
