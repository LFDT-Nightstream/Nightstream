import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs

/-!
Compile-time surface regression for the independent fixed-active outer
F-prime semantics.

| Stage path | Property under test |
|---|---|
| `fprime.active.context` | verifier setup constructs the selected NIFS context |
| `fprime.active.obligations` | only irreducible checks inhabit semantic acceptance |
| `fprime.active.output` | the complete parent-and-children output is canonical |
| `fprime.active.honest_nifs` | independent honest premises yield a selected result or exact sampler shortfall |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

#check Slot
#check Running
#check Nightstream.Protocol.FPrime.ConcretePhi81.Outer.Running.toPaper
#check Input
#check Nightstream.Protocol.FPrime.ConcretePhi81.Outer.Input.toPaper
#check Output
#check Nightstream.Protocol.FPrime.ConcretePhi81.Outer.Output.toPaper
#check Machine
#check Setup
#check invocationAt
#check contextAt
#check contextAt_runningParent
#check updatedRunning
#check updatedRunning_selected
#check updatedRunning_other
#check outputOf
#check outputOf_toPaper
#check outputOf_runningNext_selected
#check outputOf_runningNext_other
#check Obligations
#check Obligations.priorPcValid
#check Obligations.selectedIndex_eq
#check Obligations.selectedStructures_eq_expected
#check Obligations.selectedInputAuthority
#check Holds
#check HonestNifs.SemanticPremises
#check HonestNifs.SemanticPremises.exists_resultTransition_or_samplerShortfall
#check HonestNifs.Premises
#check HonestNifs.Premises.toSemanticPremises
#check HonestNifs.Premises.exists_resultTransition
