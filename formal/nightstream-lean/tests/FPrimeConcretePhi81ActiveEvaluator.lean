import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.SemanticBoundary

/-!
Compile-time surface regression for the fixed-active outer evaluator.

| Stage path | Property under test |
|---|---|
| `fprime.active.checkers` | nontrivial equalities have named exact Boolean owners |
| `fprime.active.physical` | successful execution has one complete physical characterization |
| `fprime.active.semantic` | soundness exposes named failures and closes only under explicit semantic/security premises |
| `fprime.active.completeness` | honest paper/source premises construct one accepted canonical output or exact sampler shortfall |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator

#check Certificate
#check Checkers
#check Checkers.priorLinkCheck_eq_true_iff
#check Checkers.freshStructureCheck_eq_true_iff
#check Checkers.dispatchCheck_eq_true_iff
#check OuterChecks
#check outerCheck
#check outerCheck_eq_true_iff
#check PhysicalChecks
#check run
#check run_eq_some_iff_physicalChecks
#check run_sound_or_outputUnbound_or_piCcsBadEvent
#check SoundnessClosure
#check run_sound_of_closure
#check exists_run_and_holds_or_samplerShortfall
#check run_complete_of_outer_and_honestNifs
#check exists_run_and_holds_of_outer_and_honestNifs
