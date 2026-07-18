import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap

/-!
Compile-time interface checks for the independent zero-running NIFS bootstrap.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.fixed_bootstrap.arity` | the source product is exactly one fresh plus zero running | silently modeling synthetic default children |
| `nifs.fixed_bootstrap.input.parent_absence` | semantic acceptance proves the parent carrier is absent | digest-only or unchecked bootstrap authority |
| `nifs.fixed_bootstrap.result` | parent and children share one semantic certificate | caller-supplied parent cache |
-/

open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedBootstrap

#check arity
#check arity_freshCount
#check arity_mode
#check arity_runningCount
#check arity_total
#check Context
#check Certificate
#check FoldResult
#check runningAuthority_of_parentAbsent
#check runningAuthority_iff_parentAbsent
#check resultOf
#check ResultTransition
#check ResultTransition.children_transition
#check ResultTransition.parentAbsent
#check ResultTransition.childStructure_eq_fresh
