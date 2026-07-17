import Nightstream.Protocol.FPrime.ConcretePhi81.Context

/-!
Compile-time theorem-surface regression for the fixed active outer-to-NIFS
context constructor.

| Stage path | Property under test |
|---|---|
| `fprime.nifs.input` | one fresh plus all fourteen running children are constructed, not supplied as a second product |
| `fprime.nifs.running_parent` | the exact selected derived parent is installed as `some parent` |
| `fprime.nifs.transcript` | public Split-NC input and prior transcript state pass through exactly once |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.Context

#check Invocation
#check Invocation.sourceProduct
#check Invocation.sourceProduct_fresh
#check Invocation.sourceProduct_running
#check Template
#check Template.build
#check Template.build_input
#check Template.build_runningParent
#check Template.build_piCcsInput
#check Template.build_priorState
