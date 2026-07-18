import Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext

/-!
Compile-time theorem-surface regression for the zero-running outer-to-NIFS
context constructor.

| Stage path | Property under test |
|---|---|
| `fprime.bootstrap.nifs.input` | one fresh source and no running index are constructible |
| `fprime.bootstrap.nifs.running_parent` | the parent carrier is constructed as `none` and satisfies exact authority |
| `fprime.bootstrap.nifs.transcript` | public Split-NC input and prior transcript state pass through exactly once |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext

#check Invocation
#check Invocation.sourceProduct
#check Invocation.sourceProduct_fresh
#check Invocation.noRunningSource
#check Template
#check Template.build
#check Template.build_input
#check Template.build_runningParent
#check Template.build_runningAuthority
#check Template.build_piCcsInput
#check Template.build_priorState
