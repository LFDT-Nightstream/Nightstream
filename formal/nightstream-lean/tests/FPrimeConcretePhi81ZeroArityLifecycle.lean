import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle

/-!
Compile-time theorem-surface regression for the concrete zero-arity
lifecycle.

| Stage path | Property under test |
|---|---|
| `fprime.zero_arity.state` | initial, primed, and running phases are structurally distinct |
| `fprime.zero_arity.context` | bootstrap has no parent while active carries the exact derived parent |
| `fprime.zero_arity.state.valid` | raw state is paired with explicit claim and NIFS provenance evidence |
| `fprime.zero_arity.result.provenance` | running output exposes one model-level NIFS result transition |
| `fprime.zero_arity.running.closed` | steady running cannot return to the bootstrap phase |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle

#check Fresh
#check Accumulator
#check State
#check Setup
#check bootstrapContext
#check bootstrapContext_runningParent
#check activeContext
#check activeContext_runningParent
#check NextClaimObligation
#check Transition
#check NifsOutputRealized
#check StateValid
#check Transition.output_realized
#check Transition.produces_valid
#check Reachable
#check Reachable.running_realized
#check Reachable.valid_from_initial
#check Reachable.from_running_is_running
