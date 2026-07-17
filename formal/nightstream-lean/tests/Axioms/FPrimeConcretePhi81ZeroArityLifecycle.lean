import Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle
import tests.Axioms.Support

/-! Fail-closed dependency gate for the concrete zero-arity lifecycle. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.bootstrapContext_runningParent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.bootstrapContext_runningParent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.activeContext_runningParent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.activeContext_runningParent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Transition.output_realized' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Transition.output_realized

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Transition.produces_valid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Transition.produces_valid

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Reachable.running_realized' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Reachable.running_realized

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Reachable.valid_from_initial' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Reachable.valid_from_initial

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Reachable.from_running_is_running' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ZeroArityLifecycle.Reachable.from_running_is_running
