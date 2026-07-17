import Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext
import tests.Axioms.Support

/-! Fail-closed dependency gate for the bootstrap context constructor. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Invocation.sourceProduct_fresh' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Invocation.sourceProduct_fresh

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Invocation.noRunningSource' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Invocation.noRunningSource

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Template.build_runningParent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Template.build_runningParent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Template.build_runningAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.BootstrapContext.Template.build_runningAuthority
