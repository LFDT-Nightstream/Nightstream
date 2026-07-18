import Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.HonestCompleteness
import tests.Axioms.Support

/-! Fail-closed dependency gate for branch-complete honest construction. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.HonestCompleteness.complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Semantics.HonestCompleteness.complete
