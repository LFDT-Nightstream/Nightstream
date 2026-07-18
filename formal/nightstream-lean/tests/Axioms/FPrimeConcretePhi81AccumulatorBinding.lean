import Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding
import tests.Axioms.Support

/-! Fail-closed dependency gate for compact ConcretePhi81 accumulator binding. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_failure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_commitmentFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_commitmentFailure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_canonicalParentFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.parent_children_eq_or_canonicalParentFailure
