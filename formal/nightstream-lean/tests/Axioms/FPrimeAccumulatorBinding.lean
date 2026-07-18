import Nightstream.Protocol.FPrime.AccumulatorBinding
import tests.Axioms.Support

/-! Fail-closed dependency gate for exact ordered-child accumulator binding. -/

/-- info: 'Nightstream.Protocol.FPrime.AccumulatorBinding.digest_eq_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.AccumulatorBinding.digest_eq_or_failure

/-- info: 'Nightstream.Protocol.FPrime.AccumulatorBinding.claims_eq_or_chainFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.AccumulatorBinding.claims_eq_or_chainFailure
