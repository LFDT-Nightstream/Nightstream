import Nightstream.Protocol.FPrime.Paper.PriorLink
import tests.Axioms.Support

/-! Fail-closed dependency gate for paper-level cross-step binding. -/

/-- info: 'Nightstream.Protocol.FPrime.Paper.PriorLink.preimage_eq_or_securityFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.PriorLink.preimage_eq_or_securityFailure

/-- info: 'Nightstream.Protocol.FPrime.Paper.PriorLink.running_eq_or_securityFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.PriorLink.running_eq_or_securityFailure
