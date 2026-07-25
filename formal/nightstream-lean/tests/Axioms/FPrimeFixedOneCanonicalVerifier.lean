import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality
import tests.Axioms.Support

/-! Fail-closed kernel-dependency guard for the paper-only fixed-one
executable and model-level minimality headlines. -/

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval_eq_generic' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval_eq_generic

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality.accepts_iff_target' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality.accepts_iff_target

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality.inclusionMinimalSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality.inclusionMinimalSound

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality.obligation8_classification' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.Minimality.obligation8_classification

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.eval_eq_generic' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.eval_eq_generic

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality.accepts_iff_transition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality.accepts_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality.inclusionMinimalSound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality.inclusionMinimalSound

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality.obligation8_classification' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Minimality.obligation8_classification
