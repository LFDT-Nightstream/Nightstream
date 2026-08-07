import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
import tests.Axioms.Support

/-! Fail-closed kernel-dependency guard for the executable terminal verifier. -/

open Nightstream.Protocol.FPrime.CanonicalTerminalVerifier

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.finRange_all_eq_true_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finRange_all_eq_true_iff

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.allRunningAccepted_eq_true_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms allRunningAccepted_eq_true_iff

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval_eq_true_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms eval_eq_true_iff_transition

/-- info: 'Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.evalOuter_eq_true_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evalOuter_eq_true_iff_transition
