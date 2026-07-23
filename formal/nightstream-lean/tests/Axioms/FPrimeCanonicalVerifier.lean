import Nightstream.Protocol.FPrime.CanonicalVerifier
import tests.Axioms.Support

/-!
Fail-closed dependency guard for the compact executable Construction-2
verifier.
-/

open Nightstream.Protocol.FPrime.CanonicalVerifier

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.outputFor_outputHolds' does not depend on any axioms -/
#guard_msgs in
#audit_axioms outputFor_outputHolds

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.output_eq_outputFor' does not depend on any axioms -/
#guard_msgs in
#audit_axioms output_eq_outputFor

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_implies_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepts_implies_transition

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.transition_implies_accepts' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms transition_implies_accepts

/-- info: 'Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_iff_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepts_iff_transition
