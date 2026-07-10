import Nightstream.Implementation.FPrime.Envelope

/-!
Fail-closed axiom gate: `#guard_msgs` fails the build if the axiom report of a
completed theorem ever differs from the recorded expectation — a theorem that
silently picks up `sorryAx`, `Classical.choice`, or any other axiom breaks this
file instead of printing an ignored info line.
-/

/-- info: 'Nightstream.Implementation.FPrime.Envelope.check_sound' does not depend on any axioms -/
#guard_msgs in
#print axioms Nightstream.Implementation.FPrime.Envelope.check_sound
