import Nightstream.Protocol.Nebula.PaperFinalization
import tests.Axioms.Support

/-! Fail-closed dependency guards for the paper Nebula finalizer model. -/

/-- info: 'Nightstream.Protocol.Nebula.PaperFinalization.advances_implies_layer1' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.PaperFinalization.advances_implies_layer1

/-- info: 'Nightstream.Protocol.Nebula.PaperFinalization.advances_memory_continuity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.PaperFinalization.advances_memory_continuity

/-- info: 'Nightstream.Protocol.Nebula.PaperFinalization.advances_next_state_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.PaperFinalization.advances_next_state_exact
