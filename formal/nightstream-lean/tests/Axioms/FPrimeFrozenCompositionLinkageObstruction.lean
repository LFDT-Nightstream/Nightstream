import Nightstream.Protocol.FPrime.Frozen.CompositionLinkageObstruction
import tests.Axioms.Support

/-! Fail-closed dependency guard for the frozen-game linkage countermodel. -/

/-- info: 'Nightstream.Protocol.FPrime.Frozen.CompositionLinkageObstruction.unlinked_fields_countermodel' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Frozen.CompositionLinkageObstruction.unlinked_fields_countermodel
