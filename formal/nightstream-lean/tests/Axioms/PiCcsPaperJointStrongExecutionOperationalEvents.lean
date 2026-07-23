import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
import tests.Axioms.Support

/-!
Fail-closed dependency probes for the literal operational `Pi_CCS` events.
The expected sets were recorded from a focused coordinated build.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.witnessDisagreement_implies_first_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms witnessDisagreement_implies_first_success

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.outputPhiMismatch_eq_false' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms outputPhiMismatch_eq_false

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents.extraction_or_fixedFirstBad' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms extraction_or_fixedFirstBad
