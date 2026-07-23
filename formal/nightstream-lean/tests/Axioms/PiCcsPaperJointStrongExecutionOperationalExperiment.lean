import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
import tests.Axioms.Support

/-!
Fail-closed dependency probes for the exact finite operational `Pi_CCS`
experiment. The expected sets were recorded from a focused build.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment.mem_runSupport_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms mem_runSupport_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment.successfulSupport_nonempty_of_floor' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms successfulSupport_nonempty_of_floor

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment.extraction_after_first_success' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms extraction_after_first_success
