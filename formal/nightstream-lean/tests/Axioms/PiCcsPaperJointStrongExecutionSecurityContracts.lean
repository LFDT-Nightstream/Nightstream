import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
import tests.Axioms.Support

/-!
Fail-closed dependency probes for the named causal `Pi_CCS` security
contracts. The expected sets were recorded from a focused build.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts.fixedFirstBad_eq_mixing_or_sumCheck' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fixedFirstBad_eq_mixing_or_sumCheck

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts.fixedFirstBadBound_of_securityContracts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fixedFirstBadBound_of_securityContracts

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts.extraction_after_first_success_of_securityContracts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms extraction_after_first_success_of_securityContracts
