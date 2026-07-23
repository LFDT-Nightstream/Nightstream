import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
import tests.Axioms.Support

/-! Fail-closed dependency gate for deterministic causal paper `Pi_CCS`. -/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.ambientCheck_eq_true_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ambientCheck_eq_true_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.acceptedPrefix_extracts_fixedWitness_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms acceptedPrefix_extracts_fixedWitness_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.ambientSuccess_implies_source_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ambientSuccess_implies_source_or_badEvent
