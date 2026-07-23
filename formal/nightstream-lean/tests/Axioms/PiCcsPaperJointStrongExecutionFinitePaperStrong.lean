import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the finite operational `Pi_CCS` strong game.
The expected sets were recorded from a focused probe.
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.perfectComplete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms perfectComplete

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.publicCoin' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms publicCoin

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.uniformTruncatedWorkBound_of_eligible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms uniformTruncatedWorkBound_of_eligible

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.sourceExtractionProbability_eq_of_eligible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sourceExtractionProbability_eq_of_eligible

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.outputPhiMismatchProbability_eq_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms outputPhiMismatchProbability_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong.finitePaperStrong' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finitePaperStrong
