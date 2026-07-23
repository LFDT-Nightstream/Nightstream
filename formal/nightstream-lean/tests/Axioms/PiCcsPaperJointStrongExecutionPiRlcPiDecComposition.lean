import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec
import tests.Axioms.Support

/-! Fail-closed dependency guards for the finite three-stage paper composition. -/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec.abortingReductionOfKnowledge' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms abortingReductionOfKnowledge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec.weakSuccess_iff_abortingExtractedSource' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms weakSuccess_iff_abortingExtractedSource

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec.intermediateProbability' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms intermediateProbability

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec.operationalCoupling' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms operationalCoupling

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec.finiteReductionOfKnowledge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms finiteReductionOfKnowledge
