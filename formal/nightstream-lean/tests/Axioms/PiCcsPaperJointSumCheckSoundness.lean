import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness
import tests.Axioms.Support

/-!
Fail-closed dependency probes for concrete finite SumCheck soundness.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting.roots_count_le_degree' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting.roots_count_le_degree

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting.collisions_count_le_degree' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting.collisions_count_le_degree

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CausalSumCheckBound.probability_detects_le_ratio' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CausalSumCheckBound.probability_detects_le_ratio

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts.sumCheckBadChallengeEvent_eq_true_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts.sumCheckBadChallengeEvent_eq_true_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.sumCheckFailure_implies_detects' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.sumCheckFailure_implies_detects

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.verifierDetects_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.verifierDetects_probability_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.sumCheckBadChallenge_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.sumCheckBadChallenge_probability_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.sumCheckSoundnessContract_of_rootCounting' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.sumCheckSoundnessContract_of_rootCounting

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.extraction_after_first_success_of_rootCounting' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness.extraction_after_first_success_of_rootCounting
