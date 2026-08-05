import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity
import tests.Axioms.Support

/-!
Fail-closed dependency probes for finite paper-joint alpha/gamma mixing
soundness.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MultilinearRootCounting.zeros_count_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MultilinearRootCounting.zeros_count_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CoefficientRootCounting.roots_count_le_degree' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CoefficientRootCounting.roots_count_le_degree

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts.mixingRootEvent_eq_true_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts.mixingRootEvent_eq_true_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.alphaGammaZero_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.alphaGammaZero_probability_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.verifierAlphaGamma_marginal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.verifierAlphaGamma_marginal

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.mixingRoot_probability_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.mixingRoot_probability_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.mixingRootProbabilityContract_of_rootCounting' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness.mixingRootProbabilityContract_of_rootCounting

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity.fixedFirstBadBound_of_rootCounting' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity.fixedFirstBadBound_of_rootCounting

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity.extraction_after_first_success_of_rootCounting' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity.extraction_after_first_success_of_rootCounting

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity.extraction_after_success_gate_of_rootCounting' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity.extraction_after_success_gate_of_rootCounting
