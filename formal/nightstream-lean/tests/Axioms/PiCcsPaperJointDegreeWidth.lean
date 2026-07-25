import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.InteractiveCompositionBridge
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract
import tests.Axioms.Support

/-!
Fail-closed dependency probes for paper-owned SumCheck degree-width selection.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.ConstraintPolynomial.canonicalEqualityGatedDegreeBound_le_degreeBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.ConstraintPolynomial.canonicalEqualityGatedDegreeBound_le_degreeBound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.VerifierInput.sumcheckDegreeBound_le_paperRoundDegreeCeiling_of_b_eq_two' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.VerifierInput.sumcheckDegreeBound_le_paperRoundDegreeCeiling_of_b_eq_two

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.paperDegreeWidthExact_implies_width_le_paperRoundDegreeCeiling' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.paperDegreeWidthExact_implies_width_le_paperRoundDegreeCeiling

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.statement_sumcheckDegreeBound_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.statement_sumcheckDegreeBound_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.strongExecutionContext_paperDegreeWidthExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Key.strongExecutionContext_paperDegreeWidthExact

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong.Point.sumcheckWidth_le_paperRoundDegreeCeiling' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong.Point.sumcheckWidth_le_paperRoundDegreeCeiling

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_not_paperDegreeWidthExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_not_paperDegreeWidthExact
