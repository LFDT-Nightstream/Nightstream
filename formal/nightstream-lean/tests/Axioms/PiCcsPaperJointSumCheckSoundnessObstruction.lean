import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract
import tests.Axioms.Support

/-!
Fail-closed dependency probes for the exact fixed-width SumCheck soundness
obstruction at the paper-joint causal security-contract boundary.
-/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.shape_freshCount_positive' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.shape_freshCount_positive

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.syntaxDegree_eq_four' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.syntaxDegree_eq_four

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.protocolPolynomial_eq_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.protocolPolynomial_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_messageDegree_eq_six' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_messageDegree_eq_six

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_highCoefficient_eq_one' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_highCoefficient_eq_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_not_zeroAboveSyntaxDegree_four' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_not_zeroAboveSyntaxDegree_four

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_not_zeroAbovePaperDegree_four' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.rootPolynomial_not_zeroAbovePaperDegree_four

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.collision_at' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckFixedWidthPadding.collision_at

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_syntaxDegree_lt_width' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_syntaxDegree_lt_width

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_not_paperDegreeWidthExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_not_paperDegreeWidthExact

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.paperRoundDegreeCeiling_eq_four' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.paperRoundDegreeCeiling_eq_four

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.paperSumCheckBudget_eq_four_six' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.paperSumCheckBudget_eq_four_six

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_challengeSetSize_eq_alphabet_cardinality' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.context_challengeSetSize_eq_alphabet_cardinality

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.sourceProtocolData_eq_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.sourceProtocolData_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.strategy_roundMessage_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.strategy_roundMessage_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.sumCheckFailure_execute' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.sumCheckFailure_execute

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.sumCheckFailure_probability_eq_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.sumCheckFailure_probability_eq_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.not_sumCheckSoundnessContract_of_lt_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.not_sumCheckSoundnessContract_of_lt_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.not_sumCheckSoundnessContract_at_paper_budget' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.SumCheckSoundnessContract.not_sumCheckSoundnessContract_at_paper_budget
