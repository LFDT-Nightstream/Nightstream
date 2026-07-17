import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite
import tests.Axioms.Support

/-!
Fail-closed dependency gate for typed Phi81 evaluation homomorphism results.

| Protocol | Phase | Theorem family | Guarded claim |
|---|---|---|---|
| `Pi_DEC` | base-field recomposition | `BaseLinear` | finite assignment/evaluation combinations |
| `Pi_RLC` | complete carrier | `CarrierAction` | exact block action and kernel/product identification |
| `Pi_RLC` | extension evaluation | `RingKAction` | fixed-ring row action commutes with Boolean MLE |
| `Pi_RLC` | coefficient embedding | `Embedding` | exact executable quotient multiplication is preserved |
| `Pi_RLC` | quotient-ring normal form | `RingFLaws` | basis products reduce symbolically modulo Phi81 |
| `Pi_CCS` | identity coefficient row | `RingFLaws` | constant bar basis and kernel image are exact left units |
| `Pi_RLC` | finite challenge combination | `PiRLCFinite` | the complete assignment and fixed evaluation arrays use one canonical fold |
| `Pi_RLC` | product order | `RingFLaws` | bar and challenge left actions commute on every block |
-/

/-! Complete-carrier action and coefficient image. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock_act' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.assignmentBlock_act

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.ringFMul_add_left' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.ringFMul_add_left

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.ringFMul_scale_left' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.ringFMul_scale_left

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.kernelImage_eq_ringFMul' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction.kernelImage_eq_ringFMul

/-! Extension-evaluation action. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_zero

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_add' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_add

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_scale' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_scale

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_action' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_action

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_embeddedChallenge_action' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction.evaluateRows_embeddedChallenge_action

/-! Coefficientwise quotient-multiplication embedding. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.Embedding.embedChallenge_ringFMul' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.Embedding.embedChallenge_ringFMul

/-! Complete conditional `Pi_RLC` evaluation orchestration. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_eq_zero_of_padded_row_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_eq_zero_of_padded_row_zero

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_eq_kernelImage_of_unit_padded_row' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_eq_kernelImage_of_unit_padded_row

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_eq_blockSum' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_eq_blockSum

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_act' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.rowRing_act

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.matrixEvaluation_act' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.matrixEvaluation_act

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.evaluations_act' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.evaluations_act

/-! Exact finite-batch evaluation homomorphism. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear.raw_combineAssignments_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear.raw_combineAssignments_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.raw_combineAssignments_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.raw_combineAssignments_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.raw_recomposeAssignment_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.raw_recomposeAssignment_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.matrixEvaluation_combine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.matrixEvaluation_combine

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.evaluations_hom' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.evaluations_hom

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.relation_evaluations_hom' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite.relation_evaluations_hom

/-! Symbolic executable Phi81 normal form. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_basis_basis' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_basis_basis

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_one_left' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_one_left

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.kernelImage_constant' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.kernelImage_constant

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_barBasis_productOrder' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws.ringFMul_barBasis_productOrder

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.productOrderLaw' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC.productOrderLaw
