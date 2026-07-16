import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Focused surface checks for typed Phi81 `Pi_RLC` orchestration.

| Stage path | Regression |
|---|---|
| `nifs.pi_rlc.verify.evaluation_hom.product_order` | the local law is explicit and discharged by executable Phi81 multiplication |
| `nifs.pi_rlc.verify.evaluation_hom.row.blocks` | flat carrier equals the block/row/kernel tree |
| `nifs.pi_rlc.verify.evaluation_hom.row.action` | block and complete row actions use only the proved local law |
| `nifs.pi_rlc.verify.evaluation_hom.mle` | the canonical evaluator action reaches the RingK row MLE |
| `nifs.pi_rlc.verify.evaluation_hom.matrices` | every canonical matrix preserves the same action |
| `nifs.pi_rlc.verify.evaluation_hom.finite.assignment` | the complete assignment uses the canonical finite RingF fold |
| `nifs.pi_rlc.verify.evaluation_hom.finite.arrays` | fixed-shape evaluation arrays use the identical finite fold |
| `nifs.pi_rlc.verify.evaluation_hom.algebra` | the relation theorem has the exact PiRLC algebra-field input shape |
-/

namespace tests.Phi81PiRLCEvaluationHomomorphism

open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

#check PiRLC.ProductOrderLaw
#check PiRLC.productOrderLaw
#check PiRLC.kernelImage_apply
#check PiRLC.rowRing_eq_blockSum
#check PiRLC.blockRowRing_act
#check PiRLC.rowRing_act
#check PiRLC.matrixEvaluation_eq_evaluateRows
#check PiRLC.matrixEvaluation_act
#check PiRLC.evaluations_act
#check PiRLCFinite.combineAssignments
#check PiRLCFinite.combineEvaluation
#check PiRLCFinite.combineEvaluations
#check PiRLCFinite.matrixEvaluation_combine
#check PiRLCFinite.evaluations_hom
#check PiRLCFinite.relation_evaluations_hom
end tests.Phi81PiRLCEvaluationHomomorphism
