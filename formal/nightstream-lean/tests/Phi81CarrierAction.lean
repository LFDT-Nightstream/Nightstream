import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction

/-!
Focused model-level regressions for the complete-carrier Phi81 action.

| Protocol | Phase | Family | Regression |
|---|---|---|---|
| coefficient embedding | complete carrier | block / lane | flattening and decoding round-trip |
| `Pi_RLC` | assignment action | full block | typed action reads as fixed `ringFMul` |
| coefficient embedding | kernel image | all 54 lanes | independent kernel contraction equals the fixed ring product |
-/

namespace tests.Phi81CarrierAction

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism

#check CarrierAction.decode_carrierColumn
#check CarrierAction.assignmentBlock_act
#check CarrierAction.ringFMul_add_left
#check CarrierAction.ringFMul_add_right
#check CarrierAction.ringFMul_scale_left
#check CarrierAction.ringFMul_scale_right
#check CarrierAction.ringFMul_zero_right
#check CarrierAction.kernelImage_eq_ringFMul

end tests.Phi81CarrierAction
