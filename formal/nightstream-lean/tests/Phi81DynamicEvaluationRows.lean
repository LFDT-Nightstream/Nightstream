import Nightstream.Implementation.R1CS.Correspondence.Gadgets.Phi81DynamicEvaluationRows

/-! Focused compile gate for the reference dynamic Phi81 evaluator. -/

set_option autoImplicit false

namespace tests.Phi81DynamicEvaluationRows

open Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows

#check rowsFor_length
#check rows_length
#check combineFields_eq_matrixVectorAt
#check decodeCarried_rowCarried
#check decodeTable_eq_phi81Table
#check decodePoint_eq_decodedPoint
#check rows_sound

end tests.Phi81DynamicEvaluationRows
