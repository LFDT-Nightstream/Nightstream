import Nightstream.Implementation.R1CS.Correspondence.Gadgets.Phi81SharedEvaluationRows

/-! Focused compile gate for the sparse shared-tensor Phi81 evaluator. -/

set_option autoImplicit false

namespace tests.Phi81SharedEvaluationRows

open Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows

#check rowCombination_eval
#check rowActive_false_coefficients_zero
#check inactive_matrixVector_zero
#check productRows_sound
#check localProduct_sound
#check outputRows_sound
#check lane_sound
#check rows_sound
#check rowCount

end tests.Phi81SharedEvaluationRows
