import Nightstream.Implementation.R1CS.Correspondence.Gadgets.Phi81SharedTensorRows

/-! Focused compile gate for the shared dynamic Phi81 equality tensor. -/

set_option autoImplicit false

namespace tests.Phi81SharedTensorRows

open Nightstream.Implementation.R1CS.Phi81SharedTensorRows

#check nodeRows
#check RowsSatisfied
#check rows_sound
#check multiplicationCount
#check rowCount

end tests.Phi81SharedTensorRows
