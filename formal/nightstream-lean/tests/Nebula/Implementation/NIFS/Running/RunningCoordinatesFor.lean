import Nightstream.Implementation.Nebula.NIFS.Running.RunningCoordinatesFor

/-! Focused compile gate for exponent-indexed paper-NIFS running coordinates. -/

set_option autoImplicit false

namespace tests.NebulaProductNifsRunningCoordinatesFor

open Nightstream.Implementation.Nebula.ProductNifsRunningCoordinatesFor

#check sections_exact
#check point_coordinate_bound
#check public_input_coordinate_bound
#check evaluation_coordinate_bound
#check runningCoordinate_surjective
#check runningCodecFor_point_getD
#check runningCodecFor_commitment_getD
#check runningCodecFor_publicInput_getD
#check runningCodecFor_evaluation_getD

example : pointFieldCount 26 = 52 := by decide
example : commitmentsOffset 26 = 52 := by decide
example : publicInputsOffset 26 = 54484 := by decide

end tests.NebulaProductNifsRunningCoordinatesFor
