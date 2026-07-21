import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecTypedCarrier

/-! Focused interface regression for the generic active `PiDEC` carrier. -/

namespace Nightstream.Tests.PiDecTypedCarrier

open Nightstream.Implementation.R1CS.PiDecTypedCarrier

#check Profile.childLayout
#check decodedParent
#check decodedOutput
#check Accepted
#check accepted_refines_paper
#check publicTranspose_exact
#check parentPublicSlot_lt
#check childPublicSlot_lt
#check parentEvaluationLimb_lt
#check childEvaluationLimb_lt

example : Active.shape.matrixCount = 13 := Active.matrixCount_exact

example : Active.shape.rowVariables = 24 := Active.rowVariables_exact

example : Active.shape.publicWidth = 270 := Active.publicWidth_exact

#check Active.parent_evaluation_count_exact
#check Active.child_evaluation_count_exact
#check Active.parent_point_count_exact
#check Active.child_point_count_exact

end Nightstream.Tests.PiDecTypedCarrier
