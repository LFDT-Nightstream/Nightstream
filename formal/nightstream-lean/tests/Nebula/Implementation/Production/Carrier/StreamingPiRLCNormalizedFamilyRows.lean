import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedFamilyRows

/-! Regression surface for the joint normalized production PiRLC family rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcNormalizedFamilyRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedFamilyRows.Normalized

example : finalColumns = 2484972 := rfl

#check challengeAssignment_eq
#check algebraAssignment_one
#check carryRange_implies_algebraRange
#check decodedChallenges_eq
#check ExactCarry
#check carryAccepted_implies_exact
#check algebraAccepted_implies_output
#check ReplayTransition
#check jointAccepted_implies_concrete_phase

end tests.NebulaProductionStreamingPiRlcNormalizedFamilyRows
