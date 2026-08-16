import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedCarryRows

/-! Regression surface for the normalized production PiRLC carry rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcNormalizedCarryRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedCarryRows.Normalized

example : sourceColumns = 146224 := rfl
example : finalColumns = 2484972 := rfl
example : directSourceStart = 144276 := rfl
example : finalDirectStart = 1076045 := rfl
example : productionRowCount = 1621 := productionRowCount_exact

example :
    (challengeSlot ⟨1, by decide⟩ (by decide) (by decide)).start = 702 := rfl

example :
    (challengeSlot ⟨810, by decide⟩ (by decide) (by decide)).start = 19309 := rfl

example :
    (directSlot ⟨144276, by decide⟩ (by decide)).start = 1076045 := rfl

example :
    (directSlot ⟨146223, by decide⟩ (by decide)).start = 1120826 := rfl

#check evaluate_sourceColumnForm
#check evaluate_combinationImage
#check equalityImage_accepted_iff
#check ChallengesInStrongSet
#check productionAccepted_implies_source_rows
#check productionAccepted_implies_range
#check productionAccepted_implies_exact
#check productionAccepted_implies_exact_of_strong_set
#check receipt_geometry_exact

end tests.NebulaProductionStreamingPiRlcNormalizedCarryRows
