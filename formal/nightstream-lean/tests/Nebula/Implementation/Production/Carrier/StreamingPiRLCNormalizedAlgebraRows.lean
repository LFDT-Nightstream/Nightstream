import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedAlgebraRows

/-! Regression surface for the normalized production PiRLC algebra rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcNormalizedAlgebraRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized

example : localColumns = 45415 := rfl
example : finalColumns = 2484972 := rfl
example : (selectorColumn .even).val = 648 := rfl
example : (selectorColumn .odd).val = 649 := rfl

example :
    (localSlot ⟨1, by decide⟩ (by decide)).start = 702 := rfl

example :
    (localSlot ⟨810, by decide⟩ (by decide)).width = 23 := rfl

example :
    (localSlot ⟨811, by decide⟩ (by decide)).start = 19332 := rfl

example :
    (localSlot ⟨1620, by decide⟩ (by decide)).width = 41 := rfl

example :
    (localSlot ⟨1621, by decide⟩ (by decide)).start = 52542 := rfl

example :
    (localSlot ⟨1675, by decide⟩ (by decide)).start = 53784 := rfl

example :
    (localSlot ⟨45414, by decide⟩ (by decide)).start = 1059781 := rfl

example : productionRows.length = 43794 := productionRows_length

#check evaluate_localColumnForm
#check evaluate_combinationImage
#check rowImage_accepted_iff_holds
#check satisfies_implies_source_rows
#check productionAccepted_implies_source_rows
#check productionAccepted_implies_concrete_phase

end tests.NebulaProductionStreamingPiRlcNormalizedAlgebraRows
