import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedAlgebraRows

/-! Regression surface for the normalized production PiRLC algebra rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcNormalizedAlgebraRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedAlgebraRows.Normalized

example : localColumns = 51463 := rfl
example : finalColumns = 8858862 := rfl
example : (selectorColumn .even).val = 648 := rfl
example : (selectorColumn .odd).val = 649 := rfl

example :
    (localSlot ⟨1, by decide⟩ (by decide)).start = 702 := rfl

example :
    (localSlot ⟨918, by decide⟩ (by decide)).width = 41 := rfl

example :
    (localSlot ⟨919, by decide⟩ (by decide)).start = 38340 := rfl

example :
    (localSlot ⟨1836, by decide⟩ (by decide)).width = 41 := rfl

example :
    (localSlot ⟨1837, by decide⟩ (by decide)).start = 75978 := rfl

example :
    (localSlot ⟨1891, by decide⟩ (by decide)).start = 78192 := rfl

example :
    (localSlot ⟨51462, by decide⟩ (by decide)).start = 2110603 := rfl

example : productionRows.length = 49626 := productionRows_length

#check retained_audit_geometry
#check evaluate_localColumnForm
#check evaluate_combinationImage
#check rowImage_accepted_iff_holds
#check satisfies_implies_source_rows
#check productionAccepted_implies_source_rows
#check productionAccepted_implies_concrete_phase

end tests.NebulaProductionStreamingPiRlcNormalizedAlgebraRows
