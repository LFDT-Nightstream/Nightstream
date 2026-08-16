import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCNormalizedOverlayRows

/-! Regression surface for normalized PiRLC family-overlay semantics. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcNormalizedOverlayRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcNormalizedOverlayRows.Normalized

example : sourceColumns = 33360 := rfl
example : finalColumns = 35856 := rfl

example :
    (sourceSlot ⟨1, by decide⟩ (by decide)).start = 111 := rfl

example :
    (sourceSlot ⟨33251, by decide⟩ (by decide)).width = 1 := rfl

example :
    (sourceSlot ⟨33252, by decide⟩ (by decide)).start = 33362 := rfl

example :
    (sourceSlot ⟨33359, by decide⟩ (by decide)).start = 35823 := rfl

#check accepted_coordinate_eq_linearValue
#check accepted_implies_coordinate_commitment
#check PhaseBindingPlaced
#check accepted_implies_phaseBindingPlaced
#check receipt_geometry_exact
#check production_receipt_valid

end tests.NebulaProductionStreamingPiRlcNormalizedOverlayRows
