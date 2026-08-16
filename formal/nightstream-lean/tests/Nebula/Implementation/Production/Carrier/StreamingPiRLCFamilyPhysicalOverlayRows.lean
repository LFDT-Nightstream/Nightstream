import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyPhysicalOverlayRows

/-! Regression surface for the physical production PiRLC family overlay. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingPiRlcFamilyPhysicalOverlayRows

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows

#check physical_layout_exact
#check rows_length
#check FieldLinksHold
#check fieldLinkCount_exact
#check link_run_geometry_exact
#check physicalSourceColumnsExact_of_links
#check rows_sound
#check AcceptedRows.sound

end tests.NebulaProductionStreamingPiRlcFamilyPhysicalOverlayRows
