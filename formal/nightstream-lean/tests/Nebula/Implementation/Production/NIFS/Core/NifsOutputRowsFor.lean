import Nightstream.Implementation.Nebula.Production.NIFS.Core.NifsOutputRowsFor

/-! Focused compile gate for the complete paper-NIFS output carrier. -/

set_option autoImplicit false

namespace tests.NebulaProductionProductNifsOutputRowsFor

open Nightstream.Implementation.Nebula.ProductionProductNifsOutputRowsFor

#check rows_length
#check Placed.assignment_coordinate
#check rows_sound
#check sources_of_nifs_rows
#check section_rows_sound

end tests.NebulaProductionProductNifsOutputRowsFor
