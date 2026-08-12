import Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor

/-! Focused compile gate for the complete paper-NIFS output carrier. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionProductNifsOutputRowsFor

open Nightstream.Implementation.NebulaV2.ProductionProductNifsOutputRowsFor

#check rows_length
#check Placed.assignment_coordinate
#check rows_sound
#check sources_of_nifs_rows
#check section_rows_sound

end tests.NebulaV2ProductionProductNifsOutputRowsFor
