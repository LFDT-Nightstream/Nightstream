import Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCoreRowsFor

/-! Focused compile gate for row-derived exponent-indexed terminal CE core. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperTerminalCoreRowsFor

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCoreRowsFor

#check publicRows_length
#check VerifierInputPlacement.point_exact
#check VerifierInputPlacement.evaluations_exact
#check public_row_exact
#check public_exact
#check evaluations_exact

end tests.NebulaV2ProductionPaperTerminalCoreRowsFor
