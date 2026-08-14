import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.PublicRowsFor

/-! Surface gate for exact terminal public-result rows. -/

set_option autoImplicit false

namespace tests.NebulaProductionPaperTerminalPublicRowsFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalPublicRowsFor

#check rows_length_exact
#check checks_canonical
#check rows_imply_publicChecks

end tests.NebulaProductionPaperTerminalPublicRowsFor
