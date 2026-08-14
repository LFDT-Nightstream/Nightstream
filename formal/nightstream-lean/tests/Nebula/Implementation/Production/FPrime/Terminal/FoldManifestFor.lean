import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.FoldManifestFor

/-! Surface gate for terminal trailing-fold and close extraction. -/

set_option autoImplicit false

namespace tests.NebulaProductionPaperTerminalFoldManifestFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalFoldManifestFor

#check Program.fold_satisfied
#check Program.closing_satisfied
#check Program.statement_satisfied
#check Program.public_satisfied
#check Program.rows_length_exact
#check Program.rows_imply_result
#check Result.exactInvocation
#check Program.satisfies_of_rowsIncluded

end tests.NebulaProductionPaperTerminalFoldManifestFor
