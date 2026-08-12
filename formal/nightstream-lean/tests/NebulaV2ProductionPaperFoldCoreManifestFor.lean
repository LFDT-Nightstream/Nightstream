import Nightstream.Implementation.NebulaV2.ProductionPaperFoldCoreManifestFor

/-! Surface gate for the common recursive/terminal paper-fold manifest. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionPaperFoldCoreManifestFor

open Nightstream.Implementation.NebulaV2.ProductionPaperFoldCoreManifestFor

#check Program.piCcs_satisfied
#check Program.piRlcTranscript_satisfied
#check Program.nifsOutput_satisfied
#check Program.rows_imply_recursive_rowsHold
#check Program.rows_length_exact
#check Program.satisfies_of_rowsIncluded

end tests.NebulaV2ProductionPaperFoldCoreManifestFor
