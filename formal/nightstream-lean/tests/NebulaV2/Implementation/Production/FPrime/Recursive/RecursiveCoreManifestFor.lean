import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.RecursiveCoreManifestFor

/-! Surface gate for the exact exponent-indexed recursive-core manifest. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionRecursiveCoreManifestFor

open Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreManifestFor

#check Program.fold_satisfied
#check Program.continuation_satisfied
#check Program.successor_satisfied
#check Program.currentMemory_satisfied
#check Program.rows_imply_recursive_rowsHold
#check Program.rows_length_exact
#check Program.satisfies_of_rowsIncluded

end tests.NebulaV2ProductionRecursiveCoreManifestFor
