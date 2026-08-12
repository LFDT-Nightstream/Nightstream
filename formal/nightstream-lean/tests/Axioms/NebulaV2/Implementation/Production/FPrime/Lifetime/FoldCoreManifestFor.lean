import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.FoldCoreManifestFor
import tests.Axioms.Support

/-! Dependency gate for the common recursive/terminal paper-fold manifest. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionPaperFoldCoreManifestFor

open Nightstream.Implementation.NebulaV2.ProductionPaperFoldCoreManifestFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperFoldCoreManifestFor.Program.rows_imply_recursive_rowsHold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.rows_imply_recursive_rowsHold

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperFoldCoreManifestFor.Program.satisfies_of_rowsIncluded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.satisfies_of_rowsIncluded

end tests.Axioms.NebulaV2ProductionPaperFoldCoreManifestFor
