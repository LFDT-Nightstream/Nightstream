import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.FoldCoreManifestFor
import tests.Axioms.Support

/-! Dependency gate for the common recursive/terminal paper-fold manifest. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionPaperFoldCoreManifestFor

open Nightstream.Implementation.Nebula.ProductionPaperFoldCoreManifestFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperFoldCoreManifestFor.Program.rows_imply_recursive_rowsHold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.rows_imply_recursive_rowsHold

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperFoldCoreManifestFor.Program.satisfies_of_rowsIncluded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.satisfies_of_rowsIncluded

end tests.Axioms.NebulaProductionPaperFoldCoreManifestFor
