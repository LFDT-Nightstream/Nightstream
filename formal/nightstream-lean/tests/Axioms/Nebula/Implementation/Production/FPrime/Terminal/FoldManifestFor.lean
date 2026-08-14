import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.FoldManifestFor
import tests.Axioms.Support

/-! Dependency gate for terminal trailing-fold and close extraction. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionPaperTerminalFoldManifestFor

open Nightstream.Implementation.Nebula.ProductionPaperTerminalFoldManifestFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalFoldManifestFor.Program.rows_imply_result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.rows_imply_result

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalFoldManifestFor.Result.exactInvocation' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Result.exactInvocation

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalFoldManifestFor.Program.satisfies_of_rowsIncluded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.satisfies_of_rowsIncluded

end tests.Axioms.NebulaProductionPaperTerminalFoldManifestFor
