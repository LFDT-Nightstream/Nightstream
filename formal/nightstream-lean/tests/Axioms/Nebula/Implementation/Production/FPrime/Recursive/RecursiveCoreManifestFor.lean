import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.RecursiveCoreManifestFor
import tests.Axioms.Support

/-! Dependency gate for recursive-core row containment and extraction. -/

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionRecursiveCoreManifestFor

open Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor.Program.currentMemory_satisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.currentMemory_satisfied

/-- info: 'Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor.Program.rows_imply_recursive_rowsHold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.rows_imply_recursive_rowsHold

/-- info: 'Nightstream.Implementation.Nebula.ProductionRecursiveCoreManifestFor.Program.satisfies_of_rowsIncluded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Program.satisfies_of_rowsIncluded

end tests.Axioms.NebulaProductionRecursiveCoreManifestFor
