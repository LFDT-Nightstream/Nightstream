import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.CoreRowsFor
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCoreRowsFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCoreRowsFor.public_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms public_exact

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalCoreRowsFor.evaluations_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluations_exact
