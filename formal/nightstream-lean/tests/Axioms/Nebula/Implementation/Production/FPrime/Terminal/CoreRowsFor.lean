import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.CoreRowsFor
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductionPaperTerminalCoreRowsFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCoreRowsFor.public_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms public_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalCoreRowsFor.evaluations_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluations_exact
