import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.TypedFoldRowsFor
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalTypedFoldRowsFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalTypedFoldRowsFor.Frame.rows_satisfied_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Frame.rows_satisfied_iff
