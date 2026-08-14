import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.TypedFoldRowsFor
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductionPaperTerminalTypedFoldRowsFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalTypedFoldRowsFor.Frame.rows_satisfied_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Frame.rows_satisfied_iff
