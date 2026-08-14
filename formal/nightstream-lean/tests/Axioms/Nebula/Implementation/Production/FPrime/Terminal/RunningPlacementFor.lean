import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.RunningPlacementFor
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductionPaperTerminalRunningPlacementFor

/-- info: 'Nightstream.Implementation.Nebula.ProductionPaperTerminalRunningPlacementFor.toVerifierInputPlacement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms toVerifierInputPlacement
