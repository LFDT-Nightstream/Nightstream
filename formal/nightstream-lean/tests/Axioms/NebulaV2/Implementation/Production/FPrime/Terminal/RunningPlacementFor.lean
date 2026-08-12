import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.RunningPlacementFor
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ProductionPaperTerminalRunningPlacementFor

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperTerminalRunningPlacementFor.toVerifierInputPlacement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms toVerifierInputPlacement
