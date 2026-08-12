import Nightstream.Implementation.NebulaV2.FPrime.Terminal.RunningPlacementCountermodels
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.TerminalRunningPlacementCountermodels

/-- info: 'Nightstream.Implementation.NebulaV2.TerminalRunningPlacementCountermodels.omitted_placement_allows_wrong_carrier' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms omitted_placement_allows_wrong_carrier
