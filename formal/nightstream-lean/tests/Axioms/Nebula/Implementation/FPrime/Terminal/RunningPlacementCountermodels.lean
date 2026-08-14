import Nightstream.Implementation.Nebula.FPrime.Terminal.RunningPlacementCountermodels
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.TerminalRunningPlacementCountermodels

/-- info: 'Nightstream.Implementation.Nebula.TerminalRunningPlacementCountermodels.omitted_placement_allows_wrong_carrier' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms omitted_placement_allows_wrong_carrier
