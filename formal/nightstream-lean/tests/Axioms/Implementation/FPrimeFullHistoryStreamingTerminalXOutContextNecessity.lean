import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutContextNecessity
import tests.Axioms.Support

/-! Fail-closed axiom guard for the exact terminal XOut omission counterexample. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContextNecessity

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContextNecessity.exact_removal_counterexample' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_removal_counterexample
