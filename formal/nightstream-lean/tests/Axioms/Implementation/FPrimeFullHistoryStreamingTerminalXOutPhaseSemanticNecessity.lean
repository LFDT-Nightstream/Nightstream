import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticNecessity
import tests.Axioms.Support

/-! Fail-closed axiom guard for the exact terminal phase-semantic omission counterexample. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemanticNecessity

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemanticNecessity.exact_removal_counterexample' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_removal_counterexample
