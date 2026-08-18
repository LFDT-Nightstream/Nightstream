import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLinkNecessity
import tests.Axioms.Support

/-! Fail-closed axiom guard for the exact terminal Nebula-state-digest omission counterexample. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLinkNecessity

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLinkNecessity.exact_removal_counterexample' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_removal_counterexample
