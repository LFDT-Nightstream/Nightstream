import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleSemanticLinkNecessity
import tests.Axioms.Support

/-! Fail-closed axiom guard for the lifecycle semantic-link omission proof. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLinkNecessity

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLinkNecessity.exact_removal_counterexample' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_removal_counterexample
