import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecyclePayloadDomainNecessity
import tests.Axioms.Support

/-! Fail-closed axiom guard for the lifecycle payload-domain omission proof. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecyclePayloadDomainNecessity

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecyclePayloadDomainNecessity.exact_removal_counterexample' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_removal_counterexample
