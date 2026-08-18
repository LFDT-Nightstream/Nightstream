import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmissionNecessity
import tests.Axioms.Support

/-! Fail-closed axiom guard for the base verifier-key omission certificate. -/

open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmissionNecessity

/-- info: 'Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeyOmissionNecessity.exact_removal_counterexample' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_removal_counterexample
