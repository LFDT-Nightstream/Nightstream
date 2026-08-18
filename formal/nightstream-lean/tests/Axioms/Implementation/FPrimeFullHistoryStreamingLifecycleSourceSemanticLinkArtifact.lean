import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleSourceSemanticLinkArtifact
import tests.Axioms.Support

/-! Fail-closed axiom guard for lifecycle source semantic-link refinement. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLinkArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLinkArtifact.base_rows_refine_semanticLink' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms base_rows_refine_semanticLink

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLinkArtifact.recursive_rows_refine_semanticLink' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms recursive_rows_refine_semanticLink
