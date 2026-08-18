import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeRelationArtifact
import tests.Axioms.Support

/-! Fail-closed axiom guard for exact Prelude source-row refinement. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelationArtifact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeRelationArtifact.source_rows_imply_holds' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms source_rows_imply_holds
