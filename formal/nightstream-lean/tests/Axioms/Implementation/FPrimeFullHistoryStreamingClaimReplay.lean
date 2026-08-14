import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayExecution
import tests.Axioms.Support

/-! Fail-closed axiom guard for streaming claim replay. -/

namespace NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingClaimReplay

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.artifact_valid' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms artifact_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.exact_shape' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms exact_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.poseidon2_width_attribution_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms poseidon2_width_attribution_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.canonical_call_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms canonical_call_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.poseidon2_call_refines' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms poseidon2_call_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.glue_row_holds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms glue_row_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.full_execution' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_execution

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.final_execution' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms final_execution

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.full_execution_refines' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_execution_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.final_execution_refines' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms final_execution_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.full_rows_refine_declared_runtime' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_rows_refine_declared_runtime

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.final_rows_refine_declared_runtime' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms final_rows_refine_declared_runtime

end NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingClaimReplay
