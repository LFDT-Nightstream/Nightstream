import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayExecution

/-! Focused surface for the Rust-conformant streaming claim-replay artifact. -/

namespace tests.FPrimeFullHistoryStreamingClaimReplay

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution

#check artifact_valid
#check exact_shape
#check exact_leaf_counts
#check poseidon2_width_attribution_exact
#check canonical_call_refines
#check poseidon2_call_refines
#check glue_row_holds
#check full_execution
#check final_execution
#check full_execution_refines
#check final_execution_refines
#check full_rows_refine_declared_runtime
#check final_rows_refine_declared_runtime

end tests.FPrimeFullHistoryStreamingClaimReplay
