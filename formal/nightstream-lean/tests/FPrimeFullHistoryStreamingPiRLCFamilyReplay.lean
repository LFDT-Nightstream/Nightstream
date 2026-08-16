import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyReplayArtifact

/-! Focused surface for the Rust-conformant PiRLC family replay artifact. -/

set_option autoImplicit false

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyReplay

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact

#check artifact_valid
#check exact_shape
#check execution
#check execution_refines
#check replay_eq_absorbSlice
#check FamilyStatesPlaced
#check ReplayValuesPlaced
#check family_replays_exact

end tests.FPrimeFullHistoryStreamingPiRLCFamilyReplay
