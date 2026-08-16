import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPublicArtifact

/-! Focused surface for the Rust-conformant PiRLC public-suffix artifact. -/

set_option autoImplicit false

namespace tests.FPrimeFullHistoryStreamingPiRLCFamilyPublic

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact

#check artifact_valid
#check exact_shape
#check exact_leaf_counts
#check exact_public_word_layout
#check exact_state_column_shape
#check exact_suffix_owner_chain
#check canonical_call_refines
#check poseidon2_call_refines
#check glue_row_holds

end tests.FPrimeFullHistoryStreamingPiRLCFamilyPublic
