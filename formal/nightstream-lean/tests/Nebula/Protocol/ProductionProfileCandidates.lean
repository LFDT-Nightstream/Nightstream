import Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-! Regression checks for successor field-native profile candidates. -/

set_option autoImplicit false

namespace tests.NebulaProductionProfileCandidates

open Nightstream.Protocol.Nebula.ProductionProfileCandidates

example : (identity .e8).version = 5 := rfl
example : (identity .e8).checkedStepsPerFreshClaim = 8 := rfl

#check identities_pairwise_distinct
#check exact_segment_partition
#check local_batch_end_le_segment
#check candidate_count_table
#check memorySuffixCoordinate_split_exact
#check fieldNativeEnvelopeCoordinate_table
#check fieldNativeEnvelopeCoordinate_table_at_26

example : runningFieldCoordinatesFor 26 = 83212 := by decide

end tests.NebulaProductionProfileCandidates
