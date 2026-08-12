import Nightstream.Protocol.NebulaV2.ProductionBatchedAugmentedLifecycle

/-! Regression surface for the exact batch-aware augmented F-prime chain. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionBatchedAugmentedLifecycle

open Nightstream.Protocol.NebulaV2.ProductionBatchedAugmentedLifecycle

#check ProducedChain.toApplicationChain
#check ProducedChain.rows_length
#check ProducedChain.AllLinked.every_linked
#check DelayedRun.finish_segment
#check DelayedRun.prepend_segment
#check SegmentChain.toDelayedRun
#check CompleteRun.exact_claim_count
#check CompleteRun.application_rows_length
#check CompleteRun.exact_delayed_lifecycle
#check CompleteRun.complete_schedule
#check CompleteRun.application_executes
#check CompleteRun.final_application_valid
#check CompleteRun.every_claim_accepted
#check CompleteRun.every_claim_linked_to_application_batch
#check CompleteRun.augmented_invocation_count

end tests.NebulaV2ProductionBatchedAugmentedLifecycle
