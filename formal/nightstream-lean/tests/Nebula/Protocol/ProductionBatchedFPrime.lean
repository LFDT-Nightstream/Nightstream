import Nightstream.Protocol.Nebula.ProductionBatchedFPrime

/-! Regression surface for candidate-specific batched delayed consumption. -/

set_option autoImplicit false

namespace tests.NebulaProductionBatchedFPrime

open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

#check Transition.decreases_by_exact_factor
#check Transition.before_active
#check ConsumesList.after_unique
#check VerifiedRun.full_segment_has_exact_batch_count

example (batch : SuffixBatch .e8 Nat Nat Nat) :
    batch.suffixes.length = 8 := by
  simpa [checkedStepsPerFreshClaim] using batch.length_exact

/-- A factor-four claim cannot carry the one-suffix factor-one image. -/
example :
    ¬ ∃ batch : SuffixBatch .e4 Nat Nat Nat,
      batch.suffixes.length = 1 := by
  intro existsBatch
  rcases existsBatch with ⟨batch, one⟩
  have four : batch.suffixes.length = 4 := by
    simpa [checkedStepsPerFreshClaim] using batch.length_exact
  omega

end tests.NebulaProductionBatchedFPrime
