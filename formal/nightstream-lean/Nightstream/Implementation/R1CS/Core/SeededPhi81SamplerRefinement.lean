import Nightstream.Implementation.R1CS.Core.ChaCha8Refinement
import Nightstream.Implementation.R1CS.Core.SeededPhi81

/-!
Refinement of production seeded-Phi81 sampling to the pure stream semantics.

Assurance tier: implementation correspondence. The production compact block
uses the machine-efficient stream; this file proves its complete sampled
rotation schedule equals the canonical sampler instantiated with pure ChaCha8.

Owns: fast-to-pure substitution for every seed, cursor, candidate vector,
replacement, chunk, message column, and output in a schedule.

Does not own: Rust `rand_chacha` conformance; proof that a production schedule
succeeds; verifier-owned seed derivation; Phi81 rotation; SIS security; R1CS
rows; Poseidon2; transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: the result contains only the pure stream on the semantic
side. It follows from the generic ChaCha8 theorem, not from fixture equality or
a digest of sampled coefficients.

| Protocol | Phase | Constraint family | Theorem | Exact guarantee |
|---|---|---|---|---|
| seeded SIS | coefficient sampling | complete schedule | `scheduleBaseRotations_eq_pure` | all fast sampled rotations equal pure-stream sampling |
| seeded SIS | coefficient sampling | compact block | `blockBaseRotations_eq_pure` | the block coefficient source is the pure sampled schedule |
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81SamplerRefinement

theorem scheduleBaseRotations_eq_pure
    (schedule : SeededPhi81.SeedSchedule) (messageCols : Nat) :
    schedule.baseRotations messageCols =
      SeededPhi81Sampler.Schedule.baseRotations schedule
        SeededPhi81Sampler.pureStream messageCols := by
  unfold SeededPhi81.SeedSchedule.baseRotations
  apply SeededPhi81Sampler.Schedule.baseRotations_congr
  intro seed wordStart count
  exact ChaCha8Refinement.u64s_eq seed wordStart count

theorem blockBaseRotations_eq_pure (block : SeededPhi81.Block) :
    block.baseRotations =
      (SeededPhi81Sampler.Schedule.baseRotations block.schedule
        SeededPhi81Sampler.pureStream block.messageCols).getD [] := by
  unfold SeededPhi81.Block.baseRotations
  rw [scheduleBaseRotations_eq_pure]

end Nightstream.Implementation.R1CS.SeededPhi81SamplerRefinement
