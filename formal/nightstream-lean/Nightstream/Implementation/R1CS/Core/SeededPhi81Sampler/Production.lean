import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler.Schedule
import Nightstream.Implementation.R1CS.Core.SeededPhi81SamplerRefinement

/-!
Production seeded-Phi81 blocks refine the pure unbounded sampler semantics.

Assurance tier: implementation correspondence. A production `Block.Valid`
certificate supplies bounded sampler success. Generic ChaCha8 refinement then
replaces the optimized stream with the pure stream, and the schedule theorem
derives the nested unbounded first-acceptance semantics.

Owns: the bridge from a valid compact block to pure complete-schedule
semantics; identification of the exact coefficient lists used by the SIS
linear map; and their canonicality.

Does not own: Rust `rand_chacha` conformance; verifier-owned master/chunk seed
derivation; Phi81 rotation; SIS security; R1CS row soundness; Poseidon2;
transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: `Block.Valid` is used only to establish that the bounded
sampler returned `some`. The meaning of those outputs comes from the
independent `SamplesSchedule` relation over pure ChaCha8, not from the validity
Boolean or a fixture digest.

| Protocol | Phase | Mathematical branch | Definition/theorem | Exact guarantee |
|---|---|---|---|---|
| seeded SIS | coefficient sampling | successful production execution | `PureBlockSampling.of_fastSuccess` | one bounded-success fact implies pure execution, unbounded schedule semantics, and coefficient identity |
| seeded SIS | coefficient sampling | valid production block | `Block.Valid.refines_pureSampling` | every valid compact block has a `PureBlockSampling` witness |
| seeded SIS | coefficient sampling | coefficient canonicality | `Block.Valid.baseRotations_canonical` | every coefficient actually used by the block is below the field modulus |
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81Sampler

/-- Complete semantic witness for the coefficient source of one compact SIS
block. -/
structure PureBlockSampling (block : SeededPhi81.Block)
    (outputs : List (List (List Nat))) : Prop where
  pureExecution :
    Schedule.baseRotations block.schedule pureStream block.messageCols =
      some outputs
  scheduleSemantics :
    SamplesSchedule pureStream block.messageCols block.schedule.chunkSize
      block.schedule.seedsByOutput outputs
  coefficientIdentity : block.baseRotations = outputs

/-- Bounded execution success is the only concrete premise needed to assign
the independent pure/unbounded meaning to one block's coefficient tensor. -/
theorem PureBlockSampling.of_fastSuccess
    {block : SeededPhi81.Block} {outputs : List (List (List Nat))}
    (fastSuccess :
      block.schedule.baseRotations block.messageCols = some outputs) :
    PureBlockSampling block outputs := by
  have streamRefinement :=
    SeededPhi81SamplerRefinement.scheduleBaseRotations_eq_pure
      block.schedule block.messageCols
  have pureSuccess :
      Schedule.baseRotations block.schedule pureStream block.messageCols =
        some outputs := by
    rw [← streamRefinement]
    exact fastSuccess
  have coefficientIdentity : block.baseRotations = outputs := by
    unfold SeededPhi81.Block.baseRotations
    rw [fastSuccess]
    rfl
  exact
    { pureExecution := pureSuccess
      scheduleSemantics := Schedule.baseRotations_sound pureSuccess
      coefficientIdentity := coefficientIdentity }

theorem SeededPhi81.Block.Valid.refines_pureSampling
    {block : SeededPhi81.Block} (valid : block.Valid) :
    exists outputs, PureBlockSampling block outputs := by
  rcases valid.baseRotations_success with ⟨outputs, fastSuccess⟩
  exact ⟨outputs, PureBlockSampling.of_fastSuccess fastSuccess⟩

theorem PureBlockSampling.vectors_canonical
    {block : SeededPhi81.Block} {outputs : List (List (List Nat))}
    (sampling : PureBlockSampling block outputs) :
    forall output, output ∈ outputs ->
      forall vector, vector ∈ output ->
        forall value, value ∈ vector -> value < modulus :=
  sampling.scheduleSemantics.vectors_canonical

theorem SeededPhi81.Block.Valid.baseRotations_canonical
    {block : SeededPhi81.Block} (valid : block.Valid) :
    forall output, output ∈ block.baseRotations ->
      forall vector, vector ∈ output ->
      forall value, value ∈ vector -> value < modulus := by
  rcases SeededPhi81.Block.Valid.refines_pureSampling valid with
    ⟨outputs, sampling⟩
  rw [sampling.coefficientIdentity]
  exact sampling.vectors_canonical

end Nightstream.Implementation.R1CS.SeededPhi81Sampler
