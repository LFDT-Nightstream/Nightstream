import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.SeedBinding
import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler.Production
import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler.NoRejection

/-!
Exact production `Pi_CCS` SIS sampler bridge, with the finite computation
boundary exposed rather than trusted.

Assurance tier: conditional implementation/protocol correspondence. The
generic no-rejection and bounded-to-unbounded theorems are proved. The three
fixed production runs remain explicit fields of
`ProductionInitialAcceptance`; no generated validity bit, native evaluator,
legacy circuit, row count, or digest discharges those fields.

Owns: verifier-derived production seed aliases; the exact three finite
initial-acceptance obligations; composition of those obligations into the
two complete bounded schedules; and refinement to pure unbounded
first-acceptance semantics.

Does not own: kernel certificates for the three finite acceptance fields;
Rust `rand_chacha` conformance; proof that dynamic message columns are
authoritative `Pi_CCS` outputs; Phi81 rotation correctness; Poseidon2;
transcript authority; row necessity; row removal; or cost totals.

Emits constraints: no.

Authority boundary: `ProductionInitialAcceptance` is deliberately visible in
every production theorem. Until its three fields are closed by auditable
kernel proofs over the verifier-derived seeds, the production coefficient
bridge is conditional and must not authorize a constraint removal.

| Protocol | Phase | Mathematical branch | Definition/theorem | Exact guarantee |
|---|---|---|---|---|
| `Pi_CCS` | output digest | seed authority | `primarySeed0_derived`, `primarySeed1_derived`, `compressionSeed0_derived` | each literal certificate seed equals the verifier-derived schedule seed |
| `Pi_CCS` | output digest | finite stream boundary | `ProductionInitialAcceptance` | names all and only the three fixed no-rejection facts still requiring kernel certificates |
| `Pi_CCS` | output digest | bounded execution | `primaryFastSuccess`, `compressionFastSuccess` | the exact production schedules return concrete coefficient tensors |
| `Pi_CCS` | output digest | pure semantics | `primaryBlock_pureSampling`, `compressionBlock_pureSampling` | those tensors satisfy independent unbounded first-acceptance semantics |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding

open Nightstream.Implementation.R1CS

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

def primarySeed0 : List Nat :=
  [216, 10, 232, 51, 138, 24, 205, 89,
   240, 118, 170, 228, 175, 184, 181, 191,
   147, 71, 39, 190, 69, 239, 98, 77,
   67, 15, 47, 136, 189, 79, 134, 243]

def primarySeed1 : List Nat :=
  [227, 116, 17, 69, 75, 42, 154, 78,
   131, 122, 13, 46, 159, 245, 178, 93,
   250, 165, 184, 133, 218, 223, 237, 244,
   211, 177, 48, 249, 175, 40, 115, 61]

def compressionSeed0 : List Nat :=
  [222, 97, 160, 75, 169, 146, 205, 28,
   66, 7, 37, 46, 38, 226, 240, 160,
   130, 181, 109, 118, 6, 248, 19, 168,
   202, 255, 83, 20, 122, 228, 97, 38]

/-- The first primary certificate seed is exactly verifier-derived. -/
theorem primarySeed0_derived :
    primarySeed0 =
      ((SeedDerivation.primarySchedule.seedsByOutput.getD 0 []).getD 0 []) := by
  set_option maxRecDepth 8192 in
    decide

/-- The second primary certificate seed is exactly verifier-derived. -/
theorem primarySeed1_derived :
    primarySeed1 =
      ((SeedDerivation.primarySchedule.seedsByOutput.getD 1 []).getD 0 []) := by
  set_option maxRecDepth 8192 in
    decide

/-- The compression certificate seed is exactly verifier-derived. -/
theorem compressionSeed0_derived :
    compressionSeed0 =
      ((SeedDerivation.compressionSchedule.seedsByOutput.getD 0 []).getD 0 []) := by
  set_option maxRecDepth 8192 in
    decide

/-- Irreducible finite facts needed to show that production never enters the
replacement path. Each field covers every initial coefficient in one exact
production seed stream. -/
structure ProductionInitialAcceptance : Prop where
  primary0 : SeededPhi81Sampler.InitialRunAccepted ChaCha8Fast.u64s
    primarySeed0 5075 0
  primary1 : SeededPhi81Sampler.InitialRunAccepted ChaCha8Fast.u64s
    primarySeed1 5075 0
  compression0 : SeededPhi81Sampler.InitialRunAccepted ChaCha8Fast.u64s
    compressionSeed0 82 0

private theorem sampleOutput_single_success
    {stream : SeededPhi81Sampler.WordStream} {seed : List Nat}
    {messageCols chunkSize fuel : Nat} {vectors : List (List Nat)}
    (chunkCount :
      SeededPhi81Sampler.chunkMessageCount messageCols chunkSize 0 =
        messageCols)
    (vectorsSuccess :
      SeededPhi81Sampler.sampleVectors stream seed fuel messageCols 0 =
        some vectors) :
    SeededPhi81Sampler.sampleOutput stream messageCols chunkSize fuel 0
      [seed] = some vectors := by
  simp only [SeededPhi81Sampler.sampleOutput]
  rw [chunkCount, vectorsSuccess]
  simp

private theorem sampleSchedule_two_single_success
    {stream : SeededPhi81Sampler.WordStream} {seed0 seed1 : List Nat}
    {messageCols chunkSize fuel : Nat} {output0 output1 : List (List Nat)}
    (output0Success :
      SeededPhi81Sampler.sampleOutput stream messageCols chunkSize fuel 0
        [seed0] = some output0)
    (output1Success :
      SeededPhi81Sampler.sampleOutput stream messageCols chunkSize fuel 0
        [seed1] = some output1) :
    SeededPhi81Sampler.sampleScheduleOutputs stream messageCols chunkSize fuel
      [[seed0], [seed1]] = some [output0, output1] := by
  simp only [SeededPhi81Sampler.sampleScheduleOutputs]
  rw [output0Success, output1Success]

private theorem sampleSchedule_one_single_success
    {stream : SeededPhi81Sampler.WordStream} {seed : List Nat}
    {messageCols chunkSize fuel : Nat} {output : List (List Nat)}
    (outputSuccess :
      SeededPhi81Sampler.sampleOutput stream messageCols chunkSize fuel 0
        [seed] = some output) :
    SeededPhi81Sampler.sampleScheduleOutputs stream messageCols chunkSize fuel
      [[seed]] = some [output] := by
  simp only [SeededPhi81Sampler.sampleScheduleOutputs]
  rw [outputSuccess]

/-- The two exact primary runs make the complete rank-two bounded schedule
return a coefficient tensor. -/
theorem primaryFastSuccess (accepted : ProductionInitialAcceptance) :
    exists outputs,
      ProductionBinding.primaryBlock.schedule.baseRotations
        ProductionBinding.primaryBlock.messageCols = some outputs := by
  rcases SeededPhi81Sampler.sampleVectors_exists_of_initiallyAccepted
      (fuel := ProductionBinding.primaryBlock.schedule.rejectionFuel)
      accepted.primary0 with
    ⟨output0, output0Success⟩
  rcases SeededPhi81Sampler.sampleVectors_exists_of_initiallyAccepted
      (fuel := ProductionBinding.primaryBlock.schedule.rejectionFuel)
      accepted.primary1 with
    ⟨output1, output1Success⟩
  have chunkCount :
      SeededPhi81Sampler.chunkMessageCount 5075 5075 0 = 5075 := by
    decide
  have output0Schedule := sampleOutput_single_success
    chunkCount output0Success
  have output1Schedule := sampleOutput_single_success
    chunkCount output1Success
  have scheduleSuccess := sampleSchedule_two_single_success
    output0Schedule output1Schedule
  refine ⟨[output0, output1], ?_⟩
  simpa only [SeededPhi81.SeedSchedule.baseRotations,
    SeededPhi81Sampler.Schedule.baseRotations,
    ProductionBinding.primaryBlock,
    FPrimeFullHistorySeededPhi81.block8,
    primarySeed0, primarySeed1] using scheduleSuccess

/-- The exact compression run makes the complete rank-one bounded schedule
return a coefficient tensor. -/
theorem compressionFastSuccess (accepted : ProductionInitialAcceptance) :
    exists outputs,
      ProductionBinding.compressionBlock.schedule.baseRotations
        ProductionBinding.compressionBlock.messageCols = some outputs := by
  rcases SeededPhi81Sampler.sampleVectors_exists_of_initiallyAccepted
      (fuel := ProductionBinding.compressionBlock.schedule.rejectionFuel)
      accepted.compression0 with
    ⟨output, outputSuccess⟩
  have chunkCount :
      SeededPhi81Sampler.chunkMessageCount 82 1024 0 = 82 := by
    decide
  have outputSchedule := sampleOutput_single_success
    chunkCount outputSuccess
  have scheduleSuccess := sampleSchedule_one_single_success outputSchedule
  refine ⟨[output], ?_⟩
  simpa only [SeededPhi81.SeedSchedule.baseRotations,
    SeededPhi81Sampler.Schedule.baseRotations,
    ProductionBinding.compressionBlock,
    FPrimeFullHistorySeededPhi81.block9,
    compressionSeed0] using scheduleSuccess

/-- Conditional exact coefficient meaning for the primary block. -/
theorem primaryBlock_pureSampling (accepted : ProductionInitialAcceptance) :
    exists outputs,
      SeededPhi81Sampler.PureBlockSampling
        ProductionBinding.primaryBlock outputs := by
  rcases primaryFastSuccess accepted with ⟨outputs, success⟩
  exact ⟨outputs,
    SeededPhi81Sampler.PureBlockSampling.of_fastSuccess success⟩

/-- Conditional exact coefficient meaning for the compression block. -/
theorem compressionBlock_pureSampling
    (accepted : ProductionInitialAcceptance) :
    exists outputs,
      SeededPhi81Sampler.PureBlockSampling
        ProductionBinding.compressionBlock outputs := by
  rcases compressionFastSuccess accepted with ⟨outputs, success⟩
  exact ⟨outputs,
    SeededPhi81Sampler.PureBlockSampling.of_fastSuccess success⟩

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SamplerBinding
