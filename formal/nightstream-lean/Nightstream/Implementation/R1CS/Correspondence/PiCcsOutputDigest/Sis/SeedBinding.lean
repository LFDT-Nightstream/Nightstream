import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.ProductionBinding
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.Sis.SeedDerivation

/-!
Production metadata refinement to the independent `Pi_CCS` SIS seed profile.

Assurance tier: implementation/R1CS correspondence. The theorems below
kernel-evaluate the independently specified two-stage ChaCha8 seed derivation
and prove that both production blocks carry exactly those schedules.

Owns: primary/compression chunk-width equality; exact derived seed equality;
rank/message geometry equality; and the fixed rejection-fuel metadata.

Does not own: proof that the pure Lean ChaCha8 model equals Rust
`rand_chacha`; equality of `ChaCha8Fast` coefficient streams to that model;
coefficient rejection semantics; Phi81 rotation; Poseidon2; transcript
authority; row necessity; row removal; or cost totals.

Emits constraints: no.

Authority boundary: generated block seed literals are conclusions, not
premises. The master seeds and dimensions come from `SeedDerivation`; block
metadata is proved equal to their deterministic expansion.

| Protocol | Phase | Constraint family | Theorem | Exact guarantee |
|---|---|---|---|---|
| `Pi_CCS` | output digest | primary SIS seed | `primarySeeds_derived` | both block-8 chunk seeds derive from public master `0xC3` |
| `Pi_CCS` | output digest | compression SIS seed | `compressionSeeds_derived` | block-9 chunk seed derives from public master `0xC6` |
| `Pi_CCS` | output digest | chunk geometry | `primaryGeometry` / `compressionGeometry` | rank, message width, and chunk width match the profile |
| `Pi_CCS` | output digest | sampler bound | `rejectionFuel_eq` | both compact schedules carry fuel 16 |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding

open Nightstream.Implementation.R1CS

theorem primaryGeometry :
    ProductionBinding.primaryBlock.kappa = SeedDerivation.primaryKappa /\
    ProductionBinding.primaryBlock.messageCols =
      SeedDerivation.primaryMessageCols /\
    ProductionBinding.primaryBlock.schedule.chunkSize =
      SeedDerivation.primarySchedule.chunkSize := by
  decide

theorem compressionGeometry :
    ProductionBinding.compressionBlock.kappa =
      SeedDerivation.compressionKappa /\
    ProductionBinding.compressionBlock.messageCols =
      SeedDerivation.compressionMessageCols /\
    ProductionBinding.compressionBlock.schedule.chunkSize =
      SeedDerivation.compressionSchedule.chunkSize := by
  decide

private theorem primarySeedRows_length :
    ProductionBinding.primaryBlock.schedule.seedsByOutput.length =
      SeedDerivation.primarySchedule.seedsByOutput.length := by
  decide

private theorem primarySeedRow_zero :
    ProductionBinding.primaryBlock.schedule.seedsByOutput.getD 0 [] =
      SeedDerivation.primarySchedule.seedsByOutput.getD 0 [] := by
  set_option maxRecDepth 8192 in
    decide

private theorem primarySeedRow_one :
    ProductionBinding.primaryBlock.schedule.seedsByOutput.getD 1 [] =
      SeedDerivation.primarySchedule.seedsByOutput.getD 1 [] := by
  set_option maxRecDepth 8192 in
    decide

/-- Exact production-primary seed derivation, with no generated seed on the
right-hand side. The two verifier outputs are evaluated independently so the
proof does not hide the schedule boundary inside one large reduction. -/
theorem primarySeeds_derived :
    ProductionBinding.primaryBlock.schedule.seedsByOutput =
      SeedDerivation.primarySchedule.seedsByOutput := by
  apply List.ext_get
  · exact primarySeedRows_length
  · intro index leftLt rightLt
    have indexLt : index < 2 := by
      simpa [ProductionBinding.primaryBlock,
        FPrimeFullHistorySeededPhi81.block8] using leftLt
    have indexCases : index = 0 \/ index = 1 := by omega
    rcases indexCases with rfl | rfl
    · simpa using primarySeedRow_zero
    · simpa using primarySeedRow_one

private theorem compressionSeedRows_length :
    ProductionBinding.compressionBlock.schedule.seedsByOutput.length =
      SeedDerivation.compressionSchedule.seedsByOutput.length := by
  decide

private theorem compressionSeedRow_zero :
    ProductionBinding.compressionBlock.schedule.seedsByOutput.getD 0 [] =
      SeedDerivation.compressionSchedule.seedsByOutput.getD 0 [] := by
  set_option maxRecDepth 8192 in
    decide

/-- Exact production-compression seed derivation. -/
theorem compressionSeeds_derived :
    ProductionBinding.compressionBlock.schedule.seedsByOutput =
      SeedDerivation.compressionSchedule.seedsByOutput := by
  apply List.ext_get
  · exact compressionSeedRows_length
  · intro index leftLt rightLt
    have indexEq : index = 0 := by
      have indexLt : index < 1 := by
        simpa [ProductionBinding.compressionBlock,
          FPrimeFullHistorySeededPhi81.block9] using leftLt
      omega
    subst index
    simpa using compressionSeedRow_zero

theorem rejectionFuel_eq :
    ProductionBinding.primaryBlock.schedule.rejectionFuel = 16 /\
    ProductionBinding.compressionBlock.schedule.rejectionFuel = 16 := by
  decide

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedBinding
