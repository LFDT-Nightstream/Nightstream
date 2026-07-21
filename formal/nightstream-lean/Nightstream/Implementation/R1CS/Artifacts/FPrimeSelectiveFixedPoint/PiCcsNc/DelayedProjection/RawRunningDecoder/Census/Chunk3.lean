import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Schema

/-!
Executable certificate for generated raw-running decoder shard 3.

The `native_decide` input is exactly 252 proof-free `AllocationRecord`s with
four `Nat` fields each, six scalar metadata values, and a `Fin 252` traversal.
No other shard or proof-bearing structure is traversed.

Owns: the exact executable certificate for generated shard 3, containing
exactly 252 proof-free `AllocationRecord`s.

Does not own: any other shard, global record flattening, assignment semantics,
R1CS satisfaction, protocol acceptance, transcript scheduling, or commitment
authority.

Emits constraints: none; proof-only bounded certificate.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.census.chunk3` | check exactly one 252-record decoder shard | checked artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk3

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

theorem exact : ExactChunk ⟨3, by decide⟩
    Generated.Chunk3.schemaVersion
    Generated.Chunk3.sourceArm
    Generated.Chunk3.childCount
    Generated.Chunk3.logicalColumnCount
    Generated.Chunk3.finalColumnCount
    Generated.Chunk3.allocationRecords := by
  unfold ExactChunk ExactRecords
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk3
