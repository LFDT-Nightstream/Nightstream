import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Schema

/-!
Executable certificate for generated raw-running decoder shard 1.

The `native_decide` input is exactly 252 proof-free `AllocationRecord`s with
four `Nat` fields each, six scalar metadata values, and a `Fin 252` traversal.
No other shard or proof-bearing structure is traversed.

Owns: the exact executable certificate for generated shard 1, containing
exactly 252 proof-free `AllocationRecord`s.

Does not own: any other shard, global record flattening, assignment semantics,
R1CS satisfaction, protocol acceptance, transcript scheduling, or commitment
authority.

Emits constraints: none; proof-only bounded certificate.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.census.chunk1` | check exactly one 252-record decoder shard | checked artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk1

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

theorem exact : ExactChunk ⟨1, by decide⟩
    Generated.Chunk1.schemaVersion
    Generated.Chunk1.sourceArm
    Generated.Chunk1.childCount
    Generated.Chunk1.logicalColumnCount
    Generated.Chunk1.finalColumnCount
    Generated.Chunk1.allocationRecords := by
  unfold ExactChunk ExactRecords
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk1
