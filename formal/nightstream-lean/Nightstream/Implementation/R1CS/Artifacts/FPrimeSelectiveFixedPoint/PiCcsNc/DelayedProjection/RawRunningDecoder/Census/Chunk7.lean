import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Schema

/-!
Executable certificate for generated raw-running decoder shard 7.

The `native_decide` input is exactly 252 proof-free `AllocationRecord`s with
five `Nat` fields and one `Encoding` tag each, five scalar metadata values,
and a `Fin 252` traversal.
No other shard or proof-bearing structure is traversed.

Owns: the exact executable certificate for generated shard 7, containing
exactly 252 proof-free `AllocationRecord`s.

Does not own: any other shard, global record flattening, assignment semantics,
R1CS satisfaction, protocol acceptance, transcript scheduling, or commitment
authority.

Emits constraints: none; proof-only bounded certificate.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.census.chunk7` | check exactly one 252-record decoder shard | checked artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk7

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

theorem exact : ExactChunk ⟨7, by decide⟩
    Generated.Chunk7.schemaVersion
    Generated.Chunk7.sourceArm
    Generated.Chunk7.childCount
    Generated.Chunk7.logicalColumnCount
    Generated.Chunk7.finalColumnCount
    Generated.Chunk7.allocationRecords := by
  unfold ExactChunk ExactRecords
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk7
