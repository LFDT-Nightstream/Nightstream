import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Schema

/-!
Executable certificate for generated raw-running decoder shard zero.

The `native_decide` input is exactly 252 proof-free `AllocationRecord`s with
five `Nat` fields and one `Encoding` tag each, five scalar metadata values,
and a `Fin 252` traversal.
No other shard or proof-bearing structure is traversed.

Owns: the exact executable certificate for generated shard zero, containing
exactly 252 proof-free `AllocationRecord`s.

Does not own: any other shard, global record flattening, assignment semantics,
R1CS satisfaction, protocol acceptance, transcript scheduling, or commitment
authority.

Emits constraints: none; proof-only bounded certificate.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.census.chunk0` | check exactly one 252-record decoder shard | checked artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk0

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

theorem exact : ExactChunk ⟨0, by decide⟩
    Generated.Chunk0.schemaVersion
    Generated.Chunk0.sourceArm
    Generated.Chunk0.childCount
    Generated.Chunk0.logicalColumnCount
    Generated.Chunk0.finalColumnCount
    Generated.Chunk0.allocationRecords := by
  unfold ExactChunk ExactRecords
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census.Chunk0
