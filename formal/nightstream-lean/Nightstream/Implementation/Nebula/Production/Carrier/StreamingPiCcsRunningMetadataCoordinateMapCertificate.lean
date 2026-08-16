import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateMaps

/-!
Contract: verifier-owned sampler certificate for the running-metadata PiCCS
coordinate map.

Assurance tier: executable setup certificate.

Owns the complete bounded sampler check for exactly two outputs and 47,068
message columns under the fixed `0xCA` seed profile.

Does not own Rust rows, physical columns, the statement-and-fresh map,
Module-SIS hardness, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRunningMetadataCoordinateMapCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps

/-- The closed sampler check below owns exactly 94,136 sampled vectors. It is
one complete semantic map, not a shard of a larger sampler execution. -/
theorem exact_certificate_input :
    MapKind.runningMetadata.certificateBlock.messageCols = 47_068 /\
      MapKind.runningMetadata.certificateBlock.kappa = 2 /\
      MapKind.runningMetadata.certificateBlock.kappa *
          MapKind.runningMetadata.certificateBlock.messageCols = 94_136 /\
      MapKind.runningMetadata.certificateBlock.schedule.chunkSize = 32_768 /\
      MapKind.runningMetadata.certificateBlock.schedule.seedsByOutput.length = 2 /\
      (∀ seeds ∈
        MapKind.runningMetadata.certificateBlock.schedule.seedsByOutput,
        seeds.length = 2) := by
  native_decide

theorem certificateBlock_valid :
    MapKind.runningMetadata.certificateBlock.Valid := by
  native_decide

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsRunningMetadataCoordinateMapCertificate
