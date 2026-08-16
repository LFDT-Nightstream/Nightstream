import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateMaps

/-!
Contract: verifier-owned sampler certificate for the statement-and-fresh
PiCCS metadata coordinate map.

Assurance tier: executable setup certificate.

Owns the complete bounded sampler check for exactly two outputs and 19,474
message columns under the fixed `0xC8` seed profile.

Does not own Rust rows, physical columns, the running-metadata map,
Module-SIS hardness, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementFreshCoordinateMapCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps

/-- The closed sampler check below owns exactly 38,948 sampled vectors. It is
one complete semantic map, not a shard of a larger sampler execution. -/
theorem exact_certificate_input :
    MapKind.statementFresh.certificateBlock.messageCols = 19_474 /\
      MapKind.statementFresh.certificateBlock.kappa = 2 /\
      MapKind.statementFresh.certificateBlock.kappa *
          MapKind.statementFresh.certificateBlock.messageCols = 38_948 /\
      MapKind.statementFresh.certificateBlock.schedule.chunkSize = 19_474 /\
      MapKind.statementFresh.certificateBlock.schedule.seedsByOutput.length = 2 /\
      (∀ seeds ∈
        MapKind.statementFresh.certificateBlock.schedule.seedsByOutput,
        seeds.length = 1) := by
  native_decide

theorem certificateBlock_valid :
    MapKind.statementFresh.certificateBlock.Valid := by
  native_decide

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementFreshCoordinateMapCertificate
