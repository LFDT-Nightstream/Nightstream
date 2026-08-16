import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsRunningMetadataCoordinateMapCertificate
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsStatementFreshCoordinateMapCertificate

/-!
Contract: complete verifier-owned sampler-certificate registry for the two
PiCCS metadata coordinate maps.

Assurance tier: executable setup certificate composition.

Owns exact, duplicate-free coverage of both map kinds and dispatch to their
separate bounded sampler certificates.

Does not own Rust rows, physical placement, Module-SIS hardness, or lifecycle
integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMapCertificates

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps

def allMapKinds : List MapKind := [.statementFresh, .runningMetadata]

theorem allMapKinds_exact :
    allMapKinds.length = 2 /\ allMapKinds.Nodup /\
      (∀ kind, kind ∈ allMapKinds) := by
  constructor
  · decide
  constructor
  · decide
  · intro kind
    cases kind <;> decide

theorem certificateBlock_valid (kind : MapKind) :
    kind.certificateBlock.Valid := by
  cases kind with
  | statementFresh =>
      exact ProductionStreamingPiCcsStatementFreshCoordinateMapCertificate.certificateBlock_valid
  | runningMetadata =>
      exact ProductionStreamingPiCcsRunningMetadataCoordinateMapCertificate.certificateBlock_valid

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMapCertificates
