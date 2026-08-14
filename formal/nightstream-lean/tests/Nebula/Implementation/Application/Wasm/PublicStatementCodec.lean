import Nightstream.Implementation.Nebula.Application.Wasm.PublicStatementCodec
import tests.Nebula.Implementation.Application.Wasm.ResultCodec

set_option autoImplicit false

namespace tests.NebulaWasmPublicStatementCodec

open Nightstream.Implementation.Nebula.WasmPublicStatementCodec
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Profile
open Nightstream.Protocol.Nebula.Soundness
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.Protocol.Nebula.WasmStatement

def zeroDigest := tests.NebulaWasmResultCodec.zeroDigest

def verifierKey : VerifierKeyIdentity Digest.Value where
  digest := zeroDigest
  relationManifestDigest := zeroDigest
  laneLayoutDigest := zeroDigest
  setupManifestDigest := zeroDigest
  transcriptManifestDigest := zeroDigest
  codecManifestDigest := zeroDigest
  terminalManifestDigest := zeroDigest
  applicationStateSchemaDigest := zeroDigest

def identity : StatementIdentity Digest.Value where
  profile := Profile.v2
  verifierKey := verifierKey
  applicationRelationDigest := zeroDigest
  programDigest := zeroDigest
  memoryPlanDigest := zeroDigest

def initialImage : Fin scannedCells → Nat := fun _ => 0

def base : PublicStatement Unit WasmState.AppStateVector Digest.Value where
  identity := identity
  program := ()
  initialApplicationState := tests.NebulaWasmState.running
  initialImage := initialImage
  initialImageInRange := by
    intro index
    norm_num [initialImage, valueLimit]
  segmentCount := 1
  finalGlobalTimestamp := 1
  expectedResult := tests.NebulaWasmResultCodec.result

def statement : ProductionStatement Unit where
  base := base
  initialApplicationStateValid := tests.NebulaWasmState.validRunning
  resultImage := tests.NebulaWasmResultCodec.image
  resultDecoded := tests.NebulaWasmResultCodec.decoded

def image : PublicImage := PublicImage.ofStatement statement

def decoded : image.Decodes statement where
  exactImage := rfl
  exactProfile := rfl
  segmentCountPositive := by decide
  segmentCountBound := by decide
  finalTimestampBound := by decide
  realRowsFitDeclaredSegments := by decide
  smallestSegmentCount := by decide

def decodedFor : image.DecodesFor Profile.v2 statement :=
  decoded.toDecodesFor

#check encodeIdentity_injective_on_profile
#check encode_injective_of_decodesFor

theorem exact_statement_length : (encode image).length = 7868 :=
  encode_length image

theorem accepted_statement_has_unique_encoding
    (other : PublicImage)
    (otherStatement : ProductionStatement Unit)
    (otherDecoded : other.Decodes otherStatement)
    (same : encode image = encode other) :
    image = other :=
  encode_injective_of_decodes decoded otherDecoded same

end tests.NebulaWasmPublicStatementCodec
