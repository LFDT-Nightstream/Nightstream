import Nightstream.Protocol.NebulaV2.StatementAuthority
import tests.NebulaV2WasmPublicStatementCodec

set_option autoImplicit false

namespace tests.NebulaV2StatementAuthority

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.StatementAuthority
open Nightstream.Protocol.NebulaV2.WasmStatement

def zeroDigest := tests.NebulaV2WasmResultCodec.zeroDigest

def inputs : Inputs Unit Unit Unit Unit Unit Unit Unit Unit Unit Unit where
  relationManifest := ()
  laneLayout := ()
  setupManifest := ()
  transcriptManifest := ()
  codecManifest := ()
  terminalManifest := ()
  stateSchema := ()
  applicationRelation := ()
  program := ()
  memoryPlan := ()
  initialApplicationState := tests.NebulaV2WasmState.running

def functions :
    DigestFunctions Unit Unit Unit Unit Unit Unit Unit Unit Unit Unit
      Digest.Value where
  relationManifest := fun _ => zeroDigest
  laneLayout := fun _ => zeroDigest
  setupManifest := fun _ => zeroDigest
  transcriptManifest := fun _ => zeroDigest
  codecManifest := fun _ => zeroDigest
  terminalManifest := fun _ => zeroDigest
  stateSchema := fun _ => zeroDigest
  applicationRelation := fun _ => zeroDigest
  program := fun _ => zeroDigest
  memoryPlan := fun _ => zeroDigest
  initialMemoryImage := fun _ _ => 0
  verifierKey := fun _ _ => zeroDigest

def statement := tests.NebulaV2WasmPublicStatementCodec.statement

def opening : Opens functions inputs statement where
  identity := rfl
  program := rfl
  initialApplicationState := rfl
  initialMemoryImage := rfl

theorem aggregate_key_is_not_opaque :
    statement.base.identity.verifierKey.digest =
      functions.verifierKey Profile.v2 (functions.manifestDigests inputs) :=
  opening.aggregate_key_is_recomputed

def oneImage : Fin scannedCells → Nat := fun _ => 1

def differentBase :=
  { statement.base with
    initialImage := oneImage
    initialImageInRange := by
      intro index
      norm_num [oneImage, valueLimit, valueBits] }

def differentStatement : ProductionStatement Unit where
  base := differentBase
  initialApplicationStateValid := statement.initialApplicationStateValid
  resultImage := statement.resultImage
  resultDecoded := statement.resultDecoded

/-- A matching memory-plan digest cannot authorize an independently changed
initial image. The verifier-owned preimage equality is mandatory. -/
theorem changed_initial_image_is_not_authorized :
    ¬ Opens functions inputs differentStatement := by
  intro falseOpening
  have images := falseOpening.initialMemoryImage
  have atZero := congrFun images (⟨0, by decide⟩ : Fin scannedCells)
  change 1 = 0 at atZero
  omega

end tests.NebulaV2StatementAuthority
