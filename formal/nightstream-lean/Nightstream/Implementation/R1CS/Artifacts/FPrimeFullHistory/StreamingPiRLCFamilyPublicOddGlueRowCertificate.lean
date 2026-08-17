import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicCertificateSupport

/-!
Contract: bounded glue-row geometry certificate for the Rust-emitted odd
PiRLC public-family arm.

Owns 121 compact rows as one 64-row leaf and one 57-row leaf. It owns no row
semantics or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddGlueRowCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport

def oddGlueTail0 := oddArm.glueRows
def oddGlueChunk0 := oddGlueTail0.take 64
def oddGlueTail1 := oddGlueTail0.drop 64

theorem oddGlueChunk0_checked :
    oddGlueChunk0.length = 64 ∧
      glueRowsGeometryCheck 311846 1302097 1302326 oddGlueChunk0 = true := by
  constructor <;> rfl

theorem oddGlueChunk0_valid :
    ∀ indexed ∈ oddGlueChunk0,
      311846 ≤ indexed.index ∧ indexed.index < 1302097 ∧
        rowColumnsBelow 1302326 indexed.row :=
  glueRowsGeometryCheck_sound oddGlueChunk0_checked.2

theorem oddGlueTail1_checked :
    oddGlueTail1.length = 57 ∧
      glueRowsGeometryCheck 311846 1302097 1302326 oddGlueTail1 = true := by
  constructor <;> rfl

theorem oddGlueTail1_valid :
    ∀ indexed ∈ oddGlueTail1,
      311846 ≤ indexed.index ∧ indexed.index < 1302097 ∧
        rowColumnsBelow 1302326 indexed.row :=
  glueRowsGeometryCheck_sound oddGlueTail1_checked.2

theorem oddArm_glueRows_length : oddArm.glueRows.length = 121 :=
  length_of_take_drop oddGlueChunk0_checked.1 oddGlueTail1_checked.1

theorem oddArm_glueRows_valid : oddArm.GlueRowsValid := by
  unfold RawArm.GlueRowsValid
  change ∀ indexed ∈ oddGlueTail0,
    311846 ≤ indexed.index ∧ indexed.index < 1302097 ∧
      rowColumnsBelow 1302326 indexed.row
  exact valid_of_take_drop oddGlueChunk0_valid oddGlueTail1_valid

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddGlueRowCertificate
