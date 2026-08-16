import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicCertificateSupport

/-!
Contract: bounded glue-row geometry certificate for the Rust-emitted even
PiRLC public-family arm.

Owns 121 compact rows as one 64-row leaf and one 57-row leaf. It owns no row
semantics or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenGlueRowCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport

def evenGlueTail0 := evenArm.glueRows
def evenGlueChunk0 := evenGlueTail0.take 64
def evenGlueTail1 := evenGlueTail0.drop 64

theorem evenGlueChunk0_checked :
    evenGlueChunk0.length = 64 ∧
      glueRowsGeometryCheck 275006 1232857 1233086 evenGlueChunk0 = true := by
  constructor <;> rfl

theorem evenGlueChunk0_valid :
    ∀ indexed ∈ evenGlueChunk0,
      275006 ≤ indexed.index ∧ indexed.index < 1232857 ∧
        rowColumnsBelow 1233086 indexed.row :=
  glueRowsGeometryCheck_sound evenGlueChunk0_checked.2

theorem evenGlueTail1_checked :
    evenGlueTail1.length = 57 ∧
      glueRowsGeometryCheck 275006 1232857 1233086 evenGlueTail1 = true := by
  constructor <;> rfl

theorem evenGlueTail1_valid :
    ∀ indexed ∈ evenGlueTail1,
      275006 ≤ indexed.index ∧ indexed.index < 1232857 ∧
        rowColumnsBelow 1233086 indexed.row :=
  glueRowsGeometryCheck_sound evenGlueTail1_checked.2

theorem evenArm_glueRows_length : evenArm.glueRows.length = 121 :=
  length_of_take_drop evenGlueChunk0_checked.1 evenGlueTail1_checked.1

theorem evenArm_glueRows_valid : evenArm.GlueRowsValid := by
  unfold RawArm.GlueRowsValid
  change ∀ indexed ∈ evenGlueTail0,
    275006 ≤ indexed.index ∧ indexed.index < 1232857 ∧
      rowColumnsBelow 1233086 indexed.row
  exact valid_of_take_drop evenGlueChunk0_valid evenGlueTail1_valid

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenGlueRowCertificate
