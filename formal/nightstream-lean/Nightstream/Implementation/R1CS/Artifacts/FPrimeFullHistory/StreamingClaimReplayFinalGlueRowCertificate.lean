import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayLeafCertificateSupport

/-!
Contract: structural glue-row geometry certificate for the Rust-emitted final
streaming claim-replay arm.

Assurance tier: Rust-to-Lean leaf-geometry certificate.

Owns all 323 compact glue rows as six bounded leaf certificates, including row
bounds and referenced-column bounds.

Does not own row semantics, rows outside this list, or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalGlueRowCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayLeafCertificateSupport

private theorem valid_of_take_drop
    {α : Type} {property : α → Prop} {items : List α} {count : Nat}
    (head : ∀ item ∈ items.take count, property item)
    (tail : ∀ item ∈ items.drop count, property item) :
    ∀ item ∈ items, property item := by
  intro item member
  rw [← List.take_append_drop count items] at member
  rcases List.mem_append.mp member with member | member
  · exact head item member
  · exact tail item member

private theorem length_of_take_drop
    {α : Type} {items : List α} {count headLength tailLength : Nat}
    (head : (items.take count).length = headLength)
    (tail : (items.drop count).length = tailLength) :
    items.length = headLength + tailLength := by
  have split := congrArg List.length (List.take_append_drop count items)
  simpa only [List.length_append, head, tail] using split.symm

def finalGlueTail0 := finalArm.glueRows
def finalGlueChunk0 := finalGlueTail0.take 64
def finalGlueTail1 := finalGlueTail0.drop 64
def finalGlueChunk1 := finalGlueTail1.take 64
def finalGlueTail2 := finalGlueTail1.drop 64
def finalGlueChunk2 := finalGlueTail2.take 64
def finalGlueTail3 := finalGlueTail2.drop 64
def finalGlueChunk3 := finalGlueTail3.take 64
def finalGlueTail4 := finalGlueTail3.drop 64
def finalGlueChunk4 := finalGlueTail4.take 64
def finalGlueTail5 := finalGlueTail4.drop 64

theorem finalGlueChunk0_checked :
    finalGlueChunk0.length = 64 ∧
      glueRowsGeometryCheck 343256 342464 finalGlueChunk0 = true := by
  constructor <;> rfl

theorem finalGlueChunk0_valid :
    ∀ indexed ∈ finalGlueChunk0,
      indexed.index < 343256 ∧ rowColumnsBelow 342464 indexed.row :=
  glueRowsGeometryCheck_sound finalGlueChunk0_checked.2

theorem finalGlueChunk1_checked :
    finalGlueChunk1.length = 64 ∧
      glueRowsGeometryCheck 343256 342464 finalGlueChunk1 = true := by
  constructor <;> rfl

theorem finalGlueChunk1_valid :
    ∀ indexed ∈ finalGlueChunk1,
      indexed.index < 343256 ∧ rowColumnsBelow 342464 indexed.row :=
  glueRowsGeometryCheck_sound finalGlueChunk1_checked.2

theorem finalGlueChunk2_checked :
    finalGlueChunk2.length = 64 ∧
      glueRowsGeometryCheck 343256 342464 finalGlueChunk2 = true := by
  constructor <;> rfl

theorem finalGlueChunk2_valid :
    ∀ indexed ∈ finalGlueChunk2,
      indexed.index < 343256 ∧ rowColumnsBelow 342464 indexed.row :=
  glueRowsGeometryCheck_sound finalGlueChunk2_checked.2

theorem finalGlueChunk3_checked :
    finalGlueChunk3.length = 64 ∧
      glueRowsGeometryCheck 343256 342464 finalGlueChunk3 = true := by
  constructor <;> rfl

theorem finalGlueChunk3_valid :
    ∀ indexed ∈ finalGlueChunk3,
      indexed.index < 343256 ∧ rowColumnsBelow 342464 indexed.row :=
  glueRowsGeometryCheck_sound finalGlueChunk3_checked.2

theorem finalGlueChunk4_checked :
    finalGlueChunk4.length = 64 ∧
      glueRowsGeometryCheck 343256 342464 finalGlueChunk4 = true := by
  constructor <;> rfl

theorem finalGlueChunk4_valid :
    ∀ indexed ∈ finalGlueChunk4,
      indexed.index < 343256 ∧ rowColumnsBelow 342464 indexed.row :=
  glueRowsGeometryCheck_sound finalGlueChunk4_checked.2

theorem finalGlueTail5_checked :
    finalGlueTail5.length = 3 ∧
      glueRowsGeometryCheck 343256 342464 finalGlueTail5 = true := by
  constructor <;> rfl

theorem finalGlueTail5_valid :
    ∀ indexed ∈ finalGlueTail5,
      indexed.index < 343256 ∧ rowColumnsBelow 342464 indexed.row :=
  glueRowsGeometryCheck_sound finalGlueTail5_checked.2

theorem finalArm_glueRows_length : finalArm.glueRows.length = 323 := by
  have tail4 : finalGlueTail4.length = 67 :=
    length_of_take_drop (items := finalGlueTail4) (count := 64)
      finalGlueChunk4_checked.1 finalGlueTail5_checked.1
  have tail3 : finalGlueTail3.length = 131 :=
    length_of_take_drop (items := finalGlueTail3) (count := 64)
      finalGlueChunk3_checked.1 tail4
  have tail2 : finalGlueTail2.length = 195 :=
    length_of_take_drop (items := finalGlueTail2) (count := 64)
      finalGlueChunk2_checked.1 tail3
  have tail1 : finalGlueTail1.length = 259 :=
    length_of_take_drop (items := finalGlueTail1) (count := 64)
      finalGlueChunk1_checked.1 tail2
  exact length_of_take_drop (items := finalGlueTail0) (count := 64)
    finalGlueChunk0_checked.1 tail1

theorem finalArm_glueRows_valid : finalArm.GlueRowsValid := by
  unfold RawArm.GlueRowsValid
  change ∀ indexed ∈ finalGlueTail0,
    indexed.index < 343256 ∧ rowColumnsBelow 342464 indexed.row
  exact valid_of_take_drop finalGlueChunk0_valid
    (valid_of_take_drop finalGlueChunk1_valid
      (valid_of_take_drop finalGlueChunk2_valid
        (valid_of_take_drop finalGlueChunk3_valid
          (valid_of_take_drop finalGlueChunk4_valid finalGlueTail5_valid))))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalGlueRowCertificate
