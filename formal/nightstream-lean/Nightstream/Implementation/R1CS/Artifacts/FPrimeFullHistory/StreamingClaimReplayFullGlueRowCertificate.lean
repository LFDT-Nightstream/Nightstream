import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayLeafCertificateSupport

/-!
Contract: structural glue-row geometry certificate for the Rust-emitted full
streaming claim-replay arm.

Assurance tier: Rust-to-Lean leaf-geometry certificate.

Owns all 486 compact glue rows as eight bounded leaf certificates, including
row bounds and referenced-column bounds.

Does not own row semantics, rows outside this list, or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullGlueRowCertificate

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

def fullGlueTail0 := fullArm.glueRows
def fullGlueChunk0 := fullGlueTail0.take 64
def fullGlueTail1 := fullGlueTail0.drop 64
def fullGlueChunk1 := fullGlueTail1.take 64
def fullGlueTail2 := fullGlueTail1.drop 64
def fullGlueChunk2 := fullGlueTail2.take 64
def fullGlueTail3 := fullGlueTail2.drop 64
def fullGlueChunk3 := fullGlueTail3.take 64
def fullGlueTail4 := fullGlueTail3.drop 64
def fullGlueChunk4 := fullGlueTail4.take 64
def fullGlueTail5 := fullGlueTail4.drop 64
def fullGlueChunk5 := fullGlueTail5.take 64
def fullGlueTail6 := fullGlueTail5.drop 64
def fullGlueChunk6 := fullGlueTail6.take 64
def fullGlueTail7 := fullGlueTail6.drop 64

theorem fullGlueChunk0_checked :
    fullGlueChunk0.length = 64 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueChunk0 = true := by
  constructor <;> rfl

theorem fullGlueChunk0_valid :
    ∀ indexed ∈ fullGlueChunk0,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueChunk0_checked.2

theorem fullGlueChunk1_checked :
    fullGlueChunk1.length = 64 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueChunk1 = true := by
  constructor <;> rfl

theorem fullGlueChunk1_valid :
    ∀ indexed ∈ fullGlueChunk1,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueChunk1_checked.2

theorem fullGlueChunk2_checked :
    fullGlueChunk2.length = 64 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueChunk2 = true := by
  constructor <;> rfl

theorem fullGlueChunk2_valid :
    ∀ indexed ∈ fullGlueChunk2,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueChunk2_checked.2

theorem fullGlueChunk3_checked :
    fullGlueChunk3.length = 64 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueChunk3 = true := by
  constructor <;> rfl

theorem fullGlueChunk3_valid :
    ∀ indexed ∈ fullGlueChunk3,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueChunk3_checked.2

theorem fullGlueChunk4_checked :
    fullGlueChunk4.length = 64 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueChunk4 = true := by
  constructor <;> rfl

theorem fullGlueChunk4_valid :
    ∀ indexed ∈ fullGlueChunk4,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueChunk4_checked.2

theorem fullGlueChunk5_checked :
    fullGlueChunk5.length = 64 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueChunk5 = true := by
  constructor <;> rfl

theorem fullGlueChunk5_valid :
    ∀ indexed ∈ fullGlueChunk5,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueChunk5_checked.2

theorem fullGlueChunk6_checked :
    fullGlueChunk6.length = 64 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueChunk6 = true := by
  constructor <;> rfl

theorem fullGlueChunk6_valid :
    ∀ indexed ∈ fullGlueChunk6,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueChunk6_checked.2

theorem fullGlueTail7_checked :
    fullGlueTail7.length = 38 ∧
      glueRowsGeometryCheck 307762 307491 fullGlueTail7 = true := by
  constructor <;> rfl

theorem fullGlueTail7_valid :
    ∀ indexed ∈ fullGlueTail7,
      indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row :=
  glueRowsGeometryCheck_sound fullGlueTail7_checked.2

theorem fullArm_glueRows_length : fullArm.glueRows.length = 486 := by
  have tail6 : fullGlueTail6.length = 102 :=
    length_of_take_drop (items := fullGlueTail6) (count := 64)
      fullGlueChunk6_checked.1 fullGlueTail7_checked.1
  have tail5 : fullGlueTail5.length = 166 :=
    length_of_take_drop (items := fullGlueTail5) (count := 64)
      fullGlueChunk5_checked.1 tail6
  have tail4 : fullGlueTail4.length = 230 :=
    length_of_take_drop (items := fullGlueTail4) (count := 64)
      fullGlueChunk4_checked.1 tail5
  have tail3 : fullGlueTail3.length = 294 :=
    length_of_take_drop (items := fullGlueTail3) (count := 64)
      fullGlueChunk3_checked.1 tail4
  have tail2 : fullGlueTail2.length = 358 :=
    length_of_take_drop (items := fullGlueTail2) (count := 64)
      fullGlueChunk2_checked.1 tail3
  have tail1 : fullGlueTail1.length = 422 :=
    length_of_take_drop (items := fullGlueTail1) (count := 64)
      fullGlueChunk1_checked.1 tail2
  exact length_of_take_drop (items := fullGlueTail0) (count := 64)
    fullGlueChunk0_checked.1 tail1

theorem fullArm_glueRows_valid : fullArm.GlueRowsValid := by
  unfold RawArm.GlueRowsValid
  change ∀ indexed ∈ fullGlueTail0,
    indexed.index < 307762 ∧ rowColumnsBelow 307491 indexed.row
  exact valid_of_take_drop fullGlueChunk0_valid
    (valid_of_take_drop fullGlueChunk1_valid
      (valid_of_take_drop fullGlueChunk2_valid
        (valid_of_take_drop fullGlueChunk3_valid
          (valid_of_take_drop fullGlueChunk4_valid
            (valid_of_take_drop fullGlueChunk5_valid
              (valid_of_take_drop fullGlueChunk6_valid fullGlueTail7_valid))))))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullGlueRowCertificate
