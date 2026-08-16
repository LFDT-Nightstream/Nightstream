import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: structural Poseidon2 call-geometry certificate for the Rust-emitted
full streaming claim-replay arm.

Assurance tier: Rust-to-Lean leaf-geometry certificate.

Owns all 378 compact call records as six bounded leaf certificates, including
row extent, eight-input shape, input bounds, and allocated-column bounds.

Does not own Poseidon2 semantic soundness, rows outside these calls, or row
ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullPoseidon2CallCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

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

def fullTail0 := fullArm.poseidon2Calls
def fullChunk0 := fullTail0.take 64
def fullTail1 := fullTail0.drop 64
def fullChunk1 := fullTail1.take 64
def fullTail2 := fullTail1.drop 64
def fullChunk2 := fullTail2.take 64
def fullTail3 := fullTail2.drop 64
def fullChunk3 := fullTail3.take 64
def fullTail4 := fullTail3.drop 64
def fullChunk4 := fullTail4.take 64
def fullTail5 := fullTail4.drop 64

theorem fullChunk0_length : fullChunk0.length = 64 := by rfl
theorem fullChunk1_length : fullChunk1.length = 64 := by rfl
theorem fullChunk2_length : fullChunk2.length = 64 := by rfl
theorem fullChunk3_length : fullChunk3.length = 64 := by rfl
theorem fullChunk4_length : fullChunk4.length = 64 := by rfl
theorem fullTail5_length : fullTail5.length = 58 := by rfl

theorem fullArm_poseidon2Calls_length :
    fullArm.poseidon2Calls.length = 378 := by
  have tail4 : fullTail4.length = 122 :=
    length_of_take_drop (items := fullTail4) (count := 64)
      fullChunk4_length fullTail5_length
  have tail3 : fullTail3.length = 186 :=
    length_of_take_drop (items := fullTail3) (count := 64)
      fullChunk3_length tail4
  have tail2 : fullTail2.length = 250 :=
    length_of_take_drop (items := fullTail2) (count := 64)
      fullChunk2_length tail3
  have tail1 : fullTail1.length = 314 :=
    length_of_take_drop (items := fullTail1) (count := 64)
      fullChunk1_length tail2
  exact length_of_take_drop (items := fullTail0) (count := 64)
    fullChunk0_length tail1

theorem fullChunk0_valid :
    ∀ call ∈ fullChunk0, PoseidonCallValid 307491 call := by
  norm_num [fullChunk0, fullTail0, PoseidonCallValid, fullArm]

theorem fullChunk1_valid :
    ∀ call ∈ fullChunk1, PoseidonCallValid 307491 call := by
  norm_num [fullChunk1, fullTail1, fullTail0, PoseidonCallValid, fullArm]

theorem fullChunk2_valid :
    ∀ call ∈ fullChunk2, PoseidonCallValid 307491 call := by
  norm_num [fullChunk2, fullTail2, fullTail1, fullTail0, PoseidonCallValid, fullArm]

theorem fullChunk3_valid :
    ∀ call ∈ fullChunk3, PoseidonCallValid 307491 call := by
  norm_num [fullChunk3, fullTail3, fullTail2, fullTail1, fullTail0,
    PoseidonCallValid, fullArm]

theorem fullChunk4_valid :
    ∀ call ∈ fullChunk4, PoseidonCallValid 307491 call := by
  norm_num [fullChunk4, fullTail4, fullTail3, fullTail2, fullTail1, fullTail0,
    PoseidonCallValid, fullArm]

theorem fullTail5_valid :
    ∀ call ∈ fullTail5, PoseidonCallValid 307491 call := by
  norm_num [fullTail5, fullTail4, fullTail3, fullTail2, fullTail1, fullTail0,
    PoseidonCallValid, fullArm]

theorem fullArm_poseidon2Calls_valid : fullArm.Poseidon2CallsValid := by
  unfold RawArm.Poseidon2CallsValid
  change ∀ call ∈ fullTail0, PoseidonCallValid 307491 call
  exact valid_of_take_drop fullChunk0_valid
    (valid_of_take_drop fullChunk1_valid
      (valid_of_take_drop fullChunk2_valid
        (valid_of_take_drop fullChunk3_valid
          (valid_of_take_drop fullChunk4_valid fullTail5_valid))))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullPoseidon2CallCertificate
