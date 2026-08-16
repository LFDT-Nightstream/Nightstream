import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: structural Poseidon2 call-geometry certificate for the Rust-emitted
final streaming claim-replay arm.

Assurance tier: Rust-to-Lean leaf-geometry certificate.

Owns all 367 compact call records as six bounded leaf certificates, including
row extent, eight-input shape, input bounds, and allocated-column bounds.

Does not own Poseidon2 semantic soundness, rows outside these calls, or row
ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalPoseidon2CallCertificate

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

def finalTail0 := finalArm.poseidon2Calls
def finalChunk0 := finalTail0.take 64
def finalTail1 := finalTail0.drop 64
def finalChunk1 := finalTail1.take 64
def finalTail2 := finalTail1.drop 64
def finalChunk2 := finalTail2.take 64
def finalTail3 := finalTail2.drop 64
def finalChunk3 := finalTail3.take 64
def finalTail4 := finalTail3.drop 64
def finalChunk4 := finalTail4.take 64
def finalTail5 := finalTail4.drop 64

theorem finalChunk0_length : finalChunk0.length = 64 := by rfl
theorem finalChunk1_length : finalChunk1.length = 64 := by rfl
theorem finalChunk2_length : finalChunk2.length = 64 := by rfl
theorem finalChunk3_length : finalChunk3.length = 64 := by rfl
theorem finalChunk4_length : finalChunk4.length = 64 := by rfl
theorem finalTail5_length : finalTail5.length = 47 := by rfl

theorem finalArm_poseidon2Calls_length :
    finalArm.poseidon2Calls.length = 367 := by
  have tail4 : finalTail4.length = 111 :=
    length_of_take_drop (items := finalTail4) (count := 64)
      finalChunk4_length finalTail5_length
  have tail3 : finalTail3.length = 175 :=
    length_of_take_drop (items := finalTail3) (count := 64)
      finalChunk3_length tail4
  have tail2 : finalTail2.length = 239 :=
    length_of_take_drop (items := finalTail2) (count := 64)
      finalChunk2_length tail3
  have tail1 : finalTail1.length = 303 :=
    length_of_take_drop (items := finalTail1) (count := 64)
      finalChunk1_length tail2
  exact length_of_take_drop (items := finalTail0) (count := 64)
    finalChunk0_length tail1

theorem finalChunk0_valid :
    ∀ call ∈ finalChunk0, PoseidonCallValid 342464 call := by
  norm_num [finalChunk0, finalTail0, PoseidonCallValid, finalArm]

theorem finalChunk1_valid :
    ∀ call ∈ finalChunk1, PoseidonCallValid 342464 call := by
  norm_num [finalChunk1, finalTail1, finalTail0, PoseidonCallValid, finalArm]

theorem finalChunk2_valid :
    ∀ call ∈ finalChunk2, PoseidonCallValid 342464 call := by
  norm_num [finalChunk2, finalTail2, finalTail1, finalTail0, PoseidonCallValid, finalArm]

theorem finalChunk3_valid :
    ∀ call ∈ finalChunk3, PoseidonCallValid 342464 call := by
  norm_num [finalChunk3, finalTail3, finalTail2, finalTail1, finalTail0,
    PoseidonCallValid, finalArm]

theorem finalChunk4_valid :
    ∀ call ∈ finalChunk4, PoseidonCallValid 342464 call := by
  norm_num [finalChunk4, finalTail4, finalTail3, finalTail2, finalTail1, finalTail0,
    PoseidonCallValid, finalArm]

theorem finalTail5_valid :
    ∀ call ∈ finalTail5, PoseidonCallValid 342464 call := by
  norm_num [finalTail5, finalTail4, finalTail3, finalTail2, finalTail1, finalTail0,
    PoseidonCallValid, finalArm]

theorem finalArm_poseidon2Calls_valid : finalArm.Poseidon2CallsValid := by
  unfold RawArm.Poseidon2CallsValid
  change ∀ call ∈ finalTail0, PoseidonCallValid 342464 call
  exact valid_of_take_drop finalChunk0_valid
    (valid_of_take_drop finalChunk1_valid
      (valid_of_take_drop finalChunk2_valid
        (valid_of_take_drop finalChunk3_valid
          (valid_of_take_drop finalChunk4_valid finalTail5_valid))))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalPoseidon2CallCertificate
