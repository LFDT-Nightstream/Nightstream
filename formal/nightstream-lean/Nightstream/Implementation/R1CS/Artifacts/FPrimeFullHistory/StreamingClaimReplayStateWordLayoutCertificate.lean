import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: structural state-word layout certificates for both Rust-emitted
streaming claim-replay arms.

Assurance tier: Rust-to-Lean artifact layout certificate.

Owns the exact 688-column layout, its duplicate-free structure, and its
column bounds. The proof uses two reusable rotated-range theorems.

Does not own state-word semantics, leaf rows, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateWordLayoutCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

def rotatedRange (start before after : Nat) : List Nat :=
  List.range' start before ++
    (start + before + after) :: List.range' (start + before) after

theorem rotatedRange_length (start before after : Nat) :
    (rotatedRange start before after).length = before + 1 + after := by
  simp [rotatedRange]
  omega

theorem rotatedRange_nodup (start before after : Nat) :
    (rotatedRange start before after).Nodup := by
  unfold rotatedRange
  rw [List.nodup_append']
  constructor
  · exact List.nodup_range'
  constructor
  · rw [List.nodup_cons]
    constructor
    · simp only [List.mem_range'_1]
      omega
    · exact List.nodup_range'
  · rw [List.disjoint_iff_ne]
    intro left leftMember right rightMember equal
    rw [List.mem_range'_1] at leftMember
    simp only [List.mem_cons] at rightMember
    rcases rightMember with rfl | rightMember
    · omega
    · rw [List.mem_range'_1] at rightMember
      omega

theorem rotatedRange_member_bounds
    {start before after value : Nat}
    (member : value ∈ rotatedRange start before after) :
    start ≤ value ∧ value ≤ start + before + after := by
  unfold rotatedRange at member
  rw [List.mem_append] at member
  rcases member with member | member
  · rw [List.mem_range'_1] at member
    omega
  · simp only [List.mem_cons] at member
    rcases member with rfl | member
    · omega
    · rw [List.mem_range'_1] at member
      omega

def transitionStateWordColumns : List Nat :=
  rotatedRange 1 19 324 ++ rotatedRange 411 19 324

theorem transitionStateWordColumns_length :
    transitionStateWordColumns.length = 688 := by
  simp [transitionStateWordColumns, rotatedRange_length]

theorem transitionStateWordColumns_nodup :
    transitionStateWordColumns.Nodup := by
  unfold transitionStateWordColumns
  apply List.Nodup.append
  · exact rotatedRange_nodup 1 19 324
  · exact rotatedRange_nodup 411 19 324
  · rw [List.disjoint_iff_ne]
    intro left leftMember right rightMember equal
    have leftBounds := rotatedRange_member_bounds leftMember
    have rightBounds := rotatedRange_member_bounds rightMember
    omega

theorem transitionStateWordColumns_bound
    {column : Nat} (member : column ∈ transitionStateWordColumns) :
    column ≤ 754 := by
  unfold transitionStateWordColumns at member
  rw [List.mem_append] at member
  rcases member with member | member
  · have bounds := rotatedRange_member_bounds member
    omega
  · have bounds := rotatedRange_member_bounds member
    omega

theorem fullArm_stateWordColumns_exact :
    fullArm.stateWordColumns = transitionStateWordColumns := by
  rfl

theorem finalArm_stateWordColumns_exact :
    finalArm.stateWordColumns = transitionStateWordColumns := by
  rfl

theorem arms_stateWordLayout_exact :
    fullArm.stateWordColumns.length = 688 ∧
      fullArm.stateWordColumns.take 19 = List.range' 1 19 ∧
      fullArm.stateWordColumns[19]? = some 344 ∧
      (fullArm.stateWordColumns.drop 20).take 324 = List.range' 20 324 ∧
      (fullArm.stateWordColumns.drop 344).take 19 = List.range' 411 19 ∧
      fullArm.stateWordColumns[363]? = some 754 ∧
      fullArm.stateWordColumns.drop 364 = List.range' 430 324 ∧
      finalArm.stateWordColumns = fullArm.stateWordColumns := by
  rw [fullArm_stateWordColumns_exact, finalArm_stateWordColumns_exact]
  norm_num [transitionStateWordColumns, rotatedRange, List.drop_append,
    List.take_append]

theorem fullArm_stateWordLayout_valid : fullArm.StateWordLayoutValid := by
  unfold RawArm.StateWordLayoutValid
  rw [fullArm_stateWordColumns_exact]
  exact ⟨transitionStateWordColumns_length,
    transitionStateWordColumns_nodup, by
      intro column member
      change column < 340107
      have bound := transitionStateWordColumns_bound member
      omega⟩

theorem finalArm_stateWordLayout_valid : finalArm.StateWordLayoutValid := by
  unfold RawArm.StateWordLayoutValid
  rw [finalArm_stateWordColumns_exact]
  exact ⟨transitionStateWordColumns_length,
    transitionStateWordColumns_nodup, by
      intro column member
      change column < 264104
      have bound := transitionStateWordColumns_bound member
      omega⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateWordLayoutCertificate
