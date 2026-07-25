import Nightstream.Implementation.R1CS.Ownership.FPrime.FPrimeTerminalLinkArtifact

/-!
Contract: universal soundness and completeness of the exact trailing-batch
delayed-link rows.

Every canonical-residue assignment satisfying the production terminal-link
artifact fixes the fresh affine-one slot and equates all 256 fresh public bits
to the last F' step's `x_out` bits, and fixes all thirteen carrier-completion
padding coordinates to zero. Host-side nonempty and length rejection is tested
by the Rust exporter and remains explicit in the property contract.
-/

set_option maxRecDepth 32768

namespace Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLink

structure Holds (z : Nat → Nat) : Prop where
  affineOne : z freshOneCol = 1
  linked : ∀ bit, bit < 256 →
    z (freshBitCol bit) = z (lastXOutBitCol bit)
  paddingZero : ∀ padding, padding < 13 →
    z (freshPaddingCol padding) = 0

private theorem oneRow_mem : oneRow ∈ rows := by simp [rows]

private theorem linkRow_mem {bit : Nat} (bitLt : bit < 256) :
    linkRow bit ∈ rows := by
  simp only [rows, List.mem_cons, List.mem_append]
  right
  left
  exact List.mem_map.mpr ⟨bit, List.mem_range.mpr bitLt, rfl⟩

private theorem paddingRow_mem {padding : Nat} (paddingLt : padding < 13) :
    paddingRow padding ∈ rows := by
  simp only [rows, List.mem_cons, List.mem_append]
  right
  right
  exact List.mem_map.mpr ⟨padding, List.mem_range.mpr paddingLt, rfl⟩

private theorem equality_of_link_row {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {left right : Nat}
    (holds : RowHolds z
      ⟨[(left, 1), (right, goldilocksP - 1)], [(0, 1)], []⟩) :
    z left = z right := by
  have leftLt := hcanon left
  have rightLt := hcanon right
  simp only [RowHolds, lcEval, List.foldl, hone, goldilocksP] at holds leftLt rightLt
  omega

private theorem zero_of_padding_row {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {column : Nat}
    (holds : RowHolds z
      ⟨[(column, 1)], [(0, 1)], []⟩) :
    z column = 0 := by
  have columnLt := hcanon column
  simp only [RowHolds, lcEval, List.foldl, hone, goldilocksP] at holds columnLt
  omega

private theorem equality_row_of_equal {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {left right : Nat}
    (equal : z left = z right) :
    RowHolds z
      ⟨[(left, 1), (right, goldilocksP - 1)], [(0, 1)], []⟩ := by
  have leftLt := hcanon left
  have rightLt := hcanon right
  simp only [RowHolds, lcEval, List.foldl, hone, goldilocksP]
  omega

private theorem zero_row_of_zero {z : Nat → Nat}
    (hone : z 0 = 1) {column : Nat}
    (zero : z column = 0) :
    RowHolds z
      ⟨[(column, 1)], [(0, 1)], []⟩ := by
  simp [RowHolds, lcEval, hone, zero]

/-- Exact terminal-link rows close the delayed public-input obligation. -/
theorem fPrimeTerminalLink_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z) :
    Holds z := by
  refine {
    affineOne := ?_
    linked := ?_
    paddingZero := ?_
  }
  · have row := hsat oneRow oneRow_mem
    have freshLt := hcanon freshOneCol
    simp only [oneRow, RowHolds, lcEval, List.foldl, hone, goldilocksP] at row freshLt
    omega
  · intro bit bitLt
    exact equality_of_link_row hcanon hone
      (hsat (linkRow bit) (linkRow_mem bitLt))
  · intro padding paddingLt
    exact zero_of_padding_row hcanon hone
      (hsat (paddingRow padding) (paddingRow_mem paddingLt))

/-- Semantic terminal-link validity satisfies every emitted row. -/
theorem fPrimeTerminalLink_complete {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (holds : Holds z) :
    Satisfies rows z := by
  intro row rowMember
  simp only [rows, List.mem_cons, List.mem_append,
    List.mem_map] at rowMember
  rcases rowMember with rowEqual |
      ⟨bit, bitMember, rowEqual⟩ |
      ⟨padding, paddingMember, rowEqual⟩
  · subst row
    exact equality_row_of_equal hcanon hone
      (holds.affineOne.trans hone.symm)
  · subst row
    exact equality_row_of_equal hcanon hone
      (holds.linked bit (List.mem_range.mp bitMember))
  · subst row
    exact zero_row_of_zero hone
      (holds.paddingZero padding (List.mem_range.mp paddingMember))

theorem satisfies_iff_holds {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) :
    Satisfies rows z ↔ Holds z :=
  ⟨fPrimeTerminalLink_sound hcanon hone,
    fPrimeTerminalLink_complete hcanon hone⟩

end Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound
