import Nightstream.Implementation.R1CS.Canonical.KHornerHonest

/-!
Contract: exact row ownership for a Horner evaluation.

Owns: the receipt list for the whole evaluation and the proof that the emitted
program is its image.

Does not own: conservation. That is already
`KHornerSupport.hornerRows_mentions` — every column of every emitted row is a
`beta` column, a coefficient column, or a frame at or after this step. Restating
it under a second name would be a rename, not a result, so it is cited rather
than reproved.

## The receipt has to carry an offset

For a single multiplication the receipt is just a slot. Here it must also say
*which step*, because the row a step emits depends on the value carried by the
steps after it — and that value depends on how much of the coefficient list
remains.

`receiptRow` therefore reconstructs the suffix with `List.drop`. The recursion
in `hornerRows` decrements the list and increments the step in lockstep, so at
offset `j` the remaining coefficients are exactly `coefficients.drop j`. That
lockstep is what makes the map equality provable rather than merely plausible.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KHornerOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-- One receipt per emitted row: which multiplication, and which of its three
rows. -/
abbrev Receipt := Nat × KMulOwnership.RowOwner

/-- Receipts for an evaluation with `count` multiplications.  Defined by the
same recursion `hornerRows` uses, so the map equality below aligns structurally
instead of needing range arithmetic. -/
def receipts : Nat → List Receipt
  | 0 => []
  | count + 1 =>
      KMulOwnership.allOwners.map (fun owner => (0, owner))
        ++ (receipts count).map (fun receipt => (receipt.1 + 1, receipt.2))

theorem receipts_length : ∀ count, (receipts count).length = 3 * count
  | 0 => rfl
  | count + 1 => by
      simp only [receipts, List.length_append, List.length_map,
        KMulOwnership.allOwners_length, receipts_length count]
      omega

/-- Every positional Horner receipt is unique.  The head block owns offset
zero; recursive receipts are shifted to strictly positive offsets. -/
theorem receipts_nodup : ∀ count, (receipts count).Nodup
  | 0 => by simp [receipts]
  | count + 1 => by
      rw [receipts, List.nodup_append]
      refine ⟨
        LinCombNormal.nodup_map KMulOwnership.allOwners
          (fun owner => (0, owner))
          (fun left right equal => by
            simp only [Prod.mk.injEq] at equal
            exact equal.2)
          KMulOwnership.allOwners_nodup,
        LinCombNormal.nodup_map (receipts count)
          (fun receipt => (receipt.1 + 1, receipt.2))
          (fun left right equal => by
            rcases left with ⟨leftOffset, leftOwner⟩
            rcases right with ⟨rightOffset, rightOwner⟩
            simp only [Prod.mk.injEq] at equal ⊢
            exact ⟨Nat.add_right_cancel equal.1, equal.2⟩)
          (receipts_nodup count),
        ?_⟩
      intro head headMember tail tailMember equal
      rcases List.mem_map.1 headMember with ⟨owner, _, rfl⟩
      rcases List.mem_map.1 tailMember with ⟨receipt, _, rfl⟩
      simp only [Prod.mk.injEq] at equal
      omega

/-- The row a receipt emits.  The suffix is reconstructed by `drop`, which is
what the offset is for. -/
def receiptRow (beta : Carried) (frames : Nat → Frame)
    (coefficients : List Carried) (step : Nat) (receipt : Receipt) : Row :=
  KMulOwnership.ownedRow beta
    (hornerCarried beta frames (coefficients.drop (receipt.1 + 1))
      (step + receipt.1 + 1))
    (frames (step + receipt.1)) receipt.2

/-- **The emitted evaluation is its receipt list's image.**  With
`receipts_length`, position `i` is emitted by receipt `i` and by no other. -/
theorem hornerRows_eq_map_receipts
    (beta : Carried) (frames : Nat → Frame) :
    ∀ (coefficients : List Carried) (step : Nat),
      hornerRows beta frames coefficients step
        = (receipts (coefficients.length - 1)).map
            (receiptRow beta frames coefficients step)
  | [], _ => rfl
  | [_], _ => rfl
  | c :: next :: rest, step => by
      have tail := hornerRows_eq_map_receipts beta frames (next :: rest) (step + 1)
      show KMul.rows beta _ _ ++ hornerRows beta frames (next :: rest) (step + 1)
        = _
      rw [tail, KMulOwnership.rows_eq_map_owners]
      simp only [List.length_cons, Nat.add_sub_cancel, receipts,
        List.map_append, List.map_map]
      congr 1
      · refine List.map_congr_left (fun receipt _ => ?_)
        show KMulOwnership.ownedRow beta
            (hornerCarried beta frames ((next :: rest).drop (receipt.1 + 1))
              (step + 1 + receipt.1 + 1)) (frames (step + 1 + receipt.1))
            receipt.2
          = KMulOwnership.ownedRow beta
            (hornerCarried beta frames ((c :: next :: rest).drop (receipt.1 + 1 + 1))
              (step + (receipt.1 + 1) + 1)) (frames (step + (receipt.1 + 1)))
            receipt.2
        congr 2 <;> omega

end Nightstream.Implementation.R1CS.Canonical.KHornerOwnership
