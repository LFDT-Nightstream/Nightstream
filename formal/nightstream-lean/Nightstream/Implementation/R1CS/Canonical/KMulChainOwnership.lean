import Nightstream.Implementation.R1CS.Canonical.KMulChainHonest

/-!
Contract: positional receipts and whole-program conservation for a sequential
`K` multiplication chain.

Owns: one unique `(factor offset, multiplication slot)` receipt per row, the
proof that the emitted list is exactly those receipts' image, and the complete
classification of every referenced column.

Does not own: the semantic meaning of the factors. Consumers such as PiCCS
bind those source expressions separately.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KMulChainOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KMulChain

/-! ## Positional receipts -/

abbrev Receipt := Nat × KMulOwnership.RowOwner

def receipts : Nat → List Receipt
  | 0 => []
  | count + 1 =>
      KMulOwnership.allOwners.map (fun owner => (0, owner)) ++
        (receipts count).map (fun receipt => (receipt.1 + 1, receipt.2))

theorem receipts_length : ∀ count, (receipts count).length = 3 * count
  | 0 => rfl
  | count + 1 => by
      simp only [receipts, List.length_append, List.length_map,
        KMulOwnership.allOwners_length, receipts_length count]
      omega

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

def zeroCarried : Carried := ⟨[], []⟩

/-- The factor at an offset; the default is unreachable for a valid receipt. -/
def factorAt : List Carried → Nat → Carried
  | [], _ => zeroCarried
  | factor :: _, 0 => factor
  | _ :: rest, offset + 1 => factorAt rest offset

/-- The left operand at an offset. After offset zero it is exactly the prior
frame's carried output. -/
def leftAt (initial : Carried) (frames : Nat → Frame) (step : Nat) : Nat → Carried
  | 0 => initial
  | offset + 1 => frameOutput (frames (step + offset))

theorem leftAt_shift
    (initial : Carried) (frames : Nat → Frame) (step offset : Nat) :
    leftAt (frameOutput (frames step)) frames (step + 1) offset =
      leftAt initial frames step (offset + 1) := by
  cases offset with
  | zero => rfl
  | succ offset =>
      simp only [leftAt]
      congr 2
      omega

theorem factorAt_shift (factor : Carried) (rest : List Carried) (offset : Nat) :
    factorAt rest offset = factorAt (factor :: rest) (offset + 1) := by
  rfl

def receiptRow (initial : Carried) (frames : Nat → Frame)
    (factors : List Carried) (step : Nat) (receipt : Receipt) : Row :=
  KMulOwnership.ownedRow
    (leftAt initial frames step receipt.1)
    (factorAt factors receipt.1)
    (frames (step + receipt.1))
    receipt.2

/-- The emitted chain is exactly the receipt list's image. -/
theorem rows_eq_map_receipts (frames : Nat → Frame) :
    ∀ (initial : Carried) (factors : List Carried) (step : Nat),
      rows initial frames factors step =
        (receipts factors.length).map
          (receiptRow initial frames factors step)
  | _, [], _ => rfl
  | initial, factor :: rest, step => by
      have tail :=
        rows_eq_map_receipts frames (frameOutput (frames step)) rest (step + 1)
      show KMul.rows initial factor (frames step) ++
          rows (frameOutput (frames step)) frames rest (step + 1) = _
      simp only [List.length_cons]
      rw [tail, receipts, List.map_append, List.map_map,
        KMulOwnership.rows_eq_map_owners]
      congr 1
      rw [List.map_map]
      refine List.map_congr_left (fun receipt _ => ?_)
      rcases receipt with ⟨offset, owner⟩
      unfold receiptRow
      simp only [Function.comp_apply]
      rw [leftAt_shift, factorAt_shift]
      have frameIndex : step + 1 + offset = step + (offset + 1) := by omega
      rw [frameIndex]

/-! ## Conservation -/

def InitialColumn (initial : Carried) (column : Nat) : Prop :=
  Mentions initial.low column ∨ Mentions initial.high column

def FactorColumn (factors : List Carried) (column : Nat) : Prop :=
  ∃ factor ∈ factors,
    Mentions factor.low column ∨ Mentions factor.high column

def FrameOfRun (frames : Nat → Frame) (factors : List Carried)
    (step column : Nat) : Prop :=
  ∃ later, step ≤ later ∧ later < step + factors.length ∧
    KMulOwnership.FrameColumn (frames later) column

theorem frameOutput_mentions
    (frame : Frame) (column : Nat)
    (mentioned :
      Mentions (frameOutput frame).low column ∨
        Mentions (frameOutput frame).high column) :
    KMulOwnership.FrameColumn frame column := by
  rcases mentioned with low | high
  · simp only [frameOutput, outLow, Mentions, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at low
    rcases low with rfl | rfl
    · exact Or.inl rfl
    · exact Or.inr (Or.inl rfl)
  · simp only [frameOutput, outHigh, Mentions, List.map_cons, List.map_nil,
      List.mem_cons, List.not_mem_nil, or_false] at high
    rcases high with rfl | rfl | rfl
    · exact Or.inr (Or.inr rfl)
    · exact Or.inl rfl
    · exact Or.inr (Or.inl rfl)

/-- Every row reaches only the initial operand, one listed factor, or one frame
owned by this chain. -/
theorem rows_conservation (frames : Nat → Frame) :
    ∀ (initial : Carried) (factors : List Carried) (step : Nat)
      (row : Row),
      row ∈ KMulChain.rows initial frames factors step →
      ∀ column,
        (Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) →
        InitialColumn initial column ∨ FactorColumn factors column ∨
          FrameOfRun frames factors step column
  | _, [], _, _, member, _, _ => by simp [KMulChain.rows] at member
  | initial, factor :: rest, step, row, member, column, mentioned => by
      simp only [KMulChain.rows, List.mem_append] at member
      rcases member with inHead | inTail
      · rcases KMulOwnership.rows_conservation initial factor (frames step)
          row inHead column mentioned with operand | inFrame
        · rcases operand with initialLow | initialHigh | factorLow | factorHigh
          · exact Or.inl (Or.inl initialLow)
          · exact Or.inl (Or.inr initialHigh)
          · exact Or.inr (Or.inl
              ⟨factor, by simp, Or.inl factorLow⟩)
          · exact Or.inr (Or.inl
              ⟨factor, by simp, Or.inr factorHigh⟩)
        · exact Or.inr (Or.inr
            ⟨step, Nat.le_refl _, by simp, inFrame⟩)
      · rcases rows_conservation frames (frameOutput (frames step)) rest
          (step + 1) row inTail column mentioned with
        inOutput | inFactor | inFrame
        · exact Or.inr (Or.inr
            ⟨step, Nat.le_refl _, by simp,
              frameOutput_mentions (frames step) column inOutput⟩)
        · rcases inFactor with ⟨other, memberOther, inOther⟩
          exact Or.inr (Or.inl
            ⟨other, List.mem_cons_of_mem factor memberOther, inOther⟩)
        · rcases inFrame with ⟨later, lower, upper, frameColumn⟩
          exact Or.inr (Or.inr
            ⟨later, by omega,
              by simp only [List.length_cons] at upper ⊢; omega,
              frameColumn⟩)

/-- Under the canonical allocator, every frame column is inside the exact
declared allocation. -/
theorem frameOfRun_mem_columns
    (base factorCount step column : Nat)
    (stepZero : step = 0)
    (frame :
      FrameOfRun (KFrames.frameAt base)
        (List.replicate factorCount zeroCarried) step column) :
    column ∈ KMulChain.columns base factorCount := by
  rcases frame with ⟨later, lower, upper, inFrame⟩
  subst step
  simp only [List.length_replicate] at upper
  rcases inFrame with rfl | rfl | rfl
  · exact KFrames.frameAt_columns_mem base factorCount later 0
      (by omega) (by decide)
  · exact KFrames.frameAt_columns_mem base factorCount later 1
      (by omega) (by decide)
  · exact KFrames.frameAt_columns_mem base factorCount later 2
      (by omega) (by decide)

end Nightstream.Implementation.R1CS.Canonical.KMulChainOwnership
