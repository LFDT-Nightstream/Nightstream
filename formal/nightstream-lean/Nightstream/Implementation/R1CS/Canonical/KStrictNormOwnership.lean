import Nightstream.Implementation.R1CS.Canonical.KStrictNormHonest
import Nightstream.Implementation.R1CS.Canonical.KMulChainOwnership
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Positional receipts, conservation, and exact cost for strict norm.

The program is the two-factor multiplication chain from
`KStrictNormHonest.rows_eq_chain`.  It owns exactly two canonical frames; the
checked value and constant wire are shared reads.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KStrictNormOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KStrictNorm
open Nightstream.Implementation.R1CS.Canonical.KStrictNormHonest

abbrev RowOwner := KMulChainOwnership.Receipt

def allOwners : List RowOwner :=
  KMulChainOwnership.receipts 2

def ownedRow (input : Input) (owner : RowOwner) : Row :=
  KMulChainOwnership.receiptRow
    (initial input) (KFrames.frameAt input.frameBase)
    (factors input) 0 owner

theorem allOwners_length : allOwners.length = 6 :=
  KMulChainOwnership.receipts_length 2

theorem allOwners_nodup : allOwners.Nodup :=
  KMulChainOwnership.receipts_nodup 2

theorem rows_eq_map_owners (input : Input) :
    rows input = allOwners.map (ownedRow input) := by
  rw [rows_eq_chain]
  exact KMulChainOwnership.rows_eq_map_receipts
    (KFrames.frameAt input.frameBase) (initial input) (factors input) 0

def SourceColumn (input : Input) (column : Nat) : Prop :=
  Mentions input.value.low column ∨ Mentions input.value.high column

def Allocated (input : Input) (column : Nat) : Prop :=
  column ∈ columns input

private theorem one_source_or_wire (column : Nat) :
    Mentions KLinear.oneCarried.low column ∨
      Mentions KLinear.oneCarried.high column →
    column = 0 := by
  intro mentioned
  rcases mentioned with low | high
  · simpa [KLinear.oneCarried, Mentions] using low
  · simp [KLinear.oneCarried, Mentions] at high

private theorem scale_mentions
    (combination : LinComb) (column : Nat)
    (mentioned :
      Mentions
        (Nightstream.Implementation.R1CS.LinearSubstitution.scaleTerms
          (goldilocksP - 1) combination) column) :
    Mentions combination column := by
  simpa [Nightstream.Implementation.R1CS.LinearSubstitution.scaleTerms,
    Mentions] using mentioned

private theorem linear_source_or_wire
    (input : Input) (value : Carried) (column : Nat)
    (member : value = initial input ∨ value ∈ factors input)
    (mentioned : Mentions value.low column ∨ Mentions value.high column) :
    column = 0 ∨ SourceColumn input column := by
  rcases member with rfl | inFactors
  · unfold initial KLinear.addCarried at mentioned
    rcases mentioned with low | high
    · rcases List.mem_append.1
          (by simpa only [Mentions, List.map_append] using low) with
        source | wire
      · exact Or.inr (Or.inl source)
      · exact Or.inl (one_source_or_wire column (Or.inl wire))
    · rcases List.mem_append.1
          (by simpa only [Mentions, List.map_append] using high) with
        source | wire
      · exact Or.inr (Or.inr source)
      · exact Or.inl (one_source_or_wire column (Or.inr wire))
  · simp only [factors, List.mem_cons, List.not_mem_nil, or_false] at inFactors
    rcases inFactors with rfl | rfl
    · exact Or.inr mentioned
    · unfold KLinear.subCarried KLinear.addCarried KLinear.scaleCarried
        at mentioned
      rcases mentioned with low | high
      · rcases List.mem_append.1
            (by simpa only [Mentions, List.map_append] using low) with
          source | wire
        · exact Or.inr (Or.inl source)
        · exact Or.inl
            (one_source_or_wire column (Or.inl (scale_mentions _ _ wire)))
      · rcases List.mem_append.1
            (by simpa only [Mentions, List.map_append] using high) with
          source | wire
        · exact Or.inr (Or.inr source)
        · exact Or.inl
            (one_source_or_wire column (Or.inr (scale_mentions _ _ wire)))

private theorem frame_allocated
    (input : Input) (column : Nat)
    (frame :
      KMulChainOwnership.FrameOfRun
        (KFrames.frameAt input.frameBase) (factors input) 0 column) :
    Allocated input column := by
  rcases frame with ⟨later, _, upper, inFrame⟩
  have laterBound : later < 2 := by
    simpa [factors] using upper
  unfold Allocated columns
  rcases inFrame with rfl | rfl | rfl
  · exact KFrames.frameAt_columns_mem input.frameBase 2 later 0
      laterBound (by decide)
  · exact KFrames.frameAt_columns_mem input.frameBase 2 later 1
      laterBound (by decide)
  · exact KFrames.frameAt_columns_mem input.frameBase 2 later 2
      laterBound (by decide)

/-- Every row dependency is the constant wire, the checked source value, or
one of the six exactly allocated frame columns. -/
theorem rows_conservation
    (input : Input) (row : Row) (member : row ∈ rows input)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0 ∨ SourceColumn input column ∨ Allocated input column := by
  rw [rows_eq_chain] at member
  rcases KMulChainOwnership.rows_conservation
      (KFrames.frameAt input.frameBase)
      (initial input) (factors input) 0 row member column mentioned with
    inInitial | inFactor | inFrame
  · rcases linear_source_or_wire input (initial input) column
      (Or.inl rfl) inInitial with wire | source
    · exact Or.inl wire
    · exact Or.inr (Or.inl source)
  · rcases inFactor with ⟨factor, factorMember, factorMentioned⟩
    rcases linear_source_or_wire input factor column
      (Or.inr factorMember) factorMentioned with wire | source
    · exact Or.inl wire
    · exact Or.inr (Or.inl source)
  · exact Or.inr (Or.inr (frame_allocated input column inFrame))

def cost : Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := 6
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 6

theorem cost_rows (input : Input) :
    (rows input).length = cost.recurringRows :=
  rows_length input

theorem cost_columns (input : Input) :
    (columns input).length = cost.auxiliaryColumns :=
  columns_length input

end Nightstream.Implementation.R1CS.Canonical.KStrictNormOwnership
