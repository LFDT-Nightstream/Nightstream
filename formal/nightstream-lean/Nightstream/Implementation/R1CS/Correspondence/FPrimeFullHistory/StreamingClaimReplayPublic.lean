import Mathlib.Data.List.OfFn
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigest

/-!
Contract: exact normalized public carrier for one streaming claim-replay arm.

Owns the projection from Rust source-assignment columns to the ordered
carrier `[1, ten canonical 64-bit words, seven zero padding coordinates]`.
It proves the exact 641-coordinate logical prefix, the exact 648-coordinate
aligned carrier, every source-bit position, and binary values.

Does not own low-norm compiler rows, selectors, private-coordinate
normalization, phase semantics, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest

/-- The 64 source bits of one Rust-selected public word. -/
def publicWordBits
    (assignment : Nat → Nat) (kind : ArmKind) (word : Fin 10) : List Nat :=
  List.ofFn fun bit : Fin 64 =>
    assignment (publicBitSourceColumn kind word bit)

/-- Ten words in Rust's exact digest-then-cursor order. -/
def publicWordBlocks
    (assignment : Nat → Nat) (kind : ArmKind) : List (List Nat) :=
  List.ofFn fun word : Fin 10 => publicWordBits assignment kind word

/-- Constant-one coordinate followed by the exact 640 public source bits. -/
def logicalCarrier
    (assignment : Nat → Nat) (kind : ArmKind) : List Nat :=
  [assignment 0] ++ (publicWordBlocks assignment kind).flatten

/-- The compiler-aligned public carrier. The seven final coordinates are
constrained public padding, not extra protocol state. -/
def carrier (assignment : Nat → Nat) (kind : ArmKind) : List Nat :=
  logicalCarrier assignment kind ++ List.replicate 7 0

@[simp] theorem publicWordBits_length
    (assignment : Nat → Nat) (kind : ArmKind) (word : Fin 10) :
    (publicWordBits assignment kind word).length = 64 := by
  simp [publicWordBits]

@[simp] theorem publicWordBlocks_length
    (assignment : Nat → Nat) (kind : ArmKind) :
    (publicWordBlocks assignment kind).length = 10 := by
  simp [publicWordBlocks]

theorem flattened_public_words_length
    (assignment : Nat → Nat) (kind : ArmKind) :
    (publicWordBlocks assignment kind).flatten.length = 640 := by
  rw [List.length_flatten]
  simp only [publicWordBlocks, List.map_ofFn, publicWordBits,
    Function.comp_def, List.length_ofFn, List.ofFn_const,
    List.sum_replicate]
  norm_num

@[simp] theorem logicalCarrier_length
    (assignment : Nat → Nat) (kind : ArmKind) :
    (logicalCarrier assignment kind).length = 641 := by
  rw [logicalCarrier, List.length_append, List.length_singleton,
    flattened_public_words_length]

@[simp] theorem carrier_length
    (assignment : Nat → Nat) (kind : ArmKind) :
    (carrier assignment kind).length = 648 := by
  simp [carrier]

/-- The semantic carrier uses the exact verifier-owned range boundaries. -/
theorem exact_public_carrier_layout
    (assignment : Nat → Nat) (kind : ArmKind) :
    (logicalCarrier assignment kind).length =
        productionPublicLayout.logicalColumns ∧
      (carrier assignment kind).length = productionPublicLayout.columns ∧
      productionPublicLayout.paddingStart = 641 ∧
      productionPublicLayout.paddingEnd = 648 := by
  simp [productionPublicLayout]

private theorem getD_ofFn
    {Item : Type} {count : Nat}
    (items : Fin count → Item) (index : Fin count) (default : Item) :
    (List.ofFn items).getD index.val default = items index := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp), List.getElem_ofFn]
  simp

private theorem getD_flatten_ofFn_ofFn
    {Item : Type} {outer inner : Nat}
    (items : Fin outer → Fin inner → Item)
    (outerIndex : Fin outer) (innerIndex : Fin inner) (default : Item) :
    ((List.ofFn fun outerPosition =>
        List.ofFn fun innerPosition =>
          items outerPosition innerPosition).flatten).getD
      (outerIndex.val * inner + innerIndex.val) default =
        items outerIndex innerIndex := by
  induction outer with
  | zero => exact Fin.elim0 outerIndex
  | succ outer inductionHypothesis =>
      refine Fin.cases ?_ (fun index => ?_) outerIndex
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_zero,
          Nat.zero_mul, Nat.zero_add]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_left (by simp)]
        change (List.ofFn (items 0)).getD innerIndex.val default = _
        exact getD_ofFn _ innerIndex default
      · simp only [List.ofFn_succ, List.flatten_cons, Fin.val_succ]
        simp only [List.getD_eq_getElem?_getD]
        rw [List.getElem?_append_right (by
          simp only [List.length_ofFn]
          rw [Nat.add_mul, Nat.one_mul]
          omega)]
        simp only [List.length_ofFn]
        change ((List.ofFn fun outerPosition =>
          List.ofFn fun innerPosition =>
            items outerPosition.succ innerPosition).flatten).getD
          ((index.val + 1) * inner + innerIndex.val - inner) default = _
        have indexArithmetic :
            (index.val + 1) * inner + innerIndex.val - inner =
              index.val * inner + innerIndex.val := by
          rw [Nat.add_mul, Nat.one_mul]
          omega
        rw [indexArithmetic]
        exact inductionHypothesis
          (fun outerPosition innerPosition =>
            items outerPosition.succ innerPosition)
          index

private theorem logicalCarrier_getD_bit
    (assignment : Nat → Nat) (kind : ArmKind)
    (word : Fin 10) (bit : Fin 64) :
    (logicalCarrier assignment kind).getD
        (1 + word.val * 64 + bit.val) 0 =
      assignment (publicBitSourceColumn kind word bit) := by
  have indexEqual :
      1 + word.val * 64 + bit.val =
        (word.val * 64 + bit.val) + 1 := by
    omega
  rw [indexEqual]
  simp only [logicalCarrier, List.singleton_append,
    List.getD_eq_getElem?_getD, List.getElem?_cons_succ]
  simpa only [publicWordBlocks, publicWordBits,
    List.getD_eq_getElem?_getD] using
    getD_flatten_ofFn_ofFn
      (fun outer inner =>
        assignment (publicBitSourceColumn kind outer inner)) word bit 0

/-- Exact source column at every one of the 640 logical public bit
coordinates. -/
theorem carrier_getD_bit
    (assignment : Nat → Nat) (kind : ArmKind)
    (word : Fin 10) (bit : Fin 64) :
    (carrier assignment kind).getD
        (1 + word.val * 64 + bit.val) 0 =
      assignment (publicBitSourceColumn kind word bit) := by
  simp only [carrier, List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_left (by
    rw [logicalCarrier_length]
    omega)]
  simpa only [List.getD_eq_getElem?_getD] using
    logicalCarrier_getD_bit assignment kind word bit

theorem carrier_getD_one
    (assignment : Nat → Nat) (kind : ArmKind) :
    (carrier assignment kind).getD 0 0 = assignment 0 := by
  simp [carrier, logicalCarrier]

/-- Every coordinate after the logical prefix is zero. -/
theorem carrier_getD_padding
    (assignment : Nat → Nat) (kind : ArmKind) (padding : Fin 7) :
    (carrier assignment kind).getD (641 + padding.val) 0 = 0 := by
  have prefixBound :
      (logicalCarrier assignment kind).length ≤ 641 + padding.val := by
    rw [logicalCarrier_length]
    omega
  rw [carrier, List.getD_append_right _ _ _ _ prefixBound,
    logicalCarrier_length, Nat.add_sub_cancel_left]
  exact List.getD_replicate (x := 0) padding.isLt

private theorem publicWordBits_binary
    (assignment : Nat → Nat) (kind : ArmKind) (word : Fin 10)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (digit : Nat) (member : digit ∈ publicWordBits assignment kind word) :
    digit < 2 := by
  rcases List.mem_ofFn.mp member with ⟨bit, rfl⟩
  exact public_bit_binary kind word bit assignment canonical one satisfied

/-- All 648 public coordinates are binary. This includes the affine one and
the seven compiler padding coordinates. -/
theorem carrier_binary
    (assignment : Nat → Nat) (kind : ArmKind)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (armFor kind).Satisfied assignment)
    (digit : Nat) (member : digit ∈ carrier assignment kind) :
    digit < 2 := by
  change digit ∈
    logicalCarrier assignment kind ++ List.replicate 7 0 at member
  rcases List.mem_append.mp member with logicalMember | paddingMember
  · change digit ∈ ([assignment 0] ++
      (publicWordBlocks assignment kind).flatten) at logicalMember
    rcases List.mem_append.mp logicalMember with oneMember | wordsMember
    · simp only [List.mem_singleton] at oneMember
      subst digit
      omega
    · rcases List.mem_flatten.mp wordsMember with
        ⟨block, blockMember, digitMember⟩
      obtain ⟨word, blockEq⟩ := List.mem_ofFn.mp blockMember
      subst block
      exact publicWordBits_binary assignment kind word canonical one
        satisfied digit digitMember
  · simp only [List.mem_replicate] at paddingMember
    omega

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic
