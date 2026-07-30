import Nightstream.Implementation.R1CS.Canonical.LinCombNormal
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: a Lean-owned canonical decomposition of one Goldilocks residue into
64 little-endian bits.

The row program is constructed from the mathematical equations:

* 64 Boolean bit rows;
* exact field recomposition;
* one Boolean flag for `high32 = 0xffffffff`;
* an inverse-gated equality deciding that flag; and
* `flag * low32 = 0`, which excludes the noncanonical `value + p`
  representation.

The input is an arbitrary linear combination, so a Poseidon2 output-port
expression can be decomposed without allocating a duplicate field-value
column.  The recipe owns only the 64 bits, the high-word flag, and the inverse
witness.

This file owns construction, placement, conservation, and exact cost.  The
soundness and honest-witness proofs are separate proof responsibilities; no
generated row or Rust artifact is imported here.

Assurance tier: canonical encoding construction.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

def bitCount : Nat := 64
def auxiliaryCount : Nat := 66
def highMax : Nat := 4294967295

/-- First recipe-owned column. -/
structure Layout where
  base : Nat
  input : LinCombNormal.LinComb

def bitColumn (layout : Layout) (index : Nat) : Nat :=
  layout.base + index

def highFlagColumn (layout : Layout) : Nat :=
  layout.base + 64

def inverseColumn (layout : Layout) : Nat :=
  layout.base + 65

/-- Exact list of columns allocated by this occurrence. -/
def allocation (layout : Layout) : List Nat :=
  (List.range 64).map (bitColumn layout) ++
    [highFlagColumn layout, inverseColumn layout]

theorem allocation_length (layout : Layout) :
    (allocation layout).length = auxiliaryCount := by
  simp [allocation, auxiliaryCount]

theorem allocation_nodup (layout : Layout) :
    (allocation layout).Nodup := by
  unfold allocation
  rw [List.nodup_append]
  refine ⟨?_, ?_, ?_⟩
  · exact LinCombNormal.nodup_map (List.range 64)
      (bitColumn layout)
      (fun left right equal => by
        simp only [bitColumn] at equal
        omega)
      List.nodup_range
  · simp [highFlagColumn, inverseColumn]
  · intro column inBits tailColumn inTail equal
    rcases List.mem_map.mp inBits with ⟨index, indexIn, rfl⟩
    have indexLt := List.mem_range.mp indexIn
    simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
    rcases inTail with rfl | rfl
    · simp only [bitColumn, highFlagColumn] at equal
      omega
    · simp only [bitColumn, inverseColumn] at equal
      omega

theorem allocation_nonzero
    (layout : Layout) (positive : 0 < layout.base)
    (column : Nat) (member : column ∈ allocation layout) :
    column ≠ 0 := by
  unfold allocation at member
  rcases List.mem_append.mp member with inBits | inTail
  · rcases List.mem_map.mp inBits with ⟨index, _, rfl⟩
    simp only [bitColumn]
    omega
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
    rcases inTail with rfl | rfl
    · simp only [highFlagColumn]
      omega
    · simp only [inverseColumn]
      omega

/-- Every allocated column lies in the exact contiguous 66-column window. -/
theorem allocation_in_window
    (layout : Layout) (column : Nat)
    (member : column ∈ allocation layout) :
    layout.base ≤ column ∧ column < layout.base + auxiliaryCount := by
  unfold allocation at member
  rcases List.mem_append.mp member with inBits | inTail
  · rcases List.mem_map.mp inBits with ⟨index, indexIn, rfl⟩
    have indexLt := List.mem_range.mp indexIn
    simp only [bitColumn, auxiliaryCount]
    omega
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
    rcases inTail with rfl | rfl
    · simp only [highFlagColumn, auxiliaryCount]
      omega
    · simp only [inverseColumn, auxiliaryCount]
      omega

/-- Positive powers used by the field recomposition equation. -/
def bitTerms (layout : Layout) : LinCombNormal.LinComb :=
  (List.range 64).map fun index =>
    (bitColumn layout index, 2 ^ index)

def lowTerms (layout : Layout) : LinCombNormal.LinComb :=
  (List.range 32).map fun index =>
    (bitColumn layout index, 2 ^ index)

def highTerms (layout : Layout) : LinCombNormal.LinComb :=
  (List.range 32).map fun index =>
    (bitColumn layout (32 + index), 2 ^ index)

def highDifferenceTerms (layout : Layout) : LinCombNormal.LinComb :=
  highTerms layout ++ [(0, goldilocksP - highMax)]

def oneMinusHighFlag (layout : Layout) : LinCombNormal.LinComb :=
  [(highFlagColumn layout, goldilocksP - 1), (0, 1)]

def zeroEqualityRow (terms : LinCombNormal.LinComb) : Row :=
  ⟨terms, [(0, 1)], []⟩

def bitRows (layout : Layout) : List Row :=
  (List.range 64).map fun index => bitRow (bitColumn layout index)

def recompositionRow (layout : Layout) : Row :=
  ⟨layout.input, [(0, 1)], bitTerms layout⟩

def highDefinitionRow (layout : Layout) : Row :=
  ⟨[(highFlagColumn layout, 1)], highDifferenceTerms layout, []⟩

def inverseRow (layout : Layout) : Row :=
  ⟨highDifferenceTerms layout, [(inverseColumn layout, 1)],
    oneMinusHighFlag layout⟩

def canonicalityRow (layout : Layout) : Row :=
  ⟨[(highFlagColumn layout, 1)], lowTerms layout, []⟩

/-- Complete canonical-u64 row program. -/
def rows (layout : Layout) : List Row :=
  bitRows layout ++
    [ recompositionRow layout,
      bitRow (highFlagColumn layout),
      highDefinitionRow layout,
      inverseRow layout,
      canonicalityRow layout ]

theorem bitRows_length (layout : Layout) :
    (bitRows layout).length = 64 := by
  simp [bitRows]

/-- Exact row count, derived from the constructed row list. -/
theorem rows_length (layout : Layout) :
    (rows layout).length = 69 := by
  simp [rows, bitRows]

/-- Exact intrinsic cost.  The input is a read, not a second allocation. -/
def cost : Lowering.Typed.Cost where
  recurringRows := 69
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 66

theorem cost_rows (layout : Layout) :
    (rows layout).length = cost.recurringRows :=
  rows_length layout

theorem cost_columns (layout : Layout) :
    (allocation layout).length = cost.auxiliaryColumns :=
  allocation_length layout

/-- The caller-owned input columns must precede the recipe allocation. -/
def InputBelowBase (layout : Layout) : Prop :=
  ∀ column coefficient, (column, coefficient) ∈ layout.input →
    column < layout.base

/-- Every referenced column is the shared constant wire, a caller-owned input,
or one of the 66 declared allocations. -/
theorem rows_conservation
    (layout : Layout) (row : Row) (rowMember : row ∈ rows layout)
    (column coefficient : Nat)
    (mentioned :
      (column, coefficient) ∈ row.a ∨
      (column, coefficient) ∈ row.b ∨
      (column, coefficient) ∈ row.c) :
    column = 0 ∨
      (∃ inputCoefficient, (column, inputCoefficient) ∈ layout.input) ∨
      column ∈ allocation layout := by
  unfold rows at rowMember
  rcases List.mem_append.mp rowMember with inBits | inTail
  · rcases List.mem_map.mp inBits with ⟨index, indexIn, rfl⟩
    simp only [bitRow] at mentioned
    rcases mentioned with mentioned | mentioned | mentioned
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
      rcases mentioned with ⟨rfl, rfl⟩
      right; right
      unfold allocation
      apply List.mem_append_left
      exact List.mem_map.mpr ⟨index, indexIn, rfl⟩
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
      rcases mentioned with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · right; right
        unfold allocation
        apply List.mem_append_left
        exact List.mem_map.mpr ⟨index, indexIn, rfl⟩
      · left; rfl
    · cases mentioned
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
    rcases inTail with rfl | rfl | rfl | rfl | rfl
    · simp only [recompositionRow] at mentioned
      rcases mentioned with mentioned | mentioned | mentioned
      · right; left; exact ⟨coefficient, mentioned⟩
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
        rcases mentioned with ⟨rfl, rfl⟩
        left; rfl
      · unfold bitTerms at mentioned
        rcases List.mem_map.mp mentioned with ⟨index, indexIn, pairEq⟩
        right; right
        unfold allocation
        apply List.mem_append_left
        apply List.mem_map.mpr
        exact ⟨index, indexIn, by
          simp only [Prod.mk.injEq] at pairEq
          exact pairEq.1⟩
    · simp only [bitRow] at mentioned
      rcases mentioned with mentioned | mentioned | mentioned
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
        rcases mentioned with ⟨rfl, rfl⟩
        right; right
        unfold allocation
        simp
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
        rcases mentioned with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
        · right; right
          unfold allocation
          simp
        · left; rfl
      · cases mentioned
    · simp only [highDefinitionRow] at mentioned
      rcases mentioned with mentioned | mentioned | mentioned
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
        rcases mentioned with ⟨rfl, rfl⟩
        right; right
        unfold allocation
        simp
      · unfold highDifferenceTerms highTerms at mentioned
        rcases List.mem_append.mp mentioned with inHigh | inConstant
        · rcases List.mem_map.mp inHigh with
            ⟨index, indexIn, pairEq⟩
          right; right
          unfold allocation
          apply List.mem_append_left
          apply List.mem_map.mpr
          exact ⟨32 + index, by
            apply List.mem_range.mpr
            have indexLt := List.mem_range.mp indexIn
            omega, by
              simp only [Prod.mk.injEq] at pairEq
              exact pairEq.1⟩
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at inConstant
          rcases inConstant with ⟨rfl, rfl⟩
          left; rfl
      · cases mentioned
    · simp only [inverseRow] at mentioned
      rcases mentioned with mentioned | mentioned | mentioned
      · unfold highDifferenceTerms highTerms at mentioned
        rcases List.mem_append.mp mentioned with inHigh | inConstant
        · rcases List.mem_map.mp inHigh with
            ⟨index, indexIn, pairEq⟩
          right; right
          unfold allocation
          apply List.mem_append_left
          apply List.mem_map.mpr
          exact ⟨32 + index, by
            apply List.mem_range.mpr
            have indexLt := List.mem_range.mp indexIn
            omega, by
              simp only [Prod.mk.injEq] at pairEq
              exact pairEq.1⟩
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at inConstant
          rcases inConstant with ⟨rfl, rfl⟩
          left; rfl
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
        rcases mentioned with ⟨rfl, rfl⟩
        right; right
        unfold allocation
        simp
      · unfold oneMinusHighFlag at mentioned
        simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
        rcases mentioned with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
        · right; right
          unfold allocation
          simp
        · left; rfl
    · simp only [canonicalityRow] at mentioned
      rcases mentioned with mentioned | mentioned | mentioned
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at mentioned
        rcases mentioned with ⟨rfl, rfl⟩
        right; right
        unfold allocation
        simp
      · unfold lowTerms at mentioned
        rcases List.mem_map.mp mentioned with ⟨index, indexIn, pairEq⟩
        right; right
        unfold allocation
        apply List.mem_append_left
        apply List.mem_map.mpr
        exact ⟨index, by
          apply List.mem_range.mpr
          have indexLt := List.mem_range.mp indexIn
          omega, by
            simp only [Prod.mk.injEq] at pairEq
            exact pairEq.1⟩
      · cases mentioned

end Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
