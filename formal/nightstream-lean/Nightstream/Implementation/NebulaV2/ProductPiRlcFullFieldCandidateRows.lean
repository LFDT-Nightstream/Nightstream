import Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: exact V2 row program for one full-field PiRLC sampler candidate.

The input is one caller-owned Goldilocks field expression. The occurrence:

* uses the canonical-u64 recipe to bind its unique canonical integer;
* rejects exactly the canonical value `q - 1`;
* computes the value modulo five from its 64 Boolean bits; and
* constrains the small quotient used by that modulo-five equation.

The quotient has six bits. If its top bit is one, all five lower bits must be
zero. It is therefore in `0..32`. This is sufficient because the weighted
bit sum is at most 256. No large field quotient is allocated.

This file owns construction, placement, and exact cost. Soundness and honest
witness construction are separate modules.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcFullFieldCandidateRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

def quotientBitCount : Nat := 6
def auxiliaryCount : Nat := 78

/-- First occurrence-owned column and the caller-owned candidate expression. -/
structure Layout where
  base : Nat
  candidate : LinComb

/-- The canonical-u64 recipe owns the first 66 columns. -/
def canonicalLayout (layout : Layout) : CanonicalU64Recipe.Layout where
  base := layout.base
  input := layout.candidate

def acceptColumn (layout : Layout) : Nat := layout.base + 66
def inverseColumn (layout : Layout) : Nat := layout.base + 67
def residueColumn (layout : Layout) : Nat := layout.base + 68
def productColumn (layout : Layout) (stage : Nat) : Nat :=
  layout.base + 69 + stage
def quotientBitColumn (layout : Layout) (index : Nat) : Nat :=
  layout.base + 72 + index

/-- Exact contiguous allocation of this occurrence. -/
def allocation (layout : Layout) : List Nat :=
  (List.range auxiliaryCount).map fun offset => layout.base + offset

theorem allocation_length (layout : Layout) :
    (allocation layout).length = auxiliaryCount := by
  simp [allocation]

theorem allocation_nodup (layout : Layout) :
    (allocation layout).Nodup := by
  unfold allocation
  exact LinCombNormal.nodup_map _ _ (fun _ _ equal => by omega)
    List.nodup_range

theorem allocation_mem_iff (layout : Layout) (column : Nat) :
    column ∈ allocation layout ↔
      layout.base ≤ column ∧ column < layout.base + auxiliaryCount := by
  unfold allocation
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨offset, offsetMember, rfl⟩
    have offsetLt := List.mem_range.mp offsetMember
    omega
  · rintro ⟨lower, upper⟩
    exact List.mem_map.mpr
      ⟨column - layout.base, List.mem_range.mpr (by omega), by omega⟩

theorem allocation_nonzero
    (layout : Layout) (positive : 0 < layout.base)
    (column : Nat) (member : column ∈ allocation layout) :
    column ≠ 0 := by
  have window := (allocation_mem_iff layout column).mp member
  omega

/-- The caller-owned input must precede this occurrence. -/
def InputBelowBase (layout : Layout) : Prop :=
  ∀ column coefficient, (column, coefficient) ∈ layout.candidate →
    column < layout.base

def oneMinusAccept (layout : Layout) : LinComb :=
  [(acceptColumn layout, goldilocksP - 1), (0, 1)]

/-- Since the rejected value is `q - 1`, its field difference is `value + 1`. -/
def rejectionDifference (layout : Layout) : LinComb :=
  layout.candidate ++ [(0, 1)]

def acceptanceRows (layout : Layout) : List Row :=
  [ bitRow (acceptColumn layout),
    ⟨oneMinusAccept layout, rejectionDifference layout, []⟩,
    ⟨rejectionDifference layout, [(inverseColumn layout, 1)],
      [(acceptColumn layout, 1)]⟩,
    ⟨oneMinusAccept layout, [(inverseColumn layout, 1)], []⟩ ]

/-- Four product rows restrict the residue to exactly `0,1,2,3,4`. -/
def residueRangeRows (layout : Layout) : List Row :=
  [ ⟨[(residueColumn layout, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 1)],
      [(productColumn layout 0, 1)]⟩,
    ⟨[(productColumn layout 0, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 2)],
      [(productColumn layout 1, 1)]⟩,
    ⟨[(productColumn layout 1, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 3)],
      [(productColumn layout 2, 1)]⟩,
    ⟨[(productColumn layout 2, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 4)], []⟩ ]

def quotientBitRows (layout : Layout) : List Row :=
  (List.range quotientBitCount).map fun index =>
    bitRow (quotientBitColumn layout index)

/-- If bit five is one, bits zero through four must be zero. -/
def quotientUpperRows (layout : Layout) : List Row :=
  (List.range 5).map fun index =>
    ⟨[(quotientBitColumn layout 5, 1)],
      [(quotientBitColumn layout index, 1)], []⟩

/-- Each power-of-two bit gets its residue modulo five. -/
def weightedBitTerms (layout : Layout) : LinComb :=
  (List.range 64).map fun index =>
    (CanonicalU64Recipe.bitColumn (canonicalLayout layout) index,
      (2 ^ index) % 5)

def quotientTerms (layout : Layout) : LinComb :=
  (List.range quotientBitCount).map fun index =>
    (quotientBitColumn layout index, 5 * 2 ^ index)

/-- Exact small-integer equation `weighted bits = 5 * quotient + residue`. -/
def moduloFiveRow (layout : Layout) : Row :=
  ⟨weightedBitTerms layout, [(0, 1)],
    quotientTerms layout ++ [(residueColumn layout, 1)]⟩

/-- Complete row program for one V2 full-field candidate. -/
def rows (layout : Layout) : List Row :=
  CanonicalU64Recipe.rows (canonicalLayout layout) ++
    acceptanceRows layout ++ residueRangeRows layout ++
    quotientBitRows layout ++ quotientUpperRows layout ++
    [moduloFiveRow layout]

theorem rows_length (layout : Layout) :
    (rows layout).length = 89 := by
  simp [rows, CanonicalU64Recipe.rows_length, acceptanceRows,
    residueRangeRows, quotientBitRows, quotientUpperRows, quotientBitCount]

def cost : Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := 89
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := auxiliaryCount

theorem cost_rows (layout : Layout) :
    (rows layout).length = cost.recurringRows := rows_length layout

theorem cost_columns (layout : Layout) :
    (allocation layout).length = cost.auxiliaryColumns :=
  allocation_length layout

theorem canonical_allocation_subset
    (layout : Layout) (column : Nat)
    (member : column ∈ CanonicalU64Recipe.allocation (canonicalLayout layout)) :
    column ∈ allocation layout := by
  have window := CanonicalU64Recipe.allocation_in_window
    (canonicalLayout layout) column member
  apply (allocation_mem_iff layout column).mpr
  have lower : layout.base ≤ column := by
    simpa [canonicalLayout] using window.1
  have upper : column < layout.base + 66 := by
    simpa [canonicalLayout, CanonicalU64Recipe.auxiliaryCount] using window.2
  exact ⟨lower, by simp only [auxiliaryCount]; omega⟩

end Nightstream.Implementation.NebulaV2.ProductPiRlcFullFieldCandidateRows
