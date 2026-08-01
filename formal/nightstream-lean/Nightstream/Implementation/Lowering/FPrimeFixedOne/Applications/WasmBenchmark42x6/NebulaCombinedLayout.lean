import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaFreshCarrier
import Nightstream.Implementation.Lowering.Nebula.Layout

/-!
Physical column layout for the Nebula-enabled 42-times-6 relation.

Assurance tier: model-level.

This file owns one injective placement of the existing F-prime source
columns and the selected Nebula source columns into one logical assignment.
The first 2,430 coordinates are exactly `NebulaFreshCarrier`; all native and
Nebula private coordinates occupy disjoint suffixes.

It does not own matrices, row placement, assignments, a recursive fixed
point, Rust, or a security reduction.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedLayout

open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaFreshCarrier
open Nightstream.SuperNeo.Concrete

/-- Public coordinates inserted after the existing 257-coordinate F-prime
link. This includes all Nebula public data and the four ring-alignment zeros. -/
def insertedPublicWidth : Nat :=
  alignedPublicWidth - linkWidth

/-- The selected standalone Nebula program has one constant coordinate and
1,400 public memory coordinates before its private witness suffix. -/
def nebulaPublicEnd : Nat := Layout.wasm42x6.publicEnd

/-- Private Nebula witness width after the public prefix. -/
def nebulaPrivateWidth : Nat :=
  Layout.wasm42x6.columnCount - nebulaPublicEnd

theorem staticWidths_exact :
    insertedPublicWidth = 2173 /\
      nebulaPublicEnd = 1401 /\
      nebulaPrivateWidth = 418346 := by
  decide

/-- Shape data needed to place one native F-prime source. The source must
already contain the existing 257-coordinate public link. -/
structure Dimensions where
  rowVariables : Nat
  nativeLogicalWidth : Nat
  nativePublicFits : linkWidth <= nativeLogicalWidth

namespace Dimensions

def nativePrivateWidth (dimensions : Dimensions) : Nat :=
  dimensions.nativeLogicalWidth - linkWidth

def nebulaPrivateStart (dimensions : Dimensions) : Nat :=
  alignedPublicWidth + dimensions.nativePrivateWidth

/-- Exact combined logical width. No alignment count is inferred from a
declared cost: both private suffixes appear explicitly. -/
def logicalWidth (dimensions : Dimensions) : Nat :=
  dimensions.nebulaPrivateStart + nebulaPrivateWidth

theorem logicalWidth_eq
    (dimensions : Dimensions) :
    dimensions.logicalWidth =
      alignedPublicWidth +
        (dimensions.nativeLogicalWidth - linkWidth) +
        nebulaPrivateWidth := by
  rfl

theorem publicFits (dimensions : Dimensions) :
    alignedPublicWidth <= dimensions.logicalWidth := by
  unfold logicalWidth nebulaPrivateStart
  omega

end Dimensions

/-- Native public coordinates remain fixed. Native private coordinates move
after the complete 45-ring public carrier. -/
def nativeColumn
    (dimensions : Dimensions)
    (column : Fin dimensions.nativeLogicalWidth) :
    Fin dimensions.logicalWidth :=
  if publicCoordinate : column.val < linkWidth then
    ⟨column.val, by
      have columnBound := column.isLt
      have publicFits := dimensions.publicFits
      have linkFits : linkWidth <= alignedPublicWidth := by decide
      omega⟩
  else
    ⟨alignedPublicWidth + (column.val - linkWidth), by
      have columnBound := column.isLt
      have sourceFits := dimensions.nativePublicFits
      unfold Dimensions.logicalWidth Dimensions.nebulaPrivateStart
        Dimensions.nativePrivateWidth
      omega⟩

theorem nativeColumn_public
    (dimensions : Dimensions)
    (column : Fin dimensions.nativeLogicalWidth)
    (publicCoordinate : column.val < linkWidth) :
    (nativeColumn dimensions column).val = column.val := by
  simp [nativeColumn, publicCoordinate]

theorem nativeColumn_private
    (dimensions : Dimensions)
    (column : Fin dimensions.nativeLogicalWidth)
    (privateCoordinate : linkWidth <= column.val) :
    (nativeColumn dimensions column).val =
      alignedPublicWidth + (column.val - linkWidth) := by
  simp [nativeColumn, Nat.not_lt.mpr privateCoordinate]

theorem nativeColumn_injective
    (dimensions : Dimensions) :
    Function.Injective (nativeColumn dimensions) := by
  intro left right equal
  have values := congrArg Fin.val equal
  by_cases leftPublic : left.val < linkWidth
  · rw [nativeColumn_public dimensions left leftPublic] at values
    by_cases rightPublic : right.val < linkWidth
    · rw [nativeColumn_public dimensions right rightPublic] at values
      exact Fin.ext values
    · rw [nativeColumn_private dimensions right
          (Nat.not_lt.mp rightPublic)] at values
      have rightPrivate := Nat.not_lt.mp rightPublic
      have separated : linkWidth < alignedPublicWidth := by decide
      omega
  · rw [nativeColumn_private dimensions left
        (Nat.not_lt.mp leftPublic)] at values
    by_cases rightPublic : right.val < linkWidth
    · rw [nativeColumn_public dimensions right rightPublic] at values
      have leftPrivate := Nat.not_lt.mp leftPublic
      have separated : linkWidth < alignedPublicWidth := by decide
      omega
    · rw [nativeColumn_private dimensions right
          (Nat.not_lt.mp rightPublic)] at values
      apply Fin.ext
      have leftPrivate := Nat.not_lt.mp leftPublic
      have rightPrivate := Nat.not_lt.mp rightPublic
      omega

/-- Place one standalone Nebula coordinate. Its constant wire aliases native
public coordinate zero; its 1,400 public values occupy the memory block; its
private witness occupies the final disjoint suffix. -/
def nebulaColumn
    (dimensions : Dimensions)
    (column : Fin Layout.wasm42x6.columnCount) :
    Fin dimensions.logicalWidth :=
  if constantCoordinate : column.val = 0 then
    ⟨0, by
      have publicFits := dimensions.publicFits
      exact Nat.lt_of_lt_of_le (by decide) publicFits⟩
  else if publicCoordinate : column.val < nebulaPublicEnd then
    ⟨memoryStart + (column.val - 1), by
      have columnPositive : 1 <= column.val := Nat.one_le_iff_ne_zero.mpr
        constantCoordinate
      have publicBound := publicCoordinate
      have publicFits := dimensions.publicFits
      have memoryRange :
          memoryStart + (nebulaPublicEnd - 1) <= alignedPublicWidth := by
        decide
      omega⟩
  else
    ⟨dimensions.nebulaPrivateStart + (column.val - nebulaPublicEnd), by
      have columnBound := column.isLt
      have privateCoordinate := Nat.not_lt.mp publicCoordinate
      unfold Dimensions.logicalWidth nebulaPrivateWidth
      omega⟩

@[simp] theorem nebulaColumn_zero (dimensions : Dimensions) :
    (nebulaColumn dimensions ⟨0, by
      rw [Layout.wasm42x6_columnCount]
      decide⟩).val = 0 := by
  simp [nebulaColumn]

theorem nebulaColumn_public
    (dimensions : Dimensions)
    (column : Fin Layout.wasm42x6.columnCount)
    (positive : 0 < column.val)
    (publicCoordinate : column.val < nebulaPublicEnd) :
    (nebulaColumn dimensions column).val =
      memoryStart + (column.val - 1) := by
  simp [nebulaColumn, Nat.ne_of_gt positive, publicCoordinate]

theorem nebulaColumn_private
    (dimensions : Dimensions)
    (column : Fin Layout.wasm42x6.columnCount)
    (privateCoordinate : nebulaPublicEnd <= column.val) :
    (nebulaColumn dimensions column).val =
      dimensions.nebulaPrivateStart +
        (column.val - nebulaPublicEnd) := by
  have nonzero : column.val ≠ 0 := by
    have endPositive : 0 < nebulaPublicEnd := by decide
    omega
  simp [nebulaColumn, nonzero, Nat.not_lt.mpr privateCoordinate]

theorem nebulaColumn_injective
    (dimensions : Dimensions) :
    Function.Injective (nebulaColumn dimensions) := by
  intro left right equal
  have values := congrArg Fin.val equal
  by_cases leftZero : left.val = 0
  ·
    have mappedLeft : (nebulaColumn dimensions left).val = 0 := by
      simp [nebulaColumn, leftZero]
    rw [mappedLeft] at values
    by_cases rightZero : right.val = 0
    · apply Fin.ext
      omega
    · by_cases rightPublic : right.val < nebulaPublicEnd
      · rw [nebulaColumn_public dimensions right
            (Nat.pos_of_ne_zero rightZero) rightPublic] at values
        have memoryExact : memoryStart = 257 := offsets_exact.1
        omega
      · rw [nebulaColumn_private dimensions right
            (Nat.not_lt.mp rightPublic)] at values
        have startBound : alignedPublicWidth <=
            dimensions.nebulaPrivateStart := by
          unfold Dimensions.nebulaPrivateStart
          omega
        have alignedPositive : 0 < alignedPublicWidth := by decide
        omega
  · by_cases rightZero : right.val = 0
    ·
      have mappedRight : (nebulaColumn dimensions right).val = 0 := by
        simp [nebulaColumn, rightZero]
      rw [mappedRight] at values
      by_cases leftPublic : left.val < nebulaPublicEnd
      · rw [nebulaColumn_public dimensions left
            (Nat.pos_of_ne_zero leftZero) leftPublic] at values
        have memoryExact : memoryStart = 257 := offsets_exact.1
        omega
      · rw [nebulaColumn_private dimensions left
            (Nat.not_lt.mp leftPublic)] at values
        have startBound : alignedPublicWidth <=
            dimensions.nebulaPrivateStart := by
          unfold Dimensions.nebulaPrivateStart
          omega
        have alignedPositive : 0 < alignedPublicWidth := by decide
        omega
    · by_cases leftPublic : left.val < nebulaPublicEnd
      · rw [nebulaColumn_public dimensions left
            (Nat.pos_of_ne_zero leftZero) leftPublic] at values
        by_cases rightPublic : right.val < nebulaPublicEnd
        · rw [nebulaColumn_public dimensions right
              (Nat.pos_of_ne_zero rightZero) rightPublic] at values
          apply Fin.ext
          omega
        · rw [nebulaColumn_private dimensions right
              (Nat.not_lt.mp rightPublic)] at values
          have leftUpper :
              memoryStart + (left.val - 1) < alignedPublicWidth := by
            have leftPositive := Nat.pos_of_ne_zero leftZero
            have memoryRange :
                memoryStart + (nebulaPublicEnd - 1) <=
                  alignedPublicWidth := by decide
            omega
          have rightLower : alignedPublicWidth <=
              dimensions.nebulaPrivateStart +
                (right.val - nebulaPublicEnd) := by
            unfold Dimensions.nebulaPrivateStart
            omega
          omega
      · rw [nebulaColumn_private dimensions left
            (Nat.not_lt.mp leftPublic)] at values
        by_cases rightPublic : right.val < nebulaPublicEnd
        · rw [nebulaColumn_public dimensions right
              (Nat.pos_of_ne_zero rightZero) rightPublic] at values
          have rightUpper :
              memoryStart + (right.val - 1) < alignedPublicWidth := by
            have rightPositive := Nat.pos_of_ne_zero rightZero
            have memoryRange :
                memoryStart + (nebulaPublicEnd - 1) <=
                  alignedPublicWidth := by decide
            omega
          have leftLower : alignedPublicWidth <=
              dimensions.nebulaPrivateStart +
                (left.val - nebulaPublicEnd) := by
            unfold Dimensions.nebulaPrivateStart
            omega
          omega
        · rw [nebulaColumn_private dimensions right
              (Nat.not_lt.mp rightPublic)] at values
          apply Fin.ext
          have leftPrivate := Nat.not_lt.mp leftPublic
          have rightPrivate := Nat.not_lt.mp rightPublic
          omega

/-- Native and Nebula private columns cannot collide. Their only allowed
overlap is the shared constant at coordinate zero. -/
theorem private_regions_disjoint
    (dimensions : Dimensions)
    (native : Fin dimensions.nativeLogicalWidth)
    (nebula : Fin Layout.wasm42x6.columnCount)
    (nativePrivate : linkWidth <= native.val)
    (nebulaPrivate : nebulaPublicEnd <= nebula.val) :
    nativeColumn dimensions native ≠ nebulaColumn dimensions nebula := by
  intro equal
  have values := congrArg Fin.val equal
  rw [nativeColumn_private dimensions native nativePrivate,
    nebulaColumn_private dimensions nebula nebulaPrivate] at values
  have nativeBound := native.isLt
  have nativeFits := dimensions.nativePublicFits
  have nativeUpper :
      alignedPublicWidth + (native.val - linkWidth) <
        dimensions.nebulaPrivateStart := by
    unfold Dimensions.nebulaPrivateStart Dimensions.nativePrivateWidth
    omega
  have nebulaLower :
      dimensions.nebulaPrivateStart <=
        dimensions.nebulaPrivateStart +
          (nebula.val - nebulaPublicEnd) := by
    omega
  omega

/-- A combined coordinate has no source ambiguity outside the intentionally
shared constant wire. -/
theorem source_overlap_only_constant
    (dimensions : Dimensions)
    (native : Fin dimensions.nativeLogicalWidth)
    (nebula : Fin Layout.wasm42x6.columnCount)
    (equal : nativeColumn dimensions native =
      nebulaColumn dimensions nebula) :
    native.val = 0 ∧ nebula.val = 0 := by
  by_cases nativePublic : native.val < linkWidth
  · by_cases nebulaZero : nebula.val = 0
    · exact ⟨by
        have values := congrArg Fin.val equal
        rw [nativeColumn_public dimensions native nativePublic] at values
        have nebulaMapped : (nebulaColumn dimensions nebula).val = 0 := by
          simp [nebulaColumn, nebulaZero]
        rw [nebulaMapped] at values
        exact values, nebulaZero⟩
    · by_cases nebulaPublic : nebula.val < nebulaPublicEnd
      · have values := congrArg Fin.val equal
        rw [nativeColumn_public dimensions native nativePublic,
          nebulaColumn_public dimensions nebula
            (Nat.pos_of_ne_zero nebulaZero) nebulaPublic] at values
        have nativeUpper : native.val < memoryStart := by
          have exactStart : memoryStart = linkWidth := rfl
          omega
        have nebulaLower : memoryStart <=
            memoryStart + (nebula.val - 1) := by omega
        omega
      · have values := congrArg Fin.val equal
        rw [nativeColumn_public dimensions native nativePublic,
          nebulaColumn_private dimensions nebula
            (Nat.not_lt.mp nebulaPublic)] at values
        have nativeUpper : native.val < alignedPublicWidth := by
          have linkFits : linkWidth <= alignedPublicWidth := by decide
          omega
        have nebulaLower : alignedPublicWidth <=
            dimensions.nebulaPrivateStart +
              (nebula.val - nebulaPublicEnd) := by
          unfold Dimensions.nebulaPrivateStart
          omega
        omega
  · have nebulaPrivate : nebulaPublicEnd <= nebula.val := by
      apply Nat.le_of_not_gt
      intro nebulaPublic
      by_cases nebulaZero : nebula.val = 0
      · have values := congrArg Fin.val equal
        rw [nativeColumn_private dimensions native
            (Nat.not_lt.mp nativePublic)] at values
        have nebulaMapped : (nebulaColumn dimensions nebula).val = 0 := by
          simp [nebulaColumn, nebulaZero]
        rw [nebulaMapped] at values
        have alignedPositive : 0 < alignedPublicWidth := by decide
        omega
      · have values := congrArg Fin.val equal
        rw [nativeColumn_private dimensions native
            (Nat.not_lt.mp nativePublic),
          nebulaColumn_public dimensions nebula
            (Nat.pos_of_ne_zero nebulaZero) nebulaPublic] at values
        have nativeLower : alignedPublicWidth <=
            alignedPublicWidth + (native.val - linkWidth) := by omega
        have nebulaUpper :
            memoryStart + (nebula.val - 1) < alignedPublicWidth := by
          have memoryRange :
              memoryStart + (nebulaPublicEnd - 1) <=
                alignedPublicWidth := by decide
          omega
        omega
    have contradiction :=
      private_regions_disjoint dimensions native nebula
        (Nat.not_lt.mp nativePublic) nebulaPrivate equal
    exact False.elim contradiction

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaCombinedLayout
