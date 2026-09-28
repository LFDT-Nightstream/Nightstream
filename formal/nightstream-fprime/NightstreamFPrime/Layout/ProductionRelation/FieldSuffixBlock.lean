import NightstreamFPrime.Layout.LowNormBlock

/-!
Owns a generic suffix of derived general-field source values. Existing source
columns remain a fixed prefix. The derived values and their retained slots
use one exact indexed order.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.FieldSuffixBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout

def sourceWidth (baseSourceWidth derivedCount : Nat) : Nat :=
  baseSourceWidth + derivedCount

def baseColumn (baseSourceWidth derivedCount : Nat)
    (column : Fin baseSourceWidth) : Fin (sourceWidth baseSourceWidth derivedCount) :=
  ⟨column.val, by
    unfold sourceWidth
    omega⟩

def derivedColumn (baseSourceWidth derivedCount : Nat)
    (index : Fin derivedCount) : Fin (sourceWidth baseSourceWidth derivedCount) :=
  ⟨baseSourceWidth + index.val, by
    unfold sourceWidth
    omega⟩

def sourceAssignment (baseSourceWidth derivedCount : Nat)
    (base : Fin baseSourceWidth → F) (derived : Fin derivedCount → F) :
    Fin (sourceWidth baseSourceWidth derivedCount) → F :=
  fun column =>
    if baseRegion : column.val < baseSourceWidth then
      base ⟨column.val, baseRegion⟩
    else
      derived ⟨column.val - baseSourceWidth, by
        have bound := column.isLt
        unfold sourceWidth at bound
        omega⟩

@[simp] theorem sourceAssignment_base (baseSourceWidth derivedCount : Nat)
    (base : Fin baseSourceWidth → F) (derived : Fin derivedCount → F)
    (column : Fin baseSourceWidth) :
    sourceAssignment baseSourceWidth derivedCount base derived
        (baseColumn baseSourceWidth derivedCount column) =
      base column := by
  unfold sourceAssignment baseColumn
  rw [dif_pos column.isLt]

@[simp] theorem sourceAssignment_derived (baseSourceWidth derivedCount : Nat)
    (base : Fin baseSourceWidth → F) (derived : Fin derivedCount → F)
    (index : Fin derivedCount) :
    sourceAssignment baseSourceWidth derivedCount base derived
        (derivedColumn baseSourceWidth derivedCount index) =
      derived index := by
  simp [sourceAssignment, derivedColumn]

def block (baseSourceWidth derivedCount : Nat) :
    LowNormBlock.Block (sourceWidth baseSourceWidth derivedCount) where
  kind := .field
  slotCount := derivedCount
  source := derivedColumn baseSourceWidth derivedCount

@[simp] theorem block_slotCount (baseSourceWidth derivedCount : Nat) :
    (block baseSourceWidth derivedCount).slotCount = derivedCount := by
  rfl

@[simp] theorem block_coordinateCount (baseSourceWidth derivedCount : Nat) :
    (block baseSourceWidth derivedCount).coordinateCount = derivedCount * 41 := by
  simp [block, LowNormBlock.Block.coordinateCount, LowNormSlot.Kind.width,
    BalancedTernary.width]

theorem block_sourceAssignment (baseSourceWidth derivedCount : Nat)
    (base : Fin baseSourceWidth → F) (derived : Fin derivedCount → F)
    (index : Fin derivedCount) :
    sourceAssignment baseSourceWidth derivedCount base derived
        ((block baseSourceWidth derivedCount).source index) =
      derived index := by
  exact sourceAssignment_derived baseSourceWidth derivedCount base derived index

end NightstreamFPrime.Layout.ProductionRelation.FieldSuffixBlock
