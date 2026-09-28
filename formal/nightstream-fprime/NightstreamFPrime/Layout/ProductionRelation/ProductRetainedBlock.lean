import Batteries.Data.Fin.Coding
import NightstreamFPrime.Layout.LowNormBlock

/-!
Owns the augmented source suffix and retained low-norm block for fixed
five-product invocation outputs. Each invocation contributes 33 general-field
group values in invocation-major order.

This module does not select a concrete invocation schedule.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.ProductRetainedBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout

def sourceWidth (baseSourceWidth invocationCount : Nat) : Nat :=
  baseSourceWidth + invocationCount * 33

def baseColumn (baseSourceWidth invocationCount : Nat)
    (column : Fin baseSourceWidth) :
    Fin (sourceWidth baseSourceWidth invocationCount) :=
  ⟨column.val, by
    unfold sourceWidth
    omega⟩

def groupColumn (baseSourceWidth invocationCount : Nat)
    (invocation : Fin invocationCount) (group : Fin 33) :
    Fin (sourceWidth baseSourceWidth invocationCount) :=
  ⟨baseSourceWidth + (Fin.encodeProd (invocation, group)).val, by
    have encodedBound := (Fin.encodeProd (invocation, group)).isLt
    unfold sourceWidth
    omega⟩

/-- Total augmented source assignment. -/
def sourceAssignment (baseSourceWidth invocationCount : Nat)
    (base : Fin baseSourceWidth → F)
    (groupValue : Fin invocationCount → Fin 33 → F) :
    Fin (sourceWidth baseSourceWidth invocationCount) → F :=
  fun column =>
    if baseRegion : column.val < baseSourceWidth then
      base ⟨column.val, baseRegion⟩
    else
      let offset : Fin (invocationCount * 33) :=
        ⟨column.val - baseSourceWidth, by
          have bound := column.isLt
          unfold sourceWidth at bound
          omega⟩
      let decoded : Fin invocationCount × Fin 33 := Fin.decodeProd offset
      groupValue decoded.1 decoded.2

@[simp] theorem sourceAssignment_base (baseSourceWidth invocationCount : Nat)
    (base : Fin baseSourceWidth → F)
    (groupValue : Fin invocationCount → Fin 33 → F)
    (column : Fin baseSourceWidth) :
    sourceAssignment baseSourceWidth invocationCount base groupValue
        (baseColumn baseSourceWidth invocationCount column) =
      base column := by
  unfold sourceAssignment baseColumn
  rw [dif_pos column.isLt]

@[simp] theorem sourceAssignment_group
    (baseSourceWidth invocationCount : Nat)
    (base : Fin baseSourceWidth → F)
    (groupValue : Fin invocationCount → Fin 33 → F)
    (invocation : Fin invocationCount) (group : Fin 33) :
    sourceAssignment baseSourceWidth invocationCount base groupValue
        (groupColumn baseSourceWidth invocationCount invocation group) =
      groupValue invocation group := by
  simp [sourceAssignment, groupColumn]

/-- One field slot for every derived group output. -/
def block (baseSourceWidth invocationCount : Nat) :
    LowNormBlock.Block (sourceWidth baseSourceWidth invocationCount) where
  kind := .field
  slotCount := invocationCount * 33
  source := fun slot =>
    let decoded : Fin invocationCount × Fin 33 := Fin.decodeProd slot
    groupColumn baseSourceWidth invocationCount decoded.1 decoded.2

@[simp] theorem block_slotCount (baseSourceWidth invocationCount : Nat) :
    (block baseSourceWidth invocationCount).slotCount = invocationCount * 33 := by
  rfl

@[simp] theorem block_coordinateCount (baseSourceWidth invocationCount : Nat) :
    (block baseSourceWidth invocationCount).coordinateCount =
      invocationCount * 33 * 41 := by
  simp [block, LowNormBlock.Block.coordinateCount, LowNormSlot.Kind.width,
    BalancedTernary.width]

theorem block_source (baseSourceWidth invocationCount : Nat)
    (slot : Fin (invocationCount * 33)) :
    let decoded : Fin invocationCount × Fin 33 := Fin.decodeProd slot
    ((block baseSourceWidth invocationCount).source slot).val =
      baseSourceWidth + (Fin.encodeProd decoded).val := by
  rfl

theorem block_sourceAssignment (baseSourceWidth invocationCount : Nat)
    (base : Fin baseSourceWidth → F)
    (groupValue : Fin invocationCount → Fin 33 → F)
    (slot : Fin (invocationCount * 33)) :
    let decoded : Fin invocationCount × Fin 33 := Fin.decodeProd slot
    sourceAssignment baseSourceWidth invocationCount base groupValue
        ((block baseSourceWidth invocationCount).source slot) =
      groupValue decoded.1 decoded.2 := by
  dsimp only [block]
  exact sourceAssignment_group baseSourceWidth invocationCount base groupValue
    (Fin.decodeProd slot).1 (Fin.decodeProd slot).2

end NightstreamFPrime.Layout.ProductionRelation.ProductRetainedBlock
