import Batteries.Data.Fin.Coding
import NightstreamFPrime.Layout.LowNormBlock
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedSlots

/-!
Owns the compact retained-slot block for any ordered list of canonical
Poseidon2 invocations. Slot order is invocation-major and then direct S-box
row order. Each invocation contributes exactly 86 general-field slots.

The caller owns only the invocation witness starts and proves that each full
592-column local interval is inside the source assignment.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedBlock

open NightstreamFPrime.Layout

/-- Compact field-slot block for one ordered invocation schedule. -/
def block (sourceWidth invocationCount : Nat)
    (witnessStart : Fin invocationCount → Nat)
    (witnessBound : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤
        sourceWidth) : LowNormBlock.Block sourceWidth where
  kind := .field
  slotCount := invocationCount * PoseidonRetainedSlots.rows.length
  source := fun slot =>
    let indices : Fin invocationCount × Fin PoseidonRetainedSlots.rows.length :=
      Fin.decodeProd slot
    ⟨witnessStart indices.1 +
        (PoseidonRetainedSlots.localOutput indices.2).val, by
      have invocationBound := witnessBound indices.1
      have localBound := (PoseidonRetainedSlots.localOutput indices.2).isLt
      omega⟩

@[simp] theorem block_slotCount (sourceWidth invocationCount : Nat)
    (witnessStart : Fin invocationCount → Nat)
    (witnessBound : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤
        sourceWidth) :
    (block sourceWidth invocationCount witnessStart witnessBound).slotCount =
      invocationCount * 86 := by
  simp [block]

/-- Exact direct low-norm coordinate count of an invocation block. -/
theorem block_coordinateCount (sourceWidth invocationCount : Nat)
    (witnessStart : Fin invocationCount → Nat)
    (witnessBound : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤
        sourceWidth) :
    (block sourceWidth invocationCount witnessStart
      witnessBound).coordinateCount = invocationCount * 86 * 41 := by
  simp [LowNormBlock.Block.coordinateCount, block,
    LowNormSlot.Kind.width, BalancedTernary.width]

/-- The compact source function uses the exact invocation witness start and
the exact template-local retained S-box output. -/
theorem block_source (sourceWidth invocationCount : Nat)
    (witnessStart : Fin invocationCount → Nat)
    (witnessBound : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤
        sourceWidth)
    (slot : Fin (invocationCount * PoseidonRetainedSlots.rows.length)) :
    let indices : Fin invocationCount × Fin PoseidonRetainedSlots.rows.length :=
      Fin.decodeProd slot
    ((block sourceWidth invocationCount witnessStart witnessBound).source
      slot).val = witnessStart indices.1 +
        (PoseidonRetainedSlots.localOutput indices.2).val := by
  rfl

end NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedBlock
