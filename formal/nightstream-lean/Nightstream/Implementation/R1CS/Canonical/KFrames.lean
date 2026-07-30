import Nightstream.Implementation.R1CS.Canonical.KHorner

/-!
Contract: the canonical column allocator for `K`-multiplication frames.

Owns: the placement of each Horner step's three auxiliary columns, the proof
that distinct steps never collide, and the exact allocation list.

Does not own: soundness (`KMul`, `KHorner` — neither needs disjointness), the
projection identity, or any NIFS structure.

## Why this comes before honest completeness

`KHorner.hornerRows_sound` deliberately assumes nothing about frames: each
step's `outLow_sound` reads only that step's own rows, so overlapping frames
would over-constrain the system without breaking soundness.

Honest completeness is the opposite. Building a witness means *writing* a value
to every frame column, and if two steps shared a column the witness would have
to write two different products there. So completeness, ownership and
conservation all need disjointness, and it has to be established before any of
them — which is why the allocator is its own module rather than an afterthought
inside the recipe.

## Layout

Step `s` takes columns `base + 3s`, `base + 3s + 1`, `base + 3s + 2`, in the
order `KMul.Frame` declares them. Consecutive and gapless, so `count`
multiplications occupy exactly `[base, base + 3·count)` and the allocation is a
contiguous block that a caller can place with one number.

This mirrors `Poseidon2Layout`'s S-box frames, which take four consecutive
columns from an auxiliary base. The width differs because `KMul` allocates
three products where an S-box allocates four chain values.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KFrames

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul

/-- Columns one `K` multiplication allocates. -/
def columnsPerFrame : Nat := 3

/-- The `slot`-th column of step `step`. -/
def frameColumn (base step slot : Nat) : Nat := base + columnsPerFrame * step + slot

/-- The frame for one Horner step. -/
def frameAt (base step : Nat) : Frame where
  lowLow := frameColumn base step 0
  highHigh := frameColumn base step 1
  cross := frameColumn base step 2

/-- The allocator, as a function of the step index. -/
def frames (base : Nat) : Nat → Frame := frameAt base

/-! ## Disjointness -/

/-- **Distinct steps never share a column.**  The bound on slots is what makes
this true: without `slot < 3` the blocks would overlap. -/
theorem frameColumn_step_disjoint
    (base step slot otherStep otherSlot : Nat)
    (slotLt : slot < columnsPerFrame) (otherSlotLt : otherSlot < columnsPerFrame)
    (distinct : step ≠ otherStep) :
    frameColumn base step slot ≠ frameColumn base otherStep otherSlot := by
  unfold frameColumn columnsPerFrame at *
  omega

/-- Distinct slots of the same step are distinct columns. -/
theorem frameColumn_slot_disjoint
    (base step slot otherSlot : Nat) (distinct : slot ≠ otherSlot) :
    frameColumn base step slot ≠ frameColumn base step otherSlot := by
  unfold frameColumn
  omega

/-- Every column of a frame is one of its three slots — so bounding the slots
bounds the frame. -/
theorem frameAt_slots (base step : Nat) :
    (frameAt base step).lowLow = frameColumn base step 0
      ∧ (frameAt base step).highHigh = frameColumn base step 1
      ∧ (frameAt base step).cross = frameColumn base step 2 :=
  ⟨rfl, rfl, rfl⟩

/-! ## The allocation list

Consecutive and gapless, so it is exactly `[base, base + 3·count)`. Writing it
as a mapped range rather than a `flatMap` over steps is what makes `Nodup`
provable without Mathlib. -/

def frameColumns (base count : Nat) : List Nat :=
  (List.range (columnsPerFrame * count)).map (fun offset => base + offset)

theorem frameColumns_length (base count : Nat) :
    (frameColumns base count).length = 3 * count := by
  unfold frameColumns columnsPerFrame
  rw [List.length_map, List.length_range]

/-- **No column is allocated twice.**  This is the exact-column-ownership
obligation for the evaluation program. -/
theorem frameColumns_nodup (base count : Nat) :
    (frameColumns base count).Nodup := by
  unfold frameColumns
  refine nodup_map _ _ (fun a b equal => by omega) (List.nodup_range)

/-- Every allocated column lies in the declared block, so a caller placing the
allocator at `base` knows exactly what it consumes. -/
theorem frameColumns_mem_iff (base count column : Nat) :
    column ∈ frameColumns base count
      ↔ base ≤ column ∧ column < base + 3 * count := by
  unfold frameColumns columnsPerFrame
  constructor
  · intro member
    rcases List.mem_map.1 member with ⟨offset, inRange, image⟩
    have bound := List.mem_range.1 inRange
    omega
  · intro ⟨lower, upper⟩
    exact List.mem_map.2 ⟨column - base, List.mem_range.2 (by omega), by omega⟩

/-- Each step's own columns are inside the block, provided the step is one of
the `count` the block was sized for. -/
theorem frameAt_columns_mem
    (base count step slot : Nat)
    (stepLt : step < count) (slotLt : slot < columnsPerFrame) :
    frameColumn base step slot ∈ frameColumns base count := by
  rw [frameColumns_mem_iff]
  unfold frameColumn columnsPerFrame at *
  omega

end Nightstream.Implementation.R1CS.Canonical.KFrames
