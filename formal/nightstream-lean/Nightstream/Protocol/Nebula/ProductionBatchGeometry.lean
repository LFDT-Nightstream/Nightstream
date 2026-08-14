import Nightstream.Protocol.Nebula.ProductionProfileCandidates

/-!
Contract: exact successor batch indexing and lane geometry.

Each fresh claim owns one batch. Each batch owns `E` consecutive semantic
checked steps. This file proves that batch and within-batch indexes are in
bijection with all 1,088 segment steps and fixes the unoptimized complete lane
and delayed-suffix widths used for candidate compilation.

The lane geometry concatenates complete factor-one ring blocks. A later
repacking optimization needs a new proved layout and manifest.

Does not own row generation, semantic step validity, commitments, or a selected
candidate.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ProductionBatchGeometry

open Nightstream.Protocol.Nebula.ProductionProfileCandidates

abbrev SegmentStep := Fin stepsPerSegment

structure Location (candidate : Id) where
  batch : Fin (claimsPerSegment candidate)
  within : Fin (checkedStepsPerFreshClaim candidate)
deriving DecidableEq

/-- Row-major batch location to semantic checked-step index. -/
def encode (candidate : Id) (location : Location candidate) : SegmentStep :=
  ⟨location.batch.val * checkedStepsPerFreshClaim candidate +
      location.within.val,
    by
      have endBound := local_batch_end_le_segment candidate location.batch.isLt
      exact Nat.lt_of_lt_of_le
        (Nat.add_lt_add_left location.within.isLt
          (location.batch.val * checkedStepsPerFreshClaim candidate))
        endBound⟩

/-- Semantic checked-step index to its unique batch and within-batch index. -/
def decode (candidate : Id) (step : SegmentStep) : Location candidate where
  batch :=
    ⟨step.val / checkedStepsPerFreshClaim candidate, by
      cases candidate <;>
        simp [claimsPerSegment, checkedStepsPerFreshClaim, stepsPerSegment] <;>
        omega⟩
  within :=
    ⟨step.val % checkedStepsPerFreshClaim candidate,
      Nat.mod_lt _ (checkedSteps_positive candidate)⟩

theorem encode_decode (candidate : Id) (step : SegmentStep) :
    encode candidate (decode candidate step) = step := by
  apply Fin.ext
  simp only [encode, decode]
  simpa [Nat.mul_comm] using
    Nat.div_add_mod step.val (checkedStepsPerFreshClaim candidate)

theorem decode_encode (candidate : Id) (location : Location candidate) :
    decode candidate (encode candidate location) = location := by
  cases location with
  | mk batch within =>
      simp only [decode, encode]
      congr 1
      · apply Fin.ext
        change
          (batch.val * checkedStepsPerFreshClaim candidate + within.val) /
              checkedStepsPerFreshClaim candidate = batch.val
        rw [Nat.mul_comm batch.val (checkedStepsPerFreshClaim candidate),
          Nat.mul_add_div (checkedSteps_positive candidate),
          Nat.div_eq_of_lt within.isLt,
          Nat.add_zero]
      · apply Fin.ext
        change
          (batch.val * checkedStepsPerFreshClaim candidate + within.val) %
              checkedStepsPerFreshClaim candidate = within.val
        exact Nat.mul_add_mod_of_lt within.isLt

theorem encode_injective (candidate : Id) :
    Function.Injective (encode candidate) := by
  intro left right equal
  rw [← decode_encode candidate left, ← decode_encode candidate right, equal]

theorem encode_surjective (candidate : Id) :
    Function.Surjective (encode candidate) := by
  intro step
  exact ⟨decode candidate step, encode_decode candidate step⟩

/-- Exact cover: every semantic segment step has one and only one batch
location. -/
theorem encode_bijective (candidate : Id) :
    Function.Injective (encode candidate) /\
      Function.Surjective (encode candidate) :=
  ⟨encode_injective candidate, encode_surjective candidate⟩

/-! ## Complete concatenated lane geometry -/

def operationRingColumnsPerStep : Nat := 124
def initialSnapshotRingColumnsPerStep : Nat := 66
def finalSnapshotRingColumnsPerStep : Nat := 66
def totalRingColumnsPerStep : Nat := 256

def operationRingColumns (candidate : Id) : Nat :=
  checkedStepsPerFreshClaim candidate * operationRingColumnsPerStep

def initialSnapshotRingColumns (candidate : Id) : Nat :=
  checkedStepsPerFreshClaim candidate * initialSnapshotRingColumnsPerStep

def finalSnapshotRingColumns (candidate : Id) : Nat :=
  checkedStepsPerFreshClaim candidate * finalSnapshotRingColumnsPerStep

def totalRingColumns (candidate : Id) : Nat :=
  operationRingColumns candidate + initialSnapshotRingColumns candidate +
    finalSnapshotRingColumns candidate

def delayedMemorySuffixCoordinates (candidate : Id) : Nat :=
  checkedStepsPerFreshClaim candidate *
    memorySuffixCoordinatesPerCheckedStep

theorem totalRingColumns_eq (candidate : Id) :
    totalRingColumns candidate =
      checkedStepsPerFreshClaim candidate * totalRingColumnsPerStep := by
  cases candidate <;> decide

theorem candidate_geometry_table :
    totalRingColumns .e1 = 256 /\
      delayedMemorySuffixCoordinates .e1 = 192 /\
    totalRingColumns .e4 = 1024 /\
      delayedMemorySuffixCoordinates .e4 = 768 /\
    totalRingColumns .e8 = 2048 /\
      delayedMemorySuffixCoordinates .e8 = 1536 /\
    totalRingColumns .e16 = 4096 /\
      delayedMemorySuffixCoordinates .e16 = 3072 := by
  decide

end Nightstream.Protocol.Nebula.ProductionBatchGeometry
