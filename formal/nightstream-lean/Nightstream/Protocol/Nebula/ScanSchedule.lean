import Mathlib.Data.List.FinRange
import Nightstream.Protocol.Nebula.ConcreteLaneGeometry
import Nightstream.Protocol.Nebula.GlobalFPrime
import Nightstream.Protocol.Nebula.SnapshotSlot

/-!
Contract: exact full-memory scan schedule for one V2 segment.

Assurance tier: protocol model.

Owns the proof that delayed F-prime consumption fixes claim step indexes in
strict order and that the 1,088-by-64 structural snapshot positions cover
each of the 69,632 global memory indexes exactly once.

Does not own snapshot values, snapshot boundary validity, circuit rows,
fingerprint accumulation, roots, or commitment binding.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ScanSchedule

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.FullClaim
open Nightstream.Protocol.Nebula.GlobalFPrime
open Nightstream.Protocol.Nebula.Lifecycle

def scanSlots : Nat := ConcreteLaneGeometry.scanSlots

theorem scanSlots_exact : scanSlots = 64 := by
  decide

theorem scan_capacity : claimsPerSegment * scanSlots = scannedCells := by
  decide

/-- One structural snapshot position. Both indexes are fixed by the verifier
key and are not witness-selected addresses. -/
structure Position where
  step : Fin claimsPerSegment
  slot : Fin scanSlots
deriving DecidableEq, Repr

@[ext]
theorem Position.ext
    {left right : Position}
    (step : left.step = right.step)
    (slot : left.slot = right.slot) :
    left = right := by
  cases left
  cases right
  simp_all

/-- The structural global address for one scan position. -/
def Position.globalIndex (position : Position) : Fin scannedCells :=
  ⟨position.step.val * scanSlots + position.slot.val, by
    have stepBound := position.step.isLt
    have slotBound := position.slot.isLt
    norm_num [claimsPerSegment, scanSlots, ConcreteLaneGeometry.scanSlots,
      scannedCells, romCells, ramCells] at stepBound slotBound ⊢
    omega⟩

/-- The unique scan position for one global memory index. -/
def positionOfIndex (index : Fin scannedCells) : Position :=
  { step := ⟨index.val / scanSlots, by
      apply (Nat.div_lt_iff_lt_mul (by decide : 0 < scanSlots)).2
      simpa only [scan_capacity] using index.isLt⟩
    slot := ⟨index.val % scanSlots, Nat.mod_lt _ (by decide)⟩ }

@[simp]
theorem globalIndex_positionOfIndex (index : Fin scannedCells) :
    (positionOfIndex index).globalIndex = index := by
  apply Fin.ext
  simp only [Position.globalIndex, positionOfIndex]
  simpa [Nat.mul_comm] using Nat.div_add_mod index.val scanSlots

@[simp]
theorem positionOfIndex_globalIndex (position : Position) :
    positionOfIndex position.globalIndex = position := by
  cases position with
  | mk step slot =>
      apply Position.ext <;> apply Fin.ext
      · simp only [positionOfIndex, Position.globalIndex]
        have slotBound := slot.isLt
        norm_num [scanSlots, ConcreteLaneGeometry.scanSlots] at slotBound ⊢
        omega
      · simp only [positionOfIndex, Position.globalIndex]
        have slotBound := slot.isLt
        norm_num [scanSlots, ConcreteLaneGeometry.scanSlots] at slotBound ⊢
        omega

/-- Structural scan addresses form a bijection. Therefore no address can be
omitted, repeated, or reordered through a witness-selected address field. -/
theorem globalIndex_bijective :
    Function.Bijective Position.globalIndex := by
  constructor
  · intro left right equal
    have decoded := congrArg positionOfIndex equal
    simpa only [positionOfIndex_globalIndex] using decoded
  · intro index
    exact ⟨positionOfIndex index, globalIndex_positionOfIndex index⟩

/-- Exact claim step at every list position in any run that starts active.
This theorem derives ordering from the transition relation. It does not use
the run length as an ordering assumption. -/
theorem verifiedRun_claim_step_at
    {schema : FullClaim.Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof →
      FullClaim.Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List
      (FullClaim.Verified schema Digest Challenge Products verify)}
    (run : FullClaim.VerifiedRun verify balanced before claims after)
    (active : ActiveCarry Digest Challenge Products)
    (beforeActive : before = .active active)
    (index : Fin claims.length) :
    (claims.get index).claim.memory.stepIndex.val =
      active.stepIndex.val + index.val := by
  induction run generalizing active with
  | nil =>
      exact Fin.elim0 index
  | @cons before middle after head tail step rest inductionHypothesis =>
      cases step.consumes with
      | @interior activeBefore claim agreement notLast =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · simpa using congrArg Fin.val agreement.stepIndex
          · have tailStep := inductionHypothesis
              (interiorCarry activeBefore head.claim.memory notLast) rfl
              tailIndex
            change
              (tail.get tailIndex).claim.memory.stepIndex.val =
                activeBefore.stepIndex.val + tailIndex.succ.val
            simp only [interiorCarry] at tailStep
            have successorValue : tailIndex.succ.val = tailIndex.val + 1 :=
              rfl
            rw [successorValue]
            omega
      | @close activeBefore claim agreement last checks =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          have tailEmpty := rest.from_closed_is_empty
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · simpa using congrArg Fin.val agreement.stepIndex
          · have noTail : tail.length = 0 := by simp [tailEmpty.1]
            have tailBound := tailIndex.isLt
            omega

/-- Every consumed claim in one active run carries the exact segment
timestamp bounds from the opening active carry. This is derived from the
transition chain; it is not a claim-list premise. -/
theorem verifiedRun_claim_segment_bounds_at
    {schema : FullClaim.Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof →
      FullClaim.Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {before after : Carry Digest Challenge Products}
    {claims : List
      (FullClaim.Verified schema Digest Challenge Products verify)}
    (run : FullClaim.VerifiedRun verify balanced before claims after)
    (active : ActiveCarry Digest Challenge Products)
    (beforeActive : before = .active active)
    (index : Fin claims.length) :
    (claims.get index).claim.memory.segmentStartTimestamp =
        active.segmentStartTimestamp ∧
      (claims.get index).claim.memory.segmentEndTimestamp =
        active.segmentEndTimestamp := by
  induction run generalizing active with
  | nil =>
      exact Fin.elim0 index
  | @cons before middle after head tail step rest inductionHypothesis =>
      cases step.consumes with
      | @interior activeBefore claim agreement notLast =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · exact ⟨agreement.segmentStartTimestamp,
              agreement.segmentEndTimestamp⟩
          · have tailBounds := inductionHypothesis
              (interiorCarry activeBefore head.claim.memory notLast) rfl
              tailIndex
            simpa [interiorCarry] using tailBounds
      | @close activeBefore claim agreement last checks =>
          have activeExact : activeBefore = active :=
            Carry.active.inj beforeActive
          subst active
          have tailEmpty := rest.from_closed_is_empty
          refine Fin.cases ?_ (fun tailIndex => ?_) index
          · exact ⟨agreement.segmentStartTimestamp,
              agreement.segmentEndTimestamp⟩
          · have noTail : tail.length = 0 := by simp [tailEmpty.1]
            have tailBound := tailIndex.isLt
            omega

/-- A complete segment has claim step indexes `0, 1, ..., 1087` at the exact
corresponding list positions. -/
theorem segment_claim_step_at
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before)
    (index : Fin run.claims.length) :
    (run.claims.get index).claim.memory.stepIndex.val = index.val := by
  have indexed := verifiedRun_claim_step_at run.consumed run.active rfl index
  rw [run.startsAtStepZero] at indexed
  omega

/-- Each list-position/slot pair in a complete segment selects the same global
index as its canonical scan position. -/
theorem segment_snapshot_global_index_at
    {ChallengeField : Type} [Field ChallengeField]
    {schema : FullClaim.Schema} {Digest : Type}
    {verify : V2Verifier schema Digest ChallengeField}
    {before : ClosedCarry Digest}
    (run : SegmentRun schema Digest verify before)
    (index : Fin run.claims.length)
    (slot : Fin scanSlots) :
    SnapshotSlot.globalIndex
        (run.claims.get index).claim.memory.stepIndex.val slot =
      index.val * scanSlots + slot.val := by
  rw [segment_claim_step_at run index]
  rfl

/-- Claim count alone is not an exact-cover condition. A list with the right
length can repeat step zero at every position. -/
def repeatedStepIndexes : List Nat :=
  List.replicate claimsPerSegment 0

theorem repeatedStepIndexes_has_exact_count :
    repeatedStepIndexes.length = claimsPerSegment := by
  simp [repeatedStepIndexes]

theorem repeatedStepIndexes_is_not_canonical :
    repeatedStepIndexes ≠ List.range claimsPerSegment := by
  intro equal
  have atOne := congrArg (fun values => values[1]?) equal
  norm_num [repeatedStepIndexes, claimsPerSegment] at atOne

end Nightstream.Protocol.Nebula.ScanSchedule
