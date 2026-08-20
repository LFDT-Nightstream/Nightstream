import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateMaps
import Nightstream.Implementation.R1CS.Core.SeededPhi81RingRefinement
import Nightstream.Protocol.Nebula.CompactCommit

/-!
Contract: authoritative three-map accumulator for the production PiCCS claim
frame.

Assurance tier: model-level.

Owns the exact statement/fresh, running-commitment, and running-public map
inputs, their fixed Rust seed schedules, one additive transition per claim
chunk, and completion after all 98 chunks.

Does not own sampler liveness, Rust row placement, physical link rows,
Module-SIS hardness, or lifecycle selection. Sampler liveness remains the
explicit `SetupValid` obligation and is not inferred from generated data.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
open Nightstream.Protocol.Nebula.ShiftedTernary41V1
open Nightstream.SuperNeo.Concrete

/-- The verifier-owned claim frame in exact physical field order. -/
abbrev ClaimFrame := Fin claimFrameLength → CanonicalGoldilocks

def outputWidth : Nat := 2 * SeededPhi81.dimension

theorem outputWidth_exact : outputWidth = 108 := by
  decide

abbrev Commitment := Fin outputWidth → F
abbrev State := MapKind → Commitment

/-- Canonical semantic placement of every shifted-ternary word. This is not a
physical artifact layout. It gives the compact seeded compiler one unique
column for each authoritative map-field digit. -/
def semanticWordStarts (kind : MapKind) : List Nat :=
  List.ofFn fun field : Fin kind.fieldCount => field.val * digitCount

def semanticBlock (kind : MapKind) : SeededPhi81.Block where
  rowStart := 0
  wordStarts := semanticWordStarts kind
  wordWidth := digitCount
  kappa := 2
  messageCols := kind.messageColumnCount
  outputColumns := List.range outputWidth
  superneoTransformedColumns := false
  schedule := kind.expectedSchedule

/-- Exact setup condition for all three fixed Rust sampler schedules. The
lifecycle relation does not discharge this condition by evaluation. -/
def SetupValid : Prop :=
  ∀ kind, (semanticBlock kind).Valid

def selectedDigit
    (frame : ClaimFrame) (kind : MapKind) (chunk : Fin claimChunkCount)
    (field : Fin kind.fieldCount) (digit : Fin digitCount) : Nat :=
  if kind.claimChunk field = chunk then
    fieldDigit (Nightstream.Protocol.Nebula.CompactCommit.tritAt
      (frame ⟨kind.framePosition field, kind.framePosition_lt field⟩) digit)
  else
    0

/-- Dense assignment read by the independent compact-map interpreter. Every
field outside the verifier-selected chunk has value zero. -/
def semanticAssignment
    (frame : ClaimFrame) (kind : MapKind) (chunk : Fin claimChunkCount)
    (column : Nat) : Nat :=
  if bound : column < kind.fieldCount * digitCount then
    selectedDigit frame kind chunk
      ⟨column / digitCount, by
        have positive : 0 < digitCount := by decide
        exact (Nat.div_lt_iff_lt_mul positive).2 (by
          simpa [Nat.mul_comm] using bound)⟩
      ⟨column % digitCount, Nat.mod_lt _ (by decide)⟩
  else
    0

def outputRow (output : Fin outputWidth) : Fin 2 :=
  ⟨output.val / SeededPhi81.dimension, by
    have bound := output.isLt
    unfold outputWidth at bound
    have positive : 0 < SeededPhi81.dimension := by decide
    exact (Nat.div_lt_iff_lt_mul positive).2 (by
      simpa [Nat.mul_comm] using bound)⟩

def outputCoordinate (output : Fin outputWidth) : Fin SeededPhi81.dimension :=
  ⟨output.val % SeededPhi81.dimension, Nat.mod_lt _ (by decide)⟩

/-- One exact map contribution selected by one verifier-owned claim chunk.
The coefficients come from the map's fixed Rust seed schedule. -/
def chunkContribution
    (frame : ClaimFrame) (kind : MapKind) (chunk : Fin claimChunkCount) :
    Commitment :=
  fun output => residueNat
    ((semanticBlock kind).linearValue
      (semanticAssignment frame kind chunk)
      (outputRow output).val (outputCoordinate output).val)

def zeroState : State := fun _ _ => 0

/-- All three map accumulators advance in every claim phase. An inactive map
has a zero partial contribution, so this equation also owns its exact carry. -/
def Step
    (frame : ClaimFrame) (chunk : Fin claimChunkCount)
    (before after : State) : Prop :=
  ∀ kind output,
    after kind output =
      before kind output + chunkContribution frame kind chunk output

def partialAtNat
    (frame : ClaimFrame) (index : Nat) : State :=
  if bound : index < claimChunkCount then
    fun kind => chunkContribution frame kind ⟨index, bound⟩
  else
    zeroState

/-- Canonical accumulator after the first `count` claim chunks. -/
def accumulated (frame : ClaimFrame) : Nat → State
  | 0 => zeroState
  | count + 1 => fun kind output =>
      accumulated frame count kind output +
        partialAtNat frame count kind output

theorem accumulated_zero (frame : ClaimFrame) :
    accumulated frame 0 = zeroState := by
  rfl

theorem accumulated_succ
    (frame : ClaimFrame) (chunk : Fin claimChunkCount) :
    Step frame chunk (accumulated frame chunk.val)
      (accumulated frame (chunk.val + 1)) := by
  intro kind output
  simp [accumulated, partialAtNat, chunk.isLt]

/-- Complete verifier-owned binding of all 99,520 mapped frame fields. -/
def completeBinding (frame : ClaimFrame) : State :=
  accumulated frame claimChunkCount

/-- One ordered state for all three maps across every claim phase. -/
structure AcceptedRun (frame : ClaimFrame) where
  state : Nat → State
  initial : state 0 = zeroState
  step : ∀ chunk : Fin claimChunkCount,
    Step frame chunk (state chunk.val) (state (chunk.val + 1))

namespace AcceptedRun

theorem state_eq_accumulated
    {frame : ClaimFrame} (run : AcceptedRun frame)
    (count : Nat) (bound : count ≤ claimChunkCount) :
    run.state count = accumulated frame count := by
  induction count with
  | zero =>
      rw [run.initial, accumulated_zero]
  | succ count inductionHypothesis =>
      have countBound : count < claimChunkCount := by omega
      let chunk : Fin claimChunkCount := ⟨count, countBound⟩
      funext kind output
      calc
        run.state (count + 1) kind output =
            run.state count kind output +
              chunkContribution frame kind chunk output :=
          run.step chunk kind output
        _ = accumulated frame count kind output +
              chunkContribution frame kind chunk output := by
          rw [inductionHypothesis (by omega)]
        _ = accumulated frame (count + 1) kind output := by
          rw [show chunkContribution frame kind chunk output =
              partialAtNat frame count kind output by
            simp [partialAtNat, chunk, countBound]]
          rfl

/-- A complete accepted chain cannot authorize only a self-consistent digest.
Its full three-map state equals the binding recomputed from the exact frame. -/
theorem final_eq_complete
    {frame : ClaimFrame} (run : AcceptedRun frame) :
    run.state claimChunkCount = completeBinding frame := by
  exact run.state_eq_accumulated claimChunkCount (by rfl)

end AcceptedRun

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
