import Nightstream.Implementation.Nebula.Production.Carrier.StreamingClaimReplayState
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFusedPass

/-!
Contract: authoritative semantic transition for one production claim-replay
chunk.

Assurance tier: model-level.

Owns one verifier-selected claim chunk, its Poseidon2 replay, exact frame and
program cursors, and the simultaneous update of all three PiCCS metadata
accumulators. A complete run starts from the empty replay and zero map state
and ends with the expected replay state and complete three-map binding.

Does not own generated rows, Rust column identity, supplied-frame collision
reduction, sampler liveness, state-digest rows, or lifecycle selection.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayTransition

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
open Nightstream.Protocol.Nebula.ShiftedTernary41V1
open Nightstream.SuperNeo.Concrete

abbrev ClaimFrame :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame

/-- The verifier program places claim replay after the prelude and all prior
state chunks. -/
def claimProgramStart : Nat := 1 + productionConfig.priorStateChunks

theorem claimProgramStart_exact : claimProgramStart = 95 := by
  decide

def chunkFrameIndex
    (chunk : Fin claimChunkCount)
    (offset : Fin (claimChunkFieldCount chunk)) : Fin claimFrameLength :=
  ⟨chunk.val * claimChunkWidth + offset.val, by
    have chunkBound := chunk.isLt
    have offsetBound := offset.isLt
    unfold claimChunkFieldCount at offsetBound
    split at offsetBound
    · unfold claimChunkCount claimChunkWidth claimFrameLength at *
      omega
    · unfold claimChunkCount claimChunkWidth claimFrameLength at *
      omega⟩

/-- Exact active frame values for one claim phase. The final list has 575
values; every prior list has 1,024 values. -/
def chunkValues
    (frame : ClaimFrame) (chunk : Fin claimChunkCount) : List Nat :=
  List.ofFn fun offset => (frame (chunkFrameIndex chunk offset)).val

@[simp] theorem chunkValues_length
    (frame : ClaimFrame) (chunk : Fin claimChunkCount) :
    (chunkValues frame chunk).length = claimChunkFieldCount chunk := by
  simp [chunkValues]

def duplexLaneIndex
    (lane : Fin Poseidon2Core.width) : Fin spongeWidth :=
  ⟨lane.val, by
    have bound := lane.isLt
    simpa [Poseidon2Core.width, spongeWidth] using bound⟩

/-- Canonical natural-residue view used by the existing Poseidon2 model. -/
def spongeToDuplex
    (state : ProductionStreamingClaimReplayState.SpongeState) :
    ProductPoseidon2.State where
  lanes := fun lane => (state.lanes (duplexLaneIndex lane)).val
  absorbed := state.absorbed.val

/-- One exact verifier-owned transition. The same `frame` and `chunk` feed
both `runtimeReplay` and `coordinateStep`. -/
structure PhaseStep
    (frame : ClaimFrame) (chunk : Fin claimChunkCount)
    (before after : PersistentState) : Prop where
  expectedCarry : after.expected = before.expected
  expectedAbsorbed :
    before.expected.absorbed =
      residueNat (claimFrameLength % Poseidon2Sponge.rate)
  runtimeAbsorbedZero : before.runtime.absorbed = 0
  runtimeReplay :
    spongeToDuplex after.runtime =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (chunkValues frame chunk) (spongeToDuplex before.runtime)
  frameCursorBefore :
    before.frameCursor = residueNat (chunk.val * claimChunkWidth)
  frameCursorAfter :
    after.frameCursor = residueNat
      (chunk.val * claimChunkWidth + claimChunkFieldCount chunk)
  programCursorBefore :
    before.programCursor = residueNat (claimProgramStart + chunk.val)
  programCursorAfter :
    after.programCursor = residueNat (claimProgramStart + chunk.val + 1)
  coordinateStep :
    ProductionStreamingPiCcsMetadataAccumulator.Step
      frame chunk before.coordinates after.coordinates
  initial : chunk.val = 0 →
    spongeToDuplex before.runtime = ProductPoseidon2.initialState ∧
      before.coordinates = zeroState
  final : chunk.val + 1 = claimChunkCount →
    spongeToDuplex after.runtime = spongeToDuplex after.expected

noncomputable def replayed (frame : ClaimFrame) : Nat → ProductPoseidon2.State
  | 0 => ProductPoseidon2.initialState
  | count + 1 =>
      if bound : count < claimChunkCount then
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (chunkValues frame ⟨count, bound⟩) (replayed frame count)
      else
        replayed frame count

theorem replayed_succ
    (frame : ClaimFrame) (chunk : Fin claimChunkCount) :
    replayed frame (chunk.val + 1) =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (chunkValues frame chunk) (replayed frame chunk.val) := by
  simp [replayed, chunk.isLt]

/-- One ordered state chain for all 98 verifier-owned claim chunks. -/
structure AcceptedRun (frame : ClaimFrame) where
  state : Nat → PersistentState
  step : ∀ chunk : Fin claimChunkCount,
    PhaseStep frame chunk (state chunk.val) (state (chunk.val + 1))

namespace AcceptedRun

theorem runtime_eq_replayed
    {frame : ClaimFrame} (run : AcceptedRun frame)
    (count : Nat) (bound : count ≤ claimChunkCount) :
    spongeToDuplex (run.state count).runtime = replayed frame count := by
  induction count with
  | zero =>
      have first := (run.step ⟨0, by decide⟩).initial rfl
      exact first.1
  | succ count inductionHypothesis =>
      have countBound : count < claimChunkCount := by omega
      let chunk : Fin claimChunkCount := ⟨count, countBound⟩
      calc
        spongeToDuplex (run.state (count + 1)).runtime =
            Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
              (chunkValues frame chunk)
              (spongeToDuplex (run.state count).runtime) :=
          (run.step chunk).runtimeReplay
        _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
              (chunkValues frame chunk) (replayed frame count) := by
          rw [inductionHypothesis (by omega)]
        _ = replayed frame (count + 1) := by
          exact (replayed_succ frame chunk).symm

private def coordinateRun
    {frame : ClaimFrame} (run : AcceptedRun frame) :
    ProductionStreamingPiCcsMetadataAccumulator.AcceptedRun frame where
  state := fun count => (run.state count).coordinates
  initial := ((run.step ⟨0, by decide⟩).initial rfl).2
  step := fun chunk => (run.step chunk).coordinateStep

/-- All three final accumulator values are recomputed from the exact frame. -/
theorem final_coordinates_eq_complete
    {frame : ClaimFrame} (run : AcceptedRun frame) :
    (run.state claimChunkCount).coordinates = completeBinding frame := by
  exact (coordinateRun run).final_eq_complete

/-- Final readiness binds the carried expected state to replay of the exact
99,903-field frame. -/
theorem final_expected_eq_replayed
    {frame : ClaimFrame} (run : AcceptedRun frame) :
    spongeToDuplex (run.state claimChunkCount).expected =
      replayed frame claimChunkCount := by
  have finalReady := (run.step ⟨97, by decide⟩).final (by decide)
  have runtimeExact := run.runtime_eq_replayed claimChunkCount (by rfl)
  simpa [claimChunkCount] using finalReady.symm.trans runtimeExact

end AcceptedRun

end Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayTransition
