import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding

/-!
Contract: one bounded streaming pass uses the same field chunk for Poseidon2
state binding and for phase-local algebra.

Assurance tier: model-level exact refinement and cryptographic-reduction
boundary.

Owns a compact continuation machine, exact equivalence between chunked and
monolithic accumulation, and the reduction from a different accepted frame
to one named Poseidon2 replay collision.

Does not own a specific F-prime accumulator, generated rows, Rust refinement,
Poseidon2 security, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionStreamingFusedPass

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.R1CS.Canonical

abbrev State := ProductPoseidon2.State

/-- The only generic state that persists between chunks. A concrete phase
must give its accumulator a fixed, small field encoding. -/
structure Runtime (Accumulator : Type) where
  transcript : State
  cursor : Nat
  accumulator : Accumulator

def initial {Accumulator : Type} (accumulator : Accumulator) :
    Runtime Accumulator where
  transcript := ProductPoseidon2.initialState
  cursor := 0
  accumulator

/-- One exact input chunk drives both consumers. There is no independent
digest chunk and algebra chunk. -/
noncomputable def Runtime.advance
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator)
    (runtime : Runtime Accumulator) (chunk : List Nat) :
    Runtime Accumulator where
  transcript := Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
    chunk runtime.transcript
  cursor := runtime.cursor + chunk.length
  accumulator := chunk.foldl consume runtime.accumulator

noncomputable def run
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator) :
    List (List Nat) -> Runtime Accumulator -> Runtime Accumulator
  | [], runtime => runtime
  | chunk :: rest, runtime => run consume rest (runtime.advance consume chunk)

theorem run_transcript
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator)
    (chunks : List (List Nat)) (runtime : Runtime Accumulator) :
    (run consume chunks runtime).transcript =
      absorbChunks chunks runtime.transcript := by
  induction chunks generalizing runtime with
  | nil => rfl
  | cons chunk rest inductionHypothesis =>
      exact inductionHypothesis (runtime.advance consume chunk)

theorem run_cursor
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator)
    (chunks : List (List Nat)) (runtime : Runtime Accumulator) :
    (run consume chunks runtime).cursor =
      runtime.cursor + chunks.flatten.length := by
  induction chunks generalizing runtime with
  | nil => simp [run]
  | cons chunk rest inductionHypothesis =>
      rw [run, inductionHypothesis]
      simp only [Runtime.advance, List.flatten_cons, List.length_append]
      omega

/-- Chunk boundaries do not change the phase-local fold. -/
theorem run_accumulator
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator)
    (chunks : List (List Nat)) (runtime : Runtime Accumulator) :
    (run consume chunks runtime).accumulator =
      chunks.flatten.foldl consume runtime.accumulator := by
  induction chunks generalizing runtime with
  | nil => rfl
  | cons chunk rest inductionHypothesis =>
      rw [run, inductionHypothesis]
      simp only [Runtime.advance, List.flatten_cons, List.foldl_append]

/-- A deterministic schedule gives the exact monolithic transcript, cursor,
and algebra result. This theorem is functional equality; it uses no
cryptographic assumption. -/
theorem run_schedule_exact
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator)
    {width : Nat} {values : List Nat} {chunks : List (List Nat)}
    (schedule : ChunkSchedule width values chunks)
    (runtime : Runtime Accumulator)
    (normalized : runtime.transcript.absorbed < Poseidon2Sponge.rate) :
    (run consume chunks runtime).transcript =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants values
          runtime.transcript /\
      (run consume chunks runtime).cursor =
        runtime.cursor + values.length /\
      (run consume chunks runtime).accumulator =
        values.foldl consume runtime.accumulator := by
  constructor
  · rw [run_transcript,
      absorbChunks_eq_absorbSlice_flatten chunks runtime.transcript normalized,
      schedule.flatten_eq]
  constructor
  · rw [run_cursor, schedule.flatten_eq]
  · rw [run_accumulator, schedule.flatten_eq]

/-- Exact failure event for a supplied frame that differs from the
authoritative frame but reaches the same Poseidon2 state from one specified
carried state. -/
def FrameReplayCollisionAt (prior : State) (authoritative : List Nat) : Prop :=
  exists supplied : List Nat,
    supplied ≠ authoritative /\
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants supplied
          prior =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants authoritative
          prior

/-- Initial-state specialization used by complete-frame replay. -/
abbrev FrameReplayCollision (authoritative : List Nat) : Prop :=
  FrameReplayCollisionAt ProductPoseidon2.initialState authoritative

/-- If the final transcript equals the authoritative transcript, the fused
algebra used the authoritative frame, or the execution exposes one named
Poseidon2 collision from the carried prior state. -/
theorem accepted_run_recovers_fold_or_collision_at
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator)
    {width : Nat} {authoritative supplied : List Nat}
    {chunks : List (List Nat)} (runtime : Runtime Accumulator)
    (schedule : ChunkSchedule width supplied chunks)
    (normalized : runtime.transcript.absorbed < Poseidon2Sponge.rate)
    (transcriptExact :
      (run consume chunks runtime).transcript =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants authoritative
          runtime.transcript) :
    (run consume chunks runtime).accumulator =
        authoritative.foldl consume runtime.accumulator \/
      FrameReplayCollisionAt runtime.transcript authoritative := by
  have suppliedTranscript :
      (run consume chunks runtime).transcript =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants supplied
          runtime.transcript :=
    (run_schedule_exact consume schedule runtime normalized).1
  by_cases exactFrame : supplied = authoritative
  · left
    rw [run_accumulator, schedule.flatten_eq, exactFrame]
  · right
    exact ⟨supplied, exactFrame, suppliedTranscript.symm.trans transcriptExact⟩

/-- If the final transcript equals the authoritative transcript, the fused
algebra used the authoritative frame, or the execution exposes the named
Poseidon2 collision. This is the exact security boundary needed before rows
can replace a monolithic frame. -/
theorem accepted_run_recovers_fold_or_collision
    {Accumulator : Type}
    (consume : Accumulator -> Nat -> Accumulator)
    (start : Accumulator)
    {width : Nat} {authoritative supplied : List Nat}
    {chunks : List (List Nat)}
    (schedule : ChunkSchedule width supplied chunks)
    (transcriptExact :
      (run consume chunks (initial start)).transcript =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants authoritative
          ProductPoseidon2.initialState) :
    (run consume chunks (initial start)).accumulator =
        authoritative.foldl consume start \/
      FrameReplayCollision authoritative := by
  apply accepted_run_recovers_fold_or_collision_at consume
    (initial start) schedule
  · change ProductPoseidon2.initialState.absorbed < Poseidon2Sponge.rate
    decide
  · exact transcriptExact

/-- Canonical generic continuation width. Concrete accumulators add only
their explicit encoding to the ten Poseidon2-and-cursor fields. -/
def persistentFields
    {Accumulator : Type}
    (encodeAccumulator : Accumulator -> List Nat)
    (runtime : Runtime Accumulator) : List Nat :=
  duplexStateFields runtime.transcript ++ [runtime.cursor] ++
    encodeAccumulator runtime.accumulator

theorem persistentFields_length
    {Accumulator : Type}
    (encodeAccumulator : Accumulator -> List Nat)
    (runtime : Runtime Accumulator) :
    (persistentFields encodeAccumulator runtime).length =
      10 + (encodeAccumulator runtime.accumulator).length := by
  simp [persistentFields]
  omega

end Nightstream.Implementation.Nebula.ProductionStreamingFusedPass
