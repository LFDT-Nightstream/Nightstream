import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataAccumulator

/-!
Contract: exact semantic layout of the production claim-replay transition
state.

Assurance tier: model-level.

Owns the Rust v6 order of the expected sponge, runtime sponge, two cursors,
and all three PiCCS metadata accumulators. It decodes one 688-word transition
as a 344-word before state followed by a 344-word after state.

Does not own generated column identity, row soundness, Poseidon2 execution,
coordinate-map execution, sampler liveness, or lifecycle selection.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete

def spongeWidth : Nat := 8
def spongeStateWordCount : Nat := spongeWidth + 1
def cursorWordCount : Nat := 2
def coordinateMapCount : Nat := 3

/-- One Rust v6 persistent claim-replay state. -/
def stateWordCount : Nat :=
  2 * spongeStateWordCount + cursorWordCount +
    coordinateMapCount * outputWidth

/-- Rust allocates the before state first and the after state second. -/
def transitionWordCount : Nat := 2 * stateWordCount

theorem exact_word_counts :
    spongeStateWordCount = 9 ∧
      stateWordCount = 344 ∧
      transitionWordCount = 688 := by
  decide

def expectedOffset : Nat := 0
def runtimeOffset : Nat := spongeStateWordCount
def frameCursorOffset : Nat := 2 * spongeStateWordCount
def programCursorOffset : Nat := frameCursorOffset + 1
def coordinateOffset : Nat := frameCursorOffset + cursorWordCount

/-- Exact Rust order of the three coordinate accumulators. -/
def mapOffset : MapKind → Nat
  | .statementFresh => coordinateOffset
  | .runningCommitments => coordinateOffset + outputWidth
  | .runningPublic => coordinateOffset + 2 * outputWidth

theorem exact_offsets :
    runtimeOffset = 9 ∧
      frameCursorOffset = 18 ∧
      programCursorOffset = 19 ∧
      coordinateOffset = 20 ∧
      mapOffset .statementFresh = 20 ∧
      mapOffset .runningCommitments = 128 ∧
      mapOffset .runningPublic = 236 := by
  decide

def expectedLaneIndex (lane : Fin spongeWidth) : Fin stateWordCount :=
  ⟨expectedOffset + lane.val, by
    have bound := lane.isLt
    unfold expectedOffset spongeWidth stateWordCount spongeStateWordCount
      cursorWordCount coordinateMapCount outputWidth
    omega⟩

def expectedAbsorbedIndex : Fin stateWordCount :=
  ⟨expectedOffset + spongeWidth, by
    decide⟩

def runtimeLaneIndex (lane : Fin spongeWidth) : Fin stateWordCount :=
  ⟨runtimeOffset + lane.val, by
    have bound := lane.isLt
    unfold runtimeOffset spongeWidth stateWordCount spongeStateWordCount
      cursorWordCount coordinateMapCount outputWidth
    omega⟩

def runtimeAbsorbedIndex : Fin stateWordCount :=
  ⟨runtimeOffset + spongeWidth, by
    decide⟩

def frameCursorIndex : Fin stateWordCount :=
  ⟨frameCursorOffset, by decide⟩

def programCursorIndex : Fin stateWordCount :=
  ⟨programCursorOffset, by decide⟩

def coordinateIndex
    (kind : MapKind) (output : Fin outputWidth) : Fin stateWordCount :=
  ⟨mapOffset kind + output.val, by
    have bound := output.isLt
    cases kind with
    | statementFresh =>
        simp only [mapOffset]
        norm_num [coordinateOffset, frameCursorOffset, stateWordCount,
          spongeStateWordCount, spongeWidth, cursorWordCount,
          coordinateMapCount, outputWidth, SeededPhi81.dimension,
          SeededPhi81Sampler.dimension] at *
        omega
    | runningCommitments =>
        simp only [mapOffset]
        norm_num [coordinateOffset, frameCursorOffset, stateWordCount,
          spongeStateWordCount, spongeWidth, cursorWordCount,
          coordinateMapCount, outputWidth, SeededPhi81.dimension,
          SeededPhi81Sampler.dimension] at *
        omega
    | runningPublic =>
        simp only [mapOffset]
        norm_num [coordinateOffset, frameCursorOffset, stateWordCount,
          spongeStateWordCount, spongeWidth, cursorWordCount,
          coordinateMapCount, outputWidth, SeededPhi81.dimension,
          SeededPhi81Sampler.dimension] at *
        omega⟩

structure SpongeState where
  lanes : Fin spongeWidth → F
  absorbed : F

/-- Semantic meaning of all 344 words on one transition side. -/
structure PersistentState where
  expected : SpongeState
  runtime : SpongeState
  frameCursor : F
  programCursor : F
  coordinates : State

/-- Decode the exact Rust v6 local word order. -/
def decodeState (words : Fin stateWordCount → F) : PersistentState where
  expected := {
    lanes := fun lane => words (expectedLaneIndex lane)
    absorbed := words expectedAbsorbedIndex }
  runtime := {
    lanes := fun lane => words (runtimeLaneIndex lane)
    absorbed := words runtimeAbsorbedIndex }
  frameCursor := words frameCursorIndex
  programCursor := words programCursorIndex
  coordinates := fun kind output => words (coordinateIndex kind output)

inductive Side where
  | before
  | after
deriving DecidableEq, Repr

def sideOffset : Side → Nat
  | .before => 0
  | .after => stateWordCount

def transitionIndex
    (side : Side) (word : Fin stateWordCount) : Fin transitionWordCount :=
  ⟨sideOffset side + word.val, by
    have bound := word.isLt
    cases side with
    | before =>
        simp only [sideOffset]
        unfold transitionWordCount
        omega
    | after =>
        simp only [sideOffset]
        unfold transitionWordCount
        omega⟩

def sideWords
    (words : Fin transitionWordCount → F) (side : Side) :
    Fin stateWordCount → F :=
  fun word => words (transitionIndex side word)

structure Transition where
  before : PersistentState
  after : PersistentState

/-- Decode all 688 words as the exact before/after pair emitted by Rust. -/
def decodeTransition (words : Fin transitionWordCount → F) : Transition where
  before := decodeState (sideWords words .before)
  after := decodeState (sideWords words .after)

@[simp] theorem decodeState_expected_lane
    (words : Fin stateWordCount → F) (lane : Fin spongeWidth) :
    (decodeState words).expected.lanes lane = words (expectedLaneIndex lane) := by
  rfl

@[simp] theorem decodeState_runtime_lane
    (words : Fin stateWordCount → F) (lane : Fin spongeWidth) :
    (decodeState words).runtime.lanes lane = words (runtimeLaneIndex lane) := by
  rfl

@[simp] theorem decodeState_coordinate
    (words : Fin stateWordCount → F) (kind : MapKind)
    (output : Fin outputWidth) :
    (decodeState words).coordinates kind output =
      words (coordinateIndex kind output) := by
  rfl

@[simp] theorem transitionIndex_before
    (word : Fin stateWordCount) :
    (transitionIndex .before word).val = word.val := by
  simp [transitionIndex, sideOffset]

@[simp] theorem transitionIndex_after
    (word : Fin stateWordCount) :
    (transitionIndex .after word).val = stateWordCount + word.val := by
  simp [transitionIndex, sideOffset]

end Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState
