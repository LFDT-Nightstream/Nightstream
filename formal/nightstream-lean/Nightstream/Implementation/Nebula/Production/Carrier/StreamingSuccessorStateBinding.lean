import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.SuccessorStateBinding

/-!
Contract: bounded-chunk replay of the exact recursive successor-state hashes.

Assurance tier: model-level transcript refinement and cryptographic-reduction
boundary.

Owns the exact prior-state and challenge-independent successor-prefix frames,
their deterministic 1,024-field schedules, the ten-field continuation state,
and equality with the current monolithic Poseidon2 definitions.

Does not own generated rows, phase-local source placement, Rust refinement,
Poseidon2 security, or the final recursive relation geometry.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId

/-! ## Exact frames -/

/-- The prefix that must be fixed before the memory challenge. It contains the
verifier-owned statement frame and every challenge-independent successor
field, but it omits both memory carries. -/
noncomputable def prefixFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.PreCarryValue candidate
      fullShape) : List Nat :=
  ProductPoseidon2.statementIdentifierFields statementId ++
    ProductionSuccessorStateBinding.preCarryFrame value

theorem prefixFrame_lengthFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.PreCarryValue candidate
      fullShape) :
    (prefixFrame statementId value).length =
      ProductNifsCodec.runningFieldCountFor fullShape.rowVariables + 544 := by
  simp [prefixFrame, ProductPoseidon2.statementIdentifierFields,
    ProductPoseidon2.proofPrefixFields_length,
    ProductionSuccessorStateBinding.preCarryFrame_lengthFor contract]
  omega

theorem prefixFrame_length_r26
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.PreCarryValue candidate
      fullShape) :
    (prefixFrame statementId value).length = 95636 := by
  rw [prefixFrame_lengthFor contract.toShape statementId value,
    contract.rowVariablesExact]
  decide

theorem stateFrame_length_r26
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape) :
    (ProductionSuccessorStateBinding.stateFrame statementId value).length =
      95754 := by
  rw [ProductionSuccessorStateBinding.stateFrame_lengthFor contract.toShape
      statementId value,
    contract.rowVariablesExact]
  decide

/-! ## Compact continuation machine -/

/-- Only the normalized Poseidon2 state and exact frame cursor persist between
hash phases. The frame and schedule are verifier-owned semantic data. -/
structure ReplayState where
  transcript : State
  cursor : Nat

def initialReplayState : ReplayState where
  transcript := ProductPoseidon2.initialState
  cursor := 0

noncomputable def ReplayState.advance
    (runtime : ReplayState) (chunk : List Nat) : ReplayState where
  transcript := Poseidon2Duplex.absorbSlice ProductPoseidon2.constants chunk
    runtime.transcript
  cursor := runtime.cursor + chunk.length

/-- Two field equalities reconstruct one exact replay-state advance without
inspecting the chunk. -/
theorem ReplayState.eq_advance_of_transcript_cursor
    (before after : ReplayState) (chunk : List Nat)
    (transcript : after.transcript =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants chunk
        before.transcript)
    (cursor : after.cursor = before.cursor + chunk.length) :
    after = before.advance chunk := by
  cases before with
  | mk beforeTranscript beforeCursor =>
      cases after with
      | mk afterTranscript afterCursor =>
          simp only [ReplayState.advance] at transcript cursor ⊢
          cases transcript
          cases cursor
          rfl

noncomputable def replayChunks :
    List (List Nat) -> ReplayState -> ReplayState
  | [], runtime => runtime
  | chunk :: rest, runtime => replayChunks rest (runtime.advance chunk)

theorem replayChunks_transcript
    (chunks : List (List Nat)) (runtime : ReplayState) :
    (replayChunks chunks runtime).transcript =
      absorbChunks chunks runtime.transcript := by
  induction chunks generalizing runtime with
  | nil => rfl
  | cons chunk rest inductionHypothesis =>
      exact inductionHypothesis (runtime.advance chunk)

theorem replayChunks_cursor
    (chunks : List (List Nat)) (runtime : ReplayState) :
    (replayChunks chunks runtime).cursor =
      runtime.cursor + chunks.flatten.length := by
  induction chunks generalizing runtime with
  | nil => simp [replayChunks]
  | cons chunk rest inductionHypothesis =>
      rw [replayChunks, inductionHypothesis]
      simp only [ReplayState.advance, List.flatten_cons, List.length_append]
      omega

/-- Canonical field serialization of one continuation state. -/
def persistentFields (runtime : ReplayState) : List Nat :=
  duplexStateFields runtime.transcript ++ [runtime.cursor]

@[simp] theorem persistentFields_length (runtime : ReplayState) :
    (persistentFields runtime).length = 10 := by
  simp [persistentFields]

/-! ## Exact equivalence with the current monolithic hashes -/

private theorem absorbList_append
    (left right : List Nat) (state : State) :
    Poseidon2Duplex.absorbList ProductPoseidon2.constants (left ++ right)
        state =
      Poseidon2Duplex.absorbList ProductPoseidon2.constants right
        (Poseidon2Duplex.absorbList ProductPoseidon2.constants left state) := by
  induction left generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.cons_append, Poseidon2Duplex.absorbList]
      exact inductionHypothesis _

private theorem gate_guarded (state : State) :
    Poseidon2Duplex.gate ProductPoseidon2.constants
        (Poseidon2Duplex.guarded ProductPoseidon2.constants state) =
      Poseidon2Duplex.gate ProductPoseidon2.constants state := by
  unfold Poseidon2Duplex.gate
  rw [Poseidon2Duplex.absorbElem_guarded]

theorem preCarryState_replays_prefixFrame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.PreCarryValue candidate
      fullShape) :
    ProductionSuccessorStateBinding.preCarryState statementId value =
      Poseidon2Duplex.gate ProductPoseidon2.constants
        (Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (prefixFrame statementId value) ProductPoseidon2.initialState) := by
  rw [prefixFrame, absorbList_append]
  rfl

/-- Every deterministic chunking of the exact prefix gives the same memory
challenge authority as the current monolithic definition. -/
theorem scheduled_preCarryState_eq
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (width : Nat)
    (value : ProductionSuccessorStateBinding.PreCarryValue candidate
      fullShape)
    (chunks : List (List Nat))
    (schedule : ChunkSchedule width (prefixFrame statementId value) chunks) :
    Poseidon2Duplex.gate ProductPoseidon2.constants
        (absorbChunks chunks ProductPoseidon2.initialState) =
      ProductionSuccessorStateBinding.preCarryState statementId value := by
  calc
    Poseidon2Duplex.gate ProductPoseidon2.constants
        (absorbChunks chunks ProductPoseidon2.initialState) =
        Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            chunks.flatten ProductPoseidon2.initialState) := by
      rw [absorbChunks_eq_absorbSlice_flatten chunks
        ProductPoseidon2.initialState (by decide)]
    _ = Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (prefixFrame statementId value) ProductPoseidon2.initialState) := by
      rw [schedule.flatten_eq]
    _ = Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbList ProductPoseidon2.constants
            (prefixFrame statementId value) ProductPoseidon2.initialState) := by
      unfold Poseidon2Duplex.absorbSlice
      exact gate_guarded _
    _ = ProductionSuccessorStateBinding.preCarryState statementId value :=
      (preCarryState_replays_prefixFrame statementId value).symm

/-- Every deterministic chunking of the exact complete state gives the same
public successor state as the current monolithic definition. -/
theorem scheduled_outputState_eq
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (width : Nat)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape)
    (chunks : List (List Nat))
    (schedule : ChunkSchedule width
      (ProductionSuccessorStateBinding.stateFrame statementId value) chunks) :
    Poseidon2Duplex.gate ProductPoseidon2.constants
        (absorbChunks chunks ProductPoseidon2.initialState) =
      ProductionSuccessorStateBinding.outputState statementId value := by
  calc
    Poseidon2Duplex.gate ProductPoseidon2.constants
        (absorbChunks chunks ProductPoseidon2.initialState) =
        Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            chunks.flatten ProductPoseidon2.initialState) := by
      rw [absorbChunks_eq_absorbSlice_flatten chunks
        ProductPoseidon2.initialState (by decide)]
    _ = Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (ProductionSuccessorStateBinding.stateFrame statementId value)
            ProductPoseidon2.initialState) := by
      rw [schedule.flatten_eq]
    _ = Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbList ProductPoseidon2.constants
            (ProductionSuccessorStateBinding.stateFrame statementId value)
            ProductPoseidon2.initialState) := by
      unfold Poseidon2Duplex.absorbSlice
      exact gate_guarded _
    _ = ProductionSuccessorStateBinding.outputState statementId value :=
      (ProductionSuccessorStateBinding.outputState_replays_stateFrame
        statementId value).symm

/-! ## Exact production phase counts -/

theorem production_prefix_chunk_count_exact
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.PreCarryValue candidate
      fullShape)
    (chunks : List (List Nat))
    (schedule : ChunkSchedule 1024 (prefixFrame statementId value) chunks) :
    chunks.length = 94 := by
  have lower := schedule.values_length_le_chunk_capacity
  have upper := schedule.chunk_capacity_lt_values_plus_width (by decide)
  rw [prefixFrame_length_r26 contract statementId value] at lower upper
  omega

theorem production_state_chunk_count_exact
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape)
    (chunks : List (List Nat))
    (schedule : ChunkSchedule 1024
      (ProductionSuccessorStateBinding.stateFrame statementId value) chunks) :
    chunks.length = 94 := by
  have lower := schedule.values_length_le_chunk_capacity
  have upper := schedule.chunk_capacity_lt_values_plus_width (by decide)
  rw [stateFrame_length_r26 contract statementId value] at lower upper
  omega

/-- Exact failure boundary for advice that differs from the authoritative
state frame but reaches the same gated Poseidon2 state. Generated phase rows
must either place the exact frame chunks or reduce a mismatch to this event. -/
def StateReplayCollision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId)
    (value : ProductionSuccessorStateBinding.Value candidate fullShape) : Prop :=
  exists supplied : List Nat,
    supplied ≠ ProductionSuccessorStateBinding.stateFrame statementId value /\
      Poseidon2Duplex.gate ProductPoseidon2.constants
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants supplied
            ProductPoseidon2.initialState) =
        ProductionSuccessorStateBinding.outputState statementId value

end Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming
