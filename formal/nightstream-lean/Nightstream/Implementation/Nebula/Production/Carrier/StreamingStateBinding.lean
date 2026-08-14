import Nightstream.Implementation.Nebula.Production.Carrier.StateBinding
import Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexCursor

/-!
Contract: bounded-chunk replay of the complete production claim transcript.

Assurance tier: model-level transcript refinement and cryptographic-reduction
boundary.

Owns the exact in-order chunk schedule, bounded chunk width, the compact replay
state carried between chunks, equivalence of chunked and monolithic Poseidon2
absorption, the delayed-challenge boundary, and preservation of the existing
full-claim binding theorem.

Does not own generated rows, a streamed paper-NIFS verifier, chunk-opening
constraints, Rust refinement, Poseidon2 security, terminal lifecycle rows, or
a row/column reduction claim.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionFullClaimStreaming

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId

/-- A schedule is not a prover-selected partition. Every head is the next
`width` coordinates, and recursion continues on the exact remaining suffix. -/
inductive ChunkSchedule {Value : Type} (width : Nat) :
    List Value → List (List Value) → Prop where
  | nil : ChunkSchedule width [] []
  | cons {remaining : List Value} {tail : List (List Value)}
      (positive : 0 < width) (nonempty : remaining ≠ [])
      (next : ChunkSchedule width (remaining.drop width) tail) :
      ChunkSchedule width remaining (remaining.take width :: tail)

namespace ChunkSchedule

/-- The schedule covers every coordinate once and preserves order. -/
theorem flatten_eq
    {Value : Type} {width : Nat} {values : List Value}
    {chunks : List (List Value)}
    (schedule : ChunkSchedule width values chunks) :
    chunks.flatten = values := by
  induction schedule with
  | nil => rfl
  | cons positive nonempty next inductionHypothesis =>
      simp only [List.flatten_cons]
      rw [inductionHypothesis]
      exact List.take_append_drop width _

/-- No scheduled chunk exceeds the selected circuit width. -/
theorem every_chunk_length_le
    {Value : Type} {width : Nat} {values : List Value}
    {chunks : List (List Value)}
    (schedule : ChunkSchedule width values chunks) :
    ∀ chunk ∈ chunks, chunk.length ≤ width := by
  induction schedule with
  | nil => simp
  | cons positive nonempty next inductionHypothesis =>
      intro chunk member
      simp only [List.mem_cons] at member
      rcases member with head | tail
      · subst chunk
        simp only [List.length_take]
        omega
      · exact inductionHypothesis chunk tail

theorem exact_cover_and_bound
    {Value : Type} {width : Nat} {values : List Value}
    {chunks : List (List Value)}
    (schedule : ChunkSchedule width values chunks) :
    chunks.flatten = values ∧
      ∀ chunk ∈ chunks, chunk.length ≤ width :=
  ⟨schedule.flatten_eq, schedule.every_chunk_length_le⟩

/-- Scheduled chunks have enough total capacity to contain the source list. -/
theorem values_length_le_chunk_capacity
    {Value : Type} {width : Nat} {values : List Value}
    {chunks : List (List Value)}
    (schedule : ChunkSchedule width values chunks) :
    values.length ≤ chunks.length * width := by
  induction schedule with
  | nil => simp
  | @cons remaining tail positive nonempty next inductionHypothesis =>
      simp only [List.length_cons, Nat.add_mul, Nat.one_mul]
      rw [List.length_drop] at inductionHypothesis
      by_cases full : width ≤ remaining.length
      · have restore : remaining.length - width + width =
            remaining.length := Nat.sub_add_cancel full
        omega
      · have short : remaining.length < width := Nat.lt_of_not_ge full
        omega

/-- A deterministic schedule has no unused complete chunk. Together with the
capacity bound, this fixes the number of chunks from the source length. -/
theorem chunk_capacity_lt_values_plus_width
    {Value : Type} {width : Nat} {values : List Value}
    {chunks : List (List Value)}
    (positive : 0 < width)
    (schedule : ChunkSchedule width values chunks) :
    chunks.length * width < values.length + width := by
  induction schedule with
  | nil => simpa using positive
  | @cons remaining tail schedulePositive nonempty next inductionHypothesis =>
      simp only [List.length_cons, Nat.add_mul, Nat.one_mul]
      rw [List.length_drop] at inductionHypothesis
      by_cases full : width ≤ remaining.length
      · have restore : remaining.length - width + width =
            remaining.length := Nat.sub_add_cancel full
        omega
      · have short : remaining.length ≤ width := Nat.le_of_lt (Nat.lt_of_not_ge full)
        have dropNil : remaining.drop width = [] :=
          List.drop_eq_nil_iff.mpr short
        rw [dropNil] at next
        have tailNil : tail = [] := by
          cases next with
          | nil => rfl
          | cons _ impossible _ => exact False.elim (impossible rfl)
        subst tail
        have remainingPositive : 0 < remaining.length :=
          List.length_pos_of_ne_nil nonempty
        simp only [List.length_nil, Nat.zero_mul, Nat.zero_add]
        omega

/-- The production frame length forces exactly 86 claim-replay steps. This is
an arithmetic consequence of the deterministic schedule, not a profile
assertion supplied by the prover. -/
theorem production_chunk_count_exact
    {Value : Type} {values : List Value} {chunks : List (List Value)}
    (lengthExact : values.length = 88023)
    (schedule : ChunkSchedule 1024 values chunks) :
    chunks.length = 86 := by
  have lower := schedule.values_length_le_chunk_capacity
  have upper := schedule.chunk_capacity_lt_values_plus_width (by decide)
  rw [lengthExact] at lower upper
  omega

end ChunkSchedule

noncomputable def absorbChunks : List (List Nat) → State → State
  | [], state => state
  | chunk :: rest, state =>
      absorbChunks rest
        (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants chunk state)

/-- Runtime state between public-frame chunks. The source frame is not stored
here. Its expected final duplex state is carried separately. -/
structure ReplayState where
  transcript : State
  cursor : Nat

def initialReplayState : ReplayState where
  transcript := ProductPoseidon2.initialState
  cursor := 0

noncomputable def ReplayState.advance
    (runtime : ReplayState) (chunk : List Nat) : ReplayState where
  transcript := Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
    chunk runtime.transcript
  cursor := runtime.cursor + chunk.length

noncomputable def replayChunks :
    List (List Nat) → ReplayState → ReplayState
  | [], runtime => runtime
  | chunk :: rest, runtime =>
      replayChunks rest (runtime.advance chunk)

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

/-- A challenge-capable replay state. Construction requires both exact end
coverage and equality with the verifier-authoritative expected duplex state.
The generated circuit must enforce these two fields before it enables any
challenge gate. -/
structure Ready (expected : State) (frameLength : Nat) where
  runtime : ReplayState
  cursorExact : runtime.cursor = frameLength
  transcriptExact : runtime.transcript = expected

noncomputable def Ready.squeeze
    {expected : State} {frameLength : Nat}
    (ready : Ready expected frameLength) :
    K × State :=
  ProductPoseidon2.squeezeK ready.runtime.transcript

theorem Ready.squeeze_eq_expected
    {expected : State} {frameLength : Nat}
    (ready : Ready expected frameLength) :
    ready.squeeze = ProductPoseidon2.squeezeK expected := by
  simp only [Ready.squeeze, ready.transcriptExact]

/-- Canonical field serialization of one duplex state: eight lanes and its
absorb cursor. -/
def duplexStateFields (state : State) : List Nat :=
  List.ofFn state.lanes ++ [state.absorbed]

@[simp] theorem duplexStateFields_length (state : State) :
    (duplexStateFields state).length = 9 := by
  simp [duplexStateFields,
    Nightstream.Implementation.R1CS.Canonical.Poseidon2Core.width]

/-- Variable state needed between public-frame chunks. The frame length and
phase program are verifier constants, not carried values. This is a semantic
state count, not a generated-relation column count. -/
def persistentFields (expected : State) (runtime : ReplayState) : List Nat :=
  duplexStateFields expected ++ duplexStateFields runtime.transcript ++
    [runtime.cursor]

@[simp] theorem persistentFields_length
    (expected : State) (runtime : ReplayState) :
    (persistentFields expected runtime).length = 19 := by
  simp [persistentFields]

/-- Exact failure event for an advice-supplied stream that differs from the
authoritative frame but reaches its verifier-carried duplex state. -/
def FrameReplayCollision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : Prop :=
  ∃ supplied : List Nat,
    supplied ≠ authoritativeFrame statementId degreeBound value ∧
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants supplied
          ProductPoseidon2.initialState =
        bindingState statementId degreeBound value

theorem absorbChunks_eq_absorbSlice_flatten
    (chunks : List (List Nat)) (state : State)
    (normalized : state.absorbed < Poseidon2Sponge.rate) :
    absorbChunks chunks state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        chunks.flatten state := by
  induction chunks generalizing state with
  | nil =>
      simp only [absorbChunks, List.flatten_nil,
        Poseidon2Duplex.absorbSlice, Poseidon2Duplex.absorbList]
      unfold Poseidon2Duplex.guarded
      rw [if_neg (Nat.not_le.mpr normalized)]
  | cons head tail inductionHypothesis =>
      simp only [absorbChunks, List.flatten_cons]
      have nextNormalized :
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants head state).absorbed <
            Poseidon2Sponge.rate := by
        exact Poseidon2Duplex.guarded_absorbed_lt
          ProductPoseidon2.constants _
      rw [inductionHypothesis _ nextNormalized]
      exact (Poseidon2Duplex.absorbSlice_append
        ProductPoseidon2.constants head tail.flatten state).symm

/-- A replay that is ready for a challenge recovers the exact authoritative
frame, unless it exhibits the named Poseidon2 replay collision. The expected
state is therefore checked compression, not independent authority. -/
theorem ready_replay_recovers_frame_or_collision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (ready : Ready (bindingState statementId degreeBound value)
      (authoritativeFrame statementId degreeBound value).length)
    (runtimeExact : ready.runtime =
      replayChunks chunks initialReplayState) :
    chunks.flatten = authoritativeFrame statementId degreeBound value ∨
      FrameReplayCollision statementId degreeBound value := by
  by_cases exactFrame :
      chunks.flatten = authoritativeFrame statementId degreeBound value
  · exact Or.inl exactFrame
  · apply Or.inr
    refine ⟨chunks.flatten, exactFrame, ?_⟩
    calc
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants chunks.flatten
          ProductPoseidon2.initialState =
          absorbChunks chunks ProductPoseidon2.initialState :=
        (absorbChunks_eq_absorbSlice_flatten chunks
          ProductPoseidon2.initialState (by decide)).symm
      _ = (replayChunks chunks initialReplayState).transcript := by
        symm
        simpa [initialReplayState] using
          replayChunks_transcript chunks initialReplayState
      _ = ready.runtime.transcript := by rw [← runtimeExact]
      _ = bindingState statementId degreeBound value :=
        ready.transcriptExact

theorem ready_replay_recovers_frame_of_no_collision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (ready : Ready (bindingState statementId degreeBound value)
      (authoritativeFrame statementId degreeBound value).length)
    (runtimeExact : ready.runtime =
      replayChunks chunks initialReplayState)
    (noCollision : ¬ FrameReplayCollision statementId degreeBound value) :
    chunks.flatten = authoritativeFrame statementId degreeBound value := by
  rcases ready_replay_recovers_frame_or_collision statementId degreeBound value
      chunks ready runtimeExact with exactFrame | collision
  · exact exactFrame
  · exact False.elim (noCollision collision)

noncomputable def streamedBindingState
    (chunks : List (List Nat)) : State :=
  absorbChunks chunks ProductPoseidon2.initialState

private theorem absorbElem_absorbed
    (constants : Poseidon2Schedule.Constants) (value : Nat) (state : State) :
    (Poseidon2Duplex.absorbElem constants value state).absorbed =
      SymbolicDuplexCursor.step state.absorbed := by
  unfold Poseidon2Duplex.absorbElem Poseidon2Duplex.guarded
    SymbolicDuplexCursor.step
  by_cases full : Poseidon2Sponge.rate ≤ state.absorbed
  · simp [full, Poseidon2Duplex.permute]
  · simp [full]

private theorem absorbList_absorbed
    (constants : Poseidon2Schedule.Constants) (input : List Nat)
    (state : State) :
    (Poseidon2Duplex.absorbList constants input state).absorbed =
      SymbolicDuplexCursor.after state.absorbed input.length := by
  induction input generalizing state with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [Poseidon2Duplex.absorbList, SymbolicDuplexCursor.after]
      rw [inductionHypothesis, absorbElem_absorbed]

private theorem after_one_two_mul_odd (count : Nat) :
    SymbolicDuplexCursor.after 1 (2 * count) = 1 ∨
      SymbolicDuplexCursor.after 1 (2 * count) = 3 := by
  induction count with
  | zero => exact Or.inl rfl
  | succ count inductionHypothesis =>
      rw [Nat.mul_succ, SymbolicDuplexCursor.after_add]
      rcases inductionHypothesis with one | three
      · rw [one]
        exact Or.inr (by decide)
      · rw [three]
        exact Or.inl (by decide)

private theorem authoritativeFrame_absorbList_not_full
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    (Poseidon2Duplex.absorbList ProductPoseidon2.constants
      (authoritativeFrame statementId degreeBound value)
      ProductPoseidon2.initialState).absorbed < Poseidon2Sponge.rate := by
  rw [absorbList_absorbed]
  have lengthExact := authoritativeFrame_lengthFor contract statementId
    degreeBound value
  rw [lengthExact]
  simp only [ProductNifsCodec.runningFieldCountFor]
  rw [show 366 + (17 + (83160 + 2 * fullShape.rowVariables) + 3888 + 540) =
      1 + 2 * (43985 + fullShape.rowVariables) by omega]
  rw [SymbolicDuplexCursor.after_add]
  change SymbolicDuplexCursor.after 1 (2 * (43985 + fullShape.rowVariables)) <
    Poseidon2Sponge.rate
  rcases after_one_two_mul_odd (43985 + fullShape.rowVariables) with one | three
  · rw [one]
    decide
  · rw [three]
    decide

/-- Exact scheduled replay has the same state as the current monolithic
transcript. No cryptographic assumption is used for this equality. -/
theorem scheduled_streamedBindingState_eq
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId) (degreeBound width : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (schedule : ChunkSchedule width
      (authoritativeFrame statementId degreeBound value) chunks) :
    streamedBindingState chunks =
      bindingState statementId degreeBound value := by
  calc
    streamedBindingState chunks =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          chunks.flatten ProductPoseidon2.initialState :=
      absorbChunks_eq_absorbSlice_flatten chunks ProductPoseidon2.initialState
        (by decide)
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (authoritativeFrame statementId degreeBound value)
          ProductPoseidon2.initialState :=
      congrArg
        (fun fields =>
          Poseidon2Duplex.absorbSlice ProductPoseidon2.constants fields
            ProductPoseidon2.initialState)
        schedule.flatten_eq
    _ = Poseidon2Duplex.absorbList ProductPoseidon2.constants
          (authoritativeFrame statementId degreeBound value)
          ProductPoseidon2.initialState :=
      Poseidon2Duplex.absorbSlice_eq_absorbList_of_absorbed_lt
        ProductPoseidon2.constants
        (authoritativeFrame statementId degreeBound value)
        ProductPoseidon2.initialState
        (authoritativeFrame_absorbList_not_full contract statementId
          degreeBound value)
    _ = bindingState statementId degreeBound value :=
      (bindingState_replays_authoritativeFrame statementId degreeBound value).symm

/-- Exact scheduled replay is challenge-ready only at the end of the complete
authoritative frame. -/
noncomputable def scheduledReady
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId) (degreeBound width : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (schedule : ChunkSchedule width
      (authoritativeFrame statementId degreeBound value) chunks) :
    Ready (bindingState statementId degreeBound value)
      (authoritativeFrame statementId degreeBound value).length where
  runtime := replayChunks chunks initialReplayState
  cursorExact := by
    rw [replayChunks_cursor]
    simpa [initialReplayState] using congrArg List.length schedule.flatten_eq
  transcriptExact := by
    rw [replayChunks_transcript]
    simpa [initialReplayState, streamedBindingState] using
      scheduled_streamedBindingState_eq contract statementId degreeBound width
        value chunks schedule

/-- The first challenge exposed by the streaming interface is exactly the
monolithic challenge, and it is available only through a `Ready` value. -/
theorem scheduledReady_squeeze_eq
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (statementId : StatementId) (degreeBound width : Nat)
    (value : Value candidate fullShape) (chunks : List (List Nat))
    (schedule : ChunkSchedule width
      (authoritativeFrame statementId degreeBound value) chunks) :
    (scheduledReady contract statementId degreeBound width value chunks
      schedule).squeeze =
      ProductPoseidon2.squeezeK
        (bindingState statementId degreeBound value) :=
  Ready.squeeze_eq_expected _

/-- Streaming preserves the current full-claim binding reduction. Any two
accepted streamed transcripts with the same final state recover the same
claim, or expose one of the two failures already named by the monolithic
model. -/
theorem equal_streamed_states_recovers_claim_or_named_failure
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {statementId : StatementId} {degreeBound width : Nat}
    (left right : CanonicalClaim candidate fullShape)
    (leftChunks rightChunks : List (List Nat))
    (leftSchedule : ChunkSchedule width
      (authoritativeFrame statementId degreeBound left.value) leftChunks)
    (rightSchedule : ChunkSchedule width
      (authoritativeFrame statementId degreeBound right.value) rightChunks)
    (equal : streamedBindingState leftChunks =
      streamedBindingState rightChunks) :
    left.value = right.value ∨
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate ∨
      FullClaimTranscriptCollision candidate fullShape statementId
        degreeBound := by
  apply equal_bindingState_recovers_claim_or_named_failure contract left right
  calc
    bindingState statementId degreeBound left.value =
        streamedBindingState leftChunks :=
      (scheduled_streamedBindingState_eq contract statementId degreeBound width
        left.value leftChunks leftSchedule).symm
    _ = streamedBindingState rightChunks := equal
    _ = bindingState statementId degreeBound right.value :=
      scheduled_streamedBindingState_eq contract statementId degreeBound width
        right.value rightChunks rightSchedule

/-- Exact production exponent used by the current radix-four Rust relation. -/
theorem authoritativeFrame_length_r26
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor 26 fullShape)
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) :
    (authoritativeFrame statementId degreeBound value).length = 88023 := by
  rw [authoritativeFrame_lengthFor contract.toShape statementId degreeBound
    value, contract.rowVariablesExact]
  decide

/-- A 1,024-field transcript slice needs 86 continuation steps: 85 complete
slices and one 983-field final slice. This is only transcript geometry; it is
not a complete NIFS row estimate. -/
theorem production_1024_chunk_geometry :
    88023 = 85 * 1024 + 983 ∧ 0 < 983 ∧ 983 ≤ 1024 := by
  decide

end Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
