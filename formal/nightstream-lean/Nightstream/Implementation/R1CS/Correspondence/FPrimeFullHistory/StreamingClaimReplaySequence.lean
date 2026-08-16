import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayPhase

/-!
Contract: exact 86-phase composition of the generated claim-replay arms.

Owns the claim-phase indices in the verifier program, the 85 full chunks and
one final chunk, public digest/cursor links between adjacent assignments,
same-state recovery or a named state-digest collision, replay composition,
and recovery of the authoritative claim frame or a named replay collision.

Does not own claim-local algebra, PiCCS challenge use, selector rows, the
other 314 program phases, recursive folding, or terminal verification.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Nebula.ProductionFullClaimStateBinding
open Nightstream.Implementation.Nebula.ProductionFullClaimStreaming
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPhase
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

def claimStartProgramCursor : Nat := 83

def claimPhaseCount : Nat := 86

def claimFinalIndex : Nat := 85

/-- The verifier schedule selects 85 full arms followed by one final arm. -/
def phaseKindAt (index : Nat) : ArmKind :=
  if index = claimFinalIndex then .final else .full

@[simp] theorem phaseKindAt_final :
    phaseKindAt claimFinalIndex = .final := by
  simp [phaseKindAt]

theorem phaseKindAt_full {index : Nat} (beforeFinal : index < claimFinalIndex) :
    phaseKindAt index = .full := by
  simp [phaseKindAt, Nat.ne_of_lt beforeFinal]

/-- One satisfying raw arm at its exact verifier-owned claim index. -/
structure AcceptedPhase where
  index : Nat
  indexBound : index < claimPhaseCount
  assignment : Nat → Nat
  canonical : ∀ column, assignment column < goldilocksP
  one : assignment 0 = 1
  satisfied : (armFor (phaseKindAt index)).Satisfied assignment
  scheduleCursor :
    (decodePersistent assignment (phaseKindAt index) .before).programCursor =
      claimStartProgramCursor + index

namespace AcceptedPhase

def kind (phase : AcceptedPhase) : ArmKind := phaseKindAt phase.index

def before (phase : AcceptedPhase) : Persistent :=
  decodePersistent phase.assignment phase.kind .before

def after (phase : AcceptedPhase) : Persistent :=
  decodePersistent phase.assignment phase.kind .after

def chunk (phase : AcceptedPhase) : List Nat :=
  chunkValues phase.assignment phase.kind

def activeChunk (phase : AcceptedPhase) : List Nat :=
  phase.chunk.take (activeFields phase.kind)

theorem rowsRelation (phase : AcceptedPhase) :
    RowsRelation phase.kind phase.assignment phase.canonical := by
  exact rows_imply_relation phase.kind phase.assignment phase.canonical
    phase.one phase.satisfied

theorem activeChunk_length (phase : AcceptedPhase) :
    phase.activeChunk.length = activeFields phase.kind := by
  change
    ((chunkValues phase.assignment phase.kind).take
      (activeFields phase.kind)).length = activeFields phase.kind
  rw [List.length_take, chunkValues_length]
  cases phase.kind <;> rfl

private theorem aligned_cursor_exact
    (index : Nat) (indexBound : index < claimPhaseCount) :
    (1024 * (claimStartProgramCursor + index) +
        (goldilocksP - 84992)) % goldilocksP =
      1024 * index := by
  have sumExact :
      1024 * (claimStartProgramCursor + index) +
          (goldilocksP - 84992) =
        goldilocksP + 1024 * index := by
    unfold claimStartProgramCursor goldilocksP
    omega
  rw [sumExact, Nat.add_mod, Nat.mod_self, Nat.zero_add, Nat.mod_mod]
  apply Nat.mod_eq_of_lt
  unfold claimPhaseCount at indexBound
  unfold goldilocksP
  omega

theorem before_runtime_cursor (phase : AcceptedPhase) :
    phase.before.runtime.cursor = 1024 * phase.index := by
  change
    (decodePersistent phase.assignment phase.kind .before).runtime.cursor =
      1024 * phase.index
  rw [phase.rowsRelation.phase.frameAlignment]
  change
    (1024 * phase.before.programCursor + (goldilocksP - 84992)) %
        goldilocksP =
      1024 * phase.index
  have scheduled :
      phase.before.programCursor = claimStartProgramCursor + phase.index := by
    simpa [before, kind] using phase.scheduleCursor
  rw [scheduled]
  exact aligned_cursor_exact phase.index phase.indexBound

private theorem cursor_advance_below_modulus (phase : AcceptedPhase) :
    phase.before.runtime.cursor + activeFields phase.kind < goldilocksP := by
  rw [phase.before_runtime_cursor]
  have indexBound := phase.indexBound
  unfold claimPhaseCount at indexBound
  by_cases final : phase.index = claimFinalIndex
  · simp [AcceptedPhase.kind, phaseKindAt, final, activeFields, goldilocksP,
      claimFinalIndex]
  · have beforeFinal : phase.index < claimFinalIndex := by
      unfold claimFinalIndex at final ⊢
      omega
    unfold claimFinalIndex at beforeFinal
    simp [AcceptedPhase.kind, phaseKindAt, final, activeFields]
    unfold goldilocksP
    omega

theorem after_runtime_cursor (phase : AcceptedPhase) :
    phase.after.runtime.cursor =
      phase.before.runtime.cursor + activeFields phase.kind := by
  change
    (decodePersistent phase.assignment phase.kind .after).runtime.cursor =
      (decodePersistent phase.assignment phase.kind .before).runtime.cursor +
        activeFields phase.kind
  rw [phase.rowsRelation.phase.frameAdvance]
  exact Nat.mod_eq_of_lt phase.cursor_advance_below_modulus

private theorem replayStateExt {left right : ReplayState}
    (transcript : left.transcript = right.transcript)
    (cursor : left.cursor = right.cursor) : left = right := by
  cases left
  cases right
  simp_all

/-- One accepted arm advances the exact semantic replay state on the same
active chunk. The padded final tail is not absorbed. -/
theorem runtimeAdvance (phase : AcceptedPhase) :
    phase.after.runtime = phase.before.runtime.advance phase.activeChunk := by
  apply replayStateExt
  · exact phase.rowsRelation.phase.transcriptTransition
  · simp only [ReplayState.advance]
    rw [phase.activeChunk_length, phase.after_runtime_cursor]

end AcceptedPhase

/-- Exact equality of the five shared output/input words used to connect two
adjacent phase assignments. The values are taken from the constrained public
word decomposition, not from an unauthenticated digest sidecar. -/
def PublicLinked (left right : AcceptedPhase) : Prop :=
  (∀ lane : Fin 4,
      publicWordValue left.assignment left.kind
          (digestPublicWordIndex .after lane) =
        publicWordValue right.assignment right.kind
          (digestPublicWordIndex .before lane)) ∧
    publicWordValue left.assignment left.kind
        (cursorPublicWordIndex .after) =
      publicWordValue right.assignment right.kind
        (cursorPublicWordIndex .before)

/-- Exact failure event when two different decoded persistent states have the
same independently recomputed four-word state digest. -/
def StateDigestCollision : Prop :=
  ∃ (leftKind : ArmKind) (leftAssignment : Nat → Nat)
      (leftCanonical : ∀ column, leftAssignment column < goldilocksP)
      (rightKind : ArmKind) (rightAssignment : Nat → Nat)
      (rightCanonical : ∀ column, rightAssignment column < goldilocksP),
    decodePersistent leftAssignment leftKind .after ≠
        decodePersistent rightAssignment rightKind .before ∧
      ∀ lane : Fin 4,
        (stateDigest leftAssignment leftCanonical leftKind .after lane).val =
          (stateDigest rightAssignment rightCanonical rightKind .before lane).val

theorem publicLinked_digest_values
    (left right : AcceptedPhase) (linked : PublicLinked left right) :
    (fun lane : Fin 4 =>
        (stateDigest left.assignment left.canonical left.kind .after lane).val) =
      fun lane : Fin 4 =>
        (stateDigest right.assignment right.canonical right.kind .before lane).val := by
  funext lane
  calc
    (stateDigest left.assignment left.canonical left.kind .after lane).val =
        publicWordValue left.assignment left.kind
          (digestPublicWordIndex .after lane) :=
      left.rowsRelation.publicBinding.1 lane
    _ = publicWordValue right.assignment right.kind
          (digestPublicWordIndex .before lane) := linked.1 lane
    _ = (stateDigest right.assignment right.canonical right.kind .before lane).val :=
      (right.rowsRelation.publicBinding.2.1 lane).symm

/-- Adjacent public words recover one shared persistent state, unless they
exhibit the named state-digest collision. -/
theorem publicLinked_state_eq_or_collision
    (left right : AcceptedPhase) (linked : PublicLinked left right) :
    left.after = right.before ∨ StateDigestCollision := by
  by_cases same : left.after = right.before
  · exact Or.inl same
  · apply Or.inr
    refine ⟨left.kind, left.assignment, left.canonical,
      right.kind, right.assignment, right.canonical, same, ?_⟩
    intro lane
    exact congrFun (publicLinked_digest_values left right linked) lane

/-- Exact nonempty run shape. Starting at zero forces 85 full phases and one
final phase. Each constructor carries the constrained public link to the next
raw assignment. -/
inductive AcceptedRunFrom : Nat → List AcceptedPhase → Type where
  | final (phase : AcceptedPhase)
      (indexExact : phase.index = claimFinalIndex) :
      AcceptedRunFrom claimFinalIndex [phase]
  | cons {index : Nat} {rest : List AcceptedPhase}
      (beforeFinal : index < claimFinalIndex)
      (phase next : AcceptedPhase)
      (indexExact : phase.index = index)
      (tail : AcceptedRunFrom (index + 1) (next :: rest))
      (linked : PublicLinked phase next) :
      AcceptedRunFrom index (phase :: next :: rest)

namespace AcceptedRunFrom

def first : ∀ {index phases}, AcceptedRunFrom index phases → AcceptedPhase
  | _, _, .final phase _ => phase
  | _, _, .cons _ phase _ _ _ _ => phase

def last : ∀ {index phases}, AcceptedRunFrom index phases → AcceptedPhase
  | _, _, .final phase _ => phase
  | _, _, .cons _ _ _ _ tail _ => tail.last

@[simp] theorem first_of_nonempty
    {index : Nat} {phase : AcceptedPhase} {rest : List AcceptedPhase}
    (run : AcceptedRunFrom index (phase :: rest)) :
    run.first = phase := by
  cases run <;> rfl

def activeChunks (phases : List AcceptedPhase) : List (List Nat) :=
  phases.map AcceptedPhase.activeChunk

theorem runtime_replay_of_no_collision
    {index : Nat} {phases : List AcceptedPhase}
    (run : AcceptedRunFrom index phases)
    (noCollision : ¬ StateDigestCollision) :
    run.last.after.runtime =
      replayChunks (activeChunks phases) run.first.before.runtime := by
  induction run with
  | final phase indexExact =>
      change phase.after.runtime =
        replayChunks [phase.activeChunk] phase.before.runtime
      simp [replayChunks, phase.runtimeAdvance]
  | @cons index rest beforeFinal phase next indexExact tail linked
      inductionHypothesis =>
      have stateLink : phase.after = next.before := by
        rcases publicLinked_state_eq_or_collision phase next linked with
            same | collision
        · exact same
        · exact False.elim (noCollision collision)
      have runtimeLink : phase.after.runtime = next.before.runtime :=
        congrArg Persistent.runtime stateLink
      change tail.last.after.runtime =
        replayChunks
          (phase.activeChunk :: activeChunks (next :: rest))
          phase.before.runtime
      rw [replayChunks, inductionHypothesis]
      rw [first_of_nonempty tail, ← runtimeLink, phase.runtimeAdvance]

theorem expected_carry_of_no_collision
    {index : Nat} {phases : List AcceptedPhase}
    (run : AcceptedRunFrom index phases)
    (noCollision : ¬ StateDigestCollision) :
    run.last.after.expected = run.first.before.expected := by
  induction run with
  | final phase indexExact =>
      change phase.after.expected = phase.before.expected
      exact phase.rowsRelation.phase.expectedCarry
  | @cons index rest beforeFinal phase next indexExact tail linked
      inductionHypothesis =>
      have stateLink : phase.after = next.before := by
        rcases publicLinked_state_eq_or_collision phase next linked with
            same | collision
        · exact same
        · exact False.elim (noCollision collision)
      change tail.last.after.expected = phase.before.expected
      calc
        tail.last.after.expected = tail.first.before.expected :=
          inductionHypothesis
        _ = next.before.expected := by rw [first_of_nonempty tail]
        _ = phase.after.expected :=
          (congrArg Persistent.expected stateLink).symm
        _ = phase.before.expected := phase.rowsRelation.phase.expectedCarry

theorem finalChecks
    {index : Nat} {phases : List AcceptedPhase}
    (run : AcceptedRunFrom index phases) :
    FPrimeFullHistoryStreamingClaimReplayPhase.FinalChecks .final
      run.last.before run.last.after run.last.chunk := by
  induction run with
  | final phase indexExact =>
      have kindFinal : phase.kind = .final := by
        simp [AcceptedPhase.kind, phaseKindAt, indexExact, claimFinalIndex]
      have checks := phase.rowsRelation.phase.finalChecks
      rw [kindFinal] at checks
      change FPrimeFullHistoryStreamingClaimReplayPhase.FinalChecks .final
        (decodePersistent phase.assignment phase.kind .before)
        (decodePersistent phase.assignment phase.kind .after)
        (chunkValues phase.assignment phase.kind)
      rw [kindFinal]
      exact checks
  | cons beforeFinal phase next indexExact tail linked inductionHypothesis =>
      exact inductionHypothesis

theorem final_runtime_ready
    {index : Nat} {phases : List AcceptedPhase}
    (run : AcceptedRunFrom index phases) :
    run.last.after.runtime.transcript = run.last.after.expected ∧
      run.last.after.runtime.cursor = 88023 := by
  have checks := run.finalChecks
  exact ⟨checks.2.2.2.1, checks.2.2.2.2⟩

end AcceptedRunFrom

/-- The exact 86 accepted raw assignments recover the authoritative claim
frame, unless an adjacent state digest or the complete replay has a named
Poseidon2 collision. -/
theorem accepted_run_recovers_frame_or_named_collision
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      26 fullShape)
    (statementId : ProductPoseidon2.StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape)
    {phases : List AcceptedPhase}
    (run : AcceptedRunFrom 0 phases)
    (initialExpected :
      run.first.before.expected =
        bindingState statementId degreeBound value)
    (initialRuntime : run.first.before.runtime = initialReplayState) :
    (AcceptedRunFrom.activeChunks phases).flatten =
        authoritativeFrame statementId degreeBound value ∨
      StateDigestCollision ∨
        FrameReplayCollision statementId degreeBound value := by
  by_cases stateCollision : StateDigestCollision
  · exact Or.inr (Or.inl stateCollision)
  · have replay := run.runtime_replay_of_no_collision stateCollision
    have expected := run.expected_carry_of_no_collision stateCollision
    have finalReady := run.final_runtime_ready
    have frameLength := authoritativeFrame_lengthFor contract.toShape statementId
      degreeBound value
    rw [contract.rowVariablesExact] at frameLength
    let ready : Ready (bindingState statementId degreeBound value)
        (authoritativeFrame statementId degreeBound value).length := {
      runtime := run.last.after.runtime
      cursorExact := by
        rw [frameLength]
        exact finalReady.2
      transcriptExact := by
        calc
          run.last.after.runtime.transcript = run.last.after.expected :=
            finalReady.1
          _ = run.first.before.expected := expected
          _ = bindingState statementId degreeBound value := initialExpected }
    have runtimeExact :
        ready.runtime = replayChunks
          (AcceptedRunFrom.activeChunks phases) initialReplayState := by
      change run.last.after.runtime = _
      rw [replay, initialRuntime]
    rcases ready_replay_recovers_frame_or_collision statementId degreeBound
        value (AcceptedRunFrom.activeChunks phases) ready runtimeExact with
      exactFrame | replayCollision
    · exact Or.inl exactFrame
    · exact Or.inr (Or.inr replayCollision)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence
