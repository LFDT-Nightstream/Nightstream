import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPreludeStateDigest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayDigestArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayLifecycleBridge

/-!
Contract: prior-state replay semantics inside the exact global phased lifecycle.

Owns the complete ten-field replay-state preimage, the verifier-owned prior
running-instance target, exact work-item meaning, and inclusion of the local
replay relation in the global phase authority.

Also owns the composition from the exact retained replay rows to this phase
relation. It does not own chunk authority, public-column placement,
delayed-payload encoding, Poseidon2 collision resistance, other phase kinds,
or recursive-size closure.

Assurance tier: model-level plus the existing exact replay row certificates.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayPhasedLifecycle

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayDigestArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayFinalArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayLifecycleBridge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionArtifact

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

abbrev LifecycleDigest :=
  FPrimeFullHistoryStreamingLifecycleRelation.Digest

abbrev LifecyclePublicEnvelope :=
  FPrimeFullHistoryStreamingLifecycleRelation.PublicEnvelope

abbrev LocalReplayState :=
  ProductionSuccessorStateStreaming.ReplayState

/-- Full private state needed by one prior-state replay arm. The replay state
is the complete ten-field preimage of the local phase digest. -/
structure PhaseState
    (Running Fresh Nebula : Type) where
  outer : OuterState Running Fresh Nebula
  replay : LocalReplayState
  delayed : List Fresh

/-- Protocol-bound field digest of all ten persistent replay-state fields. -/
def replayDigest (state : LocalReplayState) : LifecycleDigest :=
  FPrimeFullHistoryStreamingPreludeStateDigest.stateDigest
    (persistentFields state)

/-- Row-level view of the same protocol-bound digest computation. -/
def replaySemantics :
    FPrimeFullHistoryStreamingPriorStateReplayRelation.Semantics where
  stateDigest := fun fields => digestValues
    (FPrimeFullHistoryStreamingPreludeStateDigest.stateDigest fields)

/-- The lifecycle view recomputes the phase digest from the complete replay
state. The digest is not independent authority. -/
def stateView
    {Running Fresh Nebula : Type} :
    StateView Running Fresh Nebula (PhaseState Running Fresh Nebula) where
  outer := PhaseState.outer
  phaseState := fun state => replayDigest state.replay
  phaseInput := PhaseState.delayed

/-- Local replay public values projected from the common lifecycle envelope.
The local digest lanes use the canonical row encoding of the field digest. -/
def replayEnvelope
    (envelope : LifecyclePublicEnvelope)
    (prior next : LifecycleDigest) :
    FPrimeFullHistoryStreamingPriorStateReplayRelation.PublicEnvelope where
  beforeLocalStateDigest := digestValues prior
  afterLocalStateDigest := digestValues next
  beforeProgramCursor := envelope.beforeCursor
  afterProgramCursor := envelope.afterCursor

/-- Exact prior-state replay meaning on the full phased runtime values. The
target is derived from the active running instance. It cannot be selected by
the prover. -/
def PhaseSemantics
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length)
    (chunkAt : Nat -> List Nat) (item : WorkItem)
    (before after : PhaseState Running Fresh Nebula) : Prop :=
  exists target : LifecycleDigest,
    beforePriorStateDigest configuration.runningPriorStateDigest before.outer =
        some target /\
      Holds replaySemantics (kindAt item.index) item before.replay after.replay
        (chunkAt item.index) (digestValues target)
        (replayEnvelope (expectedPublic configuration before.outer after.outer)
          (replayDigest before.replay) (replayDigest after.replay))

/-- The prior-state replay part of the global phase authority. The selected
chunk remains an explicit input to this subrelation. -/
def Step
    {Fresh : Type uFresh}
    (chunkAt : Nat -> List Nat) (envelope : LifecyclePublicEnvelope)
    (arm : WorkArm) (priorDigest : LifecycleDigest) (_priorFresh : List Fresh)
    (nextDigest : LifecycleDigest) (_nextFresh : List Fresh) : Prop :=
  exists before after : LocalReplayState, exists target : LifecycleDigest,
    priorDigest = replayDigest before /\
      nextDigest = replayDigest after /\
        envelope.beforePriorStateDigest = some target /\
          Holds replaySemantics (kindAt (workItem arm).index) (workItem arm)
            before after (chunkAt (workItem arm).index) (digestValues target)
            (replayEnvelope envelope priorDigest nextDigest)

/-- A complete production phase authority must include the prior-state replay
subrelation. Other phase families prove their own inclusion separately. -/
def Included
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length)
    (chunkAt : Nat -> List Nat) : Prop :=
  forall (envelope : LifecyclePublicEnvelope) (arm : WorkArm)
      (priorDigest : LifecycleDigest) (priorFresh : List Fresh)
      (nextDigest : LifecycleDigest) (nextFresh : List Fresh),
    Step chunkAt envelope arm priorDigest priorFresh nextDigest nextFresh ->
      configuration.phaseAuthority.step envelope arm priorDigest priorFresh
        nextDigest nextFresh

/-- Inclusion of the exact prior-state replay subrelation discharges the
arm-local lifecycle refinement. No relation for another phase kind is used. -/
theorem phaseRefinesAt_of_included
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {chunkAt : Nat -> List Nat} {arm : WorkArm}
    {before after : Runtime (PhaseState Running Fresh Nebula)}
    (included : Included configuration chunkAt) :
    PhaseRefinesAt configuration stateView
      (PhaseSemantics configuration chunkAt) arm before after := by
  intro phase
  apply included
  rcases phase.2.2 with ⟨target, targetExact, localHolds⟩
  refine ⟨before.value.replay, after.value.replay, target, rfl, rfl, ?_, ?_⟩
  · exact targetExact
  · exact localHolds

/-! ## Exact retained-row refinement -/

/-- Full-arm chunk in the exact Rust source-column order. -/
def fullSourceChunk (assignment : Nat → Nat) : List Nat :=
  (List.range' 155 chunkWidth).map assignment

private theorem fullSourceChunk_length (assignment : Nat → Nat) :
    (fullSourceChunk assignment).length = chunkWidth := by
  simp only [fullSourceChunk, List.length_map, List.length_range']

private theorem kindAt_full_of_lt {index : Nat}
    (indexBound : index < fullChunks) : kindAt index = .full := by
  have indexLt : index < 93 := by
    simpa [fullChunks] using indexBound
  have notLast : index + 1 ≠ chunkCount := by
    intro last
    have lastExact : index + 1 = 94 := by
      simpa [chunkCount, fullChunks] using last
    omega
  simp [kindAt, notLast]

/-- Full-arm public values read directly from the Rust source assignment. -/
def fullRowEnvelope (assignment : Nat → Nat) :
    FPrimeFullHistoryStreamingPriorStateReplayRelation.PublicEnvelope where
  beforeLocalStateDigest := fun lane => assignment (157184 + lane.val)
  afterLocalStateDigest := fun lane => assignment (159599 + lane.val)
  beforeProgramCursor := assignment 21
  afterProgramCursor := assignment 88

/-- Final-arm public values read directly from the Rust source assignment. -/
def finalRowEnvelope (assignment : Nat → Nat) :
    FPrimeFullHistoryStreamingPriorStateReplayRelation.PublicEnvelope where
  beforeLocalStateDigest := fun lane => assignment (82191 + lane.val)
  afterLocalStateDigest := fun lane => assignment (84606 + lane.val)
  beforeProgramCursor := assignment 21
  afterProgramCursor := assignment 88

private theorem aligned_replay_cursor
    (assignment : Nat → Nat) (one : assignment 0 = 1)
    (index : Nat) (indexBound : index < chunkCount)
    (programExact : assignment 21 = firstProgramCursor + index)
    (alignment : assignment 10 = lcEval assignment
      [(21, 1024), (0, 18446744069414583297)]) :
    assignment 10 = index * chunkWidth := by
  rw [alignment]
  simp only [lcEval, List.foldl, one, Nat.mul_one, Nat.one_mul,
    Nat.zero_add]
  rw [programExact]
  have rawExact :
      1024 * (firstProgramCursor + index) + 18446744069414583297 =
        goldilocksP + index * chunkWidth := by
    simp [firstProgramCursor, chunkWidth, goldilocksP]
    omega
  rw [rawExact, Nat.add_mod]
  have small : index * chunkWidth < goldilocksP := by
    simp [chunkCount, fullChunks, chunkWidth, goldilocksP] at indexBound ⊢
    omega
  simp only [Nat.mod_self, Nat.zero_add, Nat.mod_mod,
    Nat.mod_eq_of_lt small]

private theorem advanced_program_cursor
    (assignment : Nat → Nat) (one : assignment 0 = 1)
    (index : Nat) (indexBound : index < chunkCount)
    (programExact : assignment 21 = firstProgramCursor + index)
    (advance : assignment 88 = lcEval assignment [(0, 1), (21, 1)]) :
    assignment 88 = assignment 21 + 1 := by
  rw [advance]
  simp only [lcEval, List.foldl, one, Nat.mul_one, Nat.one_mul,
    Nat.zero_add]
  have small : 1 + assignment 21 < goldilocksP := by
    rw [programExact]
    simp [chunkCount, fullChunks, firstProgramCursor, goldilocksP]
      at indexBound ⊢
    omega
  rw [Nat.mod_eq_of_lt small]
  omega

/-- Exact full-arm rows imply the complete local replay relation. The only
non-row facts are the verifier-selected work item and its source cursor. -/
theorem full_rows_imply_holds
    (item : WorkItem) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment)
    (itemPhase : item.phase = .priorStateReplay)
    (indexBound : item.index < fullChunks)
    (programExact : assignment 21 = firstProgramCursor + item.index)
    (target : FPrimeFullHistoryStreamingPriorStateReplayRelation.Digest) :
    Holds replaySemantics .full item
      (replayStateAt assignment 1) (replayStateAt assignment 11)
      (fullSourceChunk assignment) target (fullRowEnvelope assignment) := by
  have itemBound : item.index < chunkCount := by
    simp [chunkCount, fullChunks] at indexBound ⊢
    omega
  have cursors := full_cursor_facts assignment canonical one satisfied
  have beforeCursor := aligned_replay_cursor assignment one item.index
    itemBound programExact cursors.1
  have afterCursor := advanced_program_cursor assignment one item.index
    itemBound programExact cursors.2
  have noWrap : assignment 10 + 1024 < goldilocksP := by
    rw [beforeCursor]
    simp [chunkCount, fullChunks, chunkWidth, goldilocksP]
      at itemBound ⊢
    omega
  have transition := full_replay_state_transition assignment canonical one
    satisfied noWrap
  rw [fullSlices_flatten_eq_chunk] at transition
  have activeChunk :
      (fullSourceChunk assignment).take (activeFields .full) =
        fullSourceChunk assignment := by
    change (fullSourceChunk assignment).take chunkWidth =
      fullSourceChunk assignment
    rw [← fullSourceChunk_length assignment, List.take_length]
  refine ⟨itemPhase, itemBound, ?_, ?_,
    full_before_absorbed assignment canonical one satisfied,
    beforeCursor, ?_, ?_, ?_, programExact, afterCursor, ?_⟩
  · exact (kindAt_full_of_lt indexBound).symm
  · exact fullSourceChunk_length assignment
  · rw [activeChunk]
    exact transition
  · funext lane
    exact full_before_digest_exact assignment canonical one satisfied lane
  · funext lane
    exact full_after_digest_exact assignment canonical one satisfied lane
  · simp [FinalChecks]

/-- Exact final-arm rows imply the complete local replay relation, including
zero padding and the gated target digest. -/
theorem final_rows_imply_holds
    (item : WorkItem) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment)
    (itemPhase : item.phase = .priorStateReplay)
    (itemIndex : item.index = fullChunks) :
    Holds replaySemantics .final item
      (replayStateAt assignment 1) (replayStateAt assignment 11)
      (finalChunk assignment) (targetDigestAt assignment)
      (finalRowEnvelope assignment) := by
  have boundary := boundary_facts assignment canonical one satisfied
  have itemBound : item.index < chunkCount := by
    rw [itemIndex]
    simp [chunkCount, fullChunks]
  have programExact : assignment 21 = firstProgramCursor + item.index := by
    rw [boundary.2.1, itemIndex]
    rfl
  have cursors := final_cursor_facts assignment canonical one satisfied
  have afterCursor := advanced_program_cursor assignment one item.index
    itemBound programExact cursors.2
  have beforeCursor : assignment 10 = item.index * chunkWidth := by
    rw [boundary.1, itemIndex]
    rfl
  have noWrap : assignment 10 + 522 < goldilocksP := by
    rw [boundary.1]
    simp [goldilocksP]
  have activeChunk :
      (finalChunk assignment).take (activeFields .final) =
        (List.range' 159 522).map assignment := by
    unfold finalChunk activeFields finalFields
    rw [← List.map_take, List.take_range'_of_length_ge (by decide)]
  have transition := final_replay_state_transition assignment canonical one
    satisfied noWrap
  rw [finalSlices_flatten_eq_activeChunk, ← activeChunk] at transition
  refine ⟨itemPhase, itemBound, ?_, ?_,
    final_before_absorbed assignment canonical one satisfied,
    beforeCursor, transition, ?_, ?_, programExact, afterCursor, ?_⟩
  · rw [itemIndex]
    exact kindAt_last
  · simp [finalChunk, chunkWidth]
  · funext lane
    exact final_before_digest_exact assignment canonical one satisfied lane
  · funext lane
    exact final_after_digest_exact assignment canonical one satisfied lane
  · exact final_rows_imply_finalChecks assignment canonical one satisfied

private theorem full_row_envelope_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment)
    (envelope : LifecyclePublicEnvelope)
    (beforeCursor : assignment 21 = envelope.beforeCursor)
    (afterCursor : assignment 88 = envelope.afterCursor) :
    fullRowEnvelope assignment =
      replayEnvelope envelope (replayDigest (replayStateAt assignment 1))
        (replayDigest (replayStateAt assignment 11)) := by
  have beforeDigestExact :
      (fun lane => assignment (157184 + lane.val)) =
        digestValues (replayDigest (replayStateAt assignment 1)) := by
    funext lane
    exact full_before_digest_exact assignment canonical one satisfied lane
  have afterDigestExact :
      (fun lane => assignment (159599 + lane.val)) =
        digestValues (replayDigest (replayStateAt assignment 11)) := by
    funext lane
    exact full_after_digest_exact assignment canonical one satisfied lane
  simp [fullRowEnvelope, replayEnvelope, beforeDigestExact,
    afterDigestExact, beforeCursor, afterCursor]

private theorem final_row_envelope_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment)
    (envelope : LifecyclePublicEnvelope)
    (beforeCursor : assignment 21 = envelope.beforeCursor)
    (afterCursor : assignment 88 = envelope.afterCursor) :
    finalRowEnvelope assignment =
      replayEnvelope envelope (replayDigest (replayStateAt assignment 1))
        (replayDigest (replayStateAt assignment 11)) := by
  have beforeDigestExact :
      (fun lane => assignment (82191 + lane.val)) =
        digestValues (replayDigest (replayStateAt assignment 1)) := by
    funext lane
    exact final_before_digest_exact assignment canonical one satisfied lane
  have afterDigestExact :
      (fun lane => assignment (84606 + lane.val)) =
        digestValues (replayDigest (replayStateAt assignment 11)) := by
    funext lane
    exact final_after_digest_exact assignment canonical one satisfied lane
  simp [finalRowEnvelope, replayEnvelope, beforeDigestExact,
    afterDigestExact, beforeCursor, afterCursor]

/-- Full-arm rows refine the authoritative recursive lifecycle phase. The
two explicit placement facts are owned by the public-column and chunk-source
bridges. -/
theorem full_rows_imply_phaseSemantics
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (recursive : Recursive configuration)
    (chunkAt : Nat → List Nat) (item : WorkItem)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment)
    (itemPhase : item.phase = .priorStateReplay)
    (indexBound : item.index < fullChunks)
    (sourceCursor : assignment 21 = recursive.prior.stepCount)
    (scheduleCursor : recursive.prior.stepCount =
      firstProgramCursor + item.index)
    (chunkExact : chunkAt item.index = fullSourceChunk assignment)
    (priorDelayed nextDelayed : List Fresh) :
    PhaseSemantics configuration chunkAt item
      ⟨recursive.prior, replayStateAt assignment 1, priorDelayed⟩
      ⟨recursive.next, replayStateAt assignment 11, nextDelayed⟩ := by
  let target := configuration.runningPriorStateDigest recursive.running
  refine ⟨target, ?_, ?_⟩
  · exact (Invocation.before_prior_state_digest_exact
      recursive.toInvocation).symm.trans
        (Recursive.before_prior_state_digest_exact recursive)
  have programExact := sourceCursor.trans scheduleCursor
  have localHolds := full_rows_imply_holds item assignment canonical one satisfied
    itemPhase indexBound programExact (digestValues target)
  have itemBound : item.index < chunkCount := by
    simp [chunkCount, fullChunks] at indexBound ⊢
    omega
  have sourceAfter : assignment 88 = recursive.next.stepCount := by
    calc
      assignment 88 = assignment 21 + 1 :=
        advanced_program_cursor assignment one item.index itemBound programExact
          (full_cursor_facts assignment canonical one satisfied).2
      _ = recursive.prior.stepCount + 1 := by rw [sourceCursor]
      _ = recursive.next.stepCount :=
        (Invocation.step_count_succ recursive.toInvocation).symm
  have envelopeExact := full_row_envelope_exact assignment canonical one
    satisfied (expectedPublic configuration recursive.prior recursive.next)
    sourceCursor sourceAfter
  have kindExact : kindAt item.index = .full := by
    exact kindAt_full_of_lt indexBound
  rw [kindExact, chunkExact, ← envelopeExact]
  exact localHolds

/-- Final-arm rows refine the authoritative recursive lifecycle phase. The
target is recomputed from the complete active running instance. -/
theorem final_rows_imply_phaseSemantics
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (recursive : Recursive configuration)
    (chunkAt : Nat → List Nat) (item : WorkItem)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment)
    (itemPhase : item.phase = .priorStateReplay)
    (itemIndex : item.index = fullChunks)
    (scheduleCursor : recursive.prior.stepCount =
      firstProgramCursor + item.index)
    (chunkExact : chunkAt item.index = finalChunk assignment)
    (targetLink : TargetLink recursive assignment)
    (priorDelayed nextDelayed : List Fresh) :
    PhaseSemantics configuration chunkAt item
      ⟨recursive.prior, replayStateAt assignment 1, priorDelayed⟩
      ⟨recursive.next, replayStateAt assignment 11, nextDelayed⟩ := by
  let target := configuration.runningPriorStateDigest recursive.running
  refine ⟨target, ?_, ?_⟩
  · exact (Invocation.before_prior_state_digest_exact
      recursive.toInvocation).symm.trans
        (Recursive.before_prior_state_digest_exact recursive)
  have localHolds := final_rows_imply_holds item assignment canonical one satisfied
    itemPhase itemIndex
  have boundary := boundary_facts assignment canonical one satisfied
  have sourceCursor : assignment 21 = recursive.prior.stepCount := by
    calc
      assignment 21 = firstProgramCursor + item.index := by
        rw [boundary.2.1, itemIndex]
        rfl
      _ = recursive.prior.stepCount := scheduleCursor.symm
  have itemBound : item.index < chunkCount := by
    rw [itemIndex]
    simp [chunkCount, fullChunks]
  have sourceAfter : assignment 88 = recursive.next.stepCount := by
    calc
      assignment 88 = assignment 21 + 1 :=
        advanced_program_cursor assignment one item.index itemBound
          (sourceCursor.trans scheduleCursor)
          (final_cursor_facts assignment canonical one satisfied).2
      _ = recursive.prior.stepCount + 1 := by rw [sourceCursor]
      _ = recursive.next.stepCount :=
        (Invocation.step_count_succ recursive.toInvocation).symm
  have envelopeExact := final_row_envelope_exact assignment canonical one
    satisfied (expectedPublic configuration recursive.prior recursive.next)
    sourceCursor sourceAfter
  have targetExact : targetDigestAt assignment = digestValues target := by
    funext lane
    exact targetLink lane
  have kindExact : kindAt item.index = .final := by
    rw [itemIndex]
    exact kindAt_last
  rw [kindExact, chunkExact, ← targetExact, ← envelopeExact]
  exact localHolds

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayPhasedLifecycle
