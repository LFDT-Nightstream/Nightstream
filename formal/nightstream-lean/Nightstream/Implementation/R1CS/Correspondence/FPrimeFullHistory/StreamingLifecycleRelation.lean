import Nightstream.Implementation.Nebula.FPrime.State.OutputAuthorityRows
import Nightstream.Protocol.FPrime.DelayedTrace

/-!
Contract: smallest typed lifecycle relation for the phased 32-field XOut
profile.

Owns the stateful-with-Nebula specialization of the existing F-prime step,
the exact common before/after public envelope, one verifier-selected active
physical arm, the exact consumed and produced fresh batches used by that arm,
and base, recursive, and terminal relations. Every compact XOut field is
computed from verifier context or a checked state transition.
The accumulator digest is recomputed from the complete running value, and the
Nebula digest is recomputed from the present typed lane.
The outer semantic digest is recomputed from the exact phase-local digest and
the one-step-delayed fresh payload. It is compression, not payload authority.

The recursive relation names all retained SuperNeo obligations. A concrete
profile must prove that its executable NIFS verifier is exact for this whole
bundle. The terminal relation checks the running and trailing-fresh openings
and the final Nebula predicate. It performs no additional fold.

Does not own concrete SuperNeo predicates, application or memory semantics,
generated rows, Rust conformance, Poseidon2 security, selector matrices,
recursive-size closure, or row removal.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation

open Nightstream.HyperNova.Construction2
open Nightstream.Implementation.Nebula
open Nightstream.Protocol.FPrime

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen
  uRunningWitness uFreshWitness

/-- Four canonical Goldilocks lanes. Canonicality belongs to the later row
refinement; this model fixes the exact typed width and order. -/
abbrev Digest := Fin 4 -> Nat

abbrev OuterState
    (Running : Type uRunning) (Fresh : Type uFresh) (Nebula : Type) :=
  State Digest Running Fresh Nebula

/-- One fully typed invocation of the concrete SuperNeo NIFS checker. -/
structure NifsCall
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type) where
  context : Step.NifsContext Digest Nebula
  running : Running
  latest : List Fresh
  proof : NifsProof
  output : Running

/-- Named target relations that the concrete NIFS verifier must check.
Separating these fields prevents a row audit from silently replacing the
complete verifier with only its final accumulator equation. -/
structure NifsAuthority
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type) where
  piCcs : NifsCall Running Fresh NifsProof Nebula -> Prop
  piRlc : NifsCall Running Fresh NifsProof Nebula -> Prop
  piRlcTranscript : NifsCall Running Fresh NifsProof Nebula -> Prop
  piRlcEvaluation : NifsCall Running Fresh NifsProof Nebula -> Prop
  piRlcOpening : NifsCall Running Fresh NifsProof Nebula -> Prop
  piDec : NifsCall Running Fresh NifsProof Nebula -> Prop
  outputAccumulator : NifsCall Running Fresh NifsProof Nebula -> Prop

/-- Complete typed target of one accepted SuperNeo fold. -/
structure NifsAuthority.Complete
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    (authority : NifsAuthority Running Fresh NifsProof Nebula)
    (call : NifsCall Running Fresh NifsProof Nebula) : Prop where
  piCcs : authority.piCcs call
  piRlc : authority.piRlc call
  piRlcTranscript : authority.piRlcTranscript call
  piRlcEvaluation : authority.piRlcEvaluation call
  piRlcOpening : authority.piRlcOpening call
  piDec : authority.piDec call
  outputAccumulator : authority.outputAccumulator call

/-- One public object shared by the selected lifecycle and phase relations. -/
structure PublicEnvelope where
  beforeXOut : Digest
  afterXOut : Digest
  beforeCursor : Nat
  afterCursor : Nat
deriving DecidableEq

/-- Selected phase semantics on the same semantic digests and fresh batches
and the same public envelope used by the lifecycle relation. The consumed batch
owns the one-step-delayed Nebula payload. The produced batch owns the payload
for the next invocation. -/
structure PhaseAuthority
    (Fresh : Type uFresh) (armCount : Nat) where
  step : PublicEnvelope -> Fin armCount -> Digest -> List Fresh -> Digest ->
    List Fresh -> Prop
  terminal : Digest -> List Fresh -> Prop

/-- One fixed stateful-Nebula lifecycle profile. The exactness field makes
the seven named NIFS targets part of verifier acceptance; they are not optional
semantic annotations. -/
structure Configuration
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaOpen : Type uNebulaOpen)
    (armCount : Nat) where
  hashSemantics : XOut.Semantics Params StructureDigest Digest Digest Nebula
    Digest
  stepSemantics : Step.Semantics Digest Running Fresh NifsProof Nebula
    NebulaOpen
  context : XOut.Context Params StructureDigest Digest Digest
  phaseEnvelopeDigest : Digest -> List Fresh -> Digest
  initialPhaseState : Digest
  initialPhaseEnvelope : context.initialSemanticState =
    phaseEnvelopeDigest initialPhaseState []
  initialNebula : Nebula
  initialNebulaExact : stepSemantics.initialNebula = some initialNebula
  nifsAuthority : NifsAuthority Running Fresh NifsProof Nebula
  phaseAuthority : PhaseAuthority Fresh armCount
  nifsExact : forall call,
    stepSemantics.nifsVerify call.context call.running call.latest call.proof =
        some call.output <->
      nifsAuthority.Complete call
  armCountPositive : 0 < armCount

namespace Configuration

/-- Adapter to the already checked one-step-delayed trace relation. -/
def toTrace
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount) :
    DelayedTrace.Configuration Params StructureDigest Digest Digest Running
      Fresh NifsProof Nebula Digest NebulaOpen where
  hashSemantics := configuration.hashSemantics
  stepSemantics := configuration.stepSemantics
  mode := .stateful
  context := configuration.context

end Configuration

/-- The exact 26 non-Nebula fields of the fixed 32-field preimage, derived
only from verifier context and the typed Construction-2 state. -/
def payload
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    (state : OuterState Running Fresh Nebula) :
    StateOutputAuthorityRows.Payload where
  vkFsDigest := XOut.verifierDigest configuration.hashSemantics
    configuration.context
  piCcsHeader := configuration.context.piCcsHeader
  chunkCount := state.chunkCount
  stepCount := state.stepCount
  pc := state.pc
  currentBoundary := state.zi
  semanticState := state.semanticState
  accumulatorDigest := state.accumulatorDigest

/-- Exact 32-field message for one present Nebula state. -/
def frame
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    (state : OuterState Running Fresh Nebula)
    (nebula : Nebula) : List Nat :=
  StateOutputAuthorityRows.fullFrame (payload configuration state)
    (configuration.hashSemantics.nebulaDigest nebula)

theorem frame_length
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    (state : OuterState Running Fresh Nebula)
    (nebula : Nebula) :
    (frame configuration state nebula).length = 32 :=
  StateOutputAuthorityRows.fullFrame_length _ _

/-- The independent 32-field frame owner and the protocol XOut preimage are
the same typed message when the Nebula lane is present. -/
theorem payload_preimage_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    (state : OuterState Running Fresh Nebula)
    (nebula : Nebula)
    (present : state.nebula = some nebula) :
    (payload configuration state).toXOutPreimage
        (configuration.hashSemantics.nebulaDigest nebula) =
      XOut.preimage configuration.hashSemantics .stateful
        configuration.context state := by
  simp [payload, StateOutputAuthorityRows.Payload.toXOutPreimage,
    XOut.preimage, present]

def expectedPublic
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    (prior next : OuterState Running Fresh Nebula) : PublicEnvelope where
  beforeXOut := XOut.compute configuration.hashSemantics .stateful
    configuration.context prior
  afterXOut := XOut.compute configuration.hashSemantics .stateful
    configuration.context next
  beforeCursor := prior.stepCount
  afterCursor := next.stepCount

/-- Exact one-hot Boolean switchboard at the verifier-owned program cursor. -/
structure ActiveArm (armCount cursor : Nat) where
  selectors : Fin armCount -> Bool
  selected : Fin armCount
  selectedCursor : selected.val = cursor
  selectedActive : selectors selected = true
  inactive : forall arm, arm ≠ selected -> selectors arm = false

namespace ActiveArm

theorem selector_eq_true_iff
    {armCount cursor : Nat} (selection : ActiveArm armCount cursor)
    (arm : Fin armCount) :
    selection.selectors arm = true <-> arm = selection.selected := by
  constructor
  · intro enabled
    by_cases equal : arm = selection.selected
    · exact equal
    · have disabled := selection.inactive arm equal
      rw [disabled] at enabled
      contradiction
  · intro equal
    subst arm
    exact selection.selectedActive

theorem cursor_in_range
    {armCount cursor : Nat} (selection : ActiveArm armCount cursor) :
    cursor < armCount := by
  rw [← selection.selectedCursor]
  exact selection.selected.isLt

end ActiveArm

/-- Common relation of one physical phase invocation. Its `public` value is
the single object consumed by both lifecycle rows and the selected phase arm. -/
structure Invocation
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount) where
  prior : OuterState Running Fresh Nebula
  next : OuterState Running Fresh Nebula
  input : Step.Input Fresh Nebula NebulaOpen
  proof : Step.Proof Digest NifsProof NebulaOpen
  localHolds : Step.LocalHolds configuration.hashSemantics
    configuration.stepSemantics .stateful configuration.context prior next
      input proof
  oneFresh : input.nextLatest.length = 1
  countersAligned : prior.chunkCount = prior.stepCount
  priorNebula : Nebula
  priorNebulaExact : prior.nebula = some priorNebula
  nextNebula : Nebula
  nextNebulaExact : next.nebula = some nextNebula
  commonPublic : PublicEnvelope
  commonPublicExact : commonPublic = expectedPublic configuration prior next
  activeArm : ActiveArm armCount prior.stepCount
  phaseInput : List Fresh
  priorPhaseState : Digest
  nextPhaseState : Digest
  priorSemanticExact : prior.semanticState =
    configuration.phaseEnvelopeDigest priorPhaseState phaseInput
  nextSemanticExact : next.semanticState =
    configuration.phaseEnvelopeDigest nextPhaseState input.nextLatest
  selectedPhase : configuration.phaseAuthority.step commonPublic
    activeArm.selected priorPhaseState phaseInput nextPhaseState input.nextLatest

namespace Invocation

theorem before_public_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (invocation : Invocation configuration) :
    invocation.commonPublic.beforeXOut =
      XOut.compute configuration.hashSemantics .stateful
        configuration.context invocation.prior := by
  rw [invocation.commonPublicExact]
  rfl

theorem after_public_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (invocation : Invocation configuration) :
    invocation.commonPublic.afterXOut =
      XOut.compute configuration.hashSemantics .stateful
        configuration.context invocation.next := by
  rw [invocation.commonPublicExact]
  rfl

theorem public_cursors_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (invocation : Invocation configuration) :
    invocation.commonPublic.beforeCursor = invocation.prior.stepCount /\
      invocation.commonPublic.afterCursor = invocation.next.stepCount := by
  rw [invocation.commonPublicExact]
  exact ⟨rfl, rfl⟩

theorem prior_frame_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (invocation : Invocation configuration) :
    (payload configuration invocation.prior).toXOutPreimage
        (configuration.hashSemantics.nebulaDigest invocation.priorNebula) =
      XOut.preimage configuration.hashSemantics .stateful
        configuration.context invocation.prior :=
  payload_preimage_exact configuration invocation.prior
    invocation.priorNebula invocation.priorNebulaExact

theorem next_frame_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (invocation : Invocation configuration) :
    (payload configuration invocation.next).toXOutPreimage
        (configuration.hashSemantics.nebulaDigest invocation.nextNebula) =
      XOut.preimage configuration.hashSemantics .stateful
        configuration.context invocation.next :=
  payload_preimage_exact configuration invocation.next invocation.nextNebula
    invocation.nextNebulaExact

theorem selected_cursor_in_range
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (invocation : Invocation configuration) :
    invocation.prior.stepCount < armCount :=
  invocation.activeArm.cursor_in_range

end Invocation

/-- First physical invocation. The no-fold tag is derived from `localHolds`. -/
structure Base
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    extends Invocation configuration where
  priorInitial : prior.proof = .initial
  phaseInputEmpty : phaseInput = []
  priorPhaseStateInitial : priorPhaseState = configuration.initialPhaseState

namespace Base

theorem fold_noFold
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (base : Base configuration) : base.proof.fold = .noFold := by
  cases foldExact : base.proof.fold with
  | noFold => exact rfl
  | recursive nifsProof =>
      have localHolds := base.localHolds
      simp [Step.LocalHolds, base.priorInitial, foldExact] at localHolds

theorem baseLocalHolds
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (base : Base configuration) :
    Step.BaseLocalHolds configuration.hashSemantics
      configuration.stepSemantics .stateful configuration.context base.prior
      base.next base.input base.proof := by
  simpa [Step.LocalHolds, base.priorInitial, base.fold_noFold] using
    base.localHolds

theorem prior_counters_zero
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (base : Base configuration) :
    base.prior.chunkCount = 0 /\ base.prior.stepCount = 0 := by
  have initial := base.baseLocalHolds.1
  exact ⟨initial.2.1, initial.2.2.1⟩

/-- The base phase starts with no delayed fresh claim and produces the exact
batch installed by the base lifecycle transition. -/
theorem selected_phase_starts_empty
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (base : Base configuration) :
    configuration.phaseAuthority.step base.commonPublic base.activeArm.selected
      configuration.initialPhaseState [] base.nextPhaseState
      base.input.nextLatest := by
  simpa [base.phaseInputEmpty, base.priorPhaseStateInitial] using
    base.selectedPhase

end Base

/-- Every post-base physical invocation. The active running state and exact
NIFS proof are explicit because they are the authoritative fold inputs. -/
structure Recursive
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    extends Invocation configuration where
  running : Running
  latest : List Fresh
  nifsProof : NifsProof
  priorActive : prior.proof = .active running latest
  proofRecursive : proof.fold = .recursive nifsProof
  phaseInputExact : phaseInput = latest
  oneLatest : latest.length = 1

namespace Recursive

theorem recursiveLocalHolds
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (recursive : Recursive configuration) :
    Step.RecursiveLocalHolds configuration.hashSemantics
      configuration.stepSemantics .stateful configuration.context
      recursive.prior recursive.next recursive.input recursive.proof
      recursive.running recursive.latest recursive.nifsProof := by
  simpa [Step.LocalHolds, recursive.priorActive,
    recursive.proofRecursive] using recursive.localHolds

/-- Accepted recursive execution exposes every named SuperNeo target and the
exact running value installed in the next Construction-2 state. -/
theorem checked_fold
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (recursive : Recursive configuration) :
    exists nextRunning,
      configuration.nifsAuthority.Complete {
        context := Step.nifsContext configuration.stepSemantics
          recursive.prior recursive.input
        running := recursive.running
        latest := recursive.latest
        proof := recursive.nifsProof
        output := nextRunning } /\
      recursive.next = Step.advancedState configuration.stepSemantics
        recursive.prior nextRunning recursive.input recursive.proof := by
  rcases recursive.recursiveLocalHolds with
    ⟨active, fold, latestNonempty, priorLinked, verified⟩
  cases verifierExact : configuration.stepSemantics.nifsVerify
      (Step.nifsContext configuration.stepSemantics recursive.prior
        recursive.input) recursive.running recursive.latest
        recursive.nifsProof with
  | none => simp [verifierExact] at verified
  | some nextRunning =>
      simp only [verifierExact] at verified
      rcases verified with
        ⟨nextNonempty, semantic, nebula, nextExact, outputExact⟩
      let call : NifsCall Running Fresh NifsProof Nebula := {
        context := Step.nifsContext configuration.stepSemantics
          recursive.prior recursive.input
        running := recursive.running
        latest := recursive.latest
        proof := recursive.nifsProof
        output := nextRunning }
      have complete : configuration.nifsAuthority.Complete call :=
        (configuration.nifsExact call).1 (by
          simpa [call] using verifierExact)
      exact ⟨nextRunning, complete, nextExact⟩

/-- The selected phase consumes the exact active fresh claim checked by NIFS
and produces the exact next batch installed by the lifecycle transition. -/
theorem selected_phase_consumes_latest
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    (recursive : Recursive configuration) :
    configuration.phaseAuthority.step recursive.commonPublic
      recursive.activeArm.selected recursive.priorPhaseState recursive.latest
      recursive.nextPhaseState recursive.input.nextLatest := by
  simpa [recursive.phaseInputExact] using recursive.selectedPhase

end Recursive

/-- Complete terminal relations for the fixed-one running value, trailing
fresh batch, and present Nebula state. Main and lane commitments must open to
the same witness. The delayed finalizer consumes the same trailing batch whose
opening is checked, and returns the separate post-terminal lane on which final
memory acceptance is evaluated. -/
structure TerminalAuthority
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (RunningWitness : Type uRunningWitness)
    (FreshWitness : Type uFreshWitness)
    (Nebula : Type) where
  runningCommitment : Running -> RunningWitness -> Prop
  runningLaneCommitments : Running -> RunningWitness -> Prop
  runningPublicProjection : Running -> RunningWitness -> Prop
  runningNorm : Running -> RunningWitness -> Prop
  runningEvaluations : Running -> RunningWitness -> Prop
  freshCommitment : List Fresh -> FreshWitness -> Prop
  freshLaneCommitments : List Fresh -> FreshWitness -> Prop
  freshPublicProjection : List Fresh -> FreshWitness -> Prop
  freshNorm : List Fresh -> FreshWitness -> Prop
  freshSelectedRelation : List Fresh -> FreshWitness -> Prop
  nebulaFinalize : Nebula -> List Fresh -> FreshWitness -> Option Nebula
  nebulaFinal : Nebula -> Prop

/-- Complete selected CE relation for the carried running accumulator. The
lane commitments are a second opening of slices from this same witness. -/
structure TerminalAuthority.RunningComplete
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {Nebula : Type}
    (authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula)
    (running : Running) (witness : RunningWitness) : Prop where
  commitment : authority.runningCommitment running witness
  laneCommitments : authority.runningLaneCommitments running witness
  publicProjection : authority.runningPublicProjection running witness
  norm : authority.runningNorm running witness
  evaluations : authority.runningEvaluations running witness

/-- Complete selected CCS relation for the trailing fresh batch. The lane
commitments are recomputed from slices of this same opened witness. -/
structure TerminalAuthority.FreshComplete
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {Nebula : Type}
    (authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula)
    (fresh : List Fresh) (witness : FreshWitness) : Prop where
  commitment : authority.freshCommitment fresh witness
  laneCommitments : authority.freshLaneCommitments fresh witness
  publicProjection : authority.freshPublicProjection fresh witness
  norm : authority.freshNorm fresh witness
  selectedRelation : authority.freshSelectedRelation fresh witness

/-- Terminal consumes the trailing active state and performs no additional
NIFS fold. The same state recomputes the final public XOut. -/
structure Terminal
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount)
    (authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula) where
  state : OuterState Running Fresh Nebula
  running : Running
  latest : List Fresh
  active : Step.ActiveState configuration.hashSemantics
    configuration.stepSemantics .stateful configuration.context state running
      latest
  oneFresh : latest.length = 1
  phaseState : Digest
  semanticEnvelopeExact : state.semanticState =
    configuration.phaseEnvelopeDigest phaseState latest
  phaseComplete : configuration.phaseAuthority.terminal phaseState latest
  chunkCount : state.chunkCount = armCount
  stepCount : state.stepCount = armCount
  nebula : Nebula
  nebulaExact : state.nebula = some nebula
  runningWitness : RunningWitness
  freshWitness : FreshWitness
  runningComplete : authority.RunningComplete running runningWitness
  freshComplete : authority.FreshComplete latest freshWitness
  finalNebula : Nebula
  nebulaFinalized : authority.nebulaFinalize nebula latest freshWitness =
    some finalNebula
  nebulaFinal : authority.nebulaFinal finalNebula
  publicXOut : Digest
  publicExact : publicXOut = XOut.compute configuration.hashSemantics
    .stateful configuration.context state
  trailingLinked : Step.FreshLinked configuration.stepSemantics.freshLink
    publicXOut latest

namespace Terminal

theorem terminal_relations_complete
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula}
    (terminal : Terminal configuration authority) :
    authority.RunningComplete terminal.running terminal.runningWitness /\
      authority.FreshComplete terminal.latest terminal.freshWitness :=
  ⟨terminal.runningComplete, terminal.freshComplete⟩

theorem phase_complete
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula}
    (terminal : Terminal configuration authority) :
    configuration.phaseAuthority.terminal terminal.phaseState terminal.latest :=
  terminal.phaseComplete

theorem delayed_nebula_finalized
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula}
    (terminal : Terminal configuration authority) :
    authority.nebulaFinalize terminal.nebula terminal.latest
        terminal.freshWitness = some terminal.finalNebula /\
      authority.nebulaFinal terminal.finalNebula :=
  ⟨terminal.nebulaFinalized, terminal.nebulaFinal⟩

theorem program_counter_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula}
    (terminal : Terminal configuration authority) : terminal.state.pc = 1 :=
  terminal.active.1

theorem accumulator_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula}
    (terminal : Terminal configuration authority) :
    terminal.state.accumulatorDigest =
      configuration.stepSemantics.runningDigest terminal.running :=
  terminal.active.2.2.2.2.1

theorem state_pinned
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula}
    (terminal : Terminal configuration authority) :
    XOut.StatePinned configuration.hashSemantics .stateful
      configuration.context terminal.state :=
  terminal.active.2.2.2.2.2

theorem frame_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority : TerminalAuthority Running Fresh RunningWitness FreshWitness
      Nebula}
    (terminal : Terminal configuration authority) :
    (payload configuration terminal.state).toXOutPreimage
        (configuration.hashSemantics.nebulaDigest terminal.nebula) =
      XOut.preimage configuration.hashSemantics .stateful
        configuration.context terminal.state :=
  payload_preimage_exact configuration terminal.state terminal.nebula
    terminal.nebulaExact

end Terminal

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
