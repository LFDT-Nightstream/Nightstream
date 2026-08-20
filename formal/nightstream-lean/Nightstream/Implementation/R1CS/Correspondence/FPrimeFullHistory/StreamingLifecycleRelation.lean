import Nightstream.Implementation.Nebula.FPrime.State.OutputAuthorityRows
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
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
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Protocol.FPrime

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen
  uRunningWitness uFreshWitness

/-- One canonical Goldilocks field value. -/
abbrev Field := Fin goldilocksP

/-- Four canonical Goldilocks lanes. -/
abbrev Digest := Fin 4 -> Field

/-- Row-level natural encoding of four canonical Goldilocks lanes. -/
abbrev EncodedDigest := Fin 4 -> Nat

def digestValues (digest : Digest) : EncodedDigest :=
  fun lane => (digest lane).val

theorem digestValues_canonical (digest : Digest) (lane : Fin 4) :
    digestValues digest lane < goldilocksP :=
  (digest lane).isLt

theorem digestValues_injective : Function.Injective digestValues := by
  intro left right equal
  funext lane
  apply Fin.ext
  exact congrFun equal lane

theorem digestValues_list_canonical (digest : Digest) :
    ∀ value ∈ List.ofFn (digestValues digest), value < goldilocksP := by
  rw [List.forall_mem_ofFn_iff]
  exact digestValues_canonical digest

abbrev OuterState
    (Running : Type uRunning) (Fresh : Type uFresh) (Nebula : Type) :=
  State Digest Running Fresh Nebula

/-- The three natural state words have the exact range of their Rust `u64`
representation. This is a trust-boundary encoding condition, not payload
authority. -/
structure StateWordsBounded
    {Running : Type uRunning} {Fresh : Type uFresh} {Nebula : Type}
    (state : OuterState Running Fresh Nebula) : Prop where
  chunkCount : state.chunkCount < 2 ^ 64
  stepCount : state.stepCount < 2 ^ 64
  pc : state.pc < 2 ^ 64

/-- Exact row-level encoding of one typed field preimage. -/
def encodePreimage
    (preimage : XOut.XOutPreimage Digest Digest Digest) :
    XOut.XOutPreimage EncodedDigest EncodedDigest EncodedDigest where
  vkFsDigest := digestValues preimage.vkFsDigest
  piCcsHeader := digestValues preimage.piCcsHeader
  chunkCount := preimage.chunkCount
  stepCount := preimage.stepCount
  pc := preimage.pc
  currentBoundary := digestValues preimage.currentBoundary
  semanticState := preimage.semanticState.map digestValues
  construction2Accumulator := digestValues preimage.construction2Accumulator
  nebula := preimage.nebula.map digestValues

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
  /-- Recomputed from the complete active running instance. It is absent only
  for the initial base state, which has no prior running instance. -/
  beforePriorStateDigest : Option Digest
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
  /-- Verifier-owned recomputation of the prior-state digest from the complete
  running instance. Concrete artifact bindings must prove that this function
  is the same Poseidon2 computation as the manifest's mandatory prior-state
  boundary rows. -/
  runningPriorStateDigest : Running -> Digest
  phaseEnvelopeDigest : Digest -> List Fresh -> Digest
  initialPhaseState : Digest
  initialPhaseEnvelope : context.initialSemanticState =
    phaseEnvelopeDigest initialPhaseState []
  initialNebula : Nebula
  initialNebulaExact : stepSemantics.initialNebula = some initialNebula
  nifsAuthority : NifsAuthority Running Fresh NifsProof Nebula
  phaseAuthority : PhaseAuthority Fresh armCount
  /-- Verifier-owned global cursor for each local physical arm. -/
  armCursor : Fin armCount -> Nat
  armCursorInjective : Function.Injective armCursor
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

/-- The prior-state digest visible to a selected phase. The initial state has
no running instance. Every recursive state recomputes the value from its
complete active running instance. -/
def beforePriorStateDigest
    {Running : Type uRunning} {Fresh : Type uFresh} {Nebula : Type}
    (recompute : Running -> Digest)
    (state : OuterState Running Fresh Nebula) : Option Digest :=
  match state.proof with
  | .initial => none
  | .active running _ => some (recompute running)

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
  vkFsDigest := digestValues
    (XOut.verifierDigest configuration.hashSemantics configuration.context)
  piCcsHeader := digestValues configuration.context.piCcsHeader
  chunkCount := state.chunkCount
  stepCount := state.stepCount
  pc := state.pc
  currentBoundary := digestValues state.zi
  semanticState := digestValues state.semanticState
  accumulatorDigest := digestValues state.accumulatorDigest

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
    (digestValues (configuration.hashSemantics.nebulaDigest nebula))

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

/-- Every verifier-derived lifecycle frame is a canonical 32-field
Goldilocks message when its three Rust words are in range. -/
theorem frame_canonical
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
    (bounded : StateWordsBounded state) :
    ∀ value ∈ frame configuration state nebula, value < goldilocksP := by
  intro value member
  simp only [frame, StateOutputAuthorityRows.fullFrame,
    StateOutputAuthorityRows.payloadFields, payload, List.mem_append,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with framePrefix | nebulaDigest
  · rcases framePrefix with payloadPrefix | marker
    · rcases payloadPrefix with domain | payloadMember
      · subst value
        norm_num [StateOutputFrameRows.domainTag, goldilocksP]
      rcases payloadMember with payloadPrefix | accumulator
      · rcases payloadPrefix with payloadPrefix | semantic
        · rcases payloadPrefix with payloadPrefix | boundary
          · rcases payloadPrefix with payloadPrefix | pc
            · rcases payloadPrefix with payloadPrefix | step
              · rcases payloadPrefix with payloadPrefix | chunk
                · rcases payloadPrefix with vk | header
                  · exact digestValues_list_canonical _ value vk
                  · exact digestValues_list_canonical _ value header
                · exact U64HalvesRows.u64Halves_canonical
                    bounded.chunkCount value chunk
              · exact U64HalvesRows.u64Halves_canonical
                  bounded.stepCount value step
            · exact U64HalvesRows.u64Halves_canonical bounded.pc value pc
          · exact digestValues_list_canonical _ value boundary
        · exact digestValues_list_canonical _ value semantic
      · exact digestValues_list_canonical _ value accumulator
    · subst value
      norm_num [StateOutputFrameRows.nebulaMarker, goldilocksP]
  · exact digestValues_list_canonical _ value nebulaDigest

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
        (digestValues (configuration.hashSemantics.nebulaDigest nebula)) =
      encodePreimage
        (XOut.preimage configuration.hashSemantics .stateful
          configuration.context state) := by
  simp [payload, StateOutputAuthorityRows.Payload.toXOutPreimage,
    encodePreimage, XOut.preimage, present]

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
  beforePriorStateDigest :=
    beforePriorStateDigest configuration.runningPriorStateDigest prior

/-- Exact one-hot Boolean switchboard at the verifier-owned global phase
cursor. The selected arm is local to this phase; `armCursor` maps it into the
complete physical schedule. -/
structure ActiveArm
    (armCount : Nat) (armCursor : Fin armCount -> Nat) (cursor : Nat) where
  selectors : Fin armCount -> Bool
  selected : Fin armCount
  selectedCursor : armCursor selected = cursor
  selectedActive : selectors selected = true
  inactive : forall arm, arm ≠ selected -> selectors arm = false

namespace ActiveArm

theorem selector_eq_true_iff
    {armCount cursor : Nat} {armCursor : Fin armCount -> Nat}
    (selection : ActiveArm armCount armCursor cursor)
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

theorem selected_eq_of_cursor
    {armCount cursor : Nat} {armCursor : Fin armCount -> Nat}
    (injective : Function.Injective armCursor)
    (selection : ActiveArm armCount armCursor cursor)
    (arm : Fin armCount) (cursorExact : armCursor arm = cursor) :
    arm = selection.selected :=
  injective (cursorExact.trans selection.selectedCursor.symm)

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
  priorWordsBounded : StateWordsBounded prior
  nextWordsBounded : StateWordsBounded next
  priorNebula : Nebula
  priorNebulaExact : prior.nebula = some priorNebula
  nextNebula : Nebula
  nextNebulaExact : next.nebula = some nextNebula
  commonPublic : PublicEnvelope
  commonPublicExact : commonPublic = expectedPublic configuration prior next
  activeArm : ActiveArm armCount configuration.armCursor prior.stepCount
  phaseInput : List Fresh
  priorPhaseState : Digest
  nextPhaseState : Digest
  priorSemanticExact : prior.semanticState =
    configuration.phaseEnvelopeDigest priorPhaseState phaseInput
  nextSemanticExact : next.semanticState =
    configuration.phaseEnvelopeDigest nextPhaseState input.nextLatest

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

/-- The common phase envelope does not accept an independent prior-state
digest. It computes the optional value from the complete prior lifecycle
state. -/
theorem before_prior_state_digest_exact
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
    invocation.commonPublic.beforePriorStateDigest =
      beforePriorStateDigest configuration.runningPriorStateDigest
        invocation.prior := by
  rw [invocation.commonPublicExact]
  rfl

/-- The checked local transition and the one-fresh profile advance the global
step counter by exactly one in either valid lifecycle branch. -/
theorem step_count_succ
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
    invocation.next.stepCount = invocation.prior.stepCount + 1 := by
  have fromAdvanced (nextRunning : Running)
      (nextExact : invocation.next =
        Step.advancedState configuration.stepSemantics invocation.prior
          nextRunning invocation.input invocation.proof) :
      invocation.next.stepCount = invocation.prior.stepCount + 1 := by
    calc
      invocation.next.stepCount =
          (Step.advancedState configuration.stepSemantics invocation.prior
            nextRunning invocation.input invocation.proof).stepCount :=
        congrArg (fun state => state.stepCount) nextExact
      _ = invocation.prior.stepCount + invocation.input.nextLatest.length := rfl
      _ = invocation.prior.stepCount + 1 := by rw [invocation.oneFresh]
  have localProof := invocation.localHolds
  cases priorProof : invocation.prior.proof with
  | initial =>
      cases foldProof : invocation.proof.fold with
      | noFold =>
          have base : Step.BaseLocalHolds configuration.hashSemantics
              configuration.stepSemantics .stateful configuration.context
              invocation.prior invocation.next invocation.input
              invocation.proof := by
            simpa [Step.LocalHolds, priorProof, foldProof] using localProof
          exact fromAdvanced configuration.stepSemantics.emptyRunning
            base.2.2.2.2.2.1
      | recursive nifsProof =>
          simp [Step.LocalHolds, priorProof, foldProof] at localProof
  | active running latest =>
      cases foldProof : invocation.proof.fold with
      | noFold =>
          simp [Step.LocalHolds, priorProof, foldProof] at localProof
      | recursive nifsProof =>
          have recursive : Step.RecursiveLocalHolds
              configuration.hashSemantics configuration.stepSemantics
              .stateful configuration.context invocation.prior invocation.next
              invocation.input invocation.proof running latest nifsProof := by
            simpa [Step.LocalHolds, priorProof, foldProof] using localProof
          rcases recursive with ⟨_, _, _, _, verified⟩
          cases verifierExact : configuration.stepSemantics.nifsVerify
              (Step.nifsContext configuration.stepSemantics invocation.prior
                invocation.input) running latest nifsProof with
          | none => simp [verifierExact] at verified
          | some nextRunning =>
              simp only [verifierExact] at verified
              exact fromAdvanced nextRunning verified.2.2.2.1

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
        (digestValues
          (configuration.hashSemantics.nebulaDigest invocation.priorNebula)) =
      encodePreimage
        (XOut.preimage configuration.hashSemantics .stateful
          configuration.context invocation.prior) :=
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
        (digestValues
          (configuration.hashSemantics.nebulaDigest invocation.nextNebula)) =
      encodePreimage
        (XOut.preimage configuration.hashSemantics .stateful
          configuration.context invocation.next) :=
  payload_preimage_exact configuration invocation.next invocation.nextNebula
    invocation.nextNebulaExact

theorem prior_frame_canonical
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
    ∀ value ∈ frame configuration invocation.prior invocation.priorNebula,
      value < goldilocksP :=
  frame_canonical configuration invocation.prior invocation.priorNebula
    invocation.priorWordsBounded

theorem next_frame_canonical
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
    ∀ value ∈ frame configuration invocation.next invocation.nextNebula,
      value < goldilocksP :=
  frame_canonical configuration invocation.next invocation.nextNebula
    invocation.nextWordsBounded

theorem selected_cursor_exact
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
    configuration.armCursor invocation.activeArm.selected =
      invocation.prior.stepCount :=
  invocation.activeArm.selectedCursor

end Invocation

/-- Common lifecycle part of the first physical invocation. The selected phase
relation is separate because its rows have a different owner. -/
structure BaseCommon
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

/-- Complete first physical invocation. The no-fold tag is derived from
`localHolds`; the selected phase relation comes from the phase rows. -/
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
    extends BaseCommon configuration where
  selectedPhase : configuration.phaseAuthority.step commonPublic
    activeArm.selected priorPhaseState phaseInput nextPhaseState input.nextLatest

namespace BaseCommon

/-- Add the phase-row fact to the common base relation. -/
def complete
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
    (common : BaseCommon configuration)
    (selectedPhase : configuration.phaseAuthority.step common.commonPublic
      common.activeArm.selected common.priorPhaseState common.phaseInput
      common.nextPhaseState common.input.nextLatest) :
    Base configuration where
  toBaseCommon := common
  selectedPhase := selectedPhase

end BaseCommon

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

/-- The verifier-owned base envelope starts at step zero and advances by the
single fresh instance required by the lifecycle relation. -/
theorem public_cursors_zero_one
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
    base.commonPublic.beforeCursor = 0 /\
      base.commonPublic.afterCursor = 1 := by
  have cursors := Invocation.public_cursors_exact base.toInvocation
  have nextExact := base.baseLocalHolds.2.2.2.2.2.1
  constructor
  · exact cursors.1.trans base.prior_counters_zero.2
  · calc
      base.commonPublic.afterCursor = base.next.stepCount := cursors.2
      _ = (Step.advancedState configuration.stepSemantics base.prior
          configuration.stepSemantics.emptyRunning base.input
          base.proof).stepCount :=
        congrArg (fun state => state.stepCount) nextExact
      _ = base.prior.stepCount + base.input.nextLatest.length := rfl
      _ = 1 := by
        rw [base.prior_counters_zero.2, base.oneFresh]

/-- The base invocation has no prior running instance and therefore no
prior-state digest authority. -/
theorem before_prior_state_digest_none
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
    base.commonPublic.beforePriorStateDigest = none := by
  rw [Invocation.before_prior_state_digest_exact base.toInvocation]
  simp [beforePriorStateDigest, base.priorInitial]

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

/-- Common lifecycle part of every post-base physical invocation. The active
running state and exact NIFS proof are explicit authoritative fold inputs. -/
structure RecursiveCommon
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

/-- Complete post-base physical invocation. The selected phase relation comes
from the phase rows on the same common public input. -/
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
    extends RecursiveCommon configuration where
  selectedPhase : configuration.phaseAuthority.step commonPublic
    activeArm.selected priorPhaseState phaseInput nextPhaseState input.nextLatest

namespace RecursiveCommon

/-- Add the phase-row fact to the common recursive relation. -/
def complete
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
    (common : RecursiveCommon configuration)
    (selectedPhase : configuration.phaseAuthority.step common.commonPublic
      common.activeArm.selected common.priorPhaseState common.phaseInput
      common.nextPhaseState common.input.nextLatest) :
    Recursive configuration where
  toRecursiveCommon := common
  selectedPhase := selectedPhase

end RecursiveCommon

namespace Recursive

/-- The recursive common envelope carries the digest recomputed from the
exact active running instance consumed by NIFS. -/
theorem before_prior_state_digest_exact
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
    recursive.commonPublic.beforePriorStateDigest =
      some (configuration.runningPriorStateDigest recursive.running) := by
  rw [Invocation.before_prior_state_digest_exact recursive.toInvocation]
  simp [beforePriorStateDigest, recursive.priorActive]

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

/-- An exact result from the configured NIFS verifier fixes the running value
installed in the next Construction-2 state. -/
theorem checked_fold_of_exact_verifier_output
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
    (recursive : Recursive configuration)
    (authoritativeOutput : Running)
    (accepted :
      configuration.stepSemantics.nifsVerify
          (Step.nifsContext configuration.stepSemantics recursive.prior
            recursive.input)
          recursive.running recursive.latest recursive.nifsProof =
        some authoritativeOutput) :
    recursive.next = Step.advancedState configuration.stepSemantics
      recursive.prior authoritativeOutput recursive.input recursive.proof := by
  rcases recursive.checked_fold with
    ⟨nextRunning, complete, nextExact⟩
  have verifierNext :
      configuration.stepSemantics.nifsVerify
          (Step.nifsContext configuration.stepSemantics recursive.prior
            recursive.input)
          recursive.running recursive.latest recursive.nifsProof =
        some nextRunning :=
    (configuration.nifsExact {
      context := Step.nifsContext configuration.stepSemantics
        recursive.prior recursive.input
      running := recursive.running
      latest := recursive.latest
      proof := recursive.nifsProof
      output := nextRunning }).2 complete
  have outputExact : nextRunning = authoritativeOutput :=
    Option.some.inj (verifierNext.symm.trans accepted)
  simpa [outputExact] using nextExact

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
  wordsBounded : StateWordsBounded state
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
        (digestValues
          (configuration.hashSemantics.nebulaDigest terminal.nebula)) =
      encodePreimage
        (XOut.preimage configuration.hashSemantics .stateful
          configuration.context terminal.state) :=
  payload_preimage_exact configuration terminal.state terminal.nebula
    terminal.nebulaExact

theorem frame_canonical
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
    ∀ value ∈ frame configuration terminal.state terminal.nebula,
      value < goldilocksP :=
  FPrimeFullHistoryStreamingLifecycleRelation.frame_canonical configuration
    terminal.state terminal.nebula terminal.wordsBounded

end Terminal

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
