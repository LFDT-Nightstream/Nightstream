import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilySequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutSequence

/-!
Contract: PiRLC family semantics inside the exact global phased lifecycle.

Owns the full family-state preimages, immutable family-major inputs and
outputs, exact work-item selection, and the PiRLC subrelation included in the
global phase authority.

Does not own emitted rows, delayed-payload encoding, Poseidon2 collision
resistance, other phase kinds, terminal acceptance, or recursive-size closure.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhasedLifecycle

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyContinuity
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutContinuity

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

abbrev Family := ProductionStreamingPiRlcInputBinding.Family
abbrev Source := ProductionStreamingPiRlcInputBinding.Source
abbrev InputRings := ProductionStreamingPiRlcInputBinding.InputRings
abbrev RingF := Nightstream.SuperNeo.Concrete.RingF

/-- Full private state needed by one PiRLC family arm. The outer state and
delayed batch are typed lifecycle data. The family state is the complete
1,045-field preimage of the local four-field digest. -/
structure PhaseState
    (Running Fresh Nebula : Type) where
  outer : OuterState Running Fresh Nebula
  family : FamilyState
  delayed : List Fresh

/-- The lifecycle view recomputes the local digest from the complete family
state. The digest is not an independent authority. -/
def stateView
    {Running Fresh Nebula : Type} :
    StateView Running Fresh Nebula (PhaseState Running Fresh Nebula) where
  outer := PhaseState.outer
  phaseState := fun state => familyStateDigest state.family
  phaseInput := PhaseState.delayed

/-- Exact production encoding of the typed delayed batch in the fixed-width
phase-envelope preimage. The equation requires the lifecycle and physical
relations to use the same Poseidon2 application. -/
structure PhaseEnvelopeCompatibility
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length) where
  encode : List Fresh → PhasePayload
  digestExact : ∀ localDigest fresh,
    configuration.phaseEnvelopeDigest localDigest fresh =
      phaseEnvelopeDigest localDigest (encode fresh)

/-- Independently recomputed physical and lifecycle semantic digests recover
the complete local preimage, or expose one exact phase-envelope collision. -/
theorem phasePreimage_exact_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (compatible : PhaseEnvelopeCompatibility configuration)
    (family : FamilyState) (physicalPayload : PhasePayload)
    (phaseState : Digest) (fresh : List Fresh) (semantic : Digest)
    (physicalExact : semantic =
      phaseEnvelopeDigest (familyStateDigest family) physicalPayload)
    (lifecycleExact : semantic =
      configuration.phaseEnvelopeDigest phaseState fresh) :
    (familyStateDigest family = phaseState ∧
        physicalPayload = compatible.encode fresh) ∨
      Poseidon2PhaseEnvelopeCollision := by
  apply phase_preimage_eq_or_collision
  calc
    phaseEnvelopeDigest (familyStateDigest family) physicalPayload =
        semantic := physicalExact.symm
    _ = configuration.phaseEnvelopeDigest phaseState fresh := lifecycleExact
    _ = phaseEnvelopeDigest phaseState (compatible.encode fresh) :=
      compatible.digestExact phaseState fresh

/-- Complete before-and-after phase-envelope preimages recovered from one
accepted physical arm and one common lifecycle invocation. -/
structure AcceptedInvocationPreimages
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {setup : InputBindingSetup}
    {family : FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.PhysicalFamily}
    (compatible : PhaseEnvelopeCompatibility configuration)
    (accepted :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateArm
        Running Fresh
      configuration.hashSemantics configuration.context setup family)
    (invocation : Invocation configuration) where
  beforeFamilyDigestExact :
    familyStateDigest accepted.physical.beforeState = invocation.priorPhaseState
  afterFamilyDigestExact :
    familyStateDigest accepted.physical.afterState = invocation.nextPhaseState
  priorPayloadExact :
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.acceptedPhasePayload
        accepted.physical = compatible.encode invocation.phaseInput
  nextPayloadExact :
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.acceptedPhasePayload
        accepted.physical = compatible.encode invocation.input.nextLatest

/-- Physical and lifecycle recomputation recover both complete phase-envelope
preimages. A failure is one explicit Poseidon2 phase-envelope collision. -/
theorem acceptedInvocation_preimages_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (compatible : PhaseEnvelopeCompatibility configuration)
    {setup : InputBindingSetup}
    {family : FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.PhysicalFamily}
    (accepted :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateArm
        Running Fresh
      configuration.hashSemantics configuration.context setup family)
    (invocation : Invocation configuration)
    (beforeOuterExact : accepted.beforeOuter = invocation.prior)
    (afterOuterExact : accepted.afterOuter = invocation.next) :
    AcceptedInvocationPreimages compatible accepted invocation ∨
      Poseidon2PhaseEnvelopeCollision := by
  have beforePhysical :=
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.accepted_semantic_eq_phase_envelope
      accepted .before
  have beforeLifecycle :
      accepted.beforeOuter.semanticState =
        configuration.phaseEnvelopeDigest invocation.priorPhaseState
          invocation.phaseInput := by
    rw [beforeOuterExact]
    exact invocation.priorSemanticExact
  rcases phasePreimage_exact_or_collision compatible
      accepted.physical.beforeState
      (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.acceptedPhasePayload
        accepted.physical)
      invocation.priorPhaseState invocation.phaseInput
      accepted.beforeOuter.semanticState (by simpa using beforePhysical)
      beforeLifecycle with beforeExact | collision
  · have afterPhysical :=
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.accepted_semantic_eq_phase_envelope
        accepted .after
    have afterLifecycle :
        accepted.afterOuter.semanticState =
          configuration.phaseEnvelopeDigest invocation.nextPhaseState
            invocation.input.nextLatest := by
      rw [afterOuterExact]
      exact invocation.nextSemanticExact
    rcases phasePreimage_exact_or_collision compatible
        accepted.physical.afterState
        (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.acceptedPhasePayload
          accepted.physical)
        invocation.nextPhaseState invocation.input.nextLatest
        accepted.afterOuter.semanticState (by simpa using afterPhysical)
        afterLifecycle with afterExact | collision
    · exact Or.inl {
        beforeFamilyDigestExact := beforeExact.1
        afterFamilyDigestExact := afterExact.1
        priorPayloadExact := beforeExact.2
        nextPayloadExact := afterExact.2 }
    · exact Or.inr collision
  · exact Or.inr collision

/-- Exact PiRLC family relation selected by one typed work item. The ordinal
equality prevents a prover-selected family. -/
def FamilyStep
    (setup : InputBindingSetup) (inputs : InputRings)
    (outputs : Family → RingF) (item : WorkItem)
    (before after : FamilyState) : Prop :=
  item.phase = .piRlcFamily ∧
    ∃ ordinal : Fin exactFamilyCount,
      item.index = ordinal.val ∧
        FamilyPhaseRelation setup before after (familyAtOrdinal ordinal)
          (fun source => inputs source (familyAtOrdinal ordinal))
          (outputs (familyAtOrdinal ordinal))

/-- PiRLC family meaning on the full phased runtime values. -/
def PhaseSemantics
    {Running Fresh Nebula : Type}
    (setup : InputBindingSetup) (inputs : InputRings)
    (outputs : Family → RingF) (item : WorkItem)
    (before after : PhaseState Running Fresh Nebula) : Prop :=
  FamilyStep setup inputs outputs item before.family after.family

/-- One exact family theorem supplies the work-item relation at its canonical
ordinal. -/
theorem familyPhase_implies_familyStep
    {setup : InputBindingSetup} {inputs : InputRings}
    {outputs : Family → RingF} {item : WorkItem}
    {before after : FamilyState} {family : Family}
    (phaseKind : item.phase = .piRlcFamily)
    (familyExact : item.index = familyOrdinal family)
    (phase : FamilyPhaseRelation setup before after family
      (fun source => inputs source family) (outputs family)) :
    FamilyStep setup inputs outputs item before after := by
  refine ⟨phaseKind, familyIndex family, familyExact, ?_⟩
  simpa using phase

/-- One accepted Rust family arm refines the family step over the common
family-major input and output arrays decoded from the complete accepted run. -/
theorem acceptedRun_implies_familyStep
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {setup : InputBindingSetup}
    (run :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun
        Running Fresh configuration.hashSemantics configuration.context setup)
    (ordinal : Fin exactFamilyCount) {item : WorkItem}
    (phaseKind : item.phase = .piRlcFamily)
    (familyExact : item.index = ordinal.val) :
    FamilyStep setup
      (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.inputRings
        run)
      (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.outputs
        run)
      item (run.arm ordinal).physical.beforeState
        (run.arm ordinal).physical.afterState := by
  apply familyPhase_implies_familyStep phaseKind
    (familyExact.trans (familyOrdinal_familyAtOrdinal ordinal).symm)
  simp only [
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.inputRings,
    FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.outputs]
  rw [familyIndex_familyAtOrdinal]
  exact (run.arm ordinal).physical.phase

/-- Typed phased runtime before one accepted physical family arm. -/
def beforeRuntime
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {setup : InputBindingSetup}
    {family : FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.PhysicalFamily}
    (accepted :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateArm
        Running Fresh configuration.hashSemantics configuration.context setup
          family)
    (invocation : Invocation configuration) :
    Runtime (PhaseState Running Fresh Nebula) where
  cursor := invocation.prior.stepCount
  value := {
    outer := invocation.prior
    family := accepted.physical.beforeState
    delayed := invocation.phaseInput }

/-- Typed phased runtime after one accepted physical family arm. -/
def afterRuntime
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {setup : InputBindingSetup}
    {family : FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.PhysicalFamily}
    (accepted :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateArm
        Running Fresh configuration.hashSemantics configuration.context setup
          family)
    (invocation : Invocation configuration) :
    Runtime (PhaseState Running Fresh Nebula) where
  cursor := invocation.next.stepCount
  value := {
    outer := invocation.next
    family := accepted.physical.afterState
    delayed := invocation.input.nextLatest }

/-- One accepted Rust family arm and one common lifecycle invocation produce
the exact typed phase step. The alternative is one named collision in the
complete phase-envelope preimage. -/
theorem acceptedRun_phaseAtArm_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (compatible : PhaseEnvelopeCompatibility configuration)
    (cursorExact : ProductionCursorExact configuration)
    {setup : InputBindingSetup}
    (run :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun
        Running Fresh configuration.hashSemantics configuration.context setup)
    (ordinal : Fin exactFamilyCount) (invocation : Invocation configuration)
    (beforeOuterExact :
      (run.arm ordinal).beforeOuter = invocation.prior)
    (afterOuterExact : (run.arm ordinal).afterOuter = invocation.next)
    (workItemExact : workItem invocation.activeArm.selected =
      ({ phase := .piRlcFamily, index := ordinal.val } : WorkItem)) :
    (AcceptedInvocationPreimages compatible (run.arm ordinal) invocation ∧
      InvocationAt configuration stateView
        (beforeRuntime (run.arm ordinal) invocation)
        (afterRuntime (run.arm ordinal) invocation) invocation ∧
      PhaseAtArm
        (PhaseSemantics setup
          (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.inputRings
            run)
          (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.outputs
            run))
        invocation.activeArm.selected
        (beforeRuntime (run.arm ordinal) invocation)
        (afterRuntime (run.arm ordinal) invocation)) ∨
      Poseidon2PhaseEnvelopeCollision := by
  rcases acceptedInvocation_preimages_or_collision compatible
      (run.arm ordinal) invocation beforeOuterExact afterOuterExact with
    preimages | collision
  · refine Or.inl ⟨preimages, ?_, ?_⟩
    · exact {
        priorExact := rfl
        nextExact := rfl
        priorPhaseStateExact := preimages.beforeFamilyDigestExact.symm
        nextPhaseStateExact := preimages.afterFamilyDigestExact.symm
        phaseInputExact := rfl
        nextPhaseInputExact := rfl
        beforeCursorExact := rfl
        afterCursorExact := rfl }
    · refine ⟨?_, ?_, ?_⟩
      · exact invocation.activeArm.selectedCursor.symm.trans
          (cursorExact invocation.activeArm.selected)
      · exact Invocation.step_count_succ invocation
      · have phaseKind :
            (workItem invocation.activeArm.selected).phase = .piRlcFamily := by
            rw [workItemExact]
        have familyExact :
            (workItem invocation.activeArm.selected).index = ordinal.val := by
          rw [workItemExact]
        simpa [PhaseSemantics, beforeRuntime, afterRuntime] using
          acceptedRun_implies_familyStep run ordinal phaseKind familyExact
  · exact Or.inr collision

/-- The PiRLC part of the global phase authority. The typed delayed batches
remain in the signature because the lifecycle envelope owns them. PiRLC does
not change or authorize their encoding. -/
def Step
    {Fresh : Type uFresh}
    (setup : InputBindingSetup) (inputs : InputRings)
    (outputs : Family → RingF) (_envelope : PublicEnvelope)
    (arm : WorkArm) (priorDigest : Digest) (_priorFresh : List Fresh)
    (nextDigest : Digest) (_nextFresh : List Fresh) : Prop :=
  ∃ before after : FamilyState,
    priorDigest = familyStateDigest before ∧
      nextDigest = familyStateDigest after ∧
        FamilyStep setup inputs outputs (workItem arm) before after

/-- A complete production phase authority must include the PiRLC subrelation.
Other phase families prove their own inclusion separately. -/
def Included
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length)
    (setup : InputBindingSetup) (inputs : InputRings)
    (outputs : Family → RingF) : Prop :=
  ∀ (envelope : PublicEnvelope) (arm : WorkArm)
      (priorDigest : Digest) (priorFresh : List Fresh)
      (nextDigest : Digest) (nextFresh : List Fresh),
    Step setup inputs outputs envelope arm priorDigest priorFresh nextDigest
        nextFresh →
      configuration.phaseAuthority.step envelope arm priorDigest priorFresh
        nextDigest nextFresh

/-- Inclusion of the exact PiRLC subrelation discharges the arm-local
lifecycle refinement. No relation for another phase kind is assumed. -/
theorem phaseRefinesAt_of_included
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {setup : InputBindingSetup} {inputs : InputRings}
    {outputs : Family → RingF} {arm : WorkArm}
    {before after : Runtime (PhaseState Running Fresh Nebula)}
    (included : Included configuration setup inputs outputs) :
    PhaseRefinesAt configuration stateView
      (PhaseSemantics setup inputs outputs) arm before after := by
  intro phase
  apply included
  exact ⟨before.value.family, after.value.family, rfl, rfl, phase.2.2⟩

/-- Common recursive lifecycle rows and one accepted PiRLC family arm imply
the complete lifecycle relation for that exact arm. The common relation does
not contain `selectedPhase`, so this composition is non-circular. -/
theorem recursiveCompleteAtArm_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (compatible : PhaseEnvelopeCompatibility configuration)
    (cursorExact : ProductionCursorExact configuration)
    {setup : InputBindingSetup}
    (run :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun
        Running Fresh configuration.hashSemantics configuration.context setup)
    (included : Included configuration setup
      (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.inputRings
        run)
      (FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun.outputs
        run))
    (ordinal : Fin exactFamilyCount)
    (recursive : RecursiveCommon configuration)
    (beforeOuterExact :
      (run.arm ordinal).beforeOuter = recursive.prior)
    (afterOuterExact : (run.arm ordinal).afterOuter = recursive.next)
    (workItemExact : workItem recursive.activeArm.selected =
      ({ phase := .piRlcFamily, index := ordinal.val } : WorkItem))
    (positive : 0 < recursive.activeArm.selected.val) :
    (AcceptedInvocationPreimages compatible (run.arm ordinal)
        recursive.toInvocation ∧
      CompleteAtArm configuration stateView recursive.activeArm.selected
        (beforeRuntime (run.arm ordinal) recursive.toInvocation)
        (afterRuntime (run.arm ordinal) recursive.toInvocation)) ∨
      Poseidon2PhaseEnvelopeCollision := by
  rcases acceptedRun_phaseAtArm_or_collision compatible cursorExact run ordinal
      recursive.toInvocation beforeOuterExact afterOuterExact workItemExact with
    exactPhase | collision
  · rcases exactPhase with ⟨preimages, bound, phase⟩
    have recursiveAt : RecursiveAt configuration stateView
        (beforeRuntime (run.arm ordinal) recursive.toInvocation)
        (afterRuntime (run.arm ordinal) recursive.toInvocation) :=
      ⟨recursive, bound⟩
    have common : CommonSemantics configuration stateView
        (lifecycleCircuit recursive.activeArm.selected)
        (beforeRuntime (run.arm ordinal) recursive.toInvocation)
        (afterRuntime (run.arm ordinal) recursive.toInvocation) := by
      simpa [CommonSemantics,
        lifecycleCircuit_recursive recursive.activeArm.selected positive] using
          recursiveAt
    have refinement : PhaseRefinesAt configuration stateView
        (PhaseSemantics setup run.inputRings run.outputs)
        recursive.activeArm.selected
        (beforeRuntime (run.arm ordinal) recursive.toInvocation)
        (afterRuntime (run.arm ordinal) recursive.toInvocation) :=
      phaseRefinesAt_of_included (included := included)
    exact Or.inl ⟨preimages,
      common_and_phase_imply_completeAtArm cursorExact refinement common phase⟩
  · exact Or.inr collision

/-- One accepted full PiRLC family run completes every selected recursive arm.
The exact schedule facts stay in a separate leaf certificate. A failure at any
ordinal is represented by the same named phase-envelope collision event. -/
theorem acceptedRun_allRecursiveComplete_or_collision
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running Fresh : Type}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    (compatible : PhaseEnvelopeCompatibility configuration)
    (cursorExact : ProductionCursorExact configuration)
    {setup : InputBindingSetup}
    (run :
      FPrimeFullHistoryStreamingPiRLCFamilyXOutSequence.AcceptedFullStateRun
        Running Fresh configuration.hashSemantics configuration.context setup)
    (included : Included configuration setup run.inputRings run.outputs)
    (recursive : Fin exactFamilyCount → RecursiveCommon configuration)
    (beforeOuterExact : ∀ ordinal,
      (run.arm ordinal).beforeOuter = (recursive ordinal).prior)
    (afterOuterExact : ∀ ordinal,
      (run.arm ordinal).afterOuter = (recursive ordinal).next)
    (workItemExact : ∀ ordinal,
      workItem (recursive ordinal).activeArm.selected =
        ({ phase := .piRlcFamily, index := ordinal.val } : WorkItem))
    (positive : ∀ ordinal, 0 < (recursive ordinal).activeArm.selected.val) :
    (∀ ordinal,
      AcceptedInvocationPreimages compatible (run.arm ordinal)
          (recursive ordinal).toInvocation ∧
        CompleteAtArm configuration stateView
          (recursive ordinal).activeArm.selected
          (beforeRuntime (run.arm ordinal) (recursive ordinal).toInvocation)
          (afterRuntime (run.arm ordinal) (recursive ordinal).toInvocation)) ∨
      Poseidon2PhaseEnvelopeCollision := by
  classical
  by_cases collision : Poseidon2PhaseEnvelopeCollision
  · exact Or.inr collision
  · exact Or.inl fun ordinal =>
      (recursiveCompleteAtArm_or_collision compatible cursorExact run included
        ordinal (recursive ordinal) (beforeOuterExact ordinal)
        (afterOuterExact ordinal) (workItemExact ordinal)
        (positive ordinal)).resolve_right collision

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPhasedLifecycle
