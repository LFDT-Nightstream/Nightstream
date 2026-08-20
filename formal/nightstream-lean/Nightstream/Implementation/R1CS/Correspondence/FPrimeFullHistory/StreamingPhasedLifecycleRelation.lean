import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedOverlayRelation

/-!
Contract: same-arm composition of the phased schedule and lifecycle relation.

Owns the authoritative runtime view, the common base/recursive row target,
the production arm-cursor profile, and completion with one selected phase
relation on the same before/after values.

Does not own emitted rows, phase-local algebra, terminal acceptance,
recursive-size closure, or collision resistance.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedLifecycleRelation

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen

/-- The authoritative lifecycle fields carried by one phased runtime value. -/
structure StateView
    (Running : Type uRunning) (Fresh : Type uFresh) (Nebula : Type)
    (State : Type) where
  outer : State → OuterState Running Fresh Nebula
  phaseState : State → Digest
  phaseInput : State → List Fresh

/-- One common lifecycle invocation bound to the phased runtime values and
both schedule cursors. -/
structure InvocationAt
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
    {State : Type}
    (view : StateView Running Fresh Nebula State)
    (before after : Runtime State)
    (invocation : Invocation configuration) : Prop where
  priorExact : invocation.prior = view.outer before.value
  nextExact : invocation.next = view.outer after.value
  priorPhaseStateExact : invocation.priorPhaseState =
    view.phaseState before.value
  nextPhaseStateExact : invocation.nextPhaseState =
    view.phaseState after.value
  phaseInputExact : invocation.phaseInput = view.phaseInput before.value
  nextPhaseInputExact : invocation.input.nextLatest =
    view.phaseInput after.value
  beforeCursorExact : before.cursor = invocation.prior.stepCount
  afterCursorExact : after.cursor = invocation.next.stepCount

/-- Common-row meaning of the base lifecycle circuit. -/
def BaseAt
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
    {State : Type}
    (view : StateView Running Fresh Nebula State)
    (before after : Runtime State) : Prop :=
  ∃ common : BaseCommon configuration,
    InvocationAt configuration view before after common.toInvocation

/-- Common-row meaning of the recursive lifecycle circuit. -/
def RecursiveAt
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
    {State : Type}
    (view : StateView Running Fresh Nebula State)
    (before after : Runtime State) : Prop :=
  ∃ common : RecursiveCommon configuration,
    InvocationAt configuration view before after common.toInvocation

/-- Meaning of the two physical common circuits before phase rows are added. -/
def CommonSemantics
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
    {State : Type}
    (view : StateView Running Fresh Nebula State)
    (circuit : Fin 2) (before after : Runtime State) : Prop :=
  if circuit = 0 then
    BaseAt configuration view before after
  else
    RecursiveAt configuration view before after

/-- The selected production arm and the lifecycle arm use the same cursor. -/
def ProductionCursorExact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length) : Prop :=
  ∀ arm, configuration.armCursor arm = arm.val

/-- Phase-local semantics refine the lifecycle phase relation for one exact
schedule arm and the same public before/after state. This is the leaf proof
boundary for one row family. -/
def PhaseRefinesAt
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length)
    {State : Type}
    (view : StateView Running Fresh Nebula State)
    (phaseSemantics : WorkItem → State → State → Prop)
    (arm : WorkArm) (before after : Runtime State) : Prop :=
  PhaseAtArm phaseSemantics arm before after →
    configuration.phaseAuthority.step
      (expectedPublic configuration
        (view.outer before.value) (view.outer after.value))
      arm (view.phaseState before.value) (view.phaseInput before.value)
      (view.phaseState after.value) (view.phaseInput after.value)

/-- Every phase-local semantic relation refines the lifecycle phase relation.
Concrete phase families prove `PhaseRefinesAt` separately before this record
is assembled. -/
structure PhaseRefinement
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length)
    {State : Type}
    (view : StateView Running Fresh Nebula State)
    (phaseSemantics : WorkItem → State → State → Prop) : Prop where
  selected : ∀ arm before after,
    PhaseRefinesAt configuration view phaseSemantics arm before after

/-- Complete lifecycle meaning selected by one exact schedule arm. -/
def CompleteAtArm
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    (configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length)
    {State : Type}
    (view : StateView Running Fresh Nebula State)
    (arm : WorkArm) (before after : Runtime State) : Prop :=
  (arm.val = 0 ∧ ∃ base : Base configuration,
    InvocationAt configuration view before after base.toInvocation) ∨
  (0 < arm.val ∧ ∃ recursive : Recursive configuration,
    InvocationAt configuration view before after recursive.toInvocation)

private theorem selected_arm_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {State : Type}
    {view : StateView Running Fresh Nebula State}
    {before after : Runtime State}
    {invocation : Invocation configuration}
    {arm : WorkArm}
    (cursorExact : ProductionCursorExact configuration)
    (bound : InvocationAt configuration view before after invocation)
    (cursor : before.cursor = arm.val) :
    invocation.activeArm.selected = arm := by
  apply Fin.ext
  calc
    invocation.activeArm.selected.val =
        configuration.armCursor invocation.activeArm.selected :=
      (cursorExact invocation.activeArm.selected).symm
    _ = invocation.prior.stepCount := invocation.activeArm.selectedCursor
    _ = before.cursor := bound.beforeCursorExact.symm
    _ = arm.val := cursor

private theorem selected_phase_exact
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {State : Type}
    {view : StateView Running Fresh Nebula State}
    {phaseSemantics : WorkItem → State → State → Prop}
    {before after : Runtime State}
    {invocation : Invocation configuration}
    {arm : WorkArm}
    (cursorExact : ProductionCursorExact configuration)
    (refinement : PhaseRefinesAt configuration view phaseSemantics arm
      before after)
    (bound : InvocationAt configuration view before after invocation)
    (phase : PhaseAtArm phaseSemantics arm before after) :
    configuration.phaseAuthority.step invocation.commonPublic
      invocation.activeArm.selected invocation.priorPhaseState
      invocation.phaseInput invocation.nextPhaseState
      invocation.input.nextLatest := by
  have selected : invocation.activeArm.selected = arm :=
    selected_arm_exact cursorExact bound phase.1
  rw [selected, invocation.commonPublicExact, bound.priorExact,
    bound.nextExact, bound.priorPhaseStateExact,
    bound.nextPhaseStateExact, bound.phaseInputExact,
    bound.nextPhaseInputExact]
  exact refinement phase

/-- Common lifecycle meaning and phase meaning for one selected arm complete
the base or recursive lifecycle relation for that same arm. -/
theorem common_and_phase_imply_completeAtArm
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {State : Type}
    {view : StateView Running Fresh Nebula State}
    {phaseSemantics : WorkItem → State → State → Prop}
    {before after : Runtime State}
    {arm : WorkArm}
    (cursorExact : ProductionCursorExact configuration)
    (refinement : PhaseRefinesAt configuration view phaseSemantics arm
      before after)
    (common : CommonSemantics configuration view
      (lifecycleCircuit arm) before after)
    (phase : PhaseAtArm phaseSemantics arm before after) :
    CompleteAtArm configuration view arm before after := by
  by_cases zero : arm.val = 0
  · have baseCommon : BaseAt configuration view before after := by
      simpa [CommonSemantics, lifecycleCircuit_base arm zero] using common
    rcases baseCommon with ⟨base, bound⟩
    have selectedPhase := selected_phase_exact cursorExact refinement bound phase
    exact Or.inl ⟨zero, base.complete selectedPhase, bound⟩
  · have positive : 0 < arm.val := Nat.pos_of_ne_zero zero
    have recursiveCommon : RecursiveAt configuration view before after := by
      simpa [CommonSemantics,
        lifecycleCircuit_recursive arm positive] using common
    rcases recursiveCommon with ⟨recursive, bound⟩
    have selectedPhase := selected_phase_exact cursorExact refinement bound phase
    exact Or.inr ⟨positive, recursive.complete selectedPhase, bound⟩

/-- A same-arm selector refinement completes one authoritative lifecycle
transition. -/
theorem jointArmSemantics_implies_completeAtArm
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {State : Type}
    {view : StateView Running Fresh Nebula State}
    {phaseSemantics : WorkItem → State → State → Prop}
    {before after : Runtime State}
    (cursorExact : ProductionCursorExact configuration)
    (selected : ∃ arm,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.JointArmSemantics
        (CommonSemantics configuration view)
        phaseSemantics before after arm ∧
      PhaseRefinesAt configuration view phaseSemantics arm before after) :
    ∃ arm, CompleteAtArm configuration view arm before after := by
  rcases selected with ⟨arm, semantics, refinement⟩
  exact ⟨arm, common_and_phase_imply_completeAtArm cursorExact refinement
    semantics.1 semantics.2⟩

/-- A complete all-arm refinement discharges the local refinement required by
one same-arm selector result. -/
theorem jointArmSemantics_implies_completeAtArm_of_refinement
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen (program productionConfig).length}
    {State : Type}
    {view : StateView Running Fresh Nebula State}
    {phaseSemantics : WorkItem → State → State → Prop}
    {before after : Runtime State}
    (cursorExact : ProductionCursorExact configuration)
    (refinement : PhaseRefinement configuration view phaseSemantics)
    (selected : ∃ arm,
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.JointArmSemantics
        (CommonSemantics configuration view)
        phaseSemantics before after arm) :
    ∃ arm, CompleteAtArm configuration view arm before after := by
  apply jointArmSemantics_implies_completeAtArm cursorExact
  rcases selected with ⟨arm, semantics⟩
  exact ⟨arm, semantics, refinement.selected arm before after⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedLifecycleRelation
