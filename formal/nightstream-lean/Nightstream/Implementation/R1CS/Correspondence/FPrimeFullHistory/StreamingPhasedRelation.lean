import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.ScheduledGrouped

/-!
Contract: schedule authority for the phased production F-prime relation.

Owns the exact 400-arm selector vocabulary, its two physical lifecycle
circuits, 23 physical phase kinds, base/bootstrap/steady logical modes, the
cursor-to-arm binding, and refinement to one exact streaming-program step.

Does not own emitted rows, assignment coordinates, phase-local algebra, the
outer folding verifier, terminal proof verification, or Rust conformance.

Emits constraints: no. Concrete row families must separately prove the three
`RowsZero` equivalences used by `exactRefinement`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ScheduledGrouped

/-- One selector arm for each exact item in the production streaming program. -/
abbrev WorkArm := Fin (program productionConfig).length

theorem workArm_count : Fintype.card WorkArm = 400 := by
  rw [Fintype.card_fin, production_program_length]

/-- The verifier-owned work item selected by an arm. -/
def workItem (arm : WorkArm) : WorkItem :=
  (program productionConfig).get arm

/-- Group map used by the common-plus-phase selector composer. -/
def lifecycleGroup (arm : WorkArm) : Fin 3 :=
  lifecycleGroupAtCursor arm.val

theorem lifecycleGroup_base (arm : WorkArm) (zero : arm.val = 0) :
    lifecycleGroup arm = 0 := by
  simp [lifecycleGroup, lifecycleGroupAtCursor, zero]

theorem lifecycleGroup_bootstrap (arm : WorkArm) (one : arm.val = 1) :
    lifecycleGroup arm = 1 := by
  simp [lifecycleGroup, lifecycleGroupAtCursor, one]

theorem lifecycleGroup_steady (arm : WorkArm) (later : 2 ≤ arm.val) :
    lifecycleGroup arm = 2 := by
  have notZero : arm.val ≠ 0 := by omega
  have notOne : arm.val ≠ 1 := by omega
  simp [lifecycleGroup, lifecycleGroupAtCursor, notZero, notOne]

/-- Physical lifecycle circuit selected by one exact arm. Bootstrap and
steady recursion use the same recursive matrix. -/
def lifecycleCircuit (arm : WorkArm) : Fin 2 :=
  lifecycleCircuitAtCursor arm.val

theorem lifecycleCircuit_base (arm : WorkArm) (zero : arm.val = 0) :
    lifecycleCircuit arm = 0 := by
  simp [lifecycleCircuit, lifecycleCircuitAtCursor, zero]

theorem lifecycleCircuit_recursive (arm : WorkArm) (positive : 0 < arm.val) :
    lifecycleCircuit arm = 1 := by
  have notZero : arm.val ≠ 0 := by omega
  simp [lifecycleCircuit, lifecycleCircuitAtCursor, notZero]

/-- Physical phase circuit selected by one exact arm. Repeated PiCCS rounds
share one matrix. PiRLC families select the exact even or odd cursor shape. -/
def phaseKind (arm : WorkArm) : Fin 23 :=
  (circuitKind productionConfig (workItem arm)).code

/-- Exact semantic obligation of one selected phase arm.

The equality `before.cursor = arm.val` prevents a prover-selected phase from
acting as authority. The arm determines the work item from the fixed program. -/
def PhaseAtArm {State : Type}
    (phaseSemantics : WorkItem → State → State → Prop)
    (arm : WorkArm) (before after : Runtime State) : Prop :=
  before.cursor = arm.val ∧
    after.cursor = before.cursor + 1 ∧
      phaseSemantics (workItem arm) before.value after.value

theorem phaseAtArm_lifecycleGroup_eq_cursor {State : Type}
    {phaseSemantics : WorkItem → State → State → Prop}
    {arm : WorkArm} {before after : Runtime State}
    (phase : PhaseAtArm phaseSemantics arm before after) :
    lifecycleGroup arm = lifecycleGroupAtCursor before.cursor := by
  rw [phase.1]
  rfl

theorem phaseAtArm_to_step {State : Type}
    {phaseSemantics : WorkItem → State → State → Prop}
    {arm : WorkArm} {before after : Runtime State}
    (phase : PhaseAtArm phaseSemantics arm before after) :
    Step phaseSemantics productionConfig before after := by
  have inBounds : before.cursor < (program productionConfig).length := by
    rw [phase.1]
    exact arm.isLt
  refine ⟨inBounds, phase.2.1, ?_⟩
  have sameArm :
      (⟨before.cursor, inBounds⟩ : WorkArm) = arm := by
    apply Fin.ext
    exact phase.1
  simpa [workItem, sameArm] using phase.2.2

theorem step_to_phaseAtArm {State : Type}
    {phaseSemantics : WorkItem → State → State → Prop}
    {before after : Runtime State}
    (step : Step phaseSemantics productionConfig before after) :
    ∃ arm, PhaseAtArm phaseSemantics arm before after := by
  rcases step with ⟨inBounds, cursorAdvance, localStep⟩
  let arm : WorkArm := ⟨before.cursor, inBounds⟩
  refine ⟨arm, rfl, cursorAdvance, ?_⟩
  simpa [workItem, arm] using localStep

theorem exists_phaseAtArm_iff_step {State : Type}
    (phaseSemantics : WorkItem → State → State → Prop)
    (before after : Runtime State) :
    (∃ arm, PhaseAtArm phaseSemantics arm before after) ↔
      Step phaseSemantics productionConfig before after := by
  constructor
  · rintro ⟨arm, phase⟩
    exact phaseAtArm_to_step phase
  · exact step_to_phaseAtArm

/-- Small arm-local rows bind the shared public cursor to the exact arm. -/
def CursorAtArm {State : Type}
    (arm : WorkArm) (before after : Runtime State) : Prop :=
  before.cursor = arm.val ∧ after.cursor = before.cursor + 1

/-- Complete semantics of one arm: its physical outer circuit, physical phase
kind, and exact schedule rows hold on the same before/after values. -/
def ArmSemantics {State : Type}
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (before after : Runtime State) (arm : WorkArm) : Prop :=
  commonSemantics (lifecycleCircuit arm) before after ∧
    PhaseAtArm phaseSemantics arm before after

/-- Row-family refinement used by the executable scheduled composer.

All three row equivalences refer to the same `before` and `after` values. This
is the model-level same-assignment boundary. -/
theorem exactRefinement {State : Type}
    (commonRows : Fin 2 → ResidualFamily)
    (phaseKindRows : Fin 23 → ResidualFamily)
    (scheduleRows : WorkArm → ResidualFamily)
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (before after : Runtime State)
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (phaseKindExact : ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) ↔
        phaseSemantics (workItem arm) before.value after.value)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔ CursorAtArm arm before after) :
    ExactRefinement lifecycleCircuit phaseKind commonRows phaseKindRows
      scheduleRows
      (ArmSemantics commonSemantics phaseSemantics before after) := by
  constructor
  · intro arm commonZero phaseZero scheduleZero
    have cursor := (scheduleExact arm).mp scheduleZero
    exact ⟨(commonExact _).mp commonZero, cursor.1, cursor.2,
      (phaseKindExact arm).mp phaseZero⟩
  · intro arm semantics
    exact ⟨(commonExact _).mpr semantics.1,
      (phaseKindExact arm).mpr semantics.2.2.2,
      (scheduleExact arm).mpr ⟨semantics.2.1, semantics.2.2.1⟩⟩

/-- Exact semantic contract of scheduled selector acceptance for the
production program. -/
theorem exists_linkedAccepts_iff_armSemantics
    (noZeroProducts : NoZeroProducts)
    {State : Type}
    (commonRows : Fin 2 → ResidualFamily)
    (phaseKindRows : Fin 23 → ResidualFamily)
    (scheduleRows : WorkArm → ResidualFamily)
    (commonSemantics : Fin 2 → Runtime State → Runtime State → Prop)
    (phaseSemantics : WorkItem → State → State → Prop)
    (before after : Runtime State)
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (phaseKindExact : ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) ↔
        phaseSemantics (workItem arm) before.value after.value)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔ CursorAtArm arm before after) :
    (∃ weights lifecycleWeights phaseKindWeights,
      LinkedAccepts lifecycleCircuit phaseKind weights lifecycleWeights
        phaseKindWeights commonRows phaseKindRows scheduleRows) ↔
      ∃ arm, ArmSemantics commonSemantics phaseSemantics before after arm := by
  exact exists_linkedAccepts_iff_semantics noZeroProducts
    (exactRefinement commonRows phaseKindRows scheduleRows commonSemantics
      phaseSemantics before after commonExact phaseKindExact scheduleExact)

/-- Any accepted scheduled relation performs the verifier-selected next
program step. The shared physical rows cannot authorize a different arm. -/
theorem linkedAccepts_implies_step
    (noZeroProducts : NoZeroProducts)
    {State : Type}
    {commonRows : Fin 2 → ResidualFamily}
    {phaseKindRows : Fin 23 → ResidualFamily}
    {scheduleRows : WorkArm → ResidualFamily}
    {commonSemantics : Fin 2 → Runtime State → Runtime State → Prop}
    {phaseSemantics : WorkItem → State → State → Prop}
    {before after : Runtime State}
    (commonExact : ∀ circuit,
      RowsZero (commonRows circuit) ↔ commonSemantics circuit before after)
    (phaseKindExact : ∀ arm,
      RowsZero (phaseKindRows (phaseKind arm)) ↔
        phaseSemantics (workItem arm) before.value after.value)
    (scheduleExact : ∀ arm,
      RowsZero (scheduleRows arm) ↔ CursorAtArm arm before after)
    {weights : WorkArm → F}
    {lifecycleWeights : Fin 2 → F}
    {phaseKindWeights : Fin 23 → F}
    (accepted :
      LinkedAccepts lifecycleCircuit phaseKind weights lifecycleWeights
        phaseKindWeights commonRows phaseKindRows scheduleRows) :
    Step phaseSemantics productionConfig before after := by
  have acceptedExists :
      ∃ someWeights someLifecycleWeights somePhaseKindWeights,
        LinkedAccepts lifecycleCircuit phaseKind someWeights
          someLifecycleWeights somePhaseKindWeights commonRows phaseKindRows
          scheduleRows :=
    ⟨weights, lifecycleWeights, phaseKindWeights, accepted⟩
  have semanticsExists :=
    (exists_linkedAccepts_iff_armSemantics noZeroProducts commonRows
      phaseKindRows scheduleRows commonSemantics phaseSemantics before after
      commonExact phaseKindExact scheduleExact).mp acceptedExists
  rcases semanticsExists with ⟨arm, _, phase⟩
  exact phaseAtArm_to_step phase

/-- Terminal acceptance from the initial cursor requires exactly 400 phased
steps. -/
theorem terminal_complete_steps_exact {State : Type}
    {phaseSemantics : WorkItem → State → State → Prop}
    {steps : Nat} {startValue : State} {after : Runtime State}
    (run : Runs phaseSemantics productionConfig steps (initial startValue) after)
    (complete : Complete productionConfig after) :
    steps = 400 :=
  production_complete_run_steps_exact run complete

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedRelation
