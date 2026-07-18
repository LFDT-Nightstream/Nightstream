import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne

/-!
Closed inclusion-minimality fixture for the one-slot base plan.

Assurance tier: model-level.

Owns: one explicit machine, independent countermodels for the two retained
base equations, the exhaustive family ledger, and inclusion-minimal soundness
of the fixed-one plan.

Does not own: production decoding, recursive/NIFS obligations, Poseidon2,
Rust, R1CS, costs, global gate-count minimality, or row removal.

Emits constraints: no.

Authority boundary: the fixture reuses only typed relation data from the
independently checked 270-coordinate honest baseline. Its Boolean state and
outer machine are explicit. Each countermodel changes exactly the field owned
by the removed family.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.base.fixed_one.minimality.iteration` | without `i = 0`, iteration one is admitted | removal witness | `iterationZero_necessary` |
| `fprime.base.fixed_one.minimality.initial_state` | without `z_0 = z_i`, unequal Boolean states are admitted | removal witness | `initialState_necessary` |
| `fprime.base.fixed_one.minimality.ledger` | every family is retained or eliminated, never both | exact classification | `family_ledger` |
| `fprime.base.fixed_one.minimality.closed` | the two-family plan is inclusion-minimal for soundness | model-level theorem | `inclusionMinimalSound` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Protocol.FPrime.ConcretePhi81.Outer

namespace Baseline

def selected : Fin 1 := ⟨0, by decide⟩

abbrev ModelInput :=
  Input Unit Bool Unit ActiveSemantics.HonestBaseline.Sources.shape
    ActiveSemantics.HonestBaseline.Context.publicRingColumns
    ActiveSemantics.HonestBaseline.Context.publicFits
    ActiveSemantics.HonestBaseline.Context.verifierRows 1

/-- Construct a base input while retaining the same typed relation data. -/
def inputAt (iteration : Nat) (z0 zi : Bool) : ModelInput where
  verifierKey := ()
  iteration := iteration
  z0 := z0
  zi := zi
  running := ActiveSemantics.HonestBaseline.Context.input.running
  fresh := ActiveSemantics.HonestBaseline.Context.input.fresh
  priorPc := 1
  witness := ()

def honestInput : ModelInput := inputAt 0 false false

def nonzeroIterationInput : ModelInput := inputAt 1 false false

def unequalStateInput : ModelInput := inputAt 0 false true

/-- Explicit one-slot outer machine. Only its control function participates in
the base obligation plan; the remaining fields make the complete paper
machine visible rather than assumed. -/
def machine :
    Machine Unit Unit Bool Unit ActiveSemantics.HonestBaseline.Sources.shape
      ActiveSemantics.HonestBaseline.Context.publicRingColumns
      ActiveSemantics.HonestBaseline.Context.publicFits
      ActiveSemantics.HonestBaseline.Context.verifierRows 1 where
  control := fun _ _ =>
    Paper.ProgramCounter.ofIndex selected
  step := fun _ state _ => state
  defaultRunning :=
    ActiveSemantics.HonestBaseline.Context.input.running.toPaper
  hash := fun _ => ()
  encodeInstance := fun _ =>
    ActiveSemantics.HonestBaseline.Context.input.fresh.publicInput

def caseOf (input : ModelInput) :
    ObligationPlan.Case Unit Bool Unit
      ActiveSemantics.HonestBaseline.Sources.shape
      ActiveSemantics.HonestBaseline.Context.publicRingColumns
      ActiveSemantics.HonestBaseline.Context.publicFits
      ActiveSemantics.HonestBaseline.Context.verifierRows 1 where
  input := input

end Baseline

/-- Removing the iteration-zero equation admits an otherwise valid base
input at iteration one. -/
theorem iterationZero_necessary :
    CheckPlan.NecessaryForSoundness
      (ObligationPlan.semantics Baseline.machine Baseline.selected)
      (ObligationPlan.target Baseline.machine Baseline.selected)
      checks .iterationZero := by
  refine ⟨Baseline.caseOf Baseline.nonzeroIterationInput, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | iterationZero => exact (retained rfl).elim
    | initialState => rfl
    | dispatch => exact (dispatch_not_mem inChecks).elim
  · intro obligations
    have : (1 : Nat) = 0 := obligations.iterationZero
    omega

/-- Removing the initial-state equation admits unequal Boolean states while
retaining iteration zero. -/
theorem initialState_necessary :
    CheckPlan.NecessaryForSoundness
      (ObligationPlan.semantics Baseline.machine Baseline.selected)
      (ObligationPlan.target Baseline.machine Baseline.selected)
      checks .initialState := by
  refine ⟨Baseline.caseOf Baseline.unequalStateInput, ?_, ?_⟩
  · intro family member
    have inChecks := (CheckPlan.mem_without_iff.mp member).1
    have retained := (CheckPlan.mem_without_iff.mp member).2
    cases family with
    | iterationZero => rfl
    | initialState => exact (retained rfl).elim
    | dispatch => exact (dispatch_not_mem inChecks).elim
  · intro obligations
    exact Bool.false_ne_true obligations.initialState

/-- Exhaustive, disjoint classification of the general base family. -/
theorem family_ledger (family : Family) :
    (family ∈ checks ∨ family ∈ eliminated) ∧
      ¬(family ∈ checks ∧ family ∈ eliminated) :=
  ⟨classified family, classification_disjoint family⟩

/-- Every retained fixed-one base family has a concrete removal witness. -/
theorem retained_necessary
    (family : Family)
    (member : family ∈ checks) :
    CheckPlan.NecessaryForSoundness
      (ObligationPlan.semantics Baseline.machine Baseline.selected)
      (ObligationPlan.target Baseline.machine Baseline.selected)
      checks family := by
  cases family with
  | iterationZero => exact iterationZero_necessary
  | initialState => exact initialState_necessary
  | dispatch => exact (dispatch_not_mem member).elim

/-- The fixed-one base plan is sound and every retained leaf is
inclusion-necessary. This is not a physical row lower bound. -/
theorem inclusionMinimalSound :
    CheckPlan.InclusionMinimalSound
      (ObligationPlan.semantics Baseline.machine Baseline.selected)
      (ObligationPlan.target Baseline.machine Baseline.selected)
      checks := by
  apply CheckPlan.inclusionMinimalSound_of_witnesses
  · intro case accepted
    exact (accepts_iff_obligations Baseline.machine Baseline.selected case).1
      accepted
  · exact retained_necessary

end Nightstream.Protocol.FPrime.ConcretePhi81.BaseSemantics.ObligationPlan.FixedOne.Minimality
