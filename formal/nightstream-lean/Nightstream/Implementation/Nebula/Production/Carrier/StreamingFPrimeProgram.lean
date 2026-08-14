import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCCS
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLC
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding

/-!
Contract: verifier-owned microstep program for one recursive Nebula F-prime
transition.

Assurance tier: model-level lifecycle authority.

Owns the exact phase order, bounded phase indices, cursor advance, exclusion
of skipped or repeated work, and the production step count for the current
82-state-chunk, 86-claim-chunk, 26-round, 110-family, and
82-successor-prefix-chunk schedule.

Does not own generated rows, a Rust relation, phase-local constraint
semantics, same-assignment conformance, recursive proof integration, or a
final row or column count.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram

/-- Fixed phase selector vocabulary. The numeric code is part of the future
Rust-to-Lean relation boundary. -/
inductive Phase where
  | prelude
  | claimReplay
  | piCcsStart
  | piCcsRound
  | piCcsFinish
  | runningParentPiDec
  | piRlcStart
  | piRlcFamily
  | piRlcFinish
  | piDec
  | pointBinding
  | priorStateReplay
  | nebula
  | accumulator
  | counters
  | output
  | application
  | semanticLinks
  | successorPrefixReplay
  deriving DecidableEq, Inhabited, Repr

def Phase.code : Phase -> Fin 19
  | .prelude => 0
  | .claimReplay => 1
  | .piCcsStart => 2
  | .piCcsRound => 3
  | .piCcsFinish => 4
  | .runningParentPiDec => 5
  | .piRlcStart => 6
  | .piRlcFamily => 7
  | .piRlcFinish => 8
  | .piDec => 9
  | .pointBinding => 10
  | .priorStateReplay => 11
  | .nebula => 12
  | .accumulator => 13
  | .counters => 14
  | .output => 15
  | .application => 16
  | .semanticLinks => 17
  | .successorPrefixReplay => 18

theorem Phase.code_injective : Function.Injective Phase.code := by
  intro left right equal
  cases left <;> cases right <;> simp [Phase.code] at equal ⊢

/-- One verifier-selected unit of work. Only the three repeated phases carry
an index. -/
structure WorkItem where
  phase : Phase
  index : Nat
  deriving DecidableEq, Inhabited, Repr

def singleton (phase : Phase) : WorkItem :=
  { phase, index := 0 }

def indexed (phase : Phase) (count : Nat) : List WorkItem :=
  (List.range count).map fun index => { phase, index }

@[simp] theorem indexed_length (phase : Phase) (count : Nat) :
    (indexed phase count).length = count := by
  simp [indexed]

/-- Geometry that affects the number of microsteps. These values are
verifier-owned relation constants. -/
structure Config where
  priorStateChunks : Nat
  claimChunks : Nat
  piCcsRounds : Nat
  piRlcFamilies : Nat
  successorPrefixChunks : Nat
  deriving DecidableEq, Repr

/-- Exact recursive program order. Fixed phases remain explicit so that the
terminal check cannot accept after only the three streamed cores. -/
def program (config : Config) : List WorkItem :=
  [singleton .prelude] ++
    indexed .priorStateReplay config.priorStateChunks ++
    indexed .claimReplay config.claimChunks ++
    [singleton .piCcsStart] ++
    indexed .piCcsRound config.piCcsRounds ++
    [singleton .piCcsFinish,
      singleton .runningParentPiDec,
      singleton .piRlcStart] ++
    indexed .piRlcFamily config.piRlcFamilies ++
    [singleton .piRlcFinish,
      singleton .piDec,
      singleton .pointBinding,
      singleton .application,
      singleton .counters] ++
    indexed .successorPrefixReplay config.successorPrefixChunks ++
    [singleton .nebula,
      singleton .accumulator,
      singleton .output,
      singleton .semanticLinks]

/-- Fourteen work items are independent of the five repeated phase counts. -/
def fixedWorkItems : Nat := 14

def workItemCount (config : Config) : Nat :=
  config.priorStateChunks + config.claimChunks + config.piCcsRounds +
    config.piRlcFamilies + config.successorPrefixChunks + fixedWorkItems

@[simp] theorem program_length (config : Config) :
    (program config).length = workItemCount config := by
  simp [program, workItemCount, fixedWorkItems]
  omega

/-- Current production schedule: both 83-thousand-field state messages and
the 88,023-field claim use 1,024-field chunks. PiCCS uses 26 SumCheck rounds,
and PiRLC uses 110 narrow output-family phases. -/
def productionConfig : Config where
  priorStateChunks := 82
  claimChunks := 86
  piCcsRounds := 26
  piRlcFamilies := 110
  successorPrefixChunks := 82

theorem production_work_item_count :
    workItemCount productionConfig = 400 := by
  decide

theorem production_program_length :
    (program productionConfig).length = 400 := by
  rw [program_length, production_work_item_count]

/-- Persistent semantic value plus the verifier-checked program cursor. -/
structure Runtime (State : Type) where
  value : State
  cursor : Nat

def initial {State : Type} (value : State) : Runtime State :=
  { value, cursor := 0 }

def Complete {State : Type} (config : Config) (runtime : Runtime State) : Prop :=
  runtime.cursor = (program config).length

/-- One relation step must consume the exact program item at the current
cursor and must advance the cursor by one. -/
def Step {State : Type}
    (semantics : WorkItem -> State -> State -> Prop)
    (config : Config) (before after : Runtime State) : Prop :=
  exists inBounds : before.cursor < (program config).length,
    after.cursor = before.cursor + 1 /\
      semantics ((program config).get ⟨before.cursor, inBounds⟩)
        before.value after.value

theorem Step.cursor_succ {State : Type}
    {semantics : WorkItem -> State -> State -> Prop}
    {config : Config} {before after : Runtime State}
    (step : Step semantics config before after) :
    after.cursor = before.cursor + 1 := by
  exact step.choose_spec.1

/-- The local relation always receives the verifier-owned item at the current
cursor. A prover-supplied phase selector is not authority. -/
theorem Step.uses_exact_work_item {State : Type}
    {semantics : WorkItem -> State -> State -> Prop}
    {config : Config} {before after : Runtime State}
    (step : Step semantics config before after) :
    semantics
      ((program config).get ⟨before.cursor, step.choose⟩)
      before.value after.value := by
  exact step.choose_spec.2

theorem no_step_from_complete {State : Type}
    {semantics : WorkItem -> State -> State -> Prop}
    {config : Config} {before after : Runtime State}
    (complete : Complete config before) :
    ¬ Step semantics config before after := by
  intro step
  rcases step with ⟨inBounds, _⟩
  rw [complete] at inBounds
  omega

/-- Exact finite execution of the fixed program. -/
inductive Runs {State : Type}
    (semantics : WorkItem -> State -> State -> Prop)
    (config : Config) : Nat -> Runtime State -> Runtime State -> Prop where
  | nil (runtime : Runtime State) : Runs semantics config 0 runtime runtime
  | cons {steps : Nat} {before middle after : Runtime State}
      (head : Step semantics config before middle)
      (tail : Runs semantics config steps middle after) :
      Runs semantics config (steps + 1) before after

theorem Runs.cursor_exact {State : Type}
    {semantics : WorkItem -> State -> State -> Prop}
    {config : Config} {steps : Nat} {before after : Runtime State}
    (run : Runs semantics config steps before after) :
    after.cursor = before.cursor + steps := by
  induction run with
  | nil => simp
  | @cons steps before middle after head tail inductionHypothesis =>
      rw [inductionHypothesis, head.cursor_succ]
      omega

/-- A terminally accepted execution from the initial cursor has consumed
every work item exactly once. It cannot stop early or add an extra step. -/
theorem complete_run_steps_exact {State : Type}
    {semantics : WorkItem -> State -> State -> Prop}
    {config : Config} {steps : Nat} {startValue : State}
    {after : Runtime State}
    (run : Runs semantics config steps (initial startValue) after)
    (complete : Complete config after) :
    steps = (program config).length := by
  have cursorExact := run.cursor_exact
  simp only [initial] at cursorExact
  unfold Complete at complete
  omega

theorem production_complete_run_steps_exact {State : Type}
    {semantics : WorkItem -> State -> State -> Prop}
    {steps : Nat} {startValue : State} {after : Runtime State}
    (run : Runs semantics productionConfig steps (initial startValue) after)
    (complete : Complete productionConfig after) :
    steps = 400 := by
  rw [complete_run_steps_exact run complete, production_program_length]

end Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
