import NightstreamFPrime.Lifecycle.RandomOracle.Uniqueness
import NightstreamFPrime.Spec.RandomOracleBudget

/-! The binding reduction of the random-oracle chain as an algorithm, for any fold transcript
`Schedule` and fork game `Fork.Game`: the port of `RandomOracleReduction`.

The reduction sees the adversary only through its run records (`RunRecord`: output and
queries). So every test it makes reads records. `View` is what a base run and its retries show:
the base oracle and output, the retry answers and the retry outputs. `ValidOf` is
`Uniqueness.Valid` on a view (`valid_iff`, by definition), and `forkIndexOf` and `contextOf` are
the fork index and the resampling context computed from a record (`forkIndex_eq`,
`context_eq`).

The machine (`reduction`, a `RunMachine` that queries the adversary with whole oracles) runs the
adversary, reruns it once per `ρ` coordinate until each retry succeeds at the base `ρ` point,
checks the pair (`ValidOf`), then resamples the oracle after the fork index until a resampled
run succeeds at the same index, and extracts a second pair. A machine output (`output`) is the
fold's `kernel` of the two views. `solve` runs the machine for `budget` adversary runs
(`runs_length_le`), and its output depends only on the adversary's records at the oracles it
queries (`solve_congr`). `Solver` bounds its success and work. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Reduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.RandomOracleBudget
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point point)
open NightstreamFPrime.Lifecycle.RandomOracle.Fork (Game)

attribute [local instance low] Classical.propDecidable

/-- What a base run and its retry family show: the base oracle and output, the retry answers
and the retry outputs. -/
structure View (schedule : Schedule) (width : Nat) (Output : Type) where
  oracle : Point schedule width → Answer
  base : Output
  retries : Fin schedule.rhoCount → Answer
  forks : Fin schedule.rhoCount → Output

section Views

variable {schedule : Schedule} {width : Nat} {Output Start : Type}
  (game : Game schedule width Output Start)

/-- The `ρ` point of `index` in the transcript of `output`. -/
noncomputable def rhoPointOf (output : Output) (index : Fin schedule.rhoCount) :
    Point schedule width :=
  point (game.statement output) (game.messages output) (.rho index)

/-- The oracle of the retry at `index`. -/
noncomputable def forkedOf (view : View schedule width Output) (index : Fin schedule.rhoCount) :
    Point schedule width → Answer :=
  Function.update view.oracle (rhoPointOf game view.base index) (view.retries index)

/-- `Fork.ForkValid` on a view. -/
def ForkValidOf (view : View schedule width Output) (index : Fin schedule.rhoCount) : Prop :=
  (game.Succeeds (forkedOf game view index) (view.forks index) ∧
      rhoPointOf game (view.forks index) index = rhoPointOf game view.base index) ∧
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample (view.retries index) ≠
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
        (view.oracle (rhoPointOf game view.base index)) ∧
    game.start (view.forks index) = game.start view.base

/-- `Uniqueness.Valid` on a view. -/
def ValidOf (view : View schedule width Output) : Prop :=
  game.Succeeds view.oracle view.base ∧ ∀ index, ForkValidOf game view index

/-- The record-level calls of the checked game (`Test.checked`). -/
noncomputable def checkedOf (output : Output) (queries : List (Point schedule width)) :
    List (Point schedule width) :=
  queries ++ (Test.challenges schedule).map (point (game.statement output) (game.messages output))

/-- `Uniqueness.forkIndex` from a record. -/
noncomputable def forkIndexOf (output : Output) (queries : List (Point schedule width)) : Nat :=
  (checkedOf game output queries).findIdx (Uniqueness.Extends (game.statement output))

/-- `Uniqueness.context` from a record. -/
noncomputable def contextOf (index : Nat) (output : Output)
    (queries : List (Point schedule width)) : Finset (Point schedule width) :=
  ((checkedOf game output queries).take index).toFinset

section Valid

variable {game} {view : View schedule width Output} {index : Fin schedule.rhoCount}

/-- `Fork.ForkValid.identifies` on a view. -/
theorem ForkValidOf.identifies (valid : ForkValidOf game view index) :
    game.statement (view.forks index) = game.statement view.base ∧
      game.messages (view.forks index) = game.messages view.base :=
  Fork.rhoPoint_identifies index valid.1.2

/-- `Fork.ForkValid.rho_eq` on a view. -/
theorem ForkValidOf.rho_eq (valid : ForkValidOf game view index) (other : Fin schedule.rhoCount) :
    Fork.rho (forkedOf game view index) (game.statement (view.forks index))
        (game.messages (view.forks index)) other =
      if other = index then Fork.readRho (view.retries index)
      else Fork.rho view.oracle (game.statement view.base) (game.messages view.base) other := by
  obtain ⟨statementEq, messagesEq⟩ := valid.identifies
  rw [statementEq, messagesEq]
  exact Fork.rho_update view.oracle _ _ index other (view.retries index)

/-- `Fork.ForkValid.agree` on a view. -/
theorem ForkValidOf.agree (valid : ForkValidOf game view index) {other : Fin schedule.rhoCount}
    (different : other ≠ index) :
    Fork.rho view.oracle (game.statement view.base) (game.messages view.base) other =
      Fork.rho (forkedOf game view index) (game.statement (view.forks index))
        (game.messages (view.forks index)) other := by
  rw [valid.rho_eq other, ite_eq_right different]

/-- `Fork.ForkValid.changed` on a view. -/
theorem ForkValidOf.changed (valid : ForkValidOf game view index) :
    Fork.rho view.oracle (game.statement view.base) (game.messages view.base) index ≠
      Fork.rho (forkedOf game view index) (game.statement (view.forks index))
        (game.messages (view.forks index)) index := by
  rw [valid.rho_eq index, ite_eq_left rfl]
  exact fun same => valid.2.1 ((Fork.readRho_eq_iff _ _).mp same).symm

end Valid

variable (adversary : OracleComp (Point schedule width) Answer Output)

/-- The view of the base oracle `oracle` and the retries `retries`. -/
noncomputable def viewOf (oracle : Point schedule width → Answer)
    (retries : Fin schedule.rhoCount → Answer) : View schedule width Output :=
  ⟨oracle, adversary.run oracle, retries,
    fun index => adversary.run (Fork.forked game adversary oracle retries index)⟩

theorem valid_iff (oracle : Point schedule width → Answer)
    (retries : Fin schedule.rhoCount → Answer) :
    Uniqueness.Valid game adversary oracle retries ↔
      ValidOf game (viewOf game adversary oracle retries) :=
  Iff.rfl

theorem checked_queries (oracle : Point schedule width → Answer) :
    (Test.checked adversary game.statement game.messages).queries oracle =
      checkedOf game (adversary.run oracle) (adversary.queries oracle) :=
  OracleComp.queries_thenQueries _ oracle adversary

theorem forkIndex_eq (oracle : Point schedule width → Answer) :
    Uniqueness.forkIndex game adversary oracle =
      forkIndexOf game (adversary.run oracle) (adversary.queries oracle) := by
  unfold Uniqueness.forkIndex Uniqueness.firstIndex forkIndexOf
  rw [checked_queries]

theorem context_eq (index : Nat) (oracle : Point schedule width → Answer) :
    Uniqueness.context game adversary index oracle =
      contextOf game index (adversary.run oracle) (adversary.queries oracle) := by
  unfold Uniqueness.context queriedBefore contextOf
  rw [checked_queries]

end Views

structure Run (Oracle Record : Type) where
  oracle : Oracle
  record : Record

structure Forks (Coordinate Answer Record : Type) where
  retries : Coordinate → Answer
  records : Coordinate → Record

inductive Purpose (Run Forks : Type)
  | first
  | second (base : Run) (baseForks : Forks)

inductive State (Run Forks Coordinate : Type)
  | start
  | extract (purpose : Purpose Run Forks) (run : Run) (index : Coordinate)
      (rest : List Coordinate) (forks : Forks)
  | rerun (base : Run) (forks : Forks)
  | finished (base : Run) (forks : Forks) (other : Run) (otherForks : Forks)
  | failed

section Machine

variable {schedule : Schedule} {width : Nat} {Output Start Result : Type}
  (game : Game schedule width Output Start)

local notation "Oracle" => Point schedule width → Answer
local notation "Record" => RunRecord (Point schedule width) Output
local notation "Coordinate" => Fin schedule.rhoCount
local notation "Made" => Run Oracle Record
local notation "Retried" => Forks Coordinate Answer Record
local notation "Progress" => State Made Retried Coordinate

/-- The view of a recorded run and its recorded retries. -/
def viewAt (run : Made) (forks : Retried) : View schedule width Output :=
  ⟨run.oracle, run.record.output, forks.retries, fun index => (forks.records index).output⟩

noncomputable def rhoPointAt (run : Made) (index : Coordinate) : Point schedule width :=
  rhoPointOf game run.record.output index

noncomputable def afterExtraction : Purpose Made Retried → Made → Retried → Progress
  | .first, run, forks =>
      if ValidOf game (viewAt run forks) then .rerun run forks else .failed
  | .second base baseForks, run, forks =>
      if ValidOf game (viewAt run forks) then .finished base baseForks run forks
      else .rerun base baseForks

noncomputable def extracted (purpose : Purpose Made Retried) (run : Made) :
    List Coordinate → Retried → Progress
  | [], forks => afterExtraction game purpose run forks
  | index :: rest, forks => .extract purpose run index rest forks

noncomputable def startExtraction (purpose : Purpose Made Retried) (run : Made) : Progress :=
  extracted game purpose run (List.finRange _)
    ⟨fun index => run.oracle (rhoPointAt game run index), fun _ => run.record⟩

/-- The oracle that the machine gives the adversary next: a new answer at one `ρ` point while
it extracts, the base oracle resampled after the fork index while it reruns. -/
noncomputable def query : Progress → Oracle → Oracle
  | .extract _ run index _ _, fresh =>
      Function.update run.oracle (rhoPointAt game run index) (fresh (rhoPointAt game run index))
  | .rerun base _, fresh =>
      overlay (contextOf game (forkIndexOf game base.record.output base.record.queries)
          base.record.output base.record.queries)
        base.oracle fresh
  | _, fresh => fresh

noncomputable def next : Progress → Oracle → Record → Progress
  | .start, fresh, record =>
      if game.Succeeds fresh record.output then startExtraction game .first ⟨fresh, record⟩
      else .failed
  | .extract purpose run index rest forks, fresh, record =>
      if game.Succeeds (query game (.extract purpose run index rest forks) fresh) record.output ∧
          rhoPointOf game record.output index = rhoPointAt game run index
      then extracted game purpose run rest
        ⟨Function.update forks.retries index (fresh (rhoPointAt game run index)),
          Function.update forks.records index record⟩
      else .extract purpose run index rest forks
  | .rerun base forks, fresh, record =>
      if forkIndexOf game record.output record.queries =
          forkIndexOf game base.record.output base.record.queries ∧
        game.Succeeds (query game (.rerun base forks) fresh) record.output
      then startExtraction game (.second base forks) ⟨query game (.rerun base forks) fresh, record⟩
      else .rerun base forks
  | .finished base forks other otherForks, _, _ => .finished base forks other otherForks
  | .failed, _, _ => .failed

/-- The reduction: each step runs the adversary once on `query`'s oracle and reads its
record. -/
noncomputable def reduction : RunMachine Progress Oracle Oracle Record where
  query := query game
  next := next game
  halted
    | .finished .. => true
    | .failed => true
    | _ => false

variable (kernel : View schedule width Output → View schedule width Output → Option Result)

/-- The machine's output: the fold's `kernel` of the two extracted views. -/
noncomputable def output : Progress → Option Result
  | .finished base forks other otherForks => kernel (viewAt base forks) (viewAt other otherForks)
  | _ => none

variable (adversary : OracleComp (Point schedule width) Answer Output)

noncomputable def finalState (budget : ℕ) (tape : Fin budget → Oracle) : Progress :=
  ((reduction game).program budget .start tape).run adversary.record

/-- The oracles on which the machine runs the adversary. -/
noncomputable def runs (budget : ℕ) (tape : Fin budget → Oracle) : List Oracle :=
  ((reduction game).program budget .start tape).queries adversary.record

/-- The solver: the machine with `budget` adversary runs, then its output. -/
noncomputable def solve (budget : ℕ) (tape : Fin budget → Oracle) : Option Result :=
  output kernel (finalState game adversary budget tape)

theorem runs_length_le (budget : ℕ) (tape : Fin budget → Oracle) :
    (runs game adversary budget tape).length ≤ budget :=
  ((reduction game).program_bound budget .start tape).queries_length_le _

theorem runs_take_congr (other : OracleComp (Point schedule width) Answer Output)
    (budget : ℕ) (tape : Fin budget → Oracle) (step : ℕ)
    (agree : ∀ oracle ∈ (runs game adversary budget tape).take step,
      adversary.record oracle = other.record oracle) :
    (runs game other budget tape).take (step + 1) =
      (runs game adversary budget tape).take (step + 1) :=
  OracleComp.take_succ_queries_congr _ _ _ step agree

/-- The solver is black-box: two adversaries with the same records on the solver's runs give
the same output. -/
theorem solve_congr (other : OracleComp (Point schedule width) Answer Output)
    (budget : ℕ) (tape : Fin budget → Oracle)
    (agree : ∀ oracle ∈ runs game adversary budget tape,
      adversary.record oracle = other.record oracle) :
    solve game kernel other budget tape = solve game kernel adversary budget tape := by
  unfold solve finalState
  rw [OracleComp.run_congr _ _ _ agree]

end Machine

end NightstreamFPrime.Lifecycle.RandomOracle.Reduction
