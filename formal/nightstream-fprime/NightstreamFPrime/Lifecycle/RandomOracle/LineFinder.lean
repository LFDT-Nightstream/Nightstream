import Mathlib.Algebra.Order.Archimedean.Real.Basic
import NightstreamFPrime.Lifecycle.RandomOracle.Reduction

/-! A retry finder for one `ρ` coordinate of any fork game (`Fork.Game`): the port of the line
finder of `RandomOracleHashSolver`.

The finder (`lineFinder`, a `RunMachine` that queries the adversary with whole oracles) runs the
adversary until a run succeeds, then reruns it with new answers at that run's `ρ` point of
`index` until a rerun succeeds at the same point. It outputs `collide` of the two outputs
(`lineOutput`), for any record-level test `collide`. `lineSolve` is the finder with `budget`
adversary runs (`lineRuns_length_le`), and its output depends only on the adversary's records at
the oracles it queries (`lineSolve_congr`).

`line_le`: the expected chance that a retry at `index` of a successful run makes `collide` give a
value (`retryCollision`, a `Fork.retryChance`) is at most the finder's chance plus
`(Q + rhoCount + 1) / budget` (times `budget`, as `budget · (chance − finder) ≤ Q + rhoCount + 1`).
The work is one run plus the expected reruns of `Fork.expected_retries_at_le`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.LineFinder

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.RandomOracleBudget
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.RandomOracle.Fork (Game)
open NightstreamFPrime.Lifecycle.RandomOracle.Reduction (Run rhoPointOf rhoPointAt)

attribute [local instance low] Classical.propDecidable

inductive LineState (Run : Type)
  | start
  | retry (base : Run)
  | finished (base other : Run)
  | failed

variable {schedule : Schedule} {width : Nat} {Output Start Result : Type}
  (game : Game schedule width Output Start) (index : Fin schedule.rhoCount)

local notation "Oracle" => Point schedule width → Answer
local notation "Record" => RunRecord (Point schedule width) Output
local notation "Made" => Run Oracle Record

/-- The oracle that the finder gives the adversary next: a new answer at the base run's `ρ`
point of `index` while it retries. -/
noncomputable def lineQuery : LineState Made → Oracle → Oracle
  | .retry base, fresh =>
      Function.update base.oracle (rhoPointAt game base index) (fresh (rhoPointAt game base index))
  | _, fresh => fresh

noncomputable def lineFinder : RunMachine (LineState Made) Oracle Oracle Record where
  query := lineQuery game index
  next
    | .start, fresh, record =>
        if game.Succeeds fresh record.output then .retry ⟨fresh, record⟩ else .failed
    | .retry base, fresh, record =>
        if game.Succeeds (lineQuery game index (.retry base) fresh) record.output ∧
            rhoPointOf game record.output index = rhoPointAt game base index
        then .finished base ⟨lineQuery game index (.retry base) fresh, record⟩
        else .retry base
    | state, _, _ => state
  halted
    | .finished .. => true
    | .failed => true
    | _ => false

variable (collide : Output → Output → Option Result)

/-- The finder's output: `collide` of the base output and the retry output. -/
noncomputable def lineOutput : LineState Made → Option Result
  | .finished base other => collide base.record.output other.record.output
  | _ => none

variable (adversary : OracleComp (Point schedule width) Answer Output)

/-- Named: instance search exceeds its size limit on this type. -/
noncomputable local instance (budget : ℕ) : Fintype (Fin budget → Oracle) := Pi.instFintype

/-- The finder with `budget` adversary runs, then its output. -/
noncomputable def lineSolve (budget : ℕ) (tape : Fin budget → Oracle) : Option Result :=
  lineOutput collide (((lineFinder game index).program budget .start tape).run adversary.record)

/-- The oracles on which the finder runs the adversary. -/
noncomputable def lineRuns (budget : ℕ) (tape : Fin budget → Oracle) : List Oracle :=
  ((lineFinder game index).program budget .start tape).queries adversary.record

theorem lineRuns_length_le (budget : ℕ) (tape : Fin budget → Oracle) :
    (lineRuns game index adversary budget tape).length ≤ budget :=
  ((lineFinder game index).program_bound budget .start tape).queries_length_le _

/-- The finder is black-box: two adversaries with the same records on the finder's runs give the
same output. -/
theorem lineSolve_congr (other : OracleComp (Point schedule width) Answer Output)
    (budget : ℕ) (tape : Fin budget → Oracle)
    (agree : ∀ oracle ∈ lineRuns game index adversary budget tape,
      adversary.record oracle = other.record oracle) :
    lineSolve game index collide other budget tape =
      lineSolve game index collide adversary budget tape := by
  unfold lineSolve
  rw [OracleComp.run_congr _ _ _ agree]

/-- The chance that the finder outputs a value with `budget` adversary runs. -/
noncomputable def lineChance (budget : ℕ) : ℝ :=
  𝔼 tape : Fin budget → Oracle,
    if (lineSolve game index collide adversary budget tape).isSome then 1 else 0

/-- The chance that a retry at `index` of the run at `oracle` makes `collide` give a value. -/
noncomputable def retryCollision (oracle : Oracle) : ℝ :=
  Fork.retryChance game adversary (fun answer =>
    (collide (adversary.run oracle) (adversary.run (Function.update oracle
      (Fork.rhoPoint game adversary index oracle) answer))).isSome) index oracle

private theorem lineAverage_collision (base : Oracle) :
    lineAverage (Fork.retrySets game adversary base index) (fun answer =>
        if (collide (adversary.run base) (adversary.run (Function.update base
          (Fork.rhoPoint game adversary index base) answer))).isSome then 1 else 0) =
      retryCollision game index collide adversary base := by
  have numerator : (𝔼 answer, if answer ∈ Fork.retrySets game adversary base index then
        (if (collide (adversary.run base) (adversary.run (Function.update base
          (Fork.rhoPoint game adversary index base) answer))).isSome then (1 : ℝ) else 0)
        else 0) =
      mass (Fork.retrySets game adversary base index ∩
        {answer | (collide (adversary.run base) (adversary.run (Function.update base
          (Fork.rhoPoint game adversary index base) answer))).isSome}) :=
    Finset.expect_congr rfl fun answer _ => by
      by_cases inside : answer ∈ Fork.retrySets game adversary base index
      · rw [ite_eq_left inside]
        by_cases collides : (collide (adversary.run base) (adversary.run (Function.update base
            (Fork.rhoPoint game adversary index base) answer))).isSome
        · rw [ite_eq_left collides, ite_eq_left ⟨inside, collides⟩]
        · rw [ite_eq_right collides, ite_eq_right fun both => collides both.2]
      · rw [ite_eq_right inside, ite_eq_right fun both => inside both.1]
  exact congrArg (· / mass (Fork.retrySets game adversary base index)) numerator

private noncomputable def runOf (oracle : Oracle) : Made :=
  ⟨oracle, adversary.record oracle⟩

private noncomputable def lineSuccess : LineState Made → ℝ
  | .start => 𝔼 oracle, if Fork.Succeeds game adversary oracle then
      retryCollision game index collide adversary oracle else 0
  | .retry base => retryCollision game index collide adversary base.oracle
  | .finished base other =>
      if (collide base.record.output other.record.output).isSome then 1 else 0
  | .failed => 0

private noncomputable def lineWork : LineState Made → ℝ
  | .start => 1 + 𝔼 oracle, if Fork.Succeeds game adversary oracle then
      retryCount (Fork.Hits game adversary index) (Fork.rhoPoint game adversary index oracle) oracle
    else 0
  | .retry base => retryCount (Fork.Hits game adversary index)
      (Fork.rhoPoint game adversary index base.oracle) base.oracle
  | _ => 0

private def LineGood : LineState Made → Prop
  | .retry base => base = runOf adversary base.oracle ∧ Fork.Succeeds game adversary base.oracle
  | _ => True

private theorem retryCollision_le_one (base : Oracle) :
    retryCollision game index collide adversary base ≤ 1 :=
  div_le_one_of_le₀ (mass_mono Set.inter_subset_left) (mass_nonnegative _)

private noncomputable def lineBounds :
    ((lineFinder game index).rewinding adversary.record).Bounds
      (fun state => (lineOutput collide state).isSome) where
  good := LineGood game adversary
  success := lineSuccess game index collide adversary
  work := lineWork game index adversary
  good_step := by
    intro state good unhalted fresh
    cases state with
    | start =>
        show LineGood game adversary
          (if game.Succeeds fresh (adversary.record fresh).output then
            .retry ⟨fresh, adversary.record fresh⟩ else .failed)
        split
        · rename_i succeeds
          exact ⟨rfl, succeeds⟩
        · trivial
    | retry base =>
        show LineGood game adversary (if _ then _ else _)
        split
        · trivial
        · exact good
    | finished => simp [RunMachine.rewinding, lineFinder] at unhalted
    | failed => simp [RunMachine.rewinding, lineFinder] at unhalted
  success_le_one := by
    intro state _
    cases state with
    | start =>
        refine (Finset.expect_le_expect fun oracle _ => ?_).trans_eq
          (Finset.expect_const Finset.univ_nonempty _)
        split
        · exact retryCollision_le_one game index collide adversary oracle
        · exact zero_le_one
    | retry base => exact retryCollision_le_one game index collide adversary _
    | finished => show (if _ then (1 : ℝ) else 0) ≤ 1; split <;> norm_num
    | failed => show (0 : ℝ) ≤ 1; norm_num
  success_halted := by
    intro state _ halted
    cases state with
    | finished => exact le_rfl
    | failed => show (0 : ℝ) ≤ _; simp [lineOutput]
    | start => simp [RunMachine.rewinding, lineFinder] at halted
    | retry => simp [RunMachine.rewinding, lineFinder] at halted
  success_step := by
    intro state good unhalted
    cases state with
    | start =>
        refine le_of_eq (Finset.expect_congr rfl fun fresh _ => ?_)
        show _ = lineSuccess game index collide adversary
          (if game.Succeeds fresh (adversary.record fresh).output then
            .retry ⟨fresh, adversary.record fresh⟩ else .failed)
        rw [apply_ite (lineSuccess game index collide adversary)]
        exact if_congr Iff.rfl rfl rfl
    | retry base =>
        obtain ⟨recorded, succeeds⟩ := good
        rw [recorded]
        let run := base.oracle
        let point := Fork.rhoPoint game adversary index run
        let set := Fork.retrySets game adversary run index
        let value : Answer → ℝ := fun answer =>
          if (collide (adversary.run run)
            (adversary.run (Function.update run point answer))).isSome then 1 else 0
        have positive : mass set ≠ 0 :=
          Fork.mass_ne_zero (Fork.base_mem_retrySets game adversary succeeds index)
        have pointwise (fresh : Oracle) :
            lineSuccess game index collide adversary
                (((lineFinder game index).rewinding adversary.record).step
                  (.retry (runOf adversary run)) fresh) =
              (fun answer => if answer ∈ set then value answer else lineAverage set value)
                (fresh point) := by
          show lineSuccess game index collide adversary (if _ then _ else _) = _
          rw [apply_ite (lineSuccess game index collide adversary)]
          exact if_congr Iff.rfl rfl
            (lineAverage_collision game index collide adversary run).symm
        calc
          _ = lineAverage set value := (lineAverage_collision game index collide adversary run).symm
          _ = _ := (lineAverage_step set positive value).symm
          _ = _ := (expect_eval point _).symm
          _ = _ := Finset.expect_congr rfl fun fresh _ => (pointwise fresh).symm
          _ ≤ _ := le_rfl
    | finished => simp [RunMachine.rewinding, lineFinder] at unhalted
    | failed => simp [RunMachine.rewinding, lineFinder] at unhalted
  work_nonnegative := by
    intro state _
    cases state with
    | start =>
        refine add_nonneg zero_le_one (Finset.expect_nonneg fun oracle _ => ?_)
        split
        · exact div_nonneg zero_le_one (mass_nonnegative _)
        · exact le_rfl
    | retry => exact div_nonneg zero_le_one (mass_nonnegative _)
    | finished => exact le_rfl
    | failed => exact le_rfl
  work_step := by
    intro state good unhalted
    cases state with
    | start =>
        refine le_of_eq (congrArg (1 + ·) (Finset.expect_congr rfl fun fresh _ => ?_))
        show lineWork game index adversary
          (if game.Succeeds fresh (adversary.record fresh).output then
            .retry ⟨fresh, adversary.record fresh⟩ else .failed) = _
        rw [apply_ite (lineWork game index adversary)]
        exact if_congr Iff.rfl rfl rfl
    | retry base =>
        obtain ⟨recorded, succeeds⟩ := good
        rw [recorded]
        let run := base.oracle
        let point := Fork.rhoPoint game adversary index run
        let set := Fork.retrySets game adversary run index
        have positive : mass set ≠ 0 :=
          Fork.mass_ne_zero (Fork.base_mem_retrySets game adversary succeeds index)
        have empty : lineAverage set (fun _ => 0) = 0 := by simp [lineAverage]
        let attempt : Answer → ℝ := fun answer =>
          if answer ∈ set then 0 + 0 else 1 / mass set + 0 + lineAverage set fun _ => 0
        have pointwise (fresh : Oracle) :
            lineWork game index adversary
                (((lineFinder game index).rewinding adversary.record).step
                  (.retry (runOf adversary run)) fresh) = attempt (fresh point) := by
          show lineWork game index adversary (if _ then _ else _) = _
          rw [apply_ite (lineWork game index adversary)]
          refine if_congr Iff.rfl (by simp only [lineWork, add_zero]) ?_
          rw [empty, add_zero, add_zero]
          rfl
        calc
          _ = 1 + 𝔼 fresh : Oracle, attempt (fresh point) :=
            congrArg (1 + ·) (Finset.expect_congr rfl fun fresh _ => pointwise fresh)
          _ = 1 + 𝔼 answer, attempt answer := congrArg (1 + ·) (expect_eval point attempt)
          _ = 1 / mass set + 0 + lineAverage set (fun _ => 0) :=
            line_work_step set positive 0 fun _ => 0
          _ = 1 / mass set := by rw [empty, add_zero, add_zero]
          _ ≤ _ := le_rfl
    | finished => simp [RunMachine.rewinding, lineFinder] at unhalted
    | failed => simp [RunMachine.rewinding, lineFinder] at unhalted

/-- The expected retry collision chance of a successful run is at most the finder's chance plus
`(Q + rhoCount + 1) / budget`. -/
theorem line_le {queries : Nat} (bounded : adversary.QueryBound queries) (budget : ℕ) :
    budget * (𝔼 oracle, (if Fork.Succeeds game adversary oracle then
          retryCollision game index collide adversary oracle else 0) -
        lineChance game index collide adversary budget) ≤
      queries + schedule.rhoCount + 1 := by
  have loss := (lineBounds game index collide adversary).budget_mul_le budget
    (state := .start) trivial
  have chance : ((lineFinder game index).rewinding adversary.record).chance
      (fun state => (lineOutput collide state).isSome) budget .start =
        lineChance game index collide adversary budget :=
    Finset.expect_congr rfl fun tape _ => by
      rw [← ((lineFinder game index).run_program adversary.record budget .start tape).1]
      rfl
  have retries := Fork.expected_retries_at_le game adversary bounded index
  push_cast at retries
  rw [chance] at loss
  refine loss.trans ?_
  show 1 + _ ≤ _
  linarith

end NightstreamFPrime.Lifecycle.RandomOracle.LineFinder
