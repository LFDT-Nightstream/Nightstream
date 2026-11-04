import NightstreamFPrime.Lifecycle.RandomOracle.Reduction

/-! Success and work of the binding reduction (`Reduction`) for any fold transcript `Schedule`
and fork game `Fork.Game`: the port of `RandomOracleSolver`.

The proof follows the machine with an ideal law (`law`, a `Rewinding` on oracles): it draws
each retry from the retry sets (`Fork.retrySets`) and each resampled oracle from the uniform
law after the fork index. The law's success function at the start is
`Uniqueness.reductionChance` of an outcome, and its work function is at most
`work schedule Q = 1 + 2 · rhoCount · (Q + rhoCount) + (Q + challenge count)`: the expected
retries of the first pair (`Fork.expected_retries_le`), the expected reruns until a resampled
pair is valid (`expected_reruns_le`), and the expected retries of the second pair
(`expected_second_retries_le`). The machine simulates the law (`simulation`), so
`Rewinding.Bounds.budget_mul_le` bounds what `budget` adversary runs lose.

`reductionChance_le_solver`: for a fold kernel `kernel` on views, the reduction chance of the
outcome "the kernel of the two views exists" is at most the chance that `solve` outputs a value
plus `work schedule Q / budget` (times `budget`, as `budget · (chance − solver) ≤ work`).
`expected_runs_le`: the solver runs the adversary at most `work schedule Q` times in
expectation. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Solver

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.RandomOracleBudget
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.RandomOracle.Fork (Game)
open NightstreamFPrime.Lifecycle.RandomOracle.Fork
  (Succeeds Hits rhoPoint forked retrySets mass_ne_zero base_mem_retrySets expected_retries_le)
open NightstreamFPrime.Lifecycle.RandomOracle.Uniqueness
  (Valid weight retryMass success resampledSuccess retryChance reductionChance forkIndex context
    context_stopping retryMass_nonnegative resampledSuccess_nonnegative resampledSuccess_pos
    retryChance_le_one reductionChance_le_one success_eq sum_success forkIndex_lt_bound)
open NightstreamFPrime.Lifecycle.RandomOracle.Reduction

attribute [local instance low] Classical.propDecidable

/-- The work bound of the reduction: `1 + 2 · rhoCount · (Q + rhoCount) + (Q + challenge
count)` adversary runs. -/
def work (schedule : Schedule) (queries : Nat) : Nat :=
  1 + 2 * schedule.rhoCount * (queries + schedule.rhoCount) +
    (queries + (Test.challenges schedule).length)

variable {schedule : Schedule} {width : Nat} {Output Start : Type}
  (game : Game schedule width Output Start)
  (adversary : OracleComp (Point schedule width) Answer Output)

local notation "Oracle" => Point schedule width → Answer
local notation "Coordinate" => Fin schedule.rhoCount
local notation "Retries" => Coordinate → Answer
local notation "LawState" => State Oracle Retries Coordinate
local notation "Record" => RunRecord (Point schedule width) Output
local notation "Made" => Run Oracle Record
local notation "Retried" => Forks Coordinate Answer Record
local notation "Progress" => State Made Retried Coordinate

/-- Named: instance search exceeds its size limit on this type. -/
noncomputable local instance (budget : ℕ) : Fintype (Fin budget → Oracle) := Pi.instFintype

/-- The expected number of resamplings until a resampled pair is valid at the base fork
index: at most `Q + challenge count`. -/
theorem expected_reruns_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, retryMass game adversary oracle (Valid game adversary oracle) /
        resampledSuccess game adversary (forkIndex game adversary oracle) oracle ≤
      ((queries + (Test.challenges schedule).length : Nat) : ℝ) := by
  have spread (oracle : Oracle) :=
    sum_success game adversary bounded (fun _ _ => (1 : ℝ)) oracle
  simp only [mul_one] at spread
  simp only [spread]
  rw [Finset.expect_sum_comm]
  calc
    _ ≤ ∑ _index ∈ Finset.range (queries + (Test.challenges schedule).length), (1 : ℝ) :=
      Finset.sum_le_sum fun index _ =>
        (expect_div_resampled (context game adversary index)
          (context_stopping game adversary index) (success game adversary index)
          (fun _ => (1 : ℝ)) fun _ _ => rfl).le.trans
        ((Finset.expect_le_expect fun _ _ => by split <;> norm_num).trans
          (Finset.expect_const Finset.univ_nonempty (1 : ℝ)).le)
    _ = _ := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]

private noncomputable def lawAfter :
    Purpose Oracle Retries → Oracle → Retries → LawState
  | .first, run, retries =>
      if Valid game adversary run retries then .rerun run retries else .failed
  | .second base baseRetries, run, retries =>
      if Valid game adversary run retries then .finished base baseRetries run retries
      else .rerun base baseRetries

private noncomputable def lawExtracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    List Coordinate → Retries → LawState
  | [], retries => lawAfter game adversary purpose run retries
  | index :: rest, retries => .extract purpose run index rest retries

private noncomputable def rerunOracle (base fresh : Oracle) : Oracle :=
  overlay (context game adversary (forkIndex game adversary base) base) base fresh

private noncomputable def lawStart (purpose : Purpose Oracle Retries) (run : Oracle) :
    LawState :=
  lawExtracted game adversary purpose run (List.finRange _)
    fun index => run (rhoPoint game adversary index run)

private noncomputable def step : LawState → Oracle → LawState
  | .start, fresh =>
      if Succeeds game adversary fresh then
        lawStart game adversary .first fresh
      else .failed
  | .extract purpose run index rest retries, fresh =>
      if fresh (rhoPoint game adversary index run) ∈ retrySets game adversary run index
      then lawExtracted game adversary purpose run rest
        (Function.update retries index (fresh (rhoPoint game adversary index run)))
      else .extract purpose run index rest retries
  | .rerun base retries, fresh =>
      if forkIndex game adversary (rerunOracle game adversary base fresh) =
          forkIndex game adversary base ∧
        Succeeds game adversary (rerunOracle game adversary base fresh)
      then lawStart game adversary (.second base retries)
        (rerunOracle game adversary base fresh)
      else .rerun base retries
  | .finished base retries other otherRetries, _ => .finished base retries other otherRetries
  | .failed, _ => .failed

private noncomputable def law : Rewinding LawState Oracle where
  step := step game adversary
  halted
    | .finished .. => true
    | .failed => true
    | _ => false

private noncomputable def retriesOf (run : Oracle) : ℝ :=
  ∑ coordinate, retryCount (Hits game adversary coordinate)
    (rhoPoint game adversary coordinate run) run

private noncomputable def extractionWork (index : Nat) (run : Oracle) : ℝ :=
  if forkIndex game adversary run = index ∧ Succeeds game adversary run then
    retriesOf game adversary run
  else 0

noncomputable def rerunRetries (index : Nat) (oracle : Oracle) : ℝ :=
  𝔼 fresh, extractionWork game adversary index
    (overlay (context game adversary index oracle) oracle fresh)

private theorem retriesOf_nonnegative (run : Oracle) : 0 ≤ retriesOf game adversary run :=
  Finset.sum_nonneg fun _ _ => div_nonneg zero_le_one (mass_nonnegative _)

private theorem extractionWork_nonnegative (index : Nat) (run : Oracle) :
    0 ≤ extractionWork game adversary index run := by
  unfold extractionWork
  split
  · exact retriesOf_nonnegative game adversary run
  · exact le_rfl

private theorem rerunRetries_nonnegative (index : Nat) (oracle : Oracle) :
    0 ≤ rerunRetries game adversary index oracle :=
  Finset.expect_nonneg fun _ _ => extractionWork_nonnegative game adversary index _

theorem expected_second_retries_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, retryMass game adversary oracle (Valid game adversary oracle) *
        rerunRetries game adversary (forkIndex game adversary oracle) oracle /
        resampledSuccess game adversary (forkIndex game adversary oracle) oracle ≤
      schedule.rhoCount * ((queries + schedule.rhoCount : Nat) : ℝ) := by
  simp only [sum_success game adversary bounded (rerunRetries game adversary)]
  rw [Finset.expect_sum_comm]
  have each (index : Nat) :
      𝔼 oracle, rerunRetries game adversary index oracle *
          success game adversary index oracle /
          resampledSuccess game adversary index oracle ≤
        𝔼 oracle, extractionWork game adversary index oracle := by
    have determined (oracle fresh : Oracle) :
        rerunRetries game adversary index
            (overlay (context game adversary index oracle) oracle fresh) =
          rerunRetries game adversary index oracle := by
      unfold rerunRetries
      simp only [(overlay_overlay (context game adversary index)
        (context_stopping game adversary index) oracle fresh _).2]
    calc
      _ = 𝔼 oracle, (if 𝔼 fresh, success game adversary index
              (overlay (context game adversary index oracle) oracle fresh) = 0 then 0
            else rerunRetries game adversary index oracle) :=
        expect_div_resampled (context game adversary index)
          (context_stopping game adversary index) (success game adversary index)
          (rerunRetries game adversary index) determined
      _ ≤ 𝔼 oracle, rerunRetries game adversary index oracle :=
        Finset.expect_le_expect fun oracle _ => by
          split
          · exact rerunRetries_nonnegative game adversary index oracle
          · exact le_rfl
      _ = _ := (expect_resample (context game adversary index)
          (context_stopping game adversary index)
          (extractionWork game adversary index)).symm
  have collapse (oracle : Oracle) :
      ∑ index ∈ Finset.range (queries + (Test.challenges schedule).length),
          extractionWork game adversary index oracle =
        if Succeeds game adversary oracle then retriesOf game adversary oracle
        else 0 := by
    rw [Finset.sum_eq_single (forkIndex game adversary oracle)]
    · simp [extractionWork]
    · intro index _ different
      simp [extractionWork, Ne.symm different]
    · intro outside
      exact absurd (Finset.mem_range.mpr (forkIndex_lt_bound game adversary bounded oracle)) outside
  calc
    _ ≤ ∑ index ∈ Finset.range (queries + (Test.challenges schedule).length),
          𝔼 oracle, extractionWork game adversary index oracle :=
      Finset.sum_le_sum fun index _ => each index
    _ = 𝔼 oracle, (if Succeeds game adversary oracle then
          retriesOf game adversary oracle else 0) := by
      rw [← Finset.expect_sum_comm]
      exact Finset.expect_congr rfl fun oracle _ => collapse oracle
    _ ≤ _ := expected_retries_le game adversary bounded

variable (outcome : (Point schedule width → Answer) → (Fin schedule.rhoCount → Answer) →
  (Point schedule width → Answer) → (Fin schedule.rhoCount → Answer) → Prop)

@[implicit_reducible]
private def Found : LawState → Prop
  | .finished base retries other otherRetries => outcome base retries other otherRetries
  | _ => False

private noncomputable def rerunSuccess (base : Oracle) (retries : Retries) : ℝ :=
  retryChance game adversary (forkIndex game adversary base) base (outcome base retries)

private noncomputable def rerunWork (base : Oracle) : ℝ :=
  (1 + rerunRetries game adversary (forkIndex game adversary base) base) /
    resampledSuccess game adversary (forkIndex game adversary base) base

private noncomputable def successAfter : Purpose Oracle Retries → Oracle → Retries → ℝ
  | .first, run, retries =>
      if Valid game adversary run retries then
        rerunSuccess game adversary outcome run retries else 0
  | .second base baseRetries, run, retries =>
      if Valid game adversary run retries then
        (if outcome base baseRetries run retries then 1 else 0)
      else rerunSuccess game adversary outcome base baseRetries

private noncomputable def workAfter : Purpose Oracle Retries → Oracle → Retries → ℝ
  | .first, run, retries =>
      if Valid game adversary run retries then rerunWork game adversary run
      else 0
  | .second base _, run, retries =>
      if Valid game adversary run retries then 0 else rerunWork game adversary base

private def goodOf : LawState → Prop
  | .extract .first run _ _ _ => Succeeds game adversary run
  | .extract (.second base baseRetries) run _ _ _ =>
      Succeeds game adversary run ∧ Valid game adversary base baseRetries
  | .rerun base retries => Valid game adversary base retries
  | _ => True

private noncomputable def successOf : LawState → ℝ
  | .start => reductionChance game adversary outcome
  | .extract purpose run index rest retries =>
      lineLaw (retrySets game adversary run)
        (successAfter game adversary outcome purpose run) (index :: rest) retries
  | .rerun base retries => rerunSuccess game adversary outcome base retries
  | .finished base retries other otherRetries =>
      if outcome base retries other otherRetries then 1 else 0
  | .failed => 0

private noncomputable def workOf : LawState → ℝ
  | .start => 1 + 𝔼 oracle, (if Succeeds game adversary oracle then
        retriesOf game adversary oracle +
          retryMass game adversary oracle (Valid game adversary oracle) *
            rerunWork game adversary oracle
      else 0)
  | .extract purpose run index rest retries =>
      ((index :: rest).map fun coordinate => retryCount (Hits game adversary coordinate)
          (rhoPoint game adversary coordinate run) run).sum +
        lineLaw (retrySets game adversary run)
          (workAfter game adversary purpose run) (index :: rest) retries
  | .rerun base _ => rerunWork game adversary base
  | _ => 0

private theorem weight_sum {run : Oracle} (succeeds : Succeeds game adversary run) :
    ∑ retries, weight game adversary run retries = 1 :=
  retryWeight_sum_eq_one _ fun index =>
    mass_ne_zero (base_mem_retrySets game adversary succeeds index)

private theorem sum_weight_ite {run : Oracle} (succeeds : Succeeds game adversary run)
    (event : Retries → Prop) (value : Retries → ℝ) (other : ℝ) :
    ∑ retries, weight game adversary run retries *
        (if event retries then value retries else other) =
      ∑ retries, weight game adversary run retries *
          (if event retries then value retries else 0) +
        (1 - retryMass game adversary run event) * other := by
  have split (retries : Retries) :
      weight game adversary run retries * (if event retries then value retries else other) =
        weight game adversary run retries * (if event retries then value retries else 0) +
          (weight game adversary run retries -
            weight game adversary run retries * (if event retries then 1 else 0)) * other := by
    split <;> ring
  simp only [split]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_sub_distrib, weight_sum game adversary succeeds]
  rfl

private theorem retryMass_eq_zero {run : Oracle} {event : Retries → Prop} (never : ∀ retries, ¬ event retries) :
    retryMass game adversary run event = 0 :=
  Finset.sum_eq_zero fun retries _ => by simp [never retries]

private theorem successOf_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries, successOf game adversary outcome
        (lawExtracted game adversary purpose run pending retries) =
      lineLaw (retrySets game adversary run)
        (successAfter game adversary outcome purpose run) pending retries
  | [], retries => by
      cases purpose <;>
      · simp only [lawExtracted, lawAfter, lineLaw, successAfter]
        by_cases valid : Valid game adversary run retries
        · simp only [valid, ite_true]
          rfl
        · simp only [valid, ite_false]
          rfl
  | _ :: _, _ => rfl

private theorem workOf_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries, workOf game adversary
        (lawExtracted game adversary purpose run pending retries) =
      (pending.map fun coordinate => retryCount (Hits game adversary coordinate)
          (rhoPoint game adversary coordinate run) run).sum +
        lineLaw (retrySets game adversary run)
          (workAfter game adversary purpose run) pending retries
  | [], retries => by
      cases purpose <;> simp only [lawExtracted, lawAfter, lineLaw, workAfter, List.map_nil, List.sum_nil,
        zero_add] <;> split_ifs <;> rfl
  | _ :: _, _ => rfl

private def Started : Purpose Oracle Retries → Oracle → Prop
  | .first, run => Succeeds game adversary run
  | .second base baseRetries, run =>
      Succeeds game adversary run ∧ Valid game adversary base baseRetries

private theorem goodOf_extracted {purpose : Purpose Oracle Retries} {run : Oracle}
    (started : Started game adversary purpose run) :
    ∀ pending retries, goodOf game adversary
      (lawExtracted game adversary purpose run pending retries)
  | [], retries => by
      cases purpose with
      | first =>
          simp only [lawExtracted, lawAfter]
          split_ifs with valid
          · exact valid
          · trivial
      | second base baseRetries =>
          simp only [lawExtracted, lawAfter]
          split_ifs
          · trivial
          · exact started.2
  | _ :: _, _ => by cases purpose <;> exact started

private theorem started_succeeds {purpose : Purpose Oracle Retries} {run : Oracle}
    (started : Started game adversary purpose run) :
    Succeeds game adversary run := by
  cases purpose with
  | first => exact started
  | second => exact started.1

private theorem successOf_start {purpose : Purpose Oracle Retries} {run : Oracle} :
    successOf game adversary outcome (lawStart game adversary purpose run) =
      ∑ retries, weight game adversary run retries *
        successAfter game adversary outcome purpose run retries := by
  unfold lawStart
  rw [successOf_extracted, lineLaw_eq_retryWeight]
  rfl

private theorem workOf_start {purpose : Purpose Oracle Retries} {run : Oracle} :
    workOf game adversary (lawStart game adversary purpose run) =
      retriesOf game adversary run +
        ∑ retries, weight game adversary run retries *
          workAfter game adversary purpose run retries := by
  unfold lawStart
  rw [workOf_extracted, lineLaw_eq_retryWeight, retriesOf, Fin.sum_univ_def]
  rfl

private theorem success_start_step :
    successOf game adversary outcome .start =
      𝔼 fresh, successOf game adversary outcome (step game adversary .start fresh) := by
  conv_lhs => unfold successOf
  unfold reductionChance
  refine Finset.expect_congr rfl fun oracle _ => ?_
  simp only [step]
  by_cases succeeds : Succeeds game adversary oracle
  · rw [ite_eq_left succeeds, successOf_start]
    rfl
  · rw [ite_eq_right succeeds]
    exact Finset.sum_eq_zero fun retries _ => by
      rw [ite_eq_right fun valid => succeeds valid.1, mul_zero]

private theorem work_start_step :
    1 + 𝔼 fresh, workOf game adversary (step game adversary .start fresh) =
      workOf game adversary .start := by
  conv_rhs => unfold workOf
  congr 1
  refine Finset.expect_congr rfl fun oracle _ => ?_
  simp only [step]
  by_cases succeeds : Succeeds game adversary oracle
  · rw [ite_eq_left succeeds, ite_eq_left succeeds, workOf_start]
    refine congrArg (retriesOf game adversary oracle + ·) ?_
    unfold retryMass
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun retries _ => ?_
    simp only [workAfter]
    split <;> ring
  · rw [ite_eq_right succeeds, ite_eq_right succeeds]
    rfl

private theorem success_extract_step {purpose : Purpose Oracle Retries} {run : Oracle} {index : Coordinate}
    {rest : List Coordinate} {retries : Retries} (started : Started game adversary purpose run) :
    successOf game adversary outcome (.extract purpose run index rest retries) =
      𝔼 fresh, successOf game adversary outcome
        (step game adversary (.extract purpose run index rest retries) fresh) := by
  have positive := mass_ne_zero
    (base_mem_retrySets game adversary (started_succeeds game adversary started) index)
  let next : Answer → ℝ := fun answer =>
    lineLaw (retrySets game adversary run) (successAfter game adversary outcome purpose run)
      rest (Function.update retries index answer)
  have pointwise (fresh : Oracle) :
      successOf game adversary outcome
          (step game adversary (.extract purpose run index rest retries) fresh) =
        (fun answer => if answer ∈ retrySets game adversary run index then next answer
          else lineAverage (retrySets game adversary run index) next)
          (fresh (rhoPoint game adversary index run)) := by
    simp only [step]
    split
    · rw [successOf_extracted]
    · rfl
  calc
    _ = lineAverage (retrySets game adversary run index) next := rfl
    _ = _ := (lineAverage_step _ positive next).symm
    _ = _ := (expect_eval (rhoPoint game adversary index run) _).symm
    _ = _ := Finset.expect_congr rfl fun fresh _ => (pointwise fresh).symm

private theorem work_extract_step {purpose : Purpose Oracle Retries} {run : Oracle} {index : Coordinate}
    {rest : List Coordinate} {retries : Retries} (started : Started game adversary purpose run) :
    1 + 𝔼 fresh, workOf game adversary
        (step game adversary (.extract purpose run index rest retries) fresh) =
      workOf game adversary (.extract purpose run index rest retries) := by
  have positive := mass_ne_zero
    (base_mem_retrySets game adversary (started_succeeds game adversary started) index)
  let later : ℝ := (rest.map fun coordinate => retryCount (Hits game adversary coordinate)
    (rhoPoint game adversary coordinate run) run).sum
  let next : Answer → ℝ := fun answer =>
    lineLaw (retrySets game adversary run) (workAfter game adversary purpose run)
      rest (Function.update retries index answer)
  let attempt : Answer → ℝ := fun answer =>
    if answer ∈ retrySets game adversary run index then later + next answer
    else 1 / mass (retrySets game adversary run index) + later +
      lineAverage (retrySets game adversary run index) next
  have total : workOf game adversary (.extract purpose run index rest retries) =
      1 / mass (retrySets game adversary run index) + later +
        lineAverage (retrySets game adversary run index) next := by
    simp only [workOf, List.map_cons, List.sum_cons, lineLaw]
    unfold retryCount
    rfl
  have pointwise (fresh : Oracle) :
      workOf game adversary
          (step game adversary (.extract purpose run index rest retries) fresh) =
        attempt (fresh (rhoPoint game adversary index run)) := by
    simp only [step, attempt]
    split
    · rw [workOf_extracted]
    · exact total
  calc
    _ = 1 + 𝔼 fresh : Oracle, attempt (fresh (rhoPoint game adversary index run)) :=
      congrArg (1 + ·) (Finset.expect_congr rfl fun fresh _ => pointwise fresh)
    _ = 1 + 𝔼 answer, attempt answer :=
      congrArg (1 + ·) (expect_eval (rhoPoint game adversary index run) attempt)
    _ = _ := line_work_step _ positive later next
    _ = _ := total.symm

private theorem rerun_missed {base fresh : Oracle}
    (missed : ¬ (forkIndex game adversary (rerunOracle game adversary base fresh) =
        forkIndex game adversary base ∧
      Succeeds game adversary (rerunOracle game adversary base fresh)))
    (event : Retries → Prop) :
    retryMass game adversary (rerunOracle game adversary base fresh)
        (fun otherRetries => forkIndex game adversary (rerunOracle game adversary base fresh) =
            forkIndex game adversary base ∧
          Valid game adversary (rerunOracle game adversary base fresh) otherRetries ∧
          event otherRetries) = 0 ∧
      success game adversary (forkIndex game adversary base)
        (rerunOracle game adversary base fresh) = 0 := by
  refine ⟨retryMass_eq_zero game adversary fun _ holds => missed ⟨holds.1, holds.2.1.1⟩, ?_⟩
  rw [success_eq]
  exact retryMass_eq_zero game adversary fun _ holds => missed ⟨holds.1, holds.2.1⟩

private theorem rerun_tries {base fresh : Oracle}
    (forked : forkIndex game adversary (rerunOracle game adversary base fresh) =
      forkIndex game adversary base) :
    success game adversary (forkIndex game adversary base)
        (rerunOracle game adversary base fresh) =
      retryMass game adversary (rerunOracle game adversary base fresh)
        (Valid game adversary (rerunOracle game adversary base fresh)) := by
  rw [success_eq]
  unfold retryMass
  exact Finset.sum_congr rfl fun _ _ => by simp [forked]

private theorem success_rerun_step {base : Oracle} {retries : Retries}
    (valid : Valid game adversary base retries) :
    successOf game adversary outcome (.rerun base retries) =
      𝔼 fresh, successOf game adversary outcome
        (step game adversary (.rerun base retries) fresh) := by
  let found : Oracle → ℝ := fun fresh =>
    retryMass game adversary (rerunOracle game adversary base fresh)
      fun otherRetries => forkIndex game adversary (rerunOracle game adversary base fresh) =
          forkIndex game adversary base ∧
        Valid game adversary (rerunOracle game adversary base fresh) otherRetries ∧
        outcome base retries (rerunOracle game adversary base fresh) otherRetries
  let tries : Oracle → ℝ := fun fresh =>
    success game adversary (forkIndex game adversary base)
      (rerunOracle game adversary base fresh)
  have positive : 𝔼 fresh, tries fresh ≠ 0 := (resampledSuccess_pos game adversary valid).ne'
  have chance : successOf game adversary outcome (.rerun base retries) =
      (𝔼 fresh, found fresh) / 𝔼 fresh, tries fresh := rfl
  have pointwise (fresh : Oracle) :
      successOf game adversary outcome (step game adversary (.rerun base retries) fresh) =
        found fresh + (1 - tries fresh) * successOf game adversary outcome (.rerun base retries) := by
    simp only [step]
    split
    · rename_i forked
      have foundEq : found fresh =
          ∑ otherRetries, weight game adversary (rerunOracle game adversary base fresh)
              otherRetries *
            (if Valid game adversary (rerunOracle game adversary base fresh) otherRetries
              then (if outcome base retries (rerunOracle game adversary base fresh) otherRetries
                then 1 else 0)
              else 0) := by
        simp only [found, retryMass]
        exact Finset.sum_congr rfl fun otherRetries _ => by
          by_cases holds : Valid game adversary (rerunOracle game adversary base fresh)
              otherRetries <;> simp [holds, forked.1]
      rw [successOf_start]
      simp only [successAfter]
      rw [sum_weight_ite game adversary forked.2, foundEq]
      simp only [tries, rerun_tries game adversary forked.1]
      rfl
    · rename_i missed
      obtain ⟨noFound, noTries⟩ := rerun_missed game adversary missed
        (outcome base retries (rerunOracle game adversary base fresh))
      simp only [found, tries, noFound, noTries, sub_zero, one_mul, zero_add]
  rw [Finset.expect_congr rfl fun fresh _ => pointwise fresh, chance, restart_success_eq found tries positive]

private theorem work_rerun_step {base : Oracle} {retries : Retries}
    (valid : Valid game adversary base retries) :
    1 + 𝔼 fresh, workOf game adversary
        (step game adversary (.rerun base retries) fresh) =
      workOf game adversary (.rerun base retries) := by
  let extra : Oracle → ℝ := fun fresh =>
    extractionWork game adversary (forkIndex game adversary base)
      (rerunOracle game adversary base fresh)
  let tries : Oracle → ℝ := fun fresh =>
    success game adversary (forkIndex game adversary base)
      (rerunOracle game adversary base fresh)
  have positive : 𝔼 fresh, tries fresh ≠ 0 := (resampledSuccess_pos game adversary valid).ne'
  have total : workOf game adversary (.rerun base retries) =
      (1 + 𝔼 fresh, extra fresh) / 𝔼 fresh, tries fresh := rfl
  have pointwise (fresh : Oracle) :
      workOf game adversary (step game adversary (.rerun base retries) fresh) =
        extra fresh + (1 - tries fresh) * workOf game adversary (.rerun base retries) := by
    simp only [step]
    split
    · rename_i forked
      rw [workOf_start]
      simp only [workAfter]
      rw [sum_weight_ite game adversary forked.2]
      simp only [tries, extra, rerun_tries game adversary forked.1, extractionWork,
        ite_eq_left forked, ite_self, mul_zero, Finset.sum_const_zero, zero_add]
      rfl
    · rename_i missed
      have noTries := (rerun_missed game adversary missed fun _ => True).2
      simp only [extra, tries, extractionWork, ite_eq_right missed, noTries, sub_zero, one_mul, zero_add]
  calc
    _ = 1 + 𝔼 fresh, (extra fresh + (1 - tries fresh) * ((1 + 𝔼 fresh, extra fresh) / 𝔼 fresh, tries fresh)) := by
      rw [Finset.expect_congr rfl fun fresh _ => pointwise fresh, total]
    _ = _ := restart_work_eq extra tries positive
    _ = _ := total.symm

private theorem rerunWork_nonnegative (base : Oracle) : 0 ≤ rerunWork game adversary base :=
  div_nonneg (add_nonneg zero_le_one (rerunRetries_nonnegative game adversary _ _))
    (resampledSuccess_nonnegative game adversary _ _)

private theorem successAfter_le_one (purpose : Purpose Oracle Retries) (run : Oracle) (retries : Retries) :
    successAfter game adversary outcome purpose run retries ≤ 1 := by
  cases purpose <;> simp only [successAfter] <;> split_ifs <;>
    first
    | exact retryChance_le_one game adversary _ _ _
    | norm_num

private theorem workAfter_nonnegative (purpose : Purpose Oracle Retries) (run : Oracle) (retries : Retries) :
    0 ≤ workAfter game adversary purpose run retries := by
  cases purpose <;> simp only [workAfter] <;> split_ifs <;>
    first
    | exact rerunWork_nonnegative game adversary _
    | exact le_rfl

private theorem started_of_good {purpose : Purpose Oracle Retries} {run : Oracle} {index : Coordinate}
    {rest : List Coordinate} {retries : Retries}
    (good : goodOf game adversary (.extract purpose run index rest retries)) :
    Started game adversary purpose run := by
  cases purpose <;> exact good

private noncomputable def bounds :
    (law game adversary).Bounds (Found outcome) where
  good := goodOf game adversary
  success := successOf game adversary outcome
  work := workOf game adversary
  good_step := by
    intro state good unhalted sample
    cases state with
    | start =>
        show goodOf game adversary (step game adversary .start sample)
        simp only [step]
        split
        · rename_i succeeds
          exact goodOf_extracted game adversary (purpose := .first) succeeds _ _
        · trivial
    | extract purpose run index rest retries =>
        show goodOf game adversary
          (step game adversary (.extract purpose run index rest retries) sample)
        simp only [step]
        split
        · exact goodOf_extracted game adversary
            (started_of_good game adversary good) _ _
        · exact good
    | rerun base retries =>
        show goodOf game adversary (step game adversary (.rerun base retries) sample)
        simp only [step]
        split
        · rename_i forked
          exact goodOf_extracted game adversary (purpose := .second base retries)
            ⟨forked.2, good⟩ _ _
        · exact good
    | finished => simp [law] at unhalted
    | failed => simp [law] at unhalted
  success_le_one := by
    intro state _
    cases state with
    | start => exact reductionChance_le_one game adversary outcome
    | extract purpose run index rest retries =>
        exact lineLaw_le _ zero_le_one (successAfter_le_one game adversary outcome purpose run) _ _
    | rerun base retries =>
        show rerunSuccess game adversary outcome base retries ≤ 1
        exact retryChance_le_one game adversary _ _ _
    | finished base retries other otherRetries =>
        show (if outcome base retries other otherRetries then (1 : ℝ) else 0) ≤ 1
        split <;> norm_num
    | failed => show (0 : ℝ) ≤ 1; norm_num
  success_halted := by
    intro state _ halted
    cases state with
    | finished => exact le_rfl
    | failed => show (0 : ℝ) ≤ _; simp [Found]
    | start => simp [law] at halted
    | extract => simp [law] at halted
    | rerun => simp [law] at halted
  success_step := by
    intro state good unhalted
    cases state with
    | start => exact (success_start_step game adversary outcome).le
    | extract purpose run index rest retries =>
        exact (success_extract_step game adversary outcome
          (started_of_good game adversary good)).le
    | rerun base retries => exact (success_rerun_step game adversary outcome good).le
    | finished => simp [law] at unhalted
    | failed => simp [law] at unhalted
  work_nonnegative := by
    intro state _
    cases state with
    | start =>
        refine add_nonneg zero_le_one (Finset.expect_nonneg fun oracle _ => ?_)
        split
        · exact add_nonneg (retriesOf_nonnegative game adversary oracle)
            (mul_nonneg (retryMass_nonnegative game adversary oracle _)
              (rerunWork_nonnegative game adversary oracle))
        · exact le_rfl
    | extract purpose run index rest retries =>
        refine add_nonneg (List.sum_nonneg fun count member => ?_)
          (lineLaw_nonnegative _ (workAfter_nonnegative game adversary purpose run) _ _)
        obtain ⟨coordinate, _, rfl⟩ := List.mem_map.mp member
        exact div_nonneg zero_le_one (mass_nonnegative _)
    | rerun base retries => exact rerunWork_nonnegative game adversary base
    | finished => exact le_rfl
    | failed => exact le_rfl
  work_step := by
    intro state good unhalted
    cases state with
    | start => exact (work_start_step game adversary).le
    | extract purpose run index rest retries =>
        exact (work_extract_step game adversary
          (started_of_good game adversary good)).le
    | rerun base retries => exact (work_rerun_step game adversary good).le
    | finished => simp [law] at unhalted
    | failed => simp [law] at unhalted

private theorem work_start_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    workOf game adversary .start ≤ work schedule queries := by
  have retries : 𝔼 oracle, (if Succeeds game adversary oracle then
      retriesOf game adversary oracle else 0) ≤
      schedule.rhoCount * ((queries + schedule.rhoCount : Nat) : ℝ) :=
    expected_retries_le game adversary bounded
  have reruns := expected_reruns_le game adversary bounded
  have second := expected_second_retries_le game adversary bounded
  have split : 𝔼 oracle, (if Succeeds game adversary oracle then
        retriesOf game adversary oracle +
          retryMass game adversary oracle (Valid game adversary oracle) *
            rerunWork game adversary oracle
      else 0) ≤
      𝔼 oracle, (if Succeeds game adversary oracle then
          retriesOf game adversary oracle else 0) +
        𝔼 oracle, retryMass game adversary oracle (Valid game adversary oracle) /
          resampledSuccess game adversary (forkIndex game adversary oracle) oracle +
        𝔼 oracle, retryMass game adversary oracle (Valid game adversary oracle) *
          rerunRetries game adversary (forkIndex game adversary oracle) oracle /
          resampledSuccess game adversary (forkIndex game adversary oracle) oracle := by
    rw [← Finset.expect_add_distrib, ← Finset.expect_add_distrib]
    refine Finset.expect_le_expect fun oracle _ => ?_
    have mass := retryMass_nonnegative game adversary oracle (Valid game adversary oracle)
    have denominator := resampledSuccess_nonnegative game adversary
      (forkIndex game adversary oracle) oracle
    have extra := rerunRetries_nonnegative game adversary (forkIndex game adversary oracle) oracle
    have first : 0 ≤ retryMass game adversary oracle (Valid game adversary oracle) /
        resampledSuccess game adversary (forkIndex game adversary oracle) oracle :=
      div_nonneg mass denominator
    have later : 0 ≤ retryMass game adversary oracle (Valid game adversary oracle) *
          rerunRetries game adversary (forkIndex game adversary oracle) oracle /
        resampledSuccess game adversary (forkIndex game adversary oracle) oracle :=
      div_nonneg (mul_nonneg mass extra) denominator
    have expand : retryMass game adversary oracle (Valid game adversary oracle) *
          rerunWork game adversary oracle =
        retryMass game adversary oracle (Valid game adversary oracle) /
            resampledSuccess game adversary (forkIndex game adversary oracle) oracle +
          retryMass game adversary oracle (Valid game adversary oracle) *
            rerunRetries game adversary (forkIndex game adversary oracle) oracle /
            resampledSuccess game adversary (forkIndex game adversary oracle) oracle := by
      unfold rerunWork
      ring
    split
    · rw [expand]
      linarith
    · linarith
  show 1 + _ ≤ _
  unfold work
  push_cast at retries reruns second ⊢
  linarith

omit outcome

noncomputable def runOf (oracle : Oracle) : Made :=
  ⟨oracle, adversary.record oracle⟩

noncomputable def forksOf (oracle : Oracle) (retries : Retries) : Retried :=
  ⟨retries, fun index => adversary.record (forked game adversary oracle retries index)⟩

private noncomputable def embedPurpose : Purpose Oracle Retries → Purpose Made Retried
  | .first => .first
  | .second base retries => .second (runOf adversary base) (forksOf game adversary base retries)

@[implicit_reducible]
private noncomputable def embed : LawState → Progress
  | .start => .start
  | .extract purpose run index rest retries =>
      .extract (embedPurpose game adversary purpose) (runOf adversary run) index rest
        (forksOf game adversary run retries)
  | .rerun base retries => .rerun (runOf adversary base) (forksOf game adversary base retries)
  | .finished base retries other otherRetries =>
      .finished (runOf adversary base) (forksOf game adversary base retries)
        (runOf adversary other) (forksOf game adversary other otherRetries)
  | .failed => .failed

private theorem embed_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries,
      extracted game (embedPurpose game adversary purpose) (runOf adversary run) pending
          (forksOf game adversary run retries) =
        embed game adversary (lawExtracted game adversary purpose run pending retries)
  | [], retries => by
      cases purpose <;>
      · simp only [extracted, afterExtraction, lawExtracted, lawAfter]
        rw [apply_ite (embed game adversary)]
        exact if_congr Iff.rfl rfl rfl
  | _ :: _, _ => rfl

private theorem embed_start (purpose : Purpose Oracle Retries) (run : Oracle) :
    startExtraction game (embedPurpose game adversary purpose) (runOf adversary run) =
      embed game adversary (lawStart game adversary purpose run) := by
  have records : (⟨fun index => run (rhoPoint game adversary index run), fun _ => adversary.record run⟩ :
      Retried) = forksOf game adversary run fun index => run (rhoPoint game adversary index run) :=
    congrArg (Forks.mk _) (funext fun index => by simp only [forked, Function.update_eq_self])
  unfold startExtraction lawStart
  rw [← embed_extracted, ← records]
  rfl

private theorem forksOf_update (run : Oracle) (retries : Retries) (index : Coordinate) (answer : Answer) :
    (⟨Function.update retries index answer,
        Function.update (forksOf game adversary run retries).records index
          (adversary.record (Function.update run (rhoPoint game adversary index run) answer))⟩ :
      Retried) = forksOf game adversary run (Function.update retries index answer) := by
  refine congrArg (Forks.mk _) (funext fun other => ?_)
  by_cases same : other = index
  · subst same
    simp only [forksOf, forked, Function.update_self]
  · simp only [forksOf, forked, Function.update_of_ne same]

private theorem query_rerun (base : Oracle) (retries : Retries) (fresh : Oracle) :
    query game (.rerun (runOf adversary base) (forksOf game adversary base retries)) fresh =
      rerunOracle game adversary base fresh := by
  unfold rerunOracle
  rw [context_eq, forkIndex_eq]
  rfl

private theorem embed_step (state : LawState) (fresh : Oracle)
    (unhalted : (law game adversary).halted state = false) :
    ((reduction game).rewinding adversary.record).step (embed game adversary state) fresh =
      embed game adversary (step game adversary state fresh) := by
  cases state with
  | start =>
      show next game .start fresh (adversary.record fresh) = _
      simp only [next, step]
      rw [apply_ite (embed game adversary)]
      exact if_congr Iff.rfl (embed_start game adversary .first fresh) rfl
  | extract purpose run index rest retries =>
      let embedded : Progress := .extract (embedPurpose game adversary purpose) (runOf adversary run)
        index rest (forksOf game adversary run retries)
      show next game embedded fresh (adversary.record (query game embedded fresh)) = _
      simp only [embedded, next, step]
      rw [apply_ite (embed game adversary)]
      refine if_congr Iff.rfl ?_ rfl
      rw [← embed_extracted, ← forksOf_update]
      rfl
  | rerun base retries =>
      let embedded : Progress := .rerun (runOf adversary base) (forksOf game adversary base retries)
      show next game embedded fresh (adversary.record (query game embedded fresh)) = _
      simp only [embedded, next, step]
      rw [query_rerun, apply_ite (embed game adversary)]
      refine if_congr ?_ (embed_start game adversary (.second base retries) _) rfl
      rw [forkIndex_eq game adversary (rerunOracle game adversary base fresh),
        forkIndex_eq game adversary base]
      exact Iff.rfl
  | finished => simp [law] at unhalted
  | failed => simp [law] at unhalted

private theorem simulation (budget : ℕ) (tape : Fin budget → Oracle) :
    finalState game adversary budget tape =
        embed game adversary ((law game adversary).run budget .start tape).final ∧
      (runs game adversary budget tape).length =
        ((law game adversary).run budget .start tape).runs := by
  have same := (law game adversary).run_embed ((reduction game).rewinding adversary.record)
    (embed game adversary) (fun state => by cases state <;> rfl)
    (embed_step game adversary) budget .start tape
  have program := (reduction game).run_program adversary.record budget .start tape
  refine ⟨program.1.trans ?_, program.2.trans ?_⟩
  · exact congrArg Rewinding.Result.final same
  · exact congrArg Rewinding.Result.runs same

theorem expected_runs_le {queries : Nat} (bounded : adversary.QueryBound queries) (budget : ℕ) :
    𝔼 tape : Fin budget → Oracle, ((runs game adversary budget tape).length : ℝ) ≤
      work schedule queries := by
  calc
    _ = (law game adversary).expectedRuns budget .start :=
      Finset.expect_congr rfl fun tape _ => by rw [(simulation game adversary budget tape).2]
    _ ≤ _ := ((bounds game adversary fun _ _ _ _ => True).expect_runs_le budget
        (state := .start) trivial).trans (work_start_le game adversary bounded)

theorem reductionChance_le_program {queries : Nat} (bounded : adversary.QueryBound queries)
    (finished : Progress → Prop) [DecidablePred finished] (budget : ℕ) :
    budget * (reductionChance game adversary (fun base retries other otherRetries =>
          finished (.finished (runOf adversary base) (forksOf game adversary base retries)
            (runOf adversary other) (forksOf game adversary other otherRetries))) -
        𝔼 tape : Fin budget → Oracle,
          (if finished (finalState game adversary budget tape) then (1 : ℝ) else 0)) ≤
      work schedule queries := by
  let outcome : Oracle → Retries → Oracle → Retries → Prop := fun base retries other otherRetries =>
    finished (.finished (runOf adversary base) (forksOf game adversary base retries)
      (runOf adversary other) (forksOf game adversary other otherRetries))
  have each (state : LawState) :
      (if Found outcome state then (1 : ℝ) else 0) ≤
        if finished (embed game adversary state) then 1 else 0 := by
    cases state with
    | finished => simp only [Found, embed]; split_ifs <;> norm_num
    | _ => simp only [Found, ite_false]; split <;> norm_num
  have chance : (law game adversary).chance (Found outcome) budget .start ≤
      𝔼 tape : Fin budget → Oracle,
        (if finished (finalState game adversary budget tape) then (1 : ℝ) else 0) :=
    Finset.expect_le_expect fun tape _ => by
      rw [(simulation game adversary budget tape).1]
      exact each _
  have loss := (bounds game adversary outcome).budget_mul_le budget (state := .start) trivial
  have positive : (0 : ℝ) ≤ budget := Nat.cast_nonneg budget
  calc
    _ ≤ (budget : ℝ) * (reductionChance game adversary outcome -
          (law game adversary).chance (Found outcome) budget .start) :=
      mul_le_mul_of_nonneg_left (by linarith) positive
    _ ≤ _ := loss.trans (work_start_le game adversary bounded)


variable {Result : Type}
  (kernel : View schedule width Output → View schedule width Output → Option Result)

/-- The chance that the solver outputs a value with `budget` adversary runs. -/
noncomputable def solverChance (budget : ℕ) : ℝ :=
  𝔼 tape : Fin budget → Oracle,
    if (solve game kernel adversary budget tape).isSome then 1 else 0

/-- The reduction chance of "the kernel of the two views exists" is at most the solver's chance
plus `work schedule Q / budget`. -/
theorem reductionChance_le_solver {queries : Nat} (bounded : adversary.QueryBound queries)
    (budget : ℕ) :
    budget * (reductionChance game adversary (fun oracle retries other otherRetries =>
          (kernel (viewOf game adversary oracle retries)
            (viewOf game adversary other otherRetries)).isSome) -
        solverChance game adversary kernel budget) ≤
      work schedule queries :=
  reductionChance_le_program game adversary bounded
    (fun state => (output kernel state).isSome) budget

end NightstreamFPrime.Lifecycle.RandomOracle.Solver
