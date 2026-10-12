import NightstreamFPrime.Lifecycle.RandomOracleBinding
import NightstreamFPrime.Spec.RandomOracleBudget

/-!
Owns the binding reduction of Lemma 6 as an algorithm with a strict budget of
adversary runs (`Spec.RandomOracleBudget.Rewinding`).

Inputs: an adaptive oracle adversary with at most `Q` queries, its claims, the
Ajtai key, and a budget `T`.

Outputs:
- `solve`: the reduction on a uniform tape of `T` oracles. Each step runs the
  adversary once: the base run on a fresh oracle, a retry on the base run's
  oracle with one fresh `Π_RLC` answer, or a rerun on the fork context with
  fresh answers elsewhere. Every other test reads the outputs of runs already
  made. It returns `rerunKernel` of the two runs it finishes with, so at most
  `T` runs on every tape (`Spec.RandomOracleBudget.Rewinding.runs_le`);
- `expected_second_retries_le`: the retries of the extractions inside the
  reruns take at most `17 (Q + 17)` expected runs;
- `expected_runs_le`: for every budget, at most `35 Q + 653` expected runs:
  1 base run, `17 (Q + 17)` base retries
  (`RandomOracleExtraction.expected_retries_le`), `Q + 74` reruns
  (`RandomOracleUniqueness.expected_reruns_le`) and the second extraction;
- `kernelChance_le_solver`: `T · (kernelChance − solverChance T) ≤ 35 Q + 653`,
  so the strict algorithm loses at most `(35 Q + 653) / T` of the binding
  reduction's chance (`RandomOracleBinding.kernelChance`).

The reduction's laws are the existing ones: a retry loop ends uniform on its
line (`Spec.RandomOracleBudget.lineLaw_eq_retryWeight`), and a rerun loop
restarts until a rerun extracts at the fork index (`retryChance`).

Does not own: the hardness of MSIS for the resulting solver, or machine work
beyond counted adversary runs.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleSolver

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.RandomOracleBudget
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open NightstreamFPrime.Lifecycle.RandomOracleUniqueness
open NightstreamFPrime.Lifecycle.RandomOracleBinding
open _root_.NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

/-- What follows an extraction: the base run's rerun loop, or the end of a
rerun of `base`. -/
inductive Purpose (Oracle Retries : Type)
  | first
  | second (base : Oracle) (baseRetries : Retries)

/-- The reduction between two adversary runs. -/
inductive State (Oracle Retries Coordinate : Type)
  | start
  | extract (purpose : Purpose Oracle Retries) (run : Oracle) (index : Coordinate)
      (rest : List Coordinate) (retries : Retries)
  | rerun (base : Oracle) (retries : Retries)
  | finished (base : Oracle) (retries : Retries) (other : Oracle) (otherRetries : Retries)
  | failed

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {Output : Type}
  (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
    Answer Output)
  (claim : Output → Claim relation)

local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer
local notation "Coordinate" => Fin Nifs.PaperProfile.arity.total
local notation "Retries" => Coordinate → Answer
local notation "SolverState" => State Oracle Retries Coordinate

/-! ## The algorithm -/

/-- After a run's last retry loop: a valid base run starts its rerun loop; a
valid rerun finishes; an invalid rerun restarts the rerun loop. -/
private noncomputable def afterExtraction :
    Purpose Oracle Retries → Oracle → Retries → SolverState
  | .first, run, retries =>
      if Valid relation ajtai adversary claim run retries then .rerun run retries else .failed
  | .second base baseRetries, run, retries =>
      if Valid relation ajtai adversary claim run retries then .finished base baseRetries run retries
      else .rerun base baseRetries

/-- The retry loops still pending on `run`. -/
private noncomputable def extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    List Coordinate → Retries → SolverState
  | [], retries => afterExtraction relation ajtai adversary claim purpose run retries
  | index :: rest, retries => .extract purpose run index rest retries

/-- The rerun of `base` on the fresh oracle `fresh`: the fork context of the
base run, and `fresh` elsewhere. -/
private noncomputable def rerunOracle (base fresh : Oracle) : Oracle :=
  overlay (context relation adversary claim (forkIndex relation adversary claim base) base) base fresh

/-- Retry loops on every coordinate, starting from the run's own answers. -/
private noncomputable def startExtraction (purpose : Purpose Oracle Retries) (run : Oracle) :
    SolverState :=
  extracted relation ajtai adversary claim purpose run (List.finRange _)
    fun index => run (rhoPoint relation adversary claim index run)

/-- One step: one adversary run on an oracle built from one fresh tape entry. -/
private noncomputable def step : SolverState → Oracle → SolverState
  | .start, fresh =>
      if Succeeds relation ajtai adversary claim fresh then
        startExtraction relation ajtai adversary claim .first fresh
      else .failed
  | .extract purpose run index rest retries, fresh =>
      if fresh (rhoPoint relation adversary claim index run) ∈ retrySets relation ajtai adversary claim run index
      then extracted relation ajtai adversary claim purpose run rest
        (Function.update retries index (fresh (rhoPoint relation adversary claim index run)))
      else .extract purpose run index rest retries
  | .rerun base retries, fresh =>
      if forkIndex relation adversary claim (rerunOracle relation adversary claim base fresh) =
          forkIndex relation adversary claim base ∧
        Succeeds relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
      then startExtraction relation ajtai adversary claim (.second base retries)
        (rerunOracle relation adversary claim base fresh)
      else .rerun base retries
  | .finished base retries other otherRetries, _ => .finished base retries other otherRetries
  | .failed, _ => .failed

/-- The reduction as a rewinding machine: it halts when it finishes or fails. -/
noncomputable def machine : Rewinding SolverState Oracle where
  step := step relation ajtai adversary claim
  halted
    | .finished .. => true
    | .failed => true
    | _ => false

/-- The reduction's output: the binding reduction on the two runs it finished
with. -/
noncomputable def output :
    SolverState → Option (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound)
  | .finished base retries other otherRetries =>
      rerunKernel relation ajtai adversary claim base retries other otherRetries
  | _ => none

/-- **The strict algorithm**: the reduction with at most `budget` adversary
runs on the tape (`Rewinding.runs_le`). -/
noncomputable def solve (budget : ℕ) (tape : Fin budget → Oracle) :
    Option (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound) :=
  output relation ajtai adversary claim
    ((machine relation ajtai adversary claim).run budget .start tape).final

/-- The strict algorithm's success over a uniform tape. -/
noncomputable def solverChance (budget : ℕ) : ℝ :=
  (machine relation ajtai adversary claim).chance
    (fun state => (output relation ajtai adversary claim state).isSome) budget .start

/-! ## The second extraction -/

/-- Expected retries of one extraction on `run`. -/
private noncomputable def retriesOf (run : Oracle) : ℝ :=
  ∑ coordinate, retryCount (Hits relation ajtai adversary claim coordinate)
    (rhoPoint relation adversary claim coordinate run) run

/-- The extraction work of a run that forks at `index` and succeeds. -/
private noncomputable def extractionWork (index : Nat) (run : Oracle) : ℝ :=
  if forkIndex relation adversary claim run = index ∧ Succeeds relation ajtai adversary claim run then
    retriesOf relation ajtai adversary claim run
  else 0

/-- Expected retries of the extraction inside one rerun from the context of
`index`. -/
noncomputable def rerunRetries (index : Nat) (oracle : Oracle) : ℝ :=
  𝔼 fresh, extractionWork relation ajtai adversary claim index
    (overlay (context relation adversary claim index oracle) oracle fresh)

private theorem retriesOf_nonnegative (run : Oracle) : 0 ≤ retriesOf relation ajtai adversary claim run :=
  Finset.sum_nonneg fun _ _ => div_nonneg zero_le_one (mass_nonnegative _)

private theorem extractionWork_nonnegative (index : Nat) (run : Oracle) :
    0 ≤ extractionWork relation ajtai adversary claim index run := by
  unfold extractionWork
  split
  · exact retriesOf_nonnegative relation ajtai adversary claim run
  · exact le_rfl

private theorem rerunRetries_nonnegative (index : Nat) (oracle : Oracle) :
    0 ≤ rerunRetries relation ajtai adversary claim index oracle :=
  Finset.expect_nonneg fun _ _ => extractionWork_nonnegative relation ajtai adversary claim index _

/-- **The second extraction's work.** Averaged over the base runs, the
extractions inside the reruns take at most `17 (Q + 17)` expected retries. -/
theorem expected_second_retries_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
        rerunRetries relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle /
        resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle ≤
      (Nifs.PaperProfile.arity).total * ((queries + (Nifs.PaperProfile.arity).total : Nat) : ℝ) := by
  simp only [sum_success relation ajtai adversary claim bounded (rerunRetries relation ajtai adversary claim)]
  rw [Finset.expect_sum_comm]
  have each (index : Nat) :
      𝔼 oracle, rerunRetries relation ajtai adversary claim index oracle *
          success relation ajtai adversary claim index oracle /
          resampledSuccess relation ajtai adversary claim index oracle ≤
        𝔼 oracle, extractionWork relation ajtai adversary claim index oracle := by
    have determined (oracle fresh : Oracle) :
        rerunRetries relation ajtai adversary claim index
            (overlay (context relation adversary claim index oracle) oracle fresh) =
          rerunRetries relation ajtai adversary claim index oracle := by
      unfold rerunRetries
      simp only [(overlay_overlay (context relation adversary claim index)
        (context_stopping relation adversary claim index) oracle fresh _).2]
    calc
      _ = 𝔼 oracle, (if 𝔼 fresh, success relation ajtai adversary claim index
              (overlay (context relation adversary claim index oracle) oracle fresh) = 0 then 0
            else rerunRetries relation ajtai adversary claim index oracle) :=
        expect_div_resampled (context relation adversary claim index)
          (context_stopping relation adversary claim index) (success relation ajtai adversary claim index)
          (rerunRetries relation ajtai adversary claim index) determined
      _ ≤ 𝔼 oracle, rerunRetries relation ajtai adversary claim index oracle :=
        Finset.expect_le_expect fun oracle _ => by
          split
          · exact rerunRetries_nonnegative relation ajtai adversary claim index oracle
          · exact le_rfl
      _ = _ := (expect_resample (context relation adversary claim index)
          (context_stopping relation adversary claim index)
          (extractionWork relation ajtai adversary claim index)).symm
  have collapse (oracle : Oracle) :
      ∑ index ∈ Finset.range (queries + challenges.length),
          extractionWork relation ajtai adversary claim index oracle =
        if Succeeds relation ajtai adversary claim oracle then retriesOf relation ajtai adversary claim oracle
        else 0 := by
    rw [Finset.sum_eq_single (forkIndex relation adversary claim oracle)]
    · simp [extractionWork]
    · intro index _ different
      simp [extractionWork, Ne.symm different]
    · intro outside
      exact absurd (Finset.mem_range.mpr (forkIndex_lt_bound relation adversary claim bounded oracle)) outside
  calc
    _ ≤ ∑ index ∈ Finset.range (queries + challenges.length),
          𝔼 oracle, extractionWork relation ajtai adversary claim index oracle :=
      Finset.sum_le_sum fun index _ => each index
    _ = 𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
          retriesOf relation ajtai adversary claim oracle else 0) := by
      rw [← Finset.expect_sum_comm]
      exact Finset.expect_congr rfl fun oracle _ => collapse oracle
    _ ≤ _ := expected_retries_le relation ajtai adversary claim bounded

/-! ## Potentials -/

/-- The rerun loop's eventual success: `retryChance` of the binding outcome. -/
private noncomputable def rerunSuccess (base : Oracle) (retries : Retries) : ℝ :=
  retryChance relation ajtai adversary claim (forkIndex relation adversary claim base) base
    fun other otherRetries => (rerunKernel relation ajtai adversary claim base retries other otherRetries).isSome

/-- The rerun loop's expected remaining runs: one run and an extraction per
attempt, for `1 / resampledSuccess` expected attempts. -/
private noncomputable def rerunWork (base : Oracle) : ℝ :=
  (1 + rerunRetries relation ajtai adversary claim (forkIndex relation adversary claim base) base) /
    resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim base) base

private noncomputable def successAfter : Purpose Oracle Retries → Oracle → Retries → ℝ
  | .first, run, retries =>
      if Valid relation ajtai adversary claim run retries then
        rerunSuccess relation ajtai adversary claim run retries else 0
  | .second base baseRetries, run, retries =>
      if Valid relation ajtai adversary claim run retries then
        (if (rerunKernel relation ajtai adversary claim base baseRetries run retries).isSome then 1 else 0)
      else rerunSuccess relation ajtai adversary claim base baseRetries

private noncomputable def workAfter : Purpose Oracle Retries → Oracle → Retries → ℝ
  | .first, run, retries =>
      if Valid relation ajtai adversary claim run retries then rerunWork relation ajtai adversary claim run
      else 0
  | .second base _, run, retries =>
      if Valid relation ajtai adversary claim run retries then 0 else rerunWork relation ajtai adversary claim base

/-- The states the reduction reaches: every pending loop has positive mass, and
every rerun loop restarts from a valid base. -/
private def goodOf : SolverState → Prop
  | .extract .first run _ _ _ => Succeeds relation ajtai adversary claim run
  | .extract (.second base baseRetries) run _ _ _ =>
      Succeeds relation ajtai adversary claim run ∧ Valid relation ajtai adversary claim base baseRetries
  | .rerun base retries => Valid relation ajtai adversary claim base retries
  | _ => True

/-- The eventual chance that the unbudgeted reduction returns a vector. -/
private noncomputable def successOf : SolverState → ℝ
  | .start => kernelChance relation ajtai adversary claim
  | .extract purpose run index rest retries =>
      lineLaw (retrySets relation ajtai adversary claim run)
        (successAfter relation ajtai adversary claim purpose run) (index :: rest) retries
  | .rerun base retries => rerunSuccess relation ajtai adversary claim base retries
  | .finished base retries other otherRetries =>
      if (rerunKernel relation ajtai adversary claim base retries other otherRetries).isSome then 1 else 0
  | .failed => 0

/-- The expected remaining adversary runs of the unbudgeted reduction. -/
private noncomputable def workOf : SolverState → ℝ
  | .start => 1 + 𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
        retriesOf relation ajtai adversary claim oracle +
          retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
            rerunWork relation ajtai adversary claim oracle
      else 0)
  | .extract purpose run index rest retries =>
      ((index :: rest).map fun coordinate => retryCount (Hits relation ajtai adversary claim coordinate)
          (rhoPoint relation adversary claim coordinate run) run).sum +
        lineLaw (retrySets relation ajtai adversary claim run)
          (workAfter relation ajtai adversary claim purpose run) (index :: rest) retries
  | .rerun base _ => rerunWork relation ajtai adversary claim base
  | _ => 0

/-! ## Laws of the steps -/

private theorem weight_sum {run : Oracle} (succeeds : Succeeds relation ajtai adversary claim run) :
    ∑ retries, weight relation ajtai adversary claim run retries = 1 :=
  retryWeight_sum_eq_one _ fun index =>
    mass_ne_zero (base_mem_retrySets relation ajtai adversary claim succeeds index)

/-- After an accepted run, an outcome that reads the retries only on `event`
splits into its part on `event` and the rest of the mass. -/
private theorem sum_weight_ite {run : Oracle} (succeeds : Succeeds relation ajtai adversary claim run)
    (event : Retries → Prop) (value : Retries → ℝ) (other : ℝ) :
    ∑ retries, weight relation ajtai adversary claim run retries *
        (if event retries then value retries else other) =
      ∑ retries, weight relation ajtai adversary claim run retries *
          (if event retries then value retries else 0) +
        (1 - retryMass relation ajtai adversary claim run event) * other := by
  have split (retries : Retries) :
      weight relation ajtai adversary claim run retries * (if event retries then value retries else other) =
        weight relation ajtai adversary claim run retries * (if event retries then value retries else 0) +
          (weight relation ajtai adversary claim run retries -
            weight relation ajtai adversary claim run retries * (if event retries then 1 else 0)) * other := by
    split <;> ring
  simp only [split]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_sub_distrib, weight_sum relation ajtai adversary claim succeeds]
  rfl

/-- A mass whose event never holds is zero. -/
private theorem retryMass_eq_zero {run : Oracle} {event : Retries → Prop} (never : ∀ retries, ¬ event retries) :
    retryMass relation ajtai adversary claim run event = 0 :=
  Finset.sum_eq_zero fun retries _ => by simp [never retries]

private theorem successOf_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries, successOf relation ajtai adversary claim
        (extracted relation ajtai adversary claim purpose run pending retries) =
      lineLaw (retrySets relation ajtai adversary claim run)
        (successAfter relation ajtai adversary claim purpose run) pending retries
  | [], retries => by
      cases purpose <;>
      · simp only [extracted, afterExtraction, lineLaw, successAfter]
        by_cases valid : Valid relation ajtai adversary claim run retries
        · simp only [valid, if_true]
          rfl
        · simp only [valid, if_false]
          rfl
  | _ :: _, _ => rfl

private theorem workOf_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries, workOf relation ajtai adversary claim
        (extracted relation ajtai adversary claim purpose run pending retries) =
      (pending.map fun coordinate => retryCount (Hits relation ajtai adversary claim coordinate)
          (rhoPoint relation adversary claim coordinate run) run).sum +
        lineLaw (retrySets relation ajtai adversary claim run)
          (workAfter relation ajtai adversary claim purpose run) pending retries
  | [], retries => by
      cases purpose <;> simp only [extracted, afterExtraction, lineLaw, workAfter, List.map_nil, List.sum_nil,
        zero_add] <;> split_ifs <;> rfl
  | _ :: _, _ => rfl

/-- The reduction's good states after a run whose loops all have positive mass. -/
private def Started : Purpose Oracle Retries → Oracle → Prop
  | .first, run => Succeeds relation ajtai adversary claim run
  | .second base baseRetries, run =>
      Succeeds relation ajtai adversary claim run ∧ Valid relation ajtai adversary claim base baseRetries

private theorem goodOf_extracted {purpose : Purpose Oracle Retries} {run : Oracle}
    (started : Started relation ajtai adversary claim purpose run) :
    ∀ pending retries, goodOf relation ajtai adversary claim
      (extracted relation ajtai adversary claim purpose run pending retries)
  | [], retries => by
      cases purpose with
      | first =>
          simp only [extracted, afterExtraction]
          split_ifs with valid
          · exact valid
          · trivial
      | second base baseRetries =>
          simp only [extracted, afterExtraction]
          split_ifs
          · trivial
          · exact started.2
  | _ :: _, _ => by cases purpose <;> exact started

private theorem started_succeeds {purpose : Purpose Oracle Retries} {run : Oracle}
    (started : Started relation ajtai adversary claim purpose run) :
    Succeeds relation ajtai adversary claim run := by
  cases purpose with
  | first => exact started
  | second => exact started.1

/-- A started extraction's eventual success is the retry law's average. -/
private theorem successOf_start {purpose : Purpose Oracle Retries} {run : Oracle} :
    successOf relation ajtai adversary claim (startExtraction relation ajtai adversary claim purpose run) =
      ∑ retries, weight relation ajtai adversary claim run retries *
        successAfter relation ajtai adversary claim purpose run retries := by
  unfold startExtraction
  rw [successOf_extracted, lineLaw_eq_retryWeight]
  rfl

private theorem workOf_start {purpose : Purpose Oracle Retries} {run : Oracle} :
    workOf relation ajtai adversary claim (startExtraction relation ajtai adversary claim purpose run) =
      retriesOf relation ajtai adversary claim run +
        ∑ retries, weight relation ajtai adversary claim run retries *
          workAfter relation ajtai adversary claim purpose run retries := by
  unfold startExtraction
  rw [workOf_extracted, lineLaw_eq_retryWeight, retriesOf, Fin.sum_univ_def]
  rfl

/-! ### The base run -/

private theorem success_start_step :
    successOf relation ajtai adversary claim .start =
      𝔼 fresh, successOf relation ajtai adversary claim (step relation ajtai adversary claim .start fresh) := by
  conv_lhs => unfold successOf
  unfold kernelChance
  refine Finset.expect_congr rfl fun oracle _ => ?_
  simp only [step]
  by_cases succeeds : Succeeds relation ajtai adversary claim oracle
  · rw [if_pos succeeds, successOf_start]
    rfl
  · rw [if_neg succeeds]
    exact Finset.sum_eq_zero fun retries _ => by
      rw [if_neg fun valid => succeeds valid.1, mul_zero]

private theorem work_start_step :
    1 + 𝔼 fresh, workOf relation ajtai adversary claim (step relation ajtai adversary claim .start fresh) =
      workOf relation ajtai adversary claim .start := by
  conv_rhs => unfold workOf
  congr 1
  refine Finset.expect_congr rfl fun oracle _ => ?_
  simp only [step]
  by_cases succeeds : Succeeds relation ajtai adversary claim oracle
  · rw [if_pos succeeds, if_pos succeeds, workOf_start]
    refine congrArg (retriesOf relation ajtai adversary claim oracle + ·) ?_
    unfold retryMass
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun retries _ => ?_
    simp only [workAfter]
    split <;> ring
  · rw [if_neg succeeds, if_neg succeeds]
    rfl

/-! ### A retry -/

private theorem success_extract_step {purpose : Purpose Oracle Retries} {run : Oracle} {index : Coordinate}
    {rest : List Coordinate} {retries : Retries} (started : Started relation ajtai adversary claim purpose run) :
    successOf relation ajtai adversary claim (.extract purpose run index rest retries) =
      𝔼 fresh, successOf relation ajtai adversary claim
        (step relation ajtai adversary claim (.extract purpose run index rest retries) fresh) := by
  have positive := mass_ne_zero
    (base_mem_retrySets relation ajtai adversary claim (started_succeeds relation ajtai adversary claim started) index)
  let next : Answer → ℝ := fun answer =>
    lineLaw (retrySets relation ajtai adversary claim run) (successAfter relation ajtai adversary claim purpose run)
      rest (Function.update retries index answer)
  have pointwise (fresh : Oracle) :
      successOf relation ajtai adversary claim
          (step relation ajtai adversary claim (.extract purpose run index rest retries) fresh) =
        (fun answer => if answer ∈ retrySets relation ajtai adversary claim run index then next answer
          else lineAverage (retrySets relation ajtai adversary claim run index) next)
          (fresh (rhoPoint relation adversary claim index run)) := by
    simp only [step]
    split
    · rw [successOf_extracted]
    · rfl
  calc
    _ = lineAverage (retrySets relation ajtai adversary claim run index) next := rfl
    _ = _ := (lineAverage_step _ positive next).symm
    _ = _ := (expect_eval (rhoPoint relation adversary claim index run) _).symm
    _ = _ := Finset.expect_congr rfl fun fresh _ => (pointwise fresh).symm

private theorem work_extract_step {purpose : Purpose Oracle Retries} {run : Oracle} {index : Coordinate}
    {rest : List Coordinate} {retries : Retries} (started : Started relation ajtai adversary claim purpose run) :
    1 + 𝔼 fresh, workOf relation ajtai adversary claim
        (step relation ajtai adversary claim (.extract purpose run index rest retries) fresh) =
      workOf relation ajtai adversary claim (.extract purpose run index rest retries) := by
  have positive := mass_ne_zero
    (base_mem_retrySets relation ajtai adversary claim (started_succeeds relation ajtai adversary claim started) index)
  let later : ℝ := (rest.map fun coordinate => retryCount (Hits relation ajtai adversary claim coordinate)
    (rhoPoint relation adversary claim coordinate run) run).sum
  let next : Answer → ℝ := fun answer =>
    lineLaw (retrySets relation ajtai adversary claim run) (workAfter relation ajtai adversary claim purpose run)
      rest (Function.update retries index answer)
  let attempt : Answer → ℝ := fun answer =>
    if answer ∈ retrySets relation ajtai adversary claim run index then later + next answer
    else 1 / mass (retrySets relation ajtai adversary claim run index) + later +
      lineAverage (retrySets relation ajtai adversary claim run index) next
  have total : workOf relation ajtai adversary claim (.extract purpose run index rest retries) =
      1 / mass (retrySets relation ajtai adversary claim run index) + later +
        lineAverage (retrySets relation ajtai adversary claim run index) next := by
    simp only [workOf, List.map_cons, List.sum_cons, lineLaw]
    unfold retryCount
    rfl
  have pointwise (fresh : Oracle) :
      workOf relation ajtai adversary claim
          (step relation ajtai adversary claim (.extract purpose run index rest retries) fresh) =
        attempt (fresh (rhoPoint relation adversary claim index run)) := by
    simp only [step, attempt]
    split
    · rw [workOf_extracted]
    · exact total
  calc
    _ = 1 + 𝔼 fresh : Oracle, attempt (fresh (rhoPoint relation adversary claim index run)) :=
      congrArg (1 + ·) (Finset.expect_congr rfl fun fresh _ => pointwise fresh)
    _ = 1 + 𝔼 answer, attempt answer :=
      congrArg (1 + ·) (expect_eval (rhoPoint relation adversary claim index run) attempt)
    _ = _ := line_work_step _ positive later next
    _ = _ := total.symm

/-! ### A rerun -/

/-- A rerun that does not fork at the base's index, or does not succeed,
finds nothing and counts as no try. -/
private theorem rerun_missed {base fresh : Oracle}
    (missed : ¬ (forkIndex relation adversary claim (rerunOracle relation adversary claim base fresh) =
        forkIndex relation adversary claim base ∧
      Succeeds relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)))
    (event : Retries → Prop) :
    retryMass relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
        (fun otherRetries => forkIndex relation adversary claim (rerunOracle relation adversary claim base fresh) =
            forkIndex relation adversary claim base ∧
          Valid relation ajtai adversary claim (rerunOracle relation adversary claim base fresh) otherRetries ∧
          event otherRetries) = 0 ∧
      success relation ajtai adversary claim (forkIndex relation adversary claim base)
        (rerunOracle relation adversary claim base fresh) = 0 := by
  refine ⟨retryMass_eq_zero relation ajtai adversary claim fun _ holds => missed ⟨holds.1, holds.2.1.1⟩, ?_⟩
  rw [success_eq]
  exact retryMass_eq_zero relation ajtai adversary claim fun _ holds => missed ⟨holds.1, holds.2.1⟩

/-- A rerun that forks at the base's index and succeeds tries with the mass of
its valid extractions. -/
private theorem rerun_tries {base fresh : Oracle}
    (forked : forkIndex relation adversary claim (rerunOracle relation adversary claim base fresh) =
      forkIndex relation adversary claim base) :
    success relation ajtai adversary claim (forkIndex relation adversary claim base)
        (rerunOracle relation adversary claim base fresh) =
      retryMass relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
        (Valid relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)) := by
  rw [success_eq]
  unfold retryMass
  exact Finset.sum_congr rfl fun _ _ => by simp [forked]

private theorem success_rerun_step {base : Oracle} {retries : Retries}
    (valid : Valid relation ajtai adversary claim base retries) :
    successOf relation ajtai adversary claim (.rerun base retries) =
      𝔼 fresh, successOf relation ajtai adversary claim
        (step relation ajtai adversary claim (.rerun base retries) fresh) := by
  let outcome : Oracle → Retries → Prop := fun other otherRetries =>
    (rerunKernel relation ajtai adversary claim base retries other otherRetries).isSome
  let found : Oracle → ℝ := fun fresh =>
    retryMass relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
      fun otherRetries => forkIndex relation adversary claim (rerunOracle relation adversary claim base fresh) =
          forkIndex relation adversary claim base ∧
        Valid relation ajtai adversary claim (rerunOracle relation adversary claim base fresh) otherRetries ∧
        outcome (rerunOracle relation adversary claim base fresh) otherRetries
  let tries : Oracle → ℝ := fun fresh =>
    success relation ajtai adversary claim (forkIndex relation adversary claim base)
      (rerunOracle relation adversary claim base fresh)
  have positive : 𝔼 fresh, tries fresh ≠ 0 := (resampledSuccess_pos relation ajtai adversary claim valid).ne'
  have chance : successOf relation ajtai adversary claim (.rerun base retries) =
      (𝔼 fresh, found fresh) / 𝔼 fresh, tries fresh := rfl
  have pointwise (fresh : Oracle) :
      successOf relation ajtai adversary claim (step relation ajtai adversary claim (.rerun base retries) fresh) =
        found fresh + (1 - tries fresh) * successOf relation ajtai adversary claim (.rerun base retries) := by
    simp only [step]
    split
    · rename_i forked
      have foundEq : found fresh =
          ∑ otherRetries, weight relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
              otherRetries *
            (if Valid relation ajtai adversary claim (rerunOracle relation adversary claim base fresh) otherRetries
              then (if (rerunKernel relation ajtai adversary claim base retries
                  (rerunOracle relation adversary claim base fresh) otherRetries).isSome then 1 else 0)
              else 0) := by
        simp only [found, outcome, retryMass]
        exact Finset.sum_congr rfl fun otherRetries _ => by
          by_cases holds : Valid relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
              otherRetries <;> simp [holds, forked.1]
      rw [successOf_start]
      simp only [successAfter]
      rw [sum_weight_ite relation ajtai adversary claim forked.2, foundEq]
      simp only [tries, rerun_tries relation ajtai adversary claim forked.1]
      rfl
    · rename_i missed
      obtain ⟨noFound, noTries⟩ := rerun_missed relation ajtai adversary claim missed
        (outcome (rerunOracle relation adversary claim base fresh))
      simp only [found, tries, noFound, noTries, sub_zero, one_mul, zero_add]
  rw [Finset.expect_congr rfl fun fresh _ => pointwise fresh, chance, restart_success_eq found tries positive]

private theorem work_rerun_step {base : Oracle} {retries : Retries}
    (valid : Valid relation ajtai adversary claim base retries) :
    1 + 𝔼 fresh, workOf relation ajtai adversary claim
        (step relation ajtai adversary claim (.rerun base retries) fresh) =
      workOf relation ajtai adversary claim (.rerun base retries) := by
  let extra : Oracle → ℝ := fun fresh =>
    extractionWork relation ajtai adversary claim (forkIndex relation adversary claim base)
      (rerunOracle relation adversary claim base fresh)
  let tries : Oracle → ℝ := fun fresh =>
    success relation ajtai adversary claim (forkIndex relation adversary claim base)
      (rerunOracle relation adversary claim base fresh)
  have positive : 𝔼 fresh, tries fresh ≠ 0 := (resampledSuccess_pos relation ajtai adversary claim valid).ne'
  have total : workOf relation ajtai adversary claim (.rerun base retries) =
      (1 + 𝔼 fresh, extra fresh) / 𝔼 fresh, tries fresh := rfl
  have pointwise (fresh : Oracle) :
      workOf relation ajtai adversary claim (step relation ajtai adversary claim (.rerun base retries) fresh) =
        extra fresh + (1 - tries fresh) * workOf relation ajtai adversary claim (.rerun base retries) := by
    simp only [step]
    split
    · rename_i forked
      rw [workOf_start]
      simp only [workAfter]
      rw [sum_weight_ite relation ajtai adversary claim forked.2]
      simp only [tries, extra, rerun_tries relation ajtai adversary claim forked.1, extractionWork,
        if_pos forked, ite_self, mul_zero, Finset.sum_const_zero, zero_add]
      rfl
    · rename_i missed
      have noTries := (rerun_missed relation ajtai adversary claim missed fun _ => True).2
      simp only [extra, tries, extractionWork, if_neg missed, noTries, sub_zero, one_mul, zero_add]
  calc
    _ = 1 + 𝔼 fresh, (extra fresh + (1 - tries fresh) * ((1 + 𝔼 fresh, extra fresh) / 𝔼 fresh, tries fresh)) := by
      rw [Finset.expect_congr rfl fun fresh _ => pointwise fresh, total]
    _ = _ := restart_work_eq extra tries positive
    _ = _ := total.symm

/-! ## The certificate -/

private theorem rerunWork_nonnegative (base : Oracle) : 0 ≤ rerunWork relation ajtai adversary claim base :=
  div_nonneg (add_nonneg zero_le_one (rerunRetries_nonnegative relation ajtai adversary claim _ _))
    (resampledSuccess_nonnegative relation ajtai adversary claim _ _)

private theorem successAfter_le_one (purpose : Purpose Oracle Retries) (run : Oracle) (retries : Retries) :
    successAfter relation ajtai adversary claim purpose run retries ≤ 1 := by
  cases purpose <;> simp only [successAfter] <;> split_ifs <;>
    first
    | exact retryChance_le_one relation ajtai adversary claim _ _ _
    | norm_num

private theorem workAfter_nonnegative (purpose : Purpose Oracle Retries) (run : Oracle) (retries : Retries) :
    0 ≤ workAfter relation ajtai adversary claim purpose run retries := by
  cases purpose <;> simp only [workAfter] <;> split_ifs <;>
    first
    | exact rerunWork_nonnegative relation ajtai adversary claim _
    | exact le_rfl

private theorem started_of_good {purpose : Purpose Oracle Retries} {run : Oracle} {index : Coordinate}
    {rest : List Coordinate} {retries : Retries}
    (good : goodOf relation ajtai adversary claim (.extract purpose run index rest retries)) :
    Started relation ajtai adversary claim purpose run := by
  cases purpose <;> exact good

/-- The reduction's potentials: `successOf` is its eventual chance of a vector
and `workOf` its expected remaining runs, exactly. -/
private noncomputable def bounds :
    (machine relation ajtai adversary claim).Bounds
      (fun state => (output relation ajtai adversary claim state).isSome) where
  good := goodOf relation ajtai adversary claim
  success := successOf relation ajtai adversary claim
  work := workOf relation ajtai adversary claim
  good_step := by
    intro state good unhalted sample
    cases state with
    | start =>
        show goodOf relation ajtai adversary claim (step relation ajtai adversary claim .start sample)
        simp only [step]
        split
        · rename_i succeeds
          exact goodOf_extracted relation ajtai adversary claim (purpose := .first) succeeds _ _
        · trivial
    | extract purpose run index rest retries =>
        show goodOf relation ajtai adversary claim
          (step relation ajtai adversary claim (.extract purpose run index rest retries) sample)
        simp only [step]
        split
        · exact goodOf_extracted relation ajtai adversary claim
            (started_of_good relation ajtai adversary claim good) _ _
        · exact good
    | rerun base retries =>
        show goodOf relation ajtai adversary claim (step relation ajtai adversary claim (.rerun base retries) sample)
        simp only [step]
        split
        · rename_i forked
          exact goodOf_extracted relation ajtai adversary claim (purpose := .second base retries)
            ⟨forked.2, good⟩ _ _
        · exact good
    | finished => simp [machine] at unhalted
    | failed => simp [machine] at unhalted
  success_le_one := by
    intro state _
    cases state with
    | start => exact kernelChance_le_one relation ajtai adversary claim
    | extract purpose run index rest retries =>
        exact lineLaw_le _ zero_le_one (successAfter_le_one relation ajtai adversary claim purpose run) _ _
    | rerun base retries =>
        show rerunSuccess relation ajtai adversary claim base retries ≤ 1
        exact retryChance_le_one relation ajtai adversary claim _ _ _
    | finished base retries other otherRetries =>
        show (if (rerunKernel relation ajtai adversary claim base retries other otherRetries).isSome then (1 : ℝ)
          else 0) ≤ 1
        split <;> norm_num
    | failed => show (0 : ℝ) ≤ 1; norm_num
  success_halted := by
    intro state _ halted
    cases state with
    | finished => exact le_rfl
    | failed => simp [successOf, output]
    | start => simp [machine] at halted
    | extract => simp [machine] at halted
    | rerun => simp [machine] at halted
  success_step := by
    intro state good unhalted
    cases state with
    | start => exact (success_start_step relation ajtai adversary claim).le
    | extract purpose run index rest retries =>
        exact (success_extract_step relation ajtai adversary claim
          (started_of_good relation ajtai adversary claim good)).le
    | rerun base retries => exact (success_rerun_step relation ajtai adversary claim good).le
    | finished => simp [machine] at unhalted
    | failed => simp [machine] at unhalted
  work_nonnegative := by
    intro state _
    cases state with
    | start =>
        refine add_nonneg zero_le_one (Finset.expect_nonneg fun oracle _ => ?_)
        split
        · exact add_nonneg (retriesOf_nonnegative relation ajtai adversary claim oracle)
            (mul_nonneg (retryMass_nonnegative relation ajtai adversary claim oracle _)
              (rerunWork_nonnegative relation ajtai adversary claim oracle))
        · exact le_rfl
    | extract purpose run index rest retries =>
        refine add_nonneg (List.sum_nonneg fun count member => ?_)
          (lineLaw_nonnegative _ (workAfter_nonnegative relation ajtai adversary claim purpose run) _ _)
        obtain ⟨coordinate, _, rfl⟩ := List.mem_map.mp member
        exact div_nonneg zero_le_one (mass_nonnegative _)
    | rerun base retries => exact rerunWork_nonnegative relation ajtai adversary claim base
    | finished => exact le_rfl
    | failed => exact le_rfl
  work_step := by
    intro state good unhalted
    cases state with
    | start => exact (work_start_step relation ajtai adversary claim).le
    | extract purpose run index rest retries =>
        exact (work_extract_step relation ajtai adversary claim
          (started_of_good relation ajtai adversary claim good)).le
    | rerun base retries => exact (work_rerun_step relation ajtai adversary claim good).le
    | finished => simp [machine] at unhalted
    | failed => simp [machine] at unhalted

/-- `1 + 17 (Q + 17) + (Q + 74) + 17 (Q + 17) = 35 Q + 653`. -/
private theorem work_start_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    workOf relation ajtai adversary claim .start ≤ 35 * queries + 653 := by
  have retries : 𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
      retriesOf relation ajtai adversary claim oracle else 0) ≤ 17 * ((queries : ℝ) + 17) := by
    have bound := expected_retries_le relation ajtai adversary claim bounded
    simp only [Nifs.PaperProfile.arity_total] at bound
    push_cast at bound
    exact bound
  have reruns := expected_reruns_le relation ajtai adversary claim bounded
  rw [challenges_length] at reruns
  push_cast at reruns
  have second := expected_second_retries_le relation ajtai adversary claim bounded
  simp only [Nifs.PaperProfile.arity_total] at second
  push_cast at second
  have split : 𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
        retriesOf relation ajtai adversary claim oracle +
          retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
            rerunWork relation ajtai adversary claim oracle
      else 0) ≤
      𝔼 oracle, (if Succeeds relation ajtai adversary claim oracle then
          retriesOf relation ajtai adversary claim oracle else 0) +
        𝔼 oracle, retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) /
          resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle +
        𝔼 oracle, retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
          rerunRetries relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle /
          resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle := by
    rw [← Finset.expect_add_distrib, ← Finset.expect_add_distrib]
    refine Finset.expect_le_expect fun oracle _ => ?_
    have mass := retryMass_nonnegative relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle)
    have denominator := resampledSuccess_nonnegative relation ajtai adversary claim
      (forkIndex relation adversary claim oracle) oracle
    have extra := rerunRetries_nonnegative relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle
    have first : 0 ≤ retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) /
        resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle :=
      div_nonneg mass denominator
    have later : 0 ≤ retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
          rerunRetries relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle /
        resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle :=
      div_nonneg (mul_nonneg mass extra) denominator
    have expand : retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
          rerunWork relation ajtai adversary claim oracle =
        retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) /
            resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle +
          retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
            rerunRetries relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle /
            resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle := by
      unfold rerunWork
      ring
    split
    · rw [expand]
      linarith
    · linarith
  show 1 + _ ≤ _
  linarith

/-- **Expected work.** For every budget, the strict algorithm makes at most
`35 Q + 653` expected adversary runs. -/
theorem expected_runs_le {queries : Nat} (bounded : adversary.QueryBound queries) (budget : ℕ) :
    (machine relation ajtai adversary claim).expectedRuns budget .start ≤ 35 * queries + 653 :=
  ((bounds relation ajtai adversary claim).expect_runs_le budget (state := .start) trivial).trans
    (work_start_le relation ajtai adversary claim bounded)

/-- **The Markov loss.** With budget `T`, the strict algorithm returns a
vector with chance at least `kernelChance − (35 Q + 653) / T`. -/
theorem kernelChance_le_solver {queries : Nat} (bounded : adversary.QueryBound queries) (budget : ℕ) :
    budget * (kernelChance relation ajtai adversary claim - solverChance relation ajtai adversary claim budget) ≤
      35 * queries + 653 :=
  ((bounds relation ajtai adversary claim).budget_mul_le budget (state := .start) trivial).trans
    (work_start_le relation ajtai adversary claim bounded)

end NightstreamFPrime.Lifecycle.RandomOracleSolver
