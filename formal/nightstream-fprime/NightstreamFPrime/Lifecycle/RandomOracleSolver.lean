import NightstreamFPrime.Lifecycle.RandomOracleReduction

/-! SECURITY_MODEL.md Lemma 6, binding reduction laws and work. -/

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
open NightstreamFPrime.Lifecycle.RandomOracleReduction
open _root_.NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

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
local notation "LawState" => State Oracle Retries Coordinate
local notation "Record" => RunRecord (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Output
local notation "Made" => Run Oracle Record
local notation "Retried" => Forks Coordinate Answer Record
local notation "Progress" => State Made Retried Coordinate

noncomputable local instance (budget : ℕ) : Fintype (Fin budget → Oracle) := Pi.instFintype

private noncomputable def lawAfter :
    Purpose Oracle Retries → Oracle → Retries → LawState
  | .first, run, retries =>
      if Valid relation ajtai adversary claim run retries then .rerun run retries else .failed
  | .second base baseRetries, run, retries =>
      if Valid relation ajtai adversary claim run retries then .finished base baseRetries run retries
      else .rerun base baseRetries

private noncomputable def lawExtracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    List Coordinate → Retries → LawState
  | [], retries => lawAfter relation ajtai adversary claim purpose run retries
  | index :: rest, retries => .extract purpose run index rest retries

private noncomputable def rerunOracle (base fresh : Oracle) : Oracle :=
  overlay (context relation adversary claim (forkIndex relation adversary claim base) base) base fresh

private noncomputable def lawStart (purpose : Purpose Oracle Retries) (run : Oracle) :
    LawState :=
  lawExtracted relation ajtai adversary claim purpose run (List.finRange _)
    fun index => run (rhoPoint relation adversary claim index run)

private noncomputable def step : LawState → Oracle → LawState
  | .start, fresh =>
      if Succeeds relation ajtai adversary claim fresh then
        lawStart relation ajtai adversary claim .first fresh
      else .failed
  | .extract purpose run index rest retries, fresh =>
      if fresh (rhoPoint relation adversary claim index run) ∈ retrySets relation ajtai adversary claim run index
      then lawExtracted relation ajtai adversary claim purpose run rest
        (Function.update retries index (fresh (rhoPoint relation adversary claim index run)))
      else .extract purpose run index rest retries
  | .rerun base retries, fresh =>
      if forkIndex relation adversary claim (rerunOracle relation adversary claim base fresh) =
          forkIndex relation adversary claim base ∧
        Succeeds relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
      then lawStart relation ajtai adversary claim (.second base retries)
        (rerunOracle relation adversary claim base fresh)
      else .rerun base retries
  | .finished base retries other otherRetries, _ => .finished base retries other otherRetries
  | .failed, _ => .failed

private noncomputable def law : Rewinding LawState Oracle where
  step := step relation ajtai adversary claim
  halted
    | .finished .. => true
    | .failed => true
    | _ => false

private noncomputable def retriesOf (run : Oracle) : ℝ :=
  ∑ coordinate, retryCount (Hits relation ajtai adversary claim coordinate)
    (rhoPoint relation adversary claim coordinate run) run

private noncomputable def extractionWork (index : Nat) (run : Oracle) : ℝ :=
  if forkIndex relation adversary claim run = index ∧ Succeeds relation ajtai adversary claim run then
    retriesOf relation ajtai adversary claim run
  else 0

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

variable (outcome : (Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer) →
  (Fin Nifs.PaperProfile.arity.total → Answer) →
  (Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer) →
  (Fin Nifs.PaperProfile.arity.total → Answer) → Prop)

@[implicit_reducible]
private def Found : LawState → Prop
  | .finished base retries other otherRetries => outcome base retries other otherRetries
  | _ => False

private noncomputable def rerunSuccess (base : Oracle) (retries : Retries) : ℝ :=
  retryChance relation ajtai adversary claim (forkIndex relation adversary claim base) base (outcome base retries)

private noncomputable def rerunWork (base : Oracle) : ℝ :=
  (1 + rerunRetries relation ajtai adversary claim (forkIndex relation adversary claim base) base) /
    resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim base) base

private noncomputable def successAfter : Purpose Oracle Retries → Oracle → Retries → ℝ
  | .first, run, retries =>
      if Valid relation ajtai adversary claim run retries then
        rerunSuccess relation ajtai adversary claim outcome run retries else 0
  | .second base baseRetries, run, retries =>
      if Valid relation ajtai adversary claim run retries then
        (if outcome base baseRetries run retries then 1 else 0)
      else rerunSuccess relation ajtai adversary claim outcome base baseRetries

private noncomputable def workAfter : Purpose Oracle Retries → Oracle → Retries → ℝ
  | .first, run, retries =>
      if Valid relation ajtai adversary claim run retries then rerunWork relation ajtai adversary claim run
      else 0
  | .second base _, run, retries =>
      if Valid relation ajtai adversary claim run retries then 0 else rerunWork relation ajtai adversary claim base

private def goodOf : LawState → Prop
  | .extract .first run _ _ _ => Succeeds relation ajtai adversary claim run
  | .extract (.second base baseRetries) run _ _ _ =>
      Succeeds relation ajtai adversary claim run ∧ Valid relation ajtai adversary claim base baseRetries
  | .rerun base retries => Valid relation ajtai adversary claim base retries
  | _ => True

private noncomputable def successOf : LawState → ℝ
  | .start => reductionChance relation ajtai adversary claim outcome
  | .extract purpose run index rest retries =>
      lineLaw (retrySets relation ajtai adversary claim run)
        (successAfter relation ajtai adversary claim outcome purpose run) (index :: rest) retries
  | .rerun base retries => rerunSuccess relation ajtai adversary claim outcome base retries
  | .finished base retries other otherRetries =>
      if outcome base retries other otherRetries then 1 else 0
  | .failed => 0

private noncomputable def workOf : LawState → ℝ
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

private theorem weight_sum {run : Oracle} (succeeds : Succeeds relation ajtai adversary claim run) :
    ∑ retries, weight relation ajtai adversary claim run retries = 1 :=
  retryWeight_sum_eq_one _ fun index =>
    mass_ne_zero (base_mem_retrySets relation ajtai adversary claim succeeds index)

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

private theorem retryMass_eq_zero {run : Oracle} {event : Retries → Prop} (never : ∀ retries, ¬ event retries) :
    retryMass relation ajtai adversary claim run event = 0 :=
  Finset.sum_eq_zero fun retries _ => by simp [never retries]

private theorem successOf_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries, successOf relation ajtai adversary claim outcome
        (lawExtracted relation ajtai adversary claim purpose run pending retries) =
      lineLaw (retrySets relation ajtai adversary claim run)
        (successAfter relation ajtai adversary claim outcome purpose run) pending retries
  | [], retries => by
      cases purpose <;>
      · simp only [lawExtracted, lawAfter, lineLaw, successAfter]
        by_cases valid : Valid relation ajtai adversary claim run retries
        · simp only [valid, ite_true]
          rfl
        · simp only [valid, ite_false]
          rfl
  | _ :: _, _ => rfl

private theorem workOf_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries, workOf relation ajtai adversary claim
        (lawExtracted relation ajtai adversary claim purpose run pending retries) =
      (pending.map fun coordinate => retryCount (Hits relation ajtai adversary claim coordinate)
          (rhoPoint relation adversary claim coordinate run) run).sum +
        lineLaw (retrySets relation ajtai adversary claim run)
          (workAfter relation ajtai adversary claim purpose run) pending retries
  | [], retries => by
      cases purpose <;> simp only [lawExtracted, lawAfter, lineLaw, workAfter, List.map_nil, List.sum_nil,
        zero_add] <;> split_ifs <;> rfl
  | _ :: _, _ => rfl

private def Started : Purpose Oracle Retries → Oracle → Prop
  | .first, run => Succeeds relation ajtai adversary claim run
  | .second base baseRetries, run =>
      Succeeds relation ajtai adversary claim run ∧ Valid relation ajtai adversary claim base baseRetries

private theorem goodOf_extracted {purpose : Purpose Oracle Retries} {run : Oracle}
    (started : Started relation ajtai adversary claim purpose run) :
    ∀ pending retries, goodOf relation ajtai adversary claim
      (lawExtracted relation ajtai adversary claim purpose run pending retries)
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
    (started : Started relation ajtai adversary claim purpose run) :
    Succeeds relation ajtai adversary claim run := by
  cases purpose with
  | first => exact started
  | second => exact started.1

private theorem successOf_start {purpose : Purpose Oracle Retries} {run : Oracle} :
    successOf relation ajtai adversary claim outcome (lawStart relation ajtai adversary claim purpose run) =
      ∑ retries, weight relation ajtai adversary claim run retries *
        successAfter relation ajtai adversary claim outcome purpose run retries := by
  unfold lawStart
  rw [successOf_extracted, lineLaw_eq_retryWeight]
  rfl

private theorem workOf_start {purpose : Purpose Oracle Retries} {run : Oracle} :
    workOf relation ajtai adversary claim (lawStart relation ajtai adversary claim purpose run) =
      retriesOf relation ajtai adversary claim run +
        ∑ retries, weight relation ajtai adversary claim run retries *
          workAfter relation ajtai adversary claim purpose run retries := by
  unfold lawStart
  rw [workOf_extracted, lineLaw_eq_retryWeight, retriesOf, Fin.sum_univ_def]
  rfl

private theorem success_start_step :
    successOf relation ajtai adversary claim outcome .start =
      𝔼 fresh, successOf relation ajtai adversary claim outcome (step relation ajtai adversary claim .start fresh) := by
  conv_lhs => unfold successOf
  unfold reductionChance
  refine Finset.expect_congr rfl fun oracle _ => ?_
  simp only [step]
  by_cases succeeds : Succeeds relation ajtai adversary claim oracle
  · rw [ite_eq_left succeeds, successOf_start]
    rfl
  · rw [ite_eq_right succeeds]
    exact Finset.sum_eq_zero fun retries _ => by
      rw [ite_eq_right fun valid => succeeds valid.1, mul_zero]

private theorem work_start_step :
    1 + 𝔼 fresh, workOf relation ajtai adversary claim (step relation ajtai adversary claim .start fresh) =
      workOf relation ajtai adversary claim .start := by
  conv_rhs => unfold workOf
  congr 1
  refine Finset.expect_congr rfl fun oracle _ => ?_
  simp only [step]
  by_cases succeeds : Succeeds relation ajtai adversary claim oracle
  · rw [ite_eq_left succeeds, ite_eq_left succeeds, workOf_start]
    refine congrArg (retriesOf relation ajtai adversary claim oracle + ·) ?_
    unfold retryMass
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun retries _ => ?_
    simp only [workAfter]
    split <;> ring
  · rw [ite_eq_right succeeds, ite_eq_right succeeds]
    rfl

private theorem success_extract_step {purpose : Purpose Oracle Retries} {run : Oracle} {index : Coordinate}
    {rest : List Coordinate} {retries : Retries} (started : Started relation ajtai adversary claim purpose run) :
    successOf relation ajtai adversary claim outcome (.extract purpose run index rest retries) =
      𝔼 fresh, successOf relation ajtai adversary claim outcome
        (step relation ajtai adversary claim (.extract purpose run index rest retries) fresh) := by
  have positive := mass_ne_zero
    (base_mem_retrySets relation ajtai adversary claim (started_succeeds relation ajtai adversary claim started) index)
  let next : Answer → ℝ := fun answer =>
    lineLaw (retrySets relation ajtai adversary claim run) (successAfter relation ajtai adversary claim outcome purpose run)
      rest (Function.update retries index answer)
  have pointwise (fresh : Oracle) :
      successOf relation ajtai adversary claim outcome
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
    successOf relation ajtai adversary claim outcome (.rerun base retries) =
      𝔼 fresh, successOf relation ajtai adversary claim outcome
        (step relation ajtai adversary claim (.rerun base retries) fresh) := by
  let found : Oracle → ℝ := fun fresh =>
    retryMass relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
      fun otherRetries => forkIndex relation adversary claim (rerunOracle relation adversary claim base fresh) =
          forkIndex relation adversary claim base ∧
        Valid relation ajtai adversary claim (rerunOracle relation adversary claim base fresh) otherRetries ∧
        outcome base retries (rerunOracle relation adversary claim base fresh) otherRetries
  let tries : Oracle → ℝ := fun fresh =>
    success relation ajtai adversary claim (forkIndex relation adversary claim base)
      (rerunOracle relation adversary claim base fresh)
  have positive : 𝔼 fresh, tries fresh ≠ 0 := (resampledSuccess_pos relation ajtai adversary claim valid).ne'
  have chance : successOf relation ajtai adversary claim outcome (.rerun base retries) =
      (𝔼 fresh, found fresh) / 𝔼 fresh, tries fresh := rfl
  have pointwise (fresh : Oracle) :
      successOf relation ajtai adversary claim outcome (step relation ajtai adversary claim (.rerun base retries) fresh) =
        found fresh + (1 - tries fresh) * successOf relation ajtai adversary claim outcome (.rerun base retries) := by
    simp only [step]
    split
    · rename_i forked
      have foundEq : found fresh =
          ∑ otherRetries, weight relation ajtai adversary claim (rerunOracle relation adversary claim base fresh)
              otherRetries *
            (if Valid relation ajtai adversary claim (rerunOracle relation adversary claim base fresh) otherRetries
              then (if outcome base retries (rerunOracle relation adversary claim base fresh) otherRetries
                then 1 else 0)
              else 0) := by
        simp only [found, retryMass]
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
        (outcome base retries (rerunOracle relation adversary claim base fresh))
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
        ite_eq_left forked, ite_self, mul_zero, Finset.sum_const_zero, zero_add]
      rfl
    · rename_i missed
      have noTries := (rerun_missed relation ajtai adversary claim missed fun _ => True).2
      simp only [extra, tries, extractionWork, ite_eq_right missed, noTries, sub_zero, one_mul, zero_add]
  calc
    _ = 1 + 𝔼 fresh, (extra fresh + (1 - tries fresh) * ((1 + 𝔼 fresh, extra fresh) / 𝔼 fresh, tries fresh)) := by
      rw [Finset.expect_congr rfl fun fresh _ => pointwise fresh, total]
    _ = _ := restart_work_eq extra tries positive
    _ = _ := total.symm

private theorem rerunWork_nonnegative (base : Oracle) : 0 ≤ rerunWork relation ajtai adversary claim base :=
  div_nonneg (add_nonneg zero_le_one (rerunRetries_nonnegative relation ajtai adversary claim _ _))
    (resampledSuccess_nonnegative relation ajtai adversary claim _ _)

private theorem successAfter_le_one (purpose : Purpose Oracle Retries) (run : Oracle) (retries : Retries) :
    successAfter relation ajtai adversary claim outcome purpose run retries ≤ 1 := by
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

private noncomputable def bounds :
    (law relation ajtai adversary claim).Bounds (Found relation outcome) where
  good := goodOf relation ajtai adversary claim
  success := successOf relation ajtai adversary claim outcome
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
    | finished => simp [law] at unhalted
    | failed => simp [law] at unhalted
  success_le_one := by
    intro state _
    cases state with
    | start => exact reductionChance_le_one relation ajtai adversary claim outcome
    | extract purpose run index rest retries =>
        exact lineLaw_le _ zero_le_one (successAfter_le_one relation ajtai adversary claim outcome purpose run) _ _
    | rerun base retries =>
        show rerunSuccess relation ajtai adversary claim outcome base retries ≤ 1
        exact retryChance_le_one relation ajtai adversary claim _ _ _
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
    | start => exact (success_start_step relation ajtai adversary claim outcome).le
    | extract purpose run index rest retries =>
        exact (success_extract_step relation ajtai adversary claim outcome
          (started_of_good relation ajtai adversary claim good)).le
    | rerun base retries => exact (success_rerun_step relation ajtai adversary claim outcome good).le
    | finished => simp [law] at unhalted
    | failed => simp [law] at unhalted
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
    | finished => simp [law] at unhalted
    | failed => simp [law] at unhalted

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

omit outcome

noncomputable def runOf (oracle : Oracle) : Made :=
  ⟨oracle, adversary.record oracle⟩

noncomputable def forksOf (oracle : Oracle) (retries : Retries) : Retried :=
  ⟨retries, fun index => adversary.record (forked relation adversary claim oracle retries index)⟩

private noncomputable def embedPurpose : Purpose Oracle Retries → Purpose Made Retried
  | .first => .first
  | .second base retries => .second (runOf relation adversary base) (forksOf relation adversary claim base retries)

@[implicit_reducible]
private noncomputable def embed : LawState → Progress
  | .start => .start
  | .extract purpose run index rest retries =>
      .extract (embedPurpose relation adversary claim purpose) (runOf relation adversary run) index rest
        (forksOf relation adversary claim run retries)
  | .rerun base retries => .rerun (runOf relation adversary base) (forksOf relation adversary claim base retries)
  | .finished base retries other otherRetries =>
      .finished (runOf relation adversary base) (forksOf relation adversary claim base retries)
        (runOf relation adversary other) (forksOf relation adversary claim other otherRetries)
  | .failed => .failed

private theorem embed_extracted (purpose : Purpose Oracle Retries) (run : Oracle) :
    ∀ pending retries,
      extracted relation ajtai claim (embedPurpose relation adversary claim purpose) (runOf relation adversary run) pending
          (forksOf relation adversary claim run retries) =
        embed relation adversary claim (lawExtracted relation ajtai adversary claim purpose run pending retries)
  | [], retries => by
      cases purpose <;>
      · simp only [extracted, afterExtraction, lawExtracted, lawAfter]
        rw [apply_ite (embed relation adversary claim)]
        exact if_congr Iff.rfl rfl rfl
  | _ :: _, _ => rfl

private theorem embed_start (purpose : Purpose Oracle Retries) (run : Oracle) :
    startExtraction relation ajtai claim (embedPurpose relation adversary claim purpose) (runOf relation adversary run) =
      embed relation adversary claim (lawStart relation ajtai adversary claim purpose run) := by
  have records : (⟨fun index => run (rhoPoint relation adversary claim index run), fun _ => adversary.record run⟩ :
      Retried) = forksOf relation adversary claim run fun index => run (rhoPoint relation adversary claim index run) :=
    congrArg (Forks.mk _) (funext fun index => by simp only [forked, Function.update_eq_self])
  unfold startExtraction lawStart
  rw [← embed_extracted, ← records]
  rfl

private theorem forksOf_update (run : Oracle) (retries : Retries) (index : Coordinate) (answer : Answer) :
    (⟨Function.update retries index answer,
        Function.update (forksOf relation adversary claim run retries).records index
          (adversary.record (Function.update run (rhoPoint relation adversary claim index run) answer))⟩ :
      Retried) = forksOf relation adversary claim run (Function.update retries index answer) := by
  refine congrArg (Forks.mk _) (funext fun other => ?_)
  by_cases same : other = index
  · subst same
    simp only [forksOf, forked, Function.update_self]
  · simp only [forksOf, forked, Function.update_of_ne same]

private theorem query_rerun (base : Oracle) (retries : Retries) (fresh : Oracle) :
    query relation claim (.rerun (runOf relation adversary base) (forksOf relation adversary claim base retries)) fresh =
      rerunOracle relation adversary claim base fresh := by
  unfold rerunOracle
  rw [context_eq, forkIndex_eq]
  rfl

private theorem embed_step (state : LawState) (fresh : Oracle)
    (unhalted : (law relation ajtai adversary claim).halted state = false) :
    ((reduction relation ajtai claim).rewinding adversary.record).step (embed relation adversary claim state) fresh =
      embed relation adversary claim (step relation ajtai adversary claim state fresh) := by
  cases state with
  | start =>
      show next relation ajtai claim .start fresh (adversary.record fresh) = _
      simp only [next, step]
      rw [apply_ite (embed relation adversary claim)]
      exact if_congr Iff.rfl (embed_start relation ajtai adversary claim .first fresh) rfl
  | extract purpose run index rest retries =>
      let embedded : Progress := .extract (embedPurpose relation adversary claim purpose) (runOf relation adversary run)
        index rest (forksOf relation adversary claim run retries)
      show next relation ajtai claim embedded fresh (adversary.record (query relation claim embedded fresh)) = _
      simp only [embedded, next, step]
      rw [apply_ite (embed relation adversary claim)]
      refine if_congr Iff.rfl ?_ rfl
      rw [← embed_extracted, ← forksOf_update]
      rfl
  | rerun base retries =>
      let embedded : Progress := .rerun (runOf relation adversary base) (forksOf relation adversary claim base retries)
      show next relation ajtai claim embedded fresh (adversary.record (query relation claim embedded fresh)) = _
      simp only [embedded, next, step]
      rw [query_rerun, apply_ite (embed relation adversary claim)]
      refine if_congr ?_ (embed_start relation ajtai adversary claim (.second base retries) _) rfl
      rw [forkIndex_eq relation adversary claim (rerunOracle relation adversary claim base fresh),
        forkIndex_eq relation adversary claim base]
      exact Iff.rfl
  | finished => simp [law] at unhalted
  | failed => simp [law] at unhalted

private theorem simulation (budget : ℕ) (tape : Fin budget → Oracle) :
    finalState relation ajtai claim adversary budget tape =
        embed relation adversary claim ((law relation ajtai adversary claim).run budget .start tape).final ∧
      (runs relation ajtai claim adversary budget tape).length =
        ((law relation ajtai adversary claim).run budget .start tape).runs := by
  have same := (law relation ajtai adversary claim).run_embed ((reduction relation ajtai claim).rewinding adversary.record)
    (embed relation adversary claim) (fun state => by cases state <;> rfl)
    (embed_step relation ajtai adversary claim) budget .start tape
  have program := (reduction relation ajtai claim).run_program adversary.record budget .start tape
  refine ⟨program.1.trans ?_, program.2.trans ?_⟩
  · exact congrArg Rewinding.Result.final same
  · exact congrArg Rewinding.Result.runs same

theorem expected_runs_le {queries : Nat} (bounded : adversary.QueryBound queries) (budget : ℕ) :
    𝔼 tape : Fin budget → Oracle, ((runs relation ajtai claim adversary budget tape).length : ℝ) ≤
      35 * queries + 653 := by
  calc
    _ = (law relation ajtai adversary claim).expectedRuns budget .start :=
      Finset.expect_congr rfl fun tape _ => by rw [(simulation relation ajtai adversary claim budget tape).2]
    _ ≤ _ := ((bounds relation ajtai adversary claim fun _ _ _ _ => True).expect_runs_le budget
        (state := .start) trivial).trans (work_start_le relation ajtai adversary claim bounded)

theorem reductionChance_le_program {queries : Nat} (bounded : adversary.QueryBound queries)
    (finished : Progress → Prop) [DecidablePred finished] (budget : ℕ) :
    budget * (reductionChance relation ajtai adversary claim (fun base retries other otherRetries =>
          finished (.finished (runOf relation adversary base) (forksOf relation adversary claim base retries)
            (runOf relation adversary other) (forksOf relation adversary claim other otherRetries))) -
        𝔼 tape : Fin budget → Oracle,
          (if finished (finalState relation ajtai claim adversary budget tape) then (1 : ℝ) else 0)) ≤
      35 * queries + 653 := by
  let outcome : Oracle → Retries → Oracle → Retries → Prop := fun base retries other otherRetries =>
    finished (.finished (runOf relation adversary base) (forksOf relation adversary claim base retries)
      (runOf relation adversary other) (forksOf relation adversary claim other otherRetries))
  have each (state : LawState) :
      (if Found relation outcome state then (1 : ℝ) else 0) ≤
        if finished (embed relation adversary claim state) then 1 else 0 := by
    cases state with
    | finished => simp only [Found, embed]; split_ifs <;> norm_num
    | _ => simp only [Found, ite_false]; split <;> norm_num
  have chance : (law relation ajtai adversary claim).chance (Found relation outcome) budget .start ≤
      𝔼 tape : Fin budget → Oracle,
        (if finished (finalState relation ajtai claim adversary budget tape) then (1 : ℝ) else 0) :=
    Finset.expect_le_expect fun tape _ => by
      rw [(simulation relation ajtai adversary claim budget tape).1]
      exact each _
  have loss := (bounds relation ajtai adversary claim outcome).budget_mul_le budget (state := .start) trivial
  have positive : (0 : ℝ) ≤ budget := Nat.cast_nonneg budget
  calc
    _ ≤ (budget : ℝ) * (reductionChance relation ajtai adversary claim outcome -
          (law relation ajtai adversary claim).chance (Found relation outcome) budget .start) :=
      mul_le_mul_of_nonneg_left (by linarith) positive
    _ ≤ _ := loss.trans (work_start_le relation ajtai adversary claim bounded)

noncomputable def solverChance (budget : ℕ) : ℝ :=
  𝔼 tape : Fin budget → Oracle,
    if (solve relation ajtai claim adversary budget tape).isSome then 1 else 0

theorem kernelChance_le_solver {queries : Nat} (bounded : adversary.QueryBound queries) (budget : ℕ) :
    budget * (kernelChance relation ajtai adversary claim - solverChance relation ajtai adversary claim budget) ≤
      35 * queries + 653 :=
  reductionChance_le_program relation ajtai adversary claim bounded
    (fun state => (output relation ajtai claim state).isSome) budget

end NightstreamFPrime.Lifecycle.RandomOracleSolver
