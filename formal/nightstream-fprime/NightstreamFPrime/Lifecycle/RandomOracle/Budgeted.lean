import NightstreamFPrime.Lifecycle.RandomOracle.Uniqueness
import NightstreamFPrime.Spec.RandomOracleBudget

/-! A budgeted extractor for the fork step of any fold transcript (`Fork.Game`). The ideal
extractor of `Uniqueness.knowledge_error_le` draws its retries from the retry law
(`Uniqueness.weight`), which depends on the base oracle. The budgeted extractor draws `budget`
samples instead, independent of the base oracle. Each sample gives one answer at the `ρ` point
of the first coordinate that has no retry yet (`pick`). The extractor keeps the answer when it
lies in that coordinate's retry set (`machine`, a `RandomOracleBudget.Rewinding`). Each sample
reruns the adversary once, so the extractor reruns it at most `budget` times (`runs_le`).

`miss_le`: on a successful base run, the search fails to end with a retry family that extracts
with chance at most the ideal law's chance of no extraction, plus the expected number of samples
(`Σ 1/mass`) over `budget`. The proof is `Rewinding.Bounds.budget_mul_le` with the ideal law
(`lineLaw`, `lineLaw_eq_retryWeight`) as the success function. `failure_le`: from the ideal
knowledge bound (`knowledge`), a successful run whose budgeted answer does not extract has
probability at most the ideal knowledge error plus `rhoCount · (Q + rhoCount) / budget`
(`Fork.expected_retries_le`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Budgeted

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.RandomOracleBudget
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.RandomOracle.Fork (Game)

attribute [local instance low] Classical.propDecidable

/-- A search state: the `ρ` coordinates that have no retry yet, and the retries so far. -/
structure Search (count : Nat) where
  pending : List (Fin count)
  retries : Fin count → Answer

/-- A rewinding machine stops after at most `budget` steps. -/
theorem runs_le {State Sample : Type} (machine : Rewinding State Sample) :
    ∀ (budget : Nat) (state : State) (tape : Fin budget → Sample),
      (machine.run budget state tape).runs ≤ budget
  | 0, _, _ => le_rfl
  | budget + 1, state, tape => by
      unfold Rewinding.run
      split
      · exact Nat.zero_le _
      · exact Nat.succ_le_succ (runs_le machine budget _ _)

variable {schedule : Schedule} {width : Nat} {Output Start : Type}
  (game : Game schedule width Output Start)
  (adversary : OracleComp (Point schedule width) Answer Output)
  {Sample : Type} (pick : Sample → Point schedule width → Answer)

local notation "Oracle" => Point schedule width → Answer
local notation "Retries" => Fin schedule.rhoCount → Answer

/-- One sample tries the first pending coordinate. It keeps the sample's answer at that
coordinate's `ρ` point when the answer is in the coordinate's retry set. -/
noncomputable def machine (oracle : Oracle) : Rewinding (Search schedule.rhoCount) Sample where
  step state sample :=
    match state.pending with
    | [] => state
    | index :: rest =>
        if pick sample (Fork.rhoPoint game adversary index oracle) ∈
            Fork.retrySets game adversary oracle index then
          ⟨rest, Function.update state.retries index
            (pick sample (Fork.rhoPoint game adversary index oracle))⟩
        else state
  halted state := state.pending.isEmpty

/-- Every coordinate is pending, and the retries start at the base answers. -/
noncomputable def start (oracle : Oracle) : Search schedule.rhoCount :=
  ⟨List.finRange schedule.rhoCount,
    fun index => oracle (Fork.rhoPoint game adversary index oracle)⟩

/-- The search state after the samples `tape`. -/
noncomputable def final (budget : Nat) (oracle : Oracle) (tape : Fin budget → Sample) :
    Search schedule.rhoCount :=
  ((machine game adversary pick oracle).run budget (start game adversary oracle) tape).final

/-- The retry family after the samples `tape`, when every coordinate has a retry. -/
noncomputable def found (budget : Nat) (oracle : Oracle) (tape : Fin budget → Sample) :
    Option Retries :=
  if (final game adversary pick budget oracle tape).pending = [] then
    some (final game adversary pick budget oracle tape).retries
  else none

/-- The budgeted extractor's answer: `answer` of the found retry family, else `fallback`. -/
noncomputable def extract {Result : Type} (answer : Oracle → Retries → Result)
    (fallback : Oracle → Result) (budget : Nat) (oracle : Oracle)
    (tape : Fin budget → Sample) : Result :=
  match found game adversary pick budget oracle tape with
  | some retries => answer oracle retries
  | none => fallback oracle

theorem extract_eq {Result : Type} {answer : Oracle → Retries → Result}
    {fallback : Oracle → Result} {budget : Nat} {oracle : Oracle} {tape : Fin budget → Sample}
    {retries : Retries} (returned : found game adversary pick budget oracle tape = some retries) :
    extract game adversary pick answer fallback budget oracle tape = answer oracle retries := by
  unfold extract
  rw [returned]

private theorem lineAverage_zero (set : Set Answer) : lineAverage set (fun _ => (0 : ℝ)) = 0 := by
  simp [lineAverage]

private theorem step_cons (oracle : Point schedule width → Answer) (index : Fin schedule.rhoCount)
    (rest : List (Fin schedule.rhoCount)) (retries : Fin schedule.rhoCount → Answer)
    (sample : Sample) :
    (machine game adversary pick oracle).step ⟨index :: rest, retries⟩ sample =
      if pick sample (Fork.rhoPoint game adversary index oracle) ∈
          Fork.retrySets game adversary oracle index then
        ⟨rest, Function.update retries index
          (pick sample (Fork.rhoPoint game adversary index oracle))⟩
      else ⟨index :: rest, retries⟩ :=
  rfl

section Bounds

variable [Fintype Sample] [Nonempty Sample]
  (Extracts : (Point schedule width → Answer) → (Fin schedule.rhoCount → Answer) → Prop)

/-- The ideal law's chance that the retries extract, once the pending coordinates have
retries. -/
noncomputable def success (oracle : Oracle) (state : Search schedule.rhoCount) : ℝ :=
  lineLaw (Fork.retrySets game adversary oracle)
    (fun retries => if Extracts oracle retries then 1 else 0) state.pending state.retries

/-- The expected number of samples that the pending coordinates still need. -/
noncomputable def work (oracle : Oracle) (state : Search schedule.rhoCount) : ℝ :=
  (state.pending.map fun index => 1 / mass (Fork.retrySets game adversary oracle index)).sum

/-- Every coordinate has a retry, and the retries extract. -/
def Goal (oracle : Oracle) (state : Search schedule.rhoCount) : Prop :=
  state.pending = [] ∧ Extracts oracle state.retries

variable (uniform : ∀ (point : Point schedule width) (value : Answer → ℝ),
  𝔼 sample, value (pick sample point) = 𝔼 answer, value answer)

include uniform in
/-- The ideal law as the success function and `Σ 1/mass` as the work function of the
search, on a successful base run. -/
noncomputable def bounds {oracle : Oracle} (succeeds : Fork.Succeeds game adversary oracle) :
    (machine game adversary pick oracle).Bounds (Goal Extracts oracle) where
  good _ := True
  success := success game adversary Extracts oracle
  work := work game adversary oracle
  good_step _ _ _ _ := trivial
  success_le_one state _ :=
    lineLaw_le _ zero_le_one (fun _ => by split <;> norm_num) state.pending state.retries
  success_halted state _ halted := by
    obtain ⟨pending, retries⟩ := state
    cases pending with
    | nil =>
        by_cases extracts : Extracts oracle retries
        · simp [success, lineLaw, Goal, extracts]
        · simp [success, lineLaw, Goal, extracts]
    | cons _ _ => simp [machine] at halted
  success_step state _ unhalted := by
    obtain ⟨pending, retries⟩ := state
    cases pending with
    | nil => simp [machine] at unhalted
    | cons index rest =>
        have positive := Fork.mass_ne_zero (Fork.base_mem_retrySets game adversary succeeds index)
        let value : Answer → ℝ := fun answer =>
          lineLaw (Fork.retrySets game adversary oracle)
            (fun retries => if Extracts oracle retries then 1 else 0) rest
            (Function.update retries index answer)
        let next : Answer → ℝ := fun answer =>
          if answer ∈ Fork.retrySets game adversary oracle index then value answer
          else lineAverage (Fork.retrySets game adversary oracle index) value
        have stepped (sample : Sample) :
            success game adversary Extracts oracle
                ((machine game adversary pick oracle).step ⟨index :: rest, retries⟩ sample) =
              next (pick sample (Fork.rhoPoint game adversary index oracle)) := by
          rw [step_cons]
          split
          · next inside =>
              simp only [next, ite_eq_left inside]
              rfl
          · next outside =>
              simp only [next, ite_eq_right outside]
              rfl
        refine le_of_eq ?_
        refine (lineAverage_step _ positive value).symm.trans ?_
        exact ((Finset.expect_congr rfl fun sample _ => stepped sample).trans
          (uniform (Fork.rhoPoint game adversary index oracle) next)).symm
  work_nonnegative state _ :=
    List.sum_nonneg fun term member => by
      obtain ⟨index, _, rfl⟩ := List.mem_map.mp member
      exact div_nonneg zero_le_one (mass_nonnegative _)
  work_step state _ unhalted := by
    obtain ⟨pending, retries⟩ := state
    cases pending with
    | nil => simp [machine] at unhalted
    | cons index rest =>
        have positive := Fork.mass_ne_zero (Fork.base_mem_retrySets game adversary succeeds index)
        let later : ℝ := work game adversary oracle ⟨rest, retries⟩
        let next : Answer → ℝ := fun answer =>
          if answer ∈ Fork.retrySets game adversary oracle index then later
          else 1 / mass (Fork.retrySets game adversary oracle index) + later
        have stepped (sample : Sample) :
            work game adversary oracle
                ((machine game adversary pick oracle).step ⟨index :: rest, retries⟩ sample) =
              next (pick sample (Fork.rhoPoint game adversary index oracle)) := by
          rw [step_cons]
          split
          · next inside =>
              simp only [next, ite_eq_left inside]
              rfl
          · next outside =>
              simp only [next, ite_eq_right outside]
              rfl
        have expected := line_work_step _ positive later fun _ => 0
        simp only [lineAverage_zero, add_zero] at expected
        have total : work game adversary oracle ⟨index :: rest, retries⟩ =
            1 / mass (Fork.retrySets game adversary oracle index) + later := by
          simp only [work, List.map_cons, List.sum_cons]
          rfl
        calc
          _ = 1 + 𝔼 answer, next answer := by
            rw [(Finset.expect_congr rfl fun sample _ => stepped sample).trans
              (uniform (Fork.rhoPoint game adversary index oracle) next)]
          _ = 1 / mass (Fork.retrySets game adversary oracle index) + later := expected
          _ ≤ _ := total.symm.le

include uniform in
/-- On a successful base run, the search misses an extracting retry family with chance at most
the ideal chance of no extraction plus the expected number of samples over `budget`. -/
theorem miss_le (budget : Nat) (positive : 0 < budget) {oracle : Oracle}
    (succeeds : Fork.Succeeds game adversary oracle) :
    𝔼 tape : Fin budget → Sample,
        (if ∃ retries, found game adversary pick budget oracle tape = some retries ∧
          Extracts oracle retries then (0 : ℝ) else 1) ≤
      1 - ∑ retries, Uniqueness.weight game adversary oracle retries *
          (if Extracts oracle retries then 1 else 0) +
        (∑ index, retryCount (Fork.Hits game adversary index)
          (Fork.rhoPoint game adversary index oracle) oracle) / budget := by
  have budgeted := (bounds game adversary pick Extracts uniform succeeds).budget_mul_le budget
    (state := start game adversary oracle) trivial
  have successStart : (bounds game adversary pick Extracts uniform succeeds).success
      (start game adversary oracle) =
      ∑ retries, Uniqueness.weight game adversary oracle retries *
        (if Extracts oracle retries then 1 else 0) :=
    lineLaw_eq_retryWeight _ _ _
  have workStart : (bounds game adversary pick Extracts uniform succeeds).work
      (start game adversary oracle) =
      ∑ index, retryCount (Fork.Hits game adversary index)
        (Fork.rhoPoint game adversary index oracle) oracle :=
    (Fin.sum_univ_def _).symm
  have pointwise (tape : Fin budget → Sample) :
      (if ∃ retries, found game adversary pick budget oracle tape = some retries ∧
        Extracts oracle retries then (0 : ℝ) else 1) =
        1 - if Goal Extracts oracle (final game adversary pick budget oracle tape) then 1 else 0 := by
    have same : (∃ retries, found game adversary pick budget oracle tape = some retries ∧
        Extracts oracle retries) ↔ Goal Extracts oracle (final game adversary pick budget oracle tape) := by
      unfold found Goal
      constructor
      · rintro ⟨retries, returned, extracts⟩
        split at returned
        · next empty =>
            cases returned
            exact ⟨empty, extracts⟩
        · cases returned
      · rintro ⟨empty, extracts⟩
        exact ⟨_, ite_eq_left empty, extracts⟩
    by_cases goal : Goal Extracts oracle (final game adversary pick budget oracle tape)
    · rw [ite_eq_left (same.mpr goal), ite_eq_left goal, sub_self]
    · rw [ite_eq_right (fun found => goal (same.mp found)), ite_eq_right goal, sub_zero]
  have chance : 𝔼 tape : Fin budget → Sample,
      (if ∃ retries, found game adversary pick budget oracle tape = some retries ∧
        Extracts oracle retries then (0 : ℝ) else 1) =
      1 - (machine game adversary pick oracle).chance (Goal Extracts oracle) budget
        (start game adversary oracle) := by
    simp only [pointwise]
    rw [Finset.expect_sub_distrib, Finset.expect_const Finset.univ_nonempty]
    rfl
  rw [successStart, workStart, mul_comm (budget : ℝ)] at budgeted
  have cast : (0 : ℝ) < budget := by exact_mod_cast positive
  have divided := (le_div_iff₀ cast).mpr budgeted
  rw [chance]
  linarith

include uniform in
/-- `Uniqueness.knowledge_error_le` for the budgeted extractor: a successful run whose budgeted
answer does not extract has probability at most the ideal knowledge error `error` plus
`rhoCount · (Q + rhoCount) / budget`. -/
theorem failure_le (budget : Nat) (positive : 0 < budget) {Result : Type}
    (Extracted : Output → Result → Prop) (answer : Oracle → Retries → Result)
    (fallback : Oracle → Result)
    (extracted : ∀ {oracle retries}, Extracts oracle retries →
      Extracted (adversary.run oracle) (answer oracle retries))
    (succeedsOf : ∀ {oracle retries}, Extracts oracle retries →
      Fork.Succeeds game adversary oracle)
    (error : ℝ)
    (knowledge : 𝔼 oracle, (if Fork.Succeeds game adversary oracle then (1 : ℝ) else 0) ≤
      𝔼 oracle, ∑ retries, Uniqueness.weight game adversary oracle retries *
          (if Extracts oracle retries then 1 else 0) + error)
    {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, 𝔼 tape : Fin budget → Sample,
        (if Fork.Succeeds game adversary oracle ∧ ¬ Extracted (adversary.run oracle)
            (extract game adversary pick answer fallback budget oracle tape) then (1 : ℝ)
          else 0) ≤
      error + schedule.rhoCount * ((queries + schedule.rhoCount : Nat) : ℝ) / budget := by
  let extracting : Oracle → ℝ := fun oracle =>
    ∑ retries, Uniqueness.weight game adversary oracle retries *
      (if Extracts oracle retries then 1 else 0)
  let samples : Oracle → ℝ := fun oracle =>
    if Fork.Succeeds game adversary oracle then
      ∑ index, retryCount (Fork.Hits game adversary index)
        (Fork.rhoPoint game adversary index oracle) oracle
    else 0
  have cast : (0 : ℝ) < budget := by exact_mod_cast positive
  have pointwise (oracle : Oracle) :
      𝔼 tape : Fin budget → Sample,
          (if Fork.Succeeds game adversary oracle ∧ ¬ Extracted (adversary.run oracle)
              (extract game adversary pick answer fallback budget oracle tape) then (1 : ℝ)
            else 0) ≤
        (if Fork.Succeeds game adversary oracle then (1 : ℝ) else 0) - extracting oracle +
          samples oracle / budget := by
    by_cases succeeds : Fork.Succeeds game adversary oracle
    · have missed := miss_le game adversary pick Extracts uniform budget positive succeeds
      calc
        _ ≤ 𝔼 tape : Fin budget → Sample,
            (if ∃ retries, found game adversary pick budget oracle tape = some retries ∧
              Extracts oracle retries then (0 : ℝ) else 1) :=
          Finset.expect_le_expect fun tape _ => by
            by_cases hit : ∃ retries,
                found game adversary pick budget oracle tape = some retries ∧
                  Extracts oracle retries
            · obtain ⟨retries, returned, extracts⟩ := hit
              have holds := extracted extracts
              rw [← extract_eq game adversary pick returned (answer := answer)
                (fallback := fallback)] at holds
              rw [ite_eq_right (fun failed => failed.2 holds)]
              split <;> norm_num
            · rw [ite_eq_right hit]
              split <;> norm_num
        _ ≤ _ := by
          simp only [samples, ite_eq_left succeeds, extracting]
          exact missed
    · have none : extracting oracle = 0 :=
        Finset.sum_eq_zero fun retries _ => by
          rw [ite_eq_right fun extracts => succeeds (succeedsOf extracts), mul_zero]
      simp [succeeds, samples, none]
  have summed := Finset.expect_le_expect (s := Finset.univ)
    fun oracle (_ : oracle ∈ Finset.univ) => pointwise oracle
  rw [Finset.expect_add_distrib, Finset.expect_sub_distrib, ← Finset.expect_div] at summed
  have retries := Fork.expected_retries_le game adversary bounded
  have scaled : (𝔼 oracle, samples oracle) / budget ≤
      schedule.rhoCount * ((queries + schedule.rhoCount : Nat) : ℝ) / budget :=
    div_le_div_of_nonneg_right retries cast.le
  linarith

end Bounds

end NightstreamFPrime.Lifecycle.RandomOracle.Budgeted
