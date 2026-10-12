import NightstreamFPrime.Spec.RandomOracle
import Mathlib.Algebra.BigOperators.Fin

/-!
Owns rewinding algorithms with a strict budget, and the conversion from
expected work to strict work (Markov's inequality on the coin tape).

Inputs: a step machine that runs the adversary once per step on one fresh
uniform sample, a goal on its final states, and a certificate of potentials
on its reachable states.

Outputs:
- `Rewinding.run`, `runs_le`: the machine with budget `T` reads a tape of `T`
  samples and makes at most `T` adversary runs on every tape;
- `Bounds.expect_runs_le`: its expected runs are at most the work potential,
  for every budget;
- `Bounds.budget_mul_le`: with budget `T`, it reaches the goal with chance at
  least the success potential minus `work / T`.

The certificate holds closed forms of the eventual success and of the expected
remaining runs. It needs no law of the run count: an unfinished run has spent
the whole budget, so `T · Pr[unfinished] ≤ 𝔼 runs`.

Does not own: which machine a reduction runs, or the claim that one step is
one adversary run; each machine's module states that.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.RandomOracleBudget

open scoped BigOperators

/-- An algorithm that runs the adversary once per step, on one fresh uniform
sample, until it halts. -/
structure Rewinding (State Sample : Type*) where
  step : State → Sample → State
  halted : State → Bool

/-- Where a budgeted run ends, and how many adversary runs it made. -/
structure Rewinding.Result (State : Type*) where
  final : State
  runs : ℕ

variable {State Sample : Type*} (machine : Rewinding State Sample)

/-- The machine with a strict budget: step `k` reads `tape k`; it stops when
it halts or when the budget is spent. -/
def Rewinding.run : (budget : ℕ) → State → (Fin budget → Sample) → Rewinding.Result State
  | 0, state, _ => ⟨state, 0⟩
  | budget + 1, state, tape =>
      if machine.halted state then ⟨state, 0⟩ else
        let rest : Rewinding.Result State :=
          Rewinding.run budget (machine.step state (tape 0)) (Fin.tail tape)
        ⟨rest.final, rest.runs + 1⟩

/-- **Strict work.** No tape makes the machine exceed its budget. -/
theorem Rewinding.runs_le (budget : ℕ) (state : State) (tape : Fin budget → Sample) :
    (machine.run budget state tape).runs ≤ budget := by
  induction budget generalizing state with
  | zero => exact le_rfl
  | succ budget induction =>
      unfold Rewinding.run
      split
      · exact Nat.zero_le _
      · exact Nat.succ_le_succ (induction _ _)

/-- A run that ends unhalted has spent its whole budget. -/
private theorem runs_eq_of_not_halted (budget : ℕ) (state : State) (tape : Fin budget → Sample)
    (unfinished : machine.halted (machine.run budget state tape).final = false) :
    (machine.run budget state tape).runs = budget := by
  induction budget generalizing state with
  | zero => rfl
  | succ budget induction =>
      unfold Rewinding.run at unfinished ⊢
      by_cases halted : machine.halted state = true
      · simp [halted] at unfinished
      · simp only [halted, Bool.false_eq_true, if_false] at unfinished ⊢
        exact congrArg (· + 1) (induction _ _ unfinished)

variable [Fintype Sample] [Nonempty Sample]

omit [Nonempty Sample] in
/-- A uniform tape of `budget + 1` samples is its first sample and a uniform
tape of the rest. -/
private theorem expect_tape (budget : ℕ) (value : (Fin (budget + 1) → Sample) → ℝ) :
    𝔼 tape, value tape = 𝔼 sample, 𝔼 rest : Fin budget → Sample, value (Fin.cons sample rest) := by
  rw [← Finset.expect_product' Finset.univ Finset.univ (fun sample rest => value (Fin.cons sample rest)),
    Finset.univ_product_univ]
  exact Fintype.expect_equiv (Fin.consEquiv fun _ => Sample).symm _ _ fun tape => by
    simp [Fin.consEquiv, Fin.cons_self_tail]

omit [Nonempty Sample] in
/-- One step of the budgeted run, averaged over the tape. -/
private theorem expect_run_succ (budget : ℕ) (state : State)
    (unhalted : machine.halted state = false) (value : Rewinding.Result State → ℝ) :
    𝔼 tape, value (machine.run (budget + 1) state tape) =
      𝔼 sample, 𝔼 rest : Fin budget → Sample,
        value ⟨(machine.run budget (machine.step state sample) rest).final,
          (machine.run budget (machine.step state sample) rest).runs + 1⟩ := by
  rw [expect_tape]
  refine Finset.expect_congr rfl fun sample _ => Finset.expect_congr rfl fun rest _ => ?_
  simp [Rewinding.run, unhalted, Fin.cons_zero, Fin.tail_cons]

/-- The expected adversary runs of the budgeted run. -/
noncomputable def Rewinding.expectedRuns (budget : ℕ) (state : State) : ℝ :=
  𝔼 tape : Fin budget → Sample, ((machine.run budget state tape).runs : ℝ)

/-- The chance that the budgeted run ends in `goal`. -/
noncomputable def Rewinding.chance (goal : State → Prop) [DecidablePred goal] (budget : ℕ)
    (state : State) : ℝ :=
  𝔼 tape : Fin budget → Sample, if goal (machine.run budget state tape).final then 1 else 0

/-- A certificate on `good` states: `success` never exceeds the next step's
average (it is a lower bound on the eventual chance of `goal`), and `work` pays
one adversary run per step (an upper bound on the expected remaining runs). -/
structure Rewinding.Bounds (goal : State → Prop) [DecidablePred goal] where
  good : State → Prop
  success : State → ℝ
  work : State → ℝ
  good_step : ∀ state, good state → machine.halted state = false →
    ∀ sample, good (machine.step state sample)
  success_le_one : ∀ state, good state → success state ≤ 1
  success_halted : ∀ state, good state → machine.halted state = true →
    success state ≤ if goal state then 1 else 0
  success_step : ∀ state, good state → machine.halted state = false →
    success state ≤ 𝔼 sample, success (machine.step state sample)
  work_nonnegative : ∀ state, good state → 0 ≤ work state
  work_step : ∀ state, good state → machine.halted state = false →
    1 + 𝔼 sample, work (machine.step state sample) ≤ work state

variable {machine} {goal : State → Prop} [DecidablePred goal] (bounds : machine.Bounds goal)

/-- **Expected work.** For every budget, the expected runs from a good state
are at most its work potential. -/
theorem Rewinding.Bounds.expect_runs_le (budget : ℕ) {state : State} (good : bounds.good state) :
    machine.expectedRuns budget state ≤ bounds.work state := by
  unfold Rewinding.expectedRuns
  induction budget generalizing state with
  | zero =>
      simp only [Rewinding.run, Nat.cast_zero, Finset.expect_const_zero]
      exact bounds.work_nonnegative state good
  | succ budget induction =>
      by_cases halted : machine.halted state = true
      · simp only [Rewinding.run, halted, if_true, Nat.cast_zero, Finset.expect_const_zero]
        exact bounds.work_nonnegative state good
      · have unhalted : machine.halted state = false := by simpa using halted
        rw [expect_run_succ machine budget state unhalted (fun result => (result.runs : ℝ))]
        refine le_trans ?_ (bounds.work_step state good unhalted)
        calc
          _ = 𝔼 sample, (1 + 𝔼 rest : Fin budget → Sample,
                ((machine.run budget (machine.step state sample) rest).runs : ℝ)) := by
            refine Finset.expect_congr rfl fun sample _ => ?_
            push_cast
            rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty, add_comm]
          _ ≤ 𝔼 sample, (1 + bounds.work (machine.step state sample)) :=
            Finset.expect_le_expect fun sample _ =>
              add_le_add le_rfl (induction (bounds.good_step state good unhalted sample))
          _ = _ := by rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty]

/-- The success potential is at most the chance of the goal within the budget
plus the chance that the run ends unhalted. -/
private theorem success_le_chance_add (budget : ℕ) {state : State} (good : bounds.good state) :
    bounds.success state ≤ machine.chance goal budget state +
      𝔼 tape : Fin budget → Sample,
        (if machine.halted (machine.run budget state tape).final = false then (1 : ℝ) else 0) := by
  induction budget generalizing state with
  | zero =>
      have ended (tape : Fin 0 → Sample) : machine.run 0 state tape = ⟨state, 0⟩ := rfl
      unfold Rewinding.chance
      simp only [ended]
      rw [Fintype.expect_const, Fintype.expect_const]
      by_cases halted : machine.halted state = true
      · simp only [halted, Bool.true_eq_false, if_false, add_zero]
        exact bounds.success_halted state good halted
      · have unhalted : machine.halted state = false := by simpa using halted
        have one := bounds.success_le_one state good
        simp only [unhalted, if_true]
        split <;> linarith
  | succ budget induction =>
      by_cases halted : machine.halted state = true
      · simp only [Rewinding.chance, Rewinding.run, halted, if_true, Bool.true_eq_false, if_false,
          Fintype.expect_const, add_zero]
        exact bounds.success_halted state good halted
      · have unhalted : machine.halted state = false := by simpa using halted
        unfold Rewinding.chance
        rw [expect_run_succ machine budget state unhalted (fun result => if goal result.final then (1 : ℝ) else 0),
          expect_run_succ machine budget state unhalted
            (fun result => if machine.halted result.final = false then (1 : ℝ) else 0),
          ← Finset.expect_add_distrib]
        dsimp only
        refine (bounds.success_step state good unhalted).trans (Finset.expect_le_expect fun sample _ => ?_)
        have step := induction (bounds.good_step state good unhalted sample)
        unfold Rewinding.chance at step
        exact step

/-- **Markov on the tape.** With budget `T`, the machine reaches the goal with
chance at least the success potential minus `work / T`. -/
theorem Rewinding.Bounds.budget_mul_le (budget : ℕ) {state : State} (good : bounds.good state) :
    budget * (bounds.success state - machine.chance goal budget state) ≤ bounds.work state := by
  have split := success_le_chance_add bounds budget good
  have unfinished : (budget : ℝ) * 𝔼 tape : Fin budget → Sample,
      (if machine.halted (machine.run budget state tape).final = false then (1 : ℝ) else 0) ≤
        𝔼 tape : Fin budget → Sample, ((machine.run budget state tape).runs : ℝ) := by
    rw [Finset.mul_expect]
    refine Finset.expect_le_expect fun tape _ => ?_
    split
    · rename_i unhalted
      rw [runs_eq_of_not_halted machine budget state tape unhalted, mul_one]
    · simp
  have positive : (0 : ℝ) ≤ budget := Nat.cast_nonneg budget
  calc
    _ ≤ (budget : ℝ) * 𝔼 tape : Fin budget → Sample,
          (if machine.halted (machine.run budget state tape).final = false then (1 : ℝ) else 0) :=
      mul_le_mul_of_nonneg_left (by linarith) positive
    _ ≤ _ := unfinished.trans (bounds.expect_runs_le budget good)

/-! ## Run-until-success loops -/

section Loops

open RandomOracle

variable {Answer : Type*} [Fintype Answer]

attribute [local instance low] Classical.propDecidable

/-- The outcome law of one run-until-success loop: uniform on `set`. -/
noncomputable def lineAverage (set : Set Answer) (value : Answer → ℝ) : ℝ :=
  (𝔼 answer, if answer ∈ set then value answer else 0) / mass set

private theorem expect_mem_eq_mass (set : Set Answer) :
    𝔼 answer, (if answer ∈ set then (1 : ℝ) else 0) = mass set := rfl

/-- One attempt of a loop on `set`: an answer in `set` ends it with `value`;
any other answer starts the loop again. -/
theorem lineAverage_step [Nonempty Answer] (set : Set Answer) (positive : mass set ≠ 0) (value : Answer → ℝ) :
    𝔼 answer, (if answer ∈ set then value answer else lineAverage set value) =
      lineAverage set value := by
  have split (answer : Answer) : (if answer ∈ set then value answer else lineAverage set value) =
      (if answer ∈ set then value answer else 0) +
        (1 - if answer ∈ set then (1 : ℝ) else 0) * lineAverage set value := by
    split <;> simp
  simp only [split]
  rw [Finset.expect_add_distrib, ← Finset.expect_mul, Finset.expect_sub_distrib,
    Finset.expect_const Finset.univ_nonempty, expect_mem_eq_mass]
  unfold lineAverage
  field_simp
  ring

/-- The work of one attempt: the loop takes `1 / mass set` expected attempts
before it continues with `later + value`. -/
theorem line_work_step [Nonempty Answer] (set : Set Answer) (positive : mass set ≠ 0) (later : ℝ)
    (value : Answer → ℝ) :
    1 + 𝔼 answer, (if answer ∈ set then later + value answer
        else 1 / mass set + later + lineAverage set value) =
      1 / mass set + later + lineAverage set value := by
  have split (answer : Answer) : (if answer ∈ set then later + value answer
        else 1 / mass set + later + lineAverage set value) =
      (if answer ∈ set then value answer else 0) + later +
        (1 - if answer ∈ set then (1 : ℝ) else 0) * (1 / mass set + lineAverage set value) := by
    split <;> ring
  simp only [split]
  rw [Finset.expect_add_distrib, Finset.expect_add_distrib, ← Finset.expect_mul,
    Finset.expect_sub_distrib, Finset.expect_const Finset.univ_nonempty,
    Finset.expect_const Finset.univ_nonempty, expect_mem_eq_mass]
  unfold lineAverage
  field_simp
  ring

/-- The loop's outcome law never exceeds a bound on its values. -/
theorem lineAverage_le (set : Set Answer) {value : Answer → ℝ} {bound : ℝ} (nonnegative : 0 ≤ bound)
    (below : ∀ answer, value answer ≤ bound) : lineAverage set value ≤ bound := by
  unfold lineAverage
  rcases (mass_nonnegative set).lt_or_eq with positive | zero
  · rw [div_le_iff₀ positive, ← expect_mem_eq_mass, Finset.mul_expect]
    exact Finset.expect_le_expect fun answer _ => by split <;> simp [below]
  · rw [← zero, div_zero]
    exact nonnegative

theorem lineAverage_nonnegative (set : Set Answer) {value : Answer → ℝ}
    (nonnegative : ∀ answer, 0 ≤ value answer) : 0 ≤ lineAverage set value :=
  div_nonneg (Finset.expect_nonneg fun answer _ => by split <;> simp [nonnegative])
    (mass_nonnegative set)

/-- Loops on the coordinates in `pending`, in order; each ends uniform on its
coordinate's set. -/
noncomputable def lineLaw {Index : Type*} [DecidableEq Index] (sets : Index → Set Answer)
    (value : (Index → Answer) → ℝ) : List Index → (Index → Answer) → ℝ
  | [], retries => value retries
  | index :: rest, retries =>
      lineAverage (sets index) fun answer => lineLaw sets value rest (Function.update retries index answer)

theorem lineLaw_le {Index : Type*} [DecidableEq Index] (sets : Index → Set Answer)
    {value : (Index → Answer) → ℝ} {bound : ℝ} (nonnegative : 0 ≤ bound)
    (below : ∀ retries, value retries ≤ bound) :
    ∀ pending retries, lineLaw sets value pending retries ≤ bound
  | [], retries => below retries
  | _ :: rest, _ => lineAverage_le _ nonnegative fun _ => lineLaw_le sets nonnegative below rest _

theorem lineLaw_nonnegative {Index : Type*} [DecidableEq Index] (sets : Index → Set Answer)
    {value : (Index → Answer) → ℝ} (nonnegative : ∀ retries, 0 ≤ value retries) :
    ∀ pending retries, 0 ≤ lineLaw sets value pending retries
  | [], retries => nonnegative retries
  | _ :: rest, _ => lineAverage_nonnegative _ fun _ => lineLaw_nonnegative sets nonnegative rest _

/-- The weight of one answer in a coordinate's outcome law (`retryWeight`'s
factor). -/
private noncomputable def factor {Index : Type*} (sets : Index → Set Answer) (index : Index)
    (answer : Answer) : ℝ :=
  (if answer ∈ sets index then 1 else 0) / (Fintype.card Answer * mass (sets index))

private theorem lineAverage_eq_sum (set : Set Answer) (value : Answer → ℝ) :
    lineAverage set value =
      ∑ answer, (if answer ∈ set then 1 else 0) / (Fintype.card Answer * mass set) * value answer := by
  unfold lineAverage
  rw [Finset.expect_eq_sum_div_card, Finset.card_univ, div_div, div_eq_mul_inv, Finset.sum_mul]
  refine Finset.sum_congr rfl fun answer _ => ?_
  split <;> ring

/-- The sequential loops on a duplicate-free `pending` list give the product
law on the pending coordinates; the others keep their values. -/
private theorem lineLaw_eq_sum {n : ℕ} (sets : Fin n → Set Answer) (value : (Fin n → Answer) → ℝ) :
    ∀ pending : List (Fin n), pending.Nodup → ∀ retries,
      lineLaw sets value pending retries =
        ∑ other, (pending.map fun index => factor sets index (other index)).prod *
          (if ∀ index ∉ pending, other index = retries index then 1 else 0) * value other
  | [], _, retries => by
      have only (other : Fin n → Answer) :
          (if ∀ index ∉ ([] : List (Fin n)), other index = retries index then (1 : ℝ) else 0) =
            if other = retries then 1 else 0 := by
        simp [funext_iff]
      simp only [List.map_nil, List.prod_nil, one_mul, only, ite_mul, zero_mul, one_mul,
        Finset.sum_ite_eq', Finset.mem_univ, if_true, lineLaw]
  | index :: rest, nodup, retries => by
      rw [List.nodup_cons] at nodup
      rw [lineLaw, lineAverage_eq_sum]
      simp only [lineLaw_eq_sum sets value rest nodup.2, Finset.mul_sum]
      rw [Finset.sum_comm]
      refine Finset.sum_congr rfl fun other _ => ?_
      have condition (answer : Answer) :
          (if ∀ j ∉ rest, other j = Function.update retries index answer j then (1 : ℝ) else 0) =
            (if other index = answer then 1 else 0) *
              (if ∀ j ∉ index :: rest, other j = retries j then 1 else 0) := by
        by_cases same : other index = answer
        · have iff : (∀ j ∉ rest, other j = Function.update retries index answer j) ↔
              ∀ j ∉ index :: rest, other j = retries j := by
            constructor
            · intro holds j outside
              rw [List.mem_cons, not_or] at outside
              rw [holds j outside.2, Function.update_of_ne outside.1]
            · intro holds j outside
              by_cases at_index : j = index
              · subst at_index
                rw [Function.update_self, same]
              · rw [Function.update_of_ne at_index]
                exact holds j (by rw [List.mem_cons, not_or]; exact ⟨at_index, outside⟩)
          simp only [same, if_true, one_mul, iff]
        · have fails : ¬ ∀ j ∉ rest, other j = Function.update retries index answer j := fun holds =>
            same (by simpa using holds index nodup.1)
          simp [same, fails]
      calc
        _ = ∑ answer, factor sets index answer * (if other index = answer then 1 else 0) *
              ((rest.map fun j => factor sets j (other j)).prod *
                (if ∀ j ∉ index :: rest, other j = retries j then 1 else 0) * value other) := by
          refine Finset.sum_congr rfl fun answer _ => ?_
          rw [condition]
          unfold factor
          ring
        _ = _ := by
          rw [← Finset.sum_mul, Finset.sum_mul_boole, if_pos (Finset.mem_univ _), List.map_cons,
            List.prod_cons]
          ring

/-- **Sequential loops on every coordinate give the existing retry law**
(`RandomOracle.retryWeight`). -/
theorem lineLaw_eq_retryWeight {n : ℕ} (sets : Fin n → Set Answer) (value : (Fin n → Answer) → ℝ)
    (retries : Fin n → Answer) :
    lineLaw sets value (List.finRange n) retries = ∑ other, retryWeight sets other * value other := by
  rw [lineLaw_eq_sum sets value _ (List.nodup_finRange n)]
  refine Finset.sum_congr rfl fun other _ => ?_
  have all : ∀ index ∉ List.finRange n, other index = retries index := fun index outside =>
    absurd (List.mem_finRange index) outside
  rw [if_pos all, mul_one, retryWeight, Fin.prod_univ_def]
  rfl

end Loops

/-! ## Restarting loops -/

/-- A loop that restarts after each failed attempt: an attempt succeeds with
`found`, fails and restarts with `1 - tries`. Its success is `𝔼 found / 𝔼 tries`. -/
theorem restart_success_eq (found tries : Sample → ℝ) (positive : 𝔼 sample, tries sample ≠ 0) :
    𝔼 sample, (found sample + (1 - tries sample) * ((𝔼 s, found s) / 𝔼 s, tries s)) =
      (𝔼 s, found s) / 𝔼 s, tries s := by
  rw [Finset.expect_add_distrib, ← Finset.expect_mul, Finset.expect_sub_distrib,
    Finset.expect_const Finset.univ_nonempty]
  field_simp
  ring

/-- The work of a restarting loop: one run per attempt plus `extra` runs, for
`1 / 𝔼 tries` expected attempts. -/
theorem restart_work_eq (extra tries : Sample → ℝ) (positive : 𝔼 sample, tries sample ≠ 0) :
    1 + 𝔼 sample, (extra sample + (1 - tries sample) * ((1 + 𝔼 s, extra s) / 𝔼 s, tries s)) =
      (1 + 𝔼 s, extra s) / 𝔼 s, tries s := by
  rw [Finset.expect_add_distrib, ← Finset.expect_mul, Finset.expect_sub_distrib,
    Finset.expect_const Finset.univ_nonempty]
  field_simp
  ring

end NightstreamFPrime.Spec.RandomOracleBudget
