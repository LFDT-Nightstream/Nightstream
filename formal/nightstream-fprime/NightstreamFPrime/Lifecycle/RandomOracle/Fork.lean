import Mathlib.Algebra.Order.Archimedean.Real.Basic
import NightstreamFPrime.Lifecycle.RandomOracle.Test
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Law
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Challenge

/-! The fork step of the random-oracle chain for any fold transcript `Schedule`: the step that
`RandomOracleExtraction` states for the one-circuit fold. An oracle answer at a `ρ` point is one
draw of the production Π_RLC sampler, and the challenge is its scalar (`readRho`). One scalar
has probability at most `sampleError` (`fiber_mass_le`).

The fold supplies a `Game`: the statement and the messages that an adversary output absorbs,
the start claim that the transcript does not absorb, and the success event of an oracle run.
The extractor reruns the adversary once per `ρ` coordinate with a new answer at that
coordinate's point (`forked`), drawn from the answers that keep the run successful at the same
point (`retrySets`, weighted by `retryWeight`). A retry is valid (`ForkValid`) when it also
samples a new scalar and keeps the start claim. A valid retry has the base run's statement,
messages and Π_CCS coins, and its `ρ` challenges differ from the base run's only at its
coordinate (`ForkValid.identifies`, `ForkValid.coins_eq`, `ForkValid.rho_eq`, `ForkValid.agree`,
`ForkValid.changed`).

`fork_failure_le`: an adversary with at most `Q` queries succeeds and gets an invalid retry
family with probability at most `rhoCount · (Q + rhoCount) · sampleError` plus the expected
chance that a retry moves the start claim (`mismatchChance`). `fork_failure_le_of_mismatch`
replaces that chance by any larger per-retry chance (`retryChance`). `expected_retries_le`:
the expected number of reruns is at most `rhoCount · (Q + rhoCount)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Fork

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle.Test
  (Answer Point point point_injective coins read_challengeAddress)

attribute [local instance low] Classical.propDecidable

section Reads

open Spec.Folding.Nifs.NonInteractive.PiRlcSampler (sample)

/-- The `ρ` challenge of one oracle answer: the production sampler's scalar of the draw. -/
def readRho (answer : Answer) : RingF :=
  RingStrongSet.embedScalar (sample answer)

theorem readRho_valid (answer : Answer) :
    RingRelation.PiRLCAlgebra.Challenge.challengeValid (readRho answer) :=
  ⟨_, rfl⟩

theorem readRho_eq_iff (left right : Answer) :
    readRho left = readRho right ↔ sample left = sample right :=
  RingStrongSet.embedScalar_injective.eq_iff

/-- The sponge's `ρ` read is `readRho` of the state's sampler block. -/
theorem sponge_readRho (state : Transcript.State) :
    Sponge.readRho state =
      readRho (Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.block state) :=
  rfl

/-- The chance that a uniform answer samples one given scalar. -/
noncomputable def sampleError : ℝ :=
  1 / Spec.Folding.Nifs.NonInteractive.PiRlcSampler.scalarCount +
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance

theorem sampleError_nonnegative : 0 ≤ sampleError := by
  unfold sampleError
  have := Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_nonnegative
  positivity

theorem fiber_mass_le (scalar : RingStrongSet.Scalar) :
    mass {answer : Answer | sample answer = scalar} ≤ sampleError := by
  open Spec.Folding.Nifs.NonInteractive.PiRlcSampler in
  have card : Fintype.card Answer = drawCount := by
    simp only [Fintype.card_fun, Fintype.card_fin]
    rfl
  let test : RingStrongSet.Scalar → ℝ := fun value => if value = scalar then 1 else 0
  have sampled : mass {answer : Answer | sample answer = scalar} = sampledExpect test := by
    unfold mass sampledExpect
    rw [Finset.expect_eq_sum_div_card, Finset.card_univ, card]
    exact congrArg (· / (drawCount : ℝ))
      (Finset.sum_congr rfl fun answer _ => by
        by_cases same : sample answer = scalar <;> simp [test, same])
  have uniform : uniformExpect test = 1 / scalarCount := by
    unfold uniformExpect
    simp [test]
  have bound := (abs_le.mp (expect_difference_abs_le test (fun _ => by
    simp only [test]
    split <;> norm_num) (fun _ => by
    simp only [test]
    split <;> norm_num))).2
  rw [sampled]
  unfold sampleError
  linarith

end Reads

section Points

variable {schedule : Schedule} {width : Nat}

/-- The `ρ` challenges that the transcript at `statement` and `messages` reads from
`oracle`. -/
noncomputable def rho (oracle : Point schedule width → Answer) (statement : schedule.Statement)
    (messages : schedule.Messages width) : Fin schedule.rhoCount → RingF :=
  fun index => readRho (oracle (point statement messages (.rho index)))

theorem rho_valid (oracle : Point schedule width → Answer) (statement : schedule.Statement)
    (messages : schedule.Messages width) (index : Fin schedule.rhoCount) :
    RingRelation.PiRLCAlgebra.Challenge.challengeValid (rho oracle statement messages index) :=
  readRho_valid _

/-- Equal `ρ` points have equal statements and messages (`Schedule.proverCalls_identify`). -/
theorem rhoPoint_identifies {statement statement' : schedule.Statement}
    {messages messages' : schedule.Messages width} (index : Fin schedule.rhoCount)
    (same : point statement messages (.rho index) = point statement' messages' (.rho index)) :
    statement = statement' ∧ messages = messages' :=
  schedule.proverCalls_identify (.rho index)
    (List.append_cancel_right (congrArg (fun target => target.1.val) same))

theorem update_ne (oracle : Point schedule width → Answer) (statement : schedule.Statement)
    (messages : schedule.Messages width) {changed other : schedule.Challenge}
    (different : other ≠ changed) (answer : Answer) :
    Function.update oracle (point statement messages changed) answer
        (point statement messages other) =
      oracle (point statement messages other) :=
  Function.update_of_ne (fun same => different (point_injective statement messages same)) _ _

/-- A new answer at a `ρ` point leaves the Π_CCS coins unchanged. -/
theorem coins_update_rho (oracle : Point schedule width → Answer)
    (statement : schedule.Statement) (messages : schedule.Messages width)
    (index : Fin schedule.rhoCount) (answer : Answer) :
    coins (Function.update oracle (point statement messages (.rho index)) answer) statement
        messages =
      coins oracle statement messages := by
  have alpha (coordinate : Fin schedule.shape.cubeVariables) :
      Function.update oracle (point statement messages (.rho index)) answer
          (point statement messages (.alpha coordinate)) =
        oracle (point statement messages (.alpha coordinate)) :=
    update_ne oracle statement messages (by simp) answer
  have gamma : Function.update oracle (point statement messages (.rho index)) answer
      (point statement messages .gamma) = oracle (point statement messages .gamma) :=
    update_ne oracle statement messages (by simp) answer
  have round (round : Fin schedule.shape.cubeVariables) :
      Function.update oracle (point statement messages (.rho index)) answer
          (point statement messages (.round round)) =
        oracle (point statement messages (.round round)) :=
    update_ne oracle statement messages (by simp) answer
  simp only [coins, Schedule.coinsFrom, read_challengeAddress, alpha, gamma, round]

/-- A new answer at the `ρ` point of `index` changes only the challenge of `index`. -/
theorem rho_update (oracle : Point schedule width → Answer) (statement : schedule.Statement)
    (messages : schedule.Messages width) (index other : Fin schedule.rhoCount)
    (answer : Answer) :
    rho (Function.update oracle (point statement messages (.rho index)) answer) statement
        messages other =
      if other = index then readRho answer else rho oracle statement messages other := by
  by_cases same : other = index
  · subst same
    simp [rho]
  · rw [ite_eq_right same]
    exact congrArg readRho (update_ne oracle statement messages
      (fun equal => same (Schedule.Challenge.rho.inj equal)) answer)

end Points

section Game

/-- The fold-specific part of the fork step for adversary outputs of type `Output`. -/
structure Game (schedule : Schedule) (width : Nat) (Output Start : Type) where
  /-- What the transcript absorbs before the Π_CCS coins. -/
  statement : Output → schedule.Statement
  /-- The prover messages that the transcript absorbs after the statement. -/
  messages : Output → schedule.Messages width
  /-- The claim that the fold starts from; the transcript does not absorb it. -/
  start : Output → Start
  /-- The fold's success event on an oracle run. -/
  Succeeds : (Point schedule width → Answer) → Output → Prop

variable {schedule : Schedule} {width : Nat} {Output Start : Type}
  (game : Game schedule width Output Start)
  (adversary : OracleComp (Point schedule width) Answer Output)

local notation "Oracle" => Point schedule width → Answer
local notation "Retries" => Fin schedule.rhoCount → Answer

def Succeeds (oracle : Oracle) : Prop :=
  game.Succeeds oracle (adversary.run oracle)

noncomputable def rhoPoint (index : Fin schedule.rhoCount) (oracle : Oracle) :
    Point schedule width :=
  point (game.statement (adversary.run oracle)) (game.messages (adversary.run oracle))
    (.rho index)

def Hits (index : Fin schedule.rhoCount) (target : Point schedule width) (oracle : Oracle) :
    Prop :=
  Succeeds game adversary oracle ∧ rhoPoint game adversary index oracle = target

/-- The adversary, then a query at each `ρ` point of its output. -/
noncomputable def normalized : OracleComp (Point schedule width) Answer Output :=
  adversary.thenQueries fun output =>
    List.ofFn fun index => point (game.statement output) (game.messages output) (.rho index)

theorem normalized_bound {queries : Nat} (bounded : adversary.QueryBound queries) :
    (normalized game adversary).QueryBound (queries + schedule.rhoCount) :=
  bounded.thenQueries fun _ => by simp

theorem hits_queried (index : Fin schedule.rhoCount) (target : Point schedule width)
    (oracle : Oracle) (hits : Hits game adversary index target oracle) :
    target ∈ (normalized game adversary).queries oracle := by
  apply OracleComp.mem_queries_thenQueries
  rw [← hits.2]
  exact List.mem_ofFn.mpr ⟨index, rfl⟩

/-- The oracle of the retry at `index`: the base oracle with a new answer at that `ρ`
point. -/
noncomputable def forked (oracle : Oracle) (retries : Retries) (index : Fin schedule.rhoCount) :
    Oracle :=
  Function.update oracle (rhoPoint game adversary index oracle) (retries index)

/-- The answers at the `ρ` point of `index` that keep the run successful at that point. -/
def retrySets (oracle : Oracle) (index : Fin schedule.rhoCount) : Set Answer :=
  lineSet (Hits game adversary index) (rhoPoint game adversary index oracle) oracle

/-- The retry at `index` succeeds at the base run's `ρ` point, samples a new scalar, and keeps
the start claim. -/
def ForkValid (oracle : Oracle) (retries : Retries) (index : Fin schedule.rhoCount) : Prop :=
  (Succeeds game adversary (forked game adversary oracle retries index) ∧
      rhoPoint game adversary index (forked game adversary oracle retries index) =
        rhoPoint game adversary index oracle) ∧
    Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample (retries index) ≠
      Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
        (oracle (rhoPoint game adversary index oracle)) ∧
    game.start (adversary.run (forked game adversary oracle retries index)) =
      game.start (adversary.run oracle)

/-- The chance that a retry at `index`, drawn from `retrySets`, satisfies `event`. -/
noncomputable def retryChance (event : Answer → Prop) (index : Fin schedule.rhoCount)
    (oracle : Oracle) : ℝ :=
  mass (retrySets game adversary oracle index ∩ {answer | event answer}) /
    mass (retrySets game adversary oracle index)

/-- The chance that a retry at `index` moves the start claim. -/
noncomputable def mismatchChance (index : Fin schedule.rhoCount) (oracle : Oracle) : ℝ :=
  retryChance game adversary (fun answer =>
    game.start (adversary.run
        (Function.update oracle (rhoPoint game adversary index oracle) answer)) ≠
      game.start (adversary.run oracle)) index oracle

theorem retryChance_mono {event other : Answer → Prop} (index : Fin schedule.rhoCount)
    (oracle : Oracle)
    (implies : ∀ answer, answer ∈ retrySets game adversary oracle index → event answer →
      other answer) :
    retryChance game adversary event index oracle ≤
      retryChance game adversary other index oracle := by
  unfold retryChance
  apply div_le_div_of_nonneg_right _ (mass_nonnegative _)
  apply mass_mono
  rintro answer ⟨inside, holds⟩
  exact ⟨inside, implies answer inside holds⟩

def badRetries (oracle : Oracle) (index : Fin schedule.rhoCount) : Set Answer :=
  {answer | ¬ ForkValid game adversary oracle (fun _ => answer) index}

theorem forkValid_iff (oracle : Oracle) (retries : Retries) (index : Fin schedule.rhoCount) :
    ForkValid game adversary oracle retries index ↔
      retries index ∉ badRetries game adversary oracle index := by
  simp only [badRetries, Set.mem_ofPred_eq, not_not]
  rfl

private theorem mass_union_le (left right : Set Answer) :
    mass (left ∪ right) ≤ mass left + mass right := by
  unfold mass
  rw [← Finset.expect_add_distrib]
  exact Finset.expect_le_expect fun answer _ => by
    by_cases inLeft : answer ∈ left <;> by_cases inRight : answer ∈ right <;>
      simp [inLeft, inRight]

theorem mass_ne_zero {set : Set Answer} {answer : Answer} (inside : answer ∈ set) :
    mass set ≠ 0 := by
  unfold mass
  apply ne_of_gt
  rw [Finset.expect_eq_sum_div_card]
  apply div_pos _ (by exact_mod_cast Finset.card_pos.mpr Finset.univ_nonempty)
  calc
    (0 : ℝ) < if answer ∈ set then 1 else 0 := by simp [inside]
    _ ≤ _ := Finset.single_le_sum (f := fun answer => if answer ∈ set then (1 : ℝ) else 0)
      (fun _ _ => by split <;> norm_num) (Finset.mem_univ answer)

theorem base_mem_retrySets {oracle : Oracle} (succeeds : Succeeds game adversary oracle)
    (index : Fin schedule.rhoCount) :
    oracle (rhoPoint game adversary index oracle) ∈ retrySets game adversary oracle index := by
  change Hits game adversary index _ (Function.update oracle _ _)
  rw [Function.update_eq_self]
  exact ⟨succeeds, rfl⟩

private theorem sum_hits (index : Fin schedule.rhoCount)
    (term : Point schedule width → Oracle → ℝ) (oracle : Oracle) :
    (∑ target, if Hits game adversary index target oracle then term target oracle else 0) =
      if Succeeds game adversary oracle then
        term (rhoPoint game adversary index oracle) oracle else 0 := by
  by_cases succeeds : Succeeds game adversary oracle
  · simp only [Hits, succeeds, true_and, ite_true]
    exact (Finset.sum_ite_eq Finset.univ _ _).trans (ite_eq_left (Finset.mem_univ _))
  · simp [Hits, succeeds]

/-- `RandomOracleExtraction.fork_failure_le` on any schedule: a successful run whose retry
family is invalid has probability at most `rhoCount · (Q + rhoCount) · sampleError` plus the
expected chance that a retry moves the start claim. -/
theorem fork_failure_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Succeeds game adversary oracle then
        ∑ retries, retryWeight (retrySets game adversary oracle) retries *
          (if ∀ index, ForkValid game adversary oracle retries index then 0 else 1)
      else 0) ≤
      schedule.rhoCount * (((queries + schedule.rhoCount : Nat) : ℝ) * sampleError) +
        𝔼 oracle, (if Succeeds game adversary oracle then
          ∑ index, mismatchChance game adversary index oracle else 0) := by
  have pointwise (oracle : Oracle) :
      (if Succeeds game adversary oracle then
        ∑ retries, retryWeight (retrySets game adversary oracle) retries *
          (if ∀ index, ForkValid game adversary oracle retries index then 0 else 1)
        else 0) ≤
      (∑ index, if Succeeds game adversary oracle then
          repeatChance (Hits game adversary index)
            Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
            (rhoPoint game adversary index oracle) oracle else 0) +
        (if Succeeds game adversary oracle then
          ∑ index, mismatchChance game adversary index oracle else 0) := by
    by_cases succeeds : Succeeds game adversary oracle
    · simp only [ite_eq_left succeeds]
      have failure (retries : Retries) :
          (if ∀ index, ForkValid game adversary oracle retries index then (0 : ℝ) else 1) ≤
            ∑ index, if retries index ∈ badRetries game adversary oracle index
              then 1 else 0 := by
        by_cases every : ∀ index, ForkValid game adversary oracle retries index
        · rw [ite_eq_left every]
          exact Finset.sum_nonneg fun _ _ => by split <;> norm_num
        · rw [ite_eq_right every]
          obtain ⟨index, invalid⟩ := not_forall.mp every
          have inside : retries index ∈ badRetries game adversary oracle index :=
            by_contra fun outside =>
              invalid ((forkValid_iff game adversary oracle retries index).mpr outside)
          have single := Finset.single_le_sum
            (f := fun index => if retries index ∈ badRetries game adversary oracle index
              then (1 : ℝ) else 0) (fun _ _ => by split <;> norm_num) (Finset.mem_univ index)
          simpa [inside] using single
      refine (Finset.sum_le_sum fun retries _ => mul_le_mul_of_nonneg_left (failure retries)
        (retryWeight_nonnegative _ retries)).trans ?_
      rw [expected_bad_retries _ (fun index => mass_ne_zero (base_mem_retrySets game adversary
        succeeds index)) (badRetries game adversary oracle)]
      rw [← Finset.sum_add_distrib]
      refine Finset.sum_le_sum fun index _ => ?_
      rw [repeatChance, mismatchChance, retryChance]
      unfold retrySets
      rw [← add_div]
      apply div_le_div_of_nonneg_right _ (mass_nonnegative _)
      refine (mass_mono ?_).trans (mass_union_le _ _)
      rintro answer ⟨inside, invalid⟩
      simp only [badRetries, Set.mem_ofPred_eq] at invalid
      change ¬ (answer ∈ lineSet (Hits game adversary index)
        (rhoPoint game adversary index oracle) oracle ∧ _ ∧ _) at invalid
      simp only [not_and_or, not_not] at invalid
      rcases invalid with outside | repeated | mismatch
      · exact absurd inside outside
      · exact Or.inl ⟨inside, repeated⟩
      · exact Or.inr ⟨inside, mismatch⟩
    · simp [succeeds]
  calc
    _ ≤ 𝔼 oracle, ((∑ index, if Succeeds game adversary oracle then
            repeatChance (Hits game adversary index)
              Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample
              (rhoPoint game adversary index oracle) oracle else 0) +
          (if Succeeds game adversary oracle then
            ∑ index, mismatchChance game adversary index oracle else 0)) :=
      Finset.expect_le_expect fun oracle _ => pointwise oracle
    _ = (∑ index, 𝔼 oracle, ∑ target,
            if Hits game adversary index target oracle then
              repeatChance (Hits game adversary index)
                Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample target oracle else 0) +
          𝔼 oracle, (if Succeeds game adversary oracle then
            ∑ index, mismatchChance game adversary index oracle else 0) := by
      rw [Finset.expect_add_distrib, Finset.expect_sum_comm]
      simp only [sum_hits]
    _ ≤ ∑ _index : Fin schedule.rhoCount,
            ((queries + schedule.rhoCount : Nat) : ℝ) * sampleError +
          𝔼 oracle, (if Succeeds game adversary oracle then
            ∑ index, mismatchChance game adversary index oracle else 0) := by
      refine add_le_add (Finset.sum_le_sum fun index _ => ?_) le_rfl
      exact repeat_le (Hits game adversary index)
        Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample sampleError fiber_mass_le
        sampleError_nonnegative (normalized_bound game adversary bounded)
        (hits_queried game adversary index)
    _ = _ := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- `fork_failure_le` with each mismatch chance replaced by a larger chance `bound`, such as a
hash collision between the two runs' outputs. -/
theorem fork_failure_le_of_mismatch {queries : Nat} (bounded : adversary.QueryBound queries)
    (bound : Fin schedule.rhoCount → Oracle → ℝ)
    (dominates : ∀ oracle, Succeeds game adversary oracle → ∀ index,
      mismatchChance game adversary index oracle ≤ bound index oracle) :
    𝔼 oracle, (if Succeeds game adversary oracle then
        ∑ retries, retryWeight (retrySets game adversary oracle) retries *
          (if ∀ index, ForkValid game adversary oracle retries index then 0 else 1)
      else 0) ≤
      schedule.rhoCount * (((queries + schedule.rhoCount : Nat) : ℝ) * sampleError) +
        𝔼 oracle, (if Succeeds game adversary oracle then ∑ index, bound index oracle
          else 0) := by
  refine (fork_failure_le game adversary bounded).trans
    (add_le_add le_rfl (Finset.expect_le_expect fun oracle _ => ?_))
  by_cases succeeds : Succeeds game adversary oracle
  · simp only [ite_eq_left succeeds]
    exact Finset.sum_le_sum fun index _ => dominates oracle succeeds index
  · simp only [ite_eq_right succeeds, le_refl]

theorem expected_retries_at_le {queries : Nat} (bounded : adversary.QueryBound queries)
    (index : Fin schedule.rhoCount) :
    𝔼 oracle, (if Succeeds game adversary oracle then
        retryCount (Hits game adversary index) (rhoPoint game adversary index oracle) oracle
      else 0) ≤
      ((queries + schedule.rhoCount : Nat) : ℝ) :=
  calc
    _ = 𝔼 oracle, ∑ target,
          if Hits game adversary index target oracle then
            retryCount (Hits game adversary index) target oracle else 0 :=
      Finset.expect_congr rfl fun oracle _ => (sum_hits game adversary index _ oracle).symm
    _ ≤ _ := retries_le (Hits game adversary index) (normalized_bound game adversary bounded)
        (hits_queried game adversary index)

/-- The expected number of reruns: at most `rhoCount · (Q + rhoCount)`. -/
theorem expected_retries_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Succeeds game adversary oracle then
        ∑ index, retryCount (Hits game adversary index)
          (rhoPoint game adversary index oracle) oracle else 0) ≤
      schedule.rhoCount * ((queries + schedule.rhoCount : Nat) : ℝ) := by
  calc
    _ = ∑ index, 𝔼 oracle, (if Succeeds game adversary oracle then
          retryCount (Hits game adversary index)
            (rhoPoint game adversary index oracle) oracle else 0) := by
      rw [← Finset.expect_sum_comm]
      refine Finset.expect_congr rfl fun oracle _ => ?_
      split <;> simp
    _ ≤ ∑ _index : Fin schedule.rhoCount, ((queries + schedule.rhoCount : Nat) : ℝ) :=
      Finset.sum_le_sum fun index _ => expected_retries_at_le game adversary bounded index
    _ = _ := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Game

section Valid

variable {schedule : Schedule} {width : Nat} {Output Start : Type}
  {game : Game schedule width Output Start}
  {adversary : OracleComp (Point schedule width) Answer Output}
  {oracle : Point schedule width → Answer} {retries : Fin schedule.rhoCount → Answer}
  {index : Fin schedule.rhoCount}

local notation "Retry" => adversary.run (forked game adversary oracle retries index)
local notation "Base" => adversary.run oracle

theorem ForkValid.mem_retrySets (valid : ForkValid game adversary oracle retries index) :
    retries index ∈ retrySets game adversary oracle index :=
  valid.1

theorem ForkValid.succeeds (valid : ForkValid game adversary oracle retries index) :
    game.Succeeds (forked game adversary oracle retries index) Retry :=
  valid.1.1

theorem ForkValid.start_eq (valid : ForkValid game adversary oracle retries index) :
    game.start Retry = game.start Base :=
  valid.2.2

/-- A valid retry absorbs the base run's statement and messages. -/
theorem ForkValid.identifies (valid : ForkValid game adversary oracle retries index) :
    game.statement Retry = game.statement Base ∧ game.messages Retry = game.messages Base :=
  rhoPoint_identifies index valid.1.2

/-- A valid retry reads the base run's Π_CCS coins. -/
theorem ForkValid.coins_eq (valid : ForkValid game adversary oracle retries index) :
    coins (forked game adversary oracle retries index) (game.statement Retry)
        (game.messages Retry) =
      coins oracle (game.statement Base) (game.messages Base) := by
  obtain ⟨statementEq, messagesEq⟩ := valid.identifies
  rw [statementEq, messagesEq]
  exact coins_update_rho oracle _ _ index (retries index)

/-- A valid retry's `ρ` challenges are the base run's, except the retry's answer at
`index`. -/
theorem ForkValid.rho_eq (valid : ForkValid game adversary oracle retries index)
    (other : Fin schedule.rhoCount) :
    rho (forked game adversary oracle retries index) (game.statement Retry)
        (game.messages Retry) other =
      if other = index then readRho (retries index)
      else rho oracle (game.statement Base) (game.messages Base) other := by
  obtain ⟨statementEq, messagesEq⟩ := valid.identifies
  rw [statementEq, messagesEq]
  exact rho_update oracle _ _ index other (retries index)

theorem ForkValid.agree (valid : ForkValid game adversary oracle retries index)
    {other : Fin schedule.rhoCount} (different : other ≠ index) :
    rho oracle (game.statement Base) (game.messages Base) other =
      rho (forked game adversary oracle retries index) (game.statement Retry)
        (game.messages Retry) other := by
  rw [valid.rho_eq other, ite_eq_right different]

theorem ForkValid.changed (valid : ForkValid game adversary oracle retries index) :
    rho oracle (game.statement Base) (game.messages Base) index ≠
      rho (forked game adversary oracle retries index) (game.statement Retry)
        (game.messages Retry) index := by
  rw [valid.rho_eq index, ite_eq_left rfl]
  exact fun same => valid.2.1 ((readRho_eq_iff _ _).mp same).symm

end Valid

end NightstreamFPrime.Lifecycle.RandomOracle.Fork
