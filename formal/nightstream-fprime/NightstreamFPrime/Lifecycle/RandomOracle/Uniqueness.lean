import NightstreamFPrime.Lifecycle.RandomOracle.Fork

/-! The uniqueness step of the random-oracle chain for any fold transcript `Schedule`: the step
that `RandomOracleUniqueness` states for the one-circuit fold, on a fork `Fork.Game`.

A pair of a base oracle and a retry family is valid (`Valid`) when the base run succeeds and
every retry is valid (`Fork.ForkValid`). The fold supplies the witness that a valid pair
extracts (`witnessOf`), an event `Fails` of valid pairs, and the claimed Π_CCS protocol data of
each pair (`source`). The fork index (`forkIndex`) is the first query of the checked game
(`Test.checked`) that extends the statement calls of the run's statement. Resampling the oracle
after that index (`context`, `overlay`) keeps the statement (`statement_eq_of_fork`). For a
failing pair, a valid resampled pair at the same index has one of three outcomes
(`one_le_outcomes`): its start claim moves (`Moves`), its start claim stays and its extracted
witness differs (`Collides`), or both stay. In the third case the fold shows that the base
pair's data are false on the resampled run (`FalseFor`): a hit of the Π_CCS bad sets with the
witness fixed. `worst_error_le` bounds that event with the worst pair fixed by the context
(`worst`, `worstBad`), through `Test.checked_error_le`.

`source_error_le`: the weighted chance of a failing pair is at most
`(Q + challenge count) · testError` plus the chances of the two other outcomes
(`collisionChance`, `startChance`). `knowledge_error_le` adds the fork step
(`Fork.fork_failure_le_of_mismatch`): a successful run extracts except with the statistical
error `statisticalError` and the chances of a moved start claim and a collision. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Uniqueness

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point point)
open NightstreamFPrime.Lifecycle.RandomOracle.Fork (Game)

attribute [local instance low] Classical.propDecidable

section Lists

private noncomputable def best {Item : Type} [Fintype Item] [Nonempty Item] (value : Item → ℝ) :
    Item :=
  Classical.choose (Finset.exists_max_image Finset.univ value Finset.univ_nonempty)

private theorem le_best {Item : Type} [Fintype Item] [Nonempty Item] (value : Item → ℝ)
    (item : Item) : value item ≤ value (best value) :=
  (Classical.choose_spec (Finset.exists_max_image Finset.univ value Finset.univ_nonempty)).2 item
    (Finset.mem_univ item)

private theorem findIdx_eq_of_take {Item : Type} {test : Item → Bool} {left right : List Item}
    {index : Nat} (same : left.take (index + 1) = right.take (index + 1))
    (found : left.findIdx test ≤ index) : right.findIdx test = left.findIdx test := by
  have counted := congrArg (List.findIdx test) same
  rw [List.findIdx_take, List.findIdx_take] at counted
  omega

private theorem not_mem_take_findIdx {Item : Type} {test : Item → Bool} {items : List Item}
    {item : Item} (passes : test item = true) :
    item ∉ items.take (items.findIdx test) := by
  intro inside
  obtain ⟨position, bound, located⟩ := List.getElem_of_mem inside
  rw [List.length_take] at bound
  have before := List.not_of_lt_findIdx (p := test) (xs := items) (i := position) (by omega)
  rw [List.getElem_take] at located
  have fails : test item = false := by
    rw [← located]
    exact before
  rw [passes] at fails
  cases fails

end Lists

section Extends

variable {schedule : Schedule} {width : Nat}

/-- The calls of `target` start with the statement calls of `statement`. -/
def Extends (statement : schedule.Statement) (target : Point schedule width) : Bool :=
  (schedule.statementCalls statement).isPrefixOf target.1.val

theorem extends_point (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge) :
    Extends statement (point statement messages challenge) = true := by
  rw [Extends, List.isPrefixOf_iff_prefix]
  change schedule.statementCalls statement <+: schedule.challengeCalls statement messages challenge
  cases challenge <;>
    simp only [Schedule.challengeCalls, Schedule.proverCalls, Schedule.roundPrefixCalls,
      Schedule.outputCalls, Schedule.roundsCalls, Schedule.preRoundCalls, List.append_assoc] <;>
    exact List.prefix_append _ _

/-- Two statements whose calls both start `target` are equal. -/
theorem statement_eq_of_extends {statement statement' : schedule.Statement}
    {target : Point schedule width}
    (left : Extends statement target = true) (right : Extends statement' target = true) :
    statement = statement' := by
  simp only [Extends, List.isPrefixOf_iff_prefix] at left right
  have lengths := schedule.statementCalls_length statement statement'
  exact schedule.statementCalls_identify
    ((List.prefix_of_prefix_length_le left right lengths.le).eq_of_length lengths)

private theorem hits_mono
    {family family' : schedule.Challenge → Point schedule width →
      (Point schedule width → Answer) → Set Answer}
    {oracle : Point schedule width → Answer} {statement : schedule.Statement}
    {messages : schedule.Messages width}
    (inside : ∀ challenge, family challenge (point statement messages challenge) oracle ⊆
      family' challenge (point statement messages challenge) oracle)
    (hits : Test.Hits family oracle statement messages) :
    Test.Hits family' oracle statement messages := by
  rcases hits with ⟨index, member⟩ | member | ⟨index, member⟩
  · exact Or.inl ⟨index, inside _ member⟩
  · exact Or.inr (Or.inl (inside _ member))
  · exact Or.inr (Or.inr ⟨index, inside _ member⟩)

end Extends

section Game

variable {schedule : Schedule} {width : Nat} {Output Start Witness : Type}
  (game : Game schedule width Output Start)
  (adversary : OracleComp (Point schedule width) Answer Output)

local notation "Oracle" => Point schedule width → Answer
local notation "Retries" => Fin schedule.rhoCount → Answer
local notation "Checked" => Test.checked adversary game.statement game.messages

/-- The first query of the checked game that extends the statement calls of `statement`. -/
noncomputable def firstIndex (statement : schedule.Statement) (oracle : Oracle) : Nat :=
  ((Checked).queries oracle).findIdx (Extends statement)

/-- The fork index: the first query that extends the statement of the run's output. -/
noncomputable def forkIndex (oracle : Oracle) : Nat :=
  firstIndex game adversary (game.statement (adversary.run oracle)) oracle

/-- The points that the checked game queries before `index`. -/
noncomputable def context (index : Nat) (oracle : Oracle) : Finset (Point schedule width) :=
  queriedBefore (Checked) index oracle

theorem context_stopping (index : Nat) : Stopping (context game adversary index) :=
  queriedBefore_stopping _ index

theorem firstIndex_update (statement : schedule.Statement) (oracle : Oracle)
    {target : Point schedule width} (extends_ : Extends statement target = true)
    (answer : Answer) :
    firstIndex game adversary statement (Function.update oracle target answer) =
      firstIndex game adversary statement oracle := by
  unfold firstIndex
  apply findIdx_eq_of_take (index := ((Checked).queries oracle).findIdx (Extends statement)) _
    le_rfl
  symm
  apply OracleComp.take_succ_queries_congr
  intro asked inside
  have outside : asked ≠ target := fun same => not_mem_take_findIdx extends_ (same ▸ inside)
  exact (Function.update_of_ne outside _ _).symm

theorem not_mem_context (statement : schedule.Statement) (oracle : Oracle)
    {target : Point schedule width} (extends_ : Extends statement target = true) :
    target ∉ context game adversary (firstIndex game adversary statement oracle) oracle := by
  intro inside
  exact not_mem_take_findIdx extends_ (List.mem_toFinset.mp inside)

theorem overlay_context_congr {index : Nat} {oracle other : Oracle}
    (agree : ∀ asked ∈ context game adversary index oracle, oracle asked = other asked)
    (sample : Oracle) :
    overlay (context game adversary index other) other sample =
      overlay (context game adversary index oracle) oracle sample := by
  rw [context_stopping game adversary index oracle other agree]
  funext asked
  by_cases inside : asked ∈ context game adversary index oracle
  · simp [overlay, inside, agree asked inside]
  · simp [overlay, inside]

/-- The base run succeeds and every retry is valid. -/
def Valid (oracle : Oracle) (retries : Retries) : Prop :=
  Fork.Succeeds game adversary oracle ∧
    ∀ index, Fork.ForkValid game adversary oracle retries index

noncomputable def weight (oracle : Oracle) (retries : Retries) : ℝ :=
  retryWeight (Fork.retrySets game adversary oracle) retries

noncomputable def retryMass (oracle : Oracle) (event : Retries → Prop) : ℝ :=
  ∑ retries, weight game adversary oracle retries * (if event retries then 1 else 0)

noncomputable def success (index : Nat) (oracle : Oracle) : ℝ :=
  if forkIndex game adversary oracle = index then
    retryMass game adversary oracle (Valid game adversary oracle)
  else 0

noncomputable def resampledSuccess (index : Nat) (oracle : Oracle) : ℝ :=
  𝔼 sample, success game adversary index
    (overlay (context game adversary index oracle) oracle sample)

/-- The chance that a valid resampled pair at fork index `index` has `outcome`, among the valid
resampled pairs at that index. -/
noncomputable def retryChance (index : Nat) (oracle : Oracle)
    (outcome : Oracle → Retries → Prop) : ℝ :=
  (𝔼 sample, retryMass game adversary
      (overlay (context game adversary index oracle) oracle sample)
      (fun retries => forkIndex game adversary
          (overlay (context game adversary index oracle) oracle sample) = index ∧
        Valid game adversary (overlay (context game adversary index oracle) oracle sample)
          retries ∧
        outcome (overlay (context game adversary index oracle) oracle sample) retries)) /
    resampledSuccess game adversary index oracle

variable
  (witnessOf : (Point schedule width → Answer) → (Fin schedule.rhoCount → Answer) → Witness)

/-- The resampled pair keeps the start claim and extracts another witness. -/
def Collides (oracle : Oracle) (retries : Retries) (other : Oracle) (otherRetries : Retries) :
    Prop :=
  game.start (adversary.run other) = game.start (adversary.run oracle) ∧
    witnessOf other otherRetries ≠ witnessOf oracle retries

/-- The resampled run moves the start claim. -/
def Moves (oracle other : Oracle) (_ : Retries) : Prop :=
  game.start (adversary.run other) ≠ game.start (adversary.run oracle)

noncomputable def reductionChance (outcome : Oracle → Retries → Oracle → Retries → Prop) :
    ℝ :=
  𝔼 oracle, ∑ retries, weight game adversary oracle retries *
    (if Valid game adversary oracle retries then
      retryChance game adversary (forkIndex game adversary oracle) oracle
        (outcome oracle retries)
    else 0)

noncomputable def collisionChance : ℝ :=
  reductionChance game adversary (Collides game adversary witnessOf)

noncomputable def startChance : ℝ :=
  reductionChance game adversary fun oracle _ => Moves game adversary oracle

theorem weight_nonnegative (oracle : Oracle) (retries : Retries) :
    0 ≤ weight game adversary oracle retries :=
  retryWeight_nonnegative _ _

theorem retryMass_nonnegative (oracle : Oracle) (event : Retries → Prop) :
    0 ≤ retryMass game adversary oracle event :=
  Finset.sum_nonneg fun retries _ => mul_nonneg (weight_nonnegative game adversary oracle retries)
    (by split <;> norm_num)

theorem retryMass_le_one (oracle : Oracle) (event : Retries → Prop) :
    retryMass game adversary oracle event ≤ 1 :=
  (Finset.sum_le_sum fun retries _ => mul_le_of_le_one_right
    (weight_nonnegative game adversary oracle retries) (by split <;> norm_num)).trans
    (retryWeight_sum_le_one _)

theorem retryMass_const_le (oracle : Oracle) (event : Prop) [Decidable event] :
    retryMass game adversary oracle (fun _ => event) ≤ if event then 1 else 0 := by
  by_cases happens : event
  · rw [ite_eq_left happens]
    exact retryMass_le_one game adversary oracle _
  · simp [retryMass, happens]

theorem retryMass_le_three (oracle : Oracle) {event first second third : Retries → Prop}
    (covered : ∀ retries, event retries → first retries ∨ second retries ∨ third retries) :
    retryMass game adversary oracle event ≤
      retryMass game adversary oracle first + retryMass game adversary oracle second +
        retryMass game adversary oracle third := by
  unfold retryMass
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun retries _ => ?_
  rw [← mul_add, ← mul_add]
  refine mul_le_mul_of_nonneg_left ?_ (weight_nonnegative game adversary oracle retries)
  have one : (0 : ℝ) ≤ if first retries then 1 else 0 := by split <;> norm_num
  have two : (0 : ℝ) ≤ if second retries then 1 else 0 := by split <;> norm_num
  have three : (0 : ℝ) ≤ if third retries then 1 else 0 := by split <;> norm_num
  by_cases happens : event retries
  · rw [ite_eq_left happens]
    rcases covered retries happens with holds | holds | holds
    · have : (1 : ℝ) ≤ if first retries then 1 else 0 := by rw [ite_eq_left holds]
      linarith
    · have : (1 : ℝ) ≤ if second retries then 1 else 0 := by rw [ite_eq_left holds]
      linarith
    · have : (1 : ℝ) ≤ if third retries then 1 else 0 := by rw [ite_eq_left holds]
      linarith
  · rw [ite_eq_right happens]
    linarith

theorem weight_le_retryMass (oracle : Oracle) {event : Retries → Prop} {retries : Retries}
    (happens : event retries) :
    weight game adversary oracle retries ≤ retryMass game adversary oracle event := by
  unfold retryMass
  calc
    weight game adversary oracle retries =
        weight game adversary oracle retries * (if event retries then 1 else 0) := by
      rw [ite_eq_left happens, mul_one]
    _ ≤ _ := Finset.single_le_sum
      (f := fun retries => weight game adversary oracle retries *
        (if event retries then (1 : ℝ) else 0))
      (fun retries _ => mul_nonneg (weight_nonnegative game adversary oracle retries)
        (by split <;> norm_num)) (Finset.mem_univ retries)

theorem weight_pos {oracle : Oracle} {retries : Retries}
    (valid : Valid game adversary oracle retries) :
    0 < weight game adversary oracle retries := by
  unfold weight retryWeight
  apply Finset.prod_pos
  intro index _
  have inside := (valid.2 index).mem_retrySets
  rw [ite_eq_left inside]
  have card : (0 : ℝ) < Fintype.card Answer := by exact_mod_cast Fintype.card_pos
  have positive := lt_of_le_of_ne (mass_nonnegative _) (Fork.mass_ne_zero inside).symm
  exact div_pos one_pos (mul_pos card positive)

theorem success_eq (index : Nat) (oracle : Oracle) :
    success game adversary index oracle =
      retryMass game adversary oracle fun retries =>
        forkIndex game adversary oracle = index ∧ Valid game adversary oracle retries := by
  unfold success
  by_cases forked : forkIndex game adversary oracle = index
  · rw [ite_eq_left forked]
    simp only [forked, true_and]
  · simp [forked, retryMass]

theorem success_nonnegative (index : Nat) (oracle : Oracle) :
    0 ≤ success game adversary index oracle := by
  rw [success_eq]
  exact retryMass_nonnegative game adversary oracle _

theorem resampledSuccess_nonnegative (index : Nat) (oracle : Oracle) :
    0 ≤ resampledSuccess game adversary index oracle :=
  Finset.expect_nonneg fun _ _ => success_nonnegative game adversary index _

theorem retryChance_nonnegative (index : Nat) (oracle : Oracle)
    (outcome : Oracle → Retries → Prop) :
    0 ≤ retryChance game adversary index oracle outcome :=
  div_nonneg (Finset.expect_nonneg fun _ _ => retryMass_nonnegative game adversary _ _)
    (resampledSuccess_nonnegative game adversary index oracle)

theorem retryMass_mono (oracle : Oracle) {event event' : Retries → Prop}
    (implies : ∀ retries, event retries → event' retries) :
    retryMass game adversary oracle event ≤ retryMass game adversary oracle event' :=
  Finset.sum_le_sum fun retries _ => mul_le_mul_of_nonneg_left
    (by by_cases happens : event retries
        · rw [ite_eq_left happens, ite_eq_left (implies retries happens)]
        · rw [ite_eq_right happens]
          split <;> norm_num)
    (weight_nonnegative game adversary oracle retries)

theorem retryChance_mono (index : Nat) (oracle : Oracle)
    {outcome outcome' : Oracle → Retries → Prop}
    (implies : ∀ sample retries,
      forkIndex game adversary (overlay (context game adversary index oracle) oracle sample) =
        index →
      Valid game adversary (overlay (context game adversary index oracle) oracle sample) retries →
      outcome (overlay (context game adversary index oracle) oracle sample) retries →
      outcome' (overlay (context game adversary index oracle) oracle sample) retries) :
    retryChance game adversary index oracle outcome ≤
      retryChance game adversary index oracle outcome' :=
  div_le_div_of_nonneg_right
    (Finset.expect_le_expect fun sample _ => retryMass_mono game adversary _
      fun retries holds =>
        ⟨holds.1, holds.2.1, implies sample retries holds.1 holds.2.1 holds.2.2⟩)
    (resampledSuccess_nonnegative game adversary index oracle)

theorem retryChance_le_one (index : Nat) (oracle : Oracle) (outcome : Oracle → Retries → Prop) :
    retryChance game adversary index oracle outcome ≤ 1 :=
  div_le_one_of_le₀
    (Finset.expect_le_expect fun _ _ => by
      rw [success_eq]
      exact retryMass_mono game adversary _ fun _ holds => ⟨holds.1, holds.2.1⟩)
    (resampledSuccess_nonnegative game adversary index oracle)

theorem resampledSuccess_pos {oracle : Oracle} {retries : Retries}
    (valid : Valid game adversary oracle retries) :
    0 < resampledSuccess game adversary (forkIndex game adversary oracle) oracle := by
  have own : 0 < success game adversary (forkIndex game adversary oracle)
      (overlay (context game adversary (forkIndex game adversary oracle) oracle) oracle
        oracle) := by
    rw [overlay_self, success_eq]
    exact (weight_pos game adversary valid).trans_le
      (weight_le_retryMass game adversary oracle ⟨rfl, valid⟩)
  unfold resampledSuccess
  exact expect_pos (fun _ => success_nonnegative game adversary _ _) own

theorem reductionChance_nonnegative (outcome : Oracle → Retries → Oracle → Retries → Prop) :
    0 ≤ reductionChance game adversary outcome :=
  Finset.expect_nonneg fun oracle _ => Finset.sum_nonneg fun retries _ =>
    mul_nonneg (weight_nonnegative game adversary oracle retries) (by
      split_ifs
      · exact retryChance_nonnegative game adversary _ oracle _
      · exact le_rfl)

theorem reductionChance_le_one (outcome : Oracle → Retries → Oracle → Retries → Prop) :
    reductionChance game adversary outcome ≤ 1 := by
  unfold reductionChance
  calc
    _ ≤ 𝔼 _oracle : Oracle, (1 : ℝ) := Finset.expect_le_expect fun oracle _ =>
        (Finset.sum_le_sum fun retries _ => mul_le_of_le_one_right
          (weight_nonnegative game adversary oracle retries) (by
            split_ifs
            · exact retryChance_le_one game adversary _ oracle _
            · exact zero_le_one)).trans (retryWeight_sum_le_one _)
    _ = 1 := Finset.expect_const Finset.univ_nonempty _

/-- A larger outcome on valid resampled pairs at the base fork index has a larger chance. -/
theorem reductionChance_mono {outcome outcome' : Oracle → Retries → Oracle → Retries → Prop}
    (implies : ∀ oracle retries, Valid game adversary oracle retries → ∀ sample otherRetries,
      forkIndex game adversary (overlay (context game adversary
          (forkIndex game adversary oracle) oracle) oracle sample) =
        forkIndex game adversary oracle →
      Valid game adversary (overlay (context game adversary
        (forkIndex game adversary oracle) oracle) oracle sample) otherRetries →
      outcome oracle retries (overlay (context game adversary
        (forkIndex game adversary oracle) oracle) oracle sample) otherRetries →
      outcome' oracle retries (overlay (context game adversary
        (forkIndex game adversary oracle) oracle) oracle sample) otherRetries) :
    reductionChance game adversary outcome ≤ reductionChance game adversary outcome' := by
  unfold reductionChance
  refine Finset.expect_le_expect fun oracle _ => Finset.sum_le_sum fun retries _ =>
    mul_le_mul_of_nonneg_left ?_ (weight_nonnegative game adversary oracle retries)
  split_ifs with valid
  · exact retryChance_mono game adversary _ oracle
      fun sample otherRetries forked otherValid holds =>
        implies oracle retries valid sample otherRetries forked otherValid holds
  · exact le_rfl

theorem forkIndex_lt (oracle : Oracle) :
    forkIndex game adversary oracle < ((Checked).queries oracle).length :=
  List.findIdx_lt_length_of_exists
    ⟨_, Test.mem_checked adversary game.statement game.messages oracle .gamma,
      extends_point _ _ .gamma⟩

/-- A resampled run with the base fork index has the base run's statement. -/
theorem statement_eq_of_fork (oracle sample : Oracle)
    (forked : forkIndex game adversary
      (overlay (context game adversary (forkIndex game adversary oracle) oracle) oracle sample) =
        forkIndex game adversary oracle) :
    game.statement (adversary.run
        (overlay (context game adversary (forkIndex game adversary oracle) oracle) oracle
          sample)) =
      game.statement (adversary.run oracle) := by
  set index := forkIndex game adversary oracle with index_def
  set other := overlay (context game adversary index oracle) oracle sample with other_def
  have same : ((Checked).queries other).take (index + 1) =
      ((Checked).queries oracle).take (index + 1) :=
    OracleComp.take_succ_queries_congr oracle other _ index fun asked inside =>
      overlay_agree _ oracle sample asked (List.mem_toFinset.mpr inside)
  have baseBound : index < ((Checked).queries oracle).length :=
    forkIndex_lt game adversary oracle
  have nextBound : index < ((Checked).queries other).length := by
    have bound := forkIndex_lt game adversary other
    rwa [forked] at bound
  have samePoint : ((Checked).queries other)[index] = ((Checked).queries oracle)[index] := by
    rw [List.getElem_take' nextBound (Nat.lt_succ_self index),
      List.getElem_take' baseBound (Nat.lt_succ_self index)]
    exact List.getElem_of_eq same _
  have baseExtends := ((List.findIdx_eq (p := Extends (game.statement (adversary.run oracle)))
    baseBound).mp index_def.symm).1
  have nextExtends := ((List.findIdx_eq (p := Extends (game.statement (adversary.run other)))
    nextBound).mp forked).1
  rw [samePoint] at nextExtends
  exact statement_eq_of_extends nextExtends baseExtends

section Test

variable (source : (Point schedule width → Answer) × (Fin schedule.rhoCount → Answer) →
  schedule.Statement → ProtocolPolynomial.Data K schedule.shape)

/-- The run at `oracle` hits a Π_CCS bad set of `pair`'s claimed data. -/
def FalseFor (pair : Oracle × Retries) (oracle : Oracle) : Prop :=
  Test.Hits (Test.bad (source pair)) oracle (game.statement (adversary.run oracle))
    (game.messages (adversary.run oracle))

noncomputable def testChance (index : Nat) (oracle : Oracle) (pair : Oracle × Retries) : ℝ :=
  𝔼 sample, if forkIndex game adversary
        (overlay (context game adversary index oracle) oracle sample) = index ∧
      FalseFor game adversary source pair
        (overlay (context game adversary index oracle) oracle sample) then 1 else 0

/-- The pair whose resampled false acceptance at `index` is most likely. -/
noncomputable def worst (index : Nat) (oracle : Oracle) : Oracle × Retries :=
  best (testChance game adversary source index oracle)

theorem le_worst (index : Nat) (oracle : Oracle) (pair : Oracle × Retries) :
    testChance game adversary source index oracle pair ≤
      testChance game adversary source index oracle (worst game adversary source index oracle) :=
  le_best _ pair

theorem testChance_nonnegative (index : Nat) (oracle : Oracle) (pair : Oracle × Retries) :
    0 ≤ testChance game adversary source index oracle pair :=
  Finset.expect_nonneg fun _ _ => by split <;> norm_num

theorem testChance_congr {index : Nat} {oracle other : Oracle}
    (agree : ∀ asked ∈ context game adversary index oracle, oracle asked = other asked) :
    testChance game adversary source index other =
      testChance game adversary source index oracle := by
  funext pair
  unfold testChance
  simp only [overlay_context_congr game adversary agree]

theorem worst_congr {index : Nat} {oracle other : Oracle}
    (agree : ∀ asked ∈ context game adversary index oracle, oracle asked = other asked) :
    worst game adversary source index other = worst game adversary source index oracle := by
  unfold worst
  rw [testChance_congr game adversary source agree]

/-- The bad set of the worst pair at the first index of each statement whose calls start
`target`. -/
def worstBad (challenge : schedule.Challenge) (target : Point schedule width) (oracle : Oracle) :
    Set Answer :=
  {answer | ∃ statement, Extends statement target = true ∧
    answer ∈ Test.bad (source (worst game adversary source
      (firstIndex game adversary statement oracle) oracle)) challenge target oracle}

theorem worstBad_local (challenge : schedule.Challenge) :
    Local (worstBad game adversary source challenge) := by
  intro target oracle answer
  ext candidate
  simp only [worstBad, Set.mem_ofPred_eq]
  refine exists_congr fun statement => and_congr_right fun extends_ => ?_
  have outside := not_mem_context game adversary statement oracle extends_
  rw [firstIndex_update game adversary statement oracle extends_ answer,
    worst_congr game adversary source (index := firstIndex game adversary statement oracle)
      (oracle := oracle) (other := Function.update oracle target answer)
      (fun asked inside => (Function.update_of_ne (by rintro rfl; exact outside inside) _ _).symm),
    Test.bad_local _ challenge target oracle answer]

theorem worstBad_mass_le
    (degreeCovers : ∀ pair statement,
      (source pair statement).toVerifierInput.sumcheckDegreeBound ≤ width)
    (challenge : schedule.Challenge) (target : Point schedule width) (oracle : Oracle) :
    mass (worstBad game adversary source challenge target oracle) ≤
      Test.error schedule width challenge := by
  by_cases located : ∃ statement, Extends statement target = true
  · obtain ⟨statement₀, extends₀⟩ := located
    refine (mass_mono fun answer inside => ?_).trans (Test.bad_mass_le
      (source (worst game adversary source (firstIndex game adversary statement₀ oracle) oracle))
      (degreeCovers _) challenge target oracle)
    obtain ⟨statement, extends_, member⟩ := inside
    rwa [statement_eq_of_extends extends_ extends₀] at member
  · refine (mass_mono fun answer inside => ?_).trans (Test.bad_mass_le
      (source (worst game adversary source 0 oracle)) (degreeCovers _) challenge target oracle)
    obtain ⟨statement, extends_, _⟩ := inside
    exact absurd ⟨statement, extends_⟩ located

/-- The test step with the worst pair fixed: `Test.checked_error_le` on `worstBad`. -/
theorem worst_error_le
    (degreeCovers : ∀ pair statement,
      (source pair statement).toVerifierInput.sumcheckDegreeBound ≤ width)
    {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if FalseFor game adversary source
        (worst game adversary source (forkIndex game adversary oracle) oracle) oracle
      then (1 : ℝ) else 0) ≤
      ((queries + (Test.challenges schedule).length : Nat) : ℝ) *
        IndependentExecution.testError schedule.shape width :=
  Test.checked_error_le adversary game.statement game.messages (worstBad game adversary source)
    (worstBad_local game adversary source) (worstBad_mass_le game adversary source degreeCovers)
    bounded _ fun _ false_ => hits_mono (fun challenge _ member =>
      ⟨_, extends_point _ _ challenge, member⟩) false_

noncomputable def agreeChance (oracle : Oracle) : ℝ :=
  testChance game adversary source (forkIndex game adversary oracle) oracle
      (worst game adversary source (forkIndex game adversary oracle) oracle) /
    resampledSuccess game adversary (forkIndex game adversary oracle) oracle

theorem agreeChance_nonnegative (oracle : Oracle) :
    0 ≤ agreeChance game adversary source oracle :=
  div_nonneg (testChance_nonnegative game adversary source _ _ _)
    (resampledSuccess_nonnegative game adversary _ _)

variable (Fails : (Point schedule width → Answer) → (Fin schedule.rhoCount → Answer) → Prop)

/-- A failing pair's resampled valid pair at the base fork index moves the start claim,
collides, or agrees; agreement is a false acceptance of the failing pair's data. -/
theorem one_le_outcomes
    (failsValid : ∀ {oracle retries}, Fails oracle retries →
      Valid game adversary oracle retries)
    (agree : ∀ {oracle retries other otherRetries}, Fails oracle retries →
      Valid game adversary other otherRetries →
      game.statement (adversary.run other) = game.statement (adversary.run oracle) →
      game.start (adversary.run other) = game.start (adversary.run oracle) →
      witnessOf other otherRetries = witnessOf oracle retries →
      FalseFor game adversary source (oracle, retries) other)
    {oracle : Oracle} {retries : Retries} (fails : Fails oracle retries) :
    1 ≤ retryChance game adversary (forkIndex game adversary oracle) oracle
          (Collides game adversary witnessOf oracle retries) +
        retryChance game adversary (forkIndex game adversary oracle) oracle
          (Moves game adversary oracle) +
        agreeChance game adversary source oracle := by
  have positive := resampledSuccess_pos game adversary (failsValid fails)
  unfold retryChance agreeChance
  rw [← add_div, ← add_div, le_div_iff₀ positive, one_mul]
  refine le_trans ?_ (add_le_add le_rfl (le_worst game adversary source
    (forkIndex game adversary oracle) oracle (oracle, retries)))
  unfold resampledSuccess testChance
  rw [← Finset.expect_add_distrib, ← Finset.expect_add_distrib]
  refine Finset.expect_le_expect fun sample _ => ?_
  rw [success_eq]
  refine (retryMass_le_three game adversary _ ?_).trans
    (add_le_add le_rfl (retryMass_const_le game adversary _ _))
  rintro otherRetries ⟨forked, valid⟩
  by_cases moves : Moves game adversary oracle
      (overlay (context game adversary (forkIndex game adversary oracle) oracle) oracle sample)
      otherRetries
  · exact Or.inr (Or.inl ⟨forked, valid, moves⟩)
  by_cases collides : Collides game adversary witnessOf oracle retries
      (overlay (context game adversary (forkIndex game adversary oracle) oracle) oracle sample)
      otherRetries
  · exact Or.inl ⟨forked, valid, collides⟩
  · refine Or.inr (Or.inr ⟨forked, ?_⟩)
    have sameStart : game.start (adversary.run
        (overlay (context game adversary (forkIndex game adversary oracle) oracle) oracle
          sample)) = game.start (adversary.run oracle) :=
      Classical.byContradiction fun different => moves different
    exact agree fails valid (statement_eq_of_fork game adversary oracle sample forked) sameStart
      (Classical.byContradiction fun different => collides ⟨sameStart, different⟩)

theorem resampled_le (index : Nat) :
    𝔼 oracle, testChance game adversary source index oracle
          (worst game adversary source index oracle) *
        success game adversary index oracle / resampledSuccess game adversary index oracle ≤
      𝔼 oracle, (if forkIndex game adversary oracle = index ∧
        FalseFor game adversary source (worst game adversary source index oracle) oracle
          then (1 : ℝ) else 0) := by
  have determined (oracle sample : Oracle) :
      testChance game adversary source index
          (overlay (context game adversary index oracle) oracle sample)
          (worst game adversary source index
            (overlay (context game adversary index oracle) oracle sample)) =
        testChance game adversary source index oracle
          (worst game adversary source index oracle) := by
    have agree := overlay_agree (context game adversary index oracle) oracle sample
    rw [worst_congr game adversary source agree, testChance_congr game adversary source agree]
  calc
    _ = 𝔼 oracle, (if resampledSuccess game adversary index oracle = 0 then 0 else
          testChance game adversary source index oracle
            (worst game adversary source index oracle)) :=
      expect_div_resampled (context game adversary index)
        (context_stopping game adversary index) (success game adversary index)
        (fun oracle => testChance game adversary source index oracle
          (worst game adversary source index oracle)) determined
    _ ≤ 𝔼 oracle, testChance game adversary source index oracle
          (worst game adversary source index oracle) :=
      Finset.expect_le_expect fun oracle _ => by
        split
        · exact testChance_nonnegative game adversary source index oracle _
        · exact le_rfl
    _ = 𝔼 oracle, 𝔼 sample, (if forkIndex game adversary
            (overlay (context game adversary index oracle) oracle sample) = index ∧
          FalseFor game adversary source (worst game adversary source index
              (overlay (context game adversary index oracle) oracle sample))
            (overlay (context game adversary index oracle) oracle sample)
          then (1 : ℝ) else 0) := by
      refine Finset.expect_congr rfl fun oracle _ => ?_
      unfold testChance
      refine Finset.expect_congr rfl fun sample _ => ?_
      rw [worst_congr game adversary source
        (overlay_agree (context game adversary index oracle) oracle sample)]
    _ = _ := (expect_resample (context game adversary index)
      (context_stopping game adversary index) fun oracle =>
        if forkIndex game adversary oracle = index ∧
          FalseFor game adversary source (worst game adversary source index oracle) oracle
          then (1 : ℝ) else 0).symm

theorem forkIndex_lt_bound {queries : Nat} (bounded : adversary.QueryBound queries)
    (oracle : Oracle) :
    forkIndex game adversary oracle < queries + (Test.challenges schedule).length :=
  (forkIndex_lt game adversary oracle).trans_le
    ((Test.checked_bound adversary game.statement game.messages bounded).queries_length_le
      oracle)

theorem sum_success {queries : Nat} (bounded : adversary.QueryBound queries)
    (factor : Nat → Oracle → ℝ) (oracle : Oracle) :
    retryMass game adversary oracle (Valid game adversary oracle) *
        factor (forkIndex game adversary oracle) oracle /
        resampledSuccess game adversary (forkIndex game adversary oracle) oracle =
      ∑ index ∈ Finset.range (queries + (Test.challenges schedule).length),
        factor index oracle * success game adversary index oracle /
          resampledSuccess game adversary index oracle := by
  rw [Finset.sum_eq_single (forkIndex game adversary oracle)]
  · rw [success, ite_eq_left rfl, mul_comm (factor _ _)]
  · intro index _ different
    rw [success, ite_eq_right (Ne.symm different), mul_zero, zero_div]
  · intro outside
    exact absurd (Finset.mem_range.mpr (forkIndex_lt_bound game adversary bounded oracle))
      outside

/-- The uniqueness step: the weighted chance of a failing pair is at most
`(Q + challenge count) · testError` plus the collision and start chances. -/
theorem source_error_le
    (failsValid : ∀ {oracle retries}, Fails oracle retries →
      Valid game adversary oracle retries)
    (agree : ∀ {oracle retries other otherRetries}, Fails oracle retries →
      Valid game adversary other otherRetries →
      game.statement (adversary.run other) = game.statement (adversary.run oracle) →
      game.start (adversary.run other) = game.start (adversary.run oracle) →
      witnessOf other otherRetries = witnessOf oracle retries →
      FalseFor game adversary source (oracle, retries) other)
    (degreeCovers : ∀ pair statement,
      (source pair statement).toVerifierInput.sumcheckDegreeBound ≤ width)
    {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, ∑ retries, weight game adversary oracle retries *
        (if Fails oracle retries then 1 else 0) ≤
      ((queries + (Test.challenges schedule).length : Nat) : ℝ) *
          IndependentExecution.testError schedule.shape width +
        collisionChance game adversary witnessOf + startChance game adversary := by
  have pointwise (oracle : Oracle) (retries : Retries) :
      weight game adversary oracle retries * (if Fails oracle retries then 1 else 0) ≤
        weight game adversary oracle retries *
            (if Valid game adversary oracle retries then
              retryChance game adversary (forkIndex game adversary oracle) oracle
                (Collides game adversary witnessOf oracle retries) else 0) +
          weight game adversary oracle retries *
            (if Valid game adversary oracle retries then
              retryChance game adversary (forkIndex game adversary oracle) oracle
                (Moves game adversary oracle) else 0) +
          weight game adversary oracle retries *
            (if Valid game adversary oracle retries then 1 else 0) *
            agreeChance game adversary source oracle := by
    have nonnegative := weight_nonnegative game adversary oracle retries
    have collide := retryChance_nonnegative game adversary
      (forkIndex game adversary oracle) oracle (Collides game adversary witnessOf oracle retries)
    have move := retryChance_nonnegative game adversary
      (forkIndex game adversary oracle) oracle (Moves game adversary oracle)
    have agreement := agreeChance_nonnegative game adversary source oracle
    by_cases fails : Fails oracle retries
    · simp only [ite_eq_left fails, ite_eq_left (failsValid fails), mul_one]
      have key := mul_le_mul_of_nonneg_left
        (one_le_outcomes game adversary witnessOf source Fails failsValid agree fails) nonnegative
      rw [mul_one, mul_add, mul_add] at key
      exact key
    · rw [ite_eq_right fails, mul_zero]
      by_cases valid : Valid game adversary oracle retries
      · simp only [ite_eq_left valid, mul_one]
        have first := mul_nonneg nonnegative collide
        have second := mul_nonneg nonnegative move
        have third := mul_nonneg nonnegative agreement
        linarith
      · simp only [ite_eq_right valid, mul_zero, zero_mul, add_zero, le_refl]
  have summed :
      𝔼 oracle, ∑ retries, weight game adversary oracle retries *
          (if Fails oracle retries then 1 else 0) ≤
        collisionChance game adversary witnessOf + startChance game adversary +
          𝔼 oracle, retryMass game adversary oracle (Valid game adversary oracle) *
            agreeChance game adversary source oracle := by
    refine (Finset.expect_le_expect fun oracle _ => Finset.sum_le_sum fun retries _ =>
      pointwise oracle retries).trans (le_of_eq ?_)
    simp only [Finset.sum_add_distrib, Finset.expect_add_distrib, ← Finset.sum_mul]
    rfl
  have collapse (oracle : Oracle) :
      ∑ index ∈ Finset.range (queries + (Test.challenges schedule).length),
          (if forkIndex game adversary oracle = index ∧
            FalseFor game adversary source (worst game adversary source index oracle) oracle
            then (1 : ℝ) else 0) ≤
        if FalseFor game adversary source
            (worst game adversary source (forkIndex game adversary oracle) oracle) oracle
          then 1 else 0 := by
    rw [Finset.sum_eq_single (forkIndex game adversary oracle)]
    · by_cases happens : FalseFor game adversary source
          (worst game adversary source (forkIndex game adversary oracle) oracle) oracle <;>
        simp [happens]
    · intro index _ different
      exact ite_eq_right fun ⟨same, _⟩ => different same.symm
    · intro outside
      exact absurd (Finset.mem_range.mpr (forkIndex_lt_bound game adversary bounded oracle))
        outside
  have agreement :
      𝔼 oracle, retryMass game adversary oracle (Valid game adversary oracle) *
          agreeChance game adversary source oracle ≤
        ((queries + (Test.challenges schedule).length : Nat) : ℝ) *
          IndependentExecution.testError schedule.shape width := by
    simp only [agreeChance, ← mul_div_assoc, sum_success game adversary bounded
      (fun index oracle => testChance game adversary source index oracle
        (worst game adversary source index oracle))]
    rw [Finset.expect_sum_comm]
    refine (Finset.sum_le_sum fun index _ => resampled_le game adversary source index).trans ?_
    rw [← Finset.expect_sum_comm]
    exact (Finset.expect_le_expect fun oracle _ => collapse oracle).trans
      (worst_error_le game adversary source degreeCovers bounded)
  linarith

end Test

end Game

section Knowledge

variable {schedule : Schedule} {width : Nat} {Output Start Witness : Type}
  (game : Game schedule width Output Start)
  (adversary : OracleComp (Point schedule width) Answer Output)
  (witnessOf : (Point schedule width → Answer) → (Fin schedule.rhoCount → Answer) → Witness)
  (source : (Point schedule width → Answer) × (Fin schedule.rhoCount → Answer) →
    schedule.Statement → ProtocolPolynomial.Data K schedule.shape)

local notation "Oracle" => Point schedule width → Answer
local notation "Retries" => Fin schedule.rhoCount → Answer

variable (schedule width) in
/-- The statistical part of the knowledge error: the fork step's sampling error and the test
error. -/
noncomputable def statisticalError (queries : Nat) : ℝ :=
  schedule.rhoCount * (((queries + schedule.rhoCount : Nat) : ℝ) * Fork.sampleError) +
    ((queries + (Test.challenges schedule).length : Nat) : ℝ) *
      IndependentExecution.testError schedule.shape width

/-- `RandomOracleKnowledge.knowledge_error_le` on any schedule: a successful run extracts
(`Extracts`, weighted by the retry law), except with the statistical error, the mismatch bound
`mismatch` of the fork step, and the collision and start chances of the uniqueness step. -/
theorem knowledge_error_le (Extracts Fails : Oracle → Retries → Prop)
    (splits : ∀ {oracle retries}, Valid game adversary oracle retries →
      Extracts oracle retries ∨ Fails oracle retries)
    (failsValid : ∀ {oracle retries}, Fails oracle retries →
      Valid game adversary oracle retries)
    (agree : ∀ {oracle retries other otherRetries}, Fails oracle retries →
      Valid game adversary other otherRetries →
      game.statement (adversary.run other) = game.statement (adversary.run oracle) →
      game.start (adversary.run other) = game.start (adversary.run oracle) →
      witnessOf other otherRetries = witnessOf oracle retries →
      FalseFor game adversary source (oracle, retries) other)
    (degreeCovers : ∀ pair statement,
      (source pair statement).toVerifierInput.sumcheckDegreeBound ≤ width)
    (mismatch : Fin schedule.rhoCount → Oracle → ℝ)
    (dominates : ∀ oracle, Fork.Succeeds game adversary oracle → ∀ index,
      Fork.mismatchChance game adversary index oracle ≤ mismatch index oracle)
    {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if Fork.Succeeds game adversary oracle then (1 : ℝ) else 0) ≤
      𝔼 oracle, ∑ retries, weight game adversary oracle retries *
          (if Extracts oracle retries then 1 else 0) +
        (statisticalError schedule width queries +
          (𝔼 oracle, (if Fork.Succeeds game adversary oracle then
              ∑ index, mismatch index oracle else 0) +
            collisionChance game adversary witnessOf + startChance game adversary)) := by
  have pointwise (oracle : Oracle) :
      (if Fork.Succeeds game adversary oracle then (1 : ℝ) else 0) ≤
        ∑ retries, weight game adversary oracle retries *
            (if Extracts oracle retries then 1 else 0) +
          (if Fork.Succeeds game adversary oracle then
            ∑ retries, retryWeight (Fork.retrySets game adversary oracle) retries *
              (if ∀ index, Fork.ForkValid game adversary oracle retries index then 0 else 1)
          else 0) +
          ∑ retries, weight game adversary oracle retries *
            (if Fails oracle retries then 1 else 0) := by
    have extracts : 0 ≤ ∑ retries, weight game adversary oracle retries *
        (if Extracts oracle retries then 1 else 0) :=
      Finset.sum_nonneg fun retries _ => mul_nonneg
        (weight_nonnegative game adversary oracle retries) (by split <;> norm_num)
    have failing : 0 ≤ ∑ retries, weight game adversary oracle retries *
        (if Fails oracle retries then 1 else 0) :=
      Finset.sum_nonneg fun retries _ => mul_nonneg
        (weight_nonnegative game adversary oracle retries) (by split <;> norm_num)
    by_cases succeeds : Fork.Succeeds game adversary oracle
    · rw [ite_eq_left succeeds, ite_eq_left succeeds, ← Finset.sum_add_distrib,
        ← Finset.sum_add_distrib]
      have total : ∑ retries, weight game adversary oracle retries = 1 :=
        retryWeight_sum_eq_one _ fun index =>
          Fork.mass_ne_zero (Fork.base_mem_retrySets game adversary succeeds index)
      refine total.symm.le.trans (Finset.sum_le_sum fun retries _ => ?_)
      have nonnegative := weight_nonnegative game adversary oracle retries
      change weight game adversary oracle retries ≤
        weight game adversary oracle retries * (if Extracts oracle retries then 1 else 0) +
          weight game adversary oracle retries *
            (if ∀ index, Fork.ForkValid game adversary oracle retries index then 0 else 1) +
          weight game adversary oracle retries * (if Fails oracle retries then 1 else 0)
      rw [← mul_add, ← mul_add]
      refine le_mul_of_one_le_right nonnegative ?_
      have first : (0 : ℝ) ≤ if Extracts oracle retries then 1 else 0 := by
        split <;> norm_num
      have third : (0 : ℝ) ≤ if Fails oracle retries then 1 else 0 := by
        split <;> norm_num
      by_cases forks : ∀ index, Fork.ForkValid game adversary oracle retries index
      · rw [ite_eq_left forks]
        rcases splits ⟨succeeds, forks⟩ with extracted | failed
        · rw [ite_eq_left extracted]
          linarith
        · rw [ite_eq_left failed]
          linarith
      · rw [ite_eq_right forks]
        linarith
    · rw [ite_eq_right succeeds, ite_eq_right succeeds]
      linarith
  have forkStep := Fork.fork_failure_le_of_mismatch game adversary bounded mismatch dominates
  have uniqueness := source_error_le game adversary witnessOf source Fails failsValid agree
    degreeCovers bounded
  have summed := Finset.expect_le_expect (s := Finset.univ)
    fun oracle (_ : oracle ∈ Finset.univ) =>
    pointwise oracle
  rw [Finset.expect_add_distrib, Finset.expect_add_distrib] at summed
  unfold statisticalError
  linarith

end Knowledge

end NightstreamFPrime.Lifecycle.RandomOracle.Uniqueness
