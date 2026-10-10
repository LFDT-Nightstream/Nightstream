import NightstreamFPrime.Lifecycle.RandomOracleExtraction

/-!
Owns the uniqueness step of the production NIFS extraction in the
random-oracle model (Lemma 6 of SECURITY_MODEL.md).

Inputs: an adaptive oracle adversary and its claims, and the `Π_RLC`
extraction of `RandomOracleExtraction`.

Outputs:
- `forkIndex`: the first checked query that extends the claim's statement
  calls; every challenge of that statement comes at or after it;
- `source_error_le`: the extracted witness fails the source relation with
  probability at most `(Q + 74) * testError`, plus the chance that the
  binding reduction finds a different witness for the same running statement
  (`collisionChance`; `RandomOracleBinding.collisionChance_le_kernelChance`
  bounds it by the chance that the binding reduction returns a short kernel
  vector of the Ajtai key) or a different running statement
  (`runningChance`; `Export.Stage1.RandomOracleLink.runningChance_le` bounds
  it by a state-hash collision when the claim carries `PriorLink`);
- `expected_reruns_le`: the binding reduction takes at most `Q + 74`
  expected reruns.

The binding reduction runs the extractor, then reruns the adversary from the
fork context (the answers queried before the fork index) until a rerun
succeeds at that index, and compares the two extractions. A rerun that keeps
the running statement and the witness is a false acceptance of the first
run's witness. That witness is replaced by the worst witness of the context
(`worst`), which the context alone determines, so each bad set stays local and
`RandomOracle.escape_le` charges every query once. Averaging over the base
run removes the division by the rerun success (`expect_div_resampled`).

Does not own: the numerical hardness of MSIS or of the state hash, or an
executable form of the binding reduction.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleUniqueness

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

/-! ## Challenge lists -/

/-- Every challenge of one execution, in schedule order. -/
def challenges : List Challenge :=
  List.ofFn Challenge.alpha ++ [Challenge.gamma] ++ List.ofFn Challenge.round ++
    List.ofFn Challenge.rho

theorem mem_challenges (challenge : Challenge) : challenge ∈ challenges := by
  cases challenge <;> simp [challenges]

/-- The `74` of `Q + 74`: 28 `α` coordinates, `γ`, 28 rounds and 17 `Π_RLC`
scalars. -/
theorem challenges_length : challenges.length = 74 := by
  simp [challenges]
  rfl

/-- An item where `value` is largest. -/
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

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Whether a call list extends the statement calls of `fresh`. -/
def Extends (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) {degree : Nat}
    (target : Point logicalWidth publicFits degree) : Bool :=
  (statementCalls fresh).isPrefixOf target.val

/-- Every challenge point of an execution extends its statement calls. -/
theorem extends_point {degree : Nat}
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (proof : Proof degree) (challenge : Challenge) :
    Extends fresh (point fresh proof challenge) = true := by
  rw [Extends, List.isPrefixOf_iff_prefix]
  change statementCalls fresh <+: challengeCalls fresh proof challenge
  cases challenge <;>
    simp only [challengeCalls, proverCalls, roundPrefixCalls, outputCalls, preRoundCalls,
      List.append_assoc] <;> exact List.prefix_append _ _

/-- One call list extends the statement calls of at most one fresh statement. -/
theorem fresh_eq_of_extends {degree : Nat}
    {fresh fresh' : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)}
    {target : Point logicalWidth publicFits degree}
    (left : Extends fresh target = true) (right : Extends fresh' target = true) : fresh = fresh' := by
  simp only [Extends, List.isPrefixOf_iff_prefix] at left right
  have lengths := statementCalls_length fresh fresh'
  exact statementCalls_identify_fresh
    ((List.prefix_of_prefix_length_le left right lengths.le).eq_of_length lengths)

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

local notation "Degree" => ProductionKey.degreeBound relation
local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer

variable {Output : Type}
  (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
    Answer Output)
  (claim : Output → Claim relation)

/-! ## Checked runs and fork contexts -/

/-- The adversary followed by the verifier's query of every challenge point. -/
noncomputable def checked :
    OracleComp (Point logicalWidth publicFits Degree) Answer Output :=
  adversary.thenQueries fun output =>
    challenges.map (point (claim output).fresh (claim output).proof)

theorem checked_bound {queries : Nat} (bounded : adversary.QueryBound queries) :
    (checked relation adversary claim).QueryBound (queries + challenges.length) :=
  bounded.thenQueries fun _ => by simp

theorem mem_checked (oracle : Oracle) (challenge : Challenge) :
    point (claimed relation adversary claim oracle).fresh
        (claimed relation adversary claim oracle).proof challenge ∈
      (checked relation adversary claim).queries oracle :=
  OracleComp.mem_queries_thenQueries _ _ _
    (List.mem_map.mpr ⟨challenge, mem_challenges challenge, rfl⟩)

/-- The checked query index at which the statement of `fresh` first appears. -/
noncomputable def firstIndex (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (oracle : Oracle) : Nat :=
  ((checked relation adversary claim).queries oracle).findIdx (Extends fresh)

/-- The fork index: where the claim's own statement first appears. -/
noncomputable def forkIndex (oracle : Oracle) : Nat :=
  firstIndex relation adversary claim (claimed relation adversary claim oracle).fresh oracle

/-- The points the run has seen before query `index`: the fork context. -/
noncomputable def context (index : Nat) (oracle : Oracle) :
    Finset (Point logicalWidth publicFits Degree) :=
  queriedBefore (checked relation adversary claim) index oracle

theorem context_stopping (index : Nat) :
    Stopping (context relation adversary claim index) :=
  queriedBefore_stopping _ index

/-- The statement's first index ignores the answer at any point that extends
the statement: such a point is queried no earlier. -/
theorem firstIndex_update (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (oracle : Oracle) {target : Point logicalWidth publicFits Degree}
    (extends_ : Extends fresh target = true) (answer : Answer) :
    firstIndex relation adversary claim fresh (Function.update oracle target answer) =
      firstIndex relation adversary claim fresh oracle := by
  unfold firstIndex
  apply findIdx_eq_of_take (index := ((checked relation adversary claim).queries oracle).findIdx
    (Extends fresh)) _ le_rfl
  symm
  apply OracleComp.take_succ_queries_congr
  intro point inside
  have outside : point ≠ target := fun same => not_mem_take_findIdx extends_ (same ▸ inside)
  exact (Function.update_of_ne outside _ _).symm

/-- The target is outside the context of its statement's first index. -/
theorem not_mem_context (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (oracle : Oracle) {target : Point logicalWidth publicFits Degree}
    (extends_ : Extends fresh target = true) :
    target ∉ context relation adversary claim (firstIndex relation adversary claim fresh oracle) oracle := by
  intro inside
  exact not_mem_take_findIdx extends_ (List.mem_toFinset.mp inside)

/-- A fork keeps the context and resamples everything else; another oracle that
agrees on the context makes the same fork. -/
theorem overlay_context_congr {index : Nat} {oracle other : Oracle}
    (agree : ∀ point ∈ context relation adversary claim index oracle, oracle point = other point)
    (fresh : Oracle) :
    overlay (context relation adversary claim index other) other fresh =
      overlay (context relation adversary claim index oracle) oracle fresh := by
  rw [context_stopping relation adversary claim index oracle other agree]
  funext point
  by_cases inside : point ∈ context relation adversary claim index oracle
  · simp [overlay, inside, agree point inside]
  · simp [overlay, inside]

/-! ## Runs and their extractions -/

local notation "Retries" => Fin Nifs.PaperProfile.arity.total → Answer

/-- The run's retries form a complete `Π_RLC` fork. -/
def Valid (oracle : Oracle) (retries : Retries) : Prop :=
  Succeeds relation ajtai adversary claim oracle ∧
    ∀ index, ForkValid relation ajtai adversary claim oracle retries index

/-- The witness a valid run extracts. -/
noncomputable def witnessOf (oracle : Oracle) (retries : Retries) :
    Option (OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)) :=
  if valid : Valid relation ajtai adversary claim oracle retries then
    some (extractedWitness relation ajtai adversary claim oracle retries valid.1 valid.2)
  else none

/-- The retry law of one run. -/
noncomputable def weight (oracle : Oracle) (retries : Retries) : ℝ :=
  retryWeight (retrySets relation ajtai adversary claim oracle) retries

/-- The mass of the run's retries that satisfy `event`. -/
noncomputable def retryMass (oracle : Oracle) (event : Retries → Prop) : ℝ :=
  ∑ retries, weight relation ajtai adversary claim oracle retries *
    (if event retries then 1 else 0)

/-- The probability that a run extracts at fork index `index`. -/
noncomputable def success (index : Nat) (oracle : Oracle) : ℝ :=
  if forkIndex relation adversary claim oracle = index then
    retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle)
  else 0

/-- The probability that a run resampled after the context of `index`
extracts at that index. -/
noncomputable def resampledSuccess (index : Nat) (oracle : Oracle) : ℝ :=
  𝔼 fresh, success relation ajtai adversary claim index
    (overlay (context relation adversary claim index oracle) oracle fresh)

/-- The chance that the binding reduction's first successful retry from the
context of `index` has `outcome` against the base run. -/
noncomputable def retryChance (index : Nat) (oracle : Oracle)
    (outcome : Oracle → Retries → Prop) : ℝ :=
  (𝔼 fresh, retryMass relation ajtai adversary claim
      (overlay (context relation adversary claim index oracle) oracle fresh)
      (fun retries => forkIndex relation adversary claim
          (overlay (context relation adversary claim index oracle) oracle fresh) = index ∧
        Valid relation ajtai adversary claim
          (overlay (context relation adversary claim index oracle) oracle fresh) retries ∧
        outcome (overlay (context relation adversary claim index oracle) oracle fresh) retries)) /
    resampledSuccess relation ajtai adversary claim index oracle

/-- The retry keeps the running statement but extracts another witness. -/
def Collides (oracle : Oracle) (retries : Retries) (other : Oracle) (otherRetries : Retries) : Prop :=
  (claimed relation adversary claim other).running = (claimed relation adversary claim oracle).running ∧
    witnessOf relation ajtai adversary claim other otherRetries ≠
      witnessOf relation ajtai adversary claim oracle retries

/-- The retry changes the running statement. -/
def Moves (oracle other : Oracle) (_ : Retries) : Prop :=
  (claimed relation adversary claim other).running ≠ (claimed relation adversary claim oracle).running

/-- The binding reduction: the base run extracts, and its retry from the same
context collides. Two different witnesses for one statement give a short
kernel vector of the Ajtai key (`RandomOracleBinding.rerunKernel_isSome`). -/
noncomputable def collisionChance : ℝ :=
  𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
    (if Valid relation ajtai adversary claim oracle retries then
      retryChance relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle
        (Collides relation ajtai adversary claim oracle retries)
    else 0)

/-- The same reduction, when the retry changes the running statement. With the
prior link this is a state-hash collision
(`Export.Stage1.RandomOracleLink.runningChance_le`). -/
noncomputable def runningChance : ℝ :=
  𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
    (if Valid relation ajtai adversary claim oracle retries then
      retryChance relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle
        (Moves relation adversary claim oracle)
    else 0)

/-! ## Facts about one run -/

theorem weight_nonnegative (oracle : Oracle) (retries : Retries) :
    0 ≤ weight relation ajtai adversary claim oracle retries :=
  retryWeight_nonnegative _ _

theorem retryMass_nonnegative (oracle : Oracle) (event : Retries → Prop) :
    0 ≤ retryMass relation ajtai adversary claim oracle event :=
  Finset.sum_nonneg fun retries _ => mul_nonneg (weight_nonnegative relation ajtai adversary claim
    oracle retries) (by split <;> norm_num)

theorem retryMass_le_one (oracle : Oracle) (event : Retries → Prop) :
    retryMass relation ajtai adversary claim oracle event ≤ 1 :=
  (Finset.sum_le_sum fun retries _ => mul_le_of_le_one_right (weight_nonnegative relation ajtai
    adversary claim oracle retries) (by split <;> norm_num)).trans (retryWeight_sum_le_one _)

/-- An event that does not read the retries has mass at most its indicator. -/
theorem retryMass_const_le (oracle : Oracle) (event : Prop) [Decidable event] :
    retryMass relation ajtai adversary claim oracle (fun _ => event) ≤ if event then 1 else 0 := by
  by_cases happens : event
  · rw [if_pos happens]
    exact retryMass_le_one relation ajtai adversary claim oracle _
  · simp [retryMass, happens]

/-- The mass of an event is at most the masses of three events that cover it. -/
theorem retryMass_le_three (oracle : Oracle) {event first second third : Retries → Prop}
    (covered : ∀ retries, event retries → first retries ∨ second retries ∨ third retries) :
    retryMass relation ajtai adversary claim oracle event ≤
      retryMass relation ajtai adversary claim oracle first +
        retryMass relation ajtai adversary claim oracle second +
        retryMass relation ajtai adversary claim oracle third := by
  unfold retryMass
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun retries _ => ?_
  rw [← mul_add, ← mul_add]
  refine mul_le_mul_of_nonneg_left ?_ (weight_nonnegative relation ajtai adversary claim oracle retries)
  have one : (0 : ℝ) ≤ if first retries then 1 else 0 := by split <;> norm_num
  have two : (0 : ℝ) ≤ if second retries then 1 else 0 := by split <;> norm_num
  have three : (0 : ℝ) ≤ if third retries then 1 else 0 := by split <;> norm_num
  by_cases happens : event retries
  · rw [if_pos happens]
    rcases covered retries happens with holds | holds | holds
    · have : (1 : ℝ) ≤ if first retries then 1 else 0 := by rw [if_pos holds]
      linarith
    · have : (1 : ℝ) ≤ if second retries then 1 else 0 := by rw [if_pos holds]
      linarith
    · have : (1 : ℝ) ≤ if third retries then 1 else 0 := by rw [if_pos holds]
      linarith
  · rw [if_neg happens]
    linarith

/-- A retry weight is at most the mass of any event that holds there. -/
theorem weight_le_retryMass (oracle : Oracle) {event : Retries → Prop} {retries : Retries}
    (happens : event retries) :
    weight relation ajtai adversary claim oracle retries ≤
      retryMass relation ajtai adversary claim oracle event := by
  unfold retryMass
  calc
    weight relation ajtai adversary claim oracle retries =
        weight relation ajtai adversary claim oracle retries * (if event retries then 1 else 0) := by
      rw [if_pos happens, mul_one]
    _ ≤ _ := Finset.single_le_sum
      (f := fun retries => weight relation ajtai adversary claim oracle retries *
        (if event retries then (1 : ℝ) else 0))
      (fun retries _ => mul_nonneg (weight_nonnegative relation ajtai adversary claim oracle retries)
        (by split <;> norm_num)) (Finset.mem_univ retries)

/-- The retries of a valid run have positive weight. -/
theorem weight_pos {oracle : Oracle} {retries : Retries}
    (valid : Valid relation ajtai adversary claim oracle retries) :
    0 < weight relation ajtai adversary claim oracle retries := by
  unfold weight retryWeight
  apply Finset.prod_pos
  intro index _
  have inside : retries index ∈ retrySets relation ajtai adversary claim oracle index :=
    (valid.2 index).1
  rw [if_pos inside]
  have card : (0 : ℝ) < Fintype.card Answer := by exact_mod_cast Fintype.card_pos
  have positive := lt_of_le_of_ne (mass_nonnegative _) (mass_ne_zero inside).symm
  exact div_pos one_pos (mul_pos card positive)

/-- The success mass is the mass of the event "forks at `index` and is valid". -/
theorem success_eq (index : Nat) (oracle : Oracle) :
    success relation ajtai adversary claim index oracle =
      retryMass relation ajtai adversary claim oracle fun retries =>
        forkIndex relation adversary claim oracle = index ∧
          Valid relation ajtai adversary claim oracle retries := by
  unfold success
  by_cases forked : forkIndex relation adversary claim oracle = index
  · rw [if_pos forked]
    simp only [forked, true_and]
  · simp [forked, retryMass]

theorem success_nonnegative (index : Nat) (oracle : Oracle) :
    0 ≤ success relation ajtai adversary claim index oracle := by
  rw [success_eq]
  exact retryMass_nonnegative relation ajtai adversary claim oracle _

theorem resampledSuccess_nonnegative (index : Nat) (oracle : Oracle) :
    0 ≤ resampledSuccess relation ajtai adversary claim index oracle :=
  Finset.expect_nonneg fun _ _ => success_nonnegative relation ajtai adversary claim index _

theorem retryChance_nonnegative (index : Nat) (oracle : Oracle) (outcome : Oracle → Retries → Prop) :
    0 ≤ retryChance relation ajtai adversary claim index oracle outcome :=
  div_nonneg (Finset.expect_nonneg fun _ _ => retryMass_nonnegative relation ajtai adversary claim
    _ _) (resampledSuccess_nonnegative relation ajtai adversary claim index oracle)

/-- A larger event has at least the same mass. -/
theorem retryMass_mono (oracle : Oracle) {event event' : Retries → Prop}
    (implies : ∀ retries, event retries → event' retries) :
    retryMass relation ajtai adversary claim oracle event ≤
      retryMass relation ajtai adversary claim oracle event' :=
  Finset.sum_le_sum fun retries _ => mul_le_mul_of_nonneg_left
    (by by_cases happens : event retries
        · rw [if_pos happens, if_pos (implies retries happens)]
        · rw [if_neg happens]
          split <;> norm_num)
    (weight_nonnegative relation ajtai adversary claim oracle retries)

/-- An outcome that is larger on every valid rerun from the context of
`index` has at least the same rerun chance. -/
theorem retryChance_mono (index : Nat) (oracle : Oracle) {outcome outcome' : Oracle → Retries → Prop}
    (implies : ∀ fresh retries,
      forkIndex relation adversary claim (overlay (context relation adversary claim index oracle) oracle fresh) =
        index →
      Valid relation ajtai adversary claim (overlay (context relation adversary claim index oracle) oracle fresh)
        retries →
      outcome (overlay (context relation adversary claim index oracle) oracle fresh) retries →
      outcome' (overlay (context relation adversary claim index oracle) oracle fresh) retries) :
    retryChance relation ajtai adversary claim index oracle outcome ≤
      retryChance relation ajtai adversary claim index oracle outcome' :=
  div_le_div_of_nonneg_right
    (Finset.expect_le_expect fun fresh _ => retryMass_mono relation ajtai adversary claim _
      fun retries holds => ⟨holds.1, holds.2.1, implies fresh retries holds.1 holds.2.1 holds.2.2⟩)
    (resampledSuccess_nonnegative relation ajtai adversary claim index oracle)

/-- A rerun chance is a conditional probability: at most one. -/
theorem retryChance_le_one (index : Nat) (oracle : Oracle) (outcome : Oracle → Retries → Prop) :
    retryChance relation ajtai adversary claim index oracle outcome ≤ 1 :=
  div_le_one_of_le₀
    (Finset.expect_le_expect fun _ _ => by
      rw [success_eq]
      exact retryMass_mono relation ajtai adversary claim _ fun _ holds => ⟨holds.1, holds.2.1⟩)
    (resampledSuccess_nonnegative relation ajtai adversary claim index oracle)

/-- After a valid run, the resampled runs from its fork context succeed with
positive probability: the run itself is one of them. -/
theorem resampledSuccess_pos {oracle : Oracle} {retries : Retries}
    (valid : Valid relation ajtai adversary claim oracle retries) :
    0 < resampledSuccess relation ajtai adversary claim
      (forkIndex relation adversary claim oracle) oracle := by
  have own : 0 < success relation ajtai adversary claim (forkIndex relation adversary claim oracle)
      (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
        oracle oracle) := by
    rw [overlay_self, success_eq]
    exact (weight_pos relation ajtai adversary claim valid).trans_le
      (weight_le_retryMass relation ajtai adversary claim oracle ⟨rfl, valid⟩)
  unfold resampledSuccess
  exact expect_pos (fun fresh => success_nonnegative relation ajtai adversary claim _ _) own

/-- A run forks before the end of its checked queries. -/
theorem forkIndex_lt (oracle : Oracle) :
    forkIndex relation adversary claim oracle <
      ((checked relation adversary claim).queries oracle).length :=
  List.findIdx_lt_length_of_exists ⟨_, mem_checked relation adversary claim oracle .gamma,
    extends_point _ _ .gamma⟩

/-- Two runs with one context that both fork at its index output the same fresh
statement: the query at that index extends both statement call lists. -/
theorem fresh_eq_of_fork (oracle fresh : Oracle)
    (forked : forkIndex relation adversary claim
      (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
        oracle fresh) = forkIndex relation adversary claim oracle) :
    (claimed relation adversary claim
        (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
          oracle fresh)).fresh =
      (claimed relation adversary claim oracle).fresh := by
  set index := forkIndex relation adversary claim oracle with index_def
  set other := overlay (context relation adversary claim index oracle) oracle fresh with other_def
  have same : ((checked relation adversary claim).queries other).take (index + 1) =
      ((checked relation adversary claim).queries oracle).take (index + 1) :=
    OracleComp.take_succ_queries_congr oracle other _ index fun point inside =>
      overlay_agree _ oracle fresh point (List.mem_toFinset.mpr inside)
  have baseBound : index < ((checked relation adversary claim).queries oracle).length :=
    forkIndex_lt relation adversary claim oracle
  have nextBound : index < ((checked relation adversary claim).queries other).length := by
    have bound := forkIndex_lt relation adversary claim other
    rwa [forked] at bound
  have samePoint : ((checked relation adversary claim).queries other)[index] =
      ((checked relation adversary claim).queries oracle)[index] := by
    rw [List.getElem_take' nextBound (Nat.lt_succ_self index),
      List.getElem_take' baseBound (Nat.lt_succ_self index)]
    exact List.getElem_of_eq same _
  have baseExtends := ((List.findIdx_eq (p := Extends (claimed relation adversary claim oracle).fresh)
    baseBound).mp index_def.symm).1
  have nextExtends := ((List.findIdx_eq (p := Extends (claimed relation adversary claim other).fresh)
    nextBound).mp forked).1
  rw [samePoint] at nextExtends
  exact fresh_eq_of_extends nextExtends baseExtends

/-! ## The worst witness of a fork context -/

/-- The first run `pair` extracts a witness, and the run under `oracle` is a
false acceptance of that witness for the first run's running statement. -/
def FalseFor (pair : Oracle × Retries) (oracle : Oracle) : Prop :=
  ∃ valid : Valid relation ajtai adversary claim pair.1 pair.2,
    FalseAcceptance relation ajtai (claimed relation adversary claim pair.1).running
      (extractedWitness relation ajtai adversary claim pair.1 pair.2 valid.1 valid.2) oracle
      (claimed relation adversary claim oracle).fresh (claimed relation adversary claim oracle).proof

/-- The chance that a run resampled after the context of `index` forks at
`index` and falsely accepts the witness of `pair`. -/
noncomputable def testChance (index : Nat) (oracle : Oracle) (pair : Oracle × Retries) : ℝ :=
  𝔼 fresh, if forkIndex relation adversary claim
        (overlay (context relation adversary claim index oracle) oracle fresh) = index ∧
      FalseFor relation ajtai adversary claim pair
        (overlay (context relation adversary claim index oracle) oracle fresh) then 1 else 0

/-- A first run whose witness the resampled runs from the context of `index`
falsely accept most often. -/
noncomputable def worst (index : Nat) (oracle : Oracle) : Oracle × Retries :=
  best (testChance relation ajtai adversary claim index oracle)

theorem le_worst (index : Nat) (oracle : Oracle) (pair : Oracle × Retries) :
    testChance relation ajtai adversary claim index oracle pair ≤
      testChance relation ajtai adversary claim index oracle
        (worst relation ajtai adversary claim index oracle) :=
  le_best _ pair

theorem testChance_nonnegative (index : Nat) (oracle : Oracle) (pair : Oracle × Retries) :
    0 ≤ testChance relation ajtai adversary claim index oracle pair :=
  Finset.expect_nonneg fun _ _ => by split <;> norm_num

/-- The test chances read `oracle` only on the context of `index`. -/
theorem testChance_congr {index : Nat} {oracle other : Oracle}
    (agree : ∀ point ∈ context relation adversary claim index oracle, oracle point = other point) :
    testChance relation ajtai adversary claim index other =
      testChance relation ajtai adversary claim index oracle := by
  funext pair
  unfold testChance
  simp only [overlay_context_congr relation adversary claim agree]

theorem worst_congr {index : Nat} {oracle other : Oracle}
    (agree : ∀ point ∈ context relation adversary claim index oracle, oracle point = other point) :
    worst relation ajtai adversary claim index other = worst relation ajtai adversary claim index oracle := by
  unfold worst
  rw [testChance_congr relation ajtai adversary claim agree]

/-! ## Bad sets for the worst witness -/

/-- The answers at `target` that put `challenge` into its bad set for the
witness of `pair`. -/
def pairBad (pair : Oracle × Retries) (challenge : Challenge)
    (target : Point logicalWidth publicFits Degree) (oracle : Oracle) : Set Answer :=
  {answer | ∃ valid : Valid relation ajtai adversary claim pair.1 pair.2,
    answer ∈ bad relation ajtai (claimed relation adversary claim pair.1).running
      (extractedWitness relation ajtai adversary claim pair.1 pair.2 valid.1 valid.2) challenge target
      oracle}

theorem pairBad_local (pair : Oracle × Retries) (challenge : Challenge) :
    Local (pairBad relation ajtai adversary claim pair challenge) := by
  intro target oracle answer
  ext candidate
  simp only [pairBad, Set.mem_setOf_eq, bad_local relation ajtai _ _ challenge target oracle answer]

theorem pairBad_mass_le (pair : Oracle × Retries) (challenge : Challenge)
    (target : Point logicalWidth publicFits Degree) (oracle : Oracle) :
    mass (pairBad relation ajtai adversary claim pair challenge target oracle) ≤ error challenge := by
  by_cases valid : Valid relation ajtai adversary claim pair.1 pair.2
  · refine (mass_mono fun answer inside => ?_).trans (bad_mass_le relation ajtai
      (claimed relation adversary claim pair.1).running
      (extractedWitness relation ajtai adversary claim pair.1 pair.2 valid.1 valid.2) challenge target
      oracle)
    obtain ⟨_, member⟩ := inside
    exact member
  · have empty : pairBad relation ajtai adversary claim pair challenge target oracle = ∅ := by
      ext answer
      simp only [pairBad, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      rintro ⟨other, _⟩
      exact valid other
    rw [empty]
    simp only [mass, Set.mem_empty_iff_false, if_false, Finset.expect_const_zero]
    exact error_nonnegative challenge

/-- The answers at `target` that put `challenge` into its bad set for the
worst witness of the context where the statement of `target` first appears.
That context ends before `target`, so the set does not read `target`. -/
def worstBad (challenge : Challenge) (target : Point logicalWidth publicFits Degree)
    (oracle : Oracle) : Set Answer :=
  {answer | ∃ fresh, Extends fresh target = true ∧
    answer ∈ pairBad relation ajtai adversary claim
      (worst relation ajtai adversary claim (firstIndex relation adversary claim fresh oracle) oracle)
      challenge target oracle}

theorem worstBad_local (challenge : Challenge) :
    Local (worstBad relation ajtai adversary claim challenge) := by
  intro target oracle answer
  ext candidate
  simp only [worstBad, Set.mem_setOf_eq]
  refine exists_congr fun fresh => and_congr_right fun extends_ => ?_
  have outside := not_mem_context relation adversary claim fresh oracle extends_
  rw [firstIndex_update relation adversary claim fresh oracle extends_ answer,
    worst_congr relation ajtai adversary claim (index := firstIndex relation adversary claim fresh oracle)
      (oracle := oracle) (other := Function.update oracle target answer)
      (fun point inside => (Function.update_of_ne (by rintro rfl; exact outside inside) _ _).symm),
    pairBad_local relation ajtai adversary claim _ challenge target oracle answer]

theorem worstBad_mass_le (challenge : Challenge) (target : Point logicalWidth publicFits Degree)
    (oracle : Oracle) :
    mass (worstBad relation ajtai adversary claim challenge target oracle) ≤ error challenge := by
  by_cases located : ∃ fresh, Extends fresh target = true
  · obtain ⟨fresh₀, extends₀⟩ := located
    refine (mass_mono fun answer inside => ?_).trans (pairBad_mass_le relation ajtai adversary claim
      (worst relation ajtai adversary claim (firstIndex relation adversary claim fresh₀ oracle) oracle)
      challenge target oracle)
    obtain ⟨fresh, extends_, member⟩ := inside
    rwa [fresh_eq_of_extends extends_ extends₀] at member
  · refine (mass_mono fun answer inside => ?_).trans (pairBad_mass_le relation ajtai adversary claim
      (worst relation ajtai adversary claim 0 oracle) challenge target oracle)
    obtain ⟨fresh, extends_, _⟩ := inside
    exact absurd ⟨fresh, extends_⟩ located

/-- A run falsely accepts the worst witness of its own fork context with
probability at most `(Q + 74) * testError`. -/
theorem worst_error_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, (if FalseFor relation ajtai adversary claim
        (worst relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle) oracle
      then (1 : ℝ) else 0) ≤
      ((queries + challenges.length : Nat) : ℝ) * IndependentExecution.testError productionShape 8 := by
  refine hits_error_le relation _ (fun challenge oracle =>
      oracle (point (claimed relation adversary claim oracle).fresh
          (claimed relation adversary claim oracle).proof challenge) ∈
        worstBad relation ajtai adversary claim challenge
          (point (claimed relation adversary claim oracle).fresh
            (claimed relation adversary claim oracle).proof challenge) oracle) _ ?_ ?_
  · rintro oracle ⟨valid, false_⟩
    rcases falseAcceptance_hits relation ajtai _ _ oracle _ _ false_ with
      ⟨index, member⟩ | member | ⟨index, member⟩
    · exact Or.inl ⟨index, _, extends_point _ _ _, valid, member⟩
    · exact Or.inr (Or.inl ⟨_, extends_point _ _ _, valid, member⟩)
    · exact Or.inr (Or.inr ⟨index, _, extends_point _ _ _, valid, member⟩)
  · intro challenge
    have escape := escape_le (worstBad relation ajtai adversary claim challenge)
      (worstBad_local relation ajtai adversary claim challenge) (error challenge)
      (worstBad_mass_le relation ajtai adversary claim challenge) (error_nonnegative challenge)
      (checked_bound relation adversary claim bounded)
    refine (Finset.expect_le_expect fun oracle _ => ?_).trans escape
    by_cases isBad : oracle (point (claimed relation adversary claim oracle).fresh
          (claimed relation adversary claim oracle).proof challenge) ∈
        worstBad relation ajtai adversary claim challenge
          (point (claimed relation adversary claim oracle).fresh
            (claimed relation adversary claim oracle).proof challenge) oracle
    · have queried : ∃ asked ∈ (checked relation adversary claim).queries oracle,
          oracle asked ∈ worstBad relation ajtai adversary claim challenge asked oracle :=
        ⟨_, mem_checked relation adversary claim oracle challenge, isBad⟩
      simp only [if_pos isBad, if_pos queried]
      norm_num
    · simp only [if_neg isBad]
      split <;> norm_num

/-! ## The binding reduction's outcomes -/

/-- A valid run whose witness fails the source relation of its own statement. -/
def SourceFails (oracle : Oracle) (retries : Retries) : Prop :=
  ∃ valid : Valid relation ajtai adversary claim oracle retries,
    ¬ SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (claimed relation adversary claim oracle).running
        (claimed relation adversary claim oracle).fresh)
      (extractedWitness relation ajtai adversary claim oracle retries valid.1 valid.2)

/-- A valid run with the fresh statement, the running statement and the
witness of a failing run is a false acceptance of that witness. -/
theorem falseFor_of_agree {oracle other : Oracle} {retries otherRetries : Retries}
    (fails : SourceFails relation ajtai adversary claim oracle retries)
    (valid : Valid relation ajtai adversary claim other otherRetries)
    (sameFresh : (claimed relation adversary claim other).fresh =
      (claimed relation adversary claim oracle).fresh)
    (sameRunning : (claimed relation adversary claim other).running =
      (claimed relation adversary claim oracle).running)
    (sameWitness : witnessOf relation ajtai adversary claim other otherRetries =
      witnessOf relation ajtai adversary claim oracle retries) :
    FalseFor relation ajtai adversary claim (oracle, retries) other := by
  obtain ⟨baseValid, invalid⟩ := fails
  have witnessEq : extractedWitness relation ajtai adversary claim other otherRetries valid.1 valid.2 =
      extractedWitness relation ajtai adversary claim oracle retries baseValid.1 baseValid.2 := by
    unfold witnessOf at sameWitness
    rw [dif_pos valid, dif_pos baseValid] at sameWitness
    exact Option.some.inj sameWitness
  refine ⟨baseValid, ?_, ?_, ?_⟩
  · have accepted := valid.1.1.1
    rw [sameRunning] at accepted
    exact accepted
  · have ambient := extracted_ambient relation ajtai adversary claim other otherRetries valid.1 valid.2
    dsimp only at ambient
    rw [sameRunning] at ambient
    exact witnessEq ▸ ambient
  · rw [sameFresh]
    exact invalid

/-- The worst-witness false acceptance after the run under `oracle`, per
successful retry from its fork context. -/
noncomputable def agreeChance (oracle : Oracle) : ℝ :=
  testChance relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle
      (worst relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle) /
    resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle

theorem agreeChance_nonnegative (oracle : Oracle) :
    0 ≤ agreeChance relation ajtai adversary claim oracle :=
  div_nonneg (testChance_nonnegative relation ajtai adversary claim _ _ _)
    (resampledSuccess_nonnegative relation ajtai adversary claim _ _)

/-- After a failing run, its successful retries from the fork context collide,
change the running statement, or falsely accept its witness. -/
theorem one_le_outcomes {oracle : Oracle} {retries : Retries}
    (fails : SourceFails relation ajtai adversary claim oracle retries) :
    1 ≤ retryChance relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle
          (Collides relation ajtai adversary claim oracle retries) +
        retryChance relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle
          (Moves relation adversary claim oracle) +
        agreeChance relation ajtai adversary claim oracle := by
  have positive := resampledSuccess_pos relation ajtai adversary claim fails.1
  unfold retryChance agreeChance
  rw [← add_div, ← add_div, le_div_iff₀ positive, one_mul]
  refine le_trans ?_ (add_le_add le_rfl (le_worst relation ajtai adversary claim
    (forkIndex relation adversary claim oracle) oracle (oracle, retries)))
  unfold resampledSuccess testChance
  rw [← Finset.expect_add_distrib, ← Finset.expect_add_distrib]
  refine Finset.expect_le_expect fun fresh _ => ?_
  rw [success_eq]
  refine (retryMass_le_three relation ajtai adversary claim _ ?_).trans
    (add_le_add le_rfl (retryMass_const_le relation ajtai adversary claim _ _))
  rintro otherRetries ⟨forked, valid⟩
  by_cases moves : Moves relation adversary claim oracle
      (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
        oracle fresh) otherRetries
  · exact Or.inr (Or.inl ⟨forked, valid, moves⟩)
  by_cases collides : Collides relation ajtai adversary claim oracle retries
      (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
        oracle fresh) otherRetries
  · exact Or.inl ⟨forked, valid, collides⟩
  · refine Or.inr (Or.inr ⟨forked, ?_⟩)
    have sameRunning : (claimed relation adversary claim
        (overlay (context relation adversary claim (forkIndex relation adversary claim oracle) oracle)
          oracle fresh)).running = (claimed relation adversary claim oracle).running :=
      Classical.byContradiction fun different => moves different
    exact falseFor_of_agree relation ajtai adversary claim fails valid
      (fresh_eq_of_fork relation adversary claim oracle fresh forked) sameRunning
      (Classical.byContradiction fun different => collides ⟨sameRunning, different⟩)

/-- The resampling identity at one fork index: averaging the worst-witness
test chance per success over the base runs gives a plain probability. -/
theorem resampled_le (index : Nat) :
    𝔼 oracle, testChance relation ajtai adversary claim index oracle
          (worst relation ajtai adversary claim index oracle) *
        success relation ajtai adversary claim index oracle /
        resampledSuccess relation ajtai adversary claim index oracle ≤
      𝔼 oracle, (if forkIndex relation adversary claim oracle = index ∧
        FalseFor relation ajtai adversary claim (worst relation ajtai adversary claim index oracle)
          oracle then (1 : ℝ) else 0) := by
  have determined (oracle fresh : Oracle) :
      testChance relation ajtai adversary claim index
          (overlay (context relation adversary claim index oracle) oracle fresh)
          (worst relation ajtai adversary claim index
            (overlay (context relation adversary claim index oracle) oracle fresh)) =
        testChance relation ajtai adversary claim index oracle
          (worst relation ajtai adversary claim index oracle) := by
    have agree := overlay_agree (context relation adversary claim index oracle) oracle fresh
    rw [worst_congr relation ajtai adversary claim agree,
      testChance_congr relation ajtai adversary claim agree]
  calc
    _ = 𝔼 oracle, (if resampledSuccess relation ajtai adversary claim index oracle = 0 then 0 else
          testChance relation ajtai adversary claim index oracle
            (worst relation ajtai adversary claim index oracle)) :=
      expect_div_resampled (context relation adversary claim index)
        (context_stopping relation adversary claim index) (success relation ajtai adversary claim index)
        (fun oracle => testChance relation ajtai adversary claim index oracle
          (worst relation ajtai adversary claim index oracle)) determined
    _ ≤ 𝔼 oracle, testChance relation ajtai adversary claim index oracle
          (worst relation ajtai adversary claim index oracle) :=
      Finset.expect_le_expect fun oracle _ => by
        split
        · exact testChance_nonnegative relation ajtai adversary claim index oracle _
        · exact le_rfl
    _ = 𝔼 oracle, 𝔼 fresh, (if forkIndex relation adversary claim
            (overlay (context relation adversary claim index oracle) oracle fresh) = index ∧
          FalseFor relation ajtai adversary claim (worst relation ajtai adversary claim index
              (overlay (context relation adversary claim index oracle) oracle fresh))
            (overlay (context relation adversary claim index oracle) oracle fresh)
          then (1 : ℝ) else 0) := by
      refine Finset.expect_congr rfl fun oracle _ => ?_
      unfold testChance
      refine Finset.expect_congr rfl fun fresh _ => ?_
      rw [worst_congr relation ajtai adversary claim
        (overlay_agree (context relation adversary claim index oracle) oracle fresh)]
    _ = _ := (expect_resample (context relation adversary claim index)
      (context_stopping relation adversary claim index) fun oracle =>
        if forkIndex relation adversary claim oracle = index ∧
          FalseFor relation ajtai adversary claim (worst relation ajtai adversary claim index oracle)
            oracle then (1 : ℝ) else 0).symm

/-- The fork index lies below the checked query bound. -/
theorem forkIndex_lt_bound {queries : Nat} (bounded : adversary.QueryBound queries) (oracle : Oracle) :
    forkIndex relation adversary claim oracle < queries + challenges.length :=
  (forkIndex_lt relation adversary claim oracle).trans_le
    ((checked_bound relation adversary claim bounded).queries_length_le oracle)

/-- A quantity at the run's fork index, spread over the possible fork
indices. -/
theorem sum_success {queries : Nat} (bounded : adversary.QueryBound queries)
    (factor : Nat → Oracle → ℝ) (oracle : Oracle) :
    retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) *
        factor (forkIndex relation adversary claim oracle) oracle /
        resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle =
      ∑ index ∈ Finset.range (queries + challenges.length),
        factor index oracle * success relation ajtai adversary claim index oracle /
          resampledSuccess relation ajtai adversary claim index oracle := by
  rw [Finset.sum_eq_single (forkIndex relation adversary claim oracle)]
  · rw [success, if_pos rfl, mul_comm (factor _ _)]
  · intro index _ different
    rw [success, if_neg (Ne.symm different), mul_zero, zero_div]
  · intro outside
    exact absurd (Finset.mem_range.mpr (forkIndex_lt_bound relation adversary claim bounded oracle))
      outside

/-- The binding reduction's work. After a valid run it reruns the adversary
from the fork context until a rerun succeeds at the same fork index, which
takes `1 / resampledSuccess` expected reruns. The expected total is at most
`Q + 74`. -/
theorem expected_reruns_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, retryMass relation ajtai adversary claim oracle (Valid relation ajtai adversary claim oracle) /
        resampledSuccess relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle ≤
      ((queries + challenges.length : Nat) : ℝ) := by
  have spread (oracle : Oracle) :=
    sum_success relation ajtai adversary claim bounded (fun _ _ => (1 : ℝ)) oracle
  simp only [mul_one] at spread
  simp only [spread]
  rw [Finset.expect_sum_comm]
  calc
    _ ≤ ∑ _index ∈ Finset.range (queries + challenges.length), (1 : ℝ) :=
      Finset.sum_le_sum fun index _ =>
        (expect_div_resampled (context relation adversary claim index)
          (context_stopping relation adversary claim index) (success relation ajtai adversary claim index)
          (fun _ => (1 : ℝ)) fun _ _ => rfl).le.trans
        ((Finset.expect_le_expect fun _ _ => by split <;> norm_num).trans
          (Finset.expect_const Finset.univ_nonempty (1 : ℝ)).le)
    _ = _ := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]

/-! ## Lemma 6 -/

/-- **Lemma 6.** The extracted witness fails the source relation of the base
statement with probability at most `(Q + 74) * testError`, plus the chance
that the binding reduction finds two different witnesses for one running
statement (`collisionChance`) or a changed running statement
(`runningChance`). -/
theorem source_error_le {queries : Nat} (bounded : adversary.QueryBound queries) :
    𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
        (if SourceFails relation ajtai adversary claim oracle retries then 1 else 0) ≤
      ((queries + challenges.length : Nat) : ℝ) * IndependentExecution.testError productionShape 8 +
        collisionChance relation ajtai adversary claim + runningChance relation ajtai adversary claim := by
  have pointwise (oracle : Oracle) (retries : Retries) :
      weight relation ajtai adversary claim oracle retries *
          (if SourceFails relation ajtai adversary claim oracle retries then 1 else 0) ≤
        weight relation ajtai adversary claim oracle retries *
            (if Valid relation ajtai adversary claim oracle retries then
              retryChance relation ajtai adversary claim (forkIndex relation adversary claim oracle)
                oracle (Collides relation ajtai adversary claim oracle retries) else 0) +
          weight relation ajtai adversary claim oracle retries *
            (if Valid relation ajtai adversary claim oracle retries then
              retryChance relation ajtai adversary claim (forkIndex relation adversary claim oracle)
                oracle (Moves relation adversary claim oracle) else 0) +
          weight relation ajtai adversary claim oracle retries *
            (if Valid relation ajtai adversary claim oracle retries then 1 else 0) *
            agreeChance relation ajtai adversary claim oracle := by
    have nonnegative := weight_nonnegative relation ajtai adversary claim oracle retries
    have collide := retryChance_nonnegative relation ajtai adversary claim
      (forkIndex relation adversary claim oracle) oracle
      (Collides relation ajtai adversary claim oracle retries)
    have move := retryChance_nonnegative relation ajtai adversary claim
      (forkIndex relation adversary claim oracle) oracle (Moves relation adversary claim oracle)
    have agree := agreeChance_nonnegative relation ajtai adversary claim oracle
    by_cases fails : SourceFails relation ajtai adversary claim oracle retries
    · simp only [if_pos fails, if_pos fails.1, mul_one]
      have key := mul_le_mul_of_nonneg_left (one_le_outcomes relation ajtai adversary claim fails)
        nonnegative
      rw [mul_one, mul_add, mul_add] at key
      exact key
    · rw [if_neg fails, mul_zero]
      by_cases valid : Valid relation ajtai adversary claim oracle retries
      · simp only [if_pos valid, mul_one]
        have first := mul_nonneg nonnegative collide
        have second := mul_nonneg nonnegative move
        have third := mul_nonneg nonnegative agree
        linarith
      · simp only [if_neg valid, mul_zero, zero_mul, add_zero, le_refl]
  have summed :
      𝔼 oracle, ∑ retries, weight relation ajtai adversary claim oracle retries *
          (if SourceFails relation ajtai adversary claim oracle retries then 1 else 0) ≤
        collisionChance relation ajtai adversary claim + runningChance relation ajtai adversary claim +
          𝔼 oracle, retryMass relation ajtai adversary claim oracle
              (Valid relation ajtai adversary claim oracle) *
            agreeChance relation ajtai adversary claim oracle := by
    refine (Finset.expect_le_expect fun oracle _ => Finset.sum_le_sum fun retries _ =>
      pointwise oracle retries).trans (le_of_eq ?_)
    simp only [Finset.sum_add_distrib, Finset.expect_add_distrib, ← Finset.sum_mul]
    rfl
  have collapse (oracle : Oracle) :
      ∑ index ∈ Finset.range (queries + challenges.length),
          (if forkIndex relation adversary claim oracle = index ∧
            FalseFor relation ajtai adversary claim (worst relation ajtai adversary claim index oracle)
              oracle then (1 : ℝ) else 0) ≤
        if FalseFor relation ajtai adversary claim
            (worst relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle)
            oracle then 1 else 0 := by
    rw [Finset.sum_eq_single (forkIndex relation adversary claim oracle)]
    · by_cases happens : FalseFor relation ajtai adversary claim
          (worst relation ajtai adversary claim (forkIndex relation adversary claim oracle) oracle)
          oracle <;> simp [happens]
    · intro index _ different
      exact if_neg fun ⟨same, _⟩ => different same.symm
    · intro outside
      exact absurd (Finset.mem_range.mpr
        (forkIndex_lt_bound relation adversary claim bounded oracle)) outside
  have agreement :
      𝔼 oracle, retryMass relation ajtai adversary claim oracle
            (Valid relation ajtai adversary claim oracle) *
          agreeChance relation ajtai adversary claim oracle ≤
        ((queries + challenges.length : Nat) : ℝ) * IndependentExecution.testError productionShape 8 := by
    simp only [agreeChance, ← mul_div_assoc, sum_success relation ajtai adversary claim bounded
      (fun index oracle => testChance relation ajtai adversary claim index oracle
        (worst relation ajtai adversary claim index oracle))]
    rw [Finset.expect_sum_comm]
    refine (Finset.sum_le_sum fun index _ => resampled_le relation ajtai adversary claim index).trans ?_
    rw [← Finset.expect_sum_comm]
    exact (Finset.expect_le_expect fun oracle _ => collapse oracle).trans
      (worst_error_le relation ajtai adversary claim bounded)
  linarith

end NightstreamFPrime.Lifecycle.RandomOracleUniqueness
