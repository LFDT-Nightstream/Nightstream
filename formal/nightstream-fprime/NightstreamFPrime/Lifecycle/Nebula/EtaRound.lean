import NightstreamFPrime.Lifecycle.Nebula.EndToEnd
import NightstreamFPrime.Lifecycle.Nebula.EtaOracle
import NightstreamFPrime.Lifecycle.Nebula.MemoryBound
import NightstreamFPrime.Spec.Nebula.FieldFingerprint
import NightstreamFPrime.Spec.RandomOracle
import NightstreamFPrime.Spec.Nebula.SetupCheck

/-! The memory round in the random-oracle model of `η` (CW-10.8). A computation with `Q` queries
to an `η` oracle returns a sample. `unbalanced_le` bounds the chance over a uniform oracle that
the sample is unbalanced with the oracle's challenges (`UnbalancedIn`) by
`(Q + 1) · 2 · m_mem / q²` plus the retry term `retryChance`.

The first segment of the sample's run with unbalanced multisets (`first`) and its oracle point
(`chosen`) read the sample only. An unbalanced sample hits that point (`Hit`): the segment
passes the close checks with the oracle's answer there (`hit_of_unbalanced`). The point is then
a query of the computation followed by one query at `chosen` (`hit_queried`). For one point, the
answers that hit it are bounded by the fingerprint bound of one segment
(`FieldFingerprint.field_segment_fingerprint_bound`), because the point fixes the
segment's `η` input (`EtaOracle.inputOf_pointOf`). Two answers whose runs hit the point with
different records give a Poseidon2 collision between the two segments' chain inputs
(`retry_collision`), so `retryChance` is a collision chance that Lean does not bound. The sum
over points is the general union bound `union_le`. At the Poseidon2 oracle, the model's event is
the deployed event (`unbalancedIn_deployed`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nebula.EtaRound

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Lifecycle.Nebula.EndToEnd
open NightstreamFPrime.Lifecycle.Nebula.EtaOracle
open scoped BigOperators
open scoped NightstreamFPrime.Spec.FieldExtensionRing
open Classical

section General

variable {Point Answer Output R : Type} [Fintype Point] [DecidableEq Point] [Fintype Answer]
  [Nonempty Answer]

omit [Nonempty Answer] in
/-- A set of answers has mass at most the share of calls of `play` that err, when each of its
answers errs as the call `(answer, ())`. -/
theorem mass_le_errors (line : Set Answer) (play : Answer × Unit → Option R) (bad : R → Prop)
    (errs : ∀ answer ∈ line, ∃ r, play (answer, ()) = some r ∧ bad r) :
    mass line ≤ ((errors play bad).card : ℝ) / Fintype.card (Answer × Unit) := by
  rw [mass, Finset.expect_eq_sum_div_card, Finset.sum_boole, Finset.card_univ,
    Fintype.card_prod, Fintype.card_unit, Nat.mul_one]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  refine Nat.cast_le.mpr (Finset.card_le_card_of_injOn (fun answer => (answer, ()))
    (fun answer inside => ?_) fun _ _ _ _ same => congrArg Prod.fst same)
  exact Finset.mem_filter.mpr ⟨Finset.mem_univ _,
    errs answer (Finset.mem_filter.mp (Finset.mem_coe.mp inside)).2⟩

/-- The union bound over oracle points. If each oracle of `event` has a point of `success`, the
computation queries each point of `success`, and at each point the answers of `success` have mass
at most `ε` plus `extra`, then `event` has chance at most `bound · ε` plus the expected `extra`
over all points. -/
theorem union_le {computation : OracleComp Point Answer Output} {bound : ℕ}
    (bounded : computation.QueryBound bound) (event : (Point → Answer) → Prop)
    (success : Point → (Point → Answer) → Prop)
    (covers : ∀ oracle, event oracle → ∃ point, success point oracle)
    (queried : ∀ point oracle, success point oracle → point ∈ computation.queries oracle)
    (ε : ℝ) (extra : Point → (Point → Answer) → ℝ) (nonnegative : 0 ≤ ε)
    (extraNonnegative : ∀ point oracle, 0 ≤ extra point oracle)
    (line : ∀ point oracle, mass (lineSet success point oracle) ≤ ε + extra point oracle) :
    𝔼 oracle, (if event oracle then (1 : ℝ) else 0) ≤
      bound * ε + ∑ point, 𝔼 oracle, extra point oracle := by
  have indicator (point : Point) (oracle : Point → Answer) :
      mass (lineSet success point oracle) ≤
        (if point ∈ computation.queries oracle then ε else 0) + extra point oracle := by
    by_cases zero : mass (lineSet success point oracle) = 0
    · rw [zero]
      have := extraNonnegative point oracle
      split <;> linarith
    · rw [ite_eq_left (queried_of_mass success queried point oracle zero)]
      exact line point oracle
  calc 𝔼 oracle, (if event oracle then (1 : ℝ) else 0)
      ≤ 𝔼 oracle, ∑ point, (if success point oracle then (1 : ℝ) else 0) := by
        refine Finset.expect_le_expect fun oracle _ => ?_
        by_cases happens : event oracle
        · obtain ⟨point, hit⟩ := covers oracle happens
          have single := Finset.single_le_sum
            (f := fun point => if success point oracle then (1 : ℝ) else 0)
            (fun _ _ => by split <;> norm_num) (Finset.mem_univ point)
          rw [ite_eq_left hit] at single
          rw [ite_eq_left happens]
          exact single
        · rw [ite_eq_right happens]
          exact Finset.sum_nonneg fun _ _ => by split <;> norm_num
    _ = ∑ point, 𝔼 oracle, mass (lineSet success point oracle) := by
        rw [Finset.expect_sum_comm]
        refine Finset.sum_congr rfl fun point _ => ?_
        rw [expect_update point]
        refine Finset.expect_congr rfl fun oracle _ => ?_
        unfold mass lineSet
        rfl
    _ ≤ ∑ point, 𝔼 oracle,
          ((if point ∈ computation.queries oracle then ε else 0) + extra point oracle) :=
        Finset.sum_le_sum fun point _ =>
          Finset.expect_le_expect fun oracle _ => indicator point oracle
    _ = ε * 𝔼 oracle, (∑ point, if point ∈ computation.queries oracle then (1 : ℝ) else 0) +
          ∑ point, 𝔼 oracle, extra point oracle := by
        simp only [Finset.expect_add_distrib, Finset.sum_add_distrib, Finset.mul_expect,
          Finset.mul_sum]
        rw [Finset.expect_sum_comm]
        congr 1
        refine Finset.sum_congr rfl fun point _ => Finset.expect_congr rfl fun oracle _ => ?_
        split <;> simp
    _ ≤ bound * ε + ∑ point, 𝔼 oracle, extra point oracle := by
        rw [mul_comm (bound : ℝ) ε]
        gcongr
        exact expect_queried_le bounded

end General

variable {p : Plan} (app : PortApp p.toGeometry) {Sample : Type} (x : Experiment app.width Sample)

/-- Segment `k` of the sample's extracted run. -/
def view (s : Sample) (k : ℕ) : SegmentView Digest := segmentView (context p) (x.run app s) k

/-- The first claimed segment of the sample with unbalanced multisets, or `0` if there is none. -/
noncomputable def first (s : Sample) : ℕ :=
  if h : ∃ k, k < (x.memory s).segments ∧ ¬ ((view app x s k).multisets p.toGeometry).Balanced
  then Nat.find h else 0

/-- The oracle point of the first unbalanced segment. -/
noncomputable def chosen (s : Sample) : Point :=
  pointOf ((view app x s (first app x s)).etaInput (context p))

/-- The run at `oracle` hits `point`: its first unbalanced segment has `η` input at `point` and
passes the close checks with the oracle's answer there. -/
def Hit (computation : OracleComp Point (K × K) Sample) (point : Point) (oracle : Oracle) :
    Prop :=
  pointOf ((view app x (computation.run oracle) (first app x (computation.run oracle))).etaInput
      (context p)) = point ∧
    (view app x (computation.run oracle) (first app x (computation.run oracle))).ClosesAt
      (context p) (oracle point) ∧
    ¬ ((view app x (computation.run oracle) (first app x (computation.run oracle))).multisets
      p.toGeometry).Balanced

variable {app x}

/-- At the deployed oracle, the model's event is the deployed event. -/
theorem unbalancedIn_deployed (s : Sample) :
    x.UnbalancedIn app (etaOf deployed) s ↔ x.Unbalanced app s := by
  rw [etaOf_deployed p]
  rfl

/-- An unbalanced sample hits its chosen point. -/
theorem hit_of_unbalanced (valid : p.Valid) (computation : OracleComp Point (K × K) Sample)
    (oracle : Oracle)
    (unbalanced : x.UnbalancedIn app (etaOf oracle) (computation.run oracle)) :
    Hit app x computation (chosen app x (computation.run oracle)) oracle := by
  obtain ⟨accepted, k, below, notBalanced, -⟩ := unbalanced
  have count := accepted.steps.trans accepted.stepCount
  obtain ⟨c, final, -⟩ := accepted.terminal
  have closes := ((finalCarry_isSome_iff (ctx := (context p).withEta (etaOf oracle)) valid
    count).1 ⟨c, final⟩).2
  have exists_ : ∃ k, k < (x.memory (computation.run oracle)).segments ∧
      ¬ ((view app x (computation.run oracle) k).multisets p.toGeometry).Balanced :=
    ⟨k, below, notBalanced⟩
  have firstEq : first app x (computation.run oracle) = Nat.find exists_ := by
    unfold first
    rw [dite_eq_left exists_]
  have spec := Nat.find_spec exists_
  rw [← firstEq] at spec
  have closesFirst := closes _ spec.1
  rw [segmentView_withEta, SegmentView.eta_withEta, SegmentView.closesAt_withEta] at closesFirst
  exact ⟨rfl, closesFirst, spec.2⟩

/-- A point that the run hits is a query of the computation followed by one query at `chosen`. -/
theorem hit_queried (computation : OracleComp Point (K × K) Sample) (point : Point)
    (oracle : Oracle) (hit : Hit app x computation point oracle) :
    point ∈ (computation.thenQuery (chosen app x)).queries oracle := by
  rw [← hit.1]
  exact OracleComp.mem_queries_thenQuery (chosen app x) oracle computation

variable (app x)

/-- The records of the first unbalanced segment of the run at `oracle`. -/
noncomputable def records (computation : OracleComp Point (K × K) Sample) (oracle : Oracle) :
    List StepRecords :=
  (view app x (computation.run oracle) (first app x (computation.run oracle))).records

/-- One call of the retry count at `point`: the computation runs with the answer `c.1` at
`point`, and returns the records of its first unbalanced segment when the run hits `point`. -/
noncomputable def play (computation : OracleComp Point (K × K) Sample) (point : Point)
    (oracle : Oracle) (c : (K × K) × Unit) : Option (List StepRecords) :=
  if Hit app x computation point (Function.update oracle point c.1) then
    some (records app x computation (Function.update oracle point c.1))
  else none

/-- Records whose multisets at the `η` input of `point` are not balanced. -/
def UnbalancedAt (point : Point) (z : List StepRecords) : Prop :=
  ¬ (((inputOf point).view 0 z).multisets (context p).plan.toGeometry).Balanced

/-- The retry term at `point`: the share of pairs of a first answer, drawn from all answers, and
a retry answer, drawn from the answers that hit `point`, whose runs both hit `point` with
different records. -/
noncomputable def retryChance (computation : OracleComp Point (K × K) Sample) (point : Point)
    (oracle : Oracle) : ℝ :=
  ((disagreements (play app x computation point oracle) (UnbalancedAt (p := p) point)).card : ℝ) /
    ((Fintype.card ((K × K) × Unit) : ℝ) * (successes (play app x computation point oracle)).card)

variable {app x}

theorem retryChance_nonnegative (computation : OracleComp Point (K × K) Sample) (point : Point)
    (oracle : Oracle) : 0 ≤ retryChance app x computation point oracle := by
  unfold retryChance
  positivity

/-- A call that returns records closes them at the `η` input of `point`, with the call's answer
as the challenges, and they are unbalanced. -/
theorem play_closes (valid : p.Valid) (computation : OracleComp Point (K × K) Sample)
    (point : Point) (oracle : Oracle) (c : (K × K) × Unit) (z : List StepRecords)
    (returns : play app x computation point oracle c = some z) :
    ((inputOf point).view 0 z).ClosesAt (context p) c.1 ∧ UnbalancedAt (p := p) point z := by
  unfold play at returns
  split_ifs at returns with hit
  simp only [Option.some.injEq] at returns
  subst returns
  obtain ⟨pointEq, closes, notBalanced⟩ := hit
  rw [Function.update_self] at closes
  have inputEq : inputOf point = (view app x (computation.run (Function.update oracle point c.1))
      (first app x (computation.run (Function.update oracle point c.1)))).etaInput (context p) :=
    (congrArg inputOf pointEq.symm).trans (inputOf_pointOf (closes.openTs_lt valid))
  unfold UnbalancedAt records
  rw [inputEq]
  exact ⟨⟨closes.length, closes.rows, closes.opsRoot, closes.initialRoot, closes.finalRoot,
    closes.products⟩, notBalanced⟩

/-- Every disagreeing pair of calls at `point` gives a Poseidon2 transcript collision between the
chain inputs of the two calls' segments, which have the same `η` input. -/
theorem retry_collision (valid : p.Valid) (computation : OracleComp Point (K × K) Sample)
    (point : Point) (oracle : Oracle) :
    ∀ pair ∈ disagreements (play app x computation point oracle) (UnbalancedAt (p := p) point),
      ∃ z z', play app x computation point oracle pair.1 = some z ∧
        play app x computation point oracle pair.2 = some z' ∧
        ∃ a ∈ ((inputOf point).view 0 z).chainInputs (context p),
          ∃ b ∈ ((inputOf point).view 0 z').chainInputs (context p),
            TranscriptCollision (blocks a) (blocks b) := by
  have chains : ∀ pair ∈ disagreements (play app x computation point oracle)
      (UnbalancedAt (p := p) point),
      ∃ z z', play app x computation point oracle pair.1 = some z ∧
        play app x computation point oracle pair.2 = some z' ∧
        CollisionIn (context p).hash (((inputOf point).view 0 z).chainInputs (context p))
          (((inputOf point).view 0 z').chainInputs (context p)) :=
    (FieldFingerprint.field_segment_fingerprint_bound (ctx := context p) valid
      (inputOf point) 0 (play app x computation point oracle)
      fun c z returns => (play_closes valid computation point oracle c z returns).1).2
  intro pair _
  obtain ⟨z, z', first, second, collision⟩ := chains pair ‹_›
  exact ⟨z, z', first, second, closing_collision_transcript valid
    (play_closes valid computation point oracle _ _ first).1
    (play_closes valid computation point oracle _ _ second).1 collision⟩

/-- The answers at `point` whose runs hit it have mass at most `2 · m_mem / q²` plus the retry
term at `point`. -/
theorem mass_le (valid : p.Valid) (computation : OracleComp Point (K × K) Sample)
    (point : Point) (oracle : Oracle) :
    mass (lineSet (Hit app x computation) point oracle) ≤
      2 * (p.maxTuples : ℝ) / (fieldModulus ^ 2 : ℕ) +
        retryChance app x computation point oracle := by
  have bound : ((errors (play app x computation point oracle) (UnbalancedAt (p := p) point)).card :
        ℚ≥0) / Fintype.card ((K × K) × Unit) ≤
      2 * (p.maxTuples : ℚ≥0) / (fieldModulus ^ 2 : ℕ) +
        ((disagreements (play app x computation point oracle)
            (UnbalancedAt (p := p) point)).card : ℚ≥0) /
          ((Fintype.card ((K × K) × Unit) : ℚ≥0) *
            (successes (play app x computation point oracle)).card) :=
    (FieldFingerprint.field_segment_fingerprint_bound (ctx := context p) valid
      (inputOf point) 0 (play app x computation point oracle)
      fun c z returns => (play_closes valid computation point oracle c z returns).1).1
  have real := (NNRat.cast_le (K := ℝ)).mpr bound
  push_cast at real
  unfold retryChance
  push_cast
  refine le_trans (mass_le_errors _ _ _ fun answer inLine => ?_) real
  have hit : Hit app x computation point (Function.update oracle point answer) := inLine
  have returns : play app x computation point oracle (answer, ()) =
      some (records app x computation (Function.update oracle point answer)) := by
    unfold play
    exact ite_eq_left hit
  exact ⟨_, returns, (play_closes valid computation point oracle _ _ returns).2⟩

/-- The memory round in the random-oracle model of `η` for one computation with `queries`
queries: over a uniform oracle, the chance that the sample is unbalanced with the oracle's
challenges is at most `(queries + 1) · 2 · m_mem / q²` plus the retry term summed over points. -/
theorem unbalanced_le (valid : p.Valid) {computation : OracleComp Point (K × K) Sample}
    {queries : ℕ} (bounded : computation.QueryBound queries) :
    𝔼 oracle : Oracle,
        (if x.UnbalancedIn app (etaOf oracle) (computation.run oracle) then (1 : ℝ) else 0) ≤
      ((queries + 1 : ℕ) : ℝ) * (2 * (p.maxTuples : ℝ) / (fieldModulus ^ 2 : ℕ)) +
        ∑ point, 𝔼 oracle : Oracle, retryChance app x computation point oracle :=
  union_le (bounded.thenQuery (point := chosen app x))
    (fun oracle => x.UnbalancedIn app (etaOf oracle) (computation.run oracle))
    (Hit app x computation)
    (fun oracle unbalanced => ⟨_, hit_of_unbalanced valid computation oracle unbalanced⟩)
    (hit_queried computation) _ _ (by positivity) (retryChance_nonnegative computation)
    (mass_le valid computation)

/-! ### Randomized samplers -/

/-- A randomized computation with `queries` queries to the `η` oracle: uniform coins of a finite
type, then the computation of each coin. The adversary and the extractor together are one such
sampler, so `queries` counts every query of every rerun. -/
structure EtaSampler (Sample : Type) (queries : ℕ) where
  Coins : Type
  [coinsFintype : Fintype Coins]
  [coinsNonempty : Nonempty Coins]
  run : Coins → OracleComp Point (K × K) Sample
  bounded : ∀ coin, (run coin).QueryBound queries

attribute [instance] EtaSampler.coinsFintype EtaSampler.coinsNonempty

namespace EtaSampler

variable (app x) {queries : ℕ} (sampler : EtaSampler Sample queries)

/-- The chance over uniform coins and a uniform oracle that the sample is unbalanced with the
oracle's challenges. -/
noncomputable def unbalancedChance : ℝ :=
  𝔼 coin, 𝔼 oracle : Oracle,
    if x.UnbalancedIn app (etaOf oracle) ((sampler.run coin).run oracle) then (1 : ℝ) else 0

/-- The retry term over uniform coins. Each pair that it counts is a Poseidon2 transcript
collision (`retry_collision`). -/
noncomputable def retryCollision : ℝ :=
  𝔼 coin, ∑ point, 𝔼 oracle : Oracle, retryChance app x (sampler.run coin) point oracle

variable {app x}

/-- `unbalanced_le` over the sampler's coins. -/
theorem unbalancedChance_le (valid : p.Valid) :
    sampler.unbalancedChance app x ≤
      ((queries + 1 : ℕ) : ℝ) * (2 * (p.maxTuples : ℝ) / (fieldModulus ^ 2 : ℕ)) +
        sampler.retryCollision app x := by
  unfold unbalancedChance retryCollision
  calc 𝔼 coin, 𝔼 oracle : Oracle,
        (if x.UnbalancedIn app (etaOf oracle) ((sampler.run coin).run oracle) then (1 : ℝ) else 0)
      ≤ 𝔼 coin, (((queries + 1 : ℕ) : ℝ) * (2 * (p.maxTuples : ℝ) / (fieldModulus ^ 2 : ℕ)) +
          ∑ point, 𝔼 oracle : Oracle, retryChance app x (sampler.run coin) point oracle) :=
        Finset.expect_le_expect fun coin _ => unbalanced_le valid (sampler.bounded coin)
    _ = _ := by rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty]

/-- For a secure geometry (`SPEC.md` section 7.1, rule 6), the first term is at most
`(queries + 1) · 2^−109.91` (`Geometry.secure_iff`). -/
theorem unbalancedChance_le_secure (valid : p.Valid) (secure : p.Secure) :
    sampler.unbalancedChance app x ≤
      ((queries + 1 : ℕ) : ℝ) * (2 : ℝ) ^ (-(10991 / 100 : ℝ)) + sampler.retryCollision app x := by
  have floor := (Geometry.secure_iff p.toGeometry).mp secure
  have term : 2 * (p.maxTuples : ℝ) / (fieldModulus ^ 2 : ℕ) ≤ (2 : ℝ) ^ (-(10991 / 100 : ℝ)) := by
    push_cast
    exact floor
  refine (sampler.unbalancedChance_le valid).trans ?_
  gcongr

end EtaSampler

end NightstreamFPrime.Lifecycle.Nebula.EtaRound
