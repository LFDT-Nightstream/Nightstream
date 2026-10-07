import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
Owns the random-oracle model of the Fiat–Shamir knowledge proof: a uniformly
random function from a finite point type to a finite answer type, adaptive
oracle computations, and the escape bound.

Inputs: an oracle computation with a query bound, and a bad set of answers at
each point.

Outputs:
- `escape_le`: the probability that some queried point has its answer in its
  bad set is at most `Q * ε`;
- `pinned_le`: the same for one point of the output, at most `(Q + 1) * ε`;
- `repeat_le`, `retries_le`: a rewinding extractor that resamples one point
  until it succeeds again repeats the base answer's key with total chance at
  most `Q * ε`, after at most `Q` expected retries; each queried point is
  charged once (`expect_queried_le`, `mem_queries_update_iff`);
- `expected_bad_retries`: under independent retries (`retryWeight`), the
  expected number of bad retries is the sum of the per-coordinate chances.

Invariant: a bad set is `Local`. It may read the oracle anywhere except at its
own point, so the bad set of a later challenge may read earlier challenges.

The structure follows Ironwood's `OracleComp` and `xEscAtPoint_measure_le`
(zcash/ironwood 86e3c7026db8, Apache-2.0 or MIT), restated with finite
averages. Ironwood's bad sets do not read the oracle.

Does not own: which points a protocol queries, a claim that a concrete hash
behaves as this oracle, or quantum queries.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.RandomOracle

open scoped BigOperators

universe uPoint uAnswer uOutput

/-- An adaptive oracle algorithm: return an output, or query one point and
continue with its answer. -/
inductive OracleComp (Point : Type uPoint) (Answer : Type uAnswer) (Output : Type uOutput) :
    Type (max uPoint uAnswer uOutput) where
  | done (output : Output)
  | query (point : Point) (next : Answer → OracleComp Point Answer Output)

namespace OracleComp

variable {Point : Type uPoint} {Answer : Type uAnswer} {Output : Type uOutput}

/-- The output when `oracle` answers every query. -/
def run (oracle : Point → Answer) : OracleComp Point Answer Output → Output
  | done output => output
  | query point next => (next (oracle point)).run oracle

/-- The queried points, in order. -/
def queries (oracle : Point → Answer) : OracleComp Point Answer Output → List Point
  | done _ => []
  | query point next => point :: (next (oracle point)).queries oracle

/-- Every execution makes at most `bound` queries. -/
inductive QueryBound : OracleComp Point Answer Output → Nat → Prop
  | done (output : Output) (bound : Nat) : QueryBound (done output) bound
  | query (point : Point) (next : Answer → OracleComp Point Answer Output) (bound : Nat)
      (each : ∀ answer, QueryBound (next answer) bound) :
      QueryBound (query point next) (bound + 1)

/-- Run the computation, then query one point of its output, as a verifier
does. -/
def thenQuery (point : Output → Point) : OracleComp Point Answer Output →
    OracleComp Point Answer Output
  | done output => query (point output) fun _ => done output
  | query asked next => query asked fun answer => (next answer).thenQuery point

theorem run_thenQuery (point : Output → Point) (oracle : Point → Answer)
    (computation : OracleComp Point Answer Output) :
    (computation.thenQuery point).run oracle = computation.run oracle := by
  induction computation with
  | done => rfl
  | query asked next inductionHypothesis => exact inductionHypothesis (oracle asked)

theorem mem_queries_thenQuery (point : Output → Point) (oracle : Point → Answer)
    (computation : OracleComp Point Answer Output) :
    point (computation.run oracle) ∈ (computation.thenQuery point).queries oracle := by
  induction computation with
  | done => exact List.mem_cons_self
  | query asked next inductionHypothesis =>
      exact List.mem_cons_of_mem _ (inductionHypothesis (oracle asked))

theorem QueryBound.thenQuery {point : Output → Point}
    {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound) :
    (computation.thenQuery point).QueryBound (bound + 1) := by
  induction bounded with
  | done output bound => exact .query _ _ bound fun _ => .done output bound
  | query asked next bound _ inductionHypothesis =>
      exact .query asked _ (bound + 1) inductionHypothesis

/-- Query a list of points, then return `output`. -/
def queryAll (output : Output) : List Point → OracleComp Point Answer Output
  | [] => done output
  | point :: rest => query point fun _ => queryAll output rest

/-- Run the computation, then query every listed point of its output, as a
verifier does. -/
def thenQueries (points : Output → List Point) : OracleComp Point Answer Output →
    OracleComp Point Answer Output
  | done output => queryAll output (points output)
  | query asked next => query asked fun answer => (next answer).thenQueries points

private theorem run_queryAll (oracle : Point → Answer) (output : Output) (points : List Point) :
    (queryAll output points).run oracle = output := by
  induction points with
  | nil => rfl
  | cons point rest inductionHypothesis => exact inductionHypothesis

private theorem queries_queryAll (oracle : Point → Answer) (output : Output) (points : List Point) :
    (queryAll output points : OracleComp Point Answer Output).queries oracle = points := by
  induction points with
  | nil => rfl
  | cons point rest inductionHypothesis => exact congrArg (point :: ·) inductionHypothesis

theorem run_thenQueries (points : Output → List Point) (oracle : Point → Answer)
    (computation : OracleComp Point Answer Output) :
    (computation.thenQueries points).run oracle = computation.run oracle := by
  induction computation with
  | done output => exact run_queryAll oracle output (points output)
  | query asked next inductionHypothesis => exact inductionHypothesis (oracle asked)

theorem mem_queries_thenQueries (points : Output → List Point) (oracle : Point → Answer)
    (computation : OracleComp Point Answer Output) {point : Point}
    (member : point ∈ points (computation.run oracle)) :
    point ∈ (computation.thenQueries points).queries oracle := by
  induction computation with
  | done output =>
      change point ∈ (queryAll output (points output) : OracleComp Point Answer Output).queries oracle
      rw [queries_queryAll]
      exact member
  | query asked next inductionHypothesis =>
      exact List.mem_cons_of_mem _ (inductionHypothesis (oracle asked) member)

theorem QueryBound.thenQueries {points : Output → List Point} {count : Nat}
    (counted : ∀ output, (points output).length ≤ count)
    {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound) :
    (computation.thenQueries points).QueryBound (bound + count) := by
  induction bounded with
  | done output bound =>
      have listed : ∀ (rest : List Point) (extra : Nat), rest.length ≤ extra →
          (queryAll output rest : OracleComp Point Answer Output).QueryBound (bound + extra) := by
        intro rest
        induction rest with
        | nil => exact fun extra _ => .done output _
        | cons point rest inductionHypothesis =>
            intro extra short
            obtain ⟨smaller, rfl⟩ : ∃ smaller, extra = smaller + 1 :=
              ⟨extra - 1, by simp at short; omega⟩
            exact .query point _ (bound + smaller) fun _ =>
              inductionHypothesis smaller (by simpa using short)
      exact listed (points output) count (counted output)
  | query asked next bound _ inductionHypothesis =>
      rw [show bound + 1 + count = (bound + count) + 1 by omega]
      exact .query asked _ (bound + count) inductionHypothesis

theorem QueryBound.queries_length_le {computation : OracleComp Point Answer Output}
    {bound : Nat} (bounded : computation.QueryBound bound) (oracle : Point → Answer) :
    (computation.queries oracle).length ≤ bound := by
  induction bounded with
  | done => exact Nat.zero_le _
  | query asked next bound _ inductionHypothesis =>
      exact Nat.succ_le_succ (inductionHypothesis (oracle asked))

/-- Whether a point is queried does not depend on its own answer: the run is
the same until that point is first queried. -/
theorem mem_queries_update_iff [DecidableEq Point] (oracle : Point → Answer) (point : Point)
    (answer : Answer) (computation : OracleComp Point Answer Output) :
    point ∈ computation.queries (Function.update oracle point answer) ↔
      point ∈ computation.queries oracle := by
  induction computation with
  | done => simp [queries]
  | query asked next inductionHypothesis =>
      by_cases same : asked = point
      · subst same
        simp [queries]
      · simp only [queries, List.mem_cons, Function.update_of_ne same]
        rw [inductionHypothesis (oracle asked)]

end OracleComp

section Probability

variable {Point : Type uPoint} {Answer : Type uAnswer} {Output : Type uOutput}
  [Fintype Point] [DecidableEq Point] [Fintype Answer] [Nonempty Answer]

attribute [local instance low] Classical.propDecidable

/-- The uniform probability of a set of answers. -/
noncomputable def mass (answers : Set Answer) : ℝ :=
  𝔼 answer, if answer ∈ answers then 1 else 0

/-- A bad set reads the oracle only away from its own point. -/
def Local (bad : Point → (Point → Answer) → Set Answer) : Prop :=
  ∀ point oracle answer, bad point (Function.update oracle point answer) = bad point oracle

/-- Resampling one point of a uniform oracle leaves it uniform. -/
private theorem expect_update (point : Point) (value : (Point → Answer) → ℝ) :
    𝔼 oracle, value oracle = 𝔼 oracle, 𝔼 answer, value (Function.update oracle point answer) := by
  let swap : (Point → Answer) × Answer ≃ (Point → Answer) × Answer :=
    { toFun := fun pair => (Function.update pair.1 point pair.2, pair.1 point)
      invFun := fun pair => (Function.update pair.1 point pair.2, pair.1 point)
      left_inv := fun pair => by simp
      right_inv := fun pair => by simp }
  have sums : ∑ oracle, ∑ answer, value (Function.update oracle point answer) =
      Fintype.card Answer * ∑ oracle, value oracle := by
    rw [← Fintype.sum_prod_type', Fintype.sum_equiv swap
      (fun pair => value (Function.update pair.1 point pair.2)) (fun pair => value pair.1)
      (fun _ => rfl), Fintype.sum_prod_type, Finset.mul_sum]
    simp
  simp only [Finset.expect_eq_sum_div_card, Finset.card_univ, div_eq_mul_inv, ← Finset.sum_mul]
  rw [sums]
  have positive : (0 : ℝ) < Fintype.card Answer := by exact_mod_cast Fintype.card_pos
  field_simp

/-- Answers held fixed on `fixed`; elsewhere `oracle` answers. -/
private def over (fixed : Finset Point) (values oracle : Point → Answer) : Point → Answer :=
  fun point => if point ∈ fixed then values point else oracle point

omit [Fintype Point] [Fintype Answer] [Nonempty Answer] in
private theorem over_update {fixed : Finset Point} {point : Point} (outside : point ∉ fixed)
    (values oracle : Point → Answer) (answer : Answer) :
    over fixed values (Function.update oracle point answer) =
      over (insert point fixed) (Function.update values point answer) oracle := by
  funext other
  by_cases same : other = point
  · subst same
    simp [over, outside]
  · simp [over, same]

omit [Fintype Point] [Fintype Answer] [Nonempty Answer] in
private theorem update_over {fixed : Finset Point} {point : Point} (outside : point ∉ fixed)
    (values oracle : Point → Answer) (answer : Answer) :
    over fixed values (Function.update oracle point answer) =
      Function.update (over fixed values oracle) point answer := by
  funext other
  by_cases same : other = point
  · subst same
    simp [over, outside]
  · simp [over, same]

omit [Fintype Point] [Fintype Answer] [Nonempty Answer] in
private theorem over_empty (values oracle : Point → Answer) : over ∅ values oracle = oracle := by
  funext point
  simp [over]

/-- Some queried point outside `fixed` has its answer in its bad set. -/
private def Escapes (bad : Point → (Point → Answer) → Set Answer)
    (computation : OracleComp Point Answer Output) (fixed : Finset Point)
    (oracle : Point → Answer) : Prop :=
  ∃ point ∈ computation.queries oracle, point ∉ fixed ∧ oracle point ∈ bad point oracle

variable (bad : Point → (Point → Answer) → Set Answer) (local_ : Local bad) (ε : ℝ)
  (small : ∀ point oracle, mass (bad point oracle) ≤ ε) (nonnegative : 0 ≤ ε)

include local_ small nonnegative in
/-- The escape bound with the answers on `fixed` held at `values`. A fork
holds the answers it shares with the first run this way. -/
private theorem escape_le_over {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound) (fixed : Finset Point) (values : Point → Answer) :
    𝔼 oracle, (if Escapes bad computation fixed (over fixed values oracle) then (1 : ℝ) else 0) ≤
      bound * ε := by
  induction bounded generalizing fixed values with
  | done output bound =>
      simp only [Escapes, OracleComp.queries, List.not_mem_nil, false_and, exists_false,
        if_false, Finset.expect_const_zero]
      positivity
  | query point next bound _ inductionHypothesis =>
      by_cases inside : point ∈ fixed
      · have same (oracle : Point → Answer) :
            Escapes bad (.query point next) fixed (over fixed values oracle) ↔
              Escapes bad (next (values point)) fixed (over fixed values oracle) := by
          simp [Escapes, OracleComp.queries, over, inside]
        simp only [same]
        refine (inductionHypothesis (values point) fixed values).trans ?_
        push_cast
        linarith
      · let first (oracle : Point → Answer) : Prop :=
          over fixed values oracle point ∈ bad point (over fixed values oracle)
        let later (oracle : Point → Answer) : Prop :=
          Escapes bad (next (over fixed values oracle point)) (insert point fixed)
            (over fixed values oracle)
        have split (oracle : Point → Answer) :
            (if Escapes bad (.query point next) fixed (over fixed values oracle) then (1 : ℝ)
              else 0) ≤
              (if first oracle then 1 else 0) + (if later oracle then 1 else 0) := by
          have firstNonnegative : (0 : ℝ) ≤ if first oracle then 1 else 0 := by
            split <;> norm_num
          have laterNonnegative : (0 : ℝ) ≤ if later oracle then 1 else 0 := by
            split <;> norm_num
          by_cases escapes : Escapes bad (.query point next) fixed (over fixed values oracle)
          · rw [if_pos escapes]
            obtain ⟨other, member, outside, isBad⟩ := escapes
            by_cases same : other = point
            · rw [same] at isBad
              have holds : first oracle := isBad
              rw [if_pos holds]
              linarith
            · have tail : other ∈ (next (over fixed values oracle point)).queries
                  (over fixed values oracle) := by
                simpa [OracleComp.queries, same] using member
              have holds : later oracle :=
                ⟨other, tail, by simp [same, outside], isBad⟩
              rw [if_pos holds]
              linarith
          · rw [if_neg escapes]
            linarith
        have firstBound :
            𝔼 oracle, (if first oracle then (1 : ℝ) else 0) ≤ ε := by
          rw [expect_update point]
          simp only [first, update_over inside, Function.update_self, local_ point]
          refine (Finset.expect_le_expect fun oracle _ => small point
            (over fixed values oracle)).trans ?_
          exact (Finset.expect_const Finset.univ_nonempty ε).le
        have laterBound :
            𝔼 oracle, (if later oracle then (1 : ℝ) else 0) ≤ bound * ε := by
          rw [expect_update point]
          have reached (oracle : Point → Answer) (answer : Answer) :
              over (insert point fixed) (Function.update values point answer) oracle point =
                answer := by
            simp [over]
          simp only [later, over_update inside, reached]
          rw [Finset.expect_comm]
          refine (Finset.expect_le_expect fun answer _ => inductionHypothesis answer
            (insert point fixed) (Function.update values point answer)).trans ?_
          exact (Finset.expect_const Finset.univ_nonempty _).le
        calc
          _ ≤ 𝔼 oracle, ((if first oracle then (1 : ℝ) else 0) +
                (if later oracle then 1 else 0)) :=
            Finset.expect_le_expect fun oracle _ => split oracle
          _ = (𝔼 oracle, if first oracle then (1 : ℝ) else 0) +
                𝔼 oracle, if later oracle then (1 : ℝ) else 0 :=
            Finset.expect_add_distrib _ _ _
          _ ≤ ε + bound * ε := add_le_add firstBound laterBound
          _ = ((bound + 1 : Nat) : ℝ) * ε := by push_cast; ring

include local_ small nonnegative in
/-- Escape bound: some queried point has its answer in its bad set with
probability at most `bound * ε`. -/
theorem escape_le {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound) :
    𝔼 oracle, (if ∃ point ∈ computation.queries oracle, oracle point ∈ bad point oracle
      then (1 : ℝ) else 0) ≤ bound * ε := by
  have escape := escape_le_over bad local_ ε small nonnegative bounded ∅ (fun point => Classical.arbitrary _)
  simpa [Escapes, over_empty] using escape

include local_ small nonnegative in
/-- Pinned escape: one output point has its answer in its bad set with
probability at most `(bound + 1) * ε`. -/
theorem pinned_le {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound) (point : Output → Point) :
    𝔼 oracle, (if oracle (point (computation.run oracle)) ∈
      bad (point (computation.run oracle)) oracle then (1 : ℝ) else 0) ≤ (bound + 1) * ε := by
  have escape := escape_le bad local_ ε small nonnegative (bounded.thenQuery (point := point))
  refine (Finset.expect_le_expect fun oracle _ => ?_).trans (by exact_mod_cast escape)
  by_cases isBad : oracle (point (computation.run oracle)) ∈
      bad (point (computation.run oracle)) oracle
  · have queried : ∃ asked ∈ (computation.thenQuery point).queries oracle,
        oracle asked ∈ bad asked oracle :=
      ⟨_, computation.mem_queries_thenQuery point oracle, isBad⟩
    simp only [if_pos isBad, if_pos queried]
    norm_num
  · simp only [if_neg isBad]
    split <;> norm_num

omit [Nonempty Answer] in
theorem mass_mono {small large : Set Answer} (inside : small ⊆ large) :
    mass small ≤ mass large :=
  Finset.expect_le_expect fun answer _ => by
    by_cases member : answer ∈ small
    · simp [member, inside member]
    · simp only [member, if_false]
      split <;> norm_num

omit [Nonempty Answer] in
theorem mass_nonnegative (answers : Set Answer) : 0 ≤ mass answers :=
  Finset.expect_nonneg fun answer _ => by split <;> norm_num

omit [Nonempty Answer] in
/-- The mean over all answers of a constant on `answers` and zero elsewhere. -/
private theorem expect_indicator (answers : Set Answer) (value : ℝ) :
    𝔼 answer, (if answer ∈ answers then value else 0) = value * mass answers := by
  unfold mass
  rw [Finset.mul_expect]
  exact Finset.expect_congr rfl fun answer _ => by split <;> simp

/-- The expected number of distinct queried points is at most the query
bound. -/
theorem expect_queried_le {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound) :
    𝔼 oracle, (∑ point, if point ∈ computation.queries oracle then (1 : ℝ) else 0) ≤ bound := by
  refine (Finset.expect_le_expect fun oracle _ => ?_).trans
    (Finset.expect_const Finset.univ_nonempty (bound : ℝ)).le
  have count : (∑ point, if point ∈ computation.queries oracle then (1 : ℝ) else 0) =
      ((computation.queries oracle).toFinset.card : ℝ) := by
    rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
    congr 2
    ext point
    simp
  rw [count]
  exact_mod_cast (List.toFinset_card_le _).trans (bounded.queries_length_le oracle)

/-! ## Line retries

A rewinding extractor resamples the answer at one point and reruns the
adversary until the point succeeds again. The first success is uniform on the
point's line set. The two bounds below charge each queried point once. -/

section LineRetry

variable (success : Point → (Point → Answer) → Prop)

/-- The answers at `point` under which `point` succeeds, the rest of the
oracle unchanged. -/
def lineSet (point : Point) (oracle : Point → Answer) : Set Answer :=
  {answer | success point (Function.update oracle point answer)}

omit [Fintype Point] [Fintype Answer] [Nonempty Answer] in
theorem lineSet_update (point : Point) (oracle : Point → Answer) (answer : Answer) :
    lineSet success point (Function.update oracle point answer) = lineSet success point oracle := by
  ext candidate
  simp [lineSet]

/-- Probability that the first successful retry has the base answer's key. -/
noncomputable def repeatChance {Key : Type*} (key : Answer → Key) (point : Point)
    (oracle : Point → Answer) : ℝ :=
  mass (lineSet success point oracle ∩ {answer | key answer = key (oracle point)}) /
    mass (lineSet success point oracle)

/-- Expected number of retries until the first success. -/
noncomputable def retryCount (point : Point) (oracle : Point → Answer) : ℝ :=
  1 / mass (lineSet success point oracle)

omit [Fintype Point] [Nonempty Answer] in
/-- A line with positive mass has a success, so its point is queried. -/
private theorem queried_of_mass {computation : OracleComp Point Answer Output}
    (queried : ∀ point oracle, success point oracle → point ∈ computation.queries oracle)
    (point : Point) (oracle : Point → Answer) (positive : mass (lineSet success point oracle) ≠ 0) :
    point ∈ computation.queries oracle := by
  by_contra absent
  apply positive
  unfold mass
  refine Finset.expect_eq_zero fun answer _ => if_neg fun inside => absent ?_
  exact (OracleComp.mem_queries_update_iff oracle point answer computation).mp
    (queried point _ inside)

/-- Charge one point: average its answer, then bound the line term by the
point's query indicator. -/
private theorem line_term_le {computation : OracleComp Point Answer Output}
    (queried : ∀ point oracle, success point oracle → point ∈ computation.queries oracle)
    (point : Point) (term : Set Answer → Answer → ℝ) (bound : ℝ)
    (local_ : ∀ set answer, answer ∈ set → term set answer ≤ bound / mass set)
    (nonnegative : 0 ≤ bound) :
    𝔼 oracle, (if success point oracle then
        term (lineSet success point oracle) (oracle point) else 0) ≤
      bound * 𝔼 oracle, (if point ∈ computation.queries oracle then (1 : ℝ) else 0) := by
  rw [expect_update point]
  rw [Finset.mul_expect]
  refine Finset.expect_le_expect fun oracle _ => ?_
  simp only [Function.update_self, lineSet_update]
  have rewrite (answer : Answer) :
      success point (Function.update oracle point answer) ↔
        answer ∈ lineSet success point oracle := Iff.rfl
  simp only [rewrite]
  calc
    _ ≤ 𝔼 answer, (if answer ∈ lineSet success point oracle then
          bound / mass (lineSet success point oracle) else 0) :=
      Finset.expect_le_expect fun answer _ => by
        by_cases inside : answer ∈ lineSet success point oracle
        · simp only [if_pos inside]
          exact local_ _ answer inside
        · simp only [if_neg inside, le_refl]
    _ = bound / mass (lineSet success point oracle) * mass (lineSet success point oracle) :=
      expect_indicator _ _
    _ ≤ bound * (if point ∈ computation.queries oracle then 1 else 0) := by
      by_cases positive : mass (lineSet success point oracle) = 0
      · rw [positive, mul_zero]
        split <;> nlinarith
      · rw [div_mul_cancel₀ _ positive, if_pos (queried_of_mass success queried point oracle positive),
          mul_one]

/-- The extractor's repeated-key loss: summed over every point that succeeds,
the chance that its first retry repeats the base key is at most
`bound * ε`. -/
theorem repeat_le {Key : Type*} (key : Answer → Key) (ε : ℝ)
    (small : ∀ value : Key, mass {answer | key answer = value} ≤ ε) (nonnegative : 0 ≤ ε)
    {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound)
    (queried : ∀ point oracle, success point oracle → point ∈ computation.queries oracle) :
    𝔼 oracle, (∑ point, if success point oracle then repeatChance success key point oracle else 0) ≤
      bound * ε := by
  rw [Finset.expect_sum_comm]
  calc
    _ ≤ ∑ point, ε * 𝔼 oracle, (if point ∈ computation.queries oracle then (1 : ℝ) else 0) :=
      Finset.sum_le_sum fun point _ => by
        refine le_trans (le_of_eq ?_) (line_term_le success queried point
          (fun set answer => mass (set ∩ {other | key other = key answer}) / mass set) ε
          (fun set answer _ => div_le_div_of_nonneg_right
            ((mass_mono Set.inter_subset_right).trans (small (key answer)))
            (mass_nonnegative set)) nonnegative)
        rfl
    _ = ε * 𝔼 oracle, (∑ point, if point ∈ computation.queries oracle then (1 : ℝ) else 0) := by
      rw [Finset.expect_sum_comm, Finset.mul_sum]
    _ ≤ ε * bound := mul_le_mul_of_nonneg_left (expect_queried_le bounded) nonnegative
    _ = bound * ε := mul_comm _ _

/-- The extractor's expected retries: summed over every point that succeeds,
the expected number of retries until the next success is at most `bound`. -/
theorem retries_le {computation : OracleComp Point Answer Output} {bound : Nat}
    (bounded : computation.QueryBound bound)
    (queried : ∀ point oracle, success point oracle → point ∈ computation.queries oracle) :
    𝔼 oracle, (∑ point, if success point oracle then retryCount success point oracle else 0) ≤
      bound := by
  rw [Finset.expect_sum_comm]
  calc
    _ ≤ ∑ point, 1 * 𝔼 oracle, (if point ∈ computation.queries oracle then (1 : ℝ) else 0) :=
      Finset.sum_le_sum fun point _ =>
        line_term_le success queried point (fun set _ => 1 / mass set) 1
          (fun _ _ _ => le_rfl) zero_le_one
    _ ≤ bound := by
      simp only [one_mul]
      rw [← Finset.expect_sum_comm]
      exact expect_queried_le bounded

omit [Fintype Point] [DecidableEq Point] [Nonempty Answer] in
private theorem sum_weight (set : Set Answer) (positive : mass set ≠ 0) (extra : Set Answer) :
    ∑ answer, (if answer ∈ set then (1 : ℝ) else 0) / (Fintype.card Answer * mass set) *
        (if answer ∈ extra then 1 else 0) = mass (set ∩ extra) / mass set := by
  have nonempty : Nonempty Answer := by
    by_contra empty
    apply positive
    have none : (Finset.univ : Finset Answer) = ∅ :=
      Finset.univ_eq_empty_iff.mpr (not_nonempty_iff.mp empty)
    simp [mass, none]
  have card : (0 : ℝ) < Fintype.card Answer := by exact_mod_cast Fintype.card_pos
  have joint : mass (set ∩ extra) =
      (∑ answer, (if answer ∈ set then (1 : ℝ) else 0) * (if answer ∈ extra then 1 else 0)) /
        Fintype.card Answer := by
    rw [mass, Finset.expect_eq_sum_div_card, Finset.card_univ]
    congr 1
    exact Finset.sum_congr rfl fun answer _ => by
      by_cases left : answer ∈ set <;> by_cases right : answer ∈ extra <;> simp [left, right]
  have pull : ∑ answer, (if answer ∈ set then (1 : ℝ) else 0) / (Fintype.card Answer * mass set) *
        (if answer ∈ extra then 1 else 0) =
      (∑ answer, (if answer ∈ set then (1 : ℝ) else 0) * (if answer ∈ extra then 1 else 0)) *
        (Fintype.card Answer * mass set)⁻¹ := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun _ _ => by ring
  rw [joint, pull]
  field_simp

/-- Independent first successes: coordinate `index` is uniform on
`sets index`, as the retry loop returns it. -/
noncomputable def retryWeight {Index : Type*} [Fintype Index] (sets : Index → Set Answer)
    (retries : Index → Answer) : ℝ :=
  ∏ index, (if retries index ∈ sets index then (1 : ℝ) else 0) /
    (Fintype.card Answer * mass (sets index))

omit [Fintype Point] [DecidableEq Point] [Nonempty Answer] in
theorem retryWeight_nonnegative {Index : Type*} [Fintype Index] (sets : Index → Set Answer)
    (retries : Index → Answer) : 0 ≤ retryWeight sets retries :=
  Finset.prod_nonneg fun _ _ => div_nonneg (by split <;> norm_num)
    (mul_nonneg (Nat.cast_nonneg _) (mass_nonnegative _))

omit [Fintype Point] [DecidableEq Point] [Nonempty Answer] in
/-- Independent retries: the expected number of retries that land in their
bad sets is the sum of the per-coordinate chances. -/
theorem expected_bad_retries {Index : Type*} [Fintype Index] [DecidableEq Index]
    (sets : Index → Set Answer) (positive : ∀ index, mass (sets index) ≠ 0)
    (bad : Index → Set Answer) :
    ∑ retries : Index → Answer, retryWeight sets retries *
        (∑ index, if retries index ∈ bad index then (1 : ℝ) else 0) =
      ∑ index, mass (sets index ∩ bad index) / mass (sets index) := by
  calc
    _ = ∑ index, ∑ retries : Index → Answer, retryWeight sets retries *
          (if retries index ∈ bad index then (1 : ℝ) else 0) := by
      simp_rw [Finset.mul_sum]
      exact Finset.sum_comm
    _ = ∑ index, ∑ retries : Index → Answer, ∏ coordinate,
          (if retries coordinate ∈ sets coordinate then (1 : ℝ) else 0) /
              (Fintype.card Answer * mass (sets coordinate)) *
            (if coordinate = index then
              (if retries coordinate ∈ bad coordinate then 1 else 0) else 1) := by
      refine Finset.sum_congr rfl fun index _ => Finset.sum_congr rfl fun retries _ => ?_
      simp only [Finset.prod_mul_distrib, Finset.prod_ite_eq', Finset.mem_univ, if_true,
        retryWeight]
    _ = ∑ index, ∏ coordinate, ∑ answer : Answer,
          (if answer ∈ sets coordinate then (1 : ℝ) else 0) /
              (Fintype.card Answer * mass (sets coordinate)) *
            (if coordinate = index then (if answer ∈ bad coordinate then 1 else 0) else 1) := by
      refine Finset.sum_congr rfl fun index _ => ?_
      rw [Fintype.prod_sum]
    _ = ∑ index, mass (sets index ∩ bad index) / mass (sets index) := by
      refine Finset.sum_congr rfl fun index _ => ?_
      rw [Finset.prod_eq_single index]
      · simpa using sum_weight (sets index) (positive index) (bad index)
      · intro coordinate _ different
        simpa [different, mul_one] using
          (sum_weight (sets coordinate) (positive coordinate) Set.univ).trans
            (by simp [div_self (positive coordinate)])
      · simp

end LineRetry

end Probability

end NightstreamFPrime.Spec.RandomOracle
