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
- `pinned_le`: the same for one point of the output, at most `(Q + 1) * ε`.

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

end OracleComp

section Probability

variable {Point : Type uPoint} {Answer : Type uAnswer} {Output : Type uOutput}
  [Fintype Point] [DecidableEq Point] [Fintype Answer] [Nonempty Answer]

attribute [local instance] Classical.propDecidable

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

end Probability

end NightstreamFPrime.Spec.RandomOracle
