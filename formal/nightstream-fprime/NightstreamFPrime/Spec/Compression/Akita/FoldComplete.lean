import NightstreamFPrime.Spec.Compression.Akita.FoldCheck
import NightstreamFPrime.Spec.Compression.Game.Complete

/-! The completeness of the checks `foldCheck` of one fold of Akita (ePrint 2026/1983, revision
of 2026-11-23, Theorem 10.9: "All honest algebraic checks are exact identities", with
Theorem 3.3 for the sum-checks and Theorem 7.4, item 1, for the ring reduction).

- `sumcheckWith_winsTo` is `Game.sumcheck_winsTo` for `sumcheckWith`.
- `treeDraws`, `rangeDraws` and `foldDraws` count the draws, and `treeWidth` and `rangeWidth` are
  the largest numbers of field elements in one message. `rangeTree_winsTo`, `leafStage_winsTo`
  and `rangeCheck_winsTo` are the completeness of the range tree of Akita Section 6.1.
- `foldCheck_winsTo` states that for a witness with the relation, the digit range statement and
  the binary statement, the prover has a message for each draw sequence, and `foldCheck` ends in
  the claim `(r_2, w̃(r_2))` after `foldDraws` draws.
- `foldCheck_complete` is this statement for the probability of a strategy.

The probability that the response of a fold is not admissible is not in this file: it is the
term `ξ_h` of Akita (155). -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] (fits : Type → Prop)

omit [Algebra Zq E] in
/-- `Game.sumcheck_winsTo` for the sum-check with a second bad set in each round. -/
theorem sumcheckWith_winsTo {d bound : ℕ} {truth : List E → E} {round : List E → E[X]}
    (true_rounds : RoundTruth d bound truth round) (message : fits (Fin (d + 1) → E))
    (extra : List E → Finset E) (rest : ℕ) (tail : Option (List E × E) → ℕ → Bool → Prop) :
    ∀ (rounds : ℕ) (point : List E) (claim : E), point.length + rounds ≤ bound →
      claim = truth point →
      (∀ final : List E, final.length = point.length + rounds →
        tail (some (final, truth final)) rest true) →
      WinsTo fits (sumcheckWith d round extra rounds point claim) (rest + rounds) true tail := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _ true_claim ended
      rw [true_claim]
      exact ended point rfl
  | succ rounds step =>
      intro point claim inside true_claim ended
      have within : point.length < bound := by omega
      refine ⟨rfl, message, fun index => (round point).coeff index, ?_⟩
      dsimp only
      rw [coeffPoly_coeff (round point) (true_rounds.degree point within),
        ite_eq_left (true_claim ▸ true_rounds.halves point within)]
      intro challenge
      show WinsTo fits (sumcheckWith d round extra rounds (point ++ [challenge])
        ((round point).eval challenge)) (rest + rounds) true tail
      rw [true_rounds.binds point challenge within]
      exact step (point ++ [challenge]) _ (by simp; omega) rfl fun final length =>
        ended final (by simp at length; omega)

/-- The completeness of the fused sum-check (125) with its final check (Akita Section 7.4): for
a true claim, the prover sends the true round polynomials and the value `w̃(r_2)`. -/
theorem fusedCheck_winsTo (scalar : fits E) (cubic : fits (Fin (3 + 1) → E)) (μ : ℕ)
    (support : ℕ → Prop) [DecidablePred support] (w : Table) (weight : ℕ → E) (target : E)
    (r : List E) (sClaim ζ γ : E) (rest : ℕ)
    (true_claim : target + γ * sClaim =
      combTruth μ (fusedTables w weight (virtWeight support r γ ζ)) (fusedComb E) []) :
    WinsTo fits (fusedCheck μ support w weight target r sClaim ζ γ) (rest + μ) true
      fun result _ _ => ∃ r2 : List E, r2.length = μ ∧ result = some (r2, mle r2 (lifted E w)) := by
  refine (winsTo_bind fits _ _ _ _ _).mpr (sumcheck_winsTo fits
    (fused_roundTruth μ w weight (virtWeight support r γ ζ)) cubic rest _ μ [] _ (by simp)
    true_claim fun final length => ?_)
  simp only [List.length_nil, zero_add] at length
  have last := fusedTruth_final final w weight (virtWeight support r γ ζ)
  rw [length] at last
  refine ⟨rfl, scalar, mle final (lifted E w), ?_⟩
  dsimp only
  rw [ite_eq_left last]
  exact ⟨final, length, rfl⟩

/-- The number of the draws of the range tree for the product degrees `arities` and a last stage
with `last` draws: `μ'` rounds and one batching challenge for each product level. -/
def treeDraws (μ last : ℕ) : List ℕ → ℕ
  | [] => last
  | _ :: rest => treeDraws μ last rest + 1 + μ

/-- The largest number of field elements in one message of the range tree before its last stage:
the coefficients of a round polynomial or the claims of the children of one level. -/
def treeWidth : List ℕ → ℕ → ℕ
  | [], _ => 0
  | a :: rest, nodes => max (a + 1 + 1) (max (nodes * a) (treeWidth rest (nodes * a)))

/-- The number of the draws of `rangeCheck`. -/
def rangeDraws (μ b : ℕ) : ℕ := treeDraws μ μ (rangeArities b)

/-- The largest number of field elements in one message of `rangeCheck`. -/
def rangeWidth (b : ℕ) : ℕ := max (rangeLeaf b + 1 + 1) (treeWidth (rangeArities b) 1)

/-- The number of the draws of `foldCheck`: `α`, `τ_0`, `τ_1`, the draws of the range tree,
`ζ_bin` at a level with compression digits, `γ` and the `μ'` rounds of the fused sum-check. -/
def foldDraws (μ b : ℕ) (bin : Bool) : ℕ := 0 + μ + 1 + binDraws bin + rangeDraws μ b + 1 + 1 + 1

omit [DecidableEq E] [Algebra Zq E] in
/-- The prover wins after `drawBin` when it wins for each coefficient `ζ_bin`. -/
theorem drawBin_winsTo {β : Type} (bin : Bool) {bad : Finset E} {next : E → Game β} {rest : ℕ}
    {fresh : Bool} {tail : β → ℕ → Bool → Prop}
    (each : ∀ ζ fresh, WinsTo fits (next ζ) rest fresh tail) :
    WinsTo fits (drawBin bin bad next) (rest + binDraws bin) fresh tail := by
  cases bin
  · exact each 0 fresh
  · exact fun ζ => each ζ true

omit [Fintype E] [DecidableEq E] [Algebra Zq E] in
theorem entry_of {n i : ℕ} (f : ℕ → E) (inside : i < n) :
    entry (fun j : Fin n => f j) i = f i :=
  dite_eq_left inside

/-- The completeness of the range tree: for a true first claim, the prover sends the true round
polynomials and the values of the child tables, and the last stage starts from a true claim.
`lastWins` is the completeness of the last stage. -/
theorem rangeTree_winsTo {β : Type} (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option β)) (lastDraws rest : ℕ)
    (tail : Option β → ℕ → Bool → Prop)
    (lastWins : ∀ nodes weights point claim, point.length = μ →
      claim = ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E leaf m w) →
      WinsTo fits (last nodes weights point claim) (rest + lastDraws) true tail) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      point.length = μ → (∀ n, n ≤ treeWidth arities nodes → fits (Fin n → E)) →
      claim = ∑ m ∈ Finset.range nodes,
        weights m * mle point (nodeTable E (arities.prod * leaf) m w) →
      WinsTo fits (rangeTree leaf μ w last arities nodes weights point claim)
        (rest + treeDraws μ lastDraws arities) true tail := by
  intro arities
  induction arities with
  | nil =>
      intro nodes weights point claim long _ true_claim
      rw [List.prod_nil, one_mul] at true_claim
      exact lastWins nodes weights point claim long true_claim
  | cons a others step =>
      intro nodes weights point claim long vector true_claim
      subst long
      rw [List.prod_cons, mul_assoc, ← stageTruth_nil] at true_claim
      rw [treeDraws, ← Nat.add_assoc, ← Nat.add_assoc]
      refine (winsTo_bind fits _ _ _ _ _).mpr (sumcheck_winsTo fits
        (stage_roundTruth a (others.prod * leaf) nodes point.length weights point w)
        (vector _ (le_max_left _ _)) _ _ point.length [] claim (by simp) true_claim
        fun final length => ?_)
      simp only [List.length_nil, zero_add] at length
      refine ⟨rfl, vector _ ((le_max_left _ _).trans (le_max_right _ _)),
        fun i => mle final (nodeTable E (others.prod * leaf) i w), ?_⟩
      have inside (m t : ℕ) (node : m ∈ Finset.range nodes) (place : t ∈ Finset.range a) :
          m * a + t < nodes * a :=
        Nat.lt_of_lt_of_le (Nat.add_lt_add_left (Finset.mem_range.mp place) _)
          (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right a (Finset.mem_range.mp node))
      have last := stageTruth_final a (others.prod * leaf) nodes weights point final w
      rw [length] at last
      dsimp only
      rw [ite_eq_left (last.trans (congrArg _ (Finset.sum_congr rfl fun m node => congrArg _
        (Finset.prod_congr rfl fun t place =>
          (entry_of (fun i => mle final (nodeTable E (others.prod * leaf) i w))
            (inside m t node place)).symm))))]
      intro γ
      refine step (nodes * a) (fun m => γ ^ m) final _ length
        (fun n small => vector n ((small.trans (le_max_right _ _)).trans (le_max_right _ _))) ?_
      rw [rowPoly_eval]
      exact Finset.sum_congr rfl fun m member => congrArg _
        (entry_of (fun i => mle final (nodeTable E (others.prod * leaf) i w))
          (Finset.mem_range.mp member))

/-- The completeness of the last stage of the range tree on the coefficient route: for a true
claim, the prover sends the true round polynomials and the value `s̃(r_virt)`. -/
theorem leafStage_winsTo (scalar : fits E) (leaf μ : ℕ)
    (round : fits (Fin (leaf + 1 + 1) → E)) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) (rest : ℕ) (tail : Option (List E × E) → ℕ → Bool → Prop)
    (ended : ∀ r : List E, r.length = μ → tail (some (r, mle r (derived E w))) rest false)
    (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E) (long : point.length = μ)
    (true_claim : claim = ∑ m ∈ Finset.range nodes,
      weights m * mle point (nodeTable E leaf m w)) :
    WinsTo fits (leafStage leaf μ support w nodes weights point claim) (rest + μ) true tail := by
  subst long
  rw [← leafTruth_nil] at true_claim
  refine (winsTo_bind fits _ _ _ _ _).mpr (sumcheckWith_winsTo fits
    (leaf_roundTruth leaf nodes point.length weights point w) round _ rest _ point.length []
    claim (by simp) true_claim fun final length => ?_)
  simp only [List.length_nil, zero_add] at length
  have last := leafTruth_final leaf nodes weights point final w
  rw [length] at last
  refine ⟨rfl, scalar, mle final (derived E w), ?_⟩
  dsimp only
  rw [ite_eq_left last]
  exact ended final length

/-- The completeness of `rangeCheck`: for a witness with the digit range statement, the prover
has a message for each draw sequence, and the range tree ends in `(r_virt, s̃(r_virt))`. -/
theorem rangeCheck_winsTo (scalar : fits E) (b μ : ℕ)
    (vector : ∀ n, n ≤ rangeWidth b → fits (Fin n → E)) (support : ℕ → Prop)
    [DecidablePred support] {w : Table} (ranged : RangeHolds b μ w) (τ0 : List E)
    (long : τ0.length = μ) (rest : ℕ) (tail : Option (List E × E) → ℕ → Bool → Prop)
    (ended : ∀ r : List E, r.length = μ → tail (some (r, mle r (derived E w))) rest false) :
    WinsTo fits (rangeCheck b μ support w τ0) (rest + rangeDraws μ b) true tail := by
  refine rangeTree_winsTo fits (rangeLeaf b) μ w _ μ rest tail
    (fun nodes weights point claim => leafStage_winsTo fits scalar (rangeLeaf b) μ
      (vector _ (le_max_left _ _)) support w rest tail ended nodes weights point claim)
    (rangeArities b) 1 (fun _ => 1) τ0 0 long
    (fun n small => vector n (small.trans (le_max_right _ _))) ?_
  rw [root_sum b w τ0]
  exact (mle_rangeResidual_eq_zero ranged τ0 long).symm

/-- The completeness of the checks of one fold (Akita Theorem 10.9 for the checks of Figure 6):
for a witness with the relation of the ring reduction, the digit range statement of Akita
Section 6.1 and the binary statement of Akita Remark 6.2, the prover has a message for each draw
sequence, and `foldCheck` ends in the claim `(r_2, w̃(r_2))` with `μ'` coordinates. The draws are
`α`, `τ_0`, `τ_1`, the draws of the range tree, `ζ_bin`, `γ` and the `μ'` rounds of the fused
sum-check. A message is one element of `E`, the coefficients of one round polynomial, or the
claims of the children of one level of the range tree. -/
theorem foldCheck_winsTo (R : Reduction E) (b : ℕ) (scalar : fits E)
    (range : ∀ n, n ≤ rangeWidth b → fits (Fin n → E)) (cubic : fits (Fin (3 + 1) → E))
    (support : ℕ → Prop) [DecidablePred support] {bin : Bool} {w : Table} (holds : R.Holds w)
    (ranged : RangeHolds b R.vars w) (binary : BinaryOn support R.vars w) :
    WinsTo fits (foldCheck R b support bin w) (foldDraws R.vars b bin) true
      fun result _ _ =>
        ∃ r2 : List E, r2.length = R.vars ∧ result = some (r2, mle r2 (lifted E w)) := by
  intro α τ0 τ1
  refine (winsTo_bind fits _ _ _ _ _).mpr (rangeCheck_winsTo fits scalar b R.vars range support
    ranged (List.ofFn τ0) List.length_ofFn (0 + R.vars + 1 + binDraws bin) _
    fun final length => ?_)
  dsimp only
  refine drawBin_winsTo (E := E) fits bin fun ζ fresh => ?_
  intro γ
  refine fusedCheck_winsTo fits scalar cubic R.vars support w _ _ final _ ζ γ 0 ?_
  have virt := fusedTruth_virt support w ((R.rows α).batchWeight (List.ofFn τ1)) final γ ζ
  rw [length] at virt
  have batched : (R.rows α).Batched R.vars w (List.ofFn τ1) := R.batched_of_holds holds α _
  have zero : mle final (supportTable support w) = (0 : E) := by
    rw [mle_congr final (u := fun _ => 0) fun x small => by
      unfold supportTable
      split
      · rename_i member
        rw [derived, binary x (length ▸ small) member, map_zero]
      · rfl]
    exact mle_const final 0
  rw [virt, zero, mul_zero, add_zero]
  exact congrArg (· + γ * mle final (derived E w)) batched.symm

/-- Perfect completeness of the checks of one fold: for a witness with the relation, the digit
range statement and the binary statement, a strategy of the prover reaches the claim
`(r_2, w̃(r_2))` with probability 1. -/
theorem foldCheck_complete (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) {w : Table} (holds : R.Holds w) (ranged : RangeHolds b
      R.vars w)
    (binary : BinaryOn support R.vars w) :
    ∃ strategy : Strategy (Option (List E × E)),
      prob strategy (foldCheck R b support bin w) (fun result =>
        ∃ r2 : List E, r2.length = R.vars ∧ result = some (r2, mle r2 (lifted E w))) = 1 :=
  ⟨_, prob_honest (fun _ => True)
    (foldCheck_winsTo (fun _ => True) R b trivial (fun _ _ => trivial) trivial support holds
      ranged binary)
    fun _ _ _ ended => ended⟩

end NightstreamFPrime.Spec.Compression.Akita
