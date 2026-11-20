import Mathlib.Algebra.MvPolynomial.CommRing
import NightstreamFPrime.Spec.Compression.Lib.Combination
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Akita.Range

/-! The sum-checks of one fold of Akita (ePrint 2026/1983, revision of 2026-10-15, Section 3.4,
Theorem 3.3 and Figure 2, Section 6.1, Remark 6.2 and Section 7.4), as instances of the sum-check
of `Lib/SumCheck.lean`, `Lib/Combination.lean` and `Game/SumCheck.lean`.

- `round_eq_of_agree` is Akita Theorem 3.3, item 2, for one round: a round message that
  agrees with the true sums at `ℓ + 1` challenges is the true round polynomial.
  `check_of_agree` gives the claim of the round. Item 1 is `Game.sumcheck_avoids` with
  `Game.sumcheck_costLe`.
- `lifted` is the witness as a table over `E`, and `derived` is `s(x) = w(x)·(w(x) + 1)` of Akita
  Section 6.1. `rangeSqIn` is `Q_sq` of Akita (79) over `E`.
- `rangeComb` and `rangeTables` are the summand `ẽq(τ_0, x)·Q_sq(s(x))` of the range sum (80).
  `rangeTruth_nil` is the sum (80), `rangeTruth_final` is the final claim
  `ẽq(τ_0, r_virt)·Q_sq(s̃(r_virt))`, and `range_roundTruth` gives the true round polynomials, of
  degree at most `b⋆/2 + 1`.
- `fusedComb` and `fusedTables` are the summand of the fused sum (125), of degree 3
  (Akita Theorem 10.36). `virtWeight` is the weight
  `γ·ẽq(r_virt, x) + ζ_bin·ẽq_{I_bin}(r_virt, x)` and `supportTable` is the table
  `x ↦ 1_{I_bin}(x)·s(x)` of Akita Remark 6.2. `fusedTruth_virt` is the right side of (125),
  `fusedTruth_final` is the final claim at `r_2`, and `fused_roundTruth` gives the true round
  polynomials.
- `rangeSumcheck_avoids` and `fusedSumcheck_avoids` are Akita Theorem 3.3, item 1, for the two
  sum-checks, in the form of `Game.sumcheck_avoids`: an accepted false claim has a round with a
  challenge in its bad set. `rangeSumcheck_costLe` and `fusedSumcheck_costLe` bound the bad sets.
- `NonzeroOn`, `bindBad` and `nonzeroOn_bindLow` are the step of the binary-support point of Akita
  Lemma 10.37 (ii): a table that is nonzero on the cube stays nonzero when one variable is bound
  outside a set of at most one value.
- `sumcheckWith` is the sum-check of `Game.sumcheck` with a second bad set in each round.
  `sumcheckWith_avoids` and `sumcheckWith_costLe` are its soundness and its cost. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib Polynomial

section Special

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

/-- Akita Theorem 3.3, item 2, for one round: a message of degree at most `ℓ` that agrees with the
true sums of the continuations at `ℓ + 1` distinct challenges is the true round polynomial. -/
theorem round_eq_of_agree {d bound : ℕ} {truth : List K → K} {round : List K → K[X]}
    (true_rounds : RoundTruth d bound truth round) {point : List K} (inside : point.length < bound)
    (message : Fin (d + 1) → K) (challenges : Finset K) (many : d < challenges.card)
    (agree : ∀ c ∈ challenges, (coeffPoly message).eval c = truth (point ++ [c])) :
    coeffPoly message = round point := by
  by_contra different
  have roots : challenges ⊆ zeros (coeffPoly message - round point) := by
    intro c member
    by_contra outside
    exact eval_ne_of_not_mem_zeros different outside
      ((agree c member).trans (true_rounds.binds point c inside).symm)
  have card := (Finset.card_le_card roots).trans (zeros_card_le_of_natDegree_le
    ((natDegree_sub_le _ _).trans
      (max_le (coeffPoly_natDegree message) (true_rounds.degree point inside))))
  omega

/-- Akita Theorem 3.3, item 2: under the conditions of `round_eq_of_agree`, the claim that
the round check accepts is the true sum of the prefix. -/
theorem check_of_agree {d bound : ℕ} {truth : List K → K} {round : List K → K[X]}
    (true_rounds : RoundTruth d bound truth round) {point : List K} (inside : point.length < bound)
    (message : Fin (d + 1) → K) (challenges : Finset K) (many : d < challenges.card)
    (agree : ∀ c ∈ challenges, (coeffPoly message).eval c = truth (point ++ [c])) :
    (coeffPoly message).eval 0 + (coeffPoly message).eval 1 = truth point := by
  rw [round_eq_of_agree true_rounds inside message challenges many agree]
  exact true_rounds.halves point inside

end Special

section Tables

variable {E : Type} [Field E] [Algebra Zq E]

/-- The witness `w` as a table over `E`. -/
def lifted (E : Type) [Field E] [Algebra Zq E] (w : Table) (x : ℕ) : E := algebraMap Zq E (w x)

/-- `s(x) = w(x)·(w(x) + 1)` of Akita Section 6.1, in `E`. -/
def derived (E : Type) [Field E] [Algebra Zq E] (w : Table) (x : ℕ) : E :=
  algebraMap Zq E (w x * (w x + 1))

/-- `Q_sq(s) = Π_{j < b⋆/2} (s − j(j + 1))` of Akita (79), for `s ∈ E`. -/
def rangeSqIn (b : ℕ) (s : E) : E := ∏ j ∈ Finset.range (b / 2), (s - (j : E) * ((j : E) + 1))

theorem map_rangeSq (b : ℕ) (s : Zq) :
    algebraMap Zq E (rangeSq b s) = rangeSqIn b (algebraMap Zq E s) := by
  simp [rangeSq, rangeSqIn, map_prod]

theorem rangeResidual_eq (b : ℕ) (w : Table) (x : ℕ) :
    rangeResidual E b w x = rangeSqIn b (derived E w x) :=
  map_rangeSq b _

theorem derived_eq (w : Table) (x : ℕ) : derived E w x = lifted E w x * (lifted E w x + 1) := by
  simp [derived, lifted]

/-- The summand `ẽq(τ_0, x)·Q_sq(s(x))` of the range sum (80) as a polynomial in the table of
`ẽq(τ_0, ·)` (variable 0) and the table of `s` (variable 1). -/
noncomputable def rangeComb (E : Type) [Field E] (b : ℕ) : MvPolynomial (Fin 2) E :=
  MvPolynomial.X 0 * ∏ j ∈ Finset.range (b / 2),
    (MvPolynomial.X 1 - MvPolynomial.C ((j : E) * ((j : E) + 1)))

/-- The tables of the range sum (80): `ẽq(τ_0, ·)` and `s`. -/
def rangeTables (τ : List E) (w : Table) : Fin 2 → ℕ → E := ![eqAt τ, derived E w]

omit [Algebra Zq E] in
theorem eval_rangeComb (b : ℕ) (f : Fin 2 → E) :
    MvPolynomial.eval f (rangeComb E b) = f 0 * rangeSqIn b (f 1) := by
  simp [rangeComb, rangeSqIn, map_prod]

omit [Algebra Zq E] in
/-- The summand of the range sum (80) has the degree `b⋆/2 + 1`: the degree `d_1 = b⋆/2` of
`Q_sq` and one for the equality factor (Akita Lemma 10.37). -/
theorem rangeComb_degree (b : ℕ) : (rangeComb E b).totalDegree ≤ b / 2 + 1 := by
  unfold rangeComb
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_X, add_comm]
  refine Nat.add_le_add_right ((MvPolynomial.totalDegree_finsetProd _ _).trans ?_) 1
  calc ∑ j ∈ Finset.range (b / 2),
        (MvPolynomial.X 1 - MvPolynomial.C ((j : E) * ((j : E) + 1)) :
          MvPolynomial (Fin 2) E).totalDegree
      ≤ ∑ _j ∈ Finset.range (b / 2), 1 := Finset.sum_le_sum fun j _ =>
        (MvPolynomial.totalDegree_sub _ _).trans (max_le
          (by rw [MvPolynomial.totalDegree_X]) (by rw [MvPolynomial.totalDegree_C]; omega))
    _ = b / 2 := by simp

/-- The sum-check claim of the range check is the sum (80) at the range point `τ_0`. -/
theorem rangeTruth_nil (b : ℕ) (τ : List E) (w : Table) :
    combTruth τ.length (rangeTables τ w) (rangeComb E b) [] = mle τ (rangeResidual E b w) := by
  rw [combTruth_nil, mle_eq_cubeSum]
  congr 1
  funext x
  rw [eval_rangeComb, rangeResidual_eq]
  rfl

/-- The final claim of the range sum-check at the point `r_virt`:
`ẽq(τ_0, r_virt)·Q_sq(s̃(r_virt))` (Akita Section 6.1). `mle_eqAt` gives the first factor as
`eqEval τ_0 r_virt`. -/
theorem rangeTruth_final (b : ℕ) (τ r : List E) (w : Table) :
    combTruth r.length (rangeTables τ w) (rangeComb E b) r =
      mle r (eqAt τ) * rangeSqIn b (mle r (derived E w)) := by
  rw [combTruth_final, eval_rangeComb]
  rfl

/-- The true round polynomials of the range sum-check, of degree at most `b⋆/2 + 1`. -/
theorem range_roundTruth (b μ : ℕ) (τ : List E) (w : Table) :
    RoundTruth (b / 2 + 1) μ (combTruth μ (rangeTables τ w) (rangeComb E b))
      (combRound μ (rangeTables τ w) (rangeComb E b)) :=
  combRound_truth μ _ _ _ (rangeComb_degree b)

/-- The summand `w̃·m_{τ_1} + (γ·ẽq(r_virt, ·) + ζ_bin·ẽq_{I_bin}(r_virt, ·))·w̃·(w̃ + 1)` of the
fused sum (125) as a polynomial in the table of `w` (variable 0), the table of `m_{τ_1}`
(variable 1) and the table of the weight of `r_virt` (variable 2). -/
noncomputable def fusedComb (E : Type) [Field E] : MvPolynomial (Fin 3) E :=
  MvPolynomial.X 0 * MvPolynomial.X 1 +
    MvPolynomial.X 2 * (MvPolynomial.X 0 * (MvPolynomial.X 0 + 1))

/-- The tables of the fused sum (125): the witness, the relation weight and the weight of
`r_virt`. -/
def fusedTables (w : Table) (weight virt : ℕ → E) : Fin 3 → ℕ → E := ![lifted E w, weight, virt]

omit [Algebra Zq E] in
theorem eval_fusedComb (f : Fin 3 → E) :
    MvPolynomial.eval f (fusedComb E) = f 0 * f 1 + f 2 * (f 0 * (f 0 + 1)) := by
  simp [fusedComb]

omit [Algebra Zq E] in
/-- The summand of the fused sum (125) has the degree 3 (Akita Theorem 10.36). -/
theorem fusedComb_degree : (fusedComb E).totalDegree ≤ 3 := by
  unfold fusedComb
  have one : (MvPolynomial.X 0 + 1 : MvPolynomial (Fin 3) E).totalDegree ≤ 1 := by
    refine (MvPolynomial.totalDegree_add _ _).trans (max_le ?_ ?_)
    · rw [MvPolynomial.totalDegree_X]
    · rw [MvPolynomial.totalDegree_one]
      omega
  have two : (MvPolynomial.X 0 * (MvPolynomial.X 0 + 1) :
      MvPolynomial (Fin 3) E).totalDegree ≤ 2 := by
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_X]
    omega
  refine (MvPolynomial.totalDegree_add _ _).trans (max_le ?_ ?_)
  · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_X, MvPolynomial.totalDegree_X]
    omega
  · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_X]
    omega

/-- The weight `γ·ẽq(r_virt, x) + ζ_bin·ẽq_{I_bin}(r_virt, x)` of Akita (125) at the index `x`,
for the set `I_bin = support` of the positions of the compression digits (Akita Remark 6.2). -/
def virtWeight (support : ℕ → Prop) [DecidablePred support] (r : List E) (γ ζ : E) (x : ℕ) : E :=
  γ * eqAt r x + ζ * (if support x then eqAt r x else 0)

/-- The table `x ↦ 1_{I_bin}(x)·s(x)` of the binary-support claim (Akita Remark 6.2). -/
def supportTable (support : ℕ → Prop) [DecidablePred support] (w : Table) (x : ℕ) : E :=
  if support x then derived E w x else 0

theorem fusedTruth_nil (μ : ℕ) (w : Table) (weight virt : ℕ → E) :
    combTruth μ (fusedTables w weight virt) (fusedComb E) [] =
      ∑ x ∈ Finset.range (2 ^ μ), (lifted E w x * weight x + virt x * derived E w x) := by
  rw [combTruth_nil, cubeSum_eq_sum_range]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [eval_fusedComb, derived_eq]
  rfl

/-- The right side of the fused sum (125): the relation sum `Σ_x w(x)·m_{τ_1}(x)`, plus `γ` times
`s̃(r_virt)`, plus `ζ_bin` times the binary-support sum `Σ_{x ∈ I_bin} ẽq(r_virt, x)·s(x)`. -/
theorem fusedTruth_virt (support : ℕ → Prop) [DecidablePred support] (w : Table) (weight : ℕ → E)
    (r : List E) (γ ζ : E) :
    combTruth r.length (fusedTables w weight (virtWeight support r γ ζ)) (fusedComb E) [] =
      ∑ x ∈ Finset.range (2 ^ r.length), lifted E w x * weight x + γ * mle r (derived E w) +
        ζ * mle r (supportTable support w) := by
  rw [fusedTruth_nil, mle_eq_cubeSum, mle_eq_cubeSum, cubeSum_eq_sum_range, cubeSum_eq_sum_range,
    Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [virtWeight, supportTable]
  split_ifs <;> ring

/-- The final claim of the fused sum-check at the point `r_2`, from the values `w̃(r_2)`,
`m̃_{τ_1}(r_2)` and the value of the weight of `r_virt` (Akita Section 7.4). -/
theorem fusedTruth_final (r : List E) (w : Table) (weight virt : ℕ → E) :
    combTruth r.length (fusedTables w weight virt) (fusedComb E) r =
      mle r (lifted E w) * mle r weight +
        mle r virt * (mle r (lifted E w) * (mle r (lifted E w) + 1)) := by
  rw [combTruth_final, eval_fusedComb]
  rfl

/-- The true round polynomials of the fused sum-check, of degree at most 3. -/
theorem fused_roundTruth (μ : ℕ) (w : Table) (weight virt : ℕ → E) :
    RoundTruth 3 μ (combTruth μ (fusedTables w weight virt) (fusedComb E))
      (combRound μ (fusedTables w weight virt) (fusedComb E)) :=
  combRound_truth μ _ _ _ fusedComb_degree

end Tables

section Rounds

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

/-- Akita Theorem 3.3, item 1, for the range sum-check of the sum (80): on a path that avoids the
roots of each round, when the sum (80) is not zero at `τ_0`, the final claim is not
`ẽq(τ_0, r_virt)·Q_sq(s̃(r_virt))`. -/
theorem rangeSumcheck_avoids (b μ : ℕ) (τ : Fin μ → E) (w : Table) :
    (Game.sumcheck (b / 2 + 1) (combRound μ (rangeTables (List.ofFn τ) w) (rangeComb E b)) μ []
      0).Avoids fun result => ∀ final value, result = some (final, value) →
        mle (List.ofFn τ) (rangeResidual E b w) ≠ 0 →
          value ≠ mle final (eqAt (List.ofFn τ)) * rangeSqIn b (mle final (derived E w)) := by
  refine ((Game.sumcheck_avoids (b / 2 + 1) μ _ _ (range_roundTruth b μ (List.ofFn τ) w) μ [] 0
    (by simp)).and_ends (Game.sumcheck_ends (b / 2 + 1) _ μ [] 0)).mono ?_
  rintro result ⟨sound, ends⟩ final value equal nonzero
  obtain ⟨cs, prefixed, count⟩ := ends final value equal
  have length : final.length = μ := by
    rw [prefixed]
    simpa using count
  have last := rangeTruth_final b (List.ofFn τ) final w
  rw [length] at last
  have sum := rangeTruth_nil b (List.ofFn τ) w
  rw [List.length_ofFn] at sum
  rw [← last]
  exact sound final value equal (by rw [sum]; exact nonzero.symm)

/-- The range sum-check has `μ'` rounds with at most `b⋆/2 + 1` bad challenges each
(Akita Lemma 10.37, (187) for one stage). -/
theorem rangeSumcheck_costLe (b μ : ℕ) (τ : List E) (w : Table) (claim : E) :
    (Game.sumcheck (b / 2 + 1) (combRound μ (rangeTables τ w) (rangeComb E b)) μ []
      claim).CostLe (μ * (((b / 2 + 1 : ℕ) : ℝ) / Fintype.card E)) :=
  Game.sumcheck_costLe (b / 2 + 1) μ _ (range_roundTruth b μ τ w).degree μ [] claim (by simp)

/-- Akita Theorem 3.3, item 1, for the fused sum-check (125): on a path that avoids the roots of
each round, a claim that is not the fused sum ends in a final claim that is not the summand at
`r_2`. -/
theorem fusedSumcheck_avoids (μ : ℕ) (w : Table) (weight virt : ℕ → E) (claim : E) :
    (Game.sumcheck 3 (combRound μ (fusedTables w weight virt) (fusedComb E)) μ [] claim).Avoids
      fun result => ∀ final value, result = some (final, value) →
        claim ≠ ∑ x ∈ Finset.range (2 ^ μ), (lifted E w x * weight x + virt x * derived E w x) →
          value ≠ mle final (lifted E w) * mle final weight +
            mle final virt * (mle final (lifted E w) * (mle final (lifted E w) + 1)) := by
  refine ((Game.sumcheck_avoids 3 μ _ _ (fused_roundTruth μ w weight virt) μ [] claim
    (by simp)).and_ends (Game.sumcheck_ends 3 _ μ [] claim)).mono ?_
  rintro result ⟨sound, ends⟩ final value equal false_claim
  obtain ⟨cs, prefixed, count⟩ := ends final value equal
  have length : final.length = μ := by
    rw [prefixed]
    simpa using count
  have last := fusedTruth_final final w weight virt
  rw [length] at last
  rw [← last]
  exact sound final value equal (by rwa [fusedTruth_nil])

/-- The fused sum-check has `μ'` rounds with at most 3 bad challenges each (Akita
Theorem 10.36 and the term `3μ'` of (189)). -/
theorem fusedSumcheck_costLe (μ : ℕ) (w : Table) (weight virt : ℕ → E) (claim : E) :
    (Game.sumcheck 3 (combRound μ (fusedTables w weight virt) (fusedComb E)) μ [] claim).CostLe
      (μ * (((3 : ℕ) : ℝ) / Fintype.card E)) :=
  Game.sumcheck_costLe 3 μ _ (fused_roundTruth μ w weight virt).degree μ [] claim (by simp)

end Rounds

section Support

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

/-- The table `t` has a nonzero cell on the cube of `n` variables. -/
def NonzeroOn (n : ℕ) (t : ℕ → K) : Prop := ∃ i, i < 2 ^ n ∧ t i ≠ 0

open Classical in
/-- The values `c` of the first variable that bind a table with a nonzero cell on the cube of
`n + 1` variables to a table that is zero on the cube of `n` variables. -/
noncomputable def bindBad (n : ℕ) (t : ℕ → K) : Finset K :=
  if NonzeroOn (n + 1) t then Finset.univ.filter fun c => ¬ NonzeroOn n (bindLow c t) else ∅

theorem bindBad_card_le (n : ℕ) (t : ℕ → K) : (bindBad n t).card ≤ 1 := by
  classical
  unfold bindBad
  split
  · rename_i nonzero
    obtain ⟨i, lt, value⟩ := nonzero
    have pair : t (2 * (i / 2)) ≠ 0 ∨ t (2 * (i / 2) + 1) - t (2 * (i / 2)) ≠ 0 := by
      by_contra both
      push Not at both
      rcases Nat.mod_two_eq_zero_or_one i with even | odd
      · exact value (by rw [show i = 2 * (i / 2) by omega]; exact both.1)
      · exact value (by rw [show i = 2 * (i / 2) + 1 by omega, sub_eq_zero.mp both.2, both.1])
    refine (Finset.card_le_card fun c member => ?_).trans (zeros_card_le_of_natDegree_le
      (linePoly_natDegree (t (2 * (i / 2))) (t (2 * (i / 2) + 1) - t (2 * (i / 2)))))
    have zero : bindLow c t (i / 2) = 0 := by
      by_contra cell
      exact (Finset.mem_filter.mp member).2 ⟨i / 2, by rw [pow_succ] at lt; omega, cell⟩
    unfold zeros
    rw [ite_eq_right (linePoly_ne_zero pair)]
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, linePoly_eval]
    rw [← zero]
    simp only [bindLow]
    ring
  · simp

omit [DecidableEq K] in
/-- A table with a nonzero cell on the cube of `n + 1` variables keeps a nonzero cell when its
first variable is bound to a value outside `bindBad`. -/
theorem nonzeroOn_bindLow {n : ℕ} {t : ℕ → K} {c : K} (nonzero : NonzeroOn (n + 1) t)
    (outside : c ∉ bindBad n t) : NonzeroOn n (bindLow c t) := by
  classical
  by_contra zero
  refine outside ?_
  rw [bindBad, ite_eq_left nonzero, Finset.mem_filter]
  exact ⟨Finset.mem_univ c, zero⟩

/-- The sum-check of `Game.sumcheck` with the bad set `extra point` added to the bad set of the
draw after the prefix `point`. -/
noncomputable def sumcheckWith (d : ℕ) (round : List K → K[X]) (extra : List K → Finset K) :
    ℕ → List K → K → Game (Option (List K × K))
  | 0, point, claim => .pure (some (point, claim))
  | rounds + 1, point, claim =>
      .recv (Fin (d + 1) → K) fun message =>
        if (coeffPoly message).eval 0 + (coeffPoly message).eval 1 = claim then
          .draw K (zeros (coeffPoly message - round point) ∪ extra point) fun c =>
            sumcheckWith d round extra rounds (point ++ [c]) ((coeffPoly message).eval c)
        else .pure none

/-- On a path that avoids the bad sets, an accepted end has one coordinate for each round, a
false claim ends in a false final claim (Akita Theorem 3.3, item 1), and a property `keeps` of the
prefix that each draw outside `extra` preserves holds for the final point. -/
theorem sumcheckWith_avoids (d bound : ℕ) (truth : List K → K) (round : List K → K[X])
    (true_rounds : RoundTruth d bound truth round) (extra : List K → Finset K)
    (keeps : List K → Prop)
    (step : ∀ point c, point.length < bound → keeps point → c ∉ extra point →
      keeps (point ++ [c])) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheckWith d round extra rounds point claim).Avoids fun result =>
        ∀ final value, result = some (final, value) →
          final.length = point.length + rounds ∧ (claim ≠ truth point → value ≠ truth final) ∧
            (keeps point → keeps final) := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _ final value equal
      simp only [Option.some.injEq, Prod.mk.injEq] at equal
      rw [← equal.1, ← equal.2]
      exact ⟨rfl, id, id⟩
  | succ rounds ih =>
      intro point claim fits message
      have inside : point.length < bound := by omega
      dsimp only
      split
      · rename_i checked
        intro c outside
        rw [Finset.mem_union, not_or] at outside
        refine (ih (point ++ [c]) _ (by simp; omega)).mono ?_
        intro result transfer final value equal
        obtain ⟨length, sound, kept⟩ := transfer final value equal
        refine ⟨by rw [length, List.length_append, List.length_singleton]; omega,
          fun false_claim => sound ?_, fun holds => kept (step point c inside holds outside.2)⟩
        have differ : coeffPoly message ≠ round point := by
          intro same
          apply false_claim
          rw [← checked, same, true_rounds.halves point inside]
        rw [← true_rounds.binds point c inside]
        exact eval_ne_of_not_mem_zeros differ outside.1
      · intro final value equal
        simp at equal

/-- Each draw of `sumcheckWith` has at most `d + e` bad values, for rounds of degree `d` and sets
`extra point` of at most `e` values. -/
theorem sumcheckWith_costLe (d bound e : ℕ) (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d)
    (extra : List K → Finset K) (small : ∀ point, (extra point).card ≤ e) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheckWith d round extra rounds point claim).CostLe
        (rounds * (((d + e : ℕ) : ℝ) / Fintype.card K)) := by
  intro rounds
  induction rounds with
  | zero => intro point claim _; show (0 : ℝ) ≤ _; simp
  | succ rounds ih =>
      intro point claim fits message
      dsimp only
      split
      · refine ⟨rounds * (((d + e : ℕ) : ℝ) / Fintype.card K), ?_,
          fun c => ih (point ++ [c]) _ (by simp; omega)⟩
        have card : (zeros (coeffPoly message - round point) ∪ extra point).card ≤ d + e :=
          (Finset.card_union_le _ _).trans (Nat.add_le_add
            (zeros_card_le_of_natDegree_le ((natDegree_sub_le _ _).trans
              (max_le (coeffPoly_natDegree message) (degree point (by omega))))) (small point))
        have := Game.density_le (C := K) _ _ card
        push_cast at this ⊢
        linarith
      · show (0 : ℝ) ≤ _
        positivity

end Support

end NightstreamFPrime.Spec.Compression.Akita
