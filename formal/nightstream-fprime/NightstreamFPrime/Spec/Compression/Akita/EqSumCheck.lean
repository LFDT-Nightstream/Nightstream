import NightstreamFPrime.Spec.Compression.Akita.FoldSumCheck

/-! The equality-factored sum-check of Akita (ePrint 2026/1983, revision of 2026-11-23,
Section 3.5) in the form of the pinned Akita code (`akita-sumcheck/src/proof_stream.rs`,
`verify_eq_factored_sumcheck_rounds`; `single.rs`, `advance_eq_factored_claim`), for the stages of
the range tree (`RangeTree.lean`).

- A round message is `(q_1, …, q_D)`, the inner polynomial `q` without its constant coefficient.
  For the normalized claim `c` and the coordinate `τ_k` of the point before, `q_0` is
  `c − τ_k·Σ_i q_i` (`innerPoly`), and the next normalized claim is `q(r)`. A round has no check.
- `eqSumcheck` is this verifier. The bad set of a draw is the set of the roots of the full round
  polynomial `E·eq(τ_k, X)·q(X)` (`fullPoly`, with `E = eqEval τ point`) minus the true round
  polynomial of the full sum, with a second set `extra point`.
- `eqSumcheck_avoids` is Akita Theorem 3.3, item 1, through the full round polynomials: on a path
  that avoids the bad sets, a false claim `E·c ≠ T(point)` ends in `E_n·c_n ≠ T(final)`.
  `eqSumcheck_costLe` bounds the bad sets by `(D + 1 + e)/|K|` for each round, which is the bound
  of the full sum-check.
- `EqRoundTruth` has the true normalized round polynomials, and `innerPoly_tail` states that the
  message of a true normalized round polynomial gives it back.
- `eqTruth` and `eqRound` are the normalized sum and its round polynomials for a summand `X e·P`
  whose table `e` is `ẽq(τ, ·)`: `eqRound_truth`, `eqTruth_nil` and `eqTruth_final`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

section Rounds

variable {K : Type} [Field K]

/-- `eq(τ, X) = (1 − τ)·(1 − X) + τ·X`. -/
noncomputable def eqLine (τ : K) : K[X] := C (1 - τ) * (1 - X) + C τ * X

/-- The inner polynomial of an equality-factored message `(q_1, …, q_D)` for the normalized claim
`claim` and the coordinate `τ`: `q_0 = claim − τ·Σ_i q_i` (Akita Section 3.5). -/
noncomputable def innerPoly {D : ℕ} (claim τ : K) (message : Fin D → K) : K[X] :=
  C (claim - τ * ∑ i, message i) + X * coeffPoly message

/-- The full round polynomial `E·eq(τ, X)·q(X)` of an equality-factored message. -/
noncomputable def fullPoly {D : ℕ} (factor claim τ : K) (message : Fin D → K) : K[X] :=
  C factor * eqLine τ * innerPoly claim τ message

theorem eqLine_eval (τ x : K) : (eqLine τ).eval x = (1 - τ) * (1 - x) + τ * x := by
  simp [eqLine]

theorem innerPoly_eval {D : ℕ} (claim τ : K) (message : Fin D → K) (x : K) :
    (innerPoly claim τ message).eval x =
      claim - τ * ∑ i, message i + x * (coeffPoly message).eval x := by
  simp only [innerPoly, eval_add, eval_C, eval_mul, eval_X]

theorem coeffPoly_eval_one {D : ℕ} (message : Fin D → K) :
    (coeffPoly message).eval 1 = ∑ i, message i := by
  simp [coeffPoly_eval]

/-- `E·eq(τ, X)·q(X)` has the values `E·c` at 0 and 1 together. -/
theorem fullPoly_check {D : ℕ} (factor claim τ : K) (message : Fin D → K) :
    (fullPoly factor claim τ message).eval 0 + (fullPoly factor claim τ message).eval 1 =
      factor * claim := by
  simp only [fullPoly, eval_mul, eval_C, eqLine_eval, innerPoly_eval, coeffPoly_eval_one]
  ring

theorem fullPoly_eval {D : ℕ} (factor claim τ : K) (message : Fin D → K) (x : K) :
    (fullPoly factor claim τ message).eval x =
      factor * (eqLine τ).eval x * (innerPoly claim τ message).eval x := by
  simp only [fullPoly, eval_mul, eval_C]

theorem X_mul_coeffPoly_natDegree {D : ℕ} (message : Fin D → K) :
    (X * coeffPoly message).natDegree ≤ D := by
  cases D with
  | zero => simp [coeffPoly]
  | succ D =>
      refine natDegree_mul_le.trans ?_
      have := coeffPoly_natDegree message
      have one : (X : K[X]).natDegree ≤ 1 := natDegree_X_le
      omega

theorem innerPoly_natDegree {D : ℕ} (claim τ : K) (message : Fin D → K) :
    (innerPoly claim τ message).natDegree ≤ D :=
  (natDegree_add_le _ _).trans
    (max_le (by rw [natDegree_C]; exact Nat.zero_le _) (X_mul_coeffPoly_natDegree message))

theorem fullPoly_natDegree {D : ℕ} (factor claim τ : K) (message : Fin D → K) :
    (fullPoly factor claim τ message).natDegree ≤ D + 1 := by
  have line : (eqLine τ).natDegree ≤ 1 := by
    unfold eqLine
    refine (natDegree_add_le _ _).trans (max_le ?_ ?_)
    · exact (natDegree_C_mul_le _ _).trans
        ((natDegree_sub_le _ _).trans (max_le (by simp) natDegree_X_le))
    · exact (natDegree_C_mul_le _ _).trans natDegree_X_le
  have inner := innerPoly_natDegree claim τ message
  have first := natDegree_mul_le (p := C factor) (q := eqLine τ)
  rw [natDegree_C, zero_add] at first
  unfold fullPoly
  exact natDegree_mul_le.trans (by omega)

theorem eqEval_nil (τ : List K) : eqEval τ [] = 1 := by
  cases τ <;> rfl

/-- `ẽq(τ, point ++ [r]) = ẽq(τ, point)·eq(τ_k, r)` for `k = |point| < |τ|`. -/
theorem eqEval_append_single : ∀ (τ point : List K) (r : K), point.length < τ.length →
    eqEval τ (point ++ [r]) = eqEval τ point * (eqLine (τ.getD point.length 0)).eval r
  | [], _, _, short => absurd short (Nat.not_lt_zero _)
  | t :: τ, [], r, _ => by
      simp only [List.nil_append, eqEval, eqEval_nil, List.length_nil, List.getD_cons_zero,
        eqLine_eval]
      ring
  | t :: τ, x :: point, r, short => by
      simp only [List.cons_append, eqEval, List.length_cons, List.getD_cons_succ]
      rw [eqEval_append_single τ point r (by simpa using short)]
      ring

/-- The true normalized round polynomials `round` of the sum `truth` against `eq(τ, ·)`, of degree
at most `D`, for the first `bound` rounds: `(1 − τ_k)·q(0) + τ_k·q(1)` is the claim of the round
with `k = |point|`, and `q(c)` is the claim of the next round. -/
structure EqRoundTruth (D bound : ℕ) (τ : List K) (truth : List K → K) (round : List K → K[X]) :
    Prop where
  degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ D
  halves : ∀ point : List K, point.length < bound →
    (1 - τ.getD point.length 0) * (round point).eval 0 +
      τ.getD point.length 0 * (round point).eval 1 = truth point
  binds : ∀ point : List K, ∀ c : K, point.length < bound →
    (round point).eval c = truth (point ++ [c])

/-- The message of a normalized round polynomial `q` of degree at most `D`, `(q_1, …, q_D)`, gives
`q` back for the claim `(1 − τ)·q(0) + τ·q(1)`. -/
theorem innerPoly_tail {D : ℕ} (q : K[X]) (degree : q.natDegree ≤ D) (claim τ : K)
    (halves : (1 - τ) * q.eval 0 + τ * q.eval 1 = claim) :
    innerPoly claim τ (fun i : Fin D => q.coeff (i + 1)) = q := by
  have one : q.eval 1 = q.coeff 0 + ∑ i : Fin D, q.coeff (i + 1) := by
    rw [eval_eq_sum_range' (Nat.lt_succ_of_le degree), Finset.sum_range_succ']
    simp only [one_pow, mul_one]
    rw [Fin.sum_univ_eq_sum_range (fun i => q.coeff (i + 1)) D, add_comm]
  have zero : q.eval 0 = q.coeff 0 := (coeff_zero_eq_eval_zero q).symm
  ext n
  rcases n with _ | n
  · simp only [innerPoly, Polynomial.coeff_add, coeff_C_zero, coeff_X_mul_zero, add_zero]
    rw [← halves, one, zero]
    ring
  · simp only [innerPoly, Polynomial.coeff_add, coeff_C_succ, coeff_X_mul, zero_add, coeffPoly,
      finsetSum_coeff, coeff_C_mul_X_pow]
    by_cases small : n < D
    · rw [Finset.sum_eq_single (⟨n, small⟩ : Fin D) (fun (i : Fin D) _ other =>
        ite_eq_right fun (same : n = (i : ℕ)) => other (Fin.ext same.symm)) (by simp)]
      simp
    · rw [Finset.sum_eq_zero fun (i : Fin D) _ => ite_eq_right fun (same : n = (i : ℕ)) =>
        small (same ▸ i.isLt), coeff_eq_zero_of_natDegree_lt (by omega)]

end Rounds

section Game

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

/-- The equality-factored sum-check of `rounds` rounds with the point before `τ`
(`verify_eq_factored_sumcheck_rounds`). The round with the prefix `point` receives `(q_1, …, q_D)`
and draws `r`; the normalized claim becomes `q(r)` for the coordinate `τ_k`, `k = |point|`. It
ends in the point and the normalized claim. -/
noncomputable def eqSumcheck (D : ℕ) (round : List K → K[X]) (extra : List K → Finset K)
    (τ : List K) : ℕ → List K → K → Game (List K × K)
  | 0, point, claim => .pure (point, claim)
  | rounds + 1, point, claim =>
      .recv (Fin D → K) fun message =>
        .draw K (zeros (fullPoly (eqEval τ point) claim (τ.getD point.length 0) message -
            round point) ∪ extra point) fun r =>
          eqSumcheck D round extra τ rounds (point ++ [r])
            ((innerPoly claim (τ.getD point.length 0) message).eval r)

/-- On a path that avoids the bad sets, an end has one coordinate for each round, a false claim
`ẽq(τ, point)·c ≠ T(point)` ends in `ẽq(τ, final)·c_n ≠ T(final)` (Akita Theorem 3.3, item 1, for
the full round polynomials), and a property `keeps` that each draw outside `extra` preserves
holds for the final point. -/
theorem eqSumcheck_avoids (D bound : ℕ) (truth : List K → K) (round : List K → K[X])
    (true_rounds : RoundTruth (D + 1) bound truth round) (extra : List K → Finset K)
    (keeps : List K → Prop)
    (step : ∀ point c, point.length < bound → keeps point → c ∉ extra point →
      keeps (point ++ [c])) (τ : List K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      point.length + rounds ≤ τ.length →
      (eqSumcheck D round extra τ rounds point claim).Avoids fun result =>
        result.1.length = point.length + rounds ∧
          (eqEval τ point * claim ≠ truth point → eqEval τ result.1 * result.2 ≠ truth result.1) ∧
          (keeps point → keeps result.1) := by
  intro rounds
  induction rounds with
  | zero => exact fun point claim _ _ => ⟨rfl, id, id⟩
  | succ rounds ih =>
      intro point claim fits short message c outside
      have inside : point.length < bound := by omega
      rw [Finset.mem_union, not_or] at outside
      refine (ih (point ++ [c]) _ (by simp; omega) (by simp; omega)).mono ?_
      rintro result ⟨length, sound, kept⟩
      refine ⟨by rw [length, List.length_append, List.length_singleton]; omega,
        fun false_claim => sound ?_, fun holds => kept (step point c inside holds outside.2)⟩
      have differ : fullPoly (eqEval τ point) claim (τ.getD point.length 0) message ≠
          round point := by
        intro same
        apply false_claim
        rw [← fullPoly_check (eqEval τ point) claim (τ.getD point.length 0) message, same,
          true_rounds.halves point inside]
      rw [eqEval_append_single τ point c (by omega), ← true_rounds.binds point c inside]
      have value := eval_ne_of_not_mem_zeros differ outside.1
      rwa [fullPoly_eval] at value

/-- Each draw of `eqSumcheck` has at most `D + 1 + e` bad values, for true round polynomials of
degree `D + 1` and sets `extra point` of at most `e` values. -/
theorem eqSumcheck_costLe (D bound e : ℕ) (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ D + 1)
    (extra : List K → Finset K) (small : ∀ point, (extra point).card ≤ e) (τ : List K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (eqSumcheck D round extra τ rounds point claim).CostLe
        (rounds * (((D + 1 + e : ℕ) : ℝ) / Fintype.card K)) := by
  intro rounds
  induction rounds with
  | zero => intro point claim _; show (0 : ℝ) ≤ _; simp
  | succ rounds ih =>
      intro point claim fits message
      refine ⟨rounds * (((D + 1 + e : ℕ) : ℝ) / Fintype.card K), ?_,
        fun c => ih (point ++ [c]) _ (by simp; omega)⟩
      have card : (zeros (fullPoly (eqEval τ point) claim (τ.getD point.length 0) message -
          round point) ∪ extra point).card ≤ D + 1 + e :=
        (Finset.card_union_le _ _).trans (Nat.add_le_add
          (zeros_card_le_of_natDegree_le ((natDegree_sub_le _ _).trans
            (max_le (fullPoly_natDegree _ _ _ message) (degree point (by omega)))))
          (small point))
      have := Game.density_le (C := K) _ _ card
      push_cast at this ⊢
      linarith

end Game

section Truth

variable {K : Type} [Field K] {σ : Type}

/-- The normalized sum at the prefix `point`: `Σ_x ẽq(τ_{≥k}, x)·P(tables at point ++ x)` for
`k = |point|`. -/
noncomputable def eqTruth (n : ℕ) (τ : List K) (tables : σ → ℕ → K) (P : MvPolynomial σ K)
    (point : List K) : K :=
  cubeSum (n - point.length) fun x =>
    eqAt (τ.drop point.length) x * MvPolynomial.eval (fun j => fold point (tables j) x) P

/-- The normalized round polynomial at the prefix `point`. -/
noncomputable def eqRound (n : ℕ) (τ : List K) (tables : σ → ℕ → K) (P : MvPolynomial σ K)
    (point : List K) : K[X] :=
  cubeSum (n - point.length - 1) fun x =>
    C (eqAt (τ.drop (point.length + 1)) x) *
      MvPolynomial.aeval (fun j => lineAt (fold point (tables j)) x) P

theorem eqRound_truth (D : ℕ) (τ : List K) (tables : σ → ℕ → K) (P : MvPolynomial σ K)
    (degree : P.totalDegree ≤ D) :
    EqRoundTruth D τ.length τ (eqTruth τ.length τ tables P) (eqRound τ.length τ tables P) where
  degree point _ := natDegree_cubeSum_le _ _ _ fun x =>
    (natDegree_C_mul_le _ _).trans
      ((natDegree_aeval_le P _ fun j => lineAt_natDegree _ _).trans degree)
  halves point short := by
    obtain ⟨m, rest⟩ : ∃ m, τ.length - point.length = m + 1 :=
      ⟨τ.length - point.length - 1, by omega⟩
    have getD : τ.getD point.length 0 = τ[point.length] := by
      simp [List.getD_eq_getElem?_getD, short]
    unfold eqRound eqTruth
    rw [eval_cubeSum, eval_cubeSum, rest, show m + 1 - 1 = m by omega,
      List.drop_eq_getElem_cons short, getD]
    simp only [cubeSum, eval_mul, eval_C, eval_aeval, lineAt_eval, bindLow, eqAt, sub_zero,
      one_mul, zero_mul, add_zero, sub_self, zero_add]
    rw [← cubeSum_mul_left, ← cubeSum_mul_left]
    congr 1 <;> refine congrArg _ (funext fun x => ?_)
    · rw [ite_eq_right (by omega), show 2 * x / 2 = x by omega]
      ring
    · rw [ite_eq_left (by omega), show (2 * x + 1) / 2 = x by omega]
      ring
  binds point c short := by
    unfold eqRound eqTruth
    rw [eval_cubeSum, List.length_append, List.length_singleton,
      show τ.length - (point.length + 1) = τ.length - point.length - 1 by omega]
    refine congrArg _ (funext fun x => ?_)
    rw [eval_mul, eval_C, eval_aeval]
    exact congrArg _ (eval_congr P fun j => by rw [lineAt_eval, fold_append_single])

/-- The normalized sum at the empty prefix is the full sum of `X e·P` when the table `e` is
`ẽq(τ, ·)`. -/
theorem eqTruth_nil (n : ℕ) (τ : List K) (tables : σ → ℕ → K) (P : MvPolynomial σ K) (e : σ)
    (equality : tables e = eqAt τ) :
    eqTruth n τ tables P [] = combTruth n tables (MvPolynomial.X e * P) [] := by
  rw [combTruth_nil]
  simp only [eqTruth, List.length_nil, Nat.sub_zero, List.drop_zero, fold, map_mul,
    MvPolynomial.eval_X, equality]

/-- The normalized sum at a full point is `P` at the values of the tables. -/
theorem eqTruth_final (τ : List K) (tables : σ → ℕ → K) (P : MvPolynomial σ K) (final : List K)
    (length : final.length = τ.length) :
    eqTruth τ.length τ tables P final = MvPolynomial.eval (fun j => mle final (tables j)) P := by
  simp only [eqTruth, length, Nat.sub_self, cubeSum, List.drop_length, eqAt, one_mul, mle]

end Truth

end NightstreamFPrime.Spec.Compression.Akita
