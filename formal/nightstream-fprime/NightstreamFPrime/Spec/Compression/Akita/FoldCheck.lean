import NightstreamFPrime.Spec.Compression.Akita.RowBatch
import NightstreamFPrime.Spec.Compression.Akita.FoldSumCheck

/-! The checks of one fold of Akita after the commitment of the next witness, as a game
(ePrint 2026/1983, revision of 2026-10-15, Figure 6, Sections 6.1 and 7.4, Theorem 10.36,
Lemma 10.37 and the terms of (189) after the fold challenge).

- `rangeBad` is the bad set of the range point `τ_0`, with `density_rangeBad_le`.
- `BinaryOn support μ w` is the statement of Akita Remark 6.2: `w(x)·(w(x) + 1) = 0` at each
  position `x ∈ I_bin`.
- `foldCheck` is the verifier for the flat witness `w`: the point `α` of the ring reduction, the
  range point `τ_0`, the batching point `τ_1`, the range sum-check with the claim `s_claim`, the
  coefficients `ζ_bin` and `γ`, and the fused sum-check (125) with the claimed value `v_W`. It
  ends in `none` when a check fails, and otherwise in the claim `(r_2, v_W)` for the next level.
- `foldCheck_avoids` states that on a path that avoids the bad sets, a claim `(r_2, v_W)` with
  `v_W = w̃(r_2)` gives the relation, the digit range statement and the binary statement.
- `foldError` is the sum of the bounds of the bad sets, `foldError_eq` is its closed form, which
  is the part of Akita (189) after the fold challenge for one range stage, and `foldCheck_costLe`
  and `foldCheck_error` bound the bad sets and the probability of a false accepted claim. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

open Classical in
/-- The bad set of the range point `τ_0 ∈ E^μ` for the witness `w`: the points where the sum (80)
is zero for a witness with a cell outside `A_{b⋆}` (Akita Lemma 10.37 (i)). -/
noncomputable def rangeBad (b μ : ℕ) (w : Table) : Finset (Fin μ → E) :=
  if RangeHolds b μ w then ∅
  else Finset.univ.filter fun τ => mle (List.ofFn τ) (rangeResidual E b w) = 0

/-- The bad set of the range point has a density of at most `μ' / |E|` (Akita (189)). -/
theorem density_rangeBad_le (b μ : ℕ) (w : Table) :
    density (rangeBad b μ w : Finset (Fin μ → E)) ≤ μ / Fintype.card E := by
  unfold rangeBad
  split
  · rw [density_empty]
    positivity
  · rename_i fails
    exact density_rangeSum_le fails

/-- A sum (80) that is zero at a range point outside the bad set gives the digit range
statement. -/
theorem rangeHolds_of_sum {b μ : ℕ} {w : Table} {τ : Fin μ → E}
    (outside : τ ∉ (rangeBad b μ w : Finset (Fin μ → E)))
    (zero : mle (List.ofFn τ) (rangeResidual E b w) = 0) : RangeHolds b μ w := by
  by_contra fails
  refine outside ?_
  rw [rangeBad, ite_eq_right fails, Finset.mem_filter]
  exact ⟨Finset.mem_univ τ, zero⟩

omit [Algebra Zq E] in
/-- A combination `a + c·b` that is zero for a coefficient `c` outside the roots of the line has
`a = 0` and `b = 0`. -/
theorem eq_zero_of_not_mem_zeros_linePoly {a b c : E} (outside : c ∉ zeros (linePoly a b))
    (root : a + c * b = 0) : a = 0 ∧ b = 0 := by
  by_contra nonzero
  refine outside ?_
  unfold zeros
  rw [ite_eq_right (linePoly_ne_zero (not_and_or.mp nonzero))]
  simp [linePoly_eval, root]

/-- The fused constraint of Akita Remark 6.2 on the positions `I_bin = support` of the witness on
the cube of `μ` variables: `w(x)·(w(x) + 1) = 0`. -/
def BinaryOn (support : ℕ → Prop) (μ : ℕ) (w : Table) : Prop :=
  ∀ x < 2 ^ μ, support x → w x * (w x + 1) = 0

/-- The fused sum-check (125) of Akita for the claim `V_{α,τ_1} + γ·s_claim`, and its final check
at the point `r_2` with the claimed value `v_W` of `w̃(r_2)` (Akita Section 7.4). -/
noncomputable def fusedCheck (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) : Game (Option (List E × E)) :=
  (sumcheck 3 (combRound μ (fusedTables w weight (virtWeight support r γ ζ)) (fusedComb E)) μ []
    (target + γ * sClaim)).bind fun
    | none => .pure none
    | some (r2, value) => .recv E fun v =>
        if value = v * mle r2 weight + mle r2 (virtWeight support r γ ζ) * (v * (v + 1)) then
          .pure (some (r2, v))
        else .pure none

/-- The checks of one fold for the flat witness `w` of the next level, in the order of Akita
Figure 6. The verifier draws `α`, `τ_0` and `τ_1`, runs the range sum-check for the sum (80) to
the point `r_virt`, receives `s_claim` and checks the final claim
`ẽq(τ_0, r_virt)·Q_sq(s_claim)`, draws `ζ_bin` and `γ`, and runs the fused sum-check (125). The
bad sets are: the bad set of the ring reduction, `rangeBad`, `FieldRows.batchBad`, the roots of
each round of the range sum-check with `bindBad` for the binary-support table, the roots of the
two combinations, and the roots of each round of the fused sum-check. -/
noncomputable def foldCheck (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) : Game (Option (List E × E)) :=
  .draw E (R.bad w) fun α =>
  .draw (Fin R.vars → E) (rangeBad b R.vars w) fun τ0 =>
  .draw (Fin R.batch → E) ((R.rows α).batchBad R.vars R.batch w) fun τ1 =>
  (sumcheckWith (b / 2 + 1) (combRound R.vars (rangeTables (List.ofFn τ0) w) (rangeComb E b))
    (fun point => bindBad (R.vars - point.length - 1) (fold point (supportTable support w)))
    R.vars [] 0).bind fun
    | none => .pure none
    | some (r, value) => .recv E fun sClaim =>
        if value = mle r (eqAt (List.ofFn τ0)) * rangeSqIn b sClaim then
          .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w))
            (mle r (supportTable support w)))) fun ζ =>
          .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
            ζ * mle r (supportTable support w)) (mle r (derived E w) - sClaim))) fun γ =>
            fusedCheck R.vars support w ((R.rows α).batchWeight (List.ofFn τ1))
              ((R.rows α).batchTarget (List.ofFn τ1)) r sClaim ζ γ
        else .pure none

/-- On a path of `foldCheck` that avoids the bad sets, an accepted claim `(r_2, v_W)` that is true
for the committed witness, `v_W = w̃(r_2)`, gives the relation of the ring reduction, the digit
range statement of Akita Section 6.1 and the binary statement of Akita Remark 6.2. -/
theorem foldCheck_avoids (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) :
    (foldCheck R b support w).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w) →
        R.Holds w ∧ RangeHolds b R.vars w ∧ BinaryOn support R.vars w := by
  intro α reduce τ0 point τ1 batch
  refine (avoids_bind _ _ _).mpr ?_
  refine (sumcheckWith_avoids (b / 2 + 1) R.vars _ _ (range_roundTruth b R.vars (List.ofFn τ0) w)
    (fun point => bindBad (R.vars - point.length - 1) (fold point (supportTable support w)))
    (fun point => NonzeroOn (R.vars - point.length) (fold point (supportTable support w)))
    (fun point c inside nonzero outside => by
      rw [fold_append_single, List.length_append, List.length_singleton,
        show R.vars - (point.length + 1) = R.vars - point.length - 1 by omega]
      exact nonzeroOn_bindLow
        (by rwa [show R.vars - point.length - 1 + 1 = R.vars - point.length by omega]) outside)
    R.vars [] 0 (by simp)).mono ?_
  rintro (_ | ⟨r, value⟩) ranged
  · intro r2 v equal
    simp at equal
  intro sClaim
  dsimp only
  split
  swap
  · intro r2 v equal
    simp at equal
  rename_i checked
  intro ζ outsideζ γ outsideγ
  obtain ⟨length, rangeSound, supportKept⟩ := ranged r value rfl
  simp only [List.length_nil, zero_add] at length
  refine (avoids_bind _ _ _).mpr ?_
  refine ((sumcheck_avoids 3 R.vars _ _ (fused_roundTruth R.vars w _ _) R.vars [] _
    (by simp)).and_ends (sumcheck_ends 3 _ R.vars [] _)).mono ?_
  rintro (_ | ⟨r2, value2⟩) ⟨fusedSound, fusedEnds⟩
  · intro r2 v equal
    simp at equal
  intro v
  dsimp only
  split
  swap
  · intro r2 v equal
    simp at equal
  rename_i final
  intro r2' v' equal opening
  simp only [Option.some.injEq, Prod.mk.injEq] at equal
  obtain ⟨rfl, rfl⟩ := equal
  have length2 : r2.length = R.vars := by
    obtain ⟨cs, prefixed, count⟩ := fusedEnds r2 value2 rfl
    rw [prefixed]
    simpa using count
  have fusedTrue := not_imp_not.mp (fusedSound r2 value2 rfl) (by
    have last := fusedTruth_final r2 w ((R.rows α).batchWeight (List.ofFn τ1))
      (virtWeight support r γ ζ)
    rw [length2] at last
    rw [last, final, opening])
  have virt := fusedTruth_virt support w ((R.rows α).batchWeight (List.ofFn τ1)) r γ ζ
  rw [length] at virt
  have batched : ∑ x ∈ Finset.range (2 ^ R.vars),
      lifted E w x * (R.rows α).batchWeight (List.ofFn τ1) x -
        (R.rows α).batchTarget (List.ofFn τ1) =
      mle (List.ofFn τ1) ((R.rows α).residual R.vars w) :=
    (R.rows α).batch_residual R.vars w (List.ofFn τ1)
  have root : mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
      ζ * mle r (supportTable support w) + γ * (mle r (derived E w) - sClaim) = 0 := by
    rw [← batched]
    linear_combination (fusedTrue.trans virt).symm
  obtain ⟨first, second⟩ := eq_zero_of_not_mem_zeros_linePoly outsideγ root
  obtain ⟨relation, binary⟩ := eq_zero_of_not_mem_zeros_linePoly outsideζ first
  have rangeTrue := not_imp_not.mp rangeSound (by
    have last := rangeTruth_final b (List.ofFn τ0) r w
    rw [length] at last
    rw [last, checked, sub_eq_zero.mp second])
  have sum := rangeTruth_nil b (List.ofFn τ0) w
  rw [List.length_ofFn] at sum
  have zeroTable : ¬ NonzeroOn R.vars (supportTable (E := E) support w) := by
    intro nonzero
    have kept := supportKept (by simpa [fold] using nonzero)
    rw [length, Nat.sub_self] at kept
    obtain ⟨i, lt, cell⟩ := kept
    obtain rfl : i = 0 := by simpa using lt
    exact cell binary
  refine ⟨R.sound w α reduce ((R.rows α).holds_of_batched R.vars batch
    (((R.rows α).batched_iff R.vars w _).mpr relation)),
    rangeHolds_of_sum point (by rw [← sum, ← rangeTrue]), fun x small member => ?_⟩
  by_contra cell
  refine zeroTable ⟨x, small, ?_⟩
  rw [supportTable, ite_eq_left member, derived]
  exact fun zero => cell ((algebraMap Zq E).injective (zero.trans (map_zero _).symm))

/-- The sum of the bounds of the bad sets of `foldCheck`, in the order of the draws: the ring
reduction, the range point `τ_0`, the batching point `τ_1`, the `μ'` rounds of the range sum-check
of degree `b⋆/2 + 1` with one more value for the binary-support point, the coefficients `ζ_bin`
and `γ`, and the `μ'` rounds of the fused sum-check of degree 3 (Akita (189)). -/
noncomputable def foldError (R : Reduction E) (b : ℕ) : ℝ :=
  (R.badBound : ℝ) / Fintype.card E + ((R.vars : ℝ) / Fintype.card E +
    ((R.batch : ℝ) / Fintype.card E +
      ((R.vars : ℝ) * (((b / 2 + 1 + 1 : ℕ) : ℝ) / Fintype.card E) +
        (((1 : ℕ) : ℝ) / Fintype.card E + (((1 : ℕ) : ℝ) / Fintype.card E +
          (R.vars : ℝ) * (((3 : ℕ) : ℝ) / Fintype.card E))))))

omit [DecidableEq E] in
/-- The closed form of `foldError`. For the quotient-free reduction, with `badBound = d_max − 1`
and `batch = ⌈log n⌉`, it is the part of `ε_h` of Akita (189) after `ε_cw,h`, with `k_ring − 1`
for `k_ring`, for one range stage of the degree `d_1 = b⋆/2`, `β_h = 1` and `N^norm_h = 0`. -/
theorem foldError_eq (R : Reduction E) (b : ℕ) :
    foldError R b =
      ((R.badBound + R.batch + R.vars + R.vars + R.vars * (b / 2 + 1) + 1 + 1 + 3 * R.vars : ℕ) :
        ℝ) / Fintype.card E := by
  unfold foldError
  push_cast
  ring

theorem fusedCheck_costLe (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) :
    (fusedCheck μ support w weight target r sClaim ζ γ).CostLe
      ((μ : ℝ) * (((3 : ℕ) : ℝ) / Fintype.card E)) := by
  refine (costLe_bind (δ := 0) (sumcheck_costLe 3 μ _ (fused_roundTruth μ w _ _).degree μ [] _
    (by simp)) ?_).mono (by rw [add_zero])
  rintro (_ | ⟨r2, value⟩)
  · exact le_refl (0 : ℝ)
  · intro v
    dsimp only
    split <;> exact le_refl (0 : ℝ)

/-- The bad sets of `foldCheck` have the total bound `foldError`. -/
theorem foldCheck_costLe (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) : (foldCheck R b support w).CostLe (foldError R b) := by
  have reduce := density_le (C := E) (R.bad w) _ (R.bad_card w)
  have point := density_rangeBad_le (E := E) b R.vars w
  unfold density at point
  refine ⟨_, add_le_add_left reduce _, fun α => ⟨_, add_le_add_left point _, fun τ0 => ?_⟩⟩
  have batch := (R.rows α).density_batchBad_le R.vars (R.fits α) w
  unfold density at batch
  refine ⟨_, add_le_add_left batch _, fun τ1 => ?_⟩
  refine costLe_bind (sumcheckWith_costLe (b / 2 + 1) R.vars 1 _
    (range_roundTruth b R.vars (List.ofFn τ0) w).degree _ (fun _ => bindBad_card_le _ _)
    R.vars [] 0 (by simp)) ?_
  have rest : (0 : ℝ) ≤ ((1 : ℕ) : ℝ) / Fintype.card E + (((1 : ℕ) : ℝ) / Fintype.card E +
      (R.vars : ℝ) * (((3 : ℕ) : ℝ) / Fintype.card E)) := by positivity
  rintro (_ | ⟨r, value⟩)
  · exact rest
  · intro sClaim
    dsimp only
    split
    · exact ⟨_, add_le_add_left (density_le _ 1
          (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) _,
        fun ζ => ⟨_, add_le_add_left (density_le _ 1
          (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) _,
          fun γ => fusedCheck_costLe R.vars support w _ _ r sClaim ζ γ⟩⟩
    · exact rest

/-- For each prover, the probability that `foldCheck` ends in a claim `(r_2, v_W)` with
`v_W = w̃(r_2)` for a witness that does not satisfy the relation, the digit range statement or the
binary statement is at most `foldError`. -/
theorem foldCheck_error (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) (s : Strategy (Option (List E × E))) :
    prob s (foldCheck R b support w) (fun result =>
      ¬ ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w) →
        R.Holds w ∧ RangeHolds b R.vars w ∧ BinaryOn support R.vars w) ≤ foldError R b :=
  prob_not_le s _ _ _ (foldCheck_avoids R b support w) (foldCheck_costLe R b support w)

end NightstreamFPrime.Spec.Compression.Akita
