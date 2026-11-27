import NightstreamFPrime.Spec.Compression.Akita.RowBatch
import NightstreamFPrime.Spec.Compression.Akita.RangeTree

/-! The checks of one fold of Akita after the commitment of the next witness, as a game
(ePrint 2026/1983, revision of 2026-11-23, Figure 6, Sections 6.1 and 7.4, Theorem 10.36,
Lemma 10.37 and the terms of (189) after the fold challenge).

- `rangePoint` is the range point of the pinned code for the drawn `τ_0`: its last `low`
  coordinates first. `rangeBad` is the bad set of `τ_0`, with `density_rangeBad_le`.
- `BinaryOn support μ w` is the statement of Akita Remark 6.2: `w(x)·(w(x) + 1) = 0` at each
  position `x ∈ I_bin`.
- `foldCheck` is the verifier for the flat witness `w`: the point `α` of the ring reduction, the
  range point `τ_0`, the batching point `τ_1`, the range tree `rangeCheck` with the claim
  `s_claim`, the coefficients `ζ_bin` and `γ`, and the fused sum-check (125) with the claimed
  value `v_W`. `drawBin` is the draw of `ζ_bin`: a level without compression digits has no draw
  (`β_h = 0` in Akita (189)). It ends in `none` when a check fails, and otherwise in the claim
  `(r_2, v_W)` for the next level. `fused_sound` and `fusedCheck_avoids` are the two steps of its
  soundness after the range tree.
- `foldCheck_avoids` states that on a path that avoids the bad sets, a claim `(r_2, v_W)` with
  `v_W = w̃(r_2)` gives the relation, the digit range statement and the binary statement.
- `foldError` is the sum of the bounds of the bad sets, `foldError_eq` is its closed form, which
  is the part of Akita (189) after the fold challenge, and `foldCheck_costLe`
  and `foldCheck_error` bound the bad sets and the probability of a false accepted claim. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

section Point

variable {K : Type} {μ : ℕ}

/-- The range point of the pinned code for the drawn point `τ_0 ∈ E^μ`: `τ_0` with its last `low`
coordinates first (`DigitRangeEqualityPoint::from_column_then_ring_challenges` of
`crates/akita-types/src/proof/stage1.rs` of the Akita code). Akita (80) has a uniform point, and
this point is uniform when `τ_0` is. -/
def rangePoint (low : ℕ) (τ : Fin μ → K) : List K := (List.ofFn τ).rotate (μ - low)

@[simp] theorem rangePoint_length (low : ℕ) (τ : Fin μ → K) : (rangePoint low τ).length = μ := by
  simp [rangePoint]

theorem turn_injective (μ k : ℕ) :
    Function.Injective fun i : Fin μ => (⟨(i + k) % μ, Nat.mod_lt _ i.pos⟩ : Fin μ) :=
  fun i j same => Fin.ext (Nat.ModEq.eq_of_lt_of_lt
    (Nat.ModEq.add_right_cancel' k (Fin.mk.inj_iff.mp same)) i.isLt j.isLt)

/-- The coordinate of `τ_0` at each place of `rangePoint`. -/
noncomputable def turn (μ low : ℕ) : Equiv.Perm (Fin μ) :=
  Equiv.ofBijective _ (Finite.injective_iff_bijective.mp (turn_injective μ (μ - low)))

theorem rangePoint_eq_ofFn (low : ℕ) (τ : Fin μ → K) :
    rangePoint low τ = List.ofFn fun i => τ (turn μ low i) := by
  refine List.ext_getElem (by simp) fun i left right => ?_
  simp [rangePoint, List.getElem_rotate, turn]

end Point

open Classical in
/-- The bad set of the range point `τ_0 ∈ E^μ` for the witness `w`: the points where the sum (80)
is zero at `rangePoint` for a witness with a cell outside `A_{b⋆}` (Akita Lemma 10.37 (i)). -/
noncomputable def rangeBad (low b μ : ℕ) (w : Table) : Finset (Fin μ → E) :=
  if RangeHolds b μ w then ∅
  else Finset.univ.filter fun τ => mle (rangePoint low τ) (rangeResidual E b w) = 0

/-- The bad set of the range point has a density of at most `μ' / |E|` (Akita (189)). -/
theorem density_rangeBad_le (low b μ : ℕ) (w : Table) :
    density (rangeBad low b μ w : Finset (Fin μ → E)) ≤ μ / Fintype.card E := by
  unfold rangeBad
  split
  · rw [density_empty]
    positivity
  · rename_i fails
    refine (density_congr ((turn μ low).symm.arrowCongr (Equiv.refl E)) _ _ fun τ => ?_).le.trans
      (density_rangeSum_le fails)
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, rangePoint_eq_ofFn]
    refine Eq.congr_left (congrArg (fun p => mle p _) (congrArg List.ofFn (funext fun i => ?_)))
    simp [Equiv.arrowCongr]

/-- A sum (80) that is zero at the range point of a draw outside the bad set gives the digit
range statement. -/
theorem rangeHolds_of_sum {low b μ : ℕ} {w : Table} {τ : Fin μ → E}
    (outside : τ ∉ (rangeBad low b μ w : Finset (Fin μ → E)))
    (zero : mle (rangePoint low τ) (rangeResidual E b w) = 0) : RangeHolds b μ w := by
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

/-- The draw of the coefficient `ζ_bin` of the binary-support claim at a level with compression
digits, `bin = true`. A level without compression digits has no draw, and the coefficient is
zero (Akita (189), `β_h`). -/
def drawBin {β : Type} : Bool → Finset E → (E → Game β) → Game β
  | true, bad, next => .draw E bad next
  | false, _, next => next 0

/-- The number of the draws of `drawBin`. -/
def binDraws : Bool → ℕ
  | true => 1
  | false => 0

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

/-- The checks of one fold for the flat witness `w` of the next level, in the order of Akita Figure
6. The verifier draws `α`, `τ_0` and `τ_1`, runs the range tree `rangeCheck` for the sum (80) at
the point `rangePoint R.low τ_0` to the point `r_virt` and the value `s_claim`, draws `ζ_bin` at a level with compression digits and `γ`,
and runs the fused sum-check (125). The bad sets are: the bad set of the ring reduction, `rangeBad`,
`FieldRows.batchBad`, the bad sets of `rangeCheck`, the roots of the two combinations, and the roots
of each round of the fused sum-check. -/
noncomputable def foldCheck (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (w : Table) : Game (Option (List E × E)) :=
  .draw E (R.bad w) fun α =>
  .draw (Fin R.vars → E) (rangeBad R.low b R.vars w) fun τ0 =>
  .draw (Fin R.batch → E) ((R.rows α).batchBad R.vars R.batch w) fun τ1 =>
  (rangeCheck b R.vars support w (rangePoint R.low τ0)).bind fun
    | none => .pure none
    | some (r, sClaim) =>
        drawBin bin (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w))
          (mle r (supportTable support w)))) fun ζ =>
        .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
          ζ * mle r (supportTable support w)) (mle r (derived E w) - sClaim))) fun γ =>
          fusedCheck R.vars support w ((R.rows α).batchWeight (List.ofFn τ1))
            ((R.rows α).batchTarget (List.ofFn τ1)) r sClaim ζ γ

/-- The relation, the range sum and the binary statement from a true final claim of the fused
sum-check (125) after an end `(r_virt, s_claim)` of the range tree with `Virt`, for coefficients
`ζ_bin` and `γ` outside the roots of the two combinations. `ranged` is the soundness of the range
tree for a true `s_claim`. At a level without compression digits, `ζ_bin = 0` and `I_bin` is
empty. -/
theorem fused_sound {μ : ℕ} {support : ℕ → Prop} [DecidablePred support] {w : Table}
    {weight : ℕ → E} {target residual : E} {r : List E} {sClaim ζ γ : E}
    (virt : Virt μ support w (r, sClaim))
    (batched : ∑ x ∈ Finset.range (2 ^ μ), lifted E w x * weight x - target = residual)
    (outsideζ : ζ ∉ zeros (linePoly residual (mle r (supportTable support w))) ∨
      ζ = 0 ∧ ∀ x, ¬ support x)
    (outsideγ : γ ∉ zeros (linePoly (residual + ζ * mle r (supportTable support w))
      (mle r (derived E w) - sClaim)))
    (fused : target + γ * sClaim =
      combTruth μ (fusedTables w weight (virtWeight support r γ ζ)) (fusedComb E) []) :
    residual = 0 ∧ sClaim = mle r (derived E w) ∧ BinaryOn support μ w := by
  obtain ⟨length, nonzero⟩ := virt
  have sum := fusedTruth_virt support w weight r γ ζ
  rw [show r.length = μ from length] at sum
  have root : residual + ζ * mle r (supportTable support w) +
      γ * (mle r (derived E w) - sClaim) = 0 := by
    rw [← batched]
    linear_combination (fused.trans sum).symm
  obtain ⟨first, second⟩ := eq_zero_of_not_mem_zeros_linePoly outsideγ root
  obtain ⟨relation, binary⟩ : residual = 0 ∧ mle r (supportTable support w) = (0 : E) := by
    rcases outsideζ with outside | ⟨rfl, empty⟩
    · exact eq_zero_of_not_mem_zeros_linePoly outside first
    · have none : supportTable (E := E) support w = fun _ => 0 :=
        funext fun x => ite_eq_right (empty x)
      rw [none, mle_const] at first ⊢
      exact ⟨by simpa using first, rfl⟩
  refine ⟨relation, (sub_eq_zero.mp second).symm, fun x small member => ?_⟩
  by_contra cell
  refine nonzero ⟨x, small, ?_⟩ binary
  rw [supportTable, ite_eq_left member, derived]
  exact fun zero => cell ((algebraMap Zq E).injective (zero.trans (map_zero _).symm))

/-- On a path of `fusedCheck` that avoids the bad sets, an accepted claim `(r_2, v_W)` that is
true for the witness, `v_W = w̃(r_2)`, has a true first claim `target + γ·s_claim`. -/
theorem fusedCheck_avoids (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) :
    (fusedCheck μ support w weight target r sClaim ζ γ).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w) →
        target + γ * sClaim =
          combTruth μ (fusedTables w weight (virtWeight support r γ ζ)) (fusedComb E) [] := by
  refine (avoids_bind _ _ _).mpr ?_
  refine ((sumcheck_avoids 3 μ _ _ (fused_roundTruth μ w _ _) μ [] _
    (by simp)).and_ends (sumcheck_ends 3 _ μ [] _)).mono ?_
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
  have length2 : r2.length = μ := by
    obtain ⟨cs, prefixed, count⟩ := fusedEnds r2 value2 rfl
    rw [prefixed]
    simpa using count
  exact not_imp_not.mp (fusedSound r2 value2 rfl) (by
    have last := fusedTruth_final r2 w weight (virtWeight support r γ ζ)
    rw [length2] at last
    rw [last, final, opening])

omit [DecidableEq E] [Algebra Zq E] in
/-- A game after `drawBin` avoids its bad sets when it does so for each coefficient `ζ_bin`
outside the bad set, and for `ζ_bin = 0` at a level whose set `I_bin` is empty. -/
theorem drawBin_avoids {β : Type} {support : ℕ → Prop} {bin : Bool}
    (empty : bin = false → ∀ x, ¬ support x) {bad : Finset E} {next : E → Game β}
    {goal : β → Prop}
    (each : ∀ ζ, (ζ ∉ bad ∨ ζ = 0 ∧ ∀ x, ¬ support x) → (next ζ).Avoids goal) :
    (drawBin bin bad next).Avoids goal := by
  cases bin
  · exact each 0 (.inr ⟨rfl, empty rfl⟩)
  · exact fun ζ outside => each ζ (.inl outside)

omit [DecidableEq E] [Algebra Zq E] in
/-- A game after `drawBin` has the cost of the draw at a level with compression digits. -/
theorem drawBin_costLe {β : Type} (bin : Bool) {bad : Finset E} {next : E → Game β} {δ : ℝ}
    (card : bad.card ≤ 1) (each : ∀ ζ, (next ζ).CostLe δ) :
    (drawBin bin bad next).CostLe (((binDraws bin : ℕ) : ℝ) / Fintype.card E + δ) := by
  cases bin
  · have zero : ((binDraws false : ℕ) : ℝ) / Fintype.card E + δ = δ := by simp [binDraws]
    rw [zero]
    exact each 0
  · exact ⟨_, add_le_add_left (density_le _ 1 card) _, each⟩

/-- On a path of `foldCheck` that avoids the bad sets, an accepted claim `(r_2, v_W)` that is true
for the committed witness, `v_W = w̃(r_2)`, gives the relation of the ring reduction, the digit
range statement of Akita Section 6.1 and the binary statement of Akita Remark 6.2. -/
theorem foldCheck_avoids (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (empty : bin = false → ∀ x, ¬ support x) (w : Table) :
    (foldCheck R b support bin w).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w) →
        R.Holds w ∧ RangeHolds b R.vars w ∧ BinaryOn support R.vars w := by
  intro α reduce τ0 point τ1 batch
  refine (avoids_bind _ _ _).mpr ?_
  refine (rangeCheck_avoids b R.vars support w (rangePoint R.low τ0) (rangePoint_length _ _)).mono ?_
  rintro (_ | ⟨r, sClaim⟩) ranged
  · intro r2 v equal
    simp at equal
  obtain ⟨virt, rangeSound⟩ := ranged (r, sClaim) rfl
  refine drawBin_avoids empty fun ζ outsideζ γ outsideγ => ?_
  refine (fusedCheck_avoids R.vars support w _ _ r sClaim ζ γ).mono ?_
  intro result fused r2 v equal opening
  obtain ⟨relation, opened, binary⟩ := fused_sound virt
    ((R.rows α).batch_residual R.vars w (List.ofFn τ1)) outsideζ outsideγ
    (fused r2 v equal opening)
  exact ⟨R.sound w α reduce ((R.rows α).holds_of_batched R.vars batch
    (((R.rows α).batched_iff R.vars w _).mpr relation)),
    rangeHolds_of_sum point (rangeSound opened), binary⟩

/-- The sum of the bounds of the bad sets of `foldCheck`, in the order of the draws: the ring
reduction, the range point `τ_0`, the batching point `τ_1`, the range tree with one more value in
each round of its last stage for the binary-support point, the coefficient `ζ_bin` at a level
with compression digits, the coefficient `γ`, and the `μ'` rounds of the fused sum-check of
degree 3 (Akita (189)). -/
noncomputable def foldError (R : Reduction E) (b : ℕ) (bin : Bool) : ℝ :=
  (R.badBound : ℝ) / Fintype.card E + ((R.vars : ℝ) / Fintype.card E +
    ((R.batch : ℝ) / Fintype.card E +
      (((rangeNumerator R.vars b : ℕ) : ℝ) / Fintype.card E +
        (((binDraws bin : ℕ) : ℝ) / Fintype.card E + (((1 : ℕ) : ℝ) / Fintype.card E +
          (R.vars : ℝ) * (((3 : ℕ) : ℝ) / Fintype.card E))))))

omit [DecidableEq E] in
/-- The closed form of `foldError`. For the quotient-free reduction, with `badBound = d_max − 1`
and `batch = ⌈log n⌉`, it is the part of `ε_h` of Akita (189) after `ε_cw,h`, with `k_ring − 1`
for `k_ring` and `N^norm_h = 0`: `rangeNumerator` has the term `μ'` of the binary-support point
and the two sums of (187), and `binDraws` is the term `β_h` of the coefficient `ζ_bin`. -/
theorem foldError_eq (R : Reduction E) (b : ℕ) (bin : Bool) :
    foldError R b bin =
      ((R.badBound + R.batch + R.vars + rangeNumerator R.vars b + binDraws bin + 1 +
        3 * R.vars : ℕ) :
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
    (bin : Bool) (w : Table) : (foldCheck R b support bin w).CostLe (foldError R b bin) := by
  have reduce := density_le (C := E) (R.bad w) _ (R.bad_card w)
  have point := density_rangeBad_le (E := E) R.low b R.vars w
  unfold density at point
  refine ⟨_, add_le_add_left reduce _, fun α => ⟨_, add_le_add_left point _, fun τ0 => ?_⟩⟩
  have batch := (R.rows α).density_batchBad_le R.vars (R.fits α) w
  unfold density at batch
  refine ⟨_, add_le_add_left batch _, fun τ1 => ?_⟩
  refine costLe_bind (rangeCheck_costLe b R.vars support w (rangePoint R.low τ0)) ?_
  rintro (_ | ⟨r, sClaim⟩)
  · show (0 : ℝ) ≤ _
    positivity
  · exact drawBin_costLe bin (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))
      fun ζ => ⟨_, add_le_add_left (density_le _ 1
        (zeros_card_le_of_natDegree_le (linePoly_natDegree _ _))) _,
        fun γ => fusedCheck_costLe R.vars support w _ _ r sClaim ζ γ⟩

/-- For each prover, the probability that `foldCheck` ends in a claim `(r_2, v_W)` with
`v_W = w̃(r_2)` for a witness that does not satisfy the relation, the digit range statement or the
binary statement is at most `foldError`. -/
theorem foldCheck_error (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (empty : bin = false → ∀ x, ¬ support x) (w : Table)
    (s : Strategy (Option (List E × E))) :
    prob s (foldCheck R b support bin w) (fun result =>
      ¬ ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w) →
        R.Holds w ∧ RangeHolds b R.vars w ∧ BinaryOn support R.vars w) ≤ foldError R b bin :=
  prob_not_le s _ _ _ (foldCheck_avoids R b support bin empty w)
    (foldCheck_costLe R b support bin w)

end NightstreamFPrime.Spec.Compression.Akita
