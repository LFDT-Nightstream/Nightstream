import NightstreamFPrime.Spec.Compression.Akita.FoldCheck

/-! The direct norm check of Akita for the Euclidean route, as a game after the checks of one fold
(ePrint 2026/1983, revision of 2026-10-15, Section 6.2 with (81) to (93), Lemma 6.3, and
Lemma 10.34 with (184)).

The integer coordinates `z^Z_u = Σ_a b_z^a·ŵ(π(u, a))` of (81) are a linear map of the committed
witness `w`: `normTable μ K w` with the kernel `K(u, x)` of the cells. The check proves the sum
(85), `Σ_u (z^Z_u)² = E_resp`, in the field for a claimed integer `E_resp ≤ S_max`, and it binds
the evaluation claim of the sum-check of (85) with the claim `(r_2, v_W)` of the checks of the
fold.

- `normCheck` is the verifier. It rejects unless `|r_2| = μ`, receives `E_resp` and rejects
  unless `E_resp ≤ S_max`, runs the sum-check of degree 2 for `Σ_u z̃(u)² = E_resp` to the point
  `r_u`, receives `y = z̃(r_u)` and checks the final claim `y²`. It draws `η`, runs the sum-check
  of degree 2 for `Σ_x w(x)·(ẽq(r_2, x) + η·K_{r_u}(x)) = v_W + η·y`, receives `v` and checks the
  final claim. It ends in the claim `(r_3, v)` of `w̃(r_3)`.
- `normCheck_avoids`: on a path that avoids the bad sets, a true final claim `v = w̃(r_3)` gives
  the claim `v_W = w̃(r_2)` and the sum (85) in `Z_q` for an `E_resp ≤ S_max`. The bad sets are
  the roots of each round and the root of the binding in `η`.

Akita fuses (85) into the sum-check (125) of the fold with the coefficients `ζ_norm` and
`η_norm`, with the error `2/|E|` of Lemma 10.34 beside the errors of the rounds. This file runs
two more sum-checks after the checks of the fold, with `ν + μ` more rounds of degree 2 and one
more draw `η`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- The table `z^Z` of a linear map of the witness: `Σ_{x < 2^μ} w(x)·K(u, x)` at `u`. -/
def normTable (μ : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) : Table :=
  fun u => ∑ x ∈ Finset.range (2 ^ μ), w x * kernel u x

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

/-- The weight `K_{r_u}(x) = Σ_{u < 2^ν} ẽq(r_u, u)·K(u, x)` of the cell `x` in `z̃(r_u)`. -/
noncomputable def normWeight (ν : ℕ) (kernel : ℕ → ℕ → Zq) (r : List E) (x : ℕ) : E :=
  ∑ u ∈ Finset.range (2 ^ ν), eqAt r u * algebraMap Zq E (kernel u x)

omit [Fintype E] [DecidableEq E] in
/-- `z̃(r_u) = Σ_x w(x)·K_{r_u}(x)` for a point `r_u` of `ν` coordinates. -/
theorem mle_normTable {ν μ : ℕ} (kernel : ℕ → ℕ → Zq) (w : Table) {r : List E}
    (length : r.length = ν) :
    mle r (lifted E (normTable μ kernel w)) =
      ∑ x ∈ Finset.range (2 ^ μ), lifted E w x * normWeight ν kernel r x := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length]
  simp only [lifted, normTable, normWeight, map_sum, map_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun u _ => by ring

/-- The tables of the sum-check of (85): `z^Z` twice. -/
noncomputable def normTables (μ : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) : Fin 2 → ℕ → E :=
  ![lifted E (normTable μ kernel w), lifted E (normTable μ kernel w)]

/-- The tables of the binding sum-check: `w` and `ẽq(r_2, ·) + η·K_{r_u}`. -/
noncomputable def bindTables (ν : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) (r2 ru : List E)
    (η : E) : Fin 2 → ℕ → E :=
  ![lifted E w, fun x => eqAt r2 x + η * normWeight ν kernel ru x]

/-- The direct norm check for the witness `w` with `μ` variables, the integer coordinates
`normTable μ K w` on `ν` variables, the cap `S_max = limit` and the claim `v_W = w̃(r_2)` of the
checks of the fold (Akita Section 6.2). -/
noncomputable def normCheck (ν μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) (r2 : List E)
    (v : E) : Game (Option (List E × E)) :=
  if r2.length = μ then
    .recv ℕ fun claimed =>
      if claimed ≤ limit then
        (sumcheck 2 (combRound ν (normTables μ kernel w) productComb) ν [] (claimed : E)).bind fun
          | none => .pure none
          | some (ru, value) => .recv E fun y =>
              if value = y * y then
                .draw E (zeros (linePoly (v - mle r2 (lifted E w))
                    (y - mle ru (lifted E (normTable μ kernel w))))) fun η =>
                  (sumcheck 2 (combRound μ (bindTables ν kernel w r2 ru η) productComb) μ []
                    (v + η * y)).bind fun
                    | none => .pure none
                    | some (r3, final) => .recv E fun v3 =>
                        if final = v3 * (mle r3 (eqAt r2) + η * mle r3 (normWeight ν kernel ru))
                        then .pure (some (r3, v3))
                        else .pure none
              else .pure none
      else .pure none
  else .pure none

/-- The soundness of the direct norm check (Akita Lemma 6.3 in the field and Lemma 10.34): on a
path that avoids the bad sets, an accepted claim `(r_3, v)` that is true for the witness,
`v = w̃(r_3)`, gives the claim `v_W = w̃(r_2)` and a claimed `E_resp ≤ S_max` with
`Σ_{u < 2^ν} (z^Z_u)² = E_resp` in `Z_q`. -/
theorem normCheck_avoids (ν μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) (r2 : List E)
    (v : E) :
    (normCheck ν μ limit kernel w r2 v).Avoids fun result =>
      ∀ r3 v3, result = some (r3, v3) → v3 = mle r3 (lifted E w) →
        v = mle r2 (lifted E w) ∧ ∃ claimed ≤ limit,
          ∑ u ∈ Finset.range (2 ^ ν), normTable μ kernel w u ^ 2 = (claimed : Zq) := by
  unfold normCheck
  split
  swap
  · intro r3 v3 equal
    simp at equal
  rename_i length
  intro claimed
  dsimp only
  split
  swap
  · intro r3 v3 equal
    simp at equal
  rename_i small
  refine (avoids_bind _ _ _).mpr ?_
  refine ((sumcheck_avoids 2 ν _ _ (combRound_truth ν 2 (normTables μ kernel w) productComb
    productComb_degree) ν [] _ (by simp)).and_ends (sumcheck_ends 2 _ ν [] _)).mono ?_
  rintro (_ | ⟨ru, value⟩) ⟨normSound, normEnds⟩
  · intro r3 v3 equal
    simp at equal
  intro y
  dsimp only
  split
  swap
  · intro r3 v3 equal
    simp at equal
  rename_i squared
  intro η outside
  refine (avoids_bind _ _ _).mpr ?_
  refine ((sumcheck_avoids 2 μ _ _ (combRound_truth μ 2 (bindTables ν kernel w r2 ru η)
    productComb productComb_degree) μ [] _ (by simp)).and_ends (sumcheck_ends 2 _ μ [] _)).mono ?_
  rintro (_ | ⟨r3, final⟩) ⟨bindSound, bindEnds⟩
  · intro r3 v3 equal
    simp at equal
  intro v3
  dsimp only
  split
  swap
  · intro r3 v3 equal
    simp at equal
  rename_i checked
  intro r3' v3' equal opening
  simp only [Option.some.injEq, Prod.mk.injEq] at equal
  obtain ⟨rfl, rfl⟩ := equal
  have count : r3.length = μ := by
    obtain ⟨cs, prefixed, count⟩ := bindEnds r3 final rfl
    rw [prefixed]
    simpa using count
  have countU : ru.length = ν := by
    obtain ⟨cs, prefixed, count⟩ := normEnds ru value rfl
    rw [prefixed]
    simpa using count
  have bound := not_imp_not.mp (bindSound r3 final rfl) (by
    have last := product_truth_final r3 (bindTables ν kernel w r2 ru η)
    rw [count] at last
    rw [last, checked, opening]
    simp only [bindTables, Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [mle_add, mle_mul_left])
  rw [product_truth_nil] at bound
  have split : cubeSum μ (fun i => bindTables ν kernel w r2 ru η 0 i *
      bindTables ν kernel w r2 ru η 1 i) =
      mle r2 (lifted E w) + η * mle ru (lifted E (normTable μ kernel w)) := by
    rw [mle_eq_cubeSum r2, length, mle_normTable kernel w countU, cubeSum_eq_sum_range,
      cubeSum_eq_sum_range, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [bindTables, Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  obtain ⟨first, second⟩ := eq_zero_of_not_mem_zeros_linePoly outside
    (by linear_combination bound.trans split)
  have energy := not_imp_not.mp (normSound ru value rfl) (by
    have last := product_truth_final ru (normTables μ kernel w)
    rw [countU] at last
    rw [last, squared, sub_eq_zero.mp second]
    rfl)
  rw [product_truth_nil] at energy
  refine ⟨(sub_eq_zero.mp first), claimed, small, (algebraMap Zq E).injective ?_⟩
  rw [map_sum, map_natCast, energy, cubeSum_eq_sum_range]
  refine Finset.sum_congr rfl fun u _ => ?_
  simp only [normTables, Matrix.cons_val_zero, Matrix.cons_val_one, lifted, pow_two, map_mul]

end NightstreamFPrime.Spec.Compression.Akita
