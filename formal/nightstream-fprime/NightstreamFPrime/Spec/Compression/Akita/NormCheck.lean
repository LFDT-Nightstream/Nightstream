import NightstreamFPrime.Spec.Compression.Akita.FoldCheck

/-! The direct norm check of Akita for the Euclidean route, in the checks of one fold
(ePrint 2026/1983, revision of 2026-10-15, Sections 6.2 and 6.3 with (81) to (85) and (90) to
(93), Lemma 6.3, and Lemma 10.34 with (184)), in the order of the pinned Akita code
(`crates/akita-verifier/src/stages/physical_l2_norm.rs` and `stage1.rs`).

The integer coordinates `z^Z_u = Σ_a b_z^a·ŵ(π(u, a))` of (81) are a linear map of the committed
witness `w`: `normTable μ K w` with the kernel `K(u, x)` of the cells, on the cube of the `μ'`
variables of the witness. The norm sum (85), `Σ_u (z^Z_u)² = E_resp`, is in the last sum-check of
the range tree with the coefficient `ζ_norm` (90), and the value `z̃(r_virt)` is in the fused
sum-check (125) with the coefficient `η_norm` (91), (93).

- `normLeafStage` is the last stage of the range tree on the Euclidean route. The verifier
  receives `E_resp` and rejects unless `E_resp ≤ S_max`, draws `ζ_norm`, runs the sum-check for
  the claim `C_rng + ζ_norm·E_resp`, receives `z̃(r_virt)` and `s_claim`, and checks the final
  claim. `normLeafStage_avoids` is its soundness.
- `foldCheckSq` is `foldCheck` with this last stage, the draw of `η_norm` before `ζ_bin` and `γ`,
  and the term `η_norm·K_{r_virt}` in the weight of the fused sum-check.
  `foldCheckSq_avoids`: on a path that avoids the bad sets, a true final claim gives the
  statements of `foldCheck_avoids` and the sum (85) in `Z_q` for an `E_resp ≤ S_max`. The bad sets
  of `ζ_norm` and `η_norm` have one value each, which is the term `N^norm = 2` of Akita (189). -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- The table `z^Z` of a linear map of the witness: `Σ_{x < 2^μ} w(x)·K(u, x)` at `u`. -/
def normTable (μ : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) : Table :=
  fun u => ∑ x ∈ Finset.range (2 ^ μ), w x * kernel u x

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

/-- The weight `K_r(x) = Σ_{u < 2^ν} ẽq(r, u)·K(u, x)` of the cell `x` in `z̃(r)`
(Akita (91)). -/
noncomputable def normWeight (ν : ℕ) (kernel : ℕ → ℕ → Zq) (r : List E) (x : ℕ) : E :=
  ∑ u ∈ Finset.range (2 ^ ν), eqAt r u * algebraMap Zq E (kernel u x)

omit [Fintype E] [DecidableEq E] in
/-- `z̃(r) = Σ_x w(x)·K_r(x)` for a point `r` of `ν` coordinates. -/
theorem mle_normTable {ν μ : ℕ} (kernel : ℕ → ℕ → Zq) (w : Table) {r : List E}
    (length : r.length = ν) :
    mle r (lifted E (normTable μ kernel w)) =
      ∑ x ∈ Finset.range (2 ^ μ), lifted E w x * normWeight ν kernel r x := by
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range, length]
  simp only [lifted, normTable, normWeight, map_sum, map_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun u _ => by ring

/-- The tables of the last sum-check of the range tree on the Euclidean route: `ẽq(point, ·)`,
`s` and `z^Z`. -/
noncomputable def normLeafTables (μ : ℕ) (kernel : ℕ → ℕ → Zq) (point : List E) (w : Table) :
    Fin 3 → ℕ → E :=
  ![eqAt point, derived E w, lifted E (normTable μ kernel w)]

/-- The summand `P_rng + ζ_norm·z_int²` of Akita (90): the summand `leafComb` of the range tree
and `ζ_norm` times the square of the table of `z^Z` (variable 2). -/
noncomputable def normLeafComb (E : Type) [Field E] (leaf nodes : ℕ) (weights : ℕ → E) (ζ : E) :
    MvPolynomial (Fin 3) E :=
  MvPolynomial.rename Fin.castSucc (leafComb E leaf nodes weights) +
    MvPolynomial.C ζ * (MvPolynomial.X 2 * MvPolynomial.X 2)

omit [Fintype E] [DecidableEq E] [Algebra Zq E] in
theorem eval_normLeafComb (leaf nodes : ℕ) (weights : ℕ → E) (ζ : E) (f : Fin 3 → E) :
    MvPolynomial.eval f (normLeafComb E leaf nodes weights ζ) =
      f 0 * ∑ m ∈ Finset.range nodes, weights m * nodeIn leaf m (f 1) + ζ * (f 2 * f 2) := by
  simp [normLeafComb, MvPolynomial.eval_rename, eval_leafComb]

omit [Fintype E] [DecidableEq E] [Algebra Zq E] in
/-- The summand of (90) has the degree of the last summand of the range tree, and at least 2
(Akita Section 6.3). -/
theorem normLeafComb_degree (leaf nodes : ℕ) (weights : ℕ → E) (ζ : E) :
    (normLeafComb E leaf nodes weights ζ).totalDegree ≤ max (leaf + 1) 2 := by
  unfold normLeafComb
  refine (MvPolynomial.totalDegree_add _ _).trans (max_le_max ?_ ?_)
  · exact (MvPolynomial.totalDegree_rename_le _ _).trans (leafComb_degree leaf nodes weights)
  · refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_C, zero_add]
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_X]

omit [Fintype E] [DecidableEq E] in
/-- The claim of the sum-check of (90): `C_rng + ζ_norm·Σ_u (z^Z_u)²`. -/
theorem normLeafTruth_nil (leaf nodes : ℕ) (weights : ℕ → E) (ζ : E) (kernel : ℕ → ℕ → Zq)
    (point : List E) (w : Table) :
    combTruth point.length (normLeafTables point.length kernel point w)
        (normLeafComb E leaf nodes weights ζ) [] =
      ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E leaf m w) +
        ζ * ∑ x ∈ Finset.range (2 ^ point.length),
          lifted E (normTable point.length kernel w) x ^ 2 := by
  rw [combTruth_nil, cubeSum_eq_sum_range]
  simp only [eval_normLeafComb, Finset.sum_add_distrib]
  congr 1
  · simp_rw [mle_eq_cubeSum, cubeSum_eq_sum_range, Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun m _ => Finset.sum_congr rfl fun x _ => ?_
    simp only [normLeafTables, nodeTable, Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  · rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun x _ => ?_
    simp only [normLeafTables, Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons]
    ring

omit [Fintype E] [DecidableEq E] in
/-- The final claim of the sum-check of (90) at the point `r_virt`, from the values
`s̃(r_virt)` and `z̃(r_virt)`. -/
theorem normLeafTruth_final (leaf nodes : ℕ) (weights : ℕ → E) (ζ : E) (μ : ℕ)
    (kernel : ℕ → ℕ → Zq) (point r : List E) (w : Table) :
    combTruth r.length (normLeafTables μ kernel point w) (normLeafComb E leaf nodes weights ζ) r =
      mle r (eqAt point) * ∑ m ∈ Finset.range nodes,
        weights m * nodeIn leaf m (mle r (derived E w)) +
        ζ * (mle r (lifted E (normTable μ kernel w)) *
          mle r (lifted E (normTable μ kernel w))) := by
  rw [combTruth_final, eval_normLeafComb]
  rfl

omit [Fintype E] [DecidableEq E] in
theorem normLeaf_roundTruth (leaf nodes μ : ℕ) (weights : ℕ → E) (ζ : E)
    (kernel : ℕ → ℕ → Zq) (point : List E) (w : Table) :
    RoundTruth (max (leaf + 1) 2) μ
      (combTruth μ (normLeafTables μ kernel point w) (normLeafComb E leaf nodes weights ζ))
      (combRound μ (normLeafTables μ kernel point w) (normLeafComb E leaf nodes weights ζ)) :=
  combRound_truth μ _ _ _ (normLeafComb_degree leaf nodes weights ζ)

/-- The last stage of the range tree on the Euclidean route (Akita (90);
`verify_physical_l2_norm` of the Akita code, direct shape). The verifier receives `E_resp`,
rejects unless `E_resp ≤ S_max = limit`, draws `ζ_norm`, runs the sum-check for the claim
`C_rng + ζ_norm·E_resp`, receives the values `z̃(r_virt)` and `s_claim`, and checks the final
claim. The bad sets are the root in `ζ_norm` of the two residuals, and the roots of each round
with `bindBad` for the binary-support table. It ends in `(r_virt, s_claim, z̃(r_virt))`. -/
noncomputable def normLeafStage (leaf μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (nodes : ℕ) (weights : ℕ → E) (point : List E)
    (claim : E) : Game (Option (List E × E × E)) :=
  .recv ℕ fun claimed =>
    if claimed ≤ limit then
      .draw E (zeros (linePoly
          (claim - ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E leaf m w))
          ((claimed : E) - ∑ x ∈ Finset.range (2 ^ μ),
            lifted E (normTable μ kernel w) x ^ 2))) fun ζ =>
        (sumcheckWith (max (leaf + 1) 2) (combRound μ (normLeafTables μ kernel point w)
            (normLeafComb E leaf nodes weights ζ))
          (fun bound => bindBad (μ - bound.length - 1) (fold bound (supportTable support w)))
          μ [] (claim + ζ * claimed)).bind fun
          | none => .pure none
          | some (r, value) => .recv (E × E) fun values =>
              if value = mle r (eqAt point) * ∑ m ∈ Finset.range nodes,
                  weights m * nodeIn leaf m values.2 + ζ * (values.1 * values.1) then
                .pure (some (r, values.2, values.1))
              else .pure none
    else .pure none

/-- The claims of an end `(r_virt, s_claim, y)` of the range tree on the Euclidean route are
true: `s_claim = s̃(r_virt)` and `y = z̃(r_virt)`. -/
def OpensSq (μ : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) (out : List E × E × E) : Prop :=
  out.2.1 = mle out.1 (derived E w) ∧ out.2.2 = mle out.1 (lifted E (normTable μ kernel w))

/-- The properties of an end `(r_virt, s_claim, y)` of the range tree on the Euclidean route:
`Virt`, and for true claims the sum (85) in `Z_q` for a number `E_resp ≤ S_max`. -/
def VirtSq (μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) (out : List E × E × E) : Prop :=
  Virt μ support w (out.1, out.2.1) ∧ (OpensSq μ kernel w out → ∃ claimed ≤ limit,
    ∑ u ∈ Finset.range (2 ^ μ), normTable μ kernel w u ^ 2 = (claimed : Zq))

/-- The soundness of the last stage on the Euclidean route (Akita Lemma 10.34 for the merge with
`ζ_norm`): on a path that avoids the bad sets, an end has `VirtSq`, and for true claims the
claim of the stage is the batched sum of the leaf tables. -/
theorem normLeafStage_avoids (leaf μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (nodes : ℕ) (weights : ℕ → E) (point : List E)
    (claim : E) (long : point.length = μ) :
    (normLeafStage leaf μ limit kernel support w nodes weights point claim).Avoids fun result =>
      ∀ out, result = some out → VirtSq μ limit kernel support w out ∧
        (OpensSq μ kernel w out → claim = ∑ m ∈ Finset.range nodes,
          weights m * mle point (nodeTable E leaf m w)) := by
  subst long
  intro claimed
  dsimp only
  split
  swap
  · intro out equal
    simp at equal
  rename_i small
  intro ζ outside
  refine (Game.avoids_bind _ _ _).mpr ?_
  refine (sumcheckWith_avoids (max (leaf + 1) 2) point.length _ _
    (normLeaf_roundTruth leaf nodes point.length weights ζ kernel point w)
    (fun bound => bindBad (point.length - bound.length - 1) (fold bound (supportTable support w)))
    (fun bound => NonzeroOn (point.length - bound.length) (fold bound (supportTable support w)))
    (fun bound c inside nonzero outside => by
      rw [fold_append_single, List.length_append, List.length_singleton,
        show point.length - (bound.length + 1) = point.length - bound.length - 1 by omega]
      exact nonzeroOn_bindLow
        (by rwa [show point.length - bound.length - 1 + 1 = point.length - bound.length by omega])
        outside)
    point.length [] _ (by simp)).mono ?_
  rintro (_ | ⟨r, value⟩) ranged
  · intro out equal
    simp at equal
  intro values
  dsimp only
  split
  swap
  · intro out equal
    simp at equal
  rename_i checked
  rintro ⟨r', s, y⟩ equal
  simp only [Option.some.injEq, Prod.mk.injEq] at equal
  obtain ⟨rfl, rfl, rfl⟩ := equal
  obtain ⟨length, staged, kept⟩ := ranged r value rfl
  simp only [List.length_nil, zero_add] at length
  have merged (opens : OpensSq point.length kernel w (r, values.2, values.1)) :
      claim - ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E leaf m w) = 0 ∧
        (claimed : E) - ∑ x ∈ Finset.range (2 ^ point.length),
          lifted E (normTable point.length kernel w) x ^ 2 = 0 := by
    obtain ⟨range, norm⟩ : values.2 = mle r (derived E w) ∧
        values.1 = mle r (lifted E (normTable point.length kernel w)) := opens
    refine eq_zero_of_not_mem_zeros_linePoly outside ?_
    have true_claim := not_imp_not.mp staged (by
      have final := normLeafTruth_final leaf nodes weights ζ point.length kernel point r w
      rw [length] at final
      rw [final, checked, range, norm])
    rw [normLeafTruth_nil] at true_claim
    linear_combination true_claim
  refine ⟨⟨⟨length, fun nonzero => ?_⟩, fun opens => ⟨claimed, small,
    (algebraMap Zq E).injective ?_⟩⟩, fun opens => sub_eq_zero.mp (merged opens).1⟩
  · have cell := kept (by simpa [fold] using nonzero)
    rw [length, Nat.sub_self] at cell
    obtain ⟨i, lt, value⟩ := cell
    obtain rfl : i = 0 := by simpa using lt
    exact value
  · rw [map_sum, map_natCast, sub_eq_zero.mp (merged opens).2]
    exact Finset.sum_congr rfl fun u _ => by simp only [lifted, map_pow]

/-- The checks of one fold on the Euclidean route for the flat witness `w` of the next level
(Akita Figure 6 with Section 6.3): `foldCheck` with the last stage `normLeafStage` of the range
tree, the draw of `η_norm` before `ζ_bin` and `γ`, and the fused sum-check (125) with the weight
`m_{τ_1} + η_norm·K_{r_virt}` and the claim `V + η_norm·z̃(r_virt)` of Akita (93). The bad set of
`η_norm` is the root of the residual of the relation and the residual of (91). -/
noncomputable def foldCheckSq (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) :
    Game (Option (List E × E)) :=
  .draw E (R.bad w) fun α =>
  .draw (Fin R.vars → E) (rangeBad b R.vars w) fun τ0 =>
  .draw (Fin R.batch → E) ((R.rows α).batchBad R.vars R.batch w) fun τ1 =>
  (rangeTree (rangeLeaf b) R.vars w (normLeafStage (rangeLeaf b) R.vars limit kernel support w)
    (rangeArities b) 1 (fun _ => 1) (List.ofFn τ0) 0).bind fun
    | none => .pure none
    | some (r, sClaim, y) =>
        .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w))
          (mle r (lifted E (normTable R.vars kernel w)) - y))) fun η =>
        drawBin bin (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
          η * (mle r (lifted E (normTable R.vars kernel w)) - y))
          (mle r (supportTable support w)))) fun ζ =>
        .draw E (zeros (linePoly (mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
          η * (mle r (lifted E (normTable R.vars kernel w)) - y) +
          ζ * mle r (supportTable support w)) (mle r (derived E w) - sClaim))) fun γ =>
          fusedCheck R.vars support w
            (fun x => (R.rows α).batchWeight (List.ofFn τ1) x + η * normWeight R.vars kernel r x)
            ((R.rows α).batchTarget (List.ofFn τ1) + η * y) r sClaim ζ γ

/-- On a path of `foldCheckSq` that avoids the bad sets, an accepted claim `(r_2, v_W)` that is
true for the committed witness, `v_W = w̃(r_2)`, gives the relation of the ring reduction, the
digit range statement, the binary statement, and a number `E_resp ≤ S_max` with
`Σ_{u < 2^{μ'}} (z^Z_u)² = E_resp` in `Z_q` (Akita Lemma 6.3 in the field and Lemma 10.34). -/
theorem foldCheckSq_avoids (R : Reduction E) (b : ℕ) (support : ℕ → Prop)
    [DecidablePred support] (bin : Bool) (empty : bin = false → ∀ x, ¬ support x) (limit : ℕ)
    (kernel : ℕ → ℕ → Zq) (w : Table) :
    (foldCheckSq R b support bin limit kernel w).Avoids fun result =>
      ∀ r2 v, result = some (r2, v) → v = mle r2 (lifted E w) →
        (R.Holds w ∧ RangeHolds b R.vars w ∧ BinaryOn support R.vars w) ∧ ∃ claimed ≤ limit,
          ∑ u ∈ Finset.range (2 ^ R.vars), normTable R.vars kernel w u ^ 2 = (claimed : Zq) := by
  intro α reduce τ0 point τ1 batch
  refine (avoids_bind _ _ _).mpr ?_
  refine (rangeTree_avoids (rangeLeaf b) R.vars w _ (VirtSq R.vars limit kernel support w)
    (OpensSq R.vars kernel w)
    (fun nodes weights point claim => normLeafStage_avoids (rangeLeaf b) R.vars limit kernel
      support w nodes weights point claim) (rangeArities b) 1 (fun _ => 1) (List.ofFn τ0) 0
    List.length_ofFn).mono ?_
  rintro (_ | ⟨r, sClaim, y⟩) ranged
  · intro r2 v equal
    simp at equal
  obtain ⟨⟨virt, energy⟩, rangeSound⟩ := ranged (r, sClaim, y) rfl
  intro η outsideη
  refine drawBin_avoids empty fun ζ outsideζ γ outsideγ => ?_
  refine (fusedCheck_avoids R.vars support w _ _ r sClaim ζ γ).mono ?_
  intro result fused r2 v equal opening
  have batched : ∑ x ∈ Finset.range (2 ^ R.vars), lifted E w x *
      ((R.rows α).batchWeight (List.ofFn τ1) x + η * normWeight R.vars kernel r x) -
        ((R.rows α).batchTarget (List.ofFn τ1) + η * y) =
      mle (List.ofFn τ1) ((R.rows α).residual R.vars w) +
        η * (mle r (lifted E (normTable R.vars kernel w)) - y) := by
    have length : r.length = R.vars := virt.1
    have scaled (x : ℕ) : algebraMap Zq E (w x) * (η * normWeight R.vars kernel r x) =
        η * (algebraMap Zq E (w x) * normWeight R.vars kernel r x) := by ring
    rw [← (R.rows α).batch_residual R.vars w (List.ofFn τ1), mle_normTable kernel w length]
    simp only [lifted, mul_add, Finset.sum_add_distrib, scaled, ← Finset.mul_sum]
    ring
  obtain ⟨relation, opened, binary⟩ := fused_sound virt batched outsideζ outsideγ
    (fused r2 v equal opening)
  obtain ⟨holds, bound⟩ := eq_zero_of_not_mem_zeros_linePoly outsideη relation
  have opens : OpensSq R.vars kernel w (r, sClaim, y) := ⟨opened, (sub_eq_zero.mp bound).symm⟩
  refine ⟨⟨R.sound w α reduce ((R.rows α).holds_of_batched R.vars batch
    (((R.rows α).batched_iff R.vars w _).mpr holds)), rangeHolds_of_sum point ?_, binary⟩,
    energy opens⟩
  rw [← root_sum b w (List.ofFn τ0)]
  exact (rangeSound opens).symm

end NightstreamFPrime.Spec.Compression.Akita
