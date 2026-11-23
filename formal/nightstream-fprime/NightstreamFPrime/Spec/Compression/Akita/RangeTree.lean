import NightstreamFPrime.Spec.Compression.Akita.EqSumCheck

/-! The product tree of the digit range check of Akita (ePrint 2026/1983, revision of 2026-11-23,
Section 6.1 and Lemma 10.37 with (187)), in the order of the pinned Akita code
(`crates/akita-verifier/src/stages/stage1.rs`, `crates/akita-params/src/layout/digit_range.rs`).

- `rangeArities b⋆` has the product degrees of the levels before the leaves, and `rangeLeaf b⋆` is
  the number of the factors of one leaf polynomial. `rangeArities_prod_mul_leaf` states that the
  tree has the `b⋆/2` factors of `Q_sq`.
- `nodeIn size m s` is the value of node `m` of a level whose nodes have `size` factors each, and
  `nodeIn_mul` states that a node is the product of its children. `nodeTable` is its table on the
  cube.
- `stageComb` and `stageTables` are the summand of the sum-check of one product level, and
  `leafComb` is the summand of the last sum-check. Each is `ẽq(point, ·)` times `stageData` or
  `leafData`, and `stage_eqRoundTruth` and `leaf_eqRoundTruth` give their normalized round
  polynomials (Akita Section 3.5). `rowPoly` is the polynomial of the batching of the child claims
  with the powers of one challenge.
- `rangeTree` is the verifier for a list of product degrees and a last stage `last`. Each stage is
  the equality-factored sum-check `eqSumcheck` with the messages and the normalized end check of
  the code (`verify_eq_factored_sumcheck_rounds`). `rangeTree_avoids` is its soundness: on a path
  that avoids the bad sets, when the claims of the last stage are true, the first claim is the
  batched sum of the node tables. `rangeTree_costLe` bounds the bad sets by `treeNumerator`,
  which is (187).
- `leafStage` is the last stage on the coefficient route, with the bad set of the binary-support
  point of Akita Lemma 10.37 (ii). `rangeCheck` is the tree for the base `b⋆` from the range point
  `τ_0`, with `rangeCheck_avoids` and `rangeCheck_costLe`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial
open NightstreamFPrime.Spec.Compression (Game)

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- The product degrees of the levels of the range tree before the leaves, from the root, for the
base `b⋆` (Akita Section 6.1, the table of the shapes; `DigitRangePlan::product_stage_arities` of
the Akita code). A base that the code does not have has no product level. -/
def rangeArities (b : ℕ) : List ℕ :=
  if b = 16 then [2] else if b = 32 then [4] else if b = 64 then [2, 4] else []

/-- The number of the factors `s − j(j + 1)` of one leaf polynomial of the range tree. -/
def rangeLeaf (b : ℕ) : ℕ := b / 2 / (rangeArities b).prod

/-- The leaves of the range tree have the `b⋆/2` factors of `Q_sq`. -/
theorem rangeArities_prod_mul_leaf (b : ℕ) : (rangeArities b).prod * rangeLeaf b = b / 2 := by
  unfold rangeLeaf rangeArities
  split_ifs with sixteen thirtyTwo sixtyFour
  · subst sixteen; rfl
  · subst thirtyTwo; rfl
  · subst sixtyFour; rfl
  · simp

section Nodes

variable {E : Type} [Field E] [Algebra Zq E]

/-- The value of node `m` of a level of the range tree whose nodes have `size` factors each:
`Π_{j < size} (s − k(k + 1))` for `k = m·size + j` (Akita Section 6.1). -/
def nodeIn (size m : ℕ) (s : E) : E :=
  ∏ j ∈ Finset.range size, (s - ((m * size + j : ℕ) : E) * (((m * size + j : ℕ) : E) + 1))

omit [Algebra Zq E] in
/-- The root of the range tree is `Q_sq`. -/
theorem nodeIn_root (b : ℕ) (s : E) : nodeIn (b / 2) 0 s = rangeSqIn b s := by
  simp [nodeIn, rangeSqIn]

omit [Algebra Zq E] in
theorem prod_range_mul (f : ℕ → E) (a size start : ℕ) :
    ∏ j ∈ Finset.range (a * size), f (start + j) =
      ∏ t ∈ Finset.range a, ∏ j ∈ Finset.range size, f (start + t * size + j) := by
  induction a with
  | zero => simp
  | succ a step =>
      rw [Nat.succ_mul, Finset.prod_range_add, step, Finset.prod_range_succ]
      simp only [Nat.add_assoc]

omit [Algebra Zq E] in
/-- A node of the range tree is the product of its `a` children (Akita Section 6.1). -/
theorem nodeIn_mul (a size m : ℕ) (s : E) :
    nodeIn (a * size) m s = ∏ t ∈ Finset.range a, nodeIn size (m * a + t) s := by
  unfold nodeIn
  rw [prod_range_mul (fun k => s - ((k : ℕ) : E) * (((k : ℕ) : E) + 1)) a size (m * (a * size))]
  refine Finset.prod_congr rfl fun t _ => Finset.prod_congr rfl fun j _ => ?_
  rw [show m * (a * size) + t * size + j = (m * a + t) * size + j by ring]

/-- The table of node `m` of a level of the range tree on the cube. -/
def nodeTable (E : Type) [Field E] [Algebra Zq E] (size m : ℕ) (w : Table) (x : ℕ) : E :=
  nodeIn size m (derived E w x)

/-- The tables of the sum-check of one product level: `ẽq(point, ·)` and the tables of the
children. -/
def stageTables (size : ℕ) (point : List E) (w : Table) : Option ℕ → ℕ → E
  | none => eqAt point
  | some child => nodeTable E size child w

/-- The summand of the sum-check of one product level with `nodes` nodes of `a` children each:
`ẽq·Σ_m weight_m·Π_{t<a} child_{m·a+t}`. -/
noncomputable def stageComb (E : Type) [Field E] (a nodes : ℕ) (weights : ℕ → E) :
    MvPolynomial (Option ℕ) E :=
  MvPolynomial.X none * ∑ m ∈ Finset.range nodes,
    MvPolynomial.C (weights m) * ∏ t ∈ Finset.range a, MvPolynomial.X (some (m * a + t))

omit [Algebra Zq E] in
theorem eval_stageComb (a nodes : ℕ) (weights : ℕ → E) (f : Option ℕ → E) :
    MvPolynomial.eval f (stageComb E a nodes weights) =
      f none * ∑ m ∈ Finset.range nodes, weights m * ∏ t ∈ Finset.range a, f (some (m * a + t)) := by
  simp [stageComb, map_sum, map_prod]

/-- The factor `Σ_m weight_m·Π_{t<a} child_{m·a+t}` of `stageComb` after `ẽq`. -/
noncomputable def stageData (E : Type) [Field E] (a nodes : ℕ) (weights : ℕ → E) :
    MvPolynomial (Option ℕ) E :=
  ∑ m ∈ Finset.range nodes,
    MvPolynomial.C (weights m) * ∏ t ∈ Finset.range a, MvPolynomial.X (some (m * a + t))

omit [Algebra Zq E] in
theorem stageComb_eq (a nodes : ℕ) (weights : ℕ → E) :
    stageComb E a nodes weights = MvPolynomial.X none * stageData E a nodes weights := rfl

omit [Algebra Zq E] in
theorem eval_stageData (a nodes : ℕ) (weights : ℕ → E) (f : Option ℕ → E) :
    MvPolynomial.eval f (stageData E a nodes weights) =
      ∑ m ∈ Finset.range nodes, weights m * ∏ t ∈ Finset.range a, f (some (m * a + t)) := by
  simp [stageData, map_sum, map_prod]

omit [Algebra Zq E] in
theorem stageData_degree (a nodes : ℕ) (weights : ℕ → E) :
    (stageData E a nodes weights).totalDegree ≤ a := by
  refine MvPolynomial.totalDegree_finsetSum_le fun m _ => ?_
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_C, zero_add]
  refine (MvPolynomial.totalDegree_finsetProd _ _).trans ?_
  simp [MvPolynomial.totalDegree_X]

omit [Algebra Zq E] in
/-- The summand of a product level of degree `a` has the degree `a + 1` (Akita (187)). -/
theorem stageComb_degree (a nodes : ℕ) (weights : ℕ → E) :
    (stageComb E a nodes weights).totalDegree ≤ a + 1 := by
  rw [stageComb_eq]
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_X, add_comm]
  exact Nat.add_le_add_right (stageData_degree a nodes weights) 1

/-- The claim of the sum-check of one product level: the batched sum of the node tables at the
point of the level before. -/
theorem stageTruth_nil (a size nodes : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    combTruth point.length (stageTables size point w) (stageComb E a nodes weights) [] =
      ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E (a * size) m w) := by
  rw [combTruth_nil, cubeSum_eq_sum_range]
  simp_rw [mle_eq_cubeSum, cubeSum_eq_sum_range, eval_stageComb, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun m _ => Finset.sum_congr rfl fun x _ => ?_
  simp only [stageTables, nodeTable, nodeIn_mul]
  ring

/-- The final claim of the sum-check of one product level, from the values of the child tables
at the final point. -/
theorem stageTruth_final (a size nodes : ℕ) (weights : ℕ → E) (point r : List E) (w : Table) :
    combTruth r.length (stageTables size point w) (stageComb E a nodes weights) r =
      mle r (eqAt point) * ∑ m ∈ Finset.range nodes,
        weights m * ∏ t ∈ Finset.range a, mle r (nodeTable E size (m * a + t) w) := by
  rw [combTruth_final, eval_stageComb]
  rfl

theorem stage_roundTruth (a size nodes μ : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    RoundTruth (a + 1) μ (combTruth μ (stageTables size point w) (stageComb E a nodes weights))
      (combRound μ (stageTables size point w) (stageComb E a nodes weights)) :=
  combRound_truth μ _ _ _ (stageComb_degree a nodes weights)

/-- The normalized round polynomials of the sum-check of one product level (Akita Section 3.5). -/
theorem stage_eqRoundTruth (a size nodes : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    EqRoundTruth a point.length point
      (eqTruth point.length point (stageTables size point w) (stageData E a nodes weights))
      (eqRound point.length point (stageTables size point w) (stageData E a nodes weights)) :=
  eqRound_truth a point _ _ (stageData_degree a nodes weights)

/-- The normalized claim of the sum-check of one product level is its claim. -/
theorem stage_eqTruth_nil (a size nodes : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    eqTruth point.length point (stageTables size point w) (stageData E a nodes weights) [] =
      ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E (a * size) m w) := by
  rw [eqTruth_nil _ _ _ _ none rfl, ← stageComb_eq, stageTruth_nil]

/-- The normalized final claim of the sum-check of one product level, from the values of the
child tables at the final point. -/
theorem stage_eqTruth_final (a size nodes : ℕ) (weights : ℕ → E) (point r : List E) (w : Table)
    (length : r.length = point.length) :
    eqTruth point.length point (stageTables size point w) (stageData E a nodes weights) r =
      ∑ m ∈ Finset.range nodes,
        weights m * ∏ t ∈ Finset.range a, mle r (nodeTable E size (m * a + t) w) := by
  rw [eqTruth_final _ _ _ r length, eval_stageData]
  rfl

/-- The summand of the last sum-check of the range tree, for `nodes` leaf polynomials of `leaf`
factors each: `ẽq·Σ_m weight_m·Π_{j<leaf} (s − k(k + 1))` in the table of `ẽq` (variable 0) and
the table of `s` (variable 1). -/
noncomputable def leafComb (E : Type) [Field E] (leaf nodes : ℕ) (weights : ℕ → E) :
    MvPolynomial (Fin 2) E :=
  MvPolynomial.X 0 * ∑ m ∈ Finset.range nodes, MvPolynomial.C (weights m) *
    ∏ j ∈ Finset.range leaf, (MvPolynomial.X 1 -
      MvPolynomial.C (((m * leaf + j : ℕ) : E) * (((m * leaf + j : ℕ) : E) + 1)))

omit [Algebra Zq E] in
theorem eval_leafComb (leaf nodes : ℕ) (weights : ℕ → E) (f : Fin 2 → E) :
    MvPolynomial.eval f (leafComb E leaf nodes weights) =
      f 0 * ∑ m ∈ Finset.range nodes, weights m * nodeIn leaf m (f 1) := by
  simp [leafComb, nodeIn, map_sum, map_prod]

/-- The factor `Σ_m weight_m·Π_{j<leaf} (s − k(k + 1))` of `leafComb` after `ẽq`. -/
noncomputable def leafData (E : Type) [Field E] (leaf nodes : ℕ) (weights : ℕ → E) :
    MvPolynomial (Fin 2) E :=
  ∑ m ∈ Finset.range nodes, MvPolynomial.C (weights m) *
    ∏ j ∈ Finset.range leaf, (MvPolynomial.X 1 -
      MvPolynomial.C (((m * leaf + j : ℕ) : E) * (((m * leaf + j : ℕ) : E) + 1)))

omit [Algebra Zq E] in
theorem leafComb_eq (leaf nodes : ℕ) (weights : ℕ → E) :
    leafComb E leaf nodes weights = MvPolynomial.X 0 * leafData E leaf nodes weights := rfl

omit [Algebra Zq E] in
theorem eval_leafData (leaf nodes : ℕ) (weights : ℕ → E) (f : Fin 2 → E) :
    MvPolynomial.eval f (leafData E leaf nodes weights) =
      ∑ m ∈ Finset.range nodes, weights m * nodeIn leaf m (f 1) := by
  simp [leafData, nodeIn, map_sum, map_prod]

omit [Algebra Zq E] in
theorem leafData_degree (leaf nodes : ℕ) (weights : ℕ → E) :
    (leafData E leaf nodes weights).totalDegree ≤ leaf := by
  refine MvPolynomial.totalDegree_finsetSum_le fun m _ => ?_
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_C, zero_add]
  refine (MvPolynomial.totalDegree_finsetProd _ _).trans ?_
  calc ∑ j ∈ Finset.range leaf, (MvPolynomial.X 1 - MvPolynomial.C
          (((m * leaf + j : ℕ) : E) * (((m * leaf + j : ℕ) : E) + 1)) :
            MvPolynomial (Fin 2) E).totalDegree
      ≤ ∑ _j ∈ Finset.range leaf, 1 := Finset.sum_le_sum fun j _ =>
        (MvPolynomial.totalDegree_sub _ _).trans (max_le
          (by rw [MvPolynomial.totalDegree_X]) (by rw [MvPolynomial.totalDegree_C]; omega))
    _ = leaf := by simp

omit [Algebra Zq E] in
/-- The summand of the last sum-check has the degree `leaf + 1` (Akita (187)). -/
theorem leafComb_degree (leaf nodes : ℕ) (weights : ℕ → E) :
    (leafComb E leaf nodes weights).totalDegree ≤ leaf + 1 := by
  rw [leafComb_eq]
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_X, add_comm]
  exact Nat.add_le_add_right (leafData_degree leaf nodes weights) 1

/-- The claim of the last sum-check: the batched sum of the leaf tables. -/
theorem leafTruth_nil (leaf nodes : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    combTruth point.length (rangeTables point w) (leafComb E leaf nodes weights) [] =
      ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E leaf m w) := by
  rw [combTruth_nil, cubeSum_eq_sum_range]
  simp_rw [mle_eq_cubeSum, cubeSum_eq_sum_range, eval_leafComb, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun m _ => Finset.sum_congr rfl fun x _ => ?_
  simp only [rangeTables, nodeTable, Matrix.cons_val_zero, Matrix.cons_val_one]
  ring

/-- The final claim of the last sum-check at the point `r_virt`, from the value `s̃(r_virt)`. -/
theorem leafTruth_final (leaf nodes : ℕ) (weights : ℕ → E) (point r : List E) (w : Table) :
    combTruth r.length (rangeTables point w) (leafComb E leaf nodes weights) r =
      mle r (eqAt point) * ∑ m ∈ Finset.range nodes,
        weights m * nodeIn leaf m (mle r (derived E w)) := by
  rw [combTruth_final, eval_leafComb]
  rfl

theorem leaf_roundTruth (leaf nodes μ : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    RoundTruth (leaf + 1) μ (combTruth μ (rangeTables point w) (leafComb E leaf nodes weights))
      (combRound μ (rangeTables point w) (leafComb E leaf nodes weights)) :=
  combRound_truth μ _ _ _ (leafComb_degree leaf nodes weights)

/-- The normalized round polynomials of the last sum-check (Akita Section 3.5). -/
theorem leaf_eqRoundTruth (leaf nodes : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    EqRoundTruth leaf point.length point
      (eqTruth point.length point (rangeTables point w) (leafData E leaf nodes weights))
      (eqRound point.length point (rangeTables point w) (leafData E leaf nodes weights)) :=
  eqRound_truth leaf point _ _ (leafData_degree leaf nodes weights)

theorem leaf_eqTruth_nil (leaf nodes : ℕ) (weights : ℕ → E) (point : List E) (w : Table) :
    eqTruth point.length point (rangeTables point w) (leafData E leaf nodes weights) [] =
      ∑ m ∈ Finset.range nodes, weights m * mle point (nodeTable E leaf m w) := by
  rw [eqTruth_nil _ _ _ _ 0 (by simp [rangeTables]), ← leafComb_eq, leafTruth_nil]

theorem leaf_eqTruth_final (leaf nodes : ℕ) (weights : ℕ → E) (point r : List E) (w : Table)
    (length : r.length = point.length) :
    eqTruth point.length point (rangeTables point w) (leafData E leaf nodes weights) r =
      ∑ m ∈ Finset.range nodes, weights m * nodeIn leaf m (mle r (derived E w)) := by
  rw [eqTruth_final _ _ _ r length, eval_leafData]
  simp [rangeTables]

end Nodes

section Batch

variable {K : Type} [Field K]

/-- Entry `i` of a message of `n` field elements, and zero after its end. -/
def entry {n : ℕ} (message : Fin n → K) (i : ℕ) : K :=
  if inside : i < n then message ⟨i, inside⟩ else 0

/-- The polynomial `Σ_{i<n} c_i·X^i` of the batching of `n` claims with the powers of one
challenge (`DigitRangePlan::interstage_batch_weights` of the Akita code). -/
noncomputable def rowPoly (n : ℕ) (c : ℕ → K) : K[X] := ∑ i ∈ Finset.range n, C (c i) * X ^ i

theorem rowPoly_eval (n : ℕ) (c : ℕ → K) (γ : K) :
    (rowPoly n c).eval γ = ∑ i ∈ Finset.range n, γ ^ i * c i := by
  simp only [rowPoly, eval_finsetSum, eval_mul, eval_C, eval_pow, eval_X]
  exact Finset.sum_congr rfl fun _ _ => mul_comm _ _

theorem rowPoly_coeff {n i : ℕ} (c : ℕ → K) (inside : i < n) : (rowPoly n c).coeff i = c i := by
  simp [rowPoly, coeff_C_mul, coeff_X_pow, inside]

theorem rowPoly_natDegree_le (n : ℕ) (c : ℕ → K) : (rowPoly n c).natDegree ≤ n - 1 := by
  unfold rowPoly
  refine natDegree_sum_le_of_forall_le _ _ fun i member => ?_
  exact (natDegree_C_mul_X_pow_le _ _).trans (by have := Finset.mem_range.mp member; omega)

end Batch

section Tree

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] {β : Type}

/-- The verifier of the range tree for the product degrees `arities`, from the root, and a last
stage `last` for the leaf polynomials of `leaf` factors each. The state is the number of the
nodes of the level, their batching weights, the point of the level before and the batched claim.
One product level is the equality-factored sum-check of degree `a` against `ẽq(point, ·)`
(`eqSumcheck`), the claims of the children at its final point, the check of its normalized final
claim `Σ_m weight_m·Π_t child_{m·a+t}` (`Stage1Verifier::verify_product_prefix` of the Akita
code), and a challenge `γ` whose powers batch the claims of the children. The bad sets are the
roots of the full round polynomial of each round and the roots in `γ` of the difference between
the claims and the values of the child tables. -/
noncomputable def rangeTree (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option β)) :
    List ℕ → ℕ → (ℕ → E) → List E → E → Game (Option β)
  | [], nodes, weights, point, claim => last nodes weights point claim
  | a :: rest, nodes, weights, point, claim =>
      (eqSumcheck a (combRound μ (stageTables (rest.prod * leaf) point w)
        (stageComb E a nodes weights)) (fun _ => ∅) point μ [] claim).bind fun ended =>
        .recv (Fin (nodes * a) → E) fun child =>
          if ended.2 = ∑ m ∈ Finset.range nodes,
              weights m * ∏ t ∈ Finset.range a, entry child (m * a + t) then
            .draw E (zeros (rowPoly (nodes * a) (entry child) - rowPoly (nodes * a)
              fun m => mle ended.1 (nodeTable E (rest.prod * leaf) m w))) fun γ =>
              rangeTree leaf μ w last rest (nodes * a) (fun m => γ ^ m) ended.1
                ((rowPoly (nodes * a) (entry child)).eval γ)
          else .pure none

/-- The soundness of the range tree (Akita Lemma 10.37): on a path that avoids the bad sets, an
end `out` of the last stage has the property `Extra`, and when its claims are true (`Opens`), the
first claim is the batched sum of the node tables at the first point. `sound` is this statement
for the last stage. -/
theorem rangeTree_avoids (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option β)) (Extra Opens : β → Prop)
    (sound : ∀ nodes weights point claim, point.length = μ →
      (last nodes weights point claim).Avoids fun result => ∀ out, result = some out →
        Extra out ∧ (Opens out → claim = ∑ m ∈ Finset.range nodes,
          weights m * mle point (nodeTable E leaf m w))) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      point.length = μ →
      (rangeTree leaf μ w last arities nodes weights point claim).Avoids fun result =>
        ∀ out, result = some out → Extra out ∧ (Opens out → claim = ∑ m ∈ Finset.range nodes,
          weights m * mle point (nodeTable E (arities.prod * leaf) m w)) := by
  intro arities
  induction arities with
  | nil =>
      intro nodes weights point claim long
      simpa [rangeTree] using sound nodes weights point claim long
  | cons a rest step =>
      intro nodes weights point claim long
      subst long
      refine (Game.avoids_bind _ _ _).mpr ?_
      refine (eqSumcheck_avoids a point.length _ _
        (stage_roundTruth a (rest.prod * leaf) nodes point.length weights point w) (fun _ => ∅)
        (fun _ => True) (fun _ _ _ _ _ => trivial) point point.length [] claim (by simp)
        (by simp)).mono ?_
      rintro ⟨r, value⟩ ⟨length, staged, -⟩
      dsimp only at length staged
      rw [List.length_nil, zero_add] at length
      intro child
      dsimp only
      split
      swap
      · intro out equal
        simp at equal
      rename_i checked
      intro γ outside
      refine (step (nodes * a) (fun m => γ ^ m) r _ length).mono ?_
      intro result next out equal
      obtain ⟨extra, opened⟩ := next out equal
      refine ⟨extra, fun opens => ?_⟩
      have batched := opened opens
      have same : rowPoly (nodes * a) (entry child) =
          rowPoly (nodes * a) fun m => mle r (nodeTable E (rest.prod * leaf) m w) := by
        by_contra different
        exact eval_ne_of_not_mem_zeros different outside (by rw [batched, rowPoly_eval])
      have children (m t : ℕ) (node : m ∈ Finset.range nodes) (place : t ∈ Finset.range a) :
          entry child (m * a + t) = mle r (nodeTable E (rest.prod * leaf) (m * a + t) w) := by
        have inside : m * a + t < nodes * a :=
          Nat.lt_of_lt_of_le (Nat.add_lt_add_left (Finset.mem_range.mp place) _)
            (by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right a (Finset.mem_range.mp node))
        rw [← rowPoly_coeff (entry child) inside, same, rowPoly_coeff _ inside]
      by_contra false_claim
      refine staged ?_ ?_
      · rw [eqEval_nil, one_mul, stageTruth_nil]
        rw [List.prod_cons, mul_assoc] at false_claim
        exact false_claim
      · have final := stageTruth_final a (rest.prod * leaf) nodes weights point r w
        rw [length] at final
        rw [final, mle_eqAt point r length, checked]
        refine congrArg _ (Finset.sum_congr rfl fun m node => congrArg _
          (Finset.prod_congr rfl fun t place => children m t node place))

/-- The sum of the numerators of the bounds of the bad sets of the range tree before its last
stage: `μ'·(d + 1)` for the rounds of each product level of degree `d`, and the number of the
child claims minus 1 for each batching challenge (Akita (187)). -/
def treeNumerator (μ : ℕ) : List ℕ → ℕ → ℕ
  | [], _ => 0
  | a :: rest, nodes => μ * (a + 1) + (nodes * a - 1) + treeNumerator μ rest (nodes * a)

/-- The bad sets of the range tree have the total bound `(treeNumerator + c)/|E|` when the bad
sets of the last stage have the bound `c/|E|`. -/
theorem rangeTree_costLe (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option β)) (lastCost : ℕ)
    (cost : ∀ nodes weights point claim,
      (last nodes weights point claim).CostLe ((lastCost : ℝ) / Fintype.card E)) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      (rangeTree leaf μ w last arities nodes weights point claim).CostLe
        (((treeNumerator μ arities nodes + lastCost : ℕ) : ℝ) / Fintype.card E) := by
  intro arities
  induction arities with
  | nil =>
      intro nodes weights point claim
      simpa [rangeTree, treeNumerator] using cost nodes weights point claim
  | cons a rest step =>
      intro nodes weights point claim
      have split : ((treeNumerator μ (a :: rest) nodes + lastCost : ℕ) : ℝ) / Fintype.card E =
          (μ : ℝ) * (((a + 1 : ℕ) : ℝ) / Fintype.card E) +
            (((nodes * a - 1 : ℕ) : ℝ) / Fintype.card E +
              ((treeNumerator μ rest (nodes * a) + lastCost : ℕ) : ℝ) / Fintype.card E) := by
        simp only [treeNumerator]
        push_cast
        ring
      rw [split]
      refine Game.costLe_bind (eqSumcheck_costLe a μ 0 _
        (stage_roundTruth a (rest.prod * leaf) nodes μ weights point w).degree (fun _ => ∅)
        (fun _ => (Finset.card_empty).le) point μ [] claim (by simp)) ?_
      have rest_nonneg : (0 : ℝ) ≤ ((nodes * a - 1 : ℕ) : ℝ) / Fintype.card E +
          ((treeNumerator μ rest (nodes * a) + lastCost : ℕ) : ℝ) / Fintype.card E := by
        positivity
      rintro ⟨r, value⟩ child
      dsimp only
      split
      · exact ⟨_, add_le_add_left (Game.density_le _ _ (zeros_card_le_of_natDegree_le
          ((natDegree_sub_le _ _).trans (max_le (rowPoly_natDegree_le _ _)
            (rowPoly_natDegree_le _ _))))) _, fun γ => step _ _ _ _⟩
      · exact rest_nonneg

/-- The last stage of the range tree on the coefficient route: the equality-factored sum-check of
degree `leaf` against `ẽq(point, ·)` for the batched leaf polynomials to the point `r_virt`, the
value `s_claim`, and the check of the normalized final claim `Σ_m weight_m·Π_j (s_claim − k(k + 1))`
(`Stage1Verifier::verify` of the Akita code). The bad sets are the roots of the full round
polynomial of each round with `bindBad` for the binary-support table (Akita Lemma 10.37 (ii)). It
ends in `(r_virt, s_claim)`. -/
noncomputable def leafStage (leaf μ : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (w : Table) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E) :
    Game (Option (List E × E)) :=
  (eqSumcheck leaf (combRound μ (rangeTables point w) (leafComb E leaf nodes weights))
    (fun bound => bindBad (μ - bound.length - 1) (fold bound (supportTable support w)))
    point μ [] claim).bind fun ended =>
    .recv E fun sClaim =>
      if ended.2 = ∑ m ∈ Finset.range nodes, weights m * nodeIn leaf m sClaim then
        .pure (some (ended.1, sClaim))
      else .pure none

/-- The property of an end `(r_virt, s_claim)` of the range tree that does not need a true
claim: the point has `μ'` coordinates, and a binary-support table with a nonzero cell has a
nonzero value at `r_virt`. -/
def Virt (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table) (out : List E × E) :
    Prop :=
  out.1.length = μ ∧ (NonzeroOn μ (supportTable (E := E) support w) →
    mle out.1 (supportTable (E := E) support w) ≠ 0)

theorem leafStage_avoids (leaf μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E) (long : point.length = μ) :
    (leafStage leaf μ support w nodes weights point claim).Avoids fun result =>
      ∀ out, result = some out → Virt μ support w out ∧
        (out.2 = mle out.1 (derived E w) → claim = ∑ m ∈ Finset.range nodes,
          weights m * mle point (nodeTable E leaf m w)) := by
  subst long
  refine (Game.avoids_bind _ _ _).mpr ?_
  refine (eqSumcheck_avoids leaf point.length _ _
    (leaf_roundTruth leaf nodes point.length weights point w)
    (fun bound => bindBad (point.length - bound.length - 1) (fold bound (supportTable support w)))
    (fun bound => NonzeroOn (point.length - bound.length) (fold bound (supportTable support w)))
    (fun bound c inside nonzero outside => by
      rw [fold_append_single, List.length_append, List.length_singleton,
        show point.length - (bound.length + 1) = point.length - bound.length - 1 by omega]
      exact nonzeroOn_bindLow
        (by rwa [show point.length - bound.length - 1 + 1 = point.length - bound.length by omega])
        outside)
    point point.length [] claim (by simp) (by simp)).mono ?_
  rintro ⟨r, value⟩ ⟨length, staged, kept⟩
  dsimp only at length staged kept
  rw [List.length_nil, zero_add] at length
  intro sClaim
  dsimp only
  split
  swap
  · intro out equal
    simp at equal
  rename_i checked
  rintro ⟨r', s⟩ equal
  simp only [Option.some.injEq, Prod.mk.injEq] at equal
  obtain ⟨rfl, rfl⟩ := equal
  refine ⟨⟨length, fun nonzero => ?_⟩, fun opens => ?_⟩
  · have cell := kept (by simpa [fold] using nonzero)
    rw [length, Nat.sub_self] at cell
    obtain ⟨i, lt, value⟩ := cell
    obtain rfl : i = 0 := by simpa using lt
    exact value
  · have opened : sClaim = mle r (derived E w) := opens
    by_contra false_claim
    refine staged ?_ ?_
    · rwa [eqEval_nil, one_mul, leafTruth_nil]
    · have final := leafTruth_final leaf nodes weights point r w
      rw [length] at final
      rw [final, mle_eqAt point r length, checked, opened]

theorem leafStage_costLe (leaf μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E) :
    (leafStage leaf μ support w nodes weights point claim).CostLe
      (((μ * (leaf + 1 + 1) : ℕ) : ℝ) / Fintype.card E) := by
  have whole : ((μ * (leaf + 1 + 1) : ℕ) : ℝ) / Fintype.card E =
      (μ : ℝ) * (((leaf + 1 + 1 : ℕ) : ℝ) / Fintype.card E) + 0 := by
    push_cast
    ring
  rw [whole]
  refine Game.costLe_bind (eqSumcheck_costLe leaf μ 1 _
    (leaf_roundTruth leaf nodes μ weights point w).degree _ (fun _ => bindBad_card_le _ _)
    point μ [] claim (by simp)) ?_
  rintro ⟨r, value⟩ sClaim
  dsimp only
  split <;> exact le_refl (0 : ℝ)

/-- The digit range check for the base `b⋆` from the range point `τ_0`: the range tree of Akita
Section 6.1 with the last stage `leafStage`. It ends in `(r_virt, s_claim)`. -/
noncomputable def rangeCheck (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (τ0 : List E) : Game (Option (List E × E)) :=
  rangeTree (rangeLeaf b) μ w (leafStage (rangeLeaf b) μ support w) (rangeArities b) 1
    (fun _ => 1) τ0 0

/-- The sum of the numerators of the bounds of the bad sets of `rangeCheck`: the two sums of
Akita (187), and `μ'` for the binary-support point of Akita Lemma 10.37 (ii). -/
def rangeNumerator (μ b : ℕ) : ℕ :=
  treeNumerator μ (rangeArities b) 1 + μ * (rangeLeaf b + 1 + 1)

omit [Fintype E] [DecidableEq E] in
/-- The first claim of the range tree for the base `b⋆` at the range point `τ_0` is the sum
(80). -/
theorem root_sum (b : ℕ) (w : Table) (τ0 : List E) :
    ∑ m ∈ Finset.range 1, (1 : E) *
        mle τ0 (nodeTable E ((rangeArities b).prod * rangeLeaf b) m w) =
      mle τ0 (rangeResidual E b w) := by
  rw [rangeArities_prod_mul_leaf, Finset.sum_range_one, one_mul]
  refine mle_congr τ0 fun x _ => ?_
  rw [rangeResidual_eq, nodeTable, nodeIn_root]

/-- Akita Lemma 10.37 for `rangeCheck`: on a path that avoids the bad sets, an end
`(r_virt, s_claim)` has `Virt`, and when `s_claim = s̃(r_virt)`, the sum (80) is zero at `τ_0`. -/
theorem rangeCheck_avoids (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (τ0 : List E) (long : τ0.length = μ) :
    (rangeCheck b μ support w τ0).Avoids fun result => ∀ out, result = some out →
      Virt μ support w out ∧
        (out.2 = mle out.1 (derived E w) → mle τ0 (rangeResidual E b w) = 0) := by
  refine (rangeTree_avoids (rangeLeaf b) μ w _ (Virt μ support w)
    (fun out => out.2 = mle out.1 (derived E w))
    (fun nodes weights point claim => leafStage_avoids (rangeLeaf b) μ support w nodes weights
      point claim) (rangeArities b) 1 (fun _ => 1) τ0 0 long).mono ?_
  intro result sound out equal
  obtain ⟨virt, opened⟩ := sound out equal
  refine ⟨virt, fun opens => ?_⟩
  rw [← root_sum b w τ0]
  exact (opened opens).symm

theorem rangeCheck_costLe (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (τ0 : List E) :
    (rangeCheck b μ support w τ0).CostLe (((rangeNumerator μ b : ℕ) : ℝ) / Fintype.card E) :=
  rangeTree_costLe (rangeLeaf b) μ w _ _
    (fun nodes weights point claim => leafStage_costLe (rangeLeaf b) μ support w nodes weights
      point claim) (rangeArities b) 1 (fun _ => 1) τ0 0

end Tree

end NightstreamFPrime.Spec.Compression.Akita
