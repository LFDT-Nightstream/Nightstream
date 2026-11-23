import NightstreamFPrime.Spec.Compression.Akita.Batch
import NightstreamFPrime.Spec.Compression.Akita.Setup

/-! Reductions between instances of Module-SIS (Akita, ePrint 2026/1983, revision of 2026-11-23,
Definition 3.2, Section 4.4 with (38) and (39), and Section 10.3), and the instances of the
matrices of a commitment and of an opening.

- `ShortKernel.carry` and `ShortKernelSq.carry` move a solution for `M` to a matrix `M'` that has
  the columns of `M` among its columns, a subset of the rows of `M` and a bound that is not
  smaller. The new vector is 0 in the other columns.
- `descend` is ring descent. For `stride · d_car = d`, a matrix over `R_{q,d}` with `m` columns
  gives the matrix over `R_{q,d_car}` with `stride · m` columns whose entries are the coordinates
  of Akita (39). `ShortKernel.descend` and `ShortKernelSq.descend` move a solution, with the same
  bound. `descend_bijective` and `expect_descend` state that a uniform matrix gives a uniform
  matrix, and `ShortKernel.lower_chance` states that a solver for the degree `d` is a solver for
  the degree `d_car` with the same probability.
- `Instance.Below` orders the instances of one degree, rank and norm by width and bound.
  `Reduces list M` states that a matrix `M` has a solution that gives a solution of an instance
  of `list` for each matrix of that instance with the columns of `M`.
- `Shape.instances`, `OpeningShape.instances` and `Batch.openInstances` are the instances of the
  matrices of a key, and `Key.reduces`, `level_reduces` and `Batch.reduces` give `Reduces` for
  each solution that a witness of `R_bind` holds. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators

section Monotone

variable {d : ℕ} [NeZero d] {rows rows' cols cols' : Type} [Fintype cols] [Fintype cols']

/-- The vector `z` in the columns `place`, and 0 in each other column. -/
noncomputable def padded (place : cols ↪ cols') (z : cols → Rq d) : cols' → Rq d :=
  Function.extend place z 0

omit [NeZero d] [Fintype cols] [Fintype cols'] in
theorem padded_apply (place : cols ↪ cols') (z : cols → Rq d) (j : cols) :
    padded place z (place j) = z j :=
  place.injective.extend_apply z (0 : cols' → Rq d) j

omit [NeZero d] [Fintype cols'] in
theorem padded_outside (place : cols ↪ cols') (z : cols → Rq d) {k : cols'}
    (outside : k ∉ Finset.univ.map place) : padded place z k = 0 :=
  Function.extend_apply' z (0 : cols' → Rq d) k fun ⟨j, image⟩ =>
    outside (image ▸ Finset.mem_map_of_mem place (Finset.mem_univ j))

omit [NeZero d] in
theorem sum_padded {A : Type} [AddCommMonoid A] (place : cols ↪ cols') (z : cols → Rq d)
    (f : cols' → Rq d → A) (zero : ∀ k, f k 0 = 0) :
    ∑ k, f k (padded place z k) = ∑ j, f (place j) (z j) := by
  have outer : ∀ k ∈ Finset.univ, k ∉ Finset.univ.map place → f k (padded place z k) = 0 :=
    fun k _ outside => by rw [padded_outside place z outside, zero]
  rw [← Finset.sum_subset (Finset.subset_univ (Finset.univ.map place)) outer, Finset.sum_map]
  exact Finset.sum_congr rfl fun j _ => by rw [padded_apply]

omit [NeZero d] in
theorem mulVec_padded {M : Matrix rows cols (Rq d)} {M' : Matrix rows' cols' (Rq d)}
    (pick : rows' → rows) (place : cols ↪ cols') (same : ∀ i j, M' i (place j) = M (pick i) j)
    (z : cols → Rq d) (i : rows') : M'.mulVec (padded place z) i = M.mulVec z (pick i) := by
  simp only [Matrix.mulVec, dotProduct]
  rw [sum_padded place z (fun k x => M' i k * x) fun _ => mul_zero _]
  exact Finset.sum_congr rfl fun j _ => by rw [same]

theorem normSq_zero : normSq (0 : Rq d) = 0 := by
  simp [normSq]

theorem vecNormSq_padded (place : cols ↪ cols') (z : cols → Rq d) :
    vecNormSq (padded place z) = vecNormSq z :=
  sum_padded place z (fun _ x => normSq x) fun _ => normSq_zero

theorem vecNormInf_padded_le (place : cols ↪ cols') (z : cols → Rq d) :
    vecNormInf (padded place z) ≤ vecNormInf z := by
  refine (vecNormInf_le_iff _ _).mpr fun k => ?_
  by_cases inside : k ∈ Finset.univ.map place
  · obtain ⟨j, -, rfl⟩ := Finset.mem_map.mp inside
    rw [padded_apply]
    exact normInf_le_vecNormInf z j
  · rw [padded_outside place z inside, normInf_zero]
    exact Nat.zero_le _

omit [NeZero d] [Fintype cols] [Fintype cols'] in
theorem padded_ne_zero (place : cols ↪ cols') {z : cols → Rq d} (nonzero : z ≠ 0) :
    padded place z ≠ 0 :=
  fun zero => nonzero (funext fun j => by rw [← padded_apply place z j, zero]; rfl)

/-- Monotonicity of Module-SIS in the norm `‖·‖_∞` (Akita Definition 3.2). `M'` has the columns of
`M` at the positions `place`, each row of `M'` is the row `pick` of `M`, and `bound ≤ bound'`. A
solution for `M` with zeros in the other columns is a solution for `M'`: a matrix with more
columns, with fewer rows or with a larger bound is not harder. -/
noncomputable def ShortKernel.carry {M : Matrix rows cols (Rq d)} {M' : Matrix rows' cols' (Rq d)}
    {bound bound' : ℕ} (pick : rows' → rows) (place : cols ↪ cols')
    (same : ∀ i j, M' i (place j) = M (pick i) j) (le : bound ≤ bound') (z : ShortKernel M bound) :
    ShortKernel M' bound' where
  vector := padded place z.vector
  nonzero := padded_ne_zero place z.nonzero
  kernel := funext fun i => by
    rw [mulVec_padded pick place same, z.kernel]
    rfl
  short := (vecNormInf_padded_le place z.vector).trans (z.short.trans le)

/-- Monotonicity of Module-SIS in the norm `‖·‖_2`: `ShortKernel.carry` for a bound on
`‖z‖_2²`. -/
noncomputable def ShortKernelSq.carry {M : Matrix rows cols (Rq d)}
    {M' : Matrix rows' cols' (Rq d)} {bound bound' : ℕ} (pick : rows' → rows)
    (place : cols ↪ cols') (same : ∀ i j, M' i (place j) = M (pick i) j) (le : bound ≤ bound')
    (z : ShortKernelSq M bound) : ShortKernelSq M' bound' where
  vector := padded place z.vector
  nonzero := padded_ne_zero place z.nonzero
  kernel := funext fun i => by
    rw [mulVec_padded pick place same, z.kernel]
    rfl
  short := (vecNormSq_padded place z.vector).trans_le (z.short.trans le)

end Monotone

/-- A sum over the positions below `stride · d_car` is a sum over the coordinates `a` of
Akita (39) and the positions `j` of each coordinate. -/
theorem sum_range_planes {A : Type} [AddCommMonoid A] {stride dCar d : ℕ}
    (tower : stride * dCar = d) (f : ℕ → A) :
    ∑ i ∈ Finset.range d, f i =
      ∑ a ∈ Finset.range stride, ∑ j ∈ Finset.range dCar, f (a + stride * j) := by
  subst tower
  induction dCar with
  | zero => simp
  | succ n step =>
      rw [Nat.mul_succ, Finset.sum_range_add, step, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun a _ => by rw [Finset.sum_range_succ, Nat.add_comm (stride * n)]

section Descent

variable {d dCar stride : ℕ} [NeZero d] [NeZero dCar]

/-- The last coordinate of `X^b·x` is the coordinate `stride − 1 − b` of `x`: the product has no
wrap at `X^d` in this coordinate. -/
theorem plane_root_pow_mul_top (tower : stride * dCar = d) {b : ℕ} (low : b < stride) (x : Rq d) :
    plane dCar stride (stride - 1) (AdjoinRoot.root (modulus d) ^ b * x) =
      plane dCar stride (stride - 1 - b) x := by
  have positive := NeZero.pos dCar
  refine ext_coeff fun j small => ?_
  have address := plane_address_lt tower (show stride - 1 < stride by omega) small
  rw [coeff_plane _ _ _ small, coeff_plane _ _ _ small,
    coeff_root_pow_mul (low.le.trans (tower ▸ Nat.le_mul_of_pos_right _ positive)) _ address,
    ite_eq_right (by omega)]
  exact congrArg (coeff x) (by omega)

/-- The last coordinate of a product in the basis `(X^a)_a` of Akita (39):
`(x·z)_{stride−1} = Σ_b x_{stride−1−b}·z_b`. It has no term with a factor `Y`. -/
theorem plane_top_mul (tower : stride * dCar = d) (x z : Rq d) :
    plane dCar stride (stride - 1) (x * z) =
      ∑ b ∈ Finset.range stride, plane dCar stride (stride - 1 - b) x * plane dCar stride b z := by
  have top : stride - 1 < stride := Nat.sub_lt (stride_pos tower) Nat.one_pos
  have expanded : x * z = ∑ b ∈ Finset.range stride, ∑ j ∈ Finset.range dCar,
      coeff z (b + stride * j) • (embed stride tower (AdjoinRoot.root (modulus dCar) ^ j) *
        (AdjoinRoot.root (modulus d) ^ b * x)) := by
    conv_lhs => rw [← ofCoeffs_coeff z, ofCoeffs_eq_sum, Finset.mul_sum, sum_range_planes tower]
    refine Finset.sum_congr rfl fun b _ => Finset.sum_congr rfl fun j _ => ?_
    rw [mul_smul_comm, map_pow, embed_root, pow_add, pow_mul]
    congr 1
    ring
  rw [expanded, map_sum]
  refine Finset.sum_congr rfl fun b member => ?_
  have coordinate : plane dCar stride b z = ∑ j ∈ Finset.range dCar,
      coeff z (b + stride * j) • AdjoinRoot.root (modulus dCar) ^ j := by
    rw [← ofCoeffs_eq_sum]
    rfl
  rw [coordinate, map_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [map_smul, plane_embed_mul tower top,
    plane_root_pow_mul_top tower (Finset.mem_range.mp member), mul_smul_comm]
  exact congrArg _ (mul_comm _ _)

theorem normInf_plane_le (tower : stride * dCar = d) {a : ℕ} (low : a < stride) (x : Rq d) :
    normInf (plane dCar stride a x) ≤ normInf x :=
  (normInf_le_iff _ _).mpr fun j small => by
    rw [coeff_plane _ _ _ small]
    exact (normInf_le_iff x _).mp le_rfl _ (plane_address_lt tower low small)

/-- The coordinates of an element have its coefficients, so they have its `‖·‖_2²`. -/
theorem sum_normSq_plane (tower : stride * dCar = d) (x : Rq d) :
    ∑ a : Fin stride, normSq (plane dCar stride a x) = normSq x := by
  rw [← Finset.sum_range fun a => normSq (plane dCar stride a x)]
  unfold normSq
  rw [sum_range_planes tower fun i => magnitude (coeff x i) ^ 2]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun j small => ?_
  rw [coeff_plane _ _ _ (Finset.mem_range.mp small)]

variable {rows cols : Type}

/-- Ring descent of a matrix (Akita (39)): the entry at the column `(j, b)` is the coordinate
`stride − 1 − b` of the entry `j` in the basis `1, X, …, X^{stride−1}` over `ι(R_{q,d_car})`. -/
noncomputable def descend (dCar stride : ℕ) [NeZero dCar] (M : Matrix rows cols (Rq d)) :
    Matrix rows (cols × Fin stride) (Rq dCar) :=
  Matrix.of fun i p => plane dCar stride (stride - 1 - p.2) (M i p.1)

/-- The coordinates of a vector: the entry at `(j, b)` is the coordinate `b` of the entry `j`. -/
noncomputable def planes (dCar stride : ℕ) [NeZero dCar] (z : cols → Rq d) :
    cols × Fin stride → Rq dCar :=
  fun p => plane dCar stride p.2 (z p.1)

/-- The product of the descended matrix and the coordinates of `z` is the last coordinate of
`M z`. -/
theorem descend_mulVec [Fintype cols] (tower : stride * dCar = d) (M : Matrix rows cols (Rq d))
    (z : cols → Rq d) (i : rows) :
    (descend dCar stride M).mulVec (planes dCar stride z) i =
      plane dCar stride (stride - 1) (M.mulVec z i) := by
  simp only [Matrix.mulVec, dotProduct, descend, planes, Matrix.of_apply]
  rw [Fintype.sum_prod_type, map_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [plane_top_mul tower, Finset.sum_range]

theorem planes_ne_zero (tower : stride * dCar = d) {z : cols → Rq d} (nonzero : z ≠ 0) :
    planes dCar stride z ≠ 0 :=
  fun zero => nonzero (funext fun j => eq_of_planes tower fun a low =>
    (congrFun zero (j, ⟨a, low⟩)).trans (map_zero _).symm)

theorem vecNormInf_planes_le [Fintype cols] (tower : stride * dCar = d) (z : cols → Rq d) :
    vecNormInf (planes dCar stride z) ≤ vecNormInf z :=
  (vecNormInf_le_iff _ _).mpr fun p =>
    (normInf_plane_le tower p.2.isLt (z p.1)).trans (normInf_le_vecNormInf z p.1)

theorem vecNormSq_planes [Fintype cols] (tower : stride * dCar = d) (z : cols → Rq d) :
    vecNormSq (planes dCar stride z) = vecNormSq z := by
  unfold vecNormSq
  rw [Fintype.sum_prod_type]
  exact Finset.sum_congr rfl fun j _ => sum_normSq_plane tower (z j)

/-- Ring descent of a solution in the norm `‖·‖_∞`: for `stride · d_car = d`, a solution of
`MSIS_{q,d}(n, m, β)` for `M` gives a solution of `MSIS_{q,d_car}(n, stride·m, β)` for
`descend M`, with the same bound. -/
noncomputable def ShortKernel.descend [Fintype cols] (tower : stride * dCar = d)
    {M : Matrix rows cols (Rq d)} {bound : ℕ} (z : ShortKernel M bound) :
    ShortKernel (descend dCar stride M) bound where
  vector := planes dCar stride z.vector
  nonzero := planes_ne_zero tower z.nonzero
  kernel := funext fun i => by
    rw [descend_mulVec tower, z.kernel]
    exact map_zero _
  short := (vecNormInf_planes_le tower z.vector).trans z.short

/-- Ring descent of a solution in the norm `‖·‖_2`: the vector has the same `‖·‖_2²`. -/
noncomputable def ShortKernelSq.descend [Fintype cols] (tower : stride * dCar = d)
    {M : Matrix rows cols (Rq d)} {bound : ℕ} (z : ShortKernelSq M bound) :
    ShortKernelSq (descend dCar stride M) bound where
  vector := planes dCar stride z.vector
  nonzero := planes_ne_zero tower z.nonzero
  kernel := funext fun i => by
    rw [descend_mulVec tower, z.kernel]
    exact map_zero _
  short := (vecNormSq_planes tower z.vector).trans_le z.short

theorem descend_injective (tower : stride * dCar = d) :
    Function.Injective (descend dCar stride : Matrix rows cols (Rq d) → _) := by
  intro M N same
  ext i j
  refine eq_of_planes tower fun a low => ?_
  have entry := congrFun (congrFun same i) (j, ⟨stride - 1 - a, by omega⟩)
  simp only [descend, Matrix.of_apply] at entry
  rwa [show stride - 1 - (stride - 1 - a) = a by omega] at entry

/-- The matrix over `R_{q,d}` whose entries have the coordinates `N`. -/
noncomputable def ascend (tower : stride * dCar = d)
    (N : Matrix rows (cols × Fin stride) (Rq dCar)) : Matrix rows cols (Rq d) :=
  Matrix.of fun i j => ofCoeffs fun l =>
    coeff (N i (j, ⟨stride - 1 - l % stride, by have := stride_pos tower; omega⟩)) (l / stride)

theorem descend_ascend (tower : stride * dCar = d)
    (N : Matrix rows (cols × Fin stride) (Rq dCar)) : descend dCar stride (ascend tower N) = N := by
  have positive := stride_pos tower
  ext i ⟨j, b⟩
  refine ext_coeff fun k small => ?_
  have low : stride - 1 - (b : ℕ) < stride := by omega
  have remainder : (stride - 1 - b + stride * k) % stride = stride - 1 - b := by
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt low]
  have quotient : (stride - 1 - b + stride * k) / stride = k := by
    rw [Nat.add_mul_div_left _ _ positive, Nat.div_eq_of_lt low, Nat.zero_add]
  simp only [descend, ascend, Matrix.of_apply]
  rw [coeff_plane _ _ _ small, coeff_ofCoeffs _ (plane_address_lt tower low small), quotient]
  exact congrArg (fun c => coeff (N i (j, c)) k) (Fin.ext (by
    show stride - 1 - (stride - 1 - b + stride * k) % stride = b
    rw [remainder]
    omega))

/-- Ring descent is a bijection of the matrices: it gives each matrix over `R_{q,d_car}` with
`stride · m` columns for one matrix over `R_{q,d}` with `m` columns. -/
theorem descend_bijective (tower : stride * dCar = d) :
    Function.Bijective (descend dCar stride : Matrix rows cols (Rq d) → _) :=
  ⟨descend_injective tower, fun N => ⟨ascend tower N, descend_ascend tower N⟩⟩

theorem ascend_bijective (tower : stride * dCar = d) :
    Function.Bijective (ascend tower : Matrix rows (cols × Fin stride) (Rq dCar) → _) :=
  ⟨Function.LeftInverse.injective (descend_ascend tower),
    fun M => ⟨descend dCar stride M, descend_injective tower (descend_ascend tower _)⟩⟩

variable [Fintype rows] [DecidableEq rows] [Fintype cols] [DecidableEq cols]

/-- Ring descent of a uniform matrix over `R_{q,d}` is a uniform matrix over `R_{q,d_car}`. -/
theorem expect_descend (tower : stride * dCar = d)
    (value : Matrix rows (cols × Fin stride) (Rq dCar) → ℝ) :
    𝔼 M : Matrix rows cols (Rq d), value (descend dCar stride M) = 𝔼 N, value N :=
  Fintype.expect_bijective _ (descend_bijective tower) _ _ fun _ => rfl

/-- A solver of `MSIS_{q,d}(n, m, β)` as a solver of `MSIS_{q,d_car}(n, stride·m, β)`: it runs
`solve` on the matrix with the coordinates `N` and returns the coordinates of the solution. -/
noncomputable def ShortKernel.lower (tower : stride * dCar = d) {bound : ℕ}
    (solve : (M : Matrix rows cols (Rq d)) → Option (ShortKernel M bound))
    (N : Matrix rows (cols × Fin stride) (Rq dCar)) : Option (ShortKernel N bound) :=
  (solve (ascend tower N)).map fun z =>
    cast (congrArg (ShortKernel · bound) (descend_ascend tower N)) (z.descend tower)

/-- Ring descent for a solver in the norm `‖·‖_∞`. The probability that `ShortKernel.lower`
returns a solution for a uniform matrix over `R_{q,d_car}` with `stride · m` columns is the
probability that `solve` returns a solution for a uniform matrix over `R_{q,d}` with `m` columns.
So Module-SIS at the degree `d` and the width `m` is not easier than at the degree `d_car` and the
width `stride · m`, for one rank and one bound. -/
theorem ShortKernel.lower_chance (tower : stride * dCar = d) {bound : ℕ}
    (solve : (M : Matrix rows cols (Rq d)) → Option (ShortKernel M bound)) :
    𝔼 N, (if (ShortKernel.lower tower solve N).isSome then (1 : ℝ) else 0) =
      𝔼 M, if (solve M).isSome then (1 : ℝ) else 0 :=
  Fintype.expect_bijective _ (ascend_bijective tower) _ _ fun _ => by
    rw [ShortKernel.lower, Option.isSome_map]

/-- `ShortKernel.lower` for the norm `‖·‖_2`. -/
noncomputable def ShortKernelSq.lower (tower : stride * dCar = d) {bound : ℕ}
    (solve : (M : Matrix rows cols (Rq d)) → Option (ShortKernelSq M bound))
    (N : Matrix rows (cols × Fin stride) (Rq dCar)) : Option (ShortKernelSq N bound) :=
  (solve (ascend tower N)).map fun z =>
    cast (congrArg (ShortKernelSq · bound) (descend_ascend tower N)) (z.descend tower)

/-- Ring descent for a solver in the norm `‖·‖_2`: `ShortKernel.lower_chance` for a bound on
`‖z‖_2²`. -/
theorem ShortKernelSq.lower_chance (tower : stride * dCar = d) {bound : ℕ}
    (solve : (M : Matrix rows cols (Rq d)) → Option (ShortKernelSq M bound)) :
    𝔼 N, (if (ShortKernelSq.lower tower solve N).isSome then (1 : ℝ) else 0) =
      𝔼 M, if (solve M).isSome then (1 : ℝ) else 0 :=
  Fintype.expect_bijective _ (ascend_bijective tower) _ _ fun _ => by
    rw [ShortKernelSq.lower, Option.isSome_map]

end Descent

open Setup (Instance)

namespace Setup.Instance

/-- The numbers `(d_I, n_I, m_I, p_I, η_I)` of an instance. -/
def key (I : Instance) : ℕ × ℕ × ℕ × Bool × ℕ := (I.degree, I.rows, I.cols, I.euclidean, I.bound)

/-- `I` has the degree, the rank and the norm of `J`, and a width and a bound that are not
larger. Then a solution of `I` gives a solution of `J` (`ShortKernel.carry`). -/
def Below (I J : Instance) : Prop :=
  I.degree = J.degree ∧ I.rows = J.rows ∧ I.euclidean = J.euclidean ∧ I.cols ≤ J.cols ∧
    I.bound ≤ J.bound

instance (I J : Instance) : Decidable (I.Below J) :=
  inferInstanceAs (Decidable (I.degree = J.degree ∧ I.rows = J.rows ∧
    I.euclidean = J.euclidean ∧ I.cols ≤ J.cols ∧ I.bound ≤ J.bound))

end Setup.Instance

/-- Each instance of `views` is `Below` an instance of `list`. -/
def Covered (list views : List Instance) : Prop := ∀ v ∈ views, ∃ I ∈ list, v.Below I

section Reduces

variable {d n : ℕ} [NeZero d] {cols : Type} [Fintype cols]

/-- A solution for `M` gives a solution of an instance of `list`. The instance has the degree and
the rank of `M`, the norm `euclidean`, and a width for the columns of `M`. Each matrix `M'` of the
instance that has the columns of `M` at some positions `place` has a solution at the bound of the
instance. -/
def Reduces (list : List Instance) (M : Matrix (Fin n) cols (Rq d)) (euclidean : Bool) : Prop :=
  ∃ width bound, (⟨d, n, width, euclidean, bound⟩ : Instance) ∈ list ∧
    Fintype.card cols ≤ width ∧
    ∀ (M' : Matrix (Fin n) (Fin width) (Rq d)) (place : cols ↪ Fin width),
      (∀ i j, M' i (place j) = M i j) →
        Nonempty (Instance.Kernel ⟨d, n, width, euclidean, bound⟩ M')

/-- A solution for `M` at `bound` gives `Reduces`, when an instance of `list` is above the
instance of `M` with the width `width` and the bound `bound`. -/
theorem Reduces.of_kernel {list : List Instance} {M : Matrix (Fin n) cols (Rq d)}
    {euclidean : Bool} {width bound : ℕ} (card : Fintype.card cols = width)
    (above : ∃ I ∈ list, (⟨d, n, width, euclidean, bound⟩ : Instance).Below I)
    (z : cond euclidean (ShortKernelSq M bound) (ShortKernel M bound)) :
    Reduces list M euclidean := by
  obtain ⟨⟨degree, rows, wide, norm, large⟩, member, sameDegree, sameRows, sameNorm, fits, le⟩ :=
    above
  dsimp only at sameDegree sameRows sameNorm fits le
  subst sameDegree sameRows sameNorm
  refine ⟨wide, large, member, card ▸ fits, fun M' place same => ?_⟩
  cases euclidean
  · exact ⟨ShortKernel.carry id place same le z⟩
  · exact ⟨ShortKernelSq.carry id place same le z⟩

end Reduces

/-- The instances of the compression maps of a path, with the bound 1 (Akita Table 3). -/
def pathInstances : Option ChainShape → List Instance
  | none => []
  | some c => [⟨c.firstDegree, c.firstRank, c.firstWidth, false, 1⟩,
      ⟨c.secondDegree, c.secondRank, c.secondWidth, false, 1⟩]

/-- A compression map of the path has a solution that gives a solution of an instance of
`list`. -/
def PathReduces (list : List Instance) : (chain : Option ChainShape) → PathKey chain → Prop
  | none, _ => False
  | some _, key =>
      Reduces list (ChainKey.first key) false ∨ Reduces list (ChainKey.second key) false

theorem PathReduces.of_kernel {list : List Instance} {chain : Option ChainShape}
    {key : PathKey chain} (covered : Covered list (pathInstances chain))
    (z : PathKernel chain key) : PathReduces list chain key := by
  cases chain with
  | none => exact z.elim
  | some c =>
      rcases (z : ShortKernel (ChainKey.first key) 1 ⊕ ShortKernel (ChainKey.second key) 1) with
        z | z
      · exact Or.inl (Reduces.of_kernel (euclidean := false) (Fintype.card_fin _)
          (covered _ List.mem_cons_self) z)
      · exact Or.inr (Reduces.of_kernel (euclidean := false) (Fintype.card_fin _)
          (covered _ (List.mem_cons_of_mem _ List.mem_cons_self)) z)

namespace Shape

variable (S : Shape)

/-- The instance of `A` (Akita Table 3): the degree `d_A`, the rank `n_A` and the width
`M·δ_com`, with the norm and the bound of the route. -/
def innerInstance (euclidean : Bool) (bound : ℕ) : Instance :=
  ⟨S.innerDegree, S.innerRank, S.blockLength * S.sourceDigits, euclidean, bound⟩

/-- The instance of `B` (Akita Table 3 and (49)): the degree `d_B`, the rank `n_B`, the width
`m_B = L^blk_B·n_A·(d_A/d_B)·δ_open` and the bound `b⋆ − 1`. -/
def outerInstance : Instance :=
  ⟨S.outerDegree, S.outerRank, S.sliceLength * (S.innerRank * (S.ratio * S.openingDigits)), false,
    S.certifiedBase - 1⟩

/-- The instances of the matrices of a commitment: `A` with the norm and the bound of the route,
`B`, and the compression maps `F_1` and `F_2`. -/
def instances (euclidean : Bool) (bound : ℕ) : List Instance :=
  S.innerInstance euclidean bound :: S.outerInstance :: pathInstances S.chain

end Shape

/-- The instance of `D` for one group (Akita Table 3 and (69)): the degree `d_D`, the rank `n_D`,
the width `m_D = B·(d_op/d_D)·δ_open` and the bound `b⋆ − 1`. -/
def OpeningShape.openInstance {S : Shape} (O : OpeningShape S) : Instance :=
  ⟨O.openDegree, O.openRank, S.blocks * (O.subcolumns * S.openingDigits), false,
    S.certifiedBase - 1⟩

/-- The instances of the matrices of an opening path: `D` and the compression maps `H_1` and
`H_2`. -/
def OpeningShape.instances {S : Shape} (O : OpeningShape S) : List Instance :=
  O.openInstance :: pathInstances O.chain

section Level

variable {S : Shape} {O : OpeningShape S} {list : List Instance}

/-- A matrix of the commitment key has a solution that gives a solution of an instance of
`list`: `A` on the norm `euclidean`, `B`, or a compression map. -/
def Key.Reduces (list : List Instance) (key : Key S) (euclidean : Bool) : Prop :=
  Akita.Reduces list key.inner euclidean ∨ Akita.Reduces list key.outer false ∨
    PathReduces list S.chain key.chain

/-- A matrix of the opening key has a solution that gives a solution of an instance of `list`:
`D` or a compression map. -/
def OpeningKey.Reduces (list : List Instance) (key : OpeningKey O) : Prop :=
  Akita.Reduces list key.matrix false ∨ PathReduces list O.chain key.chain

/-- Each solution of `OuterBreak` gives `Key.Reduces`. -/
theorem Key.reduces_outer {key : Key S} {euclidean : Bool} {bound : ℕ}
    (covered : Covered list (S.instances euclidean bound)) (z : OuterBreak key) :
    key.Reduces list euclidean := by
  rcases z with z | z
  · exact Or.inr (Or.inl (Reduces.of_kernel (euclidean := false)
      (by simp only [Fintype.card_prod, Fintype.card_fin])
      (covered _ (List.mem_cons_of_mem _ List.mem_cons_self)) z))
  · exact Or.inr (Or.inr (PathReduces.of_kernel
      (fun v member => covered v (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member))) z))

/-- Each solution for `A` at the bound of the instance gives `Key.Reduces`. -/
theorem Key.reduces_inner {key : Key S} {euclidean : Bool} {bound : ℕ}
    (covered : Covered list (S.instances euclidean bound))
    (z : cond euclidean (ShortKernelSq key.inner bound) (ShortKernel key.inner bound)) :
    key.Reduces list euclidean :=
  Or.inl (Reduces.of_kernel (by simp only [Fintype.card_prod, Fintype.card_fin])
    (covered _ List.mem_cons_self) z)

/-- Each solution of `OpenBreak` gives `OpeningKey.Reduces`. -/
theorem OpeningKey.reduces {key : OpeningKey O} (covered : Covered list O.instances)
    (z : OpenBreak key) : key.Reduces list := by
  rcases z with z | z
  · exact Or.inl (Reduces.of_kernel (euclidean := false)
      (by simp only [Fintype.card_prod, Fintype.card_fin]) (covered _ List.mem_cons_self) z)
  · exact Or.inr (PathReduces.of_kernel
      (fun v member => covered v (List.mem_cons_of_mem _ member)) z)

/-- Each solution that a witness of `R_bind` of one level holds gives a solution of an instance
of `list`, when `list` covers the instances of the shapes of the level. -/
theorem level_reduces {key : Key S} {openKey : OpeningKey O} {euclidean : Bool} {bound : ℕ}
    (covered : Covered list (S.instances euclidean bound)) (coveredOpen : Covered list O.instances)
    (z : cond euclidean (ShortKernelSq key.inner bound) (ShortKernel key.inner bound) ⊕
      (OuterBreak key ⊕ OpenBreak openKey)) :
    key.Reduces list euclidean ∨ openKey.Reduces list := by
  rcases z with z | z | z
  · exact Or.inl (Key.reduces_inner covered z)
  · exact Or.inl (Key.reduces_outer covered z)
  · exact Or.inr (OpeningKey.reduces coveredOpen z)

end Level

namespace Batch

variable {G : Type} [Fintype G] (B : Batch G) {list : List Instance}

/-- The instance of the shared matrix `D` (Akita (69), (70)): the width is the sum of the widths
of the groups, and the bound is the largest `b⋆_g − 1`. -/
def openInstance : Instance :=
  ⟨B.openDegree, B.openRank,
    ∑ g, (B.shape g).blocks * ((B.opening g).subcolumns * (B.shape g).openingDigits), false,
    B.openBound⟩

/-- The instances of the matrices of the shared opening path. -/
def openInstances : List Instance := B.openInstance :: pathInstances B.chain

variable {B}

/-- A matrix of the shared opening key has a solution that gives a solution of an instance of
`list`: `D` or a compression map. -/
def OpenKey.Reduces (list : List Instance) (key : B.OpenKey) : Prop :=
  Akita.Reduces list key.matrix false ∨ PathReduces list B.chain key.chain

/-- Each solution that a witness of `R_bind` of a level with opening groups holds gives a
solution of an instance of `list`, when `list` covers the instances of the shapes of the
level. -/
theorem reduces {key : (g : G) → Key (B.shape g)} {openKey : B.OpenKey} {innerBound : G → ℕ}
    (covered : ∀ g, Covered list ((B.shape g).instances false (innerBound g)))
    (coveredOpen : Covered list B.openInstances)
    (z : (Σ g, ShortKernel (key g).inner (innerBound g)) ⊕
      ((Σ g, OuterBreak (key g)) ⊕ B.OpenBreak openKey)) :
    (∃ g, (key g).Reduces list false) ∨ openKey.Reduces list := by
  rcases z with ⟨g, z⟩ | ⟨g, z⟩ | z | z
  · exact Or.inl ⟨g, Key.reduces_inner (euclidean := false) (covered g) z⟩
  · exact Or.inl ⟨g, Key.reduces_outer (covered g) z⟩
  · exact Or.inr (Or.inl (Reduces.of_kernel (euclidean := false)
      (by simp only [Fintype.card_sigma, Fintype.card_prod, Fintype.card_fin])
      (coveredOpen _ List.mem_cons_self) z))
  · exact Or.inr (Or.inr (PathReduces.of_kernel
      (fun v member => coveredOpen v (List.mem_cons_of_mem _ member)) z))

end Batch

end NightstreamFPrime.Spec.Compression.Akita
