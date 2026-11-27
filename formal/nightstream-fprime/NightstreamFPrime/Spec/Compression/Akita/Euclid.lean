import NightstreamFPrime.Spec.Compression.Akita.Fold

/-! The Euclidean norms of Akita and the Euclidean route of the weak binding of one fold
(ePrint 2026/1983, revision of 2026-11-23, Section 3.1, Lemma 3.1, Definition 3.2, Lemma 10.23,
and Corollaries 10.26 and 10.35).

- `normSq` and `vecNormSq` are `‖·‖_2²` of a ring element and of a vector, as natural numbers.
  Each Euclidean bound of this file is a bound on the square, so the file has no square root.
- `sum_sq_le_of_le_add` is the triangle inequality `‖x − y‖_2 ≤ ‖x‖_2 + ‖y‖_2` on squares:
  `‖x − y‖_2² ≤ P + Q + 2·T` for `‖x‖_2² ≤ P`, `‖y‖_2² ≤ Q` and `P·Q ≤ T²`.
- `MulBound Γ c` is the bound `‖c‖_mul,2 ≤ Γ` of Akita Section 3.1 as the bound of an operator:
  `‖c·v‖_2² ≤ Γ²·‖v‖_2²` for each `v`. `VecMulBound Z v` is `‖v‖_mul,2 ≤ Z`:
  `‖a·v‖_2² ≤ Z²·‖a‖_2²` for each `a`. The second and the third inequality of Akita (19) are these
  definitions. The file has no proof that the spectral formula `max_t |c(ζ_t)|` of Section 3.1
  gives `MulBound`.
- `normSq_mul_le` and `vecNormSq_smul_le` are the first inequality of Akita (19):
  `‖c·v‖_2 ≤ ‖c‖_1·‖v‖_2`.
- `ShortKernelSq M β` is a solution of `MSIS^{(2)}_{q,d}(n, m, ·)` for `M` with `‖z‖_2² ≤ β`
  (Akita Definition 3.2 with `p = 2`).
- `sourceCollisionSq_isSome` is Akita Lemma 10.23 with the bound (175), and
  `sourceCollisionSq_isSome_of_bounds` and `weakCollisionSq_isSome` have the bound (177).
- `extract_euclidean` and `weakCollisionSq_isSome_of_radius` are Akita Corollary 10.26 with the
  radius (179), which is Corollary 10.35 (ii). `extract_opens_of_radius` and
  `weakCollision_isSome_of_radius` are Corollary 10.35 (i). -/

namespace NightstreamFPrime.Spec.Compression.Akita

/-- The triangle inequality of the Euclidean norm on squares: if `h ≤ f + g` at each position,
`Σ f² ≤ P`, `Σ g² ≤ Q` and `P·Q ≤ T²`, then `Σ h² ≤ P + Q + 2·T`. -/
theorem sum_sq_le_of_le_add {κ : Type} (s : Finset κ) {f g h : κ → ℕ} {P Q T : ℕ}
    (pointwise : ∀ k ∈ s, h k ≤ f k + g k) (left : ∑ k ∈ s, f k ^ 2 ≤ P)
    (right : ∑ k ∈ s, g k ^ 2 ≤ Q) (cross : P * Q ≤ T ^ 2) :
    ∑ k ∈ s, h k ^ 2 ≤ P + Q + 2 * T := by
  have schwarz : (∑ k ∈ s, f k * g k) ^ 2 ≤ T ^ 2 :=
    (Finset.sum_mul_sq_le_sq_mul_sq s f g).trans ((Nat.mul_le_mul left right).trans cross)
  have product : ∑ k ∈ s, f k * g k ≤ T := (Nat.pow_le_pow_iff_left (by decide)).mp schwarz
  calc ∑ k ∈ s, h k ^ 2 ≤ ∑ k ∈ s, (f k ^ 2 + g k ^ 2 + 2 * (f k * g k)) :=
        Finset.sum_le_sum fun k member =>
          (Nat.pow_le_pow_left (pointwise k member) 2).trans_eq (by ring)
    _ = ∑ k ∈ s, f k ^ 2 + ∑ k ∈ s, g k ^ 2 + 2 * ∑ k ∈ s, f k * g k := by
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.mul_sum]
    _ ≤ P + Q + 2 * T :=
        Nat.add_le_add (Nat.add_le_add left right) (Nat.mul_le_mul_left 2 product)

section Norms

variable {d : ℕ} [NeZero d]

/-- `‖x‖_2²`, the sum of the squares of the magnitudes of the coefficients of `x`
(Akita Section 3.1). -/
noncomputable def normSq (x : Rq d) : ℕ := ∑ i ∈ Finset.range d, magnitude (coeff x i) ^ 2

/-- `‖v‖_2²` of a vector, over all coefficients of its entries (Akita Section 3.1). -/
noncomputable def vecNormSq {ι : Type} [Fintype ι] (v : ι → Rq d) : ℕ := ∑ j, normSq (v j)

/-- The product with `X` keeps `‖·‖_2`. -/
theorem normSq_root_mul (x : Rq d) : normSq (AdjoinRoot.root (modulus d) * x) = normSq x := by
  obtain ⟨n, rfl⟩ : ∃ n, d = n + 1 := ⟨d - 1, (Nat.succ_pred_eq_of_pos (NeZero.pos d)).symm⟩
  unfold normSq
  rw [Finset.sum_range_succ', Finset.sum_range_succ]
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i member => ?_) ?_
  · rw [coeff_root_mul x (Nat.add_lt_add_right (Finset.mem_range.mp member) 1),
      ite_eq_right (Nat.succ_ne_zero i), Nat.add_sub_cancel]
  · rw [coeff_root_mul x (Nat.succ_pos n), ite_eq_left rfl, magnitude_neg, Nat.add_sub_cancel]

theorem normSq_root_pow_mul (k : ℕ) (x : Rq d) :
    normSq (AdjoinRoot.root (modulus d) ^ k * x) = normSq x := by
  induction k with
  | zero => rw [pow_zero, one_mul]
  | succ k step => rw [pow_succ', mul_assoc, normSq_root_mul, step]

/-- A coefficient of `c·s` is a sum of the coefficients of `c` times coefficients of the shifts
`X^i·s`. -/
theorem magnitude_coeff_mul_le (c s : Rq d) (k : ℕ) :
    magnitude (coeff (c * s) k) ≤ ∑ i ∈ Finset.range d,
      magnitude (coeff c i) * magnitude (coeff (AdjoinRoot.root (modulus d) ^ i * s) k) := by
  have expanded : c * s =
      ∑ i ∈ Finset.range d, coeff c i • (AdjoinRoot.root (modulus d) ^ i * s) := by
    conv_lhs => rw [← ofCoeffs_coeff c, ofCoeffs_eq_sum, Finset.sum_mul]
    exact Finset.sum_congr rfl fun i _ => smul_mul_assoc _ _ _
  rw [expanded, coeff_sum]
  refine (magnitude_sum_le _ _).trans (Finset.sum_le_sum fun i _ => ?_)
  rw [coeff_smul]
  exact magnitude_mul_le _ _

/-- Akita Lemma 3.1, inequality (19), first bound: `‖c·s‖_2 ≤ ‖c‖_1 · ‖s‖_2`. -/
theorem normSq_mul_le (c s : Rq d) : normSq (c * s) ≤ normOne c ^ 2 * normSq s := by
  calc normSq (c * s)
      ≤ ∑ k ∈ Finset.range d, (∑ i ∈ Finset.range d, magnitude (coeff c i) *
          magnitude (coeff (AdjoinRoot.root (modulus d) ^ i * s) k)) ^ 2 :=
        Finset.sum_le_sum fun k _ => Nat.pow_le_pow_left (magnitude_coeff_mul_le c s k) 2
    _ ≤ ∑ k ∈ Finset.range d, (∑ i ∈ Finset.range d, magnitude (coeff c i)) *
          ∑ i ∈ Finset.range d, magnitude (coeff c i) *
            magnitude (coeff (AdjoinRoot.root (modulus d) ^ i * s) k) ^ 2 :=
        Finset.sum_le_sum fun k _ => Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul _
          (fun _ _ => Nat.zero_le _) (fun _ _ => Nat.zero_le _) fun i _ => le_of_eq (by ring)
    _ = (∑ i ∈ Finset.range d, magnitude (coeff c i)) * ∑ i ∈ Finset.range d,
          magnitude (coeff c i) * normSq (AdjoinRoot.root (modulus d) ^ i * s) := by
        rw [← Finset.mul_sum, Finset.sum_comm]
        exact congrArg _ (Finset.sum_congr rfl fun i _ => (Finset.mul_sum _ _ _).symm)
    _ = normOne c ^ 2 * normSq s := by
        simp only [normSq_root_pow_mul]
        rw [← Finset.sum_mul, normOne]
        ring

/-- The first bound of (19) for a vector: `‖c·v‖_2 ≤ ‖c‖_1 · ‖v‖_2`. -/
theorem vecNormSq_smul_le {ι : Type} [Fintype ι] (c : Rq d) (v : ι → Rq d) :
    vecNormSq (c • v) ≤ normOne c ^ 2 * vecNormSq v := by
  rw [vecNormSq, vecNormSq, Finset.mul_sum]
  exact Finset.sum_le_sum fun j _ => normSq_mul_le c (v j)

/-- `‖x − y‖_2 ≤ ‖x‖_2 + ‖y‖_2` on squares. -/
theorem normSq_sub_le {x y : Rq d} {P Q T : ℕ} (left : normSq x ≤ P) (right : normSq y ≤ Q)
    (cross : P * Q ≤ T ^ 2) : normSq (x - y) ≤ P + Q + 2 * T :=
  sum_sq_le_of_le_add (Finset.range d)
    (fun i _ => by rw [coeff_sub]; exact magnitude_sub_le _ _) left right cross

/-- `‖v − w‖_2 ≤ ‖v‖_2 + ‖w‖_2` on squares, for vectors. -/
theorem vecNormSq_sub_le {ι : Type} [Fintype ι] {v w : ι → Rq d} {P Q T : ℕ}
    (left : vecNormSq v ≤ P) (right : vecNormSq w ≤ Q) (cross : P * Q ≤ T ^ 2) :
    vecNormSq (v - w) ≤ P + Q + 2 * T := by
  have flat (u : ι → Rq d) : vecNormSq u =
      ∑ p ∈ Finset.univ ×ˢ Finset.range d, magnitude (coeff (u p.1) p.2) ^ 2 :=
    (Finset.sum_product' Finset.univ (Finset.range d)
      fun j i => magnitude (coeff (u j) i) ^ 2).symm
  rw [flat] at left right ⊢
  exact sum_sq_le_of_le_add _
    (fun p _ => by rw [Pi.sub_apply, coeff_sub]; exact magnitude_sub_le _ _) left right cross

/-- `‖c‖_mul,2 ≤ Γ` (Akita Section 3.1) as the bound of the operator `v ↦ c·v`:
`‖c·v‖_2 ≤ Γ·‖v‖_2` for each `v`. -/
def MulBound (bound : ℕ) (c : Rq d) : Prop := ∀ v : Rq d, normSq (c * v) ≤ bound ^ 2 * normSq v

/-- `‖v‖_mul,2 ≤ Z` for a vector `v` (Akita Section 3.1) as the bound of the operator `a ↦ a·v`:
`‖a·v‖_2 ≤ ‖a‖_2·Z` for each `a`, which is the third bound of Akita (19). -/
def VecMulBound {ι : Type} [Fintype ι] (bound : ℕ) (v : ι → Rq d) : Prop :=
  ∀ a : Rq d, vecNormSq (a • v) ≤ bound ^ 2 * normSq a

/-- `‖c‖_mul,2 ≤ ‖c‖_1`. -/
theorem mulBound_of_normOne_le {bound : ℕ} {c : Rq d} (mass : normOne c ≤ bound) :
    MulBound bound c :=
  fun v => (normSq_mul_le c v).trans (Nat.mul_le_mul_right _ (Nat.pow_le_pow_left mass 2))

/-- Akita Lemma 3.1, inequality (19), second bound: `‖c·v‖_2 ≤ ‖c‖_mul,2 · ‖v‖_2` for a vector
`v`. The operator acts on each entry, so the bound has no factor for the length of `v`. -/
theorem MulBound.vecNormSq_smul_le {ι : Type} [Fintype ι] {bound : ℕ} {c : Rq d}
    (operator : MulBound bound c) (v : ι → Rq d) :
    vecNormSq (c • v) ≤ bound ^ 2 * vecNormSq v := by
  rw [vecNormSq, vecNormSq, Finset.mul_sum]
  exact Finset.sum_le_sum fun j _ => operator (v j)

/-- `‖c − c'‖_mul,2 ≤ 2Γ` for two challenges with `‖·‖_mul,2 ≤ Γ` (Akita Corollary 10.26,
proof). -/
theorem MulBound.sub {bound : ℕ} {c c' : Rq d} (left : MulBound bound c)
    (right : MulBound bound c') : MulBound (2 * bound) (c - c') := by
  intro v
  rw [sub_mul]
  exact (normSq_sub_le (left v) (right v) (sq _).ge).trans_eq (by ring)

end Norms

section ModuleSis

variable {d : ℕ} [NeZero d] {rows cols : Type} [Fintype cols]

/-- A solution of `MSIS^{(2)}_{q,d}(n, m, ·)` for the matrix `M` (Akita Definition 3.2 with
`p = 2`): a vector `z ≠ 0` with `M z = 0` and `‖z‖_2² ≤ bound`. -/
structure ShortKernelSq (M : Matrix rows cols (Rq d)) (bound : ℕ) where
  vector : cols → Rq d
  nonzero : vector ≠ 0
  kernel : M.mulVec vector = 0
  short : vecNormSq vector ≤ bound

/-- `x` and `x'` are two inputs of `M` with one image, and `‖x − x'‖_2² ≤ bound`. -/
def CollidesSq (M : Matrix rows cols (Rq d)) (bound : ℕ) (x x' : cols → Rq d) : Prop :=
  x ≠ x' ∧ M.mulVec x = M.mulVec x' ∧ vecNormSq (x - x') ≤ bound

open Classical in
/-- The solution `x − x'` of Module-SIS in the Euclidean norm for `M`, when `x` and `x'`
collide. -/
noncomputable def ShortKernelSq.ofCollision? (M : Matrix rows cols (Rq d)) (bound : ℕ)
    (x x' : cols → Rq d) : Option (ShortKernelSq M bound) :=
  if collides : CollidesSq M bound x x' then
    some ⟨x - x', sub_ne_zero.mpr collides.1,
      by rw [Matrix.mulVec_sub, collides.2.1, sub_self], collides.2.2⟩
  else none

theorem ShortKernelSq.ofCollision?_isSome {M : Matrix rows cols (Rq d)} {bound : ℕ}
    {x x' : cols → Rq d} (collides : CollidesSq M bound x x') :
    (ShortKernelSq.ofCollision? M bound x x').isSome := by
  unfold ofCollision?
  split
  · rfl
  · contradiction

end ModuleSis

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

/-- The Euclidean line of the weak certificates of Akita Definition 10.22 (i):
`‖ι(c̄_i)‖_mul,2 ≤ κ̄_mul,2` and `‖ι(c̄_i)·s_i‖_2² ≤ β̄_2²` for each source block. -/
structure Certified.Euclidean (mulBound sqBound : ℕ) (st : Certified S) : Prop where
  factorMul : ∀ i, MulBound mulBound (st.factor i)
  sourceSq : ∀ i, vecNormSq (st.factor i • st.source i) ≤ sqBound

/-- The vector `z_A` of the proof of Akita Lemma 10.23, for the first block where it is a solution
for `A` in the Euclidean norm. -/
noncomputable def sourceCollisionSq (key : Key S) (bound : ℕ) (st st' : Certified S) :
    Option (ShortKernelSq key.inner bound) :=
  (List.finRange S.blocks).findSome? fun i =>
    ShortKernelSq.ofCollision? key.inner bound (st'.factor i • (st.factor i • st.source i))
      (st.factor i • (st'.factor i • st'.source i))

section Binding

variable {key : Key S} {payload : S.Payload} {st st' : Certified S} {table table' : Table}
  {factorBound sourceBound factorBound' sourceBound' : ℕ}

/-- Akita Lemma 10.23, "Source blocks", in the Euclidean norm. `first` and `second` bound the
squares of the two summands `‖ι(c̄'_i)·r_i‖_2` and `‖ι(c̄_i)·r'_i‖_2` of `z_A`, and `balance` and
`covers` state that `bound` is at least the square of the sum of the two bounds. Each bound of
Akita (19) gives `first` and `second`, which gives the three lines of Akita (175):
`vecNormSq_smul_le`, `MulBound.vecNormSq_smul_le` and `VecMulBound`. -/
theorem sourceCollisionSq_isSome {bound firstSq secondSq cross : ℕ}
    (opens : st.Opens factorBound sourceBound key payload table)
    (opens' : st'.Opens factorBound' sourceBound' key payload table')
    (first : ∀ i, vecNormSq (st'.factor i • (st.factor i • st.source i)) ≤ firstSq)
    (second : ∀ i, vecNormSq (st.factor i • (st'.factor i • st'.source i)) ≤ secondSq)
    (balance : firstSq * secondSq ≤ cross ^ 2) (covers : firstSq + secondSq + 2 * cross ≤ bound)
    (inner : st.inner = st'.inner) (different : st.source ≠ st'.source) :
    (sourceCollisionSq key bound st st').isSome := by
  obtain ⟨i, differs⟩ := Function.ne_iff.mp different
  refine List.findSome?_isSome_iff.mpr ⟨i, List.mem_finRange i,
    ShortKernelSq.ofCollision?_isSome ⟨?_, ?_, ?_⟩⟩
  · intro same
    rw [smul_smul, smul_smul, mul_comm] at same
    exact differs (((opens.unit i).mul (opens'.unit i)).smul_left_cancel.mp same)
  · simp only [Matrix.mulVec_smul]
    rw [opens.source.innerRows, opens'.source.innerRows, inner, smul_comm]
  · exact (vecNormSq_sub_le (first i) (second i) balance).trans covers

/-- Akita Lemma 10.23 with the second line of the bound (175): for two certified decommitments
with the Euclidean certificates `(κ̄_mul,2, β̄_2²)` and `(κ̄'_mul,2, β̄'_2²)`, the solution for `A`
has `‖z_A‖_2 ≤ κ̄'_mul,2·β̄_2 + κ̄_mul,2·β̄'_2`. -/
theorem sourceCollisionSq_isSome_of_euclidean {bound cross mulBound sqBound mulBound' sqBound' : ℕ}
    (opens : st.Opens factorBound sourceBound key payload table)
    (opens' : st'.Opens factorBound' sourceBound' key payload table')
    (euclid : st.Euclidean mulBound sqBound) (euclid' : st'.Euclidean mulBound' sqBound')
    (balance : mulBound' ^ 2 * sqBound * (mulBound ^ 2 * sqBound') ≤ cross ^ 2)
    (covers : mulBound' ^ 2 * sqBound + mulBound ^ 2 * sqBound' + 2 * cross ≤ bound)
    (inner : st.inner = st'.inner) (different : st.source ≠ st'.source) :
    (sourceCollisionSq key bound st st').isSome :=
  sourceCollisionSq_isSome opens opens'
    (fun i => ((euclid'.factorMul i).vecNormSq_smul_le _).trans
      (Nat.mul_le_mul_left _ (euclid.sourceSq i)))
    (fun i => ((euclid.factorMul i).vecNormSq_smul_le _).trans
      (Nat.mul_le_mul_left _ (euclid'.sourceSq i)))
    balance covers inner different

/-- Akita Lemma 10.23 with the second line of the bound (177): for two certified decommitments
with the same bounds `κ̄_mul,2` and `β̄_2²`, the solution for `A` has
`‖z_A‖_2² ≤ (2·κ̄_mul,2·β̄_2)²`. -/
theorem sourceCollisionSq_isSome_of_bounds {mulBound sqBound : ℕ}
    (opens : st.Opens factorBound sourceBound key payload table)
    (opens' : st'.Opens factorBound' sourceBound' key payload table')
    (euclid : st.Euclidean mulBound sqBound) (euclid' : st'.Euclidean mulBound sqBound)
    (inner : st.inner = st'.inner) (different : st.source ≠ st'.source) :
    (sourceCollisionSq key (4 * mulBound ^ 2 * sqBound) st st').isSome :=
  sourceCollisionSq_isSome_of_euclidean opens opens' euclid euclid' (sq _).ge
    (le_of_eq (by ring)) inner different

end Binding

variable {E : Type} [CommRing E] [Module Zq E]

/-- A solution of Module-SIS for one matrix of a fold on the Euclidean route (Akita Table 3): for
`A` with `‖z‖_2² ≤ bound`, for `B` or `D` with the bound `b⋆ − 1`, or for `F_1`, `F_2`, `H_1` or
`H_2` with the bound 1. -/
abbrev FoldBreakSq (st : Statement O E) (bound : ℕ) : Type :=
  ShortKernelSq st.key.inner bound ⊕ (OuterBreak st.key ⊕ OpenBreak st.openKey)

/-- The reduction of Akita Lemma 10.23 for two weak openings on the Euclidean route: the solution
of the digit paths, or else the solution `z_A` for `A`. -/
noncomputable def weakCollisionSq (st : Statement O E) (bound : ℕ) (ω ω' : Weak O) :
    Option (FoldBreakSq st bound) :=
  ((carriedCollision st ω.carried ω'.carried).map Sum.inr).or
    ((sourceCollisionSq st.key bound ω.toCertified ω'.toCertified).map Sum.inl)

section Weak

variable {β : Module.Basis (Fin O.planes) Zq E} {st : Statement O E} {ω ω' : Weak O}
  {factorBound sourceBound factorBound' sourceBound' : ℕ} {table table' : Table}

/-- Akita Lemma 10.23 (weak binding of one fold) with the second line of the bound (177): two weak
openings of one statement with the Euclidean certificates `(κ̄_mul,2, β̄_2²)` that differ on a
source block or on a carried digit vector give a solution of Module-SIS: for `A` with
`‖z_A‖_2² ≤ 4·κ̄_mul,2²·β̄_2²`, for `B` or `D` with the bound `b⋆ − 1`, or for a compression map with
the bound 1. -/
theorem weakCollisionSq_isSome {mulBound sqBound : ℕ}
    (opens : ω.Opens β factorBound sourceBound st table)
    (opens' : ω'.Opens β factorBound' sourceBound' st table')
    (euclid : ω.toCertified.Euclidean mulBound sqBound)
    (euclid' : ω'.toCertified.Euclidean mulBound sqBound)
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (weakCollisionSq st (4 * mulBound ^ 2 * sqBound) ω ω').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits : ω.carried = ω'.carried
  · exact Or.inr (sourceCollisionSq_isSome_of_bounds opens.certified opens'.certified euclid
      euclid' (congrArg Carried.inner digits) (different.resolve_right (not_not.mpr digits)))
  · exact Or.inl (carriedCollision_isSome opens.carried opens'.carried digits)

/-- Akita Corollary 10.26 with the radius (179), second term, which is Corollary 10.35 (ii): two
weak openings with `κ̄_mul,2 = 2Γ` and `β̄_2² = 4·S_max` give a solution for `A` with
`‖z_A‖_2² ≤ 64·Γ²·S_max`, or a solution for another matrix. -/
theorem weakCollisionSq_isSome_of_radius {operator energy : ℕ}
    (opens : ω.Opens β factorBound sourceBound st table)
    (opens' : ω'.Opens β factorBound' sourceBound' st table')
    (euclid : ω.toCertified.Euclidean (2 * operator) (4 * energy))
    (euclid' : ω'.toCertified.Euclidean (2 * operator) (4 * energy))
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (weakCollisionSq st (64 * operator ^ 2 * energy) ω ω').isSome := by
  have collision := weakCollisionSq_isSome opens opens' euclid euclid' different
  rwa [show 4 * (2 * operator) ^ 2 * (4 * energy) = 64 * operator ^ 2 * energy by ring]
    at collision

/-- Akita Corollary 10.35 (i): two weak openings with `κ̄_1 = 2·κ_1` and `β̄_∞ = 2·Z_∞` give a
solution for `A` with `‖z_A‖_∞ ≤ 8·κ_1·Z_∞`, or a solution for another matrix. -/
theorem weakCollision_isSome_of_radius {mass response : ℕ}
    (opens : ω.Opens β (2 * mass) (2 * response) st table)
    (opens' : ω'.Opens β (2 * mass) (2 * response) st table')
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (weakCollision st (8 * mass * response) ω ω').isSome := by
  have collision := weakCollision_isSome_of_bounds opens opens' different
  rwa [show 2 * (2 * mass) * (2 * response) = 8 * mass * response by ring] at collision

end Weak

section Extraction

variable {β : Module.Basis (Fin O.planes) Zq E} {st : Statement O E} {fork : Fork O fold}
  {family : Set (Rq O.carrierDegree)} {digits : Carried O}

/-- Akita Corollary 10.26 for the extracted weak opening of Akita Theorem 10.31: when each
challenge of the family has `‖ι(c)‖_mul,2 ≤ Γ` and each response of a child has
`‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max`, the extracted opening has `κ̄_mul,2 = 2Γ` and `β̄_2² = 4·S_max`. -/
theorem extract_euclidean {operator energy : ℕ} (units : UnitDifferences family)
    (accepts : fork.Accepts β family st)
    (bounded : ∀ c ∈ family, MulBound operator (embed O.stride O.stride_tower c))
    (first : ∀ i, vecNormSq (gadget fold.responseBase (fork.first i).witness.response) ≤ energy)
    (second : ∀ i,
      vecNormSq (gadget fold.responseBase (fork.second i).witness.response) ≤ energy) :
    (extract fork digits).toCertified.Euclidean (2 * operator) (4 * energy) where
  factorMul i := by
    show MulBound (2 * operator) (embed O.stride O.stride_tower (fork.difference i))
    rw [Fork.difference, map_sub]
    exact (bounded _ (accepts.secondMember i)).sub (bounded _ (accepts.firstMember i))
  sourceSq i := by
    rw [extract_factor_smul_source fork digits i (accepts.isUnit_difference units i)]
    exact (vecNormSq_sub_le (second i) (first i) (sq _).ge).trans_eq (by ring)

/-- Akita Corollary 10.35 (i) for the extracted weak opening of Akita Theorem 10.31: when each
response of a child has `‖G_{b_z,m_A}·ẑ‖_∞ ≤ Z_∞`, the extracted opening has `κ̄_1 = 2ω` and
`β̄_∞ = 2·Z_∞`. -/
theorem extract_opens_of_radius {mass peak response : ℕ} (units : UnitDifferences family)
    (sparse : ∀ c ∈ family, Sparse mass peak c) (base : 2 ≤ fold.responseBase)
    (accepts : fork.Accepts β family st) (opens : digits.Opens β st)
    (firstDigits : ∀ i, (fork.first i).witness.toCarried = digits)
    (secondDigits : ∀ i, (fork.second i).witness.toCarried = digits)
    (first : ∀ i,
      vecNormInf (gadget fold.responseBase (fork.first i).witness.response) ≤ response)
    (second : ∀ i,
      vecNormInf (gadget fold.responseBase (fork.second i).witness.response) ≤ response) :
    (extract fork digits).Opens β (2 * mass) (2 * response) st
      (S.decode (extract fork digits).source) := by
  have opened := extract_opens units sparse base accepts opens firstDigits secondDigits
  exact {
    certified := {
      source := opened.certified.source
      unit := opened.certified.unit
      factorNorm := opened.certified.factorNorm
      sourceNorm := fun i => by
        rw [extract_factor_smul_source fork digits i (accepts.isUnit_difference units i)]
        exact (vecNormInf_sub_le _ _).trans
          ((Nat.add_le_add (second i) (first i)).trans_eq (Nat.two_mul response).symm) }
    subring := opened.subring
    openRows := opened.openRows
    planes := opened.planes
    value := opened.value
    partialDigits := opened.partialDigits
    openDigits := opened.openDigits }

end Extraction

end NightstreamFPrime.Spec.Compression.Akita
