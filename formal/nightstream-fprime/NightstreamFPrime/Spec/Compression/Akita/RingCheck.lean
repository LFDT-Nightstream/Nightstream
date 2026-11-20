import Mathlib.Algebra.Polynomial.Roots
import NightstreamFPrime.Spec.Compression.Akita.Opening

/-! The checks of the ring relations of Akita (ePrint 2026/1983, revision of 2026-10-15,
Sections 3.6 and 7.3): the quotient-lift form and the quotient-free form.

- `RingRow` is one ring row in the normal form (110): for each occurrence `c` a public multiplier
  `A_{r,c}` and an address map (108), and a public target `Y_r`. `RingRow.read` is `W_{r,c}` of
  (109) for a flat witness, `RingRow.residual` is `Z_r`, and `RingRow.Holds` is `Z_r = 0`.
- `RingRow.holds_iff_quotient` is Akita Lemma 3.4 for one row: the row holds when the identity
  (111) in `Z_q[X]` has a quotient of degree below `d − 1`, and only then.
- `RingRow.LiftCheck` is the check (112) at a point `α`. `RingRow.liftCheck_of_holds` is Akita
  Lemma 7.1, item 1. `RingRow.liftBad` is a set of at most `2d − 1` points, and
  `RingRow.holds_of_liftCheck` states that a check that passes at a point outside this set gives
  the row. `card_liftCheck_le` is Akita Lemma 7.1, item 2, for rows of different degrees.
- `reducedEval α` is `P ↦ (P mod (X^d + 1))(α)`. `residueKernel` is Akita (117),
  `residueKernel_eq` is (118), and `reducedEval_mul` is Akita Lemma 7.3 with (119).
- `RingRow.reduced` is `z_r(α)` of Akita (120) and `RingRow.FreeCheck` is the quotient-free check
  `z_r(α) = 0`. `RingRow.freeCheck_of_holds`, `RingRow.holds_of_freeCheck` and
  `card_freeCheck_le` are Akita Theorem 7.4, items 1 and 2, for the ring rows, with a set
  `RingRow.freeBad` of at most `d − 1` points.
- `reducedEval_embed` states that an embedded challenge `ι(c)` has the value `c(α^{k·h_car})`,
  which is the challenge term of Akita (114).
- The field `E` of the point `α` is a domain with an injective map from `Z_q`.
  `extension_algebraMap_injective` is this condition for `K = F_q[U]/(U² − 2)`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

section Representative

variable {d : ℕ} [NeZero d]

theorem natDegree_modulus (d : ℕ) [NeZero d] : (modulus d).natDegree = d :=
  natDegree_eq_of_degree_eq_some (degree_modulus d)

theorem natDegree_rep_lt (x : Rq d) : (rep x).natDegree < d := by
  by_cases zero : rep x = 0
  · rw [zero, natDegree_zero]
    exact NeZero.pos d
  · exact (natDegree_lt_iff_degree_lt zero).mpr (degree_rep_lt x)

variable {E : Type} [CommRing E] [Algebra Zq E]

/-- The value at `α` of the representative of `x` of degree below `d`. -/
theorem aeval_rep (x : Rq d) (α : E) :
    aeval α (rep x) = ∑ i ∈ Finset.range d, coeff x i • α ^ i :=
  aeval_eq_sum_range' (natDegree_rep_lt x) α

/-- `P ↦ (P mod (X^d + 1))(α)` (Akita Section 7.3, the quotient-free form). -/
noncomputable def reducedEval (α : E) : Rq d →ₗ[Zq] E :=
  (aeval α).toLinearMap ∘ₗ AdjoinRoot.modByMonicHom (modulus_monic d)

theorem reducedEval_apply (α : E) (x : Rq d) : reducedEval α x = aeval α (rep x) :=
  rfl

/-- The residue kernel `κ^{(d)}_{A,α}(j) = (A(X)·X^j mod (X^d + 1))(α)` of Akita (117). -/
noncomputable def residueKernel (A : Rq d) (α : E) (j : ℕ) : E :=
  reducedEval α (A * AdjoinRoot.root (modulus d) ^ j)

/-- Akita (118), with the sum over the position `i` of the product: the position `i ≥ j` has
the coefficient `a_{i−j}`, and the position `i < j` has the coefficient `−a_{d+i−j}` from the
wrap `X^d = −1`. -/
theorem residueKernel_eq (A : Rq d) (α : E) {j : ℕ} (le : j ≤ d) :
    residueKernel A α j = ∑ i ∈ Finset.range d,
      (if i < j then -coeff A (d + i - j) else coeff A (i - j)) • α ^ i := by
  rw [residueKernel, reducedEval_apply, aeval_rep]
  exact Finset.sum_congr rfl fun i member => by
    rw [mul_comm, coeff_root_pow_mul le A (Finset.mem_range.mp member)]

/-- Akita Lemma 7.3 (transpose-convolution identity), (119):
`(A·W mod (X^d + 1))(α) = Σ_j w_j · κ^{(d)}_{A,α}(j)`. -/
theorem reducedEval_mul (A W : Rq d) (α : E) :
    reducedEval α (A * W) = ∑ j ∈ Finset.range d, coeff W j • residueKernel A α j := by
  conv_lhs => rw [← ofCoeffs_coeff W, ofCoeffs_eq_sum, Finset.mul_sum]
  rw [map_sum]
  exact Finset.sum_congr rfl fun j _ => by rw [mul_smul_comm, map_smul, residueKernel]

variable {dCar : ℕ} [NeZero dCar] {stride : ℕ}

theorem rep_embed (tower : stride * dCar = d) (c : Rq dCar) :
    rep (embed stride tower c) = expand Zq stride (rep c) := by
  have positive := stride_pos tower
  have below : (expand Zq stride (rep c)).degree < d := by
    rw [degree_lt_iff_coeff_zero]
    intro m large
    rw [coeff_expand positive]
    split
    · exact coeff_eq_zero c ((Nat.le_div_iff_mul_le positive).mpr
        (by rw [Nat.mul_comm, tower]; exact large))
    · rfl
  conv_lhs => rw [← mk_rep c]
  rw [embed_mk, rep_mk_of_degree_lt below]

/-- The embedded challenge `ι(c) = c(X^{k·h_car})` has the value `c(α^{k·h_car})` at `α`, which
is the challenge term of an `A` row in Akita (114). -/
theorem reducedEval_embed (tower : stride * dCar = d) (c : Rq dCar) (α : E) :
    reducedEval α (embed stride tower c) = reducedEval (α ^ stride) c := by
  rw [reducedEval_apply, reducedEval_apply, rep_embed, expand_aeval]

end Representative

/-- One ring row in the normal form of Akita (110): the native degree `d_r`, the public
multipliers `A_{r,c}` of the occurrences `c ∈ Occ_r`, the address maps `addr_{r,c}` of (108), and
the public target `Y_r`. -/
structure RingRow where
  degree : ℕ
  [positive : NeZero degree]
  occurrences : ℕ
  multiplier : Fin occurrences → Rq degree
  address : Fin occurrences → ℕ → ℕ
  target : Rq degree

attribute [instance] RingRow.positive

namespace RingRow

variable (row : RingRow)

/-- `W_{r,c}(X) = Σ_{j<d_r} w_{addr_{r,c}(j)}·X^j` of Akita (109), for the flat witness `w`. -/
noncomputable def read (w : Table) (c : Fin row.occurrences) : Rq row.degree :=
  ofCoeffs fun j => w (row.address c j)

/-- The residual `Z_r = Σ_c A_{r,c}·W_{r,c} − Y_r` of Akita (110). -/
noncomputable def residual (w : Table) : Rq row.degree :=
  ∑ c, row.multiplier c * row.read w c - row.target

/-- The row holds for the witness `w`: `Z_r = 0` in `R_r` (Akita (110)). -/
def Holds (w : Table) : Prop := row.residual w = 0

/-- The left side of Akita (111) in `Z_q[X]`, with the representatives of degree below `d_r`. -/
noncomputable def lifted (w : Table) : Zq[X] :=
  ∑ c, rep (row.multiplier c) * rep (row.read w c) - rep row.target

theorem mk_lifted (w : Table) :
    AdjoinRoot.mk (modulus row.degree) (row.lifted w) = row.residual w := by
  simp only [lifted, residual, map_sub, map_sum, map_mul, mk_rep]

/-- Each product of (111) has a degree of at most `2d − 2`. -/
theorem natDegree_lifted_le (w : Table) : (row.lifted w).natDegree ≤ 2 * row.degree - 2 := by
  have positive := NeZero.pos row.degree
  refine (natDegree_sub_le _ _).trans (max_le
    (natDegree_sum_le_of_forall_le _ _ fun c _ => natDegree_mul_le.trans ?_) ?_)
  · have left := natDegree_rep_lt (row.multiplier c)
    have right := natDegree_rep_lt (row.read w c)
    omega
  · have target := natDegree_rep_lt row.target
    omega

theorem holds_iff_dvd (w : Table) : row.Holds w ↔ modulus row.degree ∣ row.lifted w := by
  rw [Holds, ← mk_lifted, AdjoinRoot.mk_eq_zero]

/-- Akita Lemma 3.4 (ring-switch lift) for one row: the row holds in `R_q` when there is a
quotient `r` of degree below `d − 1` with `Σ_c A_c·W_c − Y = (X^d + 1)·r` in `Z_q[X]`, and only
then. -/
theorem holds_iff_quotient (w : Table) :
    row.Holds w ↔ ∃ quotient : Zq[X],
      quotient.degree < ((row.degree - 1 : ℕ) : WithBot ℕ) ∧
        row.lifted w = modulus row.degree * quotient := by
  rw [holds_iff_dvd]
  constructor
  · rintro ⟨quotient, product⟩
    refine ⟨quotient, ?_, product⟩
    by_cases zero : quotient = 0
    · rw [zero, degree_zero]
      exact WithBot.bot_lt_coe _
    · have degree := row.natDegree_lifted_le w
      rw [product, (modulus_monic row.degree).natDegree_mul' zero, natDegree_modulus] at degree
      have positive := NeZero.pos row.degree
      have small : quotient.natDegree < row.degree - 1 := by omega
      exact (natDegree_lt_iff_degree_lt zero).mp small
  · rintro ⟨quotient, -, product⟩
    exact ⟨quotient, product⟩

variable {E : Type} [CommRing E] [Algebra Zq E]

/-- The check (112) of Akita at the point `α`, for the quotient `Q_r`:
`Σ_c Σ_{j<d_r} A_{r,c}(α)·α^j·w_{addr_{r,c}(j)} − Y_r(α) − (α^{d_r} + 1)·Q_r(α) = 0`. -/
def LiftCheck (w : Table) (quotient : Zq[X]) (α : E) : Prop :=
  ∑ c, aeval α (rep (row.multiplier c)) *
      ∑ j ∈ Finset.range row.degree, α ^ j * algebraMap Zq E (w (row.address c j)) -
    aeval α (rep row.target) - (α ^ row.degree + 1) * aeval α quotient = 0

theorem aeval_read (w : Table) (c : Fin row.occurrences) (α : E) :
    aeval α (rep (row.read w c)) =
      ∑ j ∈ Finset.range row.degree, α ^ j * algebraMap Zq E (w (row.address c j)) := by
  rw [aeval_rep]
  exact Finset.sum_congr rfl fun j member => by
    rw [read, coeff_ofCoeffs _ (Finset.mem_range.mp member), Algebra.smul_def, mul_comm]

/-- The check (112) is the identity (111) at `X = α` (Akita (24)). -/
theorem liftCheck_iff (w : Table) (quotient : Zq[X]) (α : E) :
    row.LiftCheck w quotient α ↔
      aeval α (row.lifted w - modulus row.degree * quotient) = 0 := by
  unfold LiftCheck lifted modulus
  simp only [map_sub, map_sum, map_mul, map_add, map_pow, aeval_X, map_one, aeval_read]

/-- Akita Lemma 7.1, item 1, for one row: a row that holds has a quotient of degree below
`d − 1` that passes the check (112) at each point. -/
theorem liftCheck_of_holds {w : Table} (holds : row.Holds w) :
    ∃ quotient : Zq[X], quotient.degree < ((row.degree - 1 : ℕ) : WithBot ℕ) ∧
      ∀ α : E, row.LiftCheck w quotient α := by
  obtain ⟨quotient, degree, product⟩ := (row.holds_iff_quotient w).mp holds
  exact ⟨quotient, degree, fun α =>
    (row.liftCheck_iff w quotient α).mpr (by rw [product, sub_self, map_zero])⟩

/-- `z_r(α)` of Akita (120): the witness with the residue kernels as weights, minus the value of
the target. -/
noncomputable def reduced (w : Table) (α : E) : E :=
  ∑ c, ∑ j ∈ Finset.range row.degree,
      residueKernel (row.multiplier c) α j * algebraMap Zq E (w (row.address c j)) -
    reducedEval α row.target

/-- `z_r(α)` is the value at `α` of the ring residual `Z_r` (Akita Section 7.3). -/
theorem reduced_eq (w : Table) (α : E) : row.reduced w α = reducedEval α (row.residual w) := by
  rw [reduced, residual, map_sub, map_sum]
  refine congrArg (· - reducedEval α row.target) (Finset.sum_congr rfl fun c _ => ?_)
  rw [reducedEval_mul]
  exact Finset.sum_congr rfl fun j member => by
    rw [read, coeff_ofCoeffs _ (Finset.mem_range.mp member), Algebra.smul_def, mul_comm]

/-- The quotient-free check of Akita Section 7.3 at the point `α`: `z_r(α) = 0`. -/
def FreeCheck (w : Table) (α : E) : Prop := row.reduced w α = 0

/-- Akita Theorem 7.4, item 1, for one ring row: a row that holds passes the quotient-free check
at each point. -/
theorem freeCheck_of_holds {w : Table} (holds : row.Holds w) (α : E) : row.FreeCheck w α := by
  unfold Holds at holds
  rw [FreeCheck, reduced_eq, holds, map_zero]

variable [IsDomain E]

open Classical in
/-- The roots in `E` of `Σ_c A_c·W_c − Y − (X^d + 1)·Q`: the points where the check (112) can pass
for a row that does not hold (Akita Section 3.6). -/
noncomputable def liftBad (w : Table) (quotient : Zq[X]) : Finset E :=
  ((row.lifted w - modulus row.degree * quotient).map (algebraMap Zq E)).roots.toFinset

/-- For a quotient of degree below `d`, the set `liftBad` has at most `2d − 1` points
(Akita Section 3.6). -/
theorem liftBad_card_le (w : Table) {quotient : Zq[X]}
    (degree : quotient.natDegree < row.degree) :
    (row.liftBad (E := E) w quotient).card ≤ 2 * row.degree - 1 := by
  classical
  refine (Multiset.toFinset_card_le _).trans ((card_roots' _).trans (natDegree_map_le.trans ?_))
  refine (natDegree_sub_le _ _).trans (max_le ((row.natDegree_lifted_le w).trans (by omega))
    (natDegree_mul_le.trans ?_))
  rw [natDegree_modulus]
  omega

/-- A check (112) that passes at a point outside `liftBad` gives the row (Akita Section 3.6 and
Lemma 7.1, item 2). -/
theorem holds_of_liftCheck (injective : Function.Injective (algebraMap Zq E)) {w : Table}
    {quotient : Zq[X]} {α : E} (outside : α ∉ row.liftBad w quotient)
    (check : row.LiftCheck w quotient α) : row.Holds w := by
  classical
  have root := (row.liftCheck_iff w quotient α).mp check
  have zero : row.lifted w - modulus row.degree * quotient = 0 := by
    by_contra nonzero
    refine outside (Multiset.mem_toFinset.mpr (mem_roots'.mpr
      ⟨fun mapped => nonzero ((Polynomial.map_eq_zero_iff injective).mp mapped), ?_⟩))
    rw [IsRoot.def, ← eval₂_eq_eval_map, ← aeval_def]
    exact root
  exact (row.holds_iff_dvd w).mpr ⟨quotient, sub_eq_zero.mp zero⟩

open Classical in
/-- The roots in `E` of the representative of the residual `Z_r`: the points where the
quotient-free check can pass for a row that does not hold. -/
noncomputable def freeBad (w : Table) : Finset E :=
  ((rep (row.residual w)).map (algebraMap Zq E)).roots.toFinset

/-- The set `freeBad` has at most `d − 1` points (Akita Theorem 7.4, item 2). -/
theorem freeBad_card_le (w : Table) : (row.freeBad (E := E) w).card ≤ row.degree - 1 := by
  classical
  have degree := natDegree_rep_lt (row.residual w)
  exact (Multiset.toFinset_card_le _).trans
    ((card_roots' _).trans (natDegree_map_le.trans (by omega)))

/-- A quotient-free check that passes at a point outside `freeBad` gives the row
(Akita Theorem 7.4, item 2). -/
theorem holds_of_freeCheck (injective : Function.Injective (algebraMap Zq E)) {w : Table}
    {α : E} (outside : α ∉ row.freeBad w) (check : row.FreeCheck w α) : row.Holds w := by
  classical
  have root : aeval α (rep (row.residual w)) = 0 := by
    rw [← reducedEval_apply, ← row.reduced_eq]
    exact check
  have zero : rep (row.residual w) = 0 := by
    by_contra nonzero
    refine outside (Multiset.mem_toFinset.mpr (mem_roots'.mpr
      ⟨fun mapped => nonzero ((Polynomial.map_eq_zero_iff injective).mp mapped), ?_⟩))
    rw [IsRoot.def, ← eval₂_eq_eval_map, ← aeval_def]
    exact root
  rw [Holds, ← mk_rep (row.residual w), zero, map_zero]

end RingRow

section Mixed

variable {E : Type} [CommRing E] [Algebra Zq E] [IsDomain E] {ι : Type}

/-- Akita Lemma 7.1, item 2 (mixed-dimension ring switch, soundness). The rows have the degrees
`d_b ≤ d_max` and read one witness, and each quotient has a degree below `d_b`. If one row does not
hold, the checks (116) pass together at no more than `2·d_max − 1` points `α`. -/
theorem card_liftCheck_le (injective : Function.Injective (algebraMap Zq E)) (rows : ι → RingRow)
    (quotients : ι → Zq[X]) (w : Table) {largest : ℕ} (degrees : ∀ b, (rows b).degree ≤ largest)
    (quotientDegrees : ∀ b, (quotients b).natDegree < (rows b).degree) {b : ι}
    (fails : ¬ (rows b).Holds w) (accepting : Finset E)
    (passes : ∀ α ∈ accepting, ∀ b, (rows b).LiftCheck w (quotients b) α) :
    accepting.card ≤ 2 * largest - 1 := by
  have degree := degrees b
  refine (Finset.card_le_card fun α member => ?_).trans
    (((rows b).liftBad_card_le w (quotientDegrees b)).trans (by omega))
  by_contra outside
  exact fails ((rows b).holds_of_liftCheck injective outside (passes α member b))

/-- Akita Theorem 7.4, item 2 (mixed-dimension quotient-free ring checking). The rows have the
degrees `d_r ≤ d_max` and read one witness. If one row does not hold, the residuals `z_r(α)` are
all zero at no more than `d_max − 1` points `α`. -/
theorem card_freeCheck_le (injective : Function.Injective (algebraMap Zq E)) (rows : ι → RingRow)
    (w : Table) {largest : ℕ} (degrees : ∀ b, (rows b).degree ≤ largest) {b : ι}
    (fails : ¬ (rows b).Holds w) (accepting : Finset E)
    (passes : ∀ α ∈ accepting, ∀ b, (rows b).FreeCheck w α) :
    accepting.card ≤ largest - 1 := by
  have degree := degrees b
  refine (Finset.card_le_card fun α member => ?_).trans
    (((rows b).freeBad_card_le w).trans (by omega))
  by_contra outside
  exact fails ((rows b).holds_of_freeCheck injective outside (passes α member b))

end Mixed

section Extension

open NightstreamFPrime.Spec NightstreamFPrime.Spec.FieldExtensionRing

/-- The map from `Z_q` to `K = F_q[U]/(U² − 2)` is injective. -/
theorem extension_algebraMap_injective : Function.Injective (algebraMap Zq K) :=
  carrierEquiv.ringEquiv.symm.injective.comp QuadraticAlgebra.algebraMap_injective

end Extension

end NightstreamFPrime.Spec.Compression.Akita
