import NightstreamFPrime.Spec.Compression.Akita.Commitment

/-! The honest commitment algorithm of Akita (ePrint 2026/1983, revision of 2026-11-23,
Section 4.5), and the requirement of Akita Definition 3.5 that its output satisfies the
decommitment relation.

- `balancedDigit` and `fieldDigit` are the balanced base-`b` digits of Akita Section 3.3, with the
  lift `lift` of a residue. `decompose` is `G^{-1}_{b,n}`, and `gadget_decompose` is
  `G_{b,n} G^{-1}_{b,n} = 1`.
- `binaryDigit` is a digit in `{−1, 0}` of a compression stage, and `binaryDecompose` is
  `G^{-1}_{A_F,·}` with the digit placement of Akita Appendix L.1.
- `commit` is the commitment algorithm for a full-field source, and `commit_opens` and
  `commit_certified` state that its output satisfies Akita Definition 10.21 and the certified
  relation with the factor 1. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open Finset

section Balanced

/-- Digit `u` of the integer `y` in balanced base `b`, with digits in `{−b/2, …, b/2 − 1}`. -/
def balancedDigit (b : ℕ) : ℤ → ℕ → ℤ
  | y, 0 => (y + (b / 2 : ℕ)) % b - (b / 2 : ℕ)
  | y, u + 1 => balancedDigit b ((y + (b / 2 : ℕ)) / b) u

/-- The conditions on a base `b` with `δ` digits for the decomposition of each residue: `b` is
even, `2 ≤ b < q`, and `q ≤ b^δ`. -/
structure Base (b δ : ℕ) : Prop where
  even : b % 2 = 0
  two : 2 ≤ b
  small : b < fieldModulus
  full : fieldModulus ≤ b ^ δ

theorem balancedDigit_range {b : ℕ} (even : b % 2 = 0) (two : 2 ≤ b) (y : ℤ) (u : ℕ) :
    -((b / 2 : ℕ) : ℤ) ≤ balancedDigit b y u ∧ balancedDigit b y u ≤ ((b / 2 : ℕ) : ℤ) - 1 := by
  induction u generalizing y with
  | zero =>
      have positive : (0 : ℤ) < b := by omega
      have low := Int.emod_nonneg (y + (b / 2 : ℕ)) positive.ne'
      have high := Int.emod_lt_of_pos (y + (b / 2 : ℕ)) positive
      unfold balancedDigit
      omega
  | succ u step => exact step _

/-- An integer of the interval `[−M_k, T_k]` of Akita Section 3.3 is the value of its `k` balanced
digits. -/
theorem balancedDigit_sum {b : ℕ} (even : b % 2 = 0) (two : 2 ≤ b) (k : ℕ) :
    ∀ y : ℤ, -((b / 2 : ℕ) : ℤ) * ∑ u ∈ range k, (b : ℤ) ^ u ≤ y →
      y ≤ (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ range k, (b : ℤ) ^ u →
      ∑ u ∈ range k, (b : ℤ) ^ u * balancedDigit b y u = y := by
  have positive : (0 : ℤ) < b := by omega
  have halves : (b : ℤ) = 2 * ((b / 2 : ℕ) : ℤ) := by omega
  induction k with
  | zero =>
      intro y low high
      simp only [range_zero, sum_empty, mul_zero] at low high ⊢
      omega
  | succ k step =>
      intro y low high
      have geometric : ∑ u ∈ range (k + 1), (b : ℤ) ^ u = (b : ℤ) * ∑ u ∈ range k, (b : ℤ) ^ u + 1 := by
        rw [sum_range_succ', mul_sum, pow_zero]
        exact congrArg (· + 1) (sum_congr rfl fun u _ => pow_succ' _ _)
      rw [geometric] at low high
      have next := step ((y + (b / 2 : ℕ)) / b)
        (Int.le_ediv_of_mul_le positive (by nlinarith))
        (Int.lt_add_one_iff.mp (Int.ediv_lt_of_lt_mul positive (by nlinarith)))
      rw [sum_range_succ', pow_zero, one_mul]
      have shifted : ∑ u ∈ range k, (b : ℤ) ^ (u + 1) * balancedDigit b y (u + 1) =
          (b : ℤ) * ∑ u ∈ range k, (b : ℤ) ^ u * balancedDigit b ((y + (b / 2 : ℕ)) / b) u := by
        rw [mul_sum]
        exact sum_congr rfl fun u _ => by rw [pow_succ', mul_assoc]; rfl
      rw [shifted, next]
      have split := Int.mul_ediv_add_emod (y + (b / 2 : ℕ)) b
      change (b : ℤ) * ((y + (b / 2 : ℕ)) / b) + ((y + (b / 2 : ℕ)) % b - (b / 2 : ℕ)) = y
      omega

/-- The lift of a residue `x ∈ [0, q)` to an integer (Akita Section 3.3): `x` when
`x ≤ min(T_δ, ⌊q/2⌋)`, and `x − q` otherwise. -/
def lift (b δ : ℕ) (x : Zq) : ℤ :=
  if (x.val : ℤ) ≤ min ((((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ range δ, (b : ℤ) ^ u) (fieldModulus / 2 : ℕ)
  then x.val else x.val - fieldModulus

/-- Digit `u` of the residue `x` in balanced base `b` with `δ` digits. -/
def fieldDigit (b δ : ℕ) (x : Zq) (u : ℕ) : Zq := ((balancedDigit b (lift b δ x) u : ℤ) : Zq)

theorem fieldDigit_range {b δ : ℕ} (base : Base b δ) (x : Zq) (u : ℕ) :
    -((b / 2 : ℕ) : ℤ) ≤ (fieldDigit b δ x u).valMinAbs ∧
      (fieldDigit b δ x u).valMinAbs ≤ ((b / 2 : ℕ) : ℤ) - 1 := by
  have range := balancedDigit_range base.even base.two (lift b δ x) u
  have small := base.small
  have value : (fieldDigit b δ x u).valMinAbs = balancedDigit b (lift b δ x) u :=
    (ZMod.valMinAbs_spec _ _).mpr ⟨rfl, by constructor <;> omega⟩
  rw [value]
  exact range

/-- The lift of a residue is in the interval `[−M_δ, T_δ]` when `q ≤ b^δ`. -/
theorem lift_range {b δ : ℕ} (base : Base b δ) (x : Zq) :
    -((b / 2 : ℕ) : ℤ) * ∑ u ∈ range δ, (b : ℤ) ^ u ≤ lift b δ x ∧
      lift b δ x ≤ (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ range δ, (b : ℤ) ^ u := by
  have two := base.two
  have halves : (b : ℤ) = 2 * ((b / 2 : ℕ) : ℤ) := by have := base.even; omega
  have power : (b : ℤ) ^ δ = ((b : ℤ) - 1) * ∑ u ∈ range δ, (b : ℤ) ^ u + 1 := by
    rw [mul_comm, geom_sum_mul]
    ring
  have full : (fieldModulus : ℤ) ≤ (b : ℤ) ^ δ := by exact_mod_cast base.full
  have nonneg : (0 : ℤ) ≤ ∑ u ∈ range δ, (b : ℤ) ^ u := sum_nonneg fun u _ => by positivity
  have below : (x.val : ℤ) < fieldModulus := by exact_mod_cast ZMod.val_lt x
  have odd : (fieldModulus : ℤ) ≤ 2 * (fieldModulus / 2 : ℕ) + 1 := by omega
  obtain ⟨product, value⟩ : ∃ product : ℤ,
      product = ((b / 2 : ℕ) : ℤ) * ∑ u ∈ range δ, (b : ℤ) ^ u := ⟨_, rfl⟩
  have positive : 0 ≤ product - ∑ u ∈ range δ, (b : ℤ) ^ u := by
    rw [value]
    nlinarith
  have top : (((b / 2 : ℕ) : ℤ) - 1) * ∑ u ∈ range δ, (b : ℤ) ^ u =
      product - ∑ u ∈ range δ, (b : ℤ) ^ u := by rw [value]; ring
  have width : ((b : ℤ) - 1) * ∑ u ∈ range δ, (b : ℤ) ^ u =
      2 * product - ∑ u ∈ range δ, (b : ℤ) ^ u := by rw [value, halves]; ring
  rw [width] at power
  unfold lift
  rw [neg_mul, ← value, top]
  split
  next low => exact ⟨by linarith, (le_min_iff.mp low).1⟩
  next high =>
    rcases min_lt_iff.mp (not_le.mp high) with above | above
    · exact ⟨by linarith, by linarith⟩
    · by_cases beyond : product - ∑ u ∈ range δ, (b : ℤ) ^ u < x.val
      · exact ⟨by linarith, by linarith⟩
      · exact ⟨by linarith, by linarith⟩

theorem fieldDigit_sum {b δ : ℕ} (base : Base b δ) (x : Zq) :
    ∑ u ∈ range δ, (b : Zq) ^ u * fieldDigit b δ x u = x := by
  have recomposed := balancedDigit_sum base.even base.two δ (lift b δ x) (lift_range base x).1
    (lift_range base x).2
  have cast := congrArg (Int.cast : ℤ → Zq) recomposed
  simp only [fieldDigit]
  rw [Int.cast_sum] at cast
  simp only [Int.cast_mul, Int.cast_pow, Int.cast_natCast] at cast
  rw [cast]
  unfold lift
  split
  · rw [Int.cast_natCast, ZMod.natCast_zmod_val]
  · rw [Int.cast_sub, Int.cast_natCast, Int.cast_natCast, ZMod.natCast_zmod_val, ZMod.natCast_self,
      sub_zero]

end Balanced

section Vectors

variable {d : ℕ} [NeZero d] {ι : Type}

theorem coeff_sum {κ : Type} (s : Finset κ) (f : κ → Rq d) (i : ℕ) :
    coeff (∑ u ∈ s, f u) i = ∑ u ∈ s, coeff (f u) i := by
  simp [coeff, rep]

theorem coeff_natCast_pow_mul (b u : ℕ) (x : Rq d) (i : ℕ) :
    coeff ((b : Rq d) ^ u * x) i = (b : Zq) ^ u * coeff x i := by
  have scalar : (b : Rq d) ^ u * x = ((b : Zq) ^ u) • x := by
    rw [Algebra.smul_def, map_pow, map_natCast]
  rw [scalar, coeff_smul]

/-- `G^{-1}_{b,n}`: the balanced base-`b` digits of each coefficient (Akita Section 3.3). -/
noncomputable def decompose (b : ℕ) {δ : ℕ} (v : ι → Rq d) : ι × Fin δ → Rq d :=
  fun entry => ofCoeffs fun i => fieldDigit b δ (coeff (v entry.1) i) entry.2

/-- `G_{b,n} G^{-1}_{b,n} v = v`. -/
theorem gadget_decompose {b δ : ℕ} (base : Base b δ) (v : ι → Rq d) :
    gadget b (decompose b v : ι × Fin δ → Rq d) = v := by
  funext j
  refine ext_coeff fun i small => ?_
  rw [gadget, coeff_sum]
  simp only [coeff_natCast_pow_mul, decompose, coeff_ofCoeffs _ small]
  rw [Fin.sum_univ_eq_sum_range fun u => (b : Zq) ^ u * fieldDigit b δ (coeff (v j) i) u]
  exact fieldDigit_sum base _

theorem decompose_balanced {b δ : ℕ} (base : Base b δ) (v : ι → Rq d) :
    Balanced b (decompose b v : ι × Fin δ → Rq d) := by
  intro entry i small
  rw [decompose, coeff_ofCoeffs _ small]
  exact fieldDigit_range base _ _

theorem Balanced.vecNormInf_le [Fintype ι] {b : ℕ} {v : ι → Rq d} (balanced : Balanced b v) :
    vecNormInf v ≤ b / 2 := by
  refine (vecNormInf_le_iff _ _).mpr fun j => (normInf_le_iff _ _).mpr fun i small => ?_
  have range := balanced j i small
  unfold magnitude
  omega

theorem coeff_one (i : ℕ) : coeff (1 : Rq d) i = if i = 0 then 1 else 0 := by
  have below : (1 : Polynomial Zq).degree < d := by
    rw [Polynomial.degree_one]
    exact_mod_cast NeZero.pos d
  rw [coeff, ← map_one (AdjoinRoot.mk (modulus d)), rep_mk_of_degree_lt below,
    Polynomial.coeff_one]

theorem normOne_one : normOne (1 : Rq d) = 1 := by
  have large : 2 ≤ fieldModulus := by decide
  have unit : magnitude 1 = 1 := by
    have value : (1 : Zq).valMinAbs = 1 :=
      (ZMod.valMinAbs_spec _ _).mpr ⟨by simp, by constructor <;> omega⟩
    rw [magnitude, value]
    rfl
  rw [normOne, sum_eq_single 0]
  · simp only [coeff_one, ↓reduceIte, unit]
  · intro i _ nonzero
    simp only [coeff_one, nonzero, ↓reduceIte, magnitude_zero]
  · intro absent
    exact absurd (mem_range.mpr (NeZero.pos d)) absent

theorem Balanced.mono {b b' : ℕ} (le : b ≤ b') {v : ι → Rq d} (balanced : Balanced b v) :
    Balanced b' v := by
  intro entry i small
  have range := balanced entry i small
  omega

end Vectors

section Binary

/-- Digit `u` in `{−1, 0}` of a residue `x` (Akita Section 4.5): the negation of binary digit `u`
of the representative of `−x` in `[0, q)`. -/
def binaryDigit (x : Zq) (u : ℕ) : Zq := if (-x).val / 2 ^ u % 2 = 1 then -1 else 0

theorem binary_sum (k : ℕ) : ∀ n < 2 ^ k, ∑ u ∈ range k, 2 ^ u * (n / 2 ^ u % 2) = n := by
  induction k with
  | zero =>
      intro n small
      simp only [pow_zero, Nat.lt_one_iff] at small
      simp [small]
  | succ k step =>
      intro n small
      have half := step (n / 2) (by rw [pow_succ] at small; omega)
      have shifted : ∑ u ∈ range k, 2 ^ (u + 1) * (n / 2 ^ (u + 1) % 2) =
          2 * ∑ u ∈ range k, 2 ^ u * (n / 2 / 2 ^ u % 2) := by
        rw [mul_sum]
        exact sum_congr rfl fun u _ => by rw [pow_succ', ← Nat.div_div_eq_div_mul, mul_assoc]
      rw [sum_range_succ', shifted, half]
      simp only [pow_zero, Nat.div_one, one_mul]
      omega

theorem binaryDigit_sum (x : Zq) : ∑ u ∈ range binaryDigits, (2 : Zq) ^ u * binaryDigit x u = x := by
  have digit : ∀ u ∈ range binaryDigits, (2 : Zq) ^ u * binaryDigit x u =
      -(((2 ^ u * ((-x).val / 2 ^ u % 2) : ℕ)) : Zq) := by
    intro u _
    unfold binaryDigit
    rcases Nat.mod_two_eq_zero_or_one ((-x).val / 2 ^ u) with zero | one
    · simp [zero]
    · simp [one]
  have small : (-x).val < 2 ^ binaryDigits :=
    (ZMod.val_lt (-x)).trans_le (Nat.le_pow_clog (by decide) _)
  rw [sum_congr rfl digit, sum_neg_distrib, ← Nat.cast_sum, binary_sum _ _ small,
    ZMod.natCast_zmod_val, neg_neg]

theorem binaryDigit_range (x : Zq) (u : ℕ) :
    -1 ≤ (binaryDigit x u).valMinAbs ∧ (binaryDigit x u).valMinAbs ≤ 0 := by
  have large : 2 < fieldModulus := by decide
  unfold binaryDigit
  split
  · have value : (-1 : Zq).valMinAbs = -1 :=
      (ZMod.valMinAbs_spec _ _).mpr ⟨by simp, by constructor <;> omega⟩
    rw [value]
    exact ⟨le_rfl, by decide⟩
  · rw [ZMod.valMinAbs_zero]
    exact ⟨by decide, le_rfl⟩

theorem sequence_apply {d m : ℕ} [NeZero d] (v : Fin m → Rq d) (j : Fin m) {k : ℕ}
    (small : k < d) : sequence v (j * d + k) = coeff (v j) k := by
  have positive := NeZero.pos d
  have quotient : (j * d + k) / d = j := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ positive, Nat.div_eq_of_lt small, zero_add]
  have rest : (j * d + k) % d = k := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt small]
  simp only [sequence, quotient, rest, j.isLt, ↓reduceDIte, Fin.eta]

variable {d n : ℕ} [NeZero d]

/-- `G^{-1}_{A_F,·}` of a compression stage (Akita Appendix L.1, binary digit placement): digit
`u` of coefficient `v` of the image is at position `v + s·u` of the sequence of the coefficients
of the result, for `s = n·d`, and the other positions have 0. -/
noncomputable def binaryDecompose (width inputDegree : ℕ) [NeZero inputDegree]
    (image : Fin n → Rq d) : Fin width → Rq inputDegree :=
  fun j => ofCoeffs fun i =>
    if (j * inputDegree + i) / (n * d) < binaryDigits then
      binaryDigit (sequence image ((j * inputDegree + i) % (n * d)))
        ((j * inputDegree + i) / (n * d))
    else 0

/-- `G_{A_F,·} G^{-1}_{A_F,·} u = u` when the input of the compression map has a position for
each digit. -/
theorem binaryRecompose_decompose {width inputDegree : ℕ} [NeZero inputDegree]
    (fits : n * d * binaryDigits ≤ width * inputDegree) (image : Fin n → Rq d) :
    binaryRecompose d n (binaryDecompose width inputDegree image) = image := by
  funext entry
  refine ext_coeff fun k small => ?_
  rw [binaryRecompose, coeff_ofCoeffs _ small]
  have address : entry * d + k < n * d :=
    calc entry * d + k < (entry + 1) * d := by rw [Nat.add_mul, one_mul]; omega
      _ ≤ n * d := Nat.mul_le_mul_right _ entry.isLt
  have term : ∀ u ∈ range binaryDigits,
      (2 : Zq) ^ u * sequence (binaryDecompose width inputDegree image)
        (entry * d + k + n * d * u) = 2 ^ u * binaryDigit (coeff (image entry) k) u := by
    intro u member
    have digit := mem_range.mp member
    have quotient : (entry * d + k + n * d * u) / (n * d) = u := by
      rw [Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt address), Nat.div_eq_of_lt address, zero_add]
    have rest : (entry * d + k + n * d * u) % (n * d) = entry * d + k := by
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt address]
    have inside : (entry * d + k + n * d * u) / inputDegree < width := by
      refine Nat.div_lt_of_lt_mul (lt_of_lt_of_le ?_ (fits.trans (Nat.mul_comm _ _).le))
      calc entry * d + k + n * d * u < n * d + n * d * u := by omega
        _ = n * d * (u + 1) := by ring
        _ ≤ n * d * binaryDigits := Nat.mul_le_mul_left _ digit
    have unpacked := sequence_apply (binaryDecompose width inputDegree image)
      ⟨(entry * d + k + n * d * u) / inputDegree, inside⟩
      (Nat.mod_lt (entry * d + k + n * d * u) (NeZero.pos inputDegree))
    simp only [Nat.div_add_mod'] at unpacked
    rw [unpacked, binaryDecompose, coeff_ofCoeffs _ (Nat.mod_lt _ (NeZero.pos inputDegree))]
    simp only [Nat.div_add_mod', quotient, rest, digit, ↓reduceIte, sequence_apply image entry small]
  rw [sum_congr rfl term, binaryDigit_sum]

theorem binaryDecompose_binary (width inputDegree : ℕ) [NeZero inputDegree]
    (image : Fin n → Rq d) : NegativeBinary (binaryDecompose width inputDegree image) := by
  intro j i small
  rw [binaryDecompose, coeff_ofCoeffs _ small]
  split
  · exact binaryDigit_range _ _
  · rw [ZMod.valMinAbs_zero]
    exact ⟨by decide, le_rfl⟩

end Binary

section Path

variable {d n : ℕ} [NeZero d]

/-- The compression chain of the commitment algorithm (Akita Section 4.5): `û_1 = G^{-1} u`,
`u_1 = F_1 û_1`, `û_2 = G^{-1} u_1` and the payload `F_2 û_2`, or the payload `u` without
compression maps (Akita (46)). -/
noncomputable def pathCommit : (chain : Option ChainShape) → PathKey chain → (Fin n → Rq d) →
    PathDigits chain × Payload d n chain
  | none, _, image => (PUnit.unit, image)
  | some c, key, image =>
      let first := binaryDecompose c.firstWidth c.firstDegree image
      let second := binaryDecompose c.secondWidth c.secondDegree ((ChainKey.first key).mulVec first)
      (⟨first, second⟩, (ChainKey.second key).mulVec second)

/-- The input of each compression map has a position for each binary digit of its image. -/
def PathFits (d n : ℕ) : Option ChainShape → Prop
  | none => True
  | some c => n * d * binaryDigits ≤ c.firstWidth * c.firstDegree ∧
      c.firstRank * c.firstDegree * binaryDigits ≤ c.secondWidth * c.secondDegree

theorem pathCommit_rows {chain : Option ChainShape} (fits : PathFits d n chain)
    (key : PathKey chain) (image : Fin n → Rq d) :
    Rows chain key image (pathCommit chain key image).1 (pathCommit chain key image).2 ∧
      PathDigits.Binary chain (pathCommit chain key image).1 := by
  cases chain with
  | none => exact ⟨rfl, trivial⟩
  | some c =>
      obtain ⟨first, second⟩ : _ ∧ _ := fits
      exact ⟨⟨(binaryRecompose_decompose first image).symm,
        (binaryRecompose_decompose second _).symm, rfl⟩,
        binaryDecompose_binary _ _ _, binaryDecompose_binary _ _ _⟩

end Path

section Commit

variable {S : Shape}

/-- The conditions on a shape for the commitment algorithm with a full-field source: the bases
`b` and `b_1` decompose each residue, `b_1 ≤ b⋆`, a block is not empty, and the inputs of the
compression maps have a position for each digit. -/
structure Shape.Honest (S : Shape) : Prop where
  source : Base S.sourceBase S.sourceDigits
  opening : Base S.openingBase S.openingDigits
  certified : S.openingBase ≤ S.certifiedBase
  block : 0 < S.blockLength
  path : PathFits S.outerDegree (S.slices * S.outerRank) S.chain

/-- Block `i` of the ring-packed table under coefficient packing (Akita (41)): coefficient `k` of
entry `x` is cell `(i·M + x)·d_A + k`. -/
noncomputable def Shape.pack (S : Shape) (table : Table) (i : Fin S.blocks) :
    Fin S.blockLength → Rq S.innerDegree :=
  fun x => ofCoeffs fun k => table ((i * S.blockLength + x) * S.innerDegree + k)

/-- The table with 0 in each cell after block `B − 1`. -/
def Shape.pad (S : Shape) (table : Table) : Table :=
  fun cell => if cell / (S.blockLength * S.innerDegree) < S.blocks then table cell else 0

/-- The commitment algorithm of Akita Section 4.5 for one polynomial with a full-field source:
`s_i = G^{-1}_{b,M}(f_i)`, `t̂_i = G^{-1}_{b_1,n_A}(A s_i)`, the stacked image of `B` on the slice
inputs, and the compression chain. The result is the payload and the commitment state. -/
noncomputable def commit (key : Key S) (table : Table) : S.Payload × Decommitment S :=
  let source : Fin S.blocks → S.Source := fun i => decompose S.sourceBase (S.pack table i)
  let inner : Fin S.blocks → S.InnerDigits := fun i =>
    decompose S.openingBase (key.inner.mulVec (source i))
  let path := pathCommit S.chain key.chain (stack key.outer (S.sliceInput inner))
  (path.2, ⟨source, inner, path.1⟩)

theorem decode_decompose_pack (honest : S.Honest) (table : Table) :
    (S.decode fun i => decompose S.sourceBase (S.pack table i)) = S.pad table := by
  funext cell
  unfold Shape.decode Shape.pad
  split
  next inside =>
    have positive : 0 < S.blockLength * S.innerDegree :=
      Nat.mul_pos honest.block (NeZero.pos S.innerDegree)
    have below := Nat.mod_lt cell positive
    have entry : cell % (S.blockLength * S.innerDegree) / S.innerDegree < S.blockLength :=
      Nat.div_lt_of_lt_mul (lt_of_lt_of_eq below (Nat.mul_comm _ _))
    have unpacked := sequence_apply
      (S.pack table ⟨cell / (S.blockLength * S.innerDegree), inside⟩)
      ⟨cell % (S.blockLength * S.innerDegree) / S.innerDegree, entry⟩
      (Nat.mod_lt (cell % (S.blockLength * S.innerDegree)) (NeZero.pos S.innerDegree))
    simp only [Nat.div_add_mod'] at unpacked
    rw [gadget_decompose honest.source, unpacked, Shape.pack,
      coeff_ofCoeffs _ (Nat.mod_lt _ (NeZero.pos S.innerDegree))]
    congr 1
    have whole := Nat.div_add_mod cell (S.blockLength * S.innerDegree)
    have part := Nat.div_add_mod' (cell % (S.blockLength * S.innerDegree)) S.innerDegree
    rw [Nat.add_mul, Nat.add_assoc, part, Nat.mul_assoc,
      Nat.mul_comm (cell / (S.blockLength * S.innerDegree))]
    exact whole
  next => rfl

/-- The requirement of Akita Definition 3.5 for the relation of Definition 10.21: the output of
the commitment algorithm is an algebraic decommitment of its payload to the padded table. -/
theorem commit_opens (honest : S.Honest) (key : Key S) (table : Table) :
    (commit key table).2.Opens key (commit key table).1 (S.pad table) where
  decodes := (decode_decompose_pack honest table).symm
  innerRows _ := (gadget_decompose honest.opening _).symm
  rows := (pathCommit_rows honest.path key.chain _).1
  innerDigits _ := (decompose_balanced honest.opening _).mono honest.certified
  chainDigits := (pathCommit_rows honest.path key.chain _).2

/-- The requirement of Akita Definition 3.5 for the certified relation: the output of the
commitment algorithm with the factor `c̄ = 1` satisfies `Certified.Opens` when `1 ≤ κ̄_1` and
`b/2 ≤ β̄_∞`. -/
theorem commit_certified (honest : S.Honest) (key : Key S) (table : Table)
    {factorBound sourceBound : ℕ} (unitBound : 1 ≤ factorBound)
    (digitBound : S.sourceBase / 2 ≤ sourceBound) :
    Certified.Opens factorBound sourceBound ⟨(commit key table).2, fun _ => 1⟩ key
      (commit key table).1 (S.pad table) where
  source := commit_opens honest key table
  unit _ := isUnit_one
  factorNorm _ := normOne_one.le.trans unitBound
  sourceNorm _ := by
    rw [one_smul]
    exact (decompose_balanced honest.source _).vecNormInf_le.trans digitBound

end Commit

end NightstreamFPrime.Spec.Compression.Akita
