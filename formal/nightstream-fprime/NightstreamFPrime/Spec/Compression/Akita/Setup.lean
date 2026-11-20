import NightstreamFPrime.Spec.AjtaiSetupV1.Rejection
import NightstreamFPrime.Spec.Compression.Akita.Euclid
import NightstreamFPrime.Spec.RandomOracle

/-! The shared setup of Akita and its reduction to ordinary Module-SIS (ePrint 2026/1983, revision
of 2026-10-15, Section 4.3 with (35) and (36), and Lemma 10.20 (i) with (168)).

- `Instance` is one instance `I = (d_I, n_I, m_I, p_I, η_I)` of the inventory. `view I S` is the
  prefix view `M_I(S)` of the flat setup vector `S` (35), and `Instance.Kernel` is a solution of
  Module-SIS for a matrix of the instance (Akita Definition 3.2).
- `embed I M S` puts the matrix `M` in the coordinates of the view of `I` and keeps the other
  coordinates of `S`. `view_embed`, `embed_view` and `embed_embed` state that a flat vector is its
  view of `I` and its other coordinates.
- `Finder` is the algorithm `B` of Lemma 10.20, and `embedded` is the algorithm `B_I`.
  `shared_setup_eq` is (168).
- `expand` is the flat vector of a stream of words of 64 bits with the rejection of the words
  `q` or more. `sum_expand` states that it is uniform on the streams with `N` kept words or
  more, and `expand_le` bounds the difference from a uniform flat vector by the chance of the
  other streams. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

noncomputable instance Rq.fintype (d : ℕ) [NeZero d] : Fintype (Rq d) :=
  Fintype.ofInjective (fun x (i : Fin d) => coeff x i) fun _ _ same =>
    ext_coeff fun i small => congrFun same ⟨i, small⟩

namespace Setup

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (expect_lens)

/-- One instance of the inventory `I(sch)` (Akita Section 10.3): the ring degree `d_I`, the
numbers `n_I` of rows and `m_I` of columns, the norm `p_I`, which is 2 for `euclidean` and `∞`
if not, and the bound `η_I`. For `p_I = 2` the bound is on the square of the norm. -/
structure Instance where
  degree : ℕ
  rows : ℕ
  cols : ℕ
  euclidean : Bool
  bound : ℕ
  [positive : NeZero degree]

attribute [instance] Instance.positive

namespace Instance

variable (I : Instance)

/-- The number `n_I·m_I·d_I` of the coordinates of the flat vector that the view reads. -/
def size : ℕ := I.rows * I.cols * I.degree

/-- A solution of `MSIS^{(p_I)}_{q,d_I}(n_I, m_I, η_I)` for the matrix `M` (Akita
Definition 3.2). -/
def Kernel (M : Matrix (Fin I.rows) (Fin I.cols) (Rq I.degree)) : Type :=
  cond I.euclidean (ShortKernelSq M I.bound) (ShortKernel M I.bound)

theorem cols_pos {p : ℕ} (inside : p < I.size) : 0 < I.cols := by
  rcases Nat.eq_zero_or_pos I.cols with zero | positive
  · rw [size, zero, Nat.mul_zero, Nat.zero_mul] at inside
    exact absurd inside (Nat.not_lt_zero _)
  · exact positive

theorem row_lt {p : ℕ} (inside : p < I.size) : p / I.degree / I.cols < I.rows :=
  (Nat.div_lt_iff_lt_mul (I.cols_pos inside)).mpr
    ((Nat.div_lt_iff_lt_mul (NeZero.pos _)).mpr inside)

theorem index_lt (i : Fin I.rows) (j : Fin I.cols) {l : ℕ} (small : l < I.degree) :
    (i * I.cols + j) * I.degree + l < I.size := by
  have entry : (i : ℕ) * I.cols + j + 1 ≤ I.rows * I.cols :=
    calc (i : ℕ) * I.cols + j + 1 ≤ (i + 1) * I.cols := by
          rw [Nat.add_mul, Nat.one_mul]
          exact Nat.add_le_add_left j.isLt _
      _ ≤ I.rows * I.cols := Nat.mul_le_mul_right _ i.isLt
  calc (i * I.cols + j) * I.degree + l < (i * I.cols + j + 1) * I.degree := by
        rw [Nat.add_mul _ 1, Nat.one_mul]
        exact Nat.add_lt_add_left small _
    _ ≤ I.size := Nat.mul_le_mul_right _ entry

end Instance

theorem index_div {a b d : ℕ} (small : b < d) : (a * d + b) / d = a := by
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.zero_lt_of_lt small), Nat.div_eq_of_lt small,
    Nat.zero_add]

theorem index_mod {a b d : ℕ} (small : b < d) : (a * d + b) % d = b := by
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt small]

section View

variable {N : ℕ} (I : Instance)

/-- The prefix view `M_I(S)` of the flat setup vector `S` (Akita (35)): coefficient `ℓ` of the
entry `(i, j)` is `S_{(i·m + j)·d + ℓ}`. -/
noncomputable def view (S : Fin N → Zq) : Matrix (Fin I.rows) (Fin I.cols) (Rq I.degree) :=
  Matrix.of fun i j => ofCoeffs fun l =>
    if inside : (i * I.cols + j) * I.degree + l < N then S ⟨_, inside⟩ else 0

/-- The flat vector with the matrix `M` in the coordinates of the view of `I` and the entries of
`S` in the other coordinates (the proof of Akita Lemma 10.20). -/
noncomputable def embed (M : Matrix (Fin I.rows) (Fin I.cols) (Rq I.degree)) (S : Fin N → Zq) :
    Fin N → Zq := fun p =>
  if inside : p.val < I.size then
    coeff (M ⟨p / I.degree / I.cols, I.row_lt inside⟩
      ⟨p / I.degree % I.cols, Nat.mod_lt _ (I.cols_pos inside)⟩) (p % I.degree)
  else S p

variable {I}

theorem coeff_entry (M : Matrix (Fin I.rows) (Fin I.cols) (Rq I.degree)) {a b n : ℕ}
    (row : a < I.rows) (col : b < I.cols) (i : Fin I.rows) (j : Fin I.cols) (l : ℕ)
    (sameRow : a = i) (sameCol : b = j) (sameLane : n = l) :
    coeff (M ⟨a, row⟩ ⟨b, col⟩) n = coeff (M i j) l := by
  subst sameRow sameCol sameLane
  rfl

/-- The view of `I` of `embed I M S` is `M`, when the flat vector has the coordinates of the
view. -/
theorem view_embed (fits : I.size ≤ N) (M : Matrix (Fin I.rows) (Fin I.cols) (Rq I.degree))
    (S : Fin N → Zq) : view I (embed I M S) = M := by
  ext i j
  refine ext_coeff fun l small => ?_
  have index := I.index_lt i j small
  rw [view, Matrix.of_apply, coeff_ofCoeffs _ small, dite_eq_left (index.trans_le fits)]
  unfold embed
  rw [dite_eq_left index]
  refine coeff_entry M _ _ i j l ?_ ?_ (index_mod small)
  · show (((i : ℕ) * I.cols + j) * I.degree + l) / I.degree / I.cols = i
    rw [index_div small, index_div j.isLt]
  · show (((i : ℕ) * I.cols + j) * I.degree + l) / I.degree % I.cols = j
    rw [index_div small, index_mod j.isLt]

theorem embed_view (S : Fin N → Zq) : embed I (view I S) S = S := by
  funext p
  unfold embed
  split
  · rw [view, Matrix.of_apply, coeff_ofCoeffs _ (Nat.mod_lt _ (NeZero.pos _))]
    have index : (p / I.degree / I.cols * I.cols + p / I.degree % I.cols) * I.degree +
        p % I.degree = p := by
      rw [Nat.div_add_mod', Nat.div_add_mod']
    rw [dite_eq_left (show (p / I.degree / I.cols * I.cols + p / I.degree % I.cols) * I.degree +
      p % I.degree < N by rw [index]; exact p.isLt)]
    exact congrArg S (Fin.ext index)
  · rfl

theorem embed_embed (M M' : Matrix (Fin I.rows) (Fin I.cols) (Rq I.degree)) (S : Fin N → Zq) :
    embed I M' (embed I M S) = embed I M' S := by
  funext p
  by_cases inside : p.val < I.size
  · simp only [embed, dite_eq_left inside]
  · simp only [embed, dite_eq_right inside]

/-- A uniform flat vector is a uniform matrix in the view of `I` and uniform entries in the other
coordinates (the proof of Akita Lemma 10.20: "`S` is exactly uniform because `M⋆` is"). -/
theorem expect_embed (fits : I.size ≤ N) (value : (Fin N → Zq) → ℝ) :
    𝔼 S : Fin N → Zq, value S =
      𝔼 M : Matrix (Fin I.rows) (Fin I.cols) (Rq I.degree), 𝔼 S : Fin N → Zq,
        value (embed I M S) := by
  rw [Finset.expect_comm]
  exact expect_lens (view I) (fun S M => embed I M S) (fun S M => view_embed fits M S)
    embed_view (fun S M M' => embed_embed M M' S) value

end View

section Reduction

variable {Index Aux : Type} {N : ℕ} (inventory : Index → Instance)

attribute [local instance low] Classical.propDecidable

/-- The algorithm `B` of Akita Lemma 10.20 for the inventory `inventory`. Its inputs are the flat
setup vector `S` and the auxiliary data `(z, O)`. An output is an instance of the inventory and
a solution of Module-SIS for the view of this instance. -/
abbrev Finder (N : ℕ) (Aux : Type) : Type :=
  (S : Fin N → Zq) → Aux → Option (Σ instance_, (inventory instance_).Kernel
    (view (inventory instance_) S))

variable {inventory}

/-- The algorithm `B_I` of Akita Lemma 10.20 for the challenge matrix `M`. Its random data are a
flat vector, which gives the coordinates that are not in the view of `I`, and the auxiliary data
of `B`. It runs `B` on the flat vector with `M` in the view of `I`, and it returns the solution
of `B` when `B` names `I`. -/
noncomputable def embedded (fits : ∀ instance_, (inventory instance_).size ≤ N)
    (finder : Finder inventory N Aux) (instance_ : Index)
    (M : Matrix (Fin (inventory instance_).rows) (Fin (inventory instance_).cols)
      (Rq (inventory instance_).degree))
    (coins : (Fin N → Zq) × Aux) : Option ((inventory instance_).Kernel M) :=
  (finder (embed (inventory instance_) M coins.1) coins.2).bind fun out =>
    if named : out.1 = instance_ then
      some (cast (by rw [named, view_embed (fits instance_)]) out.2)
    else none

theorem embedded_isSome (fits : ∀ instance_, (inventory instance_).size ≤ N)
    (finder : Finder inventory N Aux) (instance_ : Index)
    (M : Matrix (Fin (inventory instance_).rows) (Fin (inventory instance_).cols)
      (Rq (inventory instance_).degree))
    (coins : (Fin N → Zq) × Aux) :
    (embedded fits finder instance_ M coins).isSome ↔
      ∃ out, finder (embed (inventory instance_) M coins.1) coins.2 = some out ∧
        out.1 = instance_ := by
  unfold embedded
  rcases finder (embed (inventory instance_) M coins.1) coins.2 with _ | out
  · simp
  · by_cases named : out.1 = instance_ <;> simp [named]

variable [Fintype Index] [Fintype Aux]

/-- Akita Lemma 10.20 (i), (168), with equality. `Adv_sp(B)` is the probability that `B` returns
an instance of the inventory and a solution of Module-SIS for its view, for a uniform flat setup
vector `S` and uniform auxiliary data that are independent of `S`. It is the sum over the
instances `I` of `Adv_I(B_I)`: the probability that `B_I` returns a solution for a uniform matrix
`M⋆` of the instance. `fits`: the flat vector has the coordinates of each view. -/
theorem shared_setup_eq (fits : ∀ instance_, (inventory instance_).size ≤ N)
    (finder : Finder inventory N Aux) :
    𝔼 S : Fin N → Zq, 𝔼 aux : Aux, (if (finder S aux).isSome then (1 : ℝ) else 0) =
      ∑ instance_, 𝔼 M : Matrix (Fin (inventory instance_).rows)
          (Fin (inventory instance_).cols) (Rq (inventory instance_).degree),
        𝔼 coins : (Fin N → Zq) × Aux,
          if (embedded fits finder instance_ M coins).isSome then (1 : ℝ) else 0 := by
  have named : ∀ instance_, (𝔼 M : Matrix (Fin (inventory instance_).rows)
        (Fin (inventory instance_).cols) (Rq (inventory instance_).degree),
      𝔼 coins : (Fin N → Zq) × Aux,
        if (embedded fits finder instance_ M coins).isSome then (1 : ℝ) else 0) =
      𝔼 S : Fin N → Zq, 𝔼 aux : Aux,
        if ∃ out, finder S aux = some out ∧ out.1 = instance_ then (1 : ℝ) else 0 := by
    intro instance_
    rw [expect_embed (fits instance_) fun S => 𝔼 aux : Aux,
      if ∃ out, finder S aux = some out ∧ out.1 = instance_ then (1 : ℝ) else 0]
    refine Finset.expect_congr rfl fun M _ => ?_
    rw [← Finset.univ_product_univ, Finset.expect_product]
    refine Finset.expect_congr rfl fun S _ => Finset.expect_congr rfl fun aux _ => ?_
    simp only [embedded_isSome]
  rw [Finset.sum_congr rfl fun instance_ _ => named instance_, ← Finset.expect_sum_comm]
  refine Finset.expect_congr rfl fun S _ => ?_
  rw [← Finset.expect_sum_comm]
  refine Finset.expect_congr rfl fun aux _ => ?_
  rcases finder S aux with _ | out
  · simp
  · simp

end Reduction

section Expansion

open NightstreamFPrime.Spec.AjtaiSetupV1.Rejection

variable {count N : ℕ}

/-- The flat setup vector of the code for a stream of `count` words of 64 bits (Akita (36) with
exact rejection): entry `p` is the kept word `p`, and a word is kept when it is below `q`. An
entry is 0 when the stream has `p` kept words or fewer. -/
def expand (stream : Fin count → Word) : Fin N → Zq := fun p =>
  ZMod.finEquiv fieldModulus (streamElement stream p)

/-- The flat setup vector of the code is uniform on the streams with `N` kept words or more: each
flat vector is the vector of `startCount` of these streams. -/
theorem sum_expand (value : (Fin N → Zq) → ℝ) :
    ∑ stream : Fin count → Word, (if StreamFallback N stream then 0 else value (expand stream)) =
      startCount wordCount count N * ∑ S, value S := by
  let lift : (Fin N → F) ≃ (Fin N → Zq) :=
    Equiv.piCongrRight fun _ => (ZMod.finEquiv fieldModulus).toEquiv
  have fibers : ∀ stream : Fin count → Word,
      (if StreamFallback N stream then 0 else value (expand stream)) =
        ∑ entries : Fin N → F,
          if ¬ StreamFallback N stream ∧ streamElement stream = entries then value (lift entries)
          else 0 := by
    intro stream
    by_cases fallback : StreamFallback N stream
    · simp [fallback]
    · simp only [fallback, ite_false, not_false_eq_true, true_and, Finset.sum_ite_eq,
        Finset.mem_univ, ite_true]
      rfl
  rw [Finset.sum_congr rfl fun stream _ => fibers stream, Finset.sum_comm,
    ← Fintype.sum_equiv lift (fun entries => value (lift entries)) value fun _ => rfl,
    Finset.mul_sum]
  refine Finset.sum_congr rfl fun entries _ => ?_
  rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul,
    card_streamElement fieldModulus_le entries]

/-- The setup hybrid for the code (the term `Δ_setup` of Akita Theorem 10.46): the probability of
an event of the flat setup vector of a uniform stream of `count` words is at most its probability
for a uniform flat vector plus the chance that the stream has fewer than `N` kept words. This
chance is at most `C(count, r)·(59/2^64)^r` for `r = count − N + 1`. The code reads words until
it has `N` kept words, so it has no such stream. -/
theorem expand_le (enough : N ≤ count) (event : (Fin N → Zq) → Prop) [DecidablePred event] :
    𝔼 stream : Fin count → Word, (if event (expand stream) then (1 : ℝ) else 0) ≤
      (𝔼 S : Fin N → Zq, if event S then (1 : ℝ) else 0) +
        (count.choose (count - N + 1) : ℝ) * ((wordCount - fieldModulus : ℕ) : ℝ) ^
          (count - N + 1) / (wordCount : ℝ) ^ (count - N + 1) := by
  have streams : (0 : ℝ) < Fintype.card (Fin count → Word) := by
    exact_mod_cast Fintype.card_pos
  have split : ∀ stream : Fin count → Word, (if event (expand stream) then (1 : ℝ) else 0) ≤
      (if StreamFallback N stream then 0 else if event (expand stream) then (1 : ℝ) else 0) +
        if StreamFallback N stream then 1 else 0 := by
    intro stream
    by_cases fallback : StreamFallback N stream
    · rw [ite_eq_left fallback, ite_eq_left fallback]
      split <;> norm_num
    · rw [ite_eq_right fallback, ite_eq_right fallback, add_zero]
  have kept := sum_expand (count := count) fun S : Fin N → Zq => if event S then (1 : ℝ) else 0
  have total := sum_expand (count := count) fun _ : Fin N → Zq => (1 : ℝ)
  have fewer : (startCount wordCount count N : ℝ) * Fintype.card (Fin N → Zq) ≤
      Fintype.card (Fin count → Word) := by
    have ones : ∑ _S : Fin N → Zq, (1 : ℝ) = Fintype.card (Fin N → Zq) := by simp
    have all : ∑ _stream : Fin count → Word, (1 : ℝ) = Fintype.card (Fin count → Word) := by
      simp
    rw [← ones, ← total, ← all]
    exact Finset.sum_le_sum fun stream _ => by split <;> norm_num
  have chance := streamFallback_chance_le (bound := wordCount) (count := count) (degree := N)
    fieldModulus_le (by decide) enough
  have chanceReal : ((Finset.univ.filter (StreamFallback N (count := count)
      (bound := wordCount))).card : ℝ) / Fintype.card (Fin count → Word) ≤
      (count.choose (count - N + 1) : ℝ) * ((wordCount - fieldModulus : ℕ) : ℝ) ^
        (count - N + 1) / (wordCount : ℝ) ^ (count - N + 1) := by
    have cast := (Rat.cast_le (K := ℝ)).mpr chance
    simp only [Rat.cast_div, Rat.cast_mul, Rat.cast_pow, Rat.cast_natCast] at cast
    exact cast
  refine (Finset.expect_le_expect fun stream _ => split stream).trans ?_
  rw [Finset.expect_add_distrib]
  refine add_le_add ?_ ?_
  · rw [Finset.expect_eq_sum_div_card, kept, Finset.expect_eq_sum_div_card, Finset.card_univ,
      Finset.card_univ, div_le_div_iff₀ streams (by exact_mod_cast Fintype.card_pos)]
    have nonnegative : (0 : ℝ) ≤ ∑ S : Fin N → Zq, if event S then (1 : ℝ) else 0 :=
      Finset.sum_nonneg fun S _ => by split <;> norm_num
    nlinarith [mul_le_mul_of_nonneg_left fewer nonnegative]
  · rw [Finset.expect_eq_sum_div_card, Finset.card_univ, ← Finset.sum_filter, Finset.sum_const,
      nsmul_eq_mul, mul_one]
    exact chanceReal

end Expansion

end Setup

end NightstreamFPrime.Spec.Compression.Akita
