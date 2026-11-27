import NightstreamFPrime.Spec.Compression.Akita.ResponseTail
import NightstreamFPrime.Spec.Compression.Akita.Key20
import NightstreamFPrime.Spec.Compression.Akita.Extract.Nonce

/-! The probability that the response of an honest fold of Akita does not fit, on the coefficient
route (ePrint 2026/1983, revision of 2026-11-23, Definition 10.5, Definition 10.6 with (149),
Lemma 10.7 and Corollary K.10 with (313)), with the numbers of the schedule file
`crates/ns-compress/artifacts/key20.aks`.

- `tail_le_pow`: for `71·k ≤ 100·ℓ` and `2·V·ℓ ≤ T²`, the bound `2N·exp(−T²/(2V))` of Akita
  (303) is at most `2N/2^k`.
- `response_unfit_le`: for decoders with `SignFair` and source blocks with `‖s_i‖_∞ ≤ β`, the
  response `z = Σ_i ι(c_i)·s_i` of a uniform view does not satisfy `Enc` with probability at most
  `2·N_A/2^k`, for `N_A = M·δ_com·d_A` coefficients. This is `q^rsp` of Akita (149) for one
  response, for each source with this bound.
- `Key20.level₁_unfit` and `Key20.root_unfit` are this bound at level 1 and for the three groups
  of the root of `key20.aks`, and `Key20.level₁_search` and `Key20.root_search` are Akita
  Lemma 10.7 with (157) and the `2^12` nonces of the code: a nonce fails when a fold challenge
  does not decode or the response does not fit, and each nonce fails with probability at most
  the sum of the sampler term `ε` and the bound, to the power `2^12`. `Key20.level₁_search_total`
  and `Key20.root_search_total` are the case `ε = 0` of decoders that decode each slot.

The bound does not hold at the other levels by this proof. At level 2 of `key20.aks` the limit
of `Enc` is below the limit of Akita (313), and the levels 3 to 5 and the terminal have the
Euclidean route with a filter: there the bound is the premise (340) of Akita Appendix K.3. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass mass_mono)
open NightstreamFPrime.Spec.Compression.Akita.Extract (view responseTries mass_tries_le)

attribute [local instance low] Classical.propDecidable

/-- The bound of Akita (303) with the rule `ln 2 ≤ 71/100` of the Akita code
(`crates/akita-params/src/sis/fold_linf_cap.rs`). -/
theorem tail_le_pow {N k ℓ T V : ℕ} (enough : 71 * k ≤ 100 * ℓ) (positive : 0 < V)
    (sized : 2 * V * ℓ ≤ T ^ 2) :
    2 * (N : ℝ) * Real.exp (-((T : ℝ) ^ 2) / (2 * V)) ≤ 2 * N / 2 ^ k := by
  have real : (0 : ℝ) < 2 * V := by exact_mod_cast Nat.mul_pos (by decide) positive
  have below : (ℓ : ℝ) ≤ (T : ℝ) ^ 2 / (2 * V) := by
    rw [le_div_iff₀ real]
    have cast : ((2 * V * ℓ : ℕ) : ℝ) ≤ ((T ^ 2 : ℕ) : ℝ) := Nat.cast_le.mpr sized
    push_cast at cast
    linarith
  have small : Real.exp (-((T : ℝ) ^ 2) / (2 * V)) ≤ 1 / 2 ^ k := by
    rw [neg_div, Real.exp_neg, one_div]
    exact inv_anti₀ (by positivity)
      ((two_pow_le_exp enough).trans (Real.exp_le_exp.mpr below))
  calc 2 * (N : ℝ) * Real.exp (-((T : ℝ) ^ 2) / (2 * V)) ≤ 2 * N * (1 / 2 ^ k) :=
        mul_le_mul_of_nonneg_left small (by positivity)
    _ = 2 * N / 2 ^ k := by ring

theorem mass_or_le {Ω : Type} [Fintype Ω] (A B : Set Ω) :
    mass {ω | ω ∈ A ∨ ω ∈ B} ≤ mass A + mass B := by
  unfold mass
  rw [← Finset.expect_add_distrib]
  refine Finset.expect_le_expect fun ω _ => ?_
  by_cases a : ω ∈ A <;> by_cases b : ω ∈ B <;> simp [a, b]

theorem mass_empty_le {Ω : Type} [Fintype Ω] {A : Set Ω} (empty : ∀ ω, ω ∉ A) : mass A ≤ 0 := by
  unfold mass
  refine le_of_eq (Finset.expect_eq_zero fun ω _ => ?_)
  simp [empty ω]

section Response

variable {S : Shape} (O : OpeningShape S) {Slot Coord : Type} [Fintype Slot] [Nonempty Slot]
  [Fintype Coord] [DecidableEq Coord]

/-- Akita (149) for one response on the coefficient route, by Lemma K.3 and (313): the response
`z = Σ_i ι(c_i)·s_i` of the fold challenges of a uniform view does not satisfy `Enc` with
probability at most `2·N_A/2^k`. `fair`: each decoder has `SignFair`. `sparse`: each fold
challenge has `‖c‖_2² ≤ κ` (`normSq_of_mem` for a sparse family). `bounded`: `‖s_i‖_∞ ≤ β`.
`enough` and `sized`: `k·ln 2 ≤ ℓ ≤ T²/(2·B·κ·β²)` for the limit `T` of `Enc`. -/
theorem response_unfit_le (block : Fin S.blocks → Coord) (injective : Function.Injective block)
    (decode : Fin S.blocks → Slot → Option (Rq O.carrierDegree))
    (fair : ∀ i, SignFair (decode i)) {κ β b δ k ℓ : ℕ}
    (sparse : ∀ i slot c, decode i slot = some c → normSq c ≤ κ)
    (source : Fin S.blocks → S.Source) (bounded : ∀ i col, normInf (source i col) ≤ β)
    (two : 2 ≤ b) (enough : 71 * k ≤ 100 * ℓ) (positive : 0 < S.blocks * κ * β ^ 2)
    (sized : 2 * (S.blocks * κ * β ^ 2) * ℓ ≤ ((b / 2 - 1) * ∑ u ∈ Finset.range δ, b ^ u) ^ 2) :
    mass {look : Coord → Slot | ¬ Encodable b δ
      (response O (fun i => (decode i (look (block i))).getD 0) source)} ≤
      2 * (S.blockLength * S.sourceDigits * S.innerDegree : ℕ) / 2 ^ k := by
  have tail := decoded_tail (T := (b / 2 - 1) * ∑ u ∈ Finset.range δ, b ^ u) block injective
    decode fair sparse (embed O.stride O.stride_tower)
    (shifts_embed O.stride_tower) source bounded positive
  refine (mass_mono fun look unfit => ?_).trans (tail.trans ?_)
  · by_contra small
    refine unfit (encodable_of_magnitude two fun col j below => ?_)
    by_contra large
    exact small ⟨col, j, below, Nat.lt_of_not_le large⟩
  · have bound := tail_le_pow (N := S.blockLength * S.sourceDigits * S.innerDegree)
      enough positive sized
    have count : Fintype.card (Fin S.blockLength × Fin S.sourceDigits) * S.innerDegree =
        S.blockLength * S.sourceDigits * S.innerDegree := by
      rw [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
    have cast : (2 : ℝ) * ((Fintype.card (Fin S.blockLength × Fin S.sourceDigits) : ℝ) *
        (S.innerDegree : ℝ)) = 2 * ((S.blockLength * S.sourceDigits * S.innerDegree : ℕ) : ℝ) := by
      rw [← count]
      push_cast
      ring
    rw [cast]
    push_cast at bound ⊢
    exact bound

end Response

namespace Key20

variable {Slot Coord : Type} [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]

/-- Akita (149) at level 1 of `key20.aks`: 28 blocks, the family `(31, 10)` of `R_{q,64}` with
`‖c‖_2² = 71`, source digits of the base 8, and `Enc` of 4 digits of the base 8 with the limit
1755. The response does not fit with probability at most `2·2^17/2^67`. -/
theorem level₁_unfit (block : Fin level₁.shape.blocks → Coord)
    (injective : Function.Injective block)
    (decode : Fin level₁.shape.blocks → Slot → Option (Rq level₁.opening.carrierDegree))
    (fair : ∀ i, SignFair (decode i))
    (sparse : ∀ i slot c, decode i slot = some c → normSq c ≤ 71)
    (source : Fin level₁.shape.blocks → level₁.shape.Source)
    (bounded : ∀ i col, normInf (source i col) ≤ 4) :
    mass {look : Coord → Slot | ¬ Encodable level₁.fold.responseBase level₁.fold.responseDigits
      (response level₁.opening (fun i => (decode i (look (block i))).getD 0) source)} ≤
      1 / 2 ^ 49 := by
  refine (response_unfit_le level₁.opening block injective decode fair sparse source bounded
    (k := 67) (ℓ := 48) (by decide) (by decide) (by decide) (by decide)).trans ?_
  norm_num [level₁]

/-- Akita Lemma 10.7 with (157) at level 1 of `key20.aks` with the `2^12` nonces of the code,
for the keys `input` of the nonces after one transcript prefix. A nonce fails when a fold
challenge does not decode (the verifier needs each decode, `Extract.DecodesAt`) or the response
does not satisfy `Enc`. `sampler`: a fold challenge of one answer does not decode with probability
at most `ε`. Each nonce fails with probability at most `(ε + 2^−49)^4096`. -/
theorem level₁_search {Point : Type} [Fintype Point] [DecidableEq Point]
    (input : Fin responseTries → Point) (distinct : Function.Injective input)
    (block : Fin level₁.shape.blocks → Coord) (injective : Function.Injective block)
    (decode : Fin level₁.shape.blocks → Slot → Option (Rq level₁.opening.carrierDegree))
    (fair : ∀ i, SignFair (decode i))
    (sparse : ∀ i slot c, decode i slot = some c → normSq c ≤ 71)
    (source : Fin level₁.shape.blocks → level₁.shape.Source)
    (bounded : ∀ i col, normInf (source i col) ≤ 4) {ε : ℝ}
    (sampler : mass {look : Coord → Slot | ∃ i, decode i (look (block i)) = none} ≤ ε) :
    mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ i, decode i (view (table (input n)) (block i)) = none) ∨
        ¬ Encodable level₁.fold.responseBase level₁.fold.responseDigits
          (response level₁.opening
            (fun i => (decode i (view (table (input n)) (block i))).getD 0) source)} ≤
      (ε + 1 / 2 ^ 49) ^ responseTries :=
  mass_tries_le input distinct
    {look : Coord → Slot | look ∈ {look : Coord → Slot | ∃ i, decode i (look (block i)) = none} ∨
      look ∈ {look : Coord → Slot | ¬ Encodable level₁.fold.responseBase
        level₁.fold.responseDigits
        (response level₁.opening (fun i => (decode i (look (block i))).getD 0) source)}}
    ((mass_or_le _ _).trans
      (add_le_add sampler (level₁_unfit block injective decode fair sparse source bounded)))

/-- `level₁_search` for decoders that decode each slot (`total`): the sampler term is 0, and each
nonce fails with probability at most `(2^−49)^4096`. -/
theorem level₁_search_total {Point : Type} [Fintype Point] [DecidableEq Point]
    (input : Fin responseTries → Point) (distinct : Function.Injective input)
    (block : Fin level₁.shape.blocks → Coord) (injective : Function.Injective block)
    (decode : Fin level₁.shape.blocks → Slot → Option (Rq level₁.opening.carrierDegree))
    (total : ∀ i slot, (decode i slot).isSome)
    (fair : ∀ i, SignFair (decode i))
    (sparse : ∀ i slot c, decode i slot = some c → normSq c ≤ 71)
    (source : Fin level₁.shape.blocks → level₁.shape.Source)
    (bounded : ∀ i col, normInf (source i col) ≤ 4) :
    mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ i, decode i (view (table (input n)) (block i)) = none) ∨
        ¬ Encodable level₁.fold.responseBase level₁.fold.responseDigits
          (response level₁.opening
            (fun i => (decode i (view (table (input n)) (block i))).getD 0) source)} ≤
      (1 / 2 ^ 49) ^ responseTries := by
  have bound := level₁_search input distinct block injective decode fair sparse source bounded
    (ε := 0) (mass_empty_le fun look ⟨i, failed⟩ => by
      have decoded := total i (look (block i))
      rw [failed] at decoded
      exact Bool.false_ne_true decoded)
  rwa [zero_add] at bound

/-- Akita (149) for the root of `key20.aks`: the groups of the key polynomial, `W` and `V` with
32, 16 and 8 blocks, the family `(31, 0)` of `R_{q,128}` with `‖c‖_2² = 31` and the family
`(31, 10)` of `R_{q,64}` with `‖c‖_2² = 71`, source digits of the base `2^10`, and `Enc` of 6
digits of the base 8 with the limit 112347. The response of one group or more does not fit with
probability at most `2^−12`: the union bound over the groups (Akita Corollary K.10). -/
theorem root_unfit (block : root.Block → Coord) (injective : Function.Injective block)
    (decode : (g : Fin 3) → Fin (root.shape g).blocks → Slot → Option (Rq (root.carrierDegree g)))
    (fair : ∀ g i, SignFair (decode g i))
    (sparse : ∀ g i slot c, decode g i slot = some c → normSq c ≤ ![31, 71, 71] g)
    (source : (g : Fin 3) → Fin (root.shape g).blocks → (root.shape g).Source)
    (bounded : ∀ g i col, normInf (source g i col) ≤ 512) :
    mass {look : Coord → Slot | ∃ g, ¬ Encodable (root.fold g).responseBase
      (root.fold g).responseDigits (response (root.opening g)
        (fun i => (decode g i (look (block ⟨g, i⟩))).getD 0) (source g))} ≤ 1 / 2 ^ 12 := by
  have apart : ∀ g, Function.Injective fun i : Fin (root.shape g).blocks => block ⟨g, i⟩ :=
    fun g a b same => eq_of_heq (Sigma.mk.inj (injective same)).2
  have key := response_unfit_le (root.opening 0) (fun i => block ⟨0, i⟩) (apart 0) (decode 0)
    (fair 0) (sparse 0) (source 0) (bounded 0) (b := 8) (δ := 6) (k := 33) (ℓ := 24)
    (by decide) (by decide) (by decide) (by decide)
  have wide := response_unfit_le (root.opening 1) (fun i => block ⟨1, i⟩) (apart 1) (decode 1)
    (fair 1) (sparse 1) (source 1) (bounded 1) (b := 8) (δ := 6) (k := 29) (ℓ := 21)
    (by decide) (by decide) (by decide) (by decide)
  have inverse := response_unfit_le (root.opening 2) (fun i => block ⟨2, i⟩) (apart 2)
    (decode 2) (fair 2) (sparse 2) (source 2) (bounded 2) (b := 8) (δ := 6) (k := 59) (ℓ := 42)
    (by decide) (by decide) (by decide) (by decide)
  refine (mass_mono fun look unfit => ?_).trans ((mass_exists_le Finset.univ fun g look =>
    ¬ Encodable (root.fold g).responseBase (root.fold g).responseDigits (response (root.opening g)
      (fun i => (decode g i (look (block ⟨g, i⟩))).getD 0) (source g))).trans ?_)
  · obtain ⟨g, bad⟩ := unfit
    exact ⟨g, Finset.mem_univ g, bad⟩
  · rw [Fin.sum_univ_three]
    refine (add_le_add (add_le_add key wide) inverse).trans ?_
    norm_num [root, Shape.key20, shapeW, shapeV]

/-- Akita Lemma 10.7 with (157) for the root of `key20.aks` with the `2^12` nonces of the code.
A nonce fails when a fold challenge of a group does not decode or the response of a group does
not satisfy `Enc`. `sampler`: a fold challenge of one answer does not decode with probability at
most `ε`. Each nonce fails with probability at most `(ε + 2^−12)^4096`. -/
theorem root_search {Point : Type} [Fintype Point] [DecidableEq Point]
    (input : Fin responseTries → Point) (distinct : Function.Injective input)
    (block : root.Block → Coord) (injective : Function.Injective block)
    (decode : (g : Fin 3) → Fin (root.shape g).blocks → Slot → Option (Rq (root.carrierDegree g)))
    (fair : ∀ g i, SignFair (decode g i))
    (sparse : ∀ g i slot c, decode g i slot = some c → normSq c ≤ ![31, 71, 71] g)
    (source : (g : Fin 3) → Fin (root.shape g).blocks → (root.shape g).Source)
    (bounded : ∀ g i col, normInf (source g i col) ≤ 512) {ε : ℝ}
    (sampler : mass {look : Coord → Slot | ∃ g i, decode g i (look (block ⟨g, i⟩)) = none} ≤ ε) :
    mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ g i, decode g i (view (table (input n)) (block ⟨g, i⟩)) = none) ∨
        ∃ g, ¬ Encodable (root.fold g).responseBase (root.fold g).responseDigits
          (response (root.opening g)
            (fun i => (decode g i (view (table (input n)) (block ⟨g, i⟩))).getD 0) (source g))} ≤
      (ε + 1 / 2 ^ 12) ^ responseTries :=
  mass_tries_le input distinct
    {look : Coord → Slot |
      look ∈ {look : Coord → Slot | ∃ g i, decode g i (look (block ⟨g, i⟩)) = none} ∨
      look ∈ {look : Coord → Slot | ∃ g, ¬ Encodable (root.fold g).responseBase
        (root.fold g).responseDigits (response (root.opening g)
          (fun i => (decode g i (look (block ⟨g, i⟩))).getD 0) (source g))}}
    ((mass_or_le _ _).trans
      (add_le_add sampler (root_unfit block injective decode fair sparse source bounded)))

/-- `root_search` for decoders that decode each slot (`total`): the sampler term is 0, and each
nonce fails with probability at most `(2^−12)^4096`. -/
theorem root_search_total {Point : Type} [Fintype Point] [DecidableEq Point]
    (input : Fin responseTries → Point) (distinct : Function.Injective input)
    (block : root.Block → Coord) (injective : Function.Injective block)
    (decode : (g : Fin 3) → Fin (root.shape g).blocks → Slot → Option (Rq (root.carrierDegree g)))
    (total : ∀ g i slot, (decode g i slot).isSome)
    (fair : ∀ g i, SignFair (decode g i))
    (sparse : ∀ g i slot c, decode g i slot = some c → normSq c ≤ ![31, 71, 71] g)
    (source : (g : Fin 3) → Fin (root.shape g).blocks → (root.shape g).Source)
    (bounded : ∀ g i col, normInf (source g i col) ≤ 512) :
    mass {table : Point → Option Coord → Slot | ∀ n,
      (∃ g i, decode g i (view (table (input n)) (block ⟨g, i⟩)) = none) ∨
        ∃ g, ¬ Encodable (root.fold g).responseBase (root.fold g).responseDigits
          (response (root.opening g)
            (fun i => (decode g i (view (table (input n)) (block ⟨g, i⟩))).getD 0) (source g))} ≤
      (1 / 2 ^ 12) ^ responseTries := by
  have bound := root_search input distinct block injective decode fair sparse source bounded
    (ε := 0) (mass_empty_le fun look ⟨g, i, failed⟩ => by
      have decoded := total g i (look (block ⟨g, i⟩))
      rw [failed] at decoded
      exact Bool.false_ne_true decoded)
  rwa [zero_add] at bound

end Key20

end NightstreamFPrime.Spec.Compression.Akita
