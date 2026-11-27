import NightstreamFPrime.Spec.Compression.Akita.Key20Challenge

/-! The union bound and the symmetry step of Akita Theorem H.2 (ePrint 2026/1983, revision of
2026-11-23, Appendix H.2) for the integer predicate of the filter.

- `TrigTable.Passes T c k` is the test `U_k ≤ T²·Δ²` of Akita (288) at the frequency `k`, so
  `TrigTable.Accepts T c` is `Passes T c k` for each `k < d/2`.
- `TrigTable.Antisymmetric d g` states `|C[t + d] + C[t]| ≤ g` and `|S[t + d] + S[t]| ≤ g` for
  `t < d`, and `TrigTable.widen g` is the table with the error bound `ε + g`.
- `TrigTable.code_antisymmetric`: the tables of the code have `g = 1` for each even degree.
  `downscale_round` rounds `(lo + hi)/2` down after the sign, so `g = 0` is false
  (`TrigTable.code_cos_one`).
- `twist d m inv` is `c(X) ↦ c(X^m)` for an inverse `inv` of `m` modulo `d`. It moves the
  coefficient `i` to `m·i mod d` with the sign `(−1)^⌊m·i/d⌋`, it is injective, and it keeps
  `‖c‖_1` and each family `SparseFamily d n₁ n₂`.
- `TrigTable.upper_twist_le`: for `m = 2k + 1`, `U_k` of `c` is at most `U_0` of `twist c` for
  the widened table.
- `TrigTable.card_reject_le` is the union bound over the `d/2` frequencies for `d = 2^{n+1}`:
  `#{c ∈ C | ¬ Accepts T c} ≤ (d/2)·#{c ∈ C | ¬ (widen g).Passes T c 0}`.
- `Key20.Tail` bounds the count at the frequency 0 for the two filtered families of the code,
  and `Key20.kept_of_tail` gives `Key20.Kept`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

private theorem finite_rq (d : ℕ) [NeZero d] : Finite (Rq d) :=
  Finite.of_injective (fun x (i : Fin d) => coeff x i) fun _ _ same =>
    ext_coeff fun i small => congrFun same ⟨i, small⟩

attribute [local instance] finite_rq

/-- The sign `(−1)^⌊m·i/d⌋` of the coefficient `i` under `c(X) ↦ c(X^m)` in `R_{q,d}`. -/
def twistSign (d m i : ℕ) : ℤ := if m * i / d % 2 = 1 then -1 else 1

/-- `c(X) ↦ c(X^m)` in `R_{q,d}` (Akita Appendix H.2) for an inverse `inv` of `m` modulo `d`:
the coefficient `i` moves to `m·i mod d` with the sign `twistSign d m i`. -/
noncomputable def twist (d m inv : ℕ) [NeZero d] (c : Rq d) : Rq d :=
  ofCoeffs fun j => (twistSign d m (inv * j % d) : Zq) * coeff c (inv * j % d)

theorem valMinAbs_twistSign (d m i : ℕ) (a : Zq) :
    ((twistSign d m i : Zq) * a).valMinAbs = twistSign d m i * a.valMinAbs := by
  unfold twistSign
  split_ifs
  · have odd : fieldModulus % 2 = 1 := by decide
    rw [Int.cast_neg, Int.cast_one, neg_one_mul, neg_one_mul,
      ZMod.valMinAbs_neg_of_ne_half (a := a) (by omega)]
  · rw [Int.cast_one, one_mul, one_mul]

section Twist

variable {d m inv : ℕ} [NeZero d]

omit [NeZero d] in
private theorem index_cancel {a b : ℕ} (one : a * b % d = 1) {i : ℕ} (small : i < d) :
    b * (a * i % d) % d = i := by
  rw [Nat.mul_mod b (a * i % d) d, Nat.mod_mod (a * i) d, ← Nat.mul_mod b (a * i) d,
    ← Nat.mul_assoc b a i, Nat.mul_comm b a, Nat.mul_mod (a * b) i d, one, Nat.one_mul,
    Nat.mod_mod, Nat.mod_eq_of_lt small]

private theorem sum_twist_index {M : Type} [AddCommMonoid M] (one : m * inv % d = 1)
    (f : ℕ → M) : ∑ j ∈ Finset.range d, f j = ∑ i ∈ Finset.range d, f (m * i % d) := by
  have back : inv * m % d = 1 := by rwa [Nat.mul_comm]
  exact Finset.sum_nbij' (fun j => inv * j % d) (fun i => m * i % d)
    (fun _ _ => Finset.mem_range.mpr (Nat.mod_lt _ (NeZero.pos d)))
    (fun _ _ => Finset.mem_range.mpr (Nat.mod_lt _ (NeZero.pos d)))
    (fun j member => index_cancel back (Finset.mem_range.mp member))
    (fun i member => index_cancel one (Finset.mem_range.mp member))
    (fun j member => (congrArg f (index_cancel back (Finset.mem_range.mp member))).symm)

theorem coeff_twist (one : m * inv % d = 1) (c : Rq d) {i : ℕ} (small : i < d) :
    coeff (twist d m inv c) (m * i % d) = (twistSign d m i : Zq) * coeff c i := by
  rw [twist, coeff_ofCoeffs _ (Nat.mod_lt _ (NeZero.pos d)), index_cancel one small]

theorem magnitude_coeff_twist (one : m * inv % d = 1) (c : Rq d) {i : ℕ} (small : i < d) :
    magnitude (coeff (twist d m inv c) (m * i % d)) = magnitude (coeff c i) := by
  rw [coeff_twist one c small]
  unfold twistSign
  split_ifs
  · rw [Int.cast_neg, Int.cast_one, neg_one_mul, magnitude_neg]
  · rw [Int.cast_one, one_mul]

/-- `c ↦ c(X^m)` is injective. -/
theorem twist_injective (one : m * inv % d = 1) : Function.Injective (twist d m inv) := by
  intro c c' same
  refine ext_coeff fun i small => ?_
  have sign : (twistSign d m i : Zq) * twistSign d m i = 1 := by
    unfold twistSign
    split_ifs <;> simp
  have moved := congrArg (fun x => (twistSign d m i : Zq) * coeff x (m * i % d)) same
  simp only [coeff_twist one c small, coeff_twist one c' small, ← mul_assoc, sign,
    one_mul] at moved
  exact moved

/-- `c ↦ c(X^m)` keeps `‖c‖_1`. -/
theorem normOne_twist (one : m * inv % d = 1) (c : Rq d) :
    normOne (twist d m inv c) = normOne c := by
  unfold normOne
  refine (sum_twist_index one _).trans (Finset.sum_congr rfl fun i member => ?_)
  exact magnitude_coeff_twist one c (Finset.mem_range.mp member)

/-- `c ↦ c(X^m)` keeps each family `SparseFamily d n₁ n₂`: it is a signed permutation of the
coefficient positions (Akita Appendix H.2). -/
theorem twist_mem (one : m * inv % d = 1) {ones twos : ℕ} {c : Rq d}
    (inside : c ∈ SparseFamily d ones twos) : twist d m inv c ∈ SparseFamily d ones twos := by
  obtain ⟨short, first, second⟩ := inside
  have count (value : ℕ) :
      ((Finset.range d).filter fun j => magnitude (coeff (twist d m inv c) j) = value).card =
        ((Finset.range d).filter fun i => magnitude (coeff c i) = value).card := by
    rw [Finset.card_filter, Finset.card_filter]
    refine (sum_twist_index one _).trans (Finset.sum_congr rfl fun i member => ?_)
    simp only [magnitude_coeff_twist one c (Finset.mem_range.mp member)]
  refine ⟨(normInf_le_iff _ _).mpr fun j small => ?_, (count 1).trans first,
    (count 2).trans second⟩
  have back : inv * m % d = 1 := by rwa [Nat.mul_comm]
  rw [← index_cancel back small, magnitude_coeff_twist one c (Nat.mod_lt _ (NeZero.pos d))]
  exact (normInf_le_iff c 2).mp short _ (Nat.mod_lt _ (NeZero.pos d))

omit [NeZero d] in
private theorem exists_inverse {n : ℕ} (power : 2 ^ (n + 1) = d) (k : ℕ) :
    ∃ inv, (2 * k + 1) * inv % d = 1 := by
  have coprime : Nat.Coprime (2 * k + 1) d := by
    rw [← power]
    exact Nat.Coprime.pow_right (n + 1) (Nat.coprime_two_right.mpr (odd_two_mul_add_one k))
  have large : 1 < d := by
    rw [← power]
    exact Nat.one_lt_two_pow' n
  obtain ⟨inv, -, one⟩ := Nat.exists_mul_mod_eq_one_of_coprime coprime large
  exact ⟨inv, one⟩

/-- For `m = 2k + 1` and a table `f` with `|f[t + d] + f[t]| ≤ g` for `t < d`, the accumulator
`Σ_j c_j·f[m·j mod 2d]` of `c` differs from the accumulator at the frequency 0 of `c(X^m)` by at
most `g·‖c‖_1`. -/
private theorem sum_twist_le {f : ℕ → ℤ} {slack : ℕ} (pair : ∀ t < d, |f (t + d) + f t| ≤ slack)
    {k : ℕ} (one : (2 * k + 1) * inv % d = 1) (c : Rq d) :
    |∑ j ∈ Finset.range d, (coeff c j).valMinAbs * f ((2 * k + 1) * j % (2 * d)) -
      ∑ j ∈ Finset.range d, (coeff (twist d (2 * k + 1) inv c) j).valMinAbs *
        f ((2 * 0 + 1) * j % (2 * d))| ≤ slack * normOne c := by
  have moved : ∑ j ∈ Finset.range d, (coeff (twist d (2 * k + 1) inv c) j).valMinAbs *
      f ((2 * 0 + 1) * j % (2 * d)) = ∑ i ∈ Finset.range d,
        twistSign d (2 * k + 1) i * (coeff c i).valMinAbs * f ((2 * k + 1) * i % d) := by
    refine (sum_twist_index one _).trans (Finset.sum_congr rfl fun i member => ?_)
    have below := Nat.mod_lt ((2 * k + 1) * i) (NeZero.pos d)
    have index : (2 * 0 + 1) * ((2 * k + 1) * i % d) % (2 * d) = (2 * k + 1) * i % d := by
      rw [show 2 * 0 + 1 = 1 from rfl, Nat.one_mul]
      exact Nat.mod_eq_of_lt (by omega)
    rw [coeff_twist one c (Finset.mem_range.mp member), valMinAbs_twistSign, index]
  have mass : ((normOne c : ℕ) : ℤ) = ∑ i ∈ Finset.range d, |(coeff c i).valMinAbs| := by
    rw [normOne, Nat.cast_sum]
    exact Finset.sum_congr rfl fun i _ => Int.natCast_natAbs _
  rw [moved, ← Finset.sum_sub_distrib, mass, Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
  have split : (2 * k + 1) * i % (2 * d) =
      (2 * k + 1) * i % d + d * ((2 * k + 1) * i / d % 2) := by
    rw [Nat.mul_comm 2 d]
    exact Nat.mod_mul
  have small : (2 * k + 1) * i % d < d := Nat.mod_lt _ (NeZero.pos d)
  rw [split]
  unfold twistSign
  rcases Nat.mod_two_eq_zero_or_one ((2 * k + 1) * i / d) with even | odd
  · rw [even, Nat.mul_zero, Nat.add_zero, ite_eq_right Nat.zero_ne_one, one_mul, sub_self, abs_zero]
    positivity
  · rw [odd, Nat.mul_one, ite_eq_left rfl]
    calc |(coeff c i).valMinAbs * f ((2 * k + 1) * i % d + d) -
          -1 * (coeff c i).valMinAbs * f ((2 * k + 1) * i % d)|
        = |(coeff c i).valMinAbs| * |f ((2 * k + 1) * i % d + d) + f ((2 * k + 1) * i % d)| := by
          rw [← abs_mul]
          congr 1
          ring
      _ ≤ |(coeff c i).valMinAbs| * slack := mul_le_mul_of_nonneg_left (pair _ small) (abs_nonneg _)
      _ = slack * |(coeff c i).valMinAbs| := mul_comm _ _

end Twist

private theorem ncard_exists_le {α : Type} [Finite α] (bad : ℕ → Set α) {bound : ℕ}
    (each : ∀ k, (bad k).ncard ≤ bound) : ∀ n, {x | ∃ k < n, x ∈ bad k}.ncard ≤ n * bound
  | 0 => by simp
  | n + 1 => by
    have cover : {x | ∃ k < n + 1, x ∈ bad k} ⊆ {x | ∃ k < n, x ∈ bad k} ∪ bad n := by
      rintro x ⟨k, small, inside⟩
      obtain low | rfl : k < n ∨ k = n := by omega
      · exact Or.inl ⟨k, low, inside⟩
      · exact Or.inr inside
    calc {x | ∃ k < n + 1, x ∈ bad k}.ncard
        ≤ ({x | ∃ k < n, x ∈ bad k} ∪ bad n).ncard := Set.ncard_le_ncard cover
      _ ≤ {x | ∃ k < n, x ∈ bad k}.ncard + (bad n).ncard := Set.ncard_union_le _ _
      _ ≤ n * bound + bound := Nat.add_le_add (ncard_exists_le bad each n) (each n)
      _ = (n + 1) * bound := (Nat.succ_mul n bound).symm

namespace TrigTable

/-- `|C[t + d] + C[t]| ≤ slack` and `|S[t + d] + S[t]| ≤ slack` for `t < d`: the integer form of
`cos(π(t + d)/d) = −cos(πt/d)` and `sin(π(t + d)/d) = −sin(πt/d)`. -/
def Antisymmetric (table : TrigTable) (d slack : ℕ) : Prop :=
  ∀ t < d, |table.cos (t + d) + table.cos t| ≤ slack ∧ |table.sin (t + d) + table.sin t| ≤ slack

/-- The table with the error bound `ε + slack`. -/
def widen (table : TrigTable) (slack : ℕ) : TrigTable :=
  { table with error := table.error + slack }

variable {d : ℕ} [NeZero d]

/-- The test `U_k ≤ T²·Δ²` of Akita (288) at the frequency `k`. -/
def Passes (table : TrigTable) (threshold : ℕ) (c : Rq d) (k : ℕ) : Prop :=
  table.upper c k ≤ (threshold : ℤ) ^ 2 * (table.scale : ℤ) ^ 2

private theorem upper_eq (table : TrigTable) (c : Rq d) (k : ℕ) :
    table.upper c k = (|table.real c k| + normOne c * table.error) ^ 2 +
      (|table.imaginary c k| + normOne c * table.error) ^ 2 := by
  rw [upper, add_sq, add_sq, sq_abs, sq_abs]
  push_cast
  ring

private theorem real_widen (table : TrigTable) (slack : ℕ) (c : Rq d) (k : ℕ) :
    (table.widen slack).real c k = table.real c k := rfl

private theorem imaginary_widen (table : TrigTable) (slack : ℕ) (c : Rq d) (k : ℕ) :
    (table.widen slack).imaginary c k = table.imaginary c k := rfl

private theorem error_widen (table : TrigTable) (slack : ℕ) :
    (table.widen slack).error = table.error + slack := rfl

/-- Accumulators within `g·‖c‖_1` of those of `c'` give `U_k` of `c` at most `U_{k'}` of `c'`
for the table with the error bound `ε + g`. -/
theorem upper_le_widen {table : TrigTable} {slack k k' : ℕ} {c c' : Rq d}
    (mass : normOne c' = normOne c)
    (real : |table.real c k - table.real c' k'| ≤ slack * normOne c)
    (imaginary : |table.imaginary c k - table.imaginary c' k'| ≤ slack * normOne c) :
    table.upper c k ≤ (table.widen slack).upper c' k' := by
  have growth (x x' : ℤ) (close : |x - x'| ≤ slack * normOne c) :
      (|x| + normOne c * table.error) ^ 2 ≤
        (|x'| + normOne c' * ((table.error + slack : ℕ) : ℤ)) ^ 2 := by
    have triangle := abs_add_le (x - x') x'
    rw [sub_add_cancel] at triangle
    rw [mass]
    push_cast
    exact pow_le_pow_left₀ (by positivity) (by linarith) 2
  rw [upper_eq, upper_eq, real_widen, imaginary_widen, error_widen]
  exact add_le_add (growth _ _ real) (growth _ _ imaginary)

/-- For `m = 2k + 1`, `U_k` of `c` is at most `U_0` of `c(X^m)` for the widened table. -/
theorem upper_twist_le {table : TrigTable} {slack k inv : ℕ}
    (antisymmetric : table.Antisymmetric d slack) (one : (2 * k + 1) * inv % d = 1) (c : Rq d) :
    table.upper c k ≤ (table.widen slack).upper (twist d (2 * k + 1) inv c) 0 :=
  upper_le_widen (normOne_twist one c)
    (sum_twist_le (fun t small => (antisymmetric t small).1) one c)
    (sum_twist_le (fun t small => (antisymmetric t small).2) one c)

/-- The equal-marginal step of Akita Theorem H.2 for the integer predicate: `c ↦ c(X^{2k+1})`
maps the members of `SparseFamily d n₁ n₂` that fail the test at the frequency `k` injectively to
members that fail the test of the widened table at the frequency 0. -/
theorem card_passes_le {table : TrigTable} {slack k inv : ℕ}
    (antisymmetric : table.Antisymmetric d slack) (one : (2 * k + 1) * inv % d = 1)
    (threshold ones twos : ℕ) :
    Nat.card {c ∈ SparseFamily d ones twos | ¬ table.Passes threshold c k} ≤
      Nat.card {c ∈ SparseFamily d ones twos | ¬ (table.widen slack).Passes threshold c 0} := by
  refine Nat.card_le_card_of_injective
    (fun c => ⟨twist d (2 * k + 1) inv c.1, twist_mem one c.2.1, fun passes => c.2.2 ?_⟩)
    fun a b same => Subtype.ext (twist_injective one (congrArg Subtype.val same))
  exact (upper_twist_le antisymmetric one c.1).trans passes

/-- The union bound of Akita Theorem H.2 over the `d/2` frequencies for `d = 2^{n+1}`, with the
equal-marginal step `card_passes_le` for the integer predicate. -/
theorem card_reject_le {table : TrigTable} {n slack : ℕ} (power : 2 ^ (n + 1) = d)
    (antisymmetric : table.Antisymmetric d slack) (threshold ones twos : ℕ) :
    Nat.card {c ∈ SparseFamily d ones twos | ¬ table.Accepts threshold c} ≤
      d / 2 * Nat.card {c ∈ SparseFamily d ones twos |
        ¬ (table.widen slack).Passes threshold c 0} := by
  have each (k : ℕ) : {c ∈ SparseFamily d ones twos | ¬ table.Passes threshold c k}.ncard ≤
      {c ∈ SparseFamily d ones twos | ¬ (table.widen slack).Passes threshold c 0}.ncard := by
    obtain ⟨inv, one⟩ := exists_inverse power k
    simpa only [Nat.card_coe_set_eq] using card_passes_le antisymmetric one threshold ones twos
  have cover : {c ∈ SparseFamily d ones twos | ¬ table.Accepts threshold c} ⊆
      {c | ∃ k < d / 2, c ∈ {c ∈ SparseFamily d ones twos | ¬ table.Passes threshold c k}} := by
    rintro c ⟨inside, rejected⟩
    by_contra none
    exact rejected fun k small =>
      Classical.byContradiction fun failed => none ⟨k, small, inside, failed⟩
  simp only [Nat.card_coe_set_eq]
  exact (Set.ncard_le_ncard cover).trans (ncard_exists_le _ each (d / 2))

/-- A count of the rejected members of a family bounds the count of the kept members from
below. -/
theorem le_card_of_reject {family : Set (Rq d)} {keep : Rq d → Prop} {bound : ℕ}
    (room : Nat.card {c ∈ family | ¬ keep c} + bound ≤ Nat.card family) :
    bound ≤ Nat.card {c ∈ family | keep c} := by
  have split := Set.ncard_inter_add_ncard_sdiff_eq_ncard family {c | keep c}
  rw [show family ∩ {c | keep c} = {c ∈ family | keep c} from rfl,
    show family \ {c | keep c} = {c ∈ family | ¬ keep c} from rfl] at split
  simp only [Nat.card_coe_set_eq] at room ⊢
  omega

end TrigTable

namespace TrigBuild

private theorem downscaleRound_neg (v : ℤ × ℤ) (shift : ℕ) :
    |(downscaleRound (applySign v (-1)) shift).1 + (downscaleRound v shift).1| ≤ 1 := by
  simp only [downscaleRound, applySign, fdiv, fshr, cshr, show (-1 : ℤ) < 0 by norm_num,
    ↓reduceIte, neg_neg]
  generalize v.1 / 2 ^ shift = a
  generalize (-v.2) / 2 ^ shift = b
  rw [abs_le]
  constructor <;> omega

private theorem downscaleRound_pair (v : ℤ × ℤ) (shift : ℕ) {sign sign' : ℤ}
    (unit : sign = 1 ∨ sign = -1) (opposite : sign' = -sign) :
    |(downscaleRound (applySign v sign') shift).1 + (downscaleRound (applySign v sign) shift).1| ≤
      1 := by
  have plus : applySign v 1 = v := by simp [applySign]
  subst opposite
  rcases unit with rfl | rfl
  · rw [plus]
    exact downscaleRound_neg v shift
  · rw [neg_neg, plus, add_comm]
    exact downscaleRound_neg v shift

private theorem downscaleRound_zero (sign : ℤ) (shift : ℕ) :
    (downscaleRound (applySign (0, 0) sign) shift).1 = 0 := by
  simp [downscaleRound, applySign, fdiv, fshr, cshr]

private theorem taylorSin_zero : taylorSin (0, 0) = (0, 0) := by
  decide

private theorem firstOctant_zero (d : ℕ) : (firstOctant d 0).2 = (0, 0) := by
  simp only [firstOctant, Nat.cast_zero, mul_zero, fdiv, cdiv, neg_zero, Int.zero_ediv]
  exact taylorSin_zero

private theorem octantReduce_add {d t : ℕ} (even : 2 ∣ d) (small : t < d) :
    (octantReduce (t + d) d).index = (octantReduce t d).index ∧
      (octantReduce (t + d) d).swap = (octantReduce t d).swap ∧
      ((octantReduce (t + d) d).cosSign = -(octantReduce t d).cosSign ∨
        (octantReduce t d).index = 0 ∧ (octantReduce t d).swap = true) ∧
      ((octantReduce (t + d) d).sinSign = -(octantReduce t d).sinSign ∨
        (octantReduce t d).index = 0 ∧ (octantReduce t d).swap = false) ∧
      ((octantReduce t d).cosSign = 1 ∨ (octantReduce t d).cosSign = -1) ∧
      ((octantReduce t d).sinSign = 1 ∨ (octantReduce t d).sinSign = -1) := by
  obtain ⟨m, rfl⟩ := even
  have half : 2 * m / 2 = m := by omega
  have quarter : 2 * m / 4 = m / 2 := by omega
  have low : t % (2 * (2 * m)) = t := Nat.mod_eq_of_lt (by omega)
  have high : (t + 2 * m) % (2 * (2 * m)) = t + 2 * m := Nat.mod_eq_of_lt (by omega)
  simp only [octantReduce, half, quarter, low, high, decide_eq_decide, decide_eq_true_eq,
    decide_eq_false_iff_not]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> split_ifs <;> omega

/-- The entries `t + d` and `t` of the tables of `build_tables d q` have a sum of magnitude at
most 1 for `t < d`: the octant reduction gives the same first-octant enclosure with the opposite
sign, or the enclosure of `sin 0`, and `downscale_round` of the negation of an enclosure is the
negation of its entry or that minus 1. -/
theorem entry_pair {d q t : ℕ} (even : 2 ∣ d) (small : t < d) :
    |(entry d q (t + d)).1.1 + (entry d q t).1.1| ≤ 1 ∧
      |(entry d q (t + d)).2.1 + (entry d q t).2.1| ≤ 1 := by
  obtain ⟨index, swap, cosPair, sinPair, cosUnit, sinUnit⟩ := octantReduce_add even small
  simp only [entry, index, swap]
  constructor
  · rcases cosPair with opposite | ⟨zero, swapped⟩
    · exact downscaleRound_pair _ _ cosUnit opposite
    · simp only [zero, swapped, ↓reduceIte, firstOctant_zero, downscaleRound_zero]
      norm_num
  · rcases sinPair with opposite | ⟨zero, straight⟩
    · exact downscaleRound_pair _ _ sinUnit opposite
    · simp only [zero, straight, Bool.false_eq_true, ↓reduceIte, firstOctant_zero,
        downscaleRound_zero]
      norm_num

end TrigBuild

namespace TrigTable

/-- The tables of `build_tables d q` are antisymmetric up to 1 for each even degree `d`
(`TrigBuild.entry_pair`). -/
theorem code_antisymmetric {d : ℕ} (even : 2 ∣ d) (q : ℕ) : (code d q).Antisymmetric d 1 := by
  intro t small
  obtain ⟨cos, sin⟩ := TrigBuild.entry_pair (q := q) even small
  exact ⟨by exact_mod_cast cos, by exact_mod_cast sin⟩

/-- The bound 1 of `code_antisymmetric` is reached: `C[65] + C[1] = −1` for the degree 64 at the
scale `2^48`. -/
theorem code_cos_one : (code 64 48).cos 65 + (code 64 48).cos 1 = -1 := by
  decide +kernel

end TrigTable

namespace Key20

/-- The count of Akita Theorem H.2 at the frequency 0 for the two filtered families of the code:
`(d/2)·#{c ∈ C_0 | U_0(c) > T²·Δ²} + 2^128 ≤ |C_0|`, for the family `(31, 11)` of the degree 64
with `T = 18` and the family `(31, 0)` of the degree 128 with `T = 13`, with the tables of the
code at the scale `2^48` and the error bound `ε + 1`. -/
def Tail : Prop :=
  32 * Nat.card {c ∈ SparseFamily 64 31 11 |
        ¬ ((TrigTable.code 64 48).widen 1).Passes 18 c 0} + 2 ^ 128 ≤
      Nat.card (SparseFamily 64 31 11) ∧
    64 * Nat.card {c ∈ SparseFamily 128 31 0 |
        ¬ ((TrigTable.code 128 48).widen 1).Passes 13 c 0} + 2 ^ 128 ≤
      Nat.card (SparseFamily 128 31 0)

/-- `Tail` is an upper bound on the two counts at the frequency 0: the fractions
`36635855532104442060661780368307781632 / |C_0| ≈ 0.0242200` of the degree 64 and
`11041676670609652623885560346413367296 / |C_0| ≈ 0.0105465` of the degree 128. -/
theorem tail_iff : Tail ↔
    Nat.card {c ∈ SparseFamily 64 31 11 | ¬ ((TrigTable.code 64 48).widen 1).Passes 18 c 0} ≤
        36635855532104442060661780368307781632 ∧
      Nat.card {c ∈ SparseFamily 128 31 0 | ¬ ((TrigTable.code 128 48).widen 1).Passes 13 c 0} ≤
        11041676670609652623885560346413367296 := by
  have first : Nat.card (SparseFamily 64 31 11) = 1512629743948280609404551579217617223680 := by
    rw [card_sparseFamily]
    decide
  have second : Nat.card (SparseFamily 128 31 0) = 1046949673839956231392050469602223718400 := by
    rw [card_sparseFamily]
    decide
  have power : (2 : ℕ) ^ 128 = 340282366920938463463374607431768211456 := by norm_num
  rw [Tail, first, second, power]
  constructor
  · rintro ⟨low, high⟩
    exact ⟨Nat.lt_succ_iff.mp (Nat.lt_of_mul_lt_mul_left (a := 32) (by linarith)),
      Nat.lt_succ_iff.mp (Nat.lt_of_mul_lt_mul_left (a := 64) (by linarith))⟩
  · rintro ⟨low, high⟩
    constructor <;> linarith

/-- The union bound and the equal-marginal step of Akita Theorem H.2 for the integer predicate of
the code give `Kept` from `Tail`. -/
theorem kept_of_tail (tail : Tail) : Kept :=
  ⟨TrigTable.le_card_of_reject ((Nat.add_le_add_right (TrigTable.card_reject_le (d := 64)
      (n := 5) (by norm_num) (TrigTable.code_antisymmetric (by norm_num) 48) 18 31 11) _).trans
      tail.1),
    TrigTable.le_card_of_reject ((Nat.add_le_add_right (TrigTable.card_reject_le (d := 128)
      (n := 6) (by norm_num) (TrigTable.code_antisymmetric (by norm_num) 48) 13 31 0) _).trans
      tail.2)⟩

end Key20

end NightstreamFPrime.Spec.Compression.Akita
