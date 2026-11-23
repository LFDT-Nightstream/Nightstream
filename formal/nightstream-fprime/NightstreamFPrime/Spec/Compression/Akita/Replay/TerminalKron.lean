import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalRing

/-! A fast `Σ_{c<W} a_c·z_c` in `R_{q,d}` for ring elements `z_c` with small integer coefficients
(the rows of `A·z` of the terminal, `terminal/ntt.rs` `centered_rows`), by Kronecker
substitution: the coefficients of `z_c` are the digits of base `2^96` of one natural number, so
each coefficient of `a_c` multiplies all of them in one product of natural numbers.

- `Kron.pack f n = Σ_{s<n} f s·2^{96·s}` (`pack_eq`), and `Kron.digit x s` is digit `s` of `x`;
  `digit_sum` recovers each digit below `2^96`.
- `Kron.kron` is `Σ_c Σ_{i<d} a_{c,i}·pack(z_c)·2^{96·i}`, the digits of the full negacyclic
  convolution before the wrap (`kron_eq`).
- `Poly.dotSmall` takes the positive and the negative parts of `z`, reads the digits `k` and
  `k + d` and wraps. `toRq_dotSmall`: for `W·d ≤ 2^16` and `|z| ≤ 2^15` each digit is below
  `2^96`, and the result is `Σ_c toRq(a_c)·toRq(z_c)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

namespace Kron

def base : ℕ := 2 ^ 96

def pack (f : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => f 0 + base * pack (fun i => f (i + 1)) n

def digit (x s : ℕ) : ℕ := x / base ^ s % base

/-- `Σ_{i<n} f i`. -/
def sumNat (f : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => sumNat f n + f n

theorem sumNat_eq (f : ℕ → ℕ) : ∀ n, sumNat f n = ∑ i ∈ Finset.range n, f i
  | 0 => rfl
  | n + 1 => by rw [sumNat, sumNat_eq f n, Finset.sum_range_succ]

/-- `pack f n` from the top coefficient down (Horner), with no nested closures. -/
def packFast (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, acc => acc
  | k + 1, acc => packFast f k (f k + base * acc)

/-- `Σ_{c<W} Σ_{i<d} a_{c,i}·(pack z_c d)·2^{96·i}`, with `a_{c,i}` the word of coefficient `i`
of `a c`. -/
def kron (d W : ℕ) (a : ℕ → Array Fq) (z : ℕ → ℕ → ℕ) : ℕ :=
  sumNat (fun c =>
    let ac := a c
    let zc := packFast (z c) d 0
    sumNat (fun i => (Poly.get ac i).word.toNat * (zc <<< (96 * i))) d) W

theorem pack_eq : ∀ (f : ℕ → ℕ) (n : ℕ), pack f n = ∑ s ∈ Finset.range n, f s * base ^ s
  | _, 0 => rfl
  | f, n + 1 => by
      rw [pack, pack_eq _ n, Finset.sum_range_succ', Finset.mul_sum, pow_zero, mul_one, add_comm]
      congr 1
      refine Finset.sum_congr rfl fun s _ => ?_
      rw [pow_succ]
      ring

theorem packFast_eq (f : ℕ → ℕ) : ∀ (k acc : ℕ),
    packFast f k acc = ∑ s ∈ Finset.range k, f s * base ^ s + base ^ k * acc
  | 0, acc => by simp [packFast]
  | k + 1, acc => by
      rw [packFast, packFast_eq f k, Finset.sum_range_succ, pow_succ]
      ring

theorem digit_pack : ∀ (f : ℕ → ℕ) (n : ℕ), (∀ s, f s < base) → ∀ s,
    digit (pack f n) s = if s < n then f s else 0
  | _, 0, _, s => by simp [pack, digit]
  | f, n + 1, small, 0 => by
      simp only [digit, pack, pow_zero, Nat.div_one, Nat.add_mul_mod_self_left,
        Nat.mod_eq_of_lt (small 0), Nat.zero_lt_succ, ite_true]
  | f, n + 1, small, s + 1 => by
      have shift : (f 0 + base * pack (fun i => f (i + 1)) n) / base ^ (s + 1) =
          pack (fun i => f (i + 1)) n / base ^ s := by
        rw [pow_succ', ← Nat.div_div_eq_div_mul, Nat.add_mul_div_left _ _ (by decide : 0 < base),
          Nat.div_eq_of_lt (small 0), zero_add]
      have rest := digit_pack (fun i => f (i + 1)) n (fun s => small (s + 1)) s
      simp only [digit] at rest ⊢
      rw [pack, shift, rest]
      by_cases inside : s < n
      · rw [ite_eq_left inside, ite_eq_left (by omega)]
      · rw [ite_eq_right inside, ite_eq_right (by omega)]

theorem digit_sum (f : ℕ → ℕ) (N : ℕ) (small : ∀ s < N, f s < base) {s : ℕ} (inside : s < N) :
    digit (∑ t ∈ Finset.range N, f t * base ^ t) s = f s := by
  have same : ∑ t ∈ Finset.range N, f t * base ^ t =
      pack (fun t => if t < N then f t else 0) N := by
    rw [pack_eq]
    exact Finset.sum_congr rfl fun t member => by rw [ite_eq_left (Finset.mem_range.mp member)]
  rw [same, digit_pack _ N (fun t => by split <;> [exact small t ‹_›; exact by decide]) s,
    ite_eq_left inside, ite_eq_left inside]

/-- `Σ_{s<N} [i ≤ s < i + d]·g(s − i)·2^{96·s} = Σ_{j<d} g j·2^{96·(i + j)}`. -/
theorem shift_sum (g : ℕ → ℕ) {i d N : ℕ} (fits : i + d ≤ N) :
    ∑ s ∈ Finset.range N, (if i ≤ s ∧ s - i < d then g (s - i) else 0) * base ^ s =
      ∑ j ∈ Finset.range d, g j * base ^ (i + j) := by
  rw [show N = i + (d + (N - i - d)) by omega, Finset.sum_range_add, Finset.sum_range_add,
    Finset.sum_eq_zero (fun s member => by
      rw [ite_eq_right (by have := Finset.mem_range.mp member; omega), zero_mul]),
    zero_add, Finset.sum_eq_zero (s := Finset.range (N - i - d)) (fun s _ => by
      rw [ite_eq_right (by omega), zero_mul]), add_zero]
  exact Finset.sum_congr rfl fun j member => by
    rw [ite_eq_left ⟨by omega, by have := Finset.mem_range.mp member; omega⟩,
      Nat.add_sub_cancel_left]

/-- The convolution digits: `conv s = Σ_c Σ_{i<d} [i ≤ s < i + d]·a_{c,i}·z_c(s − i)`. -/
def conv (d W : ℕ) (a z : ℕ → ℕ → ℕ) (s : ℕ) : ℕ :=
  ∑ c ∈ Finset.range W, ∑ i ∈ Finset.range d,
    a c i * if i ≤ s ∧ s - i < d then z c (s - i) else 0

theorem kron_eq (d W : ℕ) (a : ℕ → Array Fq) (z : ℕ → ℕ → ℕ) :
    kron d W a z = ∑ s ∈ Finset.range (2 * d),
      conv d W (fun c i => (Poly.get (a c) i).word.toNat) z s * base ^ s := by
  have inner : ∀ c, ∀ i < d, (Poly.get (a c) i).word.toNat * (packFast (z c) d 0 <<< (96 * i)) =
      ∑ s ∈ Finset.range (2 * d), (Poly.get (a c) i).word.toNat *
        (if i ≤ s ∧ s - i < d then z c (s - i) else 0) * base ^ s := by
    intro c i inside
    simp only [mul_assoc]
    rw [← Finset.mul_sum, shift_sum (z c) (by omega : i + d ≤ 2 * d), Nat.shiftLeft_eq,
      packFast_eq, mul_zero, add_zero, Finset.sum_mul]
    congr 1
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [pow_add, show base ^ i = 2 ^ (96 * i) by rw [base, ← pow_mul]]
    ring
  unfold kron conv
  rw [sumNat_eq]
  simp only [sumNat_eq, Finset.sum_mul]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun c _ => ?_
  conv_rhs => rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun i member => inner c i (Finset.mem_range.mp member)

end Kron

namespace Poly

open Kron

/-- `Σ_{c<W} a_c·z_c` in `R_{q,d}` for the integer coefficients `z c j` (`j < d`). -/
def dotSmall (d W : ℕ) (a : ℕ → Array Fq) (z : ℕ → ℕ → ℤ) : Array Fq :=
  let plus := kron d W a fun c j => (z c j).toNat
  let minus := kron d W a fun c j => (-z c j).toNat
  ofFn d fun k => Fq.ofInt ((digit plus k : ℤ) - digit minus k - digit plus (k + d) +
    digit minus (k + d))

theorem conv_lt {d W : ℕ} (bound : W * d ≤ 2 ^ 16) (a z : ℕ → ℕ → ℕ)
    (word : ∀ c i, a c i < 2 ^ 64) (small : ∀ c j, z c j ≤ 2 ^ 15) (s : ℕ) :
    conv d W a z s < base := by
  have each : ∀ c i, a c i * (if i ≤ s ∧ s - i < d then z c (s - i) else 0) ≤ 2 ^ 64 * 2 ^ 15 :=
    fun c i => Nat.mul_le_mul (word c i).le (by split <;> [exact small _ _; exact Nat.zero_le _])
  have inner : ∀ c, ∑ i ∈ Finset.range d,
      a c i * (if i ≤ s ∧ s - i < d then z c (s - i) else 0) ≤ d * (2 ^ 64 * 2 ^ 15) := fun c => by
    have bounded := Finset.sum_le_card_nsmul (Finset.range d) _ _ fun i _ => each c i
    simpa using bounded
  have total : conv d W a z s ≤ W * (d * (2 ^ 64 * 2 ^ 15)) := by
    have bounded := Finset.sum_le_card_nsmul (Finset.range W) _ _ fun c _ => inner c
    simpa [conv] using bounded
  calc conv d W a z s ≤ W * (d * (2 ^ 64 * 2 ^ 15)) := total
    _ = W * d * 2 ^ 79 := by ring
    _ ≤ 2 ^ 16 * 2 ^ 79 := Nat.mul_le_mul_right _ bound
    _ < base := by decide

variable {d : ℕ} [NeZero d]

theorem toRq_dotSmall {W : ℕ} (bound : W * d ≤ 2 ^ 16) (a : ℕ → Array Fq) (z : ℕ → ℕ → ℤ)
    (small : ∀ c j, (z c j).natAbs ≤ 2 ^ 15) :
    toRq d (dotSmall d W a z) =
      ∑ c ∈ Finset.range W, toRq d (a c) * toRq d (ofFn d fun j => Fq.ofInt (z c j)) := by
  refine ext_coeff fun k smallK => ?_
  rw [coeff_toRq _ smallK]
  dsimp only [dotSmall]
  set A : ℕ → ℕ → ℕ := fun c i => (get (a c) i).word.toNat with hA
  set P : ℕ → ℕ → ℕ := fun c j => (z c j).toNat with hP
  set M : ℕ → ℕ → ℕ := fun c j => (-z c j).toNat with hM
  have word : ∀ c i, A c i < 2 ^ 64 := fun c i => UInt64.toNat_lt _
  have plusSmall : ∀ c j, P c j ≤ 2 ^ 15 := fun c j => by
    have := small c j
    simp only [hP]
    omega
  have minusSmall : ∀ c j, M c j ≤ 2 ^ 15 := fun c j => by
    have := small c j
    simp only [hM]
    omega
  have digitOf : ∀ (Z : ℕ → ℕ → ℕ), (∀ c j, Z c j ≤ 2 ^ 15) → ∀ s < 2 * d,
      digit (kron d W a Z) s = conv d W A Z s := fun Z bounded s inside => by
    rw [kron_eq]
    exact digit_sum _ _ (fun t _ => conv_lt bound A Z word bounded t) inside
  rw [get_ofFn _ smallK, Fq.toZq_ofInt, digitOf P plusSmall k (by omega),
    digitOf M minusSmall k (by omega), digitOf P plusSmall (k + d) (by omega),
    digitOf M minusSmall (k + d) (by omega), coeff_sum]
  have term : ∀ c i, i < d →
      ((A c i * (if i ≤ k ∧ k - i < d then P c (k - i) else 0) : ℕ) : ℤ) -
        (A c i * (if i ≤ k ∧ k - i < d then M c (k - i) else 0) : ℕ) -
        (A c i * (if i ≤ k + d ∧ k + d - i < d then P c (k + d - i) else 0) : ℕ) +
        (A c i * (if i ≤ k + d ∧ k + d - i < d then M c (k + d - i) else 0) : ℕ) =
      (A c i : ℤ) * (if k < i then -z c (d + k - i) else z c (k - i)) := fun c i inside => by
    have parts (n : ℤ) : (n.toNat : ℤ) - ((-n).toNat : ℤ) = n := by omega
    by_cases low : k < i
    · rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left ⟨by omega, by omega⟩,
        ite_eq_left ⟨by omega, by omega⟩, ite_eq_left low, show k + d - i = d + k - i by omega]
      have value := parts (z c (d + k - i))
      simp only [hP, hM]
      push_cast
      linear_combination (-(A c i : ℤ)) * value
    · rw [ite_eq_left ⟨by omega, by omega⟩, ite_eq_left ⟨by omega, by omega⟩,
        ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right low]
      have value := parts (z c (k - i))
      simp only [hP, hM]
      push_cast
      linear_combination (A c i : ℤ) * value
  have total : ((conv d W A P k : ℕ) : ℤ) - (conv d W A M k : ℕ) - (conv d W A P (k + d) : ℕ) +
      (conv d W A M (k + d) : ℕ) = ∑ c ∈ Finset.range W, ∑ i ∈ Finset.range d,
        (A c i : ℤ) * (if k < i then -z c (d + k - i) else z c (k - i)) := by
    unfold conv
    push_cast
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i member => ?_
    have value := term c i (Finset.mem_range.mp member)
    push_cast at value
    exact value
  rw [total]
  push_cast
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [← toRq_mul, coeff_toRq _ smallK, mul, get_ofFn _ smallK, toZq_sumFq]
  refine Finset.sum_congr rfl fun i member => ?_
  have inside := Finset.mem_range.mp member
  rw [Fq.toZq_mul]
  have wordCast : ((A c i : ℕ) : Zq) = (get (a c) i).toZq := rfl
  rw [← wordCast]
  by_cases low : k < i
  · rw [ite_eq_left low, ite_eq_left low, Fq.toZq_neg, get_ofFn _ (by omega), Fq.toZq_ofInt]
  · rw [ite_eq_right low, ite_eq_right low, get_ofFn _ (by omega), Fq.toZq_ofInt]

end Poly

end NightstreamFPrime.Spec.Compression.Akita.Replay
