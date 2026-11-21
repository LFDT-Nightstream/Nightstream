import Mathlib.Analysis.Real.Pi.Bounds
import NightstreamFPrime.Spec.Compression.Akita.OperatorFilter

/-! The integer tables of the operator-norm filter of the pinned Akita code
(`crates/akita-challenges/src/sampler/op_norm.rs`, `build_tables`; Akita, ePrint 2026/1983,
revision of 2026-10-15, Appendix H.1).

- `TrigBuild.piEnclosure` is `pi_enclosure` (Machin's formula with `arctan_inv`).
- `TrigBuild.taylorCos` and `TrigBuild.taylorSin` are `taylor_cos` and `taylor_sin`: interval
  Taylor sums with 12 terms after the first and the remainder of `add_remainder`.
- `TrigBuild.octantReduce` is `octant_reduce`, and `TrigBuild.entry` and `TrigBuild.epsRoot` are
  the loop of `build_tables`.
- `TrigTable.code d q` is the table `(base_cos, base_sin, eps_root)` of `build_tables(d, q)` at
  the scale `2^q`, and `TrigTable.code_accurate` states that it is accurate for `2 ∣ d` and
  `q ≤ 61`.
- `TrigTable.codeAccepts d T` is the strict predicate `accept_strict_parts` of the code: the
  fixed-point predicate of `TrigTable.code d 48` (the scale `OP_NORM_PREDICATE_SCALE = 48` of
  `sampler/mod.rs`). `TrigTable.filtered_code` states that it is a filter with
  `TrigTable.Filtered`.

The model computes over `ℤ` and the code over `i128`. For the degrees 64 and 128 no product of
the code reaches `2^122`. The loop of `arctan_inv` stops at a term 0 after 12 terms for `x = 5`
and after 4 terms for `x = 239` (`TrigBuild.arctan_counts`), and its products stay below
`2^127`, so no `checked_mul` fails. So the model and the code compute the same tables.
`TrigBuild.epsRoot_eq` states that `eps_root` is 1 for the degrees 64 and 128 at the scale
`2^48`. -/

namespace NightstreamFPrime.Spec.Compression.Akita

namespace TrigBuild

/-- `TRIG_ONE`: the value 1 at the scale `2^61` (`TRIG_SCALE = 61`). -/
def one : ℤ := 2 ^ 61

/-- `NTERMS`: the count of Taylor terms after the first. -/
def nterms : ℕ := 12

/-- `fdiv`: `⌊a/b⌋` for `b > 0` (`div_euclid`). -/
def fdiv (a b : ℤ) : ℤ := a / b

/-- `cdiv`: `⌈a/b⌉` for `b > 0`. -/
def cdiv (a b : ℤ) : ℤ := -((-a) / b)

/-- `fshr`: `⌊x/2^shift⌋`, the arithmetic shift. -/
def fshr (x : ℤ) (shift : ℕ) : ℤ := x / 2 ^ shift

/-- `cshr`: `⌈x/2^shift⌉`. -/
def cshr (x : ℤ) (shift : ℕ) : ℤ := -((-x) / 2 ^ shift)

/-- `imul`: the product of two intervals at the scale `2^61` with nonnegative lower ends. -/
def imul (a b : ℤ × ℤ) : ℤ × ℤ := (fshr (a.1 * b.1) 61, cshr (a.2 * b.2) 61)

/-- `iadd`: the sum of two intervals. -/
def iadd (a b : ℤ × ℤ) : ℤ × ℤ := (a.1 + b.1, a.2 + b.2)

/-- `isub`: the difference of two intervals. -/
def isub (a b : ℤ × ℤ) : ℤ × ℤ := (a.1 - b.2, a.2 - b.1)

/-- The loop of `arctan_inv x` at the step `n` with the sum `sum` and the power
`power = x^{2n+1}`, for at most `fuel` steps. A step adds the term
`⌊2^61/((2n+1)·x^{2n+1})⌋` with the sign `(−1)^n`. The loop stops at the first term 0 and
returns the sum and the count of terms. -/
def arctanLoop (x : ℤ) : ℕ → ℕ → ℤ → ℤ → ℤ × ℕ
  | 0, n, sum, _ => (sum, n)
  | fuel + 1, n, sum, power =>
    let term := one / ((2 * (n : ℤ) + 1) * power)
    if term = 0 then (sum, n)
    else arctanLoop x fuel (n + 1) (if n % 2 = 0 then sum + term else sum - term) (power * (x * x))

/-- `arctan_inv x`: the sum of the loop, widened by the count of terms plus 1. -/
def arctanInv (x : ℤ) : ℤ × ℤ :=
  let s := arctanLoop x 64 0 0 x
  (s.1 - (s.2 + 1), s.1 + (s.2 + 1))

/-- `pi_enclosure`: `π = 16·arctan(1/5) − 4·arctan(1/239)` with outward rounding, at the scale
`2^61`. -/
def piEnclosure : ℤ × ℤ :=
  (16 * (arctanInv 5).1 - 4 * (arctanInv 239).2, 16 * (arctanInv 5).2 - 4 * (arctanInv 239).1)

/-- The state of the loop of `taylor_cos` and `taylor_sin`: the sum, the power and the
factorial. -/
structure Taylor where
  sum : ℤ × ℤ
  power : ℤ × ℤ
  fact : ℤ

/-- The loop of `taylor_cos` (`odd = 0`) and of `taylor_sin` (`odd = 1`) after `n` steps, from
the sum and the power `start`. The step `n` multiplies the power by `phi2` and the factorial by
`(2n + odd − 1)·(2n + odd)`. It subtracts the term for an odd `n` and adds it for an even `n`. -/
def taylorLoop (phi2 start : ℤ × ℤ) (odd : ℕ) : ℕ → Taylor
  | 0 => ⟨start, start, 1⟩
  | n + 1 =>
    let s := taylorLoop phi2 start odd n
    let power := imul s.power phi2
    let fact := s.fact * ((2 * (n + 1) + odd - 1 : ℕ) : ℤ) * ((2 * (n + 1) + odd : ℕ) : ℤ)
    let term := (fdiv power.1 fact, cdiv power.2 fact)
    ⟨if (n + 1) % 2 = 1 then isub s.sum term else iadd s.sum term, power, fact⟩

/-- `add_remainder`: the sum widened by `⌈p/(f·m·(m+1))⌉`, for the upper end `p` of the next
power and the next factorial `f·m·(m+1)`. -/
def addRemainder (sum phi2 power : ℤ × ℤ) (fact : ℤ) (next : ℕ) : ℤ × ℤ :=
  let rem := cdiv (imul power phi2).2 (fact * next * (next + 1))
  (sum.1 - rem, sum.2 + rem)

/-- `taylor_cos`: an enclosure of `cos φ` at the scale `2^61` from an enclosure `phi` of `φ`. -/
def taylorCos (phi : ℤ × ℤ) : ℤ × ℤ :=
  let s := taylorLoop (imul phi phi) (one, one) 0 nterms
  addRemainder s.sum (imul phi phi) s.power s.fact (2 * nterms + 1)

/-- `taylor_sin`: an enclosure of `sin φ` at the scale `2^61` from an enclosure `phi` of `φ`. -/
def taylorSin (phi : ℤ × ℤ) : ℤ × ℤ :=
  let s := taylorLoop (imul phi phi) phi 1 nterms
  addRemainder s.sum (imul phi phi) s.power s.fact (2 * nterms + 2)

/-- The result of `octant_reduce`: the index `r`, the swap of `cos` and `sin`, and the signs. -/
structure Octant where
  index : ℕ
  swap : Bool
  cosSign : ℤ
  sinSign : ℤ

/-- `octant_reduce t d`: the reflections of `t mod 2d` at `d`, at `d/2` and at `d/4`. -/
def octantReduce (t d : ℕ) : Octant :=
  let t₁ := t % (2 * d)
  let t₂ := if d < t₁ then 2 * d - t₁ else t₁
  let t₃ := if d / 2 < t₂ then d - t₂ else t₂
  ⟨if d / 4 < t₃ then d / 2 - t₃ else t₃, decide (d / 4 < t₃),
    if d / 2 < t₂ then -1 else 1, if d < t₁ then -1 else 1⟩

/-- `apply_sign`: the interval times the sign `±1`. -/
def applySign (v : ℤ × ℤ) (sign : ℤ) : ℤ × ℤ := if sign < 0 then (-v.2, -v.1) else v

/-- `downscale_round`: the interval at the scale `2^61` rounded outward to the scale
`2^(61 − shift)`, the midpoint entry and its error. -/
def downscaleRound (v : ℤ × ℤ) (shift : ℕ) : ℤ × ℤ :=
  let lo := fshr v.1 shift
  let hi := cshr v.2 shift
  let entry := fdiv (lo + hi) 2
  (entry, max (entry - lo) (hi - entry))

/-- `first_octant[r]`: the enclosures of `cos(πr/d)` and `sin(πr/d)` at the scale `2^61`. -/
def firstOctant (d r : ℕ) : (ℤ × ℤ) × (ℤ × ℤ) :=
  let phi := (fdiv (piEnclosure.1 * r) d, cdiv (piEnclosure.2 * r) d)
  (taylorCos phi, taylorSin phi)

/-- The step `t` of the loop of `build_tables d q`: the entry `C[t]` and its error, and the
entry `S[t]` and its error. -/
def entry (d q t : ℕ) : (ℤ × ℤ) × (ℤ × ℤ) :=
  let o := octantReduce t d
  let p := firstOctant d o.index
  (downscaleRound (applySign (if o.swap then p.2 else p.1) o.cosSign) (61 - q),
    downscaleRound (applySign (if o.swap then p.1 else p.2) o.sinSign) (61 - q))

/-- `eps_root` of `build_tables d q`: the largest error of the entries `t < 2d`. -/
def epsRoot (d q : ℕ) : ℤ :=
  (List.range (2 * d)).foldl (fun e t => max (max e (entry d q t).1.2) (entry d q t).2.2) 0

/-- The loop of `arctan_inv` stops at a term 0 after 12 terms for `x = 5` and after 4 terms for
`x = 239`, before the fuel 64 ends. -/
theorem arctan_counts : (arctanLoop 5 64 0 0 5).2 = 12 ∧ (arctanLoop 239 64 0 0 239).2 = 4 := by
  decide

/-- The values of `pi_enclosure`. -/
theorem piEnclosure_eq : piEnclosure = (7244019458077122644, 7244019458077123100) := by
  decide

/-- `eps_root` of the tables of the code for the degrees 64 and 128 at the scale `2^48` is 1. -/
theorem epsRoot_eq : epsRoot 64 48 = 1 ∧ epsRoot 128 48 = 1 := by
  decide +kernel

private def Encloses (k : ℕ) (v : ℤ × ℤ) (x : ℝ) : Prop :=
  (v.1 : ℝ) ≤ 2 ^ k * x ∧ 2 ^ k * x ≤ v.2

private theorem Encloses.of_eq {k : ℕ} {v : ℤ × ℤ} {x y : ℝ} (enclosed : Encloses k v x)
    (same : x = y) : Encloses k v y :=
  same ▸ enclosed

private theorem fdiv_le (a : ℤ) {b : ℤ} (positive : 0 < b) : (fdiv a b : ℝ) ≤ a / b := by
  rw [le_div_iff₀ (Int.cast_pos.mpr positive)]
  exact_mod_cast Int.ediv_mul_le a positive.ne'

private theorem le_cdiv (a : ℤ) {b : ℤ} (positive : 0 < b) : (a / b : ℝ) ≤ cdiv a b := by
  have below := fdiv_le (-a) positive
  rw [fdiv, Int.cast_neg, neg_div] at below
  rw [cdiv, Int.cast_neg]
  linarith

private theorem fshr_le (x : ℤ) (shift : ℕ) : (fshr x shift : ℝ) ≤ x / 2 ^ shift := by
  have below := fdiv_le x (pow_pos (by norm_num : (0 : ℤ) < 2) shift)
  push_cast at below
  exact below

private theorem le_cshr (x : ℤ) (shift : ℕ) : (x / 2 ^ shift : ℝ) ≤ cshr x shift := by
  have above := le_cdiv x (pow_pos (by norm_num : (0 : ℤ) < 2) shift)
  push_cast at above
  exact above

private theorem imul_encloses {a b : ℤ × ℤ} {x y : ℝ} (first : Encloses 61 a x)
    (second : Encloses 61 b y) (firstNonneg : 0 ≤ a.1) (secondNonneg : 0 ≤ b.1) :
    Encloses 61 (imul a b) (x * y) ∧ 0 ≤ (imul a b).1 := by
  have a₀ : (0 : ℝ) ≤ a.1 := by exact_mod_cast firstNonneg
  have b₀ : (0 : ℝ) ≤ b.1 := by exact_mod_cast secondNonneg
  have low : (a.1 : ℝ) * b.1 ≤ 2 ^ 61 * x * (2 ^ 61 * y) :=
    mul_le_mul first.1 second.1 b₀ (a₀.trans first.1)
  have high : 2 ^ 61 * x * (2 ^ 61 * y) ≤ (a.2 : ℝ) * b.2 :=
    mul_le_mul first.2 second.2 (b₀.trans second.1) ((a₀.trans first.1).trans first.2)
  have scale : (0 : ℝ) < 2 ^ 61 := by positivity
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · have below := fshr_le (a.1 * b.1) 61
    push_cast at below
    refine below.trans ?_
    rw [div_le_iff₀ scale]
    linarith
  · have above := le_cshr (a.2 * b.2) 61
    push_cast at above
    refine le_trans ?_ above
    rw [le_div_iff₀ scale]
    linarith
  · exact Int.ediv_nonneg (mul_nonneg firstNonneg secondNonneg) (by positivity)

private theorem term_encloses {p : ℤ × ℤ} {x : ℝ} {f : ℤ} (enclosed : Encloses 61 p x)
    (positive : 0 < f) : Encloses 61 (fdiv p.1 f, cdiv p.2 f) (x / f) := by
  have f₀ : (0 : ℝ) ≤ f := by exact_mod_cast positive.le
  constructor
  · calc ((fdiv p.1 f : ℤ) : ℝ) ≤ p.1 / f := fdiv_le p.1 positive
      _ ≤ 2 ^ 61 * x / f := div_le_div_of_nonneg_right enclosed.1 f₀
      _ = 2 ^ 61 * (x / f) := mul_div_assoc _ _ _
  · calc 2 ^ 61 * (x / f) = 2 ^ 61 * x / f := (mul_div_assoc _ _ _).symm
      _ ≤ p.2 / f := div_le_div_of_nonneg_right enclosed.2 f₀
      _ ≤ cdiv p.2 f := le_cdiv p.2 positive

private theorem iadd_encloses {a b : ℤ × ℤ} {x y : ℝ} (first : Encloses 61 a x)
    (second : Encloses 61 b y) : Encloses 61 (iadd a b) (x + y) := by
  constructor
  · show ((a.1 + b.1 : ℤ) : ℝ) ≤ _
    push_cast
    linarith [first.1, second.1]
  · show _ ≤ ((a.2 + b.2 : ℤ) : ℝ)
    push_cast
    linarith [first.2, second.2]

private theorem isub_encloses {a b : ℤ × ℤ} {x y : ℝ} (first : Encloses 61 a x)
    (second : Encloses 61 b y) : Encloses 61 (isub a b) (x - y) := by
  constructor
  · show ((a.1 - b.2 : ℤ) : ℝ) ≤ _
    push_cast
    linarith [first.1, second.2]
  · show _ ≤ ((a.2 - b.1 : ℤ) : ℝ)
    push_cast
    linarith [first.2, second.1]

private noncomputable def partialSum (odd : ℕ) (x : ℝ) (n : ℕ) : ℝ :=
  ∑ k ∈ Finset.range n, (-1) ^ k * x ^ (2 * k + odd) / (2 * k + odd).factorial

private theorem taylorLoop_encloses {phi2 start : ℤ × ℤ} {odd : ℕ} {x : ℝ}
    (square : Encloses 61 phi2 (x ^ 2)) (squareNonneg : 0 ≤ phi2.1)
    (first : Encloses 61 start (x ^ odd)) (firstNonneg : 0 ≤ start.1) (small : odd ≤ 1)
    (n : ℕ) :
    Encloses 61 (taylorLoop phi2 start odd n).sum (partialSum odd x (n + 1)) ∧
      Encloses 61 (taylorLoop phi2 start odd n).power (x ^ (2 * n + odd)) ∧
      0 ≤ (taylorLoop phi2 start odd n).power.1 ∧
      (taylorLoop phi2 start odd n).fact = (2 * n + odd).factorial := by
  induction n with
  | zero =>
    have unit : odd.factorial = 1 := Nat.factorial_eq_one.mpr small
    refine ⟨first.of_eq ?_, first.of_eq (by rw [Nat.mul_zero, Nat.zero_add]), firstNonneg, ?_⟩
    · simp [partialSum, unit]
    · simp [taylorLoop, unit]
  | succ n ih =>
    obtain ⟨sum, power, powerNonneg, fact⟩ := ih
    obtain ⟨next, nextNonneg⟩ := imul_encloses power square powerNonneg squareNonneg
    have factorial : (taylorLoop phi2 start odd n).fact * ((2 * (n + 1) + odd - 1 : ℕ) : ℤ) *
        ((2 * (n + 1) + odd : ℕ) : ℤ) = ((2 * (n + 1) + odd).factorial : ℤ) := by
      rw [fact, show 2 * (n + 1) + odd - 1 = 2 * n + odd + 1 by omega,
        show 2 * (n + 1) + odd = 2 * n + odd + 1 + 1 by omega, Nat.factorial_succ,
        Nat.factorial_succ]
      push_cast
      ring
    have positive : (0 : ℤ) < ((2 * (n + 1) + odd).factorial : ℤ) := by
      exact_mod_cast Nat.factorial_pos _
    have term := term_encloses next (factorial ▸ positive)
    rw [factorial] at term
    have term' : Encloses 61 (fdiv (imul (taylorLoop phi2 start odd n).power phi2).1
        ((2 * (n + 1) + odd).factorial : ℤ), cdiv (imul (taylorLoop phi2 start odd n).power phi2).2
        ((2 * (n + 1) + odd).factorial : ℤ))
        (x ^ (2 * (n + 1) + odd) / (2 * (n + 1) + odd).factorial) := by
      refine term.of_eq ?_
      push_cast
      ring
    simp only [taylorLoop]
    rw [factorial]
    refine ⟨?_, next.of_eq (by ring), nextNonneg, rfl⟩
    have extend : partialSum odd x (n + 1 + 1) = partialSum odd x (n + 1) +
        (-1) ^ (n + 1) * (x ^ (2 * (n + 1) + odd) / (2 * (n + 1) + odd).factorial) := by
      rw [partialSum, Finset.sum_range_succ, ← partialSum]
      ring
    rw [extend]
    split_ifs with parity
    · rw [Odd.neg_one_pow (Nat.odd_iff.mpr parity), neg_one_mul, ← sub_eq_add_neg]
      exact isub_encloses sum term'
    · rw [Even.neg_one_pow (Nat.even_iff.mpr (by omega)), one_mul]
      exact iadd_encloses sum term'

private theorem sum_exp (x : ℝ) (n : ℕ) :
    ∑ m ∈ Finset.range (2 * n), ((x : ℂ) * Complex.I) ^ m / (m.factorial : ℂ) =
      (partialSum 0 x n : ℂ) + (partialSum 1 x n : ℂ) * Complex.I := by
  induction n with
  | zero => simp [partialSum]
  | succ n ih =>
    have even : ((x : ℂ) * Complex.I) ^ (2 * n) = (-1) ^ n * (x : ℂ) ^ (2 * n) := by
      rw [mul_pow, pow_mul Complex.I, Complex.I_sq, mul_comm]
    have odd : ((x : ℂ) * Complex.I) ^ (2 * n + 1) =
        (-1) ^ n * (x : ℂ) ^ (2 * n + 1) * Complex.I := by
      rw [pow_succ, even]
      ring
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ, ih,
      even, odd]
    simp only [partialSum, Finset.sum_range_succ, add_zero]
    push_cast
    ring

private theorem exp_close {x : ℝ} (small : |x| ≤ 1) :
    ‖Complex.exp (x * Complex.I) - ((partialSum 0 x 13 : ℂ) + (partialSum 1 x 13 : ℂ) *
        Complex.I)‖ ≤ |x| ^ 26 * (27 * ((Nat.factorial 26 : ℝ) * 26)⁻¹) := by
  have norm : ‖(x : ℂ) * Complex.I‖ = |x| := by simp
  have bound := Complex.exp_bound (x := (x : ℂ) * Complex.I) (norm ▸ small) (n := 26)
    (by norm_num)
  have series := sum_exp x 13
  rw [show 2 * 13 = 26 from rfl] at series
  rw [series, norm] at bound
  norm_num at bound ⊢
  exact bound

private theorem cos_close {x : ℝ} (small : |x| ≤ 1) :
    |Real.cos x - partialSum 0 x 13| ≤ |x| ^ 26 * (27 * ((Nat.factorial 26 : ℝ) * 26)⁻¹) := by
  refine le_trans (le_of_eq ?_) ((Complex.abs_re_le_norm _).trans (exp_close small))
  simp [Complex.exp_ofReal_mul_I_re]

private theorem sin_close {x : ℝ} (small : |x| ≤ 1) :
    |Real.sin x - partialSum 1 x 13| ≤ |x| ^ 26 * (27 * ((Nat.factorial 26 : ℝ) * 26)⁻¹) := by
  refine le_trans (le_of_eq ?_) ((Complex.abs_im_le_norm _).trans (exp_close small))
  simp [Complex.exp_ofReal_mul_I_im]

private theorem taylor_encloses {phi start : ℤ × ℤ} {odd : ℕ} {x f : ℝ}
    (enclosed : Encloses 61 phi x) (phiNonneg : 0 ≤ phi.1) (small : odd ≤ 1)
    (first : Encloses 61 start (x ^ odd)) (firstNonneg : 0 ≤ start.1) (unit : |x| ≤ 1)
    (close : |f - partialSum odd x 13| ≤ |x| ^ 26 * (27 * ((Nat.factorial 26 : ℝ) * 26)⁻¹)) :
    Encloses 61 (addRemainder (taylorLoop (imul phi phi) start odd nterms).sum (imul phi phi)
      (taylorLoop (imul phi phi) start odd nterms).power
      (taylorLoop (imul phi phi) start odd nterms).fact (2 * nterms + 1 + odd)) f := by
  have scale : (0 : ℝ) < 2 ^ 61 := by positivity
  have x₀ : 0 ≤ x := by
    have := (by exact_mod_cast phiNonneg : (0 : ℝ) ≤ phi.1).trans enclosed.1
    nlinarith
  obtain ⟨square, squareNonneg⟩ := imul_encloses enclosed enclosed phiNonneg phiNonneg
  obtain ⟨sum, power, powerNonneg, fact⟩ :=
    taylorLoop_encloses (square.of_eq (sq x).symm) squareNonneg first firstNonneg small nterms
  obtain ⟨next, -⟩ := imul_encloses power (square.of_eq (sq x).symm) powerNonneg squareNonneg
  have factorial : (taylorLoop (imul phi phi) start odd nterms).fact *
      ((2 * nterms + 1 + odd : ℕ) : ℤ) * (((2 * nterms + 1 + odd : ℕ) : ℤ) + 1) =
      ((26 + odd).factorial : ℤ) := by
    rw [fact, show 26 + odd = 24 + odd + 1 + 1 by omega, Nat.factorial_succ, Nat.factorial_succ]
    simp only [nterms]
    push_cast
    ring
  have positive : (0 : ℝ) < ((26 + odd).factorial : ℝ) := by exact_mod_cast Nat.factorial_pos _
  have remLower := le_cdiv (imul (taylorLoop (imul phi phi) start odd nterms).power
    (imul phi phi)).2 (b := (taylorLoop (imul phi phi) start odd nterms).fact *
      ((2 * nterms + 1 + odd : ℕ) : ℤ) * (((2 * nterms + 1 + odd : ℕ) : ℤ) + 1))
    (by rw [factorial]; exact_mod_cast Nat.factorial_pos _)
  set rem := cdiv (imul (taylorLoop (imul phi phi) start odd nterms).power (imul phi phi)).2
    ((taylorLoop (imul phi phi) start odd nterms).fact * ((2 * nterms + 1 + odd : ℕ) : ℤ) *
      (((2 * nterms + 1 + odd : ℕ) : ℤ) + 1)) with remDef
  rw [factorial] at remLower
  push_cast at remLower
  have lower : 2 ^ 61 * x ^ (26 + odd) / ((26 + odd).factorial : ℝ) ≤ rem := by
    refine le_trans ?_ remLower
    refine div_le_div_of_nonneg_right (le_trans (le_of_eq ?_) next.2) positive.le
    simp only [nterms]
    ring
  have remNonneg : (0 : ℝ) ≤ rem := le_trans (by positivity) lower
  have numeric : (2 : ℝ) ^ 61 * (27 * ((Nat.factorial 26 : ℝ) * 26)⁻¹) ≤ 1 := by
    rw [← div_eq_mul_inv, mul_div_assoc', div_le_one (by positivity)]
    exact_mod_cast (by decide : 2 ^ 61 * 27 ≤ Nat.factorial 26 * 26)
  have bound : 2 ^ 61 * |f - partialSum odd x 13| ≤ rem := by
    rcases x₀.eq_or_lt with zero | pos
    · have vanish : |x| ^ 26 = 0 := by rw [← zero]; norm_num
      rw [vanish, zero_mul] at close
      rw [le_antisymm close (abs_nonneg _), mul_zero]
      exact remNonneg
    · have above : (0 : ℝ) < rem := lt_of_lt_of_le (by positivity) lower
      have atLeast : (1 : ℝ) ≤ rem := by
        have : (0 : ℤ) < rem := by exact_mod_cast above
        exact_mod_cast this
      calc 2 ^ 61 * |f - partialSum odd x 13|
          ≤ 2 ^ 61 * (|x| ^ 26 * (27 * ((Nat.factorial 26 : ℝ) * 26)⁻¹)) :=
            mul_le_mul_of_nonneg_left close scale.le
        _ ≤ 2 ^ 61 * (27 * ((Nat.factorial 26 : ℝ) * 26)⁻¹) := by
            refine mul_le_mul_of_nonneg_left ?_ scale.le
            exact mul_le_of_le_one_left (by positivity) (pow_le_one₀ (abs_nonneg x) unit)
        _ ≤ 1 := numeric
        _ ≤ rem := atLeast
  have up : 2 ^ 61 * (f - partialSum odd x 13) ≤ rem :=
    (mul_le_mul_of_nonneg_left (le_abs_self _) scale.le).trans bound
  have down : 2 ^ 61 * (partialSum odd x 13 - f) ≤ rem :=
    (mul_le_mul_of_nonneg_left ((le_abs_self _).trans (abs_sub_comm _ _).le) scale.le).trans bound
  have total : Encloses 61 (taylorLoop (imul phi phi) start odd nterms).sum
      (partialSum odd x 13) := sum
  constructor
  · show (((taylorLoop (imul phi phi) start odd nterms).sum.1 - rem : ℤ) : ℝ) ≤ 2 ^ 61 * f
    push_cast
    linarith [total.1]
  · show 2 ^ 61 * f ≤ (((taylorLoop (imul phi phi) start odd nterms).sum.2 + rem : ℤ) : ℝ)
    push_cast
    linarith [total.2]

private theorem pi_encloses : Encloses 61 piEnclosure Real.pi ∧ 0 ≤ piEnclosure.1 := by
  rw [piEnclosure_eq]
  refine ⟨⟨?_, ?_⟩, by norm_num⟩
  · have := Real.pi_gt_d20
    norm_num at this ⊢
    linarith
  · have := Real.pi_lt_d20
    norm_num at this ⊢
    linarith

private theorem phi_encloses {d r : ℕ} (positive : 0 < d) :
    Encloses 61 (fdiv (piEnclosure.1 * r) d, cdiv (piEnclosure.2 * r) d) (Real.pi * r / d) ∧
      0 ≤ fdiv (piEnclosure.1 * r) d := by
  obtain ⟨⟨low, high⟩, nonneg⟩ := pi_encloses
  have d₀ : (0 : ℝ) < d := Nat.cast_pos.mpr positive
  have dz : (0 : ℤ) < d := by exact_mod_cast positive
  have r₀ : (0 : ℝ) ≤ r := Nat.cast_nonneg r
  refine ⟨⟨?_, ?_⟩, Int.ediv_nonneg (mul_nonneg nonneg (Int.natCast_nonneg r)) dz.le⟩
  · refine (fdiv_le _ dz).trans ?_
    push_cast
    rw [div_le_iff₀ d₀]
    calc (piEnclosure.1 : ℝ) * r ≤ 2 ^ 61 * Real.pi * r := mul_le_mul_of_nonneg_right low r₀
      _ = 2 ^ 61 * (Real.pi * r / d) * d := by field_simp
  · refine le_trans ?_ (le_cdiv _ dz)
    push_cast
    rw [le_div_iff₀ d₀]
    calc 2 ^ 61 * (Real.pi * r / d) * d = 2 ^ 61 * Real.pi * r := by field_simp
      _ ≤ (piEnclosure.2 : ℝ) * r := mul_le_mul_of_nonneg_right high r₀

private theorem firstOctant_encloses {d r : ℕ} (positive : 0 < d) (quarter : 4 * r ≤ d) :
    Encloses 61 (firstOctant d r).1 (Real.cos (Real.pi * r / d)) ∧
      Encloses 61 (firstOctant d r).2 (Real.sin (Real.pi * r / d)) := by
  obtain ⟨phi, phiNonneg⟩ := phi_encloses (r := r) positive
  have d₀ : (0 : ℝ) < d := Nat.cast_pos.mpr positive
  have x₀ : 0 ≤ Real.pi * r / d := by positivity
  have x₁ : Real.pi * r / d ≤ 1 := by
    rw [div_le_one d₀]
    have : (4 * r : ℝ) ≤ d := by exact_mod_cast quarter
    nlinarith [Real.pi_le_four, Real.pi_pos, (Nat.cast_nonneg r : (0 : ℝ) ≤ r)]
  have unit : |Real.pi * r / d| ≤ 1 := by rwa [abs_of_nonneg x₀]
  have start : Encloses 61 (one, one) ((Real.pi * r / d) ^ 0) := by
    constructor <;> norm_num [one]
  exact ⟨taylor_encloses phi phiNonneg (by norm_num) start (by norm_num [one]) unit
      (cos_close unit),
    taylor_encloses phi phiNonneg le_rfl (phi.of_eq (pow_one _).symm) phiNonneg unit
      (sin_close unit)⟩

private theorem angle_full {d t : ℕ} (positive : 0 < d) (small : t ≤ 2 * d) :
    Real.pi * t / d = 2 * Real.pi - Real.pi * (2 * d - t : ℕ) / d := by
  have d₀ : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr positive.ne'
  rw [Nat.cast_sub small]
  push_cast
  field_simp
  ring

private theorem angle_half {d t : ℕ} (positive : 0 < d) (small : t ≤ d) :
    Real.pi * t / d = Real.pi - Real.pi * (d - t : ℕ) / d := by
  have d₀ : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr positive.ne'
  rw [Nat.cast_sub small]
  field_simp
  ring

private theorem angle_quarter {m t : ℕ} (positive : 0 < m) (small : t ≤ m) :
    Real.pi * t / (2 * m : ℕ) = Real.pi / 2 - Real.pi * (m - t : ℕ) / (2 * m : ℕ) := by
  have m₀ : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr positive.ne'
  rw [Nat.cast_sub small]
  push_cast
  field_simp
  ring

private theorem octant_spec {t d : ℕ} (even : 2 ∣ d) (small : t < 2 * d) :
    4 * (octantReduce t d).index ≤ d ∧
      ((octantReduce t d).cosSign = 1 ∨ (octantReduce t d).cosSign = -1) ∧
      ((octantReduce t d).sinSign = 1 ∨ (octantReduce t d).sinSign = -1) ∧
      Real.cos (Real.pi * t / d) = (octantReduce t d).cosSign *
        (if (octantReduce t d).swap then Real.sin (Real.pi * (octantReduce t d).index / d)
          else Real.cos (Real.pi * (octantReduce t d).index / d)) ∧
      Real.sin (Real.pi * t / d) = (octantReduce t d).sinSign *
        (if (octantReduce t d).swap then Real.cos (Real.pi * (octantReduce t d).index / d)
          else Real.sin (Real.pi * (octantReduce t d).index / d)) := by
  obtain ⟨m, rfl⟩ := even
  have positive : 0 < m := by omega
  have mod : t % (2 * (2 * m)) = t := Nat.mod_eq_of_lt small
  have half : 2 * m / 2 = m := by omega
  have quarter : 2 * m / 4 = m / 2 := by omega
  simp only [octantReduce, mod, half, quarter, decide_eq_true_eq]
  have first : Real.cos (Real.pi * t / (2 * m : ℕ)) =
        Real.cos (Real.pi * (if 2 * m < t then 2 * (2 * m) - t else t : ℕ) / (2 * m : ℕ)) ∧
      Real.sin (Real.pi * t / (2 * m : ℕ)) = (if 2 * m < t then -1 else 1 : ℤ) *
        Real.sin (Real.pi * (if 2 * m < t then 2 * (2 * m) - t else t : ℕ) / (2 * m : ℕ)) := by
    split_ifs with high
    · rw [angle_full (by omega) small.le, Real.cos_two_pi_sub, Real.sin_two_pi_sub]
      exact ⟨rfl, by push_cast; ring⟩
    · simp
  have bound₂ : (if 2 * m < t then 2 * (2 * m) - t else t) ≤ 2 * m := by split_ifs <;> omega
  generalize (if 2 * m < t then 2 * (2 * m) - t else t) = t₂ at first bound₂ ⊢
  have second : Real.cos (Real.pi * t₂ / (2 * m : ℕ)) = (if m < t₂ then -1 else 1 : ℤ) *
        Real.cos (Real.pi * (if m < t₂ then 2 * m - t₂ else t₂ : ℕ) / (2 * m : ℕ)) ∧
      Real.sin (Real.pi * t₂ / (2 * m : ℕ)) =
        Real.sin (Real.pi * (if m < t₂ then 2 * m - t₂ else t₂ : ℕ) / (2 * m : ℕ)) := by
    split_ifs with high
    · rw [angle_half (by omega) bound₂, Real.cos_pi_sub, Real.sin_pi_sub]
      exact ⟨by push_cast; ring, rfl⟩
    · simp
  have bound₃ : (if m < t₂ then 2 * m - t₂ else t₂) ≤ m := by split_ifs <;> omega
  generalize (if m < t₂ then 2 * m - t₂ else t₂) = t₃ at second bound₃ ⊢
  have third : Real.cos (Real.pi * t₃ / (2 * m : ℕ)) =
        (if m / 2 < t₃ then
            Real.sin (Real.pi * (if m / 2 < t₃ then m - t₃ else t₃ : ℕ) / (2 * m : ℕ))
          else Real.cos (Real.pi * (if m / 2 < t₃ then m - t₃ else t₃ : ℕ) / (2 * m : ℕ))) ∧
      Real.sin (Real.pi * t₃ / (2 * m : ℕ)) =
        (if m / 2 < t₃ then
            Real.cos (Real.pi * (if m / 2 < t₃ then m - t₃ else t₃ : ℕ) / (2 * m : ℕ))
          else Real.sin (Real.pi * (if m / 2 < t₃ then m - t₃ else t₃ : ℕ) / (2 * m : ℕ))) := by
    split_ifs with high
    · rw [angle_quarter positive bound₃, Real.cos_pi_div_two_sub, Real.sin_pi_div_two_sub]
      exact ⟨rfl, rfl⟩
    · exact ⟨rfl, rfl⟩
  refine ⟨by split_ifs <;> omega, by split_ifs <;> simp, by split_ifs <;> simp, ?_, ?_⟩
  · rw [first.1, second.1, third.1]
  · rw [first.2, second.2, third.2]

private theorem applySign_encloses {v : ℤ × ℤ} {x : ℝ} {sign : ℤ} (enclosed : Encloses 61 v x)
    (unit : sign = 1 ∨ sign = -1) : Encloses 61 (applySign v sign) (sign * x) := by
  rcases unit with rfl | rfl
  · simpa [applySign] using enclosed
  · simp only [applySign, show (-1 : ℤ) < 0 by norm_num, ↓reduceIte]
    constructor
    · push_cast
      linarith [enclosed.2]
    · push_cast
      linarith [enclosed.1]

private theorem downscale_close {v : ℤ × ℤ} {x : ℝ} {q : ℕ} (enclosed : Encloses 61 v x)
    (fine : q ≤ 61) :
    |((downscaleRound v (61 - q)).1 : ℝ) - 2 ^ q * x| ≤ (downscaleRound v (61 - q)).2 := by
  have split : (2 : ℝ) ^ 61 = 2 ^ q * 2 ^ (61 - q) := by rw [← pow_add, Nat.add_sub_cancel' fine]
  have scale : (0 : ℝ) < 2 ^ (61 - q) := by positivity
  have low : (fshr v.1 (61 - q) : ℝ) ≤ 2 ^ q * x := by
    refine (fshr_le _ _).trans ?_
    rw [div_le_iff₀ scale]
    calc (v.1 : ℝ) ≤ 2 ^ 61 * x := enclosed.1
      _ = 2 ^ q * x * 2 ^ (61 - q) := by rw [split]; ring
  have high : 2 ^ q * x ≤ (cshr v.2 (61 - q) : ℝ) := by
    refine le_trans ?_ (le_cshr _ _)
    rw [le_div_iff₀ scale]
    calc 2 ^ q * x * 2 ^ (61 - q) = 2 ^ 61 * x := by rw [split]; ring
      _ ≤ v.2 := enclosed.2
  have left := le_max_left (fdiv (fshr v.1 (61 - q) + cshr v.2 (61 - q)) 2 - fshr v.1 (61 - q))
    (cshr v.2 (61 - q) - fdiv (fshr v.1 (61 - q) + cshr v.2 (61 - q)) 2)
  have right := le_max_right (fdiv (fshr v.1 (61 - q) + cshr v.2 (61 - q)) 2 - fshr v.1 (61 - q))
    (cshr v.2 (61 - q) - fdiv (fshr v.1 (61 - q) + cshr v.2 (61 - q)) 2)
  have left' : ((fdiv (fshr v.1 (61 - q) + cshr v.2 (61 - q)) 2 - fshr v.1 (61 - q) : ℤ) : ℝ) ≤
      (downscaleRound v (61 - q)).2 := Int.cast_le.mpr left
  have right' : ((cshr v.2 (61 - q) - fdiv (fshr v.1 (61 - q) + cshr v.2 (61 - q)) 2 : ℤ) : ℝ) ≤
      (downscaleRound v (61 - q)).2 := Int.cast_le.mpr right
  push_cast at left' right'
  have entry : (downscaleRound v (61 - q)).1 = fdiv (fshr v.1 (61 - q) + cshr v.2 (61 - q)) 2 :=
    rfl
  rw [entry, abs_sub_le_iff]
  exact ⟨by linarith, by linarith⟩

private theorem foldl_max (f g : ℕ → ℤ) : ∀ (l : List ℕ) (e : ℤ),
    e ≤ l.foldl (fun e t => max (max e (f t)) (g t)) e ∧
      ∀ t ∈ l, f t ≤ l.foldl (fun e t => max (max e (f t)) (g t)) e ∧
        g t ≤ l.foldl (fun e t => max (max e (f t)) (g t)) e
  | [], e => ⟨le_rfl, fun _ member => by simp at member⟩
  | t :: l, e => by
    obtain ⟨start, each⟩ := foldl_max f g l (max (max e (f t)) (g t))
    rw [List.foldl_cons]
    refine ⟨(le_max_left _ _).trans ((le_max_left _ _).trans start), fun s member => ?_⟩
    rcases List.mem_cons.mp member with rfl | member
    · exact ⟨(le_max_right _ _).trans ((le_max_left _ _).trans start),
        (le_max_right _ _).trans start⟩
    · exact each s member

private theorem entry_close {d q t : ℕ} (even : 2 ∣ d) (fine : q ≤ 61) (small : t < 2 * d) :
    |((entry d q t).1.1 : ℝ) - 2 ^ q * Real.cos (Real.pi * t / d)| ≤ (entry d q t).1.2 ∧
      |((entry d q t).2.1 : ℝ) - 2 ^ q * Real.sin (Real.pi * t / d)| ≤ (entry d q t).2.2 := by
  obtain ⟨quarter, cosUnit, sinUnit, cosEq, sinEq⟩ := octant_spec even small
  obtain ⟨cosEnclosed, sinEnclosed⟩ := firstOctant_encloses (by omega) quarter
  have chosen : ∀ b : Bool,
      Encloses 61 (if b then (firstOctant d (octantReduce t d).index).2
          else (firstOctant d (octantReduce t d).index).1)
        (if b then Real.sin (Real.pi * (octantReduce t d).index / d)
          else Real.cos (Real.pi * (octantReduce t d).index / d)) ∧
      Encloses 61 (if b then (firstOctant d (octantReduce t d).index).1
          else (firstOctant d (octantReduce t d).index).2)
        (if b then Real.cos (Real.pi * (octantReduce t d).index / d)
          else Real.sin (Real.pi * (octantReduce t d).index / d)) := by
    intro b
    cases b
    · exact ⟨cosEnclosed, sinEnclosed⟩
    · exact ⟨sinEnclosed, cosEnclosed⟩
  rw [cosEq, sinEq]
  exact ⟨downscale_close (applySign_encloses (chosen _).1 cosUnit) fine,
    downscale_close (applySign_encloses (chosen _).2 sinUnit) fine⟩

end TrigBuild

namespace TrigTable

open TrigBuild

/-- The table `(base_cos, base_sin, eps_root)` of `build_tables d q` in the pinned Akita code
(`crates/akita-challenges/src/sampler/op_norm.rs`), at the scale `2^q`. -/
def code (d q : ℕ) : TrigTable where
  scale := 2 ^ q
  error := (epsRoot d q).toNat
  cos t := (entry d q t).1.1
  sin t := (entry d q t).2.1

/-- The tables of `build_tables d q` are accurate for an even degree `d` and `q ≤ 61`: the
enclosure of `π`, the interval operations, the Taylor sums with their remainder and the octant
reduction are sound (Akita Appendix H.1). -/
theorem code_accurate {d q : ℕ} (even : 2 ∣ d) (fine : q ≤ 61) : (code d q).Accurate d := by
  intro t small
  obtain ⟨cosClose, sinClose⟩ := entry_close even fine small
  obtain ⟨nonneg, each⟩ := foldl_max (fun t => (entry d q t).1.2) (fun t => (entry d q t).2.2)
    (List.range (2 * d)) 0
  obtain ⟨cosError, sinError⟩ := each t (List.mem_range.mpr small)
  have error : ((code d q).error : ℝ) = (epsRoot d q : ℝ) := by
    simp only [code]
    exact_mod_cast Int.toNat_of_nonneg nonneg
  rw [error]
  simp only [code]
  push_cast
  exact ⟨cosClose.trans (by exact_mod_cast cosError), sinClose.trans (by exact_mod_cast sinError)⟩

/-- The strict predicate `accept_strict_parts` of the pinned Akita code for the ring degree `d`
and the threshold `T`: the fixed-point predicate of `code d 48`, at the scale
`OP_NORM_PREDICATE_SCALE = 48` of `crates/akita-challenges/src/sampler/mod.rs`. -/
def codeAccepts (d : ℕ) [NeZero d] (threshold : ℕ) : Rq d → Prop :=
  (code d 48).Accepts threshold

/-- The predicate of the pinned Akita code is a filter with `Filtered` for each even degree. -/
theorem filtered_code {d : ℕ} [NeZero d] (even : 2 ∣ d) (threshold : ℕ) :
    Filtered threshold (codeAccepts d threshold) :=
  ⟨code d 48, Nat.two_pow_pos 48, code_accurate even (by norm_num), fun _ kept => kept⟩

end TrigTable

end NightstreamFPrime.Spec.Compression.Akita
