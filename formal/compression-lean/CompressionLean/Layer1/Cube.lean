import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Tactic.Ring

/-!
Owns tables on the Boolean cube and their multilinear extensions, in the
layer-1 order: index bit `t` is variable `t`, low bit first, as in
`crates/neo-spartan` (`fold_low`, `eq_table`).

- `bindLow c t` binds the low variable of `t` to `c`; `fold point t` binds the
  low variables in order. `mle point t = fold point t 0`.
- `cubeSum n t` sums `t` over the `n`-cube, low bit first.
- `eqAt ρ i` is the eq weight of index `i`; `eqEval ρ x` is its extension.
-/

namespace CompressionLean.Layer1

variable {K : Type} [CommRing K]

/-- The sum of a table over the `n`-cube: even indices, then odd indices. -/
def cubeSum {M : Type} [AddCommMonoid M] : ℕ → (ℕ → M) → M
  | 0, t => t 0
  | n + 1, t => cubeSum n (fun i => t (2 * i)) + cubeSum n (fun i => t (2 * i + 1))

/-- The table with its low variable bound to `c`. -/
def bindLow (c : K) (t : ℕ → K) : ℕ → K :=
  fun i => (1 - c) * t (2 * i) + c * t (2 * i + 1)

/-- The table with its low variables bound to `point`, in order. -/
def fold : List K → (ℕ → K) → ℕ → K
  | [], t => t
  | c :: rest, t => fold rest (bindLow c t)

/-- The multilinear extension of a table at a point, low variable first. -/
def mle (point : List K) (t : ℕ → K) : K :=
  fold point t 0

/-- The eq weight of index `i` at `ρ`: `Π_t (ρ_t if bit t of i else 1 - ρ_t)`. -/
def eqAt : List K → ℕ → K
  | [], _ => 1
  | r :: rest, i => (if i % 2 = 1 then r else 1 - r) * eqAt rest (i / 2)

/-- `eq(ρ, x) = Π_t (ρ_t·x_t + (1 - ρ_t)·(1 - x_t))`. -/
def eqEval : List K → List K → K
  | r :: ρ, x :: xs => (r * x + (1 - r) * (1 - x)) * eqEval ρ xs
  | _, _ => 1

/-- The point of index `i` on the `n`-cube, low bit first. -/
def boolPoint : ℕ → ℕ → List K
  | 0, _ => []
  | n + 1, i => (if i % 2 = 1 then 1 else 0) :: boolPoint n (i / 2)

section Lemmas

theorem cubeSum_add {M : Type} [AddCommMonoid M] (n : ℕ) (t u : ℕ → M) :
    cubeSum n (fun i => t i + u i) = cubeSum n t + cubeSum n u := by
  induction n generalizing t u with
  | zero => rfl
  | succ n ih =>
      simp only [cubeSum]
      rw [ih (fun i => t (2 * i)), ih (fun i => t (2 * i + 1)), add_add_add_comm]

theorem cubeSum_zero {M : Type} [AddCommMonoid M] (n : ℕ) : cubeSum n (fun _ => (0 : M)) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [cubeSum, ih, add_zero]

theorem cubeSum_mul_left (n : ℕ) (a : K) (t : ℕ → K) :
    cubeSum n (fun i => a * t i) = a * cubeSum n t := by
  induction n generalizing t with
  | zero => rfl
  | succ n ih =>
      simp only [cubeSum]
      rw [ih (fun i => t (2 * i)), ih (fun i => t (2 * i + 1)), mul_add]

theorem cubeSum_congr {M : Type} [AddCommMonoid M] (n : ℕ) {t u : ℕ → M}
    (equal : ∀ i, i < 2 ^ n → t i = u i) : cubeSum n t = cubeSum n u := by
  induction n generalizing t u with
  | zero => exact equal 0 (by norm_num)
  | succ n ih =>
      simp only [cubeSum]
      congr 1
      · exact ih fun i lt => equal _ (by rw [pow_succ]; omega)
      · exact ih fun i lt => equal _ (by rw [pow_succ]; omega)

/-- A homomorphism of additive monoids passes through the cube sum. -/
theorem map_cubeSum {M N : Type} [AddCommMonoid M] [AddCommMonoid N] (f : M →+ N) (n : ℕ)
    (t : ℕ → M) : f (cubeSum n t) = cubeSum n (fun i => f (t i)) := by
  induction n generalizing t with
  | zero => rfl
  | succ n ih => simp only [cubeSum, map_add, ih]

/-- Binding the low variable commutes with the sum over the other variables. -/
theorem cubeSum_bindLow (n : ℕ) (c : K) (t : ℕ → K) :
    cubeSum n (bindLow c t) =
      (1 - c) * cubeSum n (fun i => t (2 * i)) + c * cubeSum n (fun i => t (2 * i + 1)) := by
  unfold bindLow
  rw [cubeSum_add, cubeSum_mul_left, cubeSum_mul_left]

theorem fold_append (p q : List K) (t : ℕ → K) : fold (p ++ q) t = fold q (fold p t) := by
  induction p generalizing t with
  | nil => rfl
  | cons c rest ih => exact ih (bindLow c t)

theorem fold_append_single (p : List K) (c : K) (t : ℕ → K) :
    fold (p ++ [c]) t = bindLow c (fold p t) := by
  rw [fold_append]
  rfl

theorem mle_append (p q : List K) (t : ℕ → K) : mle (p ++ q) t = mle q (fold p t) := by
  unfold mle
  rw [fold_append]

theorem mle_cons (c : K) (rest : List K) (t : ℕ → K) :
    mle (c :: rest) t = mle rest (bindLow c t) := rfl

theorem bindLow_add (c : K) (t u : ℕ → K) :
    bindLow c (fun i => t i + u i) = fun i => bindLow c t i + bindLow c u i := by
  funext i
  simp only [bindLow]
  ring

theorem bindLow_mul_left (c a : K) (t : ℕ → K) :
    bindLow c (fun i => a * t i) = fun i => a * bindLow c t i := by
  funext i
  simp only [bindLow]
  ring

theorem bindLow_sub (c : K) (t u : ℕ → K) :
    bindLow c (fun i => t i - u i) = fun i => bindLow c t i - bindLow c u i := by
  funext i
  simp only [bindLow]
  ring

theorem fold_add (p : List K) (t u : ℕ → K) :
    fold p (fun i => t i + u i) = fun i => fold p t i + fold p u i := by
  induction p generalizing t u with
  | nil => rfl
  | cons c rest ih =>
      simp only [fold]
      rw [bindLow_add, ih]

theorem fold_mul_left (p : List K) (a : K) (t : ℕ → K) :
    fold p (fun i => a * t i) = fun i => a * fold p t i := by
  induction p generalizing t with
  | nil => rfl
  | cons c rest ih =>
      simp only [fold]
      rw [bindLow_mul_left, ih]

theorem fold_sub (p : List K) (t u : ℕ → K) :
    fold p (fun i => t i - u i) = fun i => fold p t i - fold p u i := by
  induction p generalizing t u with
  | nil => rfl
  | cons c rest ih =>
      simp only [fold]
      rw [bindLow_sub, ih]

theorem fold_const (p : List K) (a : K) : fold p (fun _ => a) = fun _ => a := by
  induction p with
  | nil => rfl
  | cons c rest ih =>
      simp only [fold]
      have : bindLow c (fun _ => a) = fun _ => a := by
        funext i
        simp only [bindLow]
        ring
      rw [this, ih]

theorem mle_add (p : List K) (t u : ℕ → K) : mle p (fun i => t i + u i) = mle p t + mle p u := by
  unfold mle
  rw [fold_add]

theorem mle_mul_left (p : List K) (a : K) (t : ℕ → K) : mle p (fun i => a * t i) = a * mle p t := by
  unfold mle
  rw [fold_mul_left]

theorem mle_sub (p : List K) (t u : ℕ → K) : mle p (fun i => t i - u i) = mle p t - mle p u := by
  unfold mle
  rw [fold_sub]

theorem mle_const (p : List K) (a : K) : mle p (fun _ => a) = a := by
  unfold mle
  rw [fold_const]

/-- Tables equal on the cube have equal extensions. -/
theorem fold_congr (p : List K) (n : ℕ) {t u : ℕ → K}
    (equal : ∀ i, i < 2 ^ (p.length + n) → t i = u i) :
    ∀ i, i < 2 ^ n → fold p t i = fold p u i := by
  induction p generalizing t u with
  | nil => exact fun i lt => equal i (by simpa using lt)
  | cons c rest ih =>
      simp only [fold]
      apply ih
      intro i lt
      simp only [bindLow]
      have bound : 2 ^ (rest.length + 1 + n) = 2 * 2 ^ (rest.length + n) := by
        rw [show rest.length + 1 + n = rest.length + n + 1 by omega, pow_succ]
        ring
      rw [List.length_cons] at equal
      rw [equal (2 * i) (by rw [bound]; omega), equal (2 * i + 1) (by rw [bound]; omega)]

theorem mle_congr (p : List K) {t u : ℕ → K} (equal : ∀ i, i < 2 ^ p.length → t i = u i) :
    mle p t = mle p u := by
  unfold mle
  exact fold_congr p 0 (by simpa using equal) 0 (by norm_num)

/-- The extension agrees with the table on the cube. -/
theorem mle_boolPoint (n i : ℕ) (t : ℕ → K) : mle (boolPoint n i) t = t (i % 2 ^ n) := by
  induction n generalizing i t with
  | zero => simp [boolPoint, mle, fold, Nat.mod_one]
  | succ n ih =>
      simp only [boolPoint, mle_cons]
      rw [ih]
      simp only [bindLow]
      have split : i % 2 ^ (n + 1) = i % 2 + 2 * (i / 2 % 2 ^ n) := by
        rw [pow_succ, mul_comm, Nat.mod_mul]
      rw [split]
      rcases Nat.mod_two_eq_zero_or_one i with parity | parity
      · rw [if_neg (by omega), parity, zero_add]
        ring
      · rw [if_pos parity, parity, add_comm 1]
        ring

/-- The extension is the eq-weighted sum of the table. -/
theorem mle_eq_cubeSum (point : List K) (t : ℕ → K) :
    mle point t = cubeSum point.length (fun i => eqAt point i * t i) := by
  induction point generalizing t with
  | nil => simp [mle, fold, cubeSum, eqAt]
  | cons r rest ih =>
      rw [mle_cons, ih, List.length_cons]
      simp only [cubeSum, bindLow, eqAt]
      have even (i : ℕ) : (2 * i) % 2 = 0 := by omega
      have odd (i : ℕ) : (2 * i + 1) % 2 = 1 := by omega
      have half (i : ℕ) : (2 * i) / 2 = i := by omega
      have half' (i : ℕ) : (2 * i + 1) / 2 = i := by omega
      simp only [even, odd, half, half', if_true, show ¬ ((0 : ℕ) = 1) by omega, if_false]
      rw [← cubeSum_add]
      congr 1
      funext i
      ring

/-- The extension of the eq table is the eq polynomial. -/
theorem mle_eqAt (ρ x : List K) (length : x.length = ρ.length) : mle x (eqAt ρ) = eqEval ρ x := by
  induction ρ generalizing x with
  | nil =>
      obtain rfl : x = [] := List.length_eq_zero_iff.mp (by simpa using length)
      simp [mle, fold, eqAt, eqEval]
  | cons r rest ih =>
      obtain ⟨c, xs, rfl⟩ := List.exists_cons_of_length_eq_add_one (by simpa using length)
      rw [mle_cons]
      have bound : bindLow c (eqAt (r :: rest)) =
          fun i => (r * c + (1 - r) * (1 - c)) * eqAt rest i := by
        funext i
        simp only [bindLow, eqAt]
        have even : (2 * i) % 2 = 0 := by omega
        have odd : (2 * i + 1) % 2 = 1 := by omega
        have half : (2 * i) / 2 = i := by omega
        have half' : (2 * i + 1) / 2 = i := by omega
        rw [even, odd, half, half', if_pos rfl, if_neg (by omega)]
        ring
      rw [bound, mle_mul_left, ih xs (by simpa using length)]
      rfl

/-- Folding `ρ` reads the table shifted by whole copies of the `ρ`-cube. -/
theorem fold_shift (ρ : List K) (t : ℕ → K) (i : ℕ) :
    fold ρ t i = mle ρ (fun y => t (y + 2 ^ ρ.length * i)) := by
  induction ρ generalizing t i with
  | nil => simp [fold, mle]
  | cons c rest ih =>
      simp only [fold, mle_cons, List.length_cons]
      rw [ih]
      congr 1
      funext y
      simp only [bindLow, pow_succ]
      congr 2 <;> congr 1 <;> ring

/-- A top variable splits the extension into the low and high halves. -/
theorem mle_append_top (ρ : List K) (τ : K) (t : ℕ → K) :
    mle (ρ ++ [τ]) t = (1 - τ) * mle ρ t + τ * mle ρ (fun y => t (y + 2 ^ ρ.length)) := by
  rw [mle_append]
  simp only [mle, fold, bindLow]
  rw [fold_shift ρ t 0, fold_shift ρ t 1]
  simp [mle]

end Lemmas

end CompressionLean.Layer1
