import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.Ring.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Abel

namespace CompressionLean.Lib

variable {K : Type} [CommRing K]

def cubeSum {M : Type} [AddCommMonoid M] : ℕ → (ℕ → M) → M
  | 0, t => t 0
  | n + 1, t => cubeSum n (fun i => t (2 * i)) + cubeSum n (fun i => t (2 * i + 1))

def bindLow (c : K) (t : ℕ → K) : ℕ → K :=
  fun i => (1 - c) * t (2 * i) + c * t (2 * i + 1)

def fold : List K → (ℕ → K) → ℕ → K
  | [], t => t
  | c :: rest, t => fold rest (bindLow c t)

def mle (point : List K) (t : ℕ → K) : K :=
  fold point t 0

/-- The eq weight of index `i` at `ρ`: `Π_t (ρ_t if bit t of i else 1 - ρ_t)`.
Coordinate `t` goes with bit `t` of `i`, low bit first, as in Rust's `eq_at`
(`shrink/mod.rs`) and `eq_table` (`field.rs`), whose entry `i` is this
weight. -/
def eqAt : List K → ℕ → K
  | [], _ => 1
  | r :: rest, i => (if i % 2 = 1 then r else 1 - r) * eqAt rest (i / 2)

def eqEval : List K → List K → K
  | r :: ρ, x :: xs => (r * x + (1 - r) * (1 - x)) * eqEval ρ xs
  | _, _ => 1

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

theorem map_cubeSum {M N : Type} [AddCommMonoid M] [AddCommMonoid N] (f : M →+ N) (n : ℕ)
    (t : ℕ → M) : f (cubeSum n t) = cubeSum n (fun i => f (t i)) := by
  induction n generalizing t with
  | zero => rfl
  | succ n ih => simp only [cubeSum, map_add, ih]

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
      simp only [even, odd, half, half', ite_true, show ¬ ((0 : ℕ) = 1) by omega, ite_false]
      rw [← cubeSum_add]
      congr 1
      funext i
      ring

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
        rw [even, odd, half, half', ite_eq_left rfl, ite_eq_right (by omega)]
        ring
      rw [bound, mle_mul_left, ih xs (by simpa using length)]
      rfl

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

theorem mle_append_top (ρ : List K) (τ : K) (t : ℕ → K) :
    mle (ρ ++ [τ]) t = (1 - τ) * mle ρ t + τ * mle ρ (fun y => t (y + 2 ^ ρ.length)) := by
  rw [mle_append]
  simp only [mle, fold, bindLow]
  rw [fold_shift ρ t 0, fold_shift ρ t 1]
  simp [mle]

theorem mle_append_split (p q : List K) (t : ℕ → K) :
    mle (p ++ q) t = mle q (fun j => mle p (fun y => t (y + 2 ^ p.length * j))) := by
  rw [mle_append]
  congr 1
  funext j
  exact fold_shift p t j

theorem mle_delta (q : List K) (c : K) :
    mle q (fun j => if j = 0 then c else 0) = c * (q.map fun x => 1 - x).prod := by
  induction q generalizing c with
  | nil => simp [mle, fold]
  | cons x rest ih =>
      rw [mle_cons]
      have bound : bindLow x (fun j => if j = 0 then c else 0) = fun j => if j = 0 then (1 - x) * c else 0 := by
        funext j
        simp only [bindLow]
        by_cases zero : j = 0
        · subst zero
          simp
        · simp [zero, show 2 * j ≠ 0 by omega]
      rw [bound, ih]
      simp only [List.map_cons, List.prod_cons]
      ring

theorem mle_low_support (p q : List K) (t : ℕ → K) (zero : ∀ i, 2 ^ p.length ≤ i → t i = 0) :
    mle (p ++ q) t = mle p t * (q.map fun x => 1 - x).prod := by
  rw [mle_append_split]
  have column : (fun j => mle p (fun y => t (y + 2 ^ p.length * j))) =
      fun j => if j = 0 then mle p t else 0 := by
    funext j
    split
    · rename_i h
      subst h
      simp
    · rename_i h
      rw [← mle_const p 0]
      apply mle_congr
      intro y _
      apply zero
      have one : 1 ≤ j := Nat.one_le_iff_ne_zero.mpr h
      calc 2 ^ p.length = 2 ^ p.length * 1 := (mul_one _).symm
        _ ≤ 2 ^ p.length * j := Nat.mul_le_mul_left _ one
        _ ≤ y + 2 ^ p.length * j := Nat.le_add_left _ _
  rw [column]
  exact mle_delta q (mle p t)

theorem mle_append_zeros (p : List K) (k : ℕ) (t : ℕ → K) :
    mle (p ++ List.replicate k 0) t = mle p t := by
  rw [mle_append]
  induction k generalizing p t with
  | zero => simp [mle, fold]
  | succ k ih =>
      rw [List.replicate_succ, mle_cons]
      have : bindLow 0 (fold p t) = fold (p ++ [0]) t := by rw [fold_append_single]
      rw [this, ih (p ++ [0]) t]
      rw [mle_append, mle_cons]
      simp [mle, fold, bindLow]

theorem cubeSum_eq_sum_range {M : Type} [AddCommMonoid M] (n : ℕ) (t : ℕ → M) :
    cubeSum n t = ∑ i ∈ Finset.range (2 ^ n), t i := by
  induction n generalizing t with
  | zero => simp [cubeSum]
  | succ n ih =>
      simp only [cubeSum, ih]
      rw [pow_succ, mul_comm]
      induction 2 ^ n with
      | zero => simp
      | succ k ihk =>
          rw [show 2 * (k + 1) = 2 * k + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
            Finset.sum_range_succ, Finset.sum_range_succ, ← ihk]
          abel

theorem mle_finset_sum {ι : Type} (p : List K) (T : Finset ι) (f : ι → ℕ → K) :
    mle p (fun x => ∑ j ∈ T, f j x) = ∑ j ∈ T, mle p (f j) := by
  classical
  induction T using Finset.induction_on with
  | empty => simpa using mle_const p 0
  | insert a T notin ih =>
      simp only [Finset.sum_insert notin]
      rw [mle_add, ih]

theorem mle_swap (p q : List K) (t : ℕ → ℕ → K) :
    mle p (fun l => mle q (fun b => t l b)) = mle q (fun b => mle p (fun l => t l b)) := by
  simp only [mle_eq_cubeSum, cubeSum_eq_sum_range, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro b _
  apply Finset.sum_congr rfl
  intro l _
  ring

end Lemmas

end CompressionLean.Lib
