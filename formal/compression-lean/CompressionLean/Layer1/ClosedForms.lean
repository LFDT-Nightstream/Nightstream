import CompressionLean.Layer1.Late

/-!
Owns the closed forms of `crates/neo-spartan/src/mle.rs` that the layer-1
verifier evaluates, each written as the Rust loop computes it and proved
equal to the extension of its table:

- `mle_identity`: `identity`, Horner from the top coordinate;
- `lessThan_eq`: `less_than`, the scan from the top coordinate;
- `step_eq`: `step`, the telescoping of `less_than` over segments;
- `mle_chi`: `chi`, one factor per coordinate of `r`, over any commutative
  ring; `chi_split`: in `K ⊗ Ext` its two parts are the extensions of the
  model's `χ_r` tables (`Statement.chi`);
- `laneSplit_head`, `laneSplit_tail`: `lane_split`, the head recurrence and
  the tail prefix sums, against the model's `laneSplit` of one slot.

`Weights.lean` owns `scaled_eq_point` (`scaled_eq`). `EvalK.lean` owns
`eval_k`. The transcription of the Rust loops into these definitions is by
hand. `KeyStructure.lean` reads `step` over the key's run bounds and slot
segments.
-/

namespace CompressionLean.Layer1

open CompressionLean.Lib

section Ring

variable {K : Type} [CommRing K]

/-- `mle.rs::identity`: `total = 2·total + x_t`, from the top coordinate down. -/
def identityValue (p : List K) : K := p.foldr (fun x total => 2 * total + x) 0

theorem mle_identity (p : List K) : mle p (fun x => (x : K)) = identityValue p := by
  induction p with
  | nil => simp [mle, fold, identityValue]
  | cons c rest ih =>
      rw [mle_cons]
      have bound : bindLow c (fun x => (x : K)) = fun i : ℕ => 2 * (i : K) + c := by
        funext i
        simp only [bindLow]
        push_cast
        ring
      rw [bound, mle_add, mle_mul_left, mle_const, ih]
      rfl

/-- The scan of `mle.rs::less_than` from coordinate `t - 1` down to `0`, on
`(total, prefix)`: at a one bit of `bound`, `total += prefix·(1 - x)` and
`prefix *= x`; at a zero bit, `prefix *= 1 - x`. -/
def lessThanScan (p : List K) (bound : ℕ) : ℕ → K × K → K × K
  | 0, acc => acc
  | t + 1, (total, pre) =>
      if bound / 2 ^ t % 2 = 1 then
        lessThanScan p bound t (total + pre * (1 - p.getD t 0), pre * p.getD t 0)
      else lessThanScan p bound t (total, pre * (1 - p.getD t 0))

/-- `mle.rs::less_than`: `1` when `bound` is past the cube, else the scan. -/
def lessThan (p : List K) (bound : ℕ) : K :=
  if bound / 2 ^ p.length ≠ 0 then 1 else (lessThanScan p bound p.length (0, 1)).1

/-- The table of `x < bound`. -/
def below (bound : ℕ) (x : ℕ) : K := if x < bound then 1 else 0

theorem lessThanScan_eq (p : List K) (bound : ℕ) :
    ∀ t, t ≤ p.length → ∀ total pre : K,
      (lessThanScan p bound t (total, pre)).1 = total + pre * mle (p.take t) (below (bound % 2 ^ t)) := by
  intro t
  induction t with
  | zero => intro _ total pre; simp [lessThanScan, mle, fold, below, Nat.mod_one]
  | succ t ih =>
      intro short total pre
      have split : p.take (t + 1) = p.take t ++ [p.getD t 0] := by
        rw [List.take_add_one, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega)]
        simp
      have length : (p.take t).length = t := by rw [List.length_take]; omega
      have modulus : bound % 2 ^ (t + 1) = bound % 2 ^ t + 2 ^ t * (bound / 2 ^ t % 2) := by
        rw [pow_succ, Nat.mod_mul]
      have small : bound % 2 ^ t < 2 ^ t := Nat.mod_lt _ (by positivity)
      simp only [lessThanScan]
      rw [split, mle_append_top, length]
      split_ifs with bit
      · rw [ih (by omega)]
        have low : mle (p.take t) (below (bound % 2 ^ (t + 1))) = 1 := by
          rw [← mle_const (p.take t) (1 : K)]
          apply mle_congr
          intro y lt
          rw [length] at lt
          simp only [below]
          rw [if_pos (by rw [modulus, bit]; omega)]
        have high : mle (p.take t) (fun y => below (bound % 2 ^ (t + 1)) (y + 2 ^ t)) =
            mle (p.take t) (below (bound % 2 ^ t)) := by
          apply mle_congr
          intro y _
          simp only [below, modulus, bit]
          congr 1
          apply propext
          omega
        rw [low, high]
        ring
      · rw [ih (by omega)]
        have bit0 : bound / 2 ^ t % 2 = 0 := by omega
        have low : mle (p.take t) (below (bound % 2 ^ (t + 1))) = mle (p.take t) (below (bound % 2 ^ t)) := by
          rw [modulus, bit0, mul_zero, add_zero]
        have high : mle (p.take t) (fun y => below (bound % 2 ^ (t + 1)) (y + 2 ^ t)) = 0 := by
          rw [← mle_const (p.take t) (0 : K)]
          apply mle_congr
          intro y _
          simp only [below]
          rw [if_neg (by rw [modulus, bit0]; omega)]
        rw [low, high]
        ring

theorem lessThan_eq (p : List K) (bound : ℕ) : lessThan p bound = mle p (below bound) := by
  unfold lessThan
  split_ifs with past
  · rw [← mle_const p (1 : K)]
    apply mle_congr
    intro y lt
    simp only [below]
    rw [if_pos]
    by_contra more
    exact past (Nat.div_eq_of_lt (by omega))
  · rw [lessThanScan_eq p bound p.length le_rfl, List.take_length, zero_add, one_mul]
    have small : bound < 2 ^ p.length := by
      by_contra more
      exact past (Nat.div_pos (by omega) (by positivity)).ne'
    rw [Nat.mod_eq_of_lt small]

/-- The stop of the first segment: the next segment's start, or `stop`. -/
def nextStart (stop : ℕ) (rest : List (ℕ × K)) : ℕ := (rest.head?.map Prod.fst).getD stop

/-- `mle.rs::step` over segments `(start, value)`: `upper = less_than(next start)`,
`total += value·(upper - below)`, `below = upper`. -/
def stepLoop (p : List K) (stop : ℕ) : List (ℕ × K) → K → K → K
  | [], _, total => total
  | (_, v) :: rest, low, total =>
      stepLoop p stop rest (lessThan p (nextStart stop rest)) (total + v * (lessThan p (nextStart stop rest) - low))

/-- `mle.rs::step`: `below` starts at the first start, or at `stop`. -/
def stepValue (p : List K) (stop : ℕ) (segments : List (ℕ × K)) : K :=
  stepLoop p stop segments (lessThan p (nextStart stop segments)) 0

/-- The step table: each segment's value on `[start, next start)`. -/
def stepTable (stop : ℕ) : List (ℕ × K) → ℕ → K
  | [], _ => 0
  | (a, v) :: rest, x => (if a ≤ x ∧ x < nextStart stop rest then v else 0) + stepTable stop rest x

/-- Each segment starts no later than it stops. -/
def Ordered (stop : ℕ) : List (ℕ × K) → Prop
  | [] => True
  | (a, _) :: rest => a ≤ nextStart stop rest ∧ Ordered stop rest

theorem stepLoop_eq (p : List K) (stop : ℕ) :
    ∀ (segments : List (ℕ × K)) (total : K), Ordered stop segments →
      stepLoop p stop segments (lessThan p (nextStart stop segments)) total =
        total + mle p (stepTable stop segments) := by
  intro segments
  induction segments with
  | nil => intro total _; simp [stepLoop, stepTable, mle_const]
  | cons head rest ih =>
      obtain ⟨a, v⟩ := head
      intro total ordered
      obtain ⟨le, ordered⟩ := ordered
      have first : nextStart stop ((a, v) :: rest) = a := rfl
      simp only [stepLoop, first]
      rw [ih _ ordered]
      have table : stepTable stop ((a, v) :: rest) =
          fun x => (v * below (nextStart stop rest) x - v * below a x) + stepTable stop rest x := by
        funext x
        simp only [stepTable, below]
        congr 1
        split_ifs <;> first | (exfalso; omega) | ring
      rw [table, mle_add, mle_sub, mle_mul_left, mle_mul_left, ← lessThan_eq, ← lessThan_eq]
      ring

theorem step_eq (p : List K) (stop : ℕ) (segments : List (ℕ × K)) (ordered : Ordered stop segments) :
    stepValue p stop segments = mle p (stepTable stop segments) := by
  rw [stepValue, stepLoop_eq p stop segments 0 ordered, zero_add]

/-- `mle.rs::chi`: one factor per coordinate of `r`; `eq(r_t, x_t) = (1 - x) + r·(2x - 1)`
at a point coordinate, `1 - r_t` past the point. -/
def chiValue : List K → List K → K
  | [], _ => 1
  | r :: rs, x :: xs => ((1 - x) + r * (2 * x - 1)) * chiValue rs xs
  | r :: rs, [] => (1 - r) * chiValue rs []

/-- `chi` is the extension of `χ_r` at a point of at most `r.length` coordinates. -/
theorem mle_chi (r p : List K) (short : p.length ≤ r.length) : mle p (eqAt r) = chiValue r p := by
  induction p generalizing r with
  | nil =>
      simp only [mle, fold]
      clear short
      induction r with
      | nil => rfl
      | cons r0 rs ih => simp [eqAt, chiValue, ih]
  | cons x xs ih =>
      obtain ⟨r0, rs, rfl⟩ := List.exists_cons_of_length_eq_add_one
        (show r.length = (r.length - 1) + 1 by simp at short; omega)
      rw [mle_cons]
      have bound : bindLow x (eqAt (r0 :: rs)) = fun i => ((1 - x) + r0 * (2 * x - 1)) * eqAt rs i := by
        funext i
        simp only [bindLow, eqAt]
        have even : (2 * i) % 2 = 0 := by omega
        have odd : (2 * i + 1) % 2 = 1 := by omega
        have half : (2 * i) / 2 = i := by omega
        have half' : (2 * i + 1) / 2 = i := by omega
        rw [even, odd, half, half', if_pos rfl, if_neg (by omega)]
        ring
      rw [bound, mle_mul_left, ih rs (by simp at short; omega)]
      rfl

end Ring

section Lanes

variable {K : Type} [Field K]

/-- `mle.rs::lane_split`, head, by distance `d` from the top lane: `H0[D-1] = τ_{D-1}`;
then `H0[l] = τ_l + ratio·H0[l+1] - [l + length < D]·ratio^length·τ_{l+length}`. -/
def headFrom (D length : ℕ) (ratio : K) (τ : ℕ → K) : ℕ → K
  | 0 => τ (D - 1)
  | d + 1 =>
      τ (D - 2 - d) + ratio * headFrom D length ratio τ d -
        (if D - 2 - d + length < D then ratio ^ length * τ (D - 2 - d + length) else 0)

/-- `H0[l]`. -/
def headValue (D length : ℕ) (ratio : K) (τ : ℕ → K) (l : ℕ) : K :=
  headFrom D length ratio τ (D - 1 - l)

/-- `mle.rs::lane_split`, tail: `prefix[l + length - D - 1]·ratio^(D - l)` with
`prefix[k] = Σ_{i ≤ k} τ_i·ratio^i`, zero when `l + length ≤ D`. -/
def tailValue (D length : ℕ) (ratio : K) (τ : ℕ → K) (l : ℕ) : K :=
  if D + 1 ≤ l + length then (∑ i ∈ Finset.range (l + length - D), τ i * ratio ^ i) * ratio ^ (D - l) else 0

theorem sum_range_ite_lt (a b : ℕ) (g : ℕ → K) :
    ∑ k ∈ Finset.range a, (if k < b then g k else 0) = ∑ k ∈ Finset.range (min a b), g k := by
  induction a with
  | zero => simp
  | succ a ih =>
      rw [Finset.sum_range_succ, ih]
      by_cases lt : a < b
      · rw [if_pos lt, min_eq_left (by omega), min_eq_left (by omega), Finset.sum_range_succ]
      · rw [if_neg lt, add_zero, min_eq_right (by omega), min_eq_right (by omega)]

/-- The head is the run's weights inside its own block. -/
theorem headValue_eq (D length : ℕ) (ratio : K) (τ : ℕ → K) (positive : 1 ≤ length) :
    ∀ l, l < D → headValue D length ratio τ l =
      ∑ k ∈ Finset.range length, (if l + k < D then ratio ^ k * τ (l + k) else 0) := by
  have main : ∀ d, d < D → headFrom D length ratio τ d =
      ∑ k ∈ Finset.range length, (if D - 1 - d + k < D then ratio ^ k * τ (D - 1 - d + k) else 0) := by
    intro d
    induction d with
    | zero =>
        intro _
        obtain ⟨m, rfl⟩ : ∃ m, length = m + 1 := ⟨length - 1, by omega⟩
        rw [Finset.sum_range_succ', Finset.sum_eq_zero (fun k _ => if_neg (by omega))]
        simp [headFrom, show D - 1 < D by omega]
    | succ d ih =>
        intro lt
        rw [headFrom, ih (by omega), Finset.mul_sum]
        have top : D - 1 - (d + 1) = D - 2 - d := by omega
        rw [top]
        -- `ratio·H0[l+1]` covers `k = 1, ..., length`; add `τ_l`, drop `k = length`.
        have shifted : ∑ k ∈ Finset.range length,
            ratio * (if D - 1 - d + k < D then ratio ^ k * τ (D - 1 - d + k) else 0) =
            ∑ k ∈ Finset.range length,
              (if D - 2 - d + (k + 1) < D then ratio ^ (k + 1) * τ (D - 2 - d + (k + 1)) else 0) := by
          apply Finset.sum_congr rfl
          intro k _
          rw [show D - 2 - d + (k + 1) = D - 1 - d + k by omega]
          split_ifs <;> ring
        rw [shifted]
        have extend := Finset.sum_range_succ'
          (fun k => if D - 2 - d + k < D then ratio ^ k * τ (D - 2 - d + k) else 0) length
        have last := Finset.sum_range_succ
          (fun k => if D - 2 - d + k < D then ratio ^ k * τ (D - 2 - d + k) else 0) length
        simp only [add_zero, pow_zero, one_mul, show D - 2 - d < D by omega, if_true] at extend last
        linear_combination last - extend
  intro l lt
  rw [headValue, main (D - 1 - l) (by omega), show D - 1 - (D - 1 - l) = l by omega]

/-- The weights slot `σ` puts on the lanes of block `block + part`, by lane. -/
theorem laneSplit_eq {F : Type} [Field F] [Algebra F K] (s : Shape) (S : Setup F) (τ : ℕ → K) (part σ : ℕ)
    (slot : σ < S.slots) :
    laneSplit s S τ part σ = ∑ l ∈ Finset.range s.D, τ l *
      (if s.D * S.slotBlock σ + S.slotLane σ ≤ s.D * (S.slotBlock σ + part) + l ∧
          s.D * (S.slotBlock σ + part) + l < s.D * S.slotBlock σ + S.slotLane σ + S.slotLength σ then
        algebraMap F K (S.slotRatio σ) ^ (s.D * (S.slotBlock σ + part) + l - (s.D * S.slotBlock σ + S.slotLane σ))
      else 0) := by
  simp [laneSplit, slot, runWeight, slotStart]

/-- `lane_split`'s head is the model's lane split of a slot at part `0`. -/
theorem laneSplit_head {F : Type} [Field F] [Algebra F K] (s : Shape) (S : Setup F) (τ : ℕ → K) (σ : ℕ)
    (slot : σ < S.slots) (lane : S.slotLane σ < s.D) (positive : 1 ≤ S.slotLength σ) :
    laneSplit s S τ 0 σ = headValue s.D (S.slotLength σ) (algebraMap F K (S.slotRatio σ)) τ (S.slotLane σ) := by
  rw [laneSplit_eq s S τ 0 σ slot, headValue_eq _ _ _ _ positive _ lane]
  simp only [add_zero]
  obtain ⟨m, hm⟩ : ∃ m, s.D = S.slotLane σ + m := ⟨s.D - S.slotLane σ, by omega⟩
  rw [show Finset.range s.D = Finset.range (S.slotLane σ + m) by rw [hm], Finset.sum_range_add,
    Finset.sum_eq_zero (fun l member => by rw [if_neg (by simp at member; omega), mul_zero]), zero_add]
  have left : ∀ k ∈ Finset.range m, τ (S.slotLane σ + k) *
      (if s.D * S.slotBlock σ + S.slotLane σ ≤ s.D * S.slotBlock σ + (S.slotLane σ + k) ∧
          s.D * S.slotBlock σ + (S.slotLane σ + k) < s.D * S.slotBlock σ + S.slotLane σ + S.slotLength σ then
        algebraMap F K (S.slotRatio σ) ^
          (s.D * S.slotBlock σ + (S.slotLane σ + k) - (s.D * S.slotBlock σ + S.slotLane σ))
      else 0) = if k < S.slotLength σ then algebraMap F K (S.slotRatio σ) ^ k * τ (S.slotLane σ + k) else 0 := by
    intro k _
    by_cases inside : k < S.slotLength σ
    · rw [if_pos ⟨by omega, by omega⟩, if_pos inside,
        show s.D * S.slotBlock σ + (S.slotLane σ + k) - (s.D * S.slotBlock σ + S.slotLane σ) = k by omega]
      ring
    · rw [if_neg (by omega), if_neg inside, mul_zero]
  rw [Finset.sum_congr rfl left, sum_range_ite_lt]
  have right : ∀ k ∈ Finset.range (S.slotLength σ),
      (if S.slotLane σ + k < s.D then algebraMap F K (S.slotRatio σ) ^ k * τ (S.slotLane σ + k) else 0) =
      if k < m then algebraMap F K (S.slotRatio σ) ^ k * τ (S.slotLane σ + k) else 0 := by
    intro k _
    congr 1
    apply propext
    omega
  rw [Finset.sum_congr rfl right, sum_range_ite_lt, min_comm]

/-- `lane_split`'s tail is the model's lane split of a slot at part `1`. -/
theorem laneSplit_tail {F : Type} [Field F] [Algebra F K] (s : Shape) (S : Setup F) (τ : ℕ → K) (σ : ℕ)
    (slot : σ < S.slots) (lane : S.slotLane σ < s.D) (length : S.slotLength σ ≤ s.D) :
    laneSplit s S τ 1 σ = tailValue s.D (S.slotLength σ) (algebraMap F K (S.slotRatio σ)) τ (S.slotLane σ) := by
  rw [laneSplit_eq s S τ 1 σ slot]
  have each : ∀ l ∈ Finset.range s.D, τ l *
      (if s.D * S.slotBlock σ + S.slotLane σ ≤ s.D * (S.slotBlock σ + 1) + l ∧
          s.D * (S.slotBlock σ + 1) + l < s.D * S.slotBlock σ + S.slotLane σ + S.slotLength σ then
        algebraMap F K (S.slotRatio σ) ^ (s.D * (S.slotBlock σ + 1) + l - (s.D * S.slotBlock σ + S.slotLane σ))
      else 0) =
      if l < S.slotLane σ + S.slotLength σ - s.D then
        τ l * algebraMap F K (S.slotRatio σ) ^ l * algebraMap F K (S.slotRatio σ) ^ (s.D - S.slotLane σ)
      else 0 := by
    intro l _
    have expand : s.D * (S.slotBlock σ + 1) = s.D * S.slotBlock σ + s.D := by ring
    by_cases inside : l < S.slotLane σ + S.slotLength σ - s.D
    · rw [if_pos ⟨by rw [expand]; omega, by rw [expand]; omega⟩, if_pos inside, expand,
        show s.D * S.slotBlock σ + s.D + l - (s.D * S.slotBlock σ + S.slotLane σ) = l + (s.D - S.slotLane σ) by
          omega, pow_add]
      ring
    · rw [if_neg (by rw [expand]; omega), if_neg inside, mul_zero]
  rw [Finset.sum_congr rfl each, sum_range_ite_lt, min_eq_right (by omega), tailValue]
  split_ifs with long
  · rw [Finset.sum_mul]
  · rw [show S.slotLane σ + S.slotLength σ - s.D = 0 by omega, Finset.sum_range_zero]

end Lanes

section Split

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

/-- `K ⊗ Ext = Ext[u]/(u² - 7)`, the ring of `circuit/algebra.rs::Kx`. -/
abbrev QuadX (K : Type) [Field K] := QuadraticAlgebra K 7 0

/-- `Kx::from_k`: a `K` value in `K ⊗ Ext`, one coordinate at a time. -/
def fromK : Quad F →+* QuadX K where
  toFun z := ⟨ι z.re, ι z.im⟩
  map_one' := by ext <;> simp [QuadraticAlgebra.re_one, QuadraticAlgebra.im_one]
  map_mul' z w := by ext <;> simp [show ι 7 = 7 from map_ofNat ι 7]
  map_zero' := by ext <;> simp
  map_add' z w := by ext <;> simp

@[simp] theorem fromK_re (z : Quad F) : (fromK (K := K) z).re = ι z.re := rfl
@[simp] theorem fromK_im (z : Quad F) : (fromK (K := K) z).im = ι z.im := rfl

theorem quadX_re_sum {α : Type} (u : Finset α) (f : α → QuadX K) : (∑ x ∈ u, f x).re = ∑ x ∈ u, (f x).re :=
  map_sum (QuadraticAlgebra.reₗ (R := K) 7 0) f u

theorem quadX_im_sum {α : Type} (u : Finset α) (f : α → QuadX K) : (∑ x ∈ u, f x).im = ∑ x ∈ u, (f x).im :=
  map_sum (QuadraticAlgebra.imₗ (R := K) 7 0) f u

theorem eqAt_map {A B : Type} [CommRing A] [CommRing B] (f : A →+* B) (ρ : List A) (i : ℕ) :
    eqAt (ρ.map f) i = f (eqAt ρ i) := by
  induction ρ generalizing i with
  | nil => simp [eqAt]
  | cons r rest ih =>
      simp only [List.map_cons, eqAt, ih, map_mul]
      split_ifs <;> simp

/-- The extension of a `K ⊗ Ext` table at an `Ext` point is the pair of the
extensions of its parts. -/
theorem mle_split (p : List K) (T : ℕ → QuadX K) :
    mle (p.map (algebraMap K (QuadX K))) T = ⟨mle p fun i => (T i).re, mle p fun i => (T i).im⟩ := by
  rw [mle_eq_cubeSum, mle_eq_cubeSum, mle_eq_cubeSum, List.length_map]
  have weight : ∀ i, eqAt (p.map (algebraMap K (QuadX K))) i * T i =
      ⟨eqAt p i * (T i).re, eqAt p i * (T i).im⟩ := by
    intro i
    rw [eqAt_map]
    ext <;> simp [QuadraticAlgebra.algebraMap_eq]
  simp only [weight]
  ext
  · exact map_cubeSum (QuadraticAlgebra.reₗ (R := K) (7 : K) 0).toAddMonoidHom _ _
  · exact map_cubeSum (QuadraticAlgebra.imₗ (R := K) (7 : K) 0).toAddMonoidHom _ _

/-- `mle.rs::chi` in `K ⊗ Ext`, at `r` and an `Ext` point, is the pair of the
extensions of the model's `χ_r` tables. -/
theorem chi_split (st : Statement F) (p : List K) (short : p.length ≤ st.r.length) :
    chiValue (st.point.map fromK) (p.map (algebraMap K (QuadX K))) =
      ⟨mle p fun i => ι (st.chi i).1, mle p fun i => ι (st.chi i).2⟩ := by
  rw [← mle_chi _ _ (by simpa [Statement.point] using short),
    show eqAt (st.point.map fromK) = fun i => fromK (eqAt st.point i) from funext (eqAt_map (fromK (K := K)) st.point),
    mle_split]
  rfl

end Split

end CompressionLean.Layer1
