import CompressionLean.Layer1.ClosedForms
import CompressionLean.Layer1.Weights

namespace CompressionLean.Layer1

open CompressionLean.Lib

variable {A : Type} [CommRing A]

section Loop

variable (D blocks : ℕ)

def divStep (bb t rem below bit : ℕ) : Option (ℕ × ℕ × ℕ) :=
  if bb ≤ t ∧ (if D ≤ 2 * rem + bit then 1 else 0) = 1 then none
  else if below = 1 ∨ (if D ≤ 2 * rem + bit then 1 else 0) < blocks / 2 ^ t % 2 then
    some (2 * rem + bit - D * (if D ≤ 2 * rem + bit then 1 else 0), 1, if D ≤ 2 * rem + bit then 1 else 0)
  else if (if D ≤ 2 * rem + bit then 1 else 0) = blocks / 2 ^ t % 2 then
    some (2 * rem + bit - D * (if D ≤ 2 * rem + bit then 1 else 0), 0, if D ≤ 2 * rem + bit then 1 else 0)
  else none

def divFactor (r s : List A) (t bit q : ℕ) : A :=
  (if bit = 1 then r.getD t 0 else 1 - r.getD t 0) *
    (if t < s.length then (if q = 1 then s.getD t 0 else 1 - s.getD t 0) else 1)

def contribution (step : Option (ℕ × ℕ × ℕ)) (rem' below' : ℕ) (factor : ℕ → A) (x : A) : A :=
  match step with
  | some (r', b', q) => if r' = rem' ∧ b' = below' then x * factor q else 0
  | none => 0

def evalKStates (r s : List A) (cb : ℕ) : ℕ → ℕ → ℕ → A
  | 0 => fun rem below => if rem = 0 ∧ below = 0 then 1 else 0
  | k + 1 => fun rem' below' =>
      ∑ rem ∈ Finset.range D, ∑ below ∈ Finset.range 2, ∑ bit ∈ Finset.range 2,
        contribution (divStep D blocks s.length (cb - 1 - k) rem below bit) rem' below'
          (divFactor r s (cb - 1 - k) bit) (evalKStates r s cb k rem below)

def evalKValue (r s : List A) (τ : ℕ → A) (cb : ℕ) : A :=
  (∑ rem ∈ Finset.range D, evalKStates D blocks r s cb cb rem 1 * τ rem) *
    ((r.drop cb).map fun x => 1 - x).prod

end Loop

section Paths

variable (D blocks : ℕ) (r s : List A) (cb : ℕ)

def path : ℕ → ℕ → Option (ℕ × ℕ × A)
  | 0, _ => some (0, 0, 1)
  | k + 1, P =>
      match path k (P / 2) with
      | none => none
      | some (rem, below, w) =>
          match divStep D blocks s.length (cb - 1 - k) rem below (P % 2) with
          | none => none
          | some (rem', below', q) => some (rem', below', w * divFactor r s (cb - 1 - k) (P % 2) q)

def pathValue (k P rem below : ℕ) : A :=
  match path D blocks r s cb k P with
  | some (r', b', w) => if r' = rem ∧ b' = below then w else 0
  | none => 0

variable {D blocks}

theorem divStep_bounds (positive : 0 < D) {bb t rem below bit r' b' q : ℕ} (lt : rem < D) (digit : bit < 2)
    (step : divStep D blocks bb t rem below bit = some (r', b', q)) : r' < D ∧ b' < 2 := by
  unfold divStep at step
  split_ifs at step <;> simp only [Option.some.injEq, Prod.mk.injEq] at step <;> omega

theorem path_bounds (positive : 0 < D) :
    ∀ k P rem below w, path D blocks r s cb k P = some (rem, below, w) → rem < D ∧ below < 2 := by
  intro k
  induction k with
  | zero =>
      intro P rem below w equal
      simp only [path, Option.some.injEq, Prod.mk.injEq] at equal
      omega
  | succ k ih =>
      intro P rem below w equal
      simp only [path] at equal
      split at equal
      · cases equal
      · rename_i rem0 below0 w0 previous
        split at equal
        · cases equal
        · rename_i rem1 below1 q step
          simp only [Option.some.injEq, Prod.mk.injEq] at equal
          obtain ⟨rfl, rfl, -⟩ := equal
          exact divStep_bounds positive (ih _ _ _ _ previous).1 (Nat.mod_lt _ (by norm_num)) step

theorem contribution_sum {ι : Type} (T : Finset ι) (step : Option (ℕ × ℕ × ℕ)) (rem' below' : ℕ)
    (factor : ℕ → A) (x : ι → A) :
    contribution step rem' below' factor (∑ i ∈ T, x i) = ∑ i ∈ T, contribution step rem' below' factor (x i) := by
  unfold contribution
  rcases step with _ | ⟨r', b', q⟩
  · simp
  · dsimp only
    split_ifs
    · rw [Finset.sum_mul]
    · simp

theorem contribution_zero (step : Option (ℕ × ℕ × ℕ)) (rem' below' : ℕ) (factor : ℕ → A) :
    contribution step rem' below' factor 0 = 0 := by
  unfold contribution
  rcases step with _ | ⟨r', b', q⟩
  · rfl
  · dsimp only
    split_ifs <;> simp

theorem collapse (positive : 0 < D) (k P bit rem' below' : ℕ) (digit : bit < 2) :
    ∑ rem ∈ Finset.range D, ∑ below ∈ Finset.range 2,
      contribution (divStep D blocks s.length (cb - 1 - k) rem below bit) rem' below'
        (divFactor r s (cb - 1 - k) bit) (pathValue D blocks r s cb k P rem below) =
      pathValue D blocks r s cb (k + 1) (2 * P + bit) rem' below' := by
  have half : (2 * P + bit) / 2 = P := by omega
  have low : (2 * P + bit) % 2 = bit := by omega
  rcases equal : path D blocks r s cb k P with _ | ⟨rem0, below0, w⟩
  · have zero : ∀ rem below, pathValue D blocks r s cb k P rem below = 0 := by
      intro rem below
      simp [pathValue, equal]
    simp only [zero, contribution_zero, Finset.sum_const_zero]
    simp [pathValue, path, half, equal]
  · obtain ⟨remLt, belowLt⟩ := path_bounds r s cb positive k P rem0 below0 w equal
    have value : ∀ rem below, pathValue D blocks r s cb k P rem below =
        if rem0 = rem ∧ below0 = below then w else 0 := by
      intro rem below
      simp [pathValue, equal]
    simp only [value]
    rw [Finset.sum_eq_single rem0 (fun rem _ ne => Finset.sum_eq_zero fun below _ => by
        rw [ite_eq_right (fun h => ne h.1.symm), contribution_zero])
      (fun out => absurd (Finset.mem_range.mpr remLt) out),
      Finset.sum_eq_single below0 (fun below _ ne => by
        rw [ite_eq_right (fun h => ne h.2.symm), contribution_zero])
      (fun out => absurd (Finset.mem_range.mpr belowLt) out), ite_eq_left ⟨rfl, rfl⟩]
    simp only [pathValue, path, half, low, equal]
    unfold contribution
    rcases divStep D blocks s.length (cb - 1 - k) rem0 below0 bit with _ | ⟨r', b', q⟩ <;> rfl

theorem sum_swap4 {ι κ μ ν : Type} (S : Finset ι) (T : Finset κ) (U : Finset μ) (V : Finset ν)
    (f : ι → κ → μ → ν → A) :
    ∑ i ∈ S, ∑ j ∈ T, ∑ k ∈ U, ∑ l ∈ V, f i j k l = ∑ l ∈ V, ∑ k ∈ U, ∑ i ∈ S, ∑ j ∈ T, f i j k l := by
  calc ∑ i ∈ S, ∑ j ∈ T, ∑ k ∈ U, ∑ l ∈ V, f i j k l
      = ∑ i ∈ S, ∑ j ∈ T, ∑ l ∈ V, ∑ k ∈ U, f i j k l :=
        Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = ∑ i ∈ S, ∑ l ∈ V, ∑ j ∈ T, ∑ k ∈ U, f i j k l := Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = ∑ l ∈ V, ∑ i ∈ S, ∑ j ∈ T, ∑ k ∈ U, f i j k l := Finset.sum_comm
    _ = ∑ l ∈ V, ∑ k ∈ U, ∑ i ∈ S, ∑ j ∈ T, f i j k l :=
        Finset.sum_congr rfl fun _ _ => (Finset.sum_congr rfl fun _ _ => Finset.sum_comm).trans Finset.sum_comm

theorem states_eq_paths (positive : 0 < D) :
    ∀ k rem below, evalKStates D blocks r s cb k rem below =
      ∑ P ∈ Finset.range (2 ^ k), pathValue D blocks r s cb k P rem below := by
  intro k
  induction k with
  | zero =>
      intro rem below
      simp [evalKStates, pathValue, path, eq_comm]
  | succ k ih =>
      intro rem' below'
      simp only [evalKStates, ih, contribution_sum]
      rw [sum_swap4, pow_succ, mul_comm, sum_range_pairs]
      apply Finset.sum_congr rfl
      intro P _
      rw [Finset.sum_range_succ, Finset.sum_range_one, collapse r s cb positive k P 0 rem' below' (by norm_num),
        collapse r s cb positive k P 1 rem' below' (by norm_num), add_zero]

end Paths
section Meaning

variable {D blocks : ℕ} (r s : List A) (cb : ℕ)

theorem long_division (positive : 0 < D) (P bit : ℕ) (digit : bit < 2) :
    (2 * P + bit) / D = 2 * (P / D) + (if D ≤ 2 * (P % D) + bit then 1 else 0) ∧
      (2 * P + bit) % D = 2 * (P % D) + bit - D * (if D ≤ 2 * (P % D) + bit then 1 else 0) := by
  have split := Nat.div_add_mod P D
  have small := Nat.mod_lt P positive
  rw [Nat.div_mod_unique positive]
  by_cases carry : D ≤ 2 * (P % D) + bit
  · rw [ite_eq_left carry]
    have expand : D * (2 * (P / D) + 1) = 2 * (D * (P / D)) + D := by ring
    constructor <;> omega
  · rw [ite_eq_right carry]
    have expand : D * (2 * (P / D) + 0) = 2 * (D * (P / D)) := by ring
    constructor <;> omega

theorem weight_step (k t P Q bit q : ℕ) (inside : t + 1 ≤ r.length) (digit : bit < 2) (carry : q < 2) :
    eqAt ((r.drop (t + 1)).take k) P * eqAt (s.drop (t + 1)) Q * divFactor r s t bit q =
      eqAt ((r.drop t).take (k + 1)) (2 * P + bit) * eqAt (s.drop t) (2 * Q + q) := by
  have rGet : r.getD t 0 = r[t] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (show t < r.length by omega)]; rfl
  have rDrop : r.drop t = r[t] :: r.drop (t + 1) := List.drop_eq_getElem_cons (show t < r.length by omega)
  have half : (2 * P + bit) / 2 = P := by omega
  have low : (2 * P + bit) % 2 = bit := by omega
  have qHalf : (2 * Q + q) / 2 = Q := by omega
  have qLow : (2 * Q + q) % 2 = q := by omega
  have rStep : eqAt ((r.drop t).take (k + 1)) (2 * P + bit) =
      (if bit = 1 then r[t] else 1 - r[t]) * eqAt ((r.drop (t + 1)).take k) P := by
    rw [rDrop, List.take_succ_cons, eqAt, half, low]
  rw [rStep, divFactor, rGet]
  by_cases blockBit : t < s.length
  · have sGet : s.getD t 0 = s[t] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem blockBit]; rfl
    have sStep : eqAt (s.drop t) (2 * Q + q) = (if q = 1 then s[t] else 1 - s[t]) * eqAt (s.drop (t + 1)) Q := by
      rw [List.drop_eq_getElem_cons blockBit, eqAt, qHalf, qLow]
    rw [sStep, ite_eq_left blockBit, sGet]
    ring
  · have sNil : s.drop t = [] := List.drop_eq_nil_of_le (by omega)
    have sNil' : s.drop (t + 1) = [] := List.drop_eq_nil_of_le (by omega)
    rw [sNil, sNil', ite_eq_right blockBit]
    simp only [eqAt]
    ring

theorem blocks_digit (t : ℕ) : blocks / 2 ^ t = 2 * (blocks / 2 ^ (t + 1)) + blocks / 2 ^ t % 2 := by
  have := Nat.div_add_mod (blocks / 2 ^ t) 2
  rw [pow_succ, ← Nat.div_div_eq_div_mul]
  omega

theorem divStep_eq (bb t R bit Q : ℕ)
    (prefixLe : Q ≤ blocks / 2 ^ (t + 1)) (prefixLt : Q < 2 ^ (bb - (t + 1))) :
    divStep D blocks bb t R (if Q < blocks / 2 ^ (t + 1) then 1 else 0) bit =
      if 2 * Q + (if D ≤ 2 * R + bit then 1 else 0) ≤ blocks / 2 ^ t ∧
          2 * Q + (if D ≤ 2 * R + bit then 1 else 0) < 2 ^ (bb - t) then
        some (2 * R + bit - D * (if D ≤ 2 * R + bit then 1 else 0),
          if 2 * Q + (if D ≤ 2 * R + bit then 1 else 0) < blocks / 2 ^ t then 1 else 0,
          if D ≤ 2 * R + bit then 1 else 0)
      else none := by
  have bounds := blocks_digit (blocks := blocks) t
  have cSmall : blocks / 2 ^ t % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have exponent : (t + 1 ≤ bb ∧ 2 ^ (bb - t) = 2 * 2 ^ (bb - (t + 1))) ∨
      (bb ≤ t ∧ 2 ^ (bb - (t + 1)) = 1 ∧ 2 ^ (bb - t) = 1) := by
    by_cases inside : t + 1 ≤ bb
    · left
      refine ⟨inside, ?_⟩
      rw [show bb - t = bb - (t + 1) + 1 by omega, pow_succ]
      ring
    · right
      refine ⟨by omega, ?_, ?_⟩
      · rw [show bb - (t + 1) = 0 by omega, pow_zero]
      · rw [show bb - t = 0 by omega, pow_zero]
  unfold divStep
  have qSmall : (if D ≤ 2 * R + bit then 1 else 0) < 2 := by split_ifs <;> norm_num
  generalize (if D ≤ 2 * R + bit then 1 else 0) = q at qSmall ⊢
  generalize blocks / 2 ^ t % 2 = c at bounds cSmall ⊢
  generalize blocks / 2 ^ t = B' at bounds ⊢
  generalize blocks / 2 ^ (t + 1) = B at bounds prefixLe ⊢
  generalize 2 ^ (bb - (t + 1)) = X at exponent prefixLt
  generalize 2 ^ (bb - t) = X' at exponent ⊢
  rcases exponent with ⟨inside, double⟩ | ⟨outside, Xone, X'one⟩ <;>
    split_ifs <;> (try simp only [false_or, true_or, not_true_eq_false, not_and_or, not_lt, not_le] at *) <;>
      (exfalso; omega)

theorem path_eq (positive : 0 < D) (deep : s.length ≤ cb) (long : cb ≤ r.length) (strict : blocks < 2 ^ cb) :
    ∀ k, k ≤ cb → ∀ P, P < 2 ^ k →
      path D blocks r s cb k P =
        if P / D ≤ blocks / 2 ^ (cb - k) ∧ P / D < 2 ^ (s.length - (cb - k)) then
          some (P % D, if P / D < blocks / 2 ^ (cb - k) then 1 else 0,
            eqAt ((r.drop (cb - k)).take k) P * eqAt (s.drop (cb - k)) (P / D))
        else none := by
  intro k
  induction k with
  | zero =>
      intro _ P lt
      obtain rfl : P = 0 := by omega
      have none_above : blocks / 2 ^ cb = 0 := Nat.div_eq_of_lt strict
      simp [path, none_above, List.drop_eq_nil_of_le deep, eqAt]
  | succ k ih =>
      intro fits P lt
      obtain ⟨P0, bit, digit, rfl⟩ : ∃ P0 bit, bit < 2 ∧ P = 2 * P0 + bit :=
        ⟨P / 2, P % 2, Nat.mod_lt _ (by norm_num), by omega⟩
      have half : (2 * P0 + bit) / 2 = P0 := by omega
      have low : (2 * P0 + bit) % 2 = bit := by omega
      have previous := ih (by omega) P0 (by rw [pow_succ] at lt; omega)
      obtain ⟨quotient, remainder⟩ := long_division positive P0 bit digit
      set t := cb - 1 - k with t_def
      have tEq : cb - (k + 1) = t := by omega
      have tPrev : cb - k = t + 1 := by omega
      rw [tPrev] at previous
      rw [tEq, quotient, remainder]
      simp only [path, half, low, previous, ← t_def]
      have bounds := blocks_digit (blocks := blocks) t
      have exponent : (t + 1 ≤ s.length ∧ 2 ^ (s.length - t) = 2 * 2 ^ (s.length - (t + 1))) ∨
          (s.length ≤ t ∧ 2 ^ (s.length - (t + 1)) = 1 ∧ 2 ^ (s.length - t) = 1) := by
        by_cases inside : t + 1 ≤ s.length
        · left
          refine ⟨inside, ?_⟩
          rw [show s.length - t = s.length - (t + 1) + 1 by omega, pow_succ]
          ring
        · right
          refine ⟨by omega, ?_, ?_⟩
          · rw [show s.length - (t + 1) = 0 by omega, pow_zero]
          · rw [show s.length - t = 0 by omega, pow_zero]
      have qSmall : (if D ≤ 2 * (P0 % D) + bit then 1 else 0) < 2 := by split_ifs <;> norm_num
      by_cases before : P0 / D ≤ blocks / 2 ^ (t + 1) ∧ P0 / D < 2 ^ (s.length - (t + 1))
      · rw [ite_eq_left before]
        dsimp only
        rw [divStep_eq s.length t (P0 % D) bit (P0 / D) before.1 before.2]
        have weight := weight_step r s k t P0 (P0 / D) bit (if D ≤ 2 * (P0 % D) + bit then 1 else 0)
          (by omega) digit qSmall
        generalize (if D ≤ 2 * (P0 % D) + bit then 1 else 0) = q at weight ⊢
        by_cases cond : 2 * (P0 / D) + q ≤ blocks / 2 ^ t ∧ 2 * (P0 / D) + q < 2 ^ (s.length - t)
        · rw [ite_eq_left cond, ite_eq_left cond]
          dsimp only
          rw [weight]
        · rw [ite_eq_right cond, ite_eq_right cond]
      · rw [ite_eq_right before]
        dsimp only
        rw [ite_eq_right]
        rintro ⟨le, lt'⟩
        apply before
        generalize (if D ≤ 2 * (P0 % D) + bit then 1 else 0) = q at qSmall le lt'
        rcases exponent with ⟨_, double⟩ | ⟨_, Xone, X'one⟩ <;> constructor <;> omega

end Meaning
section Value

variable {D blocks : ℕ}

theorem eqAt_append (l m : List A) (c : ℕ) : eqAt (l ++ m) c = eqAt l c * eqAt m (c / 2 ^ l.length) := by
  induction l generalizing c with
  | nil => simp [eqAt]
  | cons x rest ih =>
      simp only [List.cons_append, eqAt, ih, List.length_cons, pow_succ, Nat.div_div_eq_div_mul]
      rw [mul_comm 2 (2 ^ rest.length)]
      ring

theorem eqAt_zero (m : List A) : eqAt m 0 = (m.map fun x => 1 - x).prod := by
  induction m with
  | nil => rfl
  | cons x rest ih => simp [eqAt, ih]

theorem sum_blocks_lanes (g : ℕ → A) (positive : 0 < D) (cb : ℕ) (fits : D * blocks ≤ 2 ^ cb) :
    ∑ c ∈ Finset.range (2 ^ cb), (if c / D < blocks then g c else 0) =
      ∑ b ∈ Finset.range blocks, ∑ l ∈ Finset.range D, g (D * b + l) := by
  obtain ⟨extra, hextra⟩ : ∃ extra, 2 ^ cb = D * blocks + extra := ⟨2 ^ cb - D * blocks, by omega⟩
  rw [hextra, Finset.sum_range_add, Finset.sum_eq_zero (s := Finset.range extra) (fun c _ => by
    rw [ite_eq_right]
    rw [not_lt, Nat.le_div_iff_mul_le positive]
    nlinarith), add_zero]
  clear hextra fits
  induction blocks with
  | zero => simp
  | succ n ih =>
      rw [Nat.mul_succ, Finset.sum_range_add, Finset.sum_range_succ]
      have before : ∀ c ∈ Finset.range (D * n), (if c / D < n + 1 then g c else 0) = if c / D < n then g c else 0 := by
        intro c member
        have : c / D < n := (Nat.div_lt_iff_lt_mul positive).mpr (by simp at member; linarith)
        rw [ite_eq_left (by omega), ite_eq_left this]
      rw [Finset.sum_congr rfl before, ih]
      congr 1
      apply Finset.sum_congr rfl
      intro l member
      have lane : l < D := Finset.mem_range.mp member
      rw [ite_eq_left (by rw [Nat.mul_add_div positive, Nat.div_eq_of_lt lane]; omega)]

theorem evalKValue_eq (r s : List A) (τ : ℕ → A) (cb : ℕ) (positive : 0 < D) (deep : s.length ≤ cb)
    (long : cb ≤ r.length) (blocks_le : blocks ≤ 2 ^ s.length) (fits : D * blocks ≤ 2 ^ cb)
    (strict : blocks < 2 ^ cb) :
    evalKValue D blocks r s τ cb =
      ∑ b ∈ Finset.range blocks, eqAt s b * ∑ l ∈ Finset.range D, τ l * eqAt r (D * b + l) := by
  unfold evalKValue
  have states : ∀ rem, evalKStates D blocks r s cb cb rem 1 =
      ∑ c ∈ Finset.range (2 ^ cb),
        (if c / D < blocks ∧ c % D = rem then eqAt (r.take cb) c * eqAt s (c / D) else 0) := by
    intro rem
    rw [states_eq_paths r s cb positive cb rem 1]
    apply Finset.sum_congr rfl
    intro c member
    unfold pathValue
    rw [path_eq r s cb positive deep long strict cb le_rfl c (Finset.mem_range.mp member)]
    simp only [Nat.sub_self, pow_zero, Nat.div_one, List.drop_zero, Nat.sub_zero]
    by_cases inBlocks : c / D < blocks
    · rw [ite_eq_left ⟨inBlocks.le, lt_of_lt_of_le inBlocks blocks_le⟩]
      dsimp only
      rw [ite_eq_left inBlocks]
      by_cases same : c % D = rem
      · rw [ite_eq_left ⟨same, rfl⟩, ite_eq_left ⟨inBlocks, same⟩]
      · rw [ite_eq_right (fun h => same h.1), ite_eq_right (fun h => same h.2)]
    · by_cases valid : c / D ≤ blocks ∧ c / D < 2 ^ s.length
      · rw [ite_eq_left valid]
        dsimp only
        rw [ite_eq_right inBlocks, ite_eq_right (fun h : c / D < blocks ∧ c % D = rem => inBlocks h.1)]
        simp
      · rw [ite_eq_right valid, ite_eq_right (fun h : c / D < blocks ∧ c % D = rem => inBlocks h.1)]
  have collapse : ∀ c ∈ Finset.range (2 ^ cb),
      ∑ rem ∈ Finset.range D,
        (if c / D < blocks ∧ c % D = rem then eqAt (r.take cb) c * eqAt s (c / D) else 0) * τ rem =
      if c / D < blocks then τ (c % D) * (eqAt (r.take cb) c * eqAt s (c / D)) else 0 := by
    intro c _
    by_cases inBlocks : c / D < blocks
    · rw [ite_eq_left inBlocks, Finset.sum_eq_single (c % D) (fun rem _ ne => by
          rw [ite_eq_right (fun h => ne h.2.symm), zero_mul])
        (fun out => absurd (Finset.mem_range.mpr (Nat.mod_lt c positive)) out), ite_eq_left ⟨inBlocks, rfl⟩]
      ring
    · rw [ite_eq_right inBlocks]
      exact Finset.sum_eq_zero fun rem _ => by rw [ite_eq_right (fun h => inBlocks h.1), zero_mul]
  have inner : ∑ rem ∈ Finset.range D, evalKStates D blocks r s cb cb rem 1 * τ rem =
      ∑ c ∈ Finset.range (2 ^ cb),
        (if c / D < blocks then τ (c % D) * (eqAt (r.take cb) c * eqAt s (c / D)) else 0) := by
    simp only [states, Finset.sum_mul]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl collapse
  rw [inner, sum_blocks_lanes _ positive cb fits, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro b member
  rw [Finset.sum_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro l lane
  have b_lt : b < blocks := Finset.mem_range.mp member
  have l_lt : l < D := Finset.mem_range.mp lane
  have hdiv : (D * b + l) / D = b := by rw [Nat.mul_add_div positive, Nat.div_eq_of_lt l_lt, add_zero]
  have hmod : (D * b + l) % D = l := by rw [Nat.mul_add_mod, Nat.mod_eq_of_lt l_lt]
  have small : D * b + l < 2 ^ cb := by
    have : D * (b + 1) ≤ D * blocks := Nat.mul_le_mul_left _ b_lt
    rw [Nat.mul_succ] at this
    omega
  have hr : eqAt r (D * b + l) = eqAt (r.take cb) (D * b + l) * ((r.drop cb).map fun x => 1 - x).prod := by
    conv_lhs => rw [← List.take_append_drop cb r]
    rw [eqAt_append, List.length_take, min_eq_left long, Nat.div_eq_of_lt small, eqAt_zero]
  rw [hdiv, hmod, hr]
  ring

theorem mle_evalK (r s : List A) (τ : ℕ → A) (cb : ℕ) (positive : 0 < D) (deep : s.length ≤ cb)
    (long : cb ≤ r.length) (blocks_le : blocks ≤ 2 ^ s.length) (fits : D * blocks ≤ 2 ^ cb)
    (strict : blocks < 2 ^ cb) :
    mle s (fun b => if b < blocks then ∑ l ∈ Finset.range D, τ l * eqAt r (D * b + l) else 0) =
      evalKValue D blocks r s τ cb := by
  rw [evalKValue_eq r s τ cb positive deep long blocks_le fits strict, mle_eq_cubeSum, cubeSum_eq_sum_range]
  obtain ⟨extra, hextra⟩ : ∃ extra, 2 ^ s.length = blocks + extra := ⟨2 ^ s.length - blocks, by omega⟩
  rw [hextra, Finset.sum_range_add, Finset.sum_eq_zero (s := Finset.range extra) (fun b _ => by
    rw [ite_eq_right (by omega), mul_zero]), add_zero]
  exact Finset.sum_congr rfl fun b member => by rw [ite_eq_left (Finset.mem_range.mp member)]

end Value

section Split

variable {F K : Type} [Field F] [Field K] [Algebra F K]

theorem evalK_split (s : Shape) (bar : ℕ → ℕ → F) (st : Statement F) (zeta : K) (sb : List K) (cb : ℕ)
    (positive : 0 < s.D) (deep : sb.length ≤ cb) (long : cb ≤ st.r.length)
    (blocks_le : s.blocks ≤ 2 ^ sb.length) (fits : s.D * s.blocks ≤ 2 ^ cb) (strict : s.blocks < 2 ^ cb) :
    evalKValue s.D s.blocks (st.point.map fromK) (sb.map (algebraMap K (QuadX K)))
        (fun l => algebraMap K (QuadX K) (laneWeight s bar zeta l)) cb =
      ⟨mle sb (evalKTable s bar st zeta 0), mle sb (evalKTable s bar st zeta 1)⟩ := by
  rw [← mle_evalK (D := s.D) (blocks := s.blocks) _ _ _ cb positive (by simpa using deep)
      (by simpa [Statement.point] using long) (by simpa using blocks_le) fits strict, mle_split]
  congr 1 <;> apply mle_congr <;> intro b _ <;>
    simp only [evalKTable, ↓reduceIte, Nat.one_ne_zero] <;> split_ifs <;>
    simp [quadX_re_sum, quadX_im_sum, eqAt_map, Statement.chi]

end Split

end CompressionLean.Layer1
