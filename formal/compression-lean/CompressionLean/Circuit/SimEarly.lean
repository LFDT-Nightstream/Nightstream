import CompressionLean.Circuit.SimAlgebra
import CompressionLean.Circuit.Early
import CompressionLean.Final.Fields

/-!
Owns the simulation of the layer-1 verifier's early stage (`earlyStage` of
`Verifier.lean`, with the helpers of `Early.lean`) against the replay's
(`Replay/Stages.lean`, `Replay/Formulas.lean`).

- `Sim.lessThan`, `Sim.step`, `Sim.identity`, `Sim.chi`: the closed forms of
  `mle.rs` give the replay's values. The late stage reads them too.
- The circuit and the replay may add or multiply in different orders. Their
  values agree in the challenge field `Final.Cubic`, where `Ext.toK` is
  injective (`toK_inj`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

variable {F : Type} [CommRing F]

/-! ## Exact arithmetic -/

/-- `Final.Cubic` with the root of `x³ - x - 1` (`field.rs`). -/
@[reducible] private noncomputable def cubicField : Bridge.ChallengeField Final.Cubic where
  root := AdjoinRoot.root Final.cubic
  cubic := by
    have h : AdjoinRoot.mk Final.cubic (X ^ 3 - X - 1) = 0 := AdjoinRoot.mk_self
    simp only [map_sub, map_pow, AdjoinRoot.mk_X, map_one] at h
    linear_combination h
  card := Final.card_cubic

attribute [local instance] cubicField

/-- Values with one denotation in `Final.Cubic` are equal (`field.rs`). -/
private theorem toK_inj {a b : Ext} (h : a.toK (K := Final.Cubic) = b.toK) : a = b := by
  have z : (a - b).toK (K := Final.Cubic) = 0 := by rw [Replay.Ext.toK_sub, h, sub_self]
  exact Sim.ext_sub_eq_zero.mp (Replay.Ext.eq_zero_of_toK z)

/-- `K ⊗ Ext` values with one denotation are equal (`algebra.rs`). -/
private theorem IsKx.congr {c : Ctx F} {n : ℕ} {a : KxVar} {v v' : Replay.Kx} (h : IsKx c n a v)
    (e : v.toQ (K := Final.Cubic) = v'.toQ) : IsKx c n a v' := by
  have re : v.re = v'.re := toK_inj (by simpa using congrArg QuadraticAlgebra.re e)
  have im : v.im = v'.im := toK_inj (by simpa using congrArg QuadraticAlgebra.im e)
  exact ⟨re ▸ h.1, im ▸ h.2⟩

/-! ## `mle.rs` in the replay -/

/-- One step of `less_than` at coordinate `t` (`mle.rs:36-55`). -/
private def lessThanStep (point : Array Ext) (bound t : ℕ) (acc : Ext × Ext) : Ext × Ext :=
  if (bound / 2 ^ t) % 2 = 1 then (acc.1 + acc.2 * Replay.oneMinus point[t]!, acc.2 * point[t]!)
  else (acc.1, acc.2 * Replay.oneMinus point[t]!)

/-- The scan of `less_than` (`mle.rs:36-55`) is a fold from the top coordinate. -/
private theorem lessThanScan_eq (point : Array Ext) (bound : ℕ) : ∀ s acc,
    Replay.lessThanScan point bound s acc =
      (List.range s).reverse.foldl (fun acc t => lessThanStep point bound t acc) acc
  | 0, _ => rfl
  | s + 1, (total, pre) => by
      rw [Replay.lessThanScan, List.range_succ, List.reverse_append, List.reverse_singleton, List.singleton_append,
        List.foldl_cons, ← lessThanScan_eq point bound s]
      unfold lessThanStep
      split_ifs <;> rfl

/-- Term `s` of `step`: `values[s]` times its interval (`mle.rs:57-72`). -/
private def stepTerm (gpoint : Array Ext) (starts : Array ℕ) (gvalues : Array Ext) (stop s : ℕ) : Ext :=
  gvalues[s]! * (Replay.lessThan gpoint (starts[s + 1]?.getD stop) - Replay.lessThan gpoint (starts[s]?.getD stop))

/-- `step` (`mle.rs:57-72`) is the sum of its terms; a value past the starts
has an empty interval. -/
private theorem step_eq (gpoint : Array Ext) (starts : Array ℕ) (gvalues : Array Ext) (stop : ℕ)
    (size : starts.size ≤ gvalues.size) :
    Replay.step gpoint (starts.toList.zip gvalues.toList) stop =
      Replay.sumRange gvalues.size (stepTerm gpoint starts gvalues stop) := by
  set segs := starts.toList.zip gvalues.toList
  have len : segs.length = starts.size := by simp [segs, List.length_zip]; omega
  have next : ∀ j, Replay.nextStart stop (segs.drop j) = starts[j]?.getD stop := by
    intro j
    by_cases hj : j < starts.size
    · rw [List.drop_eq_getElem_cons (by omega)]
      simp [Replay.nextStart, segs, Array.getElem?_eq_getElem hj]
    · rw [List.drop_eq_nil_of_le (by omega)]
      simp [Replay.nextStart, Array.getElem?_eq_none (by omega : starts.size ≤ j)]
  have tail : ∀ d j, starts.size - j = d → j ≤ starts.size →
      (Replay.stepSum gpoint stop (segs.drop j) (Replay.lessThan gpoint (starts[j]?.getD stop))).toK
        (K := Final.Cubic) = ∑ s ∈ Finset.Ico j starts.size, (stepTerm gpoint starts gvalues stop s).toK := by
    intro d
    induction d with
    | zero =>
        intro j hd hj
        rw [List.drop_eq_nil_of_le (by omega), show j = starts.size by omega, Finset.Ico_self, Finset.sum_empty]
        exact Replay.Ext.toK_zero
    | succ d ih =>
        intro j hd hj
        have lt : j < segs.length := by omega
        rw [List.drop_eq_getElem_cons lt, List.getElem_zip, Replay.stepSum, next, Replay.Ext.toK_add,
          ih (j + 1) (by omega) (by omega), Finset.sum_eq_sum_Ico_succ_bot (show j < starts.size by omega),
          add_left_inj, stepTerm, Array.getElem_toList, getElem!_pos gvalues j (by omega)]
  have zero : ∀ s, starts.size ≤ s → (stepTerm gpoint starts gvalues stop s).toK (K := Final.Cubic) = 0 := by
    intro s hs
    simp only [stepTerm, Array.getElem?_eq_none (by omega : starts.size ≤ s + 1),
      Array.getElem?_eq_none (by omega : starts.size ≤ s), Option.getD_none, Replay.Ext.toK_mul, Replay.Ext.toK_sub,
      sub_self, mul_zero]
  apply toK_inj
  rw [Replay.step, ← List.drop_zero (l := segs), next, tail _ 0 rfl (Nat.zero_le _), Replay.toK_sumRange,
    ← Finset.sum_range_add_sum_Ico _ size, Finset.sum_eq_zero (fun s hs => zero s (Finset.mem_Ico.mp hs).1),
    add_zero, Finset.range_eq_Ico]

/-- Factor `t` of `χ_r`: at a point coordinate `x`, or past the point (`mle.rs:84-107`). -/
private def chiFactor (r : Gl × Gl) : Option Ext → Replay.Kx
  | some x => let scaled := (Replay.Kx.ofK r).scale (x + x - 1); { scaled with re := scaled.re + Replay.oneMinus x }
  | none => (Replay.Kx.ofK r).oneMinus

/-- `χ_r` (`mle.rs:84-107`) is the product of its factors. -/
private theorem toQ_chiList : ∀ (l : List (Gl × Gl)) (xs : List Ext),
    (Replay.chiList l xs).toQ (K := Final.Cubic) = ∏ t ∈ Finset.range l.length, (chiFactor l[t]! xs[t]?).toQ
  | [], _ => by simp [Replay.chiList]
  | r :: rs, [] => by
      simp only [Replay.chiList, Replay.Kx.toQ_mul, toQ_chiList rs [], List.length_cons, Finset.prod_range_succ',
        List.getElem!_cons_zero, List.getElem!_cons_succ, List.getElem?_nil]
      rw [mul_comm]
      rfl
  | r :: rs, x :: xs => by
      simp only [Replay.chiList, Replay.Kx.toQ_mul, toQ_chiList rs xs, List.length_cons, Finset.prod_range_succ',
        List.getElem!_cons_zero, List.getElem!_cons_succ, List.getElem?_cons_zero, List.getElem?_cons_succ]
      rw [mul_comm]
      rfl

/-- A running product from one (`mle.rs:84-107`). -/
private theorem toQ_foldl_mul (f : ℕ → Replay.Kx) (k : ℕ) :
    ((List.range k).foldl (fun v t => v.mul (f t)) Replay.Kx.one).toQ (K := Final.Cubic) =
      ∏ t ∈ Finset.range k, (f t).toQ := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Replay.Kx.toQ_mul, ih,
        Finset.prod_range_succ]

/-! ## `mle.rs` in the circuit -/

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

omit [CharP F goldilocksModulus] in
/-- A loop over a list whose steps check nothing, as the coordinate loop of
`less_than` (`mle.rs:36-55`). -/
private theorem loopList {δ γ : Type} {xs : List δ} {init : γ} {body : δ → γ → Build (ForInStep γ)}
    (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Sim c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') True) :
    Sim c n (P ∧ I 0 init n) (forIn xs init body) (fun r m => I xs.length r m) True :=
  (Sim.loop I (fun _ => True) step).post (fun _ _ _ h => h) fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩

/-- `less_than`: `Σ_{x < bound} eq(point, x)`, high bit first (`mle.rs:36-55`). -/
theorem lessThan {point : Array ExtVar} {gpoint : Array Ext} (bound : ℕ) (hp : P → IsExts c n point gpoint) :
    Sim c n P (Mle.lessThan point bound) (fun r m => IsExt c m r (Replay.lessThan gpoint bound)) True := by
  unfold Mle.lessThan
  split
  · rename_i big
    exact extOne.post (fun h _ _ e => by
      rw [Replay.lessThan, ← (hp h).1, ← Nat.shiftRight_eq_div_pow, if_pos big]
      exact e) fun _ => Iff.rfl
  · rename_i small
    set xs := (List.range point.size).reverse with hxs
    set G : ℕ → Ext × Ext := fun k => (xs.take k).foldl (fun acc t => lessThanStep gpoint bound t acc) (0, 1)
    refine extZero.bind₀ fun total m₁ l₁ => extOne.bind₀ fun head m₂ l₂ => ?_
    refine ((loopList (P := (P ∧ IsExt c m₁ total 0) ∧ IsExt c m₂ head 1)
      (fun k acc m => IsExt c m acc.1 (G k).1 ∧ IsExt c m acc.2 (G k).2) fun k hk acc m lm => ?_).pre
      fun h => ⟨h, h.1.2.mono l₂, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
    · have ht : xs[k] < point.size := by
        have mem : xs[k] ∈ (List.range point.size).reverse := List.getElem_mem hk
        rw [List.mem_reverse, List.mem_range] at mem
        exact mem
      have hx : P → IsExt c n point[xs[k]]! gpoint[xs[k]]! := fun h => (hp h).2 _ ht
      have L : n ≤ m := l₁.trans (l₂.trans lm)
      have G_succ : G (k + 1) = lessThanStep gpoint bound xs[k] (G k) := by
        simp only [G]
        rw [List.take_succ_eq_append_getElem hk, List.foldl_append, List.foldl_cons, List.foldl_nil]
      dsimp only
      refine (oneMinus fun h => (hx h.1.1.1).mono L).bind₀ fun low m₃ l₃ => ?_
      split
      · rename_i bit
        have cond : (bound / 2 ^ xs[k]) % 2 = 1 := by
          simpa [Nat.testBit_eq_decide_div_mod_eq] using bit
        refine (Sim.extMul (fun h => h.1.2.2.mono l₃) fun h => h.2).bind₀ fun term m₄ l₄ => ?_
        refine (extAdd (fun h => h.1.1.2.1.mono (l₃.trans l₄)) fun h => h.2).bind₀ fun total' m₅ l₅ => ?_
        refine (Sim.extMul (fun h => h.1.1.1.2.2.mono (l₃.trans (l₄.trans l₅)))
          fun h => (hx h.1.1.1.1.1.1).mono (L.trans (l₃.trans (l₄.trans l₅)))).bind₀ fun head' m₆ l₆ => ?_
        refine Sim.pure fun h => ⟨rfl, ?_, ?_⟩ <;> rw [G_succ, lessThanStep, if_pos cond]
        · exact h.1.2.mono l₆
        · exact h.2
      · rename_i bit
        have cond : ¬(bound / 2 ^ xs[k]) % 2 = 1 := by
          simpa [Nat.testBit_eq_decide_div_mod_eq] using bit
        refine (Sim.extMul (fun h => h.1.2.2.mono l₃) fun h => h.2).bind₀ fun head' m₄ l₄ => ?_
        refine Sim.pure fun h => ⟨rfl, ?_, ?_⟩ <;> rw [G_succ, lessThanStep, if_neg cond]
        · exact h.1.1.2.1.mono (l₃.trans l₄)
        · exact h.2
    · have e := h.2.1
      simp only [G, List.take_length] at e
      rw [Replay.lessThan, ← (hp h.1.1.1).1, if_neg (by rw [← Nat.shiftRight_eq_div_pow]; exact small),
        lessThanScan_eq]
      exact e

/-- `step`: `values[s]` on `[starts[s], starts[s + 1])`, the last segment
ending at `stop` (`mle.rs:57-72`). A value past the starts adds nothing. -/
theorem step {point : Array ExtVar} {gpoint : Array Ext} {starts : Array ℕ} {values : Array ExtVar}
    {gvalues : Array Ext} (stop : ℕ) (hp : P → IsExts c n point gpoint) (hv : P → IsExts c n values gvalues)
    (size : starts.size ≤ values.size) :
    Sim c n P (Mle.step point starts values stop)
      (fun r m => IsExt c m r (Replay.step gpoint (starts.toList.zip gvalues.toList) stop)) True := by
  set L : ℕ → Ext := fun s => Replay.lessThan gpoint (starts[s]?.getD stop)
  unfold Mle.step
  refine extZero.bind₀ fun total m₁ l₁ => ?_
  refine (lessThan _ fun h => (hp h.1).mono l₁).bind₀ fun below m₂ l₂ => ?_
  refine ((loop₀ (P := (P ∧ IsExt c m₁ total 0) ∧ IsExt c m₂ below (L 0))
    (fun k acc m => IsExt c m acc.1 (Replay.sumRange k (stepTerm gpoint starts gvalues stop)) ∧
      IsExt c m acc.2 (L k)) fun k hk acc m lm => ?_).pre
    fun h => ⟨h, h.1.2.mono l₂, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hk' : k < values.size := by simpa using hk
    have L₀ : n ≤ m := l₁.trans (l₂.trans lm)
    rw [show values.zipIdx[k] = (values[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (lessThan _ fun h => (hp h.1.1.1).mono L₀).bind₀ fun upper m₃ l₃ => ?_
    refine (extSub (fun h => h.2) fun h => h.1.2.2.mono l₃).bind₀ fun interval m₄ l₄ => ?_
    refine (Sim.extMul (fun h => ((hv h.1.1.1.1.1).2 k hk').mono (L₀.trans (l₃.trans l₄))) fun h => h.2).bind₀
      fun term m₅ l₅ => ?_
    refine (extAdd (fun h => h.1.1.1.2.1.mono (l₃.trans (l₄.trans l₅))) fun h => h.2).bind₀
      fun total' m₆ l₆ => ?_
    exact Sim.pure fun h => ⟨rfl, by rw [sumRange_succ]; exact h.2, h.1.1.1.2.mono (l₄.trans (l₅.trans l₆))⟩
  · have e := h.2.1
    rw [Array.size_zipIdx, (hv h.1.1.1).1] at e
    rw [step_eq gpoint starts gvalues stop (by rw [← (hv h.1.1.1).1]; exact size)]
    exact e

/-- `identity`: `Σ_t 2^t point_t` (`mle.rs:74-82`). -/
theorem identity {point : Array ExtVar} {gpoint : Array Ext} (hp : P → IsExts c n point gpoint) :
    Sim c n P (Mle.identity point) (fun r m => IsExt c m r (Replay.identityValue gpoint)) True := by
  unfold Mle.identity
  refine extZero.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 0)
    (fun k acc m => IsExt c m acc ((gpoint.reverse.toList.take k).foldl (fun t x => t + t + x) 0))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hx : P → IsExt c n point.reverse[k] gpoint.reverse[k]! := fun h => by
      have := (hp h).reverse.2 k hk
      rwa [getElem!_pos point.reverse k hk] at this
    refine (extAdd (fun h => h.2) fun h => h.2).bind₀ fun doubled m₂ l₂ => ?_
    refine (extAdd (fun h => h.2) fun h => (hx h.1.1.1).mono (l₁.trans (lm.trans l₂))).bind₀
      fun total' m₃ l₃ => ?_
    exact Sim.pure fun h => ⟨rfl, by
      have hk' : k < gpoint.reverse.toList.length := by simpa [← (hp h.1.1.1.1).1] using hk
      rw [List.take_succ_eq_append_getElem hk', List.foldl_append, List.foldl_cons, List.foldl_nil,
        Array.getElem_toList, ← getElem!_pos]
      exact h.2⟩
  · have e := h.2
    rw [List.take_of_length_le (by simp [(hp h.1.1).1])] at e
    rw [Replay.identityValue, Array.foldr_eq_foldl_reverse, ← Array.foldl_toList]
    exact e

/-- `chi`: `Π_{t<n} eq(r_t, point_t) · Π_{t≥n} (1 - r_t)` in `K ⊗ Ext`
(`mle.rs:84-107`). -/
theorem chi {r : Array KVar} {gr : Array (Gl × Gl)} {point : Array ExtVar} {gpoint : Array Ext}
    (hr : P → r.size = gr.size ∧ ∀ i < r.size, IsWord c n r[i]!.1 gr[i]!.1 ∧ IsWord c n r[i]!.2 gr[i]!.2)
    (hp : P → IsExts c n point gpoint) :
    Sim c n P (Mle.chi r point) (fun res m => IsKx c m res (Replay.chi gr gpoint)) True := by
  set V : ℕ → Replay.Kx := fun k =>
    (List.range k).foldl (fun v t => v.mul (chiFactor gr[t]! gpoint[t]?)) Replay.Kx.one
  unfold Mle.chi
  refine KxVar.one.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsKx c m₁ total Replay.Kx.one) (fun k acc m => IsKx c m acc (V k))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun res m _ => Sim.pure fun h => ?_
  · have hk' : k < r.size := by simpa using hk
    have L₀ : n ≤ m := l₁.trans lm
    have V_succ : V (k + 1) = (V k).mul (chiFactor gr[k]! gpoint[k]?) := by
      simp only [V]
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    rw [show r.zipIdx[k] = (r[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (KxVar.fromK (fun h => ((hr h.1.1).2 k hk').1.mono L₀) fun h => ((hr h.1.1).2 k hk').2.mono L₀).bind₀
      fun rt m₂ l₂ => ?_
    split
    · rename_i x hpt
      have inside : k < point.size := by
        rw [Array.getElem?_eq_some_iff] at hpt
        exact hpt.1
      have hx : P → IsExt c n x gpoint[k]! := fun h => by
        have := (hp h).2 k inside
        rwa [getElem!_pos point k inside, (Array.getElem?_eq_some_iff.mp hpt).2] at this
      have L₂ : n ≤ m₂ := L₀.trans l₂
      refine (extAdd (fun h => (hx h.1.1.1).mono L₂) fun h => (hx h.1.1.1).mono L₂).bind₀
        fun twice m₃ l₃ => ?_
      refine extOne.bind₀ fun one m₄ l₄ => ?_
      refine (extSub (fun h => h.1.2.mono l₄) fun h => h.2).bind₀ fun slope m₅ l₅ => ?_
      refine (KxVar.scale (fun h => h.1.1.1.2.mono (l₃.trans (l₄.trans l₅))) fun h => h.2).bind₀
        fun factor m₆ l₆ => ?_
      refine (oneMinus fun h =>
        (hx h.1.1.1.1.1.1.1).mono (L₂.trans (l₃.trans (l₄.trans (l₅.trans l₆))))).bind₀
        fun low m₇ l₇ => ?_
      refine (extAdd (fun h => h.1.2.1.mono l₇) fun h => h.2).bind₀ fun re m₈ l₈ => ?_
      refine (Sim.pure (Q := fun f m' => IsKx c m' f (chiFactor gr[k]! gpoint[k]?)) fun h => ?_).bind₀
        fun factor' m₉ l₉ => ?_
      · have some : gpoint[k]? = some gpoint[k]! := by
          rw [getElem!_pos gpoint k ((hp h.1.1.1.1.1.1.1.1.1).1 ▸ inside), Array.getElem?_eq_getElem]
        rw [some]
        exact ⟨h.2, h.1.1.2.2.mono (l₇.trans l₈)⟩
      refine (KxVar.mul (fun h => h.1.1.1.1.1.1.1.1.2.mono (l₂.trans (l₃.trans (l₄.trans (l₅.trans
        (l₆.trans (l₇.trans (l₈.trans l₉)))))))) fun h => h.2).bind₀ fun total' m₁₀ l₁₀ => ?_
      exact Sim.pure fun h => ⟨rfl, V_succ ▸ h.2⟩
    · rename_i hpt
      have none : P → gpoint[k]? = none := fun h => by
        rw [Array.getElem?_eq_none_iff, ← (hp h).1]
        exact Array.getElem?_eq_none_iff.mp hpt
      refine (KxVar.oneMinus fun h => h.2).bind₀ fun factor m₃ l₃ => ?_
      refine (KxVar.mul (gb := chiFactor gr[k]! gpoint[k]?) (fun h => h.1.1.2.mono (l₂.trans l₃))
        fun h => ?_).bind₀ fun total' m₄ l₄ => ?_
      · rw [none h.1.1.1.1]
        exact h.2
      exact Sim.pure fun h => ⟨rfl, V_succ ▸ h.2⟩
  · refine h.2.congr ?_
    rw [Array.size_zipIdx, (hr h.1.1).1, toQ_foldl_mul, Replay.chi, toQ_chiList, Array.length_toList]
    refine Finset.prod_congr rfl fun t _ => ?_
    rw [Array.getElem!_toList, Array.getElem?_toList]

end Sim

end CompressionLean.Circuit
