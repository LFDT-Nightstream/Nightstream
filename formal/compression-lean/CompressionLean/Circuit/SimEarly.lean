import CompressionLean.Circuit.SimGkr
import CompressionLean.Circuit.SimViews
import CompressionLean.Bridge.Early
import CompressionLean.Final.Fields

/-!
Owns the simulation of the layer-1 verifier's early stage (`earlyStage` of
`Verifier.lean`, with the helpers of `Early.lean`) against the replay's
(`Replay/Stages.lean`, `Replay/Formulas.lean`).

- `earlyStage_ensures`: once the circuit's checks hold, the replay's early
  stage runs from the matching transcript, and `IsEarlyOut` gives what later
  stages read.
- `Sim.lessThan`, `Sim.step`, `Sim.identity`, `Sim.chi`, `checkPair_ensures`,
  `Sim.observePairs`: the parts of `mle.rs` and `matrix.rs` that the late
  stage reads too.
- The circuit and the replay may add or multiply in different orders. Their
  values agree in the challenge field `Final.Cubic`, where `Ext.toK` is
  injective (`toK_inj`). So an inverse hint that passes its check is the
  replay's inverse (`Ensures.extInv`).
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
    (size : P → starts.size ≤ values.size) :
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
    rw [step_eq gpoint starts gvalues stop (by rw [← (hv h.1.1.1).1]; exact size h.1.1.1)]
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

/-! ## The early stage in the replay -/

open CompressionLean.Replay (Relation TreeShape Runs)

/-- A nonzero value times its inverse is one (`field.rs`). -/
private theorem ext_mul_inv {a : Ext} (ne : a ≠ 0) : a * a.inv = 1 := by
  apply toK_inj
  rw [Replay.Ext.toK_mul, Replay.Ext.toK_inv, Replay.Ext.toK_one]
  exact mul_inv_cancel₀ fun z => ne (Replay.Ext.eq_zero_of_toK z)

/-- Equal values pass the replay's equality test (`field.rs`). -/
private theorem ext_beq_of_eq {a b : Ext} (e : a = b) : (a == b) = true := by
  subst e
  show (a.c0.word == a.c0.word && a.c1.word == a.c1.word && a.c2.word == a.c2.word) = true
  simp

/-- A nonzero value passes the replay's zero test (`field.rs`). -/
private theorem ext_not_isZero {a : Ext} (ne : a ≠ 0) : (!a.isZero) = true := by
  cases h : a.isZero
  · rfl
  · exact absurd (Replay.Ext.eq_of_beq h) ne

/-- Term `t` of `table_sum`: `m_t/(β - (t - H))` (`norm.rs:36-49`). -/
private def tableTerm (bound : ℕ) (counts : Array Gl) (beta : Ext) (slot : ℕ) : Ext :=
  (beta - Ext.ofGl (Gl.ofNat slot - Gl.ofNat bound)).inv.scale counts[slot]!

/-- `table_sum` (`norm.rs:36-49`) accepts when no slot is a pole. -/
private theorem tableSum_ok {bound : ℕ} {counts : Array Gl} {beta : Ext}
    (poles : ∀ s < counts.size, beta - Ext.ofGl (Gl.ofNat s - Gl.ofNat bound) ≠ 0) :
    Replay.tableSum bound counts beta = .ok (Replay.sumRange counts.size (tableTerm bound counts beta)) := by
  unfold Replay.tableSum
  suffices h : ∀ k ≤ counts.size, (List.range k).foldlM (Replay.tableSumStep bound counts beta) 0 =
      .ok (Replay.sumRange k (tableTerm bound counts beta)) from h _ le_rfl
  intro k hk
  induction k with
  | zero => rfl
  | succ k ih =>
      rw [List.range_succ, List.foldlM_append, ih (by omega), sumRange_succ]
      simp [Replay.tableSumStep, Replay.ensure, ext_not_isZero (poles k (by omega)), tableTerm]

/-- `check_early` (`matrix.rs:261-317`) accepts when its pair checks pass and
its leaf claims hold. -/
private theorem checkEarly_ok {layout : Replay.Layout} {r : Array (Gl × Gl)} {ch : Replay.EarlyChallenges}
    {claims : Replay.GkrClaims} {v : Replay.EarlyValues}
    (rows : Replay.pairCheck claims.roots[1]! claims.roots[2]! "row lookup" = .ok ())
    (slots : Replay.pairCheck claims.roots[3]! claims.roots[4]! "slot scatter" = .ok ())
    (runLeaf : claims.leafValue 1 0 = ch.lookup.1 - Replay.tuple ch.lookup.2 v.row v.eRe v.eIm)
    (rowLeaf : claims.leafValue 2 1 = ch.lookup.1 - Replay.tuple ch.lookup.2
      (Replay.identityValue claims.leafPoint[2]!) (Replay.chi r claims.leafPoint[2]!).re
      (Replay.chi r claims.leafPoint[2]!).im)
    (scatterLeaf : claims.leafValue 3 1 = v.eRe + ch.scatter.2 * v.eIm)
    (pairLeaf : claims.leafValue 4 0 = Replay.sumRange (min v.pairs.size
      (Replay.eqTable ((claims.leafPoint[4]!).extract layout.slotVariables (claims.leafPoint[4]!).size)).size)
      fun j => (Replay.eqTable ((claims.leafPoint[4]!).extract layout.slotVariables
        (claims.leafPoint[4]!).size))[j]! * ((v.pairs[j]!).1 + ch.scatter.2 * (v.pairs[j]!).2))
    (indexLeaf : claims.leafValue 4 1 = ch.scatter.1 - Replay.identityValue claims.leafPoint[4]!) :
    Replay.checkEarly layout r ch claims v = .ok (Bridge.earlyOpeningsOf layout ch claims v) := by
  unfold Replay.checkEarly
  rw [rows, slots]
  simp [Replay.ensure, ext_beq_of_eq runLeaf, ext_beq_of_eq rowLeaf, ext_beq_of_eq scatterLeaf,
    ext_beq_of_eq pairLeaf, ext_beq_of_eq indexLeaf]
  rfl

/-- The trees of `GkrView::read` (`gkr.rs:608-656`): none is empty, and each
ends by the last layer. -/
private theorem depth_bounds {proof : Replay.GkrProof} {shapes : Array TreeShape}
    (read : Replay.readGkr proof shapes = .ok ()) {u : ℕ} (inside : u < shapes.size) :
    1 ≤ shapes[u]!.depth ∧ shapes[u]!.depth ≤ Replay.layerCount shapes := by
  refine ⟨?_, ?_⟩
  · unfold Replay.readGkr at read
    split_ifs at read with bad
    · obtain ⟨_, thrown, _⟩ := Replay.Except.bind_ok read
      cases thrown
    · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or] at bad
      have := bad.1.1
      simp only [Array.any_eq_true, not_exists, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at this
      rw [getElem!_pos shapes u inside]
      have := (this u inside).1
      omega
  · unfold Replay.layerCount
    suffices h : ∀ l : List ℕ, u ∈ l → shapes[u]!.depth ≤ l.foldr (fun t d => max d shapes[t]!.depth) 0 from
      h _ (List.mem_range.mpr inside)
    intro l member
    induction l with
    | nil => cases member
    | cons t rest ih =>
        rw [List.foldr_cons]
        rcases List.mem_cons.mp member with rfl | inside
        · exact le_max_right _ _
        · exact (ih inside).trans (le_max_left _ _)

/-- A tree that ends at layer `k` gets one leaf value per part
(`gkr.rs:727-740`). -/
private theorem reducedOf_size {shapes : Array TreeShape} {k u : ℕ} {layer : Replay.GkrLayer} {tau : Ext}
    (ok : Replay.layerShapeOk shapes k layer = true) (inside : u < shapes.size) (ends : shapes[u]!.depth = k) :
    (Replay.reducedOf ((Replay.activeTrees shapes k).zip layer.children).toList tau u).size =
      shapes[u]!.factors + 1 := by
  simp only [Replay.layerShapeOk, Bool.and_eq_true, beq_iff_eq] at ok
  obtain ⟨⟨⟨-, -⟩, childrenSize⟩, childSizes⟩ := ok
  have active : u ∈ (Replay.activeTrees shapes k).toList := by
    have ends' : shapes[u].depth = k := by rw [← getElem!_pos shapes u inside]; exact ends
    unfold Replay.activeTrees
    simp [inside, ends']
  obtain ⟨s, hs, hsu⟩ := List.mem_iff_getElem.mp active
  have member : ((Replay.activeTrees shapes k)[s]'(by simpa using hs), layer.children[s]'(by
      rw [childrenSize]; simpa using hs)) ∈ ((Replay.activeTrees shapes k).zip layer.children).toList := by
    rw [Array.toList_zip]
    exact List.mem_iff_getElem.mpr ⟨s, by simp [childrenSize]; simpa using hs, by simp⟩
  unfold Replay.reducedOf
  split
  · rename_i pair found
    have key : pair.1 = u := by simpa using List.find?_some found
    have size := Array.all_eq_true'.mp childSizes pair (Array.mem_toList_iff.mp (List.mem_of_find?_eq_some found))
    simp only [beq_iff_eq] at size
    rw [Array.size_map, Array.size_range, size, key, Replay.partFactors, if_pos ends]
    omega
  · rename_i missing
    exact absurd (List.find?_eq_none.mp missing _ member) (by simp [← hsu])

/-- After the layers from `k` on, the roots are kept, and a tree that ends
among them has one leaf value per part (`gkr.rs:694-750`). -/
private theorem gkrLayers_facts {shapes : Array TreeShape} :
    ∀ (layers : List Replay.GkrLayer) (k : ℕ) (cl g : Replay.GkrClaims) (t t' : Replay.Transcript),
      Runs (Replay.gkrLayers shapes k layers cl) t g t' →
      (∀ i (h : i < layers.length), Replay.layerShapeOk shapes (k + i) layers[i] = true) →
      cl.leafValues.size = shapes.size →
      g.roots = cl.roots ∧ g.leafValues.size = shapes.size ∧ ∀ u < shapes.size,
        (g.leafValues[u]!).size = if k ≤ shapes[u]!.depth ∧ shapes[u]!.depth < k + layers.length then
          shapes[u]!.factors + 1 else (cl.leafValues[u]!).size
  | [], k, cl, g, t, t', run, _, size => by
      rw [Replay.gkrLayers] at run
      obtain ⟨rfl, rfl⟩ := Replay.run_pure run
      exact ⟨rfl, size, fun u _ => by rw [if_neg (by simp)]⟩
  | layer :: rest, k, cl, g, t, t', run, ok, size => by
      rw [Replay.gkrLayers] at run
      obtain ⟨_, t1, -, run⟩ := Replay.run_bind run
      obtain ⟨_, t2, -, run⟩ := Replay.run_bind run
      obtain ⟨result, t3, -, run⟩ := Replay.run_bind run
      obtain ⟨_, t4, -, run⟩ := Replay.run_bind run
      obtain ⟨_, t5, -, run⟩ := Replay.run_bind run
      obtain ⟨tau, t6, -, run⟩ := Replay.run_bind run
      have ok₀ : Replay.layerShapeOk shapes k layer = true := ok 0 (by simp)
      obtain ⟨roots, vsize, values⟩ := gkrLayers_facts rest (k + 1) _ g t6 t' run (fun i h => by
        have := ok (i + 1) (by simp; omega)
        simpa [Nat.add_assoc, Nat.add_comm 1 i] using this) (by simp [Replay.advance, size])
      refine ⟨roots, vsize, fun u hu => ?_⟩
      rw [values u hu]
      have adv : (Replay.advance shapes k cl ((Replay.activeTrees shapes k).zip layer.children).toList result.1
          tau).leafValues[u]! = if shapes[u]!.depth = k then
            Replay.reducedOf ((Replay.activeTrees shapes k).zip layer.children).toList tau u
          else cl.leafValues[u]! := by
        simp only [Replay.advance]
        exact Bridge.getElem!_map_range _ _ (by omega)
      rw [adv]
      by_cases e : shapes[u]!.depth = k
      · rw [if_neg (by omega), if_pos e, if_pos (by simp; omega), reducedOf_size ok₀ hu e]
      · rw [if_neg e]
        simp only [List.length_cons]
        split_ifs <;> first | rfl | omega

/-- The roots of `verify` (`gkr.rs:679-751`) are the proof's, and each tree
has one leaf value per part. -/
private theorem gkrVerify_facts {proof : Replay.GkrProof} {shapes : Array TreeShape} {t t' : Replay.Transcript}
    {g : Replay.GkrClaims} (read : Replay.readGkr proof shapes = .ok ())
    (run : Runs (Replay.gkrVerify proof shapes) t g t') :
    g.roots = proof.roots ∧ ∀ u < shapes.size, (g.leafValues[u]!).size = shapes[u]!.factors + 1 := by
  obtain ⟨-, layersSize, layersOk⟩ := Bridge.readGkr_ok read
  unfold Replay.gkrVerify at run
  obtain ⟨_, t1, -, run⟩ := Replay.run_bind run
  obtain ⟨roots, -, values⟩ := gkrLayers_facts proof.layers.toList 1 _ g t1 t' run (fun i h => by
    rw [Nat.add_comm]
    simpa using layersOk i (by simpa using h)) (by simp)
  refine ⟨roots, fun u hu => ?_⟩
  have bounds := depth_bounds read hu
  rw [values u hu, if_pos ⟨bounds.1, by rw [Array.length_toList, layersSize]; omega⟩]


/-- Tree `t`'s claim is the replay's (`TreeClaim`, `gkr.rs:70-78`). -/
private def IsTree (c : Ctx F) (n : ℕ) (v : TreeClaimView) (g : Replay.GkrClaims) (t : ℕ) : Prop :=
  IsExt c n v.root.1 (g.roots[t]!).1 ∧ IsExt c n v.root.2 (g.roots[t]!).2 ∧ IsExts c n v.point g.leafPoint[t]! ∧
    IsExts c n v.values g.leafValues[t]!

/-- `IsTree` survives later values (`gkr.rs:70-78`). -/
private theorem IsTree.mono {c : Ctx F} {n m : ℕ} {v : TreeClaimView} {g : Replay.GkrClaims} {t : ℕ}
    (h : IsTree c n v g t) (le : n ≤ m) : IsTree c m v g t :=
  ⟨h.1.mono le, h.2.1.mono le, h.2.2.1.mono le, h.2.2.2.mono le⟩

/-- Leaf value `j` of tree `t` (`gkr.rs:70-78`). -/
private theorem IsTree.value {c : Ctx F} {n : ℕ} {v : TreeClaimView} {g : Replay.GkrClaims} {t j : ℕ}
    (h : IsTree c n v g t) (hj : j < (g.leafValues[t]!).size) : IsExt c n v.values[j]! (g.leafValue t j) :=
  h.2.2.2.2 j (h.2.2.2.1 ▸ hj)

/-- What `check_early` leaves to openings (`matrix.rs:115-123`). -/
private def IsOpenings (c : Ctx F) (m : ℕ) (v : EarlyOpeningsView) (o : Replay.EarlyOpenings) : Prop :=
  IsExts c m v.runPoint o.runPoint ∧ IsExts c m v.pairPoint o.pairPoint ∧ IsExts c m v.rowPoint o.rowPoint ∧
    IsExt c m v.mult o.mult ∧ IsExt c m v.setup.1 o.rowClaim ∧ IsExt c m v.setup.2.1 o.slotClaim ∧
    IsExt c m v.setup.2.2 o.coefClaim

/-- A slice of the values (`matrix.rs:261-317`). -/
private theorem isExts_extract {c : Ctx F} {n : ℕ} {xs : Array ExtVar} {gs : Array Ext} (h : IsExts c n xs gs)
    (a b : ℕ) : IsExts c n (xs.extract a b) (gs.extract a b) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

/-- The claims after the norm tree (`verifier.rs:166`). -/
private theorem extract_one {α : Type} [Inhabited α] (xs : Array α) (i : ℕ) :
    (xs.extract 1 xs.size)[i]! = xs[i + 1]! := by
  by_cases hi : i + 1 < xs.size
  · rw [getElem!_pos _ i (by simp; omega), Array.getElem_extract, getElem!_pos xs (i + 1) hi]
    simp [Nat.add_comm]
  · rw [getElem!_neg _ i (by simp; omega), getElem!_neg xs (i + 1) hi]

/-- The entries of two related arrays (`verifier.rs:43-122`). -/
private theorem forall₂_entries {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) : xs.size = ys.size ∧ ∀ i (hi : i < xs.size), R xs[i] ys[i]! := by
  obtain ⟨len, get⟩ := List.forall₂_iff_get.mp h
  simp only [Array.length_toList] at len
  refine ⟨len, fun i hi => ?_⟩
  have := get i (by simpa using hi) (by simpa [← len] using hi)
  rw [getElem!_pos ys i (by omega)]
  simpa using this

/-- A GKR view's roots and layers, entry by entry (`gkr.rs:591-656`). -/
private theorem IsGkr.entries {c : Ctx F} {n : ℕ} {v : GkrView} {gp : Replay.GkrProof} (h : IsGkr c n v gp) :
    (v.roots.size = gp.roots.size ∧ ∀ i < v.roots.size,
      IsExt c n (v.roots[i]!).1 (gp.roots[i]!).1 ∧ IsExt c n (v.roots[i]!).2 (gp.roots[i]!).2) ∧
    (v.layers.size = gp.layers.size ∧ ∀ i (hi : i < v.layers.size),
      (v.layers[i].rounds.size = (gp.layers[i]!).rounds.size ∧
        ∀ r < v.layers[i].rounds.size, IsExts c n v.layers[i].rounds[r]! (gp.layers[i]!).rounds[r]!) ∧
      (v.layers[i].children.size = (gp.layers[i]!).children.size ∧
        ∀ s < v.layers[i].children.size, IsExts c n v.layers[i].children[s]! (gp.layers[i]!).children[s]!)) := by
  obtain ⟨roots, layers⟩ := h
  obtain ⟨rs, re⟩ := forall₂_entries roots
  obtain ⟨ls, le⟩ := forall₂_entries layers
  refine ⟨⟨rs, fun i hi => ?_⟩, ls, fun i hi => ?_⟩
  · rw [getElem!_pos v.roots i hi]
    exact re i hi
  · obtain ⟨rounds, children⟩ := le i hi
    obtain ⟨rs', re'⟩ := forall₂_entries rounds
    obtain ⟨cs', ce'⟩ := forall₂_entries children
    exact ⟨⟨rs', fun r hr => by rw [getElem!_pos _ r hr]; exact re' r hr⟩, cs',
      fun s hs => by rw [getElem!_pos _ s hs]; exact ce' s hs⟩

/-- The value of `sample_ext` from duplex `r` (`hash.rs:62-65`). -/
private def drawOf (r : Replay.Duplex) : Ext := ⟨r.sample.1, r.sample.2.sample.1, r.sample.2.sample.2.sample.1⟩

/-- The duplex after `sample_ext` (`hash.rs:62-65`). -/
private def afterOf (r : Replay.Duplex) : Replay.Duplex := r.sample.2.sample.2.sample.2

/-- `bind` after a step of `check_early` that checks nothing (`matrix.rs:261-317`). -/
private theorem Ensures.bindSim {c : Ctx F} {n : ℕ} {P : Prop} {α β : Type} {b : Build α} {f : α → Build β}
    {Q : α → ℕ → Prop} {R : β → ℕ → Prop} (hb : Sim c n P b Q True)
    (hf : ∀ a m, n ≤ m → Ensures c m (P ∧ Q a m) (f a) R) : Ensures c n P (b >>= f) R :=
  Ensures.bind (hb.ensures.post fun _ _ _ h => h.1) hf

/-- `bind` after an assertion of `check_early` (`matrix.rs:261-317`). -/
private theorem Ensures.bindCheck {c : Ctx F} {n : ℕ} {P : Prop} {α β : Type} {b : Build α} {f : α → Build β}
    {C : Prop} {R : β → ℕ → Prop} (hb : Sim c n P b (fun _ _ => True) C)
    (hf : ∀ a m, n ≤ m → Ensures c m (P ∧ C) (f a) R) : Ensures c n P (b >>= f) R :=
  Ensures.bind (hb.ensures.post fun _ _ _ h => h.2) hf

/-- `tbind` after a step that checks nothing (`verifier.rs:153-168`). -/
private theorem Ensures.tbindSim {c : Ctx F} {n : ℕ} {P : Prop} {α β : Type} {t : Transcript α}
    {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop} {Q : α × Duplex → ℕ → Prop}
    (ht : Sim c n P (t.run d) Q True) (hg : ∀ a e m, n ≤ m → Ensures c m (P ∧ Q (a, e) m) ((g a).run e) R) :
    Ensures c n P ((t >>= g).run d) R :=
  Ensures.tbind (ht.ensures.post fun _ _ _ h => h.1) hg

/-- Name the witness of an existential precondition; the instructions do not
depend on it (`gkr.rs:747-750`). -/
private theorem Ensures.fixWitness {c : Ctx F} {n : ℕ} {P : Prop} {α ι : Type} {b : Build α}
    {Q : α → ℕ → Prop} {X : ι → Prop} (i₀ : ι) (h : ∀ i, Ensures c n (P ∧ X i) b Q) :
    Ensures c n (P ∧ ∃ i, X i) b Q := by
  intro p E hp hE
  obtain ⟨new, hi, hl, -⟩ := h i₀ p E hp hE
  refine ⟨new, hi, hl, fun ⟨hP, i, hx⟩ hF hh => ?_⟩
  obtain ⟨new', hi', -, hs⟩ := h i p E hp hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs ⟨hP, hx⟩ hF hh

/-! ## The early stage in the circuit -/

section Stage

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- An inverse hint that passes its check is the inverse (`algebra.rs`). -/
private theorem inv_hint {ga : Ext} {u : Fin 3 → F} (h : extProduct (coords ga) u = ![1, 0, 0]) :
    ga ≠ 0 ∧ u = coords ga.inv := by
  have one_ne : (1 : F) ≠ 0 := by
    intro e
    have := (CharP.cast_eq_zero_iff F goldilocksModulus 1).mp (by simpa using e)
    norm_num [goldilocksModulus] at this
  by_cases z : ga = 0
  · subst z
    exfalso
    apply one_ne
    simpa [extProduct] using (congrFun h 0).symm
  · refine ⟨z, ?_⟩
    have inv : extProduct (coords (F := F) ga) (coords ga.inv) = ![1, 0, 0] := by
      rw [← coords_mul, ext_mul_inv z, coords_one]
    calc u = extProduct u ![1, 0, 0] := (extProduct_one u).symm
      _ = extProduct u (extProduct (coords (F := F) ga) (coords ga.inv)) := by rw [inv]
      _ = extProduct (extProduct (coords (F := F) ga) u) (coords ga.inv) := by
        rw [← extProduct_assoc, extProduct_comm u]
      _ = coords (F := F) ga.inv := by rw [h, extProduct_comm, extProduct_one]

/-- `ext_inv` (`algebra.rs`): once its check holds, the operand is nonzero and
the hint is its inverse. -/
theorem Ensures.extInv {a : ExtVar} {ga : Ext} (ha : P → IsExt c n a ga) :
    Ensures c n P (Build.extInv a) (fun r m => ga ≠ 0 ∧ IsExt c m r ga.inv) := by
  refine ((Sim.emit (Q := fun h m => h = n ∧ ∀ k : Fin 3, h + k < m ∧ c.val (h + k) = c.tape (h + k))
    (C := extProduct (coords ga) (fun k => c.tape (n + k)) = ![1, 0, 0]) (.extInv a) fun hP E hl hF => ?_).map
    (Q' := fun r m => ∀ k : Fin 3, r k < m ∧ c.val (r k) = c.tape (n + k))
    fun h m q k => q.1 ▸ q.2 k).ensures.post fun _ r m q => ?_
  · have hE := Ctx.follows_of_step hF
    have ra : (fun i => E.value (a i)) = coords ga := funext fun i => (ha hP i).read hE hl
    refine ⟨⟨rfl, fun k => ⟨by simp [Instr.outputs], ?_⟩⟩, ?_⟩
    · rw [← hl, Ctx.val_next hF (by simp)]
      simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
      rw [dif_pos k.isLt, Option.getD_some]
    · show E.holds ∧ extProduct (fun i => E.value (a i)) (fun k => c.tape (E.values.length + k)) = ![1, 0, 0] ↔ _
      rw [ra, hl]
  · obtain ⟨ne, hu⟩ := inv_hint q.2
    exact ⟨ne, fun k => ⟨(q.1 k).1, (q.1 k).2.trans (congrFun hu k)⟩⟩

/-- `check_pair` (`matrix.rs:402-415`): once its checks hold, the replay's
`pairCheck` accepts. -/
theorem checkPair_ensures {x y : TreeClaimView} {gx gy : Ext × Ext} (name : String)
    (hx : P → IsExt c n x.root.1 gx.1 ∧ IsExt c n x.root.2 gx.2)
    (hy : P → IsExt c n y.root.1 gy.1 ∧ IsExt c n y.root.2 gy.2) :
    Ensures c n P (checkPair x y) (fun _ _ => Replay.pairCheck gx gy name = .ok ()) := by
  obtain ⟨⟨px, qx⟩, xp, xv⟩ := x
  obtain ⟨⟨py, qy⟩, yp, yv⟩ := y
  unfold checkPair
  dsimp only at hx hy ⊢
  refine Ensures.bind (Ensures.extInv fun h => (hx h).2) fun _ m₁ l₁ => ?_
  refine Ensures.bind (Ensures.extInv fun h => ((hy h.1).2).mono l₁) fun _ m₂ l₂ => ?_
  refine Ensures.bind (Sim.extMul (fun h => ((hx h.1.1).1).mono (l₁.trans l₂))
    fun h => ((hy h.1.1).2).mono (l₁.trans l₂)).ensures fun left m₃ l₃ => ?_
  refine Ensures.bind (Sim.extMul (fun h => ((hy h.1.1.1).1).mono (l₁.trans (l₂.trans l₃)))
    fun h => ((hx h.1.1.1).2).mono (l₁.trans (l₂.trans l₃))).ensures fun right m₄ l₄ => ?_
  refine (Sim.assertExtEqual (fun h => h.1.2.1.mono l₄) fun h => h.2.1).ensures.post fun h _ _ q => ?_
  unfold Replay.pairCheck
  simp [Replay.ensure, ext_not_isZero h.1.1.1.2.1, ext_not_isZero h.1.1.2.1, ext_beq_of_eq q.2]
  rfl

/-- `table_sum` (`norm.rs:36-49`): once the inverse checks hold, no slot is a
pole, and the sum is the replay's. -/
private theorem tableSum_ensures {counts : Array Var} {gcounts : Array Gl} {beta : ExtVar} {gbeta : Ext}
    (bound : ℕ) (hc : P → IsWords c n counts gcounts) (hb : P → IsExt c n beta gbeta) :
    Ensures c n P (tableSum counts bound beta) (fun r m =>
      Replay.tableSum bound gcounts gbeta = .ok (Replay.sumRange gcounts.size (tableTerm bound gcounts gbeta)) ∧
        IsExt c m r (Replay.sumRange gcounts.size (tableTerm bound gcounts gbeta))) := by
  unfold tableSum
  refine Ensures.bind Sim.extZero.ensures fun total m₁ l₁ => ?_
  rw [← Array.forIn_toList]
  refine Ensures.bind ((Ensures.loop (P := P ∧ (IsExt c m₁ total 0 ∧ True))
    (fun k acc m => (∀ s < k, gbeta - Ext.ofGl (Gl.ofNat s - Gl.ofNat bound) ≠ 0) ∧
      IsExt c m acc (Replay.sumRange k (tableTerm bound gcounts gbeta)))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, fun s hs => absurd hs (by omega), h.2.1⟩) fun r m _ =>
      Ensures.pure fun h => ?_
  · have hk' : k < counts.size := by simpa using hk
    rw [show counts.zipIdx.toList[k] = (counts[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine Ensures.bind (Sim.extConstant (Ext.ofGl (Gl.ofNat k - Gl.ofNat bound))).ensures fun value m₂ l₂ => ?_
    refine Ensures.bind (Sim.extSub (fun h => (hb h.1.1.1).mono (l₁.trans (lm.trans l₂))) fun h => h.2.1).ensures
      fun den m₃ l₃ => ?_
    refine Ensures.bind (Ensures.extInv fun h => h.2.1) fun inv m₄ l₄ => ?_
    refine Ensures.bind (Sim.extScale (fun h => h.2.2) fun h =>
      ((hc h.1.1.1.1.1).get ((hc h.1.1.1.1.1).1 ▸ hk')).mono (l₁.trans (lm.trans (l₂.trans (l₃.trans l₄))))).ensures
      fun term m₅ l₅ => ?_
    refine Ensures.bind (Sim.extAdd (fun h => h.1.1.1.1.2.2.mono (l₂.trans (l₃.trans (l₄.trans l₅))))
      fun h => h.2.1).ensures fun total' m₆ l₆ => ?_
    refine Ensures.pure fun h => ⟨rfl, fun s hs => ?_, by rw [sumRange_succ]; exact h.2.1⟩
    rcases Nat.lt_succ_iff_lt_or_eq.mp hs with lt | rfl
    · exact h.1.1.1.1.1.2.1 s lt
    · exact h.1.1.2.1
  · have size := (hc h.1.1).1
    have e := h.2
    rw [Array.length_toList, Array.size_zipIdx, size] at e
    exact ⟨tableSum_ok e.1, e.2⟩

omit [CharP F goldilocksModulus] in
/-- `forM` over a list in a transcript (`verifier.rs:318-331`): entry `x`
absorbs the words `f g` of its replay value `g`. -/
private theorem forMList {α β : Type} [Inhabited β] {body : α → Transcript PUnit} {R : ℕ → α → β → Prop}
    (f : β → List Gl) (mono : ∀ m m' x g, R m x g → m ≤ m' → R m' x g)
    (hb : ∀ {n : ℕ} {P : Prop} {d : Duplex} {r : Replay.Duplex} x g, (P → IsDuplex c n d r) → (P → R n x g) →
      Sim c n P ((body x).run d) (fun res m => IsDuplex c m res.2 ((f g).foldl Replay.Duplex.observe r)) True) :
    ∀ (l : List α) (gl : List β) {n : ℕ} {P : Prop} {d : Duplex} {r : Replay.Duplex}, (P → IsDuplex c n d r) →
      (P → List.Forall₂ (R n) l gl) →
      Sim c n P ((l.foldlM (fun _ x => body x) PUnit.unit).run d)
        (fun res m => IsDuplex c m res.2 ((gl.flatMap f).foldl Replay.Duplex.observe r)) True
  | [], gl, n, P, d, r, hd, hl => by
      rw [List.foldlM_nil, StateT.run_pure]
      exact Sim.pure fun h => by
        rw [List.forall₂_nil_left_iff.mp (hl h)]
        exact hd h
  | x :: l, gl, n, P, d, r, hd, hl => by
      rw [List.foldlM_cons, StateT.run_bind]
      cases gl with
      | nil =>
          exact (Sim.bind₀ (hb x default hd fun h => absurd (hl h) (by simp)) fun _ _ _ =>
            forMList f mono hb l [] (fun h => h.2) fun h => absurd (hl h.1) (by simp)).post
            (fun h => absurd (hl h) (by simp)) fun h => absurd (hl h) (by simp)
      | cons g gl =>
          refine Sim.bind₀ (hb x g hd fun h => (List.forall₂_cons.mp (hl h)).1) fun res m le => ?_
          refine (forMList f mono hb l gl (fun h => h.2) fun h =>
            ((List.forall₂_cons.mp (hl h.1)).2).imp fun _ _ hr => mono _ _ _ _ hr le).post
            (fun _ _ _ q => ?_) fun _ => Iff.rfl
          rw [List.flatMap_cons, List.foldl_append]
          exact q

/-- `observe_pairs` (`verifier.rs:326-331`): the replay's `observeExts` of the
flattened pairs. -/
theorem Sim.observePairs {d : Duplex} {r : Replay.Duplex} {pairs : Array (ExtVar × ExtVar)}
    {gpairs : Array (Ext × Ext)} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    (hv : P → List.Forall₂ (IsPair c n) pairs.toList gpairs.toList) :
    Sim c n P ((Circuit.observePairs pairs).run d)
      (fun res m => IsDuplex c m res.2
        (((Replay.flatPairs gpairs).flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r)) True := by
  unfold Circuit.observePairs Array.forM
  rw [← Array.foldlM_toList]
  refine (forMList (R := fun m => IsPair c m) (fun g => [g.1.c0, g.1.c1, g.1.c2, g.2.c0, g.2.c1, g.2.c2])
    (fun _ _ _ _ h le => h.mono le) (fun x g hd hx => ?_) pairs.toList gpairs.toList hd hv).post
    (fun _ _ _ q => ?_) fun _ => Iff.rfl
  · obtain ⟨a, b⟩ := x
    dsimp only
    rw [StateT.run_bind]
    simp only [List.foldl_cons, List.foldl_nil]
    exact (Sim.observeExt hp hd fun h => (hx h).1).bind₀ fun _ _ l =>
      Sim.observeExt hp (fun h => h.2) fun h => (hx h.1).2.mono l
  · have e : ((Replay.flatPairs gpairs).flatMap fun g => #[g.c0, g.c1, g.c2]).toList =
        gpairs.toList.flatMap fun g => [g.1.c0, g.1.c1, g.1.c2, g.2.c0, g.2.c1, g.2.c2] := by
      rw [Array.toList_flatMap, Replay.flatPairs, Array.toList_flatMap, List.flatMap_assoc]
      simp
    rw [← Array.foldl_toList, e]
    exact q

omit [CharP F goldilocksModulus] in
/-- The entries of an `IsExts` (`verifier.rs:318-324`). -/
private theorem isExts_forall₂ {xs : Array ExtVar} {gs : Array Ext} (h : IsExts c n xs gs) :
    List.Forall₂ (IsExt c n) xs.toList gs.toList := by
  rw [List.forall₂_iff_get]
  refine ⟨by simp [h.1], fun i h₁ h₂ => ?_⟩
  have := h.2 i (by simpa using h₁)
  rw [getElem!_pos xs i (by simpa using h₁), getElem!_pos gs i (by simpa using h₂)] at this
  simpa using this

/-- `observe_early` (`verifier.rs:318-324`): the replay's `observeExts` of the
run values, the flattened pairs and the block. -/
private theorem observeEarly_sim {d : Duplex} {r : Replay.Duplex} {values : EarlyValuesView}
    {gv : Replay.EarlyValues} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    (hruns : P → IsExts c n values.runs #[gv.eRe, gv.eIm, gv.row])
    (hpairs : P → List.Forall₂ (IsPair c n) values.pairs.toList gv.pairs.toList)
    (hblock : P → IsExt c n values.block gv.block) :
    Sim c n P ((observeEarly values).run d) (fun res m => IsDuplex c m res.2
      (((#[gv.eRe, gv.eIm, gv.row] ++ Replay.flatPairs gv.pairs ++ #[gv.block]).flatMap
        fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r)) True := by
  unfold observeEarly
  rw [StateT.run_bind]
  refine Sim.bind₀ (P := P) (Q := fun res m => IsDuplex c m res.2
    (([gv.eRe, gv.eIm, gv.row].flatMap fun g => [g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r)) ?_
    fun _ m₁ l₁ => ?_
  · unfold Array.forM
    rw [← Array.foldlM_toList]
    exact forMList (R := fun m => IsExt c m) (fun g => [g.c0, g.c1, g.c2]) (fun _ _ _ _ h le => h.mono le)
      (fun x g hd hx => by simp only [List.foldl_cons, List.foldl_nil]; exact Sim.observeExt hp hd hx)
      values.runs.toList [gv.eRe, gv.eIm, gv.row] hd
      fun h => isExts_forall₂ (hruns h)
  rw [StateT.run_bind]
  refine Sim.bind₀ (Sim.observePairs hp (fun h => h.2) fun h => ((hpairs h.1).imp fun _ _ q => q.mono l₁))
    fun _ m₂ l₂ => ?_
  refine (Sim.observeExt hp (fun h => h.2) fun h => (hblock h.1.1).mono (l₁.trans l₂)).post (fun _ _ _ q => ?_)
    fun _ => Iff.rfl
  have e : ((#[gv.eRe, gv.eIm, gv.row] ++ Replay.flatPairs gv.pairs ++ #[gv.block]).flatMap
      fun g => #[g.c0, g.c1, g.c2]).toList = ([gv.eRe, gv.eIm, gv.row].flatMap fun g => [g.c0, g.c1, g.c2]) ++
        ((Replay.flatPairs gv.pairs).flatMap fun g => #[g.c0, g.c1, g.c2]).toList ++
          [gv.block.c0, gv.block.c1, gv.block.c2] := by
    simp [Array.toList_flatMap]
  rw [← Array.foldl_toList, e, List.foldl_append, List.foldl_append]
  rw [← Array.foldl_toList] at q
  simp only [List.foldl_cons, List.foldl_nil]
  exact q

/-- `tuple`: `i + γ·(re + γ·im)` (`matrix.rs:394-400`). -/
private theorem tuple_sim {index gamma re im : ExtVar} {gi gg gr gm : Ext} (hi : P → IsExt c n index gi)
    (hg : P → IsExt c n gamma gg) (hr : P → IsExt c n re gr) (hm : P → IsExt c n im gm) :
    Sim c n P (tuple index gamma re im) (fun r m => IsExt c m r (Replay.tuple gg gi gr gm)) True :=
  (Sim.extMul hg hm).bind₀ fun _ _ l₁ => (Sim.extAdd (fun h => (hr h.1).mono l₁) fun h => h.2).bind₀
    fun _ _ l₂ => (Sim.extMul (fun h => (hg h.1.1).mono (l₁.trans l₂)) fun h => h.2).bind₀ fun _ _ l₃ =>
    Sim.extAdd (fun h => (hi h.1.1.1).mono (l₁.trans (l₂.trans l₃))) fun h => h.2

/-- `matrix_step` (`matrix.rs:368-380`): the replay's `matrixStep`. -/
private theorem matrixStep_sim {layout : Replay.Layout} {point : Array ExtVar} {gpoint : Array Ext}
    (hp : P → IsExts c n point gpoint) :
    Sim c n P (matrixStep layout point) (fun r m => IsExt c m r (Replay.matrixStep layout gpoint)) True := by
  unfold matrixStep
  refine (Sim.rangeMapM layout.matrices (fun j a m => IsExt c m a (Ext.ofGl (Gl.ofNat j)))
    (fun _ _ _ _ h le => h.mono le) fun j _ m _ => Sim.extConstant _).bind₀ fun values m₁ l₁ => ?_
  refine (Sim.step (gvalues := (Array.range layout.matrices).map fun j => Ext.ofGl (Gl.ofNat j)) _
    (fun h => (hp h.1).mono l₁) (fun h => ⟨by simp [h.2.1], fun i hi => ?_⟩) fun h => ?_).post
    (fun _ _ _ e => ?_) fun _ => Iff.rfl
  · have hi' : i < layout.matrices := h.2.1 ▸ hi
    rw [getElem!_pos values i hi, getElem!_pos _ i (by simpa using hi'), Array.getElem_map, Array.getElem_range]
    exact h.2.2 i hi
  · rw [h.2.1, Array.size_extract]
    omega
  · rw [Replay.matrixStep]
    simpa [Array.toList_map, Array.toList_range] using e

/-- `Structure::check_early` (`matrix.rs:261-317`): once its checks hold, the
replay's `checkEarly` accepts, and the openings are the replay's. -/
private theorem checkEarly_ensures {layout : Replay.Layout} {r : Array KVar} {gr : Array (Gl × Gl)}
    {lb gamma sb mu : ExtVar} {ch : Replay.EarlyChallenges} {claims : Array TreeClaimView}
    {g : Replay.GkrClaims} {values : EarlyValuesView} {gv : Replay.EarlyValues}
    (hr : P → r.size = gr.size ∧ ∀ i < r.size, IsWord c n r[i]!.1 gr[i]!.1 ∧ IsWord c n r[i]!.2 gr[i]!.2)
    (hch : P → IsExt c n lb ch.lookup.1 ∧ IsExt c n gamma ch.lookup.2 ∧ IsExt c n sb ch.scatter.1 ∧
      IsExt c n mu ch.scatter.2)
    (htrees : P → ∀ t < 4, IsTree c n claims[t]! g (t + 1))
    (hsizes : P → 1 ≤ (g.leafValues[1]!).size ∧ 2 ≤ (g.leafValues[2]!).size ∧ 3 ≤ (g.leafValues[3]!).size ∧
      2 ≤ (g.leafValues[4]!).size)
    (hruns : P → IsExts c n values.runs #[gv.eRe, gv.eIm, gv.row])
    (hpairs : P → values.pairs.size = gv.pairs.size ∧
      ∀ j < values.pairs.size, IsPair c n values.pairs[j]! gv.pairs[j]!) :
    Ensures c n P (checkEarly layout r { lookup := (lb, gamma), scatter := (sb, mu) } claims values)
      (fun res m => Replay.checkEarly layout gr ch g gv = .ok (Bridge.earlyOpeningsOf layout ch g gv) ∧
        IsOpenings c m res (Bridge.earlyOpeningsOf layout ch g gv)) := by
  have tree₁ : ∀ {m}, P → n ≤ m → IsTree c m claims[0]! g 1 := fun h le => (htrees h 0 (by omega)).mono le
  have tree₂ : ∀ {m}, P → n ≤ m → IsTree c m claims[1]! g 2 := fun h le => (htrees h 1 (by omega)).mono le
  have tree₃ : ∀ {m}, P → n ≤ m → IsTree c m claims[2]! g 3 := fun h le => (htrees h 2 (by omega)).mono le
  have tree₄ : ∀ {m}, P → n ≤ m → IsTree c m claims[3]! g 4 := fun h le => (htrees h 3 (by omega)).mono le
  have runRe : ∀ {m}, P → n ≤ m → IsExt c m values.runs[0]! gv.eRe :=
    fun h le => ((hruns h).2 0 (by rw [(hruns h).1]; simp)).mono le
  have runIm : ∀ {m}, P → n ≤ m → IsExt c m values.runs[1]! gv.eIm :=
    fun h le => ((hruns h).2 1 (by rw [(hruns h).1]; simp)).mono le
  have runRow : ∀ {m}, P → n ≤ m → IsExt c m values.runs[2]! gv.row :=
    fun h le => ((hruns h).2 2 (by rw [(hruns h).1]; simp)).mono le
  have chs : ∀ {m}, P → n ≤ m → IsExt c m lb ch.lookup.1 ∧ IsExt c m gamma ch.lookup.2 ∧
      IsExt c m sb ch.scatter.1 ∧ IsExt c m mu ch.scatter.2 :=
    fun h le => ⟨(hch h).1.mono le, (hch h).2.1.mono le, (hch h).2.2.1.mono le, (hch h).2.2.2.mono le⟩
  have hpair : ∀ {m} j, j < values.pairs.size → P → n ≤ m → IsPair c m values.pairs[j]! gv.pairs[j]! :=
    fun j hj h le => ((hpairs h).2 j hj).mono le
  unfold checkEarly
  dsimp only
  refine Ensures.bind (checkPair_ensures (gx := g.roots[1]!) (gy := g.roots[2]!) "row lookup"
    (fun h => ⟨(tree₁ h le_rfl).1, (tree₁ h le_rfl).2.1⟩)
    fun h => ⟨(tree₂ h le_rfl).1, (tree₂ h le_rfl).2.1⟩) fun _ m₁ l₁ => ?_
  refine Ensures.bind (checkPair_ensures (gx := g.roots[3]!) (gy := g.roots[4]!) "slot scatter"
    (fun h => ⟨(tree₃ h.1 l₁).1, (tree₃ h.1 l₁).2.1⟩)
    fun h => ⟨(tree₄ h.1 l₁).1, (tree₄ h.1 l₁).2.1⟩) fun _ m₂ l₂ => ?_
  have L₂ : n ≤ m₂ := l₁.trans l₂
  refine Ensures.pre (P' := P ∧ Replay.pairCheck g.roots[1]! g.roots[2]! "row lookup" = .ok () ∧
    Replay.pairCheck g.roots[3]! g.roots[4]! "slot scatter" = .ok ()) ?_ fun h => ⟨h.1.1, h.1.2, h.2⟩
  refine Ensures.bindSim (tuple_sim (fun h => runRow h.1 L₂) (fun h => (chs h.1 L₂).2.1)
    (fun h => runRe h.1 L₂) fun h => runIm h.1 L₂) fun combined m₃ l₃ => ?_
  refine Ensures.bindSim (Sim.extSub (fun h => (chs h.1.1 (L₂.trans l₃)).1) fun h => h.2) fun expected m₄ l₄ => ?_
  refine Ensures.bindCheck (Sim.assertExtEqual (fun h => (tree₁ h.1.1.1 (L₂.trans (l₃.trans l₄))).value
    (by have := (hsizes h.1.1.1).1; omega)) fun h => h.2) fun _ m₅ l₅ => ?_
  have L₅ : n ≤ m₅ := L₂.trans (l₃.trans (l₄.trans l₅))
  refine Ensures.pre (P' := P ∧ Replay.pairCheck g.roots[1]! g.roots[2]! "row lookup" = .ok () ∧
    Replay.pairCheck g.roots[3]! g.roots[4]! "slot scatter" = .ok () ∧
    g.leafValue 1 0 = ch.lookup.1 - Replay.tuple ch.lookup.2 gv.row gv.eRe gv.eIm) ?_
    fun h => ⟨h.1.1.1.1, h.1.1.1.2.1, h.1.1.1.2.2, h.2⟩
  refine Ensures.bindSim (Sim.chi (fun h => ⟨(hr h.1).1, fun i hi => ⟨((hr h.1).2 i hi).1.mono L₅,
    ((hr h.1).2 i hi).2.mono L₅⟩⟩) fun h => (tree₂ h.1 L₅).2.2.1) fun chi m₆ l₆ => ?_
  refine Ensures.bindSim (Sim.identity fun h => (tree₂ h.1.1 (L₅.trans l₆)).2.2.1) fun index m₇ l₇ => ?_
  refine Ensures.bindSim (tuple_sim (fun h => h.2) (fun h => (chs h.1.1.1 (L₅.trans (l₆.trans l₇))).2.1)
    (fun h => h.1.2.1.mono l₇) fun h => h.1.2.2.mono l₇) fun combined' m₈ l₈ => ?_
  refine Ensures.bindSim (Sim.extSub (fun h => (chs h.1.1.1.1 (L₅.trans (l₆.trans (l₇.trans l₈)))).1)
    fun h => h.2) fun expected' m₉ l₉ => ?_
  refine Ensures.bindCheck (Sim.assertExtEqual (fun h =>
    (tree₂ h.1.1.1.1.1 (L₅.trans (l₆.trans (l₇.trans (l₈.trans l₉))))).value
    (by have := (hsizes h.1.1.1.1.1).2.1; omega)) fun h => h.2) fun _ m₁₀ l₁₀ => ?_
  have L₁₀ : n ≤ m₁₀ := L₅.trans (l₆.trans (l₇.trans (l₈.trans (l₉.trans l₁₀))))
  refine Ensures.pre (P' := P ∧ Replay.pairCheck g.roots[1]! g.roots[2]! "row lookup" = .ok () ∧
    Replay.pairCheck g.roots[3]! g.roots[4]! "slot scatter" = .ok () ∧
    g.leafValue 1 0 = ch.lookup.1 - Replay.tuple ch.lookup.2 gv.row gv.eRe gv.eIm ∧
    g.leafValue 2 1 = ch.lookup.1 - Replay.tuple ch.lookup.2 (Replay.identityValue g.leafPoint[2]!)
      (Replay.chi gr g.leafPoint[2]!).re (Replay.chi gr g.leafPoint[2]!).im) ?_
    fun h => ⟨h.1.1.1.1.1.1, h.1.1.1.1.1.2.1, h.1.1.1.1.1.2.2.1, h.1.1.1.1.1.2.2.2, h.2⟩
  refine Ensures.bindSim (Sim.extMul (fun h => (chs h.1 L₁₀).2.2.2) fun h => runIm h.1 L₁₀) fun mixed m₁₁ l₁₁ => ?_
  refine Ensures.bindSim (Sim.extAdd (fun h => runRe h.1.1 (L₁₀.trans l₁₁)) fun h => h.2) fun e m₁₂ l₁₂ => ?_
  refine Ensures.bindCheck (Sim.assertExtEqual (fun h =>
    (tree₃ h.1.1.1 (L₁₀.trans (l₁₁.trans l₁₂))).value (by have := (hsizes h.1.1.1).2.2.1; omega))
    fun h => h.2) fun _ m₁₃ l₁₃ => ?_
  have L₁₃ : n ≤ m₁₃ := L₁₀.trans (l₁₁.trans (l₁₂.trans l₁₃))
  refine Ensures.pre (P' := P ∧ Replay.pairCheck g.roots[1]! g.roots[2]! "row lookup" = .ok () ∧
    Replay.pairCheck g.roots[3]! g.roots[4]! "slot scatter" = .ok () ∧
    g.leafValue 1 0 = ch.lookup.1 - Replay.tuple ch.lookup.2 gv.row gv.eRe gv.eIm ∧
    g.leafValue 2 1 = ch.lookup.1 - Replay.tuple ch.lookup.2 (Replay.identityValue g.leafPoint[2]!)
      (Replay.chi gr g.leafPoint[2]!).re (Replay.chi gr g.leafPoint[2]!).im ∧
    g.leafValue 3 1 = gv.eRe + ch.scatter.2 * gv.eIm) ?_
    fun h => ⟨h.1.1.1.1, h.1.1.1.2.1, h.1.1.1.2.2.1, h.1.1.1.2.2.2.1, h.1.1.1.2.2.2.2, h.2⟩
  refine Ensures.bindSim (matrixStep_sim (layout := layout) fun h => (tree₃ h.1 L₁₃).2.2.1)
    fun matrix m₁₄ l₁₄ => ?_
  refine Ensures.bindSim (Sim.extScaleConstant (Gl.ofNat (2 ^ layout.slotVariables)) fun h => h.2)
    fun tag m₁₅ l₁₅ => ?_
  refine Ensures.bindSim (Sim.extSub (fun h => (chs h.1.1.1 (L₁₃.trans (l₁₄.trans l₁₅))).2.2.1)
    fun h => (tree₃ h.1.1.1 (L₁₃.trans (l₁₄.trans l₁₅))).value
      (by have := (hsizes h.1.1.1).2.2.1; omega)) fun rest m₁₆ l₁₆ => ?_
  refine Ensures.bindSim (Sim.extSub (fun h => h.2) fun h => h.1.2.mono l₁₆) fun slot m₁₇ l₁₇ => ?_
  have L₁₇ : n ≤ m₁₇ := L₁₃.trans (l₁₄.trans (l₁₅.trans (l₁₆.trans l₁₇)))
  set slotClaim := ch.scatter.1 - g.leafValue 3 2 -
    (Replay.matrixStep layout g.leafPoint[3]!).scale (Gl.ofNat (2 ^ layout.slotVariables))
  refine Ensures.pre (P' := P ∧ Replay.pairCheck g.roots[1]! g.roots[2]! "row lookup" = .ok () ∧
    Replay.pairCheck g.roots[3]! g.roots[4]! "slot scatter" = .ok () ∧
    g.leafValue 1 0 = ch.lookup.1 - Replay.tuple ch.lookup.2 gv.row gv.eRe gv.eIm ∧
    g.leafValue 2 1 = ch.lookup.1 - Replay.tuple ch.lookup.2 (Replay.identityValue g.leafPoint[2]!)
      (Replay.chi gr g.leafPoint[2]!).re (Replay.chi gr g.leafPoint[2]!).im ∧
    g.leafValue 3 1 = gv.eRe + ch.scatter.2 * gv.eIm ∧ IsExt c m₁₇ slot slotClaim) ?_
    fun h => ⟨h.1.1.1.1.1, h.1.1.1.1.2.1, h.1.1.1.1.2.2.1, h.1.1.1.1.2.2.2.1, h.1.1.1.1.2.2.2.2.1,
      h.1.1.1.1.2.2.2.2.2, h.2⟩
  set geq := Replay.eqTable ((g.leafPoint[4]!).extract layout.slotVariables (g.leafPoint[4]!).size)
  refine Ensures.bindSim (Sim.eqTable (gpoint := (g.leafPoint[4]!).extract layout.slotVariables)
    fun h => (tree₄ h.1 L₁₇).2.2.1.extract layout.slotVariables) fun eq m₁₈ l₁₈ => ?_
  refine Ensures.bindSim Sim.extZero fun combined₀ m₁₉ l₁₉ => ?_
  refine Ensures.bindSim (((Sim.loop₀ (P := ((_ ∧ IsExts c m₁₈ eq geq) ∧ IsExt c m₁₉ combined₀ 0))
    (fun k acc m => IsExt c m acc (Replay.sumRange k fun j => geq[j]! * ((gv.pairs[j]!).1 + ch.scatter.2 *
      (gv.pairs[j]!).2))) fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).post
    (Q' := fun r m => IsExt c m r (Replay.sumRange (min gv.pairs.size geq.size) fun j => geq[j]! *
      ((gv.pairs[j]!).1 + ch.scatter.2 * (gv.pairs[j]!).2))) (fun h _ _ q => ?_) fun _ => Iff.rfl)
    fun combined m₂₀ l₂₀ => ?_
  · have hk' : k < values.pairs.size ∧ k < eq.size := by simpa using hk
    rw [show (values.pairs.zip eq)[k] = (values.pairs[k]!, eq[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    have L : n ≤ m := L₁₇.trans (l₁₈.trans (l₁₉.trans lm))
    refine (Sim.extMul (fun h => (chs h.1.1.1.1 L).2.2.2) fun h => (hpair k hk'.1 h.1.1.1.1 L).2).bind₀
      fun mixed m₂ l₂ => ?_
    refine (Sim.extAdd (fun h => (hpair k hk'.1 h.1.1.1.1.1 (L.trans l₂)).1) fun h => h.2).bind₀
      fun value m₃ l₃ => ?_
    refine (Sim.extMul (fun h => ((h.1.1.1.1.2).2 k hk'.2).mono (l₁₉.trans (lm.trans (l₂.trans l₃))))
      fun h => h.2).bind₀ fun term m₄ l₄ => ?_
    refine (Sim.extAdd (fun h => h.1.1.1.2.mono (l₂.trans (l₃.trans l₄))) fun h => h.2).bind₀ fun total m₅ l₅ => ?_
    exact Sim.pure fun h => ⟨rfl, by rw [sumRange_succ]; exact h.2⟩
  · have size : (values.pairs.zip eq).size = min gv.pairs.size geq.size := by
      rw [Array.size_zip, (hpairs h.1.1.1).1, h.1.2.1]
    rw [size] at q
    exact q
  refine Ensures.bindCheck (Sim.assertExtEqual (fun h =>
    (tree₄ h.1.1.1.1 (L₁₇.trans (l₁₈.trans (l₁₉.trans l₂₀)))).value
      (by have := (hsizes h.1.1.1.1).2.2.2; omega)) fun h => h.2) fun _ m₂₁ l₂₁ => ?_
  have L₂₁ : n ≤ m₂₁ := L₁₇.trans (l₁₈.trans (l₁₉.trans (l₂₀.trans l₂₁)))
  refine Ensures.bindSim (Sim.identity fun h => (tree₄ h.1.1.1.1.1 L₂₁).2.2.1) fun index' m₂₂ l₂₂ => ?_
  refine Ensures.bindSim (Sim.extSub (fun h => (chs h.1.1.1.1.1.1 (L₂₁.trans l₂₂)).2.2.1) fun h => h.2)
    fun expected'' m₂₃ l₂₃ => ?_
  refine Ensures.bindCheck (Sim.assertExtEqual (fun h =>
    (tree₄ h.1.1.1.1.1.1.1 (L₂₁.trans (l₂₂.trans l₂₃))).value
      (by have := (hsizes h.1.1.1.1.1.1.1).2.2.2; omega)) fun h => h.2) fun _ m₂₄ l₂₄ => ?_
  refine Ensures.pure fun h => ?_
  have L₂₄ : n ≤ m₂₄ := L₂₁.trans (l₂₂.trans (l₂₃.trans l₂₄))
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hP, rows, slots, runLeaf, rowLeaf, scatterLeaf, hslot⟩, -⟩, -⟩, -⟩, pairLeaf⟩, -⟩, -⟩,
    indexLeaf⟩ := h
  refine ⟨checkEarly_ok rows slots runLeaf rowLeaf scatterLeaf pairLeaf indexLeaf, (tree₃ hP L₂₄).2.2.1,
    isExts_extract (tree₄ hP L₂₄).2.2.1 0 layout.slotVariables, (tree₂ hP L₂₄).2.2.1,
    (tree₂ hP L₂₄).value (by have := (hsizes hP).2.1; omega), runRow hP L₂₄,
    hslot.mono (l₁₈.trans (l₁₉.trans (l₂₀.trans (l₂₁.trans (l₂₂.trans (l₂₃.trans l₂₄)))))),
    (tree₃ hP L₂₄).value (by have := (hsizes hP).2.2.1; omega)⟩

/-- `observe_root` (`whir/mod.rs:221-232`): the separator words as constants,
then the root. -/
private theorem observeSeedRoot {d : Duplex} {r : Replay.Duplex} {seed : Array Gl} {root : DigestVar}
    {groot : Array Gl} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r) (hr : P → IsDigest c n root groot) :
    Sim c n P ((Circuit.observeRoot seed root).run d)
      (fun res m => IsDuplex c m res.2 (groot.foldl Replay.Duplex.observe (seed.foldl Replay.Duplex.observe r)))
      True := by
  unfold Circuit.observeRoot
  rw [← Array.forIn_toList]
  refine (Sim.tbind (C' := True) ((Sim.loopT (P := P) (d := d) (I := fun k _ e m => IsDuplex c m e
      ((seed.toList.take k).foldl Replay.Duplex.observe r)) (D := fun _ => True) fun k hk _ e m lm => ?_).pre
      fun h => by exact ⟨h, hd h⟩) fun _ e m le => ?_).post (fun _ _ _ h => h) fun _ => by simp
  · refine Sim.tbind₀ (Sim.lift (Sim.const seed.toList[k])) fun w e₁ m₁ l₁ => ?_
    refine Sim.tbind₀ (Sim.observe hp (fun h => by rw [show e₁ = e from h.2.2]; exact h.1.2.mono l₁)
      fun h => h.2.1) fun _ e₂ m₂ l₂ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_⟩
    rw [List.take_succ_eq_append_getElem hk, List.foldl_append, List.foldl_cons, List.foldl_nil]
    exact h.2
  · refine Sim.observeSlice hp (fun h => ?_) fun h => IsWords.ofFn (hr h.1).1 fun lane => ((hr h.1).2 lane).mono le
    have e := h.2
    rw [List.take_of_length_le (by simp), Array.foldl_toList] at e
    exact e

/-- What later stages read of the early stage (`verifier.rs:153-168`): the
norm tree's leaf point, `z~` at it (`β - q~`), and the matrix openings, with
the run columns `(i, σ, m)` at the run point. -/
def IsEarlyOut (c : Ctx F) (m : ℕ) (v : EarlyOut) (o : Replay.EarlyOut) : Prop :=
  IsExts c m v.normPoint o.gkr.leafPoint[0]! ∧ IsExt c m v.zLeaf o.zLeaf ∧
    IsExts c m v.openings.runPoint o.openings.runPoint ∧ IsExts c m v.openings.pairPoint o.openings.pairPoint ∧
    IsExts c m v.openings.rowPoint o.openings.rowPoint ∧ IsExt c m v.openings.mult o.openings.mult ∧
    IsExt c m v.openings.setup.1 o.openings.rowClaim ∧ IsExt c m v.openings.setup.2.1 o.openings.slotClaim ∧
    IsExt c m v.openings.setup.2.2 o.openings.coefClaim

omit [CharP F goldilocksModulus] in
/-- `IsEarlyOut` survives later values (`verifier.rs:153-168`). -/
theorem IsEarlyOut.mono {m m' : ℕ} {v : EarlyOut} {o : Replay.EarlyOut} (h : IsEarlyOut c m v o) (le : m ≤ m') :
    IsEarlyOut c m' v o :=
  ⟨h.1.mono le, h.2.1.mono le, h.2.2.1.mono le, h.2.2.2.1.mono le, h.2.2.2.2.1.mono le, h.2.2.2.2.2.1.mono le,
    h.2.2.2.2.2.2.1.mono le, h.2.2.2.2.2.2.2.1.mono le, h.2.2.2.2.2.2.2.2.mono le⟩

/-- **The early stage** (`verifier.rs:153-168`). Once the circuit's checks
hold, the replay's early stage runs from the matching transcript, the
circuit's outputs are the replay's, and the duplexes still match. -/
theorem earlyStage_ensures {rel : Relation} {statement : ClaimView} {st : Replay.Statement} {proof : ProofView}
    {w : Var → Gl} {g : Replay.Proof} {d : Duplex} {t : Replay.Transcript} (hp : Permutes c.perm)
    (hd : P → IsDuplex c n d t.duplex)
    (hpoint : P → statement.point.size = st.point.size ∧ ∀ i < statement.point.size,
      IsWord c n statement.point[i]!.1 st.point[i]!.1 ∧ IsWord c n statement.point[i]!.2 st.point[i]!.2)
    (hproof : P → IsProof c w n proof g) (read : Replay.readProof rel g = .ok ()) :
    Ensures c n P ((Circuit.earlyStage rel statement proof).run d)
      (fun res m => ∃ out t', Runs (Replay.earlyStage rel st g) t out t' ∧ IsEarlyOut c m res.1 out ∧
        IsDuplex c m res.2 t'.duplex) := by
  have shape := Bridge.readProof_ok read
  obtain ⟨rootsSize, -, -⟩ := Bridge.readGkr_ok shape.early
  set D₂ := g.histogram.foldl Replay.Duplex.observe
    (g.p0.foldl Replay.Duplex.observe (rel.commitmentSeed.foldl Replay.Duplex.observe t.duplex))
  set ch : Replay.EarlyChallenges := ⟨drawOf D₂, (drawOf (afterOf D₂), drawOf (afterOf (afterOf D₂))),
    (drawOf (afterOf (afterOf (afterOf D₂))), drawOf (afterOf (afterOf (afterOf (afterOf D₂)))))⟩
  set D₇ := afterOf (afterOf (afterOf (afterOf (afterOf D₂))))
  set t₇ : Replay.Transcript := ⟨D₇, ((t.draws.push (.ext ch.beta)).push (.pair ch.lookup.1 ch.lookup.2)).push
    (.pair ch.scatter.1 ch.scatter.2)⟩
  set T := Replay.sumRange g.histogram.size (tableTerm rel.normBound g.histogram ch.beta)
  unfold Circuit.earlyStage
  refine Ensures.tbindSim (observeSeedRoot hp hd fun h => (hproof h).1) fun _ d₁ m₁ l₁ => ?_
  refine Ensures.tbindSim (Sim.observeSlice hp (fun h => h.2) fun h => (hproof h.1).2.1.mono l₁) fun _ d₂ m₂ l₂ => ?_
  refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2) fun beta d₃ m₃ l₃ => ?_
  refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2.2) fun lookupBeta d₄ m₄ l₄ => ?_
  refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2.2) fun gamma d₅ m₅ l₅ => ?_
  refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2.2) fun scatterBeta d₆ m₆ l₆ => ?_
  refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2.2) fun mu d₇ m₇ l₇ => ?_
  have L₇ : n ≤ m₇ := l₁.trans (l₂.trans (l₃.trans (l₄.trans (l₅.trans (l₆.trans l₇)))))
  refine Ensures.pre (P' := P ∧ IsExt c m₇ beta ch.beta ∧ IsExt c m₇ lookupBeta ch.lookup.1 ∧
    IsExt c m₇ gamma ch.lookup.2 ∧ IsExt c m₇ scatterBeta ch.scatter.1 ∧ IsExt c m₇ mu ch.scatter.2 ∧
    IsDuplex c m₇ d₇ D₇) ?_ fun h => ⟨h.1.1.1.1.1.1.1, h.1.1.1.1.2.1.mono (l₄.trans (l₅.trans (l₆.trans l₇))),
      h.1.1.1.2.1.mono (l₅.trans (l₆.trans l₇)), h.1.1.2.1.mono (l₆.trans l₇), h.1.2.1.mono l₇, h.2.1, h.2.2⟩
  refine Ensures.tbind (Ensures.lift (tableSum_ensures rel.normBound (fun h => (hproof h.1).2.1.mono L₇)
    fun h => h.2.1)) fun total d₈ m₈ l₈ => ?_
  refine Ensures.tbind (gkrVerify_ensures (t := t₇) hp (fun h => by
      have e : d₈ = d₇ := h.2.2
      rw [e]; exact h.1.2.2.2.2.2.2.mono l₈)
    (fun h => (((hproof h.1.1).2.2.1).mono (L₇.trans l₈)).entries.1)
    (fun h => (((hproof h.1.1).2.2.1).mono (L₇.trans l₈)).entries.2) shape.early) fun claims d₉ m₉ l₉ => ?_
  refine (Ensures.fixWitness (X := fun x : Replay.GkrClaims × Replay.Transcript =>
      Runs (Replay.gkrVerify g.early rel.earlyShapes) t₇ x.1 x.2 ∧ IsGkrClaims c m₉ claims x.1 ∧
        IsDuplex c m₉ d₉ x.2.duplex) (⟨#[], #[], #[], #[], #[]⟩, t) fun x => ?_).pre
    fun h => ⟨h.1, let ⟨gc, t₉, hx⟩ := h.2; ⟨(gc, t₉), hx⟩⟩
  obtain ⟨gc, t₉⟩ := x
  dsimp only
  refine Ensures.pre (P' := P ∧ IsExt c m₉ beta ch.beta ∧ IsExt c m₉ lookupBeta ch.lookup.1 ∧
    IsExt c m₉ gamma ch.lookup.2 ∧ IsExt c m₉ scatterBeta ch.scatter.1 ∧ IsExt c m₉ mu ch.scatter.2 ∧
    Replay.tableSum rel.normBound g.histogram ch.beta = .ok T ∧ IsExt c m₉ total T ∧
    Runs (Replay.gkrVerify g.early rel.earlyShapes) t₇ gc t₉ ∧ IsGkrClaims c m₉ claims gc ∧
    IsDuplex c m₉ d₉ t₉.duplex) ?_ fun h => by
      obtain ⟨⟨⟨hP, hβ, hlk, hγ, hσ, hμ, -⟩, ⟨tableOk, hT⟩, -⟩, run, hc, hd₉⟩ := h
      exact ⟨hP, hβ.mono (l₈.trans l₉), hlk.mono (l₈.trans l₉), hγ.mono (l₈.trans l₉), hσ.mono (l₈.trans l₉),
        hμ.mono (l₈.trans l₉), tableOk, hT.mono l₉, run, hc, hd₉⟩
  have five : ∀ {m}, IsGkrClaims c m claims gc → Runs (Replay.gkrVerify g.early rel.earlyShapes) t₇ gc t₉ →
      claims.size = 5 := fun hc run => by
    rw [hc.1, (gkrVerify_facts shape.early run).1, rootsSize, Bridge.earlyShapes_size]
  have leaves : Runs (Replay.gkrVerify g.early rel.earlyShapes) t₇ gc t₉ → ∀ u < 5,
      (gc.leafValues[u]!).size = (rel.earlyShapes[u]!).factors + 1 :=
    fun run u hu => (gkrVerify_facts shape.early run).2 u (by rw [Bridge.earlyShapes_size]; exact hu)
  have tree : ∀ {m}, IsGkrClaims c m claims gc → claims.size = 5 → ∀ u < 5, IsTree c m claims[u]! gc u :=
    fun hc h5 u hu => ⟨(hc.2 u (by omega)).1, (hc.2 u (by omega)).2.1, (hc.2 u (by omega)).2.2.1,
      (hc.2 u (by omega)).2.2.2⟩
  refine Ensures.tbind (Ensures.lift (Ensures.extInv (ga := (gc.roots[0]!).2) fun h => by
    obtain ⟨-, -, -, -, -, -, -, -, run, hc, -⟩ := h
    exact (tree hc (five hc run) 0 (by omega)).2.1)) fun _ d₁₀ m₁₀ l₁₀ => ?_
  refine Ensures.tbindSim (Sim.lift (Sim.extMul (ga := (gc.roots[0]!).2) (gb := T) (fun h => by
      obtain ⟨⟨-, -, -, -, -, -, -, -, run, hc, -⟩, -⟩ := h
      exact (tree hc (five hc run) 0 (by omega)).2.1.mono l₁₀) fun h => by
      obtain ⟨⟨-, -, -, -, -, -, -, hT, -⟩, -⟩ := h
      exact hT.mono l₁₀)) fun expected d₁₁ m₁₁ l₁₁ => ?_
  refine Ensures.tbind (Ensures.lift (Sim.assertExtEqual (ga := (gc.roots[0]!).1) (fun h => by
      obtain ⟨⟨⟨-, -, -, -, -, -, -, -, run, hc, -⟩, -⟩, -⟩ := h
      exact (tree hc (five hc run) 0 (by omega)).1.mono (l₁₀.trans l₁₁)) fun h => h.2.1).ensures)
    fun _ d₁₂ m₁₂ l₁₂ => ?_
  refine Ensures.tbindSim (Sim.lift (Sim.extSub (fun h => by
      obtain ⟨⟨⟨⟨-, hβ, -⟩, -⟩, -⟩, -⟩ := h
      exact hβ.mono (l₁₀.trans (l₁₁.trans l₁₂))) fun h => by
      obtain ⟨⟨⟨⟨-, -, -, -, -, -, -, -, run, hc, -⟩, -⟩, -⟩, -⟩ := h
      exact ((tree hc (five hc run) 0 (by omega)).value (j := 0) (by rw [leaves run 0 (by omega)]; omega)).mono
        (l₁₀.trans (l₁₁.trans l₁₂)))) fun zLeaf d₁₃ m₁₃ l₁₃ => ?_
  have L₁₃ : m₉ ≤ m₁₃ := l₁₀.trans (l₁₁.trans (l₁₂.trans l₁₃))
  have N₁₃ : n ≤ m₁₃ := L₇.trans (l₈.trans (l₉.trans L₁₃))
  have factors : (rel.earlyShapes[1]!).factors = 0 ∧ (rel.earlyShapes[2]!).factors = 1 ∧
      (rel.earlyShapes[3]!).factors = 2 ∧ (rel.earlyShapes[4]!).factors = 1 := ⟨rfl, rfl, rfl, rfl⟩
  refine Ensures.tbind (Ensures.lift (checkEarly_ensures (gr := st.point) (ch := ch) (g := gc) (gv := g.earlyValues)
    (fun h => by
      obtain ⟨⟨⟨⟨⟨hP, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      exact ⟨(hpoint hP).1, fun i hi => ⟨((hpoint hP).2 i hi).1.mono N₁₃, ((hpoint hP).2 i hi).2.mono N₁₃⟩⟩)
    (fun h => by
      obtain ⟨⟨⟨⟨⟨-, -, hlk, hγ, hσ, hμ, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      exact ⟨hlk.mono L₁₃, hγ.mono L₁₃, hσ.mono L₁₃, hμ.mono L₁₃⟩)
    (fun h t ht => by
      obtain ⟨⟨⟨⟨⟨-, -, -, -, -, -, -, -, run, hc, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      rw [extract_one]
      exact (tree hc (five hc run) (t + 1) (by omega)).mono L₁₃)
    (fun h => by
      obtain ⟨⟨⟨⟨⟨-, -, -, -, -, -, -, -, run, -, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      rw [leaves run 1 (by omega), leaves run 2 (by omega), leaves run 3 (by omega), leaves run 4 (by omega),
        factors.1, factors.2.1, factors.2.2.1, factors.2.2.2]
      omega)
    (fun h => by
      obtain ⟨⟨⟨⟨⟨hP, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      exact (hproof hP).2.2.2.1.mono N₁₃)
    (fun h => by
      obtain ⟨⟨⟨⟨⟨hP, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      obtain ⟨size, pairs⟩ := forall₂_entries (hproof hP).2.2.2.2.1
      exact ⟨size, fun j hj => by rw [getElem!_pos _ j hj]; exact (pairs j hj).mono N₁₃⟩)))
    fun openings d₁₄ m₁₄ l₁₄ => ?_
  refine Ensures.tbindSim (observeEarly_sim (r := t₉.duplex) hp (fun h => by
      obtain ⟨⟨⟨⟨⟨⟨-, -, -, -, -, -, -, -, -, -, hd₉⟩, -, e₁₀⟩, -, e₁₁⟩, -, e₁₂⟩, -, e₁₃⟩, -, e₁₄⟩ := h
      have e : d₁₄ = d₉ := e₁₄.trans (e₁₃.trans (e₁₂.trans (e₁₁.trans e₁₀)))
      rw [e]
      exact hd₉.mono (L₁₃.trans l₁₄))
    (fun h => by
      obtain ⟨⟨⟨⟨⟨⟨hP, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      exact (hproof hP).2.2.2.1.mono (N₁₃.trans l₁₄))
    (fun h => by
      obtain ⟨⟨⟨⟨⟨⟨hP, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      exact (hproof hP).2.2.2.2.1.imp fun _ _ q => q.mono (N₁₃.trans l₁₄))
    (fun h => by
      obtain ⟨⟨⟨⟨⟨⟨hP, -⟩, -⟩, -⟩, -⟩, -⟩, -⟩ := h
      exact (hproof hP).2.2.2.2.2.1.mono (N₁₃.trans l₁₄))) fun _ d₁₅ m₁₅ l₁₅ => ?_
  rw [StateT.run_pure]
  refine Ensures.pure fun h => ?_
  obtain ⟨⟨⟨⟨⟨⟨⟨hP, -, -, -, -, -, tableOk, -, run, hc, -⟩, ⟨qne, -⟩, -⟩, -, -⟩, ⟨-, rootEq⟩, -⟩, hz, -⟩,
    ⟨checkOk, hopen⟩, -⟩, hd₁₅⟩ := h
  have N₁₅ : m₉ ≤ m₁₅ := L₁₃.trans (l₁₄.trans l₁₅)
  set words := (#[g.earlyValues.eRe, g.earlyValues.eIm, g.earlyValues.row] ++ Replay.flatPairs g.earlyValues.pairs ++
    #[g.earlyValues.block]).flatMap fun v => #[v.c0, v.c1, v.c2]
  refine ⟨⟨ch, gc, g.earlyValues, ch.beta - gc.leafValue 0 0,
      Bridge.earlyOpeningsOf rel.key.layout ch gc g.earlyValues⟩,
    { t₉ with duplex := words.foldl Replay.Duplex.observe t₉.duplex }, ?_,
    ⟨(tree hc (five hc run) 0 (by omega)).2.2.1.mono N₁₅, hz.mono (l₁₄.trans l₁₅), hopen.1.mono l₁₅,
      hopen.2.1.mono l₁₅, hopen.2.2.1.mono l₁₅, hopen.2.2.2.1.mono l₁₅, hopen.2.2.2.2.1.mono l₁₅,
      hopen.2.2.2.2.2.1.mono l₁₅, hopen.2.2.2.2.2.2.mono l₁₅⟩, hd₁₅⟩
  unfold Replay.earlyStage
  exact Replay.runs_bind (Replay.runs_observeAll _ _) (Replay.runs_bind (Replay.runs_observeAll _ _)
    (Replay.runs_bind (Replay.runs_observeAll _ _) (Replay.runs_bind (Replay.runs_sampleExt _)
    (Replay.runs_bind (Replay.runs_samplePair _) (Replay.runs_bind (Replay.runs_samplePair _)
    (Replay.runs_bind (Replay.runs_liftExcept tableOk _) (Replay.runs_bind run
    (Replay.runs_bind (Replay.runs_ensure (ext_not_isZero qne) _ _)
    (Replay.runs_bind (Replay.runs_ensure (ext_beq_of_eq rootEq) _ _)
    (Replay.runs_bind (Replay.runs_liftExcept checkOk _) (Replay.runs_bind (Replay.runs_observeExts _ _)
    (Replay.runs_pure _ _))))))))))))

end Stage

end CompressionLean.Circuit
