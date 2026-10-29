import CompressionLean.Sim.Instr
import CompressionLean.Lib.Array

/-!
Owns the simulation of the field helpers of `Algebra.lean` and of `select`
(`Ir.lean`): each program's handles are the replay's value
(`Ports/Replay/Field.lean`, `Ports/Replay/Formulas.lean`), and its assertions hold iff
the replay's equality holds.
`privateExt` leaves its proof words as raw tape cells.

An extension value's coordinates in `F` (`coords`) multiply as
`extProduct`, which is commutative and associative. A program and the
replay may multiply in different orders; their values have the same
coordinates (`IsExt.congr`).
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Coordinates -/

/-- The coordinates of `e` in `F`. -/
def coords (e : Ext) : Fin 3 → F := fun k => Word.cast (e.coord k)

theorem IsExt.congr {c : Ctx F} {n : ℕ} {a : ExtVar} {e e' : Ext} (h : IsExt c n a e)
    (he : coords (F := F) e = coords e') : IsExt c n a e' :=
  fun k => (h k).congr (congrFun he k)

theorem extProduct_comm (a b : Fin 3 → F) : extProduct a b = extProduct b a := by
  funext k
  fin_cases k <;> simp [extProduct] <;> ring

theorem extProduct_assoc (a b d : Fin 3 → F) : extProduct (extProduct a b) d = extProduct a (extProduct b d) := by
  funext k
  fin_cases k <;> simp [extProduct] <;> ring

theorem extProduct_one (a : Fin 3 → F) : extProduct a ![1, 0, 0] = a := by
  funext k
  fin_cases k <;> simp [extProduct]

section Char

variable [CharP F goldilocksModulus]

theorem coords_mul (a b : Ext) : coords (F := F) (a * b) = extProduct (coords a) (coords b) := coord_mul a b

theorem coords_add (a b : Ext) : coords (F := F) (a + b) = coords a + coords b := by
  funext k
  fin_cases k <;> exact word_add _ _

theorem coords_sub (a b : Ext) : coords (F := F) (a - b) = coords a - coords b := by
  funext k
  fin_cases k <;> exact word_sub _ _

theorem coords_scale (a : Ext) (g : Word) : coords (F := F) (a.scale g) = fun k => coords a k * Word.cast g := by
  funext k
  fin_cases k <;> exact word_mul _ _

@[simp] theorem coords_one : coords (F := F) 1 = ![1, 0, 0] := by
  funext k
  fin_cases k <;> simp [coords]

@[simp] theorem coords_zero : coords (F := F) 0 = 0 := by
  funext k
  fin_cases k <;> simp [coords]

theorem coords_ofGl (g : Word) : coords (F := F) (Ext.ofGl g) = ![Word.cast g, 0, 0] := by
  funext k
  fin_cases k <;> simp [coords, Ext.coord, Ext.ofGl]

theorem coords_injective : Function.Injective (coords (F := F)) := fun _ _ h =>
  Ext.eq_of_coord (F := F) (congrFun h)

/-- A check `a·t = 1` in `F` makes `a` nonzero and `t` the coordinates of its inverse. -/
private theorem inverse_of_check {ga : Ext} {t : Fin 3 → F} (h : extProduct (coords ga) t = ![1, 0, 0]) :
    ga ≠ 0 ∧ t = coords ga.inv := by
  have nz : ga ≠ 0 := by
    rintro rfl
    have := CharP.nontrivial_of_char_ne_one (R := F) (v := goldilocksModulus) (by decide)
    have := congrFun h 0
    simp [extProduct] at this
  refine ⟨nz, ?_⟩
  have unit : extProduct (coords (F := F) ga) (coords ga.inv) = ![1, 0, 0] := by
    rw [← coords_mul, Ext.mul_inv nz, coords_one]
  calc t = extProduct t ![1, 0, 0] := (extProduct_one t).symm
    _ = extProduct (extProduct (coords ga) t) (coords ga.inv) := by
      rw [← unit, ← extProduct_assoc, extProduct_comm t]
    _ = coords ga.inv := by rw [h, extProduct_comm, extProduct_one]

end Char

/-! ## Sums and products -/

theorem ext_mul_assoc (a b d : Ext) : a * b * d = a * (b * d) :=
  coords_injective (F := Gl) (by simp only [coords_mul, extProduct_assoc])

theorem ext_mul_one (a : Ext) : a * 1 = a :=
  coords_injective (F := Gl) (by rw [coords_mul, coords_one, extProduct_one])

theorem ext_one_mul (a : Ext) : 1 * a = a :=
  coords_injective (F := Gl) (by rw [coords_mul, coords_one, extProduct_comm, extProduct_one])

theorem gl_mul_assoc (a b d : Word) : a * b * d = a * (b * d) :=
  Word.eq_of_toF (by simp only [Word.toF_mul, mul_assoc])

theorem gl_mul_one (a : Word) : a * 1 = a := Word.eq_of_toF (by rw [Word.toF_mul, Word.toF_one, mul_one])

theorem gl_one_mul (a : Word) : 1 * a = a := Word.eq_of_toF (by rw [Word.toF_mul, Word.toF_one, one_mul])

theorem prodRange_zero (f : ℕ → Ext) : Replay.prodRange 0 f = 1 := rfl

theorem sumRange_succ (k : ℕ) (f : ℕ → Ext) : Replay.sumRange (k + 1) f = Replay.sumRange k f + f k := by
  simp only [Replay.sumRange, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem prodRange_succ (k : ℕ) (f : ℕ → Ext) : Replay.prodRange (k + 1) f = Replay.prodRange k f * f k := by
  simp only [Replay.prodRange, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem prodRangeGl_succ (k : ℕ) (f : ℕ → Word) : Replay.prodRangeGl (k + 1) f = Replay.prodRangeGl k f * f k := by
  simp only [Replay.prodRangeGl, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem prodRange_succ' (k : ℕ) (f : ℕ → Ext) :
    Replay.prodRange (k + 1) f = f 0 * Replay.prodRange k fun i => f (i + 1) := by
  induction k with
  | zero => rw [prodRange_succ, prodRange_zero, prodRange_zero, ext_one_mul, ext_mul_one]
  | succ k ih => rw [prodRange_succ, ih, prodRange_succ, ext_mul_assoc]

theorem prodRange_congr {k : ℕ} {f g : ℕ → Ext} (h : ∀ t < k, f t = g t) :
    Replay.prodRange k f = Replay.prodRange k g := by
  induction k with
  | zero => rw [prodRange_zero, prodRange_zero]
  | succ k ih => rw [prodRange_succ, prodRange_succ, ih fun t ht => h t (by omega), h k (by omega)]

theorem eqEvalList_eq : ∀ l₁ l₂ : List Ext,
    Replay.eqEvalList l₁ l₂ = Replay.prodRange (min l₁.length l₂.length) fun i => Replay.eq1 l₁[i]! l₂[i]!
  | [], _ => rfl
  | _ :: _, [] => rfl
  | a :: l₁, b :: l₂ => by
      rw [Replay.eqEvalList, eqEvalList_eq l₁ l₂, List.length_cons, List.length_cons, Nat.add_min_add_right,
        prodRange_succ']
      simp only [List.getElem!_cons_zero, List.getElem!_cons_succ]

theorem eqAt_eq (point : Array Ext) (index : ℕ) :
    Replay.eqAt point index =
      Replay.prodRange point.size fun t => if index.testBit t then point[t]! else Replay.oneMinus point[t]! := by
  suffices h : ∀ (l : List Ext) (index : ℕ), Replay.eqAtList l index =
      Replay.prodRange l.length fun t => if index.testBit t then l[t]! else Replay.oneMinus l[t]! by
    rw [Replay.eqAt, h, Array.length_toList]
    simp only [Array.getElem!_toList]
  intro l
  induction l with
  | nil => intro _; rfl
  | cons a l ih =>
      intro index
      rw [Replay.eqAtList, ih, List.length_cons, prodRange_succ']
      simp only [List.getElem!_cons_zero, List.getElem!_cons_succ, Nat.testBit_zero, Nat.testBit_succ,
        decide_eq_true_eq]

theorem powerFromBits_succ (k : ℕ) (g : Word) (index : ℕ) :
    Replay.powerFromBits g index (k + 1) =
      Replay.powerFromBits g index k * (if index / 2 ^ k % 2 = 1 then (fun g : Word => g * g)^[k] g else 1) := by
  induction k generalizing g index with
  | zero =>
      show _ * 1 = 1 * _
      rw [gl_mul_one, gl_one_mul, pow_zero, Nat.div_one]
      rfl
  | succ k ih =>
      rw [Replay.powerFromBits, ih, Replay.powerFromBits, gl_mul_assoc, Nat.div_div_eq_div_mul, ← pow_succ',
        Function.iterate_succ_apply]

theorem ext_pow_zero (a : Ext) : a.pow 0 = 1 := by
  rw [Ext.pow, ite_eq_left rfl]

/-- `a^(e+1) = a^e·a` for the replay's power by squaring. -/
theorem ext_pow_succ (a : Ext) (e : ℕ) : a.pow (e + 1) = a.pow e * a := by
  induction e using Nat.strong_induction_on generalizing a with
  | _ e ih =>
    rw [Ext.pow.eq_def a (e + 1), ite_eq_right (by omega)]
    rcases Nat.even_or_odd' e with ⟨k, rfl | rfl⟩
    · rw [ite_eq_left (by omega), show (2 * k + 1) / 2 = k by omega, Ext.pow.eq_def a (2 * k)]
      rcases k with _ | k
      · rw [ite_eq_left rfl, Ext.pow.eq_def, ite_eq_left rfl]
      · rw [ite_eq_right (by omega), ite_eq_right (by omega), show 2 * (k + 1) / 2 = k + 1 by omega]
    · rw [ite_eq_right (by omega), show (2 * k + 1 + 1) / 2 = k + 1 by omega, ih k (by omega),
        Ext.pow.eq_def a (2 * k + 1), ite_eq_right (by omega), ite_eq_left (by omega), show (2 * k + 1) / 2 = k by omega,
        ext_mul_assoc]

/-- Sums of functions that agree below `k`. -/
theorem sumRange_congr {k : ℕ} {f g : ℕ → Ext} (h : ∀ t < k, f t = g t) :
    Replay.sumRange k f = Replay.sumRange k g := by
  induction k with
  | zero => simp [Replay.sumRange]
  | succ k ih => rw [sumRange_succ, sumRange_succ, ih fun t ht => h t (by omega), h k (by omega)]

/-! ## Arrays and `K ⊗ Ext` values -/

theorem map_range_succ {α : Type} (f : ℕ → α) (j : ℕ) :
    (Array.range (j + 1)).map f = ((Array.range j).map f).push (f j) := by
  rw [Array.range_succ, ← Array.push_eq_append, Array.map_push]

/-- `xs` and `gs` have one size, and each handle is its value. -/
def IsExts (c : Ctx F) (n : ℕ) (xs : Array ExtVar) (gs : Array Ext) : Prop :=
  xs.size = gs.size ∧ ∀ i < xs.size, IsExt c n xs[i]! gs[i]!

namespace IsExts

variable {c : Ctx F} {n : ℕ} {xs : Array ExtVar} {gs : Array Ext}

theorem mono (h : IsExts c n xs gs) {m : ℕ} (le : n ≤ m := by omega) : IsExts c m xs gs :=
  ⟨h.1, fun i hi => (h.2 i hi).mono le⟩

/-- Entry `i`. -/
theorem get (h : IsExts c n xs gs) {i : ℕ} (hi : i < gs.size) : IsExt c n xs[i]! gs[i]! := h.2 i (h.1 ▸ hi)

theorem push (h : IsExts c n xs gs) {x : ExtVar} {g : Ext} (hx : IsExt c n x g) :
    IsExts c n (xs.push x) (gs.push g) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_push] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_push,
    Array.getElem_push]
  by_cases lt : i < xs.size
  · rw [dite_eq_left lt, dite_eq_left (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dite_eq_right lt, dite_eq_right (h.1 ▸ lt)]
    exact hx

theorem empty : IsExts c n #[] #[] := ⟨rfl, fun _ h => absurd h (by simp)⟩

theorem append (h : IsExts c n xs gs) {ys : Array ExtVar} {gy : Array Ext} (hy : IsExts c n ys gy) :
    IsExts c n (xs ++ ys) (gs ++ gy) := by
  refine ⟨by simp [h.1, hy.1], fun i hi => ?_⟩
  rw [Array.size_append] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1, ← hy.1]; omega), Array.getElem_append,
    Array.getElem_append]
  by_cases lt : i < xs.size
  · rw [dite_eq_left lt, dite_eq_left (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dite_eq_right lt, dite_eq_right (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos, ← h.1]
    exact hy.2 _ (by omega)

theorem extract (h : IsExts c n xs gs) (a : ℕ) : IsExts c n (xs.extract a) (gs.extract a) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

theorem reverse (h : IsExts c n xs gs) : IsExts c n xs.reverse gs.reverse := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_reverse] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_reverse,
    Array.getElem_reverse, ← getElem!_pos, ← getElem!_pos, ← h.1]
  exact h.2 _ (by omega)

/-- `k` copies of one value. -/
theorem replicate {z : ExtVar} {g : Ext} (hz : IsExt c n z g) (k : ℕ) :
    IsExts c n (Array.replicate k z) (Array.replicate k g) := by
  refine ⟨by simp, fun i hi => ?_⟩
  rw [Array.size_replicate] at hi
  rw [getElem!_pos _ _ (by simpa using hi), getElem!_pos _ _ (by simpa using hi), Array.getElem_replicate,
    Array.getElem_replicate]
  exact hz

/-- The empty prefix of a table. -/
theorem range_zero (f : ℕ → Ext) : IsExts c n #[] ((Array.range 0).map f) :=
  ⟨by simp, fun _ h => absurd h (by simp)⟩

/-- Entry `i` of a table `f`. -/
theorem get_map {k : ℕ} {f : ℕ → Ext} (h : IsExts c n xs ((Array.range k).map f)) {i : ℕ} (hi : i < k) :
    IsExt c n xs[i]! (f i) := by
  have := h.get (i := i) (by simpa using hi)
  rwa [getElem!_pos ((Array.range k).map f) i (by simpa using hi), Array.getElem_map, Array.getElem_range] at this

/-- A table of equal values. -/
theorem congr_map {k : ℕ} {f g : ℕ → Ext} (h : IsExts c n xs ((Array.range k).map f)) (e : ∀ v < k, f v = g v) :
    IsExts c n xs ((Array.range k).map g) := by
  rwa [show (Array.range k).map f = (Array.range k).map g from
    Array.ext (by simp) fun i h₁ _ => by simp at h₁; simp [e i h₁]] at h

/-- `k` copies of one value as a table. -/
theorem replicate_map {z : ExtVar} {g : Ext} (hz : IsExt c n z g) {k : ℕ} {f : ℕ → Ext} (e : ∀ v < k, f v = g) :
    IsExts c n (Array.replicate k z) ((Array.range k).map f) := by
  refine ⟨by simp, fun i hi => ?_⟩
  rw [Array.size_replicate] at hi
  rw [getElem!_pos _ _ (by simpa using hi), Array.getElem_replicate, getElem!_map_range k f hi, e i hi]
  exact hz

/-- Entry `g` of nested arrays, empty past the end. -/
theorem nested {xss : Array (Array ExtVar)} {gss : Array (Array Ext)}
    (h : xss.size = gss.size ∧ ∀ g < xss.size, IsExts c n xss[g]! gss[g]!) (g : ℕ) : IsExts c n xss[g]! gss[g]! := by
  by_cases lt : g < xss.size
  · exact h.2 g lt
  · rw [getElem!_neg xss g lt, getElem!_neg gss g (by rw [← h.1]; exact lt)]
    exact IsExts.empty

/-- The last entry. -/
theorem back (h : IsExts c n xs gs) (ne : 0 < gs.size) : IsExt c n xs.back! gs.back! := by
  have e := h.1
  rw [Array.back!, Array.back!, e]
  exact h.get (by omega)

/-- Three values. -/
theorem triple {a b d : ExtVar} {ga gb gd : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) (hd : IsExt c n d gd) :
    IsExts c n #[a, b, d] #[ga, gb, gd] :=
  ⟨rfl, fun i hi => match i, hi with
    | 0, _ => ha
    | 1, _ => hb
    | 2, _ => hd⟩

/-- A slice. -/
theorem slice (h : IsExts c n xs gs) (a b : ℕ) : IsExts c n (xs.extract a b) (gs.extract a b) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

end IsExts

/-- The two parts of a `K ⊗ Ext` value. -/
def IsKx (c : Ctx F) (n : ℕ) (a : KxVar) (v : Replay.Kx) : Prop := IsExt c n a.re v.re ∧ IsExt c n a.im v.im

theorem IsKx.mono {c : Ctx F} {n m : ℕ} {a : KxVar} {v : Replay.Kx} (h : IsKx c n a v) (le : n ≤ m := by omega) :
    IsKx c m a v :=
  ⟨h.1.mono le, h.2.mono le⟩

/-! ## Programs -/

instance extConstant_appends (e : Ext) : Appends (Algebra.extConstant e) := by
  unfold Algebra.extConstant Build.extConstant; appends
instance extZero_appends : Appends Algebra.extZero := extConstant_appends _
instance extOne_appends : Appends Algebra.extOne := extConstant_appends _
instance privateExt_appends : Appends Algebra.privateExt := by unfold Algebra.privateExt; appends
instance lift_appends (a : Var) : Appends (Algebra.lift a) := by unfold Algebra.lift; appends
instance extScaleConstant_appends (a : ExtVar) (g : Word) : Appends (Algebra.extScaleConstant a g) := by
  unfold Algebra.extScaleConstant; appends
instance oneMinus_appends (a : ExtVar) : Appends (Algebra.oneMinus a) := by unfold Algebra.oneMinus; appends
instance eq1_appends (a b : ExtVar) : Appends (Algebra.eq1 a b) := by unfold Algebra.eq1; appends
instance dot_appends (x y : Array ExtVar) : Appends (Algebra.dot x y) := by unfold Algebra.dot; appends
instance eqEval_appends (x y : Array ExtVar) : Appends (Algebra.eqEval x y) := by unfold Algebra.eqEval; appends
instance eqAt_appends (point : Array ExtVar) (index : ℕ) : Appends (Algebra.eqAt point index) := by
  unfold Algebra.eqAt; appends
instance eqTable_appends (point : Array ExtVar) : Appends (Algebra.eqTable point) := by
  unfold Algebra.eqTable; appends
instance interpolate_appends (values : Array ExtVar) (x : ExtVar) : Appends (Algebra.interpolate values x) := by
  unfold Algebra.interpolate; appends
instance horner_appends (coefficients : Array ExtVar) (x : ExtVar) : Appends (Algebra.horner coefficients x) := by
  unfold Algebra.horner; appends
instance powerFromBits_appends (g : Word) (bits : Array Var) : Appends (Algebra.powerFromBits g bits) := by
  unfold Algebra.powerFromBits; appends
instance KxVar.zero_appends : Appends Circuit.KxVar.zero := by unfold Circuit.KxVar.zero; appends
instance KxVar.one_appends : Appends Circuit.KxVar.one := by unfold Circuit.KxVar.one; appends
instance KxVar.fromK_appends (a : KVar) : Appends (Circuit.KxVar.fromK a) := by unfold Circuit.KxVar.fromK; appends
instance KxVar.add_appends (a b : KxVar) : Appends (Circuit.KxVar.add a b) := by unfold Circuit.KxVar.add; appends
instance KxVar.mul_appends (a b : KxVar) : Appends (Circuit.KxVar.mul a b) := by unfold Circuit.KxVar.mul; appends
instance KxVar.scale_appends (a : KxVar) (e : ExtVar) : Appends (Circuit.KxVar.scale a e) := by
  unfold Circuit.KxVar.scale; appends
instance KxVar.oneMinus_appends (a : KxVar) : Appends (Circuit.KxVar.oneMinus a) := by
  unfold Circuit.KxVar.oneMinus; appends

theorem ext_sub_eq_zero {a b : Ext} : a - b = 0 ↔ a = b := by
  obtain ⟨a0, a1, a2⟩ := a
  obtain ⟨b0, b1, b2⟩ := b
  show (⟨a0 - b0, a1 - b1, a2 - b2⟩ : Ext) = ⟨0, 0, 0⟩ ↔ _
  simp only [Ext.mk.injEq, gl_sub_eq_zero]

section Handles

variable {c : Ctx F} {n : ℕ}

/-- `ext_constant`. -/
theorem extConstant_ensures (e : Ext) : Ensures c n (Algebra.extConstant e) fun r m => IsExt c m r e := by
  unfold Algebra.extConstant Build.extConstant
  refine (const_ensures _).bind fun x _ _ hx => (const_ensures _).bind fun y _ _ hy =>
    (const_ensures _).bind fun z _ _ hz => Ensures.pure ?_
  intro k
  fin_cases k
  · exact hx.mono
  · exact hy.mono
  · exact hz

theorem extZero_ensures : Ensures c n Algebra.extZero fun r m => IsExt c m r 0 := extConstant_ensures 0

theorem extOne_ensures : Ensures c n Algebra.extOne fun r m => IsExt c m r 1 := extConstant_ensures 1

/-- `private_ext`: three proof words, raw tape cells. -/
theorem privateExt_ensures :
    Ensures c n Algebra.privateExt fun r m => ∀ k : Fin 3, r k = n + k ∧ r k < m ∧ c.val (r k) = c.tape (r k) := by
  unfold Algebra.privateExt
  refine priv_ensures.bind fun x m₁ _ hx => priv_ensures.bind fun y m₂ _ hy =>
    priv_ensures.bind fun z m₃ _ hz => Ensures.pure ?_
  obtain ⟨ex, e₁, vx⟩ := hx
  obtain ⟨ey, e₂, vy⟩ := hy
  obtain ⟨ez, e₃, vz⟩ := hz
  subst x m₁ y m₂ z m₃
  intro k
  fin_cases k
  · exact ⟨rfl, (by omega : n < n + 1 + 1 + 1), vx⟩
  · exact ⟨rfl, (by omega : n + 1 < n + 1 + 1 + 1), vy⟩
  · exact ⟨rfl, (by omega : n + 1 + 1 < n + 1 + 1 + 1), vz⟩

/-- `lift`: a word as an extension value. -/
theorem lift_ensures {a : Var} {g : Word} (ha : IsWord c n a g) :
    Ensures c n (Algebra.lift a) fun r m => IsExt c m r (Ext.ofGl g) := by
  unfold Algebra.lift
  refine (const_ensures 0).bind fun z _ _ hz => Ensures.pure ?_
  intro k
  fin_cases k
  · exact ha.mono
  · exact hz
  · exact hz

/-- Three coordinate calls, each of `f` at its coordinate. -/
theorem coordinatewise_ensures {f : Var → Var → Build Var} {a b : ExtVar} {e : Ext}
    (hf : ∀ (k : Fin 3) m, n ≤ m → Ensures c m (f (a k) (b k)) fun h m' => IsWord c m' h (e.coord k))
    (hfa : ∀ a b, Appends (f a b) := by appends) :
    Ensures c n (Build.coordinatewise f a b) fun r m => IsExt c m r e := by
  unfold Build.coordinatewise
  refine (hf 0 n le_rfl).bind (fun x m₁ l₁ hx => (hf 1 m₁ l₁).bind (fun y m₂ l₂ hy =>
    (hf 2 m₂ (l₁.trans l₂)).bind (fun z _ _ hz => Ensures.pure ?_) (hfa _ _) fun _ => Appends.pure _)
      (hfa _ _) fun _ => Appends.bind (hfa _ _) fun _ => Appends.pure _)
    (hfa _ _) fun _ => Appends.bind (hfa _ _) fun _ => Appends.bind (hfa _ _) fun _ => Appends.pure _
  intro k
  fin_cases k
  · exact hx.mono
  · exact hy.mono
  · exact hz

/-- `Kx::zero`. -/
theorem KxVar.zero_ensures : Ensures c n Circuit.KxVar.zero fun r m => IsKx c m r Replay.Kx.zero :=
  extZero_ensures.bind fun _ _ _ h => Ensures.pure ⟨h, h⟩

/-- `Kx::one`. -/
theorem KxVar.one_ensures : Ensures c n Circuit.KxVar.one fun r m => IsKx c m r Replay.Kx.one :=
  extOne_ensures.bind fun _ _ _ h1 => extZero_ensures.bind fun _ _ _ h0 => Ensures.pure ⟨h1.mono, h0⟩

/-- `Kx::from_k`. -/
theorem KxVar.fromK_ensures {a : KVar} {g₁ g₂ : Word} (h₁ : IsWord c n a.1 g₁) (h₂ : IsWord c n a.2 g₂) :
    Ensures c n (Circuit.KxVar.fromK a) fun r m => IsKx c m r (Replay.Kx.ofK (g₁, g₂)) :=
  (lift_ensures h₁).bind fun _ _ _ hre => (lift_ensures h₂.mono).bind fun _ _ _ him => Ensures.pure ⟨hre.mono, him⟩

end Handles

section Values

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

theorem extAdd_ensures {a b : ExtVar} {ga gb : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) :
    Ensures c n (Build.extAdd a b) fun r m => IsExt c m r (ga + gb) :=
  coordinatewise_ensures fun k m le => by
    fin_cases k
    · exact add_ensures (ha 0).mono (hb 0).mono
    · exact add_ensures (ha 1).mono (hb 1).mono
    · exact add_ensures (ha 2).mono (hb 2).mono

theorem extSub_ensures {a b : ExtVar} {ga gb : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) :
    Ensures c n (Build.extSub a b) fun r m => IsExt c m r (ga - gb) :=
  coordinatewise_ensures fun k m le => by
    fin_cases k
    · exact sub_ensures (ha 0).mono (hb 0).mono
    · exact sub_ensures (ha 1).mono (hb 1).mono
    · exact sub_ensures (ha 2).mono (hb 2).mono

/-- `ext_scale` by a word handle. -/
theorem extScale_ensures {a : ExtVar} {s : Var} {ga : Ext} {gs : Word} (ha : IsExt c n a ga) (hs : IsWord c n s gs) :
    Ensures c n (Build.extScale a s) fun r m => IsExt c m r (ga.scale gs) :=
  coordinatewise_ensures fun k m le => by
    fin_cases k
    · exact mul_ensures (ha 0).mono hs.mono
    · exact mul_ensures (ha 1).mono hs.mono
    · exact mul_ensures (ha 2).mono hs.mono

/-- `ext_scale_constant`. -/
theorem extScaleConstant_ensures {a : ExtVar} {ga : Ext} (g : Word) (ha : IsExt c n a ga) :
    Ensures c n (Algebra.extScaleConstant a g) fun r m => IsExt c m r (ga.scale g) :=
  (const_ensures g).bind fun _ _ _ hg => extScale_ensures ha.mono hg

/-- `one_minus`. -/
theorem oneMinus_ensures {a : ExtVar} {ga : Ext} (ha : IsExt c n a ga) :
    Ensures c n (Algebra.oneMinus a) fun r m => IsExt c m r (Replay.oneMinus ga) :=
  extOne_ensures.bind fun _ _ _ hone => extSub_ensures hone ha.mono

/-- `eq1`: `2xy + 1 - (x + y)`. -/
theorem eq1_ensures {a b : ExtVar} {ga gb : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) :
    Ensures c n (Algebra.eq1 a b) fun r m => IsExt c m r (Replay.eq1 ga gb) := by
  refine (extMul_ensures ha hb).bind fun both _ _ hboth => ?_
  refine (extAdd_ensures hboth hboth).bind fun twice _ _ htwice => ?_
  refine extOne_ensures.bind fun one _ _ hone => ?_
  refine (extAdd_ensures htwice.mono hone).bind fun shifted _ _ hshifted => ?_
  refine (extAdd_ensures ha.mono hb.mono).bind fun sum _ _ hsum => ?_
  exact extSub_ensures hshifted.mono hsum

/-- `select`: `whenZero + bit·(whenOne - whenZero)` is `whenOne` when the bit
is one and `whenZero` when it is zero. -/
theorem select_ensures {bit w₀ w₁ : Var} {b : Bool} {g₀ g₁ : Word} (hb : IsWord c n bit (if b then 1 else 0))
    (h₀ : IsWord c n w₀ g₀) (h₁ : IsWord c n w₁ g₁) :
    Ensures c n (Build.select bit w₀ w₁) fun r m => IsWord c m r (if b then g₁ else g₀) := by
  refine (sub_ensures h₁ h₀).bind fun d _ _ hd => (mul_ensures hb.mono hd).bind fun t _ _ ht =>
    (add_ensures h₀.mono ht).post fun r m hr => hr.congr ?_
  rw [word_add, word_mul, word_sub]
  cases b <;> simp

/-- `assert_zero` on three coordinates. -/
theorem assertExtZero_ensures {a : ExtVar} {ga : Ext} (ha : IsExt c n a ga) :
    Ensures c n (Build.assertExtZero a) fun _ _ => ga = 0 := by
  refine (assertZero_ensures (ha 0)).bind fun _ _ _ h0 => (assertZero_ensures (ha 1).mono).bind fun _ _ _ h1 =>
    (assertZero_ensures (ha 2).mono).post fun _ _ h2 => ?_
  obtain ⟨a0, a1, a2⟩ := ga
  simp only [Ext.coord] at h0 h1 h2
  subst h0 h1 h2
  rfl

theorem assertExtEqual_ensures {a b : ExtVar} {ga gb : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) :
    Ensures c n (Build.assertExtEqual a b) fun _ _ => ga = gb :=
  (extSub_ensures ha hb).bind fun _ _ _ hd => (assertExtZero_ensures hd).post fun _ _ h => ext_sub_eq_zero.mp h

/-- `ext_inverse` (`record.rs::Recorder::ext_inverse`): once its check holds, the value is
nonzero and the proof words are its inverse. -/
theorem extInv_ensures {a : ExtVar} {ga : Ext} (ha : IsExt c n a ga) :
    Ensures c n (Build.extInv a) fun r m => ga ≠ 0 ∧ IsExt c m r ga.inv :=
  (Ensures.emit (Q := fun h m => ga ≠ 0 ∧ ∀ k : Fin 3, IsWord c m (h + k) (ga.inv.coord k)) _ fun E hl hF hh => by
    have hE := Ctx.follows_of_step hF
    have ra : (fun i => E.value (a i)) = coords ga := funext fun i => (ha i).read hE hl
    have holds : E.holds ∧
        extProduct (fun i => E.value (a i)) (fun k => c.tape (E.values.length + k)) = ![1, 0, 0] := hh
    rw [ra] at holds
    obtain ⟨nz, inv⟩ := inverse_of_check holds.2
    refine ⟨nz, fun k => ⟨Nat.add_lt_add_left k.isLt n, ?_⟩⟩
    rw [← hl, Ctx.val_next hF (by simp)]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    rw [dite_eq_left k.isLt, Option.getD_some]
    exact congrFun inv k).map fun _ _ h => ⟨h.1, fun k => h.2 k⟩

/-- `dot`. -/
theorem dot_ensures {x y : Array ExtVar} {gx gy : Array Ext} (hx : IsExts c n x gx) (hy : IsExts c n y gy) :
    Ensures c n (Algebra.dot x y) fun r m => IsExt c m r (Replay.dot gx gy) := by
  unfold Algebra.dot
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  refine (Ensures.loopArray (fun k acc m => IsExt c m acc (Replay.sumRange k fun i => gx[i]! * gy[i]!)) hzero
    fun k hk acc m _ hacc => ?_).bind fun r m _ hr => Ensures.pure ?_
  · have hk' : k < x.size ∧ k < y.size := by simpa using hk
    rw [show (x.zip y)[k] = (x[k]!, y[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    refine (extMul_ensures (hx.2 k hk'.1).mono (hy.2 k hk'.2).mono).bind fun term _ _ hterm => ?_
    refine (extAdd_ensures hacc.mono hterm).bind fun total' _ _ htotal => Ensures.pure ⟨rfl, ?_⟩
    rw [sumRange_succ]
    exact htotal
  · rw [Array.size_zip, hx.1, hy.1] at hr
    exact hr

/-- `eq_eval`. -/
theorem eqEval_ensures {x y : Array ExtVar} {gx gy : Array Ext} (hx : IsExts c n x gx) (hy : IsExts c n y gy) :
    Ensures c n (Algebra.eqEval x y) fun r m => IsExt c m r (Replay.eqEval gx gy) := by
  unfold Algebra.eqEval
  refine extOne_ensures.bind fun total _ _ hone => ?_
  refine (Ensures.loopArray (fun k acc m => IsExt c m acc (Replay.prodRange k fun i => Replay.eq1 gx[i]! gy[i]!))
    hone fun k hk acc m _ hacc => ?_).bind fun r m _ hr => Ensures.pure ?_
  · have hk' : k < x.size ∧ k < y.size := by simpa using hk
    rw [show (x.zip y)[k] = (x[k]!, y[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    refine (eq1_ensures (hx.2 k hk'.1).mono (hy.2 k hk'.2).mono).bind fun factor _ _ hfactor => ?_
    refine (extMul_ensures hacc.mono hfactor).bind fun total' _ _ htotal => Ensures.pure ⟨rfl, ?_⟩
    rw [prodRange_succ]
    exact htotal
  · rw [Array.size_zip, hx.1, hy.1] at hr
    rw [Replay.eqEval, eqEvalList_eq, Array.length_toList, Array.length_toList]
    simpa only [Array.getElem!_toList] using hr

/-- `eq_at`. -/
theorem eqAt_ensures {point : Array ExtVar} {gpoint : Array Ext} (index : ℕ) (hp : IsExts c n point gpoint) :
    Ensures c n (Algebra.eqAt point index) fun r m => IsExt c m r (Replay.eqAt gpoint index) := by
  unfold Algebra.eqAt
  refine extOne_ensures.bind fun total _ _ hone => ?_
  refine (Ensures.loopArray (fun k acc m => IsExt c m acc
      (Replay.prodRange k fun t => if index.testBit t then gpoint[t]! else Replay.oneMinus gpoint[t]!)) hone
    fun k hk acc m _ hacc => ?_).bind fun r m _ hr => Ensures.pure ?_
  · have hk' : k < point.size := by simpa using hk
    rw [show point.zipIdx[k] = (point[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    split
    · rename_i hb
      rw [pure_bind]
      exact (extMul_ensures hacc (hp.2 k hk').mono).bind fun total' _ _ htotal =>
        Ensures.pure ⟨rfl, by rw [prodRange_succ, ite_eq_left hb]; exact htotal⟩
    · rename_i hb
      exact (oneMinus_ensures (hp.2 k hk').mono).bind fun factor _ _ hfactor =>
        (extMul_ensures hacc.mono hfactor).bind fun total' _ _ htotal =>
        Ensures.pure ⟨rfl, by rw [prodRange_succ, ite_eq_right hb]; exact htotal⟩
  · rw [Array.size_zipIdx, hp.1] at hr
    rw [eqAt_eq]
    exact hr

/-- `horner`. -/
theorem horner_ensures {coefficients : Array ExtVar} {x : ExtVar} {gcoefficients : Array Ext} {gx : Ext}
    (hc : IsExts c n coefficients gcoefficients) (hx : IsExt c n x gx) :
    Ensures c n (Algebra.horner coefficients x) fun r m => IsExt c m r (Replay.horner gcoefficients gx) := by
  unfold Algebra.horner
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  refine (Ensures.loopArray
    (fun k acc m => IsExt c m acc ((gcoefficients.reverse.toList.take k).foldl (fun t c => t * gx + c) 0)) hzero
    fun k hk acc m _ hacc => ?_).bind fun r m _ hr => Ensures.pure ?_
  · have hcoef : IsExt c n coefficients.reverse[k] gcoefficients.reverse[k]! := by
      have := hc.reverse.2 k hk
      rwa [getElem!_pos coefficients.reverse k hk] at this
    refine (extMul_ensures hacc hx.mono).bind fun scaled _ _ hscaled => ?_
    refine (extAdd_ensures hscaled hcoef.mono).bind fun total' _ _ htotal => Ensures.pure ⟨rfl, ?_⟩
    have hk' : k < gcoefficients.reverse.toList.length := by simpa [← hc.1] using hk
    rw [List.take_succ_eq_append_getElem hk', List.foldl_append, List.foldl_cons, List.foldl_nil,
      Array.getElem_toList, ← getElem!_pos]
    exact htotal
  · rw [List.take_of_length_le (by simp [hc.1])] at hr
    rw [Replay.horner, Array.foldr_eq_foldl_reverse, ← Array.foldl_toList]
    exact hr

/-- `power_from_bits`: bit `i` of `index` selects `generator^(2^i)`. -/
theorem powerFromBits_ensures (generator : Word) {bits : Array Var} (index : ℕ)
    (hb : ∀ i < bits.size, IsWord c n bits[i]! (Word.ofNat (index / 2 ^ i % 2))) :
    Ensures c n (Algebra.powerFromBits generator bits)
      fun r m => IsWord c m r (Replay.powerFromBits generator index bits.size) := by
  unfold Algebra.powerFromBits
  refine (const_ensures 1).bind fun total _ _ hone => ?_
  refine (Ensures.loopArray (fun k acc m => IsWord c m acc.1 (Replay.powerFromBits generator index k) ∧
      acc.2 = (fun g : Word => g * g)^[k] generator) ⟨hone, rfl⟩
    fun k hk acc m _ hacc => ?_).bind fun r m _ hr => Ensures.pure hr.1
  obtain ⟨htotal, hbase⟩ := hacc
  have hbit : IsWord c n bits[k] (Word.ofNat (index / 2 ^ k % 2)) := by
    have := hb k hk
    rwa [getElem!_pos bits k hk] at this
  refine (const_ensures 1).bind fun one _ _ hone' => ?_
  refine (scale_ensures (acc.2 - 1) hbit.mono).bind fun fm _ _ hfm => ?_
  refine (add_ensures hone'.mono hfm).bind fun factor _ _ hfactor => ?_
  refine (mul_ensures htotal.mono hfactor).bind fun total' _ _ htotal' => Ensures.pure ⟨rfl, ?_, ?_⟩
  · have e : Replay.powerFromBits generator index k * (1 + Word.ofNat (index / 2 ^ k % 2) * (acc.2 - 1)) =
        Replay.powerFromBits generator index (k + 1) := by
      have select : 1 + Word.ofNat (index / 2 ^ k % 2) * (acc.2 - 1) =
          if index / 2 ^ k % 2 = 1 then acc.2 else 1 := by
        apply Word.eq_of_toF
        rcases Nat.mod_two_eq_zero_or_one (index / 2 ^ k) with hb | hb <;>
          simp [hb, Word.toF_add, Word.toF_mul, Word.toF_sub, Word.toF_ofNat]
      rw [select, powerFromBits_succ, hbase]
    exact e ▸ htotal'
  · rw [hbase, Function.iterate_succ_apply']
    rfl

/-- `interpolate`. -/
theorem interpolate_ensures {values : Array ExtVar} {x : ExtVar} {gvalues : Array Ext} {gx : Ext}
    (hv : IsExts c n values gvalues) (hx : IsExt c n x gx) :
    Ensures c n (Algebra.interpolate values x) fun r m => IsExt c m r (Replay.interpolate gvalues gx) := by
  unfold Algebra.interpolate
  dsimp only
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  refine (Ensures.loopArray (fun k acc m => IsExt c m acc (Replay.sumRange k fun i =>
      gvalues[i]! * (Replay.prodRange gvalues.size fun j => if i = j then 1 else gx - Ext.ofGl (Word.ofNat j)).scale
        (Replay.prodRangeGl gvalues.size fun j => if i = j then 1 else Word.ofNat i - Word.ofNat j).inv)) hzero
    fun k hk acc m _ hacc => ?_).bind fun r m _ hr => Ensures.pure ?_
  · have hk' : k < values.size := by simpa using hk
    rw [show values.zipIdx[k] = (values[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine extOne_ensures.bind fun numerator _ _ hone => ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range', ← List.forIn_toArray]
    refine (Ensures.loopArray
      (fun j acc' m' =>
        IsExt c m' acc'.1 (Replay.prodRange j fun j => if k = j then 1 else gx - Ext.ofGl (Word.ofNat j)) ∧
        acc'.2 = Replay.prodRangeGl j fun j => if k = j then 1 else Word.ofNat k - Word.ofNat j) ⟨hone, rfl⟩
      fun j hj acc' m' _ hacc' => ?_).bind fun r' _ _ hr' => ?_
    · obtain ⟨hnum, hden⟩ := hacc'
      rw [show (List.range' 0 [0:values.size].size 1).toArray[j] = j by simp]
      split
      · rename_i hkj
        refine (extConstant_ensures _).bind fun node _ _ hnode => ?_
        refine (extSub_ensures hx.mono hnode).bind fun factor _ _ hfactor => ?_
        refine (extMul_ensures hnum.mono hfactor).bind fun num _ _ hnum' => Ensures.pure ⟨rfl, ?_, ?_⟩
        · rw [prodRange_succ, ite_eq_right hkj]
          exact hnum'
        · rw [prodRangeGl_succ, ite_eq_right hkj, ← hden]
          rfl
      · rename_i hkj
        exact Ensures.pure ⟨rfl, by rw [prodRange_succ, ite_eq_left (not_not.mp hkj), ext_mul_one]; exact hnum,
          by rw [prodRangeGl_succ, ite_eq_left (not_not.mp hkj), gl_mul_one]; exact hden⟩
    · obtain ⟨hnum, hden⟩ := hr'
      have hlen : (List.range' 0 [0:values.size].size 1).toArray.size = gvalues.size := by simp [hv.1]
      refine (extScaleConstant_ensures r'.2.inv hnum).bind fun weight _ _ hweight => ?_
      refine (extMul_ensures (hv.2 k hk').mono hweight).bind fun term _ _ hterm => ?_
      refine (extAdd_ensures hacc.mono hterm).bind fun total' _ _ htotal => Ensures.pure ⟨rfl, ?_⟩
      rw [hden, hlen] at htotal
      rw [sumRange_succ]
      exact htotal
  · rw [Array.size_zipIdx, hv.1] at hr
    exact hr

/-- `eq_table`. -/
theorem eqTable_ensures {point : Array ExtVar} {gpoint : Array Ext} (hp : IsExts c n point gpoint) :
    Ensures c n (Algebra.eqTable point) fun r m => IsExts c m r (Replay.eqTable gpoint) := by
  obtain ⟨E, hE⟩ : ∃ E : ℕ → ℕ → Ext,
      E = fun k i => Replay.prodRange k fun t => if i.testBit t then gpoint[t]! else Replay.oneMinus gpoint[t]! :=
    ⟨_, rfl⟩
  unfold Algebra.eqTable
  refine extOne_ensures.bind fun one _ _ hone => ?_
  dsimp only
  refine (Ensures.loopArray (fun k acc m => IsExts c m acc ((Array.range (2 ^ k)).map (E k))) ?_
    fun k hk acc m _ hacc => ?_).bind fun r m _ hr => Ensures.pure ?_
  · refine ⟨by simp, fun i hi => ?_⟩
    have i0 : i = 0 := by simpa using hi
    subst i0
    rw [getElem!_pos _ 0 (by simp), getElem!_pos _ 0 (by simp), Array.getElem_map, Array.getElem_range, hE]
    simp only [prodRange_zero]
    exact hone
  · have hx : IsExt c n point[k] gpoint[k]! := by
      have := hp.2 k hk
      rwa [getElem!_pos point k hk] at this
    have hsize : acc.size = 2 ^ k := by simpa using hacc.1
    have hval : ∀ j (hj : j < acc.size), IsExt c m acc[j] (E k j) := fun j hj => by
      have := hacc.2 j hj
      rwa [getElem!_pos acc j hj, getElem!_pos _ j (by simpa [hsize] using hj), Array.getElem_map,
        Array.getElem_range] at this
    obtain ⟨H, hH⟩ : ∃ H : ℕ → Ext,
        H = fun i => if i < 2 ^ k then E k i * Replay.oneMinus gpoint[k]! else E k (i - 2 ^ k) * gpoint[k]! :=
      ⟨_, rfl⟩
    refine (oneMinus_ensures hx.mono).bind fun low _ _ hlow => ?_
    refine (Ensures.loopArray (fun j next m' => IsExts c m' next ((Array.range j).map H)) (by simp [IsExts])
      fun j hj next m' _ hnext => ?_).bind fun next₁ _ _ hnext₁ => ?_
    · refine (extMul_ensures (hval j hj).mono hlow.mono).bind fun v _ _ hv => Ensures.pure ⟨rfl, ?_⟩
      rw [map_range_succ]
      refine hnext.mono.push ?_
      rw [hH]
      dsimp only
      rw [ite_eq_left (hsize ▸ hj)]
      exact hv
    · refine (Ensures.loopArray (fun j next m' => IsExts c m' next ((Array.range (acc.size + j)).map H)) hnext₁
        fun j hj next m' _ hnext => ?_).bind fun next₂ _ _ hnext₂ => Ensures.pure ⟨rfl, ?_⟩
      · refine (extMul_ensures (hval j hj).mono hx.mono).bind fun v _ _ hv => Ensures.pure ⟨rfl, ?_⟩
        rw [← Nat.add_assoc, map_range_succ]
        refine hnext.mono.push ?_
        rw [hH]
        dsimp only
        rw [ite_eq_right (by rw [hsize]; omega), hsize, Nat.add_sub_cancel_left]
        exact hv
      · rw [hsize, show 2 ^ k + 2 ^ k = 2 ^ (k + 1) by rw [Nat.pow_succ]; omega] at hnext₂
        have key : (Array.range (2 ^ (k + 1))).map H = (Array.range (2 ^ (k + 1))).map (E (k + 1)) :=
          Array.map_inj_left.mpr fun i hi => by
            rw [Array.mem_range] at hi
            rw [hH, hE]
            dsimp only
            rw [prodRange_succ]
            by_cases lt : i < 2 ^ k
            · rw [ite_eq_left lt, Nat.testBit_lt_two_pow lt, ite_eq_right Bool.false_ne_true]
            · obtain ⟨j, rfl⟩ : ∃ j, i = 2 ^ k + j := ⟨i - 2 ^ k, by omega⟩
              have hj : j < 2 ^ k := by rw [Nat.pow_succ] at hi; omega
              rw [ite_eq_right lt, Nat.add_sub_cancel_left, Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow hj,
                Bool.not_false, ite_eq_left rfl]
              exact congrArg (· * gpoint[k]!) (prodRange_congr fun t ht => by rw [Nat.testBit_two_pow_add_gt ht])
        rw [key] at hnext₂
        exact hnext₂
  · rw [hp.1] at hr
    rw [Replay.eqTable, show Replay.eqAt gpoint = E gpoint.size from funext fun i => by rw [eqAt_eq, hE]]
    exact hr

/-- `Kx::add`. -/
theorem KxVar.add_ensures {a b : KxVar} {ga gb : Replay.Kx} (ha : IsKx c n a ga) (hb : IsKx c n b gb) :
    Ensures c n (Circuit.KxVar.add a b) fun r m => IsKx c m r (ga.add gb) :=
  (extAdd_ensures ha.1 hb.1).bind fun _ _ _ hre => (extAdd_ensures ha.2.mono hb.2.mono).bind fun _ _ _ him =>
    Ensures.pure ⟨hre.mono, him⟩

/-- `Kx::mul`. -/
theorem KxVar.mul_ensures {a b : KxVar} {ga gb : Replay.Kx} (ha : IsKx c n a ga) (hb : IsKx c n b gb) :
    Ensures c n (Circuit.KxVar.mul a b) fun r m => IsKx c m r (ga.mul gb) := by
  refine (extMul_ensures ha.1 hb.1).bind fun rr _ _ hrr => ?_
  refine (extMul_ensures ha.2.mono hb.2.mono).bind fun ii _ _ hii => ?_
  refine (extScaleConstant_ensures 7 hii).bind fun seven _ _ hseven => ?_
  refine (extMul_ensures ha.1.mono hb.2.mono).bind fun ri _ _ hri => ?_
  refine (extMul_ensures ha.2.mono hb.1.mono).bind fun ir _ _ hir => ?_
  refine (extAdd_ensures hrr.mono hseven.mono).bind fun re _ _ hre => ?_
  exact (extAdd_ensures hri.mono hir.mono).bind fun im _ _ him => Ensures.pure ⟨hre.mono, him⟩

/-- `Kx::scale`. -/
theorem KxVar.scale_ensures {a : KxVar} {e : ExtVar} {ga : Replay.Kx} {ge : Ext} (ha : IsKx c n a ga)
    (he : IsExt c n e ge) : Ensures c n (Circuit.KxVar.scale a e) fun r m => IsKx c m r (ga.scale ge) :=
  (extMul_ensures ha.1 he).bind fun _ _ _ hre => (extMul_ensures ha.2.mono he.mono).bind fun _ _ _ him =>
    Ensures.pure ⟨hre.mono, him⟩

/-- `Kx::one_minus`. -/
theorem KxVar.oneMinus_ensures {a : KxVar} {ga : Replay.Kx} (ha : IsKx c n a ga) :
    Ensures c n (Circuit.KxVar.oneMinus a) fun r m => IsKx c m r ga.oneMinus :=
  (Sim.oneMinus_ensures ha.1).bind fun _ _ _ hre => extZero_ensures.bind fun _ _ _ hzero =>
    (extSub_ensures hzero ha.2.mono).bind fun _ _ _ him => Ensures.pure ⟨hre.mono, him⟩

end Values

end CompressionLean.Sim
