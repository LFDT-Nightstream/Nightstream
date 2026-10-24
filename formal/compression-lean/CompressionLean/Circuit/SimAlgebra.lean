import CompressionLean.Circuit.Sim

/-!
Owns the simulation of the field helpers of `Algebra.lean` and of `select`
(`Ir.lean`): each program's handles are the replay's value
(`Replay/Field.lean`, `Replay/Formulas.lean`), and its assertions hold iff
the replay's equality holds.
`privateExt` leaves its proof words as raw tape cells.

An extension value's coordinates in `F` (`coords`) multiply as
`extProduct`, which is commutative and associative. A program and the
replay may multiply in different orders; their values have the same
coordinates (`IsExt.congr`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Lowering
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

/-! ## Arrays and `K ⊗ Ext` values -/

theorem map_range_succ {α : Type} (f : ℕ → α) (j : ℕ) :
    (Array.range (j + 1)).map f = ((Array.range j).map f).push (f j) := by
  rw [Array.range_succ, ← Array.push_eq_append, Array.map_push]

/-- `xs` and `gs` have one size, and each handle is its value. -/
def IsExts (c : Ctx F) (n : ℕ) (xs : Array ExtVar) (gs : Array Ext) : Prop :=
  xs.size = gs.size ∧ ∀ i < xs.size, IsExt c n xs[i]! gs[i]!

namespace IsExts

variable {c : Ctx F} {n : ℕ} {xs : Array ExtVar} {gs : Array Ext}

theorem mono (h : IsExts c n xs gs) {m : ℕ} (le : n ≤ m) : IsExts c m xs gs :=
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
  · rw [dif_pos lt, dif_pos (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dif_neg lt, dif_neg (h.1 ▸ lt)]
    exact hx

theorem reverse (h : IsExts c n xs gs) : IsExts c n xs.reverse gs.reverse := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_reverse] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_reverse,
    Array.getElem_reverse, ← getElem!_pos, ← getElem!_pos, ← h.1]
  exact h.2 _ (by omega)

end IsExts

/-- The two parts of a `K ⊗ Ext` value. -/
def IsKx (c : Ctx F) (n : ℕ) (a : KxVar) (v : Replay.Kx) : Prop := IsExt c n a.re v.re ∧ IsExt c n a.im v.im

theorem IsKx.mono {c : Ctx F} {n m : ℕ} {a : KxVar} {v : Replay.Kx} (h : IsKx c n a v) (le : n ≤ m) : IsKx c m a v :=
  ⟨h.1.mono le, h.2.mono le⟩

/-! ## Programs -/

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

omit [CharP F goldilocksModulus] in
/-- A loop over an array whose steps check nothing. -/
theorem loop₀ {δ γ : Type} {xs : Array δ} {init : γ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.size) acc m, n ≤ m → Sim c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') True) :
    Sim c n (P ∧ I 0 init n) (forIn xs init body) (fun r m => I xs.size r m) True := by
  rw [← Array.forIn_toList]
  exact (loop I (fun _ => True) step).post (fun _ _ _ h => h) fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩

omit [CharP F goldilocksModulus] in
/-- `ext_constant`. -/
theorem extConstant (e : Ext) : Sim c n P (Algebra.extConstant e) (fun r m => IsExt c m r e) True := by
  refine ((Sim.const _).bind fun x m₁ l₁ => (Sim.const _).bind fun y m₂ l₂ =>
    (Sim.const _).bind fun z m₃ l₃ => Sim.pure fun h => ?_).post (fun _ _ _ h => h) (fun _ => by simp)
  obtain ⟨⟨⟨-, hx⟩, hy⟩, hz⟩ := h
  intro k
  fin_cases k
  · exact hx.mono (l₂.trans l₃)
  · exact hy.mono l₃
  · exact hz

omit [CharP F goldilocksModulus] in
theorem extZero : Sim c n P Algebra.extZero (fun r m => IsExt c m r 0) True := extConstant 0

omit [CharP F goldilocksModulus] in
theorem extOne : Sim c n P Algebra.extOne (fun r m => IsExt c m r 1) True := extConstant 1

omit [CharP F goldilocksModulus] in
/-- `private_ext`: three proof words, raw tape cells. -/
theorem privateExt : Sim c n P Algebra.privateExt
    (fun r m => ∀ k : Fin 3, r k = n + k ∧ r k < m ∧ c.val (r k) = c.tape (r k)) True := by
  have priv : ∀ {n' : ℕ} {P' : Prop},
      Sim c n' P' Build.priv (fun h m => h = n' ∧ m = n' + 1 ∧ c.val h = c.tape h) True :=
    fun {n'} {P'} => emit _ fun _ E hl hF => by
      refine ⟨⟨rfl, rfl, ?_⟩, Iff.rfl⟩
      rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
      rfl
  refine (priv.bind₀ fun x m₁ _ => priv.bind₀ fun y m₂ _ => priv.bind₀ fun z m₃ _ => Sim.pure fun h => ?_)
  obtain ⟨⟨⟨-, ex, e₁, vx⟩, ey, e₂, vy⟩, ez, e₃, vz⟩ := h
  subst x m₁ y m₂ z m₃
  intro k
  fin_cases k
  · exact ⟨rfl, (by omega : n < n + 1 + 1 + 1), vx⟩
  · exact ⟨rfl, (by omega : n + 1 < n + 1 + 1 + 1), vy⟩
  · exact ⟨rfl, (by omega : n + 1 + 1 < n + 1 + 1 + 1), vz⟩

omit [CharP F goldilocksModulus] in
/-- `lift`: a word as an extension value. -/
theorem liftWord {a : Var} {g : Word} (ha : P → IsWord c n a g) :
    Sim c n P (Algebra.lift a) (fun r m => IsExt c m r (Ext.ofGl g)) True := by
  refine ((Sim.const 0).bind fun z m₁ l₁ => Sim.pure fun h => ?_).post (fun _ _ _ h => h) (fun _ => by simp)
  obtain ⟨hP, hz⟩ := h
  intro k
  fin_cases k
  · exact (ha hP).mono l₁
  · exact hz
  · exact hz

omit [CharP F goldilocksModulus] in
/-- Three coordinate calls, each of `f` at its coordinate. -/
theorem coordinatewise {f : Var → Var → Build Var} {a b : ExtVar} {e : Ext}
    (hf : ∀ (k : Fin 3) {m P'}, (P' → P) → n ≤ m → Sim c m P' (f (a k) (b k)) (fun h m' => IsWord c m' h (e.coord k)) True) :
    Sim c n P (Build.coordinatewise f a b) (fun r m => IsExt c m r e) True := by
  refine ((hf 0 id le_rfl).bind fun x m₁ l₁ => (hf 1 (fun h => h.1) l₁).bind fun y m₂ l₂ =>
    (hf 2 (fun h => h.1.1) (l₁.trans l₂)).bind fun z m₃ l₃ => Sim.pure fun h => ?_).post (fun _ _ _ h => h)
    (fun _ => by simp)
  obtain ⟨⟨⟨-, hx⟩, hy⟩, hz⟩ := h
  intro k
  fin_cases k
  · exact hx.mono (l₂.trans l₃)
  · exact hy.mono l₃
  · exact hz

theorem extAdd {a b : ExtVar} {ga gb : Ext} (ha : P → IsExt c n a ga) (hb : P → IsExt c n b gb) :
    Sim c n P (Build.extAdd a b) (fun r m => IsExt c m r (ga + gb)) True :=
  coordinatewise fun k _ _ hp le => by
    fin_cases k
    · exact Sim.add (fun h => (ha (hp h) 0).mono le) (fun h => (hb (hp h) 0).mono le)
    · exact Sim.add (fun h => (ha (hp h) 1).mono le) (fun h => (hb (hp h) 1).mono le)
    · exact Sim.add (fun h => (ha (hp h) 2).mono le) (fun h => (hb (hp h) 2).mono le)

theorem extSub {a b : ExtVar} {ga gb : Ext} (ha : P → IsExt c n a ga) (hb : P → IsExt c n b gb) :
    Sim c n P (Build.extSub a b) (fun r m => IsExt c m r (ga - gb)) True :=
  coordinatewise fun k _ _ hp le => by
    fin_cases k
    · exact Sim.sub (fun h => (ha (hp h) 0).mono le) (fun h => (hb (hp h) 0).mono le)
    · exact Sim.sub (fun h => (ha (hp h) 1).mono le) (fun h => (hb (hp h) 1).mono le)
    · exact Sim.sub (fun h => (ha (hp h) 2).mono le) (fun h => (hb (hp h) 2).mono le)

/-- `ext_scale` by a word handle. -/
theorem extScale {a : ExtVar} {s : Var} {ga : Ext} {gs : Word} (ha : P → IsExt c n a ga) (hs : P → IsWord c n s gs) :
    Sim c n P (Build.extScale a s) (fun r m => IsExt c m r (ga.scale gs)) True :=
  coordinatewise fun k _ _ hp le => by
    fin_cases k
    · exact Sim.mul (fun h => (ha (hp h) 0).mono le) (fun h => (hs (hp h)).mono le)
    · exact Sim.mul (fun h => (ha (hp h) 1).mono le) (fun h => (hs (hp h)).mono le)
    · exact Sim.mul (fun h => (ha (hp h) 2).mono le) (fun h => (hs (hp h)).mono le)

/-- `ext_scale_constant`. -/
theorem extScaleConstant {a : ExtVar} {ga : Ext} (g : Word) (ha : P → IsExt c n a ga) :
    Sim c n P (Algebra.extScaleConstant a g) (fun r m => IsExt c m r (ga.scale g)) True :=
  ((Sim.const g).bind fun _ _ le => extScale (fun h => (ha h.1).mono le) fun h => h.2).post (fun _ _ _ h => h)
    (fun _ => by simp)

/-- `one_minus`. -/
theorem oneMinus {a : ExtVar} {ga : Ext} (ha : P → IsExt c n a ga) :
    Sim c n P (Algebra.oneMinus a) (fun r m => IsExt c m r (Replay.oneMinus ga)) True :=
  (extOne.bind fun _ _ le => extSub (fun h => h.2) fun h => (ha h.1).mono le).post (fun _ _ _ h => h)
    (fun _ => by simp)

/-- `eq1`: `2xy + 1 - (x + y)`. -/
theorem eq1 {a b : ExtVar} {ga gb : Ext} (ha : P → IsExt c n a ga) (hb : P → IsExt c n b gb) :
    Sim c n P (Algebra.eq1 a b) (fun r m => IsExt c m r (Replay.eq1 ga gb)) True := by
  refine ((Sim.extMul ha hb).bind fun both m₁ l₁ =>
    (extAdd (fun h => h.2) fun h => h.2).bind fun twice m₂ l₂ =>
    extOne.bind fun one m₃ l₃ =>
    (extAdd (fun h => h.1.2.mono l₃) fun h => h.2).bind fun shifted m₄ l₄ =>
    (extAdd (fun h => (ha h.1.1.1.1).mono (l₁.trans (l₂.trans (l₃.trans l₄))))
      fun h => (hb h.1.1.1.1).mono (l₁.trans (l₂.trans (l₃.trans l₄)))).bind fun sum m₅ l₅ =>
    extSub (fun h => h.1.2.mono l₅) fun h => h.2).post (fun _ _ _ h => h) (fun _ => by simp)

/-- `select`: `whenZero + bit·(whenOne - whenZero)` is `whenOne` when the bit
is one and `whenZero` when it is zero. -/
theorem select {bit w₀ w₁ : Var} {b : Bool} {g₀ g₁ : Word} (hb : P → IsWord c n bit (if b then 1 else 0))
    (h₀ : P → IsWord c n w₀ g₀) (h₁ : P → IsWord c n w₁ g₁) :
    Sim c n P (Build.select bit w₀ w₁) (fun r m => IsWord c m r (if b then g₁ else g₀)) True := by
  refine ((Sim.sub h₁ h₀).bind fun d m₁ l₁ => (Sim.mul (fun h => (hb h.1).mono l₁) fun h => h.2).bind
    fun t m₂ l₂ => Sim.add (fun h => (h₀ h.1.1).mono (l₁.trans l₂)) fun h => h.2).post
    (fun _ _ _ h => h.congr ?_) (fun _ => by simp)
  rw [word_add, word_mul, word_sub]
  cases b <;> simp

/-- `assert_zero` on three coordinates. -/
theorem assertExtZero {a : ExtVar} {ga : Ext} (ha : P → IsExt c n a ga) :
    Sim c n P (Build.assertExtZero a) (fun _ _ => True) (ga = 0) := by
  refine ((Sim.assertZero fun h => ha h 0).bind fun _ m₁ l₁ =>
    (Sim.assertZero fun h => (ha h.1 1).mono l₁).bind fun _ m₂ l₂ =>
    Sim.assertZero fun h => (ha h.1.1 2).mono (l₁.trans l₂)).post (fun _ _ _ h => h) fun _ => ?_
  obtain ⟨a0, a1, a2⟩ := ga
  exact ⟨fun ⟨h0, h1, h2⟩ => by simp only [Ext.coord] at h0 h1 h2; subst h0 h1 h2; rfl,
    fun h => by cases h; exact ⟨rfl, rfl, rfl⟩⟩

theorem ext_sub_eq_zero {a b : Ext} : a - b = 0 ↔ a = b := by
  obtain ⟨a0, a1, a2⟩ := a
  obtain ⟨b0, b1, b2⟩ := b
  show (⟨a0 - b0, a1 - b1, a2 - b2⟩ : Ext) = ⟨0, 0, 0⟩ ↔ _
  simp only [Ext.mk.injEq, gl_sub_eq_zero]

theorem assertExtEqual {a b : ExtVar} {ga gb : Ext} (ha : P → IsExt c n a ga) (hb : P → IsExt c n b gb) :
    Sim c n P (Build.assertExtEqual a b) (fun _ _ => True) (ga = gb) :=
  ((extSub ha hb).bind fun _ _ _ => assertExtZero fun h => h.2).post (fun _ _ _ h => h)
    (fun _ => by rw [true_and, ext_sub_eq_zero])

/-- `dot`. -/
theorem dot {x y : Array ExtVar} {gx gy : Array Ext} (hx : P → IsExts c n x gx) (hy : P → IsExts c n y gy) :
    Sim c n P (Algebra.dot x y) (fun r m => IsExt c m r (Replay.dot gx gy)) True := by
  unfold Algebra.dot
  refine extZero.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 0)
    (fun k acc m => IsExt c m acc (Replay.sumRange k fun i => gx[i]! * gy[i]!))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hk' : k < x.size ∧ k < y.size := by simpa using hk
    rw [show (x.zip y)[k] = (x[k]!, y[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    exact (Sim.extMul (fun h => ((hx h.1.1).2 k hk'.1).mono (l₁.trans lm))
      fun h => ((hy h.1.1).2 k hk'.2).mono (l₁.trans lm)).bind₀ fun term m₂ l₂ =>
      (extAdd (fun h => h.1.2.mono l₂) fun h => h.2).bind₀ fun total' m₃ l₃ =>
      Sim.pure fun h => ⟨rfl, by rw [sumRange_succ]; exact h.2⟩
  · have e := h.2
    rw [Array.size_zip, (hx h.1.1).1, (hy h.1.1).1] at e
    exact e

/-- `eq_eval`. -/
theorem eqEval {x y : Array ExtVar} {gx gy : Array Ext} (hx : P → IsExts c n x gx) (hy : P → IsExts c n y gy) :
    Sim c n P (Algebra.eqEval x y) (fun r m => IsExt c m r (Replay.eqEval gx gy)) True := by
  unfold Algebra.eqEval
  refine extOne.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 1)
    (fun k acc m => IsExt c m acc (Replay.prodRange k fun i => Replay.eq1 gx[i]! gy[i]!))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hk' : k < x.size ∧ k < y.size := by simpa using hk
    rw [show (x.zip y)[k] = (x[k]!, y[k]!) by simp [getElem!_pos, hk'.1, hk'.2]]
    exact (Sim.eq1 (fun h => ((hx h.1.1).2 k hk'.1).mono (l₁.trans lm))
      fun h => ((hy h.1.1).2 k hk'.2).mono (l₁.trans lm)).bind₀ fun factor m₂ l₂ =>
      (Sim.extMul (fun h => h.1.2.mono l₂) fun h => h.2).bind₀ fun total' m₃ l₃ =>
      Sim.pure fun h => ⟨rfl, by rw [prodRange_succ]; exact h.2⟩
  · have e := h.2
    rw [Array.size_zip, (hx h.1.1).1, (hy h.1.1).1] at e
    rw [Replay.eqEval, eqEvalList_eq, Array.length_toList, Array.length_toList]
    simpa only [Array.getElem!_toList] using e

/-- `eq_at`. -/
theorem eqAt {point : Array ExtVar} {gpoint : Array Ext} (index : ℕ) (hp : P → IsExts c n point gpoint) :
    Sim c n P (Algebra.eqAt point index) (fun r m => IsExt c m r (Replay.eqAt gpoint index)) True := by
  unfold Algebra.eqAt
  refine extOne.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 1)
    (fun k acc m => IsExt c m acc
      (Replay.prodRange k fun t => if index.testBit t then gpoint[t]! else Replay.oneMinus gpoint[t]!))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hk' : k < point.size := by simpa using hk
    rw [show point.zipIdx[k] = (point[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    split
    · rename_i hb
      rw [pure_bind]
      exact (Sim.extMul (fun h => h.2) fun h => ((hp h.1.1).2 k hk').mono (l₁.trans lm)).bind₀ fun total' m₂ l₂ =>
        Sim.pure fun h => ⟨rfl, by rw [prodRange_succ, if_pos hb]; exact h.2⟩
    · rename_i hb
      exact (Sim.oneMinus fun h => ((hp h.1.1).2 k hk').mono (l₁.trans lm)).bind₀ fun factor m₂ l₂ =>
        (Sim.extMul (fun h => h.1.2.mono l₂) fun h => h.2).bind₀ fun total' m₃ l₃ =>
        Sim.pure fun h => ⟨rfl, by rw [prodRange_succ, if_neg hb]; exact h.2⟩
  · have e := h.2
    rw [Array.size_zipIdx, (hp h.1.1).1] at e
    rw [eqAt_eq]
    exact e

/-- `horner`. -/
theorem horner {coefficients : Array ExtVar} {x : ExtVar} {gcoefficients : Array Ext} {gx : Ext}
    (hc : P → IsExts c n coefficients gcoefficients) (hx : P → IsExt c n x gx) :
    Sim c n P (Algebra.horner coefficients x) (fun r m => IsExt c m r (Replay.horner gcoefficients gx)) True := by
  unfold Algebra.horner
  refine extZero.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 0)
    (fun k acc m => IsExt c m acc ((gcoefficients.reverse.toList.take k).foldl (fun t c => t * gx + c) 0))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hcoef : P → IsExt c n coefficients.reverse[k] gcoefficients.reverse[k]! := fun h => by
      have := (hc h).reverse.2 k hk
      rwa [getElem!_pos coefficients.reverse k hk] at this
    exact (Sim.extMul (fun h => h.2) fun h => (hx h.1.1).mono (l₁.trans lm)).bind₀ fun scaled m₂ l₂ =>
      (extAdd (fun h => h.2) fun h => (hcoef h.1.1.1).mono (l₁.trans (lm.trans l₂))).bind₀ fun total' m₃ l₃ =>
      Sim.pure fun h => ⟨rfl, by
        have hk' : k < gcoefficients.reverse.toList.length := by simpa [← (hc h.1.1.1.1).1] using hk
        rw [List.take_succ_eq_append_getElem hk', List.foldl_append, List.foldl_cons, List.foldl_nil,
          Array.getElem_toList, ← getElem!_pos]
        exact h.2⟩
  · have e := h.2
    rw [List.take_of_length_le (by simp [(hc h.1.1).1])] at e
    rw [Replay.horner, Array.foldr_eq_foldl_reverse, ← Array.foldl_toList]
    exact e

/-- `power_from_bits`: bit `i` of `index` selects `generator^(2^i)`. -/
theorem powerFromBits (generator : Word) {bits : Array Var} (index : ℕ)
    (hb : P → ∀ i < bits.size, IsWord c n bits[i]! (Word.ofNat (index / 2 ^ i % 2))) :
    Sim c n P (Algebra.powerFromBits generator bits)
      (fun r m => IsWord c m r (Replay.powerFromBits generator index bits.size)) True := by
  unfold Algebra.powerFromBits
  refine (Sim.const 1).bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsWord c m₁ total 1)
    (fun k acc m => IsWord c m acc.1 (Replay.powerFromBits generator index k) ∧
      acc.2 = (fun g : Word => g * g)^[k] generator)
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2, rfl⟩).bind₀ fun r m _ => Sim.pure fun h => h.2.1
  have hbit : P → IsWord c n bits[k] (Word.ofNat (index / 2 ^ k % 2)) := fun h => by
    have := hb h k hk
    rwa [getElem!_pos bits k hk] at this
  refine (Sim.const 1).bind₀ fun one m₂ l₂ =>
    (Sim.scale (acc.2 - 1) fun h => (hbit h.1.1.1).mono (l₁.trans (lm.trans l₂))).bind₀ fun fm m₃ l₃ =>
    (Sim.add (fun h => h.1.2.mono l₃) fun h => h.2).bind₀ fun factor m₄ l₄ =>
    (Sim.mul (fun h => h.1.1.1.2.1.mono (l₂.trans (l₃.trans l₄))) fun h => h.2).bind₀ fun total' m₅ l₅ =>
    Sim.pure fun h => ⟨rfl, ?_, by rw [h.1.1.1.1.2.2, Function.iterate_succ_apply']; rfl⟩
  have e : Replay.powerFromBits generator index k * (1 + Word.ofNat (index / 2 ^ k % 2) * (acc.2 - 1)) =
      Replay.powerFromBits generator index (k + 1) := by
    have select : 1 + Word.ofNat (index / 2 ^ k % 2) * (acc.2 - 1) = if index / 2 ^ k % 2 = 1 then acc.2 else 1 := by
      apply Word.eq_of_toF
      rcases Nat.mod_two_eq_zero_or_one (index / 2 ^ k) with hb | hb <;>
        simp [hb, Word.toF_add, Word.toF_mul, Word.toF_sub, Word.toF_ofNat]
    rw [select, powerFromBits_succ, h.1.1.1.1.2.2]
  exact e ▸ h.2

/-- `interpolate`. -/
theorem interpolate {values : Array ExtVar} {x : ExtVar} {gvalues : Array Ext} {gx : Ext}
    (hv : P → IsExts c n values gvalues) (hx : P → IsExt c n x gx) :
    Sim c n P (Algebra.interpolate values x) (fun r m => IsExt c m r (Replay.interpolate gvalues gx)) True := by
  unfold Algebra.interpolate
  dsimp only
  refine extZero.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 0)
    (fun k acc m => IsExt c m acc (Replay.sumRange k fun i =>
      gvalues[i]! * (Replay.prodRange gvalues.size fun j => if i = j then 1 else gx - Ext.ofGl (Word.ofNat j)).scale
        (Replay.prodRangeGl gvalues.size fun j => if i = j then 1 else Word.ofNat i - Word.ofNat j).inv))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hk' : k < values.size := by simpa using hk
    rw [show values.zipIdx[k] = (values[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine extOne.bind₀ fun numerator m₂ l₂ => ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range', ← List.forIn_toArray]
    refine ((loop₀
      (fun j acc' m' =>
        IsExt c m' acc'.1 (Replay.prodRange j fun j => if k = j then 1 else gx - Ext.ofGl (Word.ofNat j)) ∧
        acc'.2 = Replay.prodRangeGl j fun j => if k = j then 1 else Word.ofNat k - Word.ofNat j)
      fun j hj acc' m' lm' => ?_).pre fun h => ⟨h, h.2, rfl⟩).bind₀ fun r' m₃ l₃ => ?_
    · rw [show (List.range' 0 [0:values.size].size 1).toArray[j] = j by simp]
      split
      · rename_i hkj
        exact (extConstant _).bind₀ fun node m₄ l₄ =>
          (extSub (fun h => (hx h.1.1.1.1.1).mono (l₁.trans (lm.trans (l₂.trans (lm'.trans l₄))))) fun h => h.2).bind₀
            fun factor m₅ l₅ =>
          (Sim.extMul (fun h => h.1.1.2.1.mono (l₄.trans l₅)) fun h => h.2).bind₀ fun num m₆ l₆ =>
          Sim.pure fun h => ⟨rfl, by rw [prodRange_succ, if_neg hkj]; exact h.2,
            by rw [prodRangeGl_succ, if_neg hkj, ← h.1.1.1.2.2]; rfl⟩
      · rename_i hkj
        exact Sim.pure fun h => ⟨rfl, by rw [prodRange_succ, if_pos (not_not.mp hkj), ext_mul_one]; exact h.2.1,
          by rw [prodRangeGl_succ, if_pos (not_not.mp hkj), gl_mul_one]; exact h.2.2⟩
    · have hlen : P → (List.range' 0 [0:values.size].size 1).toArray.size = gvalues.size := fun h => by
        simp [(hv h).1]
      exact (extScaleConstant r'.2.inv fun h => h.2.1).bind₀ fun weight m₄ l₄ =>
        (Sim.extMul (fun h => ((hv h.1.1.1.1.1).2 k hk').mono (l₁.trans (lm.trans (l₂.trans (l₃.trans l₄)))))
          fun h => h.2).bind₀ fun term m₅ l₅ =>
        (extAdd (fun h => h.1.1.1.1.2.mono (l₂.trans (l₃.trans (l₄.trans l₅)))) fun h => h.2).bind₀ fun total' m₆ l₆ =>
        Sim.pure fun h => ⟨rfl, by
          have e := h.2
          rw [h.1.1.1.2.2, hlen h.1.1.1.1.1.1.1] at e
          rw [sumRange_succ]
          exact e⟩
  · have e := h.2
    rw [Array.size_zipIdx, (hv h.1.1).1] at e
    exact e

/-- `eq_table`. -/
theorem eqTable {point : Array ExtVar} {gpoint : Array Ext} (hp : P → IsExts c n point gpoint) :
    Sim c n P (Algebra.eqTable point) (fun r m => IsExts c m r (Replay.eqTable gpoint)) True := by
  obtain ⟨E, hE⟩ : ∃ E : ℕ → ℕ → Ext,
      E = fun k i => Replay.prodRange k fun t => if i.testBit t then gpoint[t]! else Replay.oneMinus gpoint[t]! :=
    ⟨_, rfl⟩
  unfold Algebra.eqTable
  refine extOne.bind₀ fun one m₁ l₁ => ?_
  dsimp only
  refine ((loop₀ (P := P ∧ IsExt c m₁ one 1) (fun k acc m => IsExts c m acc ((Array.range (2 ^ k)).map (E k)))
    fun k hk acc m lm => ?_).pre fun h => ⟨h, ?_⟩).bind₀ fun r m _ => Sim.pure fun h => ?_
  · have hx : P → IsExt c n point[k] gpoint[k]! := fun h => by
      have := (hp h).2 k hk
      rwa [getElem!_pos point k hk] at this
    have hsize : ∀ {m'}, IsExts c m' acc ((Array.range (2 ^ k)).map (E k)) → acc.size = 2 ^ k := fun h => by
      simpa using h.1
    have hval : ∀ {m'} j (hj : j < acc.size),
        IsExts c m' acc ((Array.range (2 ^ k)).map (E k)) → IsExt c m' acc[j] (E k j) :=
      fun j hj h => by
        have := h.2 j hj
        rwa [getElem!_pos acc j hj, getElem!_pos _ j (by simpa [hsize h] using hj), Array.getElem_map,
          Array.getElem_range] at this
    obtain ⟨H, hH⟩ : ∃ H : ℕ → Ext,
        H = fun i => if i < 2 ^ k then E k i * Replay.oneMinus gpoint[k]! else E k (i - 2 ^ k) * gpoint[k]! :=
      ⟨_, rfl⟩
    refine (Sim.oneMinus fun h => (hx h.1.1).mono (l₁.trans lm)).bind₀ fun low m₂ l₂ => ?_
    refine ((loop₀ (fun j next m' => IsExts c m' next ((Array.range j).map H)) fun j hj next m' lm' => ?_).pre
      fun h => ⟨h, by simp [IsExts]⟩).bind₀ fun next₁ m₃ l₃ => ?_
    · refine (Sim.extMul (fun h => (hval j hj h.1.1.2).mono (l₂.trans lm')) fun h => h.1.2.mono lm').bind₀
        fun v m₄ l₄ => Sim.pure fun h => ⟨rfl, ?_⟩
      rw [map_range_succ]
      refine (h.1.2.mono l₄).push ?_
      rw [hH]
      dsimp only
      rw [if_pos (hsize h.1.1.1.2 ▸ hj)]
      exact h.2
    · refine ((loop₀ (fun j next m' => IsExts c m' next ((Array.range (acc.size + j)).map H))
        fun j hj next m' lm' => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun next₂ m₄ l₄ => Sim.pure fun h => ⟨rfl, ?_⟩
      · refine (Sim.extMul (fun h => (hval j hj h.1.1.1.2).mono (l₂.trans (l₃.trans lm')))
          fun h => (hx h.1.1.1.1.1).mono (l₁.trans (lm.trans (l₂.trans (l₃.trans lm'))))).bind₀
          fun v m₅ l₅ => Sim.pure fun h => ⟨rfl, ?_⟩
        rw [← Nat.add_assoc, map_range_succ]
        refine (h.1.2.mono l₅).push ?_
        rw [hH]
        dsimp only
        rw [if_neg (by rw [hsize h.1.1.1.1.2]; omega), hsize h.1.1.1.1.2, Nat.add_sub_cancel_left]
        exact h.2
      · have e := h.2
        rw [hsize h.1.1.1.2, show 2 ^ k + 2 ^ k = 2 ^ (k + 1) by rw [Nat.pow_succ]; omega] at e
        have key : (Array.range (2 ^ (k + 1))).map H = (Array.range (2 ^ (k + 1))).map (E (k + 1)) :=
          Array.map_inj_left.mpr fun i hi => by
            rw [Array.mem_range] at hi
            rw [hH, hE]
            dsimp only
            rw [prodRange_succ]
            by_cases lt : i < 2 ^ k
            · rw [if_pos lt, Nat.testBit_lt_two_pow lt, if_neg Bool.false_ne_true]
            · obtain ⟨j, rfl⟩ : ∃ j, i = 2 ^ k + j := ⟨i - 2 ^ k, by omega⟩
              have hj : j < 2 ^ k := by rw [Nat.pow_succ] at hi; omega
              rw [if_neg lt, Nat.add_sub_cancel_left, Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow hj,
                Bool.not_false, if_pos rfl]
              exact congrArg (· * gpoint[k]!) (prodRange_congr fun t ht => by rw [Nat.testBit_two_pow_add_gt ht])
        rw [key] at e
        exact e
  · refine ⟨by simp, fun i hi => ?_⟩
    have i0 : i = 0 := by simpa using hi
    subst i0
    rw [getElem!_pos _ 0 (by simp), getElem!_pos _ 0 (by simp), Array.getElem_map, Array.getElem_range, hE]
    simp only [prodRange_zero]
    exact h.2
  · have e := h.2
    rw [(hp h.1.1).1] at e
    rw [Replay.eqTable, show Replay.eqAt gpoint = E gpoint.size from funext fun i => by rw [eqAt_eq, hE]]
    exact e

omit [CharP F goldilocksModulus] in
/-- `Kx::zero`. -/
theorem KxVar.zero : Sim c n P Circuit.KxVar.zero (fun r m => IsKx c m r Replay.Kx.zero) True :=
  extZero.bind₀ fun _ _ _ => Sim.pure fun h => ⟨h.2, h.2⟩

omit [CharP F goldilocksModulus] in
/-- `Kx::one`. -/
theorem KxVar.one : Sim c n P Circuit.KxVar.one (fun r m => IsKx c m r Replay.Kx.one) True :=
  extOne.bind₀ fun _ _ _ => extZero.bind₀ fun _ _ l => Sim.pure fun h => ⟨h.1.2.mono l, h.2⟩

omit [CharP F goldilocksModulus] in
/-- `Kx::from_k`. -/
theorem KxVar.fromK {a : KVar} {g₁ g₂ : Word} (h₁ : P → IsWord c n a.1 g₁) (h₂ : P → IsWord c n a.2 g₂) :
    Sim c n P (Circuit.KxVar.fromK a) (fun r m => IsKx c m r (Replay.Kx.ofK (g₁, g₂))) True :=
  (liftWord h₁).bind₀ fun _ _ l₁ => (liftWord fun h => (h₂ h.1).mono l₁).bind₀ fun _ _ l₂ =>
    Sim.pure fun h => ⟨h.1.2.mono l₂, h.2⟩

/-- `Kx::add`. -/
theorem KxVar.add {a b : KxVar} {ga gb : Replay.Kx} (ha : P → IsKx c n a ga) (hb : P → IsKx c n b gb) :
    Sim c n P (Circuit.KxVar.add a b) (fun r m => IsKx c m r (ga.add gb)) True :=
  (extAdd (fun h => (ha h).1) fun h => (hb h).1).bind₀ fun _ _ l₁ =>
    (extAdd (fun h => (ha h.1).2.mono l₁) fun h => (hb h.1).2.mono l₁).bind₀ fun _ _ l₂ =>
    Sim.pure fun h => ⟨h.1.2.mono l₂, h.2⟩

/-- `Kx::mul`. -/
theorem KxVar.mul {a b : KxVar} {ga gb : Replay.Kx} (ha : P → IsKx c n a ga) (hb : P → IsKx c n b gb) :
    Sim c n P (Circuit.KxVar.mul a b) (fun r m => IsKx c m r (ga.mul gb)) True := by
  refine (Sim.extMul (fun h => (ha h).1) fun h => (hb h).1).bind₀ fun rr m₁ l₁ =>
    (Sim.extMul (fun h => (ha h.1).2.mono l₁) fun h => (hb h.1).2.mono l₁).bind₀ fun ii m₂ l₂ =>
    (extScaleConstant 7 fun h => h.2).bind₀ fun seven m₃ l₃ =>
    (Sim.extMul (fun h => (ha h.1.1.1).1.mono (l₁.trans (l₂.trans l₃)))
      fun h => (hb h.1.1.1).2.mono (l₁.trans (l₂.trans l₃))).bind₀ fun ri m₄ l₄ =>
    (Sim.extMul (fun h => (ha h.1.1.1.1).2.mono (l₁.trans (l₂.trans (l₃.trans l₄))))
      fun h => (hb h.1.1.1.1).1.mono (l₁.trans (l₂.trans (l₃.trans l₄)))).bind₀ fun ir m₅ l₅ =>
    (extAdd (fun h => h.1.1.1.1.2.mono (l₂.trans (l₃.trans (l₄.trans l₅)))) fun h => h.1.1.2.mono (l₄.trans l₅)).bind₀
      fun re m₆ l₆ =>
    (extAdd (fun h => h.1.1.2.mono (l₅.trans l₆)) fun h => h.1.2.mono l₆).bind₀ fun im m₇ l₇ =>
    Sim.pure fun h => ⟨h.1.2.mono l₇, h.2⟩

/-- `Kx::scale`. -/
theorem KxVar.scale {a : KxVar} {e : ExtVar} {ga : Replay.Kx} {ge : Ext} (ha : P → IsKx c n a ga)
    (he : P → IsExt c n e ge) : Sim c n P (Circuit.KxVar.scale a e) (fun r m => IsKx c m r (ga.scale ge)) True :=
  (Sim.extMul (fun h => (ha h).1) he).bind₀ fun _ _ l₁ =>
    (Sim.extMul (fun h => (ha h.1).2.mono l₁) fun h => (he h.1).mono l₁).bind₀ fun _ _ l₂ =>
    Sim.pure fun h => ⟨h.1.2.mono l₂, h.2⟩

/-- `Kx::one_minus`. -/
theorem KxVar.oneMinus {a : KxVar} {ga : Replay.Kx} (ha : P → IsKx c n a ga) :
    Sim c n P (Circuit.KxVar.oneMinus a) (fun r m => IsKx c m r ga.oneMinus) True :=
  (Sim.oneMinus fun h => (ha h).1).bind₀ fun _ _ l₁ => extZero.bind₀ fun _ _ l₂ =>
    (extSub (fun h => h.2) fun h => (ha h.1.1).2.mono (l₁.trans l₂)).bind₀ fun _ _ l₃ =>
    Sim.pure fun h => ⟨h.1.1.2.mono (l₂.trans l₃), h.2⟩

end Sim

end CompressionLean.Circuit
