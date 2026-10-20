import CompressionLean.Circuit.Sim

/-!
Owns the simulation of the field helpers of `Algebra.lean` and of `select`
(`Ir.lean`): each program's handles are the replay's value
(`Replay/Field.lean`, `Replay/Formulas.lean`), and its assertions hold iff
the replay's equality holds.

An extension value's coordinates in `F` (`coords`) multiply as
`extProduct`, which is commutative and associative. A program and the
replay may multiply in different orders; their values have the same
coordinates (`IsExt.congr`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Coordinates -/

/-- The coordinates of `e` in `F`. -/
def coords (e : Ext) : Fin 3 → F := fun k => word (e.coord k)

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

theorem coords_scale (a : Ext) (g : Gl) : coords (F := F) (a.scale g) = fun k => coords a k * word g := by
  funext k
  fin_cases k <;> exact word_mul _ _

@[simp] theorem coords_one : coords (F := F) 1 = ![1, 0, 0] := by
  funext k
  fin_cases k <;> simp [coords]

@[simp] theorem coords_zero : coords (F := F) 0 = 0 := by
  funext k
  fin_cases k <;> simp [coords]

theorem coords_ofGl (g : Gl) : coords (F := F) (Ext.ofGl g) = ![word g, 0, 0] := by
  funext k
  fin_cases k <;> simp [coords, Ext.coord, Ext.ofGl]

theorem coords_injective : Function.Injective (coords (F := F)) := fun _ _ h =>
  Ext.eq_of_coord (F := F) (congrFun h)

end Char

/-! ## Programs -/

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

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
/-- `lift`: a word as an extension value. -/
theorem liftWord {a : Var} {g : Gl} (ha : P → IsWord c n a g) :
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
theorem extScale {a : ExtVar} {s : Var} {ga : Ext} {gs : Gl} (ha : P → IsExt c n a ga) (hs : P → IsWord c n s gs) :
    Sim c n P (Build.extScale a s) (fun r m => IsExt c m r (ga.scale gs)) True :=
  coordinatewise fun k _ _ hp le => by
    fin_cases k
    · exact Sim.mul (fun h => (ha (hp h) 0).mono le) (fun h => (hs (hp h)).mono le)
    · exact Sim.mul (fun h => (ha (hp h) 1).mono le) (fun h => (hs (hp h)).mono le)
    · exact Sim.mul (fun h => (ha (hp h) 2).mono le) (fun h => (hs (hp h)).mono le)

/-- `ext_scale_constant`. -/
theorem extScaleConstant {a : ExtVar} {ga : Ext} (g : Gl) (ha : P → IsExt c n a ga) :
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
theorem select {bit w₀ w₁ : Var} {b : Bool} {g₀ g₁ : Gl} (hb : P → IsWord c n bit (if b then 1 else 0))
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

end Sim

end CompressionLean.Circuit
