import CompressionLean.Circuit.SimSumcheck
import CompressionLean.Circuit.Ring
import CompressionLean.Circuit.Mle
import CompressionLean.Final.Fields

/-!
Owns the simulation of the ring rows of `Ring.lean` (`ring.rs`) and the
closed forms of `Mle.lean` (`mle.rs`) against the replay's formulas
(`Replay/Formulas.lean`).

- `IsK`, `IsKs`, `IsClaim`: statement handles hold the statement's words.
  `IsMixing`: the powers of `λ`, one per row, as the replay's `Mixing`
  reads them.
- An identity of `Ext` or `Kx` holds when it holds in the challenge field
  `Final.Cubic` (`Ext.eq_of_toK`, `kx_eq_of_toQ`). So proof words that pass
  the inverse check are the replay's inverse (`inverse_of_check`), and
  `Ensures.scaledEqPoint` finds no pole.
- `Sim.evalK`: the circuit keeps `Option` states and skips absent ones; the
  replay sums every transition. An absent state is a zero state
  (`IsState`), and one replay step is a sum over the `4·D` digits
  `(rem, below, bit)` in the circuit's order (`digitSum`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Kx Relation D)
open CompressionLean.Bridge (ChallengeField)
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

/-! ## Identities in the challenge field -/

/-- `Final.Cubic` is a challenge field. -/
@[reducible] noncomputable def challengeCubic : ChallengeField Final.Cubic where
  root := AdjoinRoot.root Final.cubic
  cubic := by
    have h : AdjoinRoot.mk Final.cubic (X ^ 3 - X - 1) = 0 := AdjoinRoot.mk_self
    rw [map_sub, map_sub, map_pow, AdjoinRoot.mk_X, map_one] at h
    linear_combination h
  card := Final.card_cubic

attribute [local instance] challengeCubic

private theorem kx_eq_of_toQ {a b : Kx} (h : a.toQ (K := Final.Cubic) = b.toQ) : a = b := by
  obtain ⟨ar, ai⟩ := a
  obtain ⟨br, bi⟩ := b
  have hr := congrArg QuadraticAlgebra.re h
  have hi := congrArg QuadraticAlgebra.im h
  simp only [Kx.toQ_re, Kx.toQ_im] at hr hi
  rw [Ext.eq_of_toK (K := Final.Cubic) hr, Ext.eq_of_toK (K := Final.Cubic) hi]

theorem ext_pow_zero (a : Ext) : a.pow 0 = 1 := by
  rw [Ext.pow, if_pos rfl]

private theorem ext_pow_succ (a : Ext) (k : ℕ) : a.pow (k + 1) = a.pow k * a :=
  Ext.eq_of_toK (K := Final.Cubic) (by rw [Ext.toK_mul, Ext.toK_pow, Ext.toK_pow, pow_succ])

theorem ext_add_zero (a : Ext) : a + 0 = a :=
  Ext.eq_of_toK (K := Final.Cubic) (by rw [Ext.toK_add, Ext.toK_zero, add_zero])

theorem ext_add_assoc (a b d : Ext) : a + b + d = a + (b + d) :=
  Ext.eq_of_toK (K := Final.Cubic) (by rw [Ext.toK_add, Ext.toK_add, Ext.toK_add, Ext.toK_add, add_assoc])

private theorem ext_scale_zero (a : Ext) : a.scale 0 = 0 :=
  Ext.eq_of_toK (K := Final.Cubic) (by rw [Ext.toK_scale, Ext.toK_zero, Replay.Word.toF_zero, map_zero, mul_zero])

private theorem ext_mul_inv {a : Ext} (h : a ≠ 0) : a * a.inv = 1 := by
  have nz : a.toK (K := Final.Cubic) ≠ 0 := fun z => h (Ext.eq_zero_of_toK z)
  exact Ext.eq_of_toK (K := Final.Cubic) (by rw [Ext.toK_mul, Ext.toK_inv, Ext.toK_one, mul_inv_cancel₀ nz])

private theorem ext_isZero {a : Ext} (h : a ≠ 0) : a.isZero = false := by
  cases e : a.isZero
  · rfl
  · exact absurd (Ext.eq_of_beq e) h

private theorem kx_add_zero (a : Kx) : a.add Kx.zero = a :=
  kx_eq_of_toQ (by rw [Kx.toQ_add, Kx.toQ_zero, add_zero])

private theorem kx_zero_add (a : Kx) : Kx.zero.add a = a :=
  kx_eq_of_toQ (by rw [Kx.toQ_add, Kx.toQ_zero, zero_add])

private theorem kx_zero_mul (a : Kx) : Kx.zero.mul a = Kx.zero :=
  kx_eq_of_toQ (by rw [Kx.toQ_mul, Kx.toQ_zero, zero_mul])

private theorem kx_zero_scale (e : Ext) : Kx.zero.scale e = Kx.zero :=
  kx_eq_of_toQ (by rw [Kx.toQ_scale, Kx.toQ_zero, zero_mul])

/-! ## Sums and arrays -/

private theorem sumRange_zero (f : ℕ → Ext) : Replay.sumRange 0 f = 0 := rfl

private theorem sumKx_succ (k : ℕ) (f : ℕ → Kx) : Replay.sumKx (k + 1) f = (Replay.sumKx k f).add (f k) := by
  simp only [Replay.sumKx, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

private theorem sumKx_congr {k : ℕ} {f g : ℕ → Kx} (h : ∀ i < k, f i = g i) :
    Replay.sumKx k f = Replay.sumKx k g := by
  induction k with
  | zero => simp only [Replay.sumKx, List.range_zero, List.foldl_nil]
  | succ k ih => rw [sumKx_succ, sumKx_succ, ih fun i hi => h i (by omega), h k (by omega)]

/-- A sum over `4·k` digits is the sum over `(rem, below, bit)`, digit `4·rem + 2·below + bit`. -/
private theorem sumKx_digits (k : ℕ) (f : ℕ → Kx) :
    Replay.sumKx (4 * k) f = Replay.sumKx k fun rem => Replay.sumKx 2 fun below => Replay.sumKx 2 fun bit =>
      f (4 * rem + 2 * below + bit) := by
  apply kx_eq_of_toQ
  simp only [Replay.toQ_sumKx]
  induction k with
  | zero => simp
  | succ k ih =>
      rw [show 4 * (k + 1) = 4 * k + 4 by ring, Finset.sum_range_add, ih, Finset.sum_range_succ (n := k)]
      simp only [Finset.sum_range_succ, Finset.sum_range_zero]
      ring_nf

private theorem map_range_succ' {α : Type} (f : ℕ → α) (k : ℕ) :
    (Array.range (k + 1)).map f = #[f 0] ++ (Array.range k).map fun i => f (i + 1) := by
  apply Array.ext'
  simp [List.range_succ_eq_map]

/-- Entry `i` of a `mapM` result over `range k` meets `R i`. -/
private theorem forall₂_range {α : Type} [Inhabited α] {R : ℕ → α → Prop} {k : ℕ} {xs : Array α}
    (h : List.Forall₂ R (Array.range k).toList xs.toList) : xs.size = k ∧ ∀ i < k, R i xs[i]! := by
  have size := h.length_eq
  simp only [Array.length_toList, Array.size_range] at size
  refine ⟨size.symm, fun i hi => ?_⟩
  have := (List.forall₂_iff_get.mp h).2 i (by simpa using hi) (by simpa [← size] using hi)
  rw [getElem!_pos xs i (by omega)]
  simpa using this

/-! ## Statement handles and the mixing -/

variable {F : Type} [CommRing F]

/-- The two handles of `a` are the words of the `K` value `g`, `(re, im)`. -/
def IsK (c : Ctx F) (n : ℕ) (a : KVar) (g : Word × Word) : Prop := IsWord c n a.1 g.1 ∧ IsWord c n a.2 g.2

theorem IsK.mono {c : Ctx F} {n m : ℕ} {a : KVar} {g : Word × Word} (h : IsK c n a g) (le : n ≤ m) : IsK c m a g :=
  ⟨h.1.mono le, h.2.mono le⟩

/-- `xs` and `gs` have one size, and each pair of handles is its `K` value. -/
def IsKs (c : Ctx F) (n : ℕ) (xs : Array KVar) (gs : Array (Word × Word)) : Prop :=
  xs.size = gs.size ∧ ∀ i < xs.size, IsK c n xs[i]! gs[i]!

namespace IsKs

variable {c : Ctx F} {n : ℕ} {xs : Array KVar} {gs : Array (Word × Word)}

theorem mono (h : IsKs c n xs gs) {m : ℕ} (le : n ≤ m) : IsKs c m xs gs :=
  ⟨h.1, fun i hi => (h.2 i hi).mono le⟩

private theorem get (h : IsKs c n xs gs) {i : ℕ} (hi : i < gs.size) : IsK c n xs[i]! gs[i]! :=
  h.2 i (h.1 ▸ hi)

private theorem extract (h : IsKs c n xs gs) (a : ℕ) : IsKs c n (xs.extract a) (gs.extract a) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

end IsKs

/-- The statement view holds the statement's words, field by field. -/
structure IsClaim (c : Ctx F) (n : ℕ) (v : ClaimView) (st : Replay.Statement) : Prop where
  commitment : v.commitment.size = st.commitment.size ∧
    ∀ i < v.commitment.size, IsWords c n v.commitment[i]! st.commitment[i]!
  publicInput : v.publicInput.size = st.publicInput.size ∧
    ∀ i < v.publicInput.size, IsWords c n v.publicInput[i]! st.publicInput[i]!
  point : IsKs c n v.point st.point
  evalK : IsKs c n v.evalK st.evalK
  evalA : v.evalA.size = st.evalA.size ∧ ∀ j < v.evalA.size, IsKs c n v.evalA[j]! st.evalA[j]!

theorem IsClaim.mono {c : Ctx F} {n m : ℕ} {v : ClaimView} {st : Replay.Statement} (h : IsClaim c n v st)
    (le : n ≤ m) : IsClaim c m v st :=
  ⟨⟨h.commitment.1, fun i hi => (h.commitment.2 i hi).mono le⟩,
    ⟨h.publicInput.1, fun i hi => (h.publicInput.2 i hi).mono le⟩, h.point.mono le, h.evalK.mono le,
    ⟨h.evalA.1, fun j hj => (h.evalA.2 j hj).mono le⟩⟩

/-- The handles of `v` are the powers of `m.lam`, one per row, and `v` has `m`'s sizes. -/
def IsMixing (c : Ctx F) (n : ℕ) (v : MixingVar) (m : Replay.Mixing) : Prop :=
  v.kappa = m.kappa ∧ v.matrices = m.matrices ∧ ∀ i < v.powers.size, IsExt c n v.powers[i]! (m.lam.pow i)

namespace IsMixing

variable {c : Ctx F} {n : ℕ} {v : MixingVar} {m : Replay.Mixing}

theorem mono (h : IsMixing c n v m) {n' : ℕ} (le : n ≤ n') : IsMixing c n' v m :=
  ⟨h.1, h.2.1, fun i hi => (h.2.2 i hi).mono le⟩

theorem commitment (h : IsMixing c n v m) {row : ℕ} (hr : row < v.powers.size) :
    IsExt c n (v.commitment row) (m.commitment row) :=
  h.2.2 row hr

theorem evalK (h : IsMixing c n v m) (hr : m.kappa + 1 < v.powers.size) :
    IsExt c n v.evalK.1 m.evalK.1 ∧ IsExt c n v.evalK.2 m.evalK.2 := by
  simp only [MixingVar.evalK, Replay.Mixing.evalK, h.1]
  exact ⟨h.2.2 _ (by omega), h.2.2 _ hr⟩

theorem evalA (h : IsMixing c n v m) {j : ℕ} (hr : m.kappa + 3 + 2 * j < v.powers.size) :
    IsExt c n (v.evalA j).1 (m.evalA j).1 ∧ IsExt c n (v.evalA j).2 (m.evalA j).2 := by
  simp only [MixingVar.evalA, Replay.Mixing.evalA, h.1]
  exact ⟨h.2.2 _ (by omega), h.2.2 _ hr⟩

theorem «public» (h : IsMixing c n v m) {block : ℕ} (hr : m.kappa + 2 + 2 * m.matrices + block < v.powers.size) :
    IsExt c n (v.public block) (m.public block) := by
  simp only [MixingVar.«public», Replay.Mixing.«public», h.1, h.2.1]
  exact h.2.2 _ hr

end IsMixing

namespace IsExts

variable {c : Ctx F} {n : ℕ} {xs : Array ExtVar} {gs : Array Ext}

/-- Handle `i` is `g i` for each `i < k`. -/
private theorem map_range_iff {k : ℕ} {g : ℕ → Ext} :
    IsExts c n xs ((Array.range k).map g) ↔ xs.size = k ∧ ∀ i < k, IsExt c n xs[i]! (g i) := by
  refine ⟨fun h => ⟨by simpa using h.1, fun i hi => ?_⟩, fun h => ⟨by simpa using h.1, fun i hi => ?_⟩⟩
  · have size : xs.size = k := by simpa using h.1
    have := h.2 i (size ▸ hi)
    rwa [getElem!_map_range _ _ hi] at this
  · rw [getElem!_map_range _ _ (h.1 ▸ hi)]
    exact h.2 i (h.1 ▸ hi)

end IsExts

/-! ## Ring rows -/

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

omit [CharP F goldilocksModulus] in
/-- A loop over a list whose steps check nothing. -/
theorem loopList₀ {γ δ : Type} {xs : List δ} {init : γ} {body : δ → γ → Build (ForInStep γ)}
    (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → Sim c m (P ∧ I k acc m) (body xs[k] acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') True) :
    Sim c n (P ∧ I 0 init n) (forIn xs init body) (fun r m => I xs.length r m) True :=
  (loop I (fun _ => True) step).post (fun _ _ _ h => h) fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩

omit [CharP F goldilocksModulus] in
/-- A loop over `[a:b]` whose steps check nothing; step `k` reads `a + k`. -/
theorem loopRange₀ {γ : Type} {a b : ℕ} {init : γ} {body : ℕ → γ → Build (ForInStep γ)}
    (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k < b - a, ∀ acc m, n ≤ m → Sim c m (P ∧ I k acc m) (body (a + k) acc)
      (fun r m' => r = .yield r.value ∧ I (k + 1) r.value m') True) :
    Sim c n (P ∧ I 0 init n) (forIn [a:b] init body) (fun r m => I (b - a) r m) True := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have size : [a:b].size = b - a := by simp [Std.Legacy.Range.size]
  refine ((loop I (fun _ => True) fun k hk acc m le => ?_).post (fun _ _ _ h => ?_)
    fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩)
  · rw [List.getElem_range', Nat.one_mul]
    exact step k (by simpa [size] using hk) acc m le
  · simpa [size] using h

/-- `Mixing::new` (`ring.rs:48-59`): `1, λ, λ², …`, one power per row. -/
theorem mixing {lambda : ExtVar} {lam : Ext} (rel : Relation) (hl : P → IsExt c n lambda lam) :
    Sim c n P (MixingVar.new lambda rel)
      (fun v m => IsMixing c m v ⟨lam, rel.kappa, rel.matrices⟩ ∧ v.powers.size = rel.rowCount) True := by
  unfold MixingVar.new
  refine extOne.bind₀ fun one m₁ l₁ => ?_
  dsimp only
  refine ((loopRange₀ (P := P ∧ IsExt c m₁ one 1)
    (fun k acc m => IsExts c m acc ((Array.range (k + 1)).map lam.pow)) fun k hk acc m lm => ?_).pre
      fun h => ⟨h, ?_⟩).bind₀ fun powers m₂ _ => Sim.pure fun h => ?_
  · have back : P ∧ IsExts c m acc ((Array.range (k + 1)).map lam.pow) → IsExt c m acc.back! (lam.pow k) :=
      fun h => by
        obtain ⟨size, get⟩ := IsExts.map_range_iff.mp h.2
        rw [Array.back!, size, Nat.add_sub_cancel]
        exact get k (by omega)
    refine (Sim.extMul (fun h => back ⟨h.1.1, h.2⟩) fun h => (hl h.1.1).mono (l₁.trans lm)).bind₀
      fun x m' l' => Sim.pure fun h => ⟨rfl, ?_⟩
    rw [map_range_succ, ext_pow_succ]
    exact (h.1.2.mono l').push h.2
  · refine ⟨by simp, fun i hi => ?_⟩
    obtain rfl : i = 0 := by simpa using hi
    simpa [ext_pow_zero] using h.2
  · have rows : 2 ≤ rel.rowCount := by simp only [Relation.rowCount]; omega
    obtain ⟨size, get⟩ := IsExts.map_range_iff.mp h.2
    rw [show rel.rowCount - 1 + 1 = rel.rowCount by omega] at size get
    exact ⟨⟨rfl, rfl, fun i hi => get i (size ▸ hi)⟩, size⟩

/-- `project` (`ring.rs:189-193`): `weights.1·Re + weights.2·Im`. -/
theorem project {value : KVar} {gv : Word × Word} {weights : ExtVar × ExtVar} {gw : Ext × Ext}
    (hv : P → IsK c n value gv) (hw : P → IsExt c n weights.1 gw.1 ∧ IsExt c n weights.2 gw.2) :
    Sim c n P (Ring.project value weights) (fun r m => IsExt c m r (Replay.project gv gw)) True :=
  (extScale (fun h => (hw h).1) fun h => (hv h).1).bind₀ fun _ _ l₁ =>
    (extScale (fun h => (hw h.1).2.mono l₁) fun h => (hv h.1).2.mono l₁).bind₀ fun _ _ l₂ =>
    extAdd (fun h => h.1.2.mono l₂) fun h => h.2

/-- `targets` (`ring.rs:164-184`): `Ȳ = Σ_rows λ^row·y_row`, lane by lane. -/
theorem targets {v : ClaimView} {st : Replay.Statement} {mv : MixingVar} {lam : Ext} {rel : Relation}
    (fits : st.fits rel = true) (hv : P → IsClaim c n v st)
    (hm : P → IsMixing c n mv ⟨lam, rel.kappa, rel.matrices⟩ ∧ mv.powers.size = rel.rowCount) :
    Sim c n P (Ring.targets v mv) (fun r m => IsExts c m r (Replay.targets st ⟨lam, rel.kappa, rel.matrices⟩))
      True := by
  have sizes := fits
  simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq, Array.all_eq_true] at sizes
  obtain ⟨⟨⟨⟨⟨⟨⟨comSize, comRows⟩, pubSize⟩, pubRows⟩, -⟩, evalKSize⟩, evalASize⟩, evalARows⟩ :=
    sizes
  have rows : rel.rowCount = rel.kappa + 2 + 2 * rel.matrices + rel.publicBlocks := rfl
  set mix : Replay.Mixing := ⟨lam, rel.kappa, rel.matrices⟩ with hmix
  have mk : mix.kappa = rel.kappa := rfl
  have mm : mix.matrices = rel.matrices := rfl
  obtain ⟨T, hT⟩ : ∃ T : ℕ → Ext, T = fun lane =>
      Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange st.evalA.size (fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j)) +
        Replay.sumRange st.publicInput.size (fun block => (mix.public block).scale (st.publicInput[block]!)[lane]!) :=
    ⟨_, rfl⟩
  have target : Replay.targets st mix = (Array.range D).map T := by rw [hT]; rfl
  rw [target]
  unfold Ring.targets
  refine (Sim.mapM (fun lane a m => IsExt c m a (T lane)) (fun _ _ _ _ h le => h.mono le)
    fun lane hlane m lm => ?_).post (fun _ res m h => IsExts.map_range_iff.mpr (forall₂_range h)) fun _ => Iff.rfl
  have hl : lane < D := by simpa using hlane
  rw [hT]
  dsimp only
  refine extZero.bind₀ fun total m₁ l₁ => ?_
  refine ((loop₀ (P := P ∧ IsExt c m₁ total 0)
    (fun k acc m' => IsExt c m' acc
      (Replay.sumRange k fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!))
    fun k hk acc m' lm' => ?_).pre fun h => ⟨h, h.2⟩).bind₀ fun total₁ m₂ l₂ => ?_
  · have hk' : k < v.commitment.size := by simpa using hk
    rw [show v.commitment.zipIdx[k] = (v.commitment[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (extScale (ga := mix.commitment k) (gs := (st.commitment[k]!)[lane]!) (fun h => ?_) fun h => ?_).bind₀
      fun term m₃ l₃ => (extAdd (fun h => h.1.2.mono l₃) fun h => h.2).bind₀ fun total' m₄ l₄ =>
      Sim.pure fun h => ⟨rfl, by rw [sumRange_succ]; exact h.2⟩
    · have hc := (hv h.1.1).commitment
      exact ((hm h.1.1).1.commitment (by rw [(hm h.1.1).2, rows]; omega)).mono (lm.trans (l₁.trans lm'))
    · have hc := (hv h.1.1).commitment
      have row := comRows k (by omega)
      rw [← getElem!_pos st.commitment k (by omega)] at row
      exact (((hc.2 k hk').get (by rw [row]; exact hl))).mono (lm.trans (l₁.trans lm'))
  refine (project (gv := st.evalK[lane]!) (gw := mix.evalK) (fun h => ?_) fun h => ?_).bind₀ fun term m₃ l₃ => ?_
  · have hk := (hv h.1.1).evalK
    have := hk.2 lane (by rw [hk.1, evalKSize]; exact hl)
    exact this.mono (lm.trans (l₁.trans l₂))
  · exact ((hm h.1.1).1.evalK (by rw [(hm h.1.1).2, rows]; omega)).imp (·.mono (lm.trans (l₁.trans l₂)))
      (·.mono (lm.trans (l₁.trans l₂)))
  refine (extAdd (ga := Replay.sumRange st.commitment.size
      fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) (fun h => ?_) fun h => h.2).bind₀
    fun total₂ m₄ l₄ => ?_
  · have e := h.1.2
    rw [Array.size_zipIdx, (hv h.1.1.1).commitment.1] at e
    exact e.mono l₃
  refine ((loop₀ (P := P ∧ IsExt c m₄ total₂
      (Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK))
    (fun k acc m' => IsExt c m' acc
      (Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange k fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j)))
    fun k hk acc m' lm' => ?_).pre fun h => ⟨⟨h.1.1.1.1, h.2⟩, ?_⟩).bind₀ fun total₃ m₅ l₅ => ?_
  · have hk' : k < v.evalA.size := by simpa using hk
    rw [show v.evalA.zipIdx[k] = (v.evalA[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (project (gv := (st.evalA[k]!)[lane]!) (gw := mix.evalA k) (fun h => ?_) fun h => ?_).bind₀
      fun term m₃ l₃ => (extAdd (fun h => h.1.2.mono l₃) fun h => h.2).bind₀ fun total' m₄ l₄ =>
      Sim.pure fun h => ⟨rfl, by rw [sumRange_succ, ← ext_add_assoc]; exact h.2⟩
    · have ha := (hv h.1.1).evalA
      have row := evalARows k (by omega)
      rw [← getElem!_pos st.evalA k (by omega)] at row
      have := (ha.2 k hk').2 lane (by rw [(ha.2 k hk').1, row]; exact hl)
      exact this.mono (lm.trans (l₁.trans (l₂.trans (l₃.trans (l₄.trans lm')))))
    · have ha := (hv h.1.1).evalA
      exact ((hm h.1.1).1.evalA (by rw [(hm h.1.1).2, rows]; omega)).imp
        (·.mono (lm.trans (l₁.trans (l₂.trans (l₃.trans (l₄.trans lm'))))))
        (·.mono (lm.trans (l₁.trans (l₂.trans (l₃.trans (l₄.trans lm'))))))
  · rw [sumRange_zero, ext_add_zero]
    exact h.2
  refine Sim.pre (P' := P ∧ IsExt c m₅ total₃
      (Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange st.evalA.size fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j))) ?_
    fun h => ⟨h.1.1.1.1.1, by have e := h.2; rwa [Array.size_zipIdx, (hv h.1.1.1.1.1).evalA.1] at e⟩
  refine ((loop₀ (P := P ∧ IsExt c m₅ total₃
      (Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange st.evalA.size fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j)))
    (fun k acc m' => IsExt c m' acc
      (Replay.sumRange st.commitment.size (fun row => (mix.commitment row).scale (st.commitment[row]!)[lane]!) +
        Replay.project st.evalK[lane]! mix.evalK +
        Replay.sumRange st.evalA.size (fun j => Replay.project (st.evalA[j]!)[lane]! (mix.evalA j)) +
        Replay.sumRange k fun block => (mix.public block).scale (st.publicInput[block]!)[lane]!))
    fun k hk acc m' lm' => ?_).pre fun h => ⟨h, by rw [sumRange_zero, ext_add_zero]; exact h.2⟩).bind₀
    fun total₄ m₆ l₆ => Sim.pure fun h => ?_
  · have hk' : k < v.publicInput.size := by simpa using hk
    rw [show v.publicInput.zipIdx[k] = (v.publicInput[k]!, k) by simp [getElem!_pos, hk']]
    dsimp only
    refine (extScale (ga := mix.public k) (gs := (st.publicInput[k]!)[lane]!) (fun h => ?_) fun h => ?_).bind₀
      fun term m₃ l₃ => (extAdd (fun h => h.1.2.mono l₃) fun h => h.2).bind₀ fun total' m₄ l₄ =>
      Sim.pure fun h => ⟨rfl, by rw [sumRange_succ, ← ext_add_assoc]; exact h.2⟩
    · have hp := (hv h.1.1).publicInput
      exact ((hm h.1.1).1.public (by rw [(hm h.1.1).2, rows]; omega)).mono
        (lm.trans (l₁.trans (l₂.trans (l₃.trans (l₄.trans (l₅.trans lm'))))))
    · have hp := (hv h.1.1).publicInput
      have row := pubRows k (by omega)
      rw [← getElem!_pos st.publicInput k (by omega)] at row
      exact ((hp.2 k hk').get (by rw [row]; exact hl)).mono
        (lm.trans (l₁.trans (l₂.trans (l₃.trans (l₄.trans (l₅.trans lm'))))))
  · have e := h.2
    rwa [Array.size_zipIdx, (hv h.1.1).publicInput.1] at e

/-- `lifted_target` (`ring.rs:231-244`): `Ȳ(ζ) + Q(ζ)·Φ81(ζ)` with `Φ81 = X⁵⁴ + X²⁷ + 1`. -/
theorem liftedTarget {target quotient : Array ExtVar} {zeta : ExtVar} {gt gq : Array Ext} {gz : Ext}
    (ht : P → IsExts c n target gt) (hq : P → IsExts c n quotient gq) (hz : P → IsExt c n zeta gz) :
    Sim c n P (Ring.liftedTarget target quotient zeta) (fun r m => IsExt c m r (Replay.liftedTarget gt gq gz))
      True := by
  unfold Ring.liftedTarget Replay.liftedTarget
  dsimp only
  refine ((loopRange₀ (fun k acc m => IsExt c m acc (gz.pow (k + 1))) fun k _ acc m lm => ?_).pre
    fun h => ⟨h, ?_⟩).bind₀ fun middle m₁ l₁ => ?_
  · exact (Sim.extMul (fun h => h.2) fun h => (hz h.1).mono lm).bind₀ fun x m' l' =>
      Sim.pure fun h => ⟨rfl, by rw [ext_pow_succ _ (k + 1)]; exact h.2⟩
  · rw [ext_pow_succ, ext_pow_zero, ext_one_mul]
    exact hz h
  refine (Sim.extMul (fun h => h.2) fun h => h.2).bind₀ fun top m₂ l₂ => extOne.bind₀ fun one m₃ l₃ =>
    (extAdd (fun h => h.1.2.mono l₃) fun h => h.1.1.2.mono (l₂.trans l₃)).bind₀ fun sum m₄ l₄ =>
    (extAdd (fun h => h.2) fun h => h.1.2.mono l₄).bind₀ fun phi m₅ l₅ => ?_
  have up : n ≤ m₅ := l₁.trans (l₂.trans (l₃.trans (l₄.trans l₅)))
  refine (horner (fun h => (ht h.1.1.1.1.1).mono up) fun h => (hz h.1.1.1.1.1).mono up).bind₀ fun target' m₆ l₆ =>
    (horner (fun h => (hq h.1.1.1.1.1.1).mono (up.trans l₆)) fun h => (hz h.1.1.1.1.1.1).mono (up.trans l₆)).bind₀
      fun quotient' m₇ l₇ =>
    (Sim.extMul (fun h => h.2) fun h => h.1.1.2.mono (l₆.trans l₇)).bind₀ fun lifted m₈ l₈ =>
    (extAdd (fun h => h.1.1.2.mono (l₇.trans l₈)) fun h => h.2).post (fun _ _ _ h => ?_) fun _ => Iff.rfl
  rwa [show (27 - 1 + 1 : ℕ) = 27 from rfl] at h

/-- `tau` (`ring.rs:150-161`): `τ_l = Σ_p bar(p, l)·ζ^p`. -/
theorem tau {zeta : ExtVar} {gz : Ext} (hz : P → IsExt c n zeta gz) :
    Sim c n P (Ring.tau zeta) (fun r m => IsExts c m r (Replay.tau gz)) True := by
  obtain ⟨S, hS⟩ : ∃ S : ℕ → ℕ → Ext,
      S = fun p l => Replay.sumRange p fun q => (gz.pow q).scale (Replay.bar q l) := ⟨_, rfl⟩
  have tau : Replay.tau gz = (Array.range D).map (S D) := by rw [hS]; rfl
  rw [tau]
  unfold Ring.tau
  refine extZero.bind₀ fun zero m₁ l₁ => ?_
  dsimp only
  refine extOne.bind₀ fun one m₂ l₂ => ?_
  refine ((loopRange₀ (P := (P ∧ IsExt c m₁ zero 0) ∧ IsExt c m₂ one 1)
    (fun p acc m => IsExts c m acc.1 ((Array.range D).map (S p)) ∧ IsExt c m acc.2 (gz.pow p))
    fun p hp acc m lm => ?_).pre fun h => ⟨h, ?_, ?_⟩).bind₀ fun res m₃ _ => Sim.pure fun h => h.2.1
  · rw [Nat.zero_add]
    refine ((loopRange₀ (P := ((P ∧ IsExt c m₁ zero 0) ∧ IsExt c m₂ one 1) ∧ IsExt c m acc.2 (gz.pow p))
      (fun col acc' m' => IsExts c m' acc' ((Array.range D).map fun l => if l < col then S (p + 1) l else S p l))
      fun col hcol acc' m' lm' => ?_).pre fun h => ⟨⟨h.1, h.2.2⟩, ?_⟩).bind₀ fun tau' m₄ l₄ => ?_
    · rw [Nat.zero_add]
      have hc : col < D := by simpa using hcol
      have succ : S (p + 1) col = S p col + (gz.pow p).scale (Replay.bar p col) := by
        rw [hS]
        exact sumRange_succ _ _
      split
      · rename_i zero
        refine Sim.pure fun h => ⟨rfl, ?_⟩
        obtain ⟨size, get⟩ := IsExts.map_range_iff.mp h.2
        refine IsExts.map_range_iff.mpr ⟨size, fun i hi => ?_⟩
        have e := get i hi
        by_cases ic : i = col
        · subst ic
          rw [if_neg (Nat.lt_irrefl _)] at e
          rw [if_pos (Nat.lt_succ_self _), succ, Replay.Word.eq_of_beq zero, ext_scale_zero, ext_add_zero]
          exact e
        · by_cases lt : i < col
          · rwa [if_pos lt, if_pos (by omega)] at *
          · rwa [if_neg lt, if_neg (by omega)] at *
      · have entry : IsExts c m' acc' ((Array.range D).map fun l => if l < col then S (p + 1) l else S p l) →
            IsExt c m' acc'[col]! (S p col) := fun h => by
          have e := (IsExts.map_range_iff.mp h).2 col hc
          rwa [if_neg (Nat.lt_irrefl _)] at e
        refine (extScaleConstant (Replay.bar p col) fun h => h.1.2.mono lm').bind₀ fun term m₅ l₅ =>
          (extAdd (fun h => (entry h.1.2).mono l₅) fun h => h.2).bind₀ fun x m₆ l₆ =>
          Sim.pure fun h => ⟨rfl, ?_⟩
        obtain ⟨size, get⟩ := IsExts.map_range_iff.mp (h.1.1.2.mono (l₅.trans l₆))
        have hx := h.2
        rw [← succ] at hx
        show IsExts c m₆ (acc'.set! col x) _
        refine IsExts.map_range_iff.mpr ⟨by simp [size], fun i hi => ?_⟩
        rw [getElem!_set! _ _ _ _ (by simp [size]; exact hi)]
        by_cases ic : col = i
        · subst ic
          rw [if_pos rfl, if_pos (Nat.lt_succ_self _)]
          exact hx
        · rw [if_neg ic]
          have e := get i hi
          by_cases lt : i < col
          · rwa [if_pos lt, if_pos (by omega)] at *
          · rwa [if_neg lt, if_neg (by omega)] at *
    · exact IsExts.map_range_iff.mpr ⟨(IsExts.map_range_iff.mp h.2.1).1, fun i hi => by
        rw [if_neg (Nat.not_lt_zero _)]
        exact (IsExts.map_range_iff.mp h.2.1).2 i hi⟩
    · refine (Sim.extMul (fun h => h.1.2.2.mono l₄)
        fun h => (hz h.1.1.1.1).mono (l₁.trans (l₂.trans (lm.trans l₄)))).bind₀
        fun power m₅ l₅ => Sim.pure fun h => ⟨rfl, ?_, by rw [ext_pow_succ]; exact h.2⟩
      obtain ⟨size, get⟩ := IsExts.map_range_iff.mp (h.1.2.mono l₅)
      refine IsExts.map_range_iff.mpr ⟨size, fun i hi => ?_⟩
      have e := get i hi
      rwa [if_pos (show i < D - 0 from hi)] at e
  · refine IsExts.map_range_iff.mpr ⟨by simp, fun i hi => ?_⟩
    rw [getElem!_pos _ i (by simpa using hi), Array.getElem_replicate, hS]
    exact h.1.2.mono l₂
  · rw [ext_pow_zero]
    exact h.2

end Sim

/-! ## The scaled eq point -/

/-- `κ = Π_t (1 + c^{2^t})` and `x_t = c^{2^t}/(1 + c^{2^t})` when no denominator is a pole; `sq t` is
`c^{2^t}`. -/
private theorem scaledEqPoint_eq (bits : ℕ) : ∀ sq : ℕ → Ext, (∀ t, sq (t + 1) = sq t * sq t) →
    (∀ t < bits, 1 + sq t ≠ 0) →
      Replay.scaledEqPoint (sq 0) bits =
        .ok (Replay.prodRange bits fun t => 1 + sq t, (Array.range bits).map fun t => sq t * (1 + sq t).inv) := by
  induction bits with
  | zero =>
      intro sq _ _
      show Except.ok (1, #[]) = Except.ok (1, _)
      congr 2
      apply Array.ext'
      simp
  | succ bits ih =>
      intro sq hsq poles
      have p0 : (1 + sq 0).isZero = false := ext_isZero (poles 0 (Nat.succ_pos _))
      have rest := ih (fun t => sq (t + 1)) (fun t => hsq (t + 1)) fun t ht => poles (t + 1) (by omega)
      rw [Nat.zero_add, hsq 0] at rest
      rw [Replay.scaledEqPoint]
      simp only [p0, Bool.false_eq_true, if_false, rest, bind_pure_comp, Except.map_ok, prodRange_succ',
        map_range_succ']

/-- A check `a·t = 1` in `F` makes `a` nonzero and `t` the coordinates of its inverse. -/
private theorem inverse_of_check [CharP F goldilocksModulus] {ga : Ext} {t : Fin 3 → F}
    (h : extProduct (coords ga) t = ![1, 0, 0]) : ga ≠ 0 ∧ t = coords ga.inv := by
  have nz : ga ≠ 0 := by
    rintro rfl
    haveI := CharP.nontrivial_of_char_ne_one (R := F) (v := goldilocksModulus) (by decide)
    have := congrFun h 0
    simp [extProduct] at this
  refine ⟨nz, ?_⟩
  have unit : extProduct (coords (F := F) ga) (coords ga.inv) = ![1, 0, 0] := by
    rw [← coords_mul, ext_mul_inv nz, coords_one]
  calc t = extProduct t ![1, 0, 0] := (extProduct_one t).symm
    _ = extProduct (extProduct (coords ga) t) (coords ga.inv) := by
      rw [← unit, ← extProduct_assoc, extProduct_comm t]
    _ = coords ga.inv := by rw [h, extProduct_comm, extProduct_one]

namespace Sim

variable {c : Ctx F} {n : ℕ} {P : Prop}

/-- `ext_inverse` (`record.rs:379-397`): three proof words, and the check that they are the inverse. -/
private theorem extInvTape {a : ExtVar} {ga : Ext} (ha : P → IsExt c n a ga) :
    Sim c n P (Build.extInv a) (fun r m => ∀ k : Fin 3, r k < m ∧ c.val (r k) = c.tape (n + k))
      (extProduct (coords ga) (fun k => c.tape (n + k)) = ![1, 0, 0]) :=
  (emit (Q := fun h m => h = n ∧ ∀ k : Fin 3, h + k < m ∧ c.val (h + k) = c.tape (h + k)) _ fun hP E hl hF => by
    have hE := Ctx.follows_of_step hF
    have ra : (fun i => E.value (a i)) = coords ga := funext fun i => (ha hP i).read hE hl
    refine ⟨⟨rfl, fun k => ⟨by simp [Instr.outputs], ?_⟩⟩, ?_⟩
    · rw [← hl, Ctx.val_next hF (by simp)]
      simp only [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
      rw [dif_pos k.isLt, Option.getD_some]
    · show E.holds ∧ extProduct (fun i => E.value (a i)) (fun k => c.tape (E.values.length + k)) = ![1, 0, 0] ↔ _
      rw [ra, hl]).map fun _ _ h k => by
    obtain ⟨rfl, h⟩ := h
    exact h k

end Sim

namespace Ensures

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `ext_inverse` (`record.rs:379-397`): once its check holds, the value is nonzero and the proof words are
its inverse. -/
private theorem extInv {a : ExtVar} {ga : Ext} (ha : P → IsExt c n a ga) :
    Ensures c n P (Build.extInv a) (fun r m => ga ≠ 0 ∧ IsExt c m r ga.inv) :=
  (Sim.extInvTape ha).ensures.post fun _ r m h => by
    obtain ⟨nz, inv⟩ := inverse_of_check h.2
    exact ⟨nz, fun k => ⟨(h.1 k).1, by rw [(h.1 k).2]; exact congrFun inv k⟩⟩

/-- `scaled_eq_point` (`mle.rs:21-34`): once each inverse checks, no denominator is a pole, the replay returns
`κ` and the point, and the handles hold them. -/
theorem scaledEqPoint {cv : ExtVar} {gc : Ext} (bits : ℕ) (hc : P → IsExt c n cv gc) :
    Ensures c n P (Mle.scaledEqPoint cv bits)
      (fun res m => ∃ kappa point, Replay.scaledEqPoint gc bits = .ok (kappa, point) ∧
        IsExt c m res.1 kappa ∧ IsExts c m res.2 point) := by
  obtain ⟨sq, hsq0, hsq⟩ : ∃ sq : ℕ → Ext, sq 0 = gc ∧ ∀ t, sq (t + 1) = sq t * sq t :=
    ⟨fun t => (fun x : Ext => x * x)^[t] gc, rfl, fun t => Function.iterate_succ_apply' _ _ _⟩
  unfold Mle.scaledEqPoint
  refine Sim.extOne.ensures₀.bind fun one m₁ l₁ => ?_
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine ((Ensures.loop (P := P ∧ IsExt c m₁ one 1)
    (fun k acc m => IsExt c m acc.1 (Replay.prodRange k fun t => 1 + sq t) ∧
      IsExts c m acc.2.1 ((Array.range k).map fun t => sq t * (1 + sq t).inv) ∧ IsExt c m acc.2.2 (sq k) ∧
      ∀ t < k, 1 + sq t ≠ 0)
    fun k hk acc m lm => ?_).pre fun h => ⟨h, ?_⟩).bind fun res m₂ _ => Ensures.pure fun h => ?_
  · refine (Sim.extAdd (fun h => h.1.2.mono lm) fun h => h.2.2.2.1).ensures₀.bind fun d m₃ l₃ => ?_
    refine (Sim.extMul (fun h => h.1.2.1.mono l₃) fun h => h.2).ensures₀.bind fun kappa m₄ l₄ => ?_
    refine (extInv fun h => h.1.2.mono l₄).bind fun inv m₅ l₅ => ?_
    refine (Sim.extMul (fun h => h.1.1.1.2.2.2.1.mono (l₃.trans (l₄.trans l₅))) fun h => h.2.2).ensures₀.bind
      fun x m₆ l₆ => ?_
    refine (Sim.extMul (fun h => h.1.1.1.1.2.2.2.1.mono (l₃.trans (l₄.trans (l₅.trans l₆))))
      fun h => h.1.1.1.1.2.2.2.1.mono (l₃.trans (l₄.trans (l₅.trans l₆)))).ensures₀.bind fun power m₇ l₇ =>
      Ensures.pure fun h => ⟨rfl, ?_, ?_, ?_, ?_⟩
    · rw [prodRange_succ]
      exact h.1.1.1.2.mono (l₅.trans (l₆.trans l₇))
    · rw [map_range_succ]
      exact (h.1.1.1.1.1.2.2.1.mono (l₃.trans (l₄.trans (l₅.trans (l₆.trans l₇))))).push (h.1.2.mono l₇)
    · rw [hsq]
      exact h.2
    · intro t ht
      rcases Nat.lt_succ_iff_lt_or_eq.mp ht with lt | rfl
      · exact h.1.1.1.1.1.2.2.2.2 t lt
      · exact h.1.1.2.1
  · refine ⟨by rw [prodRange_zero]; exact h.2, IsExts.map_range_iff.mpr ⟨rfl, fun i hi => absurd hi (by omega)⟩,
      hsq0 ▸ (hc h.1).mono l₁, fun t ht => absurd ht (by omega)⟩
  · have len : (List.range' 0 [0:bits].size 1).length = bits := by simp [Std.Legacy.Range.size]
    rw [len] at h
    exact ⟨_, _, by rw [← hsq0]; exact scaledEqPoint_eq bits sq hsq h.2.2.2.2, h.2.1, h.2.2.1⟩

end Ensures

/-! ## Eval_K -/

/-- A long-division state of the circuit holds the replay's state; an absent state is zero. -/
private def IsState (c : Ctx F) (n : ℕ) : Option KxVar → Kx → Prop
  | none, g => g = Kx.zero
  | some a, g => IsKx c n a g

private theorem IsState.mono {c : Ctx F} {n m : ℕ} {o : Option KxVar} {g : Kx} (h : IsState c n o g) (le : n ≤ m) :
    IsState c m o g := by
  cases o with
  | none => exact h
  | some a => exact IsKx.mono h le

/-- The circuit's long-division states hold the replay's states `g rem below`. -/
private def IsStates (c : Ctx F) (n : ℕ) (states : Mle.DivisionStates) (g : ℕ → ℕ → Kx) : Prop :=
  states.size = D ∧
    ∀ rem < D, (states[rem]!).size = 2 ∧ ∀ below < 2, IsState c n (states[rem]!)[below]! (g rem below)

/-- The replay's next states after the first `i` digits of one step of the long division, digit
`4·rem + 2·below + bit` (`evalKStates`). -/
private def digitSum (gr : Array (Word × Word)) (gs : Array Ext) (blocks t : ℕ) (prev : Array (Array Kx))
    (i rem' below' : ℕ) : Kx :=
  Replay.sumKx i fun j => Replay.contribution (Replay.divStep blocks gs.size t (j / 4) (j / 2 % 2) (j % 2)) rem'
    below' (Replay.divFactor gr gs t (j % 2)) ((prev[j / 4]!)[j / 2 % 2]!)

private theorem digitSum_succ (gr : Array (Word × Word)) (gs : Array Ext) (blocks t : ℕ) (prev : Array (Array Kx))
    (i rem below : ℕ) :
    digitSum gr gs blocks t prev (i + 1) rem below = (digitSum gr gs blocks t prev i rem below).add
      (Replay.contribution (Replay.divStep blocks gs.size t (i / 4) (i / 2 % 2) (i % 2)) rem below
        (Replay.divFactor gr gs t (i % 2)) ((prev[i / 4]!)[i / 2 % 2]!)) :=
  sumKx_succ _ _

private theorem evalKStates_zero (gr : Array (Word × Word)) (gs : Array Ext) (blocks cb : ℕ) {rem below : ℕ} (hr : rem < D)
    (hb : below < 2) :
    ((Replay.evalKStates gr gs blocks cb 0)[rem]!)[below]! = if rem = 0 ∧ below = 0 then Kx.one else Kx.zero := by
  rw [Replay.evalKStates, getElem!_map_range _ _ hr, getElem!_map_range _ _ hb]

private theorem evalKStates_succ (gr : Array (Word × Word)) (gs : Array Ext) (blocks cb k : ℕ) {rem below : ℕ} (hr : rem < D)
    (hb : below < 2) :
    ((Replay.evalKStates gr gs blocks cb (k + 1))[rem]!)[below]! =
      digitSum gr gs blocks (cb - 1 - k) (Replay.evalKStates gr gs blocks cb k) (4 * D) rem below := by
  rw [Replay.evalKStates, getElem!_map_range _ _ hr, getElem!_map_range _ _ hb, digitSum, sumKx_digits]
  refine sumKx_congr fun r hr' => sumKx_congr fun b hb' => sumKx_congr fun t ht => ?_
  rw [show (4 * r + 2 * b + t) / 4 = r by omega, show (4 * r + 2 * b + t) / 2 % 2 = b by omega,
    show (4 * r + 2 * b + t) % 2 = t by omega]

private theorem contribution_absent (step : Option (ℕ × ℕ × ℕ)) (rem below : ℕ) (factor : ℕ → Kx) :
    Replay.contribution step rem below factor Kx.zero = Kx.zero := by
  rcases step with _ | ⟨r, b, q⟩
  · rfl
  · simp only [Replay.contribution]
    split_ifs
    · exact kx_zero_mul _
    · rfl

/-- `divFactor` as the circuit computes it: `1 - r_t` or `r_t`, then the weight of `s_t`. -/
private theorem divFactor_ifs (gr : Array (Word × Word)) (gs : Array Ext) (t bit q : ℕ) (hbit : bit < 2) :
    Replay.divFactor gr gs t bit q =
      if t < gs.size then
        (if bit = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]!).scale
          (if q = 1 then gs[t]! else Replay.oneMinus gs[t]!)
      else if bit = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]! := by
  unfold Replay.divFactor
  rcases (show bit = 0 ∨ bit = 1 by omega) with rfl | rfl <;> simp

/-- `divStep` as the circuit tests it: the quotient bit, then the flag. -/
private theorem divStep_ifs (blocks bb t rem below bit : ℕ) :
    Replay.divStep blocks bb t rem below bit =
      if t ≥ bb ∧ (if 2 * rem + bit ≥ D then 1 else 0) = 1 then none
      else (if below = 1 ∨ (if 2 * rem + bit ≥ D then 1 else 0) < blocks / 2 ^ t % 2 then some 1
        else if (if 2 * rem + bit ≥ D then 1 else 0) = blocks / 2 ^ t % 2 then some 0 else none).map
          fun nb => (2 * rem + bit - D * (if 2 * rem + bit ≥ D then 1 else 0), nb,
            if 2 * rem + bit ≥ D then 1 else 0) := by
  unfold Replay.divStep
  simp only [ge_iff_le]
  split_ifs <;> simp_all

namespace IsStates

variable {c : Ctx F} {n : ℕ} {next : Mle.DivisionStates} {g : ℕ → ℕ → Kx} {gr : Array (Word × Word)}
  {gs : Array Ext} {blocks t : ℕ} {prev : Array (Array Kx)} {i : ℕ}

private theorem mono (h : IsStates c n next g) {m : ℕ} (le : n ≤ m) : IsStates c m next g :=
  ⟨h.1, fun rem hr => ⟨(h.2 rem hr).1, fun below hb => ((h.2 rem hr).2 below hb).mono le⟩⟩

private theorem congr (h : IsStates c n next g) {g' : ℕ → ℕ → Kx}
    (e : ∀ rem < D, ∀ below < 2, g rem below = g' rem below) : IsStates c n next g' :=
  ⟨h.1, fun rem hr => ⟨(h.2 rem hr).1, fun below hb => e rem hr below hb ▸ (h.2 rem hr).2 below hb⟩⟩

private theorem existing (h : IsStates c n next g) {slot nb : ℕ} (hs : slot < D) (hb : nb < 2) {a : KxVar}
    (e : (next[slot]!)[nb]! = some a) : IsKx c n a (g slot nb) := by
  have := (h.2 slot hs).2 nb hb
  rwa [e] at this

private theorem absent (h : IsStates c n next g) {slot nb : ℕ} (hs : slot < D) (hb : nb < 2)
    (e : (next[slot]!)[nb]! = none) : g slot nb = Kx.zero := by
  have := (h.2 slot hs).2 nb hb
  rwa [e] at this

private theorem set (h : IsStates c n next g) (slot nb : ℕ) {a : KxVar} {v : Kx} (ha : IsKx c n a v) :
    IsStates c n (next.set! slot (next[slot]!.set! nb (some a)))
      (fun rem below => if rem = slot ∧ below = nb then v else g rem below) := by
  refine ⟨by simp [h.1], fun rem hr => ?_⟩
  rw [getElem!_set! _ _ _ _ (by rw [h.1]; exact hr)]
  by_cases e : slot = rem
  · subst e
    rw [if_pos rfl]
    refine ⟨by simp [(h.2 slot hr).1], fun below hb' => ?_⟩
    rw [getElem!_set! _ _ _ _ (by rw [(h.2 slot hr).1]; exact hb')]
    dsimp only
    by_cases e' : nb = below
    · subst e'
      rw [if_pos rfl, if_pos (⟨rfl, rfl⟩ : slot = slot ∧ nb = nb)]
      exact ha
    · rw [if_neg e', if_neg fun h' : slot = slot ∧ below = nb => e' h'.2.symm]
      exact (h.2 slot hr).2 below hb'
  · rw [if_neg e]
    exact ⟨(h.2 rem hr).1, fun below hb' => by
      dsimp only
      rw [if_neg fun h' : rem = slot ∧ below = nb => e h'.1.symm]
      exact (h.2 rem hr).2 below hb'⟩

/-- A digit that moves no state. -/
private theorem skip (h : IsStates c n next (digitSum gr gs blocks t prev i))
    (zero : ∀ rem below, Replay.contribution (Replay.divStep blocks gs.size t (i / 4) (i / 2 % 2) (i % 2)) rem below
      (Replay.divFactor gr gs t (i % 2)) ((prev[i / 4]!)[i / 2 % 2]!) = Kx.zero) :
    IsStates c n next (digitSum gr gs blocks t prev (i + 1)) :=
  h.congr fun rem _ below _ => by rw [digitSum_succ, zero, kx_add_zero]

/-- A digit `(R, B, Bt)` that adds `x·factor` to state `(slot, nb)`. -/
private theorem move {R B Bt : ℕ} (h : IsStates c n next (digitSum gr gs blocks t prev (4 * R + 2 * B + Bt)))
    (hB : B < 2) (hBt : Bt < 2) {slot nb q : ℕ} (step : Replay.divStep blocks gs.size t R B Bt = some (slot, nb, q))
    {a : KxVar} (ha : IsKx c n a ((digitSum gr gs blocks t prev (4 * R + 2 * B + Bt) slot nb).add
      (((prev[R]!)[B]!).mul (Replay.divFactor gr gs t Bt q)))) :
    IsStates c n (next.set! slot (next[slot]!.set! nb (some a)))
      (digitSum gr gs blocks t prev (4 * R + 2 * B + Bt + 1)) := by
  have d1 : (4 * R + 2 * B + Bt) / 4 = R := by omega
  have d2 : (4 * R + 2 * B + Bt) / 2 % 2 = B := by omega
  have d3 : (4 * R + 2 * B + Bt) % 2 = Bt := by omega
  refine (h.set slot nb ha).congr fun rem _ below _ => ?_
  rw [digitSum_succ, d1, d2, d3, step]
  simp only [Replay.contribution]
  by_cases e : slot = rem ∧ nb = below
  · obtain ⟨rfl, rfl⟩ := e
    rw [if_pos ⟨rfl, rfl⟩, if_pos ⟨rfl, rfl⟩]
  · rw [if_neg fun h' => e ⟨h'.1.symm, h'.2.symm⟩, if_neg e, kx_add_zero]

end IsStates

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `eval_k` (`mle.rs:146-204`): the long division's value. An absent state of the circuit is a zero state of
the replay. -/
theorem evalK {r : Array KVar} {sBlock tau : Array ExtVar} {gr : Array (Word × Word)} {gs gt : Array Ext} (blocks : ℕ)
    (rSize : Nat.log2 (D * 2 ^ gs.size).nextPowerOfTwo ≤ gr.size) (tauSize : D ≤ gt.size)
    (hr : P → IsKs c n r gr) (hs : P → IsExts c n sBlock gs) (ht : P → IsExts c n tau gt) :
    Sim c n P (Mle.evalK r sBlock tau blocks) (fun res m => IsKx c m res (Replay.evalK gr gs gt blocks)) True := by
  unfold Mle.evalK Replay.evalK
  dsimp only
  generalize hcb : (D * 2 ^ sBlock.size).nextPowerOfTwo.log2 = cb
  have bits : P → cb = (D * 2 ^ gs.size).nextPowerOfTwo.log2 := fun h => by rw [← hcb, (hs h).1]
  refine KxVar.zero.bind₀ fun zero m₁ l₁ => KxVar.one.bind₀ fun one m₂ l₂ => ?_
  refine ((loopList₀ (P := (P ∧ IsKx c m₁ zero Kx.zero) ∧ IsKx c m₂ one Kx.one)
    (fun k states m => IsStates c m states fun rem below => ((Replay.evalKStates gr gs blocks cb k)[rem]!)[below]!)
    fun k hk states m lm => ?_).pre fun h => ⟨h, ?_⟩).bind₀ fun states m₃ l₃ => ?_
  · have hk' : k < cb := by simpa using hk
    rw [show (List.range cb).reverse[k] = cb - 1 - k by simp [List.getElem_reverse]]
    generalize ht' : cb - 1 - k = t
    have hrt : P → IsK c n r[t]! gr[t]! := fun h => (hr h).get (by have := bits h; omega)
    refine (KxVar.fromK (g₁ := gr[t]!.1) (g₂ := gr[t]!.2) (fun h => (hrt h.1.1.1).1.mono (l₁.trans (l₂.trans lm)))
      fun h => (hrt h.1.1.1).2.mono (l₁.trans (l₂.trans lm))).bind₀ fun rT m₄ l₄ =>
      (KxVar.oneMinus fun h => h.2).bind₀ fun rLow m₅ l₅ => ?_
    obtain ⟨prev, hprev⟩ : ∃ prev, prev = Replay.evalKStates gr gs blocks cb k := ⟨_, rfl⟩
    rw [← hprev]
    refine ((loopRange₀ (P := P ∧ IsStates c m states (fun rem below => (prev[rem]!)[below]!) ∧
        IsKx c m₅ rT (Kx.ofK gr[t]!) ∧ IsKx c m₅ rLow (Kx.ofK gr[t]!).oneMinus)
      (fun R next m' => IsStates c m' next (digitSum gr gs blocks t prev (4 * R)))
      fun R hR next m' lm' => ?_).pre fun h => ⟨⟨h.1.1.1.1.1, h.1.1.2, h.1.2.mono l₅, h.2⟩, ?_⟩).bind₀
      fun next m₆ l₆ => Sim.pure fun h => ⟨rfl, ?_⟩
    · have hR' : R < D := by simpa using hR
      rw [Nat.zero_add]
      refine ((loopRange₀ (P := P ∧ IsStates c m states (fun rem below => (prev[rem]!)[below]!) ∧
          IsKx c m₅ rT (Kx.ofK gr[t]!) ∧ IsKx c m₅ rLow (Kx.ofK gr[t]!).oneMinus)
        (fun B next' m'' => IsStates c m'' next' (digitSum gr gs blocks t prev (4 * R + 2 * B)))
        fun B hB next' m'' lm'' => ?_).pre fun h => ⟨h.1, h.2⟩).bind₀ fun next₂ m₇ l₇ =>
        Sim.pure fun h => ⟨rfl, by rw [show 4 * (R + 1) = 4 * R + 2 * (2 - 0) by omega]; exact h.2⟩
      have hB' : B < 2 := by simpa using hB
      rw [Nat.zero_add]
      rcases hst : (states[R]!)[B]! with _ | value
      · refine Sim.pure fun h => ⟨rfl, ?_⟩
        have e := h.1.2.1.absent hR' hB' hst
        have z : ∀ i, i / 4 = R ∧ i / 2 % 2 = B → ∀ rem below,
            Replay.contribution (Replay.divStep blocks gs.size t (i / 4) (i / 2 % 2) (i % 2)) rem below
              (Replay.divFactor gr gs t (i % 2)) ((prev[i / 4]!)[i / 2 % 2]!) = Kx.zero :=
          fun i hi rem below => by rw [hi.1, hi.2, e]; exact contribution_absent _ _ _ _
        have two := (h.2.skip (z _ ⟨by omega, by omega⟩)).skip (z _ ⟨by omega, by omega⟩)
        rwa [show 4 * R + 2 * B + 1 + 1 = 4 * R + 2 * (B + 1) by ring] at two
      refine ((loopRange₀ (P := P ∧ IsStates c m states (fun rem below => (prev[rem]!)[below]!) ∧
          IsKx c m₅ rT (Kx.ofK gr[t]!) ∧ IsKx c m₅ rLow (Kx.ofK gr[t]!).oneMinus)
        (fun Bt next'' m''' => IsStates c m''' next'' (digitSum gr gs blocks t prev (4 * R + 2 * B + Bt)))
        fun Bt hBt next'' m''' lm''' => ?_).pre fun h => ⟨h.1, h.2⟩).bind₀ fun next₃ m₈ l₈ =>
        Sim.pure fun h => ⟨rfl, by rw [show 4 * R + 2 * (B + 1) = 4 * R + 2 * B + (2 - 0) by omega]; exact h.2⟩
      have hBt' : Bt < 2 := by simpa using hBt
      rw [Nat.zero_add]
      have up : m ≤ m''' := l₄.trans (l₅.trans (lm'.trans (lm''.trans lm''')))
      have up5 : m₅ ≤ m''' := lm'.trans (lm''.trans lm''')
      have upn : n ≤ m''' := l₁.trans (l₂.trans (lm.trans up))
      have dec : (4 * R + 2 * B + Bt) / 4 = R ∧ (4 * R + 2 * B + Bt) / 2 % 2 = B ∧ (4 * R + 2 * B + Bt) % 2 = Bt :=
        ⟨by omega, by omega, by omega⟩
      have hstep := divStep_ifs blocks gs.size t R B Bt
      have hslot : 2 * R + Bt - D * (if 2 * R + Bt ≥ D then 1 else 0) < D := by split_ifs <;> omega
      generalize (if 2 * R + Bt ≥ D then 1 else 0) = q at hstep hslot
      generalize hnbv : (if B = 1 ∨ q < blocks / 2 ^ t % 2 then some 1 else if q = blocks / 2 ^ t % 2 then some 0
        else none) = nbo at hstep
      by_cases skip : t ≥ sBlock.size ∧ q = 1
      · rw [if_pos skip]
        refine Sim.pure fun h => ⟨rfl, h.2.skip fun rem below => ?_⟩
        rw [dec.1, dec.2.1, dec.2.2, hstep, if_pos (by rw [← (hs h.1.1).1]; exact skip)]
        rfl
      rw [if_neg skip]
      rcases nbo with _ | nb
      · refine Sim.pure fun h => ⟨rfl, h.2.skip fun rem below => ?_⟩
        rw [dec.1, dec.2.1, dec.2.2, hstep, if_neg (by rw [← (hs h.1.1).1]; exact skip)]
        rfl
      dsimp only
      have hnb2 : nb < 2 := by
        split_ifs at hnbv <;> cases hnbv <;> omega
      have step : P → Replay.divStep blocks gs.size t R B Bt = some (2 * R + Bt - D * q, nb, q) := fun hP => by
        rw [hstep, if_neg (by rw [← (hs hP).1]; exact skip)]
        rfl
      have hX : ∀ {m₀}, IsKx c m₀ rT (Kx.ofK gr[t]!) → IsKx c m₀ rLow (Kx.ofK gr[t]!).oneMinus →
          IsKx c m₀ (if Bt = 0 then rLow else rT) (if Bt = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]!) := by
        intro m₀ a b
        split_ifs
        · exact b
        · exact a
      have fac : P → Replay.divFactor gr gs t Bt q = if t < sBlock.size then
          (if Bt = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]!).scale
            (if q = 1 then gs[t]! else Replay.oneMinus gs[t]!)
          else if Bt = 0 then (Kx.ofK gr[t]!).oneMinus else Kx.ofK gr[t]! := fun hP => by
        rw [divFactor_ifs _ _ _ _ _ hBt', (hs hP).1]
      rcases hex : (next''[2 * R + Bt - D * q]!)[nb]! with _ | existing
      · by_cases hlt : t < sBlock.size
        · rw [if_pos hlt]
          by_cases hq1 : q = 1
          · rw [if_pos hq1, pure_bind]
            refine (KxVar.scale (fun h => hX (h.1.2.2.1.mono up5) (h.1.2.2.2.mono up5))
              fun h => ((hs h.1.1).get (by rw [← (hs h.1.1).1]; exact hlt)).mono upn).bind₀ fun factor m₉ l₉ =>
              (KxVar.mul (fun h => (h.1.1.2.1.existing hR' hB' hst).mono (up.trans l₉)) fun h => h.2).bind₀
              fun product m₁₀ l₁₀ => ?_
            dsimp only
            rw [pure_bind]
            refine Sim.pure fun h => ⟨rfl, (h.1.1.2.mono (l₉.trans l₁₀)).move hB' hBt' (step h.1.1.1.1) ?_⟩
            rw [h.1.1.2.absent hslot hnb2 hex, kx_zero_add, fac h.1.1.1.1, if_pos hlt, if_pos hq1]
            exact h.2
          · rw [if_neg hq1]
            refine (Sim.oneMinus fun h => ((hs h.1.1).get (by rw [← (hs h.1.1).1]; exact hlt)).mono upn).bind₀
              fun weight m₈' l₈' => (KxVar.scale (fun h => hX (h.1.1.2.2.1.mono (up5.trans l₈'))
                (h.1.1.2.2.2.mono (up5.trans l₈'))) fun h => h.2).bind₀ fun factor m₉ l₉ =>
              (KxVar.mul (fun h => (h.1.1.1.2.1.existing hR' hB' hst).mono (up.trans (l₈'.trans l₉)))
                fun h => h.2).bind₀ fun product m₁₀ l₁₀ => ?_
            dsimp only
            rw [pure_bind]
            refine Sim.pure fun h =>
              ⟨rfl, (h.1.1.1.2.mono (l₈'.trans (l₉.trans l₁₀))).move hB' hBt' (step h.1.1.1.1.1) ?_⟩
            rw [h.1.1.1.2.absent hslot hnb2 hex, kx_zero_add, fac h.1.1.1.1.1, if_pos hlt, if_neg hq1]
            exact h.2
        · rw [if_neg hlt]
          refine (KxVar.mul (fun h => (h.1.2.1.existing hR' hB' hst).mono up)
            fun h => hX (h.1.2.2.1.mono up5) (h.1.2.2.2.mono up5)).bind₀ fun product m₁₀ l₁₀ => ?_
          dsimp only
          rw [pure_bind]
          refine Sim.pure fun h => ⟨rfl, (h.1.2.mono l₁₀).move hB' hBt' (step h.1.1.1) ?_⟩
          rw [h.1.2.absent hslot hnb2 hex, kx_zero_add, fac h.1.1.1, if_neg hlt]
          exact h.2
      · by_cases hlt : t < sBlock.size
        · rw [if_pos hlt]
          by_cases hq1 : q = 1
          · rw [if_pos hq1, pure_bind]
            refine (KxVar.scale (fun h => hX (h.1.2.2.1.mono up5) (h.1.2.2.2.mono up5))
              fun h => ((hs h.1.1).get (by rw [← (hs h.1.1).1]; exact hlt)).mono upn).bind₀ fun factor m₉ l₉ =>
              (KxVar.mul (fun h => (h.1.1.2.1.existing hR' hB' hst).mono (up.trans l₉)) fun h => h.2).bind₀
              fun product m₁₀ l₁₀ => ?_
            dsimp only
            refine (KxVar.add (fun h => (h.1.1.2.existing hslot hnb2 hex).mono (l₉.trans l₁₀)) fun h => h.2).bind₀
              fun merged m₁₁ l₁₁ => Sim.pure fun h =>
                ⟨rfl, (h.1.1.1.2.mono (l₉.trans (l₁₀.trans l₁₁))).move hB' hBt' (step h.1.1.1.1.1) ?_⟩
            rw [fac h.1.1.1.1.1, if_pos hlt, if_pos hq1]
            exact h.2
          · rw [if_neg hq1]
            refine (Sim.oneMinus fun h => ((hs h.1.1).get (by rw [← (hs h.1.1).1]; exact hlt)).mono upn).bind₀
              fun weight m₈' l₈' => (KxVar.scale (fun h => hX (h.1.1.2.2.1.mono (up5.trans l₈'))
                (h.1.1.2.2.2.mono (up5.trans l₈'))) fun h => h.2).bind₀ fun factor m₉ l₉ =>
              (KxVar.mul (fun h => (h.1.1.1.2.1.existing hR' hB' hst).mono (up.trans (l₈'.trans l₉)))
                fun h => h.2).bind₀ fun product m₁₀ l₁₀ => ?_
            dsimp only
            refine (KxVar.add (fun h => (h.1.1.1.2.existing hslot hnb2 hex).mono (l₈'.trans (l₉.trans l₁₀)))
              fun h => h.2).bind₀ fun merged m₁₁ l₁₁ => Sim.pure fun h =>
                ⟨rfl, (h.1.1.1.1.2.mono (l₈'.trans (l₉.trans (l₁₀.trans l₁₁)))).move hB' hBt'
                  (step h.1.1.1.1.1.1) ?_⟩
            rw [fac h.1.1.1.1.1.1, if_pos hlt, if_neg hq1]
            exact h.2
        · rw [if_neg hlt]
          refine (KxVar.mul (fun h => (h.1.2.1.existing hR' hB' hst).mono up)
            fun h => hX (h.1.2.2.1.mono up5) (h.1.2.2.2.mono up5)).bind₀ fun product m₁₀ l₁₀ => ?_
          dsimp only
          refine (KxVar.add (fun h => (h.1.2.existing hslot hnb2 hex).mono l₁₀) fun h => h.2).bind₀
            fun merged m₁₁ l₁₁ => Sim.pure fun h =>
              ⟨rfl, (h.1.1.2.mono (l₁₀.trans l₁₁)).move hB' hBt' (step h.1.1.1.1) ?_⟩
          rw [fac h.1.1.1.1, if_neg hlt]
          exact h.2
    · exact ⟨by simp, fun rem hr => ⟨by simp [getElem!_pos, hr], fun below hb => by
        rw [getElem!_pos _ rem (by simpa using hr), Array.getElem_replicate]
        rcases (show below = 0 ∨ below = 1 by omega) with rfl | rfl <;> rfl⟩⟩
    · refine h.2.congr fun rem hr below hb => ?_
      rw [show 4 * (D - 0) = 4 * D by omega, evalKStates_succ _ _ _ _ _ hr hb, ht', hprev]
  · refine ⟨by simp, fun rem hr => ?_⟩
    rw [getElem!_set! _ _ _ _ (by simpa using hr)]
    by_cases e : 0 = rem
    · subst e
      rw [if_pos rfl]
      refine ⟨rfl, fun below hb => ?_⟩
      beta_reduce
      rw [evalKStates_zero _ _ _ _ hr hb]
      rcases (show below = 0 ∨ below = 1 by omega) with rfl | rfl
      · show IsKx c m₂ one _
        rw [if_pos ⟨rfl, rfl⟩]
        exact h.2
      · show _ = Kx.zero
        rw [if_neg (by omega)]
    · rw [if_neg e, getElem!_pos _ rem (by simpa using hr), Array.getElem_replicate]
      refine ⟨rfl, fun below hb => ?_⟩
      beta_reduce
      rw [evalKStates_zero _ _ _ _ hr hb, if_neg (by omega)]
      rcases (show below = 0 ∨ below = 1 by omega) with rfl | rfl <;> rfl
  obtain ⟨last, hlast⟩ : ∃ last, last = Replay.evalKStates gr gs blocks cb cb := ⟨_, rfl⟩
  refine ((loop₀ (P := P ∧ IsStates c m₃ states fun rem below => (last[rem]!)[below]!)
    (fun j total m => IsKx c m total (Replay.sumKx j fun rem => ((last[rem]!)[1]!).scale gt[rem]!))
    fun j hj total m lm => ?_).pre fun h => ⟨⟨h.1.1.1, by
      have e := h.2
      rwa [show (List.range cb).reverse.length = cb by simp, ← hlast] at e⟩,
      h.1.1.2.mono (l₂.trans l₃)⟩).bind₀ fun total m₄ l₄ => ?_
  · have hj' : j < states.size := by simpa using hj
    rw [show states.zipIdx[j] = (states[j]!, j) by simp [getElem!_pos, hj']]
    dsimp only
    rcases hst : (states[j]!)[1]! with _ | value
    · refine Sim.pure fun h => ⟨rfl, ?_⟩
      rw [sumKx_succ, h.1.2.absent (h.1.2.1 ▸ hj') (by omega) hst, kx_zero_scale, kx_add_zero]
      exact h.2
    · refine (KxVar.scale (fun h => (h.1.2.existing (h.1.2.1 ▸ hj') (by omega) hst).mono lm)
        fun h => ((ht h.1.1).get (by have := h.1.2.1; omega)).mono (l₁.trans (l₂.trans (l₃.trans lm)))).bind₀
        fun term m' l' => (KxVar.add (fun h => h.1.2.mono l') fun h => h.2).bind₀ fun total' m'' l'' =>
        Sim.pure fun h => ⟨rfl, by rw [sumKx_succ]; exact h.2⟩
  have hk : ∀ j (hj : j < (r.extract cb r.size).size), P →
      IsK c n (r.extract cb r.size)[j] (gr.extract cb gr.size)[j]! :=
    fun j hj h => by
      have e := ((hr h).extract cb).get (i := j) (by rw [Array.size_extract, ← (hr h).1]; simpa using hj)
      rwa [getElem!_pos (r.extract _) j hj] at e
  refine ((loop₀ (P := P)
    (fun j acc m => IsKx c m acc (((gr.extract cb gr.size).toList.take j).foldl
      (fun total value => total.mul (Kx.ofK value).oneMinus) (Replay.sumKx D fun rem =>
        ((last[rem]!)[1]!).scale gt[rem]!)))
    fun j hj acc m lm => ?_).pre fun h => ⟨h.1.1.1.1, ?_⟩).bind₀ fun res m₅ l₅ => Sim.pure fun h => ?_
  · refine (KxVar.fromK (g₁ := ((gr.extract cb gr.size)[j]!).1) (g₂ := ((gr.extract cb gr.size)[j]!).2)
      (fun h => ((hk j hj h.1).1).mono (l₁.trans (l₂.trans (l₃.trans (l₄.trans lm)))))
      fun h => ((hk j hj h.1).2).mono (l₁.trans (l₂.trans (l₃.trans (l₄.trans lm))))).bind₀ fun x m' l' =>
      (KxVar.oneMinus fun h => h.2).bind₀ fun f m'' l'' =>
      (KxVar.mul (fun h => h.1.1.2.mono (l'.trans l'')) fun h => h.2).bind₀ fun total' m''' l''' =>
      Sim.pure fun h => ⟨rfl, ?_⟩
    have size := (hr h.1.1.1.1).1
    rw [List.take_succ_eq_append_getElem (by simp only [Array.length_toList, Array.size_extract]; simp at hj; omega),
      List.foldl_append, List.foldl_cons, List.foldl_nil, Array.getElem_toList, ← getElem!_pos]
    exact h.2
  · have e := h.2
    rw [Array.size_zipIdx, h.1.2.1] at e
    rw [List.take_zero, List.foldl_nil]
    exact e
  · have size := (hr h.1.1.1.1.1).1
    have e := h.2
    rw [List.take_of_length_le (by simp [size]), Array.foldl_toList, hlast, ← hcb, (hs h.1.1.1.1.1).1] at e
    exact e

end Sim

end CompressionLean.Circuit
