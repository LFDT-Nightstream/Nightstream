import CompressionLean.Circuit.SimAlgebra
import CompressionLean.Replay.Transcript

/-!
Owns the simulation of the hashes of `Hash.lean`: the duplex challenger
against the replay's (`Replay/Transcript.lean`), the leaf sponge, node
compression and Merkle paths against the replay's `hashLeaf`, `compress` and
`merkleRoot`. `permutes_poseidon2` gives `Permutes` for Goldilocks and
Poseidon2.

- `IsWords`: an array of handles is an array of words, entry by entry.
- `IsDuplex`: the builder's duplex is the replay's, lane by lane. `observe`,
  `sample` and their slice and extension forms keep it.
- The path words of a Merkle path are free: `merkleRoot` says that the
  handles of the computed root are the replay's root of the words that the
  path handles hold. Only the caller's root check binds those words.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Arrays -/

theorem getElem!_set! {α : Type} [Inhabited α] (xs : Array α) (j i : ℕ) (x : α) (h : i < xs.size) :
    (xs.set! j x)[i]! = if j = i then x else xs[i]! := by
  rw [Array.set!_eq_setIfInBounds, getElem!_pos _ _ (by simpa using h), Array.getElem_setIfInBounds h,
    getElem!_pos xs i h]

/-- `hs` and `gs` have one size, and each handle is its word. -/
def IsWords (c : Ctx F) (n : ℕ) (hs : Array Var) (gs : Array Gl) : Prop :=
  hs.size = gs.size ∧ ∀ i < hs.size, IsWord c n hs[i]! gs[i]!

namespace IsWords

variable {c : Ctx F} {n : ℕ} {hs : Array Var} {gs : Array Gl}

theorem mono (h : IsWords c n hs gs) {m : ℕ} (le : n ≤ m) : IsWords c m hs gs :=
  ⟨h.1, fun i hi => (h.2 i hi).mono le⟩

theorem get (h : IsWords c n hs gs) {i : ℕ} (hi : i < gs.size) : IsWord c n hs[i]! gs[i]! :=
  h.2 i (h.1 ▸ hi)

theorem empty : IsWords c n #[] #[] := ⟨rfl, fun _ h => absurd h (by simp)⟩

theorem set (h : IsWords c n hs gs) {x : Var} {g : Gl} (hx : IsWord c n x g) (j : ℕ) :
    IsWords c n (hs.set! j x) (gs.set! j g) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_set!] at hi
  rw [getElem!_set! _ _ _ _ hi, getElem!_set! _ _ _ _ (h.1 ▸ hi)]
  split_ifs
  · exact hx
  · exact h.2 i hi

theorem push (h : IsWords c n hs gs) {x : Var} {g : Gl} (hx : IsWord c n x g) :
    IsWords c n (hs.push x) (gs.push g) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_push] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_push,
    Array.getElem_push]
  by_cases lt : i < hs.size
  · rw [dif_pos lt, dif_pos (h.1 ▸ lt), ← getElem!_pos, ← getElem!_pos]
    exact h.2 i lt
  · rw [dif_neg lt, dif_neg (h.1 ▸ lt)]
    exact hx

theorem pop (h : IsWords c n hs gs) : IsWords c n hs.pop gs.pop := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_pop] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_pop,
    Array.getElem_pop, ← getElem!_pos, ← getElem!_pos]
  exact h.2 i (by omega)

theorem back (h : IsWords c n hs gs) (ne : 0 < hs.size) : IsWord c n hs.back! gs.back! := by
  have e := h.1
  rw [Array.back!, Array.back!, h.1]
  exact h.get (by omega)

theorem extract (h : IsWords c n hs gs) (a b : ℕ) : IsWords c n (hs.extract a b) (gs.extract a b) := by
  refine ⟨by simp [h.1], fun i hi => ?_⟩
  rw [Array.size_extract] at hi
  rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by simp [← h.1]; omega), Array.getElem_extract,
    Array.getElem_extract, ← getElem!_pos, ← getElem!_pos]
  exact h.2 _ (by omega)

theorem replicate {z : Var} {g : Gl} (hz : IsWord c n z g) (k : ℕ) :
    IsWords c n (Array.replicate k z) (Array.replicate k g) := by
  refine ⟨by simp, fun i hi => ?_⟩
  rw [Array.size_replicate] at hi
  rw [getElem!_pos _ _ (by simpa using hi), getElem!_pos _ _ (by simpa using hi), Array.getElem_replicate,
    Array.getElem_replicate]
  exact hz

theorem ofFn {k : ℕ} {f : Fin k → Var} {gs : Array Gl} (size : gs.size = k) (hf : ∀ i : Fin k, IsWord c n (f i) gs[i]!) :
    IsWords c n (Array.ofFn f) gs := by
  refine ⟨by simp [size], fun i hi => ?_⟩
  rw [Array.size_ofFn] at hi
  rw [getElem!_pos _ _ (by simpa using hi), Array.getElem_ofFn]
  exact hf ⟨i, hi⟩

/-- Overwrite lanes `lane i` for `i < k` with `f i` and `g i`. -/
theorem setRange (h : IsWords c n hs gs) (k : ℕ) (lane : ℕ → ℕ) {f : ℕ → Var} {g : ℕ → Gl}
    (hf : ∀ i < k, IsWord c n (f i) (g i)) :
    IsWords c n ((List.range k).foldl (fun s i => s.set! (lane i) (f i)) hs)
      ((List.range k).foldl (fun s i => s.set! (lane i) (g i)) gs) := by
  induction k with
  | zero => exact h
  | succ k ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_append]
      exact (ih fun i hi => hf i (by omega)).set (hf k (by omega)) _

end IsWords

/-- The `i`-th pair of `l.zipIdx` is `(l[i], i)`. -/
theorem foldl_zipIdx {α β : Type} [Inhabited α] (l : List α) (f : β → α → ℕ → β) (s : β) :
    l.zipIdx.foldl (fun s x => f s x.1 x.2) s = (List.range l.length).foldl (fun s i => f s l[i]! i) s := by
  induction l using List.reverseRecOn generalizing s with
  | nil => rfl
  | append_singleton l a ih =>
      rw [List.zipIdx_append, List.foldl_append, ih, List.length_append, List.length_singleton, List.range_succ,
        List.foldl_append]
      simp only [List.zipIdx_singleton, Nat.zero_add, List.foldl_cons, List.foldl_nil]
      congr 1
      · apply List.foldl_ext
        intro t i hi
        rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD,
          List.getElem?_append_left (List.mem_range.mp hi)]
      · rw [List.getElem!_eq_getElem?_getD, List.getElem?_append_right (le_refl _), Nat.sub_self]
        rfl

theorem foldl_zipIdx_set {α : Type} [Inhabited α] (l : List α) (s : Array α) :
    l.zipIdx.foldl (fun s x => s.set! x.2 x.1) s = (List.range l.length).foldl (fun s i => s.set! i l[i]!) s :=
  foldl_zipIdx l (fun s v i => s.set! i v) s

theorem size_setRange {α : Type} (k : ℕ) (lane : ℕ → ℕ) (f : ℕ → α) (s : Array α) :
    ((List.range k).foldl (fun s i => s.set! (lane i) (f i)) s).size = s.size := by
  induction k with
  | zero => rfl
  | succ k ih => rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Array.size_set!, ih]

theorem range_foldl {α : Type} (a b : ℕ) (f : ℕ → α → α) (s : α) :
    (List.range' a (([a:b] : Std.Legacy.Range).size)).foldl (fun s i => f i s) s =
      (List.range (b - a)).foldl (fun s i => f (a + i) s) s := by
  rw [List.range'_eq_map_range, List.foldl_map]
  congr 2
  simp [Std.Legacy.Range.size]

/-- A loop whose body is pure is a fold. -/
theorem forIn_pure {m : Type → Type} [Monad m] [LawfulMonad m] {α β : Type} {l : List α} {init : β}
    {body : α → β → m (ForInStep β)} (f : α → β → β) (h : ∀ x s, body x s = pure (.yield (f x s))) :
    forIn l init body = pure (l.foldl (fun s x => f x s) init) := by
  rw [show body = fun x s => pure (.yield (f x s)) from funext fun x => funext fun s => h x s]
  exact List.forIn_pure_yield_eq_foldl f init

theorem array_toList_getElem! {α : Type} [Inhabited α] (xs : Array α) (i : ℕ) : xs.toList[i]! = xs[i]! := by
  rw [List.getElem!_eq_getElem?_getD, Array.getElem?_toList, getElem!_def]
  cases xs[i]? <;> rfl

/-! ## The duplex -/

/-- The builder's duplex is the replay's, lane by lane; its state has 16
lanes. -/
structure IsDuplex (c : Ctx F) (n : ℕ) (d : Duplex) (r : Replay.Duplex) : Prop where
  state : IsWords c n d.state r.state
  input : IsWords c n d.input r.input
  output : IsWords c n d.output r.output
  width : r.state.size = 16

theorem IsDuplex.mono {c : Ctx F} {n m : ℕ} {d : Duplex} {r : Replay.Duplex} (h : IsDuplex c n d r) (le : n ≤ m) :
    IsDuplex c m d r :=
  ⟨h.state.mono le, h.input.mono le, h.output.mono le, h.width⟩

theorem permute_size (s : Array Gl) : (Replay.permute s).size = 16 := rfl

/- Unifying two terms that contain the permutation must never evaluate it. -/
attribute [local irreducible] Replay.permute

/-- The duplex after a permutation, from the replay's last state `t`. -/
theorem IsDuplex.permuted {c : Ctx F} {m : ℕ} (t : Array Gl) (outs : Fin 16 → Var)
    (h : ∀ k : Fin 16, IsWord c m (outs k) (Replay.permute t)[k]!) :
    IsDuplex c m ({ state := Array.ofFn outs, input := #[], output := (Array.ofFn outs).extract 0 rate } : Duplex)
      ⟨Replay.permute t, #[], (Replay.permute t).extract 0 Replay.rate⟩ := by
  have wo := IsWords.ofFn (permute_size _) h
  refine ⟨wo, IsWords.empty, ?_, permute_size _⟩
  dsimp only
  exact wo.extract 0 rate

theorem duplexing_output_size (r : Replay.Duplex) : r.duplexing.output.size = Replay.rate := by
  unfold Replay.Duplex.duplexing
  simp [permute_size, Replay.rate]

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

omit [CharP F goldilocksModulus] in
theorem duplexNew : Sim c n P Duplex.new (fun d m => IsDuplex c m d Replay.Duplex.new) True :=
  ((Sim.const 0).bind fun _ _ _ => Sim.pure fun h =>
    ⟨IsWords.replicate h.2 16, IsWords.empty, IsWords.empty, rfl⟩).post (fun _ _ _ h => h) fun _ => by simp

/-- `duplexing`: overwrite the absorbed lanes; after input, zero the rest of
the rate and add the length tag; permute; refill the output. -/
theorem duplexing {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r) :
    Sim c n P (Duplex.duplexing.run d) (fun res m => IsDuplex c m res.2 r.duplexing) True := by
  unfold Duplex.duplexing
  rw [StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  rw [← Array.forIn_toList, Array.toList_zipIdx, forIn_pure (fun x s => s.set! x.2 x.1) (fun x s => by cases x; rfl),
    StateT.run_bind, StateT.run_pure, pure_bind]
  dsimp only
  rw [foldl_zipIdx_set]
  simp only [array_toList_getElem!, Array.length_toList]
  generalize hs₁ : (List.range d.input.size).foldl (fun s i => s.set! i d.input[i]!) d.state = s₁
  let t₁ := (List.range d.input.size).foldl (fun s i => s.set! i r.input[i]!) r.state
  have ws₁ : P → IsWords c n s₁ t₁ := fun h => by
    rw [← hs₁]
    exact (hd h).state.setRange _ (fun i => i) fun i hi => (hd h).input.get (by rw [← (hd h).input.1]; exact hi)
  have wt₁ : P → t₁.size = 16 := fun h => by simp only [t₁, size_setRange, (hd h).width]
  by_cases hpos : d.input.size > 0
  · rw [if_pos hpos]
    refine Sim.tbind₀ (Sim.lift (Sim.const 0)) fun z e₁ m₁ l₁ => ?_
    dsimp only
    rw [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, StateT.run_bind, StateT.run_pure,
      pure_bind]
    dsimp only
    rw [range_foldl d.input.size rate (fun i (s : Array Var) => s.set! i z)]
    generalize hs₂ : (List.range (rate - d.input.size)).foldl (fun s i => s.set! (d.input.size + i) z) s₁ = s₂
    let t₂ := (List.range (rate - d.input.size)).foldl (fun s i => s.set! (d.input.size + i) (0 : Gl)) t₁
    let t₃ := t₂.set! rate (t₂[rate]! + Gl.ofNat d.input.size)
    have wt₂ : P → t₂.size = 16 := fun h => by simp only [t₂, size_setRange, wt₁ h]
    have ws₂ : P → IsWord c m₁ z 0 → IsWords c m₁ s₂ t₂ := fun h hz => by
      rw [← hs₂]
      exact ((ws₁ h).mono l₁).setRange _ (fun i => d.input.size + i) (f := fun _ => z) (g := fun _ => 0)
        fun _ _ => hz
    refine Sim.tbind₀ (Sim.lift (Sim.const (Gl.ofNat d.input.size))) fun t e₂ m₂ l₂ => ?_
    dsimp only
    have add := Sim.add (c := c) (n := m₂) (P := (P ∧ IsWord c m₁ z 0 ∧ e₁ = d) ∧
        IsWord c m₂ t (Gl.ofNat d.input.size) ∧ e₂ = e₁) (a := s₂[rate]!) (b := t) (ga := t₂[rate]!)
      (gb := Gl.ofNat d.input.size) (fun h => ((ws₂ h.1.1 h.1.2.1).mono l₂).get (by rw [wt₂ h.1.1]; decide))
      fun h => h.2.1
    refine Sim.tbind₀ (Sim.lift add) fun sum e₃ m₃ l₃ => ?_
    dsimp only
    have perm := Sim.permute (c := c) (n := m₃) (gs := t₃) (state := fun lane => (s₂.set! rate sum)[lane]!) hp
      (P := ((P ∧ IsWord c m₁ z 0 ∧ e₁ = d) ∧ IsWord c m₂ t (Gl.ofNat d.input.size) ∧ e₂ = e₁) ∧
        IsWord c m₃ sum (t₂[rate]! + Gl.ofNat d.input.size) ∧ e₃ = e₂)
      fun h k => (((ws₂ h.1.1.1 h.1.1.2.1).mono (l₂.trans l₃)).set h.2.1 rate).get
        (by rw [Array.size_set!, wt₂ h.1.1.1]; exact k.isLt)
    refine Sim.tbind₀ (Sim.lift perm) fun outs e₄ m₄ l₄ => ?_
    rw [StateT.run_set]
    refine Sim.pure fun h => ?_
    have hr : r.duplexing = ⟨Replay.permute t₃, #[], (Replay.permute t₃).extract 0 Replay.rate⟩ := by
      unfold Replay.Duplex.duplexing
      simp only [← (hd h.1.1.1.1).input.1, if_pos hpos]
      rfl
    rw [hr]
    exact IsDuplex.permuted t₃ outs h.2.1
  · rw [if_neg hpos]
    have perm := Sim.permute (c := c) (n := n) (gs := t₁) (state := fun lane => s₁[lane]!) hp (P := P)
      fun h k => (ws₁ h).get (by rw [wt₁ h]; exact k.isLt)
    refine Sim.tbind₀ (Sim.lift perm) fun outs e m l => ?_
    rw [StateT.run_set]
    refine Sim.pure fun h => ?_
    have hr : r.duplexing = ⟨Replay.permute t₁, #[], (Replay.permute t₁).extract 0 Replay.rate⟩ := by
      unfold Replay.Duplex.duplexing
      simp only [← (hd h.1).input.1, if_neg hpos]
      rfl
    rw [hr]
    exact IsDuplex.permuted t₁ outs h.2.1

/-- `observe`: push the word, clear the output, and duplex at a full rate. -/
theorem observe {d : Duplex} {r : Replay.Duplex} {v : Var} {g : Gl} (hp : Permutes c.perm)
    (hd : P → IsDuplex c n d r) (hv : P → IsWord c n v g) :
    Sim c n P ((Duplex.observe v).run d) (fun res m => IsDuplex c m res.2 (r.observe g)) True := by
  unfold Duplex.observe
  rw [StateT.run_bind, StateT.run_modify, pure_bind, StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  have hd' : P → IsDuplex c n ⟨d.state, d.input.push v, #[]⟩ ⟨r.state, r.input.push g, #[]⟩ := fun h =>
    ⟨(hd h).state, (hd h).input.push (hv h), IsWords.empty, (hd h).width⟩
  by_cases full : (d.input.push v).size = rate
  · rw [if_pos full]
    refine (duplexing hp hd').post (fun h _ _ q => ?_) fun _ => Iff.rfl
    have e : (r.input.push g).size = Replay.rate := by rw [← (hd' h).input.1]; exact full
    have hr : r.observe g = Replay.Duplex.duplexing ⟨r.state, r.input.push g, #[]⟩ := by
      simp only [Replay.Duplex.observe, e, beq_self_eq_true, if_true]
    rw [hr]
    exact q
  · rw [if_neg full, StateT.run_pure]
    refine Sim.pure fun h => ?_
    have e : ¬(r.input.push g).size = Replay.rate := by rw [← (hd' h).input.1]; exact full
    have hr : r.observe g = ⟨r.state, r.input.push g, #[]⟩ := by
      simp only [Replay.Duplex.observe, beq_iff_eq, e, if_false]
    rw [hr]
    exact hd' h

/-- `observe` of each word of a list, in order. -/
theorem observeList (hp : Permutes c.perm) :
    ∀ (l : List Var) (gl : List Gl) {n : ℕ} {P : Prop} {d : Duplex} {r : Replay.Duplex}, (P → IsDuplex c n d r) →
      (P → l.length = gl.length ∧ ∀ i < l.length, IsWord c n l[i]! gl[i]!) →
      Sim c n P ((l.foldlM (fun _ v => Duplex.observe v) ()).run d)
        (fun res m => IsDuplex c m res.2 (gl.foldl Replay.Duplex.observe r)) True
  | [], gl, n, P, d, r, hd, hl => by
      rw [List.foldlM_nil, StateT.run_pure]
      exact Sim.pure fun h => by
        rw [List.eq_nil_of_length_eq_zero (hl h).1.symm]
        exact hd h
  | v :: l, gl, n, P, d, r, hd, hl => by
      rw [List.foldlM_cons, StateT.run_bind]
      obtain _ | ⟨g, gl⟩ := gl
      · exact (Sim.bind₀ (observe hp hd (g := 0) fun h => absurd (hl h).1 (by simp)) fun _ _ _ =>
          observeList hp l [] (fun h => h.2) fun h => absurd (hl h.1).1 (by simp)).post
          (fun h => absurd (hl h).1 (by simp)) fun h => absurd (hl h).1 (by simp)
      refine Sim.bind₀ (observe hp hd fun h => by simpa using (hl h).2 0 (by simp)) fun res m le => ?_
      exact observeList hp l gl (fun h => h.2) fun h => ⟨by simpa using (hl h.1).1, fun i hi => by
        simpa using ((hl h.1).2 (i + 1) (by simp; omega)).mono le⟩

/-- `observe_slice`: the replay's `foldl observe`. -/
theorem observeSlice {d : Duplex} {r : Replay.Duplex} {vs : Array Var} {gs : Array Gl} (hp : Permutes c.perm)
    (hd : P → IsDuplex c n d r) (hv : P → IsWords c n vs gs) :
    Sim c n P ((Duplex.observeSlice vs).run d) (fun res m => IsDuplex c m res.2 (gs.foldl Replay.Duplex.observe r))
      True := by
  unfold Duplex.observeSlice Array.forM
  rw [← Array.foldlM_toList, ← Array.foldl_toList]
  exact observeList hp vs.toList gs.toList hd fun h => ⟨by simpa using (hv h).1, fun i hi => by
    rw [array_toList_getElem!, array_toList_getElem!]
    exact (hv h).2 i (by simpa using hi)⟩

/-- `observe_ext`: the three coordinates in order. -/
theorem observeExt {d : Duplex} {r : Replay.Duplex} {a : ExtVar} {e : Ext} (hp : Permutes c.perm)
    (hd : P → IsDuplex c n d r) (ha : P → IsExt c n a e) :
    Sim c n P ((Duplex.observeExt a).run d)
      (fun res m => IsDuplex c m res.2 (((r.observe e.c0).observe e.c1).observe e.c2)) True := by
  unfold Duplex.observeExt
  rw [StateT.run_bind]
  refine Sim.bind₀ (observe hp hd fun h => ha h 0) fun res₁ m₁ l₁ => ?_
  rw [StateT.run_bind]
  refine Sim.bind₀ (observe hp (fun h => h.2) fun h => (ha h.1 1).mono l₁) fun res₂ m₂ l₂ => ?_
  exact observe hp (fun h => h.2) fun h => (ha h.1.1 2).mono (l₁.trans l₂)

omit [CharP F goldilocksModulus] in
/-- Pop the last output lane of a duplex with output. -/
theorem pop {d : Duplex} {r : Replay.Duplex} (hd : P → IsDuplex c n d r) (ne : P → 0 < r.output.size) :
    Sim c n P ((do
        let d ← get
        set { d with output := d.output.pop }
        Pure.pure (d.output.back!) : Transcript Var).run d)
      (fun res m => IsWord c m res.1 r.output.back! ∧ IsDuplex c m res.2 { r with output := r.output.pop }) True := by
  rw [StateT.run_bind, StateT.run_get, pure_bind, StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure]
  refine Sim.pure fun h => ⟨(hd h).output.back (by rw [(hd h).output.1]; exact ne h), ?_⟩
  exact ⟨(hd h).state, (hd h).input, (hd h).output.pop, (hd h).width⟩

theorem sample {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r) :
    Sim c n P (Duplex.sample.run d)
      (fun res m => IsWord c m res.1 r.sample.1 ∧ IsDuplex c m res.2 r.sample.2) True := by
  unfold Duplex.sample
  rw [StateT.run_bind, StateT.run_get, pure_bind]
  dsimp only
  have same : P → ((!d.input.isEmpty || d.output.isEmpty) = (!r.input.isEmpty || r.output.isEmpty)) := fun h => by
    unfold Array.isEmpty
    rw [(hd h).input.1, (hd h).output.1]
  by_cases cond : (!d.input.isEmpty || d.output.isEmpty) = true
  · rw [if_pos cond, StateT.run_bind]
    refine Sim.bind₀ (duplexing hp hd) fun res m le => ?_
    refine (pop (fun h => h.2) fun _ => by rw [duplexing_output_size]; decide).post (fun h _ _ q => ?_) fun _ => Iff.rfl
    have e : r.sample = (r.duplexing.output.back!, { r.duplexing with output := r.duplexing.output.pop }) := by
      unfold Replay.Duplex.sample
      rw [if_pos ((same h.1).symm.trans cond)]
    rw [e]
    exact q
  · rw [if_neg cond]
    have ne : P → 0 < r.output.size := fun h => by
      have := (same h).symm.trans (eq_false_of_ne_true cond)
      simp only [Bool.or_eq_false_iff, Bool.not_eq_false', Array.isEmpty] at this
      exact Array.size_pos_iff.mpr (by simpa using this.2)
    refine (pop hd ne).post (fun h _ _ q => ?_) fun _ => Iff.rfl
    have e : r.sample = (r.output.back!, { r with output := r.output.pop }) := by
      unfold Replay.Duplex.sample
      rw [if_neg (fun x => cond ((same h).trans x))]
    rw [e]
    exact q

/-- `sample_ext`: three samples, coordinate `i` the `i`-th. -/
theorem sampleExt {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r) :
    Sim c n P (Duplex.sampleExt.run d)
      (fun res m => IsExt c m res.1 ⟨r.sample.1, r.sample.2.sample.1, r.sample.2.sample.2.sample.1⟩ ∧
        IsDuplex c m res.2 r.sample.2.sample.2.sample.2) True := by
  unfold Duplex.sampleExt
  rw [StateT.run_bind]
  refine Sim.bind₀ (sample hp hd) fun x m₁ l₁ => ?_
  rw [StateT.run_bind]
  refine Sim.bind₀ (sample hp fun h => h.2.2) fun y m₂ l₂ => ?_
  rw [StateT.run_bind]
  refine Sim.bind₀ (sample hp fun h => h.2.2) fun z m₃ l₃ => ?_
  rw [StateT.run_pure]
  refine Sim.pure fun h => ⟨fun k => ?_, h.2.2⟩
  fin_cases k
  · exact h.1.1.2.1.mono (l₂.trans l₃)
  · exact h.1.2.1.mono l₃
  · exact h.2.1

end Sim

end CompressionLean.Circuit
