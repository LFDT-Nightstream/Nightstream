import CompressionLean.Circuit.MirrorAlgebra
import CompressionLean.Circuit.MirrorRead

/-!
Owns the simulation of the batched GKR verifier (`Gkr.lean`) against the
replay's (`Replay/Gkr.lean`): the child combination (`gkrCombine`), the
batched claims of a layer, and the layer loop against the replay's
recursion.
-/

namespace CompressionLean.Circuit
open CompressionLean.Replay (Gl Ext)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

namespace Sim
variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `combine`: `p₀·q₁ + p₁·q₀ + μ·q₀·q₁` from one tree's children. -/
theorem gkrCombine {ch : Array ExtVar} {gch : Array Ext} {mu : ExtVar} {gmu : Ext} (hch : P → IsExts c n ch gch)
    (two : P → 2 ≤ gch.size) (hmu : P → IsExt c n mu gmu) :
    Sim c n P (Circuit.gkrCombine ch mu) (fun r m => IsExt c m r (Replay.combineVals gch gmu)) True := by
  unfold Circuit.gkrCombine
  simp only [Std.Legacy.Range.forIn_eq_forIn_range']
  have size : P → ch.size = gch.size := fun h => (hch h).1
  have get (i : ℕ) {m : ℕ} (hi : i < gch.size) (le : n ≤ m) : P → IsExt c m ch[i]! gch[i]! := fun h =>
    ((hch h).get hi).mono le
  have numerator (side : ℕ) (hs : side < 2) {m : ℕ} {P' : Prop} (pre : P' → P) (le : n ≤ m) :
      Sim c m P' (Algebra.extOne >>= fun total =>
          forIn (List.range' 0 ([:ch.size / 2 - 1] : Std.Legacy.Range).size) total (fun j s =>
            Build.extMul s ch[2 * j + side]! >>= fun total => Pure.pure (ForInStep.yield total)) >>= fun s =>
          Pure.pure s)
        (fun r m => IsExt c m r (Replay.prodRange (gch.size / 2 - 1) fun j => gch[2 * j + side]!)) True := by
    rw [show ([:ch.size / 2 - 1] : Std.Legacy.Range).size = ch.size / 2 - 1 by simp [Std.Legacy.Range.size],
      ← List.range_eq_range']
    have loopTrue {k : ℕ} : (∀ i < k, True) ↔ True := ⟨fun _ => trivial, fun _ _ _ => trivial⟩
    refine Sim.bind₀ extOne fun one m₁ l₁ => ?_
    refine Sim.bind₀ (((Sim.loop (c := c) (n := m₁) (P := P' ∧ IsExt c m₁ one 1) (I := fun k acc m => IsExt c m acc
      (Replay.prodRange k fun j => gch[2 * j + side]!)) (D := fun _ => True) fun k hk acc m' le' => ?_).pre
        fun (h : P' ∧ IsExt c m₁ one 1) => ⟨h, h.2⟩).post (fun _ _ _ h => h) fun _ => loopTrue)
          fun r m₂ l₂ => ?_
    swap
    · refine Sim.pure fun h => ?_
      have := h.2
      rwa [List.length_range, size (pre h.1.1)] at this
    simp only [List.getElem_range]
    have hk' : P' → 2 * k + side < gch.size := fun h => by
      have := two (pre h)
      have := size (pre h)
      simp at hk
      omega
    refine Sim.bind₀ (Sim.extMul (fun h => h.2) fun h => get _ (hk' h.1.1) (le.trans (l₁.trans le')) (pre h.1.1))
      fun t m₃ l₃ => Sim.pure fun h => ⟨rfl, ?_⟩
    rw [show Replay.prodRange (k + 1) (fun j => gch[2 * j + side]!) =
      Replay.prodRange k (fun j => gch[2 * j + side]!) * gch[2 * k + side]! from range_foldl_succ _ _ _]
    exact h.2
  have q (side : ℕ) (hs : side < 2) {m : ℕ} (le : n ≤ m) : P → IsExt c m ch[2 * (ch.size / 2 - 1) + side]!
      gch[2 * (gch.size / 2 - 1) + side]! := fun h => by
    have := two h
    rw [size h]
    exact get _ (by omega) le h
  refine Sim.bind₀ (numerator 0 (by omega) id le_rfl) fun p0 m₁ l₁ => ?_
  refine Sim.bind₀ (numerator 1 (by omega) (fun h => h.1) l₁) fun p1 m₂ l₂ => ?_
  refine Sim.bind₀ (Sim.extMul (fun h => h.1.2.mono l₂) (q 1 (by omega) (l₁.trans l₂) ∘ fun h => h.1.1))
    fun a m₃ l₃ => ?_
  refine Sim.bind₀ (Sim.extMul (fun h => h.1.2.mono l₃) (q 0 (by omega) (l₁.trans (l₂.trans l₃)) ∘
    fun h => h.1.1.1)) fun c' m₄ l₄ => ?_
  have l₄' : n ≤ m₄ := l₁.trans (l₂.trans (l₃.trans l₄))
  refine Sim.bind₀ (Sim.extMul (q 0 (by omega) l₄' ∘ fun h => h.1.1.1.1)
    (q 1 (by omega) l₄' ∘ fun h => h.1.1.1.1)) fun qq m₅ l₅ => ?_
  refine Sim.bind₀ (Sim.extMul (fun h => (hmu h.1.1.1.1.1).mono (l₄'.trans l₅)) fun h => h.2) fun d m₆ l₆ => ?_
  refine Sim.bind₀ (extAdd (fun h => h.1.1.1.2.mono (l₄.trans (l₅.trans l₆))) fun h => h.1.1.2.mono (l₅.trans l₆))
    fun sum m₇ l₇ => ?_
  refine (extAdd (fun h => h.2) fun h => h.1.2.mono l₇).post (fun h _ _ hr => hr.of_eq (toC_inj ?_)) fun _ => Iff.rfl
  simp only [Replay.combineVals, Nat.add_zero, toC_add, toC_mul]
  ring
end Sim

/-! ## Pairs of extension values -/

/-- `ps` and `gs` have one size, and each pair of handle triples is its pair. -/
def IsPairs {F : Type} [CommRing F] (c : Ctx F) (n : ℕ) (ps : Array (ExtVar × ExtVar)) (gs : Array (Ext × Ext)) :
    Prop :=
  ps.size = gs.size ∧ ∀ i < ps.size, IsExt c n ps[i]!.1 gs[i]!.1 ∧ IsExt c n ps[i]!.2 gs[i]!.2

theorem IsPairs.mono {c : Ctx F} {n m : ℕ} {ps : Array (ExtVar × ExtVar)} {gs : Array (Ext × Ext)}
    (h : IsPairs c n ps gs) (le : n ≤ m) : IsPairs c m ps gs :=
  ⟨h.1, fun i hi => ⟨(h.2 i hi).1.mono le, (h.2 i hi).2.mono le⟩⟩

theorem foldl_observe_pairs (gs : Array (Ext × Ext)) (r : Replay.Duplex) :
    (Replay.flatPairs gs).foldl observe3 r = gs.foldl (fun r p => observe3 (observe3 r p.1) p.2) r := by
  rw [Replay.flatPairs, ← Array.foldl_toList, ← Array.foldl_toList, Array.toList_flatMap]
  generalize gs.toList = l
  induction l generalizing r with
  | nil => rfl
  | cons p l ih => simp [List.flatMap_cons, ih]

namespace Sim
variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `for (p, q) in pairs do observe_ext p; observe_ext q`. -/
theorem observePairsFor {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    {ps : Array (ExtVar × ExtVar)} {gs : Array (Ext × Ext)} (hv : P → IsPairs c n ps gs) :
    Sim c n P (StateT.run (forIn (m := Transcript) ps.toList PUnit.unit fun x (_ : PUnit) =>
        Duplex.observeExt x.1 >>= fun _ => Duplex.observeExt x.2 >>= fun _ => Pure.pure (ForInStep.yield PUnit.unit))
        d)
      (fun res m => IsDuplex c m res.2 (gs.foldl (fun r p => observe3 (observe3 r p.1) p.2) r)) True := by
  have len : P → ps.toList.length = gs.toList.length := fun h => by simp [(hv h).1]
  refine ((Sim.loopT (I := fun k _ e m => IsDuplex c m e ((gs.toList.take k).foldl
    (fun r p => observe3 (observe3 r p.1) p.2) r)) (D := fun _ => True)
    fun k hk _ e m le => ?_).pre fun h => ⟨h, by simpa using hd h⟩).post (fun h _ _ q => ?_) fun _ => by simp
  · have hk' : P → k < gs.size := fun h => by have := len h; simp at this hk; omega
    have hpk : P → ps.toList[k] = ps[k]! := fun _ => by
      rw [Array.getElem_toList, ← getElem!_pos ps k (by simpa using hk)]
    refine Sim.tbind₀ (observeExt hp (e := gs[k]!.1) (fun h => h.2) fun h => ?_) fun _ e₁ m₁ l₁ => ?_
    · rw [hpk h.1]
      exact ((hv h.1).2 k (by simpa using hk)).1.mono le
    refine Sim.tbind₀ (observeExt hp (e := gs[k]!.2) (fun h => h.2) fun h => ?_) fun _ e₂ m₂ l₂ => ?_
    · rw [hpk h.1.1]
      exact ((hv h.1.1).2 k (by simpa using hk)).2.mono (le.trans l₁)
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_⟩
    have hk'' : k < gs.toList.length := by simpa using hk' h.1.1.1
    rw [List.take_succ_eq_append_getElem hk'', List.foldl_append, Array.getElem_toList,
      ← getElem!_pos gs k (by simpa using hk''), List.foldl_cons, List.foldl_nil]
    unfold observe3
    exact h.2
  · rw [len h, List.take_length, Array.foldl_toList] at q
    exact q

/-- `for values in lists do for value in values do observe_ext value`. -/
theorem observeNestedFor {d : Duplex} {r : Replay.Duplex} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    {vss : Array (Array ExtVar)} {ess : Array (Array Ext)}
    (hv : P → vss.size = ess.size ∧ ∀ i (h : i < vss.size) (h' : i < ess.size), IsExts c n vss[i] ess[i]) :
    Sim c n P (StateT.run (forIn (m := Transcript) vss.toList PUnit.unit fun values (_ : PUnit) =>
        forIn (m := Transcript) values.toList PUnit.unit (fun value (_ : PUnit) =>
          Duplex.observeExt value >>= fun _ => Pure.pure (ForInStep.yield PUnit.unit)) >>= fun _ =>
        Pure.pure (ForInStep.yield PUnit.unit)) d)
      (fun res m => IsDuplex c m res.2 (ess.foldl (fun r es => es.foldl observe3 r) r)) True := by
  have len : P → vss.toList.length = ess.toList.length := fun h => by simp [(hv h).1]
  refine ((Sim.loopT (I := fun k _ e m => IsDuplex c m e ((ess.toList.take k).foldl
    (fun r es => es.foldl observe3 r) r)) (D := fun _ => True)
    fun k hk _ e m le => ?_).pre fun h => ⟨h, by simpa using hd h⟩).post (fun h _ _ q => ?_) fun _ => by simp
  · have hk' : P → k < ess.size := fun h => by have := len h; simp at this hk; omega
    refine Sim.tbind₀ (observeExtsFor hp (es := ess[k]!) (fun h => h.2) fun h => ?_) fun _ e₁ m₁ l₁ => ?_
    · rw [Array.getElem_toList, getElem!_pos ess k (hk' h.1)]
      exact ((hv h.1).2 k (by simpa using hk) (hk' h.1)).mono le
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_⟩
    have hk'' : k < ess.toList.length := by simpa using hk' h.1.1
    rw [List.take_succ_eq_append_getElem hk'', List.foldl_append, Array.getElem_toList,
      ← getElem!_pos ess k (by simpa using hk''), List.foldl_cons, List.foldl_nil]
    exact h.2
  · rw [len h, List.take_length, Array.foldl_toList] at q
    exact q

end Sim

theorem foldl_observe_flatten (ess : Array (Array Ext)) (r : Replay.Duplex) :
    ess.flatten.foldl observe3 r = ess.foldl (fun r es => es.foldl observe3 r) r := by
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_flatten]
  generalize ess.toList = l
  induction l generalizing r with
  | nil => rfl
  | cons es l ih => simp [List.flatten_cons, List.foldl_append, ih]

theorem ext_beq_of_eq {a b : Ext} (h : a = b) : (a == b) = true := by
  subst h
  obtain ⟨x, y, z⟩ := a
  show (x.word == x.word && y.word == y.word && z.word == z.word) = true
  simp

/-! ## Active trees -/

theorem activeTrees_nodup (shapes : Array Replay.TreeShape) (k : ℕ) : (Replay.activeTrees shapes k).toList.Nodup := by
  unfold Replay.activeTrees
  rw [Array.toList_filter, Array.toList_range]
  exact (List.nodup_range).filter _

theorem activeTrees_lt (shapes : Array Replay.TreeShape) (k : ℕ) {t : ℕ} (h : t ∈ Replay.activeTrees shapes k) :
    t < shapes.size ∧ k ≤ shapes[t]!.depth := by
  unfold Replay.activeTrees at h
  simp only [Array.mem_filter, Array.mem_range, decide_eq_true_eq] at h
  exact h

theorem find_nodup {β : Type} : ∀ (l : List (ℕ × β)) (k : ℕ) (hk : k < l.length), (l.map Prod.fst).Nodup →
    l.find? (fun x => x.1 == l[k].1) = some l[k]
  | [], k, hk, _ => absurd hk (by simp)
  | x :: l, 0, _, _ => by simp
  | x :: l, k + 1, hk, nd => by
      rw [List.map_cons, List.nodup_cons] at nd
      have ne : x.1 ≠ (l[k]'(by simpa using hk)).1 := fun e =>
        nd.1 (e ▸ List.mem_map.mpr ⟨_, List.getElem_mem _, rfl⟩)
      rw [List.getElem_cons_succ, List.find?_cons_of_neg (by simpa using ne)]
      exact find_nodup l k _ nd.2

theorem zip_fst_sublist {α β : Type} : ∀ (l₁ : List α) (l₂ : List β), ((l₁.zip l₂).map Prod.fst).Sublist l₁
  | [], _ => by simp
  | _ :: _, [] => by simp
  | a :: l₁, b :: l₂ => by simpa using (zip_fst_sublist l₁ l₂).cons_cons a

/-- The reduced pairs of a tree's children. -/
theorem reducedOf_get {act : Array ℕ} {ch : Array (Array Ext)} (nd : act.toList.Nodup) (tau : Ext) {k : ℕ}
    (hk : k < act.size) (hk' : k < ch.size) :
    Replay.reducedOf (act.zip ch).toList tau act[k]! =
      (Array.range (ch[k]!.size / 2)).map fun j => Replay.reduce ((ch[k]!)[2 * j]!, (ch[k]!)[2 * j + 1]!) tau := by
  have hl : k < (act.zip ch).toList.length := by simp; omega
  have e : ((act.zip ch).toList[k]).1 = act[k]! := by
    rw [Array.getElem_toList, Array.getElem_zip, getElem!_pos act k hk]
  have nd' : ((act.zip ch).toList.map Prod.fst).Nodup := by
    rw [Array.toList_zip]
    exact nd.sublist (zip_fst_sublist _ _)
  unfold Replay.reducedOf
  rw [← e, find_nodup _ k hl nd']
  simp only [Array.getElem_toList, Array.getElem_zip, getElem!_pos ch k hk']

section Advance
variable (shapes : Array Replay.TreeShape) (k : ℕ) (claims : Replay.GkrClaims) (pairs : List (ℕ × Array Ext))
  (next : Array Ext) (tau : Ext) {t : ℕ}

theorem advance_current (ht : t < claims.current.size) :
    (Replay.advance shapes k claims pairs next tau).current[t]! = if k < shapes[t]!.depth then
      ((Replay.reducedOf pairs tau t)[0]!, (Replay.reducedOf pairs tau t)[1]!) else claims.current[t]! := by
  simp only [Replay.advance]
  rw [getElem!_pos _ t (by simpa using ht), Array.getElem_map, Array.getElem_range]

theorem advance_leafPoint (ht : t < claims.leafPoint.size) :
    (Replay.advance shapes k claims pairs next tau).leafPoint[t]! =
      if shapes[t]!.depth = k then next.push tau else claims.leafPoint[t]! := by
  simp only [Replay.advance]
  rw [getElem!_pos _ t (by simpa using ht), Array.getElem_map, Array.getElem_range]

theorem advance_leafValues (ht : t < claims.leafValues.size) :
    (Replay.advance shapes k claims pairs next tau).leafValues[t]! =
      if shapes[t]!.depth = k then Replay.reducedOf pairs tau t else claims.leafValues[t]! := by
  simp only [Replay.advance]
  rw [getElem!_pos _ t (by simpa using ht), Array.getElem_map, Array.getElem_range]

end Advance

theorem mem_take_succ {l : List ℕ} {k t : ℕ} (hk : k < l.length) :
    t ∈ l.take (k + 1) ↔ t ∈ l.take k ∨ t = l[k] := by
  rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, List.mem_append, List.mem_singleton]

theorem not_mem_take {l : List ℕ} (nd : l.Nodup) {k : ℕ} (hk : k < l.length) : l[k] ∉ l.take k := by
  intro hm
  obtain ⟨i, hi, e⟩ := List.mem_iff_getElem.mp hm
  rw [List.getElem_take] at e
  have := (List.Nodup.getElem_inj_iff nd).mp e
  simp at hi
  omega

/-- `mapM` in a transcript that leaves the duplex alone. -/
theorem Accepts.tmapM {F : Type} [CommRing F] {c : Ctx F} {P : Prop} {δ γ : Type} {f : δ → Transcript γ}
    (R : δ → γ → ℕ → Prop) (mono : ∀ x y m m', R x y m → m ≤ m' → R x y m') :
    ∀ (l : List δ) (e : Duplex) (m : ℕ), (∀ x ∈ l, ∀ e' m', m ≤ m' →
      Accepts c m' P ((f x).run e') fun r m'' => R x r.1 m'' ∧ r.2 = e') →
      Accepts c m P ((l.mapM f).run e) fun r m' => r.2 = e ∧ r.1.length = l.length ∧
        ∀ i (h : i < l.length) (h' : i < r.1.length), R l[i] r.1[i] m'
  | [], e, m, _ => by
      rw [List.mapM_nil, StateT.run_pure]
      exact Accepts.pure fun _ => ⟨rfl, rfl, fun _ h => absurd h (by simp)⟩
  | x :: l, e, m, each => by
      rw [List.mapM_cons, StateT.run_bind]
      refine ((each x (by simp) e m le_rfl)).bind fun r m₁ l₁ => ?_
      rw [StateT.run_bind]
      refine ((Accepts.tmapM R mono l r.2 m₁ fun y hy e' m' le => each y (by simp [hy]) e' m' (l₁.trans le)).pre
        fun h => h.1).bind fun rs m₂ l₂ => ?_
      rw [StateT.run_pure]
      refine Accepts.pure fun h => ⟨h.2.1.trans h.1.2.2, by simp [h.2.2.1], fun i hi hi' => ?_⟩
      cases i with
      | zero => exact mono _ _ _ _ h.1.2.1 l₂
      | succ i => exact h.2.2.2 i (by simpa using hi) (by simpa using hi')

/-! ## The layer loop -/

open CompressionLean.Replay (TreeShape GkrClaims GkrLayer)

/-- One layer of the replay's batched GKR (`gkrLayers`): `μ`, `ν`, the
sum-check, the check, the children and `τ`. -/
def gkrLayerStep (shapes : Array TreeShape) (k : ℕ) (layer : GkrLayer) (claims : GkrClaims) :
    Replay.ReplayM GkrClaims := do
  let pairs := ((Replay.activeTrees shapes k).zip layer.children).toList
  let mu ← Replay.sampleExt
  let nu ← Replay.sampleExt
  let result ← Replay.sumcheckReplay layer.rounds (Replay.batchClaim claims.current mu nu pairs)
  Replay.ensure (result.2 == Replay.eqEval claims.point result.1 * Replay.batchExpected mu nu pairs) "GKR layer"
  Replay.observeExts layer.children.flatten
  let tau ← Replay.sampleExt
  pure (Replay.advance shapes k claims pairs result.1 tau)

theorem gkrLayers_cons (shapes : Array TreeShape) (k : ℕ) (layer : GkrLayer) (rest : List GkrLayer)
    (claims : GkrClaims) : Replay.gkrLayers shapes k (layer :: rest) claims =
      gkrLayerStep shapes k layer claims >>= Replay.gkrLayers shapes (k + 1) rest := by
  simp [Replay.gkrLayers, gkrLayerStep, bind_assoc]

instance : Nonempty GkrClaims := ⟨⟨#[], #[], #[], #[], #[]⟩⟩

namespace Mirrors

variable {c : Ctx F} {σ : Type} {shapes : Array TreeShape}

/-- The layer loop's shape: it accepts with no claim when its steps do. -/
theorem layers_false {body : GkrLayerView × ℕ → σ → Transcript (ForInStep σ)}
    (step : ∀ (a : GkrLayerView × ℕ) st m, ∀ d, Accepts c m False ((body a st).run d) fun _ _ => True) :
    ∀ (l : List (GkrLayerView × ℕ)) (st : σ) (m : ℕ) (d : Duplex),
      Accepts c m False ((forIn l st body).run d) fun _ _ => True
  | [], st, m, d => by rw [List.forIn_nil, StateT.run_pure]; exact Accepts.pure fun _ => trivial
  | a :: l, st, m, d => by
      rw [List.forIn_cons]
      refine (step a st m d).tbind fun r e m' _ => ?_
      cases r with
      | done b => exact Accepts.pure fun _ => trivial
      | yield b => exact (layers_false step l b m' e).pre fun h => h.1

/-- **The layer loop against the replay's recursion.** Layer `j` of the
circuit's loop (`(layer, j)` of `zipIdx`) mirrors the replay's layer
`j + 1`. -/
theorem forIn_layers {body : GkrLayerView × ℕ → σ → Transcript (ForInStep σ)}
    (Rel : ℕ → σ → GkrClaims → ℕ → Prop) (Ok : ℕ → GkrLayerView → GkrLayer → ℕ → Prop)
    (okMono : ∀ j a b m m', Ok j a b m → m ≤ m' → Ok j a b m')
    (shape : ∀ (a : GkrLayerView × ℕ) st m, ∀ d, Accepts c m False ((body a st).run d) fun _ _ => True)
    (step : ∀ layer gl j st gc m P', Mirrors c m (P' ∧ Ok j layer gl m ∧ Rel j st gc m) (body (layer, j) st)
      (gkrLayerStep shapes (j + 1) gl gc) fun res g m' => res = .yield res.value ∧ Rel (j + 1) res.value g m') :
    ∀ (l : List GkrLayerView) (gl : List GkrLayer) (j : ℕ) (st : σ) (gc : GkrClaims) (n : ℕ) (P : Prop),
      (P → l.length = gl.length ∧ (∀ i (h : i < l.length) (h' : i < gl.length), Ok (j + i) l[i] gl[i] n) ∧
        Rel j st gc n) →
      Mirrors c n P (forIn (l.zipIdx j) st body) (Replay.gkrLayers shapes (j + 1) gl gc)
        fun res g m => Rel (j + l.length) res g m
  | [], [], j, st, gc, n, P, h => by
      rw [List.zipIdx_nil, List.forIn_nil]
      exact Mirrors.pure fun hP => (h hP).2.2
  | [], _ :: _, j, st, gc, n, P, h => by
      rw [List.zipIdx_nil, List.forIn_nil]
      exact of_false (c := c) (fun _ => by rw [StateT.run_pure]; exact Accepts.pure fun _ => trivial)
        fun hP => absurd (h hP).1 (by simp)
  | _ :: _, [], j, st, gc, n, P, h =>
      of_false (c := c) (layers_false shape _ st n) fun hP => absurd (h hP).1 (by simp)
  | layer :: l, gl₀ :: gl, j, st, gc, n, P, h => by
      rw [List.zipIdx_cons, List.forIn_cons, gkrLayers_cons]
      refine Mirrors.bind ((step layer gl₀ j st gc n P).pre fun hP =>
        ⟨hP, (h hP).2.1 0 (by simp) (by simp), (h hP).2.2⟩) fun res g m le => ?_
      cases res with
      | done b => exact of_false (fun _ => Accepts.pure fun _ => trivial) fun hP => absurd hP.2.1 (by simp)
      | yield b =>
          rw [show j + (layer :: l).length = j + 1 + l.length by simp; omega]
          exact forIn_layers Rel Ok okMono shape step l gl (j + 1) b g m _ fun hP =>
            ⟨by simpa using (h hP.1).1, fun i hi hi' => okMono _ _ _ _ _ (by
              have := (h hP.1).2.1 (i + 1) (by simpa using hi) (by simpa using hi')
              rwa [show j + (i + 1) = j + 1 + i by omega] at this) le, hP.2.2⟩

end Mirrors

/-- One step of the circuit's batch loop: the claim, the expected value and
the weight `ν^i`. -/
def batchStep (cur : Array (Ext × Ext)) (mu nu : Ext) (acc : Ext × Ext × Ext) (x : ℕ × Array Ext) :
    Ext × Ext × Ext :=
  (acc.1 + acc.2.2 * (cur[x.1]!.1 + mu * cur[x.1]!.2), acc.2.1 + acc.2.2 * Replay.combineVals x.2 mu, acc.2.2 * nu)

theorem batchStep_fold (cur : Array (Ext × Ext)) (mu nu : Ext) :
    ∀ (l : List (ℕ × Array Ext)) (acc : Ext × Ext × Ext),
      toC (l.foldl (batchStep cur mu nu) acc).1 = toC acc.1 + toC acc.2.2 * toC (Replay.batchClaim cur mu nu l) ∧
      toC (l.foldl (batchStep cur mu nu) acc).2.1 = toC acc.2.1 + toC acc.2.2 * toC (Replay.batchExpected mu nu l)
  | [], acc => by simp [Replay.batchClaim, Replay.batchExpected, toC_zero]
  | x :: l, acc => by
      obtain ⟨h1, h2⟩ := batchStep_fold cur mu nu l (batchStep cur mu nu acc x)
      rw [List.foldl_cons, h1, h2]
      simp only [batchStep, Replay.batchClaim, Replay.batchExpected, toC_add, toC_mul]
      constructor <;> ring

/-- One layer view against one replay layer, with the sizes that a step
reads: nonempty round messages, one child list per active tree, each with
a numerator and a denominator pair. -/
def LayerMatch {F : Type} [CommRing F] (c : Ctx F) (shapes : Array TreeShape) (j : ℕ) (layer : GkrLayerView)
    (gl : GkrLayer) (m : ℕ) : Prop :=
  layer.rounds.size = gl.rounds.size ∧ (∀ i (h : i < layer.rounds.size) (h' : i < gl.rounds.size),
    IsExts c m layer.rounds[i] gl.rounds[i] ∧ 0 < gl.rounds[i].size) ∧
  layer.children.size = gl.children.size ∧ gl.children.size = (Replay.activeTrees shapes (j + 1)).size ∧
  ∀ i (h : i < layer.children.size) (h' : i < gl.children.size),
    IsExts c m layer.children[i] gl.children[i] ∧
      gl.children[i].size = 2 * (Replay.partFactors shapes (j + 1) (Replay.activeTrees shapes (j + 1))[i]! + 1)

/-- The circuit's loop state after `j` layers against the replay's claims:
the current fractions, the point, and every tree that ended. -/
def LayerState {F : Type} [CommRing F] (c : Ctx F) (shapes : Array TreeShape) (roots : Array (ExtVar × ExtVar))
    (groots : Array (Ext × Ext)) (j : ℕ) (st : Array (ExtVar × ExtVar) × Array TreeClaimView × Array ExtVar)
    (gc : GkrClaims) (m : ℕ) : Prop :=
  IsPairs c m st.1 gc.current ∧ st.1.size = shapes.size ∧ IsExts c m st.2.2 gc.point ∧ gc.roots = groots ∧
    st.2.1.size = shapes.size ∧ gc.leafPoint.size = shapes.size ∧ gc.leafValues.size = shapes.size ∧
    ∀ t < shapes.size, shapes[t]!.depth ≤ j → st.2.1[t]!.root = roots[t]! ∧
      IsExts c m st.2.1[t]!.point gc.leafPoint[t]! ∧ IsExts c m st.2.1[t]!.values gc.leafValues[t]!

namespace Mirrors
variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- **The batched GKR verifier against the replay's.** -/
theorem gkr (hp : Permutes c.perm) {view : GkrView} {gp : Replay.GkrProof} {shapes : Array TreeShape}
    (hv : P → IsPairs c n view.roots gp.roots ∧ view.roots.size = shapes.size ∧ view.layers.size = gp.layers.size ∧
      ∀ j (h : j < view.layers.size) (h' : j < gp.layers.size), LayerMatch c shapes j view.layers[j] gp.layers[j] n)
    (depth : ∀ t < shapes.size, 1 ≤ shapes[t]!.depth ∧ shapes[t]!.depth ≤ gp.layers.size) :
    Mirrors c n P (gkrVerify view shapes) (Replay.gkrVerify gp shapes) fun claims gc m =>
      gc.roots = gp.roots ∧ claims.size = shapes.size ∧ ∀ t < shapes.size, claims[t]!.root = view.roots[t]! ∧
        IsExts c m claims[t]!.point gc.leafPoint[t]! ∧ IsExts c m claims[t]!.values gc.leafValues[t]! := by
  unfold gkrVerify Replay.gkrVerify
  simp only [← Array.forIn_toList, Array.toList_zipIdx]
  refine Mirrors.bind (R := fun _ _ _ => True) (fun d tr =>
    ((Sim.observePairsFor hp (fun h => h.2) fun h => (hv h.1).1).accepts).post
    fun _ _ _ q => ⟨(), _, Replay.runs_observeAll _ tr, by rw [foldl_observe_words, foldl_observe_pairs]; exact q.1,
      trivial⟩) fun _ _ m₁ l₁ => ?_
  rw [← bind_pure (Replay.gkrLayers shapes 1 _ _)]
  refine Mirrors.bind (forIn_layers (shapes := shapes) (LayerState c shapes view.roots gp.roots)
    (LayerMatch c shapes) ?mono ?shape (fun layer gl j st gc m P' => ?step) view.layers.toList gp.layers.toList 0
      (view.roots, Array.replicate shapes.size default, #[]) _ m₁ _ ?init) fun res g m₂ _ => Mirrors.pure fun h => ?done
  case mono =>
    intro j a b m m' h le
    exact ⟨h.1, fun i hi hi' => ⟨(h.2.1 i hi hi').1.mono le, (h.2.1 i hi hi').2⟩, h.2.2.1, h.2.2.2.1,
      fun i hi hi' => ⟨(h.2.2.2.2 i hi hi').1.mono le, (h.2.2.2.2 i hi hi').2⟩⟩
  case shape =>
    intro a st m d
    apply Accepts.of_grows
    apply TGrows.run
    grows
  case init =>
    intro h
    obtain ⟨roots, rsize, lsize, layers⟩ := hv h.1
    refine ⟨by simp [lsize], fun i hi hi' => ?_, (roots.mono l₁), by simp [rsize], IsExts.empty, rfl, by simp,
      by simp, by simp, fun t ht d0 => absurd d0 (by have := (depth t ht).1; omega)⟩
    simp only [Array.getElem_toList, Nat.zero_add]
    have := layers i (by simpa using hi) (by simpa using hi')
    exact ⟨this.1, fun k hk hk' => ⟨(this.2.1 k hk hk').1.mono l₁, (this.2.1 k hk hk').2⟩, this.2.2.1, this.2.2.2.1,
      fun k hk hk' => ⟨(this.2.2.2.2 k hk hk').1.mono l₁, (this.2.2.2.2 k hk hk').2⟩⟩
  case done =>
    obtain ⟨-, -, -, roots, csize, -, -, ended⟩ := h.2
    refine ⟨roots, csize, fun t ht => ended t ht ?_⟩
    have := (depth t ht).2
    have := (hv h.1.1).2.2.1
    simp only [Array.length_toList, Nat.zero_add]
    omega
  case step =>
    unfold gkrLayerStep
    dsimp only
    refine Mirrors.bind (sampleExt hp) fun mu gmu m₁ l₁ => Mirrors.bind (sampleExt hp) fun nu gnu m₂ l₂ => ?_
    refine Mirrors.left Sim.extZero.accepts fun claim m₃ l₃ => Mirrors.left Sim.extZero.accepts
      fun expected m₄ l₄ => Mirrors.left Sim.extOne.accepts fun weight m₅ l₅ => ?_
    let act := Replay.activeTrees shapes (j + 1)
    let gpairs := (act.zip gl.children).toList
    let fold (k : ℕ) := (gpairs.take k).foldl (batchStep gc.current gmu gnu) (0, 0, 1)
    have l₁₅ : m ≤ m₅ := l₁.trans (l₂.trans (l₃.trans (l₄.trans l₅)))
    refine Mirrors.leftT (Q := fun s m' => IsExt c m' s.1 (Replay.batchClaim gc.current gmu gnu gpairs) ∧
      IsExt c m' s.2.1 (Replay.batchExpected gmu gnu gpairs)) (fun d => ?_) fun s m₆ l₆ => ?_
    · have glen : layer.children.size = gl.children.size →
          ((Array.filter (fun i => decide (j + 1 ≤ shapes[i]!.depth)) (Array.range shapes.size)).zip
            layer.children).toList.length = gpairs.length :=
        fun e => by simp [gpairs, act, Replay.activeTrees, e]
      refine (((Sim.loopT (I := fun k (acc : ExtVar × ExtVar × ExtVar) e m' => e = d ∧ IsExt c m' acc.1 (fold k).1 ∧
        IsExt c m' acc.2.1 (fold k).2.1 ∧ IsExt c m' acc.2.2 (fold k).2.2) (D := fun _ => True)
          fun k hk acc e m' le => ?_).pre fun h => ⟨h, rfl, ?_, ?_, ?_⟩).accepts).post fun h _ _ q => ?_
      · rw [zip_getElem _ _ hk]
        dsimp only
        have hk₁ : k < (Array.filter (fun i => decide (j + 1 ≤ shapes[i]!.depth)) (Array.range shapes.size)).size := by
          simp only [Array.length_toList, Array.size_zip] at hk; omega
        have ht : (Array.filter (fun i => decide (j + 1 ≤ shapes[i]!.depth)) (Array.range shapes.size))[k]! <
            shapes.size := by
          rw [getElem!_pos _ k hk₁]
          have := Array.getElem_mem hk₁
          simp only [Array.mem_filter, Array.mem_range] at this
          exact this.1
        let t := (Array.filter (fun i => decide (j + 1 ≤ shapes[i]!.depth)) (Array.range shapes.size))[k]!
        let gch := gl.children[k]!
        refine Sim.pre (P' := IsExt c m' mu gmu ∧ IsExt c m' nu gnu ∧ IsExt c m' st.1[t]!.1 gc.current[t]!.1 ∧
          IsExt c m' st.1[t]!.2 gc.current[t]!.2 ∧ IsExt c m' acc.1 (fold k).1 ∧ IsExt c m' acc.2.1 (fold k).2.1 ∧
          IsExt c m' acc.2.2 (fold k).2.2 ∧ IsExts c m' layer.children[k]! gch ∧ 2 ≤ gch.size ∧ e = d ∧
          layer.children.size = gl.children.size) ?_ fun h => ?_
        · refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => h.1) fun h => h.2.2.2.1)) fun mq e₁ k₁ c₁ => ?_
          refine Sim.tbind₀ (Sim.lift (Sim.extAdd (fun h => h.1.2.2.1.mono c₁) fun h => h.2.1)) fun pq e₂ k₂ c₂ => ?_
          refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => h.1.1.2.2.2.2.2.2.1.mono (c₁.trans c₂)) fun h => h.2.1))
            fun term e₃ k₃ c₃ => ?_
          refine Sim.tbind₀ (Sim.lift (Sim.extAdd (fun h => h.1.1.1.2.2.2.2.1.mono (c₁.trans (c₂.trans c₃)))
            fun h => h.2.1)) fun claim' e₄ k₄ c₄ => ?_
          have c₁₄ : m' ≤ k₄ := c₁.trans (c₂.trans (c₃.trans c₄))
          refine Sim.tbind₀ (Sim.lift (Sim.gkrCombine (fun h => h.1.1.1.1.2.2.2.2.2.2.2.1.mono c₁₄)
            (fun h => h.1.1.1.1.2.2.2.2.2.2.2.2.1) fun h => h.1.1.1.1.1.mono c₁₄)) fun combined e₅ k₅ c₅ => ?_
          refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => h.1.1.1.1.1.2.2.2.2.2.2.1.mono (c₁₄.trans c₅))
            fun h => h.2.1)) fun term' e₆ k₆ c₆ => ?_
          refine Sim.tbind₀ (Sim.lift (Sim.extAdd (fun h => h.1.1.1.1.1.1.2.2.2.2.2.1.mono (c₁₄.trans (c₅.trans c₆)))
            fun h => h.2.1)) fun expected' e₇ k₇ c₇ => ?_
          refine Sim.tbind₀ (Sim.lift (Sim.extMul
            (fun h => h.1.1.1.1.1.1.1.2.2.2.2.2.2.1.mono (c₁₄.trans (c₅.trans (c₆.trans c₇))))
            fun h => h.1.1.1.1.1.1.1.2.1.mono (c₁₄.trans (c₅.trans (c₆.trans c₇))))) fun weight' e₈ k₈ c₈ => ?_
          rw [StateT.run_pure]
          have fstep (sizes : layer.children.size = gl.children.size) :
              fold (k + 1) = batchStep gc.current gmu gnu (fold k) (t, gch) := by
            have hk' : k < gpairs.length := by
              simp only [Array.length_toList, Array.size_zip] at hk
              simp only [gpairs, act, Replay.activeTrees, Array.length_toList, Array.size_zip]
              omega
            simp only [fold]
            rw [List.take_succ_eq_append_getElem hk', List.foldl_append]
            simp only [gpairs, zip_getElem _ _ hk', List.foldl_cons, List.foldl_nil]
            rfl
          refine Sim.pure fun h => ⟨rfl, ?_, ?_, ?_, ?_⟩ <;>
            obtain ⟨⟨⟨⟨⟨⟨⟨⟨F, x1, q1⟩, x2, q2⟩, x3, q3⟩, x4, q4⟩, x5, q5⟩, x6, q6⟩, x7, q7⟩, x8, q8⟩ := h
          · exact q8.trans (q7.trans (q6.trans (q5.trans (q4.trans (q3.trans (q2.trans (q1.trans
              F.2.2.2.2.2.2.2.2.2.1)))))))
          · rw [fstep F.2.2.2.2.2.2.2.2.2.2]
            exact x4.mono (c₅.trans (c₆.trans (c₇.trans c₈)))
          · rw [fstep F.2.2.2.2.2.2.2.2.2.2]
            exact x7.mono c₈
          · rw [fstep F.2.2.2.2.2.2.2.2.2.2]
            exact x8
        · obtain ⟨⟨⟨⟨⟨⟨⟨-, lm, ls⟩, hmu⟩, hnu⟩, -⟩, -⟩, -⟩, he, ha1, ha2, ha3⟩ := h
          have hcur := ls.1.2 t (by rw [ls.2.1]; exact ht)
          have hkc : k < layer.children.size := by simp only [Array.length_toList, Array.size_zip] at hk; omega
          have hch := lm.2.2.2.2 k hkc (by rw [← lm.2.2.1]; exact hkc)
          refine ⟨hmu.mono (by omega), hnu.mono (by omega), hcur.1.mono (by omega), hcur.2.mono (by omega), ha1, ha2,
            ha3, ?_, ?_, he, lm.2.2.1⟩
          · rw [getElem!_pos _ k hkc]
            simp only [gch, getElem!_pos gl.children k (by rw [← lm.2.2.1]; exact hkc)]
            exact hch.1.mono (by omega)
          · simp only [gch, getElem!_pos gl.children k (by rw [← lm.2.2.1]; exact hkc)]
            rw [hch.2]
            omega
      · exact (h.1.1.2.1.mono (l₄.trans l₅)).of_eq rfl
      · exact (h.1.2.1.mono l₅).of_eq rfl
      · exact h.2.1.of_eq rfl
      · obtain ⟨⟨e, hc, he, -⟩, -⟩ := q
        have L := glen h.1.1.1.1.1.2.1.2.2.1
        simp only [fold] at hc he
        rw [L, List.take_length] at hc he
        obtain ⟨b1, b2⟩ := batchStep_fold gc.current gmu gnu gpairs (0, 0, 1)
        exact ⟨⟨hc.of_eq (toC_inj (by rw [b1]; simp [toC_zero, toC_one])),
          he.of_eq (toC_inj (by rw [b2]; simp [toC_zero, toC_one]))⟩, e⟩
    · have A : _ → P' ∧ LayerMatch c shapes j layer gl m ∧ LayerState c shapes view.roots gp.roots j st gc m :=
        fun (h : (((((((P' ∧ LayerMatch c shapes j layer gl m ∧ LayerState c shapes view.roots gp.roots j st gc m) ∧
          IsExt c m₁ mu gmu) ∧ IsExt c m₂ nu gnu) ∧ IsExt c m₃ claim 0 ∧ True) ∧ IsExt c m₄ expected 0 ∧ True) ∧
          IsExt c m₅ weight 1 ∧ True) ∧ IsExt c m₆ s.1 (Replay.batchClaim gc.current gmu gnu gpairs) ∧
          IsExt c m₆ s.2.1 (Replay.batchExpected gmu gnu gpairs))) => h.1.1.1.1.1.1
      have l₁₆ : m ≤ m₆ := l₁₅.trans l₆
      refine Mirrors.bind (sumcheck hp (fun h => ⟨(A h).2.1.1, fun i hi hi' =>
        ⟨((A h).2.1.2.1 i hi hi').1.mono l₁₆, ((A h).2.1.2.1 i hi hi').2⟩⟩) fun h => h.2.1) fun x gx m₇ l₇ => ?_
      refine Mirrors.left (Sim.eqEval (fun h => (A h.1).2.2.2.2.1.mono (l₁₆.trans l₇)) fun h => h.2.1).accepts
        fun eq m₈ l₈ => ?_
      refine Mirrors.left (Sim.extMul (fun h => h.2.1) fun h => h.1.1.2.2.mono (l₇.trans l₈)).accepts
        fun target m₉ l₉ => ?_
      refine Mirrors.left (Sim.assertExtEqual (fun h => h.1.1.2.2.mono (l₈.trans l₉)) fun h => h.2.1).accepts
        fun _ m₁₀ l₁₀ => ?_
      refine Mirrors.right (S0 := fun _ => True) (fun h tr => ⟨(), tr, Replay.runs_ensure _ tr
        (ext_beq_of_eq h.2.2), rfl, trivial⟩) fun _ => ?_
      have l₆₁₀ : m₆ ≤ m₁₀ := l₇.trans (l₈.trans (l₉.trans l₁₀))
      refine Mirrors.bind (R := fun _ _ _ => True) (fun d tr => ((Sim.observeNestedFor hp (fun h => h.2)
        fun h => ?_).accepts).post fun _ _ _ q => ⟨(), _, Replay.runs_observeAll _ tr,
          by rw [foldl_observe_words, foldl_observe_flatten]; exact q.1, trivial⟩) fun _ _ m₁₁ l₁₁ => ?_
      · have lm := (A h.1.1.1.1.1.1).2.1
        exact ⟨lm.2.2.1, fun i hi hi' => ((lm.2.2.2.2 i hi hi').1.mono (l₁₆.trans l₆₁₀))⟩
      refine Mirrors.bind (sampleExt hp) fun tau gtau m₁₂ l₁₂ => ?_
      let adv := Replay.advance shapes (j + 1) gc gpairs gx.1 gtau
      let red (t : ℕ) := Replay.reducedOf gpairs gtau t
      let cur (k t : ℕ) : Ext × Ext :=
        if t ∈ act.toList.take k ∧ j + 1 < shapes[t]!.depth then ((red t)[0]!, (red t)[1]!) else gc.current[t]!
      refine Mirrors.leftT (Q := fun (s2 : Array (ExtVar × ExtVar) × Array TreeClaimView) m' =>
        s2.1.size = shapes.size ∧ s2.2.size = shapes.size ∧ ∀ t < shapes.size,
          (IsExt c m' s2.1[t]!.1 (cur act.size t).1 ∧ IsExt c m' s2.1[t]!.2 (cur act.size t).2) ∧
          (shapes[t]!.depth ≤ j → s2.2[t]! = st.2.1[t]!) ∧
          (shapes[t]!.depth = j + 1 → s2.2[t]!.root = view.roots[t]! ∧ IsExts c m' s2.2[t]!.point (gx.1.push gtau) ∧
            IsExts c m' s2.2[t]!.values (red t))) (fun d => ?_) fun s2 m₁₃ l₁₃ => Mirrors.pure fun h => ?_
      · refine ((Accepts.loopT (I := fun k (acc : Array (ExtVar × ExtVar) × Array TreeClaimView) e m' =>
          e = d ∧ acc.1.size = shapes.size ∧ acc.2.size = shapes.size ∧ ∀ t < shapes.size,
            (IsExt c m' acc.1[t]!.1 (cur k t).1 ∧ IsExt c m' acc.1[t]!.2 (cur k t).2) ∧
            (shapes[t]!.depth ≤ j → acc.2[t]! = st.2.1[t]!) ∧
            (shapes[t]!.depth = j + 1 → t ∈ act.toList.take k → acc.2[t]!.root = view.roots[t]! ∧
              IsExts c m' acc.2[t]!.point (gx.1.push gtau) ∧ IsExts c m' acc.2[t]!.values (red t)))
          fun k hk acc e m' le => ?_).pre fun h => ⟨h, rfl, ?_⟩).post fun h _ _ q => ?_
        · rw [zip_getElem _ _ hk]
          dsimp only
          have hk' : k < act.size ∧ k < layer.children.size := by
            simp only [Array.length_toList, Array.size_zip] at hk
            exact Nat.lt_min.mp hk
          let gch := gl.children[k]!
          refine Accepts.pre (P' := (IsExts c m' layer.children[k]! gch ∧ layer.children[k]!.size = gch.size ∧
            gch.size = 2 * (Replay.partFactors shapes (j + 1) act[k]! + 1) ∧ IsExt c m' tau gtau ∧
            IsExts c m' x.1 gx.1 ∧ k < gl.children.size) ∧ _) ?_ fun h => ⟨?_, h⟩
          swap
          · have lm := (A h.1.1.1.1.1.1.1.1).2.1
            have hkg : k < gl.children.size := by rw [← lm.2.2.1]; exact hk'.2
            have := lm.2.2.2.2 k hk'.2 hkg
            rw [getElem!_pos _ k hk'.2]
            simp only [gch, getElem!_pos gl.children k hkg]
            exact ⟨this.1.mono (by omega), this.1.1, this.2, h.1.2.mono le,
              h.1.1.1.1.1.1.1.2.1.mono (by omega), hkg⟩
          rw [StateT.run_bind, Array.mapM_eq_mapM_toList, StateT.run_map]
          refine (((Accepts.tmapM (R := fun jj r m'' => IsExt c m'' r (Replay.reduce (gch[2 * jj]!, gch[2 * jj + 1]!)
            gtau)) (fun _ _ _ _ h l => h.mono l) _ e m' fun jj hjj e' m'' le' => ?_).map
              (Q' := fun (r : Array ExtVar × Duplex) m'' => r.2 = e ∧ r.1.size = layer.children[k]!.size / 2 ∧
                ∀ i (h : i < r.1.size), IsExt c m'' r.1[i]! (Replay.reduce (gch[2 * i]!, gch[2 * i + 1]!) gtau))
              fun r m'' h => ?_).bind fun r m₂ l₂ => ?_)
          · have hj : jj < layer.children[k]!.size / 2 := by simpa using hjj
            have get (i : ℕ) (hi : i < 2 * jj + 2) (h : (IsExts c m' layer.children[k]! gch ∧
                layer.children[k]!.size = gch.size ∧ gch.size = 2 * (Replay.partFactors shapes (j + 1) act[k]! + 1) ∧
                IsExt c m' tau gtau ∧ IsExts c m' x.1 gx.1 ∧ k < gl.children.size)) :
                IsExt c m'' (layer.children[k]!)[i]! gch[i]! :=
              (h.1.get (by rw [← h.2.1]; omega)).mono le'
            refine (Sim.accepts (C := True) (Q := fun r m''' => IsExt c m''' r.1
              (Replay.reduce (gch[2 * jj]!, gch[2 * jj + 1]!) gtau) ∧ r.2 = e') ?_).post fun _ _ _ q => q.1
            refine Sim.tbind₀ (Sim.lift (Sim.extSub (fun h => get _ (by omega) h.1)
              (fun h => get _ (by omega) h.1))) fun d₁ e₁ k₁ c₁ => ?_
            refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => (h.1.1.2.2.2.1.mono (by omega))) fun h => h.2.1))
                fun st₁ e₂ k₂ c₂ => ?_
            refine (Sim.lift (Sim.extAdd (fun h => (get _ (by omega) h.1.1.1).mono (c₁.trans c₂))
                  fun h => h.2.1)).post (fun h _ _ q => ⟨q.1, q.2.trans (h.2.2.trans h.1.2.2)⟩) fun _ => Iff.rfl
          · obtain ⟨e', len, each⟩ := h
            refine ⟨e', by simp [len], fun i hi => ?_⟩
            have hi' : i < r.1.length := by simpa using hi
            rw [getElem!_pos _ i hi, List.getElem_toArray]
            have := each i (by simpa [len] using hi) hi'
            simpa using this
          · have nd := activeTrees_nodup shapes (j + 1)
            have hkl : k < act.toList.length := by simpa using hk'.1
            have t₀eq : act[k]! = act.toList[k] := by rw [Array.getElem_toList, getElem!_pos _ k hk'.1]
            have act₀ := activeTrees_lt shapes (j + 1) (getElem!_mem hk'.1)
            have notin : act[k]! ∉ act.toList.take k := by rw [t₀eq]; exact not_mem_take nd hkl
            have mem (t : ℕ) : t ∈ act.toList.take (k + 1) ↔ t ∈ act.toList.take k ∨ t = act[k]! := by
              rw [mem_take_succ hkl, t₀eq]
            simp only [show (Array.filter (fun i => decide (j + 1 ≤ shapes[i]!.depth)) (Array.range shapes.size)) =
              act from rfl]
            split
            · rename_i dd₀
              have dd : shapes[act[k]!]!.depth = j + 1 := dd₀
              rw [StateT.run_pure]
              refine Accepts.pure fun h => ?_
              have F2 := h.1.1
              have hJ := h.1.2.2
              obtain ⟨re, rsize, rall⟩ := h.2
              have redv : red act[k]! = (Array.range (gch.size / 2)).map fun jj =>
                  Replay.reduce (gch[2 * jj]!, gch[2 * jj + 1]!) gtau :=
                reducedOf_get nd gtau hk'.1 F2.2.2.2.2.2
              have hred : IsExts c m₂ r.1 (red act[k]!) := by
                refine ⟨by rw [redv, rsize, F2.2.1]; simp, fun i hi => ?_⟩
                have v : (red act[k]!)[i]! = Replay.reduce (gch[2 * i]!, gch[2 * i + 1]!) gtau := by
                  rw [redv, getElem!_pos _ i (by simp; rw [← F2.2.1, ← rsize]; exact hi), Array.getElem_map,
                    Array.getElem_range]
                rw [v]
                exact rall i hi
              dsimp only [ForInStep.value]
              refine ⟨rfl, re.trans hJ.1, hJ.2.1, by simp [hJ.2.2.1], fun t ht => ?_⟩
              have each := hJ.2.2.2 t ht
              have same : cur (k + 1) t = cur k t := by
                simp only [cur, mem]
                by_cases tt : t = act[k]!
                · subst tt
                  simp [notin, dd]
                · simp [tt]
              refine ⟨by rw [same]; exact ⟨each.1.1.mono l₂, each.1.2.mono l₂⟩, fun dt => ?_, fun dt hm => ?_⟩
              · rw [getElem!_set! _ _ _ _ (by rw [hJ.2.2.1]; exact ht),
                  if_neg (fun (e : act[k]! = t) => by subst e; omega)]
                exact each.2.1 dt
              · rw [getElem!_set! _ _ _ _ (by rw [hJ.2.2.1]; exact ht)]
                split
                · rename_i tt
                  subst tt
                  exact ⟨rfl, (F2.2.2.2.2.1.mono (by omega)).push (F2.2.2.2.1.mono l₂), hred⟩
                · rename_i tt
                  have old := each.2.2 dt (((mem t).mp hm).resolve_right (Ne.symm tt))
                  exact ⟨old.1, old.2.1.mono l₂, old.2.2.mono l₂⟩
            · rename_i dd₀
              have dd : ¬ shapes[act[k]!]!.depth = j + 1 := dd₀
              rw [StateT.run_pure]
              refine Accepts.pure fun h => ?_
              have F2 := h.1.1
              have hJ := h.1.2.2
              obtain ⟨re, rsize, rall⟩ := h.2
              have redv : red act[k]! = (Array.range (gch.size / 2)).map fun jj =>
                  Replay.reduce (gch[2 * jj]!, gch[2 * jj + 1]!) gtau :=
                reducedOf_get nd gtau hk'.1 F2.2.2.2.2.2
              have hred : IsExts c m₂ r.1 (red act[k]!) := by
                refine ⟨by rw [redv, rsize, F2.2.1]; simp, fun i hi => ?_⟩
                have v : (red act[k]!)[i]! = Replay.reduce (gch[2 * i]!, gch[2 * i + 1]!) gtau := by
                  rw [redv, getElem!_pos _ i (by simp; rw [← F2.2.1, ← rsize]; exact hi), Array.getElem_map,
                    Array.getElem_range]
                rw [v]
                exact rall i hi
              dsimp only [ForInStep.value]
              refine ⟨rfl, re.trans hJ.1, by simp [hJ.2.1], hJ.2.2.1, fun t ht => ?_⟩
              have each := hJ.2.2.2 t ht
              refine ⟨?_, fun dt => each.2.1 dt, fun dt hm => ?_⟩
              · rw [getElem!_set! _ _ _ _ (by rw [hJ.2.1]; exact ht)]
                split
                · rename_i tt
                  subst tt
                  have deep : j + 1 < shapes[act[k]!]!.depth := by
                    have a2 : j + 1 ≤ shapes[act[k]!]!.depth := act₀.2
                    omega
                  have two : 2 ≤ (red act[k]!).size := by
                    rw [redv, Array.size_map, Array.size_range, F2.2.2.1]
                    simp [Replay.partFactors, dd]
                  simp only [cur, mem, or_true, deep, and_self, ↓reduceIte]
                  exact ⟨hred.get (by omega), hred.get (by omega)⟩
                · rename_i tt
                  have same : cur (k + 1) t = cur k t := by simp [cur, mem, Ne.symm tt]
                  rw [same]
                  exact ⟨each.1.1.mono l₂, each.1.2.mono l₂⟩
              · by_cases tt : t = act[k]!
                · subst tt; exact absurd dt dd
                · have old := each.2.2 dt (((mem t).mp hm).resolve_right tt)
                  exact ⟨old.1, old.2.1.mono l₂, old.2.2.mono l₂⟩
        · have ls := (A h.1.1.1.1.1.1.1).2.2
          refine ⟨ls.2.1, ls.2.2.2.2.1, fun t ht => ⟨?_, fun _ => rfl, fun _ hm => by simp at hm⟩⟩
          have := ls.1.2 t (by rw [ls.2.1]; exact ht)
          simp only [cur, List.take_zero, List.not_mem_nil, false_and, ↓reduceIte]
          exact ⟨this.1.mono (by omega), this.2.mono (by omega)⟩
        · have lm := (A h.1.1.1.1.1.1.1).2.1
          obtain ⟨e, s1, s2, each⟩ := q
          have len : ((Array.filter (fun i => decide (j + 1 ≤ shapes[i]!.depth)) (Array.range shapes.size)).zip
              layer.children).toList.length = act.size := by
            simp only [Array.length_toList, Array.size_zip, lm.2.2.1, lm.2.2.2.1]
            simp [act, Replay.activeTrees]
          rw [len] at each
          refine ⟨⟨s1, s2, fun t ht => ⟨(each t ht).1, (each t ht).2.1, fun dt => (each t ht).2.2 dt ?_⟩⟩, e⟩
          rw [← Array.length_toList, List.take_length]
          simp only [act, Replay.activeTrees, Array.mem_toList_iff, Array.mem_filter, Array.mem_range,
            decide_eq_true_eq]
          exact ⟨ht, by omega⟩
      · have htau := h.1.2
        have hx := h.1.1.1.1.1.1.1.2.1
        have ls := (A h.1.1.1.1.1.1.1.1).2.2
        obtain ⟨s1size, s2size, each⟩ := h.2
        obtain ⟨⟨csize, -⟩, stsize, -, roots, clsize, lpsize, lvsize, ended⟩ := ls
        have gcs : gc.current.size = shapes.size := by rw [← csize, stsize]
        refine ⟨rfl, ⟨by simp [Replay.advance, s1size, gcs], fun t ht => ?_⟩, s1size, ?_, roots,
          s2size, by simp [Replay.advance, lpsize], by simp [Replay.advance, lvsize], fun t ht dt => ?_⟩
        · have ht' : t < shapes.size := by rw [← s1size]; exact ht
          have e : cur act.size t = adv.current[t]! := by
            rw [advance_current _ _ _ _ _ _ (by rw [gcs]; exact ht')]
            have tk : act.toList.take act.size = act.toList := by rw [← Array.length_toList, List.take_length]
            simp only [cur, tk]
            by_cases deep : j + 1 < shapes[t]!.depth
            · have mem : t ∈ act.toList := by
                simp only [act, Replay.activeTrees, Array.mem_toList_iff, Array.mem_filter, Array.mem_range,
                  decide_eq_true_eq]
                exact ⟨ht', by omega⟩
              simp only [mem, deep, and_self, ↓reduceIte]
              rfl
            · simp [deep]
          rw [← e]
          exact (each t ht').1
        · exact (hx.mono (by omega)).push (htau.mono l₁₃)
        · dsimp only [ForInStep.value]
          rcases Nat.lt_or_ge (shapes[t]!.depth) (j + 1) with lt | ge
          · have old := ended t ht (by omega)
            rw [(each t ht).2.1 (by omega), advance_leafPoint _ _ _ _ _ _ (by rw [lpsize]; exact ht),
              advance_leafValues _ _ _ _ _ _ (by rw [lvsize]; exact ht), if_neg (by omega), if_neg (by omega)]
            exact ⟨old.1, old.2.1.mono (by omega), old.2.2.mono (by omega)⟩
          · have eq : shapes[t]!.depth = j + 1 := by omega
            rw [advance_leafPoint _ _ _ _ _ _ (by rw [lpsize]; exact ht),
              advance_leafValues _ _ _ _ _ _ (by rw [lvsize]; exact ht), if_pos eq, if_pos eq]
            exact (each t ht).2.2 eq

end Mirrors
end CompressionLean.Circuit
