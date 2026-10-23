import CompressionLean.Circuit.SimSumcheck
import CompressionLean.Circuit.Gkr
import CompressionLean.Bridge.Gkr

/-!
Owns the simulation of the batched GKR verifier of `Gkr.lean` (`gkr.rs`)
against the replay's (`Replay/Gkr.lean`). `gkrVerify_ensures` gives the
replay's run, and each tree's root and leaf claim (`IsGkrClaims`). The
circuit sums `ν^i·(p + μq)` from the first active tree with a running
weight; the replay uses Horner's rule from the last tree. The sums agree.
The proof has the layers that `readGkr` checks.
-/

namespace CompressionLean.Replay

/-- Layers `l₁ ++ l₂` replay `l₁`, then `l₂` from its claims (`gkr.rs:694`). -/
private theorem gkrLayers_append (shapes : Array TreeShape) (l₁ l₂ : List GkrLayer) (k : ℕ) (claims : GkrClaims) :
    gkrLayers shapes k (l₁ ++ l₂) claims =
      gkrLayers shapes k l₁ claims >>= fun c => gkrLayers shapes (k + l₁.length) l₂ c := by
  induction l₁ generalizing k claims with
  | nil => simp [gkrLayers]
  | cons layer rest ih =>
      simp only [List.cons_append, gkrLayers, ih, bind_assoc, List.length_cons]
      rw [show k + (rest.length + 1) = k + 1 + rest.length by omega]

/-- The first `j + 1` layers run as the first `j`, then layer `j` (`gkr.rs:694`). -/
private theorem runs_gkrLayers_succ {shapes : Array TreeShape} {layers : Array GkrLayer} {j : ℕ}
    (hj : j < layers.size) {init c c' : GkrClaims} {t₀ t t' : Transcript}
    (pre : Runs (gkrLayers shapes 1 (layers.toList.take j) init) t₀ c t)
    (step : Runs (gkrLayers shapes (j + 1) [layers[j]!] c) t c' t') :
    Runs (gkrLayers shapes 1 (layers.toList.take (j + 1)) init) t₀ c' t' := by
  have e : layers.toList.take (j + 1) = layers.toList.take j ++ [layers[j]!] := by
    rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hj), Option.toList_some, Array.getElem_toList,
      ← getElem!_pos layers j hj]
  rw [e, gkrLayers_append, List.length_take, Array.length_toList, Nat.min_eq_left hj.le, Nat.add_comm 1 j]
  exact runs_bind pre step

/-- A value passes the replay's equality test with itself (`gkr.rs:718`). -/
private theorem ext_beq_self (a : Ext) : (a == a) = true := by
  show (a.c0.word == a.c0.word && a.c1.word == a.c1.word && a.c2.word == a.c2.word) = true
  simp

/-- One layer of `verify` (`gkr.rs:694-746`): draw `μ` and `ν`, replay the
sum-check, check its last claim, bind the children and draw `τ`. -/
private theorem runs_gkrLayer {shapes : Array TreeShape} {k : ℕ} {layer : GkrLayer} {claims : GkrClaims}
    {t tmu tnu ts tobs t' : Transcript} {mu nu tau final : Ext} {point : Array Ext}
    (hmu : Runs sampleExt t mu tmu) (hnu : Runs sampleExt tmu nu tnu)
    (sum : Runs (sumcheckReplay layer.rounds
      (batchClaim claims.current mu nu ((activeTrees shapes k).zip layer.children).toList)) tnu (point, final) ts)
    (check : final = eqEval claims.point point * batchExpected mu nu ((activeTrees shapes k).zip layer.children).toList)
    (hobs : Runs (observeExts layer.children.flatten) ts () tobs) (htau : Runs sampleExt tobs tau t') :
    Runs (gkrLayers shapes k [layer] claims) t
      (advance shapes k claims ((activeTrees shapes k).zip layer.children).toList point tau) t' := by
  rw [gkrLayers]
  exact runs_bind hmu (runs_bind hnu (runs_bind sum (runs_bind (runs_ensure (by rw [← check]; exact ext_beq_self _) _ _)
    (runs_bind hobs (runs_bind htau (runs_pure _ _))))))

/-- Every tree ends by the last layer (`gkr.rs:749`). -/
private theorem depth_le_layerCount (shapes : Array TreeShape) {t : ℕ} (inside : t < shapes.size) :
    shapes[t]!.depth ≤ layerCount shapes := by
  unfold layerCount
  suffices h : ∀ l : List ℕ, t ∈ l → shapes[t]!.depth ≤ l.foldr (fun t d => max d shapes[t]!.depth) 0 from
    h _ (List.mem_range.mpr inside)
  intro l member
  induction l with
  | nil => cases member
  | cons u rest ih =>
      rw [List.foldr_cons]
      rcases List.mem_cons.mp member with rfl | inside
      · exact le_max_right _ _
      · exact (ih inside).trans (le_max_left _ _)

/-- A round message has at least three values (`GkrView::read`, `gkr.rs:608-656`). -/
private theorem three_le_layerDegree (factors : Array ℕ) : 3 ≤ layerDegree factors := by
  unfold layerDegree
  rw [← Array.foldr_toList]
  induction factors.toList with
  | nil => exact le_rfl
  | cons f rest ih => exact ih.trans (le_max_left _ _)

/-- Every tree has depth at least one (`GkrView::read`, `gkr.rs:608-656`). -/
private theorem readGkr_depth {proof : GkrProof} {shapes : Array TreeShape} (read : readGkr proof shapes = .ok ())
    {t : ℕ} (inside : t < shapes.size) : 1 ≤ shapes[t]!.depth := by
  unfold readGkr at read
  split_ifs at read with bad
  · obtain ⟨_, thrown, _⟩ := Except.bind_ok read
    cases thrown
  · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or] at bad
    have := bad.1.1
    simp only [Array.any_eq_true, not_exists, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or] at this
    rw [getElem!_pos shapes t inside]
    have := (this t inside).1
    omega

/-- What `GkrView::read` (`gkr.rs:608-656`) fixes of layer `k`: `k - 1`
rounds of at least three values, and `2·(factors + 1)` children for each
active tree. -/
private theorem layerShapeOk_facts {shapes : Array TreeShape} {k : ℕ} {layer : GkrLayer}
    (ok : layerShapeOk shapes k layer = true) :
    layer.rounds.size = k - 1 ∧ (∀ m ∈ layer.rounds, 3 ≤ m.size) ∧
      layer.children.size = (activeTrees shapes k).size ∧
      ∀ s < layer.children.size,
        layer.children[s]!.size = 2 * (partFactors shapes k (activeTrees shapes k)[s]! + 1) := by
  simp only [layerShapeOk, Bool.and_eq_true, beq_iff_eq] at ok
  obtain ⟨⟨⟨rounds, degree⟩, children⟩, sizes⟩ := ok
  rw [← Array.all_toList, List.all_eq_true] at degree
  rw [Array.all_eq_true] at sizes
  refine ⟨rounds, fun m member => ?_, children, fun s hs => ?_⟩
  · have := degree m (Array.mem_toList_iff.mpr member)
    simp only [beq_iff_eq] at this
    rw [this]
    exact three_le_layerDegree _
  · have := sizes s (by simp [children]; omega)
    simp only [Array.getElem_zip, beq_iff_eq] at this
    rw [getElem!_pos _ s hs, getElem!_pos _ s (by omega)]
    exact this

end CompressionLean.Replay

namespace CompressionLean.Circuit

open CompressionLean.Replay (Ext Runs TreeShape)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-- The circuit's tree claims are the replay's GKR claims: each tree's root
pair, leaf point and leaf values (`TreeClaim`, `gkr.rs:70-78`). -/
def IsGkrClaims (c : Ctx F) (m : ℕ) (claims : Array TreeClaimView) (g : Replay.GkrClaims) : Prop :=
  claims.size = g.roots.size ∧ ∀ t < claims.size,
    IsExt c m claims[t]!.root.1 (g.roots[t]!).1 ∧ IsExt c m claims[t]!.root.2 (g.roots[t]!).2 ∧
      IsExts c m claims[t]!.point g.leafPoint[t]! ∧ IsExts c m claims[t]!.values g.leafValues[t]!

/-- The claims of `verify` (`gkr.rs:747-750`) at a larger count. -/
theorem IsGkrClaims.mono {c : Ctx F} {m m' : ℕ} {claims : Array TreeClaimView} {g : Replay.GkrClaims}
    (h : IsGkrClaims c m claims g) (le : m ≤ m') : IsGkrClaims c m' claims g :=
  ⟨h.1, fun t ht => ⟨(h.2 t ht).1.mono le, (h.2 t ht).2.1.mono le, (h.2 t ht).2.2.1.mono le,
    (h.2 t ht).2.2.2.mono le⟩⟩

/-- One step of the batched sum (`gkr.rs:704-714`): the running weight
against Horner's rule. -/
private theorem horner_step (A W a nu R : Ext) : A + W * (a + nu * R) = A + W * a + W * nu * R :=
  coords_injective (F := ZMod goldilocksModulus) (by
    simp only [coords_add, coords_mul]
    funext k
    fin_cases k <;> simp [extProduct, Matrix.vecHead, Matrix.vecTail] <;> ring)

/-- The batched sum after the last tree (`gkr.rs:704-714`). -/
private theorem add_mul_zero (A W : Ext) : A + W * 0 = A :=
  coords_injective (F := ZMod goldilocksModulus) (by
    simp only [coords_add, coords_mul, coords_zero]
    funext k
    fin_cases k <;> simp [extProduct])

/-- The batched sum before the first tree (`gkr.rs:701-703`). -/
private theorem zero_add_one_mul (R : Ext) : R = 0 + 1 * R :=
  coords_injective (F := ZMod goldilocksModulus) (by
    simp only [coords_add, coords_mul, coords_zero, coords_one]
    funext k
    fin_cases k <;> simp [extProduct])

/-- Binding each child in order (`gkr.rs:719-723`) binds the flattened values. -/
private theorem flatMap_flatten {α β : Type} (f : α → List β) :
    ∀ L : List (List α), L.flatten.flatMap f = L.flatMap fun l => l.flatMap f
  | [] => rfl
  | l :: L => by rw [List.flatten_cons, List.flatMap_append, flatMap_flatten f L, List.flatMap_cons]

/-- The reduced values of one tree (`gkr.rs:727-733`), entry by entry. -/
private theorem isExts_of_forall₂ {c : Ctx F} {n N : ℕ} {g : ℕ → Ext} {as : List ExtVar}
    (h : List.Forall₂ (fun j a => IsExt c n a (g j)) (List.range N) as) :
    IsExts c n as.toArray ((Array.range N).map g) := by
  obtain ⟨len, get⟩ := List.forall₂_iff_get.mp h
  simp only [List.length_range] at len
  refine ⟨by simp [← len], fun i hi => ?_⟩
  have hi' : i < N := by simpa [← len] using hi
  have := get i (by simpa using hi') (by simpa [← len] using hi')
  rw [getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi')]
  simpa using this

/-- A root or a current pair of `verify` (`gkr.rs:687-691`) is the replay's pair. -/
private def IsPair (c : Ctx F) (m : ℕ) (a : ExtVar × ExtVar) (g : Ext × Ext) : Prop :=
  IsExt c m a.1 g.1 ∧ IsExt c m a.2 g.2

/-- A pair of `verify` (`gkr.rs:691`) at a larger count. -/
private theorem IsPair.mono {c : Ctx F} {m m' : ℕ} {a : ExtVar × ExtVar} {g : Ext × Ext} (h : IsPair c m a g)
    (le : m ≤ m') : IsPair c m' a g :=
  ⟨h.1.mono le, h.2.mono le⟩

/-- Tree `t`'s claim (`TreeClaim`, `gkr.rs:70-78`) is the replay's root,
leaf point and leaf values. -/
private def IsTreeClaim (c : Ctx F) (m : ℕ) (v : TreeClaimView) (g : Replay.GkrClaims) (t : ℕ) : Prop :=
  IsPair c m v.root g.roots[t]! ∧ IsExts c m v.point g.leafPoint[t]! ∧ IsExts c m v.values g.leafValues[t]!

/-- A tree's claim (`gkr.rs:736-740`) at a larger count. -/
private theorem IsTreeClaim.mono {c : Ctx F} {m m' : ℕ} {v : TreeClaimView} {g : Replay.GkrClaims} {t : ℕ}
    (h : IsTreeClaim c m v g t) (le : m ≤ m') : IsTreeClaim c m' v g t :=
  ⟨h.1.mono le, h.2.1.mono le, h.2.2.mono le⟩

/-- The state of `verify` after `j` layers (`gkr.rs:691-693`): every tree's
current pair, and the claim of every tree of depth at most `j`. -/
private def IsLayerState (c : Ctx F) (m : ℕ) (shapes : Array TreeShape) (j : ℕ) (cur : Array (ExtVar × ExtVar))
    (cl : Array TreeClaimView) (g : Replay.GkrClaims) : Prop :=
  cur.size = shapes.size ∧ cl.size = shapes.size ∧ g.current.size = shapes.size ∧
    g.leafPoint.size = shapes.size ∧ g.leafValues.size = shapes.size ∧
    ∀ t < shapes.size, IsPair c m cur[t]! g.current[t]! ∧ (shapes[t]!.depth ≤ j → IsTreeClaim c m cl[t]! g t)

/-- The state of `verify` (`gkr.rs:691-693`) at a larger count. -/
private theorem IsLayerState.mono {c : Ctx F} {m m' : ℕ} {shapes : Array TreeShape} {j : ℕ}
    {cur : Array (ExtVar × ExtVar)} {cl : Array TreeClaimView} {g : Replay.GkrClaims}
    (h : IsLayerState c m shapes j cur cl g) (le : m ≤ m') : IsLayerState c m' shapes j cur cl g :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, fun t ht =>
    ⟨(h.2.2.2.2.2 t ht).1.mono le, fun hd => ((h.2.2.2.2.2 t ht).2 hd).mono le⟩⟩

section Advance

variable {shapes : Array TreeShape} {k t : ℕ} {g : Replay.GkrClaims} {pairs : List (ℕ × Array Ext)}
  {next : Array Ext} {tau : Ext}

/-- An inner tree's next pair (`gkr.rs:741-742`). -/
private theorem advance_current (inside : t < g.current.size) :
    (Replay.advance shapes k g pairs next tau).current[t]! =
      if k < shapes[t]!.depth then ((Replay.reducedOf pairs tau t)[0]!, (Replay.reducedOf pairs tau t)[1]!)
      else g.current[t]! := by
  simp only [Replay.advance]
  exact Bridge.getElem!_map_range _ _ inside

/-- A leaf tree's point (`gkr.rs:735-740`). -/
private theorem advance_leafPoint (inside : t < g.leafPoint.size) :
    (Replay.advance shapes k g pairs next tau).leafPoint[t]! =
      if shapes[t]!.depth = k then next.push tau else g.leafPoint[t]! := by
  simp only [Replay.advance]
  exact Bridge.getElem!_map_range _ _ inside

/-- A leaf tree's values (`gkr.rs:735-740`). -/
private theorem advance_leafValues (inside : t < g.leafValues.size) :
    (Replay.advance shapes k g pairs next tau).leafValues[t]! =
      if shapes[t]!.depth = k then Replay.reducedOf pairs tau t else g.leafValues[t]! := by
  simp only [Replay.advance]
  exact Bridge.getElem!_map_range _ _ inside

/-- A tree that does not end at layer `k` keeps its claim (`gkr.rs:735-743`). -/
private theorem IsTreeClaim.advance {c : Ctx F} {m : ℕ} {v : TreeClaimView} (h : IsTreeClaim c m v g t)
    (ne : shapes[t]!.depth ≠ k) (hp : t < g.leafPoint.size) (hv : t < g.leafValues.size) :
    IsTreeClaim c m v (Replay.advance shapes k g pairs next tau) t := by
  refine ⟨h.1, ?_, ?_⟩
  · rw [advance_leafPoint hp, if_neg ne]
    exact h.2.1
  · rw [advance_leafValues hv, if_neg ne]
    exact h.2.2

end Advance

/-- The active trees of layer `k` (`gkr.rs:696-698`). -/
private theorem mem_activeTrees {shapes : Array TreeShape} {k t : ℕ} :
    t ∈ (Replay.activeTrees shapes k).toList ↔ t < shapes.size ∧ k ≤ shapes[t]!.depth := by
  unfold Replay.activeTrees
  simp

/-- No tree is active twice (`gkr.rs:696-698`). -/
private theorem activeTrees_nodup (shapes : Array TreeShape) (k : ℕ) :
    (Replay.activeTrees shapes k).toList.Nodup := by
  unfold Replay.activeTrees
  rw [Array.toList_filter, Array.toList_range]
  exact List.nodup_range.filter _

/-- The trees that step `s` of the reduction (`gkr.rs:726`) adds. -/
private theorem mem_take_succ {l : List ℕ} (nodup : l.Nodup) {s : ℕ} (hs : s < l.length) :
    (∀ t, t ∈ l.take (s + 1) ↔ t ∈ l.take s ∨ t = l[s]) ∧ l[s] ∉ l.take s := by
  have e : l.take (s + 1) = l.take s ++ [l[s]] := by
    rw [List.take_add_one, List.getElem?_eq_getElem hs, Option.toList_some]
  refine ⟨fun t => by rw [e, List.mem_append, List.mem_singleton], fun member => ?_⟩
  have := (nodup.sublist (List.take_sublist (s + 1) l))
  rw [e, List.nodup_append] at this
  exact this.2.2 _ member _ (List.mem_singleton_self _) rfl

/-- The replay reduces an active tree's children (`gkr.rs:727-733`). -/
private theorem reducedOf_active {shapes : Array TreeShape} {k s : ℕ} {gchildren : Array (Array Ext)} {tau : Ext}
    (size : gchildren.size = (Replay.activeTrees shapes k).size) (hs : s < (Replay.activeTrees shapes k).size) :
    Replay.reducedOf ((Replay.activeTrees shapes k).zip gchildren).toList tau (Replay.activeTrees shapes k)[s] =
      (Array.range (gchildren[s]!.size / 2)).map fun j =>
        Replay.reduce ((gchildren[s]!)[2 * j]!, (gchildren[s]!)[2 * j + 1]!) tau := by
  have nodup : ((((Replay.activeTrees shapes k).zip gchildren).toList).map Prod.fst).Nodup := by
    rw [Array.toList_zip, List.map_fst_zip (by simp [size])]
    exact activeTrees_nodup shapes k
  have member : ((Replay.activeTrees shapes k)[s], gchildren[s]!) ∈
      ((Replay.activeTrees shapes k).zip gchildren).toList :=
    List.mem_iff_getElem.mpr ⟨s, by simp; omega, by simp [getElem!_pos gchildren s (by omega)]⟩
  unfold Replay.reducedOf
  rw [Bridge.find_of_mem nodup member]

/-- The value of `sample_ext` (`gkr.rs:699`). -/
private def drawnExt (t : Replay.Transcript) : Ext :=
  ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩

/-- The transcript after `sample_ext` (`gkr.rs:699`). -/
private def afterExt (t : Replay.Transcript) : Replay.Transcript :=
  { duplex := t.duplex.sample.2.sample.2.sample.2, draws := t.draws.push (.ext (drawnExt t)) }

namespace Ensures

variable {c : Ctx F} {n : ℕ} {P : Prop}

/-- Fix the witness of an existential precondition, such as the
sum-check's run (`gkr.rs:715`). -/
private theorem elim {α ι : Type} {b : Build α} {Q : α → ℕ → Prop} {R : ι → Prop} (x₀ : ι)
    (h : ∀ x, Ensures c n (R x) b Q) (hp : P → ∃ x, R x) : Ensures c n P b Q := by
  intro p E hp' hE
  obtain ⟨new, hi, hl, -⟩ := h x₀ p E hp' hE
  refine ⟨new, hi, hl, fun hP hF hh => ?_⟩
  obtain ⟨x, hx⟩ := hp hP
  obtain ⟨new', hi', -, hs⟩ := h x p E hp' hE
  obtain rfl : new = new' := List.toArray_inj ((Array.append_right_inj _).mp (hi.symm.trans hi'))
  exact hs hx hF hh

/-- `tbind` after a step that checks nothing, such as a draw of
`gkr.rs:699-700`. -/
private theorem tbindSim {α β : Type} {t : Transcript α} {g : α → Transcript β} {d : Duplex}
    {R : β × Duplex → ℕ → Prop} {Q : α × Duplex → ℕ → Prop} (ht : Sim c n P (t.run d) Q True)
    (hg : ∀ a e m, n ≤ m → Ensures c m (P ∧ Q (a, e) m) ((g a).run e) R) : Ensures c n P ((t >>= g).run d) R :=
  Ensures.tbind (ht.ensures.post fun _ _ _ h => h.1) hg

end Ensures

namespace Sim

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

/-- `combine` (`gkr.rs:659-677`): the replay's `combineVals`. A tree has at
least two child values. -/
private theorem gkrCombine {children : Array ExtVar} {gchildren : Array Ext} {mu : ExtVar} {gmu : Ext}
    (hc : P → IsExts c n children gchildren) (two : P → 2 ≤ gchildren.size) (hmu : P → IsExt c n mu gmu) :
    Sim c n P (Circuit.gkrCombine children mu) (fun r m => IsExt c m r (Replay.combineVals gchildren gmu)) True := by
  unfold Circuit.gkrCombine
  dsimp only
  have numerator : ∀ side ≤ 1, ∀ {m : ℕ} {P' : Prop}, (P' → P) → n ≤ m →
      Sim c m P' (do
          let total ← Algebra.extOne
          let total ← forIn [:children.size / 2 - 1] total fun j total => do
            let total ← Build.extMul total children[2 * j + side]!
            Pure.pure (ForInStep.yield total)
          Pure.pure total)
        (fun r m' => IsExt c m' r (Replay.prodRange (children.size / 2 - 1) fun j => gchildren[2 * j + side]!))
        True := by
    intro side hside m P' hp le
    refine extOne.bind₀ fun total m₁ l₁ => ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range', ← List.forIn_toArray]
    refine ((loop₀ (P := P' ∧ IsExt c m₁ total 1)
      (fun j acc m' => IsExt c m' acc (Replay.prodRange j fun j => gchildren[2 * j + side]!))
      fun j hj acc m' lm' => ?_).pre fun h => ⟨h, h.2⟩).post (fun _ _ _ h => by simpa using h) fun _ => Iff.rfl
    have hj' : j < children.size / 2 - 1 := by simpa using hj
    rw [show (List.range' [:children.size / 2 - 1].start [:children.size / 2 - 1].size
      [:children.size / 2 - 1].step).toArray[j] = j by simp]
    refine (Sim.extMul (fun h => h.2) fun h => ((hc (hp h.1.1)).2 (2 * j + side) (by omega)).mono
      (le.trans (l₁.trans lm'))).bind₀ fun total' m₃ l₃ => Sim.pure fun h => ⟨rfl, ?_⟩
    rw [prodRange_succ]
    exact h.2
  have hq : ∀ i ≤ 1, ∀ {m : ℕ} {P' : Prop}, (P' → P) → n ≤ m → P' →
      IsExt c m children[2 * (children.size / 2 - 1) + i]! gchildren[2 * (children.size / 2 - 1) + i]! :=
    fun i hi m P' hp le h => ((hc (hp h)).2 _ (by have := two (hp h); have := (hc (hp h)).1; omega)).mono le
  refine (numerator 0 (by omega) id le_rfl).bind₀ fun p0 m₁ l₁ => ?_
  refine (numerator 1 le_rfl (fun h => h.1) l₁).bind₀ fun p1 m₂ l₂ => ?_
  refine (Sim.extMul (fun h => h.1.2.mono l₂) fun h => hq 1 le_rfl (fun h => h.1.1) (l₁.trans l₂) h).bind₀
    fun a m₃ l₃ => ?_
  refine (Sim.extMul (fun h => h.1.2.mono l₃)
    fun h => hq 0 (by omega) (fun h => h.1.1.1) (l₁.trans (l₂.trans l₃)) h).bind₀ fun b m₄ l₄ => ?_
  refine (Sim.extMul (fun h => hq 0 (by omega) (fun h => h.1.1.1.1) (l₁.trans (l₂.trans (l₃.trans l₄))) h)
    fun h => hq 1 le_rfl (fun h => h.1.1.1.1) (l₁.trans (l₂.trans (l₃.trans l₄))) h).bind₀ fun qq m₅ l₅ => ?_
  refine (Sim.extMul (fun h => (hmu h.1.1.1.1.1).mono (l₁.trans (l₂.trans (l₃.trans (l₄.trans l₅)))))
    fun h => h.2).bind₀ fun d m₆ l₆ => ?_
  refine (Sim.extAdd (fun h => h.1.1.1.2.mono (l₄.trans (l₅.trans l₆))) fun h => h.1.1.2.mono (l₅.trans l₆)).bind₀
    fun sum m₇ l₇ => ?_
  refine (Sim.extAdd (fun h => h.2) fun h => h.1.2.mono l₇).post (fun h _ _ e => ?_) fun _ => Iff.rfl
  rw [(hc h.1.1.1.1.1.1.1).1] at e
  simp only [Replay.combineVals, ext_mul_assoc]
  exact e

omit [CharP F goldilocksModulus] in
/-- `mapM` in a transcript whose steps keep the duplex (the reduction of
`gkr.rs:727-733`). -/
private theorem listMapMT {δ α : Type} {f : δ → Transcript α} (R : δ → α → ℕ → Prop)
    (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m') :
    ∀ (xs : List δ) {n : ℕ} {P : Prop} {d : Duplex},
      (∀ x ∈ xs, ∀ m e, n ≤ m → Sim c m P ((f x).run e) (fun r m => R x r.1 m ∧ r.2 = e) True) →
      Sim c n P ((xs.mapM f).run d) (fun r m => List.Forall₂ (fun x a => R x a m) xs r.1 ∧ r.2 = d) True
  | [], n, P, d, _ => by
      rw [List.mapM_nil, StateT.run_pure]
      exact Sim.pure fun _ => ⟨List.Forall₂.nil, rfl⟩
  | x :: xs, n, P, d, h => by
      rw [List.mapM_cons, StateT.run_bind]
      refine (h x List.mem_cons_self n d le_rfl).bind₀ fun r m le => ?_
      rw [StateT.run_bind]
      refine (listMapMT R mono xs (P := P ∧ (R x r.1 m ∧ r.2 = d)) (d := r.2) fun y hy m' e le' =>
        (h y (List.mem_cons_of_mem _ hy) m' e (le.trans le')).pre fun h => h.1).bind₀ fun r' m' le' => ?_
      rw [StateT.run_pure]
      exact Sim.pure fun hh => ⟨List.Forall₂.cons (mono _ _ _ _ hh.1.2.1 le') hh.2.1, hh.2.2.trans hh.1.2.2⟩

/-- The reduced pairs of one tree (`gkr.rs:727-733`): the replay's `reduce`. -/
private theorem reduceValues {values : Array ExtVar} {gvalues : Array Ext} {tau : ExtVar} {gtau : Ext}
    {d : Duplex} (hv : P → IsExts c n values gvalues) (ht : P → IsExt c n tau gtau) :
    Sim c n P (((Array.range (values.size / 2)).mapM (m := Transcript) fun j => do
        let difference ← liftM (Build.extSub values[2 * j + 1]! values[2 * j]!)
        let step ← liftM (Build.extMul tau difference)
        liftM (Build.extAdd values[2 * j]! step)).run d)
      (fun res m => IsExts c m res.1
        ((Array.range (gvalues.size / 2)).map fun j => Replay.reduce (gvalues[2 * j]!, gvalues[2 * j + 1]!) gtau) ∧
        res.2 = d) True := by
  rw [Array.mapM_eq_mapM_toList, StateT.run_map]
  refine ((listMapMT (fun j a m => IsExt c m a (Replay.reduce (gvalues[2 * j]!, gvalues[2 * j + 1]!) gtau))
    (fun _ _ _ _ h le => h.mono le) _ fun j member m e le => ?_).map
      (Q' := fun r m => List.Forall₂
        (fun j a => IsExt c m a (Replay.reduce (gvalues[2 * j]!, gvalues[2 * j + 1]!) gtau))
        (List.range (values.size / 2)) r.1.toList ∧ r.2 = d) fun r m h => ?_).post (fun h r m q => ?_)
      fun _ => Iff.rfl
  · have hj : 2 * j + 1 < values.size := by
      simp only [Array.toList_range, List.mem_range] at member
      omega
    refine Sim.tbind₀ (Sim.lift (Sim.extSub (fun h => ((hv h).2 _ hj).mono le)
      fun h => ((hv h).2 _ (by omega)).mono le)) fun diff e₁ m₁ l₁ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => (ht h.1).mono (le.trans l₁)) fun h => h.2.1))
      fun step e₂ m₂ l₂ => ?_
    exact (Sim.lift (Sim.extAdd (fun h => ((hv h.1.1).2 _ (by omega)).mono (le.trans (l₁.trans l₂)))
      fun h => h.2.1)).post (fun h r m q => ⟨q.1, q.2.trans (h.2.2.trans h.1.2.2)⟩) fun _ => Iff.rfl
  · simpa using h
  · rw [(hv h).1] at q
    exact ⟨isExts_of_forall₂ q.1, q.2⟩

/-- `for root in &view.roots` (`gkr.rs:687-690`): the replay's
`observeExts (flatPairs roots)`. -/
private theorem observePairs {d : Duplex} {r : Replay.Duplex} {vs : Array (ExtVar × ExtVar)}
    {gs : Array (Ext × Ext)} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    (hv : P → vs.size = gs.size ∧ ∀ i < vs.size, IsExt c n (vs[i]!).1 (gs[i]!).1 ∧ IsExt c n (vs[i]!).2 (gs[i]!).2) :
    Sim c n P ((forIn vs PUnit.unit fun (x : ExtVar × ExtVar) _ => do
        Duplex.observeExt x.1
        Duplex.observeExt x.2
        Pure.pure (ForInStep.yield PUnit.unit)).run d)
      (fun res m => IsDuplex c m res.2
        (((Replay.flatPairs gs).flatMap fun g => #[g.c0, g.c1, g.c2]).foldl Replay.Duplex.observe r)) True := by
  rw [← Array.forIn_toList]
  refine ((loopT (fun k _ e m => IsDuplex c m e
      (((gs.toList.take k).flatMap fun g => [g.1.c0, g.1.c1, g.1.c2, g.2.c0, g.2.c1, g.2.c2]).foldl
        Replay.Duplex.observe r)) (fun _ => True)
    fun k hk _ e m le => ?_).pre fun h => ⟨h, by simpa using hd h⟩).post (fun h _ _ q => ?_)
    fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩
  · have hk' : k < vs.size := by simpa using hk
    have hx : P → IsExt c n (vs.toList[k]).1 (gs[k]!).1 ∧ IsExt c n (vs.toList[k]).2 (gs[k]!).2 := fun h => by
      rw [Array.getElem_toList, ← getElem!_pos vs k hk']
      exact (hv h).2 k hk'
    rw [StateT.run_bind]
    refine (observeExt hp (fun h => h.2) fun h => ((hx h.1).1).mono le).bind₀ fun res m₁ l₁ => ?_
    rw [StateT.run_bind]
    refine (observeExt hp (fun h => h.2) fun h => ((hx h.1.1).2).mono (le.trans l₁)).bind₀ fun res₂ m₂ l₂ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_⟩
    have hg : k < gs.size := (hv h.1.1.1).1 ▸ hk'
    rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some, List.flatMap_append,
      List.foldl_append, Array.getElem_toList, ← getElem!_pos gs k hg]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.foldl_cons, List.foldl_nil]
    exact h.2
  · rw [show vs.toList.length = gs.toList.length by simp [(hv h).1], List.take_length] at q
    rw [← Array.foldl_toList, Array.toList_flatMap, Replay.flatPairs, Array.toList_flatMap, List.flatMap_assoc]
    simpa using q

/-- `for values in children` (`gkr.rs:719-723`): the replay's `observeExts`
of the flattened children. -/
private theorem observeChildren {d : Duplex} {r : Replay.Duplex} {vs : Array (Array ExtVar)}
    {gs : Array (Array Ext)} (hp : Permutes c.perm) (hd : P → IsDuplex c n d r)
    (hv : P → vs.size = gs.size ∧ ∀ i < vs.size, IsExts c n vs[i]! gs[i]!) :
    Sim c n P ((forIn vs PUnit.unit fun values _ => do
        forIn values PUnit.unit fun value _ => do
          Duplex.observeExt value
          Pure.pure (ForInStep.yield PUnit.unit)
        Pure.pure (ForInStep.yield PUnit.unit)).run d)
      (fun res m => IsDuplex c m res.2 ((gs.flatten.flatMap fun g => #[g.c0, g.c1, g.c2]).foldl
        Replay.Duplex.observe r)) True := by
  rw [← Array.forIn_toList]
  refine ((loopT (fun k _ e m => IsDuplex c m e
      (((gs.toList.take k).flatMap fun g => g.toList.flatMap fun v => [v.c0, v.c1, v.c2]).foldl
        Replay.Duplex.observe r)) (fun _ => True)
    fun k hk _ e m le => ?_).pre fun h => ⟨h, by simpa using hd h⟩).post (fun h _ _ q => ?_)
    fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩
  · have hk' : k < vs.size := by simpa using hk
    rw [StateT.run_bind]
    refine (observeExts (gs := gs[k]!) hp (fun h => h.2) fun h => ?_).bind₀ fun res m₁ l₁ => ?_
    · rw [Array.getElem_toList, ← getElem!_pos vs k hk']
      exact ((hv h.1).2 k hk').mono le
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_⟩
    have hg : k < gs.size := (hv h.1.1).1 ▸ hk'
    rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hg), Option.toList_some, List.flatMap_append,
      List.foldl_append, Array.getElem_toList, ← getElem!_pos gs k hg]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
    have := h.2
    rw [← Array.foldl_toList, Array.toList_flatMap] at this
    exact this
  · rw [show vs.toList.length = gs.toList.length by simp [(hv h).1], List.take_length] at q
    rw [← Array.foldl_toList, Array.toList_flatMap, Array.toList_flatten, flatMap_flatten, List.flatMap_map]
    exact q

/-- The batched claim and its expected value at one layer
(`gkr.rs:704-714`): the replay's `batchClaim` and `batchExpected`. -/
private theorem batchLoop {active : Array ℕ} {children : Array (Array ExtVar)} {gchildren : Array (Array Ext)}
    {cur : Array (ExtVar × ExtVar)} {gcur : Array (Ext × Ext)} {mu nu : ExtVar} {gmu gnu : Ext}
    {init : ExtVar × ExtVar × ExtVar} {d : Duplex}
    (hmu : P → IsExt c n mu gmu) (hnu : P → IsExt c n nu gnu)
    (hcur : P → ∀ i ∈ active, IsExt c n (cur[i]!).1 (gcur[i]!).1 ∧ IsExt c n (cur[i]!).2 (gcur[i]!).2)
    (hch : P → children.size = gchildren.size ∧
      ∀ s < children.size, IsExts c n children[s]! gchildren[s]! ∧ 2 ≤ gchildren[s]!.size)
    (hinit : P → IsExt c n init.1 0 ∧ IsExt c n init.2.1 0 ∧ IsExt c n init.2.2 1) :
    Sim c n P ((forIn (m := Transcript) (active.zip children) init
        fun (x : ℕ × Array ExtVar) (acc : ExtVar × ExtVar × ExtVar) => do
          let mq ← liftM (Build.extMul mu cur[x.1]!.2)
          let pq ← liftM (Build.extAdd cur[x.1]!.1 mq)
          let term ← liftM (Build.extMul acc.2.2 pq)
          let claim ← liftM (Build.extAdd acc.1 term)
          let combined ← liftM (Circuit.gkrCombine x.2 mu)
          let term ← liftM (Build.extMul acc.2.2 combined)
          let expected ← liftM (Build.extAdd acc.2.1 term)
          let weight ← liftM (Build.extMul acc.2.2 nu)
          Pure.pure (ForInStep.yield (claim, expected, weight))).run d)
      (fun res m => IsExt c m res.1.1 (Replay.batchClaim gcur gmu gnu (active.zip gchildren).toList) ∧
        IsExt c m res.1.2.1 (Replay.batchExpected gmu gnu (active.zip gchildren).toList) ∧ res.2 = d) True := by
  rw [← Array.forIn_toList]
  set pairs := (active.zip gchildren).toList with pairs_def
  refine ((loopT (fun s acc e m => e = d ∧ ∃ A E W : Ext, IsExt c m acc.1 A ∧ IsExt c m acc.2.1 E ∧
      IsExt c m acc.2.2 W ∧
      Replay.batchClaim gcur gmu gnu pairs = A + W * Replay.batchClaim gcur gmu gnu (pairs.drop s) ∧
      Replay.batchExpected gmu gnu pairs = E + W * Replay.batchExpected gmu gnu (pairs.drop s)) (fun _ => True)
    fun s hs acc e m le => ?_).pre fun h => ⟨h, rfl, 0, 0, 1, (hinit h).1, (hinit h).2.1, (hinit h).2.2,
      zero_add_one_mul _, zero_add_one_mul _⟩).post (fun h r m q => ?_)
      fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩
  · have hsa : s < active.size := by simp at hs; omega
    have hsc : s < children.size := by simp at hs; omega
    refine Sim.elim (R := fun w : Ext × Ext × Ext => P ∧ e = d ∧ IsExt c m acc.1 w.1 ∧ IsExt c m acc.2.1 w.2.1 ∧
        IsExt c m acc.2.2 w.2.2 ∧
        Replay.batchClaim gcur gmu gnu pairs = w.1 + w.2.2 * Replay.batchClaim gcur gmu gnu (pairs.drop s) ∧
        Replay.batchExpected gmu gnu pairs = w.2.1 + w.2.2 * Replay.batchExpected gmu gnu (pairs.drop s))
      (0, 0, 0) (fun w => ?_) fun ⟨hP, he, A, E, W, h⟩ => ⟨(A, E, W), hP, he, h⟩
    obtain ⟨A, E, W⟩ := w
    dsimp only
    rw [show (active.zip children).toList[s] = (active[s], children[s]) by simp]
    dsimp only
    have hi : P → IsExt c m (cur[active[s]]!).1 (gcur[active[s]]!).1 ∧
        IsExt c m (cur[active[s]]!).2 (gcur[active[s]]!).2 :=
      fun h => ⟨((hcur h _ (Array.getElem_mem _)).1).mono le, ((hcur h _ (Array.getElem_mem _)).2).mono le⟩
    have hv : P → IsExts c m children[s] gchildren[s]! ∧ 2 ≤ gchildren[s]!.size := fun h => by
      have := (hch h).2 s hsc
      rw [getElem!_pos children s hsc] at this
      exact ⟨this.1.mono le, this.2⟩
    refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => (hmu h.1).mono le) fun h => (hi h.1).2)) fun mq e₁ m₁ l₁ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extAdd (fun h => ((hi h.1.1).1).mono l₁) fun h => h.2.1)) fun pq e₂ m₂ l₂ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extMul (fun h => h.1.1.2.2.2.2.1.mono (l₁.trans l₂)) fun h => h.2.1))
      fun t₁ e₃ m₃ l₃ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extAdd (fun h => h.1.1.1.2.2.1.mono (l₁.trans (l₂.trans l₃))) fun h => h.2.1))
      fun cl e₄ m₄ l₄ => ?_
    refine Sim.tbind₀ (Sim.lift (gkrCombine (fun h => (hv h.1.1.1.1.1).1.mono (l₁.trans (l₂.trans (l₃.trans l₄))))
      (fun h => (hv h.1.1.1.1.1).2) fun h => (hmu h.1.1.1.1.1).mono (le.trans (l₁.trans (l₂.trans (l₃.trans l₄))))))
      fun cb e₅ m₅ l₅ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extMul
      (fun h => h.1.1.1.1.1.2.2.2.2.1.mono (l₁.trans (l₂.trans (l₃.trans (l₄.trans l₅))))) fun h => h.2.1))
      fun t₂ e₆ m₆ l₆ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extAdd
      (fun h => h.1.1.1.1.1.1.2.2.2.1.mono (l₁.trans (l₂.trans (l₃.trans (l₄.trans (l₅.trans l₆)))))) fun h => h.2.1))
      fun ex e₇ m₇ l₇ => ?_
    refine Sim.tbind₀ (Sim.lift (Sim.extMul
      (fun h => h.1.1.1.1.1.1.1.2.2.2.2.1.mono (l₁.trans (l₂.trans (l₃.trans (l₄.trans (l₅.trans (l₆.trans l₇)))))))
      fun h => (hnu h.1.1.1.1.1.1.1.1).mono
        (le.trans (l₁.trans (l₂.trans (l₃.trans (l₄.trans (l₅.trans (l₆.trans l₇))))))))) fun w e₈ m₈ l₈ => ?_
    rw [StateT.run_pure]
    refine Sim.pure fun h => ⟨rfl, ?_, _, _, _, h.1.1.1.1.2.1.mono (l₅.trans (l₆.trans (l₇.trans l₈))),
      h.1.2.1.mono l₈, h.2.1, ?_, ?_⟩
    · obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨-, e0, -⟩, -, f₁⟩, -, f₂⟩, -, f₃⟩, -, f₄⟩, -, f₅⟩, -, f₆⟩, -, f₇⟩, -, f₈⟩ := h
      exact f₈.trans (f₇.trans (f₆.trans (f₅.trans (f₄.trans (f₃.trans (f₂.trans (f₁.trans e0)))))))
    · have hs : s < pairs.length := by
        have := (hch h.1.1.1.1.1.1.1.1.1).1
        simp [pairs_def, ← this, hsa, hsc]
      rw [h.1.1.1.1.1.1.1.1.2.2.2.2.2.1, List.drop_eq_getElem_cons hs, Replay.batchClaim, horner_step]
      simp [pairs_def]
    · have hs : s < pairs.length := by
        have := (hch h.1.1.1.1.1.1.1.1.1).1
        simp [pairs_def, ← this, hsa, hsc]
      rw [h.1.1.1.1.1.1.1.1.2.2.2.2.2.2, List.drop_eq_getElem_cons hs, Replay.batchExpected, horner_step]
      simp [pairs_def, getElem!_pos gchildren s (by have := (hch h.1.1.1.1.1.1.1.1.1).1; omega)]
  · obtain ⟨e, A, E, W, hA, hE, -, eA, eE⟩ := q
    have len : (active.zip children).toList.length = pairs.length := by simp [pairs_def, (hch h).1]
    rw [len, List.drop_length, Replay.batchClaim, add_mul_zero] at eA
    rw [len, List.drop_length, Replay.batchExpected, add_mul_zero] at eE
    exact ⟨eA ▸ hA, eE ▸ hE, e⟩

/-- The reduction of the active trees at `τ` (`gkr.rs:726-744`): the
replay's `advance`. -/
private theorem reduceLoop {shapes : Array TreeShape} {j : ℕ} {roots : Array (ExtVar × ExtVar)}
    {next : Array ExtVar} {tau : ExtVar} {children : Array (Array ExtVar)} {gc : Replay.GkrClaims}
    {gchildren : Array (Array Ext)} {gnext : Array Ext} {gtau : Ext} {cur : Array (ExtVar × ExtVar)}
    {cl : Array TreeClaimView} {d : Duplex}
    (hs : P → IsLayerState c n shapes j cur cl gc)
    (hroots : P → ∀ t < shapes.size, IsPair c n roots[t]! gc.roots[t]!)
    (hch : P → children.size = gchildren.size ∧ gchildren.size = (Replay.activeTrees shapes (j + 1)).size ∧
      ∀ s < children.size, IsExts c n children[s]! gchildren[s]! ∧
        gchildren[s]!.size = 2 * (Replay.partFactors shapes (j + 1) (Replay.activeTrees shapes (j + 1))[s]! + 1))
    (hnext : P → IsExts c n next (gnext.push gtau)) (htau : P → IsExt c n tau gtau) :
    Sim c n P ((forIn (m := Transcript) ((Replay.activeTrees shapes (j + 1)).zip children) (cur, cl)
        fun (x : ℕ × Array ExtVar) (acc : Array (ExtVar × ExtVar) × Array TreeClaimView) => do
          let reduced ← Array.mapM (fun i => do
              let difference ← liftM (Build.extSub x.2[2 * i + 1]! x.2[2 * i]!)
              let step ← liftM (Build.extMul tau difference)
              liftM (Build.extAdd x.2[2 * i]! step)) (Array.range (x.2.size / 2))
          if shapes[x.1]!.depth = j + 1 then
            Pure.pure (ForInStep.yield
              (acc.1, acc.2.set! x.1 { root := roots[x.1]!, point := next, values := reduced }))
          else Pure.pure (ForInStep.yield (acc.1.set! x.1 (reduced[0]!, reduced[1]!), acc.2))).run d)
      (fun res m => res.2 = d ∧ IsLayerState c m shapes (j + 1) res.1.1 res.1.2
        (Replay.advance shapes (j + 1) gc ((Replay.activeTrees shapes (j + 1)).zip gchildren).toList gnext gtau))
      True := by
  rw [← Array.forIn_toList]
  have nodup := activeTrees_nodup shapes (j + 1)
  set act := Replay.activeTrees shapes (j + 1) with act_def
  set g := Replay.advance shapes (j + 1) gc (act.zip gchildren).toList gnext gtau with g_def
  refine ((loopT (fun s acc e m => e = d ∧ acc.1.size = shapes.size ∧ acc.2.size = shapes.size ∧
      ∀ t < shapes.size, IsPair c m acc.1[t]! (if t ∈ act.toList.take s then g.current[t]! else gc.current[t]!) ∧
        (shapes[t]!.depth ≤ j ∨ (t ∈ act.toList.take s ∧ shapes[t]!.depth = j + 1) → IsTreeClaim c m acc.2[t]! g t))
    (fun _ => True) fun s hs' acc e m le => ?_).pre fun h => ⟨h, ?_⟩).post (fun h r m q => ?_)
      fun _ => ⟨fun _ => trivial, fun _ _ _ => trivial⟩
  · have hsa : s < act.size := by simp at hs'; omega
    have hsc : s < children.size := by simp at hs'; omega
    have hsl : s < act.toList.length := by simpa using hsa
    rw [show (act.zip children).toList[s] = (act[s], children[s]) by simp]
    dsimp only
    have hmem : act.toList[s] ∈ act.toList := List.getElem_mem hsl
    rw [Array.getElem_toList] at hmem
    obtain ⟨hin, hdeep⟩ := mem_activeTrees.mp hmem
    obtain ⟨split, fresh⟩ := mem_take_succ nodup hsl
    rw [Array.getElem_toList] at split fresh
    have hv : P → IsExts c m children[s] gchildren[s]! ∧
        gchildren[s]!.size = 2 * (Replay.partFactors shapes (j + 1) act[s] + 1) := fun h => by
      have := (hch h).2.2 s hsc
      rw [getElem!_pos children s hsc, getElem!_pos act s hsa] at this
      exact ⟨this.1.mono le, this.2⟩
    refine Sim.tbind₀ (reduceValues (fun h => (hv h.1).1) fun h => (htau h.1).mono le) fun reduced e₁ m₁ l₁ => ?_
    have hro : P → Replay.reducedOf (act.zip gchildren).toList gtau act[s] =
        (Array.range (gchildren[s]!.size / 2)).map fun i =>
          Replay.reduce ((gchildren[s]!)[2 * i]!, (gchildren[s]!)[2 * i + 1]!) gtau :=
      fun h => reducedOf_active (hch h).2.1 hsa
    by_cases hk : shapes[act[s]]!.depth = j + 1
    · rw [if_pos hk, StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      dsimp only [ForInStep.value]
      obtain ⟨⟨hP, he, sz1, sz2, inv⟩, hr, he₁⟩ := h
      obtain ⟨c1, c2, c3, c4, c5, c6⟩ := hs hP
      refine ⟨he₁.trans he, sz1, by rw [Array.size_set!]; exact sz2, fun t ht => ⟨?_, fun hd => ?_⟩⟩
      · by_cases ts : t = act[s]
        · subst ts
          rw [if_pos ((split _).mpr (Or.inr rfl)), advance_current (by omega), if_neg (by omega)]
          have := (inv _ ht).1
          rw [if_neg fresh] at this
          exact this.mono l₁
        · have := (inv t ht).1
          simp only [(split t).trans (or_iff_left ts)]
          exact this.mono l₁
      · rw [getElem!_set! _ _ _ _ (by omega)]
        by_cases ts : act[s] = t
        · subst ts
          rw [if_pos rfl]
          refine ⟨(hroots hP _ ht).mono (le.trans l₁), ?_, ?_⟩
          · rw [advance_leafPoint (by omega), if_pos hk]
            exact (hnext hP).mono (le.trans l₁)
          · rw [advance_leafValues (by omega), if_pos hk, hro hP]
            exact hr
        · rw [if_neg ts]
          exact ((inv t ht).2 (hd.imp_right fun hd' =>
            ⟨((split t).mp hd'.1).resolve_right (Ne.symm ts), hd'.2⟩)).mono l₁
    · rw [if_neg hk, StateT.run_pure]
      refine Sim.pure fun h => ⟨rfl, ?_⟩
      dsimp only [ForInStep.value]
      obtain ⟨⟨hP, he, sz1, sz2, inv⟩, hr, he₁⟩ := h
      obtain ⟨c1, c2, c3, c4, c5, c6⟩ := hs hP
      have inner : j + 1 < shapes[act[s]]!.depth := by omega
      refine ⟨he₁.trans he, by rw [Array.size_set!]; exact sz1, sz2, fun t ht => ⟨?_, fun hd => ?_⟩⟩
      · rw [getElem!_set! _ _ _ _ (by omega)]
        by_cases ts : act[s] = t
        · subst ts
          rw [if_pos rfl, if_pos ((split _).mpr (Or.inr rfl)), advance_current (by omega), if_pos inner, hro hP]
          have size : gchildren[s]!.size / 2 = 2 := by
            rw [(hv hP).2, Replay.partFactors, if_neg hk]
          exact ⟨hr.2 0 (by rw [hr.1]; simp [size]), hr.2 1 (by rw [hr.1]; simp [size])⟩
        · rw [if_neg ts]
          have := (inv t ht).1
          simp only [(split t).trans (or_iff_left (Ne.symm ts))]
          exact this.mono l₁
      · by_cases ts : t = act[s]
        · subst ts
          exfalso
          rcases hd with hd | hd
          · omega
          · exact hk hd.2
        · exact ((inv t ht).2 (hd.imp_right fun hd' => ⟨((split t).mp hd'.1).resolve_right ts, hd'.2⟩)).mono l₁
  · obtain ⟨c1, c2, c3, c4, c5, c6⟩ := hs h
    refine ⟨rfl, c1, c2, fun t ht => ⟨?_, fun hd => ?_⟩⟩
    · simpa using (c6 t ht).1
    · simp only [List.take_zero, List.not_mem_nil, false_and, or_false] at hd
      exact ((c6 t ht).2 hd).advance (by omega) (by omega) (by omega)
  · obtain ⟨he, sz1, sz2, inv⟩ := q
    obtain ⟨c1, c2, c3, c4, c5, c6⟩ := hs h
    have len : (act.zip children).toList.length = act.toList.length := by
      simp [(hch h).1, (hch h).2.1]
    rw [len, List.take_length] at inv
    refine ⟨he, sz1, sz2, by simp [g_def, Replay.advance, c3], by simp [g_def, Replay.advance, c4],
      by simp [g_def, Replay.advance, c5], fun t ht => ⟨?_, fun hd => ?_⟩⟩
    · have := (inv t ht).1
      split_ifs at this with member
      · exact this
      · rw [advance_current (by omega), if_neg ?_]
        · exact this
        · intro deep
          exact member (mem_activeTrees.mpr ⟨ht, by omega⟩)
    · refine (inv t ht).2 ?_
      by_cases deep : shapes[t]!.depth ≤ j
      · exact Or.inl deep
      · exact Or.inr ⟨mem_activeTrees.mpr ⟨ht, by omega⟩, by omega⟩

end Sim

/-- `gkr::verify` (`gkr.rs:679-751`): the replay's `gkrVerify` runs, and
every tree's claim is the replay's. The proof has the shapes' layers
(`readGkr`). -/
theorem gkrVerify_ensures [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop} {view : GkrView}
    {proof : Replay.GkrProof} {shapes : Array TreeShape} {d : Duplex} {t : Replay.Transcript}
    (hp : Permutes c.perm) (hd : P → IsDuplex c n d t.duplex)
    (hroots : P → view.roots.size = proof.roots.size ∧ ∀ i < view.roots.size,
      IsExt c n (view.roots[i]!).1 (proof.roots[i]!).1 ∧ IsExt c n (view.roots[i]!).2 (proof.roots[i]!).2)
    (hlayers : P → view.layers.size = proof.layers.size ∧ ∀ i (hi : i < view.layers.size),
      (view.layers[i].rounds.size = (proof.layers[i]!).rounds.size ∧
        ∀ r < view.layers[i].rounds.size, IsExts c n view.layers[i].rounds[r]! (proof.layers[i]!).rounds[r]!) ∧
      (view.layers[i].children.size = (proof.layers[i]!).children.size ∧
        ∀ s < view.layers[i].children.size,
          IsExts c n view.layers[i].children[s]! (proof.layers[i]!).children[s]!))
    (read : Replay.readGkr proof shapes = .ok ()) :
    Ensures c n P ((Circuit.gkrVerify view shapes).run d)
      (fun res m => ∃ g t', Runs (Replay.gkrVerify proof shapes) t g t' ∧ IsGkrClaims c m res.1 g ∧
        IsDuplex c m res.2 t'.duplex) := by
  obtain ⟨rootsSize, layersSize, layersOk⟩ := Bridge.readGkr_ok read
  unfold Circuit.gkrVerify
  dsimp only
  set init : Replay.GkrClaims :=
    { roots := proof.roots, point := #[], current := proof.roots,
      leafPoint := shapes.map fun _ => #[], leafValues := shapes.map fun _ => #[] } with init_def
  set t₁ : Replay.Transcript := { t with duplex :=
    ((Replay.flatPairs proof.roots).flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe t.duplex }
    with t₁_def
  refine Ensures.tbindSim (Sim.observePairs hp hd fun h => hroots h) fun _ e₀ m₀ l₀ => ?_
  refine Ensures.tbind (Q := fun r m => ∃ g tL, Runs (Replay.gkrLayers shapes 1 proof.layers.toList init) t₁ g tL ∧
      IsDuplex c m r.2 tL.duplex ∧ IsLayerState c m shapes proof.layers.size r.1.1 r.1.2.1 g ∧ g.roots = proof.roots)
    ?_ fun res e m le => ?_
  · rw [← Array.forIn_toList]
    refine ((Ensures.loopT (P := P) (fun j acc e m => ∃ g tj,
        Runs (Replay.gkrLayers shapes 1 (proof.layers.toList.take j) init) t₁ g tj ∧ IsDuplex c m e tj.duplex ∧
        IsLayerState c m shapes j acc.1 acc.2.1 g ∧ IsExts c m acc.2.2 g.point ∧ g.roots = proof.roots)
      fun j hj acc e m le => ?_).pre fun h => ⟨h.1, init, t₁, Replay.runs_pure _ _, h.2, ?_, IsExts.empty, rfl⟩).post
      fun h r m q => ?_
    · have hj' : j < view.layers.size := by simpa using hj
      have N₀ : n ≤ m := l₀.trans le
      rw [show view.layers.zipIdx.toList[j] = (view.layers[j], j) by simp]
      dsimp only
      have hL : ∀ h : P, (view.layers[j].rounds.size = (proof.layers[j]!).rounds.size ∧
          ∀ r < view.layers[j].rounds.size, IsExts c n view.layers[j].rounds[r]! (proof.layers[j]!).rounds[r]!) ∧
          (view.layers[j].children.size = (proof.layers[j]!).children.size ∧
          ∀ s < view.layers[j].children.size,
            IsExts c n view.layers[j].children[s]! (proof.layers[j]!).children[s]!) :=
        fun h => (hlayers h).2 j hj'
      have hS : P → (proof.layers[j]!).rounds.size = j ∧ (∀ r ∈ (proof.layers[j]!).rounds, 3 ≤ r.size) ∧
          (proof.layers[j]!).children.size = (Replay.activeTrees shapes (j + 1)).size ∧
          ∀ s < (proof.layers[j]!).children.size, (proof.layers[j]!).children[s]!.size =
            2 * (Replay.partFactors shapes (j + 1) (Replay.activeTrees shapes (j + 1))[s]! + 1) := fun h => by
        have hjp : j < proof.layers.size := (hlayers h).1 ▸ hj'
        have := Replay.layerShapeOk_facts (layersOk j hjp)
        rw [← getElem!_pos proof.layers j hjp, Nat.add_sub_cancel] at this
        exact this
      refine Ensures.elim (R := fun w : Replay.GkrClaims × Replay.Transcript => P ∧
          Runs (Replay.gkrLayers shapes 1 (proof.layers.toList.take j) init) t₁ w.1 w.2 ∧ IsDuplex c m e w.2.duplex ∧
          IsLayerState c m shapes j acc.1 acc.2.1 w.1 ∧ IsExts c m acc.2.2 w.1.point ∧ w.1.roots = proof.roots)
        (init, t₁) (fun w => ?_) fun ⟨hP, g, tj, h⟩ => ⟨(g, tj), hP, h⟩
      obtain ⟨gc, tj⟩ := w
      dsimp only
      refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2.2.1) fun mu e₁ m₁ l₁ => ?_
      refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2.2) fun nu e₂ m₂ l₂ => ?_
      refine Ensures.tbindSim (Sim.lift Sim.extZero) fun z₁ e₃ m₃ l₃ => ?_
      refine Ensures.tbindSim (Sim.lift Sim.extZero) fun z₂ e₄ m₄ l₄ => ?_
      refine Ensures.tbindSim (Sim.lift Sim.extOne) fun o e₅ m₅ l₅ => ?_
      have L₅ : m ≤ m₅ := l₁.trans (l₂.trans (l₃.trans (l₄.trans l₅)))
      refine Ensures.tbindSim (Sim.batchLoop (active := Replay.activeTrees shapes (j + 1))
        (gchildren := (proof.layers[j]!).children) (gcur := gc.current)
        (fun h => h.1.1.1.1.2.1.mono (l₂.trans (l₃.trans (l₄.trans l₅))))
        (fun h => h.1.1.1.2.1.mono (l₃.trans (l₄.trans l₅)))
        (fun h i hi => ((h.1.1.1.1.1.2.2.2.1.2.2.2.2.2 i
          (mem_activeTrees.mp (Array.mem_toList_iff.mpr hi)).1).1).mono L₅)
        (fun h => ⟨(hL h.1.1.1.1.1.1).2.1, fun s hs => ⟨((hL h.1.1.1.1.1.1).2.2 s hs).mono (N₀.trans L₅), by
          have := (hS h.1.1.1.1.1.1).2.2.2 s (by rw [← (hL h.1.1.1.1.1.1).2.1]; exact hs)
          omega⟩⟩)
        (fun h => ⟨h.1.1.2.1.mono (l₄.trans l₅), h.1.2.1.mono l₅, h.2.1⟩)) fun acc' e₆ m₆ l₆ => ?_
      have L₆ : m ≤ m₆ := L₅.trans l₆
      refine Ensures.tbindSim (Sim.sumcheckReplay (t := afterExt (afterExt tj)) hp
        (fun h => by
          have he : e₆ = e₂ := h.2.2.2.trans (h.1.2.2.trans (h.1.1.2.2.trans h.1.1.1.2.2))
          rw [he]
          exact h.1.1.1.1.2.2.mono (l₃.trans (l₄.trans (l₅.trans l₆))))
        (fun h => ⟨(hL h.1.1.1.1.1.1.1).1.1, fun r hr =>
          ((hL h.1.1.1.1.1.1.1).1.2 r hr).mono (N₀.trans L₆)⟩)
        (fun h r hr => by have := (hS h.1.1.1.1.1.1.1).2.1 r hr; omega)
        (fun h => h.2.1)) fun sc e₇ m₇ l₇ => ?_
      have L₇ : m ≤ m₇ := L₆.trans l₇
      refine Ensures.elim (R := fun w : Array Ext × Ext × Replay.Transcript => P ∧
          Runs (Replay.gkrLayers shapes 1 (proof.layers.toList.take j) init) t₁ gc tj ∧
          IsLayerState c m₇ shapes j acc.1 acc.2.1 gc ∧ IsExts c m₇ acc.2.2 gc.point ∧ gc.roots = proof.roots ∧
          IsExt c m₇ acc'.2.1 (Replay.batchExpected (drawnExt tj) (drawnExt (afterExt tj))
            ((Replay.activeTrees shapes (j + 1)).zip (proof.layers[j]!).children).toList) ∧
          Runs (Replay.sumcheckReplay (proof.layers[j]!).rounds (Replay.batchClaim gc.current (drawnExt tj)
            (drawnExt (afterExt tj)) ((Replay.activeTrees shapes (j + 1)).zip (proof.layers[j]!).children).toList))
            (afterExt (afterExt tj)) (w.1, w.2.1) w.2.2 ∧
          IsExts c m₇ sc.1 w.1 ∧ IsExt c m₇ sc.2 w.2.1 ∧ IsDuplex c m₇ e₇ w.2.2.duplex)
        (#[], 0, tj) (fun w => ?_) fun h => ?_
      swap
      · obtain ⟨⟨⟨⟨⟨⟨⟨⟨hP, run₀, -, state, hpt, roots⟩, -⟩, -⟩, -⟩, -⟩, -⟩, -, hexp, -⟩, point, final, ts, sum, hnext,
          hlast, hdx⟩ := h
        exact ⟨(point, final, ts), hP, run₀, state.mono L₇, hpt.mono L₇, roots, hexp.mono l₇, sum, hnext, hlast, hdx⟩
      obtain ⟨gpoint, gfinal, ts⟩ := w
      dsimp only
      refine Ensures.tbindSim (Sim.lift (Sim.eqEval (fun h => h.2.2.2.1) fun h => h.2.2.2.2.2.2.2.1))
        fun eq e₈ m₈ l₈ => ?_
      refine Ensures.tbindSim (Sim.lift (Sim.extMul (fun h => h.2.1) fun h => h.1.2.2.2.2.2.1.mono l₈))
        fun target e₉ m₉ l₉ => ?_
      refine Ensures.tbind (Sim.lift (Sim.assertExtEqual (fun h => h.1.1.2.2.2.2.2.2.2.2.1.mono (l₈.trans l₉))
        fun h => h.2.1)).ensures fun _ e₁₀ m₁₀ l₁₀ => ?_
      have L₁₀ : m₇ ≤ m₁₀ := l₈.trans (l₉.trans l₁₀)
      refine Ensures.tbindSim (Sim.observeChildren (r := ts.duplex) hp
        (fun h => by
          have he : e₁₀ = e₇ := h.2.1.2.trans (h.1.2.2.trans h.1.1.2.2)
          rw [he]
          exact h.1.1.1.2.2.2.2.2.2.2.2.2.mono L₁₀)
        (fun h => ⟨(hL h.1.1.1.1).2.1, fun s hs =>
          ((hL h.1.1.1.1).2.2 s hs).mono (N₀.trans (L₇.trans L₁₀))⟩)) fun _ e₁₁ m₁₁ l₁₁ => ?_
      refine Ensures.tbindSim (Sim.sampleExt hp fun h => h.2) fun tau e₁₂ m₁₂ l₁₂ => ?_
      have L₁₂ : m₇ ≤ m₁₂ := L₁₀.trans (l₁₁.trans l₁₂)
      refine Ensures.tbindSim (Sim.reduceLoop (gc := gc) (gchildren := (proof.layers[j]!).children)
        (gnext := gpoint)
        (fun h => h.1.1.1.1.1.2.2.1.mono L₁₂)
        (fun h u hu => by
          rw [h.1.1.1.1.1.2.2.2.2.1]
          have := (hroots h.1.1.1.1.1.1).2 u (by rw [(hroots h.1.1.1.1.1.1).1, rootsSize]; exact hu)
          exact ⟨this.1.mono (N₀.trans (L₇.trans L₁₂)), this.2.mono (N₀.trans (L₇.trans L₁₂))⟩)
        (fun h => ⟨(hL h.1.1.1.1.1.1).2.1, (hS h.1.1.1.1.1.1).2.2.1, fun s hs =>
          ⟨((hL h.1.1.1.1.1.1).2.2 s hs).mono (N₀.trans (L₇.trans L₁₂)),
            (hS h.1.1.1.1.1.1).2.2.2 s (by rw [← (hL h.1.1.1.1.1.1).2.1]; exact hs)⟩⟩)
        (fun h => (h.1.1.1.1.1.2.2.2.2.2.2.2.1.mono L₁₂).push h.2.1) fun h => h.2.1) fun st e₁₃ m₁₃ l₁₃ => ?_
      rw [StateT.run_pure]
      refine Ensures.pure fun h => ⟨rfl, ?_⟩
      dsimp only [ForInStep.value]
      obtain ⟨⟨⟨⟨⟨⟨⟨hP, run₀, -, -, roots, -, sum, hnext, -, -⟩, -⟩, -⟩, ⟨-, check⟩⟩, -⟩, hτ, hdτ⟩, he₁₃, state⟩ := h
      have hjp : j < proof.layers.size := (hlayers hP).1 ▸ hj'
      refine ⟨_, _, Replay.runs_gkrLayers_succ hjp run₀ (Replay.runs_gkrLayer (Replay.runs_sampleExt tj)
        (Replay.runs_sampleExt _) sum check (Replay.runs_observeExts _ _) (Replay.runs_sampleExt _)), ?_, state,
        (hnext.mono (L₁₂.trans l₁₃)).push (hτ.mono l₁₃), roots⟩
      have he : e₁₃ = e₁₂ := he₁₃
      rw [he]
      exact hdτ.mono l₁₃
    · have sz := (hroots h.1).1
      refine ⟨by rw [sz, rootsSize], by simp, by simp [init_def, rootsSize], by simp [init_def],
        by simp [init_def], fun u hu => ⟨?_, fun deep => absurd deep (by have := Replay.readGkr_depth read hu; omega)⟩⟩
      have := (hroots h.1).2 u (by omega)
      exact ⟨this.1.mono l₀, this.2.mono l₀⟩
    · obtain ⟨g, tL, run, hdx, state, -, roots⟩ := q
      have len : view.layers.zipIdx.toList.length = proof.layers.toList.length := by simp [(hlayers h.1).1]
      rw [len, List.take_length] at run
      rw [len, Array.length_toList] at state
      exact ⟨g, tL, run, hdx, state, roots⟩
  · rw [StateT.run_pure]
    refine Ensures.pure fun h => ?_
    obtain ⟨g, tL, run, hdx, state, roots⟩ := h.2
    obtain ⟨c1, c2, c3, c4, c5, c6⟩ := state
    refine ⟨g, tL, Replay.runs_bind (Replay.runs_observeExts _ _) run, ⟨by rw [c2, roots, rootsSize], fun u hu => ?_⟩,
      hdx⟩
    have hu' : u < shapes.size := by rw [← c2]; exact hu
    have := (c6 u hu').2 (by rw [layersSize]; exact Replay.depth_le_layerCount shapes hu')
    exact ⟨this.1.1, this.1.2, this.2.1, this.2.2⟩

end CompressionLean.Circuit
