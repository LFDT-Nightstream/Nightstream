import CompressionLean.Bridge.Formulas
import CompressionLean.Game.Path
import CompressionLean.Bridge.Run
import CompressionLean.Layer1.Cost
import CompressionLean.Lib.Array

/-!
Owns the bridge of the sum-check and GKR replays (`Replay/Gkr.lean`) to the
model's games (`Game/SumCheck.lean`, `Layer1/Gkr.lean`). A replay draw
denotes a model challenge (`drawn`). A successful sum-check replay is a
path of the model's sum-check that draws the replay's challenges and ends
at the replay's point and claim (`sumcheck_reaches`): the replay's
interpolation is the model's round polynomial. A successful GKR replay is a
path of the model's GKR that ends in claims agreeing with the replay's on
every tree (`gkr_reaches`).

It also owns facts about the replays alone that the simulation reads: a
sum-check run split at a round (`runs_sumcheckReplay_split`), a GKR run one
layer at a time (`runs_gkrLayers_succ`, `runs_gkrLayer`), and the shapes that
`readGkr` checks (`readGkr_depth`, `layerShapeOk_facts`).
-/

namespace CompressionLean.Bridge

open CompressionLean.Lib
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Gl K] [Fintype K] [ChallengeField K]

/-- A replay draw as the model's challenge: an `Ext` value in `K`, a pair in
`K × K`, the setup query words as `Fin queries → Fin goldilocksModulus`, a WHIR
query word in `Fin goldilocksModulus`. -/
def drawn (queries : ℕ) : Draw → (Σ C : Type, C)
  | .ext v => ⟨K, v.toK⟩
  | .pair a b => ⟨K × K, (a.toK, b.toK)⟩
  | .words ws => ⟨Fin queries → Fin goldilocksModulus, fun j => ⟨(ws[j.val]!).word.toNat, Word.canonical _⟩⟩
  | .sample w => ⟨Fin goldilocksModulus, ⟨w.word.toNat, w.canonical⟩⟩

/-! ## Interpolation -/

section Lagrange

variable {L : Type} [Field L]

theorem prod_erase_range (n : ℕ) (i : Fin n) (f : ℕ → L) :
    ∏ j ∈ (Finset.univ : Finset (Fin n)).erase i, f j = ∏ j ∈ Finset.range n, if (i : ℕ) = j then 1 else f j := by
  rw [← Fin.prod_univ_eq_prod_range (fun j => if (i : ℕ) = j then 1 else f j) n,
    ← Finset.prod_erase_mul _ _ (Finset.mem_univ i), if_pos rfl, mul_one]
  apply Finset.prod_congr rfl
  intro j member
  rw [if_neg fun same => (Finset.mem_erase.mp member).1 (Fin.ext same.symm)]

/-- Lagrange interpolation through `0, …, n - 1`, as the replay computes it. -/
theorem interpolate_eval_range (n : ℕ) (r : Fin n → L) (c : L) :
    (Lagrange.interpolate Finset.univ (fun i : Fin n => ((i : ℕ) : L)) r).eval c =
      ∑ i : Fin n, r i * ((∏ j ∈ Finset.range n, if (i : ℕ) = j then 1 else c - j) *
        (∏ j ∈ Finset.range n, if (i : ℕ) = j then 1 else ((i : ℕ) : L) - j)⁻¹) := by
  rw [Lagrange.interpolate_apply, Polynomial.eval_finsetSum]
  apply Finset.sum_congr rfl
  intro i _
  rw [eval_mul, eval_C, Lagrange.basis, Polynomial.eval_prod]
  congr 1
  simp only [Lagrange.basisDivisor, eval_mul, eval_C, eval_sub, eval_X]
  rw [Finset.prod_mul_distrib, Finset.prod_inv_distrib, mul_comm, prod_erase_range n i (fun j => c - j),
    prod_erase_range n i (fun j => ((i : ℕ) : L) - j)]

theorem prod_range_list (n : ℕ) (f : ℕ → L) : ∏ i ∈ Finset.range n, f i = ((List.range n).map f).prod := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.prod_range_succ, ih, List.range_succ, List.map_append, List.prod_append]; simp

end Lagrange

theorem toK_interpolate (values : Array Ext) (x : Ext) :
    (interpolate values x).toK (K := K) = ∑ i ∈ Finset.range values.size, (values[i]!).toK (K := K) *
      ((∏ j ∈ Finset.range values.size, if i = j then (1 : K) else x.toK (K := K) - j) *
        (∏ j ∈ Finset.range values.size, if i = j then (1 : K) else ((i : ℕ) : K) - j)⁻¹) := by
  rw [interpolate, toK_sumRange]
  apply Finset.sum_congr rfl
  intro i _
  rw [Ext.toK_mul, Ext.toK_scale, toK_prodRange, Word.toF_inv, map_inv₀, toF_prodRangeGl, map_prod]
  congr 2
  · apply Finset.prod_congr rfl
    intro j _
    split_ifs
    · exact Ext.toK_one
    · rw [Ext.toK_sub, Ext.toK_ofGl, Word.toF_ofNat, map_natCast]
  · congr 1
    apply Finset.prod_congr rfl
    intro j _
    split_ifs
    · rw [Word.toF_one, map_one]
    · rw [Word.toF_sub, Word.toF_ofNat, Word.toF_ofNat, map_sub, map_natCast, map_natCast]

theorem roundValues_size (d : ℕ) (m : Array Ext) (size : m.size = d + 1) (claim : Ext) :
    (Replay.roundValues m claim).size = d + 2 := by
  simp only [Replay.roundValues, Array.size_append, List.size_toArray, List.length_cons, List.length_nil,
    Array.size_extract, size]
  omega

theorem roundValues_get (d : ℕ) (m : Array Ext) (size : m.size = d + 1) (claim : Ext) (i : Fin (d + 2)) :
    ((Replay.roundValues m claim)[(i : ℕ)]!).toK (K := K) =
      Lib.roundValues d claim.toK (fun j => (m[(j : ℕ)]!).toK) i := by
  obtain ⟨i, hi⟩ := i
  rw [getElem!_pos (Replay.roundValues m claim) i (by rw [roundValues_size d m size]; exact hi)]
  simp only [Lib.roundValues]
  rcases (show i = 0 ∨ i = 1 ∨ 2 ≤ i by omega) with rfl | rfl | two
  · simp [Replay.roundValues]
  · simp [Replay.roundValues, Ext.toK_sub]
  · rw [dif_neg (by omega), if_neg (by omega)]
    simp only [Replay.roundValues]
    rw [Array.getElem_append_right (by simp; omega)]
    simp only [List.size_toArray, List.length_cons, List.length_nil, Nat.reduceAdd, Array.getElem_extract]
    rw [getElem!_pos m (i - 1) (by omega)]
    congr 2
    omega

/-- The replay's next claim is the model's round polynomial at the challenge. -/
theorem toK_round (d : ℕ) (m : Array Ext) (size : m.size = d + 1) (claim r : Ext) :
    (interpolate (Replay.roundValues m claim) r).toK (K := K) =
      (roundPoly d claim.toK (fun j => (m[(j : ℕ)]!).toK)).eval r.toK := by
  rw [toK_interpolate, roundPoly, show (node (K := K) d) = fun i : Fin (d + 2) => ((i : ℕ) : K) from rfl,
    interpolate_eval_range, roundValues_size d m size,
    ← Fin.sum_univ_eq_sum_range (fun i => ((Replay.roundValues m claim)[i]!).toK (K := K) *
      ((∏ j ∈ Finset.range (d + 2), if i = j then (1 : K) else r.toK (K := K) - j) *
        (∏ j ∈ Finset.range (d + 2), if i = j then (1 : K) else ((i : ℕ) : K) - j)⁻¹)) (d + 2)]
  apply Finset.sum_congr rfl
  intro i _
  rw [roundValues_get d m size claim i]

/-! ## Sum-check -/

theorem drawn_ext (q : ℕ) (v : Ext) : drawn (K := K) q (.ext v) = ⟨K, v.toK⟩ := rfl

theorem drawn_pair (q : ℕ) (a b : Ext) : drawn (K := K) q (.pair a b) = ⟨K × K, (a.toK, b.toK)⟩ := rfl

/-- The model's messages of the replay's sum-check rounds: each round's
values in `K`. -/
def sumcheckMsgs (d : ℕ) (rounds : List (Array Ext)) : List (Σ M : Type, M) :=
  rounds.map fun m => ⟨Fin (d + 1) → K, fun j => (m[(j : ℕ)]!).toK⟩

section SumCheck

variable [DecidableEq K]

theorem sumcheck_reaches {W : Type} (worlds : Finset W) (d : ℕ) (round : W → List K → K[X])
    (extra : List K → Finset K) (q : ℕ) :
    ∀ (rounds : List (Array Ext)) (point : Array Ext) (claim : Ext) (t t' : Transcript)
      (result : Array Ext × Ext),
      (∀ m ∈ rounds, m.size = d + 1) → Runs (sumcheckRounds rounds point claim) t result t' →
      ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds ∧
        Game.Reaches (Game.sumcheck worlds d round extra rounds.length (pointK point) claim.toK)
          (sumcheckMsgs (K := K) d rounds) (ds.map (drawn (K := K) q)) (pointK result.1, result.2.toK)
  | [], point, claim, t, t', result, _, run => by
      rw [sumcheckRounds] at run
      obtain ⟨rfl, rfl⟩ := run_pure run
      exact ⟨[], by simp, Game.Reaches.pure _⟩
  | m :: rest, point, claim, t, t', result, sizes, run => by
      rw [sumcheckRounds] at run
      obtain ⟨_, t1, h1, run⟩ := run_bind run
      obtain ⟨r, t2, h2, run⟩ := run_bind run
      obtain ⟨ds, draws, reaches⟩ := sumcheck_reaches worlds d round extra q rest _ _ t2 t' result
        (fun m' member => sizes m' (List.mem_cons_of_mem m member)) run
      refine ⟨.ext r :: ds, by rw [draws, run_sampleExt h2, run_observeExts h1]; simp, ?_⟩
      rw [List.length_cons, Game.sumcheck, List.map_cons, drawn_ext]
      apply Game.Reaches.recv (fun j : Fin (d + 1) => (m[(j : ℕ)]!).toK (K := K))
      apply Game.Reaches.draw r.toK
      rw [← pointK_push, ← toK_round d m (sizes m (by simp)) claim r]
      exact reaches

end SumCheck

theorem sumcheckRounds_size :
    ∀ (rounds : List (Array Ext)) (point : Array Ext) (claim : Ext) (t t' : Replay.Transcript)
      (result : Array Ext × Ext),
      Runs (sumcheckRounds rounds point claim) t result t' → result.1.size = point.size + rounds.length
  | [], point, claim, t, t', result, run => by
      rw [sumcheckRounds] at run
      obtain ⟨rfl, rfl⟩ := run_pure run
      simp
  | m :: rest, point, claim, t, t', result, run => by
      rw [sumcheckRounds] at run
      obtain ⟨_, t1, _, run⟩ := run_bind run
      obtain ⟨r, t2, _, run⟩ := run_bind run
      rw [sumcheckRounds_size rest _ _ t2 t' result run, Array.size_push, List.length_cons]
      omega

/-! ## GKR -/

/-- The model's tree shapes are the replay's. -/
structure ShapesAgree (gs : GkrShape) (shapes : Array TreeShape) : Prop where
  trees : gs.trees = shapes.size
  depth : ∀ t, t < shapes.size → gs.depth t = (shapes[t]!).depth
  factors : ∀ t, t < shapes.size → gs.factors t = (shapes[t]!).factors

/-- The model's GKR claims are the replay's on every tree. -/
structure ClaimsAgree (n : ℕ) (c : Replay.GkrClaims) (mc : Layer1.GkrClaims K) : Prop where
  point : mc.point = pointK c.point
  roots : ∀ t, t < n → mc.roots t = pairK (c.roots[t]!)
  current : ∀ t, t < n → mc.current t = pairK (c.current[t]!)
  leafPoint : ∀ t, t < n → mc.leafPoint t = pointK (c.leafPoint[t]!)
  leafValues : ∀ t, t < n → ∀ j, mc.leafValues t j = (c.leafValue t j).toK
  current_size : c.current.size = n
  leafPoint_size : c.leafPoint.size = n
  leafValues_size : c.leafValues.size = n

section Shapes

variable {gs : GkrShape} {shapes : Array TreeShape}

theorem active_eq (agree : ShapesAgree gs shapes) (k : ℕ) : gs.active k = (activeTrees shapes k).toList := by
  unfold GkrShape.active activeTrees
  rw [agree.trees, Array.toList_filter, Array.toList_range]
  apply List.filter_congr
  intro t member
  rw [agree.depth t (List.mem_range.mp member)]

theorem partFactors_eq (agree : ShapesAgree gs shapes) (k t : ℕ) (inside : t < shapes.size) :
    gs.partFactors k t = Replay.partFactors shapes k t := by
  unfold GkrShape.partFactors Replay.partFactors
  rw [agree.depth t inside, agree.factors t inside]

theorem active_lt (k t : ℕ) (member : t ∈ (activeTrees shapes k).toList) : t < shapes.size := by
  unfold activeTrees at member
  rw [Array.toList_filter, Array.toList_range] at member
  exact List.mem_range.mp (List.mem_filter.mp member).1

theorem foldr_max_congr (l : List ℕ) (f g : ℕ → ℕ) (b : ℕ) (same : ∀ t ∈ l, f t = g t) :
    l.foldr (fun t d => max d (f t)) b = l.foldr (fun t d => max d (g t)) b := by
  induction l with
  | nil => rfl
  | cons t rest ih =>
      simp only [List.foldr_cons]
      rw [ih fun u member => same u (List.mem_cons_of_mem t member), same t List.mem_cons_self]

theorem degree_eq (agree : ShapesAgree gs shapes) (k : ℕ) : gs.degree k = layerDegree (layerFactors shapes k) := by
  unfold GkrShape.degree layerDegree layerFactors
  rw [← Array.foldr_toList, Array.toList_map, List.foldr_map, active_eq agree]
  exact foldr_max_congr _ (fun t => gs.partFactors k t + 2) (fun t => Replay.partFactors shapes k t + 2) 3
    fun t member => by rw [partFactors_eq agree k t (active_lt k t member)]

theorem layers_eq (agree : ShapesAgree gs shapes) : gs.layers = layerCount shapes := by
  unfold GkrShape.layers layerCount
  rw [agree.trees]
  exact foldr_max_congr _ gs.depth (fun t => (shapes[t]!).depth) 0
    fun t member => agree.depth t (List.mem_range.mp member)

end Shapes

/-! ### Claims -/

/-- The model's children message: tree `t`'s part pairs from the replay. -/
def childrenOf (pairs : List (ℕ × Array Ext)) (t j : ℕ) : K × K :=
  match pairs.find? (·.1 == t) with
  | some pair => ((pair.2[2 * j]!).toK, (pair.2[2 * j + 1]!).toK)
  | none => (0, 0)

theorem find_of_mem {pairs : List (ℕ × Array Ext)} (nodup : (pairs.map Prod.fst).Nodup) {t : ℕ} {v : Array Ext}
    (member : (t, v) ∈ pairs) : pairs.find? (·.1 == t) = some (t, v) := by
  induction pairs with
  | nil => cases member
  | cons p rest ih =>
      obtain ⟨fresh, nodup⟩ := List.nodup_cons.mp nodup
      rcases List.mem_cons.mp member with same | inside
      · subst same
        simp
      · have ne : p.1 ≠ t := fun same => fresh (same ▸ List.mem_map_of_mem inside)
        rw [List.find?_cons_of_neg (by simpa using ne)]
        exact ih nodup inside

theorem childrenOf_mem {pairs : List (ℕ × Array Ext)} (nodup : (pairs.map Prod.fst).Nodup) {t : ℕ}
    {v : Array Ext} (member : (t, v) ∈ pairs) (j : ℕ) :
    childrenOf (K := K) pairs t j = ((v[2 * j]!).toK, (v[2 * j + 1]!).toK) := by
  rw [childrenOf, find_of_mem nodup member]

theorem toK_reducedOf (pairs : List (ℕ × Array Ext)) (even : ∀ p ∈ pairs, 2 ∣ p.2.size) (tau : Ext) (t j : ℕ) :
    ((reducedOf pairs tau t)[j]!).toK (K := K) = Layer1.reduce (childrenOf pairs t j) tau.toK := by
  unfold reducedOf childrenOf
  cases found : pairs.find? (·.1 == t) with
  | none =>
      simp only
      rw [getElem!_neg #[] j (by simp), toK_default]
      simp [Layer1.reduce]
  | some pair =>
      simp only
      obtain ⟨half, size⟩ := even pair (List.mem_of_find?_eq_some found)
      by_cases inside : j < pair.2.size / 2
      · rw [getElem!_pos _ j (by simpa using inside)]
        simp only [Array.getElem_map, Array.getElem_range, Replay.reduce, Layer1.reduce, Ext.toK_add, Ext.toK_mul,
          Ext.toK_sub]
      · rw [getElem!_neg _ j (by simpa using inside), toK_default,
          getElem!_neg pair.2 (2 * j) (by omega), getElem!_neg pair.2 (2 * j + 1) (by omega), toK_default]
        simp [Layer1.reduce]

theorem toK_combineVals (values : Array Ext) (mu : Ext) (f : ℕ) (size : values.size = 2 * (f + 1)) :
    (Replay.combineVals values mu).toK (K := K) =
      Layer1.combineVals f (fun j => ((values[2 * j]!).toK, (values[2 * j + 1]!).toK)) mu.toK := by
  have half : values.size / 2 - 1 = f := by omega
  simp only [Replay.combineVals, half, Layer1.combineVals, Ext.toK_add, Ext.toK_mul, toK_prodRange,
    prod_range_list]

theorem toK_batchClaim (cur : Array (Ext × Ext)) (mu nu : Ext) (mc : Layer1.GkrClaims K) (n : ℕ)
    (agree : ∀ t, t < n → mc.current t = pairK (cur[t]!)) :
    ∀ l : List (ℕ × Array Ext), (∀ p ∈ l, p.1 < n) →
      (batchClaim cur mu nu l).toK (K := K) =
        wsum nu.toK (fun t => (mc.current t).1 + mu.toK * (mc.current t).2) (l.map Prod.fst)
  | [], _ => by simp [batchClaim, wsum]
  | (t, v) :: rest, small => by
      rw [batchClaim, List.map_cons, wsum, Ext.toK_add, Ext.toK_add, Ext.toK_mul, Ext.toK_mul,
        toK_batchClaim cur mu nu mc n agree rest fun p member => small p (List.mem_cons_of_mem _ member),
        agree t (small (t, v) List.mem_cons_self)]
      rfl

theorem toK_batchExpected (mu nu : Ext) (pairs : List (ℕ × Array Ext)) (nodup : (pairs.map Prod.fst).Nodup)
    (f : ℕ → ℕ) (sizes : ∀ p ∈ pairs, p.2.size = 2 * (f p.1 + 1)) :
    ∀ l : List (ℕ × Array Ext), (∀ p ∈ l, p ∈ pairs) →
      (batchExpected mu nu l).toK (K := K) =
        wsum nu.toK (fun t => Layer1.combineVals (f t) (childrenOf pairs t) mu.toK) (l.map Prod.fst)
  | [], _ => by simp [batchExpected, wsum]
  | (t, v) :: rest, inside => by
      have member := inside (t, v) List.mem_cons_self
      rw [batchExpected, List.map_cons, wsum, Ext.toK_add, Ext.toK_mul,
        toK_batchExpected mu nu pairs nodup f sizes rest fun p m => inside p (List.mem_cons_of_mem _ m),
        toK_combineVals v mu (f t) (sizes _ member)]
      congr 2
      funext j
      rw [childrenOf_mem nodup member]

theorem advance_agrees {gs : GkrShape} {shapes : Array TreeShape} (agree : ShapesAgree gs shapes) (k : ℕ)
    {c : Replay.GkrClaims} {mc : Layer1.GkrClaims K} (claims : ClaimsAgree shapes.size c mc)
    (pairs : List (ℕ × Array Ext)) (even : ∀ p ∈ pairs, 2 ∣ p.2.size) (next : Array Ext) (tau : Ext) :
    ClaimsAgree shapes.size (Replay.advance shapes k c pairs next tau)
      (Layer1.advance gs k mc (pointK next) (childrenOf pairs) tau.toK) where
  point := by simp [Replay.advance, Layer1.advance, pointK_push]
  roots t inside := claims.roots t inside
  current t inside := by
    simp only [Replay.advance, Layer1.advance]
    rw [getElem!_map_range _ _ (by rw [claims.current_size]; exact inside), agree.depth t inside]
    split_ifs
    · simp [pairK, toK_reducedOf pairs even]
    · exact claims.current t inside
  leafPoint t inside := by
    simp only [Replay.advance, Layer1.advance]
    rw [getElem!_map_range _ _ (by rw [claims.leafPoint_size]; exact inside), agree.depth t inside]
    split_ifs
    · rw [pointK_push]
    · exact claims.leafPoint t inside
  leafValues t inside j := by
    simp only [Replay.advance, Layer1.advance, GkrClaims.leafValue]
    rw [getElem!_map_range _ _ (by rw [claims.leafValues_size]; exact inside), agree.depth t inside]
    split_ifs
    · rw [toK_reducedOf pairs even]
    · exact claims.leafValues t inside j
  current_size := by simp [Replay.advance, claims.current_size]
  leafPoint_size := by simp [Replay.advance, claims.leafPoint_size]
  leafValues_size := by simp [Replay.advance, claims.leafValues_size]

/-! ### The game -/

/-- The model's messages of the replay's GKR layers from layer `k`: each
layer's sum-check rounds, then its children. -/
def gkrLayersMsgs (shapes : Array TreeShape) : List GkrLayer → ℕ → List (Σ M : Type, M)
  | [], _ => []
  | layer :: rest, k =>
      sumcheckMsgs (K := K) (layerDegree (layerFactors shapes k) - 1) layer.rounds.toList ++
        ⟨ℕ → ℕ → K × K, childrenOf ((activeTrees shapes k).zip layer.children).toList⟩ ::
          gkrLayersMsgs shapes rest (k + 1)

/-- The model's messages of the replay's GKR proof: the roots, then every layer. -/
def gkrMsgs (shapes : Array TreeShape) (proof : GkrProof) : List (Σ M : Type, M) :=
  ⟨ℕ → K × K, fun t => pairK (proof.roots[t]!)⟩ :: gkrLayersMsgs (K := K) shapes proof.layers.toList 1

section Games

variable [DecidableEq K]

theorem readGkr_ok {proof : GkrProof} {shapes : Array TreeShape} (read : readGkr proof shapes = .ok ()) :
    proof.roots.size = shapes.size ∧ proof.layers.size = layerCount shapes ∧
      ∀ i (h : i < proof.layers.size), layerShapeOk shapes (i + 1) proof.layers[i] = true := by
  unfold readGkr at read
  simp only [] at read
  split_ifs at read with bad
  · obtain ⟨_, thrown, _⟩ := Except.bind_ok read
    cases thrown
  · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at bad
    obtain ⟨⟨_, roots⟩, layers⟩ := bad
    have all := Except.ensure_ok read
    rw [List.all_eq_true] at all
    refine ⟨roots, layers, fun i inside => ?_⟩
    have := all i (List.mem_range.mpr (by omega))
    rwa [getElem!_pos proof.layers i inside] at this

theorem gkrLayers_reaches {gs : GkrShape} {shapes : Array TreeShape} (agree : ShapesAgree gs shapes)
    {W : Type} (worlds : Finset W) (batch : W → Batch K) (extra : ℕ → List K → Finset K) (q : ℕ) :
    ∀ (layers : List GkrLayer) (k : ℕ) (c : Replay.GkrClaims) (mc : Layer1.GkrClaims K)
      (t t' : Replay.Transcript) (c' : Replay.GkrClaims),
      1 ≤ k → (∀ i (h : i < layers.length), layerShapeOk shapes (k + i) layers[i] = true) →
      ClaimsAgree shapes.size c mc → c.point.size = k - 1 →
      (∀ u, u < shapes.size → (shapes[u]!).depth < k → (c.leafPoint[u]!).size = (shapes[u]!).depth) →
      Runs (Replay.gkrLayers shapes k layers c) t c' t' →
      ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds ∧ ∃ mc', ClaimsAgree shapes.size c' mc' ∧
        (∀ u, u < shapes.size → (shapes[u]!).depth < k + layers.length →
          (c'.leafPoint[u]!).size = (shapes[u]!).depth) ∧
        Game.Reaches (Layer1.gkrLayers gs worlds batch extra layers.length k mc)
          (gkrLayersMsgs (K := K) shapes layers k) (ds.map (drawn (K := K) q)) (some mc')
  | [], k, c, mc, t, t', c', _, _, claims, _, lengths, run => by
      rw [Replay.gkrLayers] at run
      obtain ⟨rfl, rfl⟩ := run_pure run
      refine ⟨[], by simp, mc, claims, fun u inside below => lengths u inside (by simpa using below), ?_⟩
      rw [List.length_nil, Layer1.gkrLayers]
      exact Game.Reaches.pure _
  | layer :: rest, k, c, mc, t, t', c', positive, shapesOk, claims, pointSize, lengths, run => by
      have ok := shapesOk 0 (by simp)
      simp only [List.getElem_cons_zero, Nat.add_zero, layerShapeOk, Bool.and_eq_true, beq_iff_eq] at ok
      obtain ⟨⟨⟨roundsSize, roundsDegree⟩, childrenSize⟩, childSizes⟩ := ok
      rw [← Array.all_toList, List.all_eq_true] at roundsDegree childSizes
      rw [Replay.gkrLayers] at run
      obtain ⟨mu, t1, h1, run⟩ := run_bind run
      obtain ⟨nu, t2, h2, run⟩ := run_bind run
      obtain ⟨result, t3, h3, run⟩ := run_bind run
      obtain ⟨_, t4, h4, run⟩ := run_bind run
      obtain ⟨_, t5, h5, run⟩ := run_bind run
      obtain ⟨tau, t6, h6, run⟩ := run_bind run
      obtain ⟨check, same⟩ := run_ensure h4
      rw [same] at h5
      set pairs := ((activeTrees shapes k).zip layer.children).toList with pairs_def
      have firsts : pairs.map Prod.fst = gs.active k := by
        rw [pairs_def, Array.toList_zip, List.map_fst_zip (by simp [childrenSize]), active_eq agree]
      have nodup : (pairs.map Prod.fst).Nodup := by
        rw [firsts, active_eq agree, activeTrees, Array.toList_filter, Array.toList_range]
        exact List.nodup_range.filter _
      have small : ∀ p ∈ pairs, p.1 < shapes.size := by
        intro p member
        rw [pairs_def, Array.toList_zip] at member
        exact active_lt k p.1 (List.of_mem_zip member).1
      have sizes : ∀ p ∈ pairs, p.2.size = 2 * (Replay.partFactors shapes k p.1 + 1) := by
        intro p member
        have := childSizes p member
        simpa using this
      have even : ∀ p ∈ pairs, 2 ∣ p.2.size := fun p member => ⟨_, sizes p member⟩
      obtain ⟨dsS, drawsS, reachS⟩ := sumcheck_reaches (K := K) worlds (gs.degree k - 1)
        (fun w => combRound (k - 1) (layerTables gs (batch w) k mc.point)
          (layerComb gs k mu.toK nu.toK)) (extra k) q layer.rounds.toList #[]
        (batchClaim c.current mu nu pairs) t2 t3 result
        (fun m member => by
          have := roundsDegree m member
          simp only [beq_iff_eq] at this
          rw [this, ← degree_eq agree k]
          have := gs.le_degree k
          omega) h3
      have nextSize : result.1.size = k - 1 := by
        rw [sumcheckRounds_size _ _ _ _ _ _ h3, Array.length_toList, roundsSize]
        simp
      obtain ⟨dsR, drawsR, mc', claims', lengths', reachR⟩ := gkrLayers_reaches agree worlds batch extra q rest
        (k + 1) (Replay.advance shapes k c pairs result.1 tau)
        (Layer1.advance gs k mc (pointK result.1) (childrenOf pairs) tau.toK)
        t6 t' c' (by omega) (fun i inside => by
          have := shapesOk (i + 1) (by simp; omega)
          simpa [Nat.add_assoc, Nat.add_comm 1 i] using this)
        (advance_agrees agree k claims pairs even result.1 tau)
        (by simp only [Replay.advance, Array.size_push, nextSize]; omega)
        (fun u inside below => by
          simp only [Replay.advance]
          rw [getElem!_map_range _ _ (by rw [claims.leafPoint_size]; exact inside)]
          split_ifs with deep
          · rw [Array.size_push, nextSize, deep]
            have := pointSize
            omega
          · exact lengths u inside (by omega))
        run
      refine ⟨.ext mu :: .ext nu :: (dsS ++ .ext tau :: dsR), ?_, mc', claims',
        fun u inside below => lengths' u inside (by simp at below; omega), ?_⟩
      · rw [drawsR, run_sampleExt h6, run_observeExts h5, drawsS, run_sampleExt h2, run_sampleExt h1]
        simp
      · rw [List.length_cons, Layer1.gkrLayers]
        simp only [List.map_cons, List.map_append, drawn_ext]
        apply Game.Reaches.draw mu.toK
        apply Game.Reaches.draw nu.toK
        have claim0 : (batchClaim c.current mu nu pairs).toK (K := K) =
            wsum nu.toK (fun t => (mc.current t).1 + mu.toK * (mc.current t).2) (gs.active k) := by
          rw [toK_batchClaim c.current mu nu mc shapes.size claims.current pairs small, firsts]
        rw [Array.length_toList, roundsSize, claim0, show pointK (K := K) #[] = [] from rfl] at reachS
        rw [gkrLayersMsgs, ← degree_eq agree k]
        refine Game.Reaches.bind reachS ?_
        apply Game.Reaches.recv (childrenOf pairs)
        rw [if_pos]
        · apply Game.Reaches.draw tau.toK
          exact reachR
        · dsimp only
          rw [Ext.toK_of_beq check, Ext.toK_mul, toK_eqEval, ← claims.point,
            toK_batchExpected mu nu pairs nodup (Replay.partFactors shapes k) sizes pairs (fun _ m => m), firsts]
          refine congrArg (fun x => Lib.eqEval mc.point (pointK result.1) * x) (wsum_congr _ fun t member => ?_)
          rw [partFactors_eq agree k t (active_lt k t (by rwa [← active_eq agree]))]

/-- A successful GKR replay is a path of the model's GKR that receives the
proof's messages and ends in claims that agree with the replay's on every
tree. -/
theorem gkr_reaches {gs : GkrShape} {shapes : Array TreeShape} (agree : ShapesAgree gs shapes)
    {W : Type} (worlds : Finset W) (batch : W → Batch K) (extra : ℕ → List K → Finset K) (q : ℕ)
    {proof : GkrProof} (read : readGkr proof shapes = .ok ()) {t t' : Replay.Transcript} {c : Replay.GkrClaims}
    (run : Runs (gkrVerify proof shapes) t c t') :
    ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds ∧ ∃ mc, ClaimsAgree shapes.size c mc ∧
      (∀ u, u < shapes.size → (c.leafPoint[u]!).size = (shapes[u]!).depth) ∧
      Game.Reaches (Layer1.gkr gs worlds batch extra) (gkrMsgs (K := K) shapes proof) (ds.map (drawn (K := K) q))
        (some mc) := by
  obtain ⟨roots, layers, layersOk⟩ := readGkr_ok read
  unfold gkrVerify at run
  obtain ⟨_, t1, h1, run⟩ := run_bind run
  have start : ClaimsAgree shapes.size
      { roots := proof.roots, point := #[], current := proof.roots,
        leafPoint := shapes.map fun _ => #[], leafValues := shapes.map fun _ => #[] }
      ({ roots := fun t => pairK (proof.roots[t]!), point := [], current := fun t => pairK (proof.roots[t]!),
         leafPoint := fun _ => [], leafValues := fun _ _ => 0 } : Layer1.GkrClaims K) :=
    { point := rfl
      roots := fun _ _ => rfl
      current := fun _ _ => rfl
      leafPoint := fun t inside => by
        show [] = pointK _
        rw [getElem!_pos _ t (by simpa using inside)]
        simp [pointK]
      leafValues := fun t inside j => by
        show 0 = ((((shapes.map fun _ => (#[] : Array Ext))[t]!))[j]!).toK
        rw [getElem!_pos _ t (by simpa using inside)]
        simp only [Array.getElem_map]
        rw [getElem!_neg #[] j (by simp), toK_default]
      current_size := roots
      leafPoint_size := by simp
      leafValues_size := by simp }
  obtain ⟨ds, draws, mc, claims, lengths, reaches⟩ := gkrLayers_reaches agree worlds batch extra q
    proof.layers.toList 1 _ _ t1 t' c le_rfl (fun i inside => by
      rw [Nat.add_comm]
      have := layersOk i (by simpa using inside)
      simpa using this) start rfl (fun u inside below => by
        rw [getElem!_pos _ u (by simpa using inside)]
        simp only [Array.getElem_map, List.size_toArray, List.length_nil]
        omega) run
  refine ⟨ds, by rw [draws, run_observeExts h1], mc, claims, fun u inside => lengths u inside ?_, ?_⟩
  · rw [Array.length_toList, layers, ← layers_eq agree, ← agree.depth u inside]
    have := gs.depth_le_layers (show u < gs.trees by rw [agree.trees]; exact inside)
    omega
  apply Game.Reaches.recv (fun t => pairK (proof.roots[t]!))
  rw [layers_eq agree, ← layers, ← Array.length_toList]
  exact reaches

end Games

end CompressionLean.Bridge

/-! ## The replays alone -/

namespace CompressionLean.Bridge

open CompressionLean.Replay

/-- One replay of `rounds` is a replay of its first `k` rounds, then a replay of the rest from that
claim (`verifier.rs:178-180`). -/
theorem runs_sumcheckReplay_split {rounds : Array (Array Ext)} {claim : Ext} (k : ℕ)
    {t t' : Transcript} {point : Array Ext} {final : Ext}
    (run : Runs (sumcheckReplay rounds claim) t (point, final) t') :
    ∃ first middle t₁ second, Runs (sumcheckReplay (rounds.extract 0 k) claim) t (first, middle) t₁ ∧
      Runs (sumcheckReplay (rounds.extract k) middle) t₁ (second, final) t' ∧ point = first ++ second ∧
      first.size = min k rounds.size ∧ second.size = rounds.size - k := by
  have split : rounds.toList = (rounds.extract 0 k).toList ++ (rounds.extract k).toList := by
    simp only [Array.toList_extract, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero]
    rw [List.take_of_length_le (l := List.drop k rounds.toList) (by simp)]
    simp
  unfold sumcheckReplay at run
  rw [split, sumcheckRounds_append] at run
  obtain ⟨⟨first, middle⟩, t₁, h₁, h₂⟩ := run_bind run
  rw [sumcheckRounds_point] at h₂
  obtain ⟨⟨second, final'⟩, h₂, e⟩ := run_map h₂
  simp only [Prod.mk.injEq] at e
  obtain ⟨rfl, rfl⟩ := e
  refine ⟨first, middle, t₁, second, h₁, h₂, rfl, ?_, ?_⟩
  · rw [Bridge.sumcheckRounds_size _ _ _ _ _ _ h₁]
    simp
  · rw [Bridge.sumcheckRounds_size _ _ _ _ _ _ h₂]
    simp

/-- Layers `l₁ ++ l₂` replay `l₁`, then `l₂` from its claims (`gkr.rs:694`). -/
theorem gkrLayers_append (shapes : Array TreeShape) (l₁ l₂ : List GkrLayer) (k : ℕ) (claims : GkrClaims) :
    gkrLayers shapes k (l₁ ++ l₂) claims =
      gkrLayers shapes k l₁ claims >>= fun c => gkrLayers shapes (k + l₁.length) l₂ c := by
  induction l₁ generalizing k claims with
  | nil => simp [gkrLayers]
  | cons layer rest ih =>
      simp only [List.cons_append, gkrLayers, ih, bind_assoc, List.length_cons]
      rw [show k + (rest.length + 1) = k + 1 + rest.length by omega]

/-- The first `j + 1` layers run as the first `j`, then layer `j` (`gkr.rs:694`). -/
theorem runs_gkrLayers_succ {shapes : Array TreeShape} {layers : Array GkrLayer} {j : ℕ}
    (hj : j < layers.size) {init c c' : GkrClaims} {t₀ t t' : Transcript}
    (pre : Runs (gkrLayers shapes 1 (layers.toList.take j) init) t₀ c t)
    (step : Runs (gkrLayers shapes (j + 1) [layers[j]!] c) t c' t') :
    Runs (gkrLayers shapes 1 (layers.toList.take (j + 1)) init) t₀ c' t' := by
  have e : layers.toList.take (j + 1) = layers.toList.take j ++ [layers[j]!] := by
    rw [List.take_add_one, List.getElem?_eq_getElem (by simpa using hj), Option.toList_some, Array.getElem_toList,
      ← getElem!_pos layers j hj]
  rw [e, gkrLayers_append, List.length_take, Array.length_toList, Nat.min_eq_left hj.le, Nat.add_comm 1 j]
  exact runs_bind pre step

/-- One layer of `verify` (`gkr.rs:694-746`): draw `μ` and `ν`, replay the
sum-check, check its last claim, bind the children and draw `τ`. -/
theorem runs_gkrLayer {shapes : Array TreeShape} {k : ℕ} {layer : GkrLayer} {claims : GkrClaims}
    {t tmu tnu ts tobs t' : Transcript} {mu nu tau final : Ext} {point : Array Ext}
    (hmu : Runs sampleExt t mu tmu) (hnu : Runs sampleExt tmu nu tnu)
    (sum : Runs (sumcheckReplay layer.rounds
      (batchClaim claims.current mu nu ((activeTrees shapes k).zip layer.children).toList)) tnu (point, final) ts)
    (check : final = eqEval claims.point point * batchExpected mu nu ((activeTrees shapes k).zip layer.children).toList)
    (hobs : Runs (observeExts layer.children.flatten) ts () tobs) (htau : Runs sampleExt tobs tau t') :
    Runs (gkrLayers shapes k [layer] claims) t
      (advance shapes k claims ((activeTrees shapes k).zip layer.children).toList point tau) t' := by
  rw [gkrLayers]
  exact runs_bind hmu (runs_bind hnu (runs_bind sum (runs_bind (runs_ensure (by rw [← check]; exact Ext.beq_self _) _ _)
    (runs_bind hobs (runs_bind htau (runs_pure _ _))))))

/-- Every tree ends by the last layer (`gkr.rs:749`). -/
theorem depth_le_layerCount (shapes : Array TreeShape) {t : ℕ} (inside : t < shapes.size) :
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
theorem three_le_layerDegree (factors : Array ℕ) : 3 ≤ layerDegree factors := by
  unfold layerDegree
  rw [← Array.foldr_toList]
  induction factors.toList with
  | nil => exact le_rfl
  | cons f rest ih => exact ih.trans (le_max_left _ _)

/-- Every tree has depth at least one (`GkrView::read`, `gkr.rs:608-656`). -/
theorem readGkr_depth {proof : GkrProof} {shapes : Array TreeShape} (read : readGkr proof shapes = .ok ())
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
theorem layerShapeOk_facts {shapes : Array TreeShape} {k : ℕ} {layer : GkrLayer}
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

end CompressionLean.Bridge
