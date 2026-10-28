import CompressionLean.Bridge.Gkr
import CompressionLean.Bridge.Relation

/-!
Owns the bridge of the early stage (`Replay.earlyStage`) to the model's
`earlyGame`. A successful early replay is a path of `earlyGame`: it receives
the P0 candidates and the histogram, draws the replay's `β`, `(β, γ)`,
`(β, μ)` and GKR challenges, receives the early values, and ends in the
model's `EarlyOut` of the replay's values, whose `EarlyChecks` the replay's
checks give (`early_reaches`).
-/

namespace CompressionLean.Bridge

open Polynomial
open CompressionLean.Replay CompressionLean.Layer1

variable {K : Type} [Field K] [Algebra Fp K] [Fintype K] [ChallengeField K]

local notation "ι" => algebraMap Fp K

/-! ## Checks -/

theorem sum_range_list {M : Type} [AddCommMonoid M] (n : ℕ) (f : ℕ → M) :
    ∑ i ∈ Finset.range n, f i = ((List.range n).map f).sum := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ, ih, List.range_succ, List.map_append, List.sum_append]; simp

theorem toK_slot (bound slot : ℕ) :
    (Ext.ofGl (Gl.ofNat slot - Gl.ofNat bound)).toK (K := K) = slotValue bound slot := by
  rw [Ext.toK_ofGl, Gl.toF_sub, Gl.toF_ofNat, Gl.toF_ofNat, map_sub, map_natCast, map_natCast]
  rfl

theorem tableSum_ok (bound : ℕ) (counts : Array Gl) (beta : Ext) {total : Ext}
    (ok : tableSum bound counts beta = .ok total) :
    (∀ t, t < counts.size → beta.toK (K := K) ≠ slotValue bound t) ∧
      total.toK (K := K) =
        ∑ t ∈ Finset.range counts.size, ι (counts[t]!).toF / (beta.toK (K := K) - slotValue bound t) := by
  have loop : ∀ (l : List ℕ) (init total : Ext), l.foldlM (tableSumStep bound counts beta) init = .ok total →
      (∀ t ∈ l, beta.toK (K := K) ≠ slotValue bound t) ∧
        total.toK (K := K) =
          init.toK + (l.map fun t => ι (counts[t]!).toF / (beta.toK (K := K) - slotValue bound t)).sum := by
    intro l
    induction l with
    | nil =>
        intro init total h
        cases h
        simp
    | cons slot rest ih =>
        intro init total h
        rw [List.foldlM_cons] at h
        obtain ⟨mid, step, h⟩ := Except.bind_ok h
        unfold tableSumStep at step
        obtain ⟨_, nonzero, step⟩ := Except.bind_ok step
        obtain rfl := Except.pure_eq_ok step
        have pole := Ext.toK_ne_zero (K := K) (Except.ensure_ok nonzero)
        rw [Ext.toK_sub, toK_slot] at pole
        obtain ⟨poles, value⟩ := ih _ _ h
        refine ⟨fun t member => ?_, ?_⟩
        · rcases List.mem_cons.mp member with rfl | inside
          · exact sub_ne_zero.mp pole
          · exact poles t inside
        · rw [value, Ext.toK_add, Ext.toK_scale, Ext.toK_inv, Ext.toK_sub, toK_slot, List.map_cons,
            List.sum_cons, div_eq_inv_mul]
          ring
  obtain ⟨poles, value⟩ := loop _ 0 total ok
  refine ⟨fun t inside => poles t (List.mem_range.mpr inside), ?_⟩
  rw [value, Ext.toK_zero, zero_add, sum_range_list]

theorem pairCheck_ok {x y : Ext × Ext} {name : String} (ok : pairCheck x y name = .ok ()) :
    PairCheck (pairK (K := K) x) (pairK y) := by
  unfold pairCheck at ok
  obtain ⟨_, h1, ok⟩ := Except.bind_ok ok
  obtain ⟨_, h2, h3⟩ := Except.bind_ok ok
  refine ⟨Ext.toK_ne_zero (Except.ensure_ok h1), Ext.toK_ne_zero (Except.ensure_ok h2), ?_⟩
  simp only [pairK]
  rw [← Ext.toK_mul, ← Ext.toK_mul]
  exact Ext.toK_of_beq (Except.ensure_ok h3)

theorem toK_tuple (g r e₁ e₂ : Ext) :
    (Replay.tuple g r e₁ e₂).toK (K := K) = Layer1.tuple g.toK r.toK e₁.toK e₂.toK := by
  simp [Replay.tuple, Layer1.tuple, Ext.toK_add, Ext.toK_mul]

/-- The openings `checkEarly` returns. -/
def earlyOpeningsOf (layout : Layout) (ch : Replay.EarlyChallenges) (claims : Replay.GkrClaims)
    (v : Replay.EarlyValues) : EarlyOpenings where
  runPoint := claims.leafPoint[3]!
  pairPoint := (claims.leafPoint[4]!).extract 0 layout.slotVariables
  rowPoint := claims.leafPoint[2]!
  mult := claims.leafValue 2 0
  rowClaim := v.row
  slotClaim := ch.scatter.1 - claims.leafValue 3 2 -
    (matrixStep layout claims.leafPoint[3]!).scale (Gl.ofNat (2 ^ layout.slotVariables))
  coefClaim := claims.leafValue 3 0

/-- What `checkEarly` checked and returned. -/
structure EarlyChecked (layout : Layout) (r : Array (Gl × Gl)) (ch : Replay.EarlyChallenges)
    (claims : Replay.GkrClaims) (v : Replay.EarlyValues) (op : EarlyOpenings) : Prop where
  rows : pairCheck claims.roots[1]! claims.roots[2]! "row lookup" = .ok ()
  slots : pairCheck claims.roots[3]! claims.roots[4]! "slot scatter" = .ok ()
  runLeaf : claims.leafValue 1 0 = ch.lookup.1 - Replay.tuple ch.lookup.2 v.row v.eRe v.eIm
  rowLeaf : claims.leafValue 2 1 = ch.lookup.1 - Replay.tuple ch.lookup.2 (Replay.identityValue claims.leafPoint[2]!)
    (chi r claims.leafPoint[2]!).re (chi r claims.leafPoint[2]!).im
  scatterLeaf : claims.leafValue 3 1 = v.eRe + ch.scatter.2 * v.eIm
  pairLeaf : claims.leafValue 4 0 =
    sumRange (min v.pairs.size (eqTable (claims.leafPoint[4]!.extract layout.slotVariables claims.leafPoint[4]!.size)).size)
      fun j => (eqTable (claims.leafPoint[4]!.extract layout.slotVariables claims.leafPoint[4]!.size))[j]! *
        ((v.pairs[j]!).1 + ch.scatter.2 * (v.pairs[j]!).2)
  indexLeaf : claims.leafValue 4 1 = ch.scatter.1 - Replay.identityValue claims.leafPoint[4]!
  openings : op = earlyOpeningsOf layout ch claims v

theorem checkEarly_ok {layout : Layout} {r : Array (Gl × Gl)} {ch : Replay.EarlyChallenges}
    {claims : Replay.GkrClaims} {v : Replay.EarlyValues} {op : EarlyOpenings}
    (ok : checkEarly layout r ch claims v = .ok op) : EarlyChecked layout r ch claims v op := by
  unfold checkEarly at ok
  obtain ⟨_, rows, ok⟩ := Except.bind_ok ok
  obtain ⟨_, slots, ok⟩ := Except.bind_ok ok
  obtain ⟨_, runLeaf, ok⟩ := Except.bind_ok ok
  obtain ⟨_, rowLeaf, ok⟩ := Except.bind_ok ok
  obtain ⟨_, scatterLeaf, ok⟩ := Except.bind_ok ok
  obtain ⟨_, pairLeaf, ok⟩ := Except.bind_ok ok
  obtain ⟨_, indexLeaf, ok⟩ := Except.bind_ok ok
  exact ⟨rows, slots, Ext.eq_of_beq (Except.ensure_ok runLeaf), Ext.eq_of_beq (Except.ensure_ok rowLeaf),
    Ext.eq_of_beq (Except.ensure_ok scatterLeaf), Ext.eq_of_beq (Except.ensure_ok pairLeaf),
    Ext.eq_of_beq (Except.ensure_ok indexLeaf), (Except.pure_eq_ok ok).symm⟩

/-! ## The stage -/

/-- The model's early values of the replay's. -/
def valuesOf (v : Replay.EarlyValues) : Layer1.EarlyValues K :=
  ⟨v.eRe.toK, v.eIm.toK, v.row.toK, fun j => pairK (v.pairs[j]!), v.block.toK⟩

/-- The model's histogram of the proof's counts. -/
def histOf (proof : Replay.Proof) (i : ℕ) : Fp := (proof.histogram[i]!).toF

/-- The model's early challenges of the replay's. -/
def challengesOf (ch : Replay.EarlyChallenges) : Layer1.EarlyChallenges K :=
  ⟨ch.beta.toK, pairK ch.lookup, pairK ch.scatter⟩

theorem earlyShapes_size (rel : Replay.Relation) : rel.earlyShapes.size = 5 := rfl

theorem earlyShapes_agree (rel : Replay.Relation) : ShapesAgree (earlyShape (shapeOf rel)) rel.earlyShapes where
  trees := rfl
  depth t inside := by
    rw [earlyShapes_size] at inside
    interval_cases t <;>
      simp [earlyShape, Replay.Relation.earlyShapes, shapeOf, Shape.cubeVars, Replay.Relation.cubeVars, Nat.add_comm]
  factors t inside := by
    rw [earlyShapes_size] at inside
    interval_cases t <;> simp [earlyShape, Replay.Relation.earlyShapes]

theorem pointK_extract_drop (p : Array Ext) (n : ℕ) : pointK (K := K) (p.extract n p.size) = (pointK p).drop n := by
  simp [pointK, Array.toList_extract]

theorem chi_re (st : Replay.Statement) (p : Array Ext) (short : p.size ≤ st.point.size) :
    (chi st.point p).re.toK (K := K) = mle (pointK p) fun i => ι ((statementOf st).chi i).1 := by
  have split := chi_split (K := K) (statementOf st) (pointK p) (by rw [pointK_length, statement_r_length]; exact short)
  rw [statement_point, ← toQ_chi] at split
  exact congrArg QuadraticAlgebra.re split

theorem chi_im (st : Replay.Statement) (p : Array Ext) (short : p.size ≤ st.point.size) :
    (chi st.point p).im.toK (K := K) = mle (pointK p) fun i => ι ((statementOf st).chi i).2 := by
  have split := chi_split (K := K) (statementOf st) (pointK p) (by rw [pointK_length, statement_r_length]; exact short)
  rw [statement_point, ← toQ_chi] at split
  exact congrArg QuadraticAlgebra.im split

theorem toK_identity (p : Array Ext) : (Replay.identityValue p).toK (K := K) = mle (pointK p) fun x => (x : K) := by
  rw [toK_identityValue, mle_identity]

section Stage

variable [DecidableEq K]

/-- A successful early replay is a path of `earlyGame` that draws the
replay's challenges and ends in the model's `EarlyOut` of the replay's
values; along it every `EarlyChecks` condition holds. -/
theorem early_reaches (rel : Replay.Relation) (st : Replay.Statement) (proof : Replay.Proof)
    (shape : ProofShape rel proof) (fits : st.fits rel = true)
    (rowFits : rel.key.layout.rowVariables ≤ rel.pointVariables) (matricesFit : rel.matrices ≤ 2 ^ rel.matrixVars)
    (S : Setup Fp) (L0 : ℕ) (cands : List (P0Tables Fp)) (short : cands.length ≤ L0) (q : ℕ)
    {t t' : Replay.Transcript} {o : Replay.EarlyOut} (run : Runs (earlyStage rel st proof) t o t') :
    ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds ∧ ∃ mc : Layer1.GkrClaims K,
      ClaimsAgree rel.earlyShapes.size o.gkr mc ∧
      (∀ u, u < rel.earlyShapes.size → (o.gkr.leafPoint[u]!).size = (rel.earlyShapes[u]!).depth) ∧
      o.values = proof.earlyValues ∧ o.zLeaf = o.challenges.beta - o.gkr.leafValue 0 0 ∧
      o.openings = earlyOpeningsOf rel.key.layout o.challenges o.gkr proof.earlyValues ∧
      Game.Reaches (earlyGame (shapeOf rel) S (statementOf st) L0) (ds.map (drawn (K := K) q))
        (some ⟨cands, histOf proof, challengesOf o.challenges, mc, valuesOf o.values⟩) := by
  unfold earlyStage at run
  obtain ⟨_, t1, h1, run⟩ := run_bind run
  obtain ⟨_, t2, h2, run⟩ := run_bind run
  obtain ⟨_, t3, h3, run⟩ := run_bind run
  obtain ⟨beta, t4, h4, run⟩ := run_bind run
  obtain ⟨lookup, t5, h5, run⟩ := run_bind run
  obtain ⟨scatter, t6, h6, run⟩ := run_bind run
  obtain ⟨total, t7, h7, run⟩ := run_bind run
  obtain ⟨claims, t8, h8, run⟩ := run_bind run
  obtain ⟨_, t9, h9, run⟩ := run_bind run
  obtain ⟨_, t10, h10, run⟩ := run_bind run
  obtain ⟨openings, t11, h11, run⟩ := run_bind run
  obtain ⟨_, t12, h12, run⟩ := run_bind run
  obtain ⟨rfl, rfl⟩ := run_pure run
  obtain ⟨tableOk, rfl⟩ := run_liftExcept h7
  obtain ⟨nonzero, rfl⟩ := run_ensure h9
  obtain ⟨rootEq, rfl⟩ := run_ensure h10
  obtain ⟨checkOk, rfl⟩ := run_liftExcept h11
  have checked := checkEarly_ok checkOk
  obtain ⟨poles, totalValue⟩ := tableSum_ok (K := K) rel.normBound proof.histogram beta tableOk
  obtain ⟨dsG, drawsG, mc, claimsAgree, lengths, reachG⟩ := gkr_reaches (K := K) (earlyShapes_agree rel)
    (Finset.univ : Finset (Fin cands.length))
    (fun w => earlyBatch (shapeOf rel) S (statementOf st) ⟨beta.toK, pairK lookup, pairK scatter⟩ (cands.get w))
    (earlyExtra (shapeOf rel) S cands) q shape.early h8
  have fitsSizes := fits
  simp only [Replay.Statement.fits, Bool.and_eq_true, beq_iff_eq] at fitsSizes
  have pointSize : st.point.size = rel.pointVariables := fitsSizes.1.1.1.2
  have small : ∀ u, u < 5 → u < rel.earlyShapes.size := fun u lt => by rw [earlyShapes_size]; exact lt
  have rowSize : (claims.leafPoint[2]!).size = rel.key.layout.rowVariables := lengths 2 (small 2 (by norm_num))
  have pairSize : (claims.leafPoint[4]!).size = rel.key.layout.slotVariables + rel.matrixVars :=
    lengths 4 (small 4 (by norm_num))
  have leafValue : ∀ u j, u < 5 → mc.leafValues u j = (claims.leafValue u j).toK :=
    fun u j lt => claimsAgree.leafValues u (small u lt) j
  have leafPoint : ∀ u, u < 5 → mc.leafPoint u = pointK (claims.leafPoint[u]!) :=
    fun u lt => claimsAgree.leafPoint u (small u lt)
  have root : ∀ u, u < 5 → mc.roots u = pairK (claims.roots[u]!) := fun u lt => claimsAgree.roots u (small u lt)
  have checks : EarlyChecks (shapeOf rel) (statementOf st)
      (⟨cands, histOf proof, ⟨beta.toK, pairK lookup, pairK scatter⟩, mc, valuesOf proof.earlyValues⟩ :
        Layer1.EarlyOut Fp K) := by
    refine ⟨fun u inside => poles u (by rw [shape.histogram]; exact inside), ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [root 0 (by norm_num)]
      exact Ext.toK_ne_zero nonzero
    · rw [root 0 (by norm_num)]
      simp only [pairK]
      rw [Ext.toK_of_beq rootEq, Ext.toK_mul, totalValue, shape.histogram]
      rfl
    · rw [root 1 (by norm_num), root 2 (by norm_num)]
      exact pairCheck_ok checked.rows
    · rw [root 3 (by norm_num), root 4 (by norm_num)]
      exact pairCheck_ok checked.slots
    · rw [leafValue 1 0 (by norm_num), checked.runLeaf, Ext.toK_sub, toK_tuple]
      rfl
    · rw [leafValue 2 1 (by norm_num), checked.rowLeaf, Ext.toK_sub, toK_tuple, EarlyOut.rowPoint,
        leafPoint 2 (by norm_num), toK_identity, chi_re st _ (by rw [rowSize, pointSize]; exact rowFits),
        chi_im st _ (by rw [rowSize, pointSize]; exact rowFits)]
      rfl
    · rw [leafValue 3 1 (by norm_num), checked.scatterLeaf, Ext.toK_add, Ext.toK_mul]
      rfl
    · have eqSize : (eqTable ((claims.leafPoint[4]!).extract rel.key.layout.slotVariables
          (claims.leafPoint[4]!).size)).size = 2 ^ rel.matrixVars := by
        rw [eqTable_size, Array.size_extract, pairSize, min_self, Nat.add_sub_cancel_left]
      rw [leafValue 4 0 (by norm_num), checked.pairLeaf, toK_sumRange, EarlyOut.pairPoint, leafPoint 4 (by norm_num),
        eqSize, shape.pairs, min_eq_left matricesFit]
      apply Finset.sum_congr rfl
      intro j inside
      have lt : j < 2 ^ ((claims.leafPoint[4]!).extract rel.key.layout.slotVariables
          (claims.leafPoint[4]!).size).size := by
        rw [Array.size_extract, pairSize]
        have := Finset.mem_range.mp inside
        rw [show min (rel.key.layout.slotVariables + rel.matrixVars) (rel.key.layout.slotVariables + rel.matrixVars) -
          rel.key.layout.slotVariables = rel.matrixVars by omega]
        omega
      rw [Ext.toK_mul, eqTable_get _ lt, toK_eqAt, pointK_extract_drop, Ext.toK_add, Ext.toK_mul]
      rfl
    · rw [leafValue 4 1 (by norm_num), checked.indexLeaf, Ext.toK_sub, toK_identity, EarlyOut.pairPoint,
        leafPoint 4 (by norm_num)]
      rfl
  refine ⟨[.ext beta, .pair lookup.1 lookup.2, .pair scatter.1 scatter.2] ++ dsG, ?_, mc, claimsAgree, lengths,
    rfl, rfl, checked.openings, ?_⟩
  · rw [run_observeExts h12, drawsG, run_samplePair h6, run_samplePair h5, run_sampleExt h4, run_observeAll h3,
      run_observeAll h2, run_observeAll h1]
    simp
  · unfold earlyGame
    apply Game.Reaches.recv ⟨cands, short⟩
    apply Game.Reaches.recv (histOf proof)
    simp only [List.map_cons, List.cons_append, List.nil_append, drawn_ext, drawn_pair]
    apply Game.Reaches.draw beta.toK
    apply Game.Reaches.draw (pairK lookup)
    apply Game.Reaches.draw (pairK scatter)
    refine Game.Reaches.congr (Game.Reaches.bind reachG ?_) (List.append_nil _)
    apply Game.Reaches.recv (valuesOf proof.earlyValues)
    rw [if_pos checks]
    exact Game.Reaches.pure _

end Stage

end CompressionLean.Bridge
