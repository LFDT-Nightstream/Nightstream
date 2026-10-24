import CompressionLean.Sim.Late
import CompressionLean.Sim.Linear
import CompressionLean.Sim.Whir
import CompressionLean.Checks.KeyFacts

/-!
Owns `verify_ensures`: the circuit's `verify` (`Verifier.lean`, `verifier.rs:136-316`)
simulated against the replay's (`Ports/Replay/Stages.lean`). Each stage lemma tracks
its stage, so the stages chain as `Tracks` steps. The replay's runs give what later
stages read (`Tracks.runs`): the early values, the sizes of the run, lane and block
points, `τ`'s lanes and the queried values.
-/

namespace CompressionLean.Sim

open CompressionLean.Spec
open CompressionLean.Circuit
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Relation TreeShape WhirShape D foldVars runOracles)
open CompressionLean.Bridge (run_bind run_pure run_liftExcept run_sampleWords)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

private theorem IsWords.iff_forall₂ {c : Ctx F} {n : ℕ} {hs : Array Var} {gs : Array Word} :
    IsWords c n hs gs ↔ List.Forall₂ (fun v g => IsWord c n v g) hs.toList gs.toList := by
  constructor
  · intro h
    refine List.forall₂_iff_get.mpr ⟨by simpa using h.1, fun i h1 h2 => ?_⟩
    have := h.2 i (by simpa using h1)
    rw [getElem!_pos hs i (by simpa using h1), getElem!_pos gs i (by simpa using h2)] at this
    simpa using this
  · exact fun h => IsWords.of_forall₂ h.flip

private theorem forall₂_of_index {α β : Type} [Inhabited α] [Inhabited β] {Q : α → β → Prop} {xs : Array α}
    {gs : Array β} (h : xs.size = gs.size ∧ ∀ i < xs.size, Q xs[i]! gs[i]!) : List.Forall₂ Q xs.toList gs.toList := by
  refine List.forall₂_iff_get.mpr ⟨by simpa using h.1, fun i h1 h2 => ?_⟩
  have := h.2 i (by simpa using h1)
  rw [getElem!_pos xs i (by simpa using h1), getElem!_pos gs i (by simpa using h2)] at this
  simpa using this

private theorem pairs_forall₂ {c : Ctx F} {n : ℕ} {xs : List KVar} {gs : List (Word × Word)}
    (h : List.Forall₂ (IsK c n) xs gs) :
    List.Forall₂ (fun v g => IsWord c n v g) (xs.flatMap fun value => [value.1, value.2])
      (gs.flatMap fun value => [value.1, value.2]) :=
  List.rel_flatMap h fun _ _ hk => List.Forall₂.cons hk.1 (List.Forall₂.cons hk.2 List.Forall₂.nil)

private theorem rows_forall₂ {c : Ctx F} {n : ℕ} {xs : Array (Array Var)} {gs : Array (Array Word)}
    (h : xs.size = gs.size ∧ ∀ i < xs.size, IsWords c n xs[i]! gs[i]!) :
    List.Forall₂ (fun v g => IsWord c n v g) xs.flatten.toList gs.flatten.toList := by
  simp only [Array.toList_flatten]
  refine List.rel_flatten ?_
  rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff]
  exact (forall₂_of_index h).imp fun _ _ hab => IsWords.iff_forall₂.mp hab

private theorem IsClaim.words {c : Ctx F} {n : ℕ} {v : ClaimView} {st : Replay.Statement} (h : IsClaim c n v st) :
    IsWords c n v.words st.words := by
  refine IsWords.iff_forall₂.mpr ?_
  simp only [ClaimView.words, Replay.Statement.words, Array.toList_append]
  refine List.rel_append (List.rel_append (rows_forall₂ h.commitment) (rows_forall₂ h.publicInput)) ?_
  have point := forall₂_of_index h.point
  have evalK := forall₂_of_index h.evalK
  have evalA : List.Forall₂ (IsK c n) v.evalA.flatten.toList st.evalA.flatten.toList := by
    simp only [Array.toList_flatten]
    refine List.rel_flatten ?_
    rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff]
    exact (forall₂_of_index h.evalA).imp fun _ _ hab => forall₂_of_index hab
  simpa [Array.toList_flatMap, Array.toList_append, List.flatMap_append] using
    pairs_forall₂ (List.rel_append (List.rel_append point evalK) evalA)

private theorem readProof_openings {rel : Relation} {g : Replay.Proof} (read : Replay.readProof rel g = .ok ()) :
    Replay.readOpening rel.p1Whir g.p1Opening = .ok () ∧ Replay.readOpening rel.p0Whir g.p0Opening = .ok () := by
  unfold Replay.readProof at read
  obtain ⟨_, -, read⟩ := Bridge.Except.bind_ok read
  obtain ⟨_, -, read⟩ := Bridge.Except.bind_ok read
  obtain ⟨_, -, read⟩ := Bridge.Except.bind_ok read
  obtain ⟨_, -, read⟩ := Bridge.Except.bind_ok read
  obtain ⟨⟨⟩, p1, p0⟩ := Bridge.Except.bind_ok read
  exact ⟨p1, p0⟩

private theorem early_run {rel : Relation} {st : Replay.Statement} {g : Replay.Proof} {t t' : Replay.Transcript}
    {out : Replay.EarlyOut} (read : Replay.readProof rel g = .ok ())
    (run : Runs (Replay.earlyStage rel st g) t out t') :
    out.values = g.earlyValues ∧ out.openings.runPoint.size = rel.key.layout.runVariables := by
  unfold Replay.earlyStage at run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, gkr, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, check, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨rfl, rfl⟩ := run_pure run
  obtain ⟨checkOk, rfl⟩ := run_liftExcept check
  refine ⟨rfl, ?_⟩
  rw [(Bridge.checkEarly_ok checkOk).openings]
  exact ((gkrVerify_sizes (Bridge.readProof_ok read).early gkr).2 3 (by rw [Bridge.earlyShapes_size]; omega)).2

private theorem scaledEqPoint_size (bits : ℕ) : ∀ (c : Ext) {k : Ext} {point : Array Ext},
    Replay.scaledEqPoint c bits = .ok (k, point) → point.size = bits := by
  induction bits with
  | zero =>
      intro c k point ok
      rw [Replay.scaledEqPoint] at ok
      obtain ⟨-, rfl⟩ := Prod.mk.inj (Bridge.Except.pure_eq_ok ok)
      rfl
  | succ bits ih =>
      intro c k point ok
      rw [Replay.scaledEqPoint] at ok
      split_ifs at ok with pole
      · obtain ⟨_, thrown, _⟩ := Bridge.Except.bind_ok ok
        cases thrown
      · obtain ⟨⟨k', rest⟩, recursive, ok⟩ := Bridge.Except.bind_ok ok
        obtain ⟨-, rfl⟩ := Prod.mk.inj (Bridge.Except.pure_eq_ok ok)
        rw [Array.size_append, ih _ recursive]
        simp only [List.size_toArray, List.length_cons, List.length_nil]
        omega

private theorem linear_run {rel : Relation} {st : Replay.Statement} {g : Replay.Proof} {t t' : Replay.Transcript}
    {out : Replay.LinearOut} (read : Replay.readProof rel g = .ok ())
    (run : Runs (Replay.linearStage rel st g) t out t') :
    out.mixing = ⟨out.lam, rel.kappa, rel.matrices⟩ ∧ out.tau.size = D ∧ out.lanePoint.size = Replay.laneVars ∧
      out.blockPoint.size = rel.blockVars := by
  have shape := Bridge.readProof_ok read
  unfold Replay.linearStage at run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨⟨_, _⟩, _, block, run⟩ := run_bind run
  obtain ⟨⟨_, _⟩, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨⟨_, _⟩, _, lane, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨rfl, rfl⟩ := run_pure run
  obtain ⟨scaleOk, rfl⟩ := run_liftExcept lane
  refine ⟨rfl, by simp [Replay.tau], scaledEqPoint_size _ _ scaleOk, ?_⟩
  rw [Bridge.sumcheckRounds_size _ _ _ _ _ _ block]
  simp [shape.linear, Relation.cubeVars]

private theorem setupCheck_size {root : Array Word} {vars index : ℕ} {leaf : Replay.Leaf} {a b alpha : Array Ext}
    {vals : Array Ext} (ok : Replay.setupCheck root vars index leaf #[a, b] alpha = .ok vals) : vals.size = 2 := by
  by_cases path : Replay.pathOk root index leaf = true
  · rw [setupCheck_ok path] at ok
    cases ok
    rfl
  · unfold Replay.setupCheck at ok
    simp only [path, Replay.ensure] at ok
    cases ok

private theorem queried_size {root : Array Word} {vars : ℕ} {a b alpha : Array Ext} :
    ∀ (l : List (ℕ × Replay.Leaf)) (init out : Array Ext),
      l.foldlM (fun queried pair => do
        pure (queried ++ (← Replay.setupCheck root vars pair.1 pair.2 #[a, b] alpha))) init = .ok out →
        out.size = init.size + 2 * l.length
  | [], init, out, h => by
      cases h
      simp
  | pair :: rest, init, out, h => by
      rw [List.foldlM_cons] at h
      obtain ⟨_, step, h⟩ := Bridge.Except.bind_ok h
      obtain ⟨_, checked, step⟩ := Bridge.Except.bind_ok step
      obtain rfl := Bridge.Except.pure_eq_ok step
      rw [queried_size rest _ _ h, Array.size_append, setupCheck_size checked, List.length_cons]
      ring

private theorem honest_run {rel : Relation} {g : Replay.Proof} {early : Replay.EarlyOut} {lin : Replay.LinearOut}
    {t t' : Replay.Transcript} {out : Replay.HonestOut} (read : Replay.readProof rel g = .ok ())
    (run : Runs (Replay.honestStage rel g early lin) t out t') : out.queried.size = 2 * rel.queries := by
  unfold Replay.honestStage at run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, -, run⟩ := run_bind run
  obtain ⟨_, _, sampled, run⟩ := run_bind run
  obtain ⟨_, _, folded, run⟩ := run_bind run
  obtain ⟨rfl, rfl⟩ := run_pure run
  obtain ⟨foldOk, rfl⟩ := run_liftExcept folded
  rw [queried_size _ _ _ foldOk]
  simp [(run_sampleWords sampled).2, (Bridge.readProof_ok read).leaves]

instance verify_appends (rel : Relation) (p1Key p0Key : WhirKey) (seed : DigestVar) (statement : ClaimView)
    (proof : ProofView) : Appends (Circuit.verify rel p1Key p0Key seed statement proof) := by
  unfold Circuit.verify; appends

/-- **`verify`** (`verifier.rs:136-316`, called by `verify_with`, `lib.rs:692-703`).
When the inputs hold a seed, a statement that fits the relation and a proof that
it reads, and the circuit's checks hold, the replay's `verify` accepts them from
the empty transcript. -/
theorem verify_ensures [CharP F goldilocksModulus] {key : Layer1Key}
    (circuit : CircuitFacts key.relation key.p1Key key.p0Key) (relation : RelationFacts key.relation)
    (named : ∀ s ∈ key.relation.key.layout.segments,
      s.classIndex < key.relation.key.layout.classes.size ∧ s.lane < D)
    {c : Ctx F} {n : ℕ} {w : Var → Word} {seed : DigestVar} {statement : ClaimView} {proof : ProofView}
    {seedWords : Array Word} {st : Replay.Statement} {g : Replay.Proof} (hp : Permutes c.perm)
    (hseed : IsDigest c n seed seedWords) (hst : IsClaim c n statement st) (fits : st.fits key.relation = true)
    (hproof : IsProof c w n proof g) (read : Replay.readProof key.relation g = .ok ()) :
    Ensures c n (verify key.relation key.p1Key key.p0Key seed statement proof)
      fun _ _ => ∃ points t', Runs (Replay.verify key.relation seedWords st g) ⟨Replay.Duplex.new, #[]⟩ points t' := by
  obtain ⟨-, -, -, root, short, k1, k0, hp1, hp1count, hp0⟩ := circuit
  have long := relation.bits.long
  have small : key.relation.setupVars + 1 - foldVars ≤ 64 := by have := relation.high; omega
  have shape := Bridge.readProof_ok read
  obtain ⟨r1, r0⟩ := readProof_openings read
  obtain ⟨-, -, -, hruns, hpairs, hblock, hquot, hlin, hfin, hpart, hp1', hleaves, hlate, hlateValues, hp1o, hp0o⟩ :=
    id hproof
  obtain ⟨hps, hpoint⟩ := hst.point
  obtain ⟨linSize, linGet⟩ := forall₂_entries hlin
  obtain ⟨partSize, partGet⟩ := forall₂_entries hpart
  obtain ⟨leafSize, leafGet⟩ := forall₂_entries hleaves
  unfold verify
  refine Duplex.new_ensures.bind fun duplex _ _ hd => ?_
  refine Ensures.bind (Q := fun _ _ => ∃ points t', Runs (Replay.verify key.relation seedWords st g)
    ⟨Replay.Duplex.new, #[]⟩ points t') ((show Tracks c _ _ (Replay.verify key.relation seedWords st g)
      (fun _ _ _ => True) from ?_) duplex ⟨Replay.Duplex.new, #[]⟩ hd |>.post
      fun _ _ ⟨points, t', run, _, _⟩ => ⟨points, t', run⟩) fun _ _ _ h => Ensures.pure h
  unfold Replay.verify
  refine Tracks.ensure fits (Tracks.check read ?_)
  refine Tracks.bind (Duplex.observeSlice_tracks hp (IsWords.ofFn hseed.1 fun lane => (hseed.2 lane).mono))
    fun _ _ _ _ _ => ?_
  refine Tracks.bind (Duplex.observeConsts_tracks hp key.relation.words) fun _ _ _ _ _ => ?_
  refine Tracks.bind (Duplex.observeSlice_tracks hp hst.mono.words) fun _ _ _ _ _ => ?_
  refine Tracks.bind (earlyStage_tracks hp hps (fun i hi => ⟨(hpoint i hi).1.mono, (hpoint i hi).2.mono⟩)
    hproof.mono read).runs fun early eout _ _ he => ?_
  obtain ⟨he, _, _, erun⟩ := he
  obtain ⟨evalues, runSize⟩ := early_run read erun
  obtain ⟨-, zLeaf, erunPoint, -, -, emult, erow, eslot, ecoef⟩ := he
  refine Tracks.bind (linearStage_tracks hp fits hst.mono hquot.mono linSize
    (fun i hi => by rw [getElem!_pos _ i hi]; exact (linGet i hi).mono) hfin.mono shape.linear
    (fun m hm => by rw [shape.rounds m (Array.mem_toList_iff.mpr hm)]; omega) shape.finals long).runs
    fun lin lout _ _ hl => ?_
  obtain ⟨hl, _, _, lrun⟩ := hl
  obtain ⟨mixEq, tauSize, laneSize, blockSize⟩ := linear_run read lrun
  obtain ⟨⟨hmix, powers⟩, htau, hblockPoint, hlanePoint, -, hkey, homega, hz⟩ := hl
  have kappa : lout.mixing.kappa = key.relation.kappa := by rw [mixEq]
  refine Tracks.bind (honestStage_tracks (proof := g) (early := eout) (lin := lout) hp hlanePoint.mono
    hblockPoint.mono erunPoint.mono erow.mono eslot.mono ecoef.mono (by rw [evalues]; exact hblock.mono)
    ⟨by simp, fun i hi => ?_⟩ hkey.mono partSize (fun i hi => by rw [getElem!_pos _ i hi]; exact (partGet i hi).mono)
    hp1'.mono leafSize (fun q hq => ?_)
    (by rw [Array.size_append, laneSize, blockSize, Relation.setupVars, Relation.cubeVars]; omega)
    (by rw [runSize, Relation.setupVars]; omega) root small).runs fun honest hout _ _ hh => ?_
  · simp only [Array.size_map, Array.size_range] at hi
    rw [getElem!_map_range _ _ hi, getElem!_map_range _ _ hi]
    exact (hmix.commitment (by rw [powers, Relation.rowCount]; omega)).mono
  · obtain ⟨values, path⟩ := leafGet q hq
    obtain ⟨psize, pget⟩ := forall₂_entries path
    have mem : g.leaves[q]! ∈ g.leaves.toList := by
      rw [getElem!_pos _ q (leafSize ▸ hq)]
      exact Array.getElem_mem_toList _
    exact ⟨values.mono, psize, psize.trans (shape.leafShapes _ mem).2, fun k hk => by
      rw [getElem!_pos _ k hk]
      exact (pget k hk).mono⟩
  obtain ⟨hh, _, _, hrun⟩ := hh
  have qsize := honest_run read hrun
  obtain ⟨halpha, -, -, -, ibits, hqueried⟩ := hh
  refine Tracks.bind (lateStage_tracks (proof := g) (lin := lout) hp htau.mono tauSize (by simp) (fun j hj => ?_)
    hlate.mono (hlateValues.imp fun _ _ x => x.mono) read short named) fun late lateOut _ _ hlt => ?_
  · simp only [Array.size_map, Array.size_range] at hj
    rw [getElem!_map_range _ _ hj]
    have e := hmix.evalA (j := j) (by rw [powers, kappa, Relation.rowCount]; omega)
    exact ⟨e.1.mono, e.2.mono⟩
  obtain ⟨⟨-, hlblock, -, hlomega⟩, pairsEq⟩ := hlt
  refine Tracks.lift (Ensures.mapM (R := fun _ _ _ => True) (fun _ _ _ _ _ _ => trivial) fun x hx m _ => ?_)
    fun queryPoints _ _ _ => ?_
  · obtain ⟨q, hq, rfl⟩ := Array.mem_iff_getElem.mp hx
    have size := (ibits q hq).1
    rw [getElem!_pos _ q hq] at size
    refine (queryPoint_ensures (index := hout.queries[q]!) size fun i hi => ?_).post fun _ _ _ => trivial
    have word := ((ibits q hq).2 i (size ▸ hi)).mono (m := m)
    rwa [getElem!_pos _ q hq] at word
  intro d tr hd
  refine (openingsStage_tracks (proof := g) (early := eout) (lin := lout) (honest := hout) (late := lateOut) hp
    hp1o.mono hp0o.mono (hpart.imp fun _ _ x => x.mono) halpha.mono hqueried.mono homega.mono hlomega.mono
    zLeaf.mono hz.mono (by rw [evalues]; exact hruns.mono) (by rw [evalues]; exact hpairs.imp fun _ _ x => x.mono)
    (by rw [evalues]; exact hblock.mono) (by rw [pairsEq]; exact hlateValues.imp fun _ _ x => x.mono) hlblock.mono
    emult.mono r1 r0 qsize (by rw [evalues]; exact shape.pairs) (by rw [pairsEq]; exact shape.lateValues)
    k1 k0 hp1 hp1count hp0 d tr hd).post fun _ _ h => ?_
  obtain ⟨_, t', run, hdx, -⟩ := h
  obtain ⟨_, _, whir1, tail⟩ := run_bind run
  obtain ⟨_, _, agree1, tail⟩ := run_bind tail
  obtain ⟨_, _, whir0, agree0⟩ := run_bind tail
  exact ⟨_, t', Bridge.runs_bind whir1 (Bridge.runs_bind agree1 (Bridge.runs_bind whir0
    (Bridge.runs_bind agree0 (Bridge.runs_pure _ _)))), hdx, trivial⟩

end CompressionLean.Sim
