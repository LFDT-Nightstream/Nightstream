import CompressionLean.Circuit.SimLate
import CompressionLean.Circuit.SimLinear
import CompressionLean.Circuit.SimWhir

/-!
Owns `verify_ensures`: the circuit's `verify` (`Verifier.lean`, `verifier.rs:136-316`)
simulated against the replay's (`Replay/Stages.lean`). The stage lemmas give each
stage. The replay's runs give what later stages read: the early values, the sizes
of the run, lane and block points, `τ`'s lanes and the queried values.

`Ensures` also fixes the instructions, which do not depend on replay values. So a
fact that only the precondition gives (the statement's fit, the proof's shape, the
point sizes) serves as a plain premise of a stage lemma, and one run at values that
have it (zeros of the relation's shape, or an empty point) gives the instructions
(`Ensures.under`). `runs_observe3` and `isDuplex_mk` state the steps that observe
the relation words on abstract words, so that no check evaluates those words.
-/

namespace CompressionLean.Circuit

open CompressionLean.Lowering
open CompressionLean.Lib
open CompressionLean.Replay (Word Ext Relation TreeShape WhirShape D foldVars runOracles Runs run_bind run_pure
  run_liftExcept run_sampleWords)
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

private theorem Simulates.observeConstants [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop} {d : Duplex}
    {r : Replay.Duplex} (words : Array Word) (hp : Permutes c.perm) (hd : P → IsDuplex c n d r) :
    Simulates c n P (StateT.run (forIn words () fun word _ => (do
        let h ← liftM (Build.const word)
        Duplex.observe h
        Pure.pure (ForInStep.yield ()) : Transcript (ForInStep Unit))) d)
      (fun res m => IsDuplex c m res.2 (words.foldl Replay.Duplex.observe r)) True := by
  rw [← Array.forIn_toList]
  refine ((Simulates.loopT (xs := words.toList) (init := ()) (d := d) (P := P) (D := fun _ => True)
      (I := fun k _ e m => IsDuplex c m e ((words.toList.take k).foldl Replay.Duplex.observe r))
      fun k hk acc e m le => ?_).pre (fun h => ⟨h, by simpa using hd h⟩)).post
    (fun _ res m h => by
      rw [← Array.foldl_toList]
      simpa [List.take_of_length_le (le_of_eq (Array.length_toList (xs := words)))] using h) (fun _ => by simp)
  rw [StateT.run_bind]
  refine Simulates.bind₀ (Simulates.lift (Simulates.const words.toList[k])) fun x m₁ l₁ => ?_
  rw [StateT.run_bind]
  refine Simulates.bind₀ (Simulates.observe hp (fun h => h.2.2 ▸ h.1.2.mono l₁) fun h => h.2.1) fun res m₂ l₂ => ?_
  rw [StateT.run_pure]
  refine Simulates.pure fun h => ⟨rfl, ?_⟩
  rw [List.take_add_one, List.foldl_append, List.getElem?_eq_getElem hk]
  simpa [← h.1.2.2] using h.2

private def zeroGkr (shapes : Array TreeShape) : Replay.GkrProof :=
  { roots := Array.replicate shapes.size (0, 0),
    layers := (Array.range (Replay.layerCount shapes)).map fun i =>
      { rounds := Array.replicate i (Array.replicate (Replay.layerDegree (Replay.layerFactors shapes (i + 1))) 0),
        children := (Replay.activeTrees shapes (i + 1)).map fun t =>
          Array.replicate (2 * (Replay.partFactors shapes (i + 1) t + 1)) 0 } }

private theorem readGkr_zero {shapes : Array TreeShape} (h : ∀ s ∈ shapes, s.depth ≠ 0 ∧ s.factors ≤ 2) :
    Replay.readGkr (zeroGkr shapes) shapes = .ok () := by
  have any : shapes.any (fun s => s.depth == 0 || decide (s.factors > 2)) = false := by
    rw [Array.any_eq_false]
    intro i hi
    have := h shapes[i] (Array.getElem_mem hi)
    simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or]
    omega
  have layers : ((List.range (Replay.layerCount shapes)).all fun i =>
      Replay.layerShapeOk shapes (i + 1) (zeroGkr shapes).layers[i]!) = true := by
    rw [List.all_eq_true]
    intro x hx
    rw [List.mem_range] at hx
    simp only [zeroGkr]
    rw [getElem!_map_range _ _ hx]
    simp only [Replay.layerShapeOk, Bool.and_eq_true, beq_iff_eq]
    rw [Array.all_eq_true, Array.all_eq_true]
    simp
  unfold Replay.readGkr
  simp [zeroGkr, any, Replay.ensure] at layers ⊢
  rw [if_pos layers]
  rfl

private def zeroOpening (shape : WhirShape) : Replay.Opening :=
  { initialOod := Array.replicate shape.oodSamples 0, evals := shape.batches.map fun b => Array.replicate b.size 0,
    initialSumcheck := Array.replicate shape.initialSumcheck.rounds (0, 0),
    rounds := shape.rounds.map fun r =>
      { root := #[], ood := Array.replicate r.oodSamples 0, sumcheck := Array.replicate r.sumcheck.rounds (0, 0) },
    finalPoly := Array.replicate shape.finalPolyLen 0,
    finalSumcheck := Array.replicate shape.finalSumcheck.rounds (0, 0) }

private theorem readOpening_zero (shape : WhirShape) : Replay.readOpening shape (zeroOpening shape) = .ok () := by
  have batches : ((shape.batches.zip (zeroOpening shape).evals).all fun pair => pair.2.size == pair.1.size) = true := by
    rw [Array.all_eq_true]
    simp [zeroOpening]
  unfold Replay.readOpening
  rw [← Array.forIn_toList, forIn_ok]
  · simp only [batches]
    simp [zeroOpening, Replay.ensure]
  · rintro ⟨x, a⟩ hp
    have same : List.Forall₂ (fun (x : Replay.WhirRoundShape) (a : Replay.WhirRound) =>
        a.ood.size = x.oodSamples ∧ a.sumcheck.size = x.sumcheck.rounds) shape.rounds.toList
        (zeroOpening shape).rounds.toList := by
      rw [zeroOpening, Array.toList_map, List.forall₂_map_right_iff, List.forall₂_same]
      simp
    rw [Array.toList_zip] at hp
    have := List.forall₂_zip same hp
    simp [Replay.ensure, this.1, this.2]
    rfl

private def zeroProof (rel : Relation) : Replay.Proof :=
  { p0 := Array.replicate 4 0, histogram := Array.replicate (2 * rel.normBound + 1) 0,
    early := zeroGkr rel.earlyShapes, earlyValues := ⟨0, 0, 0, Array.replicate rel.matrices (0, 0), 0⟩,
    quotient := Array.replicate (D - 1) 0, linear := Array.replicate rel.cubeVars (Array.replicate 2 0),
    finals := Array.replicate 3 0, partials := Array.replicate 2 (Array.replicate (2 ^ foldVars) 0),
    p1 := Array.replicate 4 0,
    leaves := Array.replicate rel.queries
      ⟨Array.replicate ((rel.kappa + runOracles) * 2 ^ foldVars) 0, Array.replicate (rel.setupVars + 1 - foldVars) #[]⟩,
    late := zeroGkr rel.lateShapes, lateValues := Array.replicate rel.matrices (0, 0),
    p1Opening := zeroOpening rel.p1Whir, p0Opening := zeroOpening rel.p0Whir }

private theorem readProof_zero {rel : Relation} (hrun : rel.key.layout.runVariables ≠ 0)
    (hrow : rel.key.layout.rowVariables ≠ 0) (hslot : rel.key.layout.slotVariables + rel.matrixVars ≠ 0) :
    Replay.readProof rel (zeroProof rel) = .ok () := by
  have early : ∀ s ∈ rel.earlyShapes, s.depth ≠ 0 ∧ s.factors ≤ 2 := by
    simp [Relation.earlyShapes, Relation.cubeVars, Replay.laneVars, hrun, hrow]
    omega
  have late : ∀ s ∈ rel.lateShapes, s.depth ≠ 0 ∧ s.factors ≤ 2 := by
    simp [Relation.lateShapes, Relation.blockVars, Replay.bitsAtLeastOne]
  have linear : (Array.replicate rel.cubeVars (Array.replicate 2 (0 : Ext))).all (·.size == 2) = true := by
    rw [Array.all_eq_true]
    simp
  have partials : (Array.replicate 2 (Array.replicate (2 ^ foldVars) (0 : Ext))).all (·.size == 2 ^ foldVars) =
      true := by
    rw [Array.all_eq_true]
    simp
  have leaves : ((zeroProof rel).leaves.all fun leaf =>
      leaf.values.size == (rel.kappa + runOracles) * 2 ^ foldVars &&
        leaf.path.size == rel.setupVars + 1 - foldVars) = true := by
    rw [zeroProof, Array.all_eq_true]
    simp
  have gkrs : Replay.readGkr (zeroProof rel).early rel.earlyShapes = .ok () ∧
      Replay.readGkr (zeroProof rel).late rel.lateShapes = .ok () := ⟨readGkr_zero early, readGkr_zero late⟩
  have openings : Replay.readOpening rel.p1Whir (zeroProof rel).p1Opening = .ok () ∧
      Replay.readOpening rel.p0Whir (zeroProof rel).p0Opening = .ok () := ⟨readOpening_zero _, readOpening_zero _⟩
  unfold Replay.readProof
  rw [leaves, gkrs.1, gkrs.2, openings.1, openings.2]
  simp only [zeroProof, linear, partials]
  simp [Replay.ensure]
  rfl

private def zeroStatement (rel : Relation) : Replay.Statement :=
  ⟨Array.replicate rel.kappa (Array.replicate D 0), Array.replicate rel.publicBlocks (Array.replicate D 0),
    Array.replicate rel.pointVariables (0, 0), Array.replicate D (0, 0),
    Array.replicate rel.matrices (Array.replicate D (0, 0))⟩

private theorem fits_zero (rel : Relation) : (zeroStatement rel).fits rel = true := by
  simp only [zeroStatement, Replay.Statement.fits, Bool.and_eq_true]
  rw [Array.all_eq_true, Array.all_eq_true, Array.all_eq_true]
  simp

private theorem readProof_openings {rel : Relation} {g : Replay.Proof} (read : Replay.readProof rel g = .ok ()) :
    Replay.readOpening rel.p1Whir g.p1Opening = .ok () ∧ Replay.readOpening rel.p0Whir g.p0Opening = .ok () := by
  unfold Replay.readProof at read
  obtain ⟨_, -, read⟩ := Replay.Except.bind_ok read
  obtain ⟨_, -, read⟩ := Replay.Except.bind_ok read
  obtain ⟨_, -, read⟩ := Replay.Except.bind_ok read
  obtain ⟨_, -, read⟩ := Replay.Except.bind_ok read
  obtain ⟨⟨⟩, p1, p0⟩ := Replay.Except.bind_ok read
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
      obtain ⟨-, rfl⟩ := Prod.mk.inj (Replay.Except.pure_eq_ok ok)
      rfl
  | succ bits ih =>
      intro c k point ok
      rw [Replay.scaledEqPoint] at ok
      split_ifs at ok with pole
      · obtain ⟨_, thrown, _⟩ := Replay.Except.bind_ok ok
        cases thrown
      · obtain ⟨⟨k', rest⟩, recursive, ok⟩ := Replay.Except.bind_ok ok
        obtain ⟨-, rfl⟩ := Prod.mk.inj (Replay.Except.pure_eq_ok ok)
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
      obtain ⟨_, step, h⟩ := Replay.Except.bind_ok h
      obtain ⟨_, checked, step⟩ := Replay.Except.bind_ok step
      obtain rfl := Replay.Except.pure_eq_ok step
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

private theorem isDuplex_mk {c : Ctx F} {m : ℕ} {d : Duplex} {r : Replay.Duplex} {ds : Array Replay.Draw}
    (h : IsDuplex c m d r) : IsDuplex c m d (Replay.Transcript.mk r ds).duplex :=
  h

private theorem runs_observe3 {β : Type} {a b e : Array Word} {d : Replay.Duplex} {ds : Array Replay.Draw}
    {k : Replay.ReplayM β} {x : β} {t' : Replay.Transcript}
    (h : Runs k ⟨e.foldl Replay.Duplex.observe (b.foldl Replay.Duplex.observe (a.foldl Replay.Duplex.observe d)), ds⟩
      x t') :
    Runs (do Replay.observeAll a; Replay.observeAll b; Replay.observeAll e; k) ⟨d, ds⟩ x t' :=
  Replay.runs_bind (Replay.runs_observeAll _ _) (Replay.runs_bind (Replay.runs_observeAll _ _)
    (Replay.runs_bind (Replay.runs_observeAll _ _) h))

section Verify

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop}

private theorem verify_core {rel : Relation} {p1Key p0Key : WhirKey}
    (long : Nat.log2 (D * 2 ^ rel.blockVars).nextPowerOfTwo ≤ rel.pointVariables)
    (root : rel.key.root.size = 4) (small : rel.setupVars + 1 - foldVars ≤ 64)
    (short : ∀ cl ∈ rel.key.layout.classes, cl.len ≤ D)
    (named : ∀ s ∈ rel.key.layout.segments, s.classIndex < rel.key.layout.classes.size ∧ s.lane < D)
    (k1 : p1Key.batches.size = rel.p1Whir.batches.size) (k0 : p0Key.batches.size = rel.p0Whir.batches.size)
    (hp1 : ∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0)
    (hp1count : (rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4)
    (hp0 : rel.p0Whir.batches.toList.map (·.size) = [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1])
    {w : Var → Word} {seed : DigestVar} {statement : ClaimView} {proof : ProofView} {seedWords : Array Word}
    {st : Replay.Statement} {g : Replay.Proof} (hp : Permutes c.perm)
    (fits : st.fits rel = true) (read : Replay.readProof rel g = .ok ())
    (h : P → IsDigest c n seed seedWords ∧ IsClaim c n statement st ∧ IsProof c w n proof g) :
    Ensures c n P (verify rel p1Key p0Key seed statement proof)
      (fun _ _ => ∃ points t', Runs (Replay.verify rel seedWords st g) ⟨Replay.Duplex.new, #[]⟩ points t') := by
  have shape := Bridge.readProof_ok read
  obtain ⟨r1, r0⟩ := readProof_openings read
  unfold verify
  refine Ensures.bind Simulates.duplexNew.ensures fun duplex m₀ l₀ => ?_
  refine Ensures.bind (Q := fun _ _ => ∃ points t', Runs (Replay.verify rel seedWords st g)
    ⟨Replay.Duplex.new, #[]⟩ points t') ?_ fun _ _ _ => Ensures.pure fun H => H.2
  refine Ensures.tbind (Simulates.observeSlice hp (fun H => H.2.1)
    fun H => IsWords.ofFn (h H.1).1.1 fun lane => ((h H.1).1.2 lane).mono l₀).ensures fun _ d₁ m₁ l₁ => ?_
  refine Ensures.tbind (Simulates.observeConstants rel.words hp fun H => H.2.1).ensures fun _ d₂ m₂ l₂ => ?_
  refine Ensures.tbind (Simulates.observeSlice hp (fun H => H.2.1)
    fun H => ((h H.1.1.1).2.1.mono (l₀.trans (l₁.trans l₂))).words).ensures fun _ d₃ m₃ l₃ => ?_
  have pf : P → IsProof c w n proof g := fun H => (h H).2.2
  have L₃ : n ≤ m₃ := l₀.trans (l₁.trans (l₂.trans l₃))
  refine Ensures.tbind (earlyStage_ensures (t := ⟨Array.foldl Replay.Duplex.observe (Array.foldl Replay.Duplex.observe
      (Array.foldl Replay.Duplex.observe Replay.Duplex.new seedWords) rel.words) st.words, #[]⟩) hp
    (fun H => isDuplex_mk H.2.1)
    (fun H => ((h H.1.1.1.1).2.1.mono L₃).point) (fun H => (pf H.1.1.1.1).mono L₃) read) fun early d₄ m₄ l₄ => ?_
  refine Ensures.witness (A := fun out : Replay.EarlyOut => out.openings.runPoint.size ≤ rel.setupVars)
    ⟨⟨0, (0, 0), (0, 0)⟩, ⟨#[], #[], #[], #[], #[]⟩, ⟨0, 0, 0, #[], 0⟩, 0, ⟨#[], #[], #[], 0, 0, 0, 0⟩⟩
    (Nat.zero_le _) (fun out ⟨_, run, _⟩ => by
      rw [(early_run read run).2, Relation.setupVars]
      omega) fun eout runSize => ?_
  refine Ensures.exists_and ⟨Replay.Duplex.new, #[]⟩ fun t₁ => ?_
  have L₄ : n ≤ m₄ := L₃.trans l₄
  refine Ensures.pre (P' := P ∧ _) ?_ fun H => ⟨H.1.1.1.1.1, H.2⟩
  refine Ensures.tbind (linearStage_ensures (t := t₁) hp (fun H => H.2.2.2) fits (fun H => (h H.1).2.1.mono L₄)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, quotient, -⟩ := pf H.1
      exact quotient.mono L₄)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, linear, -⟩ := pf H.1
      obtain ⟨size, get⟩ := forall₂_entries linear
      exact ⟨size, fun i hi => by rw [getElem!_pos _ i hi]; exact (get i hi).mono L₄⟩)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, finals, -⟩ := pf H.1
      exact finals.mono L₄)
    (fun _ => shape.linear) (fun _ m hm => by rw [shape.rounds m (Array.mem_toList_iff.mpr hm)]; omega)
    (fun _ => shape.finals) long) fun lin d₅ m₅ l₅ => ?_
  refine Ensures.witness (A := fun out : Replay.LinearOut => (out.lanePoint ++ out.blockPoint).size ≤ rel.setupVars)
    ⟨0, 0, #[], 0, #[], 0, 0, 0, 0, #[], ⟨0, 0, 0⟩, #[]⟩ (Nat.zero_le _) (fun out ⟨_, run, _⟩ => by
      obtain ⟨-, -, lane, block⟩ := linear_run read run
      rw [Array.size_append, lane, block, Relation.setupVars, Relation.cubeVars]
      omega) fun lout keySize => ?_
  refine Ensures.exists_and t₁ fun t₂ => ?_
  have L₅ : n ≤ m₅ := L₄.trans l₅
  refine Ensures.pre (P' := P ∧ (_ ∧ IsEarlyOut c m₅ early eout) ∧ _) ?_
    fun H => ⟨H.1.1, ⟨H.1.2.1, H.1.2.2.1.mono l₅⟩, H.2⟩
  refine Ensures.tbind (honestStage_ensures (t := t₂) (proof := g) (early := eout) (lin := lout) hp
    (fun H => H.2.2.2.2)
    (fun H => by obtain ⟨-, -, -, lane, -⟩ := H.2.2.2.1; exact lane)
    (fun H => by obtain ⟨-, -, block, -⟩ := H.2.2.2.1; exact block)
    (fun H => by obtain ⟨-, -, run, -⟩ := H.2.1.2; exact run)
    (fun H => by obtain ⟨-, -, -, -, -, -, row, -⟩ := H.2.1.2; exact row)
    (fun H => by obtain ⟨-, -, -, -, -, -, -, slot, -⟩ := H.2.1.2; exact slot)
    (fun H => by obtain ⟨-, -, -, -, -, -, -, -, coef⟩ := H.2.1.2; exact coef)
    (fun H => by
      obtain ⟨-, -, -, -, -, block, -⟩ := pf H.1
      dsimp only
      rw [(early_run read H.2.1.1).1]
      exact block.mono L₅)
    (fun H => by
      obtain ⟨⟨mix, powers⟩, -⟩ := H.2.2.2.1
      dsimp only
      refine ⟨by simp, fun i hi => ?_⟩
      simp only [Array.size_map, Array.size_range] at hi
      rw [getElem!_map_range _ _ hi, getElem!_map_range _ _ hi]
      exact mix.commitment (by rw [powers, Relation.rowCount]; omega))
    (fun H => by obtain ⟨-, -, -, -, -, key, -⟩ := H.2.2.2.1; exact key)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, partials, -⟩ := pf H.1
      obtain ⟨size, get⟩ := forall₂_entries partials
      exact ⟨size, fun i hi => by rw [getElem!_pos _ i hi]; exact (get i hi).mono L₅⟩)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, -, p1, -⟩ := pf H.1
      exact p1.mono L₅)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, leaves, -⟩ := pf H.1
      obtain ⟨size, get⟩ := forall₂_entries leaves
      refine ⟨size, fun q hq => ?_⟩
      obtain ⟨values, path⟩ := get q hq
      obtain ⟨psize, pget⟩ := forall₂_entries path
      have mem : g.leaves[q]! ∈ g.leaves.toList := by
        rw [getElem!_pos _ q (size ▸ hq)]
        exact Array.getElem_mem_toList _
      exact ⟨values.mono L₅, psize, psize.trans (shape.leafShapes _ mem).2, fun k hk => by
        rw [getElem!_pos _ k hk]
        exact (pget k hk).mono L₅⟩)
    keySize runSize root small) fun honest d₆ m₆ l₆ => ?_
  refine Ensures.exists_and (⟨0, #[], #[], #[], #[]⟩ : Replay.HonestOut) fun hout => ?_
  refine Ensures.exists_and t₂ fun t₃ => ?_
  have L₆ : n ≤ m₆ := L₅.trans l₆
  refine Ensures.pre (P' := P ∧ (_ ∧ IsEarlyOut c m₆ early eout) ∧ (_ ∧ IsLinearOut c m₆ rel lin lout) ∧ _) ?_
    fun H => ⟨H.1.1, ⟨H.1.2.1.1, H.1.2.1.2.mono l₆⟩, ⟨H.1.2.2.1, H.1.2.2.2.1.mono l₆⟩, H.2⟩
  refine Ensures.tbind (lateStage_ensures (t := t₃) (proof := g) (lin := lout) hp (fun H => H.2.2.2.2.2)
    (fun H => by obtain ⟨-, tau, -⟩ := H.2.2.1.2; exact tau)
    (fun H => (linear_run read H.2.2.1.1).2.1)
    (fun H => by
      obtain ⟨⟨mix, powers⟩, -⟩ := H.2.2.1.2
      have kappa : lout.mixing.kappa = rel.kappa := by rw [(linear_run read H.2.2.1.1).1]
      refine ⟨by simp, fun j hj => ?_⟩
      simp only [Array.size_map, Array.size_range] at hj
      rw [getElem!_map_range _ _ hj]
      exact mix.evalA (by rw [powers, kappa, Relation.rowCount]; omega))
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, late, -⟩ := pf H.1
      exact late.mono L₆)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, values, -⟩ := pf H.1
      exact values.imp fun _ _ x => x.mono L₆)
    read short named) fun late d₇ m₇ l₇ => ?_
  refine Ensures.exists_and (⟨0, ⟨#[], #[], #[], #[], #[]⟩, #[], ⟨#[], 0, #[], 0⟩⟩ : Replay.LateOut)
    fun lateOut => ?_
  refine Ensures.exists_and t₃ fun t₄ => ?_
  have L₇ : n ≤ m₇ := L₆.trans l₇
  refine Ensures.pre (P' := P ∧ (_ ∧ IsEarlyOut c m₇ early eout) ∧ (_ ∧ IsLinearOut c m₇ rel lin lout) ∧
    (_ ∧ IsHonestOut c m₇ rel eout lout honest hout) ∧ _) ?_
    fun H => ⟨H.1.1, ⟨H.1.2.1.1, H.1.2.1.2.mono l₇⟩, ⟨H.1.2.2.1.1, H.1.2.2.1.2.mono l₇⟩,
      ⟨H.1.2.2.2.1, H.1.2.2.2.2.1.mono l₇⟩, H.2⟩
  refine Ensures.tbind (Ensures.lift (Simulates.mapM (R := fun _ _ _ => True) (fun _ _ _ _ _ _ => trivial)
    fun x hx m lm => by
      obtain ⟨q, hq, rfl⟩ := Array.mem_iff_getElem.mp hx
      refine (Simulates.queryPoint (index := hout.queries[q]!) (fun H => ?_) fun H => ?_).post
        (fun _ _ _ _ => trivial) fun _ => Iff.rfl
      · obtain ⟨-, -, -, -, bits, -⟩ := H.2.2.2.1.2
        have size := (bits q hq).1
        rwa [getElem!_pos _ q hq] at size
      · obtain ⟨-, -, -, -, bits, -⟩ := H.2.2.2.1.2
        intro i hi
        have size := (bits q hq).1
        rw [getElem!_pos _ q hq] at size
        have word := ((bits q hq).2 i (size ▸ hi)).mono lm
        rwa [getElem!_pos _ q hq] at word).ensures) fun _ d₈ m₈ l₈ => ?_
  have L₈ : n ≤ m₈ := L₇.trans l₈
  refine Ensures.pre (P' := P ∧ (_ ∧ IsEarlyOut c m₈ early eout) ∧ (_ ∧ IsLinearOut c m₈ rel lin lout) ∧
    (_ ∧ IsHonestOut c m₈ rel eout lout honest hout) ∧ (_ ∧ IsLateOut c m₈ late lateOut ∧ _) ∧
    IsDuplex c m₈ d₈ t₄.duplex) ?_ fun H =>
      ⟨H.1.1, ⟨H.1.2.1.1, H.1.2.1.2.mono l₈⟩, ⟨H.1.2.2.1.1, H.1.2.2.1.2.mono l₈⟩,
        ⟨H.1.2.2.2.1.1, H.1.2.2.2.1.2.mono l₈⟩, ⟨H.1.2.2.2.2.1, H.1.2.2.2.2.2.1.mono l₈, H.1.2.2.2.2.2.2.1⟩,
        (show d₈ = d₇ from H.2.2) ▸ H.1.2.2.2.2.2.2.2.mono l₈⟩
  refine (openingsStage_ensures (t := t₄) (proof := g) (early := eout) (lin := lout) (honest := hout)
    (late := lateOut) hp (fun H => H.2.2.2.2.2)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, p1, -⟩ := pf H.1
      exact p1.mono L₈)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, p0⟩ := pf H.1
      exact p0.mono L₈)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, partials, -⟩ := pf H.1
      exact partials.imp fun _ _ x => x.mono L₈)
    (fun H => H.2.2.2.1.2.1) (fun H => H.2.2.2.1.2.2.2.2.2.2)
    (fun H => by obtain ⟨-, -, -, -, -, -, omega, -⟩ := H.2.2.1.2; exact omega)
    (fun H => H.2.2.2.2.1.2.1.2.2.2) (fun H => H.2.1.2.2.1)
    (fun H => by obtain ⟨-, -, -, -, -, -, -, zValue⟩ := H.2.2.1.2; exact zValue)
    (fun H => by
      obtain ⟨-, -, -, runs, -⟩ := pf H.1
      rw [(early_run read H.2.1.1).1]
      exact runs.mono L₈)
    (fun H => by
      obtain ⟨-, -, -, -, pairs, -⟩ := pf H.1
      rw [(early_run read H.2.1.1).1]
      exact pairs.imp fun _ _ x => x.mono L₈)
    (fun H => by
      obtain ⟨-, -, -, -, -, block, -⟩ := pf H.1
      rw [(early_run read H.2.1.1).1]
      exact block.mono L₈)
    (fun H => by
      obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, values, -⟩ := pf H.1
      rw [H.2.2.2.2.1.2.2]
      exact values.imp fun _ _ x => x.mono L₈)
    (fun H => H.2.2.2.2.1.2.1.2.1)
    (fun H => by obtain ⟨-, -, -, -, -, mult, -⟩ := H.2.1.2; exact mult)
    (fun _ => r1) (fun _ => r0) (fun H => honest_run read H.2.2.2.1.1)
    (fun H => by rw [(early_run read H.2.1.1).1]; exact shape.pairs)
    (fun H => by rw [H.2.2.2.2.1.2.2]; exact shape.lateValues)
    k1 k0 hp1 hp1count hp0).post fun H _ _ ⟨t', tail, _⟩ => ?_
  obtain ⟨-, ⟨early, -⟩, ⟨linear, -⟩, ⟨honest, -⟩, ⟨late, -⟩, -⟩ := H
  obtain ⟨_, _, whir1, tail⟩ := run_bind tail
  obtain ⟨_, _, agree1, tail⟩ := run_bind tail
  obtain ⟨_, _, whir0, agree0⟩ := run_bind tail
  unfold Replay.verify
  exact ⟨_, t', Replay.runs_bind (Replay.runs_ensure fits _ _) (Replay.runs_bind (Replay.runs_liftExcept read _)
    (runs_observe3 (Replay.runs_bind early (Replay.runs_bind linear (Replay.runs_bind honest
    (Replay.runs_bind late (Replay.runs_bind whir1 (Replay.runs_bind agree1 (Replay.runs_bind whir0
    (Replay.runs_bind agree0 (Replay.runs_pure _ _)))))))))))⟩

end Verify

/-- **`verify`** (`verifier.rs:136-316`, called by `verify_with`, `lib.rs:692-703`).
When the inputs hold a seed, a statement that fits the relation and a proof that
it reads, and the circuit's checks hold, the replay's `verify` accepts them from
the empty transcript. -/
theorem verify_ensures [CharP F goldilocksModulus] {key : Layer1Input}
    (hrun : key.relation.key.layout.runVariables ≠ 0) (hrow : key.relation.key.layout.rowVariables ≠ 0)
    (hslot : key.relation.key.layout.slotVariables + key.relation.matrixVars ≠ 0)
    (long : Nat.log2 (D * 2 ^ key.relation.blockVars).nextPowerOfTwo ≤ key.relation.pointVariables)
    (root : key.relation.key.root.size = 4) (small : key.relation.setupVars + 1 - foldVars ≤ 64)
    (short : ∀ cl ∈ key.relation.key.layout.classes, cl.len ≤ D)
    (named : ∀ s ∈ key.relation.key.layout.segments,
      s.classIndex < key.relation.key.layout.classes.size ∧ s.lane < D)
    (k1 : key.p1Key.batches.size = key.relation.p1Whir.batches.size)
    (k0 : key.p0Key.batches.size = key.relation.p0Whir.batches.size)
    (hp1 : ∀ b ∈ key.relation.p1Whir.batches, b.size % 3 = 0)
    (hp1count : (key.relation.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * key.relation.queries + 4)
    (hp0 : key.relation.p0Whir.batches.toList.map (·.size) =
      [1, 1, 2, 2 * key.relation.matrices, 1, 2 * key.relation.matrices + 1, 1])
    {c : Ctx F} {n : ℕ} {P : Prop} {w : Var → Word} {seed : DigestVar} {statement : ClaimView} {proof : ProofView}
    {seedWords : Array Word} {st : Replay.Statement} {g : Replay.Proof} (hp : Permutes c.perm)
    (h : P → Decodes c w ∧ IsDigest c n seed seedWords ∧ IsClaim c n statement st ∧ st.fits key.relation = true ∧
      IsProof c w n proof g ∧ Replay.readProof key.relation g = .ok ()) :
    Ensures c n P (verify key.relation key.p1Key key.p0Key seed statement proof)
      (fun _ _ => ∃ points t', Runs (Replay.verify key.relation seedWords st g) ⟨Replay.Duplex.new, #[]⟩ points t') :=
  Ensures.under (A := st.fits key.relation = true ∧ Replay.readProof key.relation g = .ok ())
    (verify_core (P := False) (w := w) (seedWords := #[]) long root small short named k1 k0 hp1 hp1count
      hp0 hp (fits_zero _) (readProof_zero hrun hrow hslot) False.elim)
    (fun H => ⟨(h H).2.2.2.1, (h H).2.2.2.2.2⟩) fun ⟨fits, read⟩ =>
    verify_core long root small short named k1 k0 hp1 hp1count hp0 hp fits read fun H =>
      ⟨(h H).2.1, (h H).2.2.1, (h H).2.2.2.2.1⟩

end CompressionLean.Circuit
