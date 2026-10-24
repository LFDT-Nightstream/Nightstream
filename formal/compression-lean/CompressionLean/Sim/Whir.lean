import CompressionLean.Sim.Views
import CompressionLean.Sim.Honest

/-!
Owns the simulation of our WHIR verifier (`whirVerify`, `whir/mod.rs:506-661`)
and of the openings (`openingsStage`, `verifier.rs:266-315`) against the
replay. `whirVerify_tracks`: when the checks hold, the circuit's WHIR verifier
tracks `Replay.whirTranscript`, and the opened values are the opening's.
`openingsStage_tracks`: the openings track the tail of `Replay.verify`, both
WHIR transcripts and both `openingsAgree` checks.

- The arithmetic, the Merkle paths and the STIR checks are steps that the
  replay does not take. They keep the duplex (`Keeps`), so they track the
  replay's `pure ()` (`Tracks.ofKeeps`).
- A query index's inverse check keeps the sample from `p - 1`, as the replay's
  `whirQueries` checks.
- The circuit compares opened and expected values pair by pair. The replay
  also compares their sizes, which Rust asserts, so the batch sizes are
  premises.
-/

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Replay (Word Ext WhirShape SumcheckShape ReplayM)
open CompressionLean.Bridge (Runs)
open NightstreamFPrime.Spec (goldilocksModulus)

variable {F : Type} [CommRing F]

/-! ## Instruction shape -/

instance Whir.absorb_appends (words : Array Word) (d : Duplex) : Appends ((Whir.absorb words).run d) := by
  unfold Whir.absorb; appends
instance Whir.expand_appends (x : ExtVar) (k : ℕ) : Appends (Whir.expand x k) := by unfold Whir.expand; appends
instance Whir.evalEq_appends (p q : Array ExtVar) : Appends (Whir.evalEq p q) := by unfold Whir.evalEq; appends
instance Whir.evalSelect_appends (var : Var) (point : Array ExtVar) : Appends (Whir.evalSelect var point) := by
  unfold Whir.evalSelect; appends
instance Whir.mle_appends (values r : Array ExtVar) : Appends (Whir.mle values r) := by unfold Whir.mle; appends
instance Whir.horner_appends (values : Array ExtVar) (var : Var) : Appends (Whir.horner values var) := by
  unfold Whir.horner; appends
instance Whir.Constraint.powers_appends (k : Whir.Constraint) : Appends k.powers := by
  unfold Whir.Constraint.powers; appends
instance Whir.Constraint.claimed_appends (k : Whir.Constraint) (powers : Array ExtVar) :
    Appends (k.claimed powers) := by
  unfold Whir.Constraint.claimed; appends
instance Whir.Constraint.weight_appends (k : Whir.Constraint) (powers r : Array ExtVar) :
    Appends (k.weight powers r) := by
  unfold Whir.Constraint.weight; appends
instance Whir.sumcheck_appends (shape : SumcheckShape) (rounds : Array (ExtVar × ExtVar)) (claim : ExtVar)
    (d : Duplex) : Appends ((Whir.sumcheck shape rounds claim).run d) := by
  unfold Whir.sumcheck; appends
instance Whir.queryIndex_appends (width : ℕ) (d : Duplex) : Appends ((Whir.queryIndex width).run d) := by
  unfold Whir.queryIndex; appends
instance Whir.stir_appends (queries : Array (Array Var)) (draws indexBits logFoldedDomain : ℕ) (root : DigestVar)
    (folding : Array ExtVar) (base : Bool) (d : Duplex) :
    Appends ((Whir.stir queries draws indexBits logFoldedDomain root folding base).run d) := by
  unfold Whir.stir; appends
instance whirVerify_appends (shape : WhirShape) (key : WhirKey) (root : DigestVar) (view : OpeningView)
    (points : Array (Array ExtVar)) (d : Duplex) : Appends ((Circuit.whirVerify shape key root view points).run d) := by
  unfold Circuit.whirVerify; appends
instance fromCoordinates_appends (values : Array ExtVar) : Appends (Circuit.fromCoordinates values) := by
  unfold Circuit.fromCoordinates; appends
instance openingsStage_appends (rel : Replay.Relation) (p1Key p0Key : WhirKey) (o : OpeningsInput) (d : Duplex) :
    Appends ((Circuit.openingsStage rel p1Key p0Key o).run d) := by
  unfold Circuit.openingsStage; appends

/-- Pass the steps that the replay does not take. -/
local macro "quiet_steps" : tactic =>
  `(tactic| repeat (with_reducible refine Tracks.quiet Tracks.ofKeeps fun _ _ _ _ => ?_))

/-! ## Lists and replay facts -/

section Facts

variable {c : Ctx F} {n : ℕ}

variable {β : Type}

private theorem forall₂_getElem! {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : List α} {ys : List β}
    (h : List.Forall₂ R xs ys) {k : ℕ} (hk : k < xs.length) : R xs[k] ys[k]! := by
  have len := h.length_eq
  rw [getElem!_pos ys k (len ▸ hk)]
  simpa using h.get (i := k) hk (len ▸ hk)

private theorem forall₂_getElem!_array {α β : Type} [Inhabited β] {R : α → β → Prop} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ R xs.toList ys.toList) {k : ℕ} (hk : k < xs.size) : R xs[k] ys[k]! := by
  have := forall₂_getElem! h (k := k) (by simpa using hk)
  rwa [Array.getElem_toList, Array.getElem!_toList] at this

private theorem IsExts.getElem! {xs : Array (Array ExtVar)} {gs : Array (Array Ext)}
    (h : List.Forall₂ (IsExts c n) xs.toList gs.toList) (k : ℕ) : IsExts c n xs[k]! gs[k]! := by
  by_cases hk : k < xs.size
  · rw [getElem!_pos xs k hk]
    exact forall₂_getElem!_array h hk
  · have len : xs.size = gs.size := by simpa using h.length_eq
    rw [getElem!_neg xs k hk, getElem!_neg gs k (len ▸ hk)]
    exact IsExts.empty

private theorem map_eq_of_zip {α β γ : Type} {f : α → γ} {g : β → γ} :
    ∀ {l₁ : List α} {l₂ : List β}, l₁.length = l₂.length → (∀ p ∈ l₁.zip l₂, f p.1 = g p.2) →
      l₁.map f = l₂.map g
  | [], [], _, _ => rfl
  | [], _ :: _, len, _ => by simp at len
  | _ :: _, [], len, _ => by simp at len
  | a :: l₁, b :: l₂, len, h => by
      rw [List.map_cons, List.map_cons, h (a, b) (by simp),
        map_eq_of_zip (by simpa using len) fun p hp => h p (by simp [hp])]

/-- `readOpening`'s batch sizes are the shape's. -/
private theorem readOpening_sizes {shape : WhirShape} {g : Replay.Opening} (h : Replay.readOpening shape g = .ok ()) :
    g.evals.toList.map (·.size) = shape.batches.toList.map (·.size) := by
  unfold Replay.readOpening at h
  obtain ⟨_, h1, -⟩ := Bridge.Except.bind_ok h
  have h1 := Bridge.Except.ensure_ok h1
  simp only [Bool.and_eq_true, beq_iff_eq, ← Array.all_toList, List.all_eq_true, Array.toList_zip] at h1
  refine (map_eq_of_zip (by simpa using h1.1.1.2.symm) fun p hp => ?_).symm
  exact (h1.2 p hp).symm

private theorem readOpening_evals {shape : WhirShape} {g : Replay.Opening} (h : Replay.readOpening shape g = .ok ()) :
    g.evals.size = shape.batches.size := by
  simpa using congrArg List.length (readOpening_sizes h)

private theorem foldl_observePair (r : Replay.Duplex) (a b : Ext) :
    (#[a, b].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r =
      (#[b].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe
        ((#[a].flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Replay.Duplex.observe r) := by
  rw [foldl_observeExt, foldl_observeExt, ← Array.foldl_toList, Array.toList_flatMap]
  simp only [List.flatMap_cons, List.flatMap_nil, List.foldl_cons, List.foldl_nil, List.append_nil,
    List.cons_append, List.nil_append]

private theorem runs_observeAll_eq {vs : Array Word} {t t' : Replay.Transcript} {u : Unit}
    (h : Runs (Replay.observeAll vs) t u t') : t' = { t with duplex := vs.foldl Replay.Duplex.observe t.duplex } := by
  cases h
  rfl

/-- Two `observe_ext` calls are the replay's `observeExts` of the pair. -/
private theorem runs_observePair {a b : Ext} {rest : ReplayM β} {tr tr' : Replay.Transcript} {v : β}
    (h : Runs (Replay.observeExt a >>= fun _ => Replay.observeExt b >>= fun _ => rest) tr v tr') :
    Runs (Replay.observeExts #[a, b] >>= fun _ => rest) tr v tr' := by
  obtain ⟨_, t1, h1, h⟩ := Bridge.run_bind h
  obtain ⟨_, t2, h2, h3⟩ := Bridge.run_bind h
  rw [runs_observeAll_eq h1] at h2
  rw [runs_observeAll_eq h2] at h3
  refine Bridge.runs_bind (Bridge.runs_observeExts _ _) ?_
  rw [foldl_observePair]
  exact h3

private theorem isExts_iff {xs : Array ExtVar} {gs : Array Ext} :
    IsExts c n xs gs ↔ List.Forall₂ (IsExt c n) xs.toList gs.toList := by
  rw [List.forall₂_iff_get]
  constructor
  · intro h
    refine ⟨by simp [h.1], fun i h1 h2 => ?_⟩
    have := h.2 i (by simpa using h1)
    rw [getElem!_pos xs i (by simpa using h1), getElem!_pos gs i (by simpa using h2)] at this
    simpa using this
  · intro h
    have size : xs.size = gs.size := by simpa using h.1
    refine ⟨size, fun i hi => ?_⟩
    rw [getElem!_pos xs i hi, getElem!_pos gs i (size ▸ hi)]
    simpa using h.2 i (by simpa using hi) (by simpa [← size] using hi)

private theorem forall₂_zip {xs ys : List ExtVar} {gs hs : List Ext} (hx : List.Forall₂ (IsExt c n) xs gs)
    (hy : List.Forall₂ (IsExt c n) ys hs) : List.Forall₂ (IsPair c n) (xs.zip ys) (gs.zip hs) := by
  induction hx generalizing ys hs with
  | nil => exact .nil
  | cons a _ ih =>
      cases hy with
      | nil => exact .nil
      | cons b rest => exact .cons ⟨a, b⟩ (ih rest)

private theorem isExts_pair {a b : ExtVar} {ga gb : Ext} (ha : IsExt c n a ga) (hb : IsExt c n b gb) :
    IsExts c n #[a, b] #[ga, gb] :=
  (IsExts.empty.push ha).push hb

/-- The coordinate groups of opened batches whose sizes are multiples of
three (`batch.chunks(3)`, `verifier.rs:276`). -/
private theorem groups_rel {opened : Array (Array ExtVar)} {evals : Array (Array Ext)}
    (h : List.Forall₂ (IsExts c n) opened.toList evals.toList) (div : ∀ b ∈ evals.toList, b.size % 3 = 0) :
    List.Forall₂ (IsExts c n)
      (opened.flatMap fun batch => (Array.range ((batch.size + 2) / 3)).map fun i =>
        batch.extract (3 * i) (3 * i + 3)).toList
      (evals.flatMap fun batch => (Array.range (batch.size / 3)).map fun i =>
        batch.extract (3 * i) (3 * i + 3)).toList := by
  rw [Array.toList_flatMap, Array.toList_flatMap]
  generalize opened.toList = xs at h
  generalize evals.toList = ys at h div
  induction h with
  | nil => exact .nil
  | @cons x y xs ys hxy _ ih =>
      rw [List.flatMap_cons, List.flatMap_cons]
      refine List.rel_append ?_ (ih fun b hb => div b (List.mem_cons_of_mem _ hb))
      have hs : (x.size + 2) / 3 = y.size / 3 := by
        have := div y List.mem_cons_self
        rw [hxy.1]
        omega
      rw [hs, Array.toList_map, Array.toList_map, List.forall₂_map_left_iff, List.forall₂_map_right_iff,
        List.forall₂_same]
      exact fun i _ => hxy.slice _ _

/-- `flat` (`verifier.rs:295`) against the replay's `flatPairs`. -/
private theorem flat_rel {pairs : Array (ExtVar × ExtVar)} {gpairs : Array (Ext × Ext)}
    {f : ExtVar × ExtVar → Array ExtVar} (hf : ∀ a b, f (a, b) = #[a, b])
    (h : List.Forall₂ (IsPair c n) pairs.toList gpairs.toList) :
    IsExts c n (pairs.flatMap f) (Replay.flatPairs gpairs) := by
  rw [isExts_iff, Replay.flatPairs, Array.toList_flatMap, Array.toList_flatMap]
  generalize pairs.toList = xs at h
  generalize gpairs.toList = ys at h
  induction h with
  | nil => exact .nil
  | @cons x y xs ys hxy _ ih =>
      obtain ⟨a, b⟩ := x
      rw [List.flatMap_cons, List.flatMap_cons, hf]
      exact List.rel_append (.cons hxy.1 (.cons hxy.2 .nil)) ih

private theorem flatPairs_size (pairs : Array (Ext × Ext)) : (Replay.flatPairs pairs).size = 2 * pairs.size := by
  rw [Replay.flatPairs, Array.size_flatMap, ← Array.sum_toList, Array.toList_map]
  simp [List.sum_replicate, Nat.mul_comm]

/-- `openingsAgree` passes on batches of the expected sizes and equal values. -/
private theorem agree_of {opened expected : Array (Array Ext)} {what : String}
    (sizes : opened.map (·.size) = expected.map (·.size))
    (same : ∀ p ∈ (opened.flatten.zip expected.flatten).toList, p.1 = p.2) :
    Replay.openingsAgree opened expected what = .ok () := by
  have all : ((opened.flatten.zip expected.flatten).all fun pair => pair.1 == pair.2) = true := by
    rw [← Array.all_toList, List.all_eq_true]
    intro p hp
    rw [same p hp]
    exact Ext.beq_self p.2
  unfold Replay.openingsAgree
  rw [show (opened.map (·.size) == expected.map (·.size)) = true from beq_iff_eq.mpr sizes, all]
  rfl

/-- `mapM` over a list that `R` relates to `ys`: result `i` meets `S` at `ys[i]`. -/
private theorem mapM₂_ensures {δ ε α : Type} {f : δ → Build α} {R : δ → ε → ℕ → Prop} {S : α → ε → ℕ → Prop}
    (monoR : ∀ x y m m', R x y m → m ≤ m' → R x y m') (monoS : ∀ a y m m', S a y m → m ≤ m' → S a y m')
    (h : ∀ x y m, R x y m → Ensures c m (f x) (S · y)) (hf : ∀ x, Appends (f x)) :
    ∀ (xs : List δ) (ys : List ε) {n : ℕ}, List.Forall₂ (R · · n) xs ys →
      Ensures c n (xs.mapM f) fun as m => List.Forall₂ (S · · m) as ys
  | [], [], _, _ => by
      rw [List.mapM_nil]
      exact Ensures.pure .nil
  | x :: xs, y :: ys, n, hr => by
      obtain ⟨hxy, hrest⟩ := List.forall₂_cons.mp hr
      rw [List.mapM_cons]
      refine (h x y n hxy).bind (fun a m le ha => ?_) (hf x)
        fun _ => Appends.bind (Appends.mapM_list hf xs) fun _ => Appends.pure _
      exact (mapM₂_ensures monoR monoS h hf xs ys (hrest.imp fun _ _ r => monoR _ _ _ _ r le)).bind
        (fun as m' le' has => Ensures.pure (.cons (monoS _ _ _ _ ha le') has)) (Appends.mapM_list hf xs)
        fun _ => Appends.pure _
  | [], _ :: _, _, hr => nomatch hr
  | _ :: _, [], _, hr => nomatch hr

end Facts

/-! ## The WHIR verifier -/

section Verify

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `absorb` (`whir/mod.rs:221-226`): the replay's `observeAll`. -/
theorem Whir.absorb_tracks (hp : Permutes c.perm) (words : Array Word) :
    Tracks c n (Whir.absorb words) (Replay.observeAll words) fun _ _ _ => True := by
  intro d t hd
  unfold Whir.absorb
  rw [← Array.forIn_toList]
  refine Ensures.tbind (Q := fun r m => IsDuplex c m r.2 (words.foldl Replay.Duplex.observe t.duplex)) ?_
    fun _ e _ _ he => ?_
  · refine (Ensures.loopT (fun k _ e m => IsDuplex c m e ((words.toList.take k).foldl Replay.Duplex.observe t.duplex))
      hd fun k hk _ e m _ hI => ?_).post fun _ _ q => ?_
    · refine Ensures.liftBind (const_ensures words.toList[k]) fun w _ _ hw => ?_
      refine Ensures.tbind (Duplex.observe_ensures hp hI.mono hw) fun _ e' _ _ hd' => ?_
      rw [StateT.run_pure]
      refine Ensures.pure ⟨rfl, ?_⟩
      rwa [List.take_succ_eq_append_getElem hk, List.foldl_append, List.foldl_cons, List.foldl_nil]
    · rwa [List.take_of_length_le (by simp), Array.foldl_toList] at q
  · rw [StateT.run_pure]
    exact Ensures.pure ⟨(), _, Bridge.runs_observeAll words t, he, trivial⟩

/-- `query_index` (`whir/mod.rs:399-416`): a sample that its inverse check
keeps from `p - 1`, as the replay draws it. -/
private theorem Whir.queryIndex_tracks (hp : Permutes c.perm) (width : ℕ) :
    Tracks c n (Whir.queryIndex width) (do
        let word ← Replay.sampleWord
        Replay.ensure (!(word + 1 == 0)) "WHIR query sample p - 1"
        Replay.record (.sample word)) fun _ _ _ => True := by
  intro d t hd
  unfold Whir.queryIndex
  refine Ensures.tbind (Duplex.sample_ensures hp hd) fun s e₁ _ _ hs => ?_
  obtain ⟨hs, hd₁⟩ := hs
  refine Ensures.liftBind (const_ensures 1) fun one _ _ hone => ?_
  refine Ensures.liftBind (add_ensures hs.mono hone) fun shifted _ _ hshifted => ?_
  refine Ensures.liftBind (const_ensures 0) fun z _ _ hz => ?_
  refine Ensures.liftBind (extInv_ensures (ga := ⟨t.duplex.sample.1 + 1, 0, 0⟩) fun k => ?_)
    fun _ _ _ hne => ?_
  · fin_cases k
    · exact hshifted.mono
    · exact hz
    · exact hz
  refine Ensures.post Ensures.keeps fun r m h => ?_
  obtain ⟨le, e⟩ := h
  have hne' : t.duplex.sample.1 + 1 ≠ 0 := fun e => hne.1 (by rw [e]; rfl)
  refine ⟨(), { duplex := t.duplex.sample.2, draws := t.draws.push (.sample t.duplex.sample.1) },
    Bridge.runs_bind (Bridge.runs_sampleWord t) (Bridge.runs_bind (Bridge.runs_ensure ?_ _ _) (Bridge.runs_record _ _)),
    by rw [e]; exact hd₁.mono, trivial⟩
  rw [Bool.not_eq_true', Bool.eq_false_iff]
  exact fun hb => hne' (Replay.Word.eq_of_beq hb)

/-- `stir`'s draws (`whir/mod.rs:418-504`): each query index, or none when
the draw count is zero. The rest checks without the duplex. -/
private theorem Whir.stir_tracks (hp : Permutes c.perm) (queries : Array (Array Var))
    (draws indexBits logFoldedDomain : ℕ) (root : DigestVar) (folding : Array ExtVar) (base : Bool) :
    Tracks c n (Whir.stir queries draws indexBits logFoldedDomain root folding base) (Replay.whirQueries draws)
      fun _ _ _ => True := by
  unfold Whir.stir Replay.whirQueries
  dsimp only
  split
  · rename_i h0
    subst h0
    exact Tracks.replay Tracks.ofKeeps fun _ _ _ run => run
  · refine Tracks.bindLast (Q := fun _ _ _ => True) ?_ fun _ _ _ _ _ => Tracks.ofKeeps
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    refine Tracks.loop (fun _ _ _ => True) (by simp [Std.Legacy.Range.size]) trivial fun k _ acc m _ _ => ?_
    exact Tracks.bindLast (Whir.queryIndex_tracks hp _) fun _ _ _ _ _ => Tracks.pure ⟨rfl, trivial⟩

/-- `sumcheck` (`whir/mod.rs:306-332`): the replay's `whirSumcheck` of the
round pairs. -/
private theorem Whir.sumcheck_tracks (hp : Permutes c.perm) {shape : SumcheckShape}
    {rounds : Array (ExtVar × ExtVar)} {pairs : Array (Ext × Ext)} {claim : ExtVar}
    (hr : List.Forall₂ (IsPair c n) rounds.toList pairs.toList) :
    Tracks c n (Whir.sumcheck shape rounds claim) (Replay.whirSumcheck shape pairs) fun _ _ _ => True := by
  have len : rounds.size = pairs.size := by simpa using hr.length_eq
  unfold Whir.sumcheck Replay.whirSumcheck
  by_cases e : rounds.isEmpty = true
  · rw [if_pos e]
    refine (Tracks.pure (b := ()) trivial).replay fun _ _ _ run => ?_
    rw [if_pos (by rw [Array.isEmpty_iff_size_eq_zero] at e ⊢; rw [← len]; exact e)]
    exact run
  · rw [if_neg e]
    refine Tracks.replay (x := Replay.observeAll shape.seed >>= fun _ => pairs.toList.forM fun pair => do
        Replay.observeExts #[pair.1, pair.2]
        let _ ← Replay.sampleExt
        Pure.pure ()) ?_ fun _ _ _ run => ?_
    · refine Tracks.bind (Whir.absorb_tracks hp _) fun _ _ _ _ _ => ?_
      refine Tracks.lift extOne_ensures fun one _ _ _ => ?_
      dsimp only
      refine Tracks.bindLast (Q := fun _ _ _ => True) ?_ fun _ _ _ _ _ => Tracks.pure trivial
      rw [← Array.forIn_toList]
      refine Tracks.loop (fun _ _ _ => True) (by simpa using hr.length_eq) trivial fun k hk acc m _ _ => ?_
      have hpair : IsPair c m rounds.toList[k] pairs.toList[k]! := (forall₂_getElem! hr hk).mono
      generalize pairs.toList[k]! = g at hpair ⊢
      rcases hx : rounds.toList[k] with ⟨a₀, a₁⟩
      rw [hx] at hpair
      dsimp only
      refine Tracks.replay (x := Replay.observeExt g.1 >>= fun _ => Replay.observeExt g.2 >>= fun _ =>
          Replay.sampleExt >>= fun _ => Pure.pure ()) ?_ fun _ _ _ run => runs_observePair run
      refine Tracks.bind (Duplex.observeExt_tracks hp hpair.1) fun _ _ _ _ _ => ?_
      refine Tracks.bind (Duplex.observeExt_tracks hp hpair.2.mono) fun _ _ _ _ _ => ?_
      refine Tracks.bind (Duplex.sampleExt_tracks hp) fun r _ _ _ _ => ?_
      quiet_steps
      exact Tracks.pure ⟨rfl, trivial⟩
    · rw [if_neg (by rw [Array.isEmpty_iff_size_eq_zero] at e ⊢; rw [← len]; exact e)]
      exact run

/-- **`whir::verify`** (`whir/mod.rs:506-661`): when its checks hold, the
circuit tracks the replay's transcript of the opening `g`, and the opened
values are `g`'s. The key has the shape's batches, and `g` has the shape's
counts (`readOpening`, which `OpeningView::read` gives). -/
theorem whirVerify_tracks (hp : Permutes c.perm) {shape : WhirShape} {key : WhirKey} {root : DigestVar}
    {view : OpeningView} {points : Array (Array ExtVar)} {g : Replay.Opening} {w : Var → Word}
    (ho : IsOpening c w n view g) (hread : Replay.readOpening shape g = .ok ())
    (hbatches : key.batches.size = shape.batches.size) :
    Tracks c n (Circuit.whirVerify shape key root view points) (Replay.whirTranscript shape g)
      fun r _ m => List.Forall₂ (IsExts c m) r.toList g.evals.toList := by
  obtain ⟨hood, hevals, hinitial, hrounds, hfinalPoly, -, hfinal⟩ := ho
  unfold Circuit.whirVerify Replay.whirTranscript
  dsimp only
  refine Tracks.bind (Q := fun _ _ _ => True) ?_ fun _ _ _ _ _ => ?_
  · rw [← Array.forIn_toList]
    refine Tracks.loop (fun _ _ _ => True) (by simpa using hood.1) trivial fun k hk acc m _ _ => ?_
    have ha : IsExt c n view.initialOod.toList[k] g.initialOod.toList[k]! := by
      have := hood.2 k (by simpa using hk)
      rwa [getElem!_pos view.initialOod k (by simpa using hk), ← Array.getElem_toList,
        ← Array.getElem!_toList] at this
    refine Tracks.bind (Whir.absorb_tracks hp _) fun _ _ _ _ _ => ?_
    refine Tracks.bind (Duplex.sampleExt_tracks hp) fun _ _ _ _ _ => ?_
    quiet_steps
    exact Tracks.bindLast (Duplex.observeExt_tracks hp ha.mono) fun _ _ _ _ _ => Tracks.pure ⟨rfl, trivial⟩
  refine Tracks.bind (Q := fun _ _ _ => True) ?_ fun _ _ _ _ _ => ?_
  · rw [← Array.forIn_toList]
    refine Tracks.loop (ys := (shape.batches.zip g.evals).toList) (fun _ _ _ => True)
      (by simp [hbatches, readOpening_evals hread]) trivial fun k hk acc m _ _ => ?_
    have hk' : k < key.batches.size := by simpa using hk
    have hs : k < shape.batches.size := hbatches ▸ hk'
    have he : k < g.evals.size := (readOpening_evals hread).symm ▸ hs
    have e : (shape.batches.zip g.evals).toList[k]! = (shape.batches[k]!, g.evals[k]!) := by
      rw [Array.getElem!_toList, getElem!_pos (shape.batches.zip g.evals) k (by simp [hs, he]),
        Array.getElem_zip, getElem!_pos shape.batches k hs, getElem!_pos g.evals k he]
    rw [e]
    simp only [Array.getElem_toList, Array.getElem_zipIdx, Nat.zero_add]
    refine Tracks.bind (Whir.absorb_tracks hp _) fun _ _ _ _ _ => ?_
    refine Tracks.bindLast (observeExts_tracks hp (IsExts.getElem! hevals k).mono) fun _ _ _ _ _ => ?_
    quiet_steps
    exact Tracks.pure ⟨rfl, trivial⟩
  refine Tracks.bind (Whir.absorb_tracks hp _) fun _ _ _ _ _ => ?_
  refine Tracks.bind (Whir.absorb_tracks hp _) fun _ _ _ _ _ => ?_
  refine Tracks.bind (Duplex.sampleExt_tracks hp) fun _ _ _ _ _ => ?_
  quiet_steps
  refine Tracks.bind (Whir.sumcheck_tracks hp (hinitial.imp fun _ _ x => x.mono)) fun x _ _ _ _ => ?_
  obtain ⟨initial, next⟩ := x
  dsimp only
  refine Tracks.bind (Q := fun _ _ _ => True) ?_ fun _ _ _ _ _ => ?_
  · rw [← Array.forIn_toList]
    have rounds := hrounds.length_eq
    simp only [Array.length_toList] at rounds
    refine Tracks.loop (ys := (shape.rounds.zip g.rounds).toList) (fun _ _ _ => True) (by simp [rounds]) trivial
      fun k hk acc m _ _ => ?_
    have hk' : k < shape.rounds.size ∧ k < view.rounds.size := by simpa using hk
    have hg : k < g.rounds.size := by omega
    obtain ⟨hroot, hood', hsum, -⟩ : IsRound c w n view.rounds[k] g.rounds[k]! :=
      forall₂_getElem!_array hrounds hk'.2
    have e : (shape.rounds.zip g.rounds).toList[k]! = (shape.rounds[k], g.rounds[k]!) := by
      rw [Array.getElem!_toList, getElem!_pos (shape.rounds.zip g.rounds) k (by simp [hk'.1, hg]),
        Array.getElem_zip, getElem!_pos g.rounds k hg]
    rw [e]
    simp only [Array.getElem_toList, Array.getElem_zipIdx, Nat.zero_add, Array.getElem_zip]
    refine Tracks.bind (Duplex.observeSlice_tracks hp (IsWords.ofFn hroot.1 fun i => (hroot.2 i).mono))
      fun _ _ _ _ _ => ?_
    refine Tracks.bind (Q := fun _ _ _ => True) ?_ fun _ _ _ _ _ => ?_
    · rw [← Array.forIn_toList]
      refine Tracks.loop (fun _ _ _ => True) (by simpa using hood'.1) trivial fun j hj acc' m' _ _ => ?_
      have ha : IsExt c n view.rounds[k].ood.toList[j] g.rounds[k]!.ood.toList[j]! := by
        have := hood'.2 j (by simpa using hj)
        rwa [getElem!_pos view.rounds[k].ood j (by simpa using hj), ← Array.getElem_toList,
          ← Array.getElem!_toList] at this
      refine Tracks.bind (Duplex.sampleExt_tracks hp) fun _ _ _ _ _ => ?_
      quiet_steps
      exact Tracks.bindLast (Duplex.observeExt_tracks hp ha.mono) fun _ _ _ _ _ => Tracks.pure ⟨rfl, trivial⟩
    refine Tracks.bind (Whir.stir_tracks hp _ _ _ _ _ _ _) fun _ _ _ _ _ => ?_
    refine Tracks.bind (Duplex.sampleExt_tracks hp) fun _ _ _ _ _ => ?_
    quiet_steps
    refine Tracks.bindLast (Whir.sumcheck_tracks hp (hsum.imp fun _ _ x => x.mono)) fun x _ _ _ _ => ?_
    obtain ⟨folding, next⟩ := x
    exact Tracks.pure ⟨rfl, trivial⟩
  refine Tracks.bind (observeExts_tracks hp hfinalPoly.mono) fun _ _ _ _ _ => ?_
  refine Tracks.bind (Whir.stir_tracks hp _ _ _ _ _ _ _) fun _ _ _ _ _ => ?_
  quiet_steps
  refine Tracks.bindLast (Whir.sumcheck_tracks hp (hfinal.imp fun _ _ x => x.mono)) fun x _ _ _ _ => ?_
  obtain ⟨finalRandomness, next⟩ := x
  dsimp only
  quiet_steps
  exact Tracks.pure (hevals.imp fun _ _ x => x.mono)

end Verify

/-! ## The openings -/

section Openings

variable [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ}

/-- `from_coordinates` (`verifier.rs:352-362`). -/
private theorem fromCoordinates_ensures {values : Array ExtVar} {gvalues : Array Ext}
    (hv : IsExts c n values gvalues) :
    Ensures c n (Circuit.fromCoordinates values) fun r m => IsExt c m r (Replay.fromCoordinates gvalues) := by
  unfold Circuit.fromCoordinates Replay.fromCoordinates
  refine extZero_ensures.bind fun total _ _ hzero => ?_
  refine (Ensures.loopArray (fun k acc m => IsExt c m acc (Replay.sumRange k fun i => Ext.basis i * gvalues[i]!))
    hzero fun k hk acc m _ hacc => ?_).bind fun r _ _ hr => Ensures.pure ?_
  · have hk' : k < values.size := by simpa using hk
    rw [show values.zipIdx[k] = (values[k]!, k) by simp [getElem!_pos, hk']]
    refine (extConstant_ensures (Ext.basis k)).bind fun basis _ _ hbasis => ?_
    refine (extMul_ensures hbasis (hv.2 k hk').mono).bind fun term _ _ hterm => ?_
    exact (extAdd_ensures hacc.mono hterm).bind fun total' _ _ htotal =>
      Ensures.pure ⟨rfl, by rw [sumRange_succ]; exact htotal⟩
  · rwa [Array.size_zipIdx, hv.1] at hr

/-- The equality checks of `verifier.rs:280-282` and `verifier.rs:312-314`:
once they hold, each pair's values are equal. The replay takes no step. -/
private theorem assertPairs_tracks {xs : Array (ExtVar × ExtVar)} {gs : Array (Ext × Ext)}
    {body : ExtVar × ExtVar → PUnit → Transcript (ForInStep PUnit)}
    (hbody : ∀ a b u, body (a, b) u = (do liftM (Build.assertExtEqual a b); Pure.pure (ForInStep.yield PUnit.unit)))
    (hl : List.Forall₂ (IsPair c n) xs.toList gs.toList) :
    Tracks c n (forIn xs PUnit.unit body) (Pure.pure PUnit.unit) fun _ _ _ => ∀ p ∈ gs.toList, p.1 = p.2 := by
  intro d t hd
  have len := hl.length_eq
  rw [← Array.forIn_toList]
  refine (Ensures.loopT (fun k _ e m => IsDuplex c m e t.duplex ∧ ∀ j < k, (gs.toList[j]!).1 = (gs.toList[j]!).2)
    ⟨hd, fun _ h => absurd h (by omega)⟩ (fun k hk _ e m _ hI => ?_)
    fun x u _ => by obtain ⟨a, b⟩ := x; rw [hbody]; appends).post fun _ _ q => ?_
  · obtain ⟨he, hprev⟩ := hI
    have hpair : IsPair c m xs.toList[k] gs.toList[k]! := (forall₂_getElem! hl hk).mono
    rcases hx : xs.toList[k] with ⟨a, b⟩
    rw [hx] at hpair
    rw [hbody]
    refine Ensures.liftBind (assertExtEqual_ensures hpair.1 hpair.2) fun _ _ _ heq => ?_
    rw [StateT.run_pure]
    refine Ensures.pure ⟨rfl, he.mono, fun j hj => ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.mp hj with lt | rfl
    · exact hprev j lt
    · exact heq
  · obtain ⟨he, hall⟩ := q
    refine ⟨(), t, Bridge.runs_pure _ _, he, fun p hp => ?_⟩
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.mp hp
    have := hall j (by rw [len]; exact hj)
    rwa [getElem!_pos gs.toList j hj] at this

/-- **The openings** (`verifier.rs:266-315`): when the checks hold, the
circuit tracks the replay's two WHIR transcripts and its two opening checks
(`openingsAgree`), the tail of `Replay.verify`.

The handles hold the replay's values field by field. Both openings pass
`readOpening`, and the honest, early and late outputs have the proof's
sizes. The keys have their shapes' batch counts. The relation's P1 batches
are triples of coordinates, one per expected value, and its P0 batches have
the sizes of the expected values; Rust asserts both (`verifier.rs:279`,
`verifier.rs:308-311`). -/
theorem openingsStage_tracks (hp : Permutes c.perm) {rel : Replay.Relation} {p1Key p0Key : WhirKey}
    {o : OpeningsInput} {proof : Replay.Proof} {early : Replay.EarlyOut} {lin : Replay.LinearOut}
    {honest : Replay.HonestOut} {late : Replay.LateOut} {w : Var → Word}
    (h1 : IsOpening c w n o.p1Opening proof.p1Opening) (h0 : IsOpening c w n o.p0Opening proof.p0Opening)
    (hpartials : List.Forall₂ (IsExts c n) o.partials.toList proof.partials.toList)
    (halpha : IsExts c n o.alpha honest.alpha) (hqueried : IsExts c n o.queried honest.queried)
    (homega : IsExt c n o.omegaValue lin.omegaValue) (hlateOmega : IsExt c n o.lateOmega late.openings.omega)
    (hzLeaf : IsExt c n o.zLeaf early.zLeaf) (hzValue : IsExt c n o.zValue lin.zValue)
    (hruns : IsExts c n o.earlyValues.runs #[early.values.eRe, early.values.eIm, early.values.row])
    (hpairs : List.Forall₂ (IsPair c n) o.earlyValues.pairs.toList early.values.pairs.toList)
    (hblock : IsExt c n o.earlyValues.block early.values.block)
    (hlateValues : List.Forall₂ (IsPair c n) o.lateValues.toList late.pairs.toList)
    (hlateBlock : IsExt c n o.lateBlock late.openings.block) (hmult : IsExt c n o.mult early.openings.mult)
    (r1 : Replay.readOpening rel.p1Whir proof.p1Opening = .ok ())
    (r0 : Replay.readOpening rel.p0Whir proof.p0Opening = .ok ())
    (hqsize : honest.queried.size = 2 * rel.queries) (hesize : early.values.pairs.size = rel.matrices)
    (hlsize : late.pairs.size = rel.matrices)
    (k1 : p1Key.batches.size = rel.p1Whir.batches.size) (k0 : p0Key.batches.size = rel.p0Whir.batches.size)
    (hp1 : ∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0)
    (hp1count : (rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4)
    (hp0 : rel.p0Whir.batches.toList.map (·.size) = [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1]) :
    Tracks c n (Circuit.openingsStage rel p1Key p0Key o) (do
        Replay.whirTranscript rel.p1Whir proof.p1Opening
        liftExcept (Replay.openingsAgree #[Replay.p1Values proof.p1Opening.evals]
          #[Replay.p1Expected proof honest lin late] "P1 openings")
        Replay.whirTranscript rel.p0Whir proof.p0Opening
        liftExcept (Replay.openingsAgree proof.p0Opening.evals (Replay.p0Expected early lin late) "P0 openings"))
      fun _ _ _ => True := by
  have div : ∀ b ∈ proof.p1Opening.evals.toList, b.size % 3 = 0 := fun b hb => by
    have hm : b.size ∈ rel.p1Whir.batches.toList.map (·.size) := by
      rw [← readOpening_sizes r1]
      exact List.mem_map_of_mem hb
    obtain ⟨b', hb', e⟩ := List.mem_map.mp hm
    rw [← e]
    exact hp1 b' (Array.mem_toList_iff.mp hb')
  unfold Circuit.openingsStage
  dsimp only
  refine Tracks.bind (whirVerify_tracks hp h1 r1 k1) fun opened _ _ _ hopened => ?_
  refine Tracks.lift (combinePartials_ensures (IsExts.getElem! hpartials 0).mono halpha.mono)
    fun keyGroup _ _ hkey => ?_
  refine Tracks.lift (combinePartials_ensures (IsExts.getElem! hpartials 1).mono halpha.mono)
    fun runGroup _ _ hrun => ?_
  refine Tracks.lift (Q := fun a m => IsExts c m a (Replay.p1Values proof.p1Opening.evals)) ?_
    fun values _ _ hvalues => ?_
  · rw [Array.mapM_eq_mapM_toList]
    refine (mapM₂_ensures (R := fun x y m => IsExts c m x y)
      (S := fun a y m => IsExt c m a (Replay.fromCoordinates y)) (fun _ _ _ _ h le => h.mono le)
      (fun _ _ _ _ h le => h.mono le) (fun _ _ _ hq => fromCoordinates_ensures hq) (fun _ => inferInstance) _
      (proof.p1Opening.evals.flatMap fun batch => (Array.range (batch.size / 3)).map fun i =>
        batch.extract (3 * i) (3 * i + 3)).toList (groups_rel (hopened.imp fun _ _ x => x.mono) div)).map fun as m hf => ?_
    have e : Replay.p1Values proof.p1Opening.evals = (proof.p1Opening.evals.flatMap fun batch =>
        (Array.range (batch.size / 3)).map fun i => batch.extract (3 * i) (3 * i + 3)).map
          Replay.fromCoordinates := by
      rw [Array.map_flatMap]
      simp only [Array.map_map]
      rfl
    rw [isExts_iff, List.toList_toArray, e, Array.toList_map, List.forall₂_map_right_iff]
    exact hf
  refine Tracks.quiet (assertPairs_tracks (gs := (Replay.p1Values proof.p1Opening.evals).zip
      (Replay.p1Expected proof honest lin late)) (fun _ _ _ => rfl) ?_) fun _ _ _ same => ?_
  · rw [Array.toList_zip, Array.toList_zip]
    exact forall₂_zip (isExts_iff.mp hvalues) (isExts_iff.mp
      (((isExts_pair hkey.mono hrun.mono).append hqueried.mono).append (isExts_pair homega.mono hlateOmega.mono)))
  refine Tracks.check (agree_of ?_ ?_) ?_
  · have e1 : (Replay.p1Values proof.p1Opening.evals).size = 2 * rel.queries + 4 := by
      rw [Replay.p1Values, Array.size_flatMap, ← Array.sum_toList, Array.toList_map]
      simp only [Array.size_map, Array.size_range]
      rw [show (proof.p1Opening.evals.toList.map fun a => a.size / 3) =
          (proof.p1Opening.evals.toList.map (·.size)).map (· / 3) by simp [Function.comp_def],
        readOpening_sizes r1, List.map_map]
      exact hp1count
    have e2 : (Replay.p1Expected proof honest lin late).size = 2 * rel.queries + 4 := by
      simp [Replay.p1Expected, hqsize]
      omega
    simp [e1, e2]
  · rw [Array.flatten_singleton, Array.flatten_singleton]
    exact same
  refine Tracks.bind (whirVerify_tracks hp h0.mono r0 k0) fun opened _ _ _ hopened => ?_
  refine Tracks.quiet (assertPairs_tracks (gs := proof.p0Opening.evals.flatten.zip
      (Replay.p0Expected early lin late).flatten) (fun _ _ _ => rfl) ?_) fun _ _ _ same => ?_
  · rw [Array.toList_zip, Array.toList_zip, Array.toList_flatten, Array.toList_flatten, Array.toList_flatten,
      Array.toList_flatten]
    refine forall₂_zip (List.rel_flatten ?_) (List.rel_flatten ?_) <;>
      rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff]
    · exact hopened.imp fun _ _ x => isExts_iff.mp x
    · refine .cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ .nil)))))) <;>
        refine isExts_iff.mp ?_
      · exact IsExts.empty.push hzLeaf.mono
      · exact IsExts.empty.push hzValue.mono
      · exact isExts_pair ((hruns.2 0 (by simp [hruns.1])).mono) ((hruns.2 1 (by simp [hruns.1])).mono)
      · exact flat_rel (fun _ _ => rfl) (hpairs.imp fun _ _ x => x.mono)
      · exact IsExts.empty.push hblock.mono
      · exact (flat_rel (fun _ _ => rfl) (hlateValues.imp fun _ _ x => x.mono)).append
          (IsExts.empty.push hlateBlock.mono)
      · exact IsExts.empty.push hmult.mono
  rw [← bind_pure (liftExcept _ : ReplayM PUnit)]
  refine Tracks.check (agree_of ?_ same) (Tracks.pure trivial)
  apply Array.ext'
  rw [Array.toList_map, Array.toList_map, readOpening_sizes r0, hp0]
  simp [Replay.p0Expected, flatPairs_size, hesize, hlsize]

end Openings

/-! ## The old names, for the files that still use `Simulates` -/

theorem openingsStage_ensures [CharP F goldilocksModulus] {c : Ctx F} {n : ℕ} {P : Prop} (hp : Permutes c.perm)
    {rel : Replay.Relation} {p1Key p0Key : WhirKey} {o : OpeningsInput} {proof : Replay.Proof}
    {early : Replay.EarlyOut} {lin : Replay.LinearOut} {honest : Replay.HonestOut} {late : Replay.LateOut}
    {w : Var → Word} {d : Duplex} {t : Replay.Transcript} (hd : P → IsDuplex c n d t.duplex)
    (h1 : P → IsOpening c w n o.p1Opening proof.p1Opening) (h0 : P → IsOpening c w n o.p0Opening proof.p0Opening)
    (hpartials : P → List.Forall₂ (IsExts c n) o.partials.toList proof.partials.toList)
    (halpha : P → IsExts c n o.alpha honest.alpha) (hqueried : P → IsExts c n o.queried honest.queried)
    (homega : P → IsExt c n o.omegaValue lin.omegaValue) (hlateOmega : P → IsExt c n o.lateOmega late.openings.omega)
    (hzLeaf : P → IsExt c n o.zLeaf early.zLeaf) (hzValue : P → IsExt c n o.zValue lin.zValue)
    (hruns : P → IsExts c n o.earlyValues.runs #[early.values.eRe, early.values.eIm, early.values.row])
    (hpairs : P → List.Forall₂ (IsPair c n) o.earlyValues.pairs.toList early.values.pairs.toList)
    (hblock : P → IsExt c n o.earlyValues.block early.values.block)
    (hlateValues : P → List.Forall₂ (IsPair c n) o.lateValues.toList late.pairs.toList)
    (hlateBlock : P → IsExt c n o.lateBlock late.openings.block) (hmult : P → IsExt c n o.mult early.openings.mult)
    (r1 : P → Replay.readOpening rel.p1Whir proof.p1Opening = .ok ())
    (r0 : P → Replay.readOpening rel.p0Whir proof.p0Opening = .ok ())
    (hqsize : P → honest.queried.size = 2 * rel.queries) (hesize : P → early.values.pairs.size = rel.matrices)
    (hlsize : P → late.pairs.size = rel.matrices)
    (k1 : p1Key.batches.size = rel.p1Whir.batches.size) (k0 : p0Key.batches.size = rel.p0Whir.batches.size)
    (hp1 : ∀ b ∈ rel.p1Whir.batches, b.size % 3 = 0)
    (hp1count : (rel.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * rel.queries + 4)
    (hp0 : rel.p0Whir.batches.toList.map (·.size) = [1, 1, 2, 2 * rel.matrices, 1, 2 * rel.matrices + 1, 1]) :
    OldEnsures c n P ((openingsStage rel p1Key p0Key o).run d) fun res m =>
      ∃ t', Runs (do
          Replay.whirTranscript rel.p1Whir proof.p1Opening
          liftExcept (Replay.openingsAgree #[Replay.p1Values proof.p1Opening.evals]
            #[Replay.p1Expected proof honest lin late] "P1 openings")
          Replay.whirTranscript rel.p0Whir proof.p0Opening
          liftExcept (Replay.openingsAgree proof.p0Opening.evals (Replay.p0Expected early lin late) "P0 openings"))
        t () t' ∧ IsDuplex c m res.2 t'.duplex :=
  .of_ensures fun h => (openingsStage_tracks hp (h1 h) (h0 h) (hpartials h) (halpha h) (hqueried h) (homega h)
    (hlateOmega h) (hzLeaf h) (hzValue h) (hruns h) (hpairs h) (hblock h) (hlateValues h) (hlateBlock h) (hmult h)
    (r1 h) (r0 h) (hqsize h) (hesize h) (hlsize h) k1 k0 hp1 hp1count hp0 d t (hd h)).post
    fun _ _ ⟨_, t', run, hdx, _⟩ => ⟨t', run, hdx⟩

end CompressionLean.Sim
