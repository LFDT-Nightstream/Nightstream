import CompressionLean.Replay.Stages

/-!
Owns what a successful replay run says: `Runs x t a t'` when the replay
action `x`, started from the transcript `t`, returns `a` and ends in `t'`.
A run decomposes along `>>=` (`run_bind`); observing leaves the draws, a
sample appends exactly its draw, and a check that passed held. The `runs_*`
lemmas build a run forward, step by step, as the circuit's simulation does.
`Grows` says that an action only appends draws; the WHIR transcript does
(`whirTranscript_grows`), so its draws follow the model's.
-/

namespace CompressionLean.Replay

/-- The result and the final transcript of `x` run from `t`. -/
def eval {α : Type} (x : ReplayM α) (t : Transcript) : Except String α × Transcript := (x.run.run t).run

theorem eval_bind {α β : Type} (x : ReplayM α) (f : α → ReplayM β) (t : Transcript) :
    eval (x >>= f) t = match eval x t with
      | (.ok a, s) => eval (f a) s
      | (.error e, s) => (.error e, s) := by
  unfold eval
  rw [ExceptT.run_bind, StateT.run_bind, Id.run_bind]
  rcases (ExceptT.run x).run t with ⟨e, s⟩
  cases e <;> rfl

theorem eval_map {α β : Type} (f : α → β) (x : ReplayM α) (t : Transcript) :
    eval (f <$> x) t = match eval x t with
      | (.ok a, s) => (.ok (f a), s)
      | (.error e, s) => (.error e, s) := by
  unfold eval
  rw [ExceptT.run_map, StateT.run_map, Id.run_map]
  rcases (ExceptT.run x).run t with ⟨e, s⟩
  cases e <;> rfl

/-- `x` run from `t` returns `a` and ends in `t'`. -/
def Runs {α : Type} (x : ReplayM α) (t : Transcript) (a : α) (t' : Transcript) : Prop :=
  eval x t = (.ok a, t')

theorem run_bind {α β : Type} {x : ReplayM α} {f : α → ReplayM β} {t : Transcript} {b : β} {t'' : Transcript}
    (h : Runs (x >>= f) t b t'') : ∃ a t', Runs x t a t' ∧ Runs (f a) t' b t'' := by
  unfold Runs at *
  rw [eval_bind] at h
  rcases hx : eval x t with ⟨e, t'⟩
  rw [hx] at h
  cases e with
  | error err => cases h
  | ok a => exact ⟨a, t', rfl, h⟩

theorem run_map {α β : Type} {f : α → β} {x : ReplayM α} {t : Transcript} {b : β} {t' : Transcript}
    (h : Runs (f <$> x) t b t') : ∃ a, Runs x t a t' ∧ b = f a := by
  unfold Runs at *
  rw [eval_map] at h
  rcases hx : eval x t with ⟨e, s⟩
  rw [hx] at h
  cases e with
  | error err => cases h
  | ok a =>
      cases h
      exact ⟨a, rfl, rfl⟩

theorem run_pure {α : Type} {a b : α} {t t' : Transcript} (h : Runs (Pure.pure a) t b t') : b = a ∧ t' = t := by
  cases h
  exact ⟨rfl, rfl⟩

theorem run_observeAll {vs : Array Gl} {t t' : Transcript} {u : Unit} (h : Runs (observeAll vs) t u t') :
    t'.draws = t.draws := by
  cases h
  rfl

theorem run_observeExts {vs : Array Ext} {t t' : Transcript} {u : Unit} (h : Runs (observeExts vs) t u t') :
    t'.draws = t.draws :=
  run_observeAll h

theorem run_observeExt {v : Ext} {t t' : Transcript} {u : Unit} (h : Runs (observeExt v) t u t') :
    t'.draws = t.draws :=
  run_observeAll h

theorem run_sampleWord {t t' : Transcript} {w : Gl} (h : Runs sampleWord t w t') : t'.draws = t.draws := by
  cases h
  rfl

theorem run_record {t t' : Transcript} {u : Unit} {d : Draw} (h : Runs (record d) t u t') :
    t'.draws = t.draws.push d := by
  cases h
  rfl

theorem run_ensure {ok : Bool} {what : String} {t t' : Transcript} {u : Unit} (h : Runs (ensure ok what) t u t') :
    ok = true ∧ t' = t := by
  cases ok
  · cases h
  · cases h
    exact ⟨rfl, rfl⟩

theorem run_liftExcept {α : Type} {e : Except String α} {a : α} {t t' : Transcript}
    (h : Runs (liftExcept e) t a t') : e = .ok a ∧ t' = t := by
  cases e
  · cases h
  · cases h
    exact ⟨rfl, rfl⟩

theorem run_drawExt {t t' : Transcript} {v : Ext} (h : Runs drawExt t v t') : t'.draws = t.draws := by
  obtain ⟨_, t1, h1, h⟩ := run_bind h
  obtain ⟨_, t2, h2, h⟩ := run_bind h
  obtain ⟨_, t3, h3, h⟩ := run_bind h
  obtain ⟨-, rfl⟩ := run_pure h
  rw [run_sampleWord h3, run_sampleWord h2, run_sampleWord h1]

theorem run_sampleExt {t t' : Transcript} {v : Ext} (h : Runs sampleExt t v t') :
    t'.draws.toList = t.draws.toList ++ [.ext v] := by
  obtain ⟨w, t1, h1, h⟩ := run_bind h
  obtain ⟨_, t2, h2, h⟩ := run_bind h
  obtain ⟨rfl, rfl⟩ := run_pure h
  rw [run_record h2, Array.toList_push, run_drawExt h1]

theorem run_samplePair {t t' : Transcript} {v : Ext × Ext} (h : Runs samplePair t v t') :
    t'.draws.toList = t.draws.toList ++ [.pair v.1 v.2] := by
  obtain ⟨a, t1, h1, h⟩ := run_bind h
  obtain ⟨b, t2, h2, h⟩ := run_bind h
  obtain ⟨_, t3, h3, h⟩ := run_bind h
  obtain ⟨rfl, rfl⟩ := run_pure h
  rw [run_record h3, Array.toList_push, run_drawExt h2, run_drawExt h1]

theorem run_wordsLoop :
    ∀ (l : List ℕ) (init : Array Gl) (t t' : Transcript) (ws : Array Gl),
      Runs (l.foldlM (fun values _ => do pure (values.push (← sampleWord))) init) t ws t' →
        t'.draws = t.draws ∧ ws.size = init.size + l.length := by
  intro l
  induction l with
  | nil =>
      intro init t t' ws h
      obtain ⟨rfl, rfl⟩ := run_pure h
      exact ⟨rfl, by simp⟩
  | cons i rest ih =>
      intro init t t' ws h
      rw [List.foldlM_cons] at h
      obtain ⟨values, t1, h1, h⟩ := run_bind h
      obtain ⟨w, t2, h2, h3⟩ := run_bind h1
      obtain ⟨rfl, rfl⟩ := run_pure h3
      obtain ⟨draws, size⟩ := ih _ _ _ _ h
      refine ⟨by rw [draws, run_sampleWord h2], by rw [size, Array.size_push, List.length_cons]; omega⟩

theorem run_sampleWords {n : ℕ} {t t' : Transcript} {ws : Array Gl} (h : Runs (sampleWords n) t ws t') :
    t'.draws.toList = t.draws.toList ++ [.words ws] ∧ ws.size = n := by
  obtain ⟨values, t1, h1, h⟩ := run_bind h
  obtain ⟨_, t2, h2, h⟩ := run_bind h
  obtain ⟨rfl, rfl⟩ := run_pure h
  obtain ⟨draws, size⟩ := run_wordsLoop _ _ _ _ _ h1
  refine ⟨by rw [run_record h2, Array.toList_push, draws], by simpa using size⟩

/-! ## Pure checks -/

theorem Except.bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem Except.pure_eq_ok {ε α : Type} {a b : α} (h : (pure a : Except ε α) = .ok b) : a = b := by
  cases h
  rfl

theorem Except.ensure_ok {ok : Bool} {what : String} {u : Unit} (h : (ensure ok what : Except String Unit) = .ok u) :
    ok = true := by
  cases ok
  · cases h
  · rfl

/-! ## Runs forward -/

theorem runs_bind {α β : Type} {x : ReplayM α} {f : α → ReplayM β} {t t' t'' : Transcript} {a : α} {b : β}
    (hx : Runs x t a t') (hf : Runs (f a) t' b t'') : Runs (x >>= f) t b t'' := by
  unfold Runs at *
  rw [eval_bind, hx]
  exact hf

theorem runs_pure {α : Type} (a : α) (t : Transcript) : Runs (Pure.pure a) t a t := rfl

theorem runs_sampleWord (t : Transcript) :
    Runs sampleWord t t.duplex.sample.1 { t with duplex := t.duplex.sample.2 } := rfl

theorem runs_drawExt (t : Transcript) :
    Runs drawExt t ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩
      { t with duplex := t.duplex.sample.2.sample.2.sample.2 } := rfl

theorem runs_sampleExt (t : Transcript) :
    let v : Ext := ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩
    Runs sampleExt t v { duplex := t.duplex.sample.2.sample.2.sample.2, draws := t.draws.push (.ext v) } := rfl

theorem runs_samplePair (t : Transcript) :
    let first : Ext := ⟨t.duplex.sample.1, t.duplex.sample.2.sample.1, t.duplex.sample.2.sample.2.sample.1⟩
    let d := t.duplex.sample.2.sample.2.sample.2
    let second : Ext := ⟨d.sample.1, d.sample.2.sample.1, d.sample.2.sample.2.sample.1⟩
    Runs samplePair t (first, second)
      { duplex := d.sample.2.sample.2.sample.2, draws := t.draws.push (.pair first second) } := rfl

theorem runs_record (d : Draw) (t : Transcript) : Runs (record d) t () { t with draws := t.draws.push d } := rfl

theorem runs_observeAll (vs : Array Gl) (t : Transcript) :
    Runs (observeAll vs) t () { t with duplex := vs.foldl Duplex.observe t.duplex } := rfl

theorem runs_observeExts (vs : Array Ext) (t : Transcript) :
    Runs (observeExts vs) t ()
      { t with duplex := (vs.flatMap fun v => #[v.c0, v.c1, v.c2]).foldl Duplex.observe t.duplex } := rfl

theorem runs_ensure {ok : Bool} (h : ok = true) (what : String) (t : Transcript) : Runs (ensure ok what) t () t := by
  subst h
  rfl

theorem runs_liftExcept {α : Type} {e : Except String α} {a : α} (h : e = .ok a) (t : Transcript) :
    Runs (liftExcept e) t a t := by
  subst h
  rfl

/-! ## Draws only grow -/

/-- A successful run of `x` only appends draws. -/
def Grows {α : Type} (x : ReplayM α) : Prop :=
  ∀ t a t', Runs x t a t' → ∃ ds : List Draw, t'.draws.toList = t.draws.toList ++ ds

theorem grows_bind {α β : Type} {x : ReplayM α} {f : α → ReplayM β} (hx : Grows x) (hf : ∀ a, Grows (f a)) :
    Grows (x >>= f) := by
  intro t b t'' h
  obtain ⟨a, t', h1, h2⟩ := run_bind h
  obtain ⟨d1, e1⟩ := hx _ _ _ h1
  obtain ⟨d2, e2⟩ := hf a _ _ _ h2
  exact ⟨d1 ++ d2, by rw [e2, e1, List.append_assoc]⟩

theorem grows_of_draws {α : Type} {x : ReplayM α} (same : ∀ t a t', Runs x t a t' → t'.draws = t.draws) :
    Grows x := fun t a t' h => ⟨[], by rw [same t a t' h, List.append_nil]⟩

theorem grows_pure {α : Type} (a : α) : Grows (Pure.pure a : ReplayM α) :=
  grows_of_draws fun _ _ _ h => by rw [(run_pure h).2]

theorem grows_map {α β : Type} (f : α → β) {x : ReplayM α} (hx : Grows x) : Grows (f <$> x) := by
  intro t b t' h
  obtain ⟨a, h1, -⟩ := run_map h
  exact hx _ _ _ h1

theorem grows_observeAll (vs : Array Gl) : Grows (observeAll vs) :=
  grows_of_draws fun _ _ _ h => run_observeAll h

theorem grows_observeExts (vs : Array Ext) : Grows (observeExts vs) :=
  grows_of_draws fun _ _ _ h => run_observeExts h

theorem grows_observeExt (v : Ext) : Grows (observeExt v) :=
  grows_of_draws fun _ _ _ h => run_observeExt h

theorem grows_sampleWord : Grows sampleWord :=
  grows_of_draws fun _ _ _ h => run_sampleWord h

theorem grows_record (d : Draw) : Grows (record d) :=
  fun _ _ _ h => ⟨[d], by rw [run_record h, Array.toList_push]⟩

theorem grows_sampleExt : Grows sampleExt :=
  fun _ _ _ h => ⟨_, run_sampleExt h⟩

theorem grows_ensure (ok : Bool) (what : String) : Grows (ensure ok what : ReplayM Unit) :=
  grows_of_draws fun _ _ _ h => by rw [(run_ensure h).2]

theorem grows_forM {α : Type} (f : α → ReplayM PUnit) (hf : ∀ a, Grows (f a)) :
    ∀ l : List α, Grows (l.forM f) := by
  intro l
  induction l with
  | nil => exact grows_pure _
  | cons a rest ih => exact grows_bind (hf a) fun _ => ih

theorem grows_ite {α : Type} (c : Prop) [Decidable c] {x y : ReplayM α} (hx : Grows x) (hy : Grows y) :
    Grows (if c then x else y) := by
  split
  · exact hx
  · exact hy

/-- Unfold `Grows` goals along the WHIR transcript's combinators. -/
macro "grows" : tactic =>
  `(tactic| repeat (first
    | with_reducible exact grows_observeAll _ | with_reducible exact grows_observeExts _
    | with_reducible exact grows_observeExt _ | with_reducible exact grows_sampleExt
    | with_reducible exact grows_sampleWord | with_reducible exact grows_record _
    | with_reducible exact grows_ensure _ _ | with_reducible exact grows_pure _
    | with_reducible apply grows_bind | with_reducible apply grows_forM | with_reducible apply grows_ite
    | with_reducible intro))

theorem whirSumcheck_grows (shape : SumcheckShape) (pairs : Array (Ext × Ext)) :
    Grows (whirSumcheck shape pairs) := by
  unfold whirSumcheck
  grows

theorem whirQueries_grows (draws : ℕ) : Grows (whirQueries draws) := by
  unfold whirQueries
  grows

theorem whirTranscript_grows (shape : WhirShape) (o : Opening) : Grows (whirTranscript shape o) := by
  unfold whirTranscript
  repeat (first
    | with_reducible exact whirSumcheck_grows _ _ | with_reducible exact whirQueries_grows _
    | with_reducible exact grows_observeAll _ | with_reducible exact grows_observeExts _
    | with_reducible exact grows_observeExt _ | with_reducible exact grows_sampleExt
    | with_reducible exact grows_pure _
    | with_reducible apply grows_bind | with_reducible apply grows_forM | with_reducible intro)

end CompressionLean.Replay
