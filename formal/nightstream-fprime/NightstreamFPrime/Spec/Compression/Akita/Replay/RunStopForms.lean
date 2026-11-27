import NightstreamFPrime.Spec.Compression.Akita.Replay.RunProof

/-! The game theorems of `Replay/RunProof.lean` for a run of the replay that has no stop before
one operation (Akita revision `d50f700`). The theorems of `RunProof.lean` read `runProof … =
accept`; these read only the run before the operation, so they hold for a proof that the code
rejects at this operation or after it.

- `before_of_replay`, `before_le`, `before_step`: a replay without a stop has no stop before each
  operation; a run without a stop before `n` has none before `j ≤ n`, and the step of each
  operation before `n` has no stop.
- `runs_level`: when the replay has no stop, a game that runs to the result of the level function
  at `fused.end` runs to the recorded `(fused.point, witness_evaluation)`.
- `tensor_game_before`, `tensor_some_of_step`: the tensor reduction at `tensor.end` runs to the
  result of `tensorRun` on the state before it, and a step without a stop gives `some`.
- `trace_game_before`, `traceSq_game_before`: a trace level runs to the result of `foldRunIn` on
  the state before its `fused.end`.
- `runs_bind_inv`, `not_runs_bind_none`: a run of `g.bind f` on a path that starts with a run of
  `g` is a run of the continuation on the rest; when `g` runs to `none` and the continuation of
  `none` is `pure none`, the game does not run to `some`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression

namespace Game

variable {α β : Type}

/-- A run of `g.bind f` on a path that starts with a run of `g` to `a` is a run of `f a` on the
rest of the path. -/
theorem runs_bind_inv {g : Game α} {f : α → Game β} {path rest : List Move} {a : α} {b : β}
    (first : g.Runs path a) (whole : (g.bind f).Runs (path ++ rest) b) : (f a).Runs rest b := by
  induction g generalizing path with
  | pure x =>
      cases path with
      | nil =>
          have equal : x = a := first
          subst equal
          exact whole
      | cons m path => cases m <;> exact first.elim
  | draw C bad next step =>
      cases path with
      | nil => exact first.elim
      | cons m path =>
          cases m with
          | draw c =>
              obtain ⟨same, first⟩ := first
              obtain ⟨_, whole⟩ := whole
              exact step _ first whole
          | recv m => exact first.elim
  | recv M next step =>
      cases path with
      | nil => exact first.elim
      | cons m path =>
          cases m with
          | draw c => exact first.elim
          | recv m =>
              obtain ⟨same, first⟩ := first
              obtain ⟨_, whole⟩ := whole
              exact step _ first whole

/-- When `g` runs to `none` on the start of a path, a game that ends in `none` after a result
`none` of `g` does not run to `some` on the path. -/
theorem not_runs_bind_none {γ : Type} {g : Game (Option α)} {f : Option α → Game (Option γ)}
    (ends : f none = .pure none) {path rest : List Move} (first : g.Runs path none) {c : γ}
    (whole : (g.bind f).Runs (path ++ rest) (some c)) : False := by
  have run := runs_bind_inv first whole
  rw [ends] at run
  cases rest with
  | nil =>
      have equal : (none : Option γ) = some c := run
      cases equal
  | cons m rest => cases m <;> exact run.elim

end Game

namespace Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

section Forms

variable (pub : Public) (proof : ByteArray) (messages : Array Message) {ops : Array Op}

/-- A replay without a stop has no stop before each operation. -/
theorem before_of_replay (none_ : (replay pub ops proof messages).1 = none) (n : ℕ) :
    (runFrom pub proof messages (ops.zipIdx.toList.take n) (channelStart pub)).1 = .ok () :=
  runFrom_take_ok pub proof messages _ n _ ((replay_none_iff pub ops proof messages).mp none_)

/-- A run without a stop before `n` has no stop before `j ≤ n`. -/
theorem before_le {j n : ℕ} (le : j ≤ n)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take n) (channelStart pub)).1 =
      .ok ()) :
    (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok () := by
  have ran := runFrom_take_ok pub proof messages _ j _ before
  rwa [List.take_take, Nat.min_eq_left le] at ran

/-- In a run without a stop before `n`, the run before an operation `t < n` and its step have no
stop. -/
theorem before_step {t n : ℕ} {op : Op} (at_ : ops[t]? = some op) (lt : t < n)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take n) (channelStart pub)).1 =
      .ok ()) :
    (runFrom pub proof messages (ops.zipIdx.toList.take t) (channelStart pub)).1 = .ok () ∧
      (((step pub proof messages t op).run).run (stateBefore pub proof messages ops t)).1 =
        .ok () := by
  have ran := before_le pub proof messages (Nat.succ_le_of_lt lt) before
  rw [take_succ_zipIdx ops at_, runFrom_append_fst, runFrom_cons_fst] at ran
  exact ⟨ran.1, ran.2.1⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- When the replay has no stop, a game that runs to the result of the level function on the state
before a `fused.end` of the shape `foldShape` runs to the recorded
`(fused.point, witness_evaluation)`. -/
theorem runs_level (none_ : (replay pub ops proof messages).1 = none) {c : ℕ} {key : Key}
    {level : ℕ} {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    {g : Game (Option (List E × E))} {path : List Move}
    (runs : g.Runs path ((foldRunIn pub (stateBefore pub proof messages ops c) level F).map
      fun o => (o.1.map Ext.into, o.2.into))) :
    g.Runs path (some ((recordedIn (stateBefore pub proof messages ops c)
        ⟨level, "fused.point", none⟩).toList.map Ext.into,
      (firstIn (stateBefore pub proof messages ops c) ⟨level, "witness_evaluation", none⟩).into)) := by
  have found : foldRunIn pub (stateBefore pub proof messages ops c) level F =
      some ((recordedIn (stateBefore pub proof messages ops c) ⟨level, "fused.point", none⟩).toList,
        firstIn (stateBefore pub proof messages ops c) ⟨level, "witness_evaluation", none⟩) :=
    replay_level pub ops proof messages none_ at_ shape
  rw [found] at runs
  exact runs

/-- The tensor reduction at `tensor.end` with its sum-check of the shape `roundsAt`, after a run
without a stop before it: `tensorCheck` of the specification runs on the path of the recorded
values of the state before it to the result of `tensorRun`. -/
theorem tensor_game_before (aligned : ReadsAligned messages ops.zipIdx.toList 0) {c : ℕ}
    {level count : ℕ} (point opening : Key)
    (shape : roundsAt (ops.zipIdx.toList.take c) ⟨level, "tensor.sumcheck", none⟩ count 2 = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ()) (w : Table) :
    (tensorCheck count w
        ((recordedIn (stateBefore pub proof messages ops c) point).toList.map (Ext.into (E := E)))
        (firstIn (stateBefore pub proof messages ops c) opening).into).Runs
      (tensorPath E count (recordedIn (stateBefore pub proof messages ops c) point).toList
        (firstIn (stateBefore pub proof messages ops c) opening)
        ((recordedIn (stateBefore pub proof messages ops c)
          ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
        ((recordedIn (stateBefore pub proof messages ops c)
          ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
        (firstIn (stateBefore pub proof messages ops c) ⟨level, "tensor.eta", none⟩)
        (roundsIn (stateBefore pub proof messages ops c) ⟨level, "tensor.sumcheck", none⟩)
        (firstIn (stateBefore pub proof messages ops c) ⟨level, "tensor.final_claims", none⟩))
      ((tensorRun count (recordedIn (stateBefore pub proof messages ops c) point).toList
        (firstIn (stateBefore pub proof messages ops c) opening)
        ((recordedIn (stateBefore pub proof messages ops c)
          ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
        ((recordedIn (stateBefore pub proof messages ops c)
          ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
        (firstIn (stateBefore pub proof messages ops c) ⟨level, "tensor.eta", none⟩)
        (roundsIn (stateBefore pub proof messages ops c) ⟨level, "tensor.sumcheck", none⟩)
        (firstIn (stateBefore pub proof messages ops c)
          ⟨level, "tensor.final_claims", none⟩)).map fun o =>
        (o.point.map Ext.into,
          mle (o.point.map Ext.into)
            (tensorWeight ((recordedIn (stateBefore pub proof messages ops c) point).toList.tail.map
              Ext.into)
              (firstIn (stateBefore pub proof messages ops c) ⟨level, "tensor.eta", none⟩).into),
          o.claim.into)) := by
  obtain ⟨e, shaped⟩ := roundsAt_sound shape
  obtain ⟨roundCount, lengths, -⟩ := rounds_facts pub proof messages _ (channelStart_fresh pub)
    (channelStart_empty pub) _ _ count 2 e shaped (aligned_take messages _ c _ aligned) before
  exact tensorRun_runs count w _ _ _ _ _ _ _ roundCount lengths

/-- A step of `tensor.end` without a stop: `tensorRun` gives a result on the recorded values. -/
theorem tensor_some_of_step (i : ℕ) (key : Key) (level count : ℕ) (point opening : Key)
    (s : Replayer)
    (stepped : (((step pub proof messages i (.check key (.tensor level count point opening))).run).run
      s).1 = .ok ()) :
    ∃ o, tensorRun count (recordedIn s point).toList (firstIn s opening)
      ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
      ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
      (firstIn s ⟨level, "tensor.eta", none⟩) (roundsIn s ⟨level, "tensor.sumcheck", none⟩)
      (firstIn s ⟨level, "tensor.final_claims", none⟩) = some o := by
  have accepts := step_stage_ok pub proof messages i key (.tensor level count point opening) s
    (out := .tensor (tensorRun count (recordedIn s point).toList (firstIn s opening)
      ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
      ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
      (firstIn s ⟨level, "tensor.eta", none⟩) (roundsIn s ⟨level, "tensor.sumcheck", none⟩)
      (firstIn s ⟨level, "tensor.final_claims", none⟩))) rfl stepped
  exact Option.isSome_iff_exists.mp accepts

/-- A trace level on the coefficient route, after a run without a stop before its `fused.end` at
`c`: `foldCheck` of `traceReduction` runs on the path of the code to the result of `foldRunIn` on
the state before `c`. -/
theorem trace_game_before (aligned : ReadsAligned messages ops.zipIdx.toList 0) {c level : ℕ}
    {F : Fold} (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {s : Replayer} (state : stateBefore pub proof messages ops c = s)
    {T : TraceShape} (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => T.shape) T.opening.openRank
      T.opening.openDegree T.opening.chain fun _ => .witness)
    (based : 2 ^ F.logBasis = T.shape.certifiedBase) (low : F.low = T.opening.low)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (challenge : Fin T.shape.blocks → Rq T.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq T.shape.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)).getD i
        (Poly.zero T.shape.innerDegree)))
    (fits : T.opening.cells T.fold ≤ 2 ^ F.vars)
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ F.batch) (w : Table)
    (plain : F.euclidean = false) (compressed : F.compressed = T.opening.compressed T.fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase)
    (leaf : leafDegree F.logBasis = rangeLeaf T.shape.certifiedBase) :
    (foldCheck (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have st : LevelState (stateBefore pub proof messages ops c) level F :=
    level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _ level F
      game (aligned_take messages _ c _ aligned) before
  rw [state] at st
  obtain ⟨virtual, count, lengths⟩ := st.leaf plain
  exact foldRunIn_trace_runs (E := E) pub s level F shaped expanded rows based low quarter
    two challenge challenged fits rowsFit w plain compressed virtual st.vars st.tau arities
    st.stages count (fun x member => (lengths x member).trans leaf) st.fusedCount st.fusedLengths

/-- A trace level on the Euclidean route, after a run without a stop before its `fused.end` at
`c`: `foldCheckSq` of `traceReduction` runs on the path of the code to the result of `foldRunIn`
on the state before `c`. -/
theorem traceSq_game_before (aligned : ReadsAligned messages ops.zipIdx.toList 0) {c level : ℕ}
    {F : Fold} (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {s : Replayer} (state : stateBefore pub proof messages ops c = s)
    {T : TraceShape} (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => T.shape) T.opening.openRank
      T.opening.openDegree T.opening.chain fun _ => .witness)
    (low : F.low = T.opening.low) (leaf : leafDegree F.logBasis = rangeLeaf T.shape.certifiedBase)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (challenge : Fin T.shape.blocks → Rq T.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq T.shape.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)).getD i
        (Poly.zero T.shape.innerDegree)))
    (fits : T.opening.cells T.fold ≤ 2 ^ F.vars)
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ F.batch) (w : Table)
    (positive : 1 ≤ rangeLeaf T.shape.certifiedBase) (small : F.limit < fieldModulus)
    (digits : 0 < T.fold.responseDigits) (euclidean : F.euclidean = true)
    (compressed : F.compressed = T.opening.compressed T.fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase) :
    (foldCheckSq (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) F.limit
        (T.shape.normKernel T.fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have st : LevelState (stateBefore pub proof messages ops c) level F :=
    level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _ level F
      game (aligned_take messages _ c _ aligned) before
  rw [state] at st
  obtain ⟨single, virtual, count, lengths⟩ := st.norm euclidean
  exact foldRunIn_traceSq_runs (E := E) pub s level F shaped expanded rows low leaf
    quarter two challenge challenged fits rowsFit w positive small digits euclidean compressed
    single virtual st.vars st.tau arities st.stages count
    (fun x member => by rw [lengths x member, leaf]) st.fusedCount st.fusedLengths

end Forms

end Akita.Replay

end NightstreamFPrime.Spec.Compression
