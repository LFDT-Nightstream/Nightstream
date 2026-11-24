import NightstreamFPrime.Spec.Compression.Akita.Replay.RunGame

/-! The whole replay of a proof (Akita revision `d50f700`; `replay` and `verdict` of
`Replay/Transcript.lean`, `verifyBytes` of `Replay/Wire.lean`).

- `channelStart` is the state at the start of `replay`. `replay_none_iff`: the replay has no stop
  exactly when the run of all the indexed operations of the script from it has none.
- `verifyBytes_accept`: with no checks after the replay, `verifyBytes` accepts exactly when the
  proof decodes with no byte left and the replay stops neither with `noWork` nor with `reject`;
  a stop with `mismatch` gives the verdict of the checks after the replay.
- `replay_level`: when the replay has no stop, the level function of each `fused.end` check of
  the script whose operations before it have the shape `foldShape` is the recorded
  `(fused.point, witness_evaluation)` on the state of the replay at the check.
  `replay_level_reject`: when the replay stops at a `fused.end` check of this shape, the level
  function is `none` there. `replay_trace_game`: on a trace level on the coefficient route the
  game of the specification level runs to acceptance on the path of the code.
- `replay_no_mismatch`: with the messages of `decode`, a reading operation of the script that the
  run reaches has the message of its kind at `next`, so its step does not stop with `mismatch`.
- `openingHolds` is the check `tensor.opening` on the recorded values; `tensorRun_opening`: when
  `tensorRun` accepts, the check holds, and `tensorRun_of_opening_reject`: when the check stops,
  `tensorRun` is `none` on the same values (finding 16 for `tensor.opening`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

namespace Replayer

/-- The state at the start of `replay`: the sponge of the channel and nothing recorded. -/
def channelStart (pub : Public) : Replayer :=
  ⟨channel pub.session pub.descriptor, 0, {}, {}, {}, {}⟩

theorem channelStart_fresh (pub : Public) (k : Key) : recordedIn (channelStart pub) k = #[] := by
  simp [recordedIn, channelStart]

theorem channelStart_empty (pub : Public) : (channelStart pub).derived.rounds = #[] := rfl

theorem channelStart_next (pub : Public) : (channelStart pub).next = 0 := rfl

section Whole

variable (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message)

theorem replay_stop :
    (replay pub ops proof messages).1 =
      match (runFrom pub proof messages ops.zipIdx.toList (channelStart pub)).1 with
      | .ok () => none
      | .error stop => some stop := by
  unfold replay
  rw [forM_array]
  rfl

/-- The replay has no stop exactly when the run of the indexed operations of the script from the
start of the channel has none. -/
theorem replay_none_iff :
    (replay pub ops proof messages).1 = none ↔
      (runFrom pub proof messages ops.zipIdx.toList (channelStart pub)).1 = .ok () := by
  rw [replay_stop]
  rcases (runFrom pub proof messages ops.zipIdx.toList (channelStart pub)).1 with _ | ⟨⟩ <;> simp

/-- With no checks after the replay, `verifyBytes` accepts exactly when the proof decodes with no
byte left and the replay stops neither with `noWork` nor with `reject`. -/
theorem verifyBytes_accept :
    verifyBytes (verdict pub ops proof fun _ => .accept) ops proof = .verdict .accept ↔
      ∃ messages rest, decode ops proof = .ok (messages, rest) ∧ rest = 0 ∧
        ∀ op site, (replay pub ops proof messages).1 ≠ some (.noWork op site) ∧
          (replay pub ops proof messages).1 ≠ some (.reject op site) := by
  unfold verifyBytes
  rcases decode ops proof with failure | ⟨messages, rest⟩
  · simp
  · unfold verdict verdictOf
    rcases stop : (replay pub ops proof messages).1 with _ | (_ | _ | _) <;>
      by_cases empty : rest = 0 <;> simp [empty, stop]

theorem take_succ_zipIdx {c : ℕ} {op : Op} (at_ : ops[c]? = some op) :
    ops.zipIdx.toList.take (c + 1) = ops.zipIdx.toList.take c ++ [(op, c)] := by
  have inside : c < ops.zipIdx.toList.length := by
    simp only [Array.length_toList, Array.size_zipIdx]
    exact (Array.getElem?_eq_some_iff.mp at_).1
  rw [List.take_succ_eq_append_getElem inside]
  congr
  obtain ⟨below, same⟩ := Array.getElem?_eq_some_iff.mp at_
  simp [same]

theorem runFrom_take_ok (l : List (Op × ℕ)) (j : ℕ) (s : Replayer)
    (ok : (runFrom pub proof messages l s).1 = .ok ()) :
    (runFrom pub proof messages (l.take j) s).1 = .ok () := by
  rw [← List.take_append_drop j l, runFrom_append_fst] at ok
  exact ok.1

/-- When the replay has no stop, the level function of a `fused.end` check at `c` whose
operations before it have the shape `foldShape` is the recorded `(fused.point,
witness_evaluation)` on the state of the replay at the check. -/
theorem replay_level (accepted : (replay pub ops proof messages).1 = none) {c : ℕ} {key : Key}
    {level : ℕ} {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true) :
    foldRunIn pub (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).2
        level F =
      some ((recordedIn (runFrom pub proof messages (ops.zipIdx.toList.take c)
          (channelStart pub)).2 ⟨level, "fused.point", none⟩).toList,
        firstIn (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).2
          ⟨level, "witness_evaluation", none⟩) := by
  have ran := runFrom_take_ok pub proof messages _ (c + 1) _
    ((replay_none_iff pub ops proof messages).mp accepted)
  rw [take_succ_zipIdx ops at_] at ran
  exact foldRunIn_of_run pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub)
    _ key c level F shape ran

/-- When the replay stops at a `fused.end` check at `c` whose operations before it have the shape
`foldShape`, the level function is `none` on the state of the replay at the check. -/
theorem replay_level_reject {c : ℕ} {key : Key} {level : ℕ} {F : Fold}
    (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (stopped : (runFrom pub proof messages (ops.zipIdx.toList.take (c + 1))
      (channelStart pub)).1 ≠ .ok ()) :
    foldRunIn pub (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).2
      level F = none := by
  rw [take_succ_zipIdx ops at_, Ne, runFrom_append_fst, runFrom_cons_fst] at stopped
  refine foldRunIn_of_reject pub proof messages _ (channelStart_fresh pub)
    (channelStart_empty pub) _ key c level F shape before fun stepped => stopped ⟨before, ?_⟩
  exact ⟨stepped, rfl⟩

/-- With the messages of `decode`, the step of a reading operation `j` of the script that the run
reaches without a stop does not stop with `mismatch`. -/
theorem replay_no_mismatch {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {j : ℕ} {op : Op} (at_ : ops[j]? = some op) (reads : op.reads = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 =
      .ok ()) :
    (((step pub proof messages j op).run).run
      (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).2).1 ≠
        .error (.mismatch j) := by
  obtain ⟨below, same⟩ := Array.getElem?_eq_some_iff.mp at_
  have inside : j < ops.zipIdx.toList.length := by simpa using below
  have entry : ops.zipIdx.toList[j] = (op, j) := by simp [same]
  obtain ⟨m, found, fits⟩ := aligned_at messages _ 0 j inside (decode_aligned decoded)
    (by rw [entry]; exact reads)
  rw [entry] at fits
  refine step_fits_no_mismatch pub proof messages j op _ ⟨m, ?_, fits⟩
  rw [runFrom_next pub proof messages _ _ before, channelStart_next]
  exact found

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- When the proof decodes and the replay has no stop, on a trace level on the coefficient route
the game `foldCheck` of the specification level runs on the path of the code to acceptance with
the recorded `(fused.point, witness_evaluation)` on the state of the replay at its `fused.end`. -/
theorem replay_trace_game {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    (accepted : (replay pub ops proof messages).1 = none) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true) {s : Replayer}
    (state : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).2 = s)
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
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) := by
  have ran := runFrom_take_ok pub proof messages _ (c + 1) _
    ((replay_none_iff pub ops proof messages).mp accepted)
  rw [take_succ_zipIdx ops at_] at ran
  exact foldRunIn_trace_game pub proof messages _ (channelStart_fresh pub)
    (channelStart_empty pub) _ key c level F shape game
    (aligned_take messages _ c _ (decode_aligned decoded)) ran state shaped expanded rows based
    low quarter two challenge challenged fits rowsFit w plain compressed arities leaf

/-- The check `tensor.opening` on the recorded values: `v = (1 − r_0)·S_0 + r_0·S_1`. -/
def openingHolds (s : Replayer) (point value partials : Key) : Bool :=
  (firstIn s value).same ((Ext.one - (recordedIn s point).getD 0 Ext.zero) *
      (recordedIn s partials).getD 0 Ext.zero +
    (recordedIn s point).getD 0 Ext.zero * (recordedIn s partials).getD 1 Ext.zero)

theorem holds_tensorOpening (s : Replayer) (point value partials : Key) :
    ((holds pub (.tensorOpening point value partials)).run).run s =
      (.ok (openingHolds s point value partials), s) := rfl

/-- When `tensorRun` accepts on the recorded values, the check `tensor.opening` holds on them. -/
theorem tensorRun_opening (s : Replayer) (count : ℕ) (point value partials : Key) (η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext)
    (accepts : (tensorRun count (recordedIn s point).toList (firstIn s value)
      ((recordedIn s partials).getD 0 Ext.zero) ((recordedIn s partials).getD 1 Ext.zero) η rounds
        final).isSome = true) :
    openingHolds s point value partials = true := by
  unfold openingHolds
  unfold tensorRun at accepts
  cases found : (recordedIn s point).toList with
  | nil => rw [found] at accepts; cases accepts
  | cons head tail =>
      rw [found] at accepts
      have first : (recordedIn s point).getD 0 Ext.zero = head := by
        rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, found]
        rfl
      rw [first]
      simp only at accepts
      split at accepts
      · split at accepts
        · assumption
        · cases accepts
      · cases accepts

/-- When the check `tensor.opening` stops, `tensorRun` is `none` on the recorded values of the state
at the check. -/
theorem tensorRun_of_opening_reject (i : ℕ) (key : Key) (s : Replayer) (count : ℕ)
    (point value partials : Key) (η : Ext) (rounds : List (List Ext × Ext)) (final : Ext)
    (stopped : (((step pub proof messages i (.check key (.tensorOpening point value partials))).run).run
      s).1 ≠ .ok ()) :
    tensorRun count (recordedIn s point).toList (firstIn s value)
      ((recordedIn s partials).getD 0 Ext.zero) ((recordedIn s partials).getD 1 Ext.zero) η rounds
        final = none := by
  cases accepts : (tensorRun count (recordedIn s point).toList (firstIn s value)
      ((recordedIn s partials).getD 0 Ext.zero) ((recordedIn s partials).getD 1 Ext.zero) η rounds
        final) with
  | none => rfl
  | some out =>
      have holdsTrue := tensorRun_opening s count point value partials η rounds final
        (by rw [accepts]; rfl)
      refine absurd ?_ stopped
      rw [step_check, checkStep_run]
      have none : stageValue pub s (.tensorOpening point value partials) = none := rfl
      rw [none]
      simp only [checkWith]
      rw [run_bind, holds_tensorOpening, holdsTrue]
      rfl

end Whole

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
