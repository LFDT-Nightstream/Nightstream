import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeAccept

/-! Theorem B of the join, direction "←": a first stop of the replay at `fused.end`,
`tensor.opening` or `tensor.end` against a run of the game of the model (Akita revision `d50f700`;
ePrint 2026/1983, revision of 2026-11-23, Figure 4, steps 0, 3 and 5).

- `tensorRun_runs_closed`: when the opening check of the tensor reduction fails on the values of
  the code, `tensorCheck` runs on the path of the code to `none`, for each list of rounds.
- `tensorValuesIn`, `tensoredPathOf`, `tensoredPathIn`: the path of the tensor reduction of the
  level `level + 1` on the recorded values of a state.
- `level_of_ran`: the level function at a `fused.end` whose step has no stop.
- `fused_ok_of_runs`, `fused_ok_of_run`: a game that runs on the path of the code to the result of
  `foldRunIn` and to `some` (alone, or before a continuation that ends in `none` after `none`)
  gives the step of `fused.end` without a stop.
- `opening_ok_of_runs`, `tensor_ok_of_runs`: the same for the steps of `tensor.opening` and
  `tensor.end`, from the run of the fold with the tensor reduction on the paths of the state
  before a later operation `c'`. After a stop, the state before `c'` is the state of the stop
  (`stateBefore_stopped`, `check_keeps`).
- `levels20`, `frames20`, `indices20`: the shapes, the frames and the indices of a script of
  `layout20`.
- `key20_root_before_of`, `key20_level1_before_of`, `key20_trace_before_of`,
  `key20_traceSq_before_of`: the games of the levels of `key20.aks` after a run without a stop
  before the `fused.end` of the level, to the result of `foldRunIn`.
- `openingShape20`, `skeleton20_openings`, `opening_of`: the `tensor.opening` checks of the
  skeleton of `layout20` and the frames of the claim that they read. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

section Closed

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- `tensorCheck` of the specification on the path of the code ends in `none` when the opening
check `v = (1 − r_0)·S_0 + r_0·S_1` fails on the values of the code; the rounds have no
condition. -/
theorem tensorRun_runs_closed (ν : ℕ) (w : Table) (point : List Ext) (v p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext)
    (closed : ∀ head tail, point = head :: tail → tail.length = ν →
      ¬ v.same ((Ext.one - head) * p0 + head * p1) = true) :
    (tensorCheck ν w (point.map (Ext.into (E := E))) v.into).Runs
      (tensorPath E ν point v p0 p1 η rounds final) none := by
  cases point with
  | nil => rfl
  | cons head tail =>
      simp only [List.map_cons, tensorCheck, tensorPath, List.length_map]
      by_cases long : tail.length = ν
      · simp only [ite_eq_left long]
        refine runs_recv ?_
        have value : (((Ext.one - head) * p0 + head * p1).into : E) =
            (1 - head.into) * p0.into + head.into * p1.into := by
          simp only [Ext.into_add, Ext.into_mul, Ext.into_sub, Ext.into_one]
        have passes := closed head tail rfl long
        have fails : ¬ (v.into : E) = (1 - head.into) * ((p0.into, p1.into) : E × E).1 +
            head.into * ((p0.into, p1.into) : E × E).2 := fun equal =>
          passes ((same_degree2 (E := E) _ _).mpr (equal.trans value.symm))
        rw [ite_eq_right passes, ite_eq_right fails]
        rfl
      · simp only [ite_eq_right long]
        rfl

/-- The recorded values that the tensor reduction of the level `level + 1` reads on the state
`s`: the claim of the level `level`, the partials, `η`, the rounds and the final claim. -/
def tensorValuesIn (level : ℕ) (s : Replayer) :
    Array Ext × Ext × Array Ext × Ext × List (List Ext × Ext) × Ext :=
  (recordedIn s ⟨level, "fused.point", none⟩, firstIn s ⟨level, "witness_evaluation", none⟩,
    recordedIn s ⟨level + 1, "tensor.partials", none⟩, firstIn s ⟨level + 1, "tensor.eta", none⟩,
    roundsIn s ⟨level + 1, "tensor.sumcheck", none⟩,
    firstIn s ⟨level + 1, "tensor.final_claims", none⟩)

variable (E) in
/-- The path of the tensor reduction with `ν` rounds on the values `x` of `tensorValuesIn`. -/
noncomputable def tensoredPathOf (ν : ℕ)
    (x : Array Ext × Ext × Array Ext × Ext × List (List Ext × Ext) × Ext) : List Move :=
  tensoredPath E ν (some (x.1.toList, x.2.1)) (x.2.2.1.getD 0 Ext.zero) (x.2.2.1.getD 1 Ext.zero)
    x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2

variable (E) in
/-- The path of the tensor reduction of the level `level + 1` with `ν` rounds for the claim of
the level `level`, on the recorded values of the state `s`. -/
noncomputable def tensoredPathIn (ν level : ℕ) (s : Replayer) : List Move :=
  tensoredPathOf E ν (tensorValuesIn level s)

omit [Fintype E] [DecidableEq E] in
theorem tensoredPathIn_eq (ν level : ℕ) (s : Replayer) :
    tensoredPathIn E ν level s = tensoredPath E ν
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList,
        firstIn s ⟨level, "witness_evaluation", none⟩))
      ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
      ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero)
      (firstIn s ⟨level + 1, "tensor.eta", none⟩)
      (roundsIn s ⟨level + 1, "tensor.sumcheck", none⟩)
      (firstIn s ⟨level + 1, "tensor.final_claims", none⟩) := rfl

end Closed

section Stops

variable (pub : Public) (proof : ByteArray) (messages : Array Message) {ops : Array Op}

/-- The level function on the state before a `fused.end` of the shape `foldShape` whose step has
no stop is the recorded `(fused.point, witness_evaluation)`. -/
theorem level_of_ran {c : ℕ} {key : Key} {level : ℕ} {F : Fold}
    (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (ran : (runFrom pub proof messages (ops.zipIdx.toList.take (c + 1)) (channelStart pub)).1 =
      .ok ()) :
    foldRunIn pub (stateBefore pub proof messages ops c) level F =
      some ((recordedIn (stateBefore pub proof messages ops c) ⟨level, "fused.point", none⟩).toList,
        firstIn (stateBefore pub proof messages ops c) ⟨level, "witness_evaluation", none⟩) := by
  rw [take_succ_zipIdx ops at_] at ran
  exact foldRunIn_of_run pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
    key c level F shape ran

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- A first stop at `fused.end` against the model, for a level with a continuation: when a game
`g` runs on `path` to the result of `foldRunIn` on the state before `c`, and `g` with a
continuation that ends in `none` after `none` runs on a path from `path` to `some`, the step of
`fused.end` at `c` has no stop. -/
theorem fused_ok_of_runs {c : ℕ} {key : Key} {level : ℕ} {F : Fold} {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {α : Type} {g : Game (Option (List E × E))} {path rest : List Move}
    {f : Option (List E × E) → Game (Option α)}
    (fold : g.Runs path ((foldRunIn pub (stateBefore pub proof messages ops c) level F).map
      fun o => (o.1.map Ext.into, o.2.into)))
    {x : α} (model : (g.bind f).Runs (path ++ rest) (some x)) (ends : f none = .pure none) :
    (((step pub proof messages c (.check key (.fused level F))).run).run
      (stateBefore pub proof messages ops c)).1 = .ok () := by
  by_contra stopped
  have none_ : foldRunIn pub (stateBefore pub proof messages ops c) level F = none :=
    foldRunIn_of_reject pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
      key c level F shape before stopped
  rw [none_] at fold
  exact not_runs_bind_none ends fold model

omit [Fintype E] [DecidableEq E] in
/-- A first stop at `fused.end` against the model, for a level without a continuation. -/
theorem fused_ok_of_run {c : ℕ} {key : Key} {level : ℕ} {F : Fold} {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {g : Game (Option (List E × E))} {path : List Move}
    (fold : g.Runs path ((foldRunIn pub (stateBefore pub proof messages ops c) level F).map
      fun o => (o.1.map Ext.into, o.2.into)))
    {x : List E × E} (model : g.Runs path (some x)) :
    (((step pub proof messages c (.check key (.fused level F))).run).run
      (stateBefore pub proof messages ops c)).1 = .ok () := by
  by_contra stopped
  have none_ : foldRunIn pub (stateBefore pub proof messages ops c) level F = none :=
    foldRunIn_of_reject pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
      key c level F shape before stopped
  rw [none_] at fold
  exact absurd (Game.runs_unique fold model) (by simp)

/-- A first stop at `tensor.opening` against the model: the level `level` ends at `c` without a
stop, and the game of the level with the tensor reduction runs to `some` on the path of the fold
and the path of the tensor reduction on the state before `c'`. Then the step of `tensor.opening`
at `o` has no stop. -/
theorem opening_ok_of_runs {level c o c' ν : ℕ} {key₀ key : Key} {F : Fold} {r a e : ℕ}
    (at_ : ops[c]? = some (.check key₀ (.fused level F)))
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true) (after : c < o)
    (openingAt : ops[o]? = some (.check key (.tensorOpening ⟨level, "fused.point", none⟩
      ⟨level, "witness_evaluation", none⟩ ⟨level + 1, "tensor.partials", none⟩)))
    (held : frees ((ops.zipIdx.toList.take o).drop c)
      [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] = true)
    (lt : o < c')
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take o) (channelStart pub)).1 =
      .ok ())
    {g : Game (Option (List E × E))} {path : List Move}
    (fold : g.Runs path ((foldRunIn pub (stateBefore pub proof messages ops c) level F).map
      fun o => (o.1.map Ext.into, o.2.into)))
    {x : List E × E × E}
    (model : (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck ν 0 r2 v).Runs
      (path ++ tensoredPathIn E ν level (stateBefore pub proof messages ops c')) (some x)) :
    (((step pub proof messages o (.check key (.tensorOpening ⟨level, "fused.point", none⟩
      ⟨level, "witness_evaluation", none⟩ ⟨level + 1, "tensor.partials", none⟩))).run).run
        (stateBefore pub proof messages ops o)).1 = .ok () := by
  by_contra stopped
  rw [level_of_ran pub proof messages at_ shape
    (before_le pub proof messages (Nat.succ_le_of_lt after) before), Option.map_some] at fold
  obtain ⟨frames, -⟩ := stateBefore_frees pub proof messages (Nat.le_of_lt after) held
  rw [← (frames _ List.mem_cons_self).1,
    ← firstIn_congr (frames ⟨level, "witness_evaluation", none⟩ (by simp)).1] at fold
  have kept : tensoredPathIn E ν level (((step pub proof messages o (.check key (.tensorOpening
      ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩
      ⟨level + 1, "tensor.partials", none⟩))).run).run (stateBefore pub proof messages ops o)).2 =
      tensoredPathIn E ν level (stateBefore pub proof messages ops o) :=
    congrArg (tensoredPathOf E ν) (check_keeps pub proof messages (tensorValuesIn level)
      (fun _ _ _ => rfl) (fun _ _ => rfl) o key _ _)
  rw [stateBefore_stopped pub proof messages openingAt lt before stopped, kept] at model
  have model' := runs_bind_inv fold model
  have false_ := holds_of_stopped pub proof messages o key _ _ stopped
  rw [holds_tensorOpening] at false_
  have fails : openingHolds (stateBefore pub proof messages ops o) ⟨level, "fused.point", none⟩
      ⟨level, "witness_evaluation", none⟩ ⟨level + 1, "tensor.partials", none⟩ = false :=
    Except.ok.inj false_
  have notSame : ∀ head tail,
      (recordedIn (stateBefore pub proof messages ops o) ⟨level, "fused.point", none⟩).toList =
        head :: tail → tail.length = ν →
      ¬ (firstIn (stateBefore pub proof messages ops o) ⟨level, "witness_evaluation", none⟩).same
        ((Ext.one - head) *
          (recordedIn (stateBefore pub proof messages ops o)
            ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero +
        head * (recordedIn (stateBefore pub proof messages ops o)
          ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero) = true := by
    intro head tail found _ passes
    unfold openingHolds at fails
    have first : (recordedIn (stateBefore pub proof messages ops o)
        ⟨level, "fused.point", none⟩).getD 0 Ext.zero = head := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, found]
      rfl
    rw [first, passes] at fails
    cases fails
  exact absurd (Game.runs_unique (tensorRun_runs_closed (E := E) ν 0 _ _ _ _ _ _ _ notSame) model')
    (by simp)

/-- A first stop at `tensor.end` against the model: the level `level` ends at `c` without a stop,
and the game of the level with the tensor reduction runs to `some` on the path of the fold and
the path of the tensor reduction on the state before `c'`. Then the step of `tensor.end` at `t`
has no stop. -/
theorem tensor_ok_of_runs (aligned : ReadsAligned messages ops.zipIdx.toList 0)
    {level c t c' ν : ℕ} (frame : TensorFrame ops level ν c t c') (lt : t < c') {key₀ : Key}
    {F : Fold} {r a e : ℕ} (at_ : ops[c]? = some (.check key₀ (.fused level F)))
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true) (after : c < t)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take t) (channelStart pub)).1 =
      .ok ())
    {g : Game (Option (List E × E))} {path : List Move}
    (fold : g.Runs path ((foldRunIn pub (stateBefore pub proof messages ops c) level F).map
      fun o => (o.1.map Ext.into, o.2.into)))
    {x : List E × E × E}
    (model : (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck ν 0 r2 v).Runs
      (path ++ tensoredPathIn E ν level (stateBefore pub proof messages ops c')) (some x)) :
    (((step pub proof messages t (.check ⟨level + 1, "tensor.end", none⟩
      (.tensor (level + 1) ν ⟨level, "fused.point", none⟩
        ⟨level, "witness_evaluation", none⟩))).run).run
          (stateBefore pub proof messages ops t)).1 = .ok () := by
  by_contra stopped
  rw [level_of_ran pub proof messages at_ shape
    (before_le pub proof messages (Nat.succ_le_of_lt after) before), Option.map_some] at fold
  obtain ⟨frames, -⟩ := stateBefore_frees pub proof messages frame.before frame.held
  rw [← (frames _ List.mem_cons_self).1,
    ← firstIn_congr (frames ⟨level, "witness_evaluation", none⟩ (by simp)).1] at fold
  have kept : tensoredPathIn E ν level (((step pub proof messages t (.check
      ⟨level + 1, "tensor.end", none⟩ (.tensor (level + 1) ν ⟨level, "fused.point", none⟩
        ⟨level, "witness_evaluation", none⟩))).run).run (stateBefore pub proof messages ops t)).2 =
      tensoredPathIn E ν level (stateBefore pub proof messages ops t) :=
    congrArg (tensoredPathOf E ν) (check_keeps pub proof messages (tensorValuesIn level)
      (fun _ _ _ => rfl) (fun _ _ => rfl) t _ _ _)
  rw [stateBefore_stopped pub proof messages frame.at_ lt before stopped, kept] at model
  have model' := runs_bind_inv fold model
  have game := tensor_game_before (E := E) pub proof messages aligned
    ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩ frame.rounds before 0
  obtain ⟨out, ran, -⟩ := Option.map_eq_some_iff.mp (Game.runs_unique game model')
  have accepted : (StageOut.tensor (tensorRunIn (stateBefore pub proof messages ops t) (level + 1)
      ν ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩)).accepts = true :=
    Option.isSome_iff_exists.mpr ⟨out, ran⟩
  apply stopped
  rw [step_stage pub proof messages t ⟨level + 1, "tensor.end", none⟩
    (.tensor (level + 1) ν ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩)
    (stateBefore pub proof messages ops t)
    (out := .tensor (tensorRunIn (stateBefore pub proof messages ops t) (level + 1) ν
      ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩)) rfl, ite_eq_left accepted]

end Stops

section Script20

variable {zMax rice : ℕ} {pub : Public}

/-- The shapes at the `fused.end` of each level of a script of `layout20`. -/
theorem levels20 (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) (ℓ : Fin 6) :
    FusedAt pub.ops ℓ (fold20 ℓ) (fusedIndex pub.layout ℓ) := by
  have facts := script20 (layout ▸ scripted)
  rw [layout]
  exact (facts.2.2.2 ℓ).1

/-- The frames of the links of a script of `layout20` on `pub.ops`. -/
theorem frames20 (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) :
    RootFrame pub.ops (fusedIndex pub.layout 0) ∧
      PackFrame pub.ops (fusedIndex pub.layout 0) (fusedIndex pub.layout 1) ∧
      TensorFrame pub.ops 1 Key20.level₁.tail (fusedIndex pub.layout 1) 385
        (fusedIndex pub.layout 2) ∧
      TensorFrame pub.ops 2 Key20.level₂.tail (fusedIndex pub.layout 2) 649
        (fusedIndex pub.layout 3) ∧
      TensorFrame pub.ops 3 Key20.level₃.tail (fusedIndex pub.layout 3) 908
        (fusedIndex pub.layout 4) ∧
      TensorFrame pub.ops 4 Key20.level₄.tail (fusedIndex pub.layout 4) 1210
        (fusedIndex pub.layout 5) ∧
      TensorFrame pub.ops 5 Key20.level₅.tail (fusedIndex pub.layout 5) 1513
        (pub.ops.size - 1) ∧
      pub.ops[pub.ops.size - 1]? = some (.terminal 6) := by
  rw [layout] at scripted ⊢
  exact script20_frames scripted

/-- The indices of the `fused.end` of each level and of the terminal of a script of `layout20`. -/
theorem indices20 (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) :
    fusedIndex pub.layout 0 = 165 ∧ fusedIndex pub.layout 1 = 316 ∧ fusedIndex pub.layout 2 = 586 ∧
      fusedIndex pub.layout 3 = 848 ∧ fusedIndex pub.layout 4 = 1153 ∧
      fusedIndex pub.layout 5 = 1456 ∧ pub.ops.size - 1 = 1521 := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨f0, f1, f2, f3, f4, f5⟩ := fusedIndex20_values
  rw [layout]
  exact ⟨f0, f1, f2, f3, f4, f5, by rw [size_skeleton erased, skeleton20_length]⟩

/-- The check `tensor.opening` of the level `level + 1` for the claim of the level `level`. -/
def isOpeningOf (level : ℕ) : Op → Bool
  | .check key (.tensorOpening p v q) =>
      key == ⟨level + 1, "tensor.opening", none⟩ && p == ⟨level, "fused.point", none⟩ &&
        v == ⟨level, "witness_evaluation", none⟩ && q == ⟨level + 1, "tensor.partials", none⟩
  | _ => false

theorem isOpeningOf_sound {level : ℕ} {op : Op} (h : isOpeningOf level op = true) :
    op = .check ⟨level + 1, "tensor.opening", none⟩ (.tensorOpening ⟨level, "fused.point", none⟩
      ⟨level, "witness_evaluation", none⟩ ⟨level + 1, "tensor.partials", none⟩) := by
  unfold isOpeningOf at h
  split at h
  · simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩ := h
    rfl
  · cases h

/-- On the skeleton of `layout20`: the check `tensor.opening` of the level `level + 1` is at `o`,
and no operation from the `fused.end` of the level `level` at `c` to `o` writes the claim of the
level `level`. -/
def openingShape20 (level c o : ℕ) : Bool :=
  ((skeleton (layout20 0 0))[o]?).any (isOpeningOf level) &&
    frees ((sk20.take o).drop c)
      [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] []

theorem skeleton20_openings : openingShape20 1 316 321 = true ∧ openingShape20 2 586 591 = true ∧
    openingShape20 3 848 853 = true ∧ openingShape20 4 1153 1158 = true ∧
    openingShape20 5 1456 1461 = true := by
  decide +kernel

theorem opening_of {ops : Array Op} (erased : ops.toList.map Op.erase = skeleton (layout20 0 0))
    {level c o : ℕ} (shaped : openingShape20 level c o = true) :
    ops[o]? = some (.check ⟨level + 1, "tensor.opening", none⟩
        (.tensorOpening ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩
          ⟨level + 1, "tensor.partials", none⟩)) ∧
      frees ((ops.zipIdx.toList.take o).drop c)
        [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] = true := by
  unfold openingShape20 at shaped
  simp only [Bool.and_eq_true] at shaped
  exact ⟨erased_check erased (erased_matched (fun _ h => isOpeningOf_sound h) shaped.1),
    joined_frees erased _ _ _ _ shaped.2⟩

end Script20

section Games20

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The root of `key20.aks` after a run without a stop before its `fused.end`: `foldCheck` of
`Batch.liftedReduction` of the root runs on the path of the code to the result of `foldRunIn`. -/
theorem key20_root_before_of {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {messages : Array Message}
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0))
      (channelStart pub)).1 = .ok ())
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 0).setupCount = some setup)
    (β : ∀ g, Module.Basis (Fin (Key20.root.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : Key20.root.Challenge)
    (challenged : ∀ k i, challenge (groupOrder 3 k) i = Poly.toRq _
      ((drawnCarrier Key20.root s 0 k).getD i
        (Poly.zero (Key20.root.carrierDegree (groupOrder 3 k)))))
    (w : Table) :
    ∃ (fits : Key20.root.cells ≤ 2 ^ (fold20 0).vars)
      (room : Key20.root.quotientRoom Key20.quotientDigits ≤ 2 ^ (fold20 0).vars)
      (rowsFit : Key20.root.rowCount + 1 ≤ 2 ^ (fold20 0).batch),
      (foldCheck (Batch.liftedReduction β (batchStatementFor Key20.root setup (claimPointIn s 0)
          (dataIn pub s 0) (fun k => .root (groupOrder 3 k).val) (rowsIn s 0 (fold20 0))
          (claimsIn s 0 (fold20 0))) challenge (fold20 0).vars fits 8 Key20.quotientDigits room
          (fold20 0).batch rowsFit)
          8 (Key20.root.gaps Key20.quotientDigits).support Key20.root.compressed w).Runs
        (foldPathIn E pub s 0 (fold20 0) setup (fold20 0).vars (fold20 0).batch)
        ((foldRunIn pub s 0 (fold20 0)).map fun o => (o.1.map Ext.into, o.2.into)) := by
  obtain ⟨-, -, game, drawn⟩ := levels20 layout scripted 0
  exact key20_root_before pub proof aligned (layout20_fold zMax rice 0) game drawn before state
    expanded β basis challenge challenged w

/-- Level 1 of `key20.aks` after a run without a stop before its `fused.end`: `foldCheck` of
`liftedReduction` of the level runs on the path of the code to the result of `foldRunIn`. -/
theorem key20_level1_before_of {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {messages : Array Message}
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 1))
      (channelStart pub)).1 = .ok ())
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout 1) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 1).setupCount = some setup)
    (β : Module.Basis (Fin Key20.level₁.opening.planes) Zq E)
    (basis : ∀ t, β t = (extBasis.getD t Ext.zero).into)
    (challenge : Fin Key20.level₁.shape.blocks → Rq Key20.level₁.opening.carrierDegree)
    (challenged : ∀ i, challenge i = Poly.toRq _ (((drawnIn s 1 0).map fun c =>
      Poly.ofSparse Key20.level₁.opening.carrierDegree (c.map fun x =>
        (x.1 / (Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree), x.2))).getD i
          (Poly.zero Key20.level₁.opening.carrierDegree)))
    (w : Table) :
    ∃ (fits : Key20.level₁.opening.cells Key20.level₁.fold ≤ 2 ^ (fold20 1).vars)
      (room : Key20.level₁.opening.quotientRoom Key20.level₁.fold Key20.quotientDigits ≤
        2 ^ (fold20 1).vars)
      (rowsFit : Key20.level₁.opening.rowCount + 1 ≤ 2 ^ (fold20 1).batch),
      (foldCheck (liftedReduction β (packStatementFor Key20.level₁.opening setup
          (claimPointIn s 1 0) (dataIn pub s 1) .witness (rowsIn s 1 (fold20 1))
          (claimsIn s 1 (fold20 1))) challenge (fold20 1).vars fits Key20.quotientDigits room
          (fold20 1).batch rowsFit)
          Key20.level₁.shape.certifiedBase
          (Key20.level₁.opening.gaps Key20.level₁.fold Key20.quotientDigits).support
          (Key20.level₁.opening.compressed Key20.level₁.fold) w).Runs
        (foldPathIn E pub s 1 (fold20 1) setup (fold20 1).vars (fold20 1).batch)
        ((foldRunIn pub s 1 (fold20 1)).map fun o => (o.1.map Ext.into, o.2.into)) := by
  obtain ⟨-, -, game, drawn⟩ := levels20 layout scripted 1
  exact key20_level1_before pub proof aligned (layout20_fold zMax rice 1) game drawn before state
    expanded β basis challenge challenged w

/-- Level 2 of `key20.aks`, a trace level on the coefficient route, after a run without a stop
before its `fused.end`: `foldCheck` of `traceReduction` runs on the path of the code to the result
of `foldRunIn`. -/
theorem key20_trace_before_of {zMax rice : ℕ} (pub : Public) (proof : ByteArray)
    (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {messages : Array Message}
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 2))
      (channelStart pub)).1 = .ok ())
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 2).setupCount = some setup)
    (challenge : Fin Key20.level₂.shape.blocks → Rq Key20.level₂.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq Key20.level₂.shape.innerDegree
      (((drawnIn s 2 0).map (Poly.ofSparse Key20.level₂.shape.innerDegree)).getD i
        (Poly.zero Key20.level₂.shape.innerDegree)))
    (w : Table) :
    ∃ (quarter : 4 * (Key20.level₂.shape.innerDegree / 4) = Key20.level₂.shape.innerDegree)
      (fits : Key20.level₂.opening.cells Key20.level₂.fold ≤ 2 ^ (fold20 2).vars)
      (rowsFit : Key20.level₂.opening.traceRowCount + 1 ≤ 2 ^ (fold20 2).batch),
      (foldCheck (traceReduction (traceStatementFor (E := E) Key20.level₂.opening quarter setup
          (traceValuesIn s 2 (fold20 2)) (dataIn pub s 2) .witness (rowsIn s 2 (fold20 2))
          (claimsIn s 2 (fold20 2))) challenge (fold20 2).vars fits (fold20 2).batch rowsFit)
          Key20.level₂.shape.certifiedBase (Key20.level₂.opening.binarySupport Key20.level₂.fold)
          (Key20.level₂.opening.compressed Key20.level₂.fold) w).Runs
        (foldPathIn E pub s 2 (fold20 2) setup (fold20 2).vars (fold20 2).batch)
        ((foldRunIn pub s 2 (fold20 2)).map fun o => (o.1.map Ext.into, o.2.into)) := by
  obtain ⟨-, -, game, -⟩ := levels20 layout scripted 2
  have sized := Key20.traces_sized Key20.level₂ (by simp [Key20.traces])
  refine ⟨by decide, sized.1, sized.2.1, ?_⟩
  exact trace_game_before pub proof messages aligned game before state
    (T := ⟨Key20.level₂.shape, Key20.level₂.opening, Key20.level₂.fold⟩) rfl expanded rfl
    (by decide) rfl (by decide) (shape20_cast_ne_zero (by decide) (by decide))
    challenge challenged _ _ w rfl (by decide) (by decide) (by decide)

/-- The levels 3 to 5 of `key20.aks`, trace levels on the Euclidean route, after a run without a
stop before the `fused.end` of the level: `foldCheckSq` of `traceReduction` with the norm stage
runs on the path of the code to the result of `foldRunIn`. -/
theorem key20_traceSq_before_of {zMax rice : ℕ} (ℓ : Fin 6) (L : Key20.Level)
    (listed : (ℓ, L) ∈ [(3, Key20.level₃), (4, Key20.level₄), (5, Key20.level₅)])
    (pub : Public) (proof : ByteArray) (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {messages : Array Message}
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout ℓ))
      (channelStart pub)).1 = .ok ())
    {s : Replayer} (state : stateBefore pub proof messages pub.ops (fusedIndex pub.layout ℓ) = s)
    {setup : Array Fq} (expanded : pub.setupFor (fold20 ℓ).setupCount = some setup)
    (challenge : Fin L.shape.blocks → Rq L.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq L.shape.innerDegree
      (((drawnIn s ℓ 0).map (Poly.ofSparse L.shape.innerDegree)).getD i
        (Poly.zero L.shape.innerDegree)))
    (w : Table) :
    ∃ (quarter : 4 * (L.shape.innerDegree / 4) = L.shape.innerDegree)
      (fits : L.opening.cells L.fold ≤ 2 ^ (fold20 ℓ).vars)
      (rowsFit : L.opening.traceRowCount + 1 ≤ 2 ^ (fold20 ℓ).batch),
      (foldCheckSq (traceReduction (traceStatementFor (E := E) L.opening quarter setup
          (traceValuesIn s ℓ (fold20 ℓ)) (dataIn pub s ℓ) .witness (rowsIn s ℓ (fold20 ℓ))
          (claimsIn s ℓ (fold20 ℓ))) challenge (fold20 ℓ).vars fits (fold20 ℓ).batch rowsFit)
          L.shape.certifiedBase (L.opening.binarySupport L.fold) (L.opening.compressed L.fold)
          (fold20 ℓ).limit (L.shape.normKernel L.fold) w).Runs
        (foldPathIn E pub s ℓ (fold20 ℓ) setup (fold20 ℓ).vars (fold20 ℓ).batch)
        ((foldRunIn pub s ℓ (fold20 ℓ)).map fun o => (o.1.map Ext.into, o.2.into)) := by
  obtain ⟨-, -, game, -⟩ := levels20 layout scripted ℓ
  simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at listed
  have sized := Key20.traces_sized L (by
    rcases listed with ⟨-, rfl⟩ | ⟨-, rfl⟩ | ⟨-, rfl⟩ <;> simp [Key20.traces])
  rcases listed with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  all_goals
    refine ⟨by decide, sized.1, sized.2.1, ?_⟩
    exact traceSq_game_before pub proof messages aligned game before state (T := ⟨_, _, _⟩) rfl
      expanded rfl rfl (by decide) (by decide) (shape20_cast_ne_zero (by decide) (by decide))
      challenge challenged _ _ w (by decide) (by decide) (by decide) rfl (by decide) (by decide)

end Games20

end NightstreamFPrime.Spec.Compression.Akita.Replay
