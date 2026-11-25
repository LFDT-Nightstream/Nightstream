import NightstreamFPrime.Spec.Compression.Akita.Replay.CausalTrace
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageCoupling

/-! The readings of the pinned Akita code at each stage of the opening of `key20.aks` as functions
of the answers of an ideal sponge (Akita revision `d50f700`; ePrint 2026/1983, revision of
2026-11-23, Section 8.2 `[KECCAK]` and (146); `docs/akita-work/w20/design.md`, decision D3;
`Replay/CausalTrace.lean`, `Replay/JoinBits.lean`).

- `Bits20 pub`: the work of each stage `k < 473` of the script of `pub` has the work bits
  `bits20` of the model. The code computes the plan from the schedule
  (`akita-config`, `derive_transcript_grinding_plan`; `akita-verifier/src/fold/verify.rs`), and
  `Public.plan` is an input of the replay, so `Bits20` is a hypothesis on the public input.
  `bitsOk20` is its check on a script, and `bits20_of_check` its soundness.
- `work20_answer`: the answer for `eventsAt20 … k` has the work of the stage `k`.
- `fieldDraw20 k` is the key and the count of the field draw directly after the work of the stage
  `k`, and `foldDraw20 k g` the level and the family of the group `g` of the fold draw.
  `skeleton20_fields`, `skeleton20_folds` and `drawCounts20`: a stage has a fold draw exactly
  when it is in `starts20`, the stage 1 has three groups and each other fold stage has one, and
  each draw squeezes bytes.
- `challenge20_answer`: the field draw of the stage `k` records the values that `decodePool`
  gives on the answer for the squeeze of the work and `eventsAt20 … k`.
- `squeeze20_answer`: the group `g` of the fold draw of the stage `k` records the challenges that
  `foldChallenges` gives on the answer for `drawEvents20 k g` (the squeezes of the groups before
  `g`), the squeeze of the work and `eventsAt20 … k`.
- `challenge20_recorded` and `squeeze20_drawn`: on each later state of the run, the recorded
  values of the key of a field draw from the position `offset20 k` and the drawn challenges of a
  group are these values and these challenges. `weights20_answer` and `drawn20_answer` are the
  instances for two readings of `AgreesAt20` (`Replay/JoinGiven.lean`): the weights of the
  batching root and the drawn fold challenges of a level before its `fused.end`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

open Replayer

section Bits

/-- The work of each stage of the script of `pub` has the work bits `bits20` of the model. -/
def Bits20 (pub : Public) : Prop :=
  ∀ k, k < 473 → ∃ site level,
    pub.ops[stageIndex20 pub k]? = some (.work site level (bits20.flatten.getD k 0))

/-- The check of `Bits20` on the operations of a script. -/
def bitsOk20 (ops : Array Op) : Bool :=
  (List.range 473).all fun k =>
    match ops[workIndices20.getD k 0]? with
    | some (.work _ _ bits) => bits == bits20.flatten.getD k 0
    | _ => false

theorem bits20_of_check {pub : Public} (ok : bitsOk20 pub.ops = true) : Bits20 pub := by
  intro k inside
  have at_ := List.all_eq_true.mp ok k (List.mem_range.mpr inside)
  rw [stageIndex20_lt pub inside]
  split at at_
  · rename_i site level bits found
    have same := beq_iff_eq.mp at_
    subst same
    exact ⟨site, level, found⟩
  · cases at_

theorem bits20_length : bits20.flatten.length = 473 := by decide +kernel

theorem bits20_ne_zero {k : ℕ} (inside : k < 473) : bits20.flatten.getD k 0 ≠ 0 := by
  have below : k < bits20.flatten.length := by
    rw [bits20_length]
    exact inside
  have member : bits20.flatten.getD k 0 ∈ bits20.flatten := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem below]
    exact List.getElem_mem below
  exact Nat.ne_of_gt (bits20_pos _ member)

variable {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- The work of the stage `k`, for a public input with the work bits of the model, a run with no
stop up to and with this work and an ideal sponge: the answer for `eventsAt20 … k` has the work
bits of the stage. -/
theorem work20_answer (bits : Bits20 pub) {k : ℕ} (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    worked (answer (eventsAt20 pub proof messages k) 8) (bits20.flatten.getD k 0) = true := by
  obtain ⟨site, level, at_⟩ := bits k inside
  exact work_passed20 at_ (bits20_ne_zero inside) ran ideal

end Bits

section Skeleton

/-- The key and the count of the field draw directly after the work of the stage `k`. -/
def fieldDraw20 (k : ℕ) : Option (Key × ℕ) :=
  (sk20[workIndices20.getD k 0 + 1]?).bind fun x =>
    match x.1 with
    | .challenge key count => some (key, count)
    | _ => none

/-- The level and the family of the group `g` of the fold draw of the stage `k`: the operation
`g` after the work, when it is one of the draws of the stage. -/
def foldDraw20 (k g : ℕ) : Option (ℕ × FoldGroup) :=
  (sk20[workIndices20.getD k 0 + 1 + g]?).bind fun x =>
    match x.1 with
    | .squeeze level group G => if group = g ∧ g < drawCount20 k then some (level, G) else none
    | _ => none

/-- The stage `k` has a field draw of one or more values exactly when it is not a fold stage. -/
def fieldFits20 (k : ℕ) : Bool :=
  match fieldDraw20 k with
  | some (_, count) => !starts20.contains k && decide (count ≠ 0)
  | none => starts20.contains k

/-- Each draw of the stage `k` is a group of a fold draw whose message has bytes exactly when the
stage is a fold stage. -/
def foldFits20 (k : ℕ) : Bool :=
  (List.range (drawCount20 k)).all fun g =>
    match foldDraw20 k g with
    | some (_, G) => starts20.contains k && decide (G.messageBytes ≠ 0)
    | none => !starts20.contains k

/-- Each stage of the skeleton of `layout20` that is not in `starts20` has a field draw of one or
more values, and no stage of `starts20` has one. -/
theorem skeleton20_fields : (List.range 473).all fieldFits20 = true := by decide +kernel

/-- Each draw of a stage of `starts20` is a group of a fold draw whose message has bytes, and no
other stage has a fold draw. -/
theorem skeleton20_folds : (List.range 473).all foldFits20 = true := by decide +kernel

/-- The fold draw of the stage 1 has three groups, and that of each other fold stage has one. -/
theorem drawCounts20 : starts20.map drawCount20 = [3, 1, 1, 1, 1, 1, 1] := by decide +kernel

theorem fieldDraw20_count {k : ℕ} (inside : k < 473) {key : Key} {count : ℕ}
    (draw : fieldDraw20 k = some (key, count)) : starts20.contains k = false ∧ count ≠ 0 := by
  have fit := List.all_eq_true.mp skeleton20_fields k (List.mem_range.mpr inside)
  unfold fieldFits20 at fit
  rw [draw] at fit
  have both : (!starts20.contains k && decide (count ≠ 0)) = true := fit
  obtain ⟨outside, positive⟩ := Bool.and_eq_true_iff.mp both
  exact ⟨by simpa using outside, of_decide_eq_true positive⟩

theorem foldDraw20_lt {k g level : ℕ} {G : FoldGroup} (draw : foldDraw20 k g = some (level, G)) :
    g < drawCount20 k := by
  obtain ⟨x, -, picked⟩ := Option.bind_eq_some_iff.mp draw
  split at picked
  · split at picked
    · rename_i fits
      exact fits.2
    · cases picked
  · cases picked

theorem foldDraw20_bytes {k g level : ℕ} {G : FoldGroup} (inside : k < 473)
    (draw : foldDraw20 k g = some (level, G)) :
    starts20.contains k = true ∧ G.messageBytes ≠ 0 := by
  have fit := List.all_eq_true.mp skeleton20_folds k (List.mem_range.mpr inside)
  unfold foldFits20 at fit
  have fit := List.all_eq_true.mp fit g (List.mem_range.mpr (foldDraw20_lt draw))
  rw [draw] at fit
  have both : (starts20.contains k && decide (G.messageBytes ≠ 0)) = true := fit
  obtain ⟨fold, positive⟩ := Bool.and_eq_true_iff.mp both
  exact ⟨fold, of_decide_eq_true positive⟩

/-- The squeezes of the groups before the group `g` of the fold draw of the stage `k`, the last
one first. -/
def drawEvents20 (k g : ℕ) : List Duplex.Event :=
  ((sk20.take (workIndices20.getD k 0 + 1 + g)).drop (workIndices20.getD k 0 + 1)).reverse.map
    fun x => Duplex.Event.squeeze x.1.drawBytes

theorem Op.drawBytes_erase (op : Op) : op.erase.drawBytes = op.drawBytes := by cases op <;> rfl

theorem Op.erase_draw {op op' : Op} (draws : isDraw op' = true) (same : op.erase = op') :
    op = op' := by
  subst same
  cases op <;> first | rfl | cases draws

theorem poolBytes_ne_zero {len : ℕ} (positive : len ≠ 0) : poolBytes len ≠ 0 := by
  unfold poolBytes
  rw [ite_eq_right positive]
  exact Nat.mul_ne_zero (by decide)
    (Nat.ne_of_gt (Nat.lt_of_lt_of_le (Nat.pos_of_ne_zero positive) (Nat.le_add_right _ _)))

theorem slice_nil {α : Type} (l : List α) (c : ℕ) : (l.take c).drop c = [] :=
  List.drop_eq_nil_of_le (by rw [List.length_take]; exact Nat.min_le_left _ _)

variable {pub : Public}

/-- The operation of a script of `layout20` at an index of the skeleton. -/
theorem op_of_skeleton (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {i : ℕ}
    {x : Op × ℕ} (at_ : sk20[i]? = some x) : ∃ op, pub.ops[i]? = some op ∧ op.erase = x.1 := by
  have listed : ((skeleton (layout20 0 0)).zipIdx)[i]? = some x := at_
  rw [List.getElem?_zipIdx] at listed
  obtain ⟨op', found, rfl⟩ := Option.map_eq_some_iff.mp listed
  rw [← erased, List.getElem?_map] at found
  obtain ⟨op, here, rfl⟩ := Option.map_eq_some_iff.mp found
  refine ⟨op, ?_, rfl⟩
  rw [← Array.getElem?_toList]
  exact here

theorem fieldDraw20_op (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (inside : k < 473) {key : Key} {count : ℕ} (draw : fieldDraw20 k = some (key, count)) :
    pub.ops[stageIndex20 pub k + 1]? = some (.challenge key count) := by
  obtain ⟨x, found, picked⟩ := Option.bind_eq_some_iff.mp draw
  obtain ⟨op, at_, same⟩ := op_of_skeleton erased found
  split at picked
  · rename_i key' count' shape
    cases picked
    rw [stageIndex20_lt pub inside, at_, Op.erase_draw rfl (same.trans shape)]
  · cases picked

theorem foldDraw20_op (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k g : ℕ}
    (inside : k < 473) {level : ℕ} {G : FoldGroup} (draw : foldDraw20 k g = some (level, G)) :
    pub.ops[stageIndex20 pub k + 1 + g]? = some (.squeeze level g G) := by
  obtain ⟨x, found, picked⟩ := Option.bind_eq_some_iff.mp draw
  obtain ⟨op, at_, same⟩ := op_of_skeleton erased found
  split at picked
  · rename_i level' group G' shape
    split at picked
    · rename_i fits
      cases picked
      obtain ⟨rfl, -⟩ := fits
      rw [stageIndex20_lt pub inside, at_, Op.erase_draw rfl (same.trans shape)]
    · cases picked
  · cases picked

/-- On a script of `layout20`, the squeezes of the draws after the work of the stage `k` and
before the group `g` are `drawEvents20 k g`. -/
theorem drawEvents20_of (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (inside : k < 473) (g : ℕ) :
    (((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1 + g)).drop
      (stageIndex20 pub k + 1)).reverse.map fun x => Duplex.Event.squeeze x.1.drawBytes) =
        drawEvents20 k g := by
  rw [stageIndex20_lt pub inside]
  show _ = (((skeleton (layout20 0 0)).zipIdx.take (workIndices20.getD k 0 + 1 + g)).drop
    (workIndices20.getD k 0 + 1)).reverse.map fun x => Duplex.Event.squeeze x.1.drawBytes
  rw [← take_erase erased, ← List.map_drop, ← List.map_reverse, List.map_map]
  refine List.map_congr_left fun x _ => ?_
  show Duplex.Event.squeeze x.1.drawBytes = Duplex.Event.squeeze x.1.erase.drawBytes
  rw [Op.drawBytes_erase]

end Skeleton

namespace Replayer

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A step without a stop of a field draw: `decodePool` gives values on the squeezed bytes, and
the step appends them to the recorded values of the key and to the challenges. -/
theorem step_challenge_values (i : ℕ) (key : Key) (count : ℕ) (s : Replayer)
    (ran : (((step pub proof messages i (.challenge key count)).run).run s).1 = .ok ()) :
    ∃ values, decodePool (s.sponge.squeeze (poolBytes (2 * count))).1 count = some values ∧
      recordedIn (((step pub proof messages i (.challenge key count)).run).run s).2 key =
        recordedIn s key ++ values ∧
      (((step pub proof messages i (.challenge key count)).run).run s).2.derived.challenges =
        s.derived.challenges.push (key, values) := by
  simp only [step] at ran ⊢
  rw [run_bind, run_squeeze] at ran ⊢
  simp only at ran ⊢
  cases decoded : decodePool (s.sponge.squeeze (poolBytes (2 * count))).1 count with
  | none =>
      rw [decoded] at ran
      cases ran
  | some values =>
      refine ⟨values, rfl, ?_, ?_⟩
      · simp only [run_bind, run_record, run_modify]
        exact recordedIn_of_env rfl
      · simp only [run_bind, run_record, run_modify]

/-- A step without a stop of the draw of a fold group: `foldChallenges` gives challenges on the
squeezed message, and the step appends them to the drawn challenges. -/
theorem step_squeeze_values (i level group : ℕ) (G : FoldGroup) (s : Replayer)
    (ran : (((step pub proof messages i (.squeeze level group G)).run).run s).1 = .ok ()) :
    ∃ cs, foldChallenges G (s.sponge.squeeze G.messageBytes).1 = some cs ∧
      (((step pub proof messages i (.squeeze level group G)).run).run s).2.derived.drawn =
        s.derived.drawn.push (level, group, cs) := by
  rw [step_squeeze, run_bind, run_squeeze] at ran ⊢
  simp only at ran ⊢
  rw [draw_run] at ran ⊢
  cases found : foldChallenges G (s.sponge.squeeze G.messageBytes).1 with
  | none =>
      rw [found] at ran
      cases ran
  | some cs => exact ⟨cs, rfl, rfl⟩

end Replayer

section Draws

variable {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- The field draw of the stage `k`, for a script of `layout20` with the work bits of the model,
a run with no stop up to and with the draw and an ideal sponge: `decodePool` gives values on the
answer for the squeeze of the work and `eventsAt20 … k`, and the step appends these values to the
recorded values of the key and to `derived.challenges`. -/
theorem challenge20_answer (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) {k : ℕ} (inside : k < 473) {key : Key} {count : ℕ}
    (draw : fieldDraw20 k = some (key, count))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1 + 1))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    ∃ values,
      decodePool (answer (.squeeze 8 :: eventsAt20 pub proof messages k) (poolBytes (2 * count)))
        count = some values ∧
      recordedIn (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1 + 1)) key =
        recordedIn (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1)) key ++
          values ∧
      (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1 + 1)).derived.challenges =
        (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1)).derived.challenges.push
          (key, values) := by
  obtain ⟨site, level, work⟩ := bits k inside
  obtain ⟨stepped, after⟩ :=
    stateBefore_succ pub proof messages pub.ops (fieldDraw20_op erased inside draw) ran
  have before := runFrom_take_ok pub proof messages _ (stageIndex20 pub k + 1) _ ran
  rw [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at before
  obtain ⟨-, squeezed⟩ := draw_squeeze20 work (bits20_ne_zero inside) (Nat.le_refl _)
    (fun x member => by rw [slice_nil] at member; cases member) before ideal
  have pool := squeezed (poolBytes (2 * count))
    (poolBytes_ne_zero (Nat.mul_ne_zero (by decide) (fieldDraw20_count inside draw).2))
  rw [slice_nil] at pool
  obtain ⟨values, decoded, recorded, challenges⟩ :=
    step_challenge_values pub proof messages _ key count _ stepped
  refine ⟨values, ?_, ?_, ?_⟩
  · rw [← decoded, pool]
    rfl
  · rw [after]
    exact recorded
  · rw [after]
    exact challenges

/-- The group `g` of the fold draw of the stage `k`, for a script of `layout20` with the work
bits of the model, a run with no stop up to and with the draw of the group and an ideal sponge:
`foldChallenges` gives challenges on the answer for the squeezes of the groups before `g`, the
squeeze of the work and `eventsAt20 … k`, and the step appends the entry of these challenges to
`derived.drawn`. -/
theorem squeeze20_answer (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) {k g : ℕ} (inside : k < 473) {level : ℕ} {G : FoldGroup}
    (draw : foldDraw20 k g = some (level, G))
    (ran : (runFrom pub proof messages
      (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1 + g + 1)) (channelStart pub)).1 =
        .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    ∃ cs,
      foldChallenges G (answer (drawEvents20 k g ++ .squeeze 8 :: eventsAt20 pub proof messages k)
        G.messageBytes) = some cs ∧
      (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1 + g + 1)).derived.drawn =
        (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1 + g)).derived.drawn.push
          (level, g, cs) := by
  obtain ⟨site, level', work⟩ := bits k inside
  obtain ⟨stepped, after⟩ :=
    stateBefore_succ pub proof messages pub.ops (foldDraw20_op erased inside draw) ran
  have before := runFrom_take_ok pub proof messages _ (stageIndex20 pub k + 1 + g) _ ran
  rw [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at before
  obtain ⟨-, squeezed⟩ := draw_squeeze20 work (bits20_ne_zero inside) (Nat.le_add_right _ g)
    (draws20_of erased inside
      (Nat.add_le_add_left (Nat.le_of_lt (foldDraw20_lt draw)) _)) before ideal
  have message := squeezed G.messageBytes (foldDraw20_bytes inside draw).2
  rw [drawEvents20_of erased inside g] at message
  obtain ⟨cs, sampled, drawn⟩ := step_squeeze_values pub proof messages _ level g G _ stepped
  refine ⟨cs, ?_, ?_⟩
  · rw [← sampled, message]
  · rw [after]
    exact drawn

end Draws

section Later

/-- The count of the values that the draws before the field draw of the stage `k` append to its
key. -/
def offset20 (k : ℕ) : ℕ :=
  match fieldDraw20 k with
  | some (key, _) => (drawnCount key (sk20.take (workIndices20.getD k 0 + 1))).getD 0
  | none => 0

/-- Each operation before the field draw of the stage `k` that writes its key is a field draw. -/
def offsetFits20 (k : ℕ) : Bool :=
  match fieldDraw20 k with
  | some (key, _) => (drawnCount key (sk20.take (workIndices20.getD k 0 + 1))).isSome
  | none => true

/-- No operation before the group `g` of the fold draw of the stage `k` is a squeeze of this
group of its level. -/
def freshFits20 (k : ℕ) : Bool :=
  (List.range (drawCount20 k)).all fun g =>
    match foldDraw20 k g with
    | some (level, _) =>
        (squeezeOf level g (sk20.take (workIndices20.getD k 0 + 1 + g))).isNone
    | none => true

/-- In the skeleton of `layout20`, only field draws write the key of a field draw before it. -/
theorem skeleton20_offsets : (List.range 473).all offsetFits20 = true := by decide +kernel

/-- In the skeleton of `layout20`, each group of a level has one squeeze. -/
theorem skeleton20_fresh : (List.range 473).all freshFits20 = true := by decide +kernel

theorem foldDraw20_fresh {k g level : ℕ} {G : FoldGroup} (inside : k < 473)
    (draw : foldDraw20 k g = some (level, G)) :
    squeezeOf level g (sk20.take (workIndices20.getD k 0 + 1 + g)) = none := by
  have fit := List.all_eq_true.mp skeleton20_fresh k (List.mem_range.mpr inside)
  unfold freshFits20 at fit
  have fit := List.all_eq_true.mp fit g (List.mem_range.mpr (foldDraw20_lt draw))
  rw [draw] at fit
  exact Option.isNone_iff_eq_none.mp fit

variable {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- On a script of `layout20`, after a run with no stop up to and with the work of the stage `k`,
the key of the field draw of the stage has `offset20 k` recorded values. -/
theorem offset20_size (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (inside : k < 473) {key : Key} {count : ℕ} (draw : fieldDraw20 k = some (key, count))
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1))
      (channelStart pub)).1 = .ok ()) :
    (recordedIn (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1)) key).size =
      offset20 k := by
  have fit := List.all_eq_true.mp skeleton20_offsets k (List.mem_range.mpr inside)
  unfold offsetFits20 at fit
  rw [draw] at fit
  obtain ⟨n, counted⟩ := Option.isSome_iff_exists.mp fit
  have value : offset20 k = n := by
    unfold offset20
    rw [draw]
    show (drawnCount key (sk20.take (workIndices20.getD k 0 + 1))).getD 0 = n
    rw [counted]
    rfl
  have counted' : drawnCount key (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1)) =
      some n := by
    rw [← drawnCount_erase, take_erase erased, stageIndex20_lt pub inside]
    exact counted
  have size := size_drawnCount pub proof messages key _ _ counted' before
  rw [channelStart_fresh] at size
  rw [value]
  exact size.trans (Nat.zero_add n)

/-- The values of the field draw of the stage `k` in the recorded values of each later state of a
run with no stop, for a script of `layout20` with the work bits of the model and an ideal
sponge: the `count` recorded values of the key from the position `offset20 k` are the values that
`decodePool` gives on the answer for the squeeze of the work and `eventsAt20 … k`. -/
theorem challenge20_recorded (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) {k : ℕ} (inside : k < 473) {key : Key} {count : ℕ}
    (draw : fieldDraw20 k = some (key, count)) {c : ℕ} (later : stageIndex20 pub k + 1 + 1 ≤ c)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    ∃ values,
      decodePool (answer (.squeeze 8 :: eventsAt20 pub proof messages k) (poolBytes (2 * count)))
        count = some values ∧
      ((recordedIn (stateBefore pub proof messages pub.ops c) key).toList.drop (offset20 k)).take
        count = values.toList := by
  obtain ⟨first, -, cut⟩ := stateBefore_le pub proof messages pub.ops later ran
  obtain ⟨values, decoded, recorded, -⟩ := challenge20_answer erased bits inside draw first ideal
  have before := runFrom_take_ok pub proof messages _ (stageIndex20 pub k + 1) _ first
  rw [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at before
  have size := offset20_size erased inside draw before
  obtain ⟨extra, grown⟩ := (runFrom_grows pub proof messages
    ((pub.ops.zipIdx.toList.take c).drop (stageIndex20 pub k + 1 + 1))
    (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1 + 1))).prefix_ key
  rw [← cut, recorded, Array.toList_append] at grown
  refine ⟨values, decoded, ?_⟩
  rw [← grown, List.append_assoc,
    List.drop_left' (by rw [Array.length_toList]; exact size),
    List.take_left' (by rw [Array.length_toList]; exact decodePool_size decoded)]

namespace Replayer

variable (pub proof messages) in
/-- A run without a stop of operations with no squeeze of the group `group` of `level` adds no
drawn challenges of this group. -/
theorem find_drawn_none (level group : ℕ) : ∀ (l : List (Op × ℕ)) (s : Replayer),
    s.derived.drawn.find? (fun x => x.1 == level && x.2.1 == group) = none →
    squeezeOf level group l = none → (runFrom pub proof messages l s).1 = .ok () →
    (runFrom pub proof messages l s).2.derived.drawn.find?
      (fun x => x.1 == level && x.2.1 == group) = none
  | [], _, missing, _, _ => missing
  | (op, i) :: l, s, missing, first, ran => by
      obtain ⟨stepped, rest⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped]
      simp only [squeezeOf] at first
      by_cases squeezed : ∃ l' g' G', op = .squeeze l' g' G'
      · obtain ⟨l', g', G', rfl⟩ := squeezed
        obtain ⟨cs, pushed⟩ := step_squeeze_drawn pub proof messages i l' g' G' s stepped
        have here : ¬(l' = level ∧ g' = group) := by
          intro both
          simp [Op.drawsGroup, both] at first
        simp only [Op.drawsGroup, here, ↓reduceIte, Option.none_or] at first
        refine find_drawn_none level group l _ ?_ first rest
        rw [pushed, Array.find?_push, missing]
        have differ : (l' == level && g' == group) = false := by
          simpa [Bool.and_eq_true, beq_iff_eq] using here
        simp [differ]
      · have other : ∀ l' g' G', op ≠ .squeeze l' g' G' :=
          fun l' g' G' same => squeezed ⟨l', g', G', same⟩
        have quiet : op.drawsGroup level group = none := by
          cases op <;> first | rfl | exact absurd rfl (other _ _ _)
        rw [quiet, Option.none_or] at first
        refine find_drawn_none level group l _ ?_ first rest
        rw [step_drawn pub proof messages i op other s]
        exact missing

end Replayer

/-- The challenges of the group `g` of the fold draw of the stage `k` on each later state of a run
with no stop, for a script of `layout20` with the work bits of the model and an ideal sponge:
the drawn challenges of the group are the challenges that `foldChallenges` gives on the answer
for the squeezes of the groups before `g`, the squeeze of the work and `eventsAt20 … k`. -/
theorem squeeze20_drawn (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) {k g : ℕ} (inside : k < 473) {level : ℕ} {G : FoldGroup}
    (draw : foldDraw20 k g = some (level, G)) {c : ℕ} (later : stageIndex20 pub k + 1 + g + 1 ≤ c)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    foldChallenges G (answer (drawEvents20 k g ++ .squeeze 8 :: eventsAt20 pub proof messages k)
      G.messageBytes) = some (drawnIn (stateBefore pub proof messages pub.ops c) level g) := by
  obtain ⟨first, -, cut⟩ := stateBefore_le pub proof messages pub.ops later ran
  obtain ⟨cs, sampled, drawn⟩ := squeeze20_answer erased bits inside draw first ideal
  have before := runFrom_take_ok pub proof messages _ (stageIndex20 pub k + 1 + g) _ first
  rw [List.take_take, Nat.min_eq_left (Nat.le_succ _)] at before
  have fresh : squeezeOf level g (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1 + g)) =
      none := by
    rw [← squeezeOf_erase, take_erase erased, stageIndex20_lt pub inside]
    exact foldDraw20_fresh inside draw
  have missing : (stateBefore pub proof messages pub.ops
      (stageIndex20 pub k + 1 + g)).derived.drawn.find?
        (fun x => x.1 == level && x.2.1 == g) = none :=
    find_drawn_none pub proof messages level g _ _ (by simp [channelStart]) fresh before
  have found : (stateBefore pub proof messages pub.ops
      (stageIndex20 pub k + 1 + g + 1)).derived.drawn.find?
        (fun x => x.1 == level && x.2.1 == g) = some (level, g, cs) := by
    rw [drawn, Array.find?_push, missing]
    simp
  rw [sampled]
  unfold drawnIn
  rw [cut, find_drawn_grows (runFrom_grows pub proof messages _ _).drawn found]
  rfl

/-- The stage 0 of the skeleton of `layout20` draws the three values of `row_coefficients`, the
first values of this key, before the `fused.end` of the root. -/
theorem row20_first : fieldDraw20 0 = some (⟨0, "row_coefficients", none⟩, 3) ∧
    offset20 0 = 0 ∧ workIndices20.getD 0 0 + 1 + 1 ≤ 165 := by decide +kernel

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The weights of the batching root that `AgreesAt20` reads (`weights20`), for a script of
`layout20` with the work bits of the model, a run with no stop before the `fused.end` of the root
and an ideal sponge: the images of the three values that `decodePool` gives on the answer for the
squeeze of the work of the stage 0 and `eventsAt20 … 0`. -/
theorem weights20_answer {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (bits : Bits20 pub)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    ∃ values,
      decodePool (answer (.squeeze 8 :: eventsAt20 pub proof messages 0) (poolBytes (2 * 3))) 3 =
        some values ∧
      ∀ g : Fin 3, weights20 pub proof messages E g = (values.getD g Ext.zero).into := by
  have later : stageIndex20 pub 0 + 1 + 1 ≤ fusedIndex pub.layout 0 := by
    rw [layout, fusedIndex20, fusedIndex20_values.1, stageIndex20_lt pub (by decide)]
    exact row20_first.2.2
  obtain ⟨values, decoded, slice⟩ :=
    challenge20_recorded erased bits (by decide) row20_first.1 later ran ideal
  rw [row20_first.2.1, List.drop_zero] at slice
  refine ⟨values, decoded, fun g => ?_⟩
  have entry : (recordedIn (fused20 pub proof messages 0)
      ⟨0, "row_coefficients", none⟩).toList[(g : ℕ)]? = values.toList[(g : ℕ)]? := by
    rw [← slice, List.getElem?_take_of_lt g.isLt]
  unfold weights20
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, entry,
    Array.getElem?_toList]

/-- The fold stage of each level 0 to 5 of the skeleton of `layout20` is a stage, and its draws
are before the `fused.end` of the level. -/
theorem folds20_before : (List.range 6).all (fun j =>
    decide (starts20.getD j 0 < 473) &&
      decide (workIndices20.getD (starts20.getD j 0) 0 + 1 + drawCount20 (starts20.getD j 0) ≤
        fusedIndex (layout20 0 0) j)) = true := by decide +kernel

/-- The fold draw of the fold stage `j` of `starts20` is the draw of the level `j`: the levels 0
to 5 and the terminal, the level 6. -/
theorem foldLevels20 : starts20.map (fun k => (foldDraw20 k 0).map (·.1)) =
    [some 0, some 1, some 2, some 3, some 4, some 5, some 6] := by decide +kernel

/-- The fold challenges of the level `ℓ` that `AgreesAt20` reads (the drawn challenges on the
state before the `fused.end` of the level), for a script of `layout20` with the work bits of the
model, a run with no stop before this `fused.end` and an ideal sponge: the challenges that
`foldChallenges` gives on the answer for the squeezes of the groups before `g`, the squeeze of
the work of the fold stage and its `eventsAt20`. -/
theorem drawn20_answer {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (bits : Bits20 pub)
    (ℓ : Fin 6) {g level : ℕ} {G : FoldGroup}
    (draw : foldDraw20 (starts20.getD ℓ 0) g = some (level, G))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (fusedIndex pub.layout ℓ))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer) :
    foldChallenges G (answer (drawEvents20 (starts20.getD ℓ 0) g ++
        .squeeze 8 :: eventsAt20 pub proof messages (starts20.getD ℓ 0)) G.messageBytes) =
      some (drawnIn (fused20 pub proof messages ℓ) level g) := by
  have fit := List.all_eq_true.mp folds20_before ℓ (List.mem_range.mpr ℓ.isLt)
  obtain ⟨inside, bound⟩ := Bool.and_eq_true_iff.mp fit
  have inside := of_decide_eq_true inside
  have bound := of_decide_eq_true bound
  have later : stageIndex20 pub (starts20.getD ℓ 0) + 1 + g + 1 ≤ fusedIndex pub.layout ℓ := by
    rw [layout, fusedIndex20, stageIndex20_lt pub inside]
    have below := foldDraw20_lt draw
    omega
  exact squeeze20_drawn erased bits inside draw later ran ideal

end Later

end NightstreamFPrime.Spec.Compression.Akita.Replay
