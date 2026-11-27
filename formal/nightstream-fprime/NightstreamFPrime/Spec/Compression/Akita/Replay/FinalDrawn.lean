import NightstreamFPrime.Spec.Compression.Akita.Replay.RunStop
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerStage

/-! The drawn fold challenges of a group on the state of a run of the operations of the script
(Akita revision `d50f700`, `record_fold_challenges` of `fold/mod.rs`; the operation `.squeeze` of
`Build.foldChallenges` and `Replayer.draw`).

- `KeepsDrawn m`: `m` does not change the drawn challenges; `step_drawn`: each step of an operation
  that is not a squeeze keeps them. `step_squeeze_drawn`: a squeeze without a stop appends the
  challenges of the sampler on its message, moved by `ambient` of its family.
- `squeezeOf level group l` is the family of the first squeeze of the group `group` of `level` in
  `l`, and `drawnShape` states that it is the family of the layout for each group of a fold.
  `drawnIn_of_run` and `drawnIn_before`: on the state of a run without a stop, the drawn challenges
  of the group are `cs.map (ambient G)` for this family `G` (the hypothesis `drawnAt` of
  `batchAmbiented` and `packAmbiented`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The family of the fold challenges of a squeeze of the group `group` of `level`. -/
def Op.drawsGroup (level group : ℕ) : Op → Option FoldGroup
  | .squeeze l g G => if l = level ∧ g = group then some G else none
  | _ => none

/-- The family of the first squeeze of the group `group` of `level` in `l`. -/
def squeezeOf (level group : ℕ) : List (Op × ℕ) → Option FoldGroup
  | [] => none
  | x :: l => (x.1.drawsGroup level group).or (squeezeOf level group l)

/-- The first squeeze of each group `g` of a fold at `level` in `l` has the family `groups[g]`. -/
def drawnShape (l : List (Op × ℕ)) (level : ℕ) (groups : List FoldGroup) : Bool :=
  (List.range groups.length).all fun g => squeezeOf level g l == groups[g]?

theorem squeezeOf_of_shape {l : List (Op × ℕ)} {level : ℕ} {groups : List FoldGroup}
    (shaped : drawnShape l level groups = true) {g : ℕ} {G : FoldGroup}
    (listed : groups[g]? = some G) : squeezeOf level g l = some G := by
  unfold drawnShape at shaped
  rw [List.all_eq_true] at shaped
  have inside := shaped g (List.mem_range.mpr (List.getElem?_eq_some_iff.mp listed).1)
  rw [listed, beq_iff_eq] at inside
  exact inside

namespace Replayer

/-- `m` keeps the drawn fold challenges. -/
def KeepsDrawn {α : Type} (m : ReplayM α) : Prop :=
  ∀ s, ((m.run).run s).2.derived.drawn = s.derived.drawn

section Drawn

theorem keepsDrawn_bind {α β : Type} {m : ReplayM α} {f : α → ReplayM β} (hm : KeepsDrawn m)
    (hf : ∀ a, KeepsDrawn (f a)) : KeepsDrawn (m >>= f) := by
  intro s
  rw [run_bind]
  rcases found : (m.run).run s with ⟨_ | a, s₁⟩
  · have kept := hm s
    rw [found] at kept
    exact kept
  · have kept := hm s
    rw [found] at kept
    exact (hf a s₁).trans kept

theorem keepsDrawn_pure {α : Type} (a : α) : KeepsDrawn (Pure.pure a : ReplayM α) := fun _ => rfl

theorem keepsDrawn_throw {α : Type} (e : Stop) : KeepsDrawn (throw e : ReplayM α) := fun _ => rfl

theorem keepsDrawn_get : KeepsDrawn (get : ReplayM Replayer) := fun _ => rfl

theorem keepsDrawn_modify {g : Replayer → Replayer} (h : ∀ s, (g s).derived.drawn = s.derived.drawn) :
    KeepsDrawn (modify g : ReplayM Unit) := h

theorem keepsDrawn_mapM {α β : Type} {f : α → ReplayM β} (each : ∀ a, KeepsDrawn (f a)) :
    ∀ l : List α, KeepsDrawn (l.mapM f)
  | [] => keepsDrawn_pure _
  | a :: l => by
      rw [List.mapM_cons]
      exact keepsDrawn_bind (each a) fun _ =>
        keepsDrawn_bind (keepsDrawn_mapM each l) fun _ => keepsDrawn_pure _

theorem keepsDrawn_foldlM {α β : Type} {f : β → α → ReplayM β}
    (each : ∀ b a, KeepsDrawn (f b a)) : ∀ (l : List α) (b : β), KeepsDrawn (l.foldlM f b)
  | [], _ => keepsDrawn_pure _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact keepsDrawn_bind (each b a) fun c => keepsDrawn_foldlM each l c

theorem keepsDrawn_recorded (key : Key) : KeepsDrawn (recorded key) :=
  keepsDrawn_bind keepsDrawn_get fun _ => keepsDrawn_pure _

theorem keepsDrawn_first (key : Key) : KeepsDrawn (first key) :=
  keepsDrawn_bind keepsDrawn_get fun _ => keepsDrawn_pure _

theorem keepsDrawn_fieldsOf (key : Key) : KeepsDrawn (fieldsOf key) :=
  keepsDrawn_bind keepsDrawn_get fun _ => keepsDrawn_pure _

theorem keepsDrawn_pointBefore (before : Before) : KeepsDrawn (pointBefore before) :=
  keepsDrawn_bind keepsDrawn_get fun _ => keepsDrawn_pure _

theorem keepsDrawn_absorb (bytes : ByteArray) : KeepsDrawn (absorb bytes) :=
  keepsDrawn_modify fun _ => rfl

theorem keepsDrawn_record (key : Key) (values : Array Ext) : KeepsDrawn (record key values) :=
  keepsDrawn_modify fun _ => rfl

theorem keepsDrawn_squeeze (count : ℕ) : KeepsDrawn (squeeze count) := fun _ => rfl

theorem keepsDrawn_message (i : ℕ) (messages : Array Message) :
    KeepsDrawn (message i messages) := by
  intro s
  rw [run_message]
  cases messages[s.next]? <;> rfl

macro "replay_drawn" : tactic => `(tactic| repeat' (first
  | with_reducible exact keepsDrawn_pure _ | with_reducible exact keepsDrawn_throw _
  | with_reducible exact keepsDrawn_get
  | with_reducible exact keepsDrawn_squeeze _ | with_reducible exact keepsDrawn_recorded _
  | with_reducible exact keepsDrawn_first _ | with_reducible exact keepsDrawn_fieldsOf _
  | with_reducible exact keepsDrawn_pointBefore _ | with_reducible exact keepsDrawn_absorb _
  | with_reducible exact keepsDrawn_record _ _ | with_reducible exact keepsDrawn_message _ _
  | (with_reducible refine keepsDrawn_modify fun _ => ?_; rfl)
  | refine keepsDrawn_bind ?_ fun _ => ?_
  | split))

theorem keepsDrawn_stage2Of (pub : Public) (level : ℕ) (F : Fold) :
    KeepsDrawn (stage2Of pub level F) := by
  unfold stage2Of
  replay_drawn

theorem keepsDrawn_value (pub : Public) (key : Key) (v : Value) : KeepsDrawn (value pub key v) := by
  cases v with
  | recorded keys =>
      exact keepsDrawn_bind (keepsDrawn_foldlM (fun _ _ => keepsDrawn_bind (keepsDrawn_recorded _)
        fun _ => keepsDrawn_pure _) keys _) fun _ => keepsDrawn_pure _
  | fused F =>
      simp only [value]
      refine keepsDrawn_bind (keepsDrawn_stage2Of pub _ F) fun _ => ?_
      replay_drawn
  | _ => simp only [value]; replay_drawn

theorem keepsDrawn_weightsOf (weights : Weights) : KeepsDrawn (weightsOf weights) := by
  cases weights <;> simp only [weightsOf] <;> replay_drawn

theorem keepsDrawn_holds (pub : Public) (check : Check) : KeepsDrawn (holds pub check) := by
  cases check <;> simp only [holds]
  · replay_drawn
  · refine keepsDrawn_bind (keepsDrawn_first _) fun _ => keepsDrawn_bind (keepsDrawn_weightsOf _)
      fun _ => keepsDrawn_bind (keepsDrawn_recorded _) fun _ => keepsDrawn_pure _
  all_goals replay_drawn

theorem keepsDrawn_checkWith (pub : Public) (i : ℕ) (key : Key) (check : Check)
    (out : Option StageOut) : KeepsDrawn (checkWith pub i key check out) := by
  cases out with
  | some out => simp only [checkWith]; replay_drawn
  | none =>
      simp only [checkWith]
      refine keepsDrawn_bind (keepsDrawn_holds pub check) fun _ => ?_
      replay_drawn

theorem keepsDrawn_terminalWith (i level : ℕ) (r : Option (Except String TerminalValues)) :
    KeepsDrawn (terminalWith i level r) := by
  rcases r with _ | _ | _ <;> simp only [terminalWith] <;> replay_drawn

theorem keepsDrawn_decodeExts (bytes : ByteArray) : KeepsDrawn (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact keepsDrawn_pure _

end Drawn

macro "replay_drawn_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact keepsDrawn_pure _ | with_reducible exact keepsDrawn_throw _
  | with_reducible exact keepsDrawn_get
  | with_reducible exact keepsDrawn_squeeze _ | with_reducible exact keepsDrawn_recorded _
  | with_reducible exact keepsDrawn_first _ | with_reducible exact keepsDrawn_pointBefore _
  | with_reducible exact keepsDrawn_absorb _ | with_reducible exact keepsDrawn_record _ _
  | with_reducible exact keepsDrawn_message _ _
  | with_reducible exact keepsDrawn_value _ _ _
  | with_reducible exact keepsDrawn_terminalWith _ _ _
  | with_reducible exact keepsDrawn_checkWith _ _ _ _ _
  | with_reducible exact keepsDrawn_mapM (fun _ => keepsDrawn_recorded _) _
  | with_reducible exact keepsDrawn_decodeExts _
  | (with_reducible refine keepsDrawn_modify fun _ => ?_; rfl)
  | refine keepsDrawn_bind ?_ fun _ => ?_
  | split))

section Steps

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- Each step of an operation that is not a squeeze keeps the drawn fold challenges. -/
theorem step_drawn (i : ℕ) (op : Op) (other : ∀ level group G, op ≠ .squeeze level group G) :
    KeepsDrawn (step pub proof messages i op) := by
  cases op
  case squeeze level group G => exact absurd rfl (other level group G)
  all_goals simp only [step]; replay_drawn_step

/-- A draw after a read of its message that keeps the drawn challenges appends the challenges of
the sampler on the message, moved by `ambient` of its family. -/
theorem bind_draw_drawn (i level group : ℕ) (G : FoldGroup) {m : ReplayM ByteArray}
    (keep : KeepsDrawn m) (s : Replayer)
    (ran : (((m >>= fun bytes => draw i level group G bytes).run).run s).1 = .ok ()) :
    ∃ cs : List Sparse,
      (((m >>= fun bytes => draw i level group G bytes).run).run s).2.derived.drawn =
        s.derived.drawn.push (level, group, cs.map (ambient G)) := by
  rw [run_bind] at ran ⊢
  rcases found : (m.run).run s with ⟨_ | bytes, s₁⟩
  · rw [found] at ran
    cases ran
  · have kept : s₁.derived.drawn = s.derived.drawn := by
      have kept := keep s
      rw [found] at kept
      exact kept
    rw [found] at ran
    have drawOk : (((draw i level group G bytes).run).run s₁).1 = .ok () := ran
    rcases drawn : ((draw i level group G bytes).run).run s₁ with ⟨_ | ⟨⟩, s₂⟩
    · rw [drawn] at drawOk
      cases drawOk
    · obtain ⟨cs, -, same⟩ := drawn_sample drawn
      refine ⟨cs, ?_⟩
      show (((draw i level group G bytes).run).run s₁).2.derived.drawn = _
      rw [drawn, same, ← kept]
      rfl

/-- A squeeze without a stop appends the challenges of the sampler on its message, moved by
`ambient` of its family. -/
theorem step_squeeze_drawn (i level group : ℕ) (G : FoldGroup) (s : Replayer)
    (ran : (((step pub proof messages i (.squeeze level group G)).run).run s).1 = .ok ()) :
    ∃ cs : List Sparse,
      (((step pub proof messages i (.squeeze level group G)).run).run s).2.derived.drawn =
        s.derived.drawn.push (level, group, cs.map (ambient G)) := by
  rw [step_squeeze] at ran ⊢
  exact bind_draw_drawn i level group G (keepsDrawn_squeeze _) s ran

/-- A found entry of the drawn challenges stays the first entry of its group. -/
theorem find_drawn_grows {s s' : Replayer}
    (grown : s.derived.drawn.toList <+: s'.derived.drawn.toList)
    {p : ℕ × ℕ × List Sparse → Bool} {x : ℕ × ℕ × List Sparse}
    (found : s.derived.drawn.find? p = some x) : s'.derived.drawn.find? p = some x := by
  obtain ⟨extra, grown⟩ := grown
  rw [← Array.find?_toList, ← grown, List.find?_append, Array.find?_toList, found]
  rfl

/-- On the state of a run without a stop from a state with no drawn challenge of the group
`group` of `level`, when the first squeeze of this group in the operations has the family `G`,
the drawn challenges of the group are the challenges of the sampler moved by `ambient G`. -/
theorem drawnIn_of_run (level group : ℕ) (G : FoldGroup) :
    ∀ (l : List (Op × ℕ)) (s : Replayer),
      s.derived.drawn.find? (fun x => x.1 == level && x.2.1 == group) = none →
      squeezeOf level group l = some G → (runFrom pub proof messages l s).1 = .ok () →
      ∃ cs : List Sparse, drawnIn (runFrom pub proof messages l s).2 level group =
        cs.map (ambient G)
  | [], _, _, first, _ => by cases first
  | (op, i) :: l, s, missing, first, ran => by
      obtain ⟨stepped, rest⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped]
      simp only [squeezeOf] at first
      by_cases squeezed : ∃ l' g' G', op = .squeeze l' g' G'
      · obtain ⟨l', g', G', rfl⟩ := squeezed
        obtain ⟨cs, pushed⟩ := step_squeeze_drawn pub proof messages i l' g' G' s stepped
        by_cases here : l' = level ∧ g' = group
        · obtain ⟨atLevel, atGroup⟩ := here
          subst l' g'
          simp only [Op.drawsGroup, and_self, ↓reduceIte, Option.some_or,
            Option.some.injEq] at first
          subst G'
          have found : (((step pub proof messages i (.squeeze level group G)).run).run
              s).2.derived.drawn.find? (fun x => x.1 == level && x.2.1 == group) =
                some (level, group, cs.map (ambient G)) := by
            rw [pushed, Array.find?_push, missing]
            simp
          refine ⟨cs, ?_⟩
          unfold drawnIn
          rw [find_drawn_grows (runFrom_grows pub proof messages l _).drawn found]
          rfl
        · simp only [Op.drawsGroup, here, ↓reduceIte, Option.none_or] at first
          refine drawnIn_of_run level group G l _ ?_ first rest
          rw [pushed, Array.find?_push, missing]
          have differ : (l' == level && g' == group) = false := by
            simpa [Bool.and_eq_true, beq_iff_eq] using here
          simp [differ]
      · have other : ∀ l' g' G', op ≠ .squeeze l' g' G' :=
          fun l' g' G' same => squeezed ⟨l', g', G', same⟩
        have quiet : op.drawsGroup level group = none := by
          cases op <;> first | rfl | exact absurd rfl (other _ _ _)
        rw [quiet, Option.none_or] at first
        refine drawnIn_of_run level group G l _ ?_ first rest
        rw [step_drawn pub proof messages i op other s]
        exact missing

/-- `drawnIn_of_run` on the state before the operation `c` of the script. -/
theorem drawnIn_before (ops : Array Op) {c : ℕ}
    (ok : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok ())
    {level group : ℕ} {G : FoldGroup}
    (first : squeezeOf level group (ops.zipIdx.toList.take c) = some G) :
    ∃ cs : List Sparse, drawnIn (stateBefore pub proof messages ops c) level group =
      cs.map (ambient G) :=
  drawnIn_of_run pub proof messages level group G _ _ (by simp [channelStart]) first ok

end Steps

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
