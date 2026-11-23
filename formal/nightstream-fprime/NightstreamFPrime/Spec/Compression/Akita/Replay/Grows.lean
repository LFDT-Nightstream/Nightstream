import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript

/-! The replay of `Replay/Transcript.lean` only appends: a value that it reads at a check is the
value that an earlier operation recorded, when no operation between them writes its key.

- `Op.writes` are the keys whose recorded values, field message, integer or bytes an operation
  writes. `Grows W s s'`: from `s` to `s'` each recorded value is a prefix of the later one, no
  value of a key outside `W` changes, and the rounds and the drawn challenges only grow.
  `Keeps W m`: `m` grows each state with `W`, on success and on a stop.
- `step_keeps`: each step keeps its writes. `keeps_forM`: a run of operations keeps the writes of
  its operations. `runOps_grows` and `lookup_later`: after `done ++ later` each value of a key
  that no operation of `later` writes is the value after `done`. `replay_runOps`: `replay` is
  `runOps` on the indexed operations of the script. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The keys whose recorded values, field messages, integers or bytes an operation writes. -/
def Op.writes : Op → List Key
  | .absorb key _ | .fields key _ | .exts key _ | .integer key | .digest key | .bounded key _
  | .challenge key _ => [key]
  | .ended key _ _ => [⟨key.level, key.site ++ ".output", none⟩]
  | _ => []

namespace Replayer

theorem run_bind {α β : Type} (m : ReplayM α) (f : α → ReplayM β) (s : Replayer) :
    ((m >>= f).run).run s = match (m.run).run s with
      | (.ok a, s1) => ((f a).run).run s1
      | (.error e, s1) => (.error e, s1) := by
  rw [ExceptT.run_bind, StateT.run_bind]
  rcases (m.run).run s with ⟨_ | a, s1⟩ <;> rfl

/-- From `s` to `s'` the replay only appends to the recorded values of each key, it changes no
recorded value, field message, integer or bytes of a key outside `W`, and it only appends to the
rounds and the drawn challenges. -/
structure Grows (W : List Key) (s s' : Replayer) : Prop where
  prefix_ : ∀ key, (recordedIn s key).toList <+: (recordedIn s' key).toList
  same : ∀ key, key ∉ W → recordedIn s' key = recordedIn s key ∧
    s'.fields.get? key = s.fields.get? key ∧ s'.integers.get? key = s.integers.get? key ∧
    s'.derived.bytes.get? key = s.derived.bytes.get? key
  rounds : s.derived.rounds.toList <+: s'.derived.rounds.toList
  drawn : s.derived.drawn.toList <+: s'.derived.drawn.toList

theorem Grows.of_eq {W : List Key} {s s' : Replayer} (env : s'.env = s.env)
    (fields : s'.fields = s.fields) (integers : s'.integers = s.integers)
    (bytes : s'.derived.bytes = s.derived.bytes) (rounds : s'.derived.rounds = s.derived.rounds)
    (drawn : s'.derived.drawn = s.derived.drawn) : Grows W s s' :=
  ⟨fun key => by unfold recordedIn; rw [env],
    fun key _ => by unfold recordedIn; rw [env, fields, integers, bytes]; exact ⟨rfl, rfl, rfl, rfl⟩,
    by rw [rounds], by rw [drawn]⟩

theorem Grows.refl (W : List Key) (s : Replayer) : Grows W s s :=
  Grows.of_eq rfl rfl rfl rfl rfl rfl

theorem Grows.trans {W : List Key} {s₁ s₂ s₃ : Replayer} (a : Grows W s₁ s₂)
    (b : Grows W s₂ s₃) : Grows W s₁ s₃ :=
  ⟨fun k => (a.prefix_ k).trans (b.prefix_ k), fun k out => by
    obtain ⟨a1, a2, a3, a4⟩ := a.same k out
    obtain ⟨b1, b2, b3, b4⟩ := b.same k out
    exact ⟨b1.trans a1, b2.trans a2, b3.trans a3, b4.trans a4⟩,
    a.rounds.trans b.rounds, a.drawn.trans b.drawn⟩

theorem Grows.mono {W W' : List Key} {s s' : Replayer} (sub : ∀ k ∈ W, k ∈ W')
    (a : Grows W s s') : Grows W' s s' :=
  ⟨a.prefix_, fun k out => a.same k fun inside => out (sub k inside), a.rounds, a.drawn⟩

/-- `m` grows each state with the writes `W`, on success and on a stop. -/
def Keeps {α : Type} (W : List Key) (m : ReplayM α) : Prop := ∀ s, Grows W s ((m.run).run s).2

theorem keeps_bind {α β : Type} {W : List Key} {m : ReplayM α} {f : α → ReplayM β}
    (hm : Keeps W m) (hf : ∀ a, Keeps W (f a)) : Keeps W (m >>= f) := by
  intro s
  rw [run_bind]
  rcases found : (m.run).run s with ⟨_ | a, s1⟩
  · have grows := hm s
    rw [found] at grows
    exact grows
  · have grows := hm s
    rw [found] at grows
    exact grows.trans (hf a s1)

theorem keeps_pure {α : Type} {W : List Key} (a : α) : Keeps W (Pure.pure a : ReplayM α) :=
  fun s => Grows.refl W s

theorem keeps_throw {α : Type} {W : List Key} (e : Stop) : Keeps W (throw e : ReplayM α) :=
  fun s => Grows.refl W s

theorem keeps_get {W : List Key} : Keeps W (get : ReplayM Replayer) := fun s => Grows.refl W s

theorem keeps_mono {α : Type} {W W' : List Key} {m : ReplayM α} (sub : ∀ k ∈ W, k ∈ W')
    (h : Keeps W m) : Keeps W' m :=
  fun s => (h s).mono sub

theorem keeps_get_bind {α : Type} {W : List Key} {f : Replayer → ReplayM α}
    (h : ∀ s, Grows W s (((f s).run).run s).2) : Keeps W (get >>= f) :=
  fun s => h s

theorem keeps_modify {W : List Key} {g : Replayer → Replayer} (h : ∀ s, Grows W s (g s)) :
    Keeps W (modify g : ReplayM Unit) := h

theorem keeps_mapM {α β : Type} {W : List Key} {f : α → ReplayM β} (each : ∀ a, Keeps W (f a)) :
    ∀ l : List α, Keeps W (l.mapM f)
  | [] => keeps_pure _
  | a :: l => by
      rw [List.mapM_cons]
      exact keeps_bind (each a) fun _ => keeps_bind (keeps_mapM each l) fun _ => keeps_pure _

theorem keeps_foldlM {α β : Type} {W : List Key} {f : β → α → ReplayM β}
    (each : ∀ b a, Keeps W (f b a)) : ∀ (l : List α) (b : β), Keeps W (l.foldlM f b)
  | [], _ => keeps_pure _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact keeps_bind (each b a) fun c => keeps_foldlM each l c

section Primitives

variable {W : List Key}

theorem keeps_recorded (key : Key) : Keeps W (recorded key) :=
  keeps_bind keeps_get fun _ => keeps_pure _

theorem keeps_first (key : Key) : Keeps W (first key) :=
  keeps_bind keeps_get fun _ => keeps_pure _

theorem keeps_fieldsOf (key : Key) : Keeps W (fieldsOf key) :=
  keeps_bind keeps_get fun _ => keeps_pure _

theorem keeps_pointBefore (before : Before) : Keeps W (pointBefore before) :=
  keeps_bind keeps_get fun _ => keeps_pure _

theorem keeps_absorb (bytes : ByteArray) : Keeps W (absorb bytes) :=
  keeps_modify fun _ => Grows.of_eq rfl rfl rfl rfl rfl rfl

theorem keeps_squeeze (count : ℕ) : Keeps W (squeeze count) := by
  refine keeps_get_bind fun s => ?_
  exact Grows.of_eq rfl rfl rfl rfl rfl rfl

theorem keeps_message (i : ℕ) (messages : Array Message) : Keeps W (message i messages) := by
  refine keeps_get_bind fun s => ?_
  cases messages[s.next]? with
  | none => exact Grows.refl W s
  | some m => exact Grows.of_eq rfl rfl rfl rfl rfl rfl

theorem keeps_record {key : Key} (inside : key ∈ W) (values : Array Ext) :
    Keeps W (record key values) := by
  intro s
  refine ⟨fun k => ?_, fun k out => ?_, List.prefix_refl _, List.prefix_refl _⟩
  · show ((s.env.get? k).getD #[]).toList <+:
      (((s.env.insert key (((s.env.get? key).getD #[]) ++ values)).get? k).getD #[]).toList
    rw [Std.HashMap.get?_insert]
    split
    · rename_i same
      rw [eq_of_beq same]
      exact ⟨values.toList, by simp⟩
    · exact List.prefix_refl _
  · have differ : (key == k) = false := by
      simpa using fun same : key = k => out (same ▸ inside)
    refine ⟨?_, rfl, rfl, rfl⟩
    show ((s.env.insert key _).get? k).getD #[] = (s.env.get? k).getD #[]
    rw [Std.HashMap.get?_insert, differ]
    rfl

macro "replay_keeps" : tactic => `(tactic| repeat' (first
  | with_reducible exact keeps_pure _ | with_reducible exact keeps_throw _
  | with_reducible exact keeps_get
  | with_reducible exact keeps_squeeze _ | with_reducible exact keeps_message _ _
  | with_reducible exact keeps_absorb _ | with_reducible exact keeps_recorded _
  | with_reducible exact keeps_first _ | with_reducible exact keeps_fieldsOf _
  | with_reducible exact keeps_pointBefore _
  | (with_reducible refine keeps_modify fun _ => ?_; exact Grows.of_eq rfl rfl rfl rfl rfl rfl)
  | refine keeps_bind ?_ fun _ => ?_
  | split))

theorem keeps_stage2Of (pub : Public) (level : ℕ) (F : Fold) : Keeps W (stage2Of pub level F) := by
  unfold stage2Of
  replay_keeps

theorem keeps_value (pub : Public) (key : Key) (v : Value) : Keeps W (value pub key v) := by
  cases v with
  | recorded keys =>
      exact keeps_bind (keeps_foldlM (fun _ _ => keeps_bind (keeps_recorded _) fun _ =>
        keeps_pure _) keys _) fun _ => keeps_pure _
  | fused F =>
      simp only [value]
      refine keeps_bind (keeps_stage2Of pub _ F) fun _ => ?_
      replay_keeps
  | _ => simp only [value]; replay_keeps

theorem grows_fields {key : Key} (inside : key ∈ W) (values : Array Fq) (s : Replayer) :
    Grows W s { s with fields := s.fields.insert key values } := by
  refine ⟨fun _ => List.prefix_refl _, fun k out => ⟨rfl, ?_, rfl, rfl⟩, List.prefix_refl _,
    List.prefix_refl _⟩
  have differ : (key == k) = false := by simpa using fun same : key = k => out (same ▸ inside)
  show (s.fields.insert key values).get? k = s.fields.get? k
  rw [Std.HashMap.get?_insert, differ]
  rfl

theorem grows_integers {key : Key} (inside : key ∈ W) (value : ℕ) (s : Replayer) :
    Grows W s { s with integers := s.integers.insert key value } := by
  refine ⟨fun _ => List.prefix_refl _, fun k out => ⟨rfl, rfl, ?_, rfl⟩, List.prefix_refl _,
    List.prefix_refl _⟩
  have differ : (key == k) = false := by simpa using fun same : key = k => out (same ▸ inside)
  show (s.integers.insert key value).get? k = s.integers.get? k
  rw [Std.HashMap.get?_insert, differ]
  rfl

theorem grows_bytes {key : Key} (inside : key ∈ W) (value : ByteArray) (s : Replayer) :
    Grows W s { s with derived := { s.derived with bytes := s.derived.bytes.insert key value } } := by
  refine ⟨fun _ => List.prefix_refl _, fun k out => ⟨rfl, rfl, rfl, ?_⟩, List.prefix_refl _,
    List.prefix_refl _⟩
  have differ : (key == k) = false := by simpa using fun same : key = k => out (same ▸ inside)
  show (s.derived.bytes.insert key value).get? k = s.derived.bytes.get? k
  rw [Std.HashMap.get?_insert, differ]
  rfl

theorem grows_rounds (e : Key × Before × Ended) (r : Key × List (List Ext × Ext)) (s : Replayer) :
    Grows W s
      { s with derived :=
        { s.derived with ends := s.derived.ends.push e, rounds := s.derived.rounds.push r } } :=
  ⟨fun _ => List.prefix_refl _, fun _ _ => ⟨rfl, rfl, rfl, rfl⟩,
    ⟨[r], by simp⟩, List.prefix_refl _⟩

theorem grows_drawn (x : ℕ × ℕ × List Sparse) (s : Replayer) :
    Grows W s { s with derived := { s.derived with drawn := s.derived.drawn.push x } } :=
  ⟨fun _ => List.prefix_refl _, fun _ _ => ⟨rfl, rfl, rfl, rfl⟩, List.prefix_refl _,
    ⟨[x], by simp⟩⟩

theorem keeps_weightsOf (weights : Weights) : Keeps W (weightsOf weights) := by
  cases weights <;> simp only [weightsOf] <;> replay_keeps

theorem keeps_holds (pub : Public) (check : Check) : Keeps W (holds pub check) := by
  cases check <;> simp only [holds]
  · replay_keeps
  · refine keeps_bind (keeps_first _) fun _ => keeps_bind (keeps_weightsOf _) fun _ =>
      keeps_bind (keeps_recorded _) fun _ => keeps_pure _
  all_goals replay_keeps

theorem keeps_checkWith (pub : Public) (i : ℕ) (key : Key) (check : Check)
    (out : Option StageOut) : Keeps W (checkWith pub i key check out) := by
  cases out with
  | some out => simp only [checkWith]; replay_keeps
  | none =>
      simp only [checkWith]
      refine keeps_bind (keeps_holds pub check) fun _ => ?_
      replay_keeps

theorem keeps_draw (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) :
    Keeps W (draw i level group G bytes) := by
  unfold draw
  refine keeps_bind (keeps_modify fun _ => Grows.of_eq rfl rfl rfl rfl rfl rfl) fun _ => ?_
  split
  · exact keeps_bind (keeps_modify fun s => grows_drawn _ s) fun _ => keeps_pure _
  · exact keeps_throw _

theorem keeps_terminalWith (i level : ℕ) (r : Option (Except String TerminalValues)) :
    Keeps W (terminalWith i level r) := by
  rcases r with _ | _ | _ <;> simp only [terminalWith] <;> replay_keeps

theorem keeps_decodeExts (bytes : ByteArray) : Keeps W (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact keeps_pure _

macro "replay_keeps_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact keeps_pure _ | with_reducible exact keeps_throw _
  | with_reducible exact keeps_get
  | with_reducible exact keeps_squeeze _ | with_reducible exact keeps_message _ _
  | with_reducible exact keeps_absorb _ | with_reducible exact keeps_recorded _
  | with_reducible exact keeps_first _ | with_reducible exact keeps_pointBefore _
  | with_reducible exact keeps_record (by simp [Op.writes]) _
  | with_reducible exact keeps_value _ _ _ | with_reducible exact keeps_draw _ _ _ _ _
  | with_reducible exact keeps_terminalWith _ _ _ | with_reducible exact keeps_checkWith _ _ _ _ _
  | with_reducible exact keeps_mapM (fun _ => keeps_recorded _) _
  | with_reducible exact keeps_decodeExts _
  | (with_reducible refine keeps_modify fun _ => ?_; first
      | exact Grows.of_eq rfl rfl rfl rfl rfl rfl
      | exact grows_fields (by simp [Op.writes]) _ _
      | exact grows_integers (by simp [Op.writes]) _ _
      | exact grows_bytes (by simp [Op.writes]) _ _
      | exact grows_rounds _ _ _)
  | refine keeps_bind ?_ fun _ => ?_
  | split))

/-- Each step of the replay only appends to the recorded values, and it changes no recorded
value, field message, integer or bytes of a key that the operation does not write; it only
appends to the rounds and the drawn challenges. -/
theorem step_keeps (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (op : Op) : Keeps op.writes (step pub proof messages i op) := by
  cases op <;> simp only [step] <;> replay_keeps_step

end Primitives

theorem forM_append {α : Type} (f : α → ReplayM PUnit) :
    ∀ a b : List α, (a ++ b).forM f = a.forM f >>= fun _ => b.forM f
  | [], b => by simp [List.forM]
  | x :: a, b => by
      simp only [List.cons_append, List.forM]
      rw [forM_append f a b, bind_assoc]

/-- A run of the operations `l` grows each state with the writes of its operations. -/
theorem keeps_forM (pub : Public) (proof : ByteArray) (messages : Array Message) :
    ∀ l : List (Op × ℕ), Keeps (l.flatMap fun x => x.1.writes)
      (l.forM fun x => step pub proof messages x.2 x.1)
  | [] => keeps_pure _
  | x :: l => by
      simp only [List.forM]
      exact keeps_bind (keeps_mono (fun k inside => List.mem_flatMap.mpr ⟨x, List.mem_cons_self, inside⟩)
          (step_keeps pub proof messages x.2 x.1)) fun _ =>
        keeps_mono (fun k inside => by
          obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
          exact List.mem_flatMap.mpr ⟨y, List.mem_cons_of_mem _ member, found⟩)
          (keeps_forM pub proof messages l)

/-- The state after the operations `l` from `s`, in order (`replay` runs the operations of the
script so). -/
def runOps (pub : Public) (proof : ByteArray) (messages : Array Message) (l : List (Op × ℕ))
    (s : Replayer) : Replayer :=
  (((l.forM fun x => step pub proof messages x.2 x.1).run).run s).2

/-- From the state after `done` to the state after `done ++ later`, the replay grows with the
writes of `later`, on success and on a stop. -/
theorem runOps_grows (pub : Public) (proof : ByteArray) (messages : Array Message)
    (done later : List (Op × ℕ)) (s : Replayer) :
    Grows (later.flatMap fun x => x.1.writes) (runOps pub proof messages done s)
      (runOps pub proof messages (done ++ later) s) := by
  unfold runOps
  rw [forM_append, run_bind]
  rcases ((done.forM fun x => step pub proof messages x.2 x.1).run).run s with ⟨_ | _, s₁⟩
  · exact Grows.refl _ _
  · exact keeps_forM pub proof messages later s₁

/-- A value that the replay reads after the operations `done` and `later` is the value after
`done` when no operation of `later` writes its key: the recorded values, the field message, the
integer and the bytes. -/
theorem lookup_later (pub : Public) (proof : ByteArray) (messages : Array Message)
    (done later : List (Op × ℕ)) (s : Replayer) (key : Key)
    (free : ∀ x ∈ later, key ∉ x.1.writes) :
    recordedIn (runOps pub proof messages (done ++ later) s) key =
        recordedIn (runOps pub proof messages done s) key ∧
      (runOps pub proof messages (done ++ later) s).fields.get? key =
        (runOps pub proof messages done s).fields.get? key ∧
      (runOps pub proof messages (done ++ later) s).integers.get? key =
        (runOps pub proof messages done s).integers.get? key ∧
      (runOps pub proof messages (done ++ later) s).derived.bytes.get? key =
        (runOps pub proof messages done s).derived.bytes.get? key :=
  (runOps_grows pub proof messages done later s).same key fun inside => by
    obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
    exact free y member found

theorem forM_array {α : Type} (as : Array α) (f : α → ReplayM PUnit) :
    as.forM f = as.toList.forM f := by
  rw [← List.forM_toArray, Array.toArray_toList]
  rfl

/-- `replay` runs the operations of the script in order from the channel of the session: its
recorded values, field messages and integers are those of `runOps` on the indexed operations. -/
theorem replay_runOps (pub : Public) (ops : Array Op) (proof : ByteArray)
    (messages : Array Message) :
    let s := runOps pub proof messages ops.zipIdx.toList
      ⟨channel pub.session pub.descriptor, 0, {}, {}, {}, {}⟩
    (replay pub ops proof messages).2.recorded = s.env ∧
      (replay pub ops proof messages).2.fields = s.fields ∧
      (replay pub ops proof messages).2.integers = s.integers := by
  unfold replay runOps
  rw [forM_array]
  exact ⟨rfl, rfl, rfl⟩

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
