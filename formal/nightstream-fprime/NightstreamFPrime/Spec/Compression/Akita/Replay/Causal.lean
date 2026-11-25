import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStage

/-! The state of the replay before an operation is a function of the events of the sponge before
it (Akita revision `d50f700`; `Replay/Keccak.lean`, `Replay/Transcript.lean`;
`docs/akita-work/w17/design.md` section 2, track N, item 4).

- `Core` and `core s`: the recorded values, field messages, integers and bytes of a state, as
  functions of the key. `noteOf` reads a state only through its core (`noteOf_core`).
- `Alike s s'`: the two states have one sponge and one core. `Sim m m'`: the computations give
  the same result and alike states from alike states (`sim_bind` and the lemmas of the
  primitives). `step_sim`: the step of an operation that reads no message and is no check.
- `Path R` and `Moves R m`: the computation moves the sponge along the preorder `R`. `Suffix`: the
  events of the first sponge are a suffix of those of the second (`step_suffix`,
  `runFrom_suffix`). With `R = Eq`: the step of a check and of the terminal keeps the sponge
  (`step_fixed`), and it keeps the core (`step_inert`).
- `Op.input proof m` are the bytes that the step of a reading operation absorbs for the message
  `m`, `Op.tail` the squeeze after it, and `store` the change of the core. `step_read`: a step
  without a stop of a reading operation absorbs `Op.input` and changes the core by `store`.
- `Faithful … l n`: for each operation of `l` that stores a message, two messages at its index
  with the same absorbed bytes have the same data.
- `run_alike`: two runs without a stop of the same operations from alike states, with faithful
  messages and the same events at the end, end in alike states (`state_causal`, `core_faithful`,
  `noteAt20_faithful`).
- `ReadsFrom`, `decode_from`: the messages of `decode` are those of `Wire.read`. `Op.Parses`: the
  data that `Wire.read` gives is a function of the absorbed bytes (`parses_fields`, `parses_exts`,
  `parses_integer`, `parses_digest`, `parses_bounded`). `decode_faithful`: the decoded messages of
  two proofs are faithful.
- `core_causal`: the core before the operation `c` is a function of the events of the sponge
  before `c`. `noteAt20_causal`: the `Note` of the wire message `k` is a function of the events
  before the work of the stage `k`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita

/-- The parts of a state of the replay that `noteOf` and the absorbed values read: the recorded
values, the field messages, the integers and the bytes of each key. -/
structure Core where
  recorded : Key → Array Ext
  fields : Key → Option (Array Fq)
  integers : Key → Option ℕ
  bytes : Key → Option ByteArray

namespace Replayer

/-- The core of a state. -/
def core (s : Replayer) : Core :=
  ⟨recordedIn s, fun k => s.fields.get? k, fun k => s.integers.get? k,
    fun k => s.derived.bytes.get? k⟩

/-- The two states have one sponge and one core. -/
def Alike (s s' : Replayer) : Prop := s.sponge = s'.sponge ∧ core s = core s'

theorem Alike.refl (s : Replayer) : Alike s s := ⟨rfl, rfl⟩

theorem Alike.symm {s s' : Replayer} (alike : Alike s s') : Alike s' s :=
  ⟨alike.1.symm, alike.2.symm⟩

theorem Alike.trans {a b c : Replayer} (first : Alike a b) (second : Alike b c) : Alike a c :=
  ⟨first.1.trans second.1, first.2.trans second.2⟩

theorem get?_insert_fun {V : Type} (map : Std.HashMap Key V) (key : Key) (v : V) :
    (fun k => (map.insert key v).get? k) = Function.update (fun k => map.get? k) key (some v) := by
  funext k
  rw [Std.HashMap.get?_insert, Function.update_apply]
  by_cases same : key = k
  · subst same
    simp
  · simp [same, show ¬ k = key from fun e => same e.symm]

/-- The recorded values after a record of `values` at `key`. -/
theorem recordedIn_insert {s t : Replayer} {key : Key} {values : Array Ext}
    (env : t.env = s.env.insert key values) :
    recordedIn t = Function.update (recordedIn s) key values := by
  funext k
  unfold recordedIn
  rw [env, Std.HashMap.get?_insert, Function.update_apply]
  by_cases same : key = k
  · subst same
    simp
  · simp [same, show ¬ k = key from fun e => same e.symm]

section Sim

/-- The computations give the same result and alike states from alike states. -/
def Sim {α : Type} (m m' : ReplayM α) : Prop :=
  ∀ s s', Alike s s' → ((m.run).run s).1 = ((m'.run).run s').1 ∧
    Alike ((m.run).run s).2 ((m'.run).run s').2

theorem sim_bind {α β : Type} {m m' : ReplayM α} {f f' : α → ReplayM β} (hm : Sim m m')
    (hf : ∀ a, Sim (f a) (f' a)) : Sim (m >>= f) (m' >>= f') := by
  intro s s' alike
  have first := hm s s' alike
  rw [run_bind, run_bind]
  revert first
  generalize (m.run).run s = r
  generalize (m'.run).run s' = r'
  rcases r with ⟨_ | a, s₁⟩ <;> rcases r' with ⟨_ | a', s₁'⟩ <;> rintro ⟨same, after⟩
  · cases same
    exact ⟨rfl, after⟩
  · cases same
  · cases same
  · cases same
    exact hf a s₁ s₁' after

theorem sim_pure {α : Type} (a : α) : Sim (Pure.pure a : ReplayM α) (Pure.pure a) :=
  fun _ _ alike => ⟨rfl, alike⟩

theorem sim_throw {α : Type} (e : Stop) : Sim (throw e : ReplayM α) (throw e) :=
  fun _ _ alike => ⟨rfl, alike⟩

theorem sim_modify {g g' : Replayer → Replayer} (h : ∀ s s', Alike s s' → Alike (g s) (g' s')) :
    Sim (modify g : ReplayM Unit) (modify g') := fun s s' alike => ⟨rfl, h s s' alike⟩

theorem sim_mapM {α β : Type} {f : α → ReplayM β} (each : ∀ a, Sim (f a) (f a)) :
    ∀ l : List α, Sim (l.mapM f) (l.mapM f)
  | [] => sim_pure _
  | a :: l => by
      rw [List.mapM_cons]
      exact sim_bind (each a) fun _ => sim_bind (sim_mapM each l) fun _ => sim_pure _

theorem sim_foldlM {α β : Type} {f : β → α → ReplayM β} (each : ∀ b a, Sim (f b a) (f b a)) :
    ∀ (l : List α) (b : β), Sim (l.foldlM f b) (l.foldlM f b)
  | [], _ => sim_pure _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact sim_bind (each b a) fun c => sim_foldlM each l c

theorem sim_recorded (key : Key) : Sim (recorded key) (recorded key) := fun _ _ alike =>
  ⟨congrArg (fun r : Key → Array Ext => Except.ok (r key)) (congrArg Core.recorded alike.2), alike⟩

theorem sim_first (key : Key) : Sim (first key) (first key) := fun _ _ alike =>
  ⟨congrArg (fun r : Key → Array Ext => Except.ok (((r key)[0]?).getD Ext.zero))
    (congrArg Core.recorded alike.2), alike⟩

theorem sim_fieldsOf (key : Key) : Sim (fieldsOf key) (fieldsOf key) := fun _ _ alike =>
  ⟨congrArg (fun r : Key → Option (Array Fq) => Except.ok ((r key).getD #[]))
    (congrArg Core.fields alike.2), alike⟩

theorem pointIn_alike {s s' : Replayer} (same : recordedIn s = recordedIn s') (before : Before) :
    pointIn s before = pointIn s' before := by
  cases before <;> simp only [pointIn, same]

theorem sim_pointBefore (before : Before) : Sim (pointBefore before) (pointBefore before) :=
  fun _ _ alike => ⟨congrArg Except.ok (pointIn_alike (congrArg Core.recorded alike.2) before), alike⟩

theorem sim_absorb (bytes : ByteArray) : Sim (absorb bytes) (absorb bytes) := fun _ _ alike =>
  ⟨rfl, congrArg (fun d : Duplex => d.absorb bytes) alike.1, alike.2⟩

theorem sim_squeeze (count : ℕ) : Sim (squeeze count) (squeeze count) := fun _ _ alike =>
  ⟨congrArg (fun d : Duplex => Except.ok (d.squeeze count).1) alike.1,
    congrArg (fun d : Duplex => (d.squeeze count).2) alike.1, alike.2⟩

theorem sim_record (key : Key) (values : Array Ext) : Sim (record key values) (record key values) :=
  by
  intro s s' alike
  refine ⟨rfl, alike.1, ?_⟩
  have recorded := congrArg Core.recorded alike.2
  have same : recordedIn s key = recordedIn s' key := congrFun recorded key
  have left : recordedIn { s with env := s.env.insert key (recordedIn s key ++ values) } =
      Function.update (recordedIn s) key (recordedIn s key ++ values) := recordedIn_insert rfl
  have right : recordedIn { s' with env := s'.env.insert key (recordedIn s' key ++ values) } =
      Function.update (recordedIn s') key (recordedIn s' key ++ values) := recordedIn_insert rfl
  show core { s with env := s.env.insert key (recordedIn s key ++ values) } =
    core { s' with env := s'.env.insert key (recordedIn s' key ++ values) }
  unfold core
  rw [left, right, same]
  exact congrArg (fun c : Core => (⟨Function.update c.recorded key (recordedIn s' key ++ values),
    c.fields, c.integers, c.bytes⟩ : Core)) alike.2

theorem sim_decodeExts (bytes : ByteArray) :
    Sim (step.decodeExts bytes) (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact sim_pure _

macro "replay_sim" : tactic => `(tactic| repeat' (first
  | with_reducible exact sim_pure _ | with_reducible exact sim_throw _
  | with_reducible exact sim_squeeze _ | with_reducible exact sim_recorded _
  | with_reducible exact sim_first _ | with_reducible exact sim_fieldsOf _
  | with_reducible exact sim_pointBefore _ | with_reducible exact sim_absorb _
  | with_reducible exact sim_record _ _
  | with_reducible exact sim_mapM (fun _ => sim_recorded _) _
  | with_reducible exact sim_decodeExts _
  | ((with_reducible refine sim_modify fun _ _ alike => ?_) <;> exact alike)
  | refine sim_bind ?_ fun _ => ?_
  | split))

theorem sim_stage2Of (pub : Public) (level : ℕ) (F : Fold) :
    Sim (stage2Of pub level F) (stage2Of pub level F) := by
  unfold stage2Of
  replay_sim

theorem sim_value (pub : Public) (key : Key) (v : Value) :
    Sim (value pub key v) (value pub key v) := by
  cases v with
  | recorded keys =>
      exact sim_bind (sim_foldlM (fun _ _ => sim_bind (sim_recorded _) fun _ =>
        sim_pure _) keys _) fun _ => sim_pure _
  | fused F =>
      simp only [value]
      refine sim_bind (sim_stage2Of pub _ F) fun _ => ?_
      replay_sim
  | _ => simp only [value]; replay_sim

theorem sim_draw (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) :
    Sim (draw i level group G bytes) (draw i level group G bytes) := by
  unfold draw
  replay_sim

/-- The step of an operation that absorbs a value, squeezes challenges or ends a sum-check does
not read the proof: two steps give the same result and alike states from alike states. -/
theorem step_sim (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)
    (i : ℕ) (op : Op) (quiet : op.reads = false) (free : ∀ key check, op ≠ .check key check)
    (open_ : ∀ level, op ≠ .terminal level) :
    Sim (step pub proof messages i op) (step pub proof' messages' i op) := by
  cases op with
  | absorb key v =>
      simp only [step]
      refine sim_bind (sim_value pub key v) fun _ => ?_
      replay_sim
  | challenge key count => simp only [step]; replay_sim
  | squeeze level group G =>
      simp only [step]
      exact sim_bind (sim_squeeze _) fun _ => sim_draw _ _ _ _ _
  | ended key rounds before => simp only [step]; replay_sim
  | work site level bits =>
      have zero : bits = 0 := by simpa [Op.reads] using quiet
      subst zero
      exact fun _ _ alike => ⟨rfl, alike⟩
  | check key check => exact absurd rfl (free key check)
  | terminal level => exact absurd rfl (open_ level)
  | _ => simp [Op.reads] at quiet

end Sim

section Moves

/-- A preorder on the sponges. -/
structure Path (R : Duplex → Duplex → Prop) : Prop where
  refl : ∀ d, R d d
  trans : ∀ {a b c}, R a b → R b c → R a c

/-- The computation moves the sponge along `R`. -/
def Moves (R : Duplex → Duplex → Prop) {α : Type} (m : ReplayM α) : Prop :=
  ∀ s, R s.sponge ((m.run).run s).2.sponge

variable {R : Duplex → Duplex → Prop} (P : Path R)
include P

theorem moves_bind {α β : Type} {m : ReplayM α} {f : α → ReplayM β} (hm : Moves R m)
    (hf : ∀ a, Moves R (f a)) : Moves R (m >>= f) := by
  intro s
  rw [run_bind]
  rcases found : (m.run).run s with ⟨_ | a, s₁⟩
  · have kept := hm s
    rw [found] at kept
    exact kept
  · have kept := hm s
    rw [found] at kept
    exact P.trans kept (hf a s₁)

theorem moves_pure {α : Type} (a : α) : Moves R (Pure.pure a : ReplayM α) := fun _ => P.refl _

theorem moves_throw {α : Type} (e : Stop) : Moves R (throw e : ReplayM α) := fun _ => P.refl _

theorem moves_get : Moves R (get : ReplayM Replayer) := fun _ => P.refl _

theorem moves_modify {g : Replayer → Replayer} (h : ∀ s, (g s).sponge = s.sponge) :
    Moves R (modify g : ReplayM Unit) := fun s => by
  show R s.sponge (g s).sponge
  rw [h]
  exact P.refl _

theorem moves_mapM {α β : Type} {f : α → ReplayM β} (each : ∀ a, Moves R (f a)) :
    ∀ l : List α, Moves R (l.mapM f)
  | [] => moves_pure P _
  | a :: l => by
      rw [List.mapM_cons]
      exact moves_bind P (each a) fun _ => moves_bind P (moves_mapM each l) fun _ => moves_pure P _

theorem moves_foldlM {α β : Type} {f : β → α → ReplayM β} (each : ∀ b a, Moves R (f b a)) :
    ∀ (l : List α) (b : β), Moves R (l.foldlM f b)
  | [], _ => moves_pure P _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact moves_bind P (each b a) fun c => moves_foldlM each l c

theorem moves_recorded (key : Key) : Moves R (recorded key) :=
  moves_bind P (moves_get P) fun _ => moves_pure P _

theorem moves_first (key : Key) : Moves R (first key) :=
  moves_bind P (moves_get P) fun _ => moves_pure P _

theorem moves_fieldsOf (key : Key) : Moves R (fieldsOf key) :=
  moves_bind P (moves_get P) fun _ => moves_pure P _

theorem moves_pointBefore (before : Before) : Moves R (pointBefore before) :=
  moves_bind P (moves_get P) fun _ => moves_pure P _

theorem moves_record (key : Key) (values : Array Ext) : Moves R (record key values) :=
  moves_modify P fun _ => rfl

theorem moves_message (i : ℕ) (messages : Array Message) : Moves R (message i messages) := by
  intro s
  rw [run_message]
  split <;> exact P.refl _

end Moves

/-- The events of the first sponge are a suffix of the events of the second. -/
def Suffix (a b : Duplex) : Prop := ∃ new, b.events = new ++ a.events

theorem suffix_path : Path Suffix :=
  ⟨fun _ => ⟨[], rfl⟩, fun ⟨x, hx⟩ ⟨y, hy⟩ => ⟨y ++ x, by rw [hy, hx, List.append_assoc]⟩⟩

theorem eq_path : Path (fun a b : Duplex => a = b) := ⟨fun _ => rfl, fun first second => first.trans second⟩

theorem moves_absorb (bytes : ByteArray) : Moves Suffix (absorb bytes) := fun s =>
  ⟨[.absorb bytes], Duplex.absorb_events s.sponge bytes⟩

theorem moves_squeeze (count : ℕ) : Moves Suffix (squeeze count) := fun s =>
  ⟨[.squeeze count], Duplex.squeeze_events s.sponge count⟩

macro "replay_moves" P:term : tactic => `(tactic| repeat' (first
  | with_reducible exact moves_pure $P _ | with_reducible exact moves_throw $P _
  | with_reducible exact moves_get $P
  | with_reducible exact moves_squeeze _ | with_reducible exact moves_absorb _
  | with_reducible exact moves_message $P _ _
  | with_reducible exact moves_recorded $P _ | with_reducible exact moves_first $P _
  | with_reducible exact moves_fieldsOf $P _ | with_reducible exact moves_pointBefore $P _
  | with_reducible exact moves_record $P _ _
  | with_reducible exact moves_mapM $P (fun _ => moves_recorded $P _) _
  | (with_reducible refine moves_modify $P fun _ => ?_; rfl)
  | refine moves_bind $P ?_ fun _ => ?_
  | split))

section Steps

variable {R : Duplex → Duplex → Prop} (P : Path R)
include P

theorem moves_stage2Of (pub : Public) (level : ℕ) (F : Fold) : Moves R (stage2Of pub level F) := by
  unfold stage2Of
  replay_moves P

theorem moves_value (pub : Public) (key : Key) (v : Value) : Moves R (value pub key v) := by
  cases v with
  | recorded keys =>
      exact moves_bind P (moves_foldlM P (fun _ _ => moves_bind P (moves_recorded P _) fun _ =>
        moves_pure P _) keys _) fun _ => moves_pure P _
  | fused F =>
      simp only [value]
      refine moves_bind P (moves_stage2Of P pub _ F) fun _ => ?_
      replay_moves P
  | _ => simp only [value]; replay_moves P

theorem moves_weightsOf (weights : Weights) : Moves R (weightsOf weights) := by
  cases weights <;> simp only [weightsOf] <;> replay_moves P

theorem moves_holds (pub : Public) (check : Check) : Moves R (holds pub check) := by
  cases check <;> simp only [holds]
  · replay_moves P
  · refine moves_bind P (moves_first P _) fun _ => moves_bind P (moves_weightsOf P _) fun _ =>
      moves_bind P (moves_recorded P _) fun _ => moves_pure P _
  all_goals replay_moves P

theorem moves_checkWith (pub : Public) (i : ℕ) (key : Key) (check : Check)
    (out : Option StageOut) : Moves R (checkWith pub i key check out) := by
  cases out with
  | some out => simp only [checkWith]; replay_moves P
  | none =>
      simp only [checkWith]
      refine moves_bind P (moves_holds P pub check) fun _ => ?_
      replay_moves P

theorem moves_draw (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) :
    Moves R (draw i level group G bytes) := by
  unfold draw
  replay_moves P

theorem moves_terminalWith (i level : ℕ) (r : Option (Except String TerminalValues)) :
    Moves R (terminalWith i level r) := by
  rcases r with _ | _ | _ <;> simp only [terminalWith] <;> replay_moves P

theorem moves_decodeExts (bytes : ByteArray) : Moves R (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact moves_pure P _

end Steps

macro "replay_moves_step" P:term : tactic => `(tactic| repeat' (first
  | with_reducible exact moves_pure $P _ | with_reducible exact moves_throw $P _
  | with_reducible exact moves_get $P
  | with_reducible exact moves_squeeze _ | with_reducible exact moves_absorb _
  | with_reducible exact moves_message $P _ _
  | with_reducible exact moves_recorded $P _ | with_reducible exact moves_first $P _
  | with_reducible exact moves_pointBefore $P _ | with_reducible exact moves_record $P _ _
  | with_reducible exact moves_value $P _ _ _ | with_reducible exact moves_draw $P _ _ _ _ _
  | with_reducible exact moves_terminalWith $P _ _ _
  | with_reducible exact moves_checkWith $P _ _ _ _ _
  | with_reducible exact moves_mapM $P (fun _ => moves_recorded $P _) _
  | with_reducible exact moves_decodeExts $P _
  | (with_reducible refine moves_modify $P fun _ => ?_; rfl)
  | refine moves_bind $P ?_ fun _ => ?_
  | split))

section Runs

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- Each step of the replay only adds events to the sponge. -/
theorem step_suffix (i : ℕ) (op : Op) : Moves Suffix (step pub proof messages i op) := by
  cases op <;> simp only [step] <;> replay_moves_step suffix_path

/-- The step of a check keeps the sponge. -/
theorem step_fixed_check (i : ℕ) (key : Key) (check : Check) :
    Moves (fun a b : Duplex => a = b) (step pub proof messages i (.check key check)) := by
  rw [step_check]
  exact moves_bind eq_path (moves_get eq_path) fun _ => moves_checkWith eq_path _ _ _ _ _

/-- The step of the terminal keeps the sponge. -/
theorem step_fixed_terminal (i level : ℕ) :
    Moves (fun a b : Duplex => a = b) (step pub proof messages i (.terminal level)) := by
  simp only [step]
  exact moves_bind eq_path (moves_get eq_path) fun _ => moves_terminalWith eq_path _ _ _

/-- A step that writes no key keeps the core. -/
theorem step_core (i : ℕ) (op : Op) (s : Replayer) (free : op.writes = []) :
    core (((step pub proof messages i op).run).run s).2 = core s := by
  have kept := step_keeps pub proof messages i op s
  rw [free] at kept
  unfold core
  congr 1
  · funext k
    exact (kept.same k List.not_mem_nil).1
  · funext k
    exact (kept.same k List.not_mem_nil).2.1
  · funext k
    exact (kept.same k List.not_mem_nil).2.2.1
  · funext k
    exact (kept.same k List.not_mem_nil).2.2.2

/-- The step of a check or of the terminal keeps the sponge and the core. -/
theorem step_inert (i : ℕ) (op : Op) (s : Replayer)
    (still : (∃ key check, op = .check key check) ∨ ∃ level, op = .terminal level) :
    Alike s (((step pub proof messages i op).run).run s).2 := by
  rcases still with ⟨key, check, rfl⟩ | ⟨level, rfl⟩
  · exact ⟨step_fixed_check pub proof messages i key check s,
      (step_core pub proof messages i _ s rfl).symm⟩
  · exact ⟨step_fixed_terminal pub proof messages i level s,
      (step_core pub proof messages i _ s rfl).symm⟩

/-- A run of the replay only adds events to the sponge. -/
theorem runFrom_suffix : ∀ (l : List (Op × ℕ)) (s : Replayer),
    Suffix s.sponge (runFrom pub proof messages l s).2.sponge
  | [], _ => suffix_path.refl _
  | x :: l, s => by
      have stepped := step_suffix pub proof messages x.2 x.1 s
      rw [runFrom_cons]
      rcases found : ((step pub proof messages x.2 x.1).run).run s with ⟨_ | _, s₁⟩
      · rw [found] at stepped
        exact stepped
      · rw [found] at stepped
        exact suffix_path.trans stepped (runFrom_suffix l s₁)

end Runs

end Replayer

/-- The operation stores the data of its message in the core of the state. -/
def Op.stores : Op → Bool
  | .fields .. | .exts .. | .integer .. | .digest .. | .bounded .. => true
  | _ => false

/-- The bytes that the step of a reading operation absorbs for the message `m`: the nonce as one
word, the length and the bytes of the payload of `z`, and otherwise the bytes of the message. -/
def Op.input (proof : ByteArray) (m : Message) : Op → ByteArray
  | .bounded _ _ =>
      le64 (m.size - 4) ++ wordPadded (proof.extract (m.offset + 4) (m.offset + m.size))
  | .work _ _ _ | .foldNonce _ _ => match m.data with
      | .nonce nonce _ => le64 nonce
      | _ => .empty
  | _ => proof.extract m.offset (m.offset + m.size)

/-- The sponge after the squeeze that follows the absorb of a reading operation: the 8 bytes of
a work, and no squeeze otherwise. -/
def Op.tail : Op → Duplex → Duplex
  | .work _ _ _, d => (d.squeeze 8).2
  | _, d => d

/-- The core after the step of a reading operation with the data `data` of its message. -/
def store (op : Op) (data : Data) (c : Core) : Core :=
  match op with
  | .fields key _ => match data with
      | .fields values => { c with fields := Function.update c.fields key (some values) }
      | _ => c
  | .exts key _ => match data with
      | .exts values =>
          { c with recorded := Function.update c.recorded key (c.recorded key ++ values) }
      | _ => c
  | .integer key => match data with
      | .integer value =>
          { c with
            recorded := Function.update c.recorded key
              (c.recorded key ++ #[Ext.ofBase (Fq.ofNat value)])
            integers := Function.update c.integers key (some value) }
      | _ => c
  | .digest key => match data with
      | .bytes value => { c with bytes := Function.update c.bytes key (some value) }
      | _ => c
  | .bounded key _ => match data with
      | .bytes value => { c with bytes := Function.update c.bytes key (some value) }
      | _ => c
  | _ => c

theorem store_quiet {op : Op} (quiet : op.stores = false) (data : Data) (c : Core) :
    store op data c = c := by
  cases op <;> first | rfl | simp [Op.stores] at quiet

/-- The events of the sponge after the squeeze of a reading operation: fixed events, then the
events before it. -/
theorem Op.tail_events (op : Op) : ∃ pre, ∀ d : Duplex, (op.tail d).events = pre ++ d.events := by
  cases op with
  | work site level bits => exact ⟨[.squeeze 8], fun d => Duplex.squeeze_events d 8⟩
  | _ => exact ⟨[], fun _ => rfl⟩

/-- For each operation of `l` that stores a message, two messages at its index, one of each
proof, with the same absorbed bytes have the same data. `n` is the index of the first message. -/
def Faithful (proof proof' : ByteArray) (messages messages' : Array Message) :
    List (Op × ℕ) → ℕ → Prop
  | [], _ => True
  | x :: l, n => (x.1.stores = true → ∀ m m', messages[n]? = some m → messages'[n]? = some m' →
      x.1.input proof m = x.1.input proof' m' → m.data = m'.data) ∧
    Faithful proof proof' messages messages' l (n + if x.1.reads then 1 else 0)

/-- The messages of one proof are faithful with themselves. -/
theorem Faithful.refl (proof : ByteArray) (messages : Array Message) :
    ∀ (l : List (Op × ℕ)) (n : ℕ), Faithful proof proof messages messages l n
  | [], _ => trivial
  | _ :: l, _ => ⟨fun _ m m' found found' _ => by
      rw [found] at found'
      cases found'
      rfl, Faithful.refl proof messages l _⟩

namespace Replayer

section Cores

variable {s t : Replayer} {key : Key}

theorem core_fields {values : Array Fq} (env : recordedIn t = recordedIn s)
    (fields : t.fields = s.fields.insert key values) (integers : t.integers = s.integers)
    (bytes : t.derived.bytes = s.derived.bytes) :
    core t = { core s with fields := Function.update (core s).fields key (some values) } := by
  unfold core
  rw [env, fields, integers, bytes, get?_insert_fun]

theorem core_env {values : Array Ext} (env : t.env = s.env.insert key (recordedIn s key ++ values))
    (fields : t.fields = s.fields) (integers : t.integers = s.integers)
    (bytes : t.derived.bytes = s.derived.bytes) :
    core t = { core s with
      recorded := Function.update (core s).recorded key ((core s).recorded key ++ values) } := by
  unfold core
  rw [recordedIn_insert env, fields, integers, bytes]

theorem core_integer {values : Array Ext} {value : ℕ}
    (env : t.env = s.env.insert key (recordedIn s key ++ values)) (fields : t.fields = s.fields)
    (integers : t.integers = s.integers.insert key value)
    (bytes : t.derived.bytes = s.derived.bytes) :
    core t = { core s with
      recorded := Function.update (core s).recorded key ((core s).recorded key ++ values)
      integers := Function.update (core s).integers key (some value) } := by
  unfold core
  rw [recordedIn_insert env, fields, integers, bytes, get?_insert_fun]

theorem core_bytes {value : ByteArray} (env : recordedIn t = recordedIn s)
    (fields : t.fields = s.fields) (integers : t.integers = s.integers)
    (bytes : t.derived.bytes = s.derived.bytes.insert key value) :
    core t = { core s with bytes := Function.update (core s).bytes key (some value) } := by
  unfold core
  rw [env, fields, integers, bytes, get?_insert_fun]

end Cores

section Read

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A step without a stop of an operation that stores a message: the message at `next` exists,
the sponge absorbs `Op.input`, and the core changes by `store`. -/
theorem step_stores (i : ℕ) (op : Op) (s : Replayer) (stores : op.stores = true)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ()) :
    ∃ m, messages[s.next]? = some m ∧
      (((step pub proof messages i op).run).run s).2.sponge =
        op.tail (s.sponge.absorb (op.input proof m)) ∧
      core (((step pub proof messages i op).run).run s).2 = store op m.data (core s) := by
  cases found : messages[s.next]? with
  | none =>
      cases op <;> first
        | (simp [Op.stores] at stores; done)
        | (simp only [step, run_bind, run_message, found] at ran; cases ran)
  | some m =>
      refine ⟨m, rfl, ?_⟩
      cases op with
      | fields key count =>
          cases data : m.data with
          | fields values =>
              simp only [step, run_bind, run_message, found, run_absorb, data, run_modify] at ran ⊢
              exact ⟨rfl, core_fields rfl rfl rfl rfl⟩
          | _ =>
              simp only [step, run_bind, run_message, found, run_absorb, data] at ran
              cases ran
      | exts key count =>
          cases data : m.data with
          | exts values =>
              simp only [step, run_bind, run_message, found, run_absorb, data, run_record] at ran ⊢
              exact ⟨rfl, core_env rfl rfl rfl rfl⟩
          | _ =>
              simp only [step, run_bind, run_message, found, run_absorb, data] at ran
              cases ran
      | integer key =>
          cases data : m.data with
          | integer value =>
              simp only [step, run_bind, run_message, found, run_absorb, data, run_record,
                run_modify] at ran ⊢
              exact ⟨rfl, core_integer rfl rfl rfl rfl⟩
          | _ =>
              simp only [step, run_bind, run_message, found, run_absorb, data] at ran
              cases ran
      | digest key =>
          cases data : m.data with
          | bytes value =>
              simp only [step, run_bind, run_message, found, run_absorb, data, run_modify] at ran ⊢
              exact ⟨rfl, core_bytes rfl rfl rfl rfl⟩
          | _ =>
              simp only [step, run_bind, run_message, found, run_absorb, data] at ran
              cases ran
      | bounded key most =>
          cases data : m.data with
          | bytes value =>
              simp only [step, run_bind, run_message, found, run_absorb, data, run_modify] at ran ⊢
              exact ⟨rfl, core_bytes rfl rfl rfl rfl⟩
          | _ =>
              simp only [step, run_bind, run_message, found, run_absorb, data] at ran
              cases ran
      | _ => simp [Op.stores] at stores

/-- A step without a stop of a response nonce: the message at `next` exists, the sponge absorbs
the nonce, and the core does not change. -/
theorem step_foldNonce (i level width : ℕ) (s : Replayer)
    (ran : (((step pub proof messages i (.foldNonce level width)).run).run s).1 = .ok ()) :
    ∃ m, messages[s.next]? = some m ∧
      (((step pub proof messages i (.foldNonce level width)).run).run s).2.sponge =
        s.sponge.absorb ((Op.foldNonce level width).input proof m) := by
  cases found : messages[s.next]? with
  | none =>
      simp only [step, run_bind, run_message, found] at ran
      cases ran
  | some m =>
      refine ⟨m, rfl, ?_⟩
      cases data : m.data with
      | nonce nonce canonical =>
          have input : (Op.foldNonce level width).input proof m = le64 nonce := by
            simp only [Op.input, data]
          rw [input]
          simp only [step, run_bind, run_message, found, data] at ran ⊢
          split
          · rfl
          · rename_i fails
            split at ran
            · rename_i fits
              exact absurd fits fails
            · cases ran
      | _ =>
          simp only [step, run_bind, run_message, found, data] at ran
          cases ran

/-- A step without a stop of a work with `bits > 0`: the message at `next` exists, and the sponge
absorbs the nonce and squeezes 8 bytes. -/
theorem step_work (i : ℕ) (site : Site) (level bits : ℕ) (s : Replayer) (positive : bits ≠ 0)
    (ran : (((step pub proof messages i (.work site level bits)).run).run s).1 = .ok ()) :
    ∃ m, messages[s.next]? = some m ∧
      (((step pub proof messages i (.work site level bits)).run).run s).2.sponge =
        ((s.sponge.absorb ((Op.work site level bits).input proof m)).squeeze 8).2 := by
  cases found : messages[s.next]? with
  | none =>
      simp only [step, positive, ↓reduceIte, run_bind, run_message, found] at ran
      cases ran
  | some m =>
      refine ⟨m, rfl, ?_⟩
      cases data : m.data with
      | nonce nonce canonical =>
          have input : (Op.work site level bits).input proof m = le64 nonce := by
            simp only [Op.input, data]
          rw [input]
          simp only [step, positive, ↓reduceIte, run_bind, run_message, found, data] at ran ⊢
          split
          · simp only [run_bind, run_absorb, run_squeeze, run_modify] at ran ⊢
            split <;> rfl
          · rename_i fails
            split at ran
            · rename_i fits
              exact absurd fits fails
            · cases ran
      | _ =>
          simp only [step, positive, ↓reduceIte, run_bind, run_message, found, data] at ran
          cases ran

/-- A step without a stop of a reading operation: the message at `next` exists, the sponge absorbs
`Op.input` and squeezes `Op.tail`, and the core changes by `store`. -/
theorem step_read (i : ℕ) (op : Op) (s : Replayer) (reads : op.reads = true)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ()) :
    ∃ m, messages[s.next]? = some m ∧
      (((step pub proof messages i op).run).run s).2.sponge =
        op.tail (s.sponge.absorb (op.input proof m)) ∧
      core (((step pub proof messages i op).run).run s).2 = store op m.data (core s) := by
  by_cases stores : op.stores = true
  · exact step_stores pub proof messages i op s stores ran
  · cases op with
    | work site level bits =>
        have positive : bits ≠ 0 := by simpa [Op.reads] using reads
        obtain ⟨m, found, sponge⟩ := step_work pub proof messages i site level bits s positive ran
        exact ⟨m, found, sponge, step_core pub proof messages i _ s rfl⟩
    | foldNonce level width =>
        obtain ⟨m, found, sponge⟩ := step_foldNonce pub proof messages i level width s ran
        exact ⟨m, found, sponge, step_core pub proof messages i _ s rfl⟩
    | fields key count => exact absurd rfl stores
    | exts key count => exact absurd rfl stores
    | integer key => exact absurd rfl stores
    | digest key => exact absurd rfl stores
    | bounded key most => exact absurd rfl stores
    | _ => simp [Op.reads] at reads

end Read

section Causal

variable (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)

/-- Two steps without a stop of one reading operation from alike states, with faithful messages
and events after them that agree up to the events of the later operations, end in alike states. -/
theorem read_alike (i : ℕ) (op : Op) (reads : op.reads = true) {s s' : Replayer}
    (alike : Alike s s') (next : s.next = s'.next)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ())
    (ran' : (((step pub proof' messages' i op).run).run s').1 = .ok ())
    (faithful : op.stores = true → ∀ m m', messages[s.next]? = some m →
      messages'[s.next]? = some m' → op.input proof m = op.input proof' m' → m.data = m'.data)
    (later : ∃ T T', T ++ (((step pub proof messages i op).run).run s).2.sponge.events =
      T' ++ (((step pub proof' messages' i op).run).run s').2.sponge.events) :
    Alike (((step pub proof messages i op).run).run s).2
      (((step pub proof' messages' i op).run).run s').2 := by
  obtain ⟨m, found, sponge, cored⟩ := step_read pub proof messages i op s reads ran
  obtain ⟨m', found', sponge', cored'⟩ := step_read pub proof' messages' i op s' reads ran'
  rw [← next] at found'
  obtain ⟨pre, tail⟩ := op.tail_events
  obtain ⟨T, T', agree⟩ := later
  rw [sponge, sponge', tail, tail, Duplex.absorb_events, Duplex.absorb_events, ← alike.1,
    ← List.append_assoc, ← List.append_assoc] at agree
  have input : op.input proof m = op.input proof' m' := by
    have cut := (List.append_inj' agree rfl).2
    injection cut with head
    injection head
  refine ⟨by rw [sponge, sponge', alike.1, input], ?_⟩
  rw [cored, cored', alike.2]
  by_cases stores : op.stores = true
  · rw [faithful stores m m' found found' input]
  · rw [store_quiet (Bool.eq_false_iff.mpr stores), store_quiet (Bool.eq_false_iff.mpr stores)]

/-- Two runs without a stop of the same operations from alike states, with faithful messages and
events at the end that agree up to a prefix, end in alike states. -/
theorem run_alike : ∀ (l : List (Op × ℕ)) (s s' : Replayer), Alike s s' → s.next = s'.next →
    (runFrom pub proof messages l s).1 = .ok () → (runFrom pub proof' messages' l s').1 = .ok () →
    Faithful proof proof' messages messages' l s.next →
    (∃ T T', T ++ (runFrom pub proof messages l s).2.sponge.events =
      T' ++ (runFrom pub proof' messages' l s').2.sponge.events) →
    Alike (runFrom pub proof messages l s).2 (runFrom pub proof' messages' l s').2
  | [], _, _, alike, _, _, _, _, _ => alike
  | x :: l, s, s', alike, next, ran, ran', faithful, later => by
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      obtain ⟨stepped', tail'⟩ := (runFrom_cons_fst pub proof' messages' _ _ _).mp ran'
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped,
        runFrom_cons_snd pub proof' messages' _ _ _ stepped'] at later ⊢
      obtain ⟨T, T', agree⟩ := later
      obtain ⟨U, grown⟩ := runFrom_suffix pub proof messages l
        (((step pub proof messages x.2 x.1).run).run s).2
      obtain ⟨U', grown'⟩ := runFrom_suffix pub proof' messages' l
        (((step pub proof' messages' x.2 x.1).run).run s').2
      have after : Alike (((step pub proof messages x.2 x.1).run).run s).2
          (((step pub proof' messages' x.2 x.1).run).run s').2 := by
        by_cases reads : x.1.reads = true
        · refine read_alike pub proof proof' messages messages' x.2 x.1 reads alike next stepped
            stepped' faithful.1 ⟨T ++ U, T' ++ U', ?_⟩
          rw [List.append_assoc, List.append_assoc, ← grown, ← grown']
          exact agree
        · by_cases still : (∃ key check, x.1 = .check key check) ∨ ∃ level, x.1 = .terminal level
          · exact (step_inert pub proof messages x.2 x.1 s still).symm.trans
              (alike.trans (step_inert pub proof' messages' x.2 x.1 s' still))
          · exact (step_sim pub proof proof' messages messages' x.2 x.1
              (Bool.eq_false_iff.mpr reads) (fun key check same => still (Or.inl ⟨key, check, same⟩))
              (fun level same => still (Or.inr ⟨level, same⟩)) s s' alike).2
      have moved := step_next pub proof messages x.2 x.1 s stepped
      have moved' := step_next pub proof' messages' x.2 x.1 s' stepped'
      refine run_alike l _ _ after (by rw [moved, moved', next]) tail tail' ?_ ⟨T, T', agree⟩
      rw [moved]
      exact faithful.2

/-- The core and the sponge of the state before the operation `c` are a function of the events of
the sponge before `c`: two proofs whose runs have no stop before `c`, with faithful messages and
the same events of the sponge before `c`, have alike states before `c`. -/
theorem state_causal (c : ℕ)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (faithful : Faithful proof proof' messages messages' (pub.ops.zipIdx.toList.take c) 0)
    (events : (stateBefore pub proof messages pub.ops c).sponge.events =
      (stateBefore pub proof' messages' pub.ops c).sponge.events) :
    Alike (stateBefore pub proof messages pub.ops c) (stateBefore pub proof' messages' pub.ops c) :=
  run_alike pub proof proof' messages messages' _ _ _ (Alike.refl _) rfl ran ran' faithful
    ⟨[], [], events⟩

/-- The core of the state before the operation `c` is a function of the events of the sponge
before `c`. -/
theorem core_faithful (c : ℕ)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (faithful : Faithful proof proof' messages messages' (pub.ops.zipIdx.toList.take c) 0)
    (events : (stateBefore pub proof messages pub.ops c).sponge.events =
      (stateBefore pub proof' messages' pub.ops c).sponge.events) :
    core (stateBefore pub proof messages pub.ops c) =
      core (stateBefore pub proof' messages' pub.ops c) :=
  (state_causal pub proof proof' messages messages' c ran ran' faithful events).2

end Causal

end Replayer

open Replayer

section Note

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] (pub : Public)

/-- `noteOf` reads a state only through its core. -/
theorem noteOf_core {s s' : Replayer} (same : core s = core s')
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) :
    noteOf (E := E) pub s tLast eLast = noteOf pub s' tLast eLast := by
  have recorded : recordedIn s = recordedIn s' := congrArg Core.recorded same
  have fields : ∀ k, s.fields.get? k = s'.fields.get? k :=
    fun k => congrFun (congrArg Core.fields same) k
  have bytes : ∀ k, s.derived.bytes.get? k = s'.derived.bytes.get? k :=
    fun k => congrFun (congrArg Core.bytes same) k
  simp only [noteOf, rootClaimIn, claimIn₁, traceClaimIn, terminalStatementIn, responseIn,
    terminalMessagesIn, bytesIn, firstIn, recorded, fields, bytes]

/-- The `Note` of the wire message `k` of the model is a function of the public input, the events
of the sponge before the work of the stage `k` and the last blocks: two proofs whose runs have no
stop before this work, with faithful messages and the same events before it, have the same
`noteAt20 … k`. -/
theorem noteAt20_faithful (proof proof' : ByteArray) (messages messages' : Array Message)
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) (k : ℕ)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (faithful : Faithful proof proof' messages messages'
      (pub.ops.zipIdx.toList.take (stageIndex20 pub k)) 0)
    (events : (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events =
      (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k)).sponge.events) :
    noteAt20 (E := E) pub proof messages tLast eLast k =
      noteAt20 pub proof' messages' tLast eLast k :=
  noteOf_core pub (core_faithful pub proof proof' messages messages' _ ran ran' faithful events)
    tLast eLast

end Note

section Decode

/-- The reading operations of `l` have the messages that `Wire.read` gives, in order from the
index `n`. -/
def ReadsFrom (proof : ByteArray) (messages : Array Message) : List (Op × ℕ) → ℕ → Prop
  | [], _ => True
  | x :: l, n => (x.1.reads = true → ∃ m at_, messages[n]? = some m ∧
      Wire.read proof at_ x.1 = some (some m)) ∧
    ReadsFrom proof messages l (n + if x.1.reads then 1 else 0)

theorem decode_step_from (proof : ByteArray)
    (f : Array Message × ℕ → Op × ℕ → Except Malformed (Array Message × ℕ))
    (step : ∀ acc x, f acc x = match Wire.read proof acc.2 x.1 with
      | none => Except.error (⟨x.2, acc.2⟩ : Malformed)
      | some none => .ok (acc.1, acc.2)
      | some (some m) => .ok (acc.1.push m, acc.2 + m.size)) :
    ∀ (l : List (Op × ℕ)) (start : Array Message) (cursor : ℕ) (messages : Array Message)
      (last : ℕ), l.foldlM f (start, cursor) = .ok (messages, last) →
      (∀ j < start.size, messages[j]? = start[j]?) ∧ ReadsFrom proof messages l start.size
  | [], start, cursor, messages, last, folded => by
      simp only [List.foldlM_nil] at folded
      cases folded
      exact ⟨fun _ _ => rfl, trivial⟩
  | x :: l, start, cursor, messages, last, folded => by
      rw [List.foldlM_cons, step] at folded
      cases read : Wire.read proof cursor x.1 with
      | none =>
          rw [read] at folded
          cases folded
      | some found =>
          cases found with
          | none =>
              rw [read] at folded
              obtain ⟨same, aligned⟩ :=
                decode_step_from proof f step l start cursor messages last folded
              have quiet := read_none read
              have stay : start.size + (if x.1.reads = true then 1 else 0) = start.size := by
                simp [quiet]
              refine ⟨same, fun reads => ?_, ?_⟩
              · rw [quiet] at reads
                cases reads
              · rw [stay]
                exact aligned
          | some m =>
              rw [read] at folded
              obtain ⟨same, aligned⟩ := decode_step_from proof f step l (start.push m)
                (cursor + m.size) messages last folded
              obtain ⟨reads, -⟩ := read_some read
              have move : start.size + (if x.1.reads = true then 1 else 0) = start.size + 1 := by
                simp [reads]
              refine ⟨fun j below => by
                rw [same j (by rw [Array.size_push]; omega), Array.getElem?_push_lt below]
                exact (Array.getElem?_eq_getElem below).symm, fun _ => ⟨m, cursor, ?_, read⟩, ?_⟩
              · rw [same _ (by rw [Array.size_push]; omega), Array.getElem?_push_size]
              · rw [move]
                rw [Array.size_push] at aligned
                exact aligned

/-- The messages of `decode` are the messages that `Wire.read` gives for the reading operations
of the script, in order. -/
theorem decode_from {ops : Array Op} {proof : ByteArray} {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) :
    ReadsFrom proof messages ops.zipIdx.toList 0 := by
  unfold decode at decoded
  rw [← Array.foldlM_toList] at decoded
  generalize found : List.foldlM (m := Except Malformed) _ ((#[] : Array Message), 0)
    ops.zipIdx.toList = result at decoded
  rcases result with e | ⟨ms, last⟩
  · cases decoded
  · have same : ms = messages := by
      injection decoded with pair
      injection pair
    subst same
    exact (decode_step_from proof _ (fun acc x => by
      obtain ⟨_, _⟩ := acc
      obtain ⟨_, _⟩ := x
      rfl) _ _ _ _ _ found).2

/-- The data of a message that `Wire.read` gives for the operation is a function of the bytes
that the step of the operation absorbs. -/
def Op.Parses (op : Op) : Prop :=
  ∀ (proof proof' : ByteArray) (at_ at_' : ℕ) (m m' : Message),
    Wire.read proof at_ op = some (some m) → Wire.read proof' at_' op = some (some m') →
    op.input proof m = op.input proof' m' → m.data = m'.data

/-- Messages that `Wire.read` gives are faithful, when the data of each stored message is a
function of its absorbed bytes. -/
theorem faithful_of_reads (proof proof' : ByteArray) (messages messages' : Array Message) :
    ∀ (l : List (Op × ℕ)) (n : ℕ), (∀ x ∈ l, x.1.stores = true → x.1.Parses) →
      ReadsFrom proof messages l n → ReadsFrom proof' messages' l n →
      Faithful proof proof' messages messages' l n
  | [], _, _, _, _ => trivial
  | x :: l, n, parses, reads, reads' => by
      refine ⟨fun stores m m' found found' input => ?_, faithful_of_reads proof proof' messages
        messages' l _ (fun y inside => parses y (List.mem_cons_of_mem _ inside)) reads.2 reads'.2⟩
      have reading : x.1.reads = true := by
        obtain ⟨op, i⟩ := x
        cases op <;> first | rfl | simp [Op.stores] at stores
      obtain ⟨m₁, at_, found₁, read⟩ := reads.1 reading
      obtain ⟨m₂, at_', found₂, read'⟩ := reads'.1 reading
      rw [found] at found₁
      rw [found'] at found₂
      cases found₁
      cases found₂
      exact parses x List.mem_cons_self stores proof proof' at_ at_' m m' read read' input

theorem foldr_congr_mem {β : Type} (f g : ℕ → β → β) (b : β) : ∀ l : List ℕ,
    (∀ i ∈ l, ∀ acc, f i acc = g i acc) → l.foldr f b = l.foldr g b
  | [], _ => rfl
  | i :: l, same => by
      simp only [List.foldr_cons]
      rw [foldr_congr_mem f g b l fun j member => same j (List.mem_cons_of_mem _ member),
        same i List.mem_cons_self]

theorem foldlM_congr_mem {β : Type} (f g : β → ℕ → Option β) : ∀ (l : List ℕ) (b : β),
    (∀ i ∈ l, ∀ acc, f acc i = g acc i) → l.foldlM f b = l.foldlM g b
  | [], _, _ => rfl
  | i :: l, b, same => by
      simp only [List.foldlM_cons]
      rw [same i List.mem_cons_self]
      exact congrArg (g b i >>= ·) (funext fun c =>
        foldlM_congr_mem f g l c fun j member => same j (List.mem_cons_of_mem _ member))

/-- A byte of a part of a byte array is the byte of the array. -/
theorem get!_extract (bytes : ByteArray) {start stop i : ℕ} (inside : start + i < stop)
    (fits : stop ≤ bytes.size) : (bytes.extract start stop).get! i = bytes.get! (start + i) := by
  show (bytes.extract start stop).data[i]! = bytes.data[start + i]!
  rw [ByteArray.data_extract]
  have small : i < (bytes.data.extract start stop).size := by
    rw [Array.size_extract, ByteArray.size_data, Nat.min_eq_left fits]
    omega
  have big : start + i < bytes.data.size := by
    rw [ByteArray.size_data]
    omega
  rw [getElem!_pos (bytes.data.extract start stop) i small,
    getElem!_pos bytes.data (start + i) big, Array.getElem_extract]

/-- A word of a part of a byte array is the word of the array. -/
theorem word_extract (bytes : ByteArray) {start stop at_ : ℕ} (inside : start + at_ + 8 ≤ stop)
    (fits : stop ≤ bytes.size) :
    Wire.word (bytes.extract start stop) at_ = Wire.word bytes (start + at_) := by
  unfold Wire.word
  refine foldr_congr_mem _ _ _ _ fun i member acc => ?_
  have below := List.mem_range.mp member
  show (acc <<< 8) ||| ((bytes.extract start stop).get! (at_ + i)).toUInt64 =
    (acc <<< 8) ||| (bytes.get! (start + at_ + i)).toUInt64
  rw [get!_extract bytes (by omega) fits, Nat.add_assoc]

/-- A number of a part of a byte array is the number of the array. -/
theorem number_extract (bytes : ByteArray) {start stop n : ℕ} (inside : start + n ≤ stop)
    (fits : stop ≤ bytes.size) :
    Wire.number (bytes.extract start stop) 0 n = Wire.number bytes start n := by
  unfold Wire.number
  refine foldr_congr_mem _ _ _ _ fun i member acc => ?_
  have below := List.mem_range.mp member
  show acc * 256 + ((bytes.extract start stop).get! (0 + i)).toNat =
    acc * 256 + (bytes.get! (start + i)).toNat
  rw [get!_extract bytes (by omega) fits, Nat.zero_add]

/-- The field elements of a message are those of its bytes. -/
theorem fields_extract {proof : ByteArray} {at_ count : ℕ}
    (fits : ¬ proof.size < at_ + 8 * count) :
    Wire.fields (proof.extract at_ (at_ + 8 * count)) 0 count = Wire.fields proof at_ count := by
  have size : (proof.extract at_ (at_ + 8 * count)).size = 8 * count := by
    rw [ByteArray.size_extract, Nat.min_eq_left (by omega)]
    omega
  unfold Wire.fields
  rw [ite_eq_right fits, ite_eq_right (by rw [size]; omega)]
  refine foldlM_congr_mem _ _ _ _ fun i member acc => ?_
  have below := List.mem_range.mp member
  show (Fq.ofWord? (Wire.word (proof.extract at_ (at_ + 8 * count)) (0 + 8 * i))).map acc.push =
    (Fq.ofWord? (Wire.word proof (at_ + 8 * i))).map acc.push
  rw [word_extract proof (by omega) (by omega), Nat.zero_add]

theorem fields_fits {proof : ByteArray} {at_ count : ℕ} {values : Array Fq}
    (found : Wire.fields proof at_ count = some values) : ¬ proof.size < at_ + 8 * count := by
  intro short
  unfold Wire.fields at found
  rw [ite_eq_left short] at found
  cases found

theorem parses_fields (key : Key) (count : ℕ) : (Op.fields key count).Parses := by
  intro proof proof' at_ at_' m m' read read' input
  simp only [Wire.read, Option.map_eq_some_iff] at read read'
  obtain ⟨v, found, made⟩ := read
  obtain ⟨v', found', made'⟩ := read'
  cases made
  cases made'
  have same : proof.extract at_ (at_ + 8 * count) = proof'.extract at_' (at_' + 8 * count) := input
  rw [← fields_extract (fields_fits found), same, fields_extract (fields_fits found')] at found
  rw [found] at found'
  cases found'
  rfl

theorem parses_exts (key : Key) (count : ℕ) : (Op.exts key count).Parses := by
  intro proof proof' at_ at_' m m' read read' input
  simp only [Wire.read, Option.map_eq_some_iff] at read read'
  obtain ⟨w, found, made⟩ := read
  obtain ⟨w', found', made'⟩ := read'
  cases made
  cases made'
  have same : proof.extract at_ (at_ + 16 * count) = proof'.extract at_' (at_' + 16 * count) :=
    input
  rw [show 16 * count = 8 * (2 * count) by omega] at same
  unfold Wire.exts at found found'
  cases words : Wire.fields proof at_ (2 * count) with
  | none =>
      rw [words] at found
      cases found
  | some values =>
      cases words' : Wire.fields proof' at_' (2 * count) with
      | none =>
          rw [words'] at found'
          cases found'
      | some values' =>
          rw [words] at found
          rw [words'] at found'
          rw [← fields_extract (fields_fits words), same, fields_extract (fields_fits words')]
            at words
          rw [words] at words'
          cases words'
          cases found
          cases found'
          rfl

theorem parses_integer (key : Key) : (Op.integer key).Parses := by
  intro proof proof' at_ at_' m m' read read' input
  simp only [Wire.read] at read read'
  split at read
  · cases read
  · rename_i fits
    split at read'
    · cases read'
    · rename_i fits'
      cases read
      cases read'
      have same : proof.extract at_ (at_ + 16) = proof'.extract at_' (at_' + 16) := input
      show Data.integer (Wire.number proof at_ 16) = Data.integer (Wire.number proof' at_' 16)
      rw [← number_extract proof (Nat.le_refl _) (by omega), same,
        number_extract proof' (Nat.le_refl _) (by omega)]

theorem parses_digest (key : Key) : (Op.digest key).Parses := by
  intro proof proof' at_ at_' m m' read read' input
  simp only [Wire.read] at read read'
  split at read
  · cases read
  · split at read'
    · cases read'
    · cases read
      cases read'
      exact congrArg Data.bytes input

/-- The length of the payload of `z` is below `2^32`. -/
theorem number_lt (proof : ByteArray) (at_ : ℕ) : Wire.number proof at_ 4 < 2 ^ 32 := by
  have r4 : List.range 4 = [0, 1, 2, 3] := rfl
  unfold Wire.number
  rw [r4]
  simp only [List.foldr_cons, List.foldr_nil]
  have b0 := UInt8.toNat_lt (proof.get! (at_ + 0))
  have b1 := UInt8.toNat_lt (proof.get! (at_ + 1))
  have b2 := UInt8.toNat_lt (proof.get! (at_ + 2))
  have b3 := UInt8.toNat_lt (proof.get! (at_ + 3))
  omega

/-- The 8 bytes of a number below `2^64` give the number. -/
theorem le64_inj {a b : ℕ} (small : a < 2 ^ 64) (small' : b < 2 ^ 64) (same : le64 a = le64 b) :
    a = b := by
  have left := word_of_data (bytes := le64 a) (front := #[]) (rest := #[]) small (by simp)
  have right := word_of_data (bytes := le64 b) (front := #[]) (rest := #[]) small' (by simp)
  rw [same, right] at left
  have nat := congrArg UInt64.toNat left
  simp only [Nat.toUInt64, UInt64.toNat_ofNat'] at nat
  omega

theorem parses_bounded (key : Key) (most : ℕ) : (Op.bounded key most).Parses := by
  intro proof proof' at_ at_' m m' read read' input
  simp only [Wire.read] at read read'
  have small := number_lt proof at_
  have small' := number_lt proof' at_'
  generalize Wire.number proof at_ 4 = l at read small
  generalize Wire.number proof' at_' 4 = l' at read' small'
  split at read
  · cases read
  · split at read
    · cases read
    · rename_i fits
      split at read'
      · cases read'
      · split at read'
        · cases read'
        · rename_i fits'
          cases read
          cases read'
          have same : le64 (4 + l - 4) ++ wordPadded (proof.extract (at_ + 4) (at_ + (4 + l))) =
              le64 (4 + l' - 4) ++ wordPadded (proof'.extract (at_' + 4) (at_' + (4 + l'))) :=
            input
          rw [Nat.add_sub_cancel_left, Nat.add_sub_cancel_left, ← Nat.add_assoc,
            ← Nat.add_assoc] at same
          have lists := congrArg (fun b : ByteArray => b.data.toList) same
          simp only [ByteArray.data_append, Array.toList_append] at lists
          obtain ⟨heads, tails⟩ := List.append_inj lists (by simp [le64])
          have length : l = l' := le64_inj (by omega) (by omega)
            (ByteArray.ext (Array.toList_inj.mp heads))
          have size : (proof.extract (at_ + 4) (at_ + 4 + l)).size = l := by
            rw [ByteArray.size_extract, Nat.min_eq_left (by omega)]
            omega
          have size' : (proof'.extract (at_' + 4) (at_' + 4 + l')).size = l' := by
            rw [ByteArray.size_extract, Nat.min_eq_left (by omega)]
            omega
          unfold wordPadded at tails
          simp only [ByteArray.data_append, Array.toList_append] at tails
          have bytes := (List.append_inj tails (by
            rw [Array.length_toList, Array.length_toList, ByteArray.size_data, ByteArray.size_data,
              size, size', length])).1
          show Data.bytes (proof.extract (at_ + 4) (at_ + 4 + l)) =
            Data.bytes (proof'.extract (at_' + 4) (at_' + 4 + l'))
          exact congrArg Data.bytes (ByteArray.ext (Array.toList_inj.mp bytes))

/-- The data of each message that an operation stores is a function of the bytes that the step of
the operation absorbs. -/
theorem parses_of_stores {op : Op} (stores : op.stores = true) : op.Parses := by
  cases op with
  | fields key count => exact parses_fields key count
  | exts key count => exact parses_exts key count
  | integer key => exact parses_integer key
  | digest key => exact parses_digest key
  | bounded key most => exact parses_bounded key most
  | _ => simp [Op.stores] at stores

theorem Faithful.take {proof proof' : ByteArray} {messages messages' : Array Message} :
    ∀ (l : List (Op × ℕ)) (n c : ℕ), Faithful proof proof' messages messages' l n →
      Faithful proof proof' messages messages' (l.take c) n
  | [], _, _, _ => by
      rw [List.take_nil]
      trivial
  | _ :: _, _, 0, _ => trivial
  | _ :: l, _, c + 1, faithful => ⟨faithful.1, Faithful.take l _ c faithful.2⟩

/-- The decoded messages of two proofs of one script are faithful: the data of each stored
message is a function of the bytes that the replay absorbs for it. -/
theorem decode_faithful {ops : Array Op} {proof proof' : ByteArray}
    {messages messages' : Array Message} {rest rest' : ℕ}
    (decoded : decode ops proof = .ok (messages, rest))
    (decoded' : decode ops proof' = .ok (messages', rest')) :
    Faithful proof proof' messages messages' ops.zipIdx.toList 0 :=
  faithful_of_reads proof proof' messages messages' _ 0 (fun _ _ stores => parses_of_stores stores)
    (decode_from decoded) (decode_from decoded')

end Decode

section Decoded

open Replayer

variable (pub : Public) (proof proof' : ByteArray) (messages messages' : Array Message)

/-- The core of the state of the replay before the operation `c` is a function of the public
input and the events of the sponge before `c`: two proofs with their decoded messages, whose runs
have no stop before `c` and have the same events of the sponge before `c`, have the same core
before `c`. -/
theorem core_causal (c : ℕ) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (events : (stateBefore pub proof messages pub.ops c).sponge.events =
      (stateBefore pub proof' messages' pub.ops c).sponge.events) :
    core (stateBefore pub proof messages pub.ops c) =
      core (stateBefore pub proof' messages' pub.ops c) :=
  core_faithful pub proof proof' messages messages' c ran ran'
    (Faithful.take _ 0 c (decode_faithful decoded decoded')) events

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The `Note` of the wire message `k` of the model is a function of the public input, the events
of the sponge before the work of the stage `k` and the last blocks: two proofs with their decoded
messages, whose runs have no stop before this work and have the same events before it, have the
same `noteAt20 … k`. -/
theorem noteAt20_causal (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (eLast : Rq Key20.terminal.innerDegree) (k : ℕ) {rest rest' : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom pub proof' messages' (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (events : (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events =
      (stateBefore pub proof' messages' pub.ops (stageIndex20 pub k)).sponge.events) :
    noteAt20 (E := E) pub proof messages tLast eLast k =
      noteAt20 pub proof' messages' tLast eLast k :=
  noteOf_core pub (core_causal pub proof proof' messages messages' _ decoded decoded' ran ran'
    events) tLast eLast

end Decoded

end NightstreamFPrime.Spec.Compression.Akita.Replay
