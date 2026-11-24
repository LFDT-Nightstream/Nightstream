import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelIn
import NightstreamFPrime.Spec.Compression.Akita.Replay.Readback

/-! What one operation of the replay writes (`Replay/Transcript.lean`, `Replayer.step`; Akita
revision `d50f700`), and runs of indexed operations.

- `decodeExts_extBytes_array`: `decodeExts` reads back each array of elements of `E` from its
  bytes `extBytes`. An `Fq` is its word, so the round trip has no condition.
- `runFrom` is the run of indexed operations from a state, with its result; `runFrom_append`:
  a run of `a ++ b` is the run of `a`, then the run of `b` when `a` does not stop.
- `Op.ends` is the key of the end of a sum-check. `RoundsTail R s s'`: from `s` to `s'` the
  replay appends to the rounds only entries with a key in `R` (`step_tail`, `runFrom_tail`), so
  `roundsIn` of another key does not change (`roundsIn_tail`).
- `step_absorb_fused`, `step_absorb_recorded`, `step_challenge_ok`: an absorb of the fused claim
  records `(stage2In …).fused`, an absorb of recorded values records them, and a squeeze of
  challenges records `count` elements. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The bytes of `extBytes` of a list of elements of `E`. -/
def extData (l : List Ext) : List UInt8 :=
  l.flatMap fun e => (le64 e.c0.word.toNat).data.toList ++ (le64 e.c1.word.toNat).data.toList

theorem le64_size (n : ℕ) : (le64 n).data.size = 8 := by simp [le64]

theorem extData_cons (e : Ext) (l : List Ext) :
    extData (e :: l) = (le64 e.c0.word.toNat).data.toList ++
      (le64 e.c1.word.toNat).data.toList ++ extData l := by
  simp only [extData, List.flatMap_cons]

theorem extData_append (a b : List Ext) : extData (a ++ b) = extData a ++ extData b := by
  simp only [extData, List.flatMap_append]

theorem extData_length (l : List Ext) : (extData l).length = 16 * l.length := by
  induction l with
  | nil => rfl
  | cons e l ih =>
      rw [extData_cons, List.length_append, List.length_append, ih, Array.length_toList,
        Array.length_toList, le64_size, le64_size, List.length_cons]
      omega

theorem extBytes_data (xs : Array Ext) : (extBytes xs).data = (extData xs.toList).toArray := by
  have key : ∀ (l : List Ext) (out : ByteArray),
      (l.foldl (fun out e => out ++ le64 e.c0.word.toNat ++ le64 e.c1.word.toNat) out).data =
        out.data ++ (extData l).toArray := by
    intro l
    induction l with
    | nil => intro out; simp [extData]
    | cons e l ih =>
        intro out
        rw [List.foldl_cons, ih, ByteArray.data_append, ByteArray.data_append, extData_cons]
        refine Array.toList_inj.mp ?_
        simp
  unfold extBytes
  rw [← Array.foldl_toList, key]
  simp

theorem extBytes_split (xs : Array Ext) (j : ℕ) (inside : j < xs.size) :
    (extBytes xs).data = (extData (xs.toList.take j)).toArray ++
      (le64 xs[j].c0.word.toNat).data ++
      ((le64 xs[j].c1.word.toNat).data ++ (extData (xs.toList.drop (j + 1))).toArray) := by
  have cut : extData xs.toList = extData (xs.toList.take j) ++
      ((le64 xs[j].c0.word.toNat).data.toList ++ ((le64 xs[j].c1.word.toNat).data.toList ++
        extData (xs.toList.drop (j + 1)))) := by
    conv_lhs => rw [← List.take_append_drop j xs.toList,
      List.drop_eq_getElem_cons (by rw [Array.length_toList]; exact inside)]
    simp only [extData_append, extData_cons, Array.getElem_toList, List.append_assoc]
  rw [extBytes_data, cut]
  refine Array.toList_inj.mp ?_
  simp

/-- `decodeExts` reads back each array of elements of `E` from its bytes. -/
theorem decodeExts_extBytes_array (xs : Array Ext) :
    Replayer.step.decodeExts (extBytes xs) = pure xs := by
  have size : (extBytes xs).size = 16 * xs.size := by
    rw [← ByteArray.size_data, extBytes_data, List.size_toArray, extData_length, Array.length_toList]
  unfold Replayer.step.decodeExts
  rw [size, Nat.mul_div_cancel_left _ (by norm_num)]
  congr 1
  refine Array.ext (by simp) fun j h₁ _ => ?_
  have inside : j < xs.size := by simpa using h₁
  have front : (extData (xs.toList.take j)).toArray.size = 16 * j := by
    rw [List.size_toArray, extData_length, List.length_take, Array.length_toList,
      Nat.min_eq_left inside.le]
  have w0 := word_of_data xs[j].c0.word.toNat_lt (extBytes_split xs j inside)
  have w8 := word_of_data (front := (extData (xs.toList.take j)).toArray ++
      (le64 xs[j].c0.word.toNat).data) (rest := (extData (xs.toList.drop (j + 1))).toArray)
    xs[j].c1.word.toNat_lt (by rw [extBytes_split xs j inside]; simp [Array.append_assoc])
  rw [front] at w0
  rw [Array.size_append, front, le64_size] at w8
  simp only [Array.getElem_map, Array.getElem_range, w0, w8, Nat.toUInt64, UInt64.ofNat_toNat]

namespace Replayer

/-- The run of the indexed operations `l` from `s`: its result and its last state. -/
def runFrom (pub : Public) (proof : ByteArray) (messages : Array Message) (l : List (Op × ℕ))
    (s : Replayer) : Except Stop Unit × Replayer :=
  ((l.forM fun x => step pub proof messages x.2 x.1).run).run s

section Runs

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

theorem runFrom_nil (s : Replayer) : runFrom pub proof messages [] s = (.ok (), s) := rfl

theorem runOps_eq (l : List (Op × ℕ)) (s : Replayer) :
    runOps pub proof messages l s = (runFrom pub proof messages l s).2 := rfl

theorem runFrom_cons (x : Op × ℕ) (l : List (Op × ℕ)) (s : Replayer) :
    runFrom pub proof messages (x :: l) s =
      match ((step pub proof messages x.2 x.1).run).run s with
      | (.ok _, s₁) => runFrom pub proof messages l s₁
      | (.error e, s₁) => (.error e, s₁) := by
  unfold runFrom
  simp only [List.forM]
  rw [run_bind]
  rcases ((step pub proof messages x.2 x.1).run).run s with ⟨_ | _, _⟩ <;> rfl

theorem runFrom_append (a b : List (Op × ℕ)) (s : Replayer) :
    runFrom pub proof messages (a ++ b) s =
      match runFrom pub proof messages a s with
      | (.ok _, s₁) => runFrom pub proof messages b s₁
      | (.error e, s₁) => (.error e, s₁) := by
  unfold runFrom
  rw [forM_append, run_bind]
  rcases ((a.forM fun x => step pub proof messages x.2 x.1).run).run s with ⟨_ | _, _⟩ <;> rfl

/-- A run of `a ++ b` ends without a stop exactly when the run of `a` does and the run of `b`
from its last state does. -/
theorem runFrom_append_fst (a b : List (Op × ℕ)) (s : Replayer) :
    (runFrom pub proof messages (a ++ b) s).1 = .ok () ↔
      (runFrom pub proof messages a s).1 = .ok () ∧
        (runFrom pub proof messages b (runFrom pub proof messages a s).2).1 = .ok () := by
  rw [runFrom_append]
  rcases runFrom pub proof messages a s with ⟨_ | ⟨⟩, s₁⟩ <;> simp

theorem runFrom_append_snd (a b : List (Op × ℕ)) (s : Replayer)
    (ok : (runFrom pub proof messages a s).1 = .ok ()) :
    (runFrom pub proof messages (a ++ b) s).2 =
      (runFrom pub proof messages b (runFrom pub proof messages a s).2).2 := by
  rw [runFrom_append]
  rcases found : runFrom pub proof messages a s with ⟨_ | ⟨⟩, s₁⟩
  · rw [found] at ok; cases ok
  · rfl

theorem runFrom_cons_fst (x : Op × ℕ) (l : List (Op × ℕ)) (s : Replayer) :
    (runFrom pub proof messages (x :: l) s).1 = .ok () ↔
      (((step pub proof messages x.2 x.1).run).run s).1 = .ok () ∧
        (runFrom pub proof messages l (((step pub proof messages x.2 x.1).run).run s).2).1 =
          .ok () := by
  rw [runFrom_cons]
  rcases ((step pub proof messages x.2 x.1).run).run s with ⟨_ | ⟨⟩, s₁⟩ <;> simp

theorem runFrom_cons_snd (x : Op × ℕ) (l : List (Op × ℕ)) (s : Replayer)
    (ok : (((step pub proof messages x.2 x.1).run).run s).1 = .ok ()) :
    (runFrom pub proof messages (x :: l) s).2 =
      (runFrom pub proof messages l (((step pub proof messages x.2 x.1).run).run s).2).2 := by
  rw [runFrom_cons]
  rcases found : ((step pub proof messages x.2 x.1).run).run s with ⟨_ | ⟨⟩, s₁⟩
  · rw [found] at ok; cases ok
  · rfl

/-- The operation `j` of a run without a stop: the run up to it, its step and the run after it
have no stop, and the last state is the state of the run after it. -/
theorem runFrom_at (l : List (Op × ℕ)) (j : ℕ) (inside : j < l.length) (s : Replayer)
    (ok : (runFrom pub proof messages l s).1 = .ok ()) :
    (runFrom pub proof messages (l.take j) s).1 = .ok () ∧
      (((step pub proof messages l[j].2 l[j].1).run).run
        (runFrom pub proof messages (l.take j) s).2).1 = .ok () ∧
      (runFrom pub proof messages l s).2 =
        (runFrom pub proof messages (l.drop j) (runFrom pub proof messages (l.take j) s).2).2 ∧
      (runFrom pub proof messages l s).2 = (runFrom pub proof messages (l.drop (j + 1))
        (((step pub proof messages l[j].2 l[j].1).run).run
          (runFrom pub proof messages (l.take j) s).2).2).2 := by
  have cut : l = l.take j ++ l[j] :: l.drop (j + 1) := by
    rw [← List.drop_eq_getElem_cons inside, List.take_append_drop]
  have whole := ok
  rw [cut, runFrom_append_fst, runFrom_cons_fst] at whole
  obtain ⟨first, stepped, _⟩ := whole
  have later : (runFrom pub proof messages l s).2 =
      (runFrom pub proof messages (l.drop j) (runFrom pub proof messages (l.take j) s).2).2 := by
    conv_lhs => rw [← List.take_append_drop j l]
    exact runFrom_append_snd pub proof messages _ _ s first
  refine ⟨first, stepped, later, ?_⟩
  rw [later, List.drop_eq_getElem_cons inside]
  exact runFrom_cons_snd pub proof messages _ _ _ stepped

theorem runFrom_grows (l : List (Op × ℕ)) (s : Replayer) :
    Grows (l.flatMap fun x => x.1.writes) s (runFrom pub proof messages l s).2 :=
  keeps_forM pub proof messages l s

end Runs

end Replayer

/-- The key of the end of a sum-check: the operation `.ended` appends its rounds to the rounds. -/
def Op.ends : Op → List Key
  | .ended key _ _ => [key]
  | _ => []

/-- The count of elements that a step without a stop appends to `key`: `count` for a squeeze of
`count` challenges into `key`, `0` for an operation that does not write `key`, and `none`
otherwise. -/
def Op.drawsInto (key : Key) : Op → Option ℕ
  | .challenge k count => if k = key then some count else some 0
  | op => if key ∈ op.writes then none else some 0

/-- The count of elements that a run of `l` without a stop appends to `key`, when each operation
that writes `key` is a squeeze of challenges. -/
def drawnCount (key : Key) : List (Op × ℕ) → Option ℕ
  | [] => some 0
  | x :: l => (x.1.drawsInto key).bind fun c => (drawnCount key l).map (c + ·)

namespace Replayer

/-- From `s` to `s'` the replay appends to the rounds only entries with a key in `R`. -/
def RoundsTail (R : List Key) (s s' : Replayer) : Prop :=
  ∃ extra : List (Key × List (List Ext × Ext)),
    s'.derived.rounds.toList = s.derived.rounds.toList ++ extra ∧ ∀ x ∈ extra, x.1 ∈ R

theorem RoundsTail.of_eq {R : List Key} {s s' : Replayer}
    (same : s'.derived.rounds = s.derived.rounds) : RoundsTail R s s' :=
  ⟨[], by rw [same, List.append_nil], fun _ inside => absurd inside List.not_mem_nil⟩

theorem RoundsTail.refl (R : List Key) (s : Replayer) : RoundsTail R s s := RoundsTail.of_eq rfl

theorem RoundsTail.trans {R : List Key} {s₁ s₂ s₃ : Replayer} (a : RoundsTail R s₁ s₂)
    (b : RoundsTail R s₂ s₃) : RoundsTail R s₁ s₃ := by
  obtain ⟨e₁, h₁, k₁⟩ := a
  obtain ⟨e₂, h₂, k₂⟩ := b
  exact ⟨e₁ ++ e₂, by rw [h₂, h₁, List.append_assoc],
    fun x inside => (List.mem_append.mp inside).elim (k₁ x) (k₂ x)⟩

theorem RoundsTail.mono {R R' : List Key} {s s' : Replayer} (sub : ∀ k ∈ R, k ∈ R')
    (a : RoundsTail R s s') : RoundsTail R' s s' := by
  obtain ⟨e, h, k⟩ := a
  exact ⟨e, h, fun x inside => sub _ (k x inside)⟩

theorem RoundsTail.push {R : List Key} {s s' : Replayer} {x : Key × List (List Ext × Ext)}
    (pushed : s'.derived.rounds = s.derived.rounds.push x) (inside : x.1 ∈ R) :
    RoundsTail R s s' :=
  ⟨[x], by rw [pushed, Array.toList_push], fun y found => by
    rw [List.mem_singleton.mp found]; exact inside⟩

/-- `m` appends to the rounds of each state only entries with a key in `R`. -/
def KeepsTail {α : Type} (R : List Key) (m : ReplayM α) : Prop :=
  ∀ s, RoundsTail R s ((m.run).run s).2

section Tail

variable {R : List Key}

theorem tail_bind {α β : Type} {m : ReplayM α} {f : α → ReplayM β} (hm : KeepsTail R m)
    (hf : ∀ a, KeepsTail R (f a)) : KeepsTail R (m >>= f) := by
  intro s
  rw [run_bind]
  rcases found : (m.run).run s with ⟨_ | a, s₁⟩
  · have tail := hm s
    rw [found] at tail
    exact tail
  · have tail := hm s
    rw [found] at tail
    exact tail.trans (hf a s₁)

theorem tail_pure {α : Type} (a : α) : KeepsTail R (Pure.pure a : ReplayM α) :=
  fun s => RoundsTail.refl R s

theorem tail_throw {α : Type} (e : Stop) : KeepsTail R (throw e : ReplayM α) :=
  fun s => RoundsTail.refl R s

theorem tail_get : KeepsTail R (get : ReplayM Replayer) := fun s => RoundsTail.refl R s

theorem tail_get_bind {α : Type} {f : Replayer → ReplayM α}
    (h : ∀ s, RoundsTail R s (((f s).run).run s).2) : KeepsTail R (get >>= f) :=
  fun s => h s

theorem tail_modify {g : Replayer → Replayer} (h : ∀ s, RoundsTail R s (g s)) :
    KeepsTail R (modify g : ReplayM Unit) := h

theorem tail_mapM {α β : Type} {f : α → ReplayM β} (each : ∀ a, KeepsTail R (f a)) :
    ∀ l : List α, KeepsTail R (l.mapM f)
  | [] => tail_pure _
  | a :: l => by
      rw [List.mapM_cons]
      exact tail_bind (each a) fun _ => tail_bind (tail_mapM each l) fun _ => tail_pure _

theorem tail_foldlM {α β : Type} {f : β → α → ReplayM β}
    (each : ∀ b a, KeepsTail R (f b a)) : ∀ (l : List α) (b : β), KeepsTail R (l.foldlM f b)
  | [], _ => tail_pure _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact tail_bind (each b a) fun c => tail_foldlM each l c

theorem tail_recorded (key : Key) : KeepsTail R (recorded key) :=
  tail_bind tail_get fun _ => tail_pure _

theorem tail_first (key : Key) : KeepsTail R (first key) :=
  tail_bind tail_get fun _ => tail_pure _

theorem tail_fieldsOf (key : Key) : KeepsTail R (fieldsOf key) :=
  tail_bind tail_get fun _ => tail_pure _

theorem tail_pointBefore (before : Before) : KeepsTail R (pointBefore before) :=
  tail_bind tail_get fun _ => tail_pure _

theorem tail_absorb (bytes : ByteArray) : KeepsTail R (absorb bytes) :=
  tail_modify fun _ => RoundsTail.of_eq rfl

theorem tail_record (key : Key) (values : Array Ext) : KeepsTail R (record key values) :=
  tail_modify fun _ => RoundsTail.of_eq rfl

theorem tail_squeeze (count : ℕ) : KeepsTail R (squeeze count) :=
  tail_get_bind fun _ => RoundsTail.of_eq rfl

theorem tail_message (i : ℕ) (messages : Array Message) : KeepsTail R (message i messages) := by
  refine tail_get_bind fun s => ?_
  cases messages[s.next]? with
  | none => exact RoundsTail.refl R s
  | some m => exact RoundsTail.of_eq rfl

macro "replay_tail" : tactic => `(tactic| repeat' (first
  | with_reducible exact tail_pure _ | with_reducible exact tail_throw _
  | with_reducible exact tail_get
  | with_reducible exact tail_squeeze _ | with_reducible exact tail_message _ _
  | with_reducible exact tail_recorded _ | with_reducible exact tail_first _
  | with_reducible exact tail_fieldsOf _ | with_reducible exact tail_pointBefore _
  | with_reducible exact tail_absorb _ | with_reducible exact tail_record _ _
  | (with_reducible refine tail_modify fun _ => ?_; exact RoundsTail.of_eq rfl)
  | refine tail_bind ?_ fun _ => ?_
  | split))

theorem tail_stage2Of (pub : Public) (level : ℕ) (F : Fold) :
    KeepsTail R (stage2Of pub level F) := by
  unfold stage2Of
  replay_tail

theorem tail_value (pub : Public) (key : Key) (v : Value) : KeepsTail R (value pub key v) := by
  cases v with
  | recorded keys =>
      exact tail_bind (tail_foldlM (fun _ _ => tail_bind (tail_recorded _) fun _ =>
        tail_pure _) keys _) fun _ => tail_pure _
  | fused F =>
      simp only [value]
      refine tail_bind (tail_stage2Of pub _ F) fun _ => ?_
      replay_tail
  | _ => simp only [value]; replay_tail

theorem tail_weightsOf (weights : Weights) : KeepsTail R (weightsOf weights) := by
  cases weights <;> simp only [weightsOf] <;> replay_tail

theorem tail_holds (pub : Public) (check : Check) : KeepsTail R (holds pub check) := by
  cases check <;> simp only [holds]
  · replay_tail
  · refine tail_bind (tail_first _) fun _ => tail_bind (tail_weightsOf _) fun _ =>
      tail_bind (tail_recorded _) fun _ => tail_pure _
  all_goals replay_tail

theorem tail_checkWith (pub : Public) (i : ℕ) (key : Key) (check : Check)
    (out : Option StageOut) : KeepsTail R (checkWith pub i key check out) := by
  cases out with
  | some out => simp only [checkWith]; replay_tail
  | none =>
      simp only [checkWith]
      refine tail_bind (tail_holds pub check) fun _ => ?_
      replay_tail

theorem tail_draw (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) :
    KeepsTail R (draw i level group G bytes) := by
  unfold draw
  replay_tail

theorem tail_terminalWith (i level : ℕ) (r : Option (Except String TerminalValues)) :
    KeepsTail R (terminalWith i level r) := by
  rcases r with _ | _ | _ <;> simp only [terminalWith] <;> replay_tail

theorem tail_decodeExts (bytes : ByteArray) : KeepsTail R (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact tail_pure _

end Tail

macro "replay_tail_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact tail_pure _ | with_reducible exact tail_throw _
  | with_reducible exact tail_get
  | with_reducible exact tail_squeeze _ | with_reducible exact tail_message _ _
  | with_reducible exact tail_recorded _ | with_reducible exact tail_first _
  | with_reducible exact tail_pointBefore _
  | with_reducible exact tail_absorb _ | with_reducible exact tail_record _ _
  | with_reducible exact tail_value _ _ _ | with_reducible exact tail_draw _ _ _ _ _
  | with_reducible exact tail_terminalWith _ _ _ | with_reducible exact tail_checkWith _ _ _ _ _
  | with_reducible exact tail_mapM (fun _ => tail_recorded _) _
  | with_reducible exact tail_decodeExts _
  | (with_reducible refine tail_modify fun _ => ?_; first
      | exact RoundsTail.of_eq rfl
      | exact RoundsTail.push rfl (by simp [Op.ends]))
  | refine tail_bind ?_ fun _ => ?_
  | split))

/-- Each step of the replay appends to the rounds only the entry of the key of its `.ended`. -/
theorem step_tail (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (op : Op) : KeepsTail op.ends (step pub proof messages i op) := by
  cases op <;> simp only [step] <;> replay_tail_step

/-- A run of the operations `l` appends to the rounds only the entries of the keys of their
`.ended`. -/
theorem runFrom_tail (pub : Public) (proof : ByteArray) (messages : Array Message) :
    ∀ (l : List (Op × ℕ)) (s : Replayer),
      RoundsTail (l.flatMap fun x => x.1.ends) s (runFrom pub proof messages l s).2
  | [], s => RoundsTail.refl _ s
  | x :: l, s => by
      rw [runFrom_cons]
      have first := (step_tail pub proof messages x.2 x.1 s).mono
        (R' := (x :: l).flatMap fun x => x.1.ends)
        fun k inside => List.mem_flatMap.mpr ⟨x, List.mem_cons_self, inside⟩
      rcases found : ((step pub proof messages x.2 x.1).run).run s with ⟨_ | _, s₁⟩
      · rw [found] at first
        exact first
      · rw [found] at first
        exact first.trans ((runFrom_tail pub proof messages l s₁).mono fun k inside => by
          obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
          exact List.mem_flatMap.mpr ⟨y, List.mem_cons_of_mem _ member, found⟩)

/-- The first entry of a key outside `R` does not change. -/
theorem find_tail {R : List Key} {s s' : Replayer} (tail : RoundsTail R s s') {key : Key}
    (outside : key ∉ R) :
    s'.derived.rounds.find? (·.1 == key) = s.derived.rounds.find? (·.1 == key) := by
  obtain ⟨extra, grown, keys⟩ := tail
  rw [← Array.find?_toList, ← Array.find?_toList, grown, List.find?_append]
  have none : extra.find? (·.1 == key) = none := List.find?_eq_none.mpr fun x inside same =>
    outside (by rw [← beq_iff_eq.mp same]; exact keys x inside)
  rw [none, Option.or_none]

theorem roundsIn_tail {R : List Key} {s s' : Replayer} (tail : RoundsTail R s s') {key : Key}
    (outside : key ∉ R) : roundsIn s' key = roundsIn s key := by
  unfold roundsIn
  rw [find_tail tail outside]

/-- A found entry stays the first entry of its key. -/
theorem find_grows {s s' : Replayer} (grown : s.derived.rounds.toList <+: s'.derived.rounds.toList)
    {key : Key} {x : Key × List (List Ext × Ext)}
    (found : s.derived.rounds.find? (·.1 == key) = some x) :
    s'.derived.rounds.find? (·.1 == key) = some x := by
  obtain ⟨extra, grown⟩ := grown
  rw [← Array.find?_toList, ← grown, List.find?_append, Array.find?_toList, found]
  rfl

section Steps

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The recorded values of a key that no operation of `l` writes do not change in a run of `l`. -/
theorem recorded_free (l : List (Op × ℕ)) (s : Replayer) (key : Key)
    (free : ∀ y ∈ l, key ∉ y.1.writes) :
    recordedIn (runFrom pub proof messages l s).2 key = recordedIn s key :=
  ((runFrom_grows pub proof messages l s).same key fun inside => by
    obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
    exact free y member found).1

theorem fields_free (l : List (Op × ℕ)) (s : Replayer) (key : Key)
    (free : ∀ y ∈ l, key ∉ y.1.writes) :
    (runFrom pub proof messages l s).2.fields.get? key = s.fields.get? key :=
  ((runFrom_grows pub proof messages l s).same key fun inside => by
    obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
    exact free y member found).2.1

theorem integers_free (l : List (Op × ℕ)) (s : Replayer) (key : Key)
    (free : ∀ y ∈ l, key ∉ y.1.writes) :
    (runFrom pub proof messages l s).2.integers.get? key = s.integers.get? key :=
  ((runFrom_grows pub proof messages l s).same key fun inside => by
    obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
    exact free y member found).2.2.1

theorem roundsIn_free (l : List (Op × ℕ)) (s : Replayer) (key : Key)
    (free : ∀ y ∈ l, key ∉ y.1.ends) :
    roundsIn (runFrom pub proof messages l s).2 key = roundsIn s key :=
  roundsIn_tail (runFrom_tail pub proof messages l s) fun inside => by
    obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
    exact free y member found

theorem recordedIn_of_env {s s' : Replayer} {key : Key} {values : Array Ext}
    (env : s'.env = s.env.insert key (recordedIn s key ++ values)) :
    recordedIn s' key = recordedIn s key ++ values := by
  unfold recordedIn
  rw [env, Std.HashMap.get?_insert]
  simp [recordedIn]

theorem run_absorb (bytes : ByteArray) (s : Replayer) :
    ((absorb bytes).run).run s = (.ok (), { s with sponge := s.sponge.absorb bytes }) := rfl

theorem run_record (key : Key) (values : Array Ext) (s : Replayer) :
    ((record key values).run).run s =
      (.ok (), { s with env := s.env.insert key (recordedIn s key ++ values) }) := rfl

theorem run_modify (g : Replayer → Replayer) (s : Replayer) :
    ((modify g : ReplayM Unit).run).run s = (.ok (), g s) := rfl

theorem run_squeeze (count : ℕ) (s : Replayer) :
    ((squeeze count).run).run s =
      (.ok (s.sponge.squeeze count).1, { s with
        sponge := (s.sponge.squeeze count).2
        derived := { s.derived with squeezed := s.derived.squeezed + count } }) := rfl

theorem value_fused_run (key : Key) (F : Fold) (s : Replayer) :
    (((value pub key (.fused F)).run).run s).1 =
        .ok (extBytes #[(stage2In pub s key.level F).fused]) ∧
      (((value pub key (.fused F)).run).run s).2.env = s.env := by
  simp only [value]
  rw [run_bind, stage2Of_run]
  exact ⟨rfl, rfl⟩

/-- The absorb of the fused claim records `(stage2In …).fused` of the state before it, and it does
not stop. -/
theorem step_absorb_fused (i : ℕ) (key : Key) (F : Fold) (s : Replayer) :
    (((step pub proof messages i (.absorb key (.fused F))).run).run s).1 = .ok () ∧
      recordedIn (((step pub proof messages i (.absorb key (.fused F))).run).run s).2 key =
        recordedIn s key ++ #[(stage2In pub s key.level F).fused] := by
  have computed := value_fused_run pub key F s
  rcases found : ((value pub key (.fused F)).run).run s with ⟨result, s₁⟩
  rw [found] at computed
  obtain ⟨rfl, env⟩ := computed
  simp only [step]
  rw [run_bind, found]
  simp only [run_bind, run_absorb, decodeExts_extBytes, run_pure, run_record, run_modify]
  refine ⟨by trivial, ?_⟩
  rw [recordedIn_of_env (s := s₁) rfl]
  unfold recordedIn
  rw [env]

theorem foldlM_recorded_run (keys : List Key) (start : Array Ext) (s : Replayer) :
    ((keys.foldlM (fun out k => do return out ++ (← recorded k)) start).run).run s =
      (.ok (keys.foldl (fun out k => out ++ recordedIn s k) start), s) := by
  induction keys generalizing start with
  | nil => rfl
  | cons k keys ih =>
      rw [List.foldlM_cons, run_bind]
      exact ih _

/-- The absorb of recorded values records them: the values of the keys in order. -/
theorem step_absorb_recorded (i : ℕ) (key : Key) (keys : List Key) (s : Replayer) :
    (((step pub proof messages i (.absorb key (.recorded keys))).run).run s).1 = .ok () ∧
      recordedIn (((step pub proof messages i (.absorb key (.recorded keys))).run).run s).2 key =
        recordedIn s key ++ keys.foldl (fun out k => out ++ recordedIn s k) #[] := by
  simp only [step, value]
  rw [run_bind, run_bind, foldlM_recorded_run]
  simp only [run_bind, run_pure, run_absorb, decodeExts_extBytes_array, run_record, run_modify]
  exact ⟨by trivial, recordedIn_of_env rfl⟩

theorem decodePool_size {bytes : ByteArray} {len : ℕ} {values : Array Ext}
    (decoded : decodePool bytes len = some values) : values.size = len := by
  unfold decodePool at decoded
  dsimp only at decoded
  revert decoded
  generalize (forIn (m := Id) [0:bytes.size / 8] (#[] : Array Fq) _) = acc
  intro decoded
  change (if acc.size < 2 * len then none else
    some ((Array.range len).map fun i =>
      (⟨(Id.run acc)[2 * i]!, (Id.run acc)[2 * i + 1]!⟩ : Ext))) = some values
    at decoded
  split at decoded
  · cases decoded
  · cases decoded
    simp

/-- A squeeze of `count` challenges without a stop records `count` elements. -/
theorem step_challenge_ok (i : ℕ) (key : Key) (count : ℕ) (s : Replayer)
    (ran : (((step pub proof messages i (.challenge key count)).run).run s).1 = .ok ()) :
    (recordedIn (((step pub proof messages i (.challenge key count)).run).run s).2 key).size =
      (recordedIn s key).size + count := by
  simp only [step] at ran ⊢
  rw [run_bind, run_squeeze] at ran ⊢
  simp only at ran ⊢
  cases decoded : decodePool (s.sponge.squeeze (poolBytes (2 * count))).1 count with
  | none =>
      rw [decoded] at ran
      cases ran
  | some values =>
      simp only [run_bind, run_record, run_modify]
      rw [recordedIn_of_env rfl, Array.size_append, decodePool_size decoded]

theorem step_drawsInto (i : ℕ) (op : Op) (key : Key) (s : Replayer) {c : ℕ}
    (draws : op.drawsInto key = some c)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ()) :
    (recordedIn (((step pub proof messages i op).run).run s).2 key).size =
      (recordedIn s key).size + c := by
  unfold Op.drawsInto at draws
  split at draws
  · rename_i k count
    by_cases same : k = key
    · subst same
      rw [ite_eq_left rfl] at draws
      cases draws
      exact step_challenge_ok pub proof messages i _ _ s ran
    · rw [ite_eq_right same] at draws
      cases draws
      rw [((step_keeps pub proof messages i _ s).same key
        (by simpa [Op.writes] using fun h : key = k => same h.symm)).1]
      rfl
  · split at draws
    · cases draws
    · rename_i outside
      cases draws
      rw [((step_keeps pub proof messages i _ s).same key outside).1]
      rfl

/-- A run without a stop appends `drawnCount` elements to `key`. -/
theorem size_drawnCount (key : Key) : ∀ (l : List (Op × ℕ)) (s : Replayer) {n : ℕ},
    drawnCount key l = some n → (runFrom pub proof messages l s).1 = .ok () →
      (recordedIn (runFrom pub proof messages l s).2 key).size = (recordedIn s key).size + n
  | [], s, n, count, _ => by
      cases count
      rfl
  | x :: l, s, n, count, ran => by
      simp only [drawnCount, Option.bind_eq_some_iff, Option.map_eq_some_iff] at count
      obtain ⟨c, draws, m, rest, rfl⟩ := count
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped,
        size_drawnCount key l _ rest tail, step_drawsInto pub proof messages _ _ _ _ draws stepped]
      omega

/-- The end of a plain sum-check records the end claim of `plainRun` on its absorbed claim and
its recorded rounds, appends the rounds, and does not stop. -/
theorem step_ended_plain (i : ℕ) (key : Key) (n : ℕ) (s : Replayer) :
    (((step pub proof messages i (.ended key n .plain)).run).run s).1 = .ok () ∧
      recordedIn (((step pub proof messages i (.ended key n .plain)).run).run s).2
          ⟨key.level, key.site ++ ".output", none⟩ =
        recordedIn s ⟨key.level, key.site ++ ".output", none⟩ ++
          #[(plainRun (firstIn s ⟨key.level, key.site ++ ".claim", none⟩)
            (endRoundsIn s key n)).claim] ∧
      (((step pub proof messages i (.ended key n .plain)).run).run s).2.derived.rounds =
        s.derived.rounds.push (key, endRoundsIn s key n) := by
  rw [step_ended]
  exact ⟨rfl, recordedIn_of_env rfl, rfl⟩

/-- A check whose `holds` is its stage function: a step without a stop accepts. -/
theorem step_check_ok (i : ℕ) (key : Key) (c : Check) (s : Replayer)
    (only : ((holds pub c).run).run s =
      (.ok (((stageValue pub s c).map StageOut.accepts).getD false), s))
    (ran : (((step pub proof messages i (.check key c)).run).run s).1 = .ok ()) :
    ((stageValue pub s c).map StageOut.accepts).getD false = true := by
  rw [step_check, checkStep_run] at ran
  cases value : stageValue pub s c with
  | none =>
      rw [value] at ran only
      simp only [checkWith] at ran
      rw [run_bind, only] at ran
      cases ran
  | some out =>
      rw [value, checkWith_run] at ran
      cases accepts : out.accepts
      · rw [accepts] at ran
        cases ran
      · simp [accepts]

theorem holds_fused (level : ℕ) (F : Fold) (s : Replayer) :
    ((holds pub (.fused level F)).run).run s =
      (.ok (((stageValue pub s (.fused level F)).map StageOut.accepts).getD false), s) := rfl

theorem holds_stage3 (level : ℕ) (plan : SetupPlan) (s : Replayer) :
    ((holds pub (.stage3 level plan)).run).run s =
      (.ok (((stageValue pub s (.stage3 level plan)).map StageOut.accepts).getD false), s) := rfl

theorem holds_rangeCheckOf (level : ℕ) (F : Fold) (s : Replayer) :
    ((holds pub (rangeCheckOf level F)).run).run s =
      (.ok (((stageValue pub s (rangeCheckOf level F)).map StageOut.accepts).getD false), s) := by
  unfold rangeCheckOf
  cases F.euclidean <;> rfl

end Steps

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
