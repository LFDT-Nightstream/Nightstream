import NightstreamFPrime.Spec.Compression.Akita.Replay.RunLevel

/-! The prover messages that the replay reads (Akita revision `d50f700`; `decode` of
`Replay/Wire.lean`, `Replayer.step` of `Replay/Transcript.lean`).

- `Op.reads`: the operation reads a prover message. `Op.Fits op m`: `m` has the data of the kind
  of `op`, with `count` elements for `.exts _ count`. `ReadsAligned messages l n`: the reading
  operations of `l` fit the messages from `n` in order. `decode_aligned`: the messages of `decode`
  are aligned with the operations of the script from `0`.
- `step_next`: a step without a stop reads one message exactly when its operation reads.
- `step_exts_ok`, `Op.extsInto`, `extsCount`, `size_extsCount`: a run without a stop of aligned
  operations appends to a key the counts of its `.exts` operations.
- `roundsShape` is the shape of the operations of one sum-check: its end at `e` with no earlier
  end of its key, `n` challenges of its point and `d` elements in each round message before `e`,
  and no later write of its point. `rounds_facts`: then the recorded rounds have `n` entries of
  `d` elements, and their challenges are the recorded point. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The operation reads a prover message. -/
def Op.reads : Op → Bool
  | .fields .. | .exts .. | .integer .. | .digest .. | .bounded .. | .foldNonce .. => true
  | .work _ _ bits => bits != 0
  | _ => false

/-- The message `m` has the data of the kind of the operation, with `count` elements of `E` for
`.exts _ count`. -/
def Op.Fits : Op → Message → Prop
  | .exts _ count, m => ∃ values, m.data = .exts values ∧ values.size = count
  | .fields _ _, m => ∃ values, m.data = .fields values
  | .integer _, m => ∃ value, m.data = .integer value
  | .digest _, m => ∃ value, m.data = .bytes value
  | .bounded _ _, m => ∃ value, m.data = .bytes value
  | .work _ _ _, m => ∃ value canonical, m.data = .nonce value canonical
  | .foldNonce _ _, m => ∃ value canonical, m.data = .nonce value canonical
  | _, _ => False

/-- The reading operations of `l` fit the messages from the index `n` in order. -/
def ReadsAligned (messages : Array Message) : List (Op × ℕ) → ℕ → Prop
  | [], _ => True
  | x :: l, n => if x.1.reads then
      (∃ m, messages[n]? = some m ∧ x.1.Fits m) ∧ ReadsAligned messages l (n + 1)
    else ReadsAligned messages l n

theorem exts_size {proof : ByteArray} {at_ count : ℕ} {values : Array Ext}
    (decoded : Wire.exts proof at_ count = some values) : values.size = count := by
  unfold Wire.exts at decoded
  cases found : Wire.fields proof at_ (2 * count) with
  | none => rw [found] at decoded; cases decoded
  | some words =>
      rw [found] at decoded
      cases decoded
      simp

theorem read_some {proof : ByteArray} {at_ : ℕ} {op : Op} {m : Message}
    (read : Wire.read proof at_ op = some (some m)) : op.reads = true ∧ op.Fits m := by
  cases op with
  | exts key count =>
      simp only [Wire.read, Option.map_eq_some_iff, Option.some.injEq] at read
      obtain ⟨values, decoded, rfl⟩ := read
      exact ⟨rfl, values, rfl, exts_size decoded⟩
  | fields key count =>
      simp only [Wire.read, Option.map_eq_some_iff, Option.some.injEq] at read
      obtain ⟨values, _, rfl⟩ := read
      exact ⟨rfl, values, rfl⟩
  | integer key =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · cases read
        exact ⟨rfl, _, rfl⟩
  | digest key =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · cases read
        exact ⟨rfl, _, rfl⟩
  | bounded key most =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · split at read
        · cases read
        · cases read
          exact ⟨rfl, _, rfl⟩
  | work site level bits =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · rename_i nonzero
        simp only [Option.map_eq_some_iff, Option.some.injEq] at read
        obtain ⟨⟨value, size, canonical⟩, _, rfl⟩ := read
        exact ⟨by simp [Op.reads, nonzero], value, canonical, rfl⟩
  | foldNonce level width =>
      simp only [Wire.read, Option.bind_eq_some_iff] at read
      obtain ⟨⟨value, size, canonical⟩, _, found⟩ := read
      split at found
      · cases found
        exact ⟨rfl, value, canonical, rfl⟩
      · cases found
  | _ => simp [Wire.read] at read

theorem read_none {proof : ByteArray} {at_ : ℕ} {op : Op}
    (read : Wire.read proof at_ op = some none) : op.reads = false := by
  cases op with
  | exts key count => simp [Wire.read] at read
  | fields key count => simp [Wire.read] at read
  | integer key => simp only [Wire.read] at read; split at read <;> cases read
  | digest key => simp only [Wire.read] at read; split at read <;> cases read
  | bounded key most =>
      simp only [Wire.read] at read
      split at read
      · cases read
      · split at read <;> cases read
  | work site level bits =>
      simp only [Wire.read] at read
      split at read
      · rename_i zero
        simp [Op.reads, zero]
      · simp at read
  | foldNonce level width =>
      simp only [Wire.read, Option.bind_eq_some_iff] at read
      obtain ⟨_, _, found⟩ := read
      split at found <;> cases found
  | _ => rfl

theorem decode_step_aligned (proof : ByteArray)
    (f : Array Message × ℕ → Op × ℕ → Except Malformed (Array Message × ℕ))
    (step : ∀ acc x, f acc x = match Wire.read proof acc.2 x.1 with
      | none => Except.error (⟨x.2, acc.2⟩ : Malformed)
      | some none => .ok (acc.1, acc.2)
      | some (some m) => .ok (acc.1.push m, acc.2 + m.size)) :
    ∀ (l : List (Op × ℕ)) (start : Array Message) (cursor : ℕ) (messages : Array Message)
      (last : ℕ), l.foldlM f (start, cursor) = .ok (messages, last) →
      (∀ j < start.size, messages[j]? = start[j]?) ∧ ReadsAligned messages l start.size
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
                decode_step_aligned proof f step l start cursor messages last folded
              refine ⟨same, ?_⟩
              simp only [ReadsAligned, read_none read]
              exact aligned
          | some m =>
              rw [read] at folded
              obtain ⟨same, aligned⟩ := decode_step_aligned proof f step l (start.push m)
                (cursor + m.size) messages last folded
              obtain ⟨reads, fits⟩ := read_some read
              refine ⟨fun j below => by
                rw [same j (by rw [Array.size_push]; omega), Array.getElem?_push_lt below]
                exact (Array.getElem?_eq_getElem below).symm, ?_⟩
              simp only [ReadsAligned, reads, ↓reduceIte]
              refine ⟨⟨m, ?_, fits⟩, by rw [Array.size_push] at aligned; exact aligned⟩
              rw [same _ (by rw [Array.size_push]; omega), Array.getElem?_push_size]

/-- The messages of `decode` are aligned with the operations of the script. -/
theorem decode_aligned {ops : Array Op} {proof : ByteArray} {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) :
    ReadsAligned messages ops.zipIdx.toList 0 := by
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
    exact (decode_step_aligned proof _ (fun acc x => by
      obtain ⟨_, _⟩ := acc
      obtain ⟨_, _⟩ := x
      rfl) _ _ _ _ _ found).2

/-- The count of elements that a step without a stop appends to `key`: `count` for a message of
`count` elements of `E` into `key`, `0` for an operation that does not write `key`, and `none`
otherwise. -/
def Op.extsInto (key : Key) : Op → Option ℕ
  | .exts k count => if k = key then some count else some 0
  | op => if key ∈ op.writes then none else some 0

/-- The count of elements that a run of `l` without a stop appends to `key`, when each operation
that writes `key` receives elements of `E`. -/
def extsCount (key : Key) : List (Op × ℕ) → Option ℕ
  | [] => some 0
  | x :: l => (x.1.extsInto key).bind fun c => (extsCount key l).map (c + ·)

/-- The end of the sum-check `key` with `n` rounds. -/
def isEndOf (key : Key) (n : ℕ) : Op → Bool
  | .ended k m _ => k == key && m == n
  | _ => false

/-- The shape of the operations `l` of the sum-check `key` with `n` rounds of `d` elements: its
end at `e`, no earlier end of `key`, `n` challenges of its point and `d` elements of each round
message before `e`, and no write of its point from `e` on. -/
def roundsShape (l : List (Op × ℕ)) (key : Key) (n d e : ℕ) : Bool :=
  (l[e]?.any fun x => isEndOf key n x.1) && frees (l.take e) [] [key] &&
    drawnCount ⟨key.level, key.site ++ ".point", none⟩ (l.take e) == some n &&
    (List.range n).all (fun r =>
      extsCount ⟨key.level, key.site ++ ".round", some r⟩ (l.take e) == some d) &&
    frees (l.drop e) [⟨key.level, key.site ++ ".point", none⟩] []

/-- `roundsShape` at the first end of the sum-check `key`. -/
def roundsAt (l : List (Op × ℕ)) (key : Key) (n d : ℕ) : Bool :=
  (l.findIdx? fun x => isEndOf key n x.1).any fun e => roundsShape l key n d e

theorem isEndOf_sound {key : Key} {n : ℕ} {op : Op} (h : isEndOf key n op = true) :
    ∃ before, op = .ended key n before := by
  unfold isEndOf at h
  split at h
  next k m before =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨before, rfl⟩
  next => cases h

theorem roundsAt_sound {l : List (Op × ℕ)} {key : Key} {n d : ℕ} (h : roundsAt l key n d = true) :
    ∃ e, roundsShape l key n d e = true := by
  obtain ⟨e, _, shaped⟩ := (Option.any_eq_true _ _).mp h
  exact ⟨e, shaped⟩

theorem aligned_take (messages : Array Message) :
    ∀ (l : List (Op × ℕ)) (j n : ℕ), ReadsAligned messages l n →
      ReadsAligned messages (l.take j) n
  | [], _, _, _ => by simp [ReadsAligned]
  | _ :: _, 0, _, _ => by simp [ReadsAligned]
  | x :: l, j + 1, n, aligned => by
      simp only [List.take_succ_cons, ReadsAligned] at aligned ⊢
      split
      · rename_i reads
        rw [ite_eq_left reads] at aligned
        exact ⟨aligned.1, aligned_take messages l j (n + 1) aligned.2⟩
      · rename_i quiet
        rw [ite_eq_right quiet] at aligned
        exact aligned_take messages l j n aligned

namespace Replayer

/-- `m` keeps the count of the messages that the replay has read. -/
def KeepsNext {α : Type} (m : ReplayM α) : Prop := ∀ s, ((m.run).run s).2.next = s.next

section Next

theorem next_bind {α β : Type} {m : ReplayM α} {f : α → ReplayM β} (hm : KeepsNext m)
    (hf : ∀ a, KeepsNext (f a)) : KeepsNext (m >>= f) := by
  intro s
  rw [run_bind]
  rcases found : (m.run).run s with ⟨_ | a, s₁⟩
  · have kept := hm s
    rw [found] at kept
    exact kept
  · have kept := hm s
    rw [found] at kept
    exact (hf a s₁).trans kept

theorem next_pure {α : Type} (a : α) : KeepsNext (Pure.pure a : ReplayM α) := fun _ => rfl

theorem next_throw {α : Type} (e : Stop) : KeepsNext (throw e : ReplayM α) := fun _ => rfl

theorem next_get : KeepsNext (get : ReplayM Replayer) := fun _ => rfl

theorem next_get_bind {α : Type} {f : Replayer → ReplayM α}
    (h : ∀ s, (((f s).run).run s).2.next = s.next) : KeepsNext (get >>= f) := fun s => h s

theorem next_modify {g : Replayer → Replayer} (h : ∀ s, (g s).next = s.next) :
    KeepsNext (modify g : ReplayM Unit) := h

theorem next_mapM {α β : Type} {f : α → ReplayM β} (each : ∀ a, KeepsNext (f a)) :
    ∀ l : List α, KeepsNext (l.mapM f)
  | [] => next_pure _
  | a :: l => by
      rw [List.mapM_cons]
      exact next_bind (each a) fun _ => next_bind (next_mapM each l) fun _ => next_pure _

theorem next_foldlM {α β : Type} {f : β → α → ReplayM β} (each : ∀ b a, KeepsNext (f b a)) :
    ∀ (l : List α) (b : β), KeepsNext (l.foldlM f b)
  | [], _ => next_pure _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact next_bind (each b a) fun c => next_foldlM each l c

theorem next_recorded (key : Key) : KeepsNext (recorded key) :=
  next_bind next_get fun _ => next_pure _

theorem next_first (key : Key) : KeepsNext (first key) := next_bind next_get fun _ => next_pure _

theorem next_fieldsOf (key : Key) : KeepsNext (fieldsOf key) :=
  next_bind next_get fun _ => next_pure _

theorem next_pointBefore (before : Before) : KeepsNext (pointBefore before) :=
  next_bind next_get fun _ => next_pure _

theorem next_absorb (bytes : ByteArray) : KeepsNext (absorb bytes) := next_modify fun _ => rfl

theorem next_record (key : Key) (values : Array Ext) : KeepsNext (record key values) :=
  next_modify fun _ => rfl

theorem next_squeeze (count : ℕ) : KeepsNext (squeeze count) := next_get_bind fun _ => rfl

macro "replay_next" : tactic => `(tactic| repeat' (first
  | with_reducible exact next_pure _ | with_reducible exact next_throw _
  | with_reducible exact next_get
  | with_reducible exact next_squeeze _ | with_reducible exact next_recorded _
  | with_reducible exact next_first _ | with_reducible exact next_fieldsOf _
  | with_reducible exact next_pointBefore _ | with_reducible exact next_absorb _
  | with_reducible exact next_record _ _
  | (with_reducible refine next_modify fun _ => ?_; rfl)
  | refine next_bind ?_ fun _ => ?_
  | split))

theorem next_stage2Of (pub : Public) (level : ℕ) (F : Fold) : KeepsNext (stage2Of pub level F) := by
  unfold stage2Of
  replay_next

theorem next_value (pub : Public) (key : Key) (v : Value) : KeepsNext (value pub key v) := by
  cases v with
  | recorded keys =>
      exact next_bind (next_foldlM (fun _ _ => next_bind (next_recorded _) fun _ =>
        next_pure _) keys _) fun _ => next_pure _
  | fused F =>
      simp only [value]
      refine next_bind (next_stage2Of pub _ F) fun _ => ?_
      replay_next
  | _ => simp only [value]; replay_next

theorem next_weightsOf (weights : Weights) : KeepsNext (weightsOf weights) := by
  cases weights <;> simp only [weightsOf] <;> replay_next

theorem next_holds (pub : Public) (check : Check) : KeepsNext (holds pub check) := by
  cases check <;> simp only [holds]
  · replay_next
  · refine next_bind (next_first _) fun _ => next_bind (next_weightsOf _) fun _ =>
      next_bind (next_recorded _) fun _ => next_pure _
  all_goals replay_next

theorem next_checkWith (pub : Public) (i : ℕ) (key : Key) (check : Check)
    (out : Option StageOut) : KeepsNext (checkWith pub i key check out) := by
  cases out with
  | some out => simp only [checkWith]; replay_next
  | none =>
      simp only [checkWith]
      refine next_bind (next_holds pub check) fun _ => ?_
      replay_next

theorem next_draw (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) :
    KeepsNext (draw i level group G bytes) := by
  unfold draw
  replay_next

theorem next_terminalWith (i level : ℕ) (r : Option (Except String TerminalValues)) :
    KeepsNext (terminalWith i level r) := by
  rcases r with _ | _ | _ <;> simp only [terminalWith] <;> replay_next

theorem next_decodeExts (bytes : ByteArray) : KeepsNext (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact next_pure _

end Next

macro "replay_next_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact next_pure _ | with_reducible exact next_throw _
  | with_reducible exact next_get
  | with_reducible exact next_squeeze _ | with_reducible exact next_recorded _
  | with_reducible exact next_first _ | with_reducible exact next_pointBefore _
  | with_reducible exact next_absorb _ | with_reducible exact next_record _ _
  | with_reducible exact next_value _ _ _ | with_reducible exact next_draw _ _ _ _ _
  | with_reducible exact next_terminalWith _ _ _ | with_reducible exact next_checkWith _ _ _ _ _
  | with_reducible exact next_mapM (fun _ => next_recorded _) _
  | with_reducible exact next_decodeExts _
  | (with_reducible refine next_modify fun _ => ?_; rfl)
  | refine next_bind ?_ fun _ => ?_
  | split))

section Reads

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

theorem get_bind_run {α : Type} (f : Replayer → ReplayM α) (s : Replayer) :
    ((get >>= f : ReplayM α).run).run s = ((f s).run).run s := rfl

theorem run_message (i : ℕ) (s : Replayer) :
    ((message i messages).run).run s = match messages[s.next]? with
      | none => (.error (.mismatch i), s)
      | some m => (.ok m, { s with next := s.next + 1 }) := by
  cases found : messages[s.next]? <;> simp only [message, get_bind_run, found] <;> rfl

theorem next_message_bind (i : ℕ) {f : Message → ReplayM Unit}
    (rest : ∀ m, KeepsNext (f m)) (s : Replayer)
    (ran : (((message i messages >>= f).run).run s).1 = .ok ()) :
    (((message i messages >>= f).run).run s).2.next = s.next + 1 := by
  rw [run_bind, run_message] at ran ⊢
  cases found : messages[s.next]? with
  | none => rw [found] at ran; cases ran
  | some m => exact rest m _

theorem step_quiet (i : ℕ) (op : Op) (quiet : op.reads = false) :
    KeepsNext (step pub proof messages i op) := by
  cases op <;> simp [Op.reads] at quiet
  case work site level bits =>
    subst quiet
    intro s
    rfl
  all_goals simp only [step]; replay_next_step

/-- A step without a stop reads one message exactly when its operation reads. -/
theorem step_next (i : ℕ) (op : Op) (s : Replayer)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ()) :
    (((step pub proof messages i op).run).run s).2.next = s.next + if op.reads then 1 else 0 := by
  cases reads : op.reads
  · rw [step_quiet pub proof messages i op reads s]
    rfl
  · simp only [ite_true]
    cases op <;> simp [Op.reads] at reads
    case work site level bits =>
      simp only [step, reads, ↓reduceIte] at ran ⊢
      exact next_message_bind messages i (fun _ => by replay_next_step) s ran
    all_goals
      simp only [step] at ran ⊢
      exact next_message_bind messages i (fun _ => by replay_next_step) s ran

/-- A message of elements of `E` without a stop: the message at `next` has these elements, and the
replay records them. -/
theorem step_exts_ok (i : ℕ) (key : Key) (count : ℕ) (s : Replayer)
    (ran : (((step pub proof messages i (.exts key count)).run).run s).1 = .ok ()) :
    ∃ m values, messages[s.next]? = some m ∧ m.data = .exts values ∧
      recordedIn (((step pub proof messages i (.exts key count)).run).run s).2 key =
        recordedIn s key ++ values := by
  simp only [step] at ran ⊢
  rw [run_bind, run_message] at ran ⊢
  cases found : messages[s.next]? with
  | none => rw [found] at ran; cases ran
  | some m =>
      rw [found] at ran
      simp only [run_bind, run_absorb] at ran ⊢
      cases data : m.data with
      | exts values =>
          rw [data] at ran
          simp only [run_record] at ran ⊢
          exact ⟨m, values, rfl, data, recordedIn_of_env rfl⟩
      | _ => rw [data] at ran; cases ran

theorem step_extsInto (i : ℕ) (op : Op) (key : Key) (s : Replayer) {c : ℕ}
    (into : op.extsInto key = some c)
    (fits : op.reads = true → ∃ m, messages[s.next]? = some m ∧ op.Fits m)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ()) :
    (recordedIn (((step pub proof messages i op).run).run s).2 key).size =
      (recordedIn s key).size + c := by
  unfold Op.extsInto at into
  split at into
  · rename_i k count
    by_cases same : k = key
    · subst same
      rw [ite_eq_left rfl] at into
      cases into
      obtain ⟨m, values, found, data, recorded⟩ := step_exts_ok pub proof messages i k _ s ran
      obtain ⟨m', found', values', data', size⟩ := fits rfl
      rw [found] at found'
      cases found'
      rw [data] at data'
      cases data'
      rw [recorded, Array.size_append, size]
    · rw [ite_eq_right same] at into
      cases into
      rw [((step_keeps pub proof messages i _ s).same key
        (by simpa [Op.writes] using fun h : key = k => same h.symm)).1]
      rfl
  · split at into
    · cases into
    · rename_i outside
      cases into
      rw [((step_keeps pub proof messages i _ s).same key outside).1]
      rfl

/-- A run of aligned operations without a stop appends `extsCount` elements to `key`. -/
theorem size_extsCount (key : Key) : ∀ (l : List (Op × ℕ)) (s : Replayer) {n : ℕ},
    extsCount key l = some n → ReadsAligned messages l s.next →
      (runFrom pub proof messages l s).1 = .ok () →
        (recordedIn (runFrom pub proof messages l s).2 key).size = (recordedIn s key).size + n
  | [], s, n, count, _, _ => by
      cases count
      rfl
  | x :: l, s, n, count, aligned, ran => by
      simp only [extsCount, Option.bind_eq_some_iff, Option.map_eq_some_iff] at count
      obtain ⟨c, into, m, rest, rfl⟩ := count
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      have next := step_next pub proof messages x.2 x.1 s stepped
      have later : ReadsAligned messages l
          (((step pub proof messages x.2 x.1).run).run s).2.next := by
        simp only [ReadsAligned] at aligned
        rw [next]
        split at aligned
        · rename_i reads
          rw [ite_eq_left reads]
          exact aligned.2
        · rename_i quiet
          rw [ite_eq_right quiet]
          exact aligned
      have fits : x.1.reads = true → ∃ m, messages[s.next]? = some m ∧ x.1.Fits m := by
        intro reads
        simp only [ReadsAligned, reads, ↓reduceIte] at aligned
        exact aligned.1
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped,
        size_extsCount key l _ rest later tail,
        step_extsInto pub proof messages _ _ _ _ into fits stepped]
      omega

/-- A reading operation with the message at `next` of its kind does not stop with `mismatch`. -/
theorem step_fits_no_mismatch (i : ℕ) (op : Op) (s : Replayer)
    (fits : ∃ m, messages[s.next]? = some m ∧ op.Fits m) :
    (((step pub proof messages i op).run).run s).1 ≠ .error (.mismatch i) := by
  obtain ⟨m, found, fits⟩ := fits
  intro stopped
  cases op with
  | exts key count =>
      obtain ⟨values, data, _⟩ := fits
      simp only [step, run_bind, run_message, found, run_absorb, data, run_record] at stopped
      cases stopped
  | fields key count =>
      obtain ⟨values, data⟩ := fits
      simp only [step, run_bind, run_message, found, run_absorb, data, run_modify] at stopped
      cases stopped
  | integer key =>
      obtain ⟨value, data⟩ := fits
      simp only [step, run_bind, run_message, found, run_absorb, data, run_record,
        run_modify] at stopped
      cases stopped
  | digest key =>
      obtain ⟨value, data⟩ := fits
      simp only [step, run_bind, run_message, found, run_absorb, data, run_modify] at stopped
      cases stopped
  | bounded key most =>
      obtain ⟨value, data⟩ := fits
      simp only [step, run_bind, run_message, found, run_absorb, data, run_modify] at stopped
      cases stopped
  | work site level bits =>
      obtain ⟨value, canonical, data⟩ := fits
      by_cases zero : bits = 0
      · simp only [step, zero, ↓reduceIte] at stopped
        cases stopped
      · simp only [step, zero, ↓reduceIte, run_bind, run_message, found, data] at stopped
        split at stopped
        · simp only [run_bind, run_absorb, run_squeeze, run_modify] at stopped
          split at stopped <;> cases stopped
        · cases stopped
  | foldNonce level width =>
      obtain ⟨value, canonical, data⟩ := fits
      simp only [step, run_bind, run_message, found, data] at stopped
      split at stopped
      · simp only [run_absorb] at stopped
        cases stopped
      · cases stopped
  | _ => exact fits

/-- A run without a stop reads one message for each reading operation. -/
theorem runFrom_next : ∀ (l : List (Op × ℕ)) (s : Replayer),
    (runFrom pub proof messages l s).1 = .ok () →
      (runFrom pub proof messages l s).2.next = s.next + (l.filter (·.1.reads)).length
  | [], _, _ => rfl
  | x :: l, s, ran => by
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages _ _ _).mp ran
      rw [runFrom_cons_snd pub proof messages _ _ _ stepped, runFrom_next l _ tail,
        step_next pub proof messages _ _ _ stepped, List.filter_cons]
      split
      · simp_all
        omega
      · simp_all

/-- In aligned operations, the message at `n` plus the count of the reading operations before a
reading operation fits it. -/
theorem aligned_at : ∀ (l : List (Op × ℕ)) (n j : ℕ) (inside : j < l.length),
    ReadsAligned messages l n → l[j].1.reads = true →
      ∃ m, messages[n + ((l.take j).filter (·.1.reads)).length]? = some m ∧ l[j].1.Fits m
  | x :: l, n, 0, _, aligned, reads => by
      simp only [ReadsAligned, List.getElem_cons_zero] at aligned reads ⊢
      rw [reads, ite_eq_left rfl] at aligned
      simpa using aligned.1
  | x :: l, n, j + 1, inside, aligned, reads => by
      simp only [ReadsAligned] at aligned
      simp only [List.getElem_cons_succ] at reads ⊢
      simp only [List.take_succ_cons, List.filter_cons]
      split at aligned
      · rename_i first
        obtain ⟨m, found, fits⟩ := aligned_at l (n + 1) j (by simpa using inside) aligned.2 reads
        refine ⟨m, ?_, fits⟩
        rw [ite_eq_left first, List.length_cons, ← found]
        congr 1
        omega
      · rename_i first
        obtain ⟨m, found, fits⟩ := aligned_at l n j (by simpa using inside) aligned reads
        rw [ite_eq_right first]
        exact ⟨m, found, fits⟩

/-- The end of a sum-check without a stop appends its rounds to the rounds. -/
theorem step_ended_ok (i : ℕ) (key : Key) (n : ℕ) (before : Before) (s : Replayer)
    (ran : (((step pub proof messages i (.ended key n before)).run).run s).1 = .ok ()) :
    (((step pub proof messages i (.ended key n before)).run).run s).2.derived.rounds =
      s.derived.rounds.push (key, endRoundsIn s key n) := by
  rw [step_ended] at ran ⊢
  cases found : sumcheckRun (formIn s before) (firstIn s ⟨key.level, key.site ++ ".claim", none⟩)
      (endRoundsIn s key n) with
  | none => rw [found] at ran; cases ran
  | some e => rfl

/-- The rounds of a sum-check of the shape `roundsShape` after a run of aligned operations
without a stop, from a state with no recorded value and no rounds: `n` rounds of `d` elements,
whose challenges are the recorded point. -/
theorem rounds_facts (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (l : List (Op × ℕ)) (key : Key) (n d e : ℕ)
    (shape : roundsShape l key n d e = true) (aligned : ReadsAligned messages l s₀.next)
    (ok : (runFrom pub proof messages l s₀).1 = .ok ()) :
    (roundsIn (runFrom pub proof messages l s₀).2 key).length = n ∧
      (∀ x ∈ roundsIn (runFrom pub proof messages l s₀).2 key, x.1.length = d) ∧
      (roundsIn (runFrom pub proof messages l s₀).2 key).map (·.2) =
        (recordedIn (runFrom pub proof messages l s₀).2
          ⟨key.level, key.site ++ ".point", none⟩).toList := by
  simp only [roundsShape, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at shape
  obtain ⟨⟨⟨⟨atE, noEnd⟩, points⟩, roundCounts⟩, pointAfter⟩ := shape
  obtain ⟨eIn, eRec⟩ := any_at atE
  obtain ⟨before, opE⟩ := isEndOf_sound eRec
  obtain ⟨okE, stepE, lastE, nextE⟩ := runFrom_at pub proof messages l e eIn s₀ ok
  rw [opE] at stepE nextE
  have alignedE := aligned_take messages l e s₀.next aligned
  have rounds : roundsIn (runFrom pub proof messages l s₀).2 key =
      endRoundsIn (runFrom pub proof messages (l.take e) s₀).2 key n := by
    have missing : (runFrom pub proof messages (l.take e) s₀).2.derived.rounds.find?
        (·.1 == key) = none := by
      rw [find_tail (runFrom_tail pub proof messages _ s₀) fun inside => by
        obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
        exact frees_ends noEnd List.mem_cons_self y member found, empty]
      rfl
    have found : (((step pub proof messages l[e].2 (.ended key n before)).run).run
        (runFrom pub proof messages (l.take e) s₀).2).2.derived.rounds.find? (·.1 == key) =
          some (key, endRoundsIn (runFrom pub proof messages (l.take e) s₀).2 key n) := by
      rw [step_ended_ok pub proof messages _ _ _ _ _ stepE, Array.find?_push, missing]
      simp
    unfold roundsIn
    rw [nextE, find_grows (runFrom_grows pub proof messages _ _).rounds found]
    rfl
  have pointSize : (recordedIn (runFrom pub proof messages (l.take e) s₀).2
      ⟨key.level, key.site ++ ".point", none⟩).size = n := by
    rw [size_drawnCount pub proof messages _ _ _ points okE, fresh]
    simp
  have roundSize : ∀ r < n, (recordedIn (runFrom pub proof messages (l.take e) s₀).2
      ⟨key.level, key.site ++ ".round", some r⟩).size = d := fun r below => by
    rw [size_extsCount pub proof messages _ _ _ (roundCounts r (List.mem_range.mpr below))
      alignedE okE, fresh]
    simp
  have point : recordedIn (runFrom pub proof messages l s₀).2
      ⟨key.level, key.site ++ ".point", none⟩ = recordedIn (runFrom pub proof messages (l.take e)
        s₀).2 ⟨key.level, key.site ++ ".point", none⟩ := by
    rw [lastE]
    exact recorded_free pub proof messages _ _ _ (frees_writes pointAfter List.mem_cons_self)
  refine ⟨?_, ?_, ?_⟩
  · rw [rounds]
    unfold endRoundsIn
    simp [pointSize]
  · intro x member
    rw [rounds] at member
    unfold endRoundsIn at member
    obtain ⟨r, inside, same⟩ := List.mem_map.mp (List.of_mem_zip member).1
    rw [← same, Array.length_toList, roundSize r (List.mem_range.mp inside)]
  · rw [rounds, point]
    exact List.map_snd_zip (by simp [pointSize])

end Reads

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
