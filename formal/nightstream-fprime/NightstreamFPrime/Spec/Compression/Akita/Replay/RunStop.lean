import NightstreamFPrime.Spec.Compression.Akita.Replay.RunWhole
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalJoin

/-! The stops of the replay (Akita revision `d50f700`; `Replayer.step` and `replay` of
`Replay/Transcript.lean`).

- `Stop.op` is the operation that a stop names. `Throws P m`: each stop of `m` has the property
  `P`. `step_stopsAt`: each stop of the step of the operation `i` names `i`.
- `stateBefore` is the state of the replay before an operation. `runFrom_stop`: a run that stops
  has a first operation whose step stops, after a run of the operations before it without a stop.
  `replay_stop_at`: the stop of the replay is the stop of the step of the operation that it names,
  on the state before it.
- `step_mismatch`: a step stops with `mismatch` only at an operation that reads a message, at the
  end of a sum-check whose point before has not the count of its rounds (`sumcheckRun = none`),
  and at the terminal without the setup prefix (`terminalValue = none`).
- `endsFit` is a Bool on the operations of a script: at each end of a sum-check in the equality
  form, the challenges of the point before and of its own point have the counts that
  `sumcheckRun` takes. `replay_matches`: with the messages of `decode`, `endsFit` and the setup
  prefix of the terminal, the replay does not stop with `mismatch`. `Public.setupFor_of_le`: the
  setup prefix exists when `Public.setup` is long enough. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The operation that a stop names. -/
def Stop.op : Stop → ℕ
  | .noWork op _ | .reject op _ | .mismatch op => op

def Stop.isMismatch : Stop → Bool
  | .mismatch _ => true
  | _ => false

def Stop.isNoWork : Stop → Bool
  | .noWork _ _ => true
  | _ => false

def Stop.isReject : Stop → Bool
  | .reject _ _ => true
  | _ => false

/-- The operations whose step can stop with `noWork`: a work, a squeeze of challenges, a response
nonce and a draw of fold challenges. -/
def Op.grinds : Op → Bool
  | .work .. | .challenge .. | .foldNonce .. | .squeeze .. => true
  | _ => false

/-- The key of the point before a sum-check in the equality form. -/
def Before.key? : Before → Option Key
  | .plain => none
  | .point k | .turned k _ => some k

/-- At the end `op` of a sum-check in the equality form after the operations `pre`, the point
before gets `p` challenges and the point of the sum-check `q`, with `0 < p = min n q`. -/
def endFits (pre : List (Op × ℕ)) : Op → Bool
  | .ended key n before => match before.key? with
    | none => true
    | some k => match drawnCount k pre, drawnCount ⟨key.level, key.site ++ ".point", none⟩ pre with
      | some p, some q => 0 < p && p == min n q
      | _, _ => false
  | _ => true

/-- `endFits` at each operation of `l`. -/
def endsFit (l : List (Op × ℕ)) : Bool :=
  (List.range l.length).all fun j => (l[j]?.map fun x => endFits (l.take j) x.1).getD true

/-- With a setup prefix of at least `count` elements in `Public.setup`, `setupFor` gives its first
`count` elements: an instance of the hypothesis on the setup prefix of the terminal. -/
theorem Public.setupFor_of_le (pub : Public) {count : ℕ} (long : count ≤ pub.setup.size) :
    pub.setupFor count = some (pub.setup.extract 0 count) := by
  simp [Public.setupFor, long]

namespace Replayer

/-- Each stop of `m` has the property `P`. -/
def Throws {α : Type} (P : Stop → Prop) (m : ReplayM α) : Prop :=
  ∀ s stop, ((m.run).run s).1 = .error stop → P stop

section Throws

variable {P : Stop → Prop}

theorem throws_bind {α β : Type} {m : ReplayM α} {f : α → ReplayM β} (hm : Throws P m)
    (hf : ∀ a, Throws P (f a)) : Throws P (m >>= f) := by
  intro s stop stopped
  rw [run_bind] at stopped
  split at stopped
  · rename_i a s₁ found
    exact hf a s₁ stop stopped
  · rename_i e s₁ found
    have same : e = stop := by simpa using stopped
    subst same
    exact hm s e (by rw [found])

theorem throws_pure {α : Type} (a : α) : Throws P (Pure.pure a : ReplayM α) := by
  intro s stop stopped
  cases stopped

theorem throws_throw {α : Type} {e : Stop} (h : P e) : Throws P (throw e : ReplayM α) := by
  intro s stop stopped
  cases stopped
  exact h

theorem throws_get : Throws P (get : ReplayM Replayer) := by
  intro s stop stopped
  cases stopped

theorem throws_modify (g : Replayer → Replayer) : Throws P (modify g : ReplayM Unit) := by
  intro s stop stopped
  cases stopped

theorem throws_mapM {α β : Type} {f : α → ReplayM β} (each : ∀ a, Throws P (f a)) :
    ∀ l : List α, Throws P (l.mapM f)
  | [] => throws_pure _
  | a :: l => by
      rw [List.mapM_cons]
      exact throws_bind (each a) fun _ => throws_bind (throws_mapM each l) fun _ => throws_pure _

theorem throws_foldlM {α β : Type} {f : β → α → ReplayM β}
    (each : ∀ b a, Throws P (f b a)) : ∀ (l : List α) (b : β), Throws P (l.foldlM f b)
  | [], _ => throws_pure _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact throws_bind (each b a) fun c => throws_foldlM each l c

theorem throws_recorded (key : Key) : Throws P (recorded key) :=
  throws_bind throws_get fun _ => throws_pure _

theorem throws_first (key : Key) : Throws P (first key) :=
  throws_bind throws_get fun _ => throws_pure _

theorem throws_fieldsOf (key : Key) : Throws P (fieldsOf key) :=
  throws_bind throws_get fun _ => throws_pure _

theorem throws_pointBefore (before : Before) : Throws P (pointBefore before) :=
  throws_bind throws_get fun _ => throws_pure _

theorem throws_absorb (bytes : ByteArray) : Throws P (absorb bytes) := throws_modify _

theorem throws_record (key : Key) (values : Array Ext) : Throws P (record key values) :=
  throws_modify _

theorem throws_squeeze (count : ℕ) : Throws P (squeeze count) := by
  intro s stop stopped
  cases stopped

theorem throws_message {i : ℕ} (messages : Array Message) (h : P (.mismatch i)) :
    Throws P (message i messages) := by
  intro s stop stopped
  rw [run_message] at stopped
  split at stopped
  · have same : Stop.mismatch i = stop := by simpa using stopped
    subst same
    exact h
  · simp at stopped

macro "replay_throws" : tactic => `(tactic| repeat' (first
  | with_reducible exact throws_pure _ | ((with_reducible refine throws_throw ?_) <;> rfl)
  | with_reducible exact throws_get
  | with_reducible exact throws_squeeze _
  | ((with_reducible refine throws_message _ ?_) <;> rfl)
  | with_reducible exact throws_recorded _ | with_reducible exact throws_first _
  | with_reducible exact throws_fieldsOf _ | with_reducible exact throws_pointBefore _
  | with_reducible exact throws_absorb _ | with_reducible exact throws_record _ _
  | with_reducible exact throws_modify _
  | refine throws_bind ?_ fun _ => ?_
  | split))

theorem throws_stage2Of (pub : Public) (level : ℕ) (F : Fold) : Throws P (stage2Of pub level F) := by
  unfold stage2Of
  replay_throws

theorem throws_value (pub : Public) (key : Key) (v : Value) : Throws P (value pub key v) := by
  cases v with
  | recorded keys =>
      exact throws_bind (throws_foldlM (fun _ _ => throws_bind (throws_recorded _) fun _ =>
        throws_pure _) keys _) fun _ => throws_pure _
  | fused F =>
      simp only [value]
      refine throws_bind (throws_stage2Of pub _ F) fun _ => ?_
      replay_throws
  | _ => simp only [value]; replay_throws

theorem throws_weightsOf (weights : Weights) : Throws P (weightsOf weights) := by
  cases weights <;> simp only [weightsOf] <;> replay_throws

theorem throws_holds (pub : Public) (check : Check) : Throws P (holds pub check) := by
  cases check <;> simp only [holds]
  · replay_throws
  · refine throws_bind (throws_first _) fun _ => throws_bind (throws_weightsOf _) fun _ =>
      throws_bind (throws_recorded _) fun _ => throws_pure _
  all_goals replay_throws

theorem throws_checkWith (pub : Public) {i : ℕ} (key : Key) (check : Check)
    (out : Option StageOut) (h : P (.reject i key.site)) :
    Throws P (checkWith pub i key check out) := by
  cases out with
  | some out =>
      simp only [checkWith]
      refine throws_bind (throws_modify _) fun _ => ?_
      split
      · exact throws_modify _
      · exact throws_bind (throws_throw h) fun _ => throws_modify _
  | none =>
      simp only [checkWith]
      refine throws_bind (throws_holds pub check) fun b => ?_
      split
      · exact throws_modify _
      · exact throws_bind (throws_throw h) fun _ => throws_modify _

theorem throws_draw {i : ℕ} (level group : ℕ) (G : FoldGroup) (bytes : ByteArray)
    (h : ∀ site, P (.noWork i site)) : Throws P (draw i level group G bytes) := by
  unfold draw
  refine throws_bind (throws_modify _) fun _ => ?_
  split
  · exact throws_modify _
  · exact throws_throw (h _)

theorem throws_terminalWith {i : ℕ} (level : ℕ) (r : Option (Except String TerminalValues))
    (none_ : r = none → P (.mismatch i)) (h : ∀ site, P (.reject i site)) :
    Throws P (terminalWith i level r) := by
  rcases r with _ | _ | _ <;> simp only [terminalWith]
  · exact throws_throw (none_ rfl)
  · exact throws_throw (h _)
  · exact throws_modify _

theorem throws_decodeExts (bytes : ByteArray) : Throws P (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact throws_pure _

end Throws

macro "replay_throws_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact throws_pure _ | ((with_reducible refine throws_throw ?_) <;> rfl)
  | with_reducible exact throws_get
  | with_reducible exact throws_squeeze _
  | ((with_reducible refine throws_message _ ?_) <;> rfl)
  | with_reducible exact throws_recorded _ | with_reducible exact throws_first _
  | with_reducible exact throws_fieldsOf _ | with_reducible exact throws_pointBefore _
  | with_reducible exact throws_absorb _ | with_reducible exact throws_record _ _
  | with_reducible exact throws_modify _
  | with_reducible exact throws_value _ _ _ | with_reducible exact throws_decodeExts _
  | with_reducible exact throws_mapM (fun _ => throws_recorded _) _
  | ((with_reducible refine throws_draw _ _ _ _ ?_) <;> intro _ <;> rfl)
  | ((with_reducible refine throws_checkWith _ _ _ _ ?_) <;> rfl)
  | ((with_reducible refine throws_terminalWith _ _ ?_ ?_) <;> intro _ <;> rfl)
  | refine throws_bind ?_ fun _ => ?_
  | split))


section Steps

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- Each stop of the step of the operation `i` names `i`. -/
theorem step_stopsAt (i : ℕ) (op : Op) :
    Throws (fun e => e.op = i) (step pub proof messages i op) := by
  cases op <;> simp only [step] <;> replay_throws_step

/-- The state of the replay before the operation `c`. -/
def stateBefore (ops : Array Op) (c : ℕ) : Replayer :=
  (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).2

/-- A run that stops has a first operation whose step stops, after a run of the operations before
it without a stop. -/
theorem runFrom_stop : ∀ (l : List (Op × ℕ)) (s : Replayer) {stop : Stop},
    (runFrom pub proof messages l s).1 = .error stop →
      ∃ j, ∃ inside : j < l.length, (runFrom pub proof messages (l.take j) s).1 = .ok () ∧
        (((step pub proof messages l[j].2 l[j].1).run).run
          (runFrom pub proof messages (l.take j) s).2).1 = .error stop
  | [], s, stop, stopped => by
      rw [runFrom_nil] at stopped
      cases stopped
  | x :: l, s, stop, stopped => by
      rw [runFrom_cons] at stopped
      rcases found : ((step pub proof messages x.2 x.1).run).run s with ⟨_ | _, s₁⟩
      · rw [found] at stopped
        refine ⟨0, by simp, rfl, ?_⟩
        simp only [List.take_zero, runFrom_nil, List.getElem_cons_zero]
        rw [found]
        exact stopped
      · rw [found] at stopped
        obtain ⟨j, inside, before, at_⟩ := runFrom_stop l s₁ stopped
        refine ⟨j + 1, by simpa using inside, ?_, ?_⟩
        · rw [List.take_succ_cons, runFrom_cons, found]
          exact before
        · rw [List.take_succ_cons, runFrom_cons, found]
          simpa using at_

theorem zipIdx_getElem (ops : Array Op) {j : ℕ} (inside : j < ops.zipIdx.toList.length) :
    ops.zipIdx.toList[j] = (ops[j]'(by simpa using inside), j) := by
  simp

/-- The stop of the replay is the stop of the step of the operation that it names, on the state of
the replay before it. -/
theorem replay_stop_at (ops : Array Op) {stop : Stop}
    (stopped : (replay pub ops proof messages).1 = some stop) :
    ∃ op, ops[stop.op]? = some op ∧
      (runFrom pub proof messages (ops.zipIdx.toList.take stop.op) (channelStart pub)).1 =
        .ok () ∧
      (((step pub proof messages stop.op op).run).run
        (stateBefore pub proof messages ops stop.op)).1 = .error stop := by
  rw [replay_stop] at stopped
  have ran : (runFrom pub proof messages ops.zipIdx.toList (channelStart pub)).1 = .error stop := by
    revert stopped
    rcases (runFrom pub proof messages ops.zipIdx.toList (channelStart pub)).1 with e | ⟨⟩
    · intro same
      cases same
      rfl
    · intro same
      cases same
  obtain ⟨j, inside, before, at_⟩ := runFrom_stop pub proof messages _ _ ran
  rw [zipIdx_getElem ops inside] at at_
  have named : stop.op = j := step_stopsAt pub proof messages j _ _ _ at_
  subst named
  exact ⟨_, by simp, before, at_⟩

/-- A step stops with `mismatch` only at an operation that reads a message, at the end of a
sum-check where `sumcheckRun` is `none`, and at the terminal without the setup prefix. -/
theorem step_mismatch (i : ℕ) (op : Op) (s : Replayer) {k : ℕ}
    (stopped : (((step pub proof messages i op).run).run s).1 = .error (.mismatch k)) :
    op.reads = true ∨
      (∃ key n before, op = .ended key n before ∧
        sumcheckRun (formIn s before) (firstIn s ⟨key.level, key.site ++ ".claim", none⟩)
          (endRoundsIn s key n) = none) ∨
      ∃ level, op = .terminal level ∧ terminalValue pub s level = none := by
  have quiet : ∀ (m : ReplayM Unit), Throws (fun e => e.isMismatch = false) m →
      ((m.run).run s).1 ≠ .error (.mismatch k) := fun m h same => by
    have := h s _ same
    cases this
  cases op with
  | absorb key v =>
      refine absurd stopped (quiet _ ?_)
      simp only [step]
      replay_throws_step
  | challenge key count =>
      refine absurd stopped (quiet _ ?_)
      simp only [step]
      replay_throws_step
  | squeeze level group G =>
      refine absurd stopped (quiet _ ?_)
      simp only [step]
      replay_throws_step
  | check key check =>
      refine absurd stopped (quiet _ ?_)
      simp only [step]
      replay_throws_step
  | work site level bits =>
      by_cases zero : bits = 0
      · simp only [step, zero, ↓reduceIte] at stopped
        cases stopped
      · left
        simp [Op.reads, zero]
  | ended key n before =>
      rw [step_ended] at stopped
      split at stopped
      · rename_i found
        exact Or.inr (Or.inl ⟨key, n, before, rfl, found⟩)
      · cases stopped
  | terminal level =>
      rw [step_terminal] at stopped
      cases found : terminalValue pub s level with
      | none => exact Or.inr (Or.inr ⟨level, rfl, found⟩)
      | some r =>
          rw [found] at stopped
          rcases r with _ | _ <;> cases stopped
  | _ => left; rfl

/-- A step stops with `noWork` only at a work, a squeeze of challenges, a response nonce and a
draw of fold challenges. -/
theorem step_noWork (i : ℕ) (op : Op) (s : Replayer) {k : ℕ} {site : String}
    (stopped : (((step pub proof messages i op).run).run s).1 = .error (.noWork k site)) :
    op.grinds = true := by
  have quiet : ∀ (m : ReplayM Unit), Throws (fun e => e.isNoWork = false) m →
      ((m.run).run s).1 ≠ .error (.noWork k site) := fun m h same => by
    have := h s _ same
    cases this
  cases op <;> first
    | rfl
    | (refine absurd stopped (quiet _ ?_); simp only [step]; replay_throws_step)

/-- A step stops with `reject` only at a check of the name of the stop and at the terminal, where
`terminalValue` is the error of the stop. -/
theorem step_reject (i : ℕ) (op : Op) (s : Replayer) {k : ℕ} {site : String}
    (stopped : (((step pub proof messages i op).run).run s).1 = .error (.reject k site)) :
    (∃ key check, op = .check key check ∧ key.site = site) ∨
      ∃ level, op = .terminal level ∧ terminalValue pub s level = some (.error site) := by
  have quiet : ∀ (m : ReplayM Unit), Throws (fun e => e.isReject = false) m →
      ((m.run).run s).1 ≠ .error (.reject k site) := fun m h same => by
    have := h s _ same
    cases this
  cases op with
  | check key check =>
      have named : Throws (fun e => ∀ j t, e = .reject j t → t = key.site)
          (step pub proof messages i (.check key check)) := by
        simp only [step, checkStep]
        exact throws_bind throws_get fun _ => throws_checkWith pub _ _ _ (by
          intro j t same
          cases same
          rfl)
      exact Or.inl ⟨key, check, rfl, (named s _ stopped k site rfl).symm⟩
  | terminal level =>
      rw [step_terminal] at stopped
      cases found : terminalValue pub s level with
      | none =>
          rw [found] at stopped
          cases stopped
      | some r =>
          rw [found] at stopped
          rcases r with t | _
          · have same : t = site := by
              simp only [terminalWith] at stopped
              cases stopped
              rfl
            exact Or.inr ⟨level, rfl, by rw [found, same]⟩
          · cases stopped
  | _ => exact absurd stopped (quiet _ (by simp only [step]; replay_throws_step))

/-- At the end of a sum-check of the shape `endFits`, after a run without a stop from a state with
no recorded value, `sumcheckRun` is not `none`. -/
theorem ended_fits (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[]) (pre : List (Op × ℕ))
    (key : Key) (n : ℕ) (before : Before) (fits : endFits pre (.ended key n before) = true)
    (ok : (runFrom pub proof messages pre s₀).1 = .ok ()) :
    sumcheckRun (formIn (runFrom pub proof messages pre s₀).2 before)
        (firstIn (runFrom pub proof messages pre s₀).2 ⟨key.level, key.site ++ ".claim", none⟩)
        (endRoundsIn (runFrom pub proof messages pre s₀).2 key n) ≠ none := by
  set s := (runFrom pub proof messages pre s₀).2
  have sized : ∀ (k : Key) {p : ℕ}, drawnCount k pre = some p → (recordedIn s k).size = p := by
    intro k p count
    rw [size_drawnCount pub proof messages k pre s₀ count ok, fresh]
    simp
  have point : ∀ k, (pointIn s (.point k)).length = (recordedIn s k).size := by
    intro k
    simp [pointIn]
  have turned : ∀ k low, (pointIn s (.turned k low)).length = (recordedIn s k).size := by
    intro k low
    simp [pointIn]
  have rounds : ∀ {q : ℕ},
      drawnCount ⟨key.level, key.site ++ ".point", none⟩ pre = some q →
        (endRoundsIn s key n).length = min n q := by
    intro q count
    simp [endRoundsIn, sized _ count]
  intro none_
  obtain ⟨b, form, empty⟩ := (sumcheckRun_none _ _ _).mp none_
  have long : ∀ k, before.key? = some k → b.length = (recordedIn s k).size := by
    intro k found
    cases before with
    | plain => cases found
    | point k' =>
        cases found
        cases form
        exact point _
    | turned k' low =>
        cases found
        cases form
        exact turned _ _
  cases keyed : before.key? with
  | none =>
      cases before with
      | plain => cases form
      | point _ => cases keyed
      | turned _ _ => cases keyed
  | some k =>
      simp only [endFits, keyed] at fits
      split at fits
      · rename_i p q countP countQ
        simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at fits
        obtain ⟨positive, same⟩ := fits
        rw [long k keyed, sized k countP, rounds countQ] at empty
        rcases empty with empty | differ
        · rw [← List.length_eq_zero_iff, long k keyed, sized k countP] at empty
          omega
        · exact differ same
      · cases fits

/-- With the messages of `decode`, `endsFit` on the operations and the setup prefix of the
terminal, the replay does not stop with `mismatch`. -/
theorem replay_matches (ops : Array Op) {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) (ends : endsFit ops.zipIdx.toList = true)
    {setup : Array Fq} (expanded : pub.setupFor pub.layout.terminal.setupCount = some setup)
    (k : ℕ) : (replay pub ops proof messages).1 ≠ some (.mismatch k) := by
  intro stopped
  obtain ⟨op, at_, before, stepped⟩ := replay_stop_at pub proof messages ops stopped
  simp only [Stop.op] at at_ before stepped
  rcases step_mismatch pub proof messages k op _ stepped with reads | ⟨key, n, b, rfl, none_⟩ |
    ⟨level, rfl, none_⟩
  · exact replay_no_mismatch pub ops proof messages decoded at_ reads before stepped
  · obtain ⟨below, _⟩ := Array.getElem?_eq_some_iff.mp at_
    have fits := List.all_eq_true.mp ends k (List.mem_range.mpr (by simpa using below))
    have entry : ops.zipIdx.toList[k]? = some (.ended key n b, k) := by
      rw [List.getElem?_eq_getElem (by simpa using below), zipIdx_getElem ops (by simpa using below)]
      simp only [Option.some.injEq, Prod.mk.injEq, and_true]
      exact (Array.getElem?_eq_some_iff.mp at_).2
    rw [entry] at fits
    exact ended_fits pub proof messages _ (channelStart_fresh pub) _ key n b fits before none_
  · unfold terminalValue at none_
    dsimp only at none_
    rw [expanded] at none_
    simp at none_

end Steps

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
