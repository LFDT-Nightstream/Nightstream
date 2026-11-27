import NightstreamFPrime.Spec.Compression.Akita.Replay.CausalTrace
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalJoin

/-! The replay of an Akita proof reads the answers of an ideal sponge only at the squeeze of a
work and of a draw (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23,
Section 8.2 `[KECCAK]` and Lemma 10.47; `Replay/Keccak.lean`, `Duplex.ideal`;
`docs/akita-work/w21/design.md`, section 5).

- `Replayer.reideal s ideal` is the state `s` with the answers `ideal` in its sponge.
  `Blind m`: the computation `m` gives the same result and the same state for each answers.
- `asks proof messages s op` is the query of the step of `op` on `s` when this step squeezes: the
  events of the sponge before the squeeze and the count of the bytes. A work with bits asks after
  the absorb of the bytes of its message, a draw asks on the events of `s`, and each other
  operation asks nothing.
- `step_reideal`: the step of an operation gives the same result and the same state for two
  answer functions that agree on its query.
- `step_asked`: after a step with no stop of a work with bits or of a draw, the events of the
  sponge are the squeeze of the query and the events of the query. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita

/-- The sponge with the answers `ideal`. -/
def Duplex.reideal (d : Duplex) (ideal : Option (List Duplex.Event → ℕ → ByteArray)) : Duplex :=
  { d with ideal }

theorem Duplex.absorb_reideal (d : Duplex) (ideal : Option (List Duplex.Event → ℕ → ByteArray))
    (input : ByteArray) : (d.reideal ideal).absorb input = (d.absorb input).reideal ideal := by
  by_cases empty : input.size = 0
  · simp only [Duplex.absorb, empty, ↓reduceIte]
    rfl
  · simp only [Duplex.absorb, empty, ↓reduceIte]
    rfl

/-- A squeeze of a sponge with other answers that agree at its events. -/
theorem Duplex.squeeze_reideal (d : Duplex) {a a' : List Duplex.Event → ℕ → ByteArray}
    (ideal : d.ideal = some a) (count : ℕ) (same : a d.events count = a' d.events count) :
    (d.reideal (some a')).squeeze count =
      ((d.squeeze count).1, (d.squeeze count).2.reideal (some a')) := by
  obtain ⟨lanes, absorbPos, squeezePos, events, answers⟩ := d
  cases ideal
  by_cases zero : count = 0
  · simp only [Duplex.squeeze, Duplex.reideal, zero, ↓reduceIte]
  · simp only [Duplex.squeeze, Duplex.reideal, zero, ↓reduceIte]
    exact Prod.ext same.symm rfl

namespace Replayer

/-- The state with the answers `ideal` in its sponge. -/
def reideal (s : Replayer) (ideal : Option (List Duplex.Event → ℕ → ByteArray)) : Replayer :=
  { s with sponge := s.sponge.reideal ideal }

/-- On the state `s`, the computation gives the same result and the same state with the answers
`ideal`. -/
def BlindAt {α : Type} (m : ReplayM α) (s : Replayer)
    (ideal : Option (List Duplex.Event → ℕ → ByteArray)) : Prop :=
  (m.run).run (reideal s ideal) = (((m.run).run s).1, reideal ((m.run).run s).2 ideal)

/-- The computation does not read the answers of the sponge. -/
def Blind {α : Type} (m : ReplayM α) : Prop := ∀ s ideal, BlindAt m s ideal

theorem run_throw {α : Type} (e : Stop) (s : Replayer) :
    ((throw e : ReplayM α).run).run s = (.error e, s) := rfl

section Blind

variable {ideal : Option (List Duplex.Event → ℕ → ByteArray)}

/-- A computation with the property on `s`, then computations that do not read the answers on
each state after it. -/
theorem bindAt {α β : Type} {m : ReplayM α} {f : α → ReplayM β} {s : Replayer}
    (hm : BlindAt m s ideal)
    (hf : ∀ a s₁, (m.run).run s = (.ok a, s₁) → BlindAt (f a) s₁ ideal) :
    BlindAt (m >>= f) s ideal := by
  unfold BlindAt at hm ⊢
  rw [run_bind m f (reideal s ideal), run_bind m f s, hm]
  rcases found : (m.run).run s with ⟨_ | a, s₁⟩
  · rfl
  · exact hf a s₁ found

theorem blind_bind {α β : Type} {m : ReplayM α} {f : α → ReplayM β} (hm : Blind m)
    (hf : ∀ a, Blind (f a)) : Blind (m >>= f) :=
  fun s ideal => bindAt (hm s ideal) fun a s₁ _ => hf a s₁ ideal

theorem blind_pure {α : Type} (a : α) : Blind (Pure.pure a : ReplayM α) := fun _ _ => rfl

theorem blind_throw {α : Type} (e : Stop) : Blind (throw e : ReplayM α) := fun _ _ => rfl

theorem blind_modify {g : Replayer → Replayer}
    (h : ∀ s ideal, g (reideal s ideal) = reideal (g s) ideal) :
    Blind (modify g : ReplayM Unit) :=
  fun s ideal => congrArg (Prod.mk (Except.ok () : Except Stop Unit)) (h s ideal)

/-- A value of the state that does not read the answers. -/
theorem blind_read {α : Type} (g : Replayer → α) (h : ∀ s ideal, g (reideal s ideal) = g s) :
    Blind (get >>= fun s => Pure.pure (g s) : ReplayM α) :=
  fun s ideal => congrArg (fun a => ((Except.ok a : Except Stop α), reideal s ideal)) (h s ideal)

/-- A computation on a value of the state that does not read the answers. -/
theorem blind_reads {α β : Type} (g : Replayer → β) (h : ∀ s ideal, g (reideal s ideal) = g s)
    {f : β → ReplayM α} (hf : ∀ b, Blind (f b)) : Blind (get >>= fun s => f (g s)) := by
  intro s ideal
  have at_ := hf (g s) s ideal
  unfold BlindAt at at_ ⊢
  show ((f (g (reideal s ideal))).run).run (reideal s ideal) = _
  rw [h]
  exact at_

theorem blind_mapM {α β : Type} {f : α → ReplayM β} (each : ∀ a, Blind (f a)) :
    ∀ l : List α, Blind (l.mapM f)
  | [] => blind_pure _
  | a :: l => by
      rw [List.mapM_cons]
      exact blind_bind (each a) fun _ => blind_bind (blind_mapM each l) fun _ => blind_pure _

theorem blind_foldlM {α β : Type} {f : β → α → ReplayM β} (each : ∀ b a, Blind (f b a)) :
    ∀ (l : List α) (b : β), Blind (l.foldlM f b)
  | [], _ => blind_pure _
  | a :: l, b => by
      rw [List.foldlM_cons]
      exact blind_bind (each b a) fun c => blind_foldlM each l c

theorem blind_recorded (key : Key) : Blind (recorded key) :=
  blind_read (fun s => recordedIn s key) fun _ _ => rfl

theorem blind_first (key : Key) : Blind (first key) :=
  blind_read (fun s => firstIn s key) fun _ _ => rfl

theorem blind_fieldsOf (key : Key) : Blind (fieldsOf key) :=
  blind_read (fun s => (s.fields.get? key).getD #[]) fun _ _ => rfl

theorem blind_pointBefore (before : Before) : Blind (pointBefore before) :=
  blind_read (fun s => pointIn s before) fun _ _ => rfl

theorem blind_absorb (bytes : ByteArray) : Blind (absorb bytes) :=
  blind_modify fun s ideal =>
    congrArg (fun d => ({ s with sponge := d } : Replayer))
      (Duplex.absorb_reideal s.sponge ideal bytes)

theorem blind_record (key : Key) (values : Array Ext) : Blind (record key values) :=
  blind_modify fun _ _ => rfl

theorem blind_message (i : ℕ) (messages : Array Message) : Blind (message i messages) := by
  intro s ideal
  have next : (reideal s ideal).next = s.next := rfl
  unfold BlindAt
  rw [run_message, run_message, next]
  cases messages[s.next]? <;> rfl

theorem blind_decodeExts (bytes : ByteArray) : Blind (step.decodeExts bytes) := by
  unfold step.decodeExts
  exact blind_pure _

end Blind

macro "replay_blind" : tactic => `(tactic| repeat' (first
  | with_reducible exact blind_pure _ | with_reducible exact blind_throw _
  | with_reducible exact blind_absorb _ | with_reducible exact blind_message _ _
  | with_reducible exact blind_recorded _ | with_reducible exact blind_first _
  | with_reducible exact blind_fieldsOf _ | with_reducible exact blind_pointBefore _
  | with_reducible exact blind_record _ _
  | with_reducible exact blind_mapM (fun _ => blind_recorded _) _
  | with_reducible exact blind_decodeExts _
  | ((with_reducible refine blind_modify fun _ _ => ?_) <;> rfl)
  | ((with_reducible refine blind_read _ fun _ _ => ?_) <;> rfl)
  | refine blind_bind ?_ fun _ => ?_
  | split))

section Steps

theorem blind_stage2Of (pub : Public) (level : ℕ) (F : Fold) : Blind (stage2Of pub level F) := by
  unfold stage2Of
  replay_blind

theorem blind_value (pub : Public) (key : Key) (v : Value) : Blind (value pub key v) := by
  cases v with
  | recorded keys =>
      exact blind_bind (blind_foldlM (fun _ _ => blind_bind (blind_recorded _) fun _ =>
        blind_pure _) keys _) fun _ => blind_pure _
  | fused F =>
      simp only [value]
      refine blind_bind (blind_stage2Of pub _ F) fun _ => ?_
      replay_blind
  | _ => simp only [value]; replay_blind

theorem blind_weightsOf (weights : Weights) : Blind (weightsOf weights) := by
  cases weights <;> simp only [weightsOf] <;> replay_blind

theorem blind_holds (pub : Public) (check : Check) : Blind (holds pub check) := by
  cases check <;> simp only [holds]
  · replay_blind
  · refine blind_bind (blind_first _) fun _ => blind_bind (blind_weightsOf _) fun _ =>
      blind_bind (blind_recorded _) fun _ => blind_pure _
  all_goals replay_blind

theorem blind_checkWith (pub : Public) (i : ℕ) (key : Key) (check : Check)
    (out : Option StageOut) : Blind (checkWith pub i key check out) := by
  cases out with
  | some out => simp only [checkWith]; replay_blind
  | none =>
      simp only [checkWith]
      refine blind_bind (blind_holds pub check) fun _ => ?_
      replay_blind

theorem blind_checkStep (pub : Public) (i : ℕ) (key : Key) (check : Check) :
    Blind (checkStep pub i key check) :=
  blind_reads (fun s => stageValue pub s check) (fun _ _ => by cases check <;> rfl)
    fun out => blind_checkWith pub i key check out

theorem blind_draw (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) :
    Blind (draw i level group G bytes) := by
  unfold draw
  replay_blind

theorem blind_terminalWith (i level : ℕ) (r : Option (Except String TerminalValues)) :
    Blind (terminalWith i level r) := by
  rcases r with _ | _ | _ <;> simp only [terminalWith] <;> replay_blind

end Steps

macro "replay_blind_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact blind_pure _ | with_reducible exact blind_throw _
  | with_reducible exact blind_absorb _ | with_reducible exact blind_message _ _
  | with_reducible exact blind_recorded _ | with_reducible exact blind_first _
  | with_reducible exact blind_pointBefore _ | with_reducible exact blind_record _ _
  | with_reducible exact blind_value _ _ _
  | with_reducible exact blind_mapM (fun _ => blind_recorded _) _
  | with_reducible exact blind_decodeExts _
  | ((with_reducible refine blind_modify fun _ _ => ?_) <;> rfl)
  | refine blind_bind ?_ fun _ => ?_
  | split))

section Asks

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The query of the step of an operation on a state, when this step squeezes: the events of the
sponge before the squeeze, and the count of the bytes. -/
def asks (s : Replayer) : Op → Option (List Duplex.Event × ℕ)
  | .work site level bits =>
      if bits = 0 then none
      else messages[s.next]?.map fun m =>
        (.absorb ((Op.work site level bits).input proof m) :: s.sponge.events, 8)
  | .challenge _ count => some (s.sponge.events, poolBytes (2 * count))
  | .squeeze _ _ G => some (s.sponge.events, G.messageBytes)
  | _ => none

theorem asks_reideal (s : Replayer) (ideal : Option (List Duplex.Event → ℕ → ByteArray))
    (op : Op) : asks proof messages (reideal s ideal) op = asks proof messages s op := by
  cases op <;> rfl

/-- An operation that is not a work and not a draw asks nothing. -/
theorem asks_quiet (s : Replayer) {op : Op} (work : isWork op = false)
    (draw : isDraw op = false) : asks proof messages s op = none := by
  cases op <;> first | rfl | exact Bool.noConfusion work | exact Bool.noConfusion draw

/-- A squeeze, then computations that do not read the answers, for two answer functions that
agree on the events of the state. -/
theorem squeeze_bindAt {β : Type} (count : ℕ) {f : ByteArray → ReplayM β}
    (hf : ∀ bytes, Blind (f bytes)) (s : Replayer) {a a' : List Duplex.Event → ℕ → ByteArray}
    (ideal : s.sponge.ideal = some a)
    (same : a s.sponge.events count = a' s.sponge.events count) :
    BlindAt (squeeze count >>= f) s (some a') := by
  refine bindAt ?_ fun bytes s₁ _ => hf bytes s₁ _
  unfold BlindAt
  rw [run_squeeze, run_squeeze]
  show (Except.ok ((s.sponge.reideal (some a')).squeeze count).1,
    ({ s with
      sponge := ((s.sponge.reideal (some a')).squeeze count).2
      derived := { s.derived with squeezed := s.derived.squeezed + count } } : Replayer)) = _
  rw [Duplex.squeeze_reideal s.sponge ideal count same]
  rfl

/-- The step of an operation gives the same result and the same state for two answer functions
that agree on the query of the step. -/
theorem step_reideal (i : ℕ) (op : Op) (s : Replayer)
    {a a' : List Duplex.Event → ℕ → ByteArray} (ideal : s.sponge.ideal = some a)
    (same : ∀ p, asks proof messages s op = some p → a p.1 p.2 = a' p.1 p.2) :
    BlindAt (step pub proof messages i op) s (some a') := by
  cases op with
  | work site level bits =>
      by_cases zero : bits = 0
      · subst zero
        refine (?_ : Blind _) s _
        simp only [step, ↓reduceIte]
        exact blind_pure _
      · simp only [step, zero, ↓reduceIte]
        refine bindAt (blind_message i messages s _) fun m s₁ ran => ?_
        rw [run_message] at ran
        cases found : messages[s.next]? with
        | none =>
            simp only [found] at ran
            cases ran
        | some m' =>
            simp only [found] at ran
            cases ran
            split
            · rename_i nonce canonical data
              have input : (Op.work site level bits).input proof m = le64 nonce := by
                simp only [Op.input, data]
              have agree := same
                (.absorb ((Op.work site level bits).input proof m) :: s.sponge.events, 8)
                (by simp only [asks, zero, ↓reduceIte, found, Option.map_some])
              rw [input] at agree
              split
              · refine bindAt (blind_absorb _ _ _) fun _ s₃ ran₃ => ?_
                rw [run_absorb] at ran₃
                cases ran₃
                refine squeeze_bindAt 8 (fun _ => by replay_blind_step) _
                  (a := a) ((Duplex.absorb_ideal _ _).trans ideal) ?_
                show a (s.sponge.absorb (le64 nonce)).events 8 =
                  a' (s.sponge.absorb (le64 nonce)).events 8
                rw [Duplex.absorb_events]
                exact agree
              · refine bindAt (blind_throw _ _ _) fun _ _ ran₂ => ?_
                rw [run_throw] at ran₂
                cases ran₂
            · exact blind_throw _ _ _
  | challenge key count =>
      simp only [step]
      exact squeeze_bindAt _ (fun _ => by replay_blind_step) s ideal (same _ rfl)
  | squeeze level group G =>
      simp only [step]
      exact squeeze_bindAt _ (fun bytes => blind_draw i level group G bytes) s ideal (same _ rfl)
  | check key check => exact blind_checkStep pub i key check s _
  | terminal level =>
      exact blind_reads (fun s => terminalValue pub s level) (fun _ _ => rfl)
        (fun r => blind_terminalWith i level r) s _
  | _ =>
      refine (?_ : Blind _) s _
      simp only [step]
      replay_blind_step

/-- After a step with no stop of a work with bits or of a draw, the events of the sponge are the
squeeze of the query of the step and the events of this query. -/
theorem step_asked (i : ℕ) (op : Op) (s : Replayer)
    (squeezes : (∃ site level bits, op = .work site level bits ∧ bits ≠ 0) ∨ isDraw op = true)
    (ran : (((step pub proof messages i op).run).run s).1 = .ok ()) :
    ∃ events count, asks proof messages s op = some (events, count) ∧
      (((step pub proof messages i op).run).run s).2.sponge.events =
        .squeeze count :: events := by
  rcases squeezes with ⟨site, level, bits, rfl, positive⟩ | draws
  · obtain ⟨m, found, sponge⟩ := step_work pub proof messages i site level bits s positive ran
    refine ⟨.absorb ((Op.work site level bits).input proof m) :: s.sponge.events, 8,
      by simp only [asks, positive, ↓reduceIte, found, Option.map_some], ?_⟩
    rw [sponge, Duplex.squeeze_events, Duplex.absorb_events]
  · have sponge := step_draw pub proof messages i op draws s
    cases op with
    | challenge key count =>
        exact ⟨_, _, rfl, by rw [sponge, Duplex.squeeze_events]; rfl⟩
    | squeeze level group G =>
        exact ⟨_, _, rfl, by rw [sponge, Duplex.squeeze_events]; rfl⟩
    | _ => cases draws

end Asks

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
