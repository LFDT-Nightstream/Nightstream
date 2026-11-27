import NightstreamFPrime.Spec.Compression.Akita.Replay.VerifierRun

/-! The address of a sponge key of `key20.aks` reads the answers of an ideal sponge only at the
squeezes inside the key (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23,
Lemma 10.14, Appendix E.1 (the stage input `T^J(k)` reads the table at the keys of earlier
stages) and Lemma 10.47; `docs/akita-work/w21/design.md`, section 6; `Replay/Resolve.lean`,
`Replay/VerifierRunTwin.lean`, `Replay/Simulate.lean`).

A page `stagePage20 … j` has values that the code computes from earlier challenges and does not
absorb (the weights of the claim of the page 1 are `row_coefficients`). So the pages are not a
function of the events only: they are a function of the events and of the answers at the
squeezes inside the events.

- `AgreeOn events a a'`: the answer functions `a` and `a'` agree at each squeeze of `events`.
  `agreeOn_answerOf`: two tables with the same answers at the base of each squeeze of the events
  give answer functions that agree on the events.
- `Public.reideal pub ideal` is the public input with the answers `ideal` in its sponge.
- `runFrom_reideal`: a run with no stop gives the same result and the same state for two answer
  functions that agree on the events of its last state. `stateBefore_reideal`,
  `eventsAt20_reideal`, `pageAt20_reideal`, `stagePage20_reideal`, `addressAt20_reideal`,
  `stage20_reideal`: the states, the events, the pages and the address of a stage.
- `addressAt20_of_events` (events give pages): two runs with the same events before the work of
  a stage, with answer functions that agree on these events, have the same address.
  `eventsAt20_of_address` (pages give events): two runs with the same address have the same
  events, when the answer functions agree on the events of one of the two runs.
- `resolve20_reideal`: `resolve20` reads the answers only at the squeezes of its events.
- `resolveWith20 total pub answer` is `resolve20` with the answers `answer`, with
  `resolveWith20_agree`, `resolveWith20_back` and `resolveWith20_eq`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Address)
open Sponge (split answerOf spongeOf)
open Replayer

section Agree

/-- The answer functions `a` and `a'` agree at each squeeze of `events`. -/
def AgreeOn (events : List Duplex.Event) (a a' : List Duplex.Event → ℕ → ByteArray) : Prop :=
  ∀ before count, (Duplex.Event.squeeze count :: before) <:+ events →
    a before count = a' before count

variable {events early : List Duplex.Event} {a a' : List Duplex.Event → ℕ → ByteArray}

theorem AgreeOn.refl (events : List Duplex.Event) (a : List Duplex.Event → ℕ → ByteArray) :
    AgreeOn events a a := fun _ _ _ => rfl

theorem AgreeOn.symm (agree : AgreeOn events a a') : AgreeOn events a' a :=
  fun before count inside => (agree before count inside).symm

theorem AgreeOn.suffix (suffix : early <:+ events) (agree : AgreeOn events a a') :
    AgreeOn early a a' :=
  fun before count inside => agree before count (inside.trans suffix)

/-- Two tables with the same answers at the base of each squeeze of the events give answer
functions that agree on the events. -/
theorem agreeOn_answerOf {L B C S : ℕ} {fs fs' : Sponge.Base L B C → Sponge.Y S C}
    (same : ∀ before count (base : Sponge.Base L B C) (sub : Sponge.Sub S C),
      (Duplex.Event.squeeze count :: before) <:+ events →
      split before count = some (base, sub) → fs base = fs' base) :
    AgreeOn events (answerOf fs) (answerOf fs') := by
  intro before count inside
  cases found : (split before count : Option (Sponge.Base L B C × Sponge.Sub S C)) with
  | none => rw [Sponge.answerOf_none fs found, Sponge.answerOf_none fs' found]
  | some pair =>
      obtain ⟨base, sub⟩ := pair
      rw [Sponge.answerOf_some fs found, Sponge.answerOf_some fs' found,
        same before count base sub inside found]

end Agree

/-- The public input with the answers `ideal` in its sponge. -/
abbrev Public.reideal (pub : Public) (ideal : Option (List Duplex.Event → ℕ → ByteArray)) :
    Public :=
  pub.withSponge (pub.sponge.reideal ideal)

theorem Public.reideal_self {pub : Public} {a : List Duplex.Event → ℕ → ByteArray}
    (ideal : pub.sponge.ideal = some a) : pub.reideal (some a) = pub := by
  have sponge : pub.sponge.reideal (some a) = pub.sponge := by
    rw [← ideal]
    rfl
  show ({ pub with sponge := pub.sponge.reideal (some a) } : Public) = pub
  rw [sponge]

/-- The sponge of the code with the answers of a table is the empty sponge with these answers. -/
theorem Public.withSponge_spongeOf {L B C S : ℕ} (pub : Public)
    (fs : Sponge.Base L B C → Sponge.Y S C) :
    pub.withSponge (spongeOf fs) = (pub.withSponge Duplex.empty).reideal (some (answerOf fs)) :=
  rfl

namespace Replayer

theorem core_reideal (s : Replayer) (ideal : Option (List Duplex.Event → ℕ → ByteArray)) :
    core (reideal s ideal) = core s := rfl

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- An operation that is not a work with bits and not a draw asks nothing. -/
theorem asks_none (s : Replayer) {op : Op}
    (quiet : ¬((∃ site level bits, op = .work site level bits ∧ bits ≠ 0) ∨ isDraw op = true)) :
    asks proof messages s op = none := by
  cases op with
  | work site level bits =>
      have zero : bits = 0 := by
        by_contra positive
        exact quiet (Or.inl ⟨site, level, bits, rfl, positive⟩)
      simp only [asks, zero, ↓reduceIte]
  | challenge key count => exact absurd (Or.inr rfl) quiet
  | squeeze level group G => exact absurd (Or.inr rfl) quiet
  | _ => rfl

/-- A run with no stop gives the same result and the same state for two answer functions that
agree at each squeeze of the events of its last state. -/
theorem runFrom_reideal : ∀ (l : List (Op × ℕ)) (s : Replayer)
    {a a' : List Duplex.Event → ℕ → ByteArray}, s.sponge.ideal = some a →
    (runFrom pub proof messages l s).1 = .ok () →
    AgreeOn (runFrom pub proof messages l s).2.sponge.events a a' →
    runFrom pub proof messages l (reideal s (some a')) =
      (.ok (), reideal (runFrom pub proof messages l s).2 (some a'))
  | [], _, _, _, _, _, _ => rfl
  | x :: l, s, a, a', ideal, ok, agree => by
      obtain ⟨stepped, tail⟩ := (runFrom_cons_fst pub proof messages x l s).mp ok
      rw [runFrom_cons_snd pub proof messages x l s stepped] at agree ⊢
      obtain ⟨grown, later⟩ := runFrom_suffix pub proof messages l
        (((step pub proof messages x.2 x.1).run).run s).2
      have blind : BlindAt (step pub proof messages x.2 x.1) s (some a') := by
        refine step_reideal pub proof messages x.2 x.1 s ideal fun p asked => ?_
        by_cases squeezes : (∃ site level bits, x.1 = .work site level bits ∧ bits ≠ 0) ∨
            isDraw x.1 = true
        · obtain ⟨events, count, found, after⟩ :=
            step_asked pub proof messages x.2 x.1 s squeezes stepped
          obtain rfl : p = (events, count) := Option.some.inj (asked.symm.trans found)
          refine agree events count ⟨grown, ?_⟩
          rw [later, after]
        · rw [asks_none proof messages s squeezes] at asked
          cases asked
      have kept : (((step pub proof messages x.2 x.1).run).run s).2.sponge.ideal = some a :=
        (Eq.symm (step_kept pub proof messages x.2 x.1 s)).trans ideal
      unfold BlindAt at blind
      rw [stepped] at blind
      rw [runFrom_cons, blind]
      exact runFrom_reideal l _ kept tail agree

end Replayer

section States

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  {a a' : List Duplex.Event → ℕ → ByteArray}

theorem channelStart_reideal (ideal : Option (List Duplex.Event → ℕ → ByteArray)) :
    channelStart (pub.reideal ideal) = reideal (channelStart pub) ideal :=
  congrArg (fun d => (⟨d, 0, {}, {}, {}, {}⟩ : Replayer))
    (channel_reideal pub.session pub.descriptor pub.sponge ideal)

variable {pub}

/-- The run before the operation `c` with the answers `a'` in the sponge of the public input: it
has no stop, and its last state is the state of the run with the answers `a`, when `a` and `a'`
agree on the events of this state. -/
theorem stateBefore_reideal (ideal : pub.sponge.ideal = some a) (c : ℕ)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (agree : AgreeOn (stateBefore pub proof messages pub.ops c).sponge.events a a') :
    (runFrom (pub.reideal (some a')) proof messages
        ((pub.reideal (some a')).ops.zipIdx.toList.take c)
        (channelStart (pub.reideal (some a')))).1 = .ok () ∧
      stateBefore (pub.reideal (some a')) proof messages (pub.reideal (some a')).ops c =
        reideal (stateBefore pub proof messages pub.ops c) (some a') := by
  have start : (channelStart pub).sponge.ideal = some a := by
    simp only [channelStart, channel, Duplex.absorb_ideal]
    exact ideal
  have run := runFrom_reideal pub proof messages _ (channelStart pub) start ran agree
  have moved : runFrom (pub.reideal (some a')) proof messages (pub.ops.zipIdx.toList.take c)
      (channelStart (pub.reideal (some a'))) =
      (.ok (), reideal (stateBefore pub proof messages pub.ops c) (some a')) := by
    rw [channelStart_reideal]
    exact (runFrom_withSponge pub proof messages _ _ _).trans run
  exact ⟨congrArg Prod.fst moved, congrArg Prod.snd moved⟩

variable {proof messages}

/-- The events before the squeeze of the work of a stage for two answer functions that agree on
them. -/
theorem eventsAt20_reideal (ideal : pub.sponge.ideal = some a) {k : ℕ}
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (agree : AgreeOn (eventsAt20 pub proof messages k) a a') :
    eventsAt20 (pub.reideal (some a')) proof messages k = eventsAt20 pub proof messages k := by
  have inner : AgreeOn (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events
      a a' :=
    agree.suffix (show (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events
      <:+ eventsAt20 pub proof messages k from List.suffix_append _ _)
  have state : stateBefore (pub.reideal (some a')) proof messages (pub.reideal (some a')).ops
      (stageIndex20 (pub.reideal (some a')) k) =
      reideal (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) (some a') :=
    (stateBefore_reideal proof messages ideal _ ran inner).2
  unfold eventsAt20
  rw [state]
  rfl

/-- The digests of the last blocks in the state before the work of a stage. -/
theorem digestAt_reideal (ideal : pub.sponge.ideal = some a) {k : ℕ}
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (agree : AgreeOn (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events
      a a') :
    tDigestAt (pub.reideal (some a')) proof messages k = tDigestAt pub proof messages k ∧
      eDigestAt (pub.reideal (some a')) proof messages k = eDigestAt pub proof messages k := by
  have state : stateBefore (pub.reideal (some a')) proof messages (pub.reideal (some a')).ops
      (stageIndex20 (pub.reideal (some a')) k) =
      reideal (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) (some a') :=
    (stateBefore_reideal proof messages ideal _ ran agree).2
  unfold tDigestAt eDigestAt
  rw [state]
  exact ⟨rfl, rfl⟩

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The page of a stage for two answer functions that agree on the events before the work of
the stage. -/
theorem pageAt20_reideal (ideal : pub.sponge.ideal = some a) {k : ℕ}
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (agree : AgreeOn (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events
      a a')
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) :
    pageAt20 (E := E) (pub.reideal (some a')) proof messages tLast eLast k =
      pageAt20 pub proof messages tLast eLast k := by
  have state : stateBefore (pub.reideal (some a')) proof messages (pub.reideal (some a')).ops
      (stageIndex20 (pub.reideal (some a')) k) =
      reideal (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) (some a') :=
    (stateBefore_reideal proof messages ideal _ ran agree).2
  have note : noteOf (E := E) (pub.reideal (some a'))
      (reideal (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) (some a')) tLast
        eLast =
      noteOf pub (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) tLast eLast :=
    noteOf_core pub (core_reideal _ _) tLast eLast
  have recvs : recvs20 E (pub.reideal (some a'))
      (reideal (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) (some a')) k =
      recvs20 E pub (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) k :=
    recvs20_core pub E (core_reideal _ _) k
  unfold pageAt20 noteAt20 cellsAt20
  rw [state, note, recvs]
  rfl

variable (pub proof messages) in
/-- The events of the state before the work of a stage `j ≤ k` are a suffix of the events before
the squeeze of the work of the stage `k`. -/
theorem stateBefore_suffix20 {j k : ℕ} (le : j ≤ k) (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ()) :
    (stateBefore pub proof messages pub.ops (stageIndex20 pub j)).sponge.events <:+
      eventsAt20 pub proof messages k := by
  obtain ⟨-, -, cut⟩ := stateBefore_le pub proof messages pub.ops
    (stageIndex20_mono pub le inside) ran
  obtain ⟨grown, later⟩ := runFrom_suffix pub proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub k)).drop (stageIndex20 pub j))
    (stateBefore pub proof messages pub.ops (stageIndex20 pub j))
  rw [← cut] at later
  exact (List.IsSuffix.trans ⟨grown, later.symm⟩ (List.suffix_append _ _) :
    _ <:+ _ ++ (stateBefore pub proof messages pub.ops (stageIndex20 pub k)).sponge.events)

/-- The page `j ≤ k` of the key of the stage `k` for two answer functions that agree on the
events before the squeeze of the work of the stage `k`. -/
theorem stagePage20_reideal (ideal : pub.sponge.ideal = some a) {j k : ℕ} (le : j ≤ k)
    (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (agree : AgreeOn (eventsAt20 pub proof messages k) a a')
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) :
    stagePage20 (E := E) (pub.reideal (some a')) proof messages tLast eLast j =
      stagePage20 pub proof messages tLast eLast j :=
  pageAt20_reideal ideal
    (stateBefore_le pub proof messages pub.ops (stageIndex20_mono pub le inside) ran).1
    (agree.suffix (stateBefore_suffix20 pub proof messages le inside ran)) _ _

/-- The key of a stage for two answer functions that agree on the events before the squeeze of
its work. -/
theorem addressAt20_reideal (ideal : pub.sponge.ideal = some a) {k : ℕ} (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (agree : AgreeOn (eventsAt20 pub proof messages k) a a')
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
    (total : ℕ) :
    addressAt20 (E := E) (pub.reideal (some a')) proof messages tLast eLast total k =
      addressAt20 pub proof messages tLast eLast total k := by
  refine Address.ext (stagePage20_reideal ideal (Nat.zero_le k) inside ran agree tLast eLast) ?_
  show (List.range (min k total)).map _ = (List.range (min k total)).map _
  refine List.map_congr_left fun j member => ?_
  have below := List.mem_range.mp member
  exact stagePage20_reideal ideal (by omega) inside ran agree tLast eLast

/-- A stage of a run with the answers `a` is a stage of the run with the answers `a'`, when `a`
and `a'` agree on its events. -/
theorem stage20_reideal (ideal : pub.sponge.ideal = some a) {events : List Duplex.Event} {k : ℕ}
    (agree : AgreeOn events a a') (stage : Stage20 pub events k proof messages) :
    Stage20 (pub.reideal (some a')) events k proof messages := by
  obtain ⟨inside, decoded, ran, same⟩ := stage
  subst same
  exact ⟨inside, decoded,
    (stateBefore_reideal proof messages ideal _ ran
      (agree.suffix (stateBefore_suffix20 pub proof messages (Nat.le_refl k) inside ran))).1,
    eventsAt20_reideal ideal ran agree⟩

/-- A stage of a run with the answers `a'` is a stage of the run with the answers `a`, when `a`
and `a'` agree on its events. -/
theorem stage20_of_reideal (ideal : pub.sponge.ideal = some a) {events : List Duplex.Event}
    {k : ℕ} (agree : AgreeOn events a a')
    (stage : Stage20 (pub.reideal (some a')) events k proof messages) :
    Stage20 pub events k proof messages := by
  have back := stage20_reideal (pub := pub.reideal (some a')) (a := a') (a' := a) rfl
    agree.symm stage
  have twice : (pub.reideal (some a')).reideal (some a) = pub := Public.reideal_self ideal
  rwa [twice] at back

/-- The key of a stage with the last blocks of a record for two answer functions that agree on
the events of the stage. -/
theorem recordAddress20_reideal (ideal : pub.sponge.ideal = some a) {events : List Duplex.Event}
    {k : ℕ} (agree : AgreeOn events a a') (stage : Stage20 pub events k proof messages)
    (record : List (ByteArray × ByteArray)) (total : ℕ) :
    recordAddress20 (E := E) (pub.reideal (some a')) record total k proof messages =
      recordAddress20 pub record total k proof messages := by
  obtain ⟨inside, -, ran, same⟩ := stage
  subst same
  have digests := digestAt_reideal ideal ran
    (agree.suffix (stateBefore_suffix20 pub proof messages (Nat.le_refl k) inside ran))
  unfold recordAddress20
  rw [digests.1, digests.2]
  exact addressAt20_reideal ideal inside ran agree _ _ total

end States

section Functions

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  {pub : Public} {proof proof' : ByteArray} {messages messages' : Array Message}
  {a a' : List Duplex.Event → ℕ → ByteArray}

/-- Events give pages: two runs with the same events before the squeeze of the work of a stage,
one with the answers `a` and one with the answers `a'`, have the same key of this stage, when
`a` and `a'` agree at each squeeze of these events. -/
theorem addressAt20_of_events (ideal : pub.sponge.ideal = some a) {events : List Duplex.Event}
    {k : ℕ} (agree : AgreeOn events a a') (stage : Stage20 pub events k proof messages)
    (stage' : Stage20 (pub.reideal (some a')) events k proof' messages')
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
    (total : ℕ) :
    addressAt20 (E := E) (pub.reideal (some a')) proof' messages' tLast eLast total k =
      addressAt20 pub proof messages tLast eLast total k := by
  obtain ⟨inside, decoded, ran, same⟩ := stage20_of_reideal ideal agree stage'
  have moved : addressAt20 (E := E) (pub.reideal (some a')) proof' messages' tLast eLast total k =
      addressAt20 pub proof' messages' tLast eLast total k :=
    addressAt20_reideal ideal inside ran (same ▸ agree) tLast eLast total
  exact moved.trans (addressAt20_causal ⟨inside, decoded, ran, same⟩ stage tLast eLast total)

/-- Pages give events: on a script of `layout20`, two runs with the same key of a stage, one with
the answers `a` and one with the answers `a'`, have the same events before the squeeze of the
work of this stage, when `a` and `a'` agree at each squeeze of the events of the second run. -/
theorem eventsAt20_of_address (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (ideal : pub.sponge.ideal = some a) {k : ℕ} (inside : k < 473) {total : ℕ} (long : k ≤ total)
    {rest rest' : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest))
    (decoded' : decode pub.ops proof' = .ok (messages', rest'))
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (ran' : (runFrom (pub.reideal (some a')) proof' messages'
      (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart (pub.reideal (some a')))).1 = .ok ())
    (agree : AgreeOn (eventsAt20 (pub.reideal (some a')) proof' messages' k) a a')
    {tLast tLast' : ℕ → Rq Key20.terminal.innerDegree}
    {eLast eLast' : Rq Key20.terminal.innerDegree}
    (same : addressAt20 (E := E) pub proof messages tLast eLast total k =
      addressAt20 (pub.reideal (some a')) proof' messages' tLast' eLast' total k) :
    eventsAt20 pub proof messages k = eventsAt20 (pub.reideal (some a')) proof' messages' k := by
  have back := stage20_of_reideal ideal agree ⟨inside, ⟨rest', decoded'⟩, ran', rfl⟩
  obtain ⟨-, -, ranBack, events⟩ := back
  have moved : addressAt20 (E := E) (pub.reideal (some a')) proof' messages' tLast' eLast' total
      k = addressAt20 pub proof' messages' tLast' eLast' total k :=
    addressAt20_reideal ideal inside ranBack (events ▸ agree) tLast' eLast' total
  rw [← events]
  exact addressAt20_events erased inside long decoded decoded' ran ranBack (same.trans moved)

/-- `resolve20` reads the answers of the sponge only at the squeezes of its events. -/
theorem resolve20_reideal (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a)
    {events : List Duplex.Event} (agree : AgreeOn events a a')
    (record : List (ByteArray × ByteArray)) (total : ℕ) :
    resolve20 (E := E) total (pub.reideal (some a')) record events =
      resolve20 total pub record events := by
  by_cases found : ∃ stage : ℕ × ByteArray × Array Message,
      Stage20 pub events stage.1 stage.2.1 stage.2.2
  · obtain ⟨⟨k, proof, messages⟩, stage⟩ := found
    have bits' : Bits20 (pub.reideal (some a')) := bits
    rw [resolve20_stage bits stage record total,
      resolve20_stage bits' (stage20_reideal ideal agree stage) record total]
    exact congrArg some (recordAddress20_reideal ideal agree stage record total)
  · have missing : ¬∃ stage : ℕ × ByteArray × Array Message,
        Stage20 (pub.reideal (some a')) events stage.1 stage.2.1 stage.2.2 := by
      rintro ⟨⟨k, proof, messages⟩, stage⟩
      exact found ⟨(k, proof, messages), stage20_of_reideal ideal agree stage⟩
    unfold resolve20
    rw [dite_eq_right found, dite_eq_right missing]

/-- `resolve` of Akita Lemma 10.47 for `key20.aks` with the answers of the sponge as an
argument: `resolve20` for the public input with the answers `answer` in its sponge. -/
noncomputable def resolveWith20 (total : ℕ) (pub : Public)
    (answer : List Duplex.Event → ℕ → ByteArray) (record : List (ByteArray × ByteArray))
    (events : List Duplex.Event) : Option (Address (Key20.Wire E) total) :=
  resolve20 total (pub.reideal (some answer)) record events

/-- `resolveWith20` reads the answers only at the squeezes of its events. -/
theorem resolveWith20_agree (bits : Bits20 pub) {events : List Duplex.Event}
    (agree : AgreeOn events a a') (record : List (ByteArray × ByteArray)) (total : ℕ) :
    resolveWith20 (E := E) total pub a record events = resolveWith20 total pub a' record events :=
  (resolve20_reideal (pub := pub.reideal (some a)) bits rfl agree record total).symm

/-- On a script of `layout20`, for each answers and each record: a key that `resolveWith20`
gives for `events` has `back20` equal to `events`. -/
theorem resolveWith20_back (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {total : ℕ} (long : 472 ≤ total) (record : List (ByteArray × ByteArray))
    (events : List Duplex.Event) (address : Address (Key20.Wire E) total)
    (resolved : resolveWith20 total pub a record events = some address) :
    back20 (pub.reideal (some a)) address = events :=
  resolves20 (pub := pub.reideal (some a)) erased long record events address resolved

/-- For each answer function `a'` and each run with it: `resolveWith20` with answers `a` that
agree with `a'` at each squeeze of the events of the stage `k` of the run is the key of this
stage, with the conditions of `resolve20_eq` on the record of the digest queries. -/
theorem resolveWith20_eq (bits : Bits20 pub) {k : ℕ} (inside : k < 473) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (ran : (runFrom (pub.reideal (some a')) proof messages
      (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart (pub.reideal (some a')))).1 = .ok ())
    (agree : AgreeOn (eventsAt20 (pub.reideal (some a')) proof messages k) a a')
    {record : List (ByteArray × ByteArray)} {tLast : ℕ → Rq Key20.terminal.innerDegree}
    {eLast : Rq Key20.terminal.innerDegree}
    (tListed : 379 ≤ k →
      (tInput Key20.terminal tLast, tDigestAt (pub.reideal (some a')) proof messages k) ∈ record ∧
      ∀ input, (input, tDigestAt (pub.reideal (some a')) proof messages k) ∈ record →
        input = tInput Key20.terminal tLast)
    (eListed : 472 ≤ k →
      (eInput eLast, eDigestAt (pub.reideal (some a')) proof messages k) ∈ record ∧
      ∀ input, (input, eDigestAt (pub.reideal (some a')) proof messages k) ∈ record →
        input = eInput eLast)
    (total : ℕ) :
    resolveWith20 (E := E) total pub a record (eventsAt20 (pub.reideal (some a')) proof messages k) =
      some (addressAt20 (pub.reideal (some a')) proof messages tLast eLast total k) := by
  have bits' : Bits20 (pub.reideal (some a')) := bits
  rw [resolveWith20_agree bits agree record total]
  exact resolve20_eq bits' inside decoded ran tListed eListed total

end Functions

end NightstreamFPrime.Spec.Compression.Akita.Replay
