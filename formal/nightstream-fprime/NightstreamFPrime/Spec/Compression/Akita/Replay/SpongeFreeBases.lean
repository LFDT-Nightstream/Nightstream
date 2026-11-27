import NightstreamFPrime.Spec.Compression.Akita.Replay.SpongeFree

/-! Each squeeze inside the key of a stage of `key20.aks` is a squeeze of an earlier stage (Akita
revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23, Appendix E.1: the stage input
`T^J(k)` reads the table at the keys of earlier stages; `Replay/SpongeFree.lean`,
`Replay/CausalTrace.lean`).

- `Replayer.Absorbs`, `step_absorbs`, `runFrom_absorbs`: an operation that is not a work and not
  a draw adds only absorbs to the events of the sponge.
- `skeleton20_quiet`, `skeleton20_start`: in the skeleton of `layout20`, no operation after the
  draws of a stage and before the next work, and no operation before the first work, is a work
  or a draw. `quiet20_of` and `start20_of` are these facts on a script of `layout20`.
- `eventsAt20_zero_parts`, `eventsAt20_succ_parts`: the events before the squeeze of the work of
  the stage `k + 1` are absorbs, the squeezes of the draws and of the work of the stage `k`, and
  the events of the stage `k`.
- `squeeze_stage20`: the events before a squeeze inside `eventsAt20 … k` are squeezes and
  `eventsAt20 … j` of a stage `j < k`.
- `agreeOn_stages`, `agreeOn_tables`: two answer functions (two tables) with the same answers at
  the squeezes (at the bases) of the stages `j < k` agree on `eventsAt20 … k`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open Sponge (split answerOf)
open Replayer

namespace Replayer

/-- The events of the second sponge are new absorbs and the events of the first. -/
def Absorbs (a b : Duplex) : Prop :=
  ∃ new : List ByteArray, b.events = new.map Duplex.Event.absorb ++ a.events

theorem absorbs_path : Path Absorbs :=
  ⟨fun _ => ⟨[], rfl⟩, fun ⟨x, hx⟩ ⟨y, hy⟩ =>
    ⟨y ++ x, by rw [hy, hx, List.map_append, List.append_assoc]⟩⟩

theorem absorbs_absorb (bytes : ByteArray) : Moves Absorbs (absorb bytes) := fun s =>
  ⟨[bytes], Duplex.absorb_events s.sponge bytes⟩

macro "replay_absorbs_step" : tactic => `(tactic| repeat' (first
  | with_reducible exact moves_pure absorbs_path _
  | with_reducible exact moves_throw absorbs_path _
  | with_reducible exact moves_get absorbs_path
  | with_reducible exact absorbs_absorb _
  | with_reducible exact moves_message absorbs_path _ _
  | with_reducible exact moves_recorded absorbs_path _
  | with_reducible exact moves_first absorbs_path _
  | with_reducible exact moves_pointBefore absorbs_path _
  | with_reducible exact moves_record absorbs_path _ _
  | with_reducible exact moves_value absorbs_path _ _ _
  | with_reducible exact moves_terminalWith absorbs_path _ _ _
  | with_reducible exact moves_checkWith absorbs_path _ _ _ _ _
  | with_reducible exact moves_mapM absorbs_path (fun _ => moves_recorded absorbs_path _) _
  | with_reducible exact moves_decodeExts absorbs_path _
  | (with_reducible refine moves_modify absorbs_path fun _ => ?_; rfl)
  | refine moves_bind absorbs_path ?_ fun _ => ?_
  | split))

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The step of an operation that is not a work and not a draw adds only absorbs. -/
theorem step_absorbs (i : ℕ) (op : Op) (quiet : isWork op = false) (calm : isDraw op = false) :
    Moves Absorbs (step pub proof messages i op) := by
  cases op <;> first
    | exact Bool.noConfusion quiet
    | exact Bool.noConfusion calm
    | (simp only [step]; replay_absorbs_step)

/-- A run of operations that are not works and not draws adds only absorbs. -/
theorem runFrom_absorbs : ∀ (l : List (Op × ℕ)) (s : Replayer),
    (∀ x ∈ l, isWork x.1 = false ∧ isDraw x.1 = false) →
    Absorbs s.sponge (runFrom pub proof messages l s).2.sponge
  | [], _, _ => absorbs_path.refl _
  | x :: l, s, quiet => by
      have stepped := step_absorbs pub proof messages x.2 x.1 (quiet x List.mem_cons_self).1
        (quiet x List.mem_cons_self).2 s
      rw [runFrom_cons]
      rcases found : ((step pub proof messages x.2 x.1).run).run s with ⟨_ | _, s₁⟩
      · rw [found] at stepped
        exact stepped
      · rw [found] at stepped
        exact absorbs_path.trans stepped
          (runFrom_absorbs l s₁ fun y member => quiet y (List.mem_cons_of_mem _ member))

end Replayer

section Skeleton

/-- In the skeleton of `layout20`, the draws of the stage `k` end before the work of the stage
`k + 1`, and no operation after them and before this work is a work or a draw. -/
def quietFit20 (k : ℕ) : Bool :=
  decide (workIndices20.getD k 0 + 1 + drawCount20 k ≤ workIndices20.getD (k + 1) 0) &&
    ((sk20.take (workIndices20.getD (k + 1) 0)).drop
      (workIndices20.getD k 0 + 1 + drawCount20 k)).all fun x => !isWork x.1 && !isDraw x.1

theorem skeleton20_quiet : (List.range 472).all quietFit20 = true := by decide +kernel

/-- In the skeleton of `layout20`, no operation before the first work is a work or a draw. -/
theorem skeleton20_start :
    ((sk20.take (workIndices20.getD 0 0)).all fun x => !isWork x.1 && !isDraw x.1) = true := by
  decide +kernel

variable {pub : Public}

/-- On a script of `layout20`, the draws of the stage `k` end before the work of the stage
`k + 1`, and no operation after them and before this work is a work or a draw. -/
theorem quiet20_of (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) {k : ℕ}
    (inside : k + 1 < 473) :
    stageIndex20 pub k + 1 + drawCount20 k ≤ stageIndex20 pub (k + 1) ∧
      ∀ x ∈ (pub.ops.zipIdx.toList.take (stageIndex20 pub (k + 1))).drop
        (stageIndex20 pub k + 1 + drawCount20 k), isWork x.1 = false ∧ isDraw x.1 = false := by
  have fit := List.all_eq_true.mp skeleton20_quiet k (List.mem_range.mpr (by omega))
  unfold quietFit20 at fit
  rw [Bool.and_eq_true] at fit
  rw [stageIndex20_lt pub (by omega : k < 473), stageIndex20_lt pub inside]
  refine ⟨of_decide_eq_true fit.1, fun x member => ?_⟩
  have erasedMember : eraseAt x ∈ (sk20.take (workIndices20.getD (k + 1) 0)).drop
      (workIndices20.getD k 0 + 1 + drawCount20 k) := by
    rw [← take_erase erased, ← List.map_drop]
    exact List.mem_map_of_mem member
  have both := List.all_eq_true.mp fit.2 _ erasedMember
  rw [Bool.and_eq_true] at both
  have works : isWork x.1.erase = isWork x.1 := by cases x.1 <;> rfl
  exact ⟨works.symm.trans (by simpa using both.1),
    (isDraw_erase x.1).symm.trans (by simpa using both.2)⟩

/-- On a script of `layout20`, no operation before the first work is a work or a draw. -/
theorem start20_of (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) :
    ∀ x ∈ pub.ops.zipIdx.toList.take (stageIndex20 pub 0),
      isWork x.1 = false ∧ isDraw x.1 = false := by
  intro x member
  rw [stageIndex20_lt pub (by omega : 0 < 473)] at member
  have erasedMember : eraseAt x ∈ sk20.take (workIndices20.getD 0 0) := by
    rw [← take_erase erased]
    exact List.mem_map_of_mem member
  have both := List.all_eq_true.mp skeleton20_start _ erasedMember
  rw [Bool.and_eq_true] at both
  have works : isWork x.1.erase = isWork x.1 := by cases x.1 <;> rfl
  exact ⟨works.symm.trans (by simpa using both.1),
    (isDraw_erase x.1).symm.trans (by simpa using both.2)⟩

end Skeleton

section Lists

variable {count : ℕ} {before : List Duplex.Event}

theorem absorbs_events_trans {x y z : List Duplex.Event} :
    (∃ n : List ByteArray, y = n.map Duplex.Event.absorb ++ x) →
    (∃ n : List ByteArray, z = n.map Duplex.Event.absorb ++ y) →
    ∃ n : List ByteArray, z = n.map Duplex.Event.absorb ++ x
  | ⟨n, h⟩, ⟨m, g⟩ => ⟨m ++ n, by rw [g, h, List.map_append, List.append_assoc]⟩

/-- A suffix of absorbs and a list that starts with a squeeze is a suffix of the list. -/
theorem squeeze_absorbs : ∀ (new : List ByteArray) {rest : List Duplex.Event},
    (Duplex.Event.squeeze count :: before) <:+ (new.map Duplex.Event.absorb ++ rest) →
    (Duplex.Event.squeeze count :: before) <:+ rest
  | [], _, inner => inner
  | b :: new, rest, inner => by
      rw [List.map_cons, List.cons_append, List.suffix_cons_iff] at inner
      rcases inner with same | later
      · exact Duplex.Event.noConfusion (List.cons.inj same).1
      · exact squeeze_absorbs new later

/-- A suffix of squeezes and a list `base` that starts with a squeeze: the events after its
first squeeze are squeezes and `base`, or it is a suffix of `base`. -/
theorem squeeze_squeezes {base : List Duplex.Event} : ∀ (counts : List ℕ),
    (Duplex.Event.squeeze count :: before) <:+ (counts.map Duplex.Event.squeeze ++ base) →
    (∃ later : List ℕ, before = later.map Duplex.Event.squeeze ++ base) ∨
      (Duplex.Event.squeeze count :: before) <:+ base
  | [], inner => Or.inr inner
  | c :: counts, inner => by
      rw [List.map_cons, List.cons_append, List.suffix_cons_iff] at inner
      rcases inner with same | later
      · exact Or.inl ⟨counts, (List.cons.inj same).2⟩
      · exact squeeze_squeezes counts later

end Lists

section Events

variable {pub : Public} {proof : ByteArray} {messages : Array Message}
  {a : List Duplex.Event → ℕ → ByteArray}

theorem channelStart_absorbs (pub : Public) : Absorbs pub.sponge (channelStart pub).sponge :=
  ⟨[protocolTag ++ bytesMsg pub.descriptor, bytesMsg pub.session, sessionId], by
    show (channel pub.session pub.descriptor pub.sponge).events = _
    unfold channel
    rw [Duplex.absorb_events, Duplex.absorb_events, Duplex.absorb_events]
    rfl⟩

variable (proof messages) in
/-- On a script of `layout20`, the events before the squeeze of the first work are absorbs and
the events of the sponge of the public input. -/
theorem eventsAt20_zero_parts (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) :
    ∃ new : List ByteArray, eventsAt20 pub proof messages 0 =
      new.map Duplex.Event.absorb ++ pub.sponge.events :=
  absorbs_events_trans
    (absorbs_events_trans (channelStart_absorbs pub)
      (runFrom_absorbs pub proof messages _ (channelStart pub) (start20_of erased)))
    (⟨_, rfl⟩ : ∃ n : List ByteArray, eventsAt20 pub proof messages 0 =
      n.map Duplex.Event.absorb ++
        (stateBefore pub proof messages pub.ops (stageIndex20 pub 0)).sponge.events)

/-- On a script of `layout20` with the work bits of the model and an ideal sponge, the events
before the squeeze of the work of the stage `k + 1` of a run with no stop before this work are
absorbs, the squeezes of the draws and of the work of the stage `k`, and the events before the
squeeze of the work of the stage `k`. -/
theorem eventsAt20_succ_parts (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) {k : ℕ} (inside : k + 1 < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub (k + 1)))
      (channelStart pub)).1 = .ok ()) :
    ∃ (new : List ByteArray) (counts : List ℕ), eventsAt20 pub proof messages (k + 1) =
      new.map Duplex.Event.absorb ++
        (counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages k) := by
  obtain ⟨le, quiet⟩ := quiet20_of erased inside
  obtain ⟨site, level, at_⟩ := bits k (by omega)
  obtain ⟨first, -, cut⟩ := stateBefore_le pub proof messages pub.ops le ran
  have drawn := (draw_squeeze20 at_ (bits20_ne_zero (by omega : k < 473))
    (Nat.le_add_right _ _) (draws20_of erased (by omega : k < 473) (Nat.le_refl _)) first
    ideal).1
  obtain ⟨counts, shape⟩ : ∃ counts : List ℕ,
      (stateBefore pub proof messages pub.ops
        (stageIndex20 pub k + 1 + drawCount20 k)).sponge.events =
        counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages k :=
    ⟨(((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1 + drawCount20 k)).drop
        (stageIndex20 pub k + 1)).reverse.map fun x => x.1.drawBytes) ++ [8], by
      rw [drawn, List.map_append, List.map_map, List.append_assoc]
      rfl⟩
  obtain ⟨n1, e1⟩ := runFrom_absorbs pub proof messages
    ((pub.ops.zipIdx.toList.take (stageIndex20 pub (k + 1))).drop
      (stageIndex20 pub k + 1 + drawCount20 k))
    (stateBefore pub proof messages pub.ops (stageIndex20 pub k + 1 + drawCount20 k)) quiet
  rw [← cut, shape] at e1
  obtain ⟨n, e⟩ := absorbs_events_trans ⟨n1, e1⟩
    (⟨_, rfl⟩ : ∃ n : List ByteArray, eventsAt20 pub proof messages (k + 1) =
      n.map Duplex.Event.absorb ++
        (stateBefore pub proof messages pub.ops (stageIndex20 pub (k + 1))).sponge.events)
  exact ⟨n, counts, e⟩

/-- On a script of `layout20` with the work bits of the model and an ideal sponge with no event
before the channel: the events before a squeeze inside the events before the squeeze of the
work of the stage `k` are squeezes and the events before the squeeze of the work of a stage
`j < k`. -/
theorem squeeze_stage20 (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = []) :
    ∀ (k : ℕ), k < 473 →
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
        (channelStart pub)).1 = .ok () →
      ∀ {before : List Duplex.Event} {count : ℕ},
        (Duplex.Event.squeeze count :: before) <:+ eventsAt20 pub proof messages k →
        ∃ j, j < k ∧ ∃ counts : List ℕ,
          before = counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages j
  | 0, _, _, before, count, inner => by
      obtain ⟨new, events⟩ := eventsAt20_zero_parts proof messages erased
      rw [events, fresh] at inner
      exact absurd (List.suffix_nil.mp (squeeze_absorbs new inner)) (List.cons_ne_nil _ _)
  | k + 1, inside, ran, before, count, inner => by
      obtain ⟨new, counts, events⟩ := eventsAt20_succ_parts erased bits ideal inside ran
      rw [events] at inner
      rcases squeeze_squeezes counts (squeeze_absorbs new inner) with ⟨later, same⟩ | deeper
      · exact ⟨k, Nat.lt_succ_self k, later, same⟩
      · obtain ⟨j, lt, found⟩ := squeeze_stage20 erased bits ideal fresh k (by omega)
          (stateBefore_le pub proof messages pub.ops
            (stageIndex20_mono pub (Nat.le_succ k) inside) ran).1 deeper
        exact ⟨j, Nat.lt_succ_of_lt lt, found⟩

/-- Two answer functions with the same answers at the squeezes of the stages `j < k` agree on
the events before the squeeze of the work of the stage `k`. -/
theorem agreeOn_stages (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = [])
    {k : ℕ} (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    {a' : List Duplex.Event → ℕ → ByteArray}
    (same : ∀ j, j < k → ∀ (counts : List ℕ) (count : ℕ),
      a (counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages j) count =
        a' (counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages j) count) :
    AgreeOn (eventsAt20 pub proof messages k) a a' := by
  intro before count inner
  obtain ⟨j, lt, counts, found⟩ := squeeze_stage20 erased bits ideal fresh k inside ran inner
  rw [found]
  exact same j lt counts count

/-- Two tables of the sponge with the same answers at the base of each stage `j < k` give answer
functions that agree on the events before the squeeze of the work of the stage `k`. -/
theorem agreeOn_tables {L B C S : ℕ} {fs fs' : Sponge.Base L B C → Sponge.Y S C}
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (bits : Bits20 pub)
    (ideal : pub.sponge.ideal = some (answerOf fs)) (fresh : pub.sponge.events = []) {k : ℕ}
    (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    (same : ∀ j, j < k → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 pub proof messages j → fs base = fs' base) :
    AgreeOn (eventsAt20 pub proof messages k) (answerOf fs) (answerOf fs') := by
  refine agreeOn_answerOf fun before count base sub inner found => ?_
  obtain ⟨j, lt, counts, shape⟩ := squeeze_stage20 erased bits ideal fresh k inside ran inner
  obtain ⟨parts, -⟩ := Sponge.split_eq_some.mp found
  obtain ⟨site, level, at_⟩ := bits j (by omega)
  obtain ⟨m, -, events, -, -⟩ := work_squeeze20 at_ (bits20_ne_zero (by omega : j < 473))
    (stateBefore_le pub proof messages pub.ops (stageIndex20_succ_le pub lt inside) ran).1 ideal
  have head : ∀ c rest, eventsAt20 pub proof messages j ≠ Duplex.Event.squeeze c :: rest := by
    intro c rest equal
    rw [events] at equal
    exact Duplex.Event.noConfusion (List.cons.inj equal).1
  exact same j lt base (Sponge.squeezes_append_inj base.events_ne head
    (show sub.counts.map Duplex.Event.squeeze ++ base.events =
      counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages j from
      parts.symm.trans shape)).2

end Events

end NightstreamFPrime.Spec.Compression.Akita.Replay
