import NightstreamFPrime.Spec.Compression.Akita.Replay.SimulateDeps
import NightstreamFPrime.Spec.Compression.Akita.Replay.SpongeFreeBases
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageMaps

/-! The `key20.aks` instance of the recording prover of Akita (ePrint 2026/1983, revision of
2026-11-23, Lemma 10.47 and Appendix E.1, (F1)): the keys that the address of a sponge key
reads, the stage of a key, the stage map and the address (`docs/akita-work/w21/design.md`,
sections 7 and 8; `Replay/SimulateDeps.lean`, `Replay/SpongeFree.lean`,
`Replay/SpongeFreeBases.lean`, `Replay/StageMaps.lean`).

- `deps20 S base`: the bases of the squeezes inside the events of `base`, the oldest first.
  `deps20_closed` is `Simulate.DepsClosed`, and `length_deps20` bounds the count by `L`.
- `stageCount events`: the count of the squeezes directly after an absorb. `stageOf base` is this
  count for the events of `base`. `stageOf_base`: a base with the events `eventsAt20 … k` of a
  run with an ideal sponge has the stage `k`.
- `phiAt … base` is the stage map `phi20 … (stageOf base)`, with the fibres `card_phiAt`.
- `resolveAt total pub fs record base` is `resolveWith20` with the answers of the table `fs`.
  `resolveAt_local` is `Simulate.DepsLocal`, and `resolveAt_back` gives the hypothesis
  `resolves` of `Simulate.simulateDeps_law` with `backAt`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Address)
open Sponge (split answerOf)
open Replayer

section Deps

variable {L B C : ℕ} (S : ℕ)

/-- The bases of the squeezes inside `events` (the newest event first), the oldest squeeze
first. A squeeze whose query is not in the bounds of the key types gives no base. -/
noncomputable def depsOf : List Duplex.Event → List (Sponge.Base L B C)
  | [] => []
  | .absorb _ :: before => depsOf before
  | .squeeze count :: before =>
      depsOf before ++
        ((split before count : Option (Sponge.Base L B C × Sponge.Sub S C)).map Prod.fst).toList

theorem depsOf_cons (e : Duplex.Event) (before : List Duplex.Event) :
    ∃ extra : List (Sponge.Base L B C), extra.length ≤ 1 ∧
      depsOf S (e :: before) = depsOf S before ++ extra := by
  cases e with
  | absorb input => exact ⟨[], Nat.zero_le _, (List.append_nil _).symm⟩
  | squeeze count =>
      refine ⟨((split before count : Option (Sponge.Base L B C × Sponge.Sub S C)).map
        Prod.fst).toList, ?_, rfl⟩
      cases (split before count : Option (Sponge.Base L B C × Sponge.Sub S C)) <;> simp

/-- The bases of a suffix of the events are the first bases of the events. -/
theorem depsOf_prefix {early : List Duplex.Event} : ∀ {events : List Duplex.Event},
    early <:+ events → (depsOf S early : List (Sponge.Base L B C)) <+: depsOf S events
  | [], suffix => by
      rw [List.suffix_nil.mp suffix]
  | e :: rest, suffix => by
      rcases List.suffix_cons_iff.mp suffix with same | inner
      · subst same
        exact List.prefix_refl _
      · obtain ⟨extra, -, grown⟩ := depsOf_cons (L := L) (B := B) (C := C) S e rest
        rw [grown]
        exact (depsOf_prefix inner).trans (List.prefix_append _ _)

/-- The base of each squeeze inside the events is in `depsOf`. -/
theorem mem_depsOf {events before : List Duplex.Event} {count : ℕ} {base : Sponge.Base L B C}
    {sub : Sponge.Sub S C} (inside : (Duplex.Event.squeeze count :: before) <:+ events)
    (found : split before count = some (base, sub)) : base ∈ depsOf S events := by
  refine (depsOf_prefix S inside).subset ?_
  show base ∈ depsOf S before ++
    ((split before count : Option (Sponge.Base L B C × Sponge.Sub S C)).map Prod.fst).toList
  rw [found]
  simp

/-- The bases that a base of `depsOf events` names are before it in `depsOf events`. -/
theorem depsOf_closed : ∀ (events : List Duplex.Event) (pre : List (Sponge.Base L B C))
    (key : Sponge.Base L B C) (post : List (Sponge.Base L B C)),
    depsOf S events = pre ++ key :: post → ∀ named ∈ depsOf S key.events, named ∈ pre
  | [], pre, key, post, parts =>
      (List.cons_ne_nil key post
        (List.append_eq_nil_iff.mp (show pre ++ key :: post = [] from parts.symm)).2).elim
  | .absorb _ :: before, pre, key, post, parts => depsOf_closed before pre key post parts
  | .squeeze count :: before, pre, key, post, parts => by
      intro named member
      cases found : (split before count : Option (Sponge.Base L B C × Sponge.Sub S C)) with
      | none =>
          have same : depsOf S (.squeeze count :: before) =
              (depsOf S before : List (Sponge.Base L B C)) := by
            show depsOf S before ++ ((split before count :
              Option (Sponge.Base L B C × Sponge.Sub S C)).map Prod.fst).toList = _
            rw [found]
            exact List.append_nil _
          exact depsOf_closed before pre key post (same.symm.trans parts) named member
      | some pair =>
          obtain ⟨base, sub⟩ := pair
          have grown : depsOf S (.squeeze count :: before) =
              (depsOf S before : List (Sponge.Base L B C)) ++ [base] := by
            show depsOf S before ++ ((split before count :
              Option (Sponge.Base L B C × Sponge.Sub S C)).map Prod.fst).toList = _
            rw [found]
            rfl
          rw [grown] at parts
          rcases List.eq_nil_or_concat post with empty | ⟨front, last, longer⟩
          · subst empty
            obtain ⟨same, single⟩ := List.append_inj' parts rfl
            have equal : base = key := List.head_eq_of_cons_eq single
            subst equal
            rw [← same]
            have suffix : base.events <:+ before := by
              rw [(Sponge.split_eq_some.mp found).1]
              exact List.suffix_append _ _
            exact (depsOf_prefix S suffix).subset member
          · subst longer
            have regroup : pre ++ key :: (front ++ [last]) = (pre ++ key :: front) ++ [last] := by
              simp
            rw [List.concat_eq_append, regroup] at parts
            exact depsOf_closed before pre key front (List.append_inj' parts rfl).1 named member

theorem length_depsOf : ∀ events : List Duplex.Event,
    (depsOf S events : List (Sponge.Base L B C)).length ≤ events.length
  | [] => Nat.le_refl 0
  | e :: rest => by
      obtain ⟨extra, short, grown⟩ := depsOf_cons (L := L) (B := B) (C := C) S e rest
      rw [grown, List.length_append, List.length_cons]
      exact Nat.add_le_add (length_depsOf rest) short

/-- The keys at which the address of the key `base` reads the table of the sponge: the bases of
the squeezes inside its events, the oldest first (Akita Appendix E.1, (F1)). -/
noncomputable def deps20 (base : Sponge.Base L B C) : List (Sponge.Base L B C) :=
  depsOf S base.events

theorem deps20_closed : Simulate.DepsClosed (deps20 (L := L) (B := B) (C := C) S) :=
  fun point pre key post parts => depsOf_closed S point.events pre key post parts

/-- A key names at most `L` keys. -/
theorem length_deps20 (point : Sponge.Base L B C) : (deps20 S point).length ≤ L :=
  (length_depsOf S point.events).trans point.length_events

end Deps

section Count

/-- The first event is an absorb. -/
def startsAbsorb : List Duplex.Event → Bool
  | .absorb _ :: _ => true
  | _ => false

/-- The count of the squeezes directly after an absorb, in events with the newest first. -/
def stageCount : List Duplex.Event → ℕ
  | [] => 0
  | .absorb _ :: before => stageCount before
  | .squeeze _ :: before => stageCount before + (if startsAbsorb before then 1 else 0)

theorem stageCount_absorbs : ∀ (new : List ByteArray) (rest : List Duplex.Event),
    stageCount (new.map Duplex.Event.absorb ++ rest) = stageCount rest
  | [], _ => rfl
  | _ :: new, rest => stageCount_absorbs new rest

theorem stageCount_squeezes {rest : List Duplex.Event} (starts : startsAbsorb rest = true) :
    ∀ counts : List ℕ, counts ≠ [] →
      stageCount (counts.map Duplex.Event.squeeze ++ rest) = stageCount rest + 1
  | [], empty => absurd rfl empty
  | [c], _ => by
      have one : (if startsAbsorb rest then 1 else 0) = 1 := by simp [starts]
      show stageCount rest + (if startsAbsorb rest then 1 else 0) = _
      rw [one]
  | c :: c' :: counts, _ => by
      have step : stageCount (Duplex.Event.squeeze c ::
          ((c' :: counts).map Duplex.Event.squeeze ++ rest)) =
          stageCount ((c' :: counts).map Duplex.Event.squeeze ++ rest) := rfl
      exact step.trans (stageCount_squeezes starts (c' :: counts) (List.cons_ne_nil _ _))

variable {pub : Public} {proof : ByteArray} {messages : Array Message}
  {a : List Duplex.Event → ℕ → ByteArray}

/-- `eventsAt20_succ_parts` with one squeeze or more: the squeeze of the work of the stage `k`
is in the events before the squeeze of the work of the stage `k + 1`. -/
theorem eventsAt20_succ_squeezes (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) {k : ℕ} (inside : k + 1 < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub (k + 1)))
      (channelStart pub)).1 = .ok ()) :
    ∃ (new : List ByteArray) (counts : List ℕ), counts ≠ [] ∧
      eventsAt20 pub proof messages (k + 1) = new.map Duplex.Event.absorb ++
        (counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages k) := by
  obtain ⟨le, quiet⟩ := quiet20_of erased inside
  obtain ⟨site, level, at_⟩ := bits k (by omega)
  obtain ⟨first, -, cut⟩ := stateBefore_le pub proof messages pub.ops le ran
  have drawn := (draw_squeeze20 at_ (bits20_ne_zero (by omega : k < 473))
    (Nat.le_add_right _ _) (draws20_of erased (by omega : k < 473) (Nat.le_refl _)) first
    ideal).1
  obtain ⟨counts, nonempty, shape⟩ : ∃ counts : List ℕ, counts ≠ [] ∧
      (stateBefore pub proof messages pub.ops
        (stageIndex20 pub k + 1 + drawCount20 k)).sponge.events =
        counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages k :=
    ⟨(((pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1 + drawCount20 k)).drop
        (stageIndex20 pub k + 1)).reverse.map fun x => x.1.drawBytes) ++ [8],
      fun empty => by simp at empty, by
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
  exact ⟨n, counts, nonempty, e⟩

/-- On a script of `layout20` with the work bits of the model and an ideal sponge with no event
before the channel: the events before the squeeze of the work of the stage `k` of a run with no
stop before this work have `k` squeezes directly after an absorb. -/
theorem stageCount_eventsAt20 (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = []) :
    ∀ (k : ℕ), k < 473 →
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
        (channelStart pub)).1 = .ok () →
      stageCount (eventsAt20 pub proof messages k) = k
  | 0, _, _ => by
      obtain ⟨new, events⟩ := eventsAt20_zero_parts proof messages erased
      rw [events, fresh]
      exact stageCount_absorbs new []
  | k + 1, inside, ran => by
      obtain ⟨new, counts, nonempty, events⟩ :=
        eventsAt20_succ_squeezes erased bits ideal inside ran
      obtain ⟨site, level, at_⟩ := bits k (by omega)
      obtain ⟨m, -, head, -, -⟩ := work_squeeze20 at_ (bits20_ne_zero (by omega : k < 473))
        (stateBefore_le pub proof messages pub.ops
          (stageIndex20_succ_le pub (Nat.lt_succ_self k) inside) ran).1 ideal
      have starts : startsAbsorb (eventsAt20 pub proof messages k) = true := by
        rw [head]
        rfl
      have before := stageCount_eventsAt20 erased bits ideal fresh k (by omega)
        (stateBefore_le pub proof messages pub.ops
          (stageIndex20_mono pub (Nat.le_succ k) inside) ran).1
      rw [events, stageCount_absorbs, stageCount_squeezes starts counts nonempty, before]

variable {L B C : ℕ}

/-- The stage of a key: the count of the squeezes directly after an absorb in its events. -/
def stageOf (base : Sponge.Base L B C) : ℕ := stageCount base.events

/-- A base with the events before the squeeze of the work of the stage `k` has the stage `k`. -/
theorem stageOf_base (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = [])
    {k : ℕ} (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
      (channelStart pub)).1 = .ok ())
    {base : Sponge.Base L B C} (events : base.events = eventsAt20 pub proof messages k) :
    stageOf base = k := by
  unfold stageOf
  rw [events]
  exact stageCount_eventsAt20 erased bits ideal fresh k inside ran

end Count

section Maps

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {L B C S : ℕ}
  (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
  (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
  (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
  (blockL : Fin Key20.terminal.blocks → Coord)

local notation "Reply" => (Option Coord → CodeSlot E)

/-- The map `φ` of the recording prover at the key `base`: the stage map of the stage of the
key. -/
noncomputable def phiAt (base : Sponge.Base L B C) : Reply × Sponge.Y S C → Sponge.Y S C :=
  phi20 block block₁ block₂ block₃ block₄ block₅ blockL (stageOf base)

/-- Each fibre of `phiAt` has `|Reply|` elements. -/
theorem card_phiAt (base : Sponge.Base L B C) (value : Sponge.Y S C) :
    Nat.card {x : Reply × Sponge.Y S C //
        phiAt block block₁ block₂ block₃ block₄ block₅ blockL base x = value} =
      Fintype.card Reply :=
  card_phi20 block block₁ block₂ block₃ block₄ block₅ blockL (stageOf base) value

end Maps

section Resolve

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {L B C S D : ℕ} {pub : Public}

/-- A record of digest queries as byte arrays. -/
def recordOf (record : List (Sponge.Str D × Bytes 32)) : List (ByteArray × ByteArray) :=
  record.map fun entry => (entry.1.array, Bytes.array entry.2)

/-- `resolve` of the recording prover for `key20.aks`: the address of the key `base` with the
answers of the table `fs` of the sponge and the last blocks of the record. -/
noncomputable def resolveAt (total : ℕ) (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C)
    (record : List (Sponge.Str D × Bytes 32)) (base : Sponge.Base L B C) :
    Option (Address (Key20.Wire E) total) :=
  resolveWith20 total pub (answerOf fs) (recordOf record) base.events

/-- `resolveAt` reads the table of the sponge only at the keys of `deps20`. -/
theorem resolveAt_local (bits : Bits20 pub) (total : ℕ) :
    Simulate.DepsLocal (resolveAt (E := E) (S := S) (D := D) total pub)
      (deps20 (L := L) (B := B) (C := C) S) := by
  intro fs other record point same
  exact resolveWith20_agree bits
    (agreeOn_answerOf fun before count base sub inside found =>
      same base (mem_depsOf S inside found)) _ total

open Classical in
/-- The key of an address for the table `fs`: the base with the events `back20`, and the empty
base when no base has these events. -/
noncomputable def backAt {total : ℕ} (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C)
    (address : Address (Key20.Wire E) total) : Sponge.Base L B C :=
  if found : ∃ base : Sponge.Base L B C,
      base.events = back20 (pub.reideal (some (answerOf fs))) address then
    Classical.choose found
  else Sponge.Base.empty

/-- On a script of `layout20`, for each table and each record: an address that `resolveAt` gives
for a key is the address of this key only. -/
theorem resolveAt_back (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {total : ℕ} (long : 472 ≤ total) (fs : Sponge.Base L B C → Sponge.Y S C)
    (record : List (Sponge.Str D × Bytes 32)) (point : Sponge.Base L B C)
    (address : Address (Key20.Wire E) total)
    (resolved : resolveAt total pub fs record point = some address) :
    backAt pub fs address = point := by
  have events := resolveWith20_back (a := answerOf fs) erased long (recordOf record)
    point.events address resolved
  have found : ∃ base : Sponge.Base L B C,
      base.events = back20 (pub.reideal (some (answerOf fs))) address := ⟨point, events.symm⟩
  unfold backAt
  rw [dite_eq_left found]
  exact Sponge.Base.events_injective ((Classical.choose_spec found).trans events)

end Resolve

end NightstreamFPrime.Spec.Compression.Akita.Replay
