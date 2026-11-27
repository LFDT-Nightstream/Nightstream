import NightstreamFPrime.Spec.Compression.Akita.Replay.Coupling
import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerFair

/-! The ideal sponge of the experiment of the byte verifier as a finite uniform table (Akita,
ePrint 2026/1983, revision of 2026-11-23, Section 8.2 `[KECCAK]` and Lemma 10.47;
`Replay/Keccak.lean`, `Duplex.ideal`; `docs/akita-work/w21/design.md`, section 2). No Akita
schedule number is in this file.

- `Short α n` are the lists of at most `n` elements. `Str B` are the byte strings of at most `B`
  bytes, and `Ev B C` the absorbs of at most `B` bytes and the squeezes of at most `C` bytes.
- A query of the sponge is a pair of an event list (the newest first) and a count. `Base L B C`
  are the lists of at most `L` events whose first event is not a squeeze. `Sub S C` has the counts
  of at most `S` squeezes and the count of the query, each at most `C`. `key` gives the query of a
  base and a sub-key: the squeezes of the sub-key, then the events of the base.
- `split` is the partial inverse of `key` (`split_eq_some`), and gives `none` for a query that is
  not in the bounds. `split_of` gives the base and the sub-key of a query in this form, and
  `split_of_bounds` shows that each query in the bounds has them.
- `Y S C` is the answer type of one base: a string of `C` bytes for each sub-key.
  `answerOf fs` is the answer function of the table `fs`: the first `count` bytes of the string
  of the base and the sub-key, and `count` bytes `0xFF` for a query that is not in the bounds.
  `worked_answerOf_none`: the work check with one bit or more fails on these bytes.
  `spongeOf fs` is the sponge of the code with these answers.
- `Lens Y Y₀` is a part `Y₀` of `Y` that can be read and written. `Lens.lift` puts a map on the
  part on `Y`, and `Lens.card_lift` keeps the sizes of the fibres. `Lens.ofEquiv` is the lens of
  an equivalence `Y ≃ Y₀ × Z`, and `Lens.exists_lift` is the lift for it. `Lens.prefixes` is the
  lens of the first bytes of the strings at different keys.
- `stageLens` is the lens of a stage with a work squeeze of 8 bytes and `n` draws of `m i`
  bytes, at the sub-keys `workSub` and `drawSub`. `oneLens` is the lens of a stage with one draw.
  `answerOf_stageLift` and `answerOf_oneLift`: the answers of a table whose entry at a base is
  the lift of `φ₀` are the strings of `φ₀`, and `card_stageLift`, `card_oneLift` give the fibres.
- `expect_table`: a uniform table of answers for each base is a uniform table on the pairs of a
  base and a sub-key. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay.Sponge

open NightstreamFPrime.Spec.Compression.Akita
open scoped BigOperators
open Bytes

/-- The lists of at most `n` elements. -/
abbrev Short (α : Type) (n : ℕ) : Type := Σ k : Fin (n + 1), List.Vector α k

namespace Short

variable {α : Type} {n : ℕ}

def toList (x : Short α n) : List α := x.2.1

theorem length_toList (x : Short α n) : x.toList.length ≤ n :=
  x.2.2.trans_le (Nat.le_of_lt_succ x.1.isLt)

def ofList (l : List α) (short : l.length ≤ n) : Short α n :=
  ⟨⟨l.length, Nat.lt_succ_of_le short⟩, ⟨l, rfl⟩⟩

theorem toList_ofList (l : List α) (short : l.length ≤ n) : (ofList l short).toList = l := rfl

theorem toList_injective : Function.Injective (toList (α := α) (n := n)) := by
  intro x y same
  have first : x.1 = y.1 :=
    Fin.ext (x.2.2.symm.trans ((congrArg List.length same).trans y.2.2))
  obtain ⟨i, v⟩ := x
  obtain ⟨i', v'⟩ := y
  obtain rfl : i = i' := first
  exact congrArg (Sigma.mk i) (Subtype.ext same)

end Short

/-- The byte strings of at most `B` bytes. -/
abbrev Str (B : ℕ) : Type := Short (Fin 256) B

theorem byte_injective : Function.Injective byte := fun a b same =>
  Fin.ext (show (byte a).toNat = (byte b).toNat from congrArg UInt8.toNat same)

/-- The `ByteArray` of a string. -/
def Str.array {B : ℕ} (x : Str B) : ByteArray := ⟨(x.toList.map byte).toArray⟩

theorem Str.array_injective {B : ℕ} : Function.Injective (Str.array (B := B)) := fun x y same =>
  Short.toList_injective (List.map_injective_iff.mpr byte_injective
    (show x.toList.map byte = y.toList.map byte from
      congrArg (fun b : ByteArray => b.data.toList) same))

theorem Str.size_array {B : ℕ} (x : Str B) : x.array.size ≤ B := by
  have size : x.array.size = x.toList.length := by
    show (x.toList.map byte).toArray.size = x.toList.length
    rw [List.size_toArray, List.length_map]
  exact size.trans_le x.length_toList

/-- Each `ByteArray` of at most `B` bytes is the `ByteArray` of a string. -/
theorem Str.exists_array {B : ℕ} (bytes : ByteArray) (short : bytes.size ≤ B) :
    ∃ x : Str B, x.array = bytes := by
  refine ⟨Short.ofList (bytes.data.toList.map fun u => (⟨u.toNat, u.toNat_lt⟩ : Fin 256)) ?_, ?_⟩
  · rw [List.length_map, Array.length_toList]
    exact short
  · show (⟨((bytes.data.toList.map fun u => (⟨u.toNat, u.toNat_lt⟩ : Fin 256)).map
      byte).toArray⟩ : ByteArray) = bytes
    have back : (byte ∘ fun u : UInt8 => (⟨u.toNat, u.toNat_lt⟩ : Fin 256)) = id :=
      funext fun u => UInt8.toNat_inj.mp rfl
    rw [List.map_map, back, List.map_id, Array.toArray_toList]

/-- An absorb of at most `B` bytes or a squeeze of at most `C` bytes. -/
abbrev Ev (B C : ℕ) : Type := Str B ⊕ Fin (C + 1)

/-- A squeeze, and not an absorb. -/
def isSqueeze : Duplex.Event → Bool
  | .squeeze _ => true
  | .absorb _ => false

/-- The event of the sponge. -/
def Ev.event {B C : ℕ} : Ev B C → Duplex.Event
  | .inl input => .absorb input.array
  | .inr count => .squeeze count

theorem Ev.event_injective {B C : ℕ} : Function.Injective (Ev.event (B := B) (C := C)) := by
  rintro (x | c) (y | d) same
  · exact congrArg Sum.inl (Str.array_injective (Duplex.Event.absorb.inj same))
  · exact (Bool.false_ne_true (congrArg isSqueeze same)).elim
  · exact (Bool.false_ne_true (congrArg isSqueeze same.symm)).elim
  · exact congrArg Sum.inr (Fin.ext (Duplex.Event.squeeze.inj same))

/-- Each event in the bounds is the event of an element of `Ev B C`. -/
theorem Ev.exists_event {B C : ℕ} (e : Duplex.Event)
    (absorbs : ∀ input, e = .absorb input → input.size ≤ B)
    (squeezes : ∀ c, e = .squeeze c → c ≤ C) : ∃ ev : Ev B C, ev.event = e := by
  cases e with
  | absorb input =>
      obtain ⟨x, same⟩ := Str.exists_array input (absorbs input rfl)
      exact ⟨.inl x, congrArg Duplex.Event.absorb same⟩
  | squeeze c => exact ⟨.inr ⟨c, Nat.lt_succ_of_le (squeezes c rfl)⟩, rfl⟩

/-- The lists of at most `L` events whose first event is not a squeeze. -/
abbrev Base (L B C : ℕ) : Type :=
  {l : Short (Ev B C) L // l.toList.head?.all Sum.isLeft = true}

instance (L B C : ℕ) : DecidableEq (Base L B C) := fun a b =>
  decidable_of_iff (a.1 = b.1) Subtype.ext_iff.symm

/-- The squeeze counts of at most `S` squeezes and the count of the query, each at most `C`. -/
abbrev Sub (S C : ℕ) : Type := Short (Fin (C + 1)) S × Fin (C + 1)

/-- The answers of one base: a string of `C` bytes for each sub-key. -/
abbrev Y (S C : ℕ) : Type := Sub S C → Bytes C

variable {L B C S : ℕ}

/-- The events of a base, the newest first. -/
def Base.events (base : Base L B C) : List Duplex.Event := base.1.toList.map Ev.event

theorem Base.events_injective : Function.Injective (Base.events (L := L) (B := B) (C := C)) :=
  fun _ _ same => Subtype.ext (Short.toList_injective
    (List.map_injective_iff.mpr Ev.event_injective same))

theorem Base.length_events (base : Base L B C) : base.events.length ≤ L :=
  (List.length_map _).trans_le base.1.length_toList

/-- The first event of a base is not a squeeze. -/
theorem Base.events_ne (base : Base L B C) (c : ℕ) (rest : List Duplex.Event) :
    base.events ≠ .squeeze c :: rest := by
  intro same
  have ok := base.2
  have mapped : base.1.toList.map Ev.event = .squeeze c :: rest := same
  cases list : base.1.toList with
  | nil =>
      rw [list, List.map_nil] at mapped
      cases mapped
  | cons e tail =>
      rw [list, List.map_cons] at mapped
      rw [list] at ok
      cases e with
      | inl x => exact Bool.false_ne_true (congrArg isSqueeze (List.cons.inj mapped).1)
      | inr d => exact Bool.false_ne_true ok

/-- Each list of events in the bounds whose first event is not a squeeze is the list of a
base. -/
theorem Base.exists_events (events : List Duplex.Event) (short : events.length ≤ L)
    (head : ∀ c rest, events ≠ .squeeze c :: rest)
    (absorbs : ∀ input, .absorb input ∈ events → input.size ≤ B)
    (squeezes : ∀ c, .squeeze c ∈ events → c ≤ C) :
    ∃ base : Base L B C, base.events = events := by
  have lists : ∀ l : List Duplex.Event, (∀ input, .absorb input ∈ l → input.size ≤ B) →
      (∀ c, .squeeze c ∈ l → c ≤ C) → ∃ evs : List (Ev B C), evs.map Ev.event = l := by
    intro l
    induction l with
    | nil => exact fun _ _ => ⟨[], rfl⟩
    | cons e rest induction =>
        intro absorbs squeezes
        obtain ⟨ev, same⟩ := Ev.exists_event (B := B) (C := C) e
          (fun input equal => absorbs input (List.mem_cons.mpr (Or.inl equal.symm)))
          (fun c equal => squeezes c (List.mem_cons.mpr (Or.inl equal.symm)))
        obtain ⟨evs, others⟩ := induction
          (fun input member => absorbs input (List.mem_cons_of_mem _ member))
          (fun c member => squeezes c (List.mem_cons_of_mem _ member))
        exact ⟨ev :: evs, by rw [List.map_cons, same, others]⟩
  obtain ⟨evs, same⟩ := lists events absorbs squeezes
  have length : evs.length ≤ L := by
    have sizes := congrArg List.length same
    rw [List.length_map] at sizes
    rw [sizes]
    exact short
  refine ⟨⟨Short.ofList evs length, ?_⟩, same⟩
  show evs.head?.all Sum.isLeft = true
  cases evs with
  | nil => rfl
  | cons ev tail =>
      cases ev with
      | inl x => rfl
      | inr d => exact (head d _ same.symm).elim

/-- The base with no event. -/
def Base.empty : Base L B C := ⟨Short.ofList [] (Nat.zero_le L), rfl⟩

theorem Base.events_empty : (Base.empty : Base L B C).events = [] := rfl

/-- The squeeze counts of a sub-key, the newest first. -/
def Sub.counts (sub : Sub S C) : List ℕ := sub.1.toList.map Fin.val

/-- The count of the query of a sub-key. -/
def Sub.count (sub : Sub S C) : ℕ := sub.2

/-- The squeezes of a sub-key, the newest first. -/
def Sub.events (sub : Sub S C) : List Duplex.Event := sub.counts.map Duplex.Event.squeeze

theorem Sub.ext {sub sub' : Sub S C} (counts : sub.counts = sub'.counts)
    (count : sub.count = sub'.count) : sub = sub' :=
  Prod.ext (Short.toList_injective (List.map_injective_iff.mpr Fin.val_injective counts))
    (Fin.ext count)

/-- The sub-key of the squeeze counts `counts` and the count `count` of the query. -/
def Sub.of (counts : List ℕ) (count : ℕ) (short : counts.length ≤ S)
    (each : ∀ c ∈ counts, c ≤ C) (le : count ≤ C) : Sub S C :=
  (Short.ofList (counts.attach.map fun c => ⟨c.1, Nat.lt_succ_of_le (each c.1 c.2)⟩)
    (by rw [List.length_map, List.length_attach]; exact short), ⟨count, Nat.lt_succ_of_le le⟩)

theorem Sub.counts_of (counts : List ℕ) (count : ℕ) (short : counts.length ≤ S)
    (each : ∀ c ∈ counts, c ≤ C) (le : count ≤ C) :
    (Sub.of counts count short each le).counts = counts := by
  show (counts.attach.map fun c => (⟨c.1, Nat.lt_succ_of_le (each c.1 c.2)⟩ : Fin (C + 1))).map
    Fin.val = counts
  rw [List.map_map]
  exact List.attach_map_subtype_val counts

theorem Sub.count_of (counts : List ℕ) (count : ℕ) (short : counts.length ≤ S)
    (each : ∀ c ∈ counts, c ≤ C) (le : count ≤ C) :
    (Sub.of counts count short each le).count = count := rfl

/-- The query of a base and a sub-key: the squeezes of the sub-key, then the events of the base,
and the count. -/
def key (p : Base L B C × Sub S C) : List Duplex.Event × ℕ :=
  (p.2.events ++ p.1.events, p.2.count)

/-- Squeezes before a list whose first event is not a squeeze give the squeezes and the list. -/
theorem squeezes_append_inj {tail tail' : List Duplex.Event}
    (head : ∀ c rest, tail ≠ .squeeze c :: rest) (head' : ∀ c rest, tail' ≠ .squeeze c :: rest) :
    ∀ {cs cs' : List ℕ}, cs.map Duplex.Event.squeeze ++ tail =
      cs'.map Duplex.Event.squeeze ++ tail' → cs = cs' ∧ tail = tail'
  | [], [], same => ⟨rfl, same⟩
  | [], c :: rest, same => (head c _ same).elim
  | c :: rest, [], same => (head' c _ same.symm).elim
  | c :: rest, c' :: rest', same => by
      simp only [List.map_cons, List.cons_append] at same
      obtain ⟨first, others⟩ := List.cons.inj same
      obtain ⟨equal, tails⟩ := squeezes_append_inj head head' others
      exact ⟨by rw [Duplex.Event.squeeze.inj first, equal], tails⟩

theorem key_injective : Function.Injective (key (L := L) (B := B) (C := C) (S := S)) := by
  rintro ⟨base, sub⟩ ⟨base', sub'⟩ same
  obtain ⟨counts, events⟩ := squeezes_append_inj base.events_ne base'.events_ne
    (show sub.counts.map Duplex.Event.squeeze ++ base.events =
      sub'.counts.map Duplex.Event.squeeze ++ base'.events from congrArg Prod.fst same)
  obtain rfl := Base.events_injective events
  obtain rfl := Sub.ext counts (congrArg Prod.snd same)
  rfl

open Classical in
/-- The base and the sub-key of a query, and `none` for a query that is not in the bounds. -/
noncomputable def split (events : List Duplex.Event) (count : ℕ) :
    Option (Base L B C × Sub S C) :=
  if h : ∃ p, key p = (events, count) then some h.choose else none

open Classical in
theorem split_eq_some {events : List Duplex.Event} {count : ℕ} {base : Base L B C}
    {sub : Sub S C} :
    split events count = some (base, sub) ↔
      events = sub.events ++ base.events ∧ count = sub.count := by
  have query : key (base, sub) = (events, count) ↔
      events = sub.events ++ base.events ∧ count = sub.count :=
    ⟨fun same => ⟨(congrArg Prod.fst same).symm, (congrArg Prod.snd same).symm⟩,
      fun parts => Prod.ext parts.1.symm parts.2.symm⟩
  refine Iff.trans ⟨fun found => ?_, fun same => ?_⟩ query
  · by_cases h : ∃ p : Base L B C × Sub S C, key p = (events, count)
    · rw [split, dite_eq_left h] at found
      exact (congrArg key (Option.some.inj found)).symm.trans h.choose_spec
    · rw [split, dite_eq_right h] at found
      cases found
  · have h : ∃ p : Base L B C × Sub S C, key p = (events, count) := ⟨_, same⟩
    rw [split, dite_eq_left h]
    exact congrArg some (key_injective (h.choose_spec.trans same.symm))

/-- Two queries with one base and one sub-key are equal. -/
theorem split_injective {events events' : List Duplex.Event} {count count' : ℕ}
    {p : Base L B C × Sub S C} (first : split events count = some p)
    (second : split events' count' = some p) : events = events' ∧ count = count' :=
  ⟨(split_eq_some.mp first).1.trans (split_eq_some.mp second).1.symm,
    (split_eq_some.mp first).2.trans (split_eq_some.mp second).2.symm⟩

/-- The squeezes `counts` before the events of a base have this base and the sub-key of
`counts`. -/
theorem split_of (base : Base L B C) (counts : List ℕ) (count : ℕ) (short : counts.length ≤ S)
    (each : ∀ c ∈ counts, c ≤ C) (le : count ≤ C) :
    split (counts.map Duplex.Event.squeeze ++ base.events) count =
      some (base, Sub.of counts count short each le) :=
  split_eq_some.mpr ⟨by rw [Sub.events, Sub.counts_of], rfl⟩

/-- A query in the bounds has a base and a sub-key: squeezes `counts` before a list `tail` of
events whose first event is not a squeeze. -/
theorem split_of_bounds (counts : List ℕ) (tail : List Duplex.Event) (count : ℕ)
    (few : counts.length ≤ S) (each : ∀ c ∈ counts, c ≤ C) (le : count ≤ C)
    (short : tail.length ≤ L) (head : ∀ c rest, tail ≠ .squeeze c :: rest)
    (absorbs : ∀ input, .absorb input ∈ tail → input.size ≤ B)
    (squeezes : ∀ c, .squeeze c ∈ tail → c ≤ C) :
    ∃ base : Base L B C, base.events = tail ∧
      split (counts.map Duplex.Event.squeeze ++ tail) count =
        some (base, Sub.of counts count few each le) := by
  obtain ⟨base, rfl⟩ := Base.exists_events tail short head absorbs squeezes
  exact ⟨base, rfl, split_of base counts count few each le⟩

/-- A query with a base and a sub-key is in the bounds. -/
theorem split_bounds {events : List Duplex.Event} {count : ℕ} {base : Base L B C} {sub : Sub S C}
    (found : split events count = some (base, sub)) :
    events.length ≤ S + L ∧ count ≤ C := by
  obtain ⟨parts, same⟩ := split_eq_some.mp found
  refine ⟨?_, same ▸ Nat.le_of_lt_succ sub.2.isLt⟩
  rw [parts, List.length_append, Sub.events, Sub.counts, List.length_map, List.length_map]
  exact Nat.add_le_add sub.1.length_toList base.length_events

/-- The first `n` bytes of a string. -/
def take {n : ℕ} (le : n ≤ C) (u : Bytes C) : Bytes n := fun i => u (Fin.castLE le i)

/-- The answer function of the table `fs`: the first `count` bytes of the string of the base and
the sub-key of the query, and `count` bytes `0xFF` for a query that is not in the bounds. -/
noncomputable def answerOf (fs : Base L B C → Y S C) (events : List Duplex.Event) (count : ℕ) :
    ByteArray :=
  match (split events count : Option (Base L B C × Sub S C)) with
  | some (base, sub) => array (take (Nat.le_of_lt_succ sub.2.isLt) (fs base sub))
  | none => array fun _ : Fin count => (255 : Fin 256)

theorem answerOf_some (fs : Base L B C → Y S C) {events : List Duplex.Event} {count : ℕ}
    {base : Base L B C} {sub : Sub S C} (found : split events count = some (base, sub)) :
    answerOf fs events count = array (take (Nat.le_of_lt_succ sub.2.isLt) (fs base sub)) := by
  simp only [answerOf, found]

theorem answerOf_none (fs : Base L B C → Y S C) {events : List Duplex.Event} {count : ℕ}
    (missing : (split events count : Option (Base L B C × Sub S C)) = none) :
    answerOf fs events count = array fun _ : Fin count => (255 : Fin 256) := by
  simp only [answerOf, missing]

/-- The answer for the squeezes `counts` before the events of a base. -/
theorem answerOf_of (fs : Base L B C → Y S C) (base : Base L B C) (counts : List ℕ) (count : ℕ)
    (short : counts.length ≤ S) (each : ∀ c ∈ counts, c ≤ C) (le : count ≤ C) :
    answerOf fs (counts.map Duplex.Event.squeeze ++ base.events) count =
      array (take le (fs base (Sub.of counts count short each le))) :=
  answerOf_some fs (split_of base counts count short each le)

theorem size_answerOf (fs : Base L B C → Y S C) (events : List Duplex.Event) (count : ℕ) :
    (answerOf fs events count).size = count := by
  cases found : (split events count : Option (Base L B C × Sub S C)) with
  | none => rw [answerOf_none fs found, size_array]
  | some p =>
      obtain ⟨base, sub⟩ := p
      rw [answerOf_some fs found, size_array]
      exact (split_eq_some.mp found).2.symm

/-- The work check with one bit or more fails on 8 bytes `0xFF`. -/
theorem not_worked_ones {bits : ℕ} (positive : 1 ≤ bits) :
    worked (array fun _ : Fin 8 => (255 : Fin 256)) bits ≠ true := by
  intro passed
  have zero : value (fun _ : Fin 8 => (255 : Fin 256)) / 2 ^ (0 * bits) % 2 ^ bits = 0 :=
    congrArg Fin.val (congrFun ((worked_iff _ bits).mp passed) 0)
  rw [Nat.zero_mul, pow_zero, Nat.div_one] at zero
  have low := value_digit (fun _ : Fin 8 => (255 : Fin 256)) 0
  rw [pow_zero, Nat.div_one, show byteAt (fun _ : Fin 8 => (255 : Fin 256)) 0 = 255 from rfl]
    at low
  have two : 2 ∣ value (fun _ : Fin 8 => (255 : Fin 256)) :=
    (dvd_pow_self 2 (Nat.one_le_iff_ne_zero.mp positive)).trans (Nat.dvd_of_mod_eq_zero zero)
  omega

/-- The work check with one bit or more fails on the answer for a query that is not in the
bounds. -/
theorem worked_answerOf_none (fs : Base L B C → Y S C) {events : List Duplex.Event}
    (missing : (split events 8 : Option (Base L B C × Sub S C)) = none) {bits : ℕ}
    (positive : 1 ≤ bits) : worked (answerOf fs events 8) bits = false := by
  rw [answerOf_none fs missing]
  exact Bool.eq_false_iff.mpr (not_worked_ones positive)

/-- A work check with one bit or more that passes on the answer for a query gives a base and a
sub-key of the query. -/
theorem split_of_worked (fs : Base L B C → Y S C) {events : List Duplex.Event} {bits : ℕ}
    (positive : 1 ≤ bits) (passed : worked (answerOf fs events 8) bits = true) :
    ∃ (base : Base L B C) (sub : Sub S C), split events 8 = some (base, sub) := by
  cases found : (split events 8 : Option (Base L B C × Sub S C)) with
  | none => rw [worked_answerOf_none fs found positive] at passed; cases passed
  | some p => exact ⟨p.1, p.2, rfl⟩

/-- The sponge of the code with the answers of the table `fs`. -/
noncomputable def spongeOf (fs : Base L B C → Y S C) : Duplex :=
  { Duplex.empty with ideal := some (answerOf fs) }

theorem spongeOf_ideal (fs : Base L B C → Y S C) : (spongeOf fs).ideal = some (answerOf fs) :=
  rfl

theorem spongeOf_events (fs : Base L B C → Y S C) : (spongeOf fs).events = [] := rfl

/-- A part `Y₀` of `Y` that can be read and written. -/
structure Lens (Y Y₀ : Type) where
  get : Y → Y₀
  put : Y₀ → Y → Y
  get_put : ∀ a y, get (put a y) = a
  put_get : ∀ y, put (get y) y = y
  put_put : ∀ a b y, put a (put b y) = put a y

namespace Lens

variable {R Y Y₀ Y₁ Z : Type}

/-- The map `φ₀` of the part, on `Y`: it changes the part only. -/
def lift (ℓ : Lens Y Y₀) (φ₀ : R × Y₀ → Y₀) (x : R × Y) : Y := ℓ.put (φ₀ (x.1, ℓ.get x.2)) x.2

theorem get_lift (ℓ : Lens Y Y₀) (φ₀ : R × Y₀ → Y₀) (r : R) (y : Y) :
    ℓ.get (ℓ.lift φ₀ (r, y)) = φ₀ (r, ℓ.get y) :=
  ℓ.get_put _ _

/-- A fibre of the lift has the size of a fibre of the map of the part. -/
theorem card_lift (ℓ : Lens Y Y₀) (φ₀ : R × Y₀ → Y₀) (y : Y) :
    Nat.card {x // ℓ.lift φ₀ x = y} = Nat.card {x // φ₀ x = ℓ.get y} :=
  Nat.card_congr
    { toFun := fun x => ⟨(x.1.1, ℓ.get x.1.2),
        (ℓ.get_lift φ₀ x.1.1 x.1.2).symm.trans (congrArg ℓ.get x.2)⟩
      invFun := fun x => ⟨(x.1.1, ℓ.put x.1.2 y), by
        show ℓ.put (φ₀ (x.1.1, ℓ.get (ℓ.put x.1.2 y))) (ℓ.put x.1.2 y) = y
        rw [ℓ.get_put, ℓ.put_put, x.2, ℓ.put_get]⟩
      left_inv := fun x => Subtype.ext (Prod.ext rfl (by
        have image : y = ℓ.put (φ₀ (x.1.1, ℓ.get x.1.2)) x.1.2 := x.2.symm
        show ℓ.put (ℓ.get x.1.2) y = x.1.2
        exact (congrArg (ℓ.put (ℓ.get x.1.2)) image).trans
          ((ℓ.put_put _ _ _).trans (ℓ.put_get _))))
      right_inv := fun x => Subtype.ext (Prod.ext rfl (ℓ.get_put _ _)) }

/-- The lens of the first component of an equivalence. -/
def ofEquiv (e : Y ≃ Y₀ × Z) : Lens Y Y₀ where
  get y := (e y).1
  put a y := e.symm (a, (e y).2)
  get_put a y := congrArg Prod.fst (e.apply_symm_apply (a, (e y).2))
  put_get y := e.symm_apply_apply y
  put_put a b y := by
    show e.symm (a, (e (e.symm (b, (e y).2))).2) = e.symm (a, (e y).2)
    rw [e.apply_symm_apply]

/-- For an equivalence `e : Y ≃ Y₀ × Z` and a map `φ₀` on `Y₀` whose fibres have `|R|` elements,
a map on `Y` exists that is `φ₀` on the first component, keeps the second component and whose
fibres have `|R|` elements. -/
theorem exists_lift [Fintype R] (e : Y ≃ Y₀ × Z) (φ₀ : R × Y₀ → Y₀)
    (fibres : ∀ y₀, Nat.card {x // φ₀ x = y₀} = Fintype.card R) :
    ∃ φ : R × Y → Y, (∀ r y, (e (φ (r, y))).1 = φ₀ (r, (e y).1)) ∧
      (∀ r y, (e (φ (r, y))).2 = (e y).2) ∧ ∀ y, Nat.card {x // φ x = y} = Fintype.card R :=
  ⟨(ofEquiv e).lift φ₀, fun r y => (ofEquiv e).get_lift φ₀ r y,
    fun r y => by
      show (e (e.symm (φ₀ (r, (e y).1), (e y).2))).2 = (e y).2
      rw [e.apply_symm_apply],
    fun y => ((ofEquiv e).card_lift φ₀ y).trans (fibres _)⟩

/-- The lens with the part in another form. -/
def map (ℓ : Lens Y Y₀) (ε : Y₀ ≃ Y₁) : Lens Y Y₁ where
  get y := ε (ℓ.get y)
  put a y := ℓ.put (ε.symm a) y
  get_put a y := (congrArg ε (ℓ.get_put (ε.symm a) y)).trans (ε.apply_symm_apply a)
  put_get y := (congrArg (fun a => ℓ.put a y) (ε.symm_apply_apply (ℓ.get y))).trans (ℓ.put_get y)
  put_put _ _ _ := ℓ.put_put _ _ _

section Prefixes

variable {K ι : Type} {C : ℕ} {m : ι → ℕ}

/-- The first `m i` bytes of the string at the key `s i`, for each `i`. -/
def read (s : ι → K) (le : ∀ i, m i ≤ C) (y : K → Bytes C) : ∀ i, Bytes (m i) :=
  fun i => take (le i) (y (s i))

open Classical in
/-- The strings `y` with the first `m i` bytes of the string at the key `s i` from `a`. -/
noncomputable def write (s : ι → K) (a : ∀ i, Bytes (m i)) (y : K → Bytes C) : K → Bytes C :=
  fun k j => if h : ∃ i, s i = k ∧ (j : ℕ) < m i then a h.choose ⟨j, h.choose_spec.2⟩ else y k j

theorem at_eq (a : ∀ i, Bytes (m i)) {i i' : ι} (same : i' = i) {j : ℕ} (p : j < m i')
    (q : j < m i) : a i' ⟨j, p⟩ = a i ⟨j, q⟩ := by
  subst same
  rfl

open Classical in
theorem read_write {s : ι → K} (injective : Function.Injective s) (le : ∀ i, m i ≤ C)
    (a : ∀ i, Bytes (m i)) (y : K → Bytes C) : read s le (write s a y) = a := by
  funext i j
  have h : ∃ i', s i' = s i ∧ (j : ℕ) < m i' := ⟨i, rfl, j.isLt⟩
  show (if h : ∃ i', s i' = s i ∧ (j : ℕ) < m i' then a h.choose ⟨j, h.choose_spec.2⟩
    else y (s i) (Fin.castLE (le i) j)) = a i j
  rw [dite_eq_left h]
  exact at_eq a (injective h.choose_spec.1) _ j.isLt

theorem write_read (s : ι → K) (le : ∀ i, m i ≤ C) (y : K → Bytes C) :
    write s (read s le y) y = y := by
  funext k j
  by_cases h : ∃ i, s i = k ∧ (j : ℕ) < m i
  · simp only [write, dite_eq_left h]
    exact congrFun (congrArg y h.choose_spec.1) j
  · simp only [write, dite_eq_right h]

theorem write_write (s : ι → K) (a b : ∀ i, Bytes (m i)) (y : K → Bytes C) :
    write s a (write s b y) = write s a y := by
  funext k j
  by_cases h : ∃ i, s i = k ∧ (j : ℕ) < m i
  · simp only [write, dite_eq_left h]
  · simp only [write, dite_eq_right h]

/-- The lens of the first `m i` bytes of the strings at the different keys `s i`. -/
noncomputable def prefixes (s : ι → K) (injective : Function.Injective s) (m : ι → ℕ)
    (le : ∀ i, m i ≤ C) : Lens (K → Bytes C) (∀ i, Bytes (m i)) where
  get := read s le
  put := write s
  get_put := read_write injective le
  put_get := write_read s le
  put_put := write_write s

end Prefixes

end Lens

section Stage

variable {n : ℕ}

/-- The counts of the draws before the draw `i`, the newest first. -/
def before (m : Fin n → ℕ) (i : ℕ) : List ℕ := ((List.ofFn m).take i).reverse

theorem length_before (m : Fin n → ℕ) {i : ℕ} (le : i ≤ n) : (before m i).length = i := by
  rw [before, List.length_reverse, List.length_take, List.length_ofFn]
  exact Nat.min_eq_left le

theorem before_le {m : Fin n → ℕ} (le : ∀ i, m i ≤ C) (i : ℕ) : ∀ c ∈ before m i, c ≤ C := by
  intro c member
  obtain ⟨j, rfl⟩ := List.mem_ofFn.mp (List.mem_of_mem_take (List.mem_reverse.mp member))
  exact le j

/-- The sub-key of the work squeeze of a stage: no squeeze before it, and 8 bytes. -/
def workSub (le8 : 8 ≤ C) : Sub S C :=
  Sub.of [] 8 (Nat.zero_le S) (fun _ member => nomatch member) le8

/-- The sub-key of the draw `i` of a stage: the squeezes of the draws before it and of the work,
and `m i` bytes. -/
def drawSub (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C) (i : Fin n) :
    Sub S C :=
  Sub.of (before m i ++ [8]) (m i)
    (by
      rw [List.length_append, length_before m i.isLt.le, List.length_singleton]
      exact (Nat.succ_le_of_lt i.isLt).trans short)
    (fun c member => (List.mem_append.mp member).elim (before_le le i c) fun single => by
      rw [List.mem_singleton.mp single]
      exact le8)
    (le i)

theorem counts_workSub (le8 : 8 ≤ C) : (workSub le8 : Sub S C).counts = [] :=
  Sub.counts_of [] 8 (Nat.zero_le S) (fun _ member => nomatch member) le8

theorem counts_drawSub (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C)
    (i : Fin n) : (drawSub m short le8 le i).counts = before m i ++ [8] :=
  Sub.counts_of _ _ _ _ _

/-- The answer for the work squeeze after the events of a base. -/
theorem answerOf_work (fs : Base L B C → Y S C) (base : Base L B C) (le8 : 8 ≤ C) :
    answerOf fs base.events 8 = array (take le8 (fs base (workSub le8))) :=
  answerOf_of fs base [] 8 (Nat.zero_le S) (fun _ member => nomatch member) le8

/-- The answer for the draw `i` after the squeezes of the draws before it, the squeeze of the
work and the events of a base. -/
theorem answerOf_draw (fs : Base L B C → Y S C) (base : Base L B C) (m : Fin n → ℕ)
    (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C) (i : Fin n) :
    answerOf fs ((before m i).map Duplex.Event.squeeze ++ .squeeze 8 :: base.events) (m i) =
      array (take (le i) (fs base (drawSub m short le8 le i))) := by
  have events : (before m i).map Duplex.Event.squeeze ++ .squeeze 8 :: base.events =
      (before m i ++ [8]).map Duplex.Event.squeeze ++ base.events := by
    simp only [List.map_append, List.map_cons, List.map_nil, List.append_assoc,
      List.cons_append, List.nil_append]
  rw [events]
  exact answerOf_of fs base (before m i ++ [8]) (m i) _ _ (le i)

/-- The sub-keys of a stage: the work, then the draws. -/
def stageSub (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C)
    (o : Option (Fin n)) : Sub S C :=
  o.elim (workSub le8) (drawSub m short le8 le)

theorem stageSub_injective (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C) :
    Function.Injective (stageSub (S := S) m short le8 le) := by
  have lengths : ∀ o : Option (Fin n), (stageSub (S := S) m short le8 le o).counts.length =
      o.elim 0 fun i => (i : ℕ) + 1 := by
    rintro (_ | i)
    · exact congrArg List.length (counts_workSub le8)
    · refine (congrArg List.length (counts_drawSub m short le8 le i)).trans ?_
      rw [List.length_append, length_before m i.isLt.le, List.length_singleton]
      rfl
  intro o o' same
  have equal := (lengths o).symm.trans
    ((congrArg (fun sub : Sub S C => sub.counts.length) same).trans (lengths o'))
  rcases o with _ | i <;> rcases o' with _ | i'
  · rfl
  · exact (Nat.succ_ne_zero _ equal.symm).elim
  · exact (Nat.succ_ne_zero _ equal).elim
  · exact congrArg some (Fin.ext (Nat.succ.inj equal))

/-- The strings of a stage in the answers of a base: the first 8 bytes at the sub-key of the
work, and the first `m i` bytes at the sub-key of the draw `i`. -/
def stageRead (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C) (y : Y S C) :
    Bytes 8 × ∀ i, Bytes (m i) :=
  (take le8 (y (workSub le8)), fun i => take (le i) (y (drawSub m short le8 le i)))

/-- The lens of the strings of a stage with a work squeeze and `n` draws of `m i` bytes. -/
noncomputable def stageLens (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C) :
    Lens (Y S C) (Bytes 8 × ∀ i, Bytes (m i)) :=
  (Lens.prefixes (stageSub m short le8 le) (stageSub_injective m short le8 le)
    (fun o : Option (Fin n) => o.elim 8 m) (by
      rintro (_ | i)
      exacts [le8, le i])).map
    (Equiv.piOptionEquivProd (β := fun o : Option (Fin n) => Bytes (o.elim 8 m)))

theorem stageLens_get (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C)
    (y : Y S C) : (stageLens m short le8 le).get y = stageRead m short le8 le y := rfl

variable {R : Type}

/-- The map `φ₀` of the strings of a stage, on the answers of a base. -/
noncomputable def stageLift (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C)
    (φ₀ : R × (Bytes 8 × ∀ i, Bytes (m i)) → Bytes 8 × ∀ i, Bytes (m i)) : R × Y S C → Y S C :=
  (stageLens m short le8 le).lift φ₀

/-- The fibres of the lift of a map whose fibres have `|R|` elements have `|R|` elements. -/
theorem card_stageLift [Fintype R] (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C)
    (le : ∀ i, m i ≤ C)
    (φ₀ : R × (Bytes 8 × ∀ i, Bytes (m i)) → Bytes 8 × ∀ i, Bytes (m i))
    (fibres : ∀ y, Nat.card {x // φ₀ x = y} = Fintype.card R) (y : Y S C) :
    Nat.card {x // stageLift m short le8 le φ₀ x = y} = Fintype.card R :=
  ((stageLens m short le8 le).card_lift φ₀ y).trans (fibres _)

/-- The answers of a table whose entry at `base` is the lift of `φ₀` at `(r, y)`: the work
squeeze after the events of the base gives the first string of `φ₀` on `r` and the strings of
the stage in `y`, and the draw `i`, after the squeezes of the draws before it and of the work,
gives the string `i`. -/
theorem answerOf_stageLift (m : Fin n → ℕ) (short : n ≤ S) (le8 : 8 ≤ C) (le : ∀ i, m i ≤ C)
    (φ₀ : R × (Bytes 8 × ∀ i, Bytes (m i)) → Bytes 8 × ∀ i, Bytes (m i))
    (fs : Base L B C → Y S C) (base : Base L B C) (r : R) (y : Y S C)
    (entry : fs base = stageLift m short le8 le φ₀ (r, y)) :
    answerOf fs base.events 8 = array (φ₀ (r, stageRead m short le8 le y)).1 ∧
      ∀ i : Fin n, answerOf fs ((before m i).map Duplex.Event.squeeze ++
          .squeeze 8 :: base.events) (m i) =
        array ((φ₀ (r, stageRead m short le8 le y)).2 i) := by
  have read := (stageLens m short le8 le).get_lift φ₀ r y
  refine ⟨(answerOf_work fs base le8).trans (congrArg array ?_),
    fun i => (answerOf_draw fs base m short le8 le i).trans (congrArg array ?_)⟩
  · rw [entry]
    exact congrArg Prod.fst read
  · rw [entry]
    exact congrFun (congrArg Prod.snd read) i

/-- The strings of a stage with one draw of `m` bytes in the answers of a base. -/
def oneRead {m : ℕ} (short : 1 ≤ S) (le8 : 8 ≤ C) (le : m ≤ C) (y : Y S C) : Bytes 8 × Bytes m :=
  (take le8 (y (workSub le8)), take le (y (drawSub (fun _ : Fin 1 => m) short le8 (fun _ => le) 0)))

/-- The lens of the strings of a stage with a work squeeze and one draw of `m` bytes. -/
noncomputable def oneLens {m : ℕ} (short : 1 ≤ S) (le8 : 8 ≤ C) (le : m ≤ C) :
    Lens (Y S C) (Bytes 8 × Bytes m) :=
  (stageLens (fun _ : Fin 1 => m) short le8 fun _ => le).map
    (Equiv.prodCongr (Equiv.refl _) (Equiv.funUnique (Fin 1) (Bytes m)))

theorem oneLens_get {m : ℕ} (short : 1 ≤ S) (le8 : 8 ≤ C) (le : m ≤ C) (y : Y S C) :
    (oneLens short le8 le).get y = oneRead short le8 le y := rfl

/-- The map `φ₀` of the strings of a stage with one draw, on the answers of a base. -/
noncomputable def oneLift {m : ℕ} (short : 1 ≤ S) (le8 : 8 ≤ C) (le : m ≤ C)
    (φ₀ : R × (Bytes 8 × Bytes m) → Bytes 8 × Bytes m) : R × Y S C → Y S C :=
  (oneLens short le8 le).lift φ₀

/-- The fibres of the lift of a map whose fibres have `|R|` elements have `|R|` elements. -/
theorem card_oneLift [Fintype R] {m : ℕ} (short : 1 ≤ S) (le8 : 8 ≤ C) (le : m ≤ C)
    (φ₀ : R × (Bytes 8 × Bytes m) → Bytes 8 × Bytes m)
    (fibres : ∀ y, Nat.card {x // φ₀ x = y} = Fintype.card R) (y : Y S C) :
    Nat.card {x // oneLift short le8 le φ₀ x = y} = Fintype.card R :=
  ((oneLens short le8 le).card_lift φ₀ y).trans (fibres _)

/-- The answers of a table whose entry at `base` is the lift of `φ₀` at `(r, y)`: the work
squeeze after the events of the base gives the first string of `φ₀` on `r` and the strings of
the stage in `y`, and the draw after the squeeze of the work gives the second string. -/
theorem answerOf_oneLift {m : ℕ} (short : 1 ≤ S) (le8 : 8 ≤ C) (le : m ≤ C)
    (φ₀ : R × (Bytes 8 × Bytes m) → Bytes 8 × Bytes m)
    (fs : Base L B C → Y S C) (base : Base L B C) (r : R) (y : Y S C)
    (entry : fs base = oneLift short le8 le φ₀ (r, y)) :
    answerOf fs base.events 8 = array (φ₀ (r, oneRead short le8 le y)).1 ∧
      answerOf fs (.squeeze 8 :: base.events) m = array (φ₀ (r, oneRead short le8 le y)).2 := by
  have read := (oneLens short le8 le).get_lift φ₀ r y
  refine ⟨(answerOf_work fs base le8).trans (congrArg array ?_),
    (answerOf_draw fs base (fun _ : Fin 1 => m) short le8 (fun _ => le) 0).trans
      (congrArg array ?_)⟩
  · rw [entry]
    exact congrArg Prod.fst read
  · rw [entry]
    exact congrArg Prod.snd read

end Stage

/-- A uniform table of functions is a uniform table on the pairs. -/
theorem expect_curry {A K V : Type} [Fintype A] [DecidableEq A] [Fintype K] [DecidableEq K]
    [Fintype V] (F : (A × K → V) → ℝ) :
    𝔼 fs : A → K → V, F (fun x => fs x.1 x.2) = 𝔼 g : A × K → V, F g :=
  (Fintype.expect_equiv (Equiv.curry A K V) F (fun fs => F fun x => fs x.1 x.2)
    fun _ => rfl).symm

/-- A uniform table of answers for each base is a uniform table on the pairs of a base and a
sub-key. -/
theorem expect_table (F : (Base L B C × Sub S C → Bytes C) → ℝ) :
    𝔼 fs : Base L B C → Y S C, F (fun x => fs x.1 x.2) = 𝔼 g, F g :=
  expect_curry F

end NightstreamFPrime.Spec.Compression.Akita.Replay.Sponge
