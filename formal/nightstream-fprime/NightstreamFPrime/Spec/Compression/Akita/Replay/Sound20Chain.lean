import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Positions

/-! The keys of one chain of `key20.aks`: the keys that one sponge query settles on a closed run
(Akita, ePrint 2026/1983, revision of 2026-11-23, Lemma 10.47 and Appendix E.1, (F1);
`docs/akita-work/w22/design.md`, the section "Item 2"; `Replay/Sound20Deps.lean`,
`Replay/SpongeFreeBases.lean`, `Replay/Resolve.lean`, `Replay/StageDigest.lean`,
`Replay/Sound20Positions.lean`).

- `depsOf_mem`, `depsOf_nested`, `chain_nested`: two keys of the chain `deps20 S point ++ [point]`
  are equal, or one is the base of a squeeze inside the events of the other.
- `stage20_base`, `stage20_dep`, `stage20_nested`: the base of a squeeze inside the events of a
  stage `k` of a proof has the events of a stage `j < k` of the same proof.
- `recordAddress20_trim`, `resolveAt_chain`: the addresses of two keys of one chain have the same
  key of each stage that the two have.
- `stageDigests20_subset`, `positions20_chain`: the keys of one chain have at most 2 different
  strings in their digest positions. `fresh20`: the hypothesis `fresh` for the public input of
  the assembly. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Address)
open Sponge (split answerOf)
open Replayer

section Lists

/-- A list with one element or more, of lists of which each two are in the relation `⊆` in one
direction, has an element that contains each other. -/
theorem exists_widest {α β : Type} (f : α → List β) : ∀ (l : List α), l ≠ [] →
    (∀ x ∈ l, ∀ y ∈ l, f x ⊆ f y ∨ f y ⊆ f x) → ∃ m ∈ l, ∀ x ∈ l, f x ⊆ f m
  | [], nonempty, _ => absurd rfl nonempty
  | [x], _, _ => ⟨x, List.mem_singleton.mpr rfl, fun y member => by
      rw [List.mem_singleton.mp member]⟩
  | x :: y :: rest, _, total => by
      obtain ⟨m, member, top⟩ := exists_widest f (y :: rest) (List.cons_ne_nil _ _)
        fun u first v second =>
          total u (List.mem_cons_of_mem _ first) v (List.mem_cons_of_mem _ second)
      rcases total x List.mem_cons_self m (List.mem_cons_of_mem _ member) with le | ge
      · refine ⟨m, List.mem_cons_of_mem _ member, fun u listed => ?_⟩
        rcases List.mem_cons.mp listed with same | later
        · rw [same]
          exact le
        · exact top u later
      · refine ⟨x, List.mem_cons_self, fun u listed => ?_⟩
        rcases List.mem_cons.mp listed with same | later
        · rw [same]
        · exact List.Subset.trans (top u later) ge

/-- The key of the stage `level` that a key extends, when `trim` gives it. -/
theorem trim_eq {Wire : Type} {stages level : ℕ} {key anchor : Address Wire stages}
    (found : key.trim level = some anchor) : anchor = key.take level := by
  unfold Address.trim at found
  split at found
  · exact (Option.some.inj found).symm
  · exact absurd found (by simp)

end Lists

section Deps

variable {L B C : ℕ} (S : ℕ)

/-- Each base of `depsOf` is the base of a squeeze inside the events. -/
theorem depsOf_mem : ∀ {events : List Duplex.Event} {base : Sponge.Base L B C},
    base ∈ depsOf S events →
      ∃ (before : List Duplex.Event) (count : ℕ) (sub : Sponge.Sub S C),
        (Duplex.Event.squeeze count :: before) <:+ events ∧ split before count = some (base, sub)
  | [], _, member => absurd member List.not_mem_nil
  | .absorb _ :: rest, _, member => by
      obtain ⟨before, count, sub, inside, found⟩ := depsOf_mem (events := rest) member
      exact ⟨before, count, sub, inside.trans (List.suffix_cons _ _), found⟩
  | .squeeze c :: rest, base, member => by
      have parts : base ∈ depsOf S rest ++
          ((split rest c : Option (Sponge.Base L B C × Sponge.Sub S C)).map Prod.fst).toList :=
        member
      rcases List.mem_append.mp parts with early | last
      · obtain ⟨before, count, sub, inside, found⟩ := depsOf_mem (events := rest) early
        exact ⟨before, count, sub, inside.trans (List.suffix_cons _ _), found⟩
      · cases found : (split rest c : Option (Sponge.Base L B C × Sponge.Sub S C)) with
        | none =>
            rw [found] at last
            simp at last
        | some pair =>
            obtain ⟨other, sub⟩ := pair
            rw [found] at last
            have equal : base = other := by simpa using last
            rw [equal]
            exact ⟨rest, c, sub, List.suffix_refl _, found⟩

variable {S}

/-- Two squeezes with a base, the first inside the events up to and with the second: the two
bases are equal, or the first is the base of a squeeze inside the events of the second base. -/
theorem depsOf_step {before before' : List Duplex.Event} {count count' : ℕ}
    {a b : Sponge.Base L B C} {sub sub' : Sponge.Sub S C}
    (found : split before count = some (a, sub)) (found' : split before' count' = some (b, sub'))
    (inner : (Duplex.Event.squeeze count :: before) <:+
      (Duplex.Event.squeeze count' :: before')) :
    a = b ∨ a ∈ depsOf S b.events := by
  obtain ⟨parts, -⟩ := Sponge.split_eq_some.mp found
  obtain ⟨parts', -⟩ := Sponge.split_eq_some.mp found'
  have same_of : ∀ later : List ℕ,
      before = later.map Duplex.Event.squeeze ++ b.events → a = b := fun later shape =>
    Sponge.Base.events_injective (Sponge.squeezes_append_inj a.events_ne b.events_ne
      (show sub.counts.map Duplex.Event.squeeze ++ a.events =
        later.map Duplex.Event.squeeze ++ b.events from parts.symm.trans shape)).2
  rcases List.suffix_cons_iff.mp inner with equal | deeper
  · exact Or.inl (same_of sub'.counts ((List.cons.inj equal).2.trans parts'))
  · rw [parts'] at deeper
    rcases squeeze_squeezes sub'.counts deeper with ⟨later, shape⟩ | within
    · exact Or.inl (same_of later shape)
    · exact Or.inr (mem_depsOf S within found)

variable (S)

/-- Two bases of `depsOf events` are equal, or one is a base of `depsOf` of the events of the
other. -/
theorem depsOf_nested {events : List Duplex.Event} {a b : Sponge.Base L B C}
    (first : a ∈ depsOf S events) (second : b ∈ depsOf S events) :
    a = b ∨ a ∈ depsOf S b.events ∨ b ∈ depsOf S a.events := by
  obtain ⟨before, count, sub, inside, found⟩ := depsOf_mem S first
  obtain ⟨before', count', sub', inside', found'⟩ := depsOf_mem S second
  rcases List.suffix_or_suffix_of_suffix inside inside' with le | ge
  · rcases depsOf_step found found' le with same | deeper
    · exact Or.inl same
    · exact Or.inr (Or.inl deeper)
  · rcases depsOf_step found' found ge with same | deeper
    · exact Or.inl same.symm
    · exact Or.inr (Or.inr deeper)

/-- Two keys of one chain are equal, or one is a key that the other names. -/
theorem chain_nested {point a b : Sponge.Base L B C} (first : a ∈ deps20 S point ++ [point])
    (second : b ∈ deps20 S point ++ [point]) : a = b ∨ a ∈ deps20 S b ∨ b ∈ deps20 S a := by
  rcases List.mem_append.mp first with early | last
  · rcases List.mem_append.mp second with early' | last'
    · exact depsOf_nested S early early'
    · rw [List.mem_singleton.mp last']
      exact Or.inr (Or.inl early)
  · rw [List.mem_singleton.mp last]
    rcases List.mem_append.mp second with early' | last'
    · exact Or.inr (Or.inr early')
    · exact Or.inl (List.mem_singleton.mp last').symm

end Deps

section Stages

variable {L B C S : ℕ} {pub : Public} {proof proof' : ByteArray}
  {messages messages' : Array Message} {a : List Duplex.Event → ℕ → ByteArray}

/-- On a script of `layout20` with the work bits of the model and an ideal sponge with no event
before the channel: the base of a squeeze inside the events of the stage `k` of a proof has the
events of a stage `j < k` of the same proof. -/
theorem stage20_base (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = [])
    {events : List Duplex.Event} {k : ℕ} (stage : Stage20 pub events k proof messages)
    {before : List Duplex.Event} {count : ℕ}
    (inner : (Duplex.Event.squeeze count :: before) <:+ events) {base : Sponge.Base L B C}
    {sub : Sponge.Sub S C} (found : split before count = some (base, sub)) :
    ∃ j, j < k ∧ Stage20 pub base.events j proof messages := by
  obtain ⟨inside, decoded, ran, same⟩ := stage
  subst same
  obtain ⟨j, lt, counts, shape⟩ := squeeze_stage20 erased bits ideal fresh k inside ran inner
  obtain ⟨parts, -⟩ := Sponge.split_eq_some.mp found
  obtain ⟨site, level, at_⟩ := bits j (by omega)
  obtain ⟨m, -, head, -, -⟩ := work_squeeze20 at_ (bits20_ne_zero (by omega : j < 473))
    (stateBefore_le pub proof messages pub.ops (stageIndex20_succ_le pub lt inside) ran).1 ideal
  have first : ∀ c rest, eventsAt20 pub proof messages j ≠ Duplex.Event.squeeze c :: rest := by
    intro c rest equal
    rw [head] at equal
    exact Duplex.Event.noConfusion (List.cons.inj equal).1
  have equal := (Sponge.squeezes_append_inj base.events_ne first
    (show sub.counts.map Duplex.Event.squeeze ++ base.events =
      counts.map Duplex.Event.squeeze ++ eventsAt20 pub proof messages j from
      parts.symm.trans shape)).2
  exact ⟨j, lt, by omega, decoded,
    (stateBefore_le pub proof messages pub.ops
      (stageIndex20_mono pub (Nat.le_of_lt lt) inside) ran).1, equal.symm⟩

/-- A key that the key of the stage `k` of a proof names has the events of a stage `j < k` of the
same proof. -/
theorem stage20_dep (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = [])
    {big small : Sponge.Base L B C} {k : ℕ} (stage : Stage20 pub big.events k proof messages)
    (named : small ∈ deps20 S big) :
    ∃ j, j < k ∧ Stage20 pub small.events j proof messages := by
  obtain ⟨before, count, sub, inner, found⟩ := depsOf_mem S named
  exact stage20_base erased bits ideal fresh stage inner found

/-- A key with a stage `j` of a proof, that the key of the stage `k` of a second proof names:
`j < k`, and the key has the events of the stage `j` of the second proof. -/
theorem stage20_nested (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (ideal : pub.sponge.ideal = some a) (fresh : pub.sponge.events = [])
    {big small : Sponge.Base L B C} {k j : ℕ} (stage : Stage20 pub big.events k proof messages)
    (named : small ∈ deps20 S big) (own : Stage20 pub small.events j proof' messages') :
    j < k ∧ small.events = eventsAt20 pub proof messages j ∧
      Stage20 pub small.events j proof messages := by
  obtain ⟨i, lt, moved⟩ := stage20_dep erased bits ideal fresh stage named
  obtain rfl : j = i := stage20_unique bits own moved
  exact ⟨lt, moved.2.2.2.symm, moved⟩

end Stages

section Addresses

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {L B C S D : ℕ} {pub : Public}

/-- `resolve20` gives a key only for the events of a stage of a proof. -/
theorem stage20_of_resolved {total : ℕ} {record : List (ByteArray × ByteArray)}
    {events : List Duplex.Event} {address : Address (Key20.Wire E) total}
    (resolved : resolve20 total pub record events = some address) :
    ∃ (k : ℕ) (proof : ByteArray) (messages : Array Message),
      Stage20 pub events k proof messages := by
  by_contra missing
  have none_ : ¬∃ stage : ℕ × ByteArray × Array Message,
      Stage20 pub events stage.1 stage.2.1 stage.2.2 :=
    fun ⟨⟨k, proof, messages⟩, stage⟩ => missing ⟨k, proof, messages, stage⟩
  unfold resolve20 at resolved
  rw [dite_eq_right none_] at resolved
  cases resolved

/-- On a script of `layout20`, the keys of two stages `j ≤ k` of one proof with the last blocks
of one record have the same key of each stage `level` that the two have: the page `level` reads
the last block of `t` only for `379 ≤ level`, and the digest of this block is the same before
the work of each stage from 379; the same for the last block of `e` and 472. -/
theorem recordAddress20_trim (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {j k : ℕ} (le : j ≤ k) (record : List (ByteArray × ByteArray)) (total : ℕ)
    (proof : ByteArray) (messages : Array Message) {level : ℕ}
    {u v : Address (Key20.Wire E) total}
    (trimmed : (recordAddress20 pub record total j proof messages).trim level = some u)
    (trimmed' : (recordAddress20 pub record total k proof messages).trim level = some v) :
    u = v := by
  have length : (recordAddress20 (E := E) pub record total j proof messages).rest.length =
      min j total := by
    unfold recordAddress20
    show ((List.range (min j total)).map _).length = _
    rw [List.length_map, List.length_range]
  have short : level ≤ j := by
    have below := (Address.trim_rank trimmed).2
    omega
  rw [trim_eq trimmed, trim_eq trimmed']
  unfold recordAddress20
  rw [addressAt20_take _ _ short total, addressAt20_take _ _ (short.trans le) total]
  refine addressAt20_congr (fun late => ?_) (fun late => ?_) total
  · rw [tDigestAt_terminal pub proof messages erased (late.trans short),
      tDigestAt_terminal pub proof messages erased ((late.trans short).trans le)]
  · rw [eDigestAt_terminal pub proof messages erased (late.trans short),
      eDigestAt_terminal pub proof messages erased ((late.trans short).trans le)]

/-- The addresses of a key and of a key that it names have the same key of each stage that the
two have. -/
theorem resolveAt_dep (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (fresh : pub.sponge.events = []) (total : ℕ)
    (fs : Sponge.Base L B C → Sponge.Y S C) (record : List (Sponge.Str D × Bytes 32))
    {small big : Sponge.Base L B C} (named : small ∈ deps20 S big)
    {x y : Address (Key20.Wire E) total}
    (resolved : resolveAt total pub fs record small = some x)
    (resolved' : resolveAt total pub fs record big = some y) {level : ℕ}
    {u v : Address (Key20.Wire E) total} (trimmed : x.trim level = some u)
    (trimmed' : y.trim level = some v) : u = v := by
  have bits' : Bits20 (pub.reideal (some (answerOf fs))) := bits
  have atSmall : resolve20 (E := E) total (pub.reideal (some (answerOf fs))) (recordOf record)
      small.events = some x := resolved
  have atBig : resolve20 (E := E) total (pub.reideal (some (answerOf fs))) (recordOf record)
      big.events = some y := resolved'
  obtain ⟨k, proof, messages, stage⟩ := stage20_of_resolved atBig
  obtain ⟨j, lt, stage'⟩ := stage20_dep (pub := pub.reideal (some (answerOf fs))) erased bits'
    rfl fresh stage named
  rw [resolve20_stage bits' stage' (recordOf record) total] at atSmall
  rw [resolve20_stage bits' stage (recordOf record) total] at atBig
  rw [← Option.some.inj atSmall] at trimmed
  rw [← Option.some.inj atBig] at trimmed'
  exact recordAddress20_trim (pub := pub.reideal (some (answerOf fs))) erased (Nat.le_of_lt lt)
    (recordOf record) total proof messages trimmed trimmed'

/-- The addresses of the keys of one chain are coherent: for one table and one record, the
addresses of two keys of the chain `deps20 S point ++ [point]` have the same key of each stage
`level` that the two have (Akita Appendix E.1, (F1): one sponge query settles a chain of nested
stage inputs of one proof). -/
theorem resolveAt_chain (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (fresh : pub.sponge.events = []) (total : ℕ)
    (fs : Sponge.Base L B C → Sponge.Y S C) (record : List (Sponge.Str D × Bytes 32))
    (point : Sponge.Base L B C) {a b : Sponge.Base L B C}
    (first : a ∈ deps20 S point ++ [point]) (second : b ∈ deps20 S point ++ [point])
    {x y : Address (Key20.Wire E) total} (resolved : resolveAt total pub fs record a = some x)
    (resolved' : resolveAt total pub fs record b = some y) {level : ℕ}
    {u v : Address (Key20.Wire E) total} (trimmed : x.trim level = some u)
    (trimmed' : y.trim level = some v) : u = v := by
  rcases chain_nested S first second with same | named | named
  · rw [same] at resolved
    rw [Option.some.inj (resolved.symm.trans resolved')] at trimmed
    exact Option.some.inj (trimmed.symm.trans trimmed')
  · exact resolveAt_dep erased bits fresh total fs record named resolved resolved' trimmed
      trimmed'
  · exact (resolveAt_dep erased bits fresh total fs record named resolved' resolved trimmed'
      trimmed).symm

end Addresses

section Positions

variable {L B C S D : ℕ} {pub : Public}

/-- On a script of `layout20`, the strings in the digest positions of the key of a stage `j` of
a proof are strings of the key of each later stage `k` of this proof. -/
theorem stageDigests20_subset (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    {j k : ℕ} (le : j ≤ k) (proof : ByteArray) (messages : Array Message) :
    stageDigests20 pub proof messages j ⊆ stageDigests20 pub proof messages k := by
  intro string member
  unfold stageDigests20 at member ⊢
  rcases List.mem_append.mp member with t | e
  · by_cases late : 379 ≤ j
    · rw [ite_eq_left late, tDigestAt_terminal pub proof messages erased late] at t
      refine List.mem_append_left _ ?_
      rw [ite_eq_left (late.trans le),
        tDigestAt_terminal pub proof messages erased (late.trans le)]
      exact t
    · rw [ite_eq_right late] at t
      exact absurd t List.not_mem_nil
  · by_cases late : 472 ≤ j
    · rw [ite_eq_left late, eDigestAt_terminal pub proof messages erased late] at e
      refine List.mem_append_right _ ?_
      rw [ite_eq_left (late.trans le),
        eDigestAt_terminal pub proof messages erased (late.trans le)]
      exact e
    · rw [ite_eq_right late] at e
      exact absurd e List.not_mem_nil

/-- The digest positions of a key that a second key names have only strings of the second key,
or the second key has no string. -/
theorem positions20_dep (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (bits : Bits20 pub) (fresh : pub.sponge.events = [])
    (fs : Sponge.Base L B C → Sponge.Y S C) {small big : Sponge.Base L B C}
    (named : small ∈ deps20 S big) :
    positions20 (D := D) pub fs (.inl small) ⊆ positions20 (D := D) pub fs (.inl big) ∨
      positions20 (D := D) pub fs (.inl big) = [] := by
  have bits' : Bits20 (pub.reideal (some (answerOf fs))) := bits
  by_cases found : ∃ stage : ℕ × ByteArray × Array Message,
      Stage20 (pub.reideal (some (answerOf fs))) big.events stage.1 stage.2.1 stage.2.2
  · obtain ⟨⟨k, proof, messages⟩, staged⟩ := found
    have stage : Stage20 (pub.reideal (some (answerOf fs))) big.events k proof messages := staged
    obtain ⟨j, lt, stage'⟩ := stage20_dep (pub := pub.reideal (some (answerOf fs))) erased bits'
      rfl fresh stage named
    left
    rw [positions20_stage bits stage', positions20_stage bits stage]
    exact stageDigests20_subset (pub := pub.reideal (some (answerOf fs))) erased
      (Nat.le_of_lt lt) proof messages
  · right
    show keyDigests20 pub fs big = []
    unfold keyDigests20
    rw [dite_eq_right found]

/-- The keys of one chain have at most 2 different strings in their digest positions, for each
table of the sponge (Akita Lemma 10.47, `V_o = 2` in (213), for the keys that one sponge query
settles). -/
theorem positions20_chain [DecidableEq (Bytes 32)]
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (bits : Bits20 pub)
    (fresh : pub.sponge.events = []) (fs : Sponge.Base L B C → Sponge.Y S C)
    (point : Sponge.Base L B C) :
    ((deps20 S point ++ [point]).flatMap fun key =>
      positions20 (D := D) pub fs (.inl key)).toFinset.card ≤ 2 := by
  have total : ∀ a ∈ deps20 S point ++ [point], ∀ b ∈ deps20 S point ++ [point],
      positions20 (D := D) pub fs (.inl a) ⊆ positions20 (D := D) pub fs (.inl b) ∨
        positions20 (D := D) pub fs (.inl b) ⊆ positions20 (D := D) pub fs (.inl a) := by
    intro a first b second
    rcases chain_nested S first second with same | named | named
    · rw [same]
      exact Or.inl (List.Subset.refl _)
    · rcases positions20_dep (D := D) erased bits fresh fs named with sub | empty
      · exact Or.inl sub
      · exact Or.inr (by rw [empty]; exact List.nil_subset _)
    · rcases positions20_dep (D := D) erased bits fresh fs named with sub | empty
      · exact Or.inr sub
      · exact Or.inl (by rw [empty]; exact List.nil_subset _)
  obtain ⟨top, -, widest⟩ := exists_widest
    (fun key => positions20 (D := D) pub fs (.inl key)) (deps20 S point ++ [point])
    (by simp) total
  have inside : ((deps20 S point ++ [point]).flatMap fun key =>
      positions20 (D := D) pub fs (.inl key)).toFinset ⊆
      (positions20 (D := D) pub fs (.inl top)).toFinset := by
    intro string member
    obtain ⟨key, listed, hit⟩ := List.mem_flatMap.mp (List.mem_toFinset.mp member)
    exact List.mem_toFinset.mpr (widest key listed hit)
  exact (Finset.card_le_card inside).trans
    ((List.toFinset_card_le _).trans (length_positions20 pub fs top))

/-- The public input of `resolveAt` in the assembly has a sponge with no event. -/
theorem fresh20 (pubV : Public) : (pubV.withSponge Duplex.empty).sealed.sponge.events = [] :=
  rfl

end Positions

end NightstreamFPrime.Spec.Compression.Akita.Replay
