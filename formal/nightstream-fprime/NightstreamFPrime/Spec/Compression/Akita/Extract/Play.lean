import NightstreamFPrime.Spec.Compression.Akita.Extract.PairComplete

/-! The honest prover of the checks of one level on a Fiat–Shamir transcript, for checks with more
than one message between two draws (Akita, ePrint 2026/1983, revision of 2026-10-15,
Theorem 10.9: "All honest algebraic checks are exact identities").

The checks of a level with the direct norm check or with the tensor reduction send two or three
messages before one draw, and the claimed energy of the norm check is a natural number. `WinsTo`
has one message between two draws and a condition on the type of a message.

- `WinsAt fits slots game draws index tail` states that a prover has a message for each draw
  sequence such that the game ends with `tail`. `index` is the number of the messages after the
  last draw, which is less than `slots` at each message, and `fits` is a condition on each
  message. `winsAt_of_winsTo` gives it from `WinsTo`.
- `Carries` states that a wire message can have each value of the other data of the proof with
  each list of at most `slots` messages that satisfy `fits`.
- `play_carried` gives the wire messages of the honest prover for a game with `WinsAt`.
- `Tier.drawn` is the fold challenge vector that a table of the oracle gives after a transcript
  prefix, and `Tier.run` gives the transcript prefix of the honest prover through the stages of
  one level: the game of the checks ends in the goal, and each transcript with this prefix has
  the stages of the level, the fold challenges `Tier.drawn` and this game. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

section Wins

variable {α β : Type} (fits : (M : Type) → M → Prop) (slots : ℕ)

/-- A prover has a message for each draw sequence such that the game ends in a result, a number
of remaining draws and a number of messages after the last draw that satisfy `tail`. A draw uses
one of the `draws` draws and sets `index` to 0. A message needs `index < slots` and `fits`, and
adds 1 to `index`. -/
def WinsAt : Game α → ℕ → ℕ → (α → ℕ → ℕ → Prop) → Prop
  | Game.pure result, draws, index, tail => tail result draws index
  | Game.draw _ _ _, 0, _, _ => False
  | Game.draw _ _ next, rest + 1, _, tail => ∀ challenge, WinsAt (next challenge) rest 0 tail
  | Game.recv M next, draws, index, tail =>
      index < slots ∧ ∃ message, fits M message ∧ WinsAt (next message) draws (index + 1) tail

theorem winsAt_bind (game : Game α) (after : α → Game β) (draws index : ℕ)
    (tail : β → ℕ → ℕ → Prop) :
    WinsAt fits slots (game.bind after) draws index tail ↔
      WinsAt fits slots game draws index fun result rest place =>
        WinsAt fits slots (after result) rest place tail := by
  induction game generalizing draws index with
  | pure result => exact Iff.rfl
  | draw C bad next induction =>
      cases draws with
      | zero => exact Iff.rfl
      | succ rest => exact forall_congr' fun challenge => induction challenge rest 0
  | recv M next induction =>
      exact and_congr_right fun _ => exists_congr fun message =>
        and_congr_right fun _ => induction message draws (index + 1)

variable {fits slots}

theorem WinsAt.mono {game : Game α} {draws index : ℕ} {tail other : α → ℕ → ℕ → Prop}
    (wins : WinsAt fits slots game draws index tail)
    (weaker : ∀ result rest place, tail result rest place → other result rest place) :
    WinsAt fits slots game draws index other := by
  induction game generalizing draws index with
  | pure result => exact weaker _ _ _ wins
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest => exact fun challenge => induction challenge (wins challenge)
  | recv M next induction =>
      obtain ⟨room, message, sendable, won⟩ := wins
      exact ⟨room, message, sendable, induction message won⟩

/-- A game with `WinsTo` has `WinsAt`: it has at most one message after `index` messages, or
after a draw. The game ends with at most `index + 1` messages after the last draw when it starts
with `fresh`, and at most `max index 1` when it does not. -/
theorem winsAt_of_winsTo {admits : Type → Prop}
    (each : ∀ (M : Type) (message : M), admits M → fits M message) (one : 1 ≤ slots)
    {game : Game α} {draws : ℕ} {fresh : Bool} {index : ℕ} {tail : α → ℕ → Bool → Prop}
    (wins : WinsTo admits game draws fresh tail) (room : fresh = true → index < slots) :
    WinsAt fits slots game draws index fun result rest place =>
      place ≤ max (index + fresh.toNat) 1 ∧ ∃ ready, tail result rest ready := by
  induction game generalizing draws fresh index with
  | pure result => exact ⟨le_max_of_le_left (Nat.le_add_right _ _), fresh, wins⟩
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest =>
          exact fun challenge => (induction challenge (wins challenge) fun _ => one).mono
            fun _ _ _ ended => ⟨ended.1.trans (le_max_right _ _), ended.2⟩
  | recv M next induction =>
      obtain ⟨ready, sendable, message, won⟩ := wins
      subst ready
      exact ⟨room rfl, message, each M message sendable,
        (induction message won fun impossible => absurd impossible (by decide)).mono
          fun _ _ _ ended => ended⟩

/-- `winsAt_of_winsTo` for a game that starts after `index` messages and whose end condition is a
goal of the result: it ends with at most `index + 1` messages after the last draw. -/
theorem winsAt_of_fresh {admits : Type → Prop}
    (each : ∀ (M : Type) (message : M), admits M → fits M message) {game : Game α}
    {draws index : ℕ} {goal : α → Prop}
    (wins : WinsTo admits game draws true fun result _ _ => goal result)
    (room : index < slots) :
    WinsAt fits slots game draws index fun result _ place => place ≤ index + 1 ∧ goal result :=
  (winsAt_of_winsTo each (Nat.one_le_of_lt room) wins fun _ => room).mono
    fun _ _ _ ended => ⟨ended.1.trans (max_le le_rfl (Nat.le_add_left _ _)),
      ended.2.elim fun _ good => good⟩

/-- A game with `WinsAt` for `draws` draws has it for `draws + extra` draws, and it ends with
`extra` remaining draws or more. -/
theorem WinsAt.more {game : Game α} {draws index : ℕ} {tail : α → ℕ → Prop}
    (wins : WinsAt fits slots game draws index fun result _ place => tail result place)
    (extra : ℕ) :
    WinsAt fits slots game (draws + extra) index fun result rest place =>
      extra ≤ rest ∧ tail result place := by
  induction game generalizing draws index with
  | pure result => exact ⟨Nat.le_add_left _ _, wins⟩
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest =>
          rw [Nat.add_right_comm]
          exact fun challenge => induction challenge (wins challenge)
  | recv M next induction =>
      obtain ⟨room, message, sendable, won⟩ := wins
      exact ⟨room, message, sendable, induction message won⟩

/-- A game with `WinsAt` after `index` messages has it after fewer messages, for an end condition
that a smaller number of messages keeps. -/
theorem WinsAt.lower {game : Game α} {draws index fewer : ℕ} {tail : α → ℕ → ℕ → Prop}
    (wins : WinsAt fits slots game draws index tail) (below : fewer ≤ index)
    (keeps : ∀ result rest place smaller, smaller ≤ place → tail result rest place →
      tail result rest smaller) :
    WinsAt fits slots game draws fewer tail := by
  induction game generalizing draws index fewer with
  | pure result => exact keeps _ _ _ _ below wins
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest => exact wins
  | recv M next induction =>
      obtain ⟨room, message, sendable, won⟩ := wins
      exact ⟨Nat.lt_of_le_of_lt below room, message, sendable,
        induction message won (Nat.succ_le_succ below)⟩

/-- The honest prover of two games in sequence: the first game ends in `goal` with at most `bound`
messages after its last draw, and the second game has `WinsAt` after `bound` messages for each
result with `goal`. -/
theorem WinsAt.bind {game : Game α} {after : α → Game β} {first second index bound last : ℕ}
    {goal : α → Prop} {ended : β → Prop}
    (wins : WinsAt fits slots game first index fun result _ place => place ≤ bound ∧ goal result)
    (next : ∀ result, goal result → WinsAt fits slots (after result) second bound
      fun result _ place => place ≤ last ∧ ended result) :
    WinsAt fits slots (game.bind after) (first + second) index
      fun result _ place => place ≤ last ∧ ended result :=
  (winsAt_bind fits slots game after _ _ _).mpr ((wins.more second).mono
    fun result rest place ⟨spare, early, good⟩ => by
      obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le spare
      exact (((next result good).more extra).mono fun _ _ _ kept => kept.2).lower early
        fun _ _ _ _ smaller kept => ⟨smaller.trans kept.1, kept.2⟩)

end Wins

/-- One message of a game, with its type. -/
structure Sent : Type 1 where
  M : Type
  [inhabited : Nonempty M]
  message : M

section Play

variable {Wire Data Reply α : Type} [Fintype Reply] (fits : (M : Type) → M → Prop) (slots : ℕ)
  (read : Reader Wire) {goal : α → Prop}

/-- A wire message can have each value of the data `dataOf` with each list of at most `slots`
messages of a game that satisfy `fits`: message `index` of the list is the message that `read`
gives at `index`. -/
def Carries (dataOf : Wire → Data) : Prop :=
  ∀ (data : Data) (sent : List Sent), sent.length ≤ slots →
    (∀ letter ∈ sent, fits letter.M letter.message) →
    ∃ wire, dataOf wire = data ∧ ∀ (index : ℕ) (inside : index < sent.length),
      @read sent[index].M sent[index].inhabited wire index = sent[index].message

/-- The messages of the honest prover before a draw: when the prover has a message for each draw
sequence, a list of messages gives the game `after` at its next draw or at its end, for each wire
message that has this list from `index`. -/
theorem settle_winsAt {game : Game α} {draws index : ℕ}
    (wins : WinsAt fits slots game draws index fun result _ _ => goal result) :
    ∃ (sent : List Sent) (after : Game α), sent.length ≤ slots - index ∧
      (∀ letter ∈ sent, fits letter.M letter.message) ∧
      WinsAt fits slots after draws 0 (fun result _ _ => goal result) ∧
      ∀ wire, (∀ (place : ℕ) (inside : place < sent.length),
        @read sent[place].M sent[place].inhabited wire (index + place) = sent[place].message) →
        settle read wire game index = after := by
  induction game generalizing index with
  | pure result => exact ⟨[], .pure result, Nat.zero_le _, by simp, wins, fun _ _ => rfl⟩
  | draw C bad next _ =>
      refine ⟨[], .draw C bad next, Nat.zero_le _, by simp, ?_, fun _ _ => rfl⟩
      cases draws <;> exact wins
  | recv M next induction =>
      obtain ⟨room, message, sendable, won⟩ := wins
      obtain ⟨sent, after, short, each, later, reads⟩ := induction message won
      refine ⟨⟨M, message⟩ :: sent, after, by rw [List.length_cons]; omega, ?_, later,
        fun wire sends => ?_⟩
      · intro letter member
        rcases List.mem_cons.mp member with rfl | member
        · exact sendable
        · exact each _ member
      · have head : read M wire index = message := sends 0 (Nat.succ_pos _)
        show settle read wire (next (read M wire index)) (index + 1) = after
        rw [head]
        exact reads wire fun place inside => by
          have sentAt := sends (place + 1) (Nat.succ_lt_succ inside)
          rwa [show index + (place + 1) = index + 1 + place by omega] at sentAt

variable {fits slots}

theorem advance_winsAt {game : Game α} {draws : ℕ} (reply : Reply) (settled : Settled game)
    (wins : WinsAt fits slots game (draws + 1) 0 fun result _ _ => goal result) :
    WinsAt fits slots (advance reply game) draws 0 fun result _ _ => goal result := by
  cases game with
  | pure result => exact wins
  | draw C bad next => exact wins _
  | recv M next => exact settled.elim

theorem ended_of_winsAt {game : Game α} (settled : Settled game)
    (wins : WinsAt fits slots game 0 0 fun result _ _ => goal result) :
    ∃ result, game = .pure result ∧ goal result := by
  cases game with
  | pure result => exact ⟨result, rfl, wins⟩
  | draw C bad next => exact wins.elim
  | recv M next => exact settled.elim

variable {read}

/-- The transcript prefix of the honest prover for a game with `draws` draws and `WinsAt`: for
the answers `reply` of the oracle, which can read the earlier messages, wire messages exist such
that the game ends in the goal. Wire message `index` has the data `data index after` for the game
`after` at its next draw or at its end, and `invariant` is a property of the prefix that each
step preserves. -/
theorem play_carried {dataOf : Wire → Data} (carries : Carries fits slots read dataOf)
    {start : Game α} {draws : ℕ}
    (wins : WinsAt fits slots start draws 0 fun result _ _ => goal result)
    (data : ℕ → Game α → Data) (reply : Wire → List (Reply × Wire) → Reply)
    (invariant : Wire → List (Reply × Wire) → Prop)
    (base : ∀ first, dataOf first = data 0 (settle read first start 0) → invariant first [])
    (step : ∀ first moves wire, moves.length < draws → invariant first moves →
      dataOf wire = data (moves.length + 1)
        (position read start first (moves ++ [(reply first moves, wire)])) →
      invariant first (moves ++ [(reply first moves, wire)])) :
    ∃ first moves result, moves.length = draws ∧ invariant first moves ∧
      position read start first moves = .pure result ∧ goal result := by
  have build : ∀ count ≤ draws, ∃ first moves, moves.length = count ∧ invariant first moves ∧
      Settled (position read start first moves) ∧
      WinsAt fits slots (position read start first moves) (draws - count) 0
        fun result _ _ => goal result := by
    intro count
    induction count with
    | zero =>
        intro _
        obtain ⟨sent, after, short, each, won, reads⟩ := settle_winsAt fits slots read wins
        obtain ⟨first, held, sends⟩ := carries (data 0 after) sent short each
        have same : settle read first start 0 = after := reads first fun place inside => by
          rw [Nat.zero_add]
          exact sends place inside
        refine ⟨first, [], rfl, base first (by rw [same]; exact held),
          settle_settled read first start 0, ?_⟩
        show WinsAt fits slots (settle read first start 0) _ 0 _
        rw [same]
        exact won
    | succ count induction =>
        intro inside
        obtain ⟨first, moves, length, kept, settled, won⟩ := induction (by omega)
        rw [show draws - count = draws - (count + 1) + 1 by omega] at won
        obtain ⟨sent, after, short, each, later, reads⟩ := settle_winsAt fits slots read
          (advance_winsAt (reply first moves) settled won)
        obtain ⟨wire, held, sends⟩ := carries (data (count + 1) after) sent short each
        have same : position read start first (moves ++ [(reply first moves, wire)]) = after := by
          rw [position_append]
          exact reads wire fun place inside => by
            rw [Nat.zero_add]
            exact sends place inside
        refine ⟨first, moves ++ [(reply first moves, wire)], by simp [length],
          step first moves wire (by omega) kept (by rw [same, length]; exact held), ?_, ?_⟩
        · rw [position_append]
          exact settle_settled read wire _ 0
        · rw [same]
          exact later
  obtain ⟨first, moves, length, kept, settled, won⟩ := build draws le_rfl
  rw [Nat.sub_self] at won
  obtain ⟨result, ended, good⟩ := ended_of_winsAt settled won
  exact ⟨first, moves, result, length, kept, ended, good⟩

end Play

section Last

variable {Wire Reply : Type} {total : ℕ}

/-- The last wire message of a transcript prefix. -/
def lastWire (first : Wire) (moves : List (Reply × Wire)) : Wire :=
  (moves.getLast?.map Prod.snd).getD first

theorem lastWire_concat (first : Wire) (moves : List (Reply × Wire)) (move : Reply × Wire) :
    lastWire first (moves ++ [move]) = move.2 := by
  simp [lastWire]

theorem lastWire_append_cons (first : Wire) (before after : List (Reply × Wire))
    (move : Reply × Wire) :
    lastWire first (before ++ move :: after) = lastWire move.2 after := by
  rcases List.eq_nil_or_concat' after with rfl | ⟨init, last, rfl⟩
  · exact lastWire_concat first before move
  · rw [lastWire_concat, ← List.cons_append, ← List.append_assoc, lastWire_concat]

theorem Prefix.wire_length {point : Prefix Wire Reply total} {before after : List (Reply × Wire)}
    (shape : point.rest = before ++ after) :
    point.wire before.length = lastWire point.first before := by
  rcases List.eq_nil_or_concat' before with rfl | ⟨init, move, rfl⟩
  · rfl
  · rw [lastWire_concat]
    exact point.wire_of (before := init) (after := after)
      (by rw [shape, List.append_assoc, List.singleton_append]) (by simp)

end Last

namespace Tier

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Wire Coord Slot Result Source Data : Type} [∀ b, Zero (Ring b)]
  (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source) {total : ℕ}
  [Fintype Slot] [Nonempty Slot] [Fintype Coord] [DecidableEq Coord]
  {accept : Prefix Wire (Coord → Slot) total → Prop}
  (table : Prefix Wire (Coord → Slot) total → Coord → Slot)

omit [Fintype Slot] [Fintype Coord] [DecidableEq Coord] in
/-- The fold challenges of the level that the table of the oracle gives after the transcript
prefix with the message `first` and the moves `early`. -/
noncomputable def drawn (first : Wire) (early : List ((Coord → Slot) × Wire)) (b : Block) :
    Ring b :=
  (T.decode b (answer table first early (T.block b))).getD 0

omit [Fintype Slot] [Fintype Coord] [DecidableEq Coord] in
theorem drawn_mem {first : Wire} {early : List ((Coord → Slot) × Wire)}
    (short : early.length ≤ total)
    (decodes : ∀ input b, (T.decode b (table input (T.block b))).isSome) (b : Block) :
    T.decode b (answer table first early (T.block b)) = some (T.drawn table first early b) :=
  decoded (by
    rw [answer_eq table first short]
    exact decodes _ b)

variable {T table}

/-- The transcript prefix of the honest prover through the stages of one level. `early` has the
moves before the fold challenges of the level, on the path of the table of the oracle. `wins`:
the prover has a message for each draw sequence of the checks for the claim of the last message of
`early`, the fold challenges `Tier.drawn` and the source `source`. Then a message `wire` and
moves exist such that the prefix is on the path, `wire` and the last message have the data
`data`, the game of the checks ends in the goal, and each transcript with this prefix has the
stages of the level, fold challenges that decode to `Tier.drawn`, and the game `position` of
these moves for each source. -/
theorem run {fits : (M : Type) → M → Prop} {slots : ℕ} {dataOf : Wire → Data}
    (carries : Carries fits slots T.read dataOf) {goal : Result → Prop} {first : Wire}
    {early : List ((Coord → Slot) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = T.start) (room : T.finish ≤ total)
    (decodes : ∀ input b, (T.decode b (table input (T.block b))).isSome) (source : Source)
    (wins : WinsAt fits slots
      (T.check (lastWire first early) (T.drawn table first early) source) T.costs.length 0
      fun result _ _ => goal result)
    (data : ℕ → Game Result → Data) :
    ∃ (wire : Wire) (moves : List ((Coord → Slot) × Wire)) (result : Result),
      moves.length = T.costs.length ∧ goal result ∧
      OnPath accept table first (early ++ (answer table first early, wire) :: moves) ∧
      dataOf wire = data 0 (settle T.read wire
        (T.check (lastWire first early) (T.drawn table first early) source) 0) ∧
      dataOf (lastWire wire moves) = data moves.length (.pure result) ∧
      position T.read (T.check (lastWire first early) (T.drawn table first early) source) wire
        moves = .pure result ∧
      ∀ stage : Prefix Wire (Coord → Slot) total, stage.first = first →
        (∃ later, stage.rest = (early ++ (answer table first early, wire) :: moves) ++ later) →
        T.finish ≤ stage.rest.length ∧ T.Decodes stage ∧
        stage.wire T.start = lastWire first early ∧ stage.wire (T.start + 1) = wire ∧
        stage.wire T.finish = lastWire wire moves ∧
        ∀ candidate, T.game candidate stage =
          position T.read (T.check (lastWire first early) (T.drawn table first early) candidate)
            wire moves := by
  have ends := T.finish_eq
  obtain ⟨reply0, answered⟩ : ∃ reply, reply = answer table first early := ⟨_, rfl⟩
  obtain ⟨wire, moves, result, counted, ⟨on', held, last⟩, ended, good⟩ :=
    play_carried carries wins data
      (fun wire moves => answer table first (early ++ (reply0, wire) :: moves))
      (fun wire moves => OnPath accept table first (early ++ (reply0, wire) :: moves) ∧
        dataOf wire = data 0 (settle T.read wire
          (T.check (lastWire first early) (T.drawn table first early) source) 0) ∧
        dataOf (lastWire wire moves) = data moves.length (position T.read
          (T.check (lastWire first early) (T.drawn table first early) source) wire moves))
      (fun wire kept => ⟨by
        have next := on.snoc (by omega) wire
        rwa [← answered] at next, kept, kept⟩)
      (fun wire moves later inside kept linked => ⟨by
        have next := kept.1.snoc (by rw [List.length_append, List.length_cons]; omega) later
        rwa [List.append_assoc, List.cons_append] at next, kept.2.1, by
        rw [lastWire_concat, List.length_append, List.length_singleton]
        exact linked⟩)
  rw [ended, counted] at last
  refine ⟨wire, moves, result, counted, good, answered ▸ on', held, counted ▸ last, ended,
    fun stage rooted ⟨later, shape⟩ => ?_⟩
  rw [← answered] at shape
  have head : stage.rest = early ++ (reply0, wire) :: (moves ++ later) := by
    rw [shape, List.append_assoc, List.cons_append]
  have replied : stage.reply T.start = some reply0 := stage.reply_of head starts.symm
  have short : early.length ≤ total := by omega
  have whole : (early ++ (reply0, wire) :: moves).length = T.finish := by
    rw [List.length_append, List.length_cons]
    omega
  have fold : T.challenge stage = T.drawn table first early := by
    funext b
    show ((stage.reply T.start).bind fun reply => T.decode b (reply (T.block b))).getD 0 = _
    rw [replied, answered]
    rfl
  have claimed : stage.wire T.start = lastWire first early := by
    rw [← starts, ← rooted]
    exact Prefix.wire_length head
  have opened : stage.wire (T.start + 1) = wire :=
    stage.wire_of head (congrArg (· + 1) starts.symm)
  refine ⟨by rw [shape, List.length_append, whole]; omega,
    ⟨reply0, replied, fun b => Option.isSome_iff_exists.mpr
      ⟨_, answered ▸ T.drawn_mem table short decodes b⟩⟩, claimed, opened, ?_,
    fun candidate => ?_⟩
  · rw [← whole, Prefix.wire_length shape, lastWire_append_cons]
  · have moved : (stage.rest.take T.finish).drop (T.start + 1) = moves := by
      rw [shape, List.take_left' whole, List.append_cons]
      exact List.drop_left' (by rw [List.length_append, List.length_singleton, starts])
    show position T.read (T.check (stage.wire T.start) (T.challenge stage) candidate)
      (stage.wire (T.start + 1)) ((stage.rest.take T.finish).drop (T.start + 1)) = _
    rw [moved, fold, claimed, opened]

end Tier

end NightstreamFPrime.Spec.Compression.Akita.Extract
