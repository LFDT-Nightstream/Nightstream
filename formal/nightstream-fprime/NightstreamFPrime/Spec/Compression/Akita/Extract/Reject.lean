import NightstreamFPrime.Spec.Compression.Akita.Extract.Play
import NightstreamFPrime.Spec.Compression.Akita.Extract.Segment

/-! The honest prover of a public-coin game with reject sets on a Fiat–Shamir transcript
(SPEC.md Algorithm 9, Rejects; Akita, ePrint 2026/1983, revision of 2026-10-15, Theorem 10.9 and
Definition 10.16 (v)).

The honest prover of the compression argument fails when a lookup challenge is a table value and
when `ω` has no row point. `WinsAt` is for each draw sequence, so it is false for these games.

- `ends_position`: a property of each end of a game holds for its result after a transcript.
- `Hit set C challenge` states that the challenge of the type `C` is a value of `set`.
- `WinsOff reject` is `WinsAt` for the draw sequences with no challenge in `reject`, and
  `WinsAt.off` gives it from `WinsAt`.
- `play_off` gives the wire messages of the honest prover for a game with `WinsOff` that starts
  at a draw, for answers of the oracle that decode to no challenge in `reject`.
- `Segment.run` gives the transcript prefix of the honest prover through the stages of a
  checkpoint segment: the game of the segment ends in the goal for each transcript with this
  prefix. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

section Ends

variable {Wire Reply α : Type} [Fintype Reply] (read : Reader Wire)

theorem settle_of_settled (wire : Wire) {game : Game α} (settled : Settled game) (index : ℕ) :
    settle read wire game index = game := by
  cases game with
  | pure result => rfl
  | draw C bad next => rfl
  | recv M next => exact settled.elim

theorem ends_settle (wire : Wire) {game : Game α} {goal : α → Prop} (ends : game.Ends goal)
    (index : ℕ) : (settle read wire game index).Ends goal := by
  induction game generalizing index with
  | pure result => exact ends
  | draw C bad next _ => exact ends
  | recv M next induction => exact induction _ (ends _) _

theorem ends_advance {game : Game α} {goal : α → Prop} (ends : game.Ends goal) (reply : Reply) :
    (advance reply game).Ends goal := by
  cases game with
  | pure result => exact ends
  | draw C bad next => exact ends _
  | recv M next => exact ends

/-- A property of each end of a game holds for the game after a transcript. -/
theorem ends_position {start : Game α} {goal : α → Prop} (ends : start.Ends goal) (first : Wire)
    (moves : List (Reply × Wire)) : (position read start first moves).Ends goal := by
  induction moves using List.reverseRecOn with
  | nil => exact ends_settle read first ends 0
  | append_singleton moves move induction =>
      obtain ⟨reply, wire⟩ := move
      rw [position_append]
      exact ends_settle read wire (ends_advance induction reply) 0

end Ends

/-- The challenge of the type `C` is a value of the set `set` of challenges of the type `T`. -/
def Hit {T : Type} (set : Set T) (C : Type) (challenge : C) : Prop :=
  ∃ same : C = T, cast same challenge ∈ set

theorem hit_self {T : Type} {set : Set T} {challenge : T} :
    Hit set T challenge ↔ challenge ∈ set :=
  ⟨fun ⟨_, member⟩ => member, fun member => ⟨rfl, member⟩⟩

section Wins

variable {α β : Type} (reject : (C : Type) → C → Prop) (fits : (M : Type) → M → Prop)
  (slots : ℕ)

/-- `WinsAt` for the draw sequences with no challenge in `reject`: a prover has a message for
each such sequence such that the game ends with `tail`. -/
def WinsOff : Game α → ℕ → ℕ → (α → ℕ → ℕ → Prop) → Prop
  | Game.pure result, draws, index, tail => tail result draws index
  | Game.draw _ _ _, 0, _, _ => False
  | Game.draw C _ next, rest + 1, _, tail =>
      ∀ challenge, ¬ reject C challenge → WinsOff (next challenge) rest 0 tail
  | Game.recv M next, draws, index, tail =>
      index < slots ∧ ∃ message, fits M message ∧ WinsOff (next message) draws (index + 1) tail

theorem winsOff_bind (game : Game α) (after : α → Game β) (draws index : ℕ)
    (tail : β → ℕ → ℕ → Prop) :
    WinsOff reject fits slots (game.bind after) draws index tail ↔
      WinsOff reject fits slots game draws index fun result rest place =>
        WinsOff reject fits slots (after result) rest place tail := by
  induction game generalizing draws index with
  | pure result => exact Iff.rfl
  | draw C bad next induction =>
      cases draws with
      | zero => exact Iff.rfl
      | succ rest =>
          exact forall_congr' fun challenge => imp_congr_right fun _ => induction challenge rest 0
  | recv M next induction =>
      exact and_congr_right fun _ => exists_congr fun message =>
        and_congr_right fun _ => induction message draws (index + 1)

variable {reject fits slots}

theorem WinsOff.mono {game : Game α} {draws index : ℕ} {tail other : α → ℕ → ℕ → Prop}
    (wins : WinsOff reject fits slots game draws index tail)
    (weaker : ∀ result rest place, tail result rest place → other result rest place) :
    WinsOff reject fits slots game draws index other := by
  induction game generalizing draws index with
  | pure result => exact weaker _ _ _ wins
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest => exact fun challenge allowed => induction challenge (wins challenge allowed)
  | recv M next induction =>
      obtain ⟨room, message, sendable, won⟩ := wins
      exact ⟨room, message, sendable, induction message won⟩

/-- A game with `WinsAt` has `WinsOff` for each reject predicate. -/
theorem WinsAt.off {game : Game α} {draws index : ℕ} {tail : α → ℕ → ℕ → Prop}
    (wins : WinsAt fits slots game draws index tail) :
    WinsOff reject fits slots game draws index tail := by
  induction game generalizing draws index with
  | pure result => exact wins
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest => exact fun challenge _ => induction challenge (wins challenge)
  | recv M next induction =>
      obtain ⟨room, message, sendable, won⟩ := wins
      exact ⟨room, message, sendable, induction message won⟩

/-- A game with `WinsOff` for `draws` draws has it for `draws + extra` draws, and it ends with
`extra` remaining draws or more. -/
theorem WinsOff.more {game : Game α} {draws index : ℕ} {tail : α → ℕ → Prop}
    (wins : WinsOff reject fits slots game draws index fun result _ place => tail result place)
    (extra : ℕ) :
    WinsOff reject fits slots game (draws + extra) index fun result rest place =>
      extra ≤ rest ∧ tail result place := by
  induction game generalizing draws index with
  | pure result => exact ⟨Nat.le_add_left _ _, wins⟩
  | draw C bad next induction =>
      cases draws with
      | zero => exact wins.elim
      | succ rest =>
          rw [Nat.add_right_comm]
          exact fun challenge allowed => induction challenge (wins challenge allowed)
  | recv M next induction =>
      obtain ⟨room, message, sendable, won⟩ := wins
      exact ⟨room, message, sendable, induction message won⟩

end Wins

section Play

variable {Wire Data Reply α : Type} [Fintype Reply] {reject : (C : Type) → C → Prop}
  {fits : (M : Type) → M → Prop} {slots : ℕ} (read : Reader Wire) {goal : α → Prop}

/-- `settle_winsAt` for `WinsOff`. -/
theorem settle_winsOff {game : Game α} {draws index : ℕ}
    (wins : WinsOff reject fits slots game draws index fun result _ _ => goal result) :
    ∃ (sent : List Sent) (after : Game α), sent.length ≤ slots - index ∧
      (∀ letter ∈ sent, fits letter.M letter.message) ∧
      WinsOff reject fits slots after draws 0 (fun result _ _ => goal result) ∧
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

variable {read}

theorem advance_winsOff {game : Game α} {draws : ℕ} (reply : Reply) (settled : Settled game)
    (dodges : ∀ (C : Type) [Fintype C] [Nonempty C], ¬ reject C (sample Reply C reply))
    (wins : WinsOff reject fits slots game (draws + 1) 0 fun result _ _ => goal result) :
    WinsOff reject fits slots (advance reply game) draws 0 fun result _ _ => goal result := by
  cases game with
  | pure result => exact wins
  | draw C bad next => exact wins _ (dodges C)
  | recv M next => exact settled.elim

omit [Fintype Reply] in
theorem ended_of_winsOff {game : Game α} (settled : Settled game)
    (wins : WinsOff reject fits slots game 0 0 fun result _ _ => goal result) :
    ∃ result, game = .pure result ∧ goal result := by
  cases game with
  | pure result => exact ⟨result, rfl, wins⟩
  | draw C bad next => exact wins.elim
  | recv M next => exact settled.elim

/-- The moves of the honest prover for a game with `draws` draws and `WinsOff` that starts at a
draw, after the wire message `first`. The answers `reply` of the oracle decode to no challenge in
`reject`. Moves exist such that the game ends in the goal. Each wire message has the data `data`
of the game at its next draw or at its end, and `invariant` is a property of the moves that each
step preserves. -/
theorem play_off {dataOf : Wire → Data} (carries : Carries fits slots read dataOf)
    {start : Game α} {draws : ℕ} (settled : Settled start)
    (wins : WinsOff reject fits slots start draws 0 fun result _ _ => goal result)
    (first : Wire) (data : Game α → Data) (reply : List (Reply × Wire) → Reply)
    (dodges : ∀ moves, moves.length < draws → ∀ (C : Type) [Fintype C] [Nonempty C],
      ¬ reject C (sample Reply C (reply moves)))
    (invariant : List (Reply × Wire) → Prop) (base : invariant [])
    (step : ∀ moves wire, moves.length < draws → invariant moves →
      dataOf wire = data (position read start first (moves ++ [(reply moves, wire)])) →
      invariant (moves ++ [(reply moves, wire)])) :
    ∃ moves result, moves.length = draws ∧ invariant moves ∧
      position read start first moves = .pure result ∧ goal result := by
  have same : settle read first start 0 = start := settle_of_settled read first settled 0
  have build : ∀ count ≤ draws, ∃ moves, moves.length = count ∧ invariant moves ∧
      Settled (position read start first moves) ∧
      WinsOff reject fits slots (position read start first moves) (draws - count) 0
        fun result _ _ => goal result := by
    intro count
    induction count with
    | zero =>
        intro _
        refine ⟨[], rfl, base, ?_, ?_⟩
        · show Settled (settle read first start 0)
          rw [same]
          exact settled
        · show WinsOff reject fits slots (settle read first start 0) _ 0 _
          rw [same]
          exact wins
    | succ count induction =>
        intro inside
        obtain ⟨moves, length, kept, ready, won⟩ := induction (by omega)
        rw [show draws - count = draws - (count + 1) + 1 by omega] at won
        obtain ⟨sent, after, short, each, later, reads⟩ := settle_winsOff read
          (advance_winsOff (reply moves) ready (dodges moves (by omega)) won)
        obtain ⟨wire, held, sends⟩ := carries (data after) sent short each
        have next : position read start first (moves ++ [(reply moves, wire)]) = after := by
          rw [position_append]
          exact reads wire fun place inside => by
            rw [Nat.zero_add]
            exact sends place inside
        refine ⟨moves ++ [(reply moves, wire)], by simp [length],
          step moves wire (by omega) kept (by rw [next]; exact held), ?_, ?_⟩
        · rw [position_append]
          exact settle_settled read wire _ 0
        · rw [next]
          exact later
  obtain ⟨moves, length, kept, ready, won⟩ := build draws le_rfl
  rw [Nat.sub_self] at won
  obtain ⟨result, ended, good⟩ := ended_of_winsOff ready won
  exact ⟨moves, result, length, kept, ended, good⟩

end Play

namespace Segment

variable {Wire Coord Slot Result Cand Data : Type} {total : ℕ}
  (G : Segment Wire Coord Slot Result Cand total) [Fintype Slot] [Nonempty Slot] [Fintype Coord]
  [DecidableEq Coord] {accept : Prefix Wire (Coord → Slot) total → Prop}
  {table : Prefix Wire (Coord → Slot) total → Coord → Slot}

/-- The transcript prefix of the honest prover through the stages of a checkpoint segment.
`early` has the moves before the segment, on the path of the table of the oracle. `settled` and
`wins`: the game of the segment for the stage input of `early` and the candidate starts at a
draw, and the prover has a message for each draw sequence with no challenge in `reject`.
`dodges`: an answer of the table decodes to no challenge in `reject`. Then moves exist such that
the prefix is on the path, the last message has the data `data` of the end of the game, and for
each transcript with this prefix the game of the segment ends in a result with the goal. -/
theorem run {reject : (C : Type) → C → Prop} {fits : (M : Type) → M → Prop} {slots : ℕ}
    {dataOf : Wire → Data} (carries : Carries fits slots G.read dataOf) {goal : Result → Prop}
    {first : Wire} {early : List ((Coord → Slot) × Wire)} (on : OnPath accept table first early)
    (starts : early.length = G.start) (room : G.finish ≤ total) (short : early.length ≤ total)
    (candidate : Cand) (settled : Settled (G.check ⟨first, early, short⟩ candidate))
    (wins : WinsOff reject fits slots (G.check ⟨first, early, short⟩ candidate) G.costs.length 0
      fun result _ _ => goal result)
    (dodges : ∀ input (C : Type) [Fintype C] [Nonempty C],
      ¬ reject C (sample (Coord → Slot) C (table input)))
    (draws : G.costs ≠ []) (data : Game Result → Data) :
    ∃ (moves : List ((Coord → Slot) × Wire)) (result : Result),
      moves.length = G.costs.length ∧ goal result ∧ OnPath accept table first (early ++ moves) ∧
      dataOf (lastWire first (early ++ moves)) = data (.pure result) ∧
      ∀ stage : Prefix Wire (Coord → Slot) total, stage.first = first →
        (∃ later, stage.rest = early ++ moves ++ later) →
        stage.wire G.finish = lastWire first (early ++ moves) ∧
          G.game candidate stage = .pure result := by
  have ends : G.finish = G.start + G.costs.length := rfl
  obtain ⟨moves, result, counted, ⟨on', held⟩, ended, good⟩ :=
    play_off (read := G.read) carries settled wins (lastWire first early) data
      (fun moves => answer table first (early ++ moves))
      (fun moves inside C _ _ => by
        rw [answer_eq table first (show (early ++ moves).length ≤ total by
          rw [List.length_append]; omega)]
        exact dodges _ C)
      (fun moves => OnPath accept table first (early ++ moves) ∧
        (moves ≠ [] → dataOf (lastWire first (early ++ moves)) =
          data (position G.read (G.check ⟨first, early, short⟩ candidate) (lastWire first early)
            moves)))
      ⟨by rwa [List.append_nil], fun different => (different rfl).elim⟩
      (fun moves wire inside kept linked => ⟨by
        have next := kept.1.snoc (by rw [List.length_append]; omega) wire
        rwa [List.append_assoc] at next, fun _ => by
        rw [← List.append_assoc, lastWire_concat]
        exact linked⟩)
  have some : moves ≠ [] := fun empty => draws (List.length_eq_zero_iff.mp (by
    rw [← counted, empty]
    rfl))
  have whole : (early ++ moves).length = G.finish := by
    rw [List.length_append, starts, counted, ends]
  refine ⟨moves, result, counted, good, on', by rw [held some, ended],
    fun stage rooted ⟨later, shape⟩ => ⟨?_, ?_⟩⟩
  · rw [← whole, ← rooted]
    exact Prefix.wire_length shape
  · have taken : stage.take G.start = ⟨first, early, short⟩ := Prefix.ext rooted (by
      show stage.rest.take G.start = early
      rw [shape, List.append_assoc]
      exact List.take_left' starts)
    have claimed : stage.wire G.start = lastWire first early := by
      rw [← starts, ← rooted]
      exact Prefix.wire_length (by rw [shape, List.append_assoc])
    have moved : (stage.rest.take G.finish).drop G.start = moves := by
      rw [shape, List.take_left' whole]
      exact List.drop_left' starts
    show position G.read (G.check (stage.take G.start) candidate) (stage.wire G.start)
      ((stage.rest.take G.finish).drop G.start) = _
    rw [moved, taken, claimed]
    exact ended

end Segment

end NightstreamFPrime.Spec.Compression.Akita.Extract
