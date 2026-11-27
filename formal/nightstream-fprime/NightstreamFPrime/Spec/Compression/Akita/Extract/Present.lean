import NightstreamFPrime.Spec.Compression.Game.Path
import NightstreamFPrime.Spec.Compression.Game.SumCheck
import NightstreamFPrime.Spec.Compression.Akita.Extract.Tier
import NightstreamFPrime.Spec.Compression.Akita.Extract.Decoders

/-! A run of a game on a path of challenges and messages (`Game.Runs`) and the transcript form of the soundness claims
(`position`, `Segment.game`, `Tier.game`; Akita, ePrint 2026/1983, revision of 2026-11-23,
Section 10.3).

- `Fair.mass_eq`: a fair decoder gives each challenge the mass `1/|C|`, so `sample` gives each
  challenge for some answer (`sample_surjective`).
- `pathOf read game wire index moves` is the path of a game on a transcript segment: a message is
  `read.message` of the current wire message at the index of `settle`, and a draw is `sample` of
  the next answer, after which the next wire message starts at index 0. The path stops at the end
  of the game and at a draw with no answer left. `Presents read first moves path game` states that
  this path from the wire message `first` is `path`.
- `position_eq_pure_iff`: the game ends in `result` after the segment exactly when it runs to
  `result` on its path. `Presents.position_iff`, `Segment.game_eq_pure_iff` and
  `Tier.game_eq_pure_iff` are this statement for a presented path.
- `chunkReader Chunk` reads message `index` of a wire message `ℕ → Chunk` from its chunk `index`
  by a map onto the type of the message. `Presentable Reply Chunk game` states that each draw of
  the game has a fair decoder from `Reply` and that `Chunk` maps onto the type of each message.
  `exists_presents`: each path on which such a game runs has a transcript segment that presents it.
- `listChunk_onto`: the lists of `E` map onto `E`, `E × E`, `ℕ` and `Fin n → E`, the message
  types of the checks. `presentable_sumcheck`: `Game.sumcheck` is presentable for the answers
  `Coord → Slot` with `FieldSlot Slot K` and the chunks `List K`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (mass)
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Game (Move)

attribute [local instance low] Classical.propDecidable

section Sampler

variable {Reply C : Type} [Fintype Reply] [Nonempty Reply] [Fintype C] [Nonempty C]

/-- A fair decoder gives each challenge the mass `1/|C|`. -/
theorem Fair.mass_eq {decode : Reply → C} (fair : Fair decode) (c : C) :
    mass {reply | decode reply = c} = 1 / Fintype.card C := by
  have upper := fair {c}
  have lower := fair (Finset.univ.erase c)
  simp only [Finset.mem_singleton, Finset.card_singleton, Nat.cast_one] at upper
  have total : mass {reply | decode reply = c} +
      mass {reply | decode reply ∈ Finset.univ.erase c} = 1 := by
    unfold mass
    rw [← Finset.expect_add_distrib]
    calc _ = 𝔼 _reply : Reply, (1 : ℝ) := Finset.expect_congr rfl fun reply _ => by
            by_cases same : decode reply = c <;> simp [same]
      _ = 1 := Finset.expect_const Finset.univ_nonempty 1
  have positive : (0 : ℝ) < Fintype.card C := by exact_mod_cast Fintype.card_pos
  rw [Finset.card_erase_of_mem (Finset.mem_univ c), Finset.card_univ,
    Nat.cast_sub (Nat.one_le_iff_ne_zero.mpr Fintype.card_ne_zero), Nat.cast_one, sub_div,
    div_self positive.ne'] at lower
  exact le_antisymm upper (by linarith)

/-- With a fair decoder from `Reply` to `C`, `sample` gives each challenge for some answer. -/
theorem sample_surjective (fair : ∃ decode : Reply → C, Fair decode) :
    Function.Surjective (sample Reply C) := by
  intro c
  by_contra missed
  have empty : {reply | sample Reply C reply = c} = (∅ : Set Reply) :=
    Set.eq_empty_of_forall_notMem fun reply same => missed ⟨reply, same⟩
  have value := (sample_fair fair).mass_eq c
  rw [empty] at value
  have zero : mass (∅ : Set Reply) = 0 := by simp [mass]
  have positive : (0 : ℝ) < 1 / Fintype.card C :=
    one_div_pos.mpr (by exact_mod_cast Fintype.card_pos)
  linarith

end Sampler

section Path

variable {Wire Reply α : Type} [Fintype Reply] (read : Reader Wire)

/-- The path of a game on a transcript segment: the messages of the wire message `wire` from
message `index`, and for each answer and wire message of `moves` the challenge that `sample` gives
for the answer and then the messages of this wire message from message 0. -/
noncomputable def pathOf : Game α → Wire → ℕ → List (Reply × Wire) → List Move
  | .pure _, _, _, _ => []
  | @Game.draw _ _ _ _ _ _, _, _, [] => []
  | @Game.draw _ C finite nonempty _ next, _, _, move :: moves =>
      Move.draw (@sample Reply _ C finite nonempty move.1) ::
        pathOf (next (@sample Reply _ C finite nonempty move.1)) move.2 0 moves
  | @Game.recv _ M nonempty next, wire, index, moves =>
      Move.recv (@Reader.message _ read M nonempty wire index) ::
        pathOf (next (@Reader.message _ read M nonempty wire index)) wire (index + 1) moves

/-- The segment with the wire message `first` and the answers and wire messages `moves` presents
the path `path` of the game: `path` is the path of the game on the segment. -/
def Presents (first : Wire) (moves : List (Reply × Wire)) (path : List Move) (game : Game α) :
    Prop :=
  pathOf read game first 0 moves = path

theorem foldl_pure (moves : List (Reply × Wire)) (result : α) :
    moves.foldl (fun game move => settle read move.2 (advance move.1 game) 0) (.pure result) =
      .pure result := by
  induction moves with
  | nil => rfl
  | cons move moves induction => exact induction

theorem foldl_settle_eq_pure_iff (game : Game α) (wire : Wire) (index : ℕ)
    (moves : List (Reply × Wire)) (result : α) :
    moves.foldl (fun game move => settle read move.2 (advance move.1 game) 0)
        (settle read wire game index) = .pure result ↔
      game.Runs (pathOf read game wire index moves) result := by
  induction game generalizing wire index moves with
  | pure a =>
      show moves.foldl _ (Game.pure a) = Game.pure result ↔ a = result
      rw [foldl_pure]
      exact ⟨Game.pure.inj, congrArg Game.pure⟩
  | draw C bad next induction =>
      cases moves with
      | nil =>
          show Game.draw C bad next = .pure result ↔ False
          exact iff_false_intro fun ended => by cases ended
      | cons move moves =>
          exact (induction (sample Reply C move.1) move.2 0 moves).trans
            ⟨fun runs => ⟨rfl, runs⟩, fun ⟨_, runs⟩ => by rwa [cast_eq] at runs⟩
  | recv M next induction =>
      exact (induction (read.message M wire index) wire (index + 1) moves).trans
        ⟨fun runs => ⟨rfl, runs⟩, fun ⟨_, runs⟩ => by rwa [cast_eq] at runs⟩

/-- The game ends in `result` after a transcript segment exactly when it runs to `result` on its
path on the segment. -/
theorem position_eq_pure_iff (start : Game α) (first : Wire) (moves : List (Reply × Wire))
    (result : α) :
    position read start first moves = .pure result ↔
      start.Runs (pathOf read start first 0 moves) result :=
  foldl_settle_eq_pure_iff read start first 0 moves result

variable {read}

/-- On a presented path, the game ends in `result` after the segment exactly when it runs to
`result` on the path. -/
theorem Presents.position_iff {first : Wire} {moves : List (Reply × Wire)} {path : List Move}
    {game : Game α} (presents : Presents read first moves path game) (result : α) :
    position read game first moves = .pure result ↔ game.Runs path result := by
  rw [Extract.position_eq_pure_iff, show pathOf read game first 0 moves = path from presents]

/-- A run of a game on a presented path ends the game after the segment in its result. -/
theorem position_of_runs {first : Wire} {moves : List (Reply × Wire)} {path : List Move}
    {game : Game α} {result : α} (runs : game.Runs path result)
    (presents : Presents read first moves path game) :
    position read game first moves = .pure result :=
  (presents.position_iff result).mpr runs

end Path

section Forms

variable {Wire Coord Slot Result : Type} {total : ℕ} [Fintype Slot] [Fintype Coord]
  [DecidableEq Coord]

/-- `Presents.position_iff` for the game of a checkpoint segment: its moves are the answers
without the work slot and the wire messages of the stages of the segment. -/
theorem Segment.game_eq_pure_iff {Cand : Type} (G : Segment Wire Coord Slot Result Cand total)
    (candidate : Cand) (point : Prefix Wire (Option Coord → Slot) total) {path : List Move}
    (presents : Presents G.read (point.wire G.start)
      (((point.rest.take G.finish).drop G.start).map fun move => (view move.1, move.2)) path
      (G.check (point.take G.start) candidate))
    (result : Result) :
    G.game candidate point = .pure result ↔
      (G.check (point.take G.start) candidate).Runs path result :=
  presents.position_iff result

variable {Block : Type} {Ring : Block → Type} {Stmt Brk Wit Weak Source : Type}
  [∀ b, Zero (Ring b)] (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source)

/-- `Presents.position_iff` for the game of the checks of a level: its moves are the answers
without the work slot and the wire messages of the stages after the fold challenges. -/
theorem Tier.game_eq_pure_iff (source : Source) (point : Prefix Wire (Option Coord → Slot) total)
    {path : List Move}
    (presents : Presents T.read (point.wire (T.start + 1))
      (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2)) path
      (T.check (point.wire T.start) (T.challenge point) source))
    (result : Result) :
    T.game source point = .pure result ↔
      (T.check (point.wire T.start) (T.challenge point) source).Runs path result :=
  presents.position_iff result

/-- A run of the checks of a level on a presented path to a result that is true for the source,
on a transcript with the stages of the level and the checks `Tier.Decodes`, gives
`Tier.Passed`. -/
theorem Tier.passed_of_runs {source : Source} {point : Prefix Wire (Option Coord → Slot) total}
    {path : List Move} {result : Result} (long : T.finish ≤ point.rest.length)
    (decodes : T.Decodes point)
    (runs : (T.check (point.wire T.start) (T.challenge point) source).Runs path result)
    (presents : Presents T.read (point.wire (T.start + 1))
      (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2)) path
      (T.check (point.wire T.start) (T.challenge point) source))
    (passes : T.passes source result) : T.Passed source point :=
  ⟨long, decodes, result, (T.game_eq_pure_iff source point presents result).mpr runs, passes⟩

end Forms

section Chunk

variable {Reply Chunk α : Type} [Fintype Reply]

/-- A map from `Chunk` onto the type `M`, when one exists. -/
noncomputable def unchunk (Chunk M : Type) [Nonempty M] : Chunk → M :=
  if onto : ∃ map : Chunk → M, Function.Surjective map then onto.choose
  else fun _ => Classical.arbitrary M

theorem unchunk_surjective {M : Type} [Nonempty M]
    (onto : ∃ map : Chunk → M, Function.Surjective map) :
    Function.Surjective (unchunk Chunk M) := by
  unfold unchunk
  rw [dite_eq_left onto]
  exact onto.choose_spec

variable (Chunk) in
/-- The reader of the wire messages `ℕ → Chunk`: message `index` is `unchunk` of chunk `index`,
and the nonce of the response search is 0. -/
noncomputable def chunkReader : Reader (ℕ → Chunk) where
  message M _ wire index := unchunk Chunk M (wire index)
  nonce _ := 0

theorem chunkReader_message (M : Type) [Nonempty M] (wire : ℕ → Chunk) (index : ℕ) :
    (chunkReader Chunk).message M wire index = unchunk Chunk M (wire index) :=
  rfl

variable (Reply Chunk) in
/-- Each draw of the game has a fair decoder from the answers `Reply`, and `Chunk` maps onto the
type of each message of the game. -/
def Presentable : Game α → Prop
  | .pure _ => True
  | @Game.draw _ C finite _ _ next =>
      (∃ decode : Reply → C, @Fair Reply _ C finite decode) ∧ ∀ c, Presentable (next c)
  | @Game.recv _ M _ next =>
      (∃ map : Chunk → M, Function.Surjective map) ∧ ∀ m, Presentable (next m)

theorem Presentable.bind {β : Type} {game : Game α} {after : α → Game β}
    (first : Presentable Reply Chunk game) (second : ∀ a, Presentable Reply Chunk (after a)) :
    Presentable Reply Chunk (game.bind after) := by
  induction game with
  | pure a => exact second a
  | draw C bad next induction => exact ⟨first.1, fun c => induction c (first.2 c)⟩
  | recv M next induction => exact ⟨first.1, fun m => induction m (first.2 m)⟩

theorem Presentable.ite {c : Prop} [Decidable c] {yes no : Game α}
    (first : Presentable Reply Chunk yes) (second : Presentable Reply Chunk no) :
    Presentable Reply Chunk (if c then yes else no) := by
  split
  · exact first
  · exact second

/-- Each path on which a presentable game runs is the path of the game on wire messages `ℕ → Chunk`
and answers, from message `index` of a wire message that agrees with `wire` below `index`. -/
theorem exists_pathOf [Nonempty Reply] [Nonempty Chunk] (game : Game α) (path : List Move)
    (result : α) (runs : game.Runs path result) (presentable : Presentable Reply Chunk game)
    (wire : ℕ → Chunk) (index : ℕ) :
    ∃ (other : ℕ → Chunk) (moves : List (Reply × (ℕ → Chunk))),
      (∀ earlier < index, other earlier = wire earlier) ∧
        pathOf (chunkReader Chunk) game other index moves = path := by
  induction game generalizing path wire index with
  | pure a =>
      cases path with
      | nil => exact ⟨wire, [], fun _ _ => rfl, rfl⟩
      | cons move path => cases move <;> exact runs.elim
  | draw C bad next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | recv m => exact runs.elim
          | draw c =>
              obtain ⟨same, runs⟩ := runs
              subst same
              obtain ⟨reply, sampled⟩ := sample_surjective presentable.1 c
              obtain ⟨other, moves, -, presented⟩ := induction c path runs (presentable.2 c)
                (fun _ => Classical.arbitrary Chunk) 0
              refine ⟨wire, (reply, other) :: moves, fun _ _ => rfl, ?_⟩
              simp only [pathOf]
              rw [sampled, presented]
  | recv M next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | draw c => exact runs.elim
          | recv m =>
              obtain ⟨same, runs⟩ := runs
              subst same
              obtain ⟨chunk, decoded⟩ := unchunk_surjective presentable.1 m
              obtain ⟨other, moves, agree, presented⟩ := induction m path runs (presentable.2 m)
                (Function.update wire index chunk) (index + 1)
              have here : other index = chunk :=
                (agree index (Nat.lt_succ_self _)).trans (Function.update_self ..)
              refine ⟨other, moves, fun earlier below =>
                (agree earlier (Nat.lt_succ_of_lt below)).trans
                  (Function.update_of_ne (Nat.ne_of_lt below) ..), ?_⟩
              simp only [pathOf, chunkReader_message]
              rw [here, decoded, presented]

/-- Each path on which a presentable game runs has a transcript segment with wire messages
`ℕ → Chunk` that presents it, and the game ends in its result after this segment. -/
theorem exists_presents [Nonempty Reply] [Nonempty Chunk] {game : Game α} {path : List Move}
    {result : α} (runs : game.Runs path result) (presentable : Presentable Reply Chunk game) :
    ∃ (first : ℕ → Chunk) (moves : List (Reply × (ℕ → Chunk))),
      Presents (chunkReader Chunk) first moves path game ∧
        position (chunkReader Chunk) game first moves = .pure result := by
  obtain ⟨first, moves, -, presented⟩ :=
    exists_pathOf game path result runs presentable (fun _ => Classical.arbitrary Chunk) 0
  exact ⟨first, moves, presented, position_of_runs runs presented⟩

end Chunk

section Instances

/-- The lists of `E` map onto `E`, `E × E`, `ℕ` and `Fin n → E`: the types of the messages of the
checks. -/
theorem listChunk_onto {E : Type} [Zero E] (n : ℕ) :
    (∃ map : List E → E, Function.Surjective map) ∧
      (∃ map : List E → E × E, Function.Surjective map) ∧
      (∃ map : List E → ℕ, Function.Surjective map) ∧
      ∃ map : List E → Fin n → E, Function.Surjective map :=
  ⟨⟨fun list => list.headD 0, fun x => ⟨[x], rfl⟩⟩,
    ⟨fun list => (list.headD 0, list.tail.headD 0), fun pair => ⟨[pair.1, pair.2], rfl⟩⟩,
    ⟨List.length, fun count => ⟨List.replicate count 0, by simp⟩⟩,
    ⟨fun list index => list.getD index 0, fun vector => ⟨List.ofFn vector, funext fun index => by
      simp [List.getD_eq_getElem?_getD]⟩⟩⟩

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] {Coord : Type} [Fintype Coord]
  [DecidableEq Coord] [Nonempty Coord] {Slot : Type} [Fintype Slot] [Nonempty Slot]
  [FieldSlot Slot K]

/-- `Game.sumcheck` is presentable for the answers `Coord → Slot` and the chunks `List K`: each
draw reads one slot (`exists_fair_field`), and each message is a vector of coefficients. -/
theorem presentable_sumcheck (d : ℕ) (round : List K → Polynomial K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      Presentable (Coord → Slot) (List K) (Game.sumcheck d round rounds point claim) := by
  intro rounds
  induction rounds with
  | zero => exact fun _ _ => trivial
  | succ rounds induction =>
      exact fun _ _ => ⟨(listChunk_onto (d + 1)).2.2.2, fun _ =>
        Presentable.ite ⟨exists_fair_field, fun _ => induction _ _⟩ trivial⟩

end Instances

end NightstreamFPrime.Spec.Compression.Akita.Extract
