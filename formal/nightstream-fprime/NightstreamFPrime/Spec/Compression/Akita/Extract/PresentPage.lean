import NightstreamFPrime.Spec.Compression.Akita.Extract.Present
import NightstreamFPrime.Spec.Compression.Akita.Extract.Page
import NightstreamFPrime.Spec.Compression.Akita.Extract.Play
import NightstreamFPrime.Spec.Compression.Akita.Extract.Root
import NightstreamFPrime.Spec.Compression.Akita.Extract.TerminalSuffix

/-! Transcripts on the finite wire `Page` that present a path of a game (Akita, ePrint 2026/1983,
revision of 2026-11-23, Section 10.3, "Extraction with checkpoints"): the form of
`exists_presents` for the reader `Page.read` and the stages of one `Tier`.

- `Drawable Reply game`: each draw of the game has a fair decoder from `Reply`
  (`Presentable.drawable`). `Packs E width cap slots path index`: each message of the path is the
  message of a cell (`Cell.Sendable`), and the message `index` after a draw has `index < slots`.
  A number is the message of a cell exactly up to `cap` (`Cell.sendable_nat_iff`). `countDraws`
  is the number of the draws of a path, and `nextIndex` the index of the next message after it.
- `exists_cells`: for a run on such a path, cells for the messages before the first draw and an
  answer and cells for each draw exist, such that each page with these cells and each list of
  moves that starts with these answers and cells present the path, for each data and nonce of the
  pages.
- `withWork work rest` is the answer with the work slot `work` and the view `rest`.
  `exists_mem_workZeros_iff` and `exists_not_mem_workZeros_iff` give the slots with and without
  `g` zero work bits (`workZeros_card`). `foldAnswer` has given slots at the fold coordinates, and
  `foldAnswer_reads` gives its fold challenges and its check `DecodesAt`. `exists_weights` gives
  the weights of a batching root, and `Terminal.fold_of_reply` the fold stage of the terminal.
- `Prefix.split_of_take` and `Prefix.reads_of_split` read the stages of a block of a transcript,
  and `Prefix.ofBlocks` is the transcript of a list of blocks with `Prefix.ofBlocks_block`.
- `Tier.exists_block`: the block of the stages of one level on `Page.read` for a run of its
  checks: each transcript with this block has the stages of the level, the fold challenges of
  the fold answer, the check `Tier.Decodes` exactly for the work slots and the nonce, the
  presented path, and the result of the run. `Tier.game_eq_pure_iff_of_runs` is the converse
  reading on a presented path. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Game (Move)

section Paths

/-- The number of the draws of a path. -/
def countDraws : List Move → ℕ
  | [] => 0
  | Move.draw _ :: path => countDraws path + 1
  | Move.recv _ :: path => countDraws path

theorem countDraws_append (first second : List Move) :
    countDraws (first ++ second) = countDraws first + countDraws second := by
  induction first with
  | nil => exact (Nat.zero_add _).symm
  | cons move first induction =>
      cases move with
      | draw c =>
          show countDraws (first ++ second) + 1 = countDraws first + 1 + countDraws second
          rw [induction]
          omega
      | recv m => exact induction

/-- The index of the next message after a path that starts at the message `index`: a draw sets
it to 0, and a message adds 1. -/
def nextIndex : List Move → ℕ → ℕ
  | [], index => index
  | Move.draw _ :: path, _ => nextIndex path 0
  | Move.recv _ :: path, index => nextIndex path (index + 1)

variable (E : Type) [Zero E] (width cap slots : ℕ) in
/-- The messages of a path fit the cells of the pages: each message is the message of a cell,
and the message `index` after a draw has `index < slots`. -/
def Packs : List Move → ℕ → Prop
  | [], _ => True
  | Move.draw _ :: path, _ => Packs path 0
  | @Move.recv M message :: path, index =>
      index < slots ∧ Cell.Sendable E width cap M message ∧ Packs path (index + 1)

variable {E : Type} [Zero E] {width cap slots : ℕ}

theorem packs_draw {C : Type} (c : C) (path : List Move) (index : ℕ) :
    Packs E width cap slots (Move.draw c :: path) index ↔ Packs E width cap slots path 0 :=
  Iff.rfl

theorem packs_recv {M : Type} (message : M) (path : List Move) (index : ℕ) :
    Packs E width cap slots (Move.recv message :: path) index ↔
      index < slots ∧ Cell.Sendable E width cap M message ∧
        Packs E width cap slots path (index + 1) :=
  Iff.rfl

theorem Packs.append (first second : List Move) (index : ℕ) :
    Packs E width cap slots (first ++ second) index ↔
      Packs E width cap slots first index ∧
        Packs E width cap slots second (nextIndex first index) := by
  induction first generalizing index with
  | nil => exact ⟨fun packs => ⟨trivial, packs⟩, fun packs => packs.2⟩
  | cons move first induction =>
      cases move with
      | draw c => exact induction 0
      | recv m =>
          rw [List.cons_append, packs_recv, packs_recv, induction]
          exact ⟨fun ⟨a, b, c, d⟩ => ⟨⟨a, b, c⟩, d⟩, fun ⟨⟨a, b, c⟩, d⟩ => ⟨a, b, c, d⟩⟩

/-- A number is the message of a cell exactly when it is at most `cap`. -/
theorem Cell.sendable_nat_iff (message : ℕ) :
    Cell.Sendable E width cap ℕ message ↔ message ≤ cap := by
  constructor
  · intro sendable
    obtain ⟨cell, sent⟩ := sendable ⟨0⟩
    have value : Cell.message ℕ cell = (cell.2 : ℕ) := dite_eq_left rfl
    rw [← value.symm.trans sent]
    exact Nat.le_of_lt_succ cell.2.isLt
  · intro small _
    exact ⟨(fun _ => 0, ⟨message, Nat.lt_succ_of_le small⟩), dite_eq_left rfl⟩

variable (Reply : Type) [Fintype Reply] in
/-- Each draw of the game has a fair decoder from the answers `Reply`. -/
def Drawable {α : Type} : Game α → Prop
  | .pure _ => True
  | @Game.draw _ C finite _ _ next =>
      (∃ decode : Reply → C, @Fair Reply _ C finite decode) ∧ ∀ c, Drawable (next c)
  | @Game.recv _ _ _ next => ∀ m, Drawable (next m)

variable {Reply α : Type} [Fintype Reply]

theorem Presentable.drawable {Chunk : Type} {game : Game α}
    (presentable : Presentable Reply Chunk game) : Drawable Reply game := by
  induction game with
  | pure a => trivial
  | draw C bad next induction => exact ⟨presentable.1, fun c => induction c (presentable.2 c)⟩
  | recv M next induction => exact fun m => induction m (presentable.2 m)

theorem Drawable.bind {β : Type} {game : Game α} {after : α → Game β}
    (first : Drawable Reply game) (second : ∀ a, Drawable Reply (after a)) :
    Drawable Reply (game.bind after) := by
  induction game with
  | pure a => exact second a
  | draw C bad next induction => exact ⟨first.1, fun c => induction c (first.2 c)⟩
  | recv M next induction => exact fun m => induction m (first m)

variable [Nonempty Reply] {Data : Type}

/-- A run on a path whose draws have fair decoders and whose messages fit the cells from the
message `index`: cells from `index` and an answer and cells for each draw exist, and each page and
moves with them present the path from `index`, for each data and nonce of the pages. -/
theorem exists_cells_from (game : Game α) (path : List Move) (result : α)
    (runs : game.Runs path result) (drawable : Drawable Reply game) (index : ℕ)
    (packs : Packs E width cap slots path index) :
    ∃ (first : Fin slots → Cell E width cap)
      (moves : List (Reply × (Fin slots → Cell E width cap))),
      moves.length = countDraws path ∧
      ∀ (page : Page Data E slots width cap) (pages : List (Reply × Page Data E slots width cap)),
        (∀ j : Fin slots, index ≤ j → page.cells j = first j) →
        moves <+: pages.map (fun move => (move.1, move.2.cells)) →
        pathOf Page.read game page index pages = path := by
  induction game generalizing path index with
  | pure a =>
      cases path with
      | nil => exact ⟨fun _ => (fun _ => 0, 0), [], rfl, fun _ _ _ _ => rfl⟩
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
              obtain ⟨reply, sampled⟩ := sample_surjective drawable.1 c
              obtain ⟨first, moves, counted, presents⟩ :=
                induction c path runs (drawable.2 c) 0 packs
              refine ⟨fun _ => (fun _ => 0, 0), (reply, first) :: moves,
                congrArg (· + 1) counted, fun page pages _ leading => ?_⟩
              rcases pages with _ | ⟨⟨answer, later⟩, pages⟩
              · exact absurd (List.prefix_nil.mp leading) (List.cons_ne_nil _ _)
              · simp only [List.map_cons, List.cons_prefix_cons, Prod.mk.injEq] at leading
                obtain ⟨⟨rfl, cells⟩, leading⟩ := leading
                simp only [pathOf]
                rw [sampled, presents later pages (fun j _ => (congrFun cells j).symm) leading]
  | recv M next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | draw c => exact runs.elim
          | recv m =>
              obtain ⟨same, runs⟩ := runs
              obtain rfl := same.symm
              obtain ⟨inside, sendable, packs⟩ := packs
              obtain ⟨cell, sent⟩ := sendable inferInstance
              obtain ⟨first, moves, counted, presents⟩ :=
                induction m path runs (drawable m) (index + 1) packs
              refine ⟨Function.update first ⟨index, inside⟩ cell, moves, counted,
                fun page pages agree leading => ?_⟩
              have here : page.cells ⟨index, inside⟩ = cell :=
                (agree _ le_rfl).trans (Function.update_self ..)
              have read : Page.read.message M page index = m := by
                show (if inside : index < slots then (page.cells ⟨index, inside⟩).message M
                  else Classical.arbitrary M) = m
                rw [dite_eq_left inside, here]
                exact sent
              simp only [pathOf, read]
              rw [presents page pages (fun j later => (agree j (Nat.le_of_succ_le later)).trans
                (Function.update_of_ne (Fin.ne_of_val_ne (by show (j : ℕ) ≠ index; omega)) ..))
                leading]

/-- `exists_cells_from` from the message 0: the first page has the cells `first`. -/
theorem exists_cells {game : Game α} {path : List Move} {result : α}
    (runs : game.Runs path result) (drawable : Drawable Reply game)
    (packs : Packs E width cap slots path 0) :
    ∃ (first : Fin slots → Cell E width cap)
      (moves : List (Reply × (Fin slots → Cell E width cap))),
      moves.length = countDraws path ∧
      ∀ (page : Page Data E slots width cap) (pages : List (Reply × Page Data E slots width cap)),
        page.cells = first → moves <+: pages.map (fun move => (move.1, move.2.cells)) →
        pathOf Page.read game page 0 pages = path := by
  obtain ⟨first, moves, counted, presents⟩ :=
    exists_cells_from (Data := Data) game path result runs drawable 0 packs
  exact ⟨first, moves, counted, fun page pages cells leading =>
    presents page pages (fun j _ => congrFun cells j) leading⟩

/-- A list is a prefix of the list of its first `n` entries for `n` at least its length. -/
theorem prefix_ofFn_getD {β : Type} {list : List β} {n : ℕ} (short : list.length ≤ n)
    (fallback : β) : list <+: List.ofFn fun i : Fin n => list.getD i fallback := by
  rw [List.prefix_iff_eq_take]
  refine List.ext_getElem (by simp [Nat.min_eq_left short]) fun i inside _ => ?_
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inside]

end Paths

section Answers

variable {Coord Slot : Type}

/-- The answer with the work slot `work` and the slot `rest c` at each coordinate `c`. -/
def withWork (work : Slot) (rest : Coord → Slot) : Option Coord → Slot
  | none => work
  | some coordinate => rest coordinate

theorem view_withWork (work : Slot) (rest : Coord → Slot) : view (withWork work rest) = rest :=
  rfl

theorem withWork_none (work : Slot) (rest : Coord → Slot) : withWork work rest none = work :=
  rfl

variable (Slot) [Fintype Slot] in
/-- A slot with `g` zero work bits exists exactly when `2^g ≤ |Slot|`. -/
theorem exists_mem_workZeros_iff (g : ℕ) :
    (∃ slot, slot ∈ workZeros Slot g) ↔ 2 ^ g ≤ Fintype.card Slot := by
  constructor
  · rintro ⟨slot, member⟩
    have positive : 0 < (workZeros Slot g).card := Finset.card_pos.mpr ⟨slot, member⟩
    rw [workZeros_card] at positive
    by_contra small
    rw [Nat.div_eq_of_lt (Nat.lt_of_not_le small)] at positive
    exact Nat.lt_irrefl 0 positive
  · intro large
    have positive : 0 < (workZeros Slot g).card := by
      rw [workZeros_card]
      exact Nat.div_pos large (Nat.two_pow_pos g)
    exact Finset.card_pos.mp positive

variable (Slot) [Fintype Slot] in
/-- A slot without `g` zero work bits exists exactly when `⌊|Slot|/2^g⌋ < |Slot|`. -/
theorem exists_not_mem_workZeros_iff (g : ℕ) :
    (∃ slot, slot ∉ workZeros Slot g) ↔ Fintype.card Slot / 2 ^ g < Fintype.card Slot := by
  rw [← workZeros_card, Finset.card_lt_iff_ne_univ, Ne, Finset.eq_univ_iff_forall, not_forall]

variable (Slot) [Fintype Slot] [Nonempty Slot] in
/-- A slot without `g` zero work bits exists for `g > 0`. -/
theorem exists_not_mem_workZeros {g : ℕ} (positive : 0 < g) :
    ∃ slot, slot ∉ workZeros Slot g :=
  (exists_not_mem_workZeros_iff Slot g).mpr
    (Nat.div_lt_self Fintype.card_pos (Nat.one_lt_two_pow positive.ne'))

variable {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)] {Wire : Type} {total : ℕ}

/-- The fold answer with the slot `s b` at the coordinate `coordinate b`, the work slot `work`,
and the slot `rest c` at each other coordinate `c`. -/
noncomputable def foldAnswer (coordinate : Block → Coord) (s : Block → Slot) (work : Slot)
    (rest : Coord → Slot) : Option Coord → Slot :=
  withWork work (Function.extend coordinate s rest)

theorem foldAnswer_some {coordinate : Block → Coord} (injective : Function.Injective coordinate)
    (s : Block → Slot) (work : Slot) (rest : Coord → Slot) (b : Block) :
    foldAnswer coordinate s work rest (some (coordinate b)) = s b :=
  injective.extend_apply s rest b

theorem foldAnswer_none (coordinate : Block → Coord) (s : Block → Slot) (work : Slot)
    (rest : Coord → Slot) : foldAnswer coordinate s work rest none = work :=
  rfl

omit [∀ b, Zero (Ring b)] in
/-- The fold answer decodes to the challenges `c` at the fold coordinates. -/
theorem foldAnswer_decode (decode : (b : Block) → Slot → Option (Ring b))
    {block : Block → Option Coord} {coordinate : Block → Coord}
    (placed : ∀ b, block b = some (coordinate b)) (injective : Function.Injective coordinate)
    {s : Block → Slot} {c : (b : Block) → Ring b} (decoded : ∀ b, decode b (s b) = some (c b))
    (work : Slot) (rest : Coord → Slot) (b : Block) :
    decode b (foldAnswer coordinate s work rest (block b)) = some (c b) := by
  rw [placed, foldAnswer_some injective, decoded]

/-- A stage input whose answer of the stage `start` decodes to `c` at the fold coordinates has
the fold challenges `c`. -/
theorem challengeAt_of_reply (decode : (b : Block) → Slot → Option (Ring b))
    (block : Block → Option Coord) (start : ℕ) {point : Prefix Wire (Option Coord → Slot) total}
    {reply : Option Coord → Slot} (answered : point.reply start = some reply)
    {c : (b : Block) → Ring b} (decoded : ∀ b, decode b (reply (block b)) = some (c b)) :
    challengeAt decode block start point = c := by
  funext b
  simp [challengeAt, answered, decoded]

omit [∀ b, Zero (Ring b)] in
/-- When the answer of the stage `start` decodes at each fold coordinate, the verifier accepts
the fold stage exactly when its work slot has `bits` zero work bits. -/
theorem decodesAt_iff_of_reply [Fintype Slot] (decode : (b : Block) → Slot → Option (Ring b))
    (block : Block → Option Coord) (start bits : ℕ)
    {point : Prefix Wire (Option Coord → Slot) total} {reply : Option Coord → Slot}
    (answered : point.reply start = some reply)
    (decoded : ∀ b, (decode b (reply (block b))).isSome) :
    DecodesAt decode block start bits point ↔ reply none ∈ workZeros Slot bits := by
  constructor
  · rintro ⟨other, same, worked, _⟩
    obtain rfl := Option.some.inj (same.symm.trans answered)
    exact worked
  · exact fun worked => ⟨reply, answered, worked, decoded⟩

/-- A stage input whose answer of the stage `start` is a fold answer for the slots `s` with
`decode b (s b) = some (c b)` has the fold challenges `c`, and the verifier accepts the fold
stage exactly when the work slot has `bits` zero work bits. -/
theorem foldAnswer_reads [Fintype Slot] (decode : (b : Block) → Slot → Option (Ring b))
    {block : Block → Option Coord} {coordinate : Block → Coord}
    (placed : ∀ b, block b = some (coordinate b)) (injective : Function.Injective coordinate)
    {s : Block → Slot} {c : (b : Block) → Ring b} (decoded : ∀ b, decode b (s b) = some (c b))
    (work : Slot) (rest : Coord → Slot) (start bits : ℕ)
    {point : Prefix Wire (Option Coord → Slot) total}
    (answered : point.reply start = some (foldAnswer coordinate s work rest)) :
    challengeAt decode block start point = c ∧
      (DecodesAt decode block start bits point ↔ work ∈ workZeros Slot bits) :=
  ⟨challengeAt_of_reply decode block start answered
      (foldAnswer_decode decode placed injective decoded work rest),
    decodesAt_iff_of_reply decode block start bits answered fun b => by
      simp [foldAnswer_decode decode placed injective decoded work rest]⟩

/-- For each vector `ϑ` of weights, an answer with the work slot `work` gives the weights `ϑ`
at the stage of each batching root. -/
theorem exists_weights {E : Type} [Field E] [Fintype E] [DecidableEq E] [Fintype Coord]
    [DecidableEq Coord] [Fintype Slot] [Nonempty Slot] {count : ℕ}
    (fair : ∃ decode : (Coord → Slot) → Fin count → E, Fair decode) (ϑ : Fin count → E)
    (work : Slot) :
    ∃ rest : Coord → Slot, ∀ {Wire Cand : Type} {total : ℕ}
      (R : Root E Wire Coord Slot Cand count total)
      (point : Prefix Wire (Option Coord → Slot) total),
      point.reply R.start = some (withWork work rest) → R.weights point = ϑ := by
  obtain ⟨rest, sampled⟩ := sample_surjective fair ϑ
  exact ⟨rest, fun R point answered => by simp [Root.weights, answered, view_withWork, sampled]⟩

/-- The fold stage of the terminal: an answer of the stage `start` that decodes to `c` at the
slots of the blocks gives the fold challenges `c`, and the check `DecodesAt` of
`Terminal.Passed` holds exactly when its work slot has `foldBits` zero work bits. -/
theorem Terminal.fold_of_reply {S : Shape} {E : Type} [CommRing E] [Module Zq E] {route : Route}
    [Fintype Slot] (L : Terminal S E Wire Coord Slot total route)
    {point : Prefix Wire (Option Coord → Slot) total} {reply : Option Coord → Slot}
    (answered : point.reply L.start = some reply) {c : Fin S.blocks → Rq S.innerDegree}
    (decoded : ∀ i, L.decode (reply (some (L.block i))) = some (c i)) :
    L.challenge point = c ∧
      (DecodesAt (fun _ => L.decode) L.slot L.start L.foldBits point ↔
        reply none ∈ workZeros Slot L.foldBits) :=
  ⟨challengeAt_of_reply (fun _ => L.decode) L.slot L.start answered decoded,
    decodesAt_iff_of_reply (fun _ => L.decode) L.slot L.start L.foldBits answered fun i => by
      show (L.decode (reply (some (L.block i)))).isSome
      simp [decoded]⟩

end Answers

section Blocks

variable {Wire Reply : Type} {total : ℕ}

/-- A transcript whose stages from `start` begin with a nonempty block `block` is the stages
before `start`, then `block`, then the later stages. -/
theorem Prefix.split_of_take {point : Prefix Wire Reply total} {start : ℕ}
    {block : List (Reply × Wire)} (taken : (point.rest.drop start).take block.length = block)
    (nonempty : block ≠ []) :
    ∃ before after, before.length = start ∧ point.rest = before ++ (block ++ after) := by
  have long := congrArg List.length taken
  rw [List.length_take, List.length_drop] at long
  have positive : 0 < block.length := by
    cases block with
    | nil => exact absurd rfl nonempty
    | cons _ _ => exact Nat.succ_pos _
  have whole := List.take_append_drop start point.rest
  have tail := List.take_append_drop block.length (point.rest.drop start)
  rw [taken] at tail
  rw [← tail] at whole
  exact ⟨_, _, by rw [List.length_take]; omega, whole.symm⟩

/-- The stages of a block of a transcript after the stages `before`: the wire message at its
start is the last wire message of `before`, and stage `before.length + i` has the answer and the
next wire message of entry `i` of the block. -/
theorem Prefix.reads_of_split {point : Prefix Wire Reply total}
    {before block after : List (Reply × Wire)} (shape : point.rest = before ++ (block ++ after)) :
    point.wire before.length = lastWire point.first before ∧
      ∀ i (inside : i < block.length), point.reply (before.length + i) = some block[i].1 ∧
        point.wire (before.length + i + 1) = block[i].2 := by
  refine ⟨Prefix.wire_length (after := block ++ after) shape, fun i inside => ?_⟩
  have found : point.rest[before.length + i]? = some block[i] := by
    rw [shape, List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left,
      List.getElem?_append_left inside, List.getElem?_eq_getElem inside]
  exact ⟨by simp [Prefix.reply, found], by simp [Prefix.wire, found]⟩

/-- `Prefix.reads_of_split` for the stages from `start` that begin with a nonempty block. -/
theorem Prefix.reads_of_take {point : Prefix Wire Reply total} {start : ℕ}
    {block : List (Reply × Wire)} (taken : (point.rest.drop start).take block.length = block)
    (nonempty : block ≠ []) :
    start + block.length ≤ point.rest.length ∧
      ∀ i (inside : i < block.length), point.reply (start + i) = some block[i].1 ∧
        point.wire (start + i + 1) = block[i].2 := by
  obtain ⟨before, after, length, shape⟩ := Prefix.split_of_take taken nonempty
  obtain ⟨-, each⟩ := Prefix.reads_of_split shape
  rw [length] at each
  refine ⟨?_, each⟩
  rw [shape, List.length_append, List.length_append, length]
  omega

/-- The transcript with the wire message `first` and the stages of the blocks `blocks` in
order. -/
def Prefix.ofBlocks (first : Wire) (blocks : List (List (Reply × Wire)))
    (short : (blocks.map List.length).sum ≤ total) : Prefix Wire Reply total :=
  ⟨first, blocks.flatten, by rw [List.length_flatten]; exact short⟩

/-- Block `k` of `Prefix.ofBlocks` starts at the stage `Σ_{j<k} |block_j|`: the stages from
there begin with the block, the wire message there is the last wire message of the earlier
blocks, and each stage of the block has its answer and its next wire message. -/
theorem Prefix.ofBlocks_block (first : Wire) (blocks : List (List (Reply × Wire)))
    (short : (blocks.map List.length).sum ≤ total) (k : ℕ) (inside : k < blocks.length) :
    ((Prefix.ofBlocks first blocks short).rest.drop
        ((blocks.take k).map List.length).sum).take blocks[k].length = blocks[k] ∧
      (Prefix.ofBlocks first blocks short).wire ((blocks.take k).map List.length).sum =
        lastWire first (blocks.take k).flatten ∧
      ∀ i (within : i < blocks[k].length),
        (Prefix.ofBlocks first blocks short).reply (((blocks.take k).map List.length).sum + i) =
            some blocks[k][i].1 ∧
          (Prefix.ofBlocks first blocks short).wire
            (((blocks.take k).map List.length).sum + i + 1) = blocks[k][i].2 := by
  have shape : (Prefix.ofBlocks first blocks short).rest =
      (blocks.take k).flatten ++ (blocks[k] ++ (blocks.drop (k + 1)).flatten) := by
    show blocks.flatten = _
    conv_lhs => rw [← List.take_append_drop k blocks, List.drop_eq_getElem_cons inside]
    rw [List.flatten_append, List.flatten_cons]
  have length : ((blocks.take k).flatten).length = ((blocks.take k).map List.length).sum :=
    List.length_flatten
  obtain ⟨wire, each⟩ := Prefix.reads_of_split shape
  rw [length] at wire each
  refine ⟨?_, wire, each⟩
  rw [shape, List.drop_left' length, List.take_left' rfl]

end Blocks

section Level

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [∀ b, Zero (Ring b)] [Fintype Slot]
  [Fintype Coord] [DecidableEq Coord]

/-- On a transcript whose stages of the level present a path on which the checks run to
`result`, the game of the level ends in `other` exactly when `other` is `result`. -/
theorem Tier.game_eq_pure_iff_of_runs
    (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source) (source : Source)
    {total : ℕ} (point : Prefix Wire (Option Coord → Slot) total) {path : List Move}
    {result : Result}
    (presents : Presents T.read (point.wire (T.start + 1))
      (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2)) path
      (T.check (point.wire T.start) (T.challenge point) source))
    (runs : (T.check (point.wire T.start) (T.challenge point) source).Runs path result)
    (other : Result) : T.game source point = .pure other ↔ other = result :=
  (T.game_eq_pure_iff source point presents other).trans
    ⟨fun ran => Game.runs_unique ran runs, fun same => same ▸ runs⟩

variable [Nonempty Slot] {Data E : Type} [Zero E] {slots width cap : ℕ}
  (T : Tier Block Ring Stmt Brk Wit Weak (Page Data E slots width cap) Coord Slot Result Source)

/-- The block of the stages of one level on `Page.read` for a run of its checks for `wire₀`, the
fold challenges `c` of `fold` and `source`. `work i` is the work slot of the stage `i` of the
checks, and `data i` and `nonce i` are the fields of the wire message after the stage `i` of the
block. Each transcript with `wire₀` before the stage `T.start` and the block from there has the
stages of the level, the challenges `c`, `Tier.Decodes` exactly for the work slots and the nonce
of `wire₀`, the path and the result of the run. Stages after the end of the run are padding. -/
theorem Tier.exists_block (reads : T.read = Page.read) (wire₀ : Page Data E slots width cap)
    (fold : Option Coord → Slot) {c : (b : Block) → Ring b}
    (decoded : ∀ b, T.decode b (fold (T.block b)) = some (c b)) (source : Source)
    {path : List Move} {result : Result} (runs : (T.check wire₀ c source).Runs path result)
    (drawable : Drawable (Coord → Slot) (T.check wire₀ c source))
    (packs : Packs E width cap slots path 0) (few : countDraws path ≤ T.costs.length)
    (work : ℕ → Slot) (data : ℕ → Data) (nonce : ℕ → Fin (2 ^ 32)) :
    ∃ block : List ((Option Coord → Slot) × Page Data E slots width cap),
      block.length = T.costs.length + 1 ∧
      (∀ i (inside : i < block.length), block[i].2.data = data i ∧ block[i].2.nonce = nonce i) ∧
      block.head?.map Prod.fst = some fold ∧
      (∀ i (inside : i + 1 < block.length), block[i + 1].1 none = work i) ∧
      ∀ {total : ℕ} (point : Prefix (Page Data E slots width cap) (Option Coord → Slot) total),
        point.wire T.start = wire₀ →
        (point.rest.drop T.start).take (T.costs.length + 1) = block →
        T.finish ≤ point.rest.length ∧ T.challenge point = c ∧
        (T.Decodes point ↔ fold none ∈ workZeros Slot T.foldBits ∧
          (∀ i < T.costs.length, work i ∈ workZeros Slot (T.bits.getD i 0)) ∧
          (wire₀.nonce : ℕ) < responseTries) ∧
        Presents T.read (point.wire (T.start + 1))
          (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2))
          path (T.check (point.wire T.start) (T.challenge point) source) ∧
        T.game source point = .pure result := by
  obtain ⟨first, moves, counted, presents⟩ := exists_cells (Data := Data) runs drawable packs
  have short : moves.length ≤ T.costs.length := by rw [counted]; exact few
  obtain ⟨blank, -⟩ : ∃ blank : (Coord → Slot) × (Fin slots → Cell E width cap),
      blank = (view fold, fun _ => (fun _ => 0, 0)) := ⟨_, rfl⟩
  obtain ⟨later, listed⟩ : ∃ later : List ((Option Coord → Slot) × Page Data E slots width cap),
      later = List.ofFn fun i : Fin T.costs.length =>
        (withWork (work i) (moves.getD i blank).1,
          ⟨data (i + 1), (moves.getD i blank).2, nonce (i + 1)⟩) := ⟨_, rfl⟩
  have sized : later.length = T.costs.length := by rw [listed, List.length_ofFn]
  have entry : ∀ i (inside : i < later.length), later[i] =
      (withWork (work i) (moves.getD i blank).1,
        ⟨data (i + 1), (moves.getD i blank).2, nonce (i + 1)⟩) := by
    intro i inside
    subst listed
    rw [List.getElem_ofFn]
  have mapped : ((later.map fun move => (view move.1, move.2)).map
      fun move => (move.1, move.2.cells)) =
        List.ofFn fun i : Fin T.costs.length => moves.getD i blank := by
    subst listed
    simp only [List.map_ofFn]
    rfl
  have leading : moves <+: ((later.map fun move => (view move.1, move.2)).map
      fun move => (move.1, move.2.cells)) := by
    rw [mapped]
    exact prefix_ofFn_getD short blank
  let page₀ : Page Data E slots width cap := ⟨data 0, first, nonce 0⟩
  have size : ((fold, page₀) :: later).length = T.costs.length + 1 := by
    rw [List.length_cons, sized]
  have worked : ∀ i (inside : i + 1 < ((fold, page₀) :: later).length),
      ((fold, page₀) :: later)[i + 1].1 none = work i := by
    intro i inside
    rw [List.getElem_cons_succ, entry i (Nat.lt_of_succ_lt_succ inside)]
    rfl
  refine ⟨(fold, page₀) :: later, size, fun i inside => ?_, rfl, worked,
    fun {total} point wired taken => ?_⟩
  · cases i with
    | zero => exact ⟨rfl, rfl⟩
    | succ i =>
        rw [List.getElem_cons_succ, entry i (Nat.lt_of_succ_lt_succ inside)]
        exact ⟨rfl, rfl⟩
  rw [← size] at taken
  obtain ⟨before, after, length, shape⟩ := Prefix.split_of_take taken (List.cons_ne_nil _ _)
  obtain ⟨-, each⟩ := Prefix.reads_of_split shape
  rw [length] at each
  have answered : point.reply T.start = some fold := (each 0 (Nat.succ_pos _)).1
  have opened : point.wire (T.start + 1) = page₀ := (each 0 (Nat.succ_pos _)).2
  have challenged : T.challenge point = c :=
    challengeAt_of_reply T.decode T.block T.start answered decoded
  have whole : (before ++ (fold, page₀) :: later).length = T.finish := by
    rw [List.length_append, List.length_cons, length, sized]
    exact T.finish_eq.symm
  have moved : (point.rest.take T.finish).drop (T.start + 1) = later := by
    rw [shape, ← List.append_assoc, List.take_left' whole, List.append_cons]
    exact List.drop_left' (by rw [List.length_append, List.length_singleton, length])
  have presented : Presents T.read (point.wire (T.start + 1))
      (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2))
      path (T.check (point.wire T.start) (T.challenge point) source) := by
    unfold Presents
    rw [moved, opened, challenged, wired, reads]
    exact presents page₀ _ rfl leading
  refine ⟨?_, challenged, ?_, presented,
    (T.game_eq_pure_iff source point presented result).mpr (by rw [wired, challenged]; exact runs)⟩
  · rw [shape, List.length_append, List.length_append, List.length_cons, length, sized]
    unfold Tier.finish
    omega
  · unfold Tier.Decodes
    refine and_congr (decodesAt_iff_of_reply T.decode T.block T.start T.foldBits answered
      fun b => by simp [decoded b]) (and_congr ⟨fun checked i inside => ?_,
        fun good i inside reply found => ?_⟩ (by rw [wired, reads]; rfl))
    · have room : i + 1 < ((fold, page₀) :: later).length := by rw [size]; omega
      have known := (each (i + 1) room).1
      rw [show T.start + (i + 1) = T.start + 1 + i by omega] at known
      have here := checked i inside _ known
      rwa [worked i room] at here
    · have room : i + 1 < ((fold, page₀) :: later).length := by rw [size]; omega
      have known := (each (i + 1) room).1
      rw [show T.start + (i + 1) = T.start + 1 + i by omega] at known
      obtain rfl := Option.some.inj (found.symm.trans known)
      rw [worked i room]
      exact good i inside

end Level

end NightstreamFPrime.Spec.Compression.Akita.Extract
