import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentPage

/-! A transcript on the finite wire `Page` with given answers presents a path of a game (Akita,
ePrint 2026/1983, revision of 2026-11-23, Section 10.3, "Extraction with checkpoints"): the form
of `exists_cells` and `Tier.exists_block` in which the answers are given and the cells are a fixed
function of the messages of the path.

- `Samples Reply game path views`: `path` is a path of the game, and each draw of `path` is the
  challenge that `sample` gives for the next answer of `views`. `exists_samples`: a run of a
  drawable game has such answers, and `Samples.append` adds answers after them.
  `Presents.samples`: the answers of a segment that presents a path of a run have `Samples`.
- `cellFor message` is a fixed cell with the message `message`, when one exists
  (`cellFor_message`). `segments path j` is the list of the messages of `path` after its draw
  `j` and before the next draw (`j = 0`: before the first draw), and `cellsOf path j` has the
  cell `cellFor` of the message `i` of this list at the index `i`.
- `pathOf_of_samples`: a page with the cells `cellsOf path 0` and moves whose answers have
  `Samples` and whose page `j` has the cells `cellsOf path (j + 1)` present the path.
- `Tier.presents_of_samples`: the stages of one level on `Page.read`, for a transcript whose
  answer of the fold stage decodes to `c`, whose later answers have `Samples` for the checks, and
  whose wire messages have the cells `cellsOf`: the level has the fold challenges `c` and
  presents the path. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Game (Move)

section Samples

variable (Reply : Type) [Fintype Reply] {α : Type}

/-- `path` is a path of the game on the answers `views`: each draw of `path` is the challenge
that `sample` gives for the next answer, each receipt takes the next message, and the path ends
with the game. Answers after the last draw are free. -/
def Samples : Game α → List Move → List Reply → Prop
  | .pure _, [], _ => True
  | @Game.draw _ C finite nonempty _ next, Move.draw (C := C') c :: path, reply :: views =>
      ∃ same : C' = C, @sample Reply _ C finite nonempty reply = cast same c ∧
        Samples (next (cast same c)) path views
  | @Game.recv _ M _ next, Move.recv (M := M') m :: path, views =>
      ∃ same : M' = M, Samples (next (cast same m)) path views
  | _, _, _ => False

variable {Reply}

/-- Answers after the answers of the draws of a path keep `Samples`. -/
theorem Samples.append {game : Game α} {path : List Move} {views : List Reply}
    (samples : Samples Reply game path views) (extra : List Reply) :
    Samples Reply game path (views ++ extra) := by
  induction game generalizing path views with
  | pure a =>
      cases path with
      | nil => trivial
      | cons move path => cases move <;> exact samples.elim
  | draw C bad next induction =>
      cases path with
      | nil => exact samples.elim
      | cons move path =>
          cases move with
          | recv m => exact samples.elim
          | draw c =>
              cases views with
              | nil => exact samples.elim
              | cons reply views =>
                  obtain ⟨same, sampled, samples⟩ := samples
                  exact ⟨same, sampled, induction _ samples⟩
  | recv M next induction =>
      cases path with
      | nil => exact samples.elim
      | cons move path =>
          cases move with
          | draw c => exact samples.elim
          | recv m =>
              obtain ⟨same, samples⟩ := samples
              exact ⟨same, induction _ samples⟩

/-- A run of a game whose draws have fair decoders has answers with `Samples`, one for each
draw. -/
theorem exists_samples [Nonempty Reply] (game : Game α) (path : List Move) (result : α)
    (runs : game.Runs path result) (drawable : Drawable Reply game) :
    ∃ views : List Reply, views.length = countDraws path ∧ Samples Reply game path views := by
  induction game generalizing path with
  | pure a =>
      cases path with
      | nil => exact ⟨[], rfl, trivial⟩
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
              obtain ⟨views, counted, samples⟩ := induction c path runs (drawable.2 c)
              exact ⟨reply :: views, congrArg (· + 1) counted, rfl, sampled, samples⟩
  | recv M next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | draw c => exact runs.elim
          | recv m =>
              obtain ⟨same, runs⟩ := runs
              subst same
              obtain ⟨views, counted, samples⟩ := induction m path runs (drawable m)
              exact ⟨views, counted, rfl, samples⟩

variable {Wire : Type}

theorem samples_pathOf (read : Reader Wire) (game : Game α) (wire : Wire) (index : ℕ)
    (moves : List (Reply × Wire)) (result : α)
    (runs : game.Runs (pathOf read game wire index moves) result) :
    Samples Reply game (pathOf read game wire index moves) (moves.map Prod.fst) := by
  induction game generalizing wire index moves with
  | pure a => trivial
  | draw C bad next induction =>
      cases moves with
      | nil => exact runs.elim
      | cons move moves =>
          obtain ⟨same, runs⟩ := runs
          exact ⟨rfl, rfl, induction _ _ _ _ runs⟩
  | recv M next induction =>
      obtain ⟨same, runs⟩ := runs
      exact ⟨rfl, induction _ _ _ _ runs⟩

/-- The answers of a segment that presents a path on which the game runs have `Samples`. -/
theorem Presents.samples {read : Reader Wire} {first : Wire} {moves : List (Reply × Wire)}
    {path : List Move} {game : Game α} {result : α} (presents : Presents read first moves path game)
    (runs : game.Runs path result) : Samples Reply game path (moves.map Prod.fst) := by
  subst presents
  exact samples_pathOf read game first 0 moves result runs

end Samples

section Cells

variable {E : Type} [Zero E] {width cap slots : ℕ}

open Classical in
/-- A fixed cell with the message `message`, when one exists. -/
noncomputable def cellFor {M : Type} (message : M) : Cell E width cap :=
  if sent : ∃ cell : Cell E width cap, @Cell.message E width cap M ⟨message⟩ cell = message
  then sent.choose else (fun _ => 0, 0)

theorem cellFor_message {M : Type} [Nonempty M] {message : M}
    (sendable : Cell.Sendable E width cap M message) :
    (cellFor message : Cell E width cap).message M = message := by
  unfold cellFor
  rw [dite_eq_left (sendable ⟨message⟩)]
  exact (sendable ⟨message⟩).choose_spec

/-- The cells of the messages of a path before its first draw, from the cell `index`. -/
noncomputable def cellsFrom : List Move → ℕ → Fin slots → Cell E width cap
  | Move.recv message :: path, index => fun j =>
      if (j : ℕ) = index then cellFor message else cellsFrom path (index + 1) j
  | _, _ => fun _ => (fun _ => 0, 0)

/-- The path after its draw `j`, and the whole path for `j = 0`. -/
def afterDraws : List Move → ℕ → List Move
  | path, 0 => path
  | [], _ + 1 => []
  | Move.draw _ :: path, j + 1 => afterDraws path j
  | Move.recv _ :: path, j + 1 => afterDraws path (j + 1)

/-- The messages of a path before its first draw. -/
def leading : List Move → List Move
  | Move.recv message :: path => Move.recv message :: leading path
  | _ => []

/-- The messages of a path after its draw `j` and before the next draw; for `j = 0` the messages
before the first draw. -/
def segments (path : List Move) (j : ℕ) : List Move := leading (afterDraws path j)

/-- The cells of the wire message after the draw `j` of a path: the cell `i` is `cellFor` of the
message `i` of `segments path j`. -/
noncomputable def cellsOf (path : List Move) (j : ℕ) : Fin slots → Cell E width cap :=
  cellsFrom (segments path j) 0

/-- `cellsOf path j` is a function of the messages `segments path j`. -/
theorem cellsOf_congr {path other : List Move} {j i : ℕ} (same : segments path j = segments other i) :
    (cellsOf path j : Fin slots → Cell E width cap) = cellsOf other i := by
  unfold cellsOf
  rw [same]

theorem afterDraws_zero (path : List Move) : afterDraws path 0 = path := by
  cases path with
  | nil => rfl
  | cons move path => cases move <;> rfl

theorem cellsFrom_leading (path : List Move) (index : ℕ) :
    (cellsFrom (leading path) index : Fin slots → Cell E width cap) = cellsFrom path index := by
  induction path generalizing index with
  | nil => rfl
  | cons move path induction =>
      cases move with
      | draw c => rfl
      | recv m =>
          funext j
          show (if (j : ℕ) = index then cellFor m else cellsFrom (leading path) (index + 1) j) =
            if (j : ℕ) = index then cellFor m else cellsFrom path (index + 1) j
          rw [induction]

theorem cellsOf_zero (path : List Move) :
    (cellsOf path 0 : Fin slots → Cell E width cap) = cellsFrom path 0 := by
  unfold cellsOf segments
  rw [afterDraws_zero, cellsFrom_leading]

variable {Reply α Data : Type} [Fintype Reply]

theorem pathOf_of_samples_from (game : Game α) (path : List Move) (index : ℕ)
    (page : Page Data E slots width cap) (pages : List (Reply × Page Data E slots width cap))
    (samples : Samples Reply game path (pages.map Prod.fst))
    (packs : Packs E width cap slots path index)
    (first : ∀ j : Fin slots, index ≤ j → page.cells j = cellsFrom path index j)
    (later : ∀ j < countDraws path, ∀ move, pages[j]? = some move →
      move.2.cells = cellsOf path (j + 1)) :
    pathOf Page.read game page index pages = path := by
  induction game generalizing path index page pages with
  | pure a =>
      cases path with
      | nil => rfl
      | cons move path => cases move <;> exact samples.elim
  | draw C bad next induction =>
      cases path with
      | nil => exact samples.elim
      | cons move path =>
          cases move with
          | recv m => exact samples.elim
          | draw c =>
              cases pages with
              | nil => exact samples.elim
              | cons move pages =>
                  obtain ⟨same, sampled, samples⟩ := samples
                  obtain rfl := same.symm
                  have drawn : sample Reply C move.1 = c := sampled
                  have opened : move.2.cells = cellsFrom path 0 :=
                    (later 0 (Nat.succ_pos _) move List.getElem?_cons_zero).trans
                      (cellsOf_zero path)
                  have rest := induction c path 0 move.2 pages samples packs
                    (fun j _ => congrFun opened j)
                    (fun j below other found => later (j + 1) (Nat.succ_lt_succ below) other
                      (List.getElem?_cons_succ.trans found))
                  simp only [pathOf]
                  rw [drawn, rest]
  | recv M next induction =>
      cases path with
      | nil => exact samples.elim
      | cons move path =>
          cases move with
          | draw c => exact samples.elim
          | recv m =>
              obtain ⟨same, samples⟩ := samples
              obtain rfl := same.symm
              obtain ⟨inside, sendable, packs⟩ := packs
              have here : page.cells ⟨index, inside⟩ = cellFor m :=
                (first ⟨index, inside⟩ le_rfl).trans (ite_eq_left rfl)
              have read : Page.read.message M page index = m := by
                show (if inside : index < slots then (page.cells ⟨index, inside⟩).message M
                  else Classical.arbitrary M) = m
                rw [dite_eq_left inside, here]
                exact cellFor_message sendable
              have rest := induction m path (index + 1) page pages samples packs
                (fun j after => (first j (Nat.le_of_succ_le after)).trans
                  (ite_eq_right (by show (j : ℕ) ≠ index; omega)))
                (fun j below other found => later j below other found)
              simp only [pathOf, read]
              rw [rest]

/-- A page with the cells `cellsOf path 0` and moves whose answers have `Samples` for the path
and whose page `j` has the cells `cellsOf path (j + 1)`, for each draw `j` of the path, present
the path, for each data and nonce of the pages. -/
theorem pathOf_of_samples {game : Game α} {path : List Move}
    {page : Page Data E slots width cap} {pages : List (Reply × Page Data E slots width cap)}
    (samples : Samples Reply game path (pages.map Prod.fst))
    (packs : Packs E width cap slots path 0) (first : page.cells = cellsOf path 0)
    (later : ∀ j < countDraws path, ∀ move, pages[j]? = some move →
      move.2.cells = cellsOf path (j + 1)) :
    pathOf Page.read game page 0 pages = path :=
  pathOf_of_samples_from game path 0 page pages samples packs
    (fun j _ => congrFun (first.trans (cellsOf_zero path)) j) later

end Cells

section Level

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Coord Slot Result Source : Type} [∀ b, Zero (Ring b)] [Fintype Slot]
  [Fintype Coord] [DecidableEq Coord] {Data E : Type} [Zero E] {slots width cap : ℕ}
  (T : Tier Block Ring Stmt Brk Wit Weak (Page Data E slots width cap) Coord Slot Result Source)

/-- The stages of one level on `Page.read` for a transcript with given answers: when the answer
of the fold stage decodes to `c`, the answers of the later stages of the level have `Samples` for
the checks on `path`, and the wire message `T.start + 1 + j` of the level has the cells
`cellsOf path j` for each draw `j` of the path, the level has the fold challenges `c` and
presents `path`. Stages after the last draw of the path are free. -/
theorem Tier.presents_of_samples (reads : T.read = Page.read) {total : ℕ}
    {point : Prefix (Page Data E slots width cap) (Option Coord → Slot) total}
    {reply : Option Coord → Slot} (answered : point.reply T.start = some reply)
    {c : (b : Block) → Ring b} (decoded : ∀ b, T.decode b (reply (T.block b)) = some (c b))
    (source : Source) {path : List Move} (packs : Packs E width cap slots path 0)
    (samples : Samples (Coord → Slot) (T.check (point.wire T.start) c source) path
      (((point.rest.take T.finish).drop (T.start + 1)).map fun move => view move.1))
    (cells : ∀ j ≤ countDraws path, T.start + 1 + j ≤ T.finish →
      (point.wire (T.start + 1 + j)).cells = cellsOf path j) :
    T.challenge point = c ∧
      Presents T.read (point.wire (T.start + 1))
        (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2))
        path (T.check (point.wire T.start) (T.challenge point) source) := by
  have challenged : T.challenge point = c :=
    challengeAt_of_reply T.decode T.block T.start answered decoded
  refine ⟨challenged, ?_⟩
  unfold Presents
  rw [challenged, reads]
  refine pathOf_of_samples (by rw [List.map_map]; exact samples) packs
    (cells 0 (Nat.zero_le _) (Nat.le_add_right _ _)) fun j below move found => ?_
  rw [List.getElem?_map] at found
  obtain ⟨x, at_, rfl⟩ := Option.map_eq_some_iff.mp found
  rw [List.getElem?_drop, List.getElem?_take] at at_
  have early : T.start + 1 + j < T.finish := by
    by_contra late
    rw [ite_eq_right late] at at_
    cases at_
  rw [ite_eq_left early] at at_
  have wired : point.wire (T.start + 1 + j + 1) = x.2 := by
    show ((point.rest[T.start + 1 + j]?).map Prod.snd).getD point.first = x.2
    rw [at_]
    rfl
  exact (congrArg Page.cells wired).symm.trans (cells (j + 1) below early)

end Level

end NightstreamFPrime.Spec.Compression.Akita.Extract
