import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStage
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentGiven

/-! Theorem (R) of the join for `key20.aks` (Akita revision `d50f700`; ePrint 2026/1983, revision
of 2026-11-23, Section 10.3): each transcript of `tower20` whose wire messages are the pages of
the proof and whose answers have the readings of the code presents the bytes.

- `Tier.Agrees T c path source point`: the answer of the fold stage of the level decodes to `c`,
  and the answers of its later stages have `Extract.Samples` for its checks on `path`.
  `Tier.Agrees.shows` gives `Tier.Shows` on pages with the cells `Extract.cellsOf`
  (`Extract.Tier.presents_of_samples`). `Tier.Shows.agrees` is the converse for a run, and
  `Tier.Agrees.mapWire` keeps `Tier.Agrees` after a change of the wire messages that keeps the
  checks of the level (`Tier.Shows.agrees_mapWire` has the two steps).
- `stage20 k` is the level `ℓ` and the draw `j` of the wire message `k` of the model: the wire
  message `start_ℓ + 1 + j` up to the fold stage of the next level. `page20 … k` is the page `k`
  of a proof: the data `noteAt20 … k`, the cells `Extract.cellsOf (path20 … ℓ) j`, the nonce 0.
- `AgreesAt20 o …` and `Agrees20` (for `tower20`): the readings of the model on the answers of a
  transcript are the readings of the code: each work slot has the work bits of its stage, the
  weights of the batching root are `row_coefficients`, and each level and the terminal have the
  fold challenges and the draws of the code.
- `presents20_of_agrees`: theorem (R). `exists_agrees20`: an accepted proof has a transcript with
  the pages `page20` and `Agrees20`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix Packs Samples
  countDraws)

open Replayer

section Agrees

variable {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)]
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [Fintype Coord] [DecidableEq Coord]
  [Fintype Slot] {total : ℕ}

/-- The answers of a transcript have the readings `c` and `path` of a level: the answer of its
fold stage decodes at each fold coordinate to `c`, and the answers of its later stages have
`Extract.Samples` for its checks for the source `source` on `path`. -/
def Tier.Agrees (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source)
    (c : (b : Block) → Ring b) (path : List Move) (source : Source)
    (point : Prefix Wire (Option Coord → Slot) total) : Prop :=
  (∃ reply, point.reply T.start = some reply ∧ ∀ b, T.decode b (reply (T.block b)) = some (c b)) ∧
    Samples (Coord → Slot) (T.check (point.wire T.start) c source) path
      (((point.rest.take T.finish).drop (T.start + 1)).map fun move => view move.1)

/-- A level that shows the challenges `c` and a path on which its checks run has `Tier.Agrees`. -/
theorem Tier.Shows.agrees {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source}
    {c : (b : Block) → Ring b} {path : List Move} {source : Source}
    {point : Prefix Wire (Option Coord → Slot) total} {result : Result}
    (runs : (T.check (point.wire T.start) c source).Runs path result)
    (shows : Tier.Shows T c path source point) : Tier.Agrees T c path source point := by
  obtain ⟨⟨reply, answered, decoded⟩, presented⟩ := shows
  have challenged : T.challenge point = c :=
    Extract.challengeAt_of_reply T.decode T.block T.start answered decoded
  rw [challenged] at presented
  have samples := Extract.Presents.samples presented runs
  rw [List.map_map] at samples
  exact ⟨⟨reply, answered, decoded⟩, samples⟩

/-- The answers of a transcript after a change of its wire messages. -/
theorem Prefix.mapWire_views {Reply β : Type} (f : ℕ → Wire → Wire)
    (point : Prefix Wire Reply total) (h : Reply → β) :
    (Prefix.mapWire f point).rest.map (fun move => h move.1) =
      point.rest.map fun move => h move.1 := by
  refine List.ext_getElem? fun k => ?_
  show ((point.rest.zipIdx.map fun x => (x.1.1, f (x.2 + 1) x.1.2)).map fun move => h move.1)[k]? =
    _
  rw [List.getElem?_map, List.getElem?_map, List.getElem?_zipIdx, List.getElem?_map]
  cases point.rest[k]? <;> rfl

omit [∀ b, Zero (Ring b)] in
/-- `Tier.Agrees` holds on the transcript with the wire messages `f k`, when `f` keeps the checks
of the level at its fold stage. -/
theorem Tier.Agrees.mapWire {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source}
    {c : (b : Block) → Ring b} {path : List Move} {source : Source}
    {point : Prefix Wire (Option Coord → Slot) total} (f : ℕ → Wire → Wire)
    (long : T.start ≤ point.rest.length)
    (check : T.check (f T.start (point.wire T.start)) = T.check (point.wire T.start))
    (agrees : Tier.Agrees T c path source point) :
    Tier.Agrees T c path source (Prefix.mapWire f point) := by
  obtain ⟨⟨reply, answered, decoded⟩, samples⟩ := agrees
  refine ⟨⟨reply, (Prefix.mapWire_reply f point _).trans answered, decoded⟩, ?_⟩
  rw [Prefix.mapWire_wire f point long, check, List.map_drop, List.map_take,
    Prefix.mapWire_views f point view, ← List.map_take, ← List.map_drop]
  exact samples

/-- A level on `Extract.Page.read` with `Tier.Agrees` shows the challenges and the path, when
the level has the stages from `start` to `finish`, the path fits the cells, and the wire message
`start + 1 + j` up to `finish` has the cells `Extract.cellsOf path j`. -/
theorem Tier.Agrees.shows {Data F : Type} [Zero F] {slots width cap : ℕ}
    {T : Tier Block Ring Stmt Brk Wit Weak (Extract.Page Data F slots width cap) Coord Slot Result
      Source}
    (reads : T.read = Extract.Page.read) {c : (b : Block) → Ring b} {path : List Move}
    {source : Source}
    {point : Prefix (Extract.Page Data F slots width cap) (Option Coord → Slot) total}
    {start finish next : ℕ} (started : T.start = start) (finished : T.finish = finish)
    (fits : Fits F width cap slots path 0 next)
    (cells : ∀ j, start + 1 + j ≤ finish →
      (point.wire (start + 1 + j)).cells = Extract.cellsOf path j)
    (agrees : Tier.Agrees T c path source point) : Tier.Shows T c path source point := by
  subst started finished
  obtain ⟨⟨reply, answered, decoded⟩, samples⟩ := agrees
  exact ⟨⟨reply, answered, decoded⟩,
    (Extract.Tier.presents_of_samples T reads answered decoded source fits.1 samples
      fun j _ le => cells j le).2⟩

/-- `Tier.Shows.agrees` and `Tier.Agrees.mapWire` for a level whose checks run on the path from
a wire message `w` with the checks of the wire message of the fold stage. -/
theorem Tier.Shows.agrees_mapWire
    {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source}
    {c : (b : Block) → Ring b} {path : List Move} {source : Source}
    {point : Prefix Wire (Option Coord → Slot) total} (f : ℕ → Wire → Wire)
    (long : T.start ≤ point.rest.length)
    (check : T.check (f T.start (point.wire T.start)) = T.check (point.wire T.start)) {w : Wire}
    (same : T.check w = T.check (point.wire T.start))
    (runs : ∃ result, (T.check w c source).Runs path result)
    (shows : Tier.Shows T c path source point) :
    Tier.Agrees T c path source (Prefix.mapWire f point) := by
  obtain ⟨result, runs⟩ := runs
  rw [same] at runs
  exact Tier.Agrees.mapWire f long check (Tier.Shows.agrees runs shows)

end Agrees

section Pages

/-- The level `ℓ` and the draw `j` of the wire message `k` of the model: the wire messages 2 to
51 are after the draws 0 to 49 of the level 0, and so on to the wire message 472 of the level 5.
The wire messages 0, 1 and 473 have no message of the checks. -/
def stage20 (k : ℕ) : Option (Fin 6 × ℕ) :=
  if k < 2 then none else if k ≤ 51 then some (0, k - 2) else if k ≤ 120 then some (1, k - 52)
  else if k ≤ 203 then some (2, k - 121) else if k ≤ 284 then some (3, k - 204)
  else if k ≤ 378 then some (4, k - 285) else if k ≤ 472 then some (5, k - 379) else none

theorem stage20_zeroth (j : ℕ) (le : 1 + 1 + j ≤ 51) : stage20 (1 + 1 + j) = some (0, j) := by
  unfold stage20
  rw [ite_eq_right (by omega : ¬1 + 1 + j < 2), ite_eq_left le, show 1 + 1 + j - 2 = j by omega]

theorem stage20_first (j : ℕ) (le : 51 + 1 + j ≤ 120) : stage20 (51 + 1 + j) = some (1, j) := by
  unfold stage20
  rw [ite_eq_right (by omega : ¬51 + 1 + j < 2), ite_eq_right (by omega : ¬51 + 1 + j ≤ 51),
    ite_eq_left le, show 51 + 1 + j - 52 = j by omega]

theorem stage20_second (j : ℕ) (le : 120 + 1 + j ≤ 203) : stage20 (120 + 1 + j) = some (2, j) := by
  unfold stage20
  rw [ite_eq_right (by omega : ¬120 + 1 + j < 2), ite_eq_right (by omega : ¬120 + 1 + j ≤ 51),
    ite_eq_right (by omega : ¬120 + 1 + j ≤ 120), ite_eq_left le,
    show 120 + 1 + j - 121 = j by omega]

theorem stage20_third (j : ℕ) (le : 203 + 1 + j ≤ 284) : stage20 (203 + 1 + j) = some (3, j) := by
  unfold stage20
  rw [ite_eq_right (by omega : ¬203 + 1 + j < 2), ite_eq_right (by omega : ¬203 + 1 + j ≤ 51),
    ite_eq_right (by omega : ¬203 + 1 + j ≤ 120), ite_eq_right (by omega : ¬203 + 1 + j ≤ 203),
    ite_eq_left le, show 203 + 1 + j - 204 = j by omega]

theorem stage20_fourth (j : ℕ) (le : 284 + 1 + j ≤ 378) : stage20 (284 + 1 + j) = some (4, j) := by
  unfold stage20
  rw [ite_eq_right (by omega : ¬284 + 1 + j < 2), ite_eq_right (by omega : ¬284 + 1 + j ≤ 51),
    ite_eq_right (by omega : ¬284 + 1 + j ≤ 120), ite_eq_right (by omega : ¬284 + 1 + j ≤ 203),
    ite_eq_right (by omega : ¬284 + 1 + j ≤ 284), ite_eq_left le,
    show 284 + 1 + j - 285 = j by omega]

theorem stage20_fifth (j : ℕ) (le : 378 + 1 + j ≤ 472) : stage20 (378 + 1 + j) = some (5, j) := by
  unfold stage20
  rw [ite_eq_right (by omega : ¬378 + 1 + j < 2), ite_eq_right (by omega : ¬378 + 1 + j ≤ 51),
    ite_eq_right (by omega : ¬378 + 1 + j ≤ 120), ite_eq_right (by omega : ¬378 + 1 + j ≤ 203),
    ite_eq_right (by omega : ¬378 + 1 + j ≤ 284), ite_eq_right (by omega : ¬378 + 1 + j ≤ 378),
    ite_eq_left le, show 378 + 1 + j - 379 = j by omega]

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)

variable (E) in
/-- The cells of the page `k` of a proof: `Extract.cellsOf` of the path of the code of the level
of `stage20 k` after its draw, and cells of zeros for a wire message without a level. -/
noncomputable def cells20 (k : ℕ) : Fin 4 → Extract.Cell E 8 3029804899 :=
  match stage20 k with
  | some (ℓ, j) => Extract.cellsOf (path20 pub proof messages E ℓ) j
  | none => fun _ => (fun _ => 0, 0)

/-- The page `k` of a proof, the wire message `k` of the model: the data `noteAt20 … k`, the
cells `cells20 … k` and the nonce 0. -/
noncomputable def page20 (k : ℕ) : Key20.Wire E :=
  ⟨noteAt20 pub proof messages tLast eLast k, cells20 E pub proof messages k, 0⟩

omit [Fintype E] [DecidableEq E] in
theorem page20_cells {k : ℕ} {ℓ : Fin 6} {j : ℕ} (at_ : stage20 k = some (ℓ, j)) :
    (page20 (E := E) pub proof messages tLast eLast k).cells =
      Extract.cellsOf (path20 pub proof messages E ℓ) j := by
  show cells20 E pub proof messages k = _
  unfold cells20
  rw [at_]

end Pages

section At

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "code₂" =>
  (CodeSlot.open Key20.open128 Key20.open128_fits :
    CodeSlot E → Option (Rq Key20.level₂.shape.innerDegree))

local notation "code₃" =>
  (CodeSlot.kept128 : CodeSlot E → Option (Rq Key20.level₃.shape.innerDegree))

local notation "code₄" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₄.shape.innerDegree))

local notation "code₅" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₅.shape.innerDegree))

local notation "zeroth" =>
  Extract.liftBatch o.β o.key o.openKey o.claim Key20.rootVars (And.left Key20.root_sized) 8
    Key20.quotientDigits Key20.root_room Key20.rootBatch (And.left (And.right Key20.root_sized))
    o.block (Key20.rootDecode E) o.read Key20.rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (Extract.liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ Key20.level₁.vars
    (And.left Key20.level₁_sized) Key20.quotientDigits (And.left (And.right Key20.level₁_sized))
    Key20.level₁.batch (And.left (And.right (And.right Key20.level₁_sized))) o.block₁ code₁ o.read
    51 (Tier.finish zeroth)) Key20.level₁.tail

local notation "second" =>
  Tier.tensored (Extract.committedTrace (fold := Key20.level₂.fold) o.key₂ o.openKey₂ o.claim₂
    Key20.level₂.vars (And.left Key20.sized₂) Key20.level₂.batch (And.left (And.right Key20.sized₂))
    o.block₂ code₂ o.read 31 (Tier.finish first)) Key20.level₂.tail

local notation "third" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₃.fold) o.key₃ o.openKey₃ o.claim₃
    Key20.level₃.vars (And.left Key20.sized₃) Key20.level₃.batch
    (And.left (And.right Key20.sized₃)) o.block₃ code₃ o.read 31 Key20.norm₃.operator
    Key20.norm₃.limit (Tier.finish second)) Key20.level₃.tail

local notation "fourth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₄.fold) o.key₄ o.openKey₄ o.claim₄
    Key20.level₄.vars (And.left Key20.sized₄) Key20.level₄.batch
    (And.left (And.right Key20.sized₄)) o.block₄ code₄ o.read 53 Key20.norm₄.operator
    Key20.norm₄.limit (Tier.finish third)) Key20.level₄.tail

local notation "fifth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₅.fold) o.key₅ o.openKey₅ o.claim₅
    Key20.level₅.vars (And.left Key20.sized₅) Key20.level₅.batch
    (And.left (And.right Key20.sized₅)) o.block₅ code₅ o.read 53 Key20.norm₅.operator
    Key20.norm₅.limit (Tier.finish fourth)) Key20.level₅.tail

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- The readings of the model on the answers of the transcript `point` of the tower data `o` are
the readings of the code on a proof of `key20.aks`. `work`: the work slot of the stage `k` has
the work bits `bits20` of the model. `weights`: the weights of the batching root are
`row_coefficients`. `level₀` to `level₅`: the fold answer of each level decodes to the fold
challenges of the code, and the answers of its later stages give the draws of the path of the
code (`Tier.Agrees`), for the source 0. `terminal`: the fold answer of the terminal decodes to
the fold challenges of the code. A wire message occurs only as the argument of the checks of a
level at its fold stage. -/
structure AgreesAt20 (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total) :
    Prop where
  work : ∀ k < 473, ∀ reply, point.reply k = some reply →
    reply none ∈ Extract.workZeros (CodeSlot E) (bits20.flatten.getD k 0)
  weights : (Extract.claimsRoot (Coord := Coord) (Slot := CodeSlot E) (total := total) 0 o.read
    o.value o.place).weights point = weights20 pub proof messages E
  level₀ : Tier.Agrees (zeroth) (challenge20₀ pub proof messages) (path20 pub proof messages E 0) 0
    point
  level₁ : Tier.Agrees (first) (challenge20₁ pub proof messages) (path20 pub proof messages E 1) 0
    point
  level₂ : Tier.Agrees (second) (challengeTrace20 pub proof messages 2 Key20.level₂)
    (path20 pub proof messages E 2) 0 point
  level₃ : Tier.Agrees (third) (challengeTrace20 pub proof messages 3 Key20.level₃)
    (path20 pub proof messages E 3) 0 point
  level₄ : Tier.Agrees (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄)
    (path20 pub proof messages E 4) 0 point
  level₅ : Tier.Agrees (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅)
    (path20 pub proof messages E 5) 0 point
  terminal : ∃ reply, point.reply o.terminal.start = some reply ∧
    ∀ i, o.terminal.decode (reply (some (o.terminal.block i))) =
      some (challenge20L pub proof messages i)

/-- Theorem (R) for the tower data `o` of the replay: a transcript with the 473 stages, the wire
messages `page20` and the readings of the code on its answers (`AgreesAt20`) presents the bytes
of an accepted proof of `layout20`. -/
theorem presentsAt20_of_agrees {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (long : 473 ≤ point.rest.length)
    (wires : ∀ k ≤ 473, point.wire k = page20 pub proof messages tLast eLast k)
    (agrees : AgreesAt20 o pub proof messages point) :
    PresentsAt20 o pub proof messages tLast eLast point := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have staged := noteAt20_reads (E := E) pub proof messages tLast eLast layout accepted
  have fits := fun ℓ : Fin 6 => fits_path20 (E := E) layout accepted terminalSetup decoded ℓ
  have read : o.read = Extract.Page.read := by subst tower; rfl
  have reads : Reads20 pub proof messages tLast eLast fun k => (point.wire k).data :=
    Reads20.congr pub proof messages tLast eLast
      (fun k le => congrArg Extract.Page.data (wires k le)) staged
  have nonce : ∀ j < 7, ((point.wire (starts20.getD j 0)).nonce : ℕ) < Extract.responseTries ↔
      StepOk pub proof messages (nonceIndices20.getD j 0) := fun j below => by
    refine iff_of_true ?_ (ok _)
    rw [wires _ ((by decide : ∀ j < 7, starts20.getD j 0 ≤ 473) j below)]
    exact (by decide : ((0 : Fin (2 ^ 32)) : ℕ) < Extract.responseTries)
  have level₀ : Tier.Shows (zeroth) (challenge20₀ pub proof messages)
      (path20 pub proof messages E 0) 0 point :=
    Tier.Agrees.shows read (zeroth_numbers o).1 (zeroth_numbers o).2.2.1 (fits 0)
      (fun j le => (congrArg Extract.Page.cells (wires _ (le.trans (by decide)))).trans
        (page20_cells pub proof messages tLast eLast (stage20_zeroth j le))) agrees.level₀
  have level₁ : Tier.Shows (first) (challenge20₁ pub proof messages)
      (path20 pub proof messages E 1) 0 point :=
    Tier.Agrees.shows read (first20_numbers o).1 (first20_numbers o).2.2.1 (fits 1)
      (fun j le => (congrArg Extract.Page.cells (wires _ (le.trans (by decide)))).trans
        (page20_cells pub proof messages tLast eLast (stage20_first j le))) agrees.level₁
  have level₂ : Tier.Shows (second) (challengeTrace20 pub proof messages 2 Key20.level₂)
      (path20 pub proof messages E 2) 0 point :=
    Tier.Agrees.shows read (second20_numbers o).1 (second20_numbers o).2.2.1 (fits 2)
      (fun j le => (congrArg Extract.Page.cells (wires _ (le.trans (by decide)))).trans
        (page20_cells pub proof messages tLast eLast (stage20_second j le))) agrees.level₂
  have level₃ : Tier.Shows (third) (challengeTrace20 pub proof messages 3 Key20.level₃)
      (path20 pub proof messages E 3) 0 point :=
    Tier.Agrees.shows read (third20_numbers o).1 (third20_numbers o).2.2.1 (fits 3)
      (fun j le => (congrArg Extract.Page.cells (wires _ (le.trans (by decide)))).trans
        (page20_cells pub proof messages tLast eLast (stage20_third j le))) agrees.level₃
  have level₄ : Tier.Shows (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄)
      (path20 pub proof messages E 4) 0 point :=
    Tier.Agrees.shows read (fourth20_numbers o).1 (fourth20_numbers o).2.2.1 (fits 4)
      (fun j le => (congrArg Extract.Page.cells (wires _ (le.trans (by decide)))).trans
        (page20_cells pub proof messages tLast eLast (stage20_fourth j le))) agrees.level₄
  have level₅ : Tier.Shows (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅)
      (path20 pub proof messages E 5) 0 point :=
    Tier.Agrees.shows read (fifth20_numbers o).1 (fifth20_numbers o).2.2.1 (fits 5)
      (fun j le => (congrArg Extract.Page.cells (wires _ (le.trans (by decide)))).trans
        (page20_cells pub proof messages tLast eLast (stage20_fifth j le))) agrees.level₅
  exact ⟨long, reads, nonce,
    fun k below reply answered => iff_of_true (agrees.work k below reply answered) (ok _),
    agrees.weights, level₀, level₁, level₂, level₃, level₄, level₅, agrees.terminal⟩

/-- A transcript that presents the bytes of an accepted proof of `layout20` has the readings of
the code on its answers also with the pages `page20` in place of its wire messages. -/
theorem agreesAt20_of_presents {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    AgreesAt20 o pub proof messages
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have staged := noteAt20_reads (E := E) pub proof messages tLast eLast layout accepted
  have long := presents.long
  have old := presents.reads
  obtain ⟨started₀, -⟩ := zeroth_numbers o
  obtain ⟨started₁, -⟩ := first20_numbers o
  obtain ⟨started₂, -⟩ := second20_numbers o
  obtain ⟨started₃, -⟩ := third20_numbers o
  obtain ⟨started₄, -⟩ := fourth20_numbers o
  obtain ⟨started₅, -⟩ := fifth20_numbers o
  have claim₀ : ∀ w w' : Key20.Wire E, w.data.claim = w'.data.claim →
      o.claim w = o.claim w' := by
    subst tower
    exact fun _ _ same => same
  have claim₁ : ∀ w w' : Key20.Wire E, w.data.claim₁ = w'.data.claim₁ →
      o.claim₁ w = o.claim₁ w' := by
    subst tower
    exact fun _ _ same => same
  have claim₂ : ∀ w w' : Key20.Wire E, w.data.claim₂ = w'.data.claim₂ →
      o.claim₂ w = o.claim₂ w' := by
    subst tower
    exact fun _ _ same => same
  have claim₃ : ∀ w w' : Key20.Wire E, w.data.claim₃ = w'.data.claim₃ →
      o.claim₃ w = o.claim₃ w' := by
    subst tower
    exact fun _ _ same => same
  have claim₄ : ∀ w w' : Key20.Wire E, w.data.claim₄ = w'.data.claim₄ →
      o.claim₄ w = o.claim₄ w' := by
    subst tower
    exact fun _ _ same => same
  have claim₅ : ∀ w w' : Key20.Wire E, w.data.claim₅ = w'.data.claim₅ →
      o.claim₅ w = o.claim₅ w' := by
    subst tower
    exact fun _ _ same => same
  have level₀ : Tier.Agrees (zeroth) (challenge20₀ pub proof messages)
      (path20 pub proof messages E 0) 0
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) :=
    Tier.Shows.agrees_mapWire _ (by rw [started₀]; omega)
      (zeroth_check o (claim₀ _ _ (by
        rw [started₀]
        exact staged.claim.trans old.claim.symm)))
      (zeroth_check o (claim₀ _ _ (by rw [started₀]; exact old.claim.symm)))
      (runs20_zeroth o pub proof messages tLast eLast layout accepted terminalSetup decoded points
      values (expanded 0) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl) presents.level₀
  have level₁ : Tier.Agrees (first) (challenge20₁ pub proof messages)
      (path20 pub proof messages E 1) 0
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) :=
    Tier.Shows.agrees_mapWire _ (by rw [started₁]; omega)
      (first_check o (claim₁ _ _ (by
        rw [started₁]
        exact staged.claim₁.trans old.claim₁.symm)))
      (first_check o (claim₁ _ _ (by rw [started₁]; exact old.claim₁.symm)))
      (runs20_first o pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 1) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl) presents.level₁
  have level₂ : Tier.Agrees (second) (challengeTrace20 pub proof messages 2 Key20.level₂)
      (path20 pub proof messages E 2) 0
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) :=
    Tier.Shows.agrees_mapWire _ (by rw [started₂]; omega)
      (second_check o (claim₂ _ _ (by
        rw [started₂]
        exact staged.claim₂.trans old.claim₂.symm)))
      (second_check o (claim₂ _ _ (by rw [started₂]; exact old.claim₂.symm)))
      (runs20_second o pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 2) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl) presents.level₂
  have level₃ : Tier.Agrees (third) (challengeTrace20 pub proof messages 3 Key20.level₃)
      (path20 pub proof messages E 3) 0
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) :=
    Tier.Shows.agrees_mapWire _ (by rw [started₃]; omega)
      (third_check o (claim₃ _ _ (by
        rw [started₃]
        exact staged.claim₃.trans old.claim₃.symm)))
      (third_check o (claim₃ _ _ (by rw [started₃]; exact old.claim₃.symm)))
      (runs20_third o pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 3) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl) presents.level₃
  have level₄ : Tier.Agrees (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄)
      (path20 pub proof messages E 4) 0
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) :=
    Tier.Shows.agrees_mapWire _ (by rw [started₄]; omega)
      (fourth_check o (claim₄ _ _ (by
        rw [started₄]
        exact staged.claim₄.trans old.claim₄.symm)))
      (fourth_check o (claim₄ _ _ (by rw [started₄]; exact old.claim₄.symm)))
      (runs20_fourth o pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 4) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl) presents.level₄
  have level₅ : Tier.Agrees (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅)
      (path20 pub proof messages E 5) 0
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) :=
    Tier.Shows.agrees_mapWire _ (by rw [started₅]; omega)
      (fifth_check o (claim₅ _ _ (by
        rw [started₅]
        exact staged.claim₅.trans old.claim₅.symm)))
      (fifth_check o (claim₅ _ _ (by rw [started₅]; exact old.claim₅.symm)))
      (runs20_fifth o pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 5) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl) presents.level₅
  refine ⟨fun k below reply answered => (presents.work k below reply
      ((Prefix.mapWire_reply _ _ k).symm.trans answered)).mpr (ok _), ?_, level₀, level₁, level₂,
    level₃, level₄, level₅, ?_⟩
  · refine Eq.trans ?_ presents.weights
    unfold Extract.Root.weights
    rw [Prefix.mapWire_reply]
  · obtain ⟨reply, answered, decoded⟩ := presents.terminal
    exact ⟨reply, (Prefix.mapWire_reply _ _ _).trans answered, decoded⟩

end At

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- The answers of a transcript of `tower20` have the readings of the code on a proof of
`key20.aks` with the decoded messages `messages`: `AgreesAt20` for the tower data of the replay
with the fold challenge of the terminal at the stage 472. -/
def Agrees20 (pub : Public) (proof : ByteArray) (messages : Array Message)
    (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
    (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
    (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
    (blockL : Fin Key20.terminal.blocks → Coord) {total : ℕ}
    (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total) : Prop :=
  AgreesAt20 (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof
    messages point

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem (R) of the join for `key20.aks`: for an accepted proof of `layout20` and each last
blocks `tLast`, `eLast` of the terminal, each transcript of `tower20` with the 473 stages, whose
wire message `k ≤ 473` is the page `page20 … k` of the proof and whose answers have the readings
of the code (`Agrees20`), presents the bytes (`Presents20`). The answers are given: the theorem
chooses no answer and no wire message. -/
theorem presents20_of_agrees {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) {total : ℕ}
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (long : 473 ≤ point.rest.length)
    (wires : ∀ k ≤ 473, point.wire k = page20 pub proof messages tLast eLast k)
    (agrees : Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL point) :
    Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅ blockL
      point :=
  presentsAt20_of_agrees _ pub proof messages tLast eLast layout accepted terminalSetup decoded rfl
    long wires agrees

/-- The hypotheses of theorem (R) hold for a transcript: for an accepted proof of `layout20` and
each last blocks `tLast`, `eLast` of the terminal, a transcript of `tower20` with `total ≥ 473`
stages has the 473 stages, the wire messages `page20` and the readings of the code on its answers
(`Agrees20`). The hypotheses are those of `exists_presents20`. -/
theorem exists_agrees20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord) {total : ℕ}
    (inside : 473 ≤ total) :
    ∃ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total,
      473 ≤ point.rest.length ∧
        (∀ k ≤ 473, point.wire k = page20 pub proof messages tLast eLast k) ∧
        Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL point := by
  obtain ⟨point, presents⟩ := exists_presents20 (E := E) pub proof messages tLast eLast layout
    accepted terminalSetup expanded decoded points values injective injective₁ injective₂
    injective₃ injective₄ injective₅ injectiveL slots inside
  exact ⟨Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point,
    by rw [Prefix.mapWire_length]; exact presents.long,
    fun k le => Prefix.mapWire_wire _ _ (le.trans presents.long),
    agreesAt20_of_presents _ pub proof messages tLast eLast layout accepted terminalSetup expanded
      decoded points values rfl presents⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
