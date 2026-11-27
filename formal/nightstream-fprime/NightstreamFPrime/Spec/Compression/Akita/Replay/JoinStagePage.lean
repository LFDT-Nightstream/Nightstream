import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBound
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageDrawsGuard
import NightstreamFPrime.Spec.Compression.Akita.Replay.Resolve

/-! Theorem (R), `Agrees20` from the readings of each stage, and the bound of the claims for
`key20.aks`, with the wire messages `stagePage20` (Akita revision `d50f700`; ePrint 2026/1983,
revision of 2026-11-23, Theorem 10.17, Lemma 10.47 and Section 10.3;
`docs/akita-work/w21/design.md`, sections 4 and 5; `Replay/JoinGiven.lean`,
`Replay/JoinBound.lean`, `Replay/StageDrawsAgrees.lean`, `Replay/Resolve.lean`).

A page `page20 … k` has the last blocks of the terminal for each `k`, so it is not a function of
the events before the stage `k`. The wire messages here are `stagePage20 … k`.

- `Tier.Agrees.congr`, `agreesAt20_congr`: `Tier.Agrees` and `AgreesAt20` read a transcript only
  through its answers and through the claim of each level in the wire message of its fold stage.
- `presentsAt20_of_cells`: theorem (R) for wire messages with the readings `Reads20` in their
  data, a nonce below `responseTries` and the cells `cells20`.
- `presents20_of_agrees_staged`: theorem (R) for the wire messages `stagePage20`.
  `exists_agrees20_staged`: an accepted proof has a transcript with these wire messages and
  `Agrees20`.
- `Readings20.mapWire`, `agrees20_of_reads_staged`: `Agrees20` from `Readings20` for the wire
  messages `stagePage20`, with no hypothesis `Bits20` (`bits20_of_scripted`).
  `exists_readings20_staged`: an accepted proof with an ideal sponge has a transcript with these
  wire messages and `Readings20`.
- `verifyProof_bound20_of_staged`: the bound of `verifyProof_bound20` with the hypothesis
  `reads`: the wire messages of the transcript of the prover are the pages `stagePage20` of the
  bytes, the sponge of the table is ideal, and the answers of the table have the readings of the
  code on the answers of the sponge (`Readings20`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open scoped BigOperators
open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix Packs Samples
  countDraws Address)
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

open Replayer

section Congr

variable {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)]
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [Fintype Coord] [DecidableEq Coord]
  [Fintype Slot] {total : ℕ}

/-- Two transcripts with the same answers have the same readings of their answers. -/
theorem Prefix.views_of_replies {Reply β : Type} {point point' : Prefix Wire Reply total}
    (replies : ∀ k, point'.reply k = point.reply k) (h : Reply → β) :
    (point'.rest.map fun move => h move.1) = point.rest.map fun move => h move.1 := by
  refine List.ext_getElem? fun k => ?_
  have same : (point'.rest[k]?).map Prod.fst = (point.rest[k]?).map Prod.fst := replies k
  have step := congrArg (Option.map h) same
  rw [Option.map_map, Option.map_map] at step
  rw [List.getElem?_map, List.getElem?_map]
  exact step

omit [∀ b, Zero (Ring b)] in
/-- `Tier.Agrees` holds on a second transcript with the same answers and the same checks of the
level at its fold stage. -/
theorem Tier.Agrees.congr {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source}
    {c : (b : Block) → Ring b} {path : List Move} {source : Source}
    {point point' : Prefix Wire (Option Coord → Slot) total}
    (replies : ∀ k, point'.reply k = point.reply k)
    (check : T.check (point'.wire T.start) = T.check (point.wire T.start))
    (agrees : Tier.Agrees T c path source point) : Tier.Agrees T c path source point' := by
  obtain ⟨⟨reply, answered, decoded⟩, samples⟩ := agrees
  refine ⟨⟨reply, (replies _).trans answered, decoded⟩, ?_⟩
  rw [check, List.map_drop, List.map_take, Prefix.views_of_replies replies view,
    ← List.map_take, ← List.map_drop]
  exact samples

end Congr

section Pages

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)

theorem cells20_stage {k : ℕ} {ℓ : Fin 6} {j : ℕ} (at_ : stage20 k = some (ℓ, j)) :
    cells20 E pub proof messages k = Extract.cellsOf (path20 pub proof messages E ℓ) j := by
  unfold cells20
  rw [at_]

theorem stagePage20_cells (k : ℕ) :
    (stagePage20 (E := E) pub proof messages tLast eLast k).cells =
      cellsAt20 E pub proof messages k := rfl

theorem stagePage20_nonce (k : ℕ) :
    (stagePage20 (E := E) pub proof messages tLast eLast k).nonce = 0 := rfl

end Pages

section Readings

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {pub : Public} {proof : ByteArray} {messages : Array Message}
  {answer : List Duplex.Event → ℕ → ByteArray}
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

omit [DecidableEq E] in
/-- `Readings20` reads a transcript only through its answers: it holds after a change of the
wire messages. -/
theorem Readings20.mapWire (f : ℕ → Key20.Wire E → Key20.Wire E)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (reads : Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
      point) :
    Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
      (Prefix.mapWire f point) :=
  ⟨fun k below reply at_ =>
      reads.work k below reply ((Prefix.mapWire_reply f point k).symm.trans at_),
    fun reply at_ => reads.weights reply ((Prefix.mapWire_reply f point 0).symm.trans at_),
    fun ℓ j d drawn reply at_ =>
      reads.field ℓ j d drawn reply ((Prefix.mapWire_reply f point _).symm.trans at_),
    fun reply at_ => reads.root reply ((Prefix.mapWire_reply f point 1).symm.trans at_),
    fun reply at_ => reads.level₁ reply ((Prefix.mapWire_reply f point 51).symm.trans at_),
    fun reply at_ => reads.level₂ reply ((Prefix.mapWire_reply f point 120).symm.trans at_),
    fun reply at_ => reads.level₃ reply ((Prefix.mapWire_reply f point 203).symm.trans at_),
    fun reply at_ => reads.level₄ reply ((Prefix.mapWire_reply f point 284).symm.trans at_),
    fun reply at_ => reads.level₅ reply ((Prefix.mapWire_reply f point 378).symm.trans at_),
    fun reply at_ => reads.terminal reply ((Prefix.mapWire_reply f point 472).symm.trans at_)⟩

end Readings

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

/-- Theorem (R) for the tower data `o` of the replay and wire messages that are given by their
readings: a transcript with the 473 stages, whose wire messages `k ≤ 473` have the readings
`Reads20` in their data, a nonce below `responseTries` and the cells `cells20 … k`, and whose
answers have the readings of the code (`AgreesAt20`), presents the bytes of an accepted proof of
`layout20`. -/
theorem presentsAt20_of_cells {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (long : 473 ≤ point.rest.length)
    (reads : Reads20 pub proof messages tLast eLast fun k => (point.wire k).data)
    (nonces : ∀ k ≤ 473, ((point.wire k).nonce : ℕ) < Extract.responseTries)
    (cells : ∀ k ≤ 473, (point.wire k).cells = cells20 E pub proof messages k)
    (agrees : AgreesAt20 o pub proof messages point) :
    PresentsAt20 o pub proof messages tLast eLast point := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have fits := fun ℓ : Fin 6 => fits_path20 (E := E) layout accepted terminalSetup decoded ℓ
  have read : o.read = Extract.Page.read := by subst tower; rfl
  have nonce : ∀ j < 7, ((point.wire (starts20.getD j 0)).nonce : ℕ) < Extract.responseTries ↔
      StepOk pub proof messages (nonceIndices20.getD j 0) := fun j below =>
    iff_of_true (nonces _ ((by decide : ∀ j < 7, starts20.getD j 0 ≤ 473) j below)) (ok _)
  have level₀ : Tier.Shows (zeroth) (challenge20₀ pub proof messages)
      (path20 pub proof messages E 0) 0 point :=
    Tier.Agrees.shows read (zeroth_numbers o).1 (zeroth_numbers o).2.2.1 (fits 0)
      (fun j le => (cells _ (le.trans (by decide))).trans
        (cells20_stage pub proof messages (stage20_zeroth j le))) agrees.level₀
  have level₁ : Tier.Shows (first) (challenge20₁ pub proof messages)
      (path20 pub proof messages E 1) 0 point :=
    Tier.Agrees.shows read (first20_numbers o).1 (first20_numbers o).2.2.1 (fits 1)
      (fun j le => (cells _ (le.trans (by decide))).trans
        (cells20_stage pub proof messages (stage20_first j le))) agrees.level₁
  have level₂ : Tier.Shows (second) (challengeTrace20 pub proof messages 2 Key20.level₂)
      (path20 pub proof messages E 2) 0 point :=
    Tier.Agrees.shows read (second20_numbers o).1 (second20_numbers o).2.2.1 (fits 2)
      (fun j le => (cells _ (le.trans (by decide))).trans
        (cells20_stage pub proof messages (stage20_second j le))) agrees.level₂
  have level₃ : Tier.Shows (third) (challengeTrace20 pub proof messages 3 Key20.level₃)
      (path20 pub proof messages E 3) 0 point :=
    Tier.Agrees.shows read (third20_numbers o).1 (third20_numbers o).2.2.1 (fits 3)
      (fun j le => (cells _ (le.trans (by decide))).trans
        (cells20_stage pub proof messages (stage20_third j le))) agrees.level₃
  have level₄ : Tier.Shows (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄)
      (path20 pub proof messages E 4) 0 point :=
    Tier.Agrees.shows read (fourth20_numbers o).1 (fourth20_numbers o).2.2.1 (fits 4)
      (fun j le => (cells _ (le.trans (by decide))).trans
        (cells20_stage pub proof messages (stage20_fourth j le))) agrees.level₄
  have level₅ : Tier.Shows (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅)
      (path20 pub proof messages E 5) 0 point :=
    Tier.Agrees.shows read (fifth20_numbers o).1 (fifth20_numbers o).2.2.1 (fits 5)
      (fun j le => (cells _ (le.trans (by decide))).trans
        (cells20_stage pub proof messages (stage20_fifth j le))) agrees.level₅
  exact ⟨long, reads, nonce,
    fun k below reply answered => iff_of_true (agrees.work k below reply answered) (ok _),
    agrees.weights, level₀, level₁, level₂, level₃, level₄, level₅, agrees.terminal⟩

/-- `AgreesAt20` reads a transcript only through its answers and through the claim of each level
in the wire message of its fold stage: it holds on a second transcript with the same answers,
when the data of the wire messages of the two transcripts have the readings `Reads20`. -/
theorem agreesAt20_congr
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point point' : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (replies : ∀ k, point'.reply k = point.reply k)
    (old : Reads20 pub proof messages tLast eLast fun k => (point.wire k).data)
    (new : Reads20 pub proof messages tLast eLast fun k => (point'.wire k).data)
    (agrees : AgreesAt20 o pub proof messages point) : AgreesAt20 o pub proof messages point' := by
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
      (path20 pub proof messages E 0) 0 point' :=
    Tier.Agrees.congr replies
      (zeroth_check o (claim₀ _ _ (by
        rw [started₀]
        exact new.claim.trans old.claim.symm))) agrees.level₀
  have level₁ : Tier.Agrees (first) (challenge20₁ pub proof messages)
      (path20 pub proof messages E 1) 0 point' :=
    Tier.Agrees.congr replies
      (first_check o (claim₁ _ _ (by
        rw [started₁]
        exact new.claim₁.trans old.claim₁.symm))) agrees.level₁
  have level₂ : Tier.Agrees (second) (challengeTrace20 pub proof messages 2 Key20.level₂)
      (path20 pub proof messages E 2) 0 point' :=
    Tier.Agrees.congr replies
      (second_check o (claim₂ _ _ (by
        rw [started₂]
        exact new.claim₂.trans old.claim₂.symm))) agrees.level₂
  have level₃ : Tier.Agrees (third) (challengeTrace20 pub proof messages 3 Key20.level₃)
      (path20 pub proof messages E 3) 0 point' :=
    Tier.Agrees.congr replies
      (third_check o (claim₃ _ _ (by
        rw [started₃]
        exact new.claim₃.trans old.claim₃.symm))) agrees.level₃
  have level₄ : Tier.Agrees (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄)
      (path20 pub proof messages E 4) 0 point' :=
    Tier.Agrees.congr replies
      (fourth_check o (claim₄ _ _ (by
        rw [started₄]
        exact new.claim₄.trans old.claim₄.symm))) agrees.level₄
  have level₅ : Tier.Agrees (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅)
      (path20 pub proof messages E 5) 0 point' :=
    Tier.Agrees.congr replies
      (fifth_check o (claim₅ _ _ (by
        rw [started₅]
        exact new.claim₅.trans old.claim₅.symm))) agrees.level₅
  refine ⟨fun k below reply answered =>
      agrees.work k below reply ((replies k).symm.trans answered), ?_, level₀, level₁, level₂,
    level₃, level₄, level₅, ?_⟩
  · refine Eq.trans ?_ agrees.weights
    unfold Extract.Root.weights
    rw [replies]
  · obtain ⟨reply, answered, decoded⟩ := agrees.terminal
    exact ⟨reply, (replies _).trans answered, decoded⟩

end At

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem (R) of the join for `key20.aks` with the wire messages of the model prover: for an
accepted proof of `layout20` and each last blocks `tLast`, `eLast` of the terminal, each
transcript of `tower20` with the 473 stages, whose wire message `k ≤ 473` is the page
`stagePage20 … k` of the proof and whose answers have the readings of the code (`Agrees20`),
presents the bytes (`Presents20`). -/
theorem presents20_of_agrees_staged {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) {total : ℕ}
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (long : 473 ≤ point.rest.length)
    (wires : ∀ k ≤ 473, point.wire k = stagePage20 pub proof messages tLast eLast k)
    (agrees : Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL point) :
    Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅ blockL
      point :=
  presentsAt20_of_cells _ pub proof messages tLast eLast layout accepted terminalSetup decoded rfl
    long
    (Reads20.congr pub proof messages tLast eLast
      (fun k le => congrArg Extract.Page.data (wires k le))
      (stagePage20_reads (E := E) layout accepted tLast eLast))
    (fun k le => by
      rw [wires k le, stagePage20_nonce]
      exact (by decide : ((0 : Fin (2 ^ 32)) : ℕ) < Extract.responseTries))
    (fun k le => by
      rw [wires k le, stagePage20_cells]
      exact cellsAt20_eq E layout accepted terminalSetup decoded k)
    agrees

/-- `Agrees20` from the readings of each stage with the wire messages of the model prover: for
an accepted proof of `layout20` with the ideal sponge `answer`, a transcript of `tower20` with
the 473 stages, whose wire message `k ≤ 473` is the page `stagePage20 … k` of the proof and whose
answers have the readings of the code on the answers of the sponge (`Readings20`), has
`Agrees20`. The work bits of the model are those of the script (`bits20_of_scripted`). -/
theorem agrees20_of_reads_staged {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    {total : ℕ} {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (long : 473 ≤ point.rest.length)
    (wires : ∀ k ≤ 473, point.wire k = stagePage20 pub proof messages tLast eLast k)
    (reads : Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
      point) :
    Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL point := by
  have staged := noteAt20_reads (E := E) pub proof messages tLast eLast layout accepted
  have paged : Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL
      (Prefix.mapWire (fun k _ => page20 pub proof messages tLast eLast k) point) :=
    agrees20_of_reads pub proof messages tLast eLast layout accepted terminalSetup expanded decoded
      points values (bits20_of_scripted layout (verifyProof_scripted accepted)) ideal
      (by rw [Prefix.mapWire_length]; exact long)
      (fun k le => Prefix.mapWire_wire _ _ (le.trans long)) (Readings20.mapWire _ reads)
  exact agreesAt20_congr _ pub proof messages tLast eLast rfl
    (fun k => (Prefix.mapWire_reply _ point k).symm)
    (Reads20.congr pub proof messages tLast eLast
      (fun k le => congrArg Extract.Page.data
        (Prefix.mapWire_wire (fun k _ => page20 pub proof messages tLast eLast k) point
          (le.trans long))) staged)
    (Reads20.congr pub proof messages tLast eLast
      (fun k le => congrArg Extract.Page.data (wires k le))
      (stagePage20_reads (E := E) layout accepted tLast eLast)) paged

/-- The hypotheses of `presents20_of_agrees_staged` hold for a transcript: for an accepted proof
of `layout20` and each last blocks `tLast`, `eLast` of the terminal, a transcript of `tower20`
with `total ≥ 473` stages has the 473 stages, the wire messages `stagePage20` and `Agrees20`. -/
theorem exists_agrees20_staged {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
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
        (∀ k ≤ 473, point.wire k = stagePage20 pub proof messages tLast eLast k) ∧
        Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL point := by
  obtain ⟨point, long, wires, agrees⟩ := exists_agrees20 (E := E) pub proof messages tLast eLast
    layout accepted terminalSetup expanded decoded points values injective injective₁ injective₂
    injective₃ injective₄ injective₅ injectiveL slots inside
  have moved : ∀ k ≤ 473,
      (Prefix.mapWire (fun k _ => stagePage20 pub proof messages tLast eLast k) point).wire k =
        stagePage20 pub proof messages tLast eLast k :=
    fun k le => Prefix.mapWire_wire _ _ (le.trans long)
  exact ⟨Prefix.mapWire (fun k _ => stagePage20 pub proof messages tLast eLast k) point,
    by rw [Prefix.mapWire_length]; exact long, moved,
    agreesAt20_congr _ pub proof messages tLast eLast rfl
      (fun k => Prefix.mapWire_reply _ point k)
      (Reads20.congr pub proof messages tLast eLast
        (fun k le => congrArg Extract.Page.data (wires k le))
        (noteAt20_reads (E := E) pub proof messages tLast eLast layout accepted))
      (Reads20.congr pub proof messages tLast eLast
        (fun k le => congrArg Extract.Page.data (moved k le))
        (stagePage20_reads (E := E) layout accepted tLast eLast)) agrees⟩

/-- The hypotheses of `agrees20_of_reads_staged` hold for a transcript: for an accepted proof of
`layout20` with the ideal sponge `answer`, and each last blocks `tLast`, `eLast` of the terminal,
a transcript of `tower20` with `total ≥ 473` stages has the 473 stages, the wire messages
`stagePage20` and `Readings20`. -/
theorem exists_readings20_staged {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord) {total : ℕ}
    (inside : 473 ≤ total) :
    ∃ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total,
      473 ≤ point.rest.length ∧
        (∀ k ≤ 473, point.wire k = stagePage20 pub proof messages tLast eLast k) ∧
        Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
          point := by
  obtain ⟨point, long, -, reads⟩ := exists_readings20 (E := E) pub proof messages tLast eLast
    layout accepted terminalSetup expanded decoded points values
    (bits20_of_scripted layout (verifyProof_scripted accepted)) ideal injective injective₁
    injective₂ injective₃ injective₄ injective₅ injectiveL slots inside
  exact ⟨Prefix.mapWire (fun k _ => stagePage20 pub proof messages tLast eLast k) point,
    by rw [Prefix.mapWire_length]; exact long,
    fun k le => Prefix.mapWire_wire _ _ (le.trans long), Readings20.mapWire _ reads⟩

section Bound

attribute [local instance low] Classical.propDecidable

variable [Nonempty Coord] {total : ℕ}

/-- The bound of `verifyProof_bound20` with the wire messages of the model prover and the
readings of each stage. `reads`: when `verifyProof` with the sponge of a table accepts the bytes
of the table, the sponge of the table is the ideal sponge of `answer`, the transcript of `prover`
with the answers of the table has the 473 stages, its wire message `k ≤ 473` is the page
`stagePage20 … k` of the bytes, and its answers have the readings of the code on the answers of
the sponge (`Readings20`), for last blocks of the code. -/
theorem verifyProof_bound20_of_staged (pub : Public) {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice) {full : Array Fq}
    (terminalSetup : pub.sealed.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.sealed.setupFor (fold20 ℓ).setupCount).isSome)
    (root : pub.root = root20)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord)
    (inside : 472 < total) (kept : Key20.Kept)
    (sponge : (Address (Key20.Wire E) total → Option Coord → CodeSlot E) → Duplex)
    (bytes : (Address (Key20.Wire E) total → Option Coord → CodeSlot E) → ByteArray)
    (prover : OracleComp (Address (Key20.Wire E) total) (Option Coord → CodeSlot E)
      (ℕ → Key20.Wire E)) {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ}
    (positive : 0 < budget)
    (reads : ∀ table, verifyProof (pub.withSponge (sponge table)) (bytes table) =
        .verdict .accept →
      ∃ (messages : Array Message) (rest : ℕ) (tLast : ℕ → Rq Key20.terminal.innerDegree)
        (eLast : Rq Key20.terminal.innerDegree) (answer : List Duplex.Event → ℕ → ByteArray),
        decode pub.ops (bytes table) = .ok (messages, rest) ∧
        (sponge table).ideal = some answer ∧
        Last20 (E := E) (pub.withSponge (sponge table)).sealed (bytes table) messages tLast
          eLast ∧
        473 ≤ ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
          472).body.protocol.final prover table).rest.length ∧
        (∀ k ≤ 473, ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL
            total 472).body.protocol.final prover table).wire k =
          stagePage20 (pub.withSponge (sponge table)).sealed (bytes table) messages tLast
            eLast k) ∧
        Readings20 (pub.withSponge (sponge table)).sealed (bytes table) messages answer
          block block₁ block₂ block₃ block₄ block₅ blockL
          ((tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.final prover table)) :
    𝔼 table, 𝔼 tape : Tape (Address (Key20.Wire E) total) (Option Coord) (CodeSlot E) budget
          (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total
            472).body.protocol.stages,
        (if verifyProof (pub.withSponge (sponge table)) (bytes table) = .verdict .accept ∧
          Checkpoint.extract (tower20 (E := E) pub.sealed block block₁ block₂ block₃ block₄ block₅
            blockL total 472).body prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (317479 / (1024 * Fintype.card E) + 135 * (1 / (fieldModulus : ℝ) ^ 2)) +
        (queries + 1) * (142 : ℕ) / budget :=
  verifyProof_bound20 pub layout terminalSetup expanded root injective injective₁ injective₂
    injective₃ injective₄ injective₅ injectiveL slots inside kept sponge bytes prover bounded
    positive fun table accepted => by
      obtain ⟨messages, rest, tLast, eLast, answer, decoded, ideal, last, long, wires, readings⟩ :=
        reads table accepted
      obtain ⟨points, values, -⟩ := shaped20 (pub := pub.withSponge (sponge table)) root
        ((verifyProof_accept_iff _ _).mp accepted).1
      have ran := ((verifyProof_accept_iff _ _).mp accepted).2
      exact ⟨messages, rest, tLast, eLast, decoded, last,
        presents20_of_agrees_staged (pub.withSponge (sponge table)).sealed (bytes table) messages
          tLast eLast layout ran terminalSetup decoded long wires
          (agrees20_of_reads_staged (pub.withSponge (sponge table)).sealed (bytes table) messages
            tLast eLast layout ran terminalSetup expanded decoded points values ideal long wires
            readings)⟩

end Bound

end NightstreamFPrime.Spec.Compression.Akita.Replay
