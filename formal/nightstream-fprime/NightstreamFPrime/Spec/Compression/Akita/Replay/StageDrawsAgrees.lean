import NightstreamFPrime.Spec.Compression.Akita.Replay.StageDraws

/-! `Agrees20` from the readings of each stage (Akita revision `d50f700`; ePrint 2026/1983,
revision of 2026-11-23, Section 8.2 `[KECCAK]` and Section 10.3;
`docs/akita-work/w21/design.md`, track Q; `Replay/JoinGiven.lean`, `Replay/StageCoupling.lean`,
`Replay/StageDraws.lean`).

- `FieldReads20 … k d reply`: the challenge that the model reads from the answer `reply` of the
  stage `k` is `Draw.read` of the values that `decodePool` gives on the answer of the sponge for
  the squeeze of the work of the stage and `eventsAt20` (the form of the conclusion of
  `stage_field`).
- `FoldReads20 … k g decode block reply`: the decoder of the model on the slots `block` of the
  answer `reply` gives the challenges of the sampler of the code on the answer of the sponge for
  the group `g` of the fold draw of the stage `k` (the form of the conclusion of `FoldCoupling`
  and `FoldsCoupling`).
- `Readings20 … answer … point`: for each stage `k < 473` of the transcript `point`, the reading
  of the model on its answer is the reading of the code on the answer of the sponge: the work
  slot, the weights of the batching root, each field draw of each level, and each fold draw.
- `Tier.agrees_of_reads`: `Tier.Agrees` for one level, from its fold reading and `FieldReads20`
  at each draw of its path.
- `agreesAt20_of_reads`, `agrees20_of_reads`: `Agrees20` for an accepted proof of `layout20`
  with an ideal sponge and a transcript with the pages `page20` and `Readings20`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix Packs Samples
  countDraws draws)

open Replayer

/-- The fold draws of the skeleton of `layout20`: the level and the family of each group of each
fold stage, and the draw of the terminal is before the terminal. -/
theorem foldDraws20 :
    foldDraw20 1 0 = some (0, ⟨32, 1024, 128, 31, 0, false⟩) ∧
      foldDraw20 1 1 = some (0, ⟨16, 512, 64, 31, 10, false⟩) ∧
      foldDraw20 1 2 = some (0, ⟨8, 1024, 64, 31, 10, false⟩) ∧
      foldDraw20 51 0 = some (1, ⟨28, 1024, 64, 31, 10, false⟩) ∧
      foldDraw20 120 0 = some (2, ⟨21, 128, 128, 31, 0, false⟩) ∧
      foldDraw20 203 0 = some (3, ⟨11, 128, 128, 31, 0, true⟩) ∧
      foldDraw20 284 0 = some (4, ⟨7, 64, 64, 31, 11, true⟩) ∧
      foldDraw20 378 0 = some (5, ⟨7, 64, 64, 31, 11, true⟩) ∧
      foldDraw20 472 0 = some (6, ⟨5, 64, 64, 31, 11, true⟩) ∧
      workIndices20.getD 472 0 + 1 + 0 + 1 ≤ 1521 := by decide +kernel

theorem first20_le : ∀ ℓ : Fin 6, first20 ℓ ≤ bound20 ℓ := by decide

/-- The weights of a batching root on a transcript with the answer of its stage. -/
theorem Root.weights_of_reply {E : Type} [Field E] [Fintype E] [DecidableEq E]
    {Wire Coord Slot Cand : Type} {count total : ℕ} [Fintype Slot] [Nonempty Slot]
    [Fintype Coord] [DecidableEq Coord] (R : Extract.Root E Wire Coord Slot Cand count total)
    {point : Prefix Wire (Option Coord → Slot) total} {reply : Option Coord → Slot}
    (answered : point.reply R.start = some reply) :
    R.weights point = Extract.sample (Coord → Slot) (Fin count → E) (view reply) := by
  simp [Extract.Root.weights, answered]

section Reads

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (answer : List Duplex.Event → ℕ → ByteArray)

/-- The challenge that the model reads from the answer `reply` of the stage `k` is `Draw.read`
of the values that `decodePool` gives on the answer of the sponge for the squeeze of the work of
the stage `k` and `eventsAt20`. -/
def FieldReads20 (k : ℕ) (d : Draw) (reply : Option Coord → CodeSlot E) : Prop :=
  ∀ values,
    decodePool (answer (.squeeze 8 :: eventsAt20 pub proof messages k) (poolBytes (2 * d.count)))
      d.count = some values →
    ∀ (finite : Fintype (d.Space E)) (nonempty : Nonempty (d.Space E)),
      @Extract.sample (Coord → CodeSlot E) _ (d.Space E) finite nonempty (view reply) =
        d.read E values

/-- The decoder `decode` of the model on the slots `block` of the answer `reply` gives the
challenges of the sampler of the code on the answer of the sponge for the squeezes of the groups
before `g`, the squeeze of the work of the fold stage `k` and `eventsAt20`. -/
def FoldReads20 (k g : ℕ) {d n : ℕ} [NeZero d] (decode : CodeSlot E → Option (Rq d))
    (block : Fin n → Coord) (reply : Option Coord → CodeSlot E) : Prop :=
  ∀ level G cs, foldDraw20 k g = some (level, G) →
    sample G (answer (drawEvents20 k g ++ .squeeze 8 :: eventsAt20 pub proof messages k)
      G.messageBytes) = some cs →
    ∀ i : Fin n, decode (reply (some (block i))) = (cs[(i : ℕ)]?).map (Sparse.ring d)

variable {pub proof messages answer}

omit [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E] [Fintype Coord]
  [DecidableEq Coord] in
/-- A fold reading on the drawn challenges of a state: the drawn challenges are the challenges
`cs` of the sampler moved by `ambient`, one for each slot, and the decoder of the model gives
`cs`. -/
theorem FoldReads20.decoded {k g level : ℕ} {G : FoldGroup} [NeZero G.degree] {d n : ℕ}
    [NeZero d] {decode : CodeSlot E → Option (Rq d)} {block : Fin n → Coord}
    {reply : Option Coord → CodeSlot E}
    (reads : FoldReads20 pub proof messages answer k g decode block reply)
    (draw : foldDraw20 k g = some (level, G)) {s : Replayer}
    (drawn : foldChallenges G (answer (drawEvents20 k g ++
      .squeeze 8 :: eventsAt20 pub proof messages k) G.messageBytes) = some (drawnIn s level g))
    (count : G.coordinates = n) :
    ∃ cs : List Sparse, drawnIn s level g = cs.map (ambient G) ∧ cs.length = n ∧
      ∀ (i : Fin n) (inside : (i : ℕ) < cs.length),
        decode (reply (some (block i))) = some (Sparse.ring d cs[i]) := by
  unfold foldChallenges at drawn
  obtain ⟨cs, sampled, mapped⟩ := Option.map_eq_some_iff.mp drawn
  refine ⟨cs, mapped.symm, (sample_mem sampled).1.trans count, fun i inside => ?_⟩
  rw [reads level G cs draw sampled i, List.getElem?_eq_getElem inside]
  rfl

variable (pub proof messages answer)

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

local notation "codeL" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.terminal.innerDegree))

/-- For each stage `k < 473` of the transcript `point`, the reading of the model on the answer of
the stage is the reading of the code on the answer `answer` of the sponge. `work`: the work slot
has the work bits of the stage when the work of the code passes. `weights`: the stage 0 gives
the three values of `row_coefficients`. `field`: the stage of the draw `j` of the level `ℓ`
gives this draw (`FieldReads20`). `root`, `level₁` to `level₅`, `terminal`: the fold stage of
each level and of the terminal gives the challenges of the sampler (`FoldReads20`). -/
structure Readings20 (block : Key20.root.Block → Coord)
    (block₁ : Fin Key20.level₁.shape.blocks → Coord)
    (block₂ : Fin Key20.level₂.shape.blocks → Coord)
    (block₃ : Fin Key20.level₃.shape.blocks → Coord)
    (block₄ : Fin Key20.level₄.shape.blocks → Coord)
    (block₅ : Fin Key20.level₅.shape.blocks → Coord) (blockL : Fin Key20.terminal.blocks → Coord)
    (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total) : Prop where
  work : ∀ k < 473, ∀ reply, point.reply k = some reply →
    worked (answer (eventsAt20 pub proof messages k) 8) (bits20.flatten.getD k 0) = true →
    reply none ∈ Extract.workZeros (CodeSlot E) (bits20.flatten.getD k 0)
  weights : ∀ reply, point.reply 0 = some reply → ∀ values,
    decodePool (answer (.squeeze 8 :: eventsAt20 pub proof messages 0) (poolBytes (2 * 3))) 3 =
      some values →
    ∀ finite : Fintype (Fin 3 → E),
      @Extract.sample (Coord → CodeSlot E) _ (Fin 3 → E) finite _ (view reply) =
        fun g : Fin 3 => ((values.getD g Ext.zero).into : E)
  field : ∀ (ℓ : Fin 6) (j : ℕ) (d : Draw), (draws20 ℓ)[j]? = some d →
    ∀ reply, point.reply (first20 ℓ + j) = some reply →
      FieldReads20 pub proof messages answer (first20 ℓ + j) d reply
  root : ∀ reply, point.reply 1 = some reply → ∀ g : Fin 3,
    FoldReads20 pub proof messages answer 1 g (Key20.rootDecode E g)
      (fun i : Fin (Key20.root.shape g).blocks => block ⟨g, i⟩) reply
  level₁ : ∀ reply, point.reply 51 = some reply →
    FoldReads20 pub proof messages answer 51 0 code₁ block₁ reply
  level₂ : ∀ reply, point.reply 120 = some reply →
    FoldReads20 pub proof messages answer 120 0 code₂ block₂ reply
  level₃ : ∀ reply, point.reply 203 = some reply →
    FoldReads20 pub proof messages answer 203 0 code₃ block₃ reply
  level₄ : ∀ reply, point.reply 284 = some reply →
    FoldReads20 pub proof messages answer 284 0 code₄ block₄ reply
  level₅ : ∀ reply, point.reply 378 = some reply →
    FoldReads20 pub proof messages answer 378 0 code₅ block₅ reply
  terminal : ∀ reply, point.reply 472 = some reply →
    FoldReads20 pub proof messages answer 472 0 codeL blockL reply

variable {pub proof messages answer}

omit [DecidableEq E] in
/-- `Tier.Agrees` for the level `ℓ` of an accepted proof of `layout20` with the work bits of the
model and an ideal sponge: when the checks of the level run on the path of the code, the answer
of its fold stage decodes to `c`, and the answer of the stage of each draw of the path gives the
draw (`FieldReads20`). -/
theorem Tier.agrees_of_reads {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)]
    {Stmt Brk Wit Weak Result Source : Type}
    {T : Tier Block Ring Stmt Brk Wit Weak (Key20.Wire E) Coord (CodeSlot E) Result Source}
    {c : (b : Block) → Ring b} {source : Source}
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (bits : Bits20 pub)
    (ideal : pub.sponge.ideal = some answer) (ℓ : Fin 6) (started : T.start + 1 = first20 ℓ)
    (finished : T.finish = bound20 ℓ) (long : 473 ≤ point.rest.length) {result : Result}
    (runs : (T.check (point.wire T.start) c source).Runs (path20 pub proof messages E ℓ) result)
    (fold : ∀ reply, point.reply T.start = some reply →
      ∀ b, T.decode b (reply (T.block b)) = some (c b))
    (field : ∀ (j : ℕ) (d : Draw), (draws20 ℓ)[j]? = some d →
      ∀ reply, point.reply (first20 ℓ + j) = some reply →
        FieldReads20 pub proof messages answer (first20 ℓ + j) d reply) :
    Tier.Agrees T c (path20 pub proof messages E ℓ) source point := by
  obtain ⟨erased, ran⟩ := ran_of_accept layout accepted terminalSetup decoded
  have last := (bound20_last ℓ).1
  have le := first20_le ℓ
  have inside : T.start < point.rest.length := by omega
  refine ⟨⟨(point.rest[T.start]).1, ?_, fold _ ?_⟩, ?_⟩
  · show (point.rest[T.start]?).map Prod.fst = _
    rw [List.getElem?_eq_getElem inside]
    rfl
  · show (point.rest[T.start]?).map Prod.fst = _
    rw [List.getElem?_eq_getElem inside]
    rfl
  refine Extract.samples_of_draws _ _ _ runs fun j C c' drawn => ?_
  rw [path20_draws E layout accepted terminalSetup decoded ℓ, List.getElem?_map] at drawn
  obtain ⟨d, at_, moved⟩ := Option.map_eq_some_iff.mp drawn
  obtain ⟨values, pooled, same⟩ := draw20_value E erased bits ℓ at_ (ran _) ideal
  have bounded := (draws20_checked ℓ at_).2.2.2
  have here : first20 ℓ + j < point.rest.length := by omega
  have answered : point.reply (first20 ℓ + j) = some (point.rest[first20 ℓ + j]).1 := by
    show (point.rest[first20 ℓ + j]?).map Prod.fst = _
    rw [List.getElem?_eq_getElem here]
    rfl
  have sampled := field j d at_ _ answered values pooled
  cases moved
  refine ⟨view (point.rest[first20 ℓ + j]).1, ?_, fun finite nonempty =>
    (sampled finite nonempty).trans same.symm⟩
  rw [List.getElem?_map, List.getElem?_drop, List.getElem?_take_of_lt (by omega), started,
    List.getElem?_eq_getElem here]
  rfl

end Reads

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

local notation "codeL" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.terminal.innerDegree))

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

/-- `AgreesAt20` from the readings of each stage, for the tower data `o` of the replay: an
accepted proof of `layout20` with the work bits of the model and the ideal sponge `answer`, and
a transcript with the 473 stages, the wire messages `page20` and `Readings20`. -/
theorem agreesAt20_of_reads {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3) (bits : Bits20 pub)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (long : 473 ≤ point.rest.length)
    (wires : ∀ k ≤ 473, point.wire k = page20 pub proof messages tLast eLast k)
    (reads : Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
      point) :
    AgreesAt20 o pub proof messages point := by
  obtain ⟨erased, ran⟩ := ran_of_accept layout accepted terminalSetup decoded
  have staged := noteAt20_reads (E := E) pub proof messages tLast eLast layout accepted
  have old : Reads20 pub proof messages tLast eLast fun k => (point.wire k).data :=
    Reads20.congr pub proof messages tLast eLast
      (fun k le => congrArg Extract.Page.data (wires k le)) staged
  obtain ⟨started₀, -, finished₀, -⟩ := zeroth_numbers o
  obtain ⟨started₁, -, finished₁, -⟩ := first20_numbers o
  obtain ⟨started₂, -, finished₂, -⟩ := second20_numbers o
  obtain ⟨started₃, -, finished₃, -⟩ := third20_numbers o
  obtain ⟨started₄, -, finished₄, -⟩ := fourth20_numbers o
  obtain ⟨started₅, -, finished₅, -⟩ := fifth20_numbers o
  obtain ⟨d00, d01, d02, d1, d2, d3, d4, d5, dL, beforeL⟩ := foldDraws20
  have answered : ∀ k (below : k < 473),
      point.reply k = some (point.rest[k]'(Nat.lt_of_lt_of_le below long)).1 := by
    intro k below
    show (point.rest[k]?).map Prod.fst = _
    rw [List.getElem?_eq_getElem (Nat.lt_of_lt_of_le below long)]
    rfl
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
  have inner₀ : ∀ reply, point.reply 1 = some reply → ∀ b : Key20.root.Block,
      Key20.rootDecode E b.1 (reply (some (block b))) =
        some (challenge20₀ pub proof messages b) := by
    intro reply at_
    have group₀ : ∀ i : Fin (Key20.root.shape 0).blocks,
        Key20.rootDecode E 0 (reply (some (block ⟨0, i⟩))) =
          some (challenge20₀ pub proof messages ⟨0, i⟩) := by
      obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.root reply at_ 0) d00
        (drawn20_answer layout erased bits 0 d00 (ran _) ideal) rfl
      intro i
      have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
      rw [challenge20₀_eq pub proof messages 0 i mapped (by decide) (by decide) inside]
      exact decodes i inside
    have group₁ : ∀ i : Fin (Key20.root.shape 1).blocks,
        Key20.rootDecode E 1 (reply (some (block ⟨1, i⟩))) =
          some (challenge20₀ pub proof messages ⟨1, i⟩) := by
      obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.root reply at_ 1) d01
        (drawn20_answer layout erased bits 0 d01 (ran _) ideal) rfl
      intro i
      have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
      rw [challenge20₀_eq pub proof messages 1 i mapped (by decide) (by decide) inside]
      exact decodes i inside
    have group₂ : ∀ i : Fin (Key20.root.shape 2).blocks,
        Key20.rootDecode E 2 (reply (some (block ⟨2, i⟩))) =
          some (challenge20₀ pub proof messages ⟨2, i⟩) := by
      obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.root reply at_ 2) d02
        (drawn20_answer layout erased bits 0 d02 (ran _) ideal) rfl
      intro i
      have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
      rw [challenge20₀_eq pub proof messages 2 i mapped (by decide) (by decide) inside]
      exact decodes i inside
    rintro ⟨g, i⟩
    fin_cases g
    exacts [group₀ i, group₁ i, group₂ i]
  have inner₁ : ∀ reply, point.reply 51 = some reply → ∀ i : Fin Key20.level₁.shape.blocks,
      code₁ (reply (some (block₁ i))) = some (challenge20₁ pub proof messages i) := by
    intro reply at_
    obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.level₁ reply at_) d1
      (drawn20_answer layout erased bits 1 d1 (ran _) ideal) rfl
    intro i
    have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
    rw [challenge20₁_eq pub proof messages i mapped (by decide) (by decide) inside]
    exact decodes i inside
  have inner₂ : ∀ reply, point.reply 120 = some reply → ∀ i : Fin Key20.level₂.shape.blocks,
      code₂ (reply (some (block₂ i))) =
        some (challengeTrace20 pub proof messages 2 Key20.level₂ i) := by
    intro reply at_
    obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.level₂ reply at_) d2
      (drawn20_answer layout erased bits 2 d2 (ran _) ideal) rfl
    intro i
    have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 2 Key20.level₂ i mapped rfl (by decide) inside]
    exact decodes i inside
  have inner₃ : ∀ reply, point.reply 203 = some reply → ∀ i : Fin Key20.level₃.shape.blocks,
      code₃ (reply (some (block₃ i))) =
        some (challengeTrace20 pub proof messages 3 Key20.level₃ i) := by
    intro reply at_
    obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.level₃ reply at_) d3
      (drawn20_answer layout erased bits 3 d3 (ran _) ideal) rfl
    intro i
    have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 3 Key20.level₃ i mapped rfl (by decide) inside]
    exact decodes i inside
  have inner₄ : ∀ reply, point.reply 284 = some reply → ∀ i : Fin Key20.level₄.shape.blocks,
      code₄ (reply (some (block₄ i))) =
        some (challengeTrace20 pub proof messages 4 Key20.level₄ i) := by
    intro reply at_
    obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.level₄ reply at_) d4
      (drawn20_answer layout erased bits 4 d4 (ran _) ideal) rfl
    intro i
    have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 4 Key20.level₄ i mapped rfl (by decide) inside]
    exact decodes i inside
  have inner₅ : ∀ reply, point.reply 378 = some reply → ∀ i : Fin Key20.level₅.shape.blocks,
      code₅ (reply (some (block₅ i))) =
        some (challengeTrace20 pub proof messages 5 Key20.level₅ i) := by
    intro reply at_
    obtain ⟨cs, mapped, length, decodes⟩ := FoldReads20.decoded (reads.level₅ reply at_) d5
      (drawn20_answer layout erased bits 5 d5 (ran _) ideal) rfl
    intro i
    have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
    rw [challengeTrace20_eq pub proof messages 5 Key20.level₅ i mapped rfl (by decide) inside]
    exact decodes i inside
  have innerL : ∀ i : Fin Key20.terminal.blocks,
      codeL ((point.rest[472]'(by omega)).1 (some (blockL i))) =
        some (challenge20L pub proof messages i) := by
    have later : stageIndex20 pub 472 + 1 + 0 + 1 ≤ pub.ops.size - 1 := by
      rw [stageIndex20_lt pub (by decide), size_skeleton erased, skeleton20_length]
      exact beforeL
    obtain ⟨cs, mapped, length, decodes⟩ :=
      FoldReads20.decoded (reads.terminal _ (answered 472 (by decide))) dL
        (squeeze20_drawn erased bits (by decide) dL later (ran _) ideal) rfl
    intro i
    have inside : (i : ℕ) < cs.length := by rw [length]; exact i.isLt
    rw [challenge20L_eq pub proof messages i mapped rfl (by decide) inside]
    exact decodes i inside
  have fold₀ : ∀ reply, point.reply (zeroth).start = some reply →
      ∀ b, (zeroth).decode b (reply ((zeroth).block b)) =
        some (challenge20₀ pub proof messages b) := by
    subst tower
    exact fun reply at_ b => inner₀ reply at_ b
  have fold₁ : ∀ reply, point.reply (first).start = some reply →
      ∀ b, (first).decode b (reply ((first).block b)) =
        some (challenge20₁ pub proof messages b) := by
    subst tower
    exact fun reply at_ b => inner₁ reply at_ b
  have fold₂ : ∀ reply, point.reply (second).start = some reply →
      ∀ b, (second).decode b (reply ((second).block b)) =
        some (challengeTrace20 pub proof messages 2 Key20.level₂ b) := by
    subst tower
    exact fun reply at_ b => inner₂ reply at_ b
  have fold₃ : ∀ reply, point.reply (third).start = some reply →
      ∀ b, (third).decode b (reply ((third).block b)) =
        some (challengeTrace20 pub proof messages 3 Key20.level₃ b) := by
    subst tower
    exact fun reply at_ b => inner₃ reply at_ b
  have fold₄ : ∀ reply, point.reply (fourth).start = some reply →
      ∀ b, (fourth).decode b (reply ((fourth).block b)) =
        some (challengeTrace20 pub proof messages 4 Key20.level₄ b) := by
    rw [started₄]
    subst tower
    exact fun reply at_ b => inner₄ reply at_ b
  have fold₅ : ∀ reply, point.reply (fifth).start = some reply →
      ∀ b, (fifth).decode b (reply ((fifth).block b)) =
        some (challengeTrace20 pub proof messages 5 Key20.level₅ b) := by
    rw [started₅]
    subst tower
    exact fun reply at_ b => inner₅ reply at_ b
  have level₀ : Tier.Agrees (zeroth) (challenge20₀ pub proof messages)
      (path20 pub proof messages E 0) 0 point := by
    obtain ⟨result, runs⟩ := runs20_zeroth o pub proof messages tLast eLast layout accepted
      terminalSetup decoded points values (expanded 0) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl
    have same : (zeroth).check
        (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) =
          (zeroth).check (point.wire (zeroth).start) :=
      zeroth_check o (claim₀ _ _ (by rw [started₀]; exact old.claim.symm))
    rw [same] at runs
    exact Tier.agrees_of_reads layout accepted terminalSetup decoded bits ideal 0
      (by rw [started₀]; rfl) (by rw [finished₀]; rfl) long runs fold₀ (reads.field 0)
  have level₁ : Tier.Agrees (first) (challenge20₁ pub proof messages)
      (path20 pub proof messages E 1) 0 point := by
    obtain ⟨result, runs⟩ := runs20_first o pub proof messages tLast eLast layout accepted
      terminalSetup decoded (expanded 1) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl
    have same : (first).check
        (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) =
          (first).check (point.wire (first).start) :=
      first_check o (claim₁ _ _ (by rw [started₁]; exact old.claim₁.symm))
    rw [same] at runs
    exact Tier.agrees_of_reads layout accepted terminalSetup decoded bits ideal 1
      (by rw [started₁]; rfl) (by rw [finished₁]; rfl) long runs fold₁ (reads.field 1)
  have level₂ : Tier.Agrees (second) (challengeTrace20 pub proof messages 2 Key20.level₂)
      (path20 pub proof messages E 2) 0 point := by
    obtain ⟨result, runs⟩ := runs20_second o pub proof messages tLast eLast layout accepted
      terminalSetup decoded (expanded 2) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl
    have same : (second).check
        (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) =
          (second).check (point.wire (second).start) :=
      second_check o (claim₂ _ _ (by rw [started₂]; exact old.claim₂.symm))
    rw [same] at runs
    exact Tier.agrees_of_reads layout accepted terminalSetup decoded bits ideal 2
      (by rw [started₂]; rfl) (by rw [finished₂]; rfl) long runs fold₂ (reads.field 2)
  have level₃ : Tier.Agrees (third) (challengeTrace20 pub proof messages 3 Key20.level₃)
      (path20 pub proof messages E 3) 0 point := by
    obtain ⟨result, runs⟩ := runs20_third o pub proof messages tLast eLast layout accepted
      terminalSetup decoded (expanded 3) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl
    have same : (third).check
        (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) =
          (third).check (point.wire (third).start) :=
      third_check o (claim₃ _ _ (by rw [started₃]; exact old.claim₃.symm))
    rw [same] at runs
    exact Tier.agrees_of_reads layout accepted terminalSetup decoded bits ideal 3
      (by rw [started₃]; rfl) (by rw [finished₃]; rfl) long runs fold₃ (reads.field 3)
  have level₄ : Tier.Agrees (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄)
      (path20 pub proof messages E 4) 0 point := by
    obtain ⟨result, runs⟩ := runs20_fourth o pub proof messages tLast eLast layout accepted
      terminalSetup decoded (expanded 4) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl
    have same : (fourth).check
        (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) =
          (fourth).check (point.wire (fourth).start) :=
      fourth_check o (claim₄ _ _ (by rw [started₄]; exact old.claim₄.symm))
    rw [same] at runs
    exact Tier.agrees_of_reads layout accepted terminalSetup decoded bits ideal 4
      (by rw [started₄]; rfl) (by rw [finished₄]; rfl) long runs fold₄ (reads.field 4)
  have level₅ : Tier.Agrees (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅)
      (path20 pub proof messages E 5) 0 point := by
    obtain ⟨result, runs⟩ := runs20_fifth o pub proof messages tLast eLast layout accepted
      terminalSetup decoded (expanded 5) tower
      (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) rfl
    have same : (fifth).check
        (renote (fun _ => note20 pub proof messages tLast eLast) 0 point.first) =
          (fifth).check (point.wire (fifth).start) :=
      fifth_check o (claim₅ _ _ (by rw [started₅]; exact old.claim₅.symm))
    rw [same] at runs
    exact Tier.agrees_of_reads layout accepted terminalSetup decoded bits ideal 5
      (by rw [started₅]; rfl) (by rw [finished₅]; rfl) long runs fold₅ (reads.field 5)
  refine ⟨fun k below reply at_ => reads.work k below reply at_
      (work20_answer bits below (ran _) ideal), ?_, level₀, level₁, level₂, level₃, level₄,
    level₅, ?_⟩
  · obtain ⟨vs, pooled, same⟩ := weights20_answer (E := E) layout erased bits (ran _) ideal
    refine (Root.weights_of_reply _ (answered 0 (by decide))).trans ?_
    rw [reads.weights _ (answered 0 (by decide)) vs pooled]
    exact funext fun g => (same g).symm
  · subst tower
    exact ⟨_, answered 472 (by decide), innerL⟩

end At

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- `Agrees20` from the readings of each stage: for an accepted proof of `layout20` with the work
bits of the model and the ideal sponge `answer`, a transcript of `tower20` with the 473 stages,
whose wire message `k ≤ 473` is the page `page20 … k` of the proof and whose answers have the
readings of the code on the answers of the sponge (`Readings20`), has `Agrees20`. -/
theorem agrees20_of_reads {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3) (bits : Bits20 pub)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    {total : ℕ} {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (long : 473 ≤ point.rest.length)
    (wires : ∀ k ≤ 473, point.wire k = page20 pub proof messages tLast eLast k)
    (reads : Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
      point) :
    Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL point :=
  agreesAt20_of_reads _ pub proof messages tLast eLast layout accepted terminalSetup expanded
    decoded points values bits ideal rfl long wires reads

end NightstreamFPrime.Spec.Compression.Akita.Replay
