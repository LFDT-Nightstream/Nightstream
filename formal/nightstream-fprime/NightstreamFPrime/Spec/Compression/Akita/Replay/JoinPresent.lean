import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinShape
import NightstreamFPrime.Spec.Compression.Akita.Extract.PresentPage

/-! A transcript of the model that presents the bytes of a proof of `key20.aks` (Akita revision
`d50f700`; ePrint 2026/1983, revision of 2026-11-23, Section 10.3; `docs/akita-work/w16/design.md`
section 2).

- `workIndices20` and `nonceIndices20` are the indices of the 473 works and the 7 response nonces
  of the skeleton of `layout20`, and `starts20` the fold stages of the model (`stages20`).
- `StepOk` is the step of the replay at an index without a stop: the check of the code there
  holds.
- `challenge20₀`, `challenge20₁`, `challengeTrace20` and `challenge20L` are the fold challenges that
  the sampler of the code drew at the levels 0, 1, 2 to 5 and at the terminal, and `path20 ℓ` is
  the path of the code of the level `ℓ` (`foldPathIn`, with the tensor reduction of the next level).
- `Tier.Shows T c path source point`: the answer of the fold stage decodes at each coordinate to
  `c`, and the checks of `T` present `path` on the stages of `T`.
- `Reads20 … note`: the readings of the `Note` of the wire messages by `Key20.Tower.accepts`, for
  the notes `note k`, are those of `note20`.
- `PresentsAt20 o …` and `Presents20` (for `tower20`): each reading of the model on the transcript
  is the value of the replay: the 473 stages; the readings of the `Note` of the wire messages by
  `Key20.Tower.accepts` are those of `note20`; the nonce before each fold stage is below
  `responseTries` exactly when the code accepts the response nonce of that fold; the work slot of each stage has the work bits of the model exactly when the
  code accepts its work; the weights of the batching root are `row_coefficients`; each level shows
  the challenges and the path of the code; the fold answer of the terminal decodes to the challenges
  of the code. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

section Indices

/-- The operation is a work. -/
def isWork : Op → Bool
  | .work .. => true
  | _ => false

/-- The operation is a response nonce. -/
def isFoldNonce : Op → Bool
  | .foldNonce .. => true
  | _ => false

/-- The indices of the works of the skeleton of `layout20`: one for each stage of the model. -/
def workIndices20 : List ℕ := (sk20.filter fun x => isWork x.1).map (·.2)

/-- The indices of the response nonces of the skeleton of `layout20`: one for each fold. -/
def nonceIndices20 : List ℕ := (sk20.filter fun x => isFoldNonce x.1).map (·.2)

/-- The fold stages of the model: the levels 0 to 5 and the terminal. -/
def starts20 : List ℕ := [1, 51, 120, 203, 284, 378, 472]

theorem workIndices20_length : workIndices20.length = 473 := by decide +kernel

theorem nonceIndices20_length : nonceIndices20.length = 7 := by decide +kernel

end Indices

section Code

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The step of the replay at the index `c` has no stop: the check of the code at `c` holds on the
state before it. -/
def StepOk (c : ℕ) : Prop :=
  ∀ op, pub.ops[c]? = some op →
    (((step pub proof messages c op).run).run (stateBefore pub proof messages pub.ops c)).1 = .ok ()

/-- The state of the replay before the `fused.end` of the level `ℓ`. -/
abbrev fused20 (ℓ : ℕ) : Replayer := stateBefore pub proof messages pub.ops (fusedIndex pub.layout ℓ)

/-- The index where the tensor reduction of the claim of the level `ℓ` has ended: the `fused.end`
of the next level, or the terminal after the level 5. -/
def receiver20 (ℓ : ℕ) : ℕ := if ℓ < 5 then fusedIndex pub.layout (ℓ + 1) else pub.ops.size - 1

/-- The rounds `ν` of the tensor reduction of the claim of the level `ℓ`. -/
def tail20 : ℕ → ℕ
  | 1 => Key20.level₁.tail
  | 2 => Key20.level₂.tail
  | 3 => Key20.level₃.tail
  | 4 => Key20.level₄.tail
  | _ => Key20.level₅.tail

variable (E : Type) [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The path of the code of the level `ℓ`: the fold on the state before its `fused.end`, and for
`ℓ ≥ 1` the tensor reduction of its claim on the state before `receiver20 ℓ`. -/
noncomputable def path20 (ℓ : Fin 6) : List Move :=
  let s := fused20 pub proof messages ℓ
  let s' := stateBefore pub proof messages pub.ops (receiver20 pub ℓ)
  foldPathIn E pub s ℓ (fold20 ℓ) (setup20 pub ℓ) (fold20 ℓ).vars (fold20 ℓ).batch ++
    if ℓ.val = 0 then [] else
      tensoredPath E (tail20 ℓ)
        (some ((recordedIn s' ⟨ℓ, "fused.point", none⟩).toList,
          firstIn s' ⟨ℓ, "witness_evaluation", none⟩))
        ((recordedIn s' ⟨ℓ + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
        ((recordedIn s' ⟨ℓ + 1, "tensor.partials", none⟩).getD 1 Ext.zero)
        (firstIn s' ⟨ℓ + 1, "tensor.eta", none⟩) (roundsIn s' ⟨ℓ + 1, "tensor.sumcheck", none⟩)
        (firstIn s' ⟨ℓ + 1, "tensor.final_claims", none⟩)

/-- The fold challenges that the sampler of the code drew at the root fold, in the subring of
each group (`challenged` of `verifyProof_key20_root_of`). -/
noncomputable def challenge20₀ (b : Key20.root.Block) : Rq (Key20.root.carrierDegree b.1) :=
  Poly.toRq _ ((drawnCarrier Key20.root (fused20 pub proof messages 0) 0
    ((groupOrder 3).symm b.1)).getD b.2 (Poly.zero (Key20.root.carrierDegree b.1)))

/-- The fold challenges that the sampler of the code drew at the level 1, in its subring. -/
noncomputable def challenge20₁ (i : Fin Key20.level₁.shape.blocks) :
    Rq Key20.level₁.opening.carrierDegree :=
  Poly.toRq _ (((drawnIn (fused20 pub proof messages 1) 1 0).map fun c =>
    Poly.ofSparse Key20.level₁.opening.carrierDegree (c.map fun x =>
      (x.1 / (Key20.level₁.shape.innerDegree / Key20.level₁.opening.carrierDegree), x.2))).getD i
        (Poly.zero Key20.level₁.opening.carrierDegree))

/-- The fold challenges that the sampler of the code drew at the trace level `ℓ` of the shape
`L`. -/
noncomputable def challengeTrace20 (ℓ : ℕ) (L : Key20.Level) (i : Fin L.shape.blocks) :
    Rq L.shape.innerDegree :=
  Poly.toRq L.shape.innerDegree (((drawnIn (fused20 pub proof messages ℓ) ℓ 0).map
    (Poly.ofSparse L.shape.innerDegree)).getD i (Poly.zero L.shape.innerDegree))

/-- The fold challenges that the sampler of the code drew at the terminal. -/
noncomputable def challenge20L : Fin Key20.terminal.blocks → Rq Key20.terminal.innerDegree :=
  challengesOf Key20.terminal (terminalChallengesIn (terminalState20 pub proof messages) 6)

/-- The weights `ϑ` of the batching root that the code drew (`row_coefficients`). -/
noncomputable def weights20 (g : Fin 3) : E :=
  ((recordedIn (fused20 pub proof messages 0) ⟨0, "row_coefficients", none⟩).getD g Ext.zero).into

end Code

section Shows

variable {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)]
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [Fintype Coord] [DecidableEq Coord]
  [Fintype Slot] [Nonempty Slot] {total : ℕ}

/-- A level shows the challenges `c` and the path `path` of the code on a transcript: the answer of
its fold stage decodes at each fold coordinate to `c`, and its checks for the source `source`
present `path` on its stages. -/
def Tier.Shows (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source)
    (c : (b : Block) → Ring b) (path : List Move) (source : Source)
    (point : Prefix Wire (Option Coord → Slot) total) : Prop :=
  (∃ reply, point.reply T.start = some reply ∧ ∀ b, T.decode b (reply (T.block b)) = some (c b)) ∧
    Presents T.read (point.wire (T.start + 1))
      (((point.rest.take T.finish).drop (T.start + 1)).map fun move => (view move.1, move.2)) path
      (T.check (point.wire T.start) (T.challenge point) source)

end Shows

section Presents

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

/-- The readings of the `Note` of the wire messages by `Key20.Tower.accepts` (`towerLinks`:
`RootLinked`, `Tier.Linked`, `Tier.TraceLinked`, `Terminal.Linked`, `Terminal.Passed`,
`Tier.game`), for the notes `note k` of the wire messages `k`, are the readings of `note20` of the
replay: the commitments, values and points at the wire message 0, the claim of each level at its
fold stage (1, 51, 120, 203, 284, 378), the commitment of the next level in the message after it
(2, 52, 121, 204, 285), the state of the terminal at 379, its statement and partials at its fold
stage 472 and its response at 473. -/
structure Reads20 (note : ℕ → Key20.Note E) : Prop where
  commit : (note 0).commit = (note20 (E := E) pub proof messages tLast eLast).commit
  value : (note 0).value = (note20 (E := E) pub proof messages tLast eLast).value
  place : (note 0).point = (note20 (E := E) pub proof messages tLast eLast).point
  claim : (note 1).claim = (note20 (E := E) pub proof messages tLast eLast).claim
  commit₁ : (note 2).commit₁ = (note20 (E := E) pub proof messages tLast eLast).commit₁
  claim₁ : (note 51).claim₁ = (note20 (E := E) pub proof messages tLast eLast).claim₁
  commit₂ : (note 52).commit₂ = (note20 (E := E) pub proof messages tLast eLast).commit₂
  claim₂ : (note 120).claim₂ = (note20 (E := E) pub proof messages tLast eLast).claim₂
  commit₃ : (note 121).commit₃ = (note20 (E := E) pub proof messages tLast eLast).commit₃
  claim₃ : (note 203).claim₃ = (note20 (E := E) pub proof messages tLast eLast).claim₃
  commit₄ : (note 204).commit₄ = (note20 (E := E) pub proof messages tLast eLast).commit₄
  claim₄ : (note 284).claim₄ = (note20 (E := E) pub proof messages tLast eLast).claim₄
  commit₅ : (note 285).commit₅ = (note20 (E := E) pub proof messages tLast eLast).commit₅
  claim₅ : (note 378).claim₅ = (note20 (E := E) pub proof messages tLast eLast).claim₅
  state : (note 379).state = (note20 (E := E) pub proof messages tLast eLast).state
  statement : (note 472).statement = (note20 (E := E) pub proof messages tLast eLast).statement
  partials : (note 472).partials = (note20 (E := E) pub proof messages tLast eLast).partials
  response : (note 473).response = (note20 (E := E) pub proof messages tLast eLast).response

omit [Fintype E] [DecidableEq E] in
/-- Notes that are `note20` at each wire message have the readings of `note20`. -/
theorem Reads20.of_all {note : ℕ → Key20.Note E}
    (all : ∀ k, note k = note20 pub proof messages tLast eLast) :
    Reads20 pub proof messages tLast eLast note :=
  ⟨congrArg Key20.Note.commit (all 0), congrArg Key20.Note.value (all 0),
    congrArg Key20.Note.point (all 0), congrArg Key20.Note.claim (all 1),
    congrArg Key20.Note.commit₁ (all 2), congrArg Key20.Note.claim₁ (all 51),
    congrArg Key20.Note.commit₂ (all 52), congrArg Key20.Note.claim₂ (all 120),
    congrArg Key20.Note.commit₃ (all 121), congrArg Key20.Note.claim₃ (all 203),
    congrArg Key20.Note.commit₄ (all 204), congrArg Key20.Note.claim₄ (all 284),
    congrArg Key20.Note.commit₅ (all 285), congrArg Key20.Note.claim₅ (all 378),
    congrArg Key20.Note.state (all 379), congrArg Key20.Note.statement (all 472),
    congrArg Key20.Note.partials (all 472), congrArg Key20.Note.response (all 473)⟩

/-- The transcript `point` of the tower data `o` presents the replay of a proof of `key20.aks`
(design section 2, items 1 to 4). `long`: it has the 473 stages. `reads`: each reading of the
`Note` of a wire message by `Key20.Tower.accepts` is the reading of `note20` of the replay
(`Reads20`). `nonce`: the nonce of the message before the fold stage `j` is
below `responseTries` exactly when the code accepts the response nonce of the fold `j`. `work`:
the work slot of the stage `k` has the work bits `bits20` of the model exactly when the code
accepts the work `k`. `weights`: the weights of the batching root are `row_coefficients`. `level₀` to
`level₅`: each level shows the fold challenges and the path of the code, for the source 0.
`terminal`: the fold answer of the terminal decodes to the fold challenges of the code. -/
structure PresentsAt20 (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total) :
    Prop where
  long : 473 ≤ point.rest.length
  reads : Reads20 pub proof messages tLast eLast fun k => (point.wire k).data
  nonce : ∀ j < 7, ((point.wire (starts20.getD j 0)).nonce : ℕ) < Extract.responseTries ↔
    StepOk pub proof messages (nonceIndices20.getD j 0)
  work : ∀ k < 473, ∀ reply, point.reply k = some reply →
    (reply none ∈ Extract.workZeros (CodeSlot E) (bits20.flatten.getD k 0) ↔
      StepOk pub proof messages (workIndices20.getD k 0))
  weights : (Extract.claimsRoot (Coord := Coord) (Slot := CodeSlot E) (total := total) 0 o.read
    o.value o.place).weights point = weights20 pub proof messages E
  level₀ : Tier.Shows (zeroth) (challenge20₀ pub proof messages) (path20 pub proof messages E 0) 0
    point
  level₁ : Tier.Shows (first) (challenge20₁ pub proof messages) (path20 pub proof messages E 1) 0
    point
  level₂ : Tier.Shows (second) (challengeTrace20 pub proof messages 2 Key20.level₂)
    (path20 pub proof messages E 2) 0 point
  level₃ : Tier.Shows (third) (challengeTrace20 pub proof messages 3 Key20.level₃)
    (path20 pub proof messages E 3) 0 point
  level₄ : Tier.Shows (fourth) (challengeTrace20 pub proof messages 4 Key20.level₄)
    (path20 pub proof messages E 4) 0 point
  level₅ : Tier.Shows (fifth) (challengeTrace20 pub proof messages 5 Key20.level₅)
    (path20 pub proof messages E 5) 0 point
  terminal : ∃ reply, point.reply o.terminal.start = some reply ∧
    ∀ i, o.terminal.decode (reply (some (o.terminal.block i))) =
      some (challenge20L pub proof messages i)

end Presents

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]

/-- A transcript of `tower20` presents the bytes of a proof of `key20.aks` with the decoded
messages `messages` and the last blocks `tLast` and `eLast` of the terminal: `PresentsAt20` for
the tower data of the replay with the fold challenge of the terminal at the stage 472. -/
def Presents20 (pub : Public) (proof : ByteArray) (messages : Array Message)
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
    (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
    (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
    (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
    (blockL : Fin Key20.terminal.blocks → Coord) {total : ℕ}
    (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total) : Prop :=
  PresentsAt20 (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof
    messages tLast eLast point

end NightstreamFPrime.Spec.Compression.Akita.Replay
