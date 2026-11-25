import NightstreamFPrime.Spec.Compression.Akita.Replay.StageDrawsAgrees

/-! The converse of `agrees20_of_reads`, and a transcript with `Readings20` (Akita revision
`d50f700`; ePrint 2026/1983, revision of 2026-11-23, Section 10.3;
`docs/akita-work/w21/design.md`, track Q; `Replay/StageDrawsAgrees.lean`).

- `Extract.draws_of_samples`: answers with `Extract.Samples` for a path give each draw of the
  path.
- `Tier.Agrees.field`: a level with `Tier.Agrees` on the path of the code has `FieldReads20` at
  the stage of each draw. `foldReads20_of_decoded`: a fold answer that decodes to the challenges
  of the code has `FoldReads20`.
- `readingsAt20_of_agrees`, `readings20_of_agrees`: `Agrees20` gives `Readings20`, for an
  accepted proof of `layout20` with an ideal sponge.
- `exists_readings20`: the hypothesis `Readings20` of `agrees20_of_reads` holds for a transcript
  of each accepted proof, with the pages `page20`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Game (Move)

/-- Answers with `Samples` for a path give each draw of the path. -/
theorem draws_of_samples {Reply : Type} [Fintype Reply] {α : Type} (game : Game α)
    (path : List Move) (views : List Reply) (samples : Samples Reply game path views) (j : ℕ)
    (C : Type) (c : C) (at_ : (draws path)[j]? = some (Move.draw c)) :
    ∃ reply, views[j]? = some reply ∧ ∀ (finite : Fintype C) (nonempty : Nonempty C),
      @sample Reply _ C finite nonempty reply = c := by
  induction game generalizing path views j C c with
  | pure a =>
      cases path with
      | nil => cases at_
      | cons move path => cases move <;> exact samples.elim
  | draw D bad next induction =>
      cases path with
      | nil => exact samples.elim
      | cons move path =>
          cases move with
          | recv m => exact samples.elim
          | draw c₀ =>
              cases views with
              | nil => exact samples.elim
              | cons reply views =>
                  obtain ⟨same, sampled, samples⟩ := samples
                  obtain rfl := same.symm
                  cases j with
                  | zero =>
                      have first : some (Move.draw c₀) = some (Move.draw c) := at_
                      cases first
                      have read : sample Reply D reply = c₀ := sampled
                      refine ⟨reply, rfl, fun finite nonempty => ?_⟩
                      convert read
                  | succ j =>
                      obtain ⟨other, found, read⟩ := induction c₀ path views samples j C c at_
                      exact ⟨other, found, read⟩
  | recv M next induction =>
      cases path with
      | nil => exact samples.elim
      | cons move path =>
          cases move with
          | draw c₀ => exact samples.elim
          | recv m =>
              obtain ⟨same, samples⟩ := samples
              obtain rfl := same.symm
              exact induction m path views samples j C c at_

end NightstreamFPrime.Spec.Compression.Akita.Extract

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix Packs Samples
  countDraws draws)

open Replayer

section Reads

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {pub : Public} {proof : ByteArray} {messages : Array Message}
  {answer : List Duplex.Event → ℕ → ByteArray}

omit [DecidableEq E] in
/-- A level with `Tier.Agrees` on the path of the code of an accepted proof of `layout20` has
`FieldReads20` at the stage of each draw of the path. -/
theorem Tier.Agrees.field {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)]
    {Stmt Brk Wit Weak Result Source : Type}
    {T : Tier Block Ring Stmt Brk Wit Weak (Key20.Wire E) Coord (CodeSlot E) Result Source}
    {c : (b : Block) → Ring b} {source : Source}
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (bits : Bits20 pub)
    (ideal : pub.sponge.ideal = some answer) (ℓ : Fin 6) (started : T.start + 1 = first20 ℓ)
    (finished : T.finish = bound20 ℓ)
    (agrees : Tier.Agrees T c (path20 pub proof messages E ℓ) source point) {j : ℕ} {d : Draw}
    (at_ : (draws20 ℓ)[j]? = some d) {reply : Option Coord → CodeSlot E}
    (answered : point.reply (first20 ℓ + j) = some reply) :
    FieldReads20 pub proof messages answer (first20 ℓ + j) d reply := by
  intro values pooled finite nonempty
  obtain ⟨bounded, values', pooled', drawn⟩ :=
    path20_draw E layout accepted terminalSetup decoded bits ideal ℓ at_
  obtain rfl : values' = values := Option.some.inj (pooled'.symm.trans pooled)
  obtain ⟨-, samples⟩ := agrees
  obtain ⟨r, found, sampled⟩ := Extract.draws_of_samples _ _ _ samples j _ _ drawn
  rw [List.getElem?_map, List.getElem?_drop, List.getElem?_take_of_lt (by omega), started]
    at found
  obtain ⟨x, here, rfl⟩ := Option.map_eq_some_iff.mp found
  have first : (point.rest[first20 ℓ + j]?).map Prod.fst = some reply := answered
  rw [here] at first
  obtain rfl : x.1 = reply := Option.some.inj first
  exact sampled finite nonempty

omit [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E] [Fintype Coord]
  [DecidableEq Coord] in
/-- A fold answer that decodes to the challenges `c` has `FoldReads20`, when `c` are the
challenges of the sampler for the drawn challenges of a state. -/
theorem foldReads20_of_decoded {k g level : ℕ} {G : FoldGroup} [NeZero G.degree] {d n : ℕ}
    [NeZero d] {decode : CodeSlot E → Option (Rq d)} {block : Fin n → Coord}
    {reply : Option Coord → CodeSlot E} (draw : foldDraw20 k g = some (level, G))
    {s : Replayer}
    (drawn : foldChallenges G (answer (drawEvents20 k g ++
      .squeeze 8 :: eventsAt20 pub proof messages k) G.messageBytes) = some (drawnIn s level g))
    (count : G.coordinates = n) {c : Fin n → Rq d}
    (challenge : ∀ cs : List Sparse, drawnIn s level g = cs.map (ambient G) →
      ∀ (i : Fin n) (inside : (i : ℕ) < cs.length), c i = Sparse.ring d cs[i])
    (decoded : ∀ i, decode (reply (some (block i))) = some (c i)) :
    FoldReads20 pub proof messages answer k g decode block reply := by
  intro level' G' cs draw' sampled i
  have same : some (level', G') = some (level, G) := draw'.symm.trans draw
  cases same
  have mapped : drawnIn s level g = cs.map (ambient G) := by
    unfold foldChallenges at drawn
    rw [sampled] at drawn
    exact (Option.some.inj drawn).symm
  have inside : (i : ℕ) < cs.length := by
    rw [(sample_mem sampled).1, count]
    exact i.isLt
  rw [decoded i, challenge cs mapped i inside, List.getElem?_eq_getElem inside]
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
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- The converse of `agreesAt20_of_reads`: a transcript with `AgreesAt20` for the tower data `o`
of the replay has `Readings20`, for an accepted proof of `layout20` with the work bits of the
model and the ideal sponge `answer`. -/
theorem readingsAt20_of_agrees {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (bits : Bits20 pub)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (agrees : AgreesAt20 o pub proof messages point) :
    Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
      point := by
  obtain ⟨erased, ran⟩ := ran_of_accept layout accepted terminalSetup decoded
  obtain ⟨started₀, -, finished₀, -⟩ := zeroth_numbers o
  obtain ⟨started₁, -, finished₁, -⟩ := first20_numbers o
  obtain ⟨started₂, -, finished₂, -⟩ := second20_numbers o
  obtain ⟨started₃, -, finished₃, -⟩ := third20_numbers o
  obtain ⟨started₄, -, finished₄, -⟩ := fourth20_numbers o
  obtain ⟨started₅, -, finished₅, -⟩ := fifth20_numbers o
  obtain ⟨d00, d01, d02, d1, d2, d3, d4, d5, dL, beforeL⟩ := foldDraws20
  have dec₀ : ∀ reply, point.reply 1 = some reply → ∀ b : Key20.root.Block,
      Key20.rootDecode E b.1 (reply (some (block b))) =
        some (challenge20₀ pub proof messages b) := by
    obtain ⟨⟨r, at_r, decodes⟩, -⟩ := agrees.level₀
    rw [started₀] at at_r
    intro reply at_
    obtain rfl : r = reply := Option.some.inj (at_r.symm.trans at_)
    subst tower
    exact decodes
  have dec₁ : ∀ reply, point.reply 51 = some reply → ∀ i : Fin Key20.level₁.shape.blocks,
      code₁ (reply (some (block₁ i))) = some (challenge20₁ pub proof messages i) := by
    obtain ⟨⟨r, at_r, decodes⟩, -⟩ := agrees.level₁
    rw [started₁] at at_r
    intro reply at_
    obtain rfl : r = reply := Option.some.inj (at_r.symm.trans at_)
    subst tower
    exact decodes
  have dec₂ : ∀ reply, point.reply 120 = some reply → ∀ i : Fin Key20.level₂.shape.blocks,
      code₂ (reply (some (block₂ i))) =
        some (challengeTrace20 pub proof messages 2 Key20.level₂ i) := by
    obtain ⟨⟨r, at_r, decodes⟩, -⟩ := agrees.level₂
    rw [started₂] at at_r
    intro reply at_
    obtain rfl : r = reply := Option.some.inj (at_r.symm.trans at_)
    subst tower
    exact decodes
  have dec₃ : ∀ reply, point.reply 203 = some reply → ∀ i : Fin Key20.level₃.shape.blocks,
      code₃ (reply (some (block₃ i))) =
        some (challengeTrace20 pub proof messages 3 Key20.level₃ i) := by
    obtain ⟨⟨r, at_r, decodes⟩, -⟩ := agrees.level₃
    rw [started₃] at at_r
    intro reply at_
    obtain rfl : r = reply := Option.some.inj (at_r.symm.trans at_)
    subst tower
    exact decodes
  have dec₄ : ∀ reply, point.reply 284 = some reply → ∀ i : Fin Key20.level₄.shape.blocks,
      code₄ (reply (some (block₄ i))) =
        some (challengeTrace20 pub proof messages 4 Key20.level₄ i) := by
    obtain ⟨⟨r, at_r, decodes⟩, -⟩ := agrees.level₄
    rw [started₄] at at_r
    intro reply at_
    obtain rfl : r = reply := Option.some.inj (at_r.symm.trans at_)
    subst tower
    exact decodes
  have dec₅ : ∀ reply, point.reply 378 = some reply → ∀ i : Fin Key20.level₅.shape.blocks,
      code₅ (reply (some (block₅ i))) =
        some (challengeTrace20 pub proof messages 5 Key20.level₅ i) := by
    obtain ⟨⟨r, at_r, decodes⟩, -⟩ := agrees.level₅
    rw [started₅] at at_r
    intro reply at_
    obtain rfl : r = reply := Option.some.inj (at_r.symm.trans at_)
    subst tower
    exact decodes
  have decL : ∀ reply, point.reply 472 = some reply → ∀ i : Fin Key20.terminal.blocks,
      codeL (reply (some (blockL i))) = some (challenge20L pub proof messages i) := by
    obtain ⟨r, at_r, decodes⟩ := agrees.terminal
    subst tower
    intro reply at_
    obtain rfl : r = reply := Option.some.inj (at_r.symm.trans at_)
    exact decodes
  have laterL : stageIndex20 pub 472 + 1 + 0 + 1 ≤ pub.ops.size - 1 := by
    rw [stageIndex20_lt pub (by decide), size_skeleton erased, skeleton20_length]
    exact beforeL
  have root₀ : ∀ reply, point.reply 1 = some reply →
      FoldReads20 pub proof messages answer 1 ((0 : Fin 3) : ℕ) (Key20.rootDecode E 0)
        (fun i : Fin (Key20.root.shape 0).blocks => block ⟨0, i⟩) reply := fun reply at_ =>
    foldReads20_of_decoded d00 (drawn20_answer layout erased bits 0 d00 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challenge20₀_eq pub proof messages 0 i mapped (by decide) (by decide) inside)
      fun i => dec₀ reply at_ ⟨0, i⟩
  have root₁ : ∀ reply, point.reply 1 = some reply →
      FoldReads20 pub proof messages answer 1 ((1 : Fin 3) : ℕ) (Key20.rootDecode E 1)
        (fun i : Fin (Key20.root.shape 1).blocks => block ⟨1, i⟩) reply := fun reply at_ =>
    foldReads20_of_decoded d01 (drawn20_answer layout erased bits 0 d01 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challenge20₀_eq pub proof messages 1 i mapped (by decide) (by decide) inside)
      fun i => dec₀ reply at_ ⟨1, i⟩
  have root₂ : ∀ reply, point.reply 1 = some reply →
      FoldReads20 pub proof messages answer 1 ((2 : Fin 3) : ℕ) (Key20.rootDecode E 2)
        (fun i : Fin (Key20.root.shape 2).blocks => block ⟨2, i⟩) reply := fun reply at_ =>
    foldReads20_of_decoded d02 (drawn20_answer layout erased bits 0 d02 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challenge20₀_eq pub proof messages 2 i mapped (by decide) (by decide) inside)
      fun i => dec₀ reply at_ ⟨2, i⟩
  refine ⟨fun k below reply at_ _ => agrees.work k below reply at_, ?_, ?_, ?_,
    fun reply at_ => foldReads20_of_decoded d1
      (drawn20_answer layout erased bits 1 d1 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challenge20₁_eq pub proof messages i mapped (by decide) (by decide) inside)
      (dec₁ reply at_),
    fun reply at_ => foldReads20_of_decoded d2
      (drawn20_answer layout erased bits 2 d2 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challengeTrace20_eq pub proof messages 2 Key20.level₂ i mapped rfl (by decide) inside)
      (dec₂ reply at_),
    fun reply at_ => foldReads20_of_decoded d3
      (drawn20_answer layout erased bits 3 d3 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challengeTrace20_eq pub proof messages 3 Key20.level₃ i mapped rfl (by decide) inside)
      (dec₃ reply at_),
    fun reply at_ => foldReads20_of_decoded d4
      (drawn20_answer layout erased bits 4 d4 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challengeTrace20_eq pub proof messages 4 Key20.level₄ i mapped rfl (by decide) inside)
      (dec₄ reply at_),
    fun reply at_ => foldReads20_of_decoded d5
      (drawn20_answer layout erased bits 5 d5 (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challengeTrace20_eq pub proof messages 5 Key20.level₅ i mapped rfl (by decide) inside)
      (dec₅ reply at_),
    fun reply at_ => foldReads20_of_decoded dL
      (squeeze20_drawn erased bits (by decide) dL laterL (ran _) ideal) rfl
      (fun cs mapped i inside =>
        challenge20L_eq pub proof messages i mapped rfl (by decide) inside)
      (decL reply at_)⟩
  · intro reply at_ vs pooled finite
    obtain ⟨vs', pooled', same⟩ := weights20_answer (E := E) layout erased bits (ran _) ideal
    obtain rfl : vs' = vs := Option.some.inj (pooled'.symm.trans pooled)
    have weighed := agrees.weights.symm.trans (Root.weights_of_reply
      (Extract.claimsRoot (Coord := Coord) (Slot := CodeSlot E) (total := total) 0 o.read o.value
        o.place) at_)
    have free : ∀ other : Fintype (Fin 3 → E),
        @Extract.sample (Coord → CodeSlot E) _ (Fin 3 → E) finite _ (view reply) =
          @Extract.sample (Coord → CodeSlot E) _ (Fin 3 → E) other _ (view reply) :=
      fun other => by rw [Subsingleton.elim finite other]
    exact (free _).trans (weighed.symm.trans (funext same))
  · intro ℓ j d at_ reply answered
    match ℓ with
    | 0 =>
        exact Tier.Agrees.field layout accepted terminalSetup decoded bits ideal 0
          (by rw [started₀]; rfl) (by rw [finished₀]; rfl) agrees.level₀ at_ answered
    | 1 =>
        exact Tier.Agrees.field layout accepted terminalSetup decoded bits ideal 1
          (by rw [started₁]; rfl) (by rw [finished₁]; rfl) agrees.level₁ at_ answered
    | 2 =>
        exact Tier.Agrees.field layout accepted terminalSetup decoded bits ideal 2
          (by rw [started₂]; rfl) (by rw [finished₂]; rfl) agrees.level₂ at_ answered
    | 3 =>
        exact Tier.Agrees.field layout accepted terminalSetup decoded bits ideal 3
          (by rw [started₃]; rfl) (by rw [finished₃]; rfl) agrees.level₃ at_ answered
    | 4 =>
        exact Tier.Agrees.field layout accepted terminalSetup decoded bits ideal 4
          (by rw [started₄]; rfl) (by rw [finished₄]; rfl) agrees.level₄ at_ answered
    | 5 =>
        exact Tier.Agrees.field layout accepted terminalSetup decoded bits ideal 5
          (by rw [started₅]; rfl) (by rw [finished₅]; rfl) agrees.level₅ at_ answered
  · intro reply at_ g
    fin_cases g
    exacts [root₀ reply at_, root₁ reply at_, root₂ reply at_]

end At

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- The converse of `agrees20_of_reads`: a transcript of `tower20` with `Agrees20` has
`Readings20`, for an accepted proof of `layout20` with the work bits of the model and the ideal
sponge `answer`. -/
theorem readings20_of_agrees {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (bits : Bits20 pub)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    {total : ℕ} {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (agrees : Agrees20 pub proof messages block block₁ block₂ block₃ block₄ block₅ blockL point) :
    Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL point :=
  readingsAt20_of_agrees _ pub proof messages layout accepted terminalSetup decoded bits ideal rfl
    agrees

/-- The hypotheses of `agrees20_of_reads` hold for a transcript: for an accepted proof of
`layout20` with the work bits of the model and the ideal sponge `answer`, and each last blocks
`tLast`, `eLast` of the terminal, a transcript of `tower20` with `total ≥ 473` stages has the 473
stages, the wire messages `page20` and `Readings20`. The hypotheses are those of
`exists_agrees20`, `Bits20` and the ideal sponge. -/
theorem exists_readings20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3) (bits : Bits20 pub)
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord) {total : ℕ}
    (inside : 473 ≤ total) :
    ∃ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total,
      473 ≤ point.rest.length ∧
        (∀ k ≤ 473, point.wire k = page20 pub proof messages tLast eLast k) ∧
        Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
          point := by
  obtain ⟨point, long, wires, agrees⟩ := exists_agrees20 (E := E) pub proof messages tLast eLast
    layout accepted terminalSetup expanded decoded points values injective injective₁ injective₂
    injective₃ injective₄ injective₅ injectiveL slots inside
  exact ⟨point, long, wires,
    readings20_of_agrees pub proof messages layout accepted terminalSetup decoded bits ideal
      agrees⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
