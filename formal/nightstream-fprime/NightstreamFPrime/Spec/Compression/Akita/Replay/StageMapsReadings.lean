import NightstreamFPrime.Spec.Compression.Akita.Replay.StageMaps

/-! `Readings20` for a transcript whose answers are coupled with the table of the sponge by the
stage maps `phi20` (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23, Section
8.2 `[KECCAK]` and Lemma 10.47; `docs/akita-work/w21/design.md`, section 5;
`Replay/StageMaps.lean`, `Replay/StageDrawsAgrees.lean`).

- `FoldAt.reads`, `RootAt.reads`: the readings `FoldAt` and `RootAt` of a table entry at the base
  of a fold stage give `FoldReads20` and the work slot, on a run with no stop.
- `readings20_of_coupled`: an accepted proof of `layout20` with the sponge of the table `fs` and
  a transcript whose answer at each stage `k < 473` gives the entry of `fs` at the base of the
  stage under `phi20 … k` has `Readings20` on the answers of `fs`.
- `exists_coupled20`: each table has a transcript with 473 stages and this property. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (view Prefix)
open Bytes

open Replayer

section Folds

variable {E : Type} [Fintype E] {Coord : Type} {L B C S : ℕ}
  {pub : Public} {proof : ByteArray} {messages : Array Message}
  {fs : Sponge.Base L B C → Sponge.Y S C} {base : Sponge.Base L B C}
  {reply : Option Coord → CodeSlot E} {y : Sponge.Y S C} {bits : ℕ}
  {φ : (Option Coord → CodeSlot E) × Sponge.Y S C → Sponge.Y S C}

/-- The reading `FoldAt` of the entry of a table at the base of the fold stage `k` gives
`FoldReads20` for the group of the stage, and the work slot on a run with no stop up to and with
the draw. -/
theorem FoldAt.reads {k level : ℕ} {G : FoldGroup} [NeZero G.degree]
    {block : Fin G.coordinates → Coord} {decode : CodeSlot E → Option (Rq G.degree)}
    (fold : FoldAt bits G block decode φ) (entry : fs base = φ (reply, y))
    (passed : worked (Sponge.answerOf fs base.events 8) bits = true)
    (events : base.events = eventsAt20 pub proof messages k)
    (draw : foldDraw20 k 0 = some (level, G))
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (work : Bits20 pub)
    (inside : k < 473)
    (ran : ∀ c, (runFrom pub proof messages (pub.ops.zipIdx.toList.take c)
      (channelStart pub)).1 = .ok ())
    (ideal : pub.sponge.ideal = some (Sponge.answerOf fs)) :
    FoldReads20 pub proof messages (Sponge.answerOf fs) k 0 decode block reply ∧
      reply none ∈ Extract.workZeros (CodeSlot E) bits := by
  refine ⟨fun level' G' cs draw' sampled i => ?_, ?_⟩
  · obtain ⟨-, rfl⟩ := Prod.mk.inj (Option.some.inj (draw.symm.trans draw'))
    rw [drawEvents20_zero, List.nil_append, ← events] at sampled
    exact (fold fs base reply y entry passed cs sampled).2 i
  · obtain ⟨cs, drawn, -⟩ := squeeze20_answer erased work inside draw (ran _) ideal
    unfold foldChallenges at drawn
    obtain ⟨cs₀, sampled, -⟩ := Option.map_eq_some_iff.mp drawn
    rw [drawEvents20_zero, List.nil_append, ← events] at sampled
    exact (fold fs base reply y entry passed cs₀ sampled).1

/-- The reading `RootAt` of the entry of a table at the base of the fold stage of the root gives
the work slot and `FoldReads20` for each group of the root, on a run with no stop. -/
theorem RootAt.reads {block : Key20.root.Block → Coord} (root : RootAt bits block φ)
    (entry : fs base = φ (reply, y))
    (passed : worked (Sponge.answerOf fs base.events 8) bits = true)
    (events : base.events = eventsAt20 pub proof messages 1)
    (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0)) (work : Bits20 pub)
    (ran : ∀ c, (runFrom pub proof messages (pub.ops.zipIdx.toList.take c)
      (channelStart pub)).1 = .ok ())
    (ideal : pub.sponge.ideal = some (Sponge.answerOf fs)) :
    reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
      ∀ g : Fin 3, FoldReads20 pub proof messages (Sponge.answerOf fs) 1 g
        (Key20.rootDecode E g) (fun i : Fin (Key20.root.shape g).blocks => block ⟨g, i⟩) reply := by
  have each : ∀ g : Fin 3, ∃ cs, sample (rootGroup20 g) (Sponge.answerOf fs
      (drawEvents20 1 g ++ .squeeze 8 :: base.events) (rootGroup20 g).messageBytes) =
        some cs := by
    intro g
    obtain ⟨cs, drawn, -⟩ := squeeze20_answer erased work (by decide : 1 < 473)
      (foldDraw20_root g) (ran _) ideal
    unfold foldChallenges at drawn
    obtain ⟨cs₀, sampled, -⟩ := Option.map_eq_some_iff.mp drawn
    rw [← events] at sampled
    exact ⟨cs₀, sampled⟩
  choose cs sampled using each
  obtain ⟨zero, decodes⟩ := root fs base reply y entry passed cs sampled
  refine ⟨zero, fun g level G cs' draw sampled' i => ?_⟩
  obtain ⟨-, rfl⟩ := Prod.mk.inj (Option.some.inj ((foldDraw20_root g).symm.trans draw))
  rw [← events] at sampled'
  obtain rfl : cs g = cs' := Option.some.inj ((sampled g).symm.trans sampled')
  exact decodes g i

end Folds

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {L B C S : ℕ}
  {pub : Public} {proof : ByteArray} {messages : Array Message}
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- `Readings20` from the stage maps: for an accepted proof of `layout20` with the sponge of the
table `fs`, a transcript whose answer `reply` at each stage `k < 473` gives the entry of `fs` at
the base of the stage as `phi20 … k (reply, y)` for some `y` has the readings of the code on the
answers of `fs`. The table has strings of `C ≥ drawBytes20` bytes and sub-keys of `S ≥ 3`
squeezes, the block maps are injective and the answer has 22 coordinates or more. -/
theorem readings20_of_coupled {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (fs : Sponge.Base L B C → Sponge.Y S C)
    (ideal : pub.sponge.ideal = some (Sponge.answerOf fs)) (sizes : drawBytes20 ≤ C)
    (three : 3 ≤ S) (injective : Function.Injective block)
    (injective₁ : Function.Injective block₁) (injective₂ : Function.Injective block₂)
    (injective₃ : Function.Injective block₃) (injective₄ : Function.Injective block₄)
    (injective₅ : Function.Injective block₅) (injectiveL : Function.Injective blockL)
    (slots : 22 ≤ Fintype.card Coord) {total : ℕ}
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (coupled : ∀ k < 473, ∀ reply, point.reply k = some reply → ∀ base : Sponge.Base L B C,
      base.events = eventsAt20 pub proof messages k →
      ∃ y, fs base = phi20 block block₁ block₂ block₃ block₄ block₅ blockL k (reply, y)) :
    Readings20 pub proof messages (Sponge.answerOf fs) block block₁ block₂ block₃ block₄ block₅
      blockL point := by
  obtain ⟨erased, ran⟩ := ran_of_accept layout accepted terminalSetup decoded
  have work := bits20_of_scripted layout (verifyProof_script20 layout accepted).1
  obtain ⟨-, -, -, d1, d2, d3, d4, d5, dL, -⟩ := foldDraws20
  have staged : ∀ k (inside : k < 473) reply, point.reply k = some reply →
      ∃ (base : Sponge.Base L B C) (y : Sponge.Y S C),
        base.events = eventsAt20 pub proof messages k ∧
        fs base = phi20 block block₁ block₂ block₃ block₄ block₅ blockL k (reply, y) ∧
        worked (Sponge.answerOf fs base.events 8) (bits20.flatten.getD k 0) = true := by
    intro k inside reply at_
    obtain ⟨base, events⟩ := base20 fs work inside (ran _) ideal
    obtain ⟨y, entry⟩ := coupled k inside reply at_ base events
    refine ⟨base, y, events, entry, ?_⟩
    rw [events]
    exact work20_answer work inside (ran _) ideal
  have maps : ∀ k (inside : k < 473), Coupled20 block block₁ block₂ block₃ block₄ block₅ blockL k
      (phi20 block block₁ block₂ block₃ block₄ block₅ blockL k :
        (Option Coord → CodeSlot E) × Sponge.Y S C → Sponge.Y S C) :=
    fun k inside => phi20_coupled sizes three injective injective₁ injective₂ injective₃
      injective₄ injective₅ injectiveL slots inside
  refine
    { work := ?_, weights := ?_, field := ?_, root := ?_, level₁ := ?_, level₂ := ?_,
      level₃ := ?_, level₄ := ?_, level₅ := ?_, terminal := ?_ }
  · intro k inside reply at_ _
    obtain ⟨base, y, events, entry, passed⟩ := staged k inside reply at_
    rcases stage20_cases inside with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | ⟨ℓ, j, d, at', rfl⟩
    · obtain ⟨values, pooled, -⟩ :=
        challenge20_answer erased work inside row20_first.1 (ran _) ideal
      rw [← events] at pooled
      exact ((maps 0 inside).weights rfl fs base reply y entry passed values pooled).1
    · exact (RootAt.reads ((maps 1 inside).root rfl) entry passed events erased work ran ideal).1
    · exact (FoldAt.reads ((maps 51 inside).level₁ rfl) entry passed events d1 erased work inside ran
        ideal).2
    · exact (FoldAt.reads ((maps 120 inside).level₂ rfl) entry passed events d2 erased work inside ran
        ideal).2
    · exact (FoldAt.reads ((maps 203 inside).level₃ rfl) entry passed events d3 erased work inside ran
        ideal).2
    · exact (FoldAt.reads ((maps 284 inside).level₄ rfl) entry passed events d4 erased work inside ran
        ideal).2
    · exact (FoldAt.reads ((maps 378 inside).level₅ rfl) entry passed events d5 erased work inside ran
        ideal).2
    · exact (FoldAt.reads ((maps 472 inside).terminal rfl) entry passed events dL erased work inside ran
        ideal).2
    · obtain ⟨values, pooled, -⟩ :=
        challenge20_answer erased work inside (draws20_checked ℓ at').1 (ran _) ideal
      rw [← events] at pooled
      exact ((maps _ inside).field ℓ j d at' rfl fs base reply y entry passed values pooled).1
  · intro reply at_ values pooled finite
    obtain ⟨base, y, events, entry, passed⟩ := staged 0 (by decide) reply at_
    rw [← events] at pooled
    exact ((maps 0 (by decide)).weights rfl fs base reply y entry passed values pooled).2 finite
  · intro ℓ j d at' reply at_ values pooled finite nonempty
    have inside := (draws20_stage ℓ at').1
    obtain ⟨base, y, events, entry, passed⟩ := staged _ inside reply at_
    rw [← events] at pooled
    exact ((maps _ inside).field ℓ j d at' rfl fs base reply y entry passed values pooled).2
      finite nonempty
  · intro reply at_
    obtain ⟨base, y, events, entry, passed⟩ := staged 1 (by decide) reply at_
    exact (RootAt.reads ((maps 1 (by decide)).root rfl) entry passed events erased work ran ideal).2
  · intro reply at_
    obtain ⟨base, y, events, entry, passed⟩ := staged 51 (by decide) reply at_
    exact (FoldAt.reads ((maps 51 (by decide)).level₁ rfl) entry passed events d1 erased work
      (by decide) ran ideal).1
  · intro reply at_
    obtain ⟨base, y, events, entry, passed⟩ := staged 120 (by decide) reply at_
    exact (FoldAt.reads ((maps 120 (by decide)).level₂ rfl) entry passed events d2 erased work
      (by decide) ran ideal).1
  · intro reply at_
    obtain ⟨base, y, events, entry, passed⟩ := staged 203 (by decide) reply at_
    exact (FoldAt.reads ((maps 203 (by decide)).level₃ rfl) entry passed events d3 erased work
      (by decide) ran ideal).1
  · intro reply at_
    obtain ⟨base, y, events, entry, passed⟩ := staged 284 (by decide) reply at_
    exact (FoldAt.reads ((maps 284 (by decide)).level₄ rfl) entry passed events d4 erased work
      (by decide) ran ideal).1
  · intro reply at_
    obtain ⟨base, y, events, entry, passed⟩ := staged 378 (by decide) reply at_
    exact (FoldAt.reads ((maps 378 (by decide)).level₅ rfl) entry passed events d5 erased work
      (by decide) ran ideal).1
  · intro reply at_
    obtain ⟨base, y, events, entry, passed⟩ := staged 472 (by decide) reply at_
    exact (FoldAt.reads ((maps 472 (by decide)).terminal rfl) entry passed events dL erased work
      (by decide) ran ideal).1

/-- The hypothesis `coupled` of `readings20_of_coupled` holds for a transcript: each table `fs`
has a transcript with the 473 stages whose answer at each stage `k < 473` gives the entry of `fs`
at each base with the events `eventsAt20 … k` under `phi20 … k`. -/
theorem exists_coupled20 (pub : Public) (proof : ByteArray) (messages : Array Message)
    (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
    (block₂ : Fin Key20.level₂.shape.blocks → Coord)
    (block₃ : Fin Key20.level₃.shape.blocks → Coord)
    (block₄ : Fin Key20.level₄.shape.blocks → Coord)
    (block₅ : Fin Key20.level₅.shape.blocks → Coord) (blockL : Fin Key20.terminal.blocks → Coord)
    (fs : Sponge.Base L B C → Sponge.Y S C) (first : Key20.Wire E) {total : ℕ}
    (inside : 473 ≤ total) :
    ∃ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total,
      473 ≤ point.rest.length ∧
        ∀ k < 473, ∀ reply, point.reply k = some reply → ∀ base : Sponge.Base L B C,
          base.events = eventsAt20 pub proof messages k →
          ∃ y, fs base = phi20 block block₁ block₂ block₃ block₄ block₅ blockL k (reply, y) := by
  classical
  have onto : ∀ k (target : Sponge.Y S C), ∃ x : (Option Coord → CodeSlot E) × Sponge.Y S C,
      phi20 block block₁ block₂ block₃ block₄ block₅ blockL k x = target := by
    intro k target
    by_contra none_
    have empty : IsEmpty {x : (Option Coord → CodeSlot E) × Sponge.Y S C //
        phi20 block block₁ block₂ block₃ block₄ block₅ blockL k x = target} :=
      ⟨fun x => none_ ⟨x.1, x.2⟩⟩
    have count := card_phi20 (E := E) block block₁ block₂ block₃ block₄ block₅ blockL k target
    rw [Nat.card_of_isEmpty] at count
    exact Nat.ne_of_lt Fintype.card_pos count
  choose pick picked using onto
  let answerAt : ℕ → (Option Coord → CodeSlot E) := fun k =>
    if found : ∃ base : Sponge.Base L B C, base.events = eventsAt20 pub proof messages k then
      (pick k (fs found.choose)).1
    else Classical.arbitrary _
  refine ⟨⟨first, (List.range total).map fun k => (answerAt k, first), ?_⟩, ?_, ?_⟩
  · rw [List.length_map, List.length_range]
  · show 473 ≤ ((List.range total).map fun k => (answerAt k, first)).length
    rw [List.length_map, List.length_range]
    exact inside
  · intro k below reply at_ base events
    have here : k < total := Nat.lt_of_lt_of_le below inside
    have value : reply = answerAt k := by
      have listed : (((List.range total).map fun k => (answerAt k, first))[k]?).map Prod.fst =
          some reply := at_
      rw [List.getElem?_map, List.getElem?_range here] at listed
      exact (Option.some.inj listed).symm
    have found : ∃ base : Sponge.Base L B C, base.events = eventsAt20 pub proof messages k :=
      ⟨base, events⟩
    have same : found.choose = base :=
      Sponge.Base.events_injective (found.choose_spec.trans events.symm)
    refine ⟨(pick k (fs base)).2, ?_⟩
    rw [value]
    show fs base = phi20 block block₁ block₂ block₃ block₄ block₅ blockL k
      ((if found : ∃ base : Sponge.Base L B C,
          base.events = eventsAt20 pub proof messages k then
        (pick k (fs found.choose)).1 else Classical.arbitrary _), (pick k (fs base)).2)
    rw [dite_eq_left found, same]
    exact (picked k (fs base)).symm

end NightstreamFPrime.Spec.Compression.Akita.Replay
