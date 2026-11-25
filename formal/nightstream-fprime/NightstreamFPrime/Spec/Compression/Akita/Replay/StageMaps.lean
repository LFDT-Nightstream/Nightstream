import NightstreamFPrime.Spec.Compression.Akita.Replay.SpongeModel
import NightstreamFPrime.Spec.Compression.Akita.Replay.StageDrawsAgrees

/-! The stage maps of the opening of `key20.aks` on the answer type of the finite sponge (Akita
revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23, Section 8.2 `[KECCAK]`, (146) and
Lemma 10.47; `docs/akita-work/w21/design.md`, sections 2 and 5; `Replay/SpongeModel.lean`,
`Replay/StageCoupling.lean`, `Replay/StageDraws.lean`).

- `Sponge.base_of_worked`, `eventsAt20_head`, `base20`: the first event of `eventsAt20 … k` is the
  absorb of the nonce of the work, so a work that passes on the answers of a table gives a base
  with the events `eventsAt20 … k`.
- `rootGroup20`, `foldDraw20_root`, `drawEvents20_zero`, `drawEvents20_root`: the squeezes before
  the group `g` of a fold draw are the squeezes of the sub-key of the draw `g`.
- `drawBytes20`, `poolBytes20_le`, `fieldDraw20_le`, `foldBytes20_le`: each draw of a stage
  squeezes at most 5608 bytes.
- `draws20_stage`, `draws20_unique`, `stage20_cases`: each stage `k < 473` is the stage 0, a fold
  stage, or the stage of exactly one draw of `draws20`.
- `FieldAt`, `WeightsAt`, `FoldAt`, `RootAt`: the reading of the model on the answer `reply` is the
  reading of the code on the answers of a table whose entry at a base is `φ (reply, y)`.
  `exists_drawMap`, `exists_foldMap`, `exists_rootMap`: the lifts of the couplings of
  `Replay/StageCoupling.lean` have these readings and fibres of the size `|Reply|`.
- `Coupled20 … k φ` is the reading of the stage `k`. `phi20 … k` is the stage map of the stage
  `k`: `card_phi20` gives its fibres and `phi20_coupled` its reading. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (view)
open Bytes

open Replayer

section Base

variable {L B C S : ℕ}

/-- The base of a query whose first event is not a squeeze has the events of the query. -/
theorem Sponge.events_of_split {events : List Duplex.Event} {count : ℕ} {base : Sponge.Base L B C}
    {sub : Sponge.Sub S C} (head : ∀ c rest, events ≠ .squeeze c :: rest)
    (found : Sponge.split events count = some (base, sub)) : base.events = events := by
  have parts := (Sponge.split_eq_some.mp found).1
  unfold Sponge.Sub.events at parts
  cases counts : sub.counts with
  | nil =>
      rw [counts] at parts
      exact parts.symm
  | cons c rest =>
      rw [counts] at parts
      exact (head c _ parts).elim

/-- A work check with one bit or more that passes on the answer for events whose first event is
not a squeeze gives a base with these events. -/
theorem Sponge.base_of_worked (fs : Sponge.Base L B C → Sponge.Y S C)
    {events : List Duplex.Event} (head : ∀ c rest, events ≠ .squeeze c :: rest) {bits : ℕ}
    (positive : 1 ≤ bits) (passed : worked (Sponge.answerOf fs events 8) bits = true) :
    ∃ base : Sponge.Base L B C, base.events = events := by
  obtain ⟨base, sub, found⟩ := Sponge.split_of_worked fs positive passed
  exact ⟨base, Sponge.events_of_split head found⟩

variable {pub : Public} {proof : ByteArray} {messages : Array Message}

/-- The first event of `eventsAt20 … k` is the absorb of the nonce of the work, for a public input
with the work bits of the model and a run with no stop up to and with the work of the stage. -/
theorem eventsAt20_head (bits : Bits20 pub) {k : ℕ} (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1))
      (channelStart pub)).1 = .ok ())
    {answer : List Duplex.Event → ℕ → ByteArray} (ideal : pub.sponge.ideal = some answer)
    (c : ℕ) (rest : List Duplex.Event) :
    eventsAt20 pub proof messages k ≠ .squeeze c :: rest := by
  obtain ⟨site, level, at_⟩ := bits k inside
  obtain ⟨m, -, events, -, -⟩ := work_squeeze20 at_ (bits20_ne_zero inside) ran ideal
  rw [events]
  intro same
  cases (List.cons.inj same).1

/-- The base of the stage `k`: for a public input with the work bits of the model, a run with no
stop up to and with the work of the stage and the sponge of the table `fs`, a base has the events
`eventsAt20 … k`. -/
theorem base20 (fs : Sponge.Base L B C → Sponge.Y S C) (bits : Bits20 pub) {k : ℕ}
    (inside : k < 473)
    (ran : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k + 1))
      (channelStart pub)).1 = .ok ())
    (ideal : pub.sponge.ideal = some (Sponge.answerOf fs)) :
    ∃ base : Sponge.Base L B C, base.events = eventsAt20 pub proof messages k :=
  Sponge.base_of_worked fs (eventsAt20_head bits inside ran ideal)
    (Nat.one_le_iff_ne_zero.mpr (bits20_ne_zero inside)) (work20_answer bits inside ran ideal)

end Base

section Keys

/-- The group `g` of the code of the root of `key20.aks`. -/
abbrev rootGroup20 (g : Fin 3) : FoldGroup :=
  ⟨(Key20.root.shape g).blocks, ![1024, 512, 1024] g, Key20.root.carrierDegree g, 31,
    Key20.rootTwos g, false⟩

/-- The fold draw of the stage 1 has the three groups of the root. -/
theorem foldDraw20_root : ∀ g : Fin 3, foldDraw20 1 g = some (0, rootGroup20 g) := by
  decide +kernel

/-- No squeeze is before the first group of a fold draw. -/
theorem drawEvents20_zero (k : ℕ) : drawEvents20 k 0 = [] := by
  show ((sk20.take (workIndices20.getD k 0 + 1)).drop (workIndices20.getD k 0 + 1)).reverse.map
    _ = []
  rw [slice_nil]
  rfl

theorem drawCounts20_root : ∀ g : Fin 3,
    (((sk20.take (workIndices20.getD 1 0 + 1 + g)).drop (workIndices20.getD 1 0 + 1)).reverse.map
      fun x => x.1.drawBytes) = Sponge.before (fun i => (rootGroup20 i).messageBytes) g := by
  decide +kernel

/-- The squeezes before the group `g` of the fold draw of the root are the squeezes of the
sub-key of the draw `g` of a stage with the three messages of the root. -/
theorem drawEvents20_root (g : Fin 3) :
    drawEvents20 1 g =
      (Sponge.before (fun i => (rootGroup20 i).messageBytes) g).map Duplex.Event.squeeze := by
  rw [← drawCounts20_root g, List.map_map]
  rfl

/-- The largest count of the bytes that a draw of the opening of `key20.aks` squeezes. -/
def drawBytes20 : ℕ := 5608

/-- The pool of a field draw of at most 22 values has at most `drawBytes20` bytes. -/
theorem poolBytes20_le : ∀ n : Fin 23, poolBytes (2 * n) ≤ drawBytes20 := by decide +kernel

/-- The field draw of the stage `k` has at most 22 values. -/
def countFits20 (k : ℕ) : Bool :=
  match fieldDraw20 k with
  | some (_, count) => decide (count ≤ 22)
  | none => true

theorem skeleton20_counts : (List.range 473).all countFits20 = true := by decide +kernel

theorem fieldDraw20_le {k : ℕ} (inside : k < 473) {key : Key} {count : ℕ}
    (draw : fieldDraw20 k = some (key, count)) : count ≤ 22 := by
  have fit := List.all_eq_true.mp skeleton20_counts k (List.mem_range.mpr inside)
  unfold countFits20 at fit
  rw [draw] at fit
  have small : decide (count ≤ 22) = true := fit
  exact of_decide_eq_true small

/-- The message of each fold group of the opening of `key20.aks` has at most `drawBytes20`
bytes. -/
theorem foldBytes20_le : (∀ g, (rootGroup20 g).messageBytes ≤ drawBytes20) ∧
    FoldGroup.messageBytes ⟨28, 1024, 64, 31, 10, false⟩ ≤ drawBytes20 ∧
    FoldGroup.messageBytes ⟨21, 128, 128, 31, 0, false⟩ ≤ drawBytes20 ∧
    FoldGroup.messageBytes ⟨11, 128, 128, 31, 0, true⟩ ≤ drawBytes20 ∧
    FoldGroup.messageBytes ⟨7, 64, 64, 31, 11, true⟩ ≤ drawBytes20 ∧
    FoldGroup.messageBytes ⟨5, 64, 64, 31, 11, true⟩ ≤ drawBytes20 := by decide +kernel

/-- Each work bit count of the opening of `key20.aks` is at most 64. -/
theorem bits20_le_64 (k : ℕ) : bits20.flatten.getD k 0 ≤ 64 := by
  rcases Nat.lt_or_ge k bits20.flatten.length with below | above
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem below]
    exact (bits20_le_word _ (List.getElem_mem below)).trans (by decide)
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none above]
    exact Nat.zero_le _

end Keys

section Stages

/-- The path of each level has one draw for each stage between its fold stage and the fold stage
of the next level. -/
theorem draws20_length : ∀ ℓ : Fin 6, (draws20 ℓ).length = bound20 ℓ - first20 ℓ := by
  decide +kernel

theorem ranges20 : ∀ ℓ ℓ' : Fin 6, ℓ ≠ ℓ' →
    bound20 ℓ ≤ first20 ℓ' ∨ bound20 ℓ' ≤ first20 ℓ := by decide

theorem special20 : ∀ ℓ : Fin 6, ∀ s ∈ 0 :: starts20, ¬(first20 ℓ ≤ s ∧ s < bound20 ℓ) := by
  decide

/-- The stage of a draw of `draws20` is a stage, the draw has at most 22 values, and the stage is
not the stage 0 and not a fold stage. -/
theorem draws20_stage (ℓ : Fin 6) {j : ℕ} {d : Draw} (at_ : (draws20 ℓ)[j]? = some d) :
    first20 ℓ + j < 473 ∧ d.count ≤ 22 ∧ ∀ k, k = first20 ℓ + j → k ∉ 0 :: starts20 := by
  obtain ⟨draw, -, -, bounded⟩ := draws20_checked ℓ at_
  have inside : first20 ℓ + j < 473 := Nat.lt_trans bounded (bound20_last ℓ).1
  refine ⟨inside, fieldDraw20_le inside draw, fun k same member => ?_⟩
  subst same
  exact special20 ℓ _ member ⟨Nat.le_add_right _ _, bounded⟩

/-- One stage has one draw of `draws20`. -/
theorem draws20_unique {ℓ ℓ' : Fin 6} {j j' : ℕ} {d d' : Draw}
    (at_ : (draws20 ℓ)[j]? = some d) (at' : (draws20 ℓ')[j']? = some d')
    (same : first20 ℓ + j = first20 ℓ' + j') : d = d' := by
  have bounded := (draws20_checked ℓ at_).2.2.2
  have bounded' := (draws20_checked ℓ' at').2.2.2
  have level : ℓ = ℓ' := by
    by_contra differ
    rcases ranges20 ℓ ℓ' differ with le | le <;> omega
  subst level
  have index : j = j' := by omega
  subst index
  exact Option.some.inj (at_.symm.trans at')

/-- Each stage `k < 473` is the stage 0, a fold stage, or the stage of a draw of `draws20`. -/
theorem stage20_cases {k : ℕ} (inside : k < 473) :
    k = 0 ∨ k = 1 ∨ k = 51 ∨ k = 120 ∨ k = 203 ∨ k = 284 ∨ k = 378 ∨ k = 472 ∨
      ∃ (ℓ : Fin 6) (j : ℕ) (d : Draw), (draws20 ℓ)[j]? = some d ∧ k = first20 ℓ + j := by
  have covered : ∀ ℓ : Fin 6, first20 ℓ ≤ k → k < bound20 ℓ →
      ∃ (ℓ : Fin 6) (j : ℕ) (d : Draw), (draws20 ℓ)[j]? = some d ∧ k = first20 ℓ + j := by
    intro ℓ low high
    have below : k - first20 ℓ < (draws20 ℓ).length := by
      rw [draws20_length]
      omega
    exact ⟨ℓ, k - first20 ℓ, (draws20 ℓ)[k - first20 ℓ], List.getElem?_eq_getElem below,
      by omega⟩
  have split : k = 0 ∨ k = 1 ∨ k = 51 ∨ k = 120 ∨ k = 203 ∨ k = 284 ∨ k = 378 ∨ k = 472 ∨
      (2 ≤ k ∧ k < 51) ∨ (52 ≤ k ∧ k < 120) ∨ (121 ≤ k ∧ k < 203) ∨ (204 ≤ k ∧ k < 284) ∨
      (285 ≤ k ∧ k < 378) ∨ (379 ≤ k ∧ k < 472) := by omega
  rcases split with h | h | h | h | h | h | h | h | h | h | h | h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr (Or.inl h)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h)))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h)))))))
  all_goals refine Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ?_)))))))
  · exact covered 0 h.1 h.2
  · exact covered 1 h.1 h.2
  · exact covered 2 h.1 h.2
  · exact covered 3 h.1 h.2
  · exact covered 4 h.1 h.2
  · exact covered 5 h.1 h.2

end Stages

section Fields

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

theorem listFn_into (out : Array Ext) {count i : ℕ} (small : i < out.size) (lt : i < count) :
    listFn count (out.toList.map (Ext.into (E := E))) ⟨i, lt⟩ = (out[i]).into := by
  show (out.toList.map (Ext.into (E := E))).getD i 0 = _
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList,
    Array.getElem?_eq_getElem small]
  rfl

/-- Two outputs of `decodePool` with the same elements of `E` are equal. -/
theorem decodePool_ext {bytes bytes' : ByteArray} {count : ℕ} {out out' : Array Ext}
    (read : decodePool bytes count = some out) (read' : decodePool bytes' count = some out')
    (same : ∀ i (small : i < out.size) (small' : i < out'.size),
      ((out[i]).into : E) = (out'[i]).into) : out = out' := by
  obtain ⟨size, canonical⟩ := decodePool_canonical read
  obtain ⟨size', canonical'⟩ := decodePool_canonical read'
  refine Array.ext (size.trans size'.symm) fun i small small' => ?_
  exact into_injective (canonical _ (Array.getElem_mem small)).1
    (canonical _ (Array.getElem_mem small)).2 (canonical' _ (Array.getElem_mem small')).1
    (canonical' _ (Array.getElem_mem small')).2 (same i small small')

end Fields

/-- The pairs with one second component are as many as the first components. -/
theorem card_second {R Y : Type} [Fintype R] (y : Y) :
    Nat.card {x : R × Y // x.2 = y} = Fintype.card R :=
  (Nat.card_congr
    { toFun := fun x => x.1.1
      invFun := fun r => ⟨(r, y), rfl⟩
      left_inv := fun x => Subtype.ext (Prod.ext rfl x.2.symm)
      right_inv := fun _ => rfl }).trans Nat.card_eq_fintype_card

section Lifts

variable {E : Type} {Coord : Type} {S C : ℕ}

local notation "Reply" => (Option Coord → CodeSlot E)

/-- The reading of a stage with a field draw `d`, for a table whose entry at a base is
`φ (reply, y)`: when the work of `bits` bits passes on the answer for the events of the base and
`decodePool` gives values on the answer for the squeeze of the work, the work slot of `reply` has
`bits` zero bits and the model reads the challenge `d.read` of the values. -/
def FieldAt [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] [Fintype Coord] [DecidableEq Coord]
    (bits : ℕ) (d : Draw) (φ : Reply × Sponge.Y S C → Sponge.Y S C) : Prop :=
  ∀ {L B : ℕ} (fs : Sponge.Base L B C → Sponge.Y S C) (base : Sponge.Base L B C)
    (reply : Reply) (y : Sponge.Y S C), fs base = φ (reply, y) →
    worked (Sponge.answerOf fs base.events 8) bits = true →
    ∀ values, decodePool (Sponge.answerOf fs (.squeeze 8 :: base.events)
        (poolBytes (2 * d.count))) d.count = some values →
      reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
        ∀ (finite : Fintype (d.Space E)) (nonempty : Nonempty (d.Space E)),
          @Extract.sample (Coord → CodeSlot E) _ (d.Space E) finite nonempty (view reply) =
            d.read E values

/-- The reading of the stage 0: the reading of `FieldAt` for the three values of
`row_coefficients`, in the form of the weights of the batching root. -/
def WeightsAt [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] [Fintype Coord]
    [DecidableEq Coord] (bits : ℕ) (φ : Reply × Sponge.Y S C → Sponge.Y S C) : Prop :=
  ∀ {L B : ℕ} (fs : Sponge.Base L B C → Sponge.Y S C) (base : Sponge.Base L B C)
    (reply : Reply) (y : Sponge.Y S C), fs base = φ (reply, y) →
    worked (Sponge.answerOf fs base.events 8) bits = true →
    ∀ values, decodePool (Sponge.answerOf fs (.squeeze 8 :: base.events) (poolBytes (2 * 3))) 3 =
        some values →
      reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
        ∀ finite : Fintype (Fin 3 → E),
          @Extract.sample (Coord → CodeSlot E) _ (Fin 3 → E) finite _ (view reply) =
            fun g : Fin 3 => ((values.getD g Ext.zero).into : E)

/-- The reading of a fold stage with one group `G`: when the work passes and the sampler of the
code gives the challenges `cs` on the answer for the squeeze of the work, the work slot of `reply`
has `bits` zero bits and the decoder of the model gives `cs` at the slots `block`. -/
def FoldAt [Fintype E] (bits : ℕ) (G : FoldGroup) [NeZero G.degree]
    (block : Fin G.coordinates → Coord) (decode : CodeSlot E → Option (Rq G.degree))
    (φ : Reply × Sponge.Y S C → Sponge.Y S C) : Prop :=
  ∀ {L B : ℕ} (fs : Sponge.Base L B C → Sponge.Y S C) (base : Sponge.Base L B C)
    (reply : Reply) (y : Sponge.Y S C), fs base = φ (reply, y) →
    worked (Sponge.answerOf fs base.events 8) bits = true →
    ∀ cs, sample G (Sponge.answerOf fs (.squeeze 8 :: base.events) G.messageBytes) = some cs →
      reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
        ∀ i : Fin G.coordinates, decode (reply (some (block i))) =
          (cs[(i : ℕ)]?).map (Sparse.ring G.degree)

/-- The reading of the fold stage of the root: when the work passes and the sampler of the code
gives the challenges `cs g` of each group on the answer for the squeezes of the groups before `g`
and of the work, the work slot of `reply` has `bits` zero bits and the decoders of the model give
`cs`. -/
def RootAt [Fintype E] (bits : ℕ) (block : Key20.root.Block → Coord)
    (φ : Reply × Sponge.Y S C → Sponge.Y S C) : Prop :=
  ∀ {L B : ℕ} (fs : Sponge.Base L B C → Sponge.Y S C) (base : Sponge.Base L B C)
    (reply : Reply) (y : Sponge.Y S C), fs base = φ (reply, y) →
    worked (Sponge.answerOf fs base.events 8) bits = true →
    ∀ cs : Fin 3 → List Sparse,
      (∀ g : Fin 3, sample (rootGroup20 g) (Sponge.answerOf fs
        (drawEvents20 1 g ++ .squeeze 8 :: base.events) (rootGroup20 g).messageBytes) =
          some (cs g)) →
      reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
        ∀ (g : Fin 3) (i : Fin (Key20.root.shape g).blocks),
          Key20.rootDecode E g (reply (some (block ⟨g, i⟩))) =
            ((cs g)[(i : ℕ)]?).map (Sparse.ring (Key20.root.carrierDegree g))

/-- The lift of the coupling `stage_field` to the answers of a base. -/
theorem exists_fieldMap [Fintype E] [Nonempty E] [Fintype Coord] [DecidableEq Coord] {T : Type}
    [Fintype T] [Nonempty T] {bits : ℕ} (le : bits ≤ 64) (count : ℕ)
    (size : Fintype.card T = fieldModulus ^ (2 * count))
    (fair : ∃ decode : (Coord → CodeSlot E) → T, Extract.Fair decode) (δ : Array Ext → T)
    (injective : ∀ bytes bytes' out out', decodePool bytes count = some out →
      decodePool bytes' count = some out' → δ out = δ out' → out = out')
    (short : 1 ≤ S) (le8 : 8 ≤ C) (fits : poolBytes (2 * count) ≤ C) :
    ∃ φ : Reply × Sponge.Y S C → Sponge.Y S C,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card Reply) ∧
      ∀ {L B : ℕ} (fs : Sponge.Base L B C → Sponge.Y S C) (base : Sponge.Base L B C)
        (reply : Reply) (y : Sponge.Y S C), fs base = φ (reply, y) →
        worked (Sponge.answerOf fs base.events 8) bits = true →
        ∀ out, decodePool (Sponge.answerOf fs (.squeeze 8 :: base.events)
            (poolBytes (2 * count))) count = some out →
          reply none ∈ Extract.workZeros (CodeSlot E) bits ∧
            Extract.sample (Coord → CodeSlot E) T (view reply) = δ out := by
  obtain ⟨φ₀, fibres, reads⟩ :=
    stage_field (Coord := Coord) (E := E) le count size fair δ injective
  refine ⟨Sponge.oneLift short le8 fits φ₀, Sponge.card_oneLift short le8 fits φ₀ fibres, ?_⟩
  intro L B fs base reply y entry passed out decoded
  obtain ⟨work, draw⟩ := Sponge.answerOf_oneLift short le8 fits φ₀ fs base reply y entry
  rw [work] at passed
  rw [draw] at decoded
  exact reads reply _ out passed decoded

/-- A map with the fibres of the size `|Reply|` and the reading `FieldAt` of the draw `d`: the
lift of `stage_field` for the challenge space and the reader of `d`. -/
theorem exists_drawMap [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] [Fintype Coord]
    [DecidableEq Coord] (d : Draw) {bits : ℕ} (le : bits ≤ 64)
    (few : d.count ≤ Fintype.card Coord) (inhabited : Nonempty Coord) (short : 1 ≤ S)
    (le8 : 8 ≤ C) (fits : poolBytes (2 * d.count) ≤ C) :
    ∃ φ : Reply × Sponge.Y S C → Sponge.Y S C,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card Reply) ∧ FieldAt bits d φ := by
  cases d with
  | one key index =>
      obtain ⟨φ, fibres, reads⟩ := exists_fieldMap (E := E) (Coord := Coord) (S := S) (C := C)
        (T := E) le 1 Key20.card_field Extract.exists_fair_field (fun out => (out[0]!).into)
        (fun _ _ out out' read read' same => decodePool_ext (E := E) read read'
          fun i small small' => by
            have size := (decodePool_canonical read).1
            have first : ((out[0]!).into : E) = (out'[0]!).into := same
            obtain rfl : i = 0 := by omega
            rw [getElem!_pos out 0 small, getElem!_pos out' 0 small'] at first
            exact first) short le8 fits
      refine ⟨φ, fibres, fun fs base reply y entry passed values decoded => ?_⟩
      obtain ⟨zero, sampled⟩ := reads fs base reply y entry passed values decoded
      refine ⟨zero, fun finite nonempty => ?_⟩
      obtain rfl : finite = (inferInstance : Fintype E) := Subsingleton.elim _ _
      exact sampled
  | many key count =>
      obtain ⟨φ, fibres, reads⟩ := exists_fieldMap (E := E) (Coord := Coord) (S := S) (C := C)
        (T := Fin count → E) le count
        (by rw [Fintype.card_fun, Fintype.card_fin, Key20.card_field, ← pow_mul])
        (Extract.exists_fair_fields few) (fun out => listFn count (out.toList.map Ext.into))
        (fun _ _ out out' read read' same => decodePool_ext (E := E) read read'
          fun i small small' => by
            have lt : i < count := Nat.lt_of_lt_of_eq small (decodePool_canonical read).1
            have entry : listFn count (out.toList.map (Ext.into (E := E))) ⟨i, lt⟩ =
                listFn count (out'.toList.map (Ext.into (E := E))) ⟨i, lt⟩ :=
              congrFun same ⟨i, lt⟩
            rw [listFn_into out small lt, listFn_into out' small' lt] at entry
            exact entry) short le8 fits
      refine ⟨φ, fibres, fun fs base reply y entry passed values decoded => ?_⟩
      obtain ⟨zero, sampled⟩ := reads fs base reply y entry passed values decoded
      refine ⟨zero, fun finite nonempty => ?_⟩
      obtain rfl : finite = (inferInstance : Fintype (Fin count → E)) := Subsingleton.elim _ _
      exact sampled

/-- The reading of a draw of the first three values of a key is the reading of the weights of the
batching root. -/
theorem FieldAt.weights [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] [Fintype Coord]
    [DecidableEq Coord] {bits : ℕ} {key : Key} {φ : Reply × Sponge.Y S C → Sponge.Y S C}
    (field : FieldAt bits (.many key 3) φ) : WeightsAt bits φ := by
  intro L B fs base reply y entry passed values decoded
  obtain ⟨zero, sampled⟩ := field fs base reply y entry passed values decoded
  refine ⟨zero, fun finite => (sampled finite (inferInstanceAs (Nonempty (Fin 3 → E)))).trans (funext fun g => ?_)⟩
  have small : (g : ℕ) < values.size := by
    rw [decodePool_size decoded]
    exact g.isLt
  refine (listFn_into (E := E) values small g.isLt).trans ?_
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem small]
  rfl

/-- The lift of a `FoldCoupling` to the answers of a base. -/
theorem exists_foldMap [Fintype E] [Fintype Coord] [DecidableEq Coord] {G : FoldGroup}
    [NeZero G.degree] {bits : ℕ} {block : Fin G.coordinates → Coord}
    {decode : CodeSlot E → Option (Rq G.degree)} (coupling : FoldCoupling E G bits block decode)
    (short : 1 ≤ S) (le8 : 8 ≤ C) (fits : G.messageBytes ≤ C) :
    ∃ φ : Reply × Sponge.Y S C → Sponge.Y S C,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card Reply) ∧ FoldAt bits G block decode φ := by
  obtain ⟨φ₀, fibres, reads⟩ := coupling
  refine ⟨Sponge.oneLift short le8 fits φ₀, Sponge.card_oneLift short le8 fits φ₀ fibres, ?_⟩
  intro L B fs base reply y entry passed cs sampled
  obtain ⟨work, draw⟩ := Sponge.answerOf_oneLift short le8 fits φ₀ fs base reply y entry
  rw [work] at passed
  rw [draw] at sampled
  exact reads reply _ cs passed sampled

/-- The lift of the coupling `stage_folds_root` of the three groups of the root to the answers of
a base. -/
theorem exists_rootMap [Fintype E] [Nonempty E] [Fintype Coord] [DecidableEq Coord] {bits : ℕ}
    (le : bits ≤ 64) {block : Key20.root.Block → Coord} (injective : Function.Injective block)
    (short : 3 ≤ S) (le8 : 8 ≤ C) (fits : ∀ g, (rootGroup20 g).messageBytes ≤ C) :
    ∃ φ : Reply × Sponge.Y S C → Sponge.Y S C,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card Reply) ∧ RootAt bits block φ := by
  obtain ⟨φ₀, fibres, reads⟩ := stage_folds_root (E := E)
    (fun g => (Key20.root.shape g).blocks) ![1024, 512, 1024] le (fun g i => block ⟨g, i⟩)
    (fun x x' same => injective same)
  refine ⟨Sponge.stageLift (fun g => (rootGroup20 g).messageBytes) short le8 fits φ₀,
    Sponge.card_stageLift _ short le8 fits φ₀ fibres, ?_⟩
  intro L B fs base reply y entry passed cs sampled
  obtain ⟨work, draws⟩ := Sponge.answerOf_stageLift
    (fun g => (rootGroup20 g).messageBytes) short le8 fits φ₀ fs base reply y entry
  rw [work] at passed
  refine reads reply _ cs passed fun g => ?_
  have drawn := sampled g
  rw [drawEvents20_root g, draws g] at drawn
  exact drawn

end Lifts

section Maps

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {S C : ℕ}
  (block : Key20.root.Block → Coord) (block₁ : Fin Key20.level₁.shape.blocks → Coord)
  (block₂ : Fin Key20.level₂.shape.blocks → Coord) (block₃ : Fin Key20.level₃.shape.blocks → Coord)
  (block₄ : Fin Key20.level₄.shape.blocks → Coord) (block₅ : Fin Key20.level₅.shape.blocks → Coord)
  (blockL : Fin Key20.terminal.blocks → Coord)

local notation "Reply" => (Option Coord → CodeSlot E)

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

local notation "bitsAt" k => bits20.flatten.getD k 0

/-- The reading of the stage `k` of the opening of `key20.aks` under the map `φ`: the weights of
the batching root at the stage 0, the draw of `draws20` at its stage, and the fold challenges of
the root, of each level and of the terminal at the fold stages. -/
structure Coupled20 (k : ℕ) (φ : Reply × Sponge.Y S C → Sponge.Y S C) : Prop where
  weights : k = 0 → WeightsAt (bitsAt k) φ
  field : ∀ (ℓ : Fin 6) (j : ℕ) (d : Draw), (draws20 ℓ)[j]? = some d → k = first20 ℓ + j →
    FieldAt (bitsAt k) d φ
  root : k = 1 → RootAt (bitsAt k) block φ
  level₁ : k = 51 → FoldAt (bitsAt k) ⟨28, 1024, 64, 31, 10, false⟩ block₁ code₁ φ
  level₂ : k = 120 → FoldAt (bitsAt k) ⟨21, 128, 128, 31, 0, false⟩ block₂ code₂ φ
  level₃ : k = 203 → FoldAt (bitsAt k) ⟨11, 128, 128, 31, 0, true⟩ block₃ code₃ φ
  level₄ : k = 284 → FoldAt (bitsAt k) ⟨7, 64, 64, 31, 11, true⟩ block₄ code₄ φ
  level₅ : k = 378 → FoldAt (bitsAt k) ⟨7, 64, 64, 31, 11, true⟩ block₅ code₅ φ
  terminal : k = 472 → FoldAt (bitsAt k) ⟨5, 64, 64, 31, 11, true⟩ blockL codeL φ

local macro "other" : term => `(fun h => absurd h (by decide))

/-- Each stage has a map with the fibres of the size `|Reply|`. When each draw fits a string of
the table, the sub-keys have three squeezes or more, the block maps are injective and the answer
has 22 coordinates or more, the map has the reading `Coupled20` of the stage. -/
theorem exists_stage20 (k : ℕ) :
    ∃ φ : Reply × Sponge.Y S C → Sponge.Y S C,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card Reply) ∧
      (drawBytes20 ≤ C → 3 ≤ S → Function.Injective block → Function.Injective block₁ →
        Function.Injective block₂ → Function.Injective block₃ → Function.Injective block₄ →
        Function.Injective block₅ → Function.Injective blockL → 22 ≤ Fintype.card Coord →
        Coupled20 block block₁ block₂ block₃ block₄ block₅ blockL k φ) := by
  by_cases given : drawBytes20 ≤ C ∧ 3 ≤ S ∧ Function.Injective block ∧
      Function.Injective block₁ ∧ Function.Injective block₂ ∧ Function.Injective block₃ ∧
      Function.Injective block₄ ∧ Function.Injective block₅ ∧ Function.Injective blockL ∧
      22 ≤ Fintype.card Coord
  swap
  · exact ⟨Prod.snd, card_second, fun a b c c₁ c₂ c₃ c₄ c₅ cL s =>
      (given ⟨a, b, c, c₁, c₂, c₃, c₄, c₅, cL, s⟩).elim⟩
  obtain ⟨sizes, three, injective, injective₁, injective₂, injective₃, injective₄, injective₅,
    injectiveL, slots⟩ := given
  suffices found : ∃ φ : Reply × Sponge.Y S C → Sponge.Y S C,
      (∀ y, Nat.card {x // φ x = y} = Fintype.card Reply) ∧
        Coupled20 block block₁ block₂ block₃ block₄ block₅ blockL k φ by
    obtain ⟨φ, fibres, coupled⟩ := found
    exact ⟨φ, fibres, fun _ _ _ _ _ _ _ _ _ _ => coupled⟩
  have le := bits20_le_64 k
  have le8 : 8 ≤ C := Nat.le_trans (by decide) sizes
  have one : 1 ≤ S := Nat.le_trans (by decide) three
  have inhabited : Nonempty Coord := ⟨blockL ⟨0, by decide⟩⟩
  obtain ⟨bytes, bytes₁, bytes₂, bytes₃, bytes₄, bytesL⟩ := foldBytes20_le
  by_cases k0 : k = 0
  · subst k0
    obtain ⟨φ, fibres, field⟩ := exists_drawMap (E := E) (Coord := Coord) (S := S) (C := C)
      (.many ⟨0, "row_coefficients", none⟩ 3) le (Nat.le_trans (by decide) slots) inhabited one
      le8 (Nat.le_trans (poolBytes20_le 3) sizes)
    exact ⟨φ, fibres,
      { weights := fun _ => field.weights
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := other, level₁ := other, level₂ := other, level₃ := other, level₄ := other
        level₅ := other, terminal := other }⟩
  by_cases k1 : k = 1
  · subst k1
    obtain ⟨φ, fibres, root⟩ := exists_rootMap (E := E) (S := S) (C := C) le injective three le8
      fun g => Nat.le_trans (bytes g) sizes
    exact ⟨φ, fibres,
      { weights := other
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := fun _ => root, level₁ := other, level₂ := other, level₃ := other
        level₄ := other, level₅ := other, terminal := other }⟩
  by_cases k51 : k = 51
  · subst k51
    obtain ⟨φ, fibres, fold⟩ := exists_foldMap (S := S) (C := C)
      (stage_fold_open (E := E) 28 1024 (G := Key20.open64) Key20.open64_fits rfl (by decide) le
        block₁ injective₁) one le8 (Nat.le_trans bytes₁ sizes)
    exact ⟨φ, fibres,
      { weights := other
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := other, level₁ := fun _ => fold, level₂ := other, level₃ := other
        level₄ := other, level₅ := other, terminal := other }⟩
  by_cases k120 : k = 120
  · subst k120
    obtain ⟨φ, fibres, fold⟩ := exists_foldMap (S := S) (C := C)
      (stage_fold_open (E := E) 21 128 (G := Key20.open128) Key20.open128_fits rfl (by decide)
        le block₂ injective₂) one le8 (Nat.le_trans bytes₂ sizes)
    exact ⟨φ, fibres,
      { weights := other
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := other, level₁ := other, level₂ := fun _ => fold, level₃ := other
        level₄ := other, level₅ := other, terminal := other }⟩
  by_cases k203 : k = 203
  · subst k203
    obtain ⟨φ, fibres, fold⟩ := exists_foldMap (S := S) (C := C)
      (stage_fold_kept128 (E := E) 11 128 le block₃ injective₃) one le8
      (Nat.le_trans bytes₃ sizes)
    exact ⟨φ, fibres,
      { weights := other
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := other, level₁ := other, level₂ := other, level₃ := fun _ => fold
        level₄ := other, level₅ := other, terminal := other }⟩
  by_cases k284 : k = 284
  · subst k284
    obtain ⟨φ, fibres, fold⟩ := exists_foldMap (S := S) (C := C)
      (stage_fold_kept64 (E := E) 7 64 le block₄ injective₄) one le8 (Nat.le_trans bytes₄ sizes)
    exact ⟨φ, fibres,
      { weights := other
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := other, level₁ := other, level₂ := other, level₃ := other
        level₄ := fun _ => fold, level₅ := other, terminal := other }⟩
  by_cases k378 : k = 378
  · subst k378
    obtain ⟨φ, fibres, fold⟩ := exists_foldMap (S := S) (C := C)
      (stage_fold_kept64 (E := E) 7 64 le block₅ injective₅) one le8 (Nat.le_trans bytes₄ sizes)
    exact ⟨φ, fibres,
      { weights := other
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := other, level₁ := other, level₂ := other, level₃ := other, level₄ := other
        level₅ := fun _ => fold, terminal := other }⟩
  by_cases k472 : k = 472
  · subst k472
    obtain ⟨φ, fibres, fold⟩ := exists_foldMap (S := S) (C := C)
      (stage_fold_kept64 (E := E) 5 64 le blockL injectiveL) one le8 (Nat.le_trans bytesL sizes)
    exact ⟨φ, fibres,
      { weights := other
        field := fun ℓ j d at_ same => absurd (by decide) ((draws20_stage ℓ at_).2.2 _ same)
        root := other, level₁ := other, level₂ := other, level₃ := other, level₄ := other
        level₅ := other, terminal := fun _ => fold }⟩
  by_cases drawn : ∃ (ℓ : Fin 6) (j : ℕ) (d : Draw),
      (draws20 ℓ)[j]? = some d ∧ k = first20 ℓ + j
  · obtain ⟨ℓ₀, j₀, d₀, at₀, same₀⟩ := drawn
    obtain ⟨-, few, -⟩ := draws20_stage ℓ₀ at₀
    obtain ⟨φ, fibres, field⟩ := exists_drawMap (E := E) (Coord := Coord) (S := S) (C := C) d₀
      le (Nat.le_trans few slots) inhabited one le8
      (Nat.le_trans (poolBytes20_le ⟨d₀.count, Nat.lt_succ_of_le few⟩) sizes)
    exact ⟨φ, fibres,
      { weights := fun h => absurd h k0
        field := fun ℓ j d at_ same => by
          obtain rfl := draws20_unique at₀ at_ (same₀.symm.trans same)
          exact field
        root := fun h => absurd h k1, level₁ := fun h => absurd h k51
        level₂ := fun h => absurd h k120, level₃ := fun h => absurd h k203
        level₄ := fun h => absurd h k284, level₅ := fun h => absurd h k378
        terminal := fun h => absurd h k472 }⟩
  · exact ⟨Prod.snd, card_second,
      { weights := fun h => absurd h k0
        field := fun ℓ j d at_ same => (drawn ⟨ℓ, j, d, at_, same⟩).elim
        root := fun h => absurd h k1, level₁ := fun h => absurd h k51
        level₂ := fun h => absurd h k120, level₃ := fun h => absurd h k203
        level₄ := fun h => absurd h k284, level₅ := fun h => absurd h k378
        terminal := fun h => absurd h k472 }⟩

/-- The stage map of the stage `k` of the opening of `key20.aks` on the answers of a base of the
sponge: a map of `exists_stage20` for `k < 473`, and the second projection for the other `k`. -/
noncomputable def phi20 (k : ℕ) : Reply × Sponge.Y S C → Sponge.Y S C :=
  if k < 473 then
    Classical.choose (exists_stage20 (E := E) (S := S) (C := C) block block₁ block₂ block₃ block₄
      block₅ blockL k)
  else Prod.snd

/-- The stage map of a stage `k ≥ 473` is the second projection. -/
theorem phi20_last {k : ℕ} (outside : 473 ≤ k) :
    (phi20 block block₁ block₂ block₃ block₄ block₅ blockL k :
      Reply × Sponge.Y S C → Sponge.Y S C) = Prod.snd :=
  ite_eq_right (Nat.not_lt.mpr outside)

/-- Each fibre of a stage map has `|Reply|` elements. -/
theorem card_phi20 (k : ℕ) (y : Sponge.Y S C) :
    Nat.card {x : Reply × Sponge.Y S C //
        phi20 block block₁ block₂ block₃ block₄ block₅ blockL k x = y} =
      Fintype.card Reply := by
  by_cases inside : k < 473
  · have chosen : (phi20 block block₁ block₂ block₃ block₄ block₅ blockL k :
        Reply × Sponge.Y S C → Sponge.Y S C) = Classical.choose (exists_stage20 (E := E)
          (S := S) (C := C) block block₁ block₂ block₃ block₄ block₅ blockL k) := ite_eq_left inside
    rw [chosen]
    exact (Classical.choose_spec (exists_stage20 (E := E) (S := S) (C := C) block block₁ block₂
      block₃ block₄ block₅ blockL k)).1 y
  · rw [phi20_last block block₁ block₂ block₃ block₄ block₅ blockL (Nat.le_of_not_lt inside)]
    exact card_second y

variable {block block₁ block₂ block₃ block₄ block₅ blockL}

/-- The stage map of a stage `k < 473` has the reading `Coupled20` of the stage, when each draw
fits a string of the table (`drawBytes20 ≤ C`), the sub-keys have three squeezes or more, the
block maps are injective and the answer has 22 coordinates or more. -/
theorem phi20_coupled (sizes : drawBytes20 ≤ C) (three : 3 ≤ S)
    (injective : Function.Injective block) (injective₁ : Function.Injective block₁)
    (injective₂ : Function.Injective block₂) (injective₃ : Function.Injective block₃)
    (injective₄ : Function.Injective block₄) (injective₅ : Function.Injective block₅)
    (injectiveL : Function.Injective blockL) (slots : 22 ≤ Fintype.card Coord) {k : ℕ}
    (inside : k < 473) :
    Coupled20 block block₁ block₂ block₃ block₄ block₅ blockL k
      (phi20 block block₁ block₂ block₃ block₄ block₅ blockL k :
        Reply × Sponge.Y S C → Sponge.Y S C) := by
  have chosen : (phi20 block block₁ block₂ block₃ block₄ block₅ blockL k :
      Reply × Sponge.Y S C → Sponge.Y S C) = Classical.choose (exists_stage20 (E := E)
        (S := S) (C := C) block block₁ block₂ block₃ block₄ block₅ blockL k) := ite_eq_left inside
  rw [chosen]
  exact (Classical.choose_spec (exists_stage20 (E := E) (S := S) (C := C) block block₁ block₂
    block₃ block₄ block₅ blockL k)).2 sizes three injective injective₁ injective₂ injective₃
    injective₄ injective₅ injectiveL slots

end Maps

end NightstreamFPrime.Spec.Compression.Akita.Replay
