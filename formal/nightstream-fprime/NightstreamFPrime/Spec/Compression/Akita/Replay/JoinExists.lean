import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBlock

/-! Theorem A of the join for `key20.aks` (Akita revision `d50f700`; ePrint 2026/1983, revision of
2026-11-23, Section 10.3): each accepted proof has a
transcript of `tower20` that presents its bytes, and `Key20.Tower.accepts` accepts it.

- `Prefix.wire_all`, `lastWire_all`, `ofBlocks_at`, `Chained`, `Chained.getElem`: the wire messages
  and the answers of a transcript of blocks (`Extract.Prefix.ofBlocks`).
- `bits20_le`: each work bit count of the model is at most 131, so each stage has a work slot with
  these zero bits (`Key20.card_codeSlot`).
- `exists_presents20`: theorem A. The transcript is the stage 0 of the batching root with the
  weights of the code (`Extract.exists_weights`), the blocks `block20_zeroth` .. `block20_fifth`,
  and the fold stage of the terminal with the slots of `slots20L`; each wire message has the data
  `note20` and the nonce 0, and each answer has a work slot with the zero bits of its stage.
- `verifyProof_exists_accepts20`: an accepted proof has last blocks `Last20` of the terminal and a
  transcript that presents its bytes with them, and `Key20.Tower.accepts` of `tower20` accepts this
  transcript (`presents20_accepts`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix lastWire)

open Replayer

section Blocks

variable {Wire Reply : Type} {total : ℕ}

/-- A property of the first message and of each message of the stages holds for each wire
message of a transcript. -/
theorem Prefix.wire_all (point : Prefix Wire Reply total) (P : Wire → Prop) (first : P point.first)
    (each : ∀ x ∈ point.rest, P x.2) : ∀ k, P (point.wire k)
  | 0 => first
  | k + 1 => by
      show P ((point.rest[k]?.map Prod.snd).getD point.first)
      cases found : point.rest[k]? with
      | none => exact first
      | some x => exact each x (List.mem_of_getElem? found)

/-- A property of the first message and of each message of a list of stages holds for its last
wire message. -/
theorem lastWire_all (P : Wire → Prop) (first : Wire) (moves : List (Reply × Wire))
    (start : P first) (each : ∀ x ∈ moves, P x.2) : P (lastWire first moves) := by
  unfold lastWire
  cases found : moves.getLast? with
  | none => exact start
  | some x => exact each x (List.mem_of_getLast? found)

/-- The block `b` of a transcript of blocks after the blocks `before`: the stages from
`|before.flatten|` begin with `b`, and the wire message there is the last one of `before`. -/
theorem ofBlocks_at (first : Wire) (before : List (List (Reply × Wire))) (b : List (Reply × Wire))
    (after : List (List (Reply × Wire)))
    (short : ((before ++ b :: after).map List.length).sum ≤ total) :
    ((Prefix.ofBlocks first (before ++ b :: after) short).rest.drop
        before.flatten.length).take b.length = b ∧
      (Prefix.ofBlocks first (before ++ b :: after) short).wire before.flatten.length =
        lastWire first before.flatten := by
  have shape : (Prefix.ofBlocks first (before ++ b :: after) short).rest =
      before.flatten ++ (b ++ after.flatten) := by
    show (before ++ b :: after).flatten = _
    rw [List.flatten_append, List.flatten_cons]
  refine ⟨?_, Extract.Prefix.wire_length shape⟩
  rw [shape, List.drop_left, List.take_left]

/-- `P (start + i) x` holds for the entry `x` of index `i` of each block, from the start `start`
of the first block. -/
def Chained {α : Type} (P : ℕ → α → Prop) : ℕ → List (List α) → Prop
  | _, [] => True
  | start, b :: rest => (∀ i (inside : i < b.length), P (start + i) b[i]) ∧
      Chained P (start + b.length) rest

/-- For chained blocks, `P (start + k)` holds for the entry `k` of the flattened list. -/
theorem Chained.getElem {α : Type} {P : ℕ → α → Prop} : ∀ {start : ℕ} {blocks : List (List α)},
    Chained P start blocks → ∀ k (inside : k < blocks.flatten.length),
      P (start + k) blocks.flatten[k]
  | _, [], _, k, inside => absurd inside (by simp)
  | start, b :: rest, ⟨here, later⟩, k, inside => by
      simp only [List.flatten_cons] at inside ⊢
      by_cases front : k < b.length
      · rw [List.getElem_append_left front]
        exact here k front
      · rw [List.getElem_append_right (Nat.le_of_not_lt front)]
        have long : k - b.length < rest.flatten.length := by
          rw [List.length_append] at inside
          omega
        have entry := Chained.getElem later (k - b.length) long
        rwa [show start + b.length + (k - b.length) = start + k by omega] at entry

end Blocks

/-- Each work bit count of the stages of the model is at most 131. -/
theorem bits20_le (k : ℕ) : bits20.flatten.getD k 0 ≤ 131 := by
  rcases lt_or_ge k bits20.flatten.length with inside | outside
  · rw [List.getD_eq_getElem _ _ inside]
    exact (by decide +kernel : ∀ x ∈ bits20.flatten, x ≤ 131) _ (List.getElem_mem inside)
  · rw [List.getD_eq_default _ _ outside]
    omega

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord]
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem A of the join for `key20.aks`: for an
accepted proof of `layout20` and each last blocks `tLast`, `eLast` of the terminal, a transcript
of `tower20` with `total ≥ 473` stages presents the bytes (`Presents20`). The hypotheses are
those of `presents20_accepts` without `Last20`, injective coordinates of the fold challenges, and
an answer with 22 slots or more. -/
theorem exists_presents20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
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
      Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅ blockL
        point := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have work : ∀ k, ∃ slot : CodeSlot E,
      slot ∈ Extract.workZeros (CodeSlot E) (bits20.flatten.getD k 0) := fun k =>
    (Extract.exists_mem_workZeros_iff (CodeSlot E) _).mpr
      ((Nat.pow_le_pow_right (by decide) (bits20_le k)).trans Key20.card_codeSlot)
  choose W worked using work
  obtain ⟨restR, weighed⟩ := Extract.exists_weights (Coord := Coord) (Slot := CodeSlot E)
    (Extract.exists_fair_field_family (E := E) (Index := Fin 3)
      (by rw [Fintype.card_fin]; omega)) (weights20 pub proof messages E) (W 0)
  obtain ⟨sL, decodesL⟩ := slots20L (E := E) layout accepted terminalSetup decoded
  obtain ⟨page, paged⟩ : ∃ page : Key20.Wire E,
      page = ⟨note20 pub proof messages tLast eLast, fun _ => (fun _ => 0, 0), 0⟩ := ⟨_, rfl⟩
  have pageData : page.data = note20 pub proof messages tLast eLast := by rw [paged]
  have pageNonce : page.nonce = 0 := by rw [paged]
  have flat : ∀ blocks : List (List ((Option Coord → CodeSlot E) × Key20.Wire E)),
      (∀ l ∈ blocks, ∀ x ∈ l, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) →
      ∀ x ∈ blocks.flatten, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0 :=
    fun blocks each x member => by
      obtain ⟨l, listed, inl⟩ := List.mem_flatten.mp member
      exact each l listed x inl
  have lastData : ∀ blocks : List (List ((Option Coord → CodeSlot E) × Key20.Wire E)),
      (∀ l ∈ blocks, ∀ x ∈ l, x.2.data = note20 pub proof messages tLast eLast ∧ x.2.nonce = 0) →
      (lastWire page blocks.flatten).data = note20 pub proof messages tLast eLast :=
    fun blocks each => lastWire_all (fun wire => wire.data = note20 pub proof messages tLast eLast)
      page _ pageData fun x member => (flat blocks each x member).1
  obtain ⟨b0, length0, fields0, worked0, shows0⟩ := block20_zeroth
    (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof messages tLast
    eLast layout accepted terminalSetup decoded points values (expanded 0) rfl injective slots W
    (lastWire page [[(Extract.withWork (W 0) restR, page)]].flatten) (lastData _ (by
      simp only [List.mem_singleton, forall_eq]; exact ⟨pageData, pageNonce⟩))
  obtain ⟨b1, length1, fields1, worked1, shows1⟩ := block20_first
    (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof messages tLast
    eLast layout accepted terminalSetup decoded (expanded 1) rfl injective₁ slots W
    (lastWire page [[(Extract.withWork (W 0) restR, page)], b0].flatten) (lastData _ (by
      simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
      exact ⟨⟨pageData, pageNonce⟩, fields0⟩))
  obtain ⟨b2, length2, fields2, worked2, shows2⟩ := block20_second
    (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof messages tLast
    eLast layout accepted terminalSetup decoded (expanded 2) rfl injective₂ slots W
    (lastWire page [[(Extract.withWork (W 0) restR, page)], b0, b1].flatten) (lastData _ (by
      simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
      exact ⟨⟨pageData, pageNonce⟩, fields0, fields1⟩))
  obtain ⟨b3, length3, fields3, worked3, shows3⟩ := block20_third
    (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof messages tLast
    eLast layout accepted terminalSetup decoded (expanded 3) rfl injective₃ slots W
    (lastWire page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2].flatten) (lastData _ (by
      simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
      exact ⟨⟨pageData, pageNonce⟩, fields0, fields1, fields2⟩))
  obtain ⟨b4, length4, fields4, worked4, shows4⟩ := block20_fourth
    (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof messages tLast
    eLast layout accepted terminalSetup decoded (expanded 4) rfl injective₄ slots W
    (lastWire page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3].flatten) (lastData _ (by
      simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
      exact ⟨⟨pageData, pageNonce⟩, fields0, fields1, fields2, fields3⟩))
  obtain ⟨b5, length5, fields5, worked5, shows5⟩ := block20_fifth
    (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472) pub proof messages tLast
    eLast layout accepted terminalSetup decoded (expanded 5) rfl injective₅ slots W
    (lastWire page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4].flatten)
    (lastData _ (by
      simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
      exact ⟨⟨pageData, pageNonce⟩, fields0, fields1, fields2, fields3, fields4⟩))
  have short : ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
      [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]].map List.length).sum ≤ total := by
    simp only [List.map_cons, List.map_nil, List.length_singleton, length0, length1, length2,
      length3, length4, length5, List.sum_cons, List.sum_nil]
    omega
  refine ⟨Prefix.ofBlocks page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short, ?_⟩
  have fields := flat [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] (by
    simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨⟨pageData, pageNonce⟩, fields0, fields1, fields2, fields3, fields4, fields5, ⟨pageData, pageNonce⟩⟩)
  have lengthAll : ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
      [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]].flatten).length = 473 := by
    simp only [List.flatten_cons, List.flatten_nil, List.length_append, List.length_singleton,
      length0, length1, length2, length3, length4, length5, List.length_nil]
  have chained : Chained (fun i (x : (Option Coord → CodeSlot E) × Key20.Wire E) => x.1 none = W i)
      0 [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
        [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] := by
    refine ⟨fun i inside => ?_, fun i inside => ?_, fun i inside => ?_, fun i inside => ?_,
      fun i inside => ?_, fun i inside => ?_, fun i inside => ?_, fun i inside => ?_, trivial⟩
    · obtain rfl : i = 0 := by simpa using inside
      rfl
    · dsimp only
      rw [worked0 i inside]
      exact congrArg W (by simp only [List.length_singleton])
    · dsimp only
      rw [worked1 i inside]
      exact congrArg W (by simp only [List.length_singleton, length0])
    · dsimp only
      rw [worked2 i inside]
      exact congrArg W (by simp only [List.length_singleton, length0, length1])
    · dsimp only
      rw [worked3 i inside]
      exact congrArg W (by simp only [List.length_singleton, length0, length1, length2])
    · dsimp only
      rw [worked4 i inside]
      exact congrArg W (by
        simp only [List.length_singleton, length0, length1, length2, length3])
    · dsimp only
      rw [worked5 i inside]
      exact congrArg W (by
        simp only [List.length_singleton, length0, length1, length2, length3, length4])
    · obtain rfl : i = 0 := by simpa using inside
      exact congrArg W (by
        simp only [List.length_singleton, length0, length1, length2, length3, length4, length5])
  have at0 := ofBlocks_at page [[(Extract.withWork (W 0) restR, page)]] b0 [b1, b2, b3, b4, b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short
  have at1 := ofBlocks_at page [[(Extract.withWork (W 0) restR, page)], b0] b1 [b2, b3, b4, b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short
  have at2 := ofBlocks_at page [[(Extract.withWork (W 0) restR, page)], b0, b1] b2 [b3, b4, b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short
  have at3 := ofBlocks_at page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2] b3 [b4, b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short
  have at4 := ofBlocks_at page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3] b4 [b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short
  have at5 := ofBlocks_at page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4] b5
    [[(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short
  have atL := ofBlocks_at page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5]
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)] [] short
  have atR := ofBlocks_at page [] [(Extract.withWork (W 0) restR, page)] [b0, b1, b2, b3, b4, b5,
    [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short
  have wires : ∀ k, ((Prefix.ofBlocks page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
      [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short).wire k).data =
      note20 pub proof messages tLast eLast ∧
      ((Prefix.ofBlocks page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
      [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short).wire k).nonce = 0 :=
    Prefix.wire_all (Prefix.ofBlocks page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
      [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short)
      (fun wire => wire.data = note20 pub proof messages tLast eLast ∧ wire.nonce = 0)
      ⟨pageData, pageNonce⟩ fields
  have replyR : (Prefix.ofBlocks page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4,
      b5, [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short).reply 0 =
      some (Extract.withWork (W 0) restR) :=
    ((Extract.Prefix.reads_of_take atR.1 (List.cons_ne_nil _ _)).2 0 (Nat.succ_pos 0)).1
  have replyL : (Prefix.ofBlocks page [[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4,
      b5, [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]] short).reply 472 =
      some (Extract.foldAnswer blockL sL (W 472) fun _ => W 0) := by
    have reads := ((Extract.Prefix.reads_of_take atL.1 (List.cons_ne_nil _ _)).2 0 (Nat.succ_pos 0)).1
    rw [show ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5].flatten).length + 0 =
      472 by simp only [List.flatten_cons, List.flatten_nil, List.length_append,
        List.length_singleton, length0, length1, length2, length3, length4, length5,
        List.length_nil]] at reads
    exact reads
  rw [show ([[(Extract.withWork (W 0) restR, page)]]).flatten.length = 1 by simp, length0] at at0
  rw [show ([[(Extract.withWork (W 0) restR, page)], b0]).flatten.length = 51
    by simp [length0], length1] at at1
  rw [show ([[(Extract.withWork (W 0) restR, page)], b0, b1]).flatten.length = 120
    by simp [length0, length1], length2] at at2
  rw [show ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2]).flatten.length = 203
    by simp [length0, length1, length2], length3] at at3
  rw [show ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3]).flatten.length = 284
    by simp [length0, length1, length2, length3], length4] at at4
  rw [show ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4]).flatten.length = 378
    by simp [length0, length1, length2, length3, length4], length5] at at5
  refine ⟨?_, Reads20.of_all _ _ _ _ _ fun k => (wires k).1, fun j _ => ?_,
    fun k below reply answered => ?_, ?_, shows0 _ at0.2 at0.1,
    shows1 _ at1.2 at1.1, shows2 _ at2.2 at2.1, shows3 _ at3.2 at3.1, shows4 _ at4.2 at4.1,
    shows5 _ at5.2 at5.1, ⟨_, replyL, fun i => ?_⟩⟩
  · show 473 ≤ (List.flatten _).length
    rw [lengthAll]
  · refine iff_of_true ?_ (ok _)
    rw [(wires _).2]
    decide
  · refine iff_of_true ?_ (ok _)
    have inside' : k < ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
        [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]].flatten).length := by
      rw [lengthAll]
      exact below
    have entry := chained.getElem k inside'
    have found : reply = ([[(Extract.withWork (W 0) restR, page)], b0, b1, b2, b3, b4, b5,
        [(Extract.foldAnswer blockL sL (W 472) fun _ => W 0, page)]].flatten)[k].1 := by
      have same : (Prefix.ofBlocks page _ short).reply k = _ := answered
      simp only [Prefix.reply, Prefix.ofBlocks, List.getElem?_eq_getElem inside', Option.map_some,
        Option.some.injEq] at same
      exact same.symm
    rw [found, entry, Nat.zero_add]
    exact worked k
  · exact weighed _ _ replyR
  · show CodeSlot.kept64 (Extract.foldAnswer blockL sL (W 472) (fun _ => W 0) (some (blockL i))) = _
    rw [Extract.foldAnswer_some injectiveL]
    exact decodesL i

/-- Theorem A with theorem B "→" for `key20.aks`: an accepted proof of `layout20` has last blocks
`tLast`, `eLast` of the terminal with `Last20` and a transcript of `tower20` that presents its
bytes with them, and `Key20.Tower.accepts` of `tower20` accepts this transcript. -/
theorem verifyProof_exists_accepts20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
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
    ∃ (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
      (point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total),
      Last20 (E := E) pub proof messages tLast eLast ∧
        Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅ blockL
          point ∧
        (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts point := by
  obtain ⟨tLast, eLast, last⟩ := last20_of_accept (E := E) layout accepted terminalSetup decoded
  obtain ⟨point, presents⟩ := exists_presents20 (E := E) pub proof messages tLast eLast layout accepted
    terminalSetup expanded decoded points values injective injective₁ injective₂ injective₃
    injective₄ injective₅ injectiveL slots inside
  exact ⟨tLast, eLast, point, last, presents, presents20_accepts pub proof messages tLast eLast
    layout accepted terminalSetup expanded decoded points values last presents⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
