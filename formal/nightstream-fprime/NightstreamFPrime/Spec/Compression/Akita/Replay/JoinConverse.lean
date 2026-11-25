import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeAccept

/-! Theorem B of the join, direction "←", for `key20.aks`: the works (Akita revision `d50f700`,
`VerifierTranscript::check_grind`, `read_fold_response`, `VerifierFoldDraw`; ePrint 2026/1983,
revision of 2026-11-23, (146) and Figure 4, step 2; `docs/akita-work/w17/brief-c.md`).

- `Tier.works_of_decodes`, `Tier.tried_of_decodes`: the check `Tier.Decodes` of a level gives
  the work bits of each of its stages and the bound of the nonce of its response search.
- `stepOk20_of_accepts`: on a transcript that presents the bytes and that `Key20.Tower.accepts`
  accepts, the step of each work and of each response nonce of the replay has no stop.
- `isPool`, `Pools20`: the step of each squeeze of field challenges and of each draw of fold
  challenges has no stop (`decodePool` and `foldChallenges` give challenges); the model does not
  have these stops. `pools20_of_accept`: an accepted proof has `Pools20`.
- `index20_of`, `works20_of_stepOk`, `works20_of_accepts`: with `Pools20`, the works of the replay
  hold on such a transcript.
- `verifyProof_accept_of_presents20`: `verifyProof_accept_of_levels` with these works. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

section Works

variable {Block : Type} {Ring : Block → Type}
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [Fintype Slot] {total : ℕ}

/-- The check `Tier.Decodes` of a level with the fold stage `start`, `len` check stages and the
work bits `bits` of its stages: the work slot of the answer of each stage of the level has its
work bits. -/
theorem Tier.works_of_decodes (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source)
    {point : Prefix Wire (Option Coord → Slot) total} (decodes : T.Decodes point)
    {start len : ℕ} {bits : ℕ → ℕ} (started : T.start = start) (length : T.costs.length = len)
    (fold : T.foldBits = bits start)
    (checks : ∀ index < len, T.bits.getD index 0 = bits (start + 1 + index)) {k : ℕ}
    (lo : start ≤ k) (hi : k < start + 1 + len) {reply : Option Coord → Slot}
    (answered : point.reply k = some reply) :
    reply none ∈ Extract.workZeros Slot (bits k) := by
  obtain ⟨⟨first, firstAt, firstWork, -⟩, later, -⟩ := decodes
  rw [started] at firstAt later
  rw [length] at later
  by_cases same : k = start
  · subst same
    obtain rfl := Option.some.inj (firstAt.symm.trans answered)
    rw [← fold]
    exact firstWork
  · have at_ : start + 1 + (k - (start + 1)) = k := by omega
    have known := later (k - (start + 1)) (by omega) reply (by rw [at_]; exact answered)
    rw [checks _ (by omega), at_] at known
    exact known

/-- The check `Tier.Decodes` of a level with the fold stage `start`: the nonce of the response
search in the message before the stage `start` is below `responseTries`. -/
theorem Tier.tried_of_decodes (T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source)
    {point : Prefix Wire (Option Coord → Slot) total} (decodes : T.Decodes point) {start : ℕ}
    (started : T.start = start) : T.read.nonce (point.wire start) < Extract.responseTries := by
  subst started
  exact decodes.2.2

end Works

/-- The operation is a squeeze of field challenges or the draw of the fold challenges of a
group. -/
def isPool : Op → Bool
  | .challenge .. | .squeeze .. => true
  | _ => false

/-- The pools of the code give challenges: the step of the replay at each squeeze of field
challenges (`decodePool`) and at each draw of the fold challenges of a group (`foldChallenges`)
has no stop. -/
def Pools20 (pub : Public) (proof : ByteArray) (messages : Array Message) : Prop :=
  ∀ c op, pub.ops[c]? = some op → isPool op = true → StepOk pub proof messages c

/-- An accepted proof has `Pools20`. -/
theorem pools20_of_accept {pub : Public} {proof : ByteArray}
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (ends : endsFit pub.ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) : Pools20 pub proof messages :=
  fun c _ _ _ => stepOk_of_accept scripted ends terminalSetup accepted decoded c

/-- The index of an operation whose erased form has `p`, in operations whose erased forms are the
skeleton of `layout20`, is an entry of the indices of `p` in the skeleton. -/
theorem index20_of {ops : Array Op} (erased : ops.toList.map Op.erase = skeleton (layout20 0 0))
    {p : Op → Bool} {c : ℕ} {op : Op} (at_ : ops[c]? = some op) (holds : p op.erase = true) :
    ∃ k < ((sk20.filter fun x => p x.1).map (·.2)).length,
      ((sk20.filter fun x => p x.1).map (·.2)).getD k 0 = c := by
  have entry : (skeleton (layout20 0 0))[c]? = some op.erase := by
    rw [← erased, List.getElem?_map, Array.getElem?_toList, at_]
    rfl
  have member : c ∈ (sk20.filter fun x => p x.1).map (·.2) :=
    List.mem_map.mpr ⟨(op.erase, c), List.mem_filter.mpr
      ⟨List.mem_of_getElem? (i := c) (by rw [List.getElem?_zipIdx, entry]; simp), holds⟩, rfl⟩
  obtain ⟨k, inside, same⟩ := List.getElem_of_mem member
  exact ⟨k, inside, by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem inside, same]
    rfl⟩

/-- The works of the replay of a script of `layout20` hold when the step of each work, of each
response nonce and of each pool has no stop. -/
theorem works20_of_stepOk {zMax rice : ℕ} {pub : Public} {proof : ByteArray}
    {messages : Array Message} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (worked : ∀ k < 473, StepOk pub proof messages (workIndices20.getD k 0))
    (nonces : ∀ j < 7, StepOk pub proof messages (nonceIndices20.getD j 0))
    (pools : Pools20 pub proof messages) : works pub pub.ops proof messages = true := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted, layout, skeleton20]
  rcases stop : (replay pub pub.ops proof messages).1 with _ | (⟨i, site⟩ | ⟨i, site⟩ | i)
  · unfold works
    rw [stop]
  · exfalso
    obtain ⟨op, at_, -, stepped⟩ := replay_stop_at pub proof messages pub.ops stop
    simp only [Stop.op] at at_ stepped
    have grinds := step_noWork pub proof messages i op _ stepped
    have ok : StepOk pub proof messages i := by
      cases op with
      | work site level bits =>
          obtain ⟨k, inside, same⟩ := index20_of erased (p := isWork) at_ rfl
          have below : k < 473 := by
            have h : k < workIndices20.length := inside
            rwa [workIndices20_length] at h
          have known := worked k below
          rwa [show workIndices20.getD k 0 = i from same] at known
      | foldNonce level width =>
          obtain ⟨k, inside, same⟩ := index20_of erased (p := isFoldNonce) at_ rfl
          have below : k < 7 := by
            have h : k < nonceIndices20.length := inside
            rwa [nonceIndices20_length] at h
          have known := nonces k below
          rwa [show nonceIndices20.getD k 0 = i from same] at known
      | challenge key count => exact pools i _ at_ rfl
      | squeeze level group G => exact pools i _ at_ rfl
      | _ => cases grinds
    have ran := ok op at_
    rw [stepped] at ran
    cases ran
  · unfold works
    rw [stop]
  · unfold works
    rw [stop]

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

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem B "←" for the works (Akita (146) and Figure 4, step 2): on a transcript that presents
the bytes of a proof and that `Key20.Tower.accepts` accepts, the step of the replay at each of the
473 works and at each of the 7 response nonces has no stop. The checks of the model are the work
bits of `RootLinked`, `Tier.Decodes` of each level and `Terminal.Passed`, and the nonce of the
response search of each fold. -/
theorem stepOk20_of_accepts
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) (accepts : o.accepts point) :
    (∀ k < 473, StepOk pub proof messages (workIndices20.getD k 0)) ∧
      ∀ j < 7, StepOk pub proof messages (nonceIndices20.getD j 0) := by
  obtain ⟨s0, n0, -, f0, b0⟩ := zeroth_numbers o
  obtain ⟨s1, n1, -, f1, b1⟩ := first20_numbers o
  obtain ⟨s2, n2, -, f2, b2⟩ := second20_numbers o
  obtain ⟨s3, n3, -, f3, b3⟩ := third20_numbers o
  obtain ⟨s4, n4, -, f4, b4⟩ := fourth20_numbers o
  obtain ⟨s5, n5, -, f5, b5⟩ := fifth20_numbers o
  subst tower
  obtain ⟨-, rooted, ⟨-, d0, -⟩, ⟨-, d1, -⟩, ⟨-, d2, -⟩, ⟨-, d3, -⟩, ⟨-, d4, -⟩, ⟨-, d5, -⟩,
    passed⟩ := accepts
  have long := presents.long
  refine ⟨fun k below => ?_, fun j below => ?_⟩
  · obtain ⟨⟨reply, wire⟩, found⟩ : ∃ entry, point.rest[k]? = some entry :=
      ⟨_, List.getElem?_eq_getElem (by omega)⟩
    have answered : point.reply k = some reply := by simp [Prefix.reply, found]
    refine (presents.work k below reply answered).mp ?_
    by_cases h0 : k = 0
    · subst h0
      obtain ⟨-, -, -, -, other, at_, work⟩ := rooted
      obtain rfl := Option.some.inj (at_.symm.trans answered)
      exact work
    by_cases h1 : k < 51
    · exact Tier.works_of_decodes _ d0 (bits := fun k => bits20.flatten.getD k 0) s0 n0 f0 b0
        (by omega) (by omega) answered
    by_cases h2 : k < 120
    · exact Tier.works_of_decodes _ d1 (bits := fun k => bits20.flatten.getD k 0) s1 n1 f1 b1
        (by omega) (by omega) answered
    by_cases h3 : k < 203
    · exact Tier.works_of_decodes _ d2 (bits := fun k => bits20.flatten.getD k 0) s2 n2 f2 b2
        (by omega) (by omega) answered
    by_cases h4 : k < 284
    · exact Tier.works_of_decodes _ d3 (bits := fun k => bits20.flatten.getD k 0) s3 n3 f3 b3
        (by omega) (by omega) answered
    by_cases h5 : k < 378
    · exact Tier.works_of_decodes _ d4 (bits := fun k => bits20.flatten.getD k 0) s4 n4 f4 b4
        (by omega) (by omega) answered
    by_cases h6 : k < 472
    · exact Tier.works_of_decodes _ d5 (bits := fun k => bits20.flatten.getD k 0) s5 n5 f5 b5
        (by omega) (by omega) answered
    obtain rfl : k = 472 := by omega
    obtain ⟨other, at_, work, -⟩ := passed.decodes
    obtain rfl := Option.some.inj (at_.symm.trans answered)
    have bitsEq : Extract.Terminal.foldBits (tower20 (E := E) pub block block₁ block₂ block₃ block₄
        block₅ blockL total 472).terminal = bits20.flatten.getD 472 0 := by
      simp only [Extract.Terminal.foldBits]
      decide
    rw [← bitsEq]
    exact work
  · interval_cases j
    · exact (presents.nonce 0 (by decide)).mp (Tier.tried_of_decodes _ d0 s0)
    · exact (presents.nonce 1 (by decide)).mp (Tier.tried_of_decodes _ d1 s1)
    · exact (presents.nonce 2 (by decide)).mp (Tier.tried_of_decodes _ d2 s2)
    · exact (presents.nonce 3 (by decide)).mp (Tier.tried_of_decodes _ d3 s3)
    · exact (presents.nonce 4 (by decide)).mp (Tier.tried_of_decodes _ d4 s4)
    · exact (presents.nonce 5 (by decide)).mp (Tier.tried_of_decodes _ d5 s5)
    · exact (presents.nonce 6 (by decide)).mp passed.tried

/-- On a transcript that presents the bytes of a proof of `layout20` and that
`Key20.Tower.accepts` accepts, with `Pools20`, the works of the replay hold. -/
theorem works20_of_accepts {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) (pools : Pools20 pub proof messages)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅
      blockL point)
    (accepts : (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts
      point) :
    works pub pub.ops proof messages = true := by
  obtain ⟨worked, nonces⟩ := stepOk20_of_accepts _ pub proof messages tLast eLast rfl presents
    accepts
  exact works20_of_stepOk layout scripted worked nonces pools

/-- `verifyProof_accept_of_levels` with the works of the tower: a proof of `layout20` that decodes
with no byte left is accepted when a transcript that presents its bytes is accepted by
`Key20.Tower.accepts`, `Pools20` holds, and the level functions, the tensor reductions and the
terminal of `verifyProof_accept_of_levels` accept. -/
theorem verifyProof_accept_of_presents20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (decoded : decode pub.ops proof = .ok (messages, 0)) (pools : Pools20 pub proof messages)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅
      blockL point)
    (accepts : (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts
      point)
    (levels : ∀ ℓ : Fin 6,
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take (levelStart pub.layout ℓ))
        (channelStart pub)).1 = .ok () →
      foldRunIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ℓ)) ℓ
        (fold20 ℓ) ≠ none)
    (tensors : ∀ c key level count point opening,
      pub.ops[c]? = some (.check key (.tensor level count point opening)) →
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take (levelStart pub.layout level))
        (channelStart pub)).1 = .ok () →
      (tensorRunIn (stateBefore pub proof messages pub.ops c) level count point opening).isSome =
        true)
    (terminal : ∀ c level, pub.ops[c]? = some (.terminal level) →
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
      ∃ out, terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
        pub.layout.terminal.params full
        (terminalMessagesIn (stateBefore pub proof messages pub.ops c) level)
        (terminalChallengesIn (stateBefore pub proof messages pub.ops c) level) = .ok out) :
    runProof pub proof = .verdict .accept :=
  verifyProof_accept_of_levels pub proof layout scripted terminalSetup decoded
    (works20_of_accepts pub proof messages tLast eLast layout scripted pools presents accepts)
    levels tensors terminal

end NightstreamFPrime.Spec.Compression.Akita.Replay
