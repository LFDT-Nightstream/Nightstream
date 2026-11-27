import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseLevels
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseRange

/-! Theorem B of the join, direction "←", for `key20.aks`: the works (Akita revision `d50f700`,
`VerifierTranscript::check_grind`, `read_fold_response`, `VerifierFoldDraw`; ePrint 2026/1983,
revision of 2026-11-23, (146) and Figure 4, step 2).

- `Tier.works_of_decodes`, `Tier.tried_of_decodes`: the check `Tier.Decodes` of a level gives
  the work bits of each of its stages and the bound of the nonce of its response search.
- `stepOk20_of_accepts`: on a transcript that presents the bytes and that `Key20.Tower.accepts`
  accepts, the step of each work and of each response nonce of the replay has no stop.
- `isPool`, `Pools20`: the step of each squeeze of field challenges and of each draw of fold
  challenges has no stop (`decodePool` and `foldChallenges` give challenges); the model does not
  have these stops. `pools20_of_accept`: an accepted proof has `Pools20`.
- `index20_of`, `works20_of_stepOk`, `works20_of_accepts`: with `Pools20`, the works of the replay
  hold on such a transcript.
- `verifyProof_accept_of_presents20`: `verifyProof_accept_of_levels` with these works.
- `mem20_of`, `checks20_lists`: the indices of the `fused.end`, `tensor.opening`, `tensor.end` and
  `stage3.end` checks and of the terminal in the skeleton of `layout20`.
- `isRangeOp`, `ranges20_list`: the indices of the 15 checks of the range proofs of the levels
  (`range.k.end`, `range.leaf.end` or `norm.end`, `norm.integer`) in the skeleton of `layout20`.
- `checks20_of_accepts`: on a transcript that presents the bytes and that `Key20.Tower.accepts`
  accepts, the step of each check of the script has no stop when the run before it has none. A
  stop at a check of a range proof leaves a path of the level 5 with too few draws for its game
  (`range20_ok`, `model20_draws5`).
- `verifyProof_accept_of_tower20`: theorem B "←" for `key20.aks`, with the conjuncts `Pools20`,
  `Payload20` and `Digests20`.
- `verifyProof_iff_tower20`: for last blocks `Last20`, `runProof` accepts exactly when the layout
  takes the plan, `Pools20` and `Payload20` hold and `Key20.Tower.accepts` accepts a transcript
  that presents the bytes. -/

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

/-- The index of an operation whose erased form has `p`, in operations whose erased forms are the
skeleton of `layout20`, is a member of the indices of `p` in the skeleton. -/
theorem mem20_of {ops : Array Op} (erased : ops.toList.map Op.erase = skeleton (layout20 0 0))
    {p : Op → Bool} {c : ℕ} {op : Op} (at_ : ops[c]? = some op) (holds : p op.erase = true) :
    c ∈ (sk20.filter fun x => p x.1).map (·.2) := by
  have entry : (skeleton (layout20 0 0))[c]? = some op.erase := by
    rw [← erased, List.getElem?_map, Array.getElem?_toList, at_]
    rfl
  exact List.mem_map.mpr ⟨(op.erase, c), List.mem_filter.mpr
    ⟨List.mem_of_getElem? (i := c) (by rw [List.getElem?_zipIdx, entry]; simp), holds⟩, rfl⟩

/-- The operation is a `fused.end` check. -/
def isFusedOp : Op → Bool
  | .check _ (.fused ..) => true
  | _ => false

/-- The operation is a `tensor.opening` check. -/
def isOpeningOp : Op → Bool
  | .check _ (.tensorOpening ..) => true
  | _ => false

/-- The operation is a `tensor.end` check. -/
def isTensorOp : Op → Bool
  | .check _ (.tensor ..) => true
  | _ => false

/-- The operation is a `stage3.end` check. -/
def isStage3Op : Op → Bool
  | .check _ (.stage3 ..) => true
  | _ => false

/-- The operation is the terminal. -/
def isTerminalOp : Op → Bool
  | .terminal _ => true
  | _ => false

/-- The indices of the `fused.end`, `tensor.opening`, `tensor.end` and `stage3.end` checks and of
the terminal in the skeleton of `layout20`. -/
theorem checks20_lists :
    (sk20.filter fun x => isFusedOp x.1).map (·.2) = [165, 316, 586, 848, 1153, 1456] ∧
      (sk20.filter fun x => isOpeningOp x.1).map (·.2) = [321, 591, 853, 1158, 1461] ∧
      (sk20.filter fun x => isTensorOp x.1).map (·.2) = [385, 649, 908, 1210, 1513] ∧
      (sk20.filter fun x => isStage3Op x.1).map (·.2) = [] ∧
      (sk20.filter fun x => isTerminalOp x.1).map (·.2) = [1521] := by
  decide +kernel

/-- The operation is a check of the range proof of a level: `range.k.end`, the last check
`range.leaf.end` or `norm.end`, or `norm.integer`. -/
def isRangeOp : Op → Bool
  | .check _ (.range ..) | .check _ (.rangeNorm ..) | .check _ (.stage ..)
  | .check _ (.energy ..) => true
  | _ => false

/-- The indices of the checks of the range proofs of the levels in the skeleton of `layout20`. -/
theorem ranges20_list : (sk20.filter fun x => isRangeOp x.1).map (·.2) =
    [91, 245, 458, 521, 719, 783, 784, 975, 1032, 1093, 1094, 1278, 1335, 1396, 1397] := by
  decide +kernel

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
      ∃ out, terminalRun pub.layout.terminal.shape pub.digest pub.layout.terminal.limit
        pub.layout.terminal.params full
        (terminalMessagesIn (stateBefore pub proof messages pub.ops c) level)
        (terminalChallengesIn (stateBefore pub proof messages pub.ops c) level) = .ok out) :
    runProof pub proof = .verdict .accept :=
  verifyProof_accept_of_levels pub proof layout scripted terminalSetup decoded
    (works20_of_accepts pub proof messages tLast eLast layout scripted pools presents accepts)
    levels tensors terminal

/-- Theorem B "←" for the checks of the script: on a transcript that presents the bytes of a
proof of `layout20` and that `Key20.Tower.accepts` accepts, the step of each check has no stop
when the run before it has none. The steps of `fused.end`, `tensor.opening` and `tensor.end` are
from the links of the levels of the model (`level20_ok₀` to `level20_ok₅`); there is no
`stage3.end`. The steps of the checks of the range proofs are from the draws of the game of the
level 5 (`range20_ok`). -/
theorem checks20_of_accepts {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome)
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅
      blockL point)
    (accepts : (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts
      point)
    {c : ℕ} {key : Key} {check : Check} (at_ : pub.ops[c]? = some (.check key check))
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ()) :
    (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages pub.ops c)).1 = .ok () := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  obtain ⟨f0, f1, f2, f3, f4, f5, -⟩ := indices20 layout scripted
  obtain ⟨fusedList, openingList, tensorList, stage3List, -⟩ := checks20_lists
  have accepts' := accepts
  obtain ⟨-, -, l0, l1, l2, l3, l4, l5, -⟩ := accepts'
  have ok₀ := level20_ok₀ _ pub proof messages tLast eLast layout scripted aligned points values
    (expanded 0) rfl presents l0
  have ok₁ := level20_ok₁ _ pub proof messages tLast eLast layout scripted aligned (expanded 1) rfl
    presents l1
  have ok₂ := level20_ok₂ _ pub proof messages tLast eLast layout scripted aligned (expanded 2) rfl
    presents l2
  have ok₃ := level20_ok₃ _ pub proof messages tLast eLast layout scripted aligned (expanded 3) rfl
    presents l3
  have ok₄ := level20_ok₄ _ pub proof messages tLast eLast layout scripted aligned (expanded 4) rfl
    presents l4
  have ok₅ := level20_ok₅ _ pub proof messages tLast eLast layout scripted aligned (expanded 5) rfl
    presents l5
  rw [f0] at ok₀
  rw [f1] at ok₁
  rw [f2] at ok₂
  rw [f3] at ok₃
  rw [f4] at ok₄
  rw [f5] at ok₅
  have need := model20_draws5 _ pub proof messages tLast eLast rfl presents l5
  have early : ∀ i ∈ (sk20.filter fun x => isRangeOp x.1).map (·.2), i ≤ 1397 := by
    rw [ranges20_list]
    decide
  cases check with
  | fused level F =>
      have member := mem20_of erased (p := isFusedOp) at_ rfl
      rw [fusedList] at member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl | rfl | rfl | rfl
      · exact ok₀ before _ at_
      · exact ok₁.1 before _ at_
      · exact ok₂.1 before _ at_
      · exact ok₃.1 before _ at_
      · exact ok₄.1 before _ at_
      · exact ok₅.1 before _ at_
  | tensorOpening point value partials =>
      have member := mem20_of erased (p := isOpeningOp) at_ rfl
      rw [openingList] at member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl | rfl | rfl
      · exact ok₁.2.1 before _ at_
      · exact ok₂.2.1 before _ at_
      · exact ok₃.2.1 before _ at_
      · exact ok₄.2.1 before _ at_
      · exact ok₅.2.1 before _ at_
  | tensor lv count point opening =>
      have member := mem20_of erased (p := isTensorOp) at_ rfl
      rw [tensorList] at member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl | rfl | rfl
      · exact ok₁.2.2 before _ at_
      · exact ok₂.2.2 before _ at_
      · exact ok₃.2.2 before _ at_
      · exact ok₄.2.2 before _ at_
      · exact ok₅.2.2 before _ at_
  | stage3 lv plan =>
      have member := mem20_of erased (p := isStage3Op) at_ rfl
      rw [stage3List] at member
      exact absurd member List.not_mem_nil
  | range lv base arities point =>
      exact range20_ok pub proof messages layout scripted at_
        (early c (mem20_of erased (p := isRangeOp) at_ rfl)) before need
  | rangeNorm lv leaf limit arities point =>
      exact range20_ok pub proof messages layout scripted at_
        (early c (mem20_of erased (p := isRangeOp) at_ rfl)) before need
  | stage nodes arity weights output children =>
      exact range20_ok pub proof messages layout scripted at_
        (early c (mem20_of erased (p := isRangeOp) at_ rfl)) before need
  | energy energy limit =>
      exact range20_ok pub proof messages layout scripted at_
        (early c (mem20_of erased (p := isRangeOp) at_ rfl)) before need

/-- Theorem B of the join, direction "←", for `key20.aks`: a proof of `layout20` that decodes
with no byte left is accepted by `runProof` when `Key20.Tower.accepts` accepts a transcript that
presents its bytes (`Presents20`) and the conjuncts that the model does not have hold. `pools`:
the squeezes of field challenges and the draws of fold challenges give challenges. `payload`: the
payload of `z` decodes. `digests`: the digests of the proof are those of the last blocks of the
transcript. `terminalSetup` and `expanded`: the setup prefixes exist. `points` and `values`: the
shape of the public statement. -/
theorem verifyProof_accept_of_tower20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome)
    (decoded : decode pub.ops proof = .ok (messages, 0))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (pools : Pools20 pub proof messages) (payload : Payload20 pub proof messages)
    (digests : Digests20 pub proof messages tLast eLast)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅
      blockL point)
    (accepts : (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts
      point) :
    runProof pub proof = .verdict .accept := by
  have aligned := decode_aligned decoded
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase (layout ▸ scripted), skeleton20]
  have worked := works20_of_accepts pub proof messages tLast eLast layout scripted pools presents
    accepts
  refine (verifyProof_accept_iff20 pub proof layout scripted terminalSetup).mpr
    ⟨messages, decoded, ?_⟩
  rcases stop : (replay pub pub.ops proof messages).1 with _ | st
  · rfl
  · exfalso
    obtain ⟨op, at_, before, stepped⟩ := replay_stop_at pub proof messages pub.ops stop
    rcases st with ⟨i, site⟩ | ⟨i, site⟩ | k
    · unfold works at worked
      rw [stop] at worked
      cases worked
    · simp only [Stop.op] at at_ before stepped
      rcases step_reject pub proof messages i op _ stepped with ⟨key, check, rfl, -⟩ |
        ⟨level, rfl, found⟩
      · have ok := checks20_of_accepts pub proof messages tLast eLast layout scripted aligned
          expanded points values presents accepts at_ before
        rw [stepped] at ok
        cases ok
      · have member := mem20_of erased (p := isTerminalOp) at_ rfl
        rw [checks20_lists.2.2.2.2] at member
        obtain ⟨-, -, -, -, -, -, size⟩ := indices20 layout scripted
        obtain rfl : i = pub.ops.size - 1 := (List.mem_singleton.mp member).trans size.symm
        have last := (frames20 layout scripted).2.2.2.2.2.2.2
        rw [at_] at last
        cases last
        have accepts' := accepts
        obtain ⟨-, -, -, -, -, -, -, -, passed⟩ := accepts'
        obtain ⟨out, accepted⟩ := terminal20_of_last pub proof messages tLast eLast layout
          scripted terminalSetup aligned payload
          (last20_of_passed _ pub proof messages tLast eLast digests rfl presents passed) before
        unfold terminalValue at found
        dsimp only at found
        rw [terminalSetup, Option.map_some, accepted] at found
        cases found
    · exact replay_matches20 layout scripted decoded terminalSetup k stop

/-- Theorem B of the join for `key20.aks` as an equivalence: for a proof of `layout20` that
decodes with no byte left, last blocks `Last20` of the terminal of the code and a transcript that
presents the bytes with them, `runProof` accepts exactly when the layout takes the plan, the
conjuncts `Pools20` and `Payload20` hold and `Key20.Tower.accepts` accepts the transcript. -/
theorem verifyProof_iff_tower20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome)
    (decoded : decode pub.ops proof = .ok (messages, 0))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (last : Last20 (E := E) pub proof messages tLast eLast)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅
      blockL point) :
    runProof pub proof = .verdict .accept ↔
      script pub.layout pub.plan = .ok pub.ops ∧ Pools20 pub proof messages ∧
        Payload20 pub proof messages ∧
        (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts point := by
  constructor
  · intro accepted
    obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
    exact ⟨scripted, pools20_of_accept scripted ends terminalSetup accepted decoded,
      payload20_guard layout accepted terminalSetup decoded,
      presents20_accepts pub proof messages tLast eLast layout accepted terminalSetup expanded
        decoded points values last presents⟩
  · rintro ⟨scripted, pools, payload, accepts⟩
    exact verifyProof_accept_of_tower20 pub proof messages tLast eLast layout scripted
      terminalSetup expanded decoded points values pools payload last.digests presents
      accepts

end NightstreamFPrime.Spec.Compression.Akita.Replay
