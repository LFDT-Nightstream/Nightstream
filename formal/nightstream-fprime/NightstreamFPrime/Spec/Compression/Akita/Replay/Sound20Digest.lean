import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverseTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sound20Deps

/-! The stage data of the replay of `key20.aks` do not read the hash `Public.digest` of the two
last-block digests (Akita revision `d50f700`, `akita-verifier/src/terminal/direct.rs`; ePrint
2026/1983, revision of 2026-11-23, Lemma 10.47; `docs/akita-work/w22/design.md`;
`Replay/DigestFree.lean`, `Replay/Resolve.lean`, `Replay/Sound20Deps.lean`).

The only hypothesis is `pub.layout = layout20 zMax rice`. A script of `layout20` has its only
terminal at the end (`script20_free`), and a layout that does not take the plan has no operation.

- `run20_withDigest`: the run before the work of each stage, and before the terminal for
  `473 ≤ k`, is the same for each hash. `eventsAt20_withDigest`, `tDigestAt_withDigest`,
  `eDigestAt_withDigest`, `pageAt20_withDigest`, `stagePage20_withDigest`,
  `addressAt20_withDigest`: the same for the events, the digests and the pages, for each `k`.
- `stage20_withDigest`, `recordAddress20_withDigest`, `resolve20_withDigest`,
  `resolveWith20_withDigest`, `back20_withDigest`, `resolveAt_withDigest`, `backAt_withDigest`:
  the address of a key and its inverse.
- `tower20_withDigest`, `readings20_withDigest`: the tower and the readings.
- `last20_withDigest`: `Last20` reads the hash only in its two digest equations. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Address Prefix)
open Sponge (answerOf)
open Replayer

section Free

/-- No operation before the index `c` is the terminal when no key before `c` has a terminal. -/
theorem take_free (ops : Array Op) (c : ℕ)
    (free : ∀ j < c, ∀ level, ops[j]? ≠ some (.terminal level)) :
    ∀ x ∈ ops.zipIdx.toList.take c, x.1.isTerminal = false := by
  intro x member
  obtain ⟨j, inside, at_⟩ := List.getElem_of_mem member
  have short : j < c ∧ j < ops.zipIdx.toList.length := by
    rw [List.length_take] at inside
    omega
  rw [List.getElem_take, zipIdx_getElem ops short.2] at at_
  subst at_
  cases found : ops[j]'(by simpa using short.2) with
  | terminal level =>
      exact absurd ((Array.getElem?_eq_getElem _).trans (congrArg some found))
        (free j short.1 level)
  | _ => rfl

variable {zMax rice : ℕ} (pub : Public)

/-- The operations of a public input of `layout20` are a script of `layout20`, or the layout does
not take the plan and there is no operation. -/
theorem ops_scripted20 (layout : pub.layout = layout20 zMax rice) :
    pub.ops = #[] ∨ pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
  unfold Public.ops
  rw [layout]
  cases scripted : script (layout20 zMax rice) pub.plan with
  | error _ => exact Or.inl rfl
  | ok ops => exact Or.inr ((script_erase scripted).trans (skeleton20 zMax rice))

/-- No operation before the last one of a public input of `layout20` is the terminal. -/
theorem ops_free20 (layout : pub.layout = layout20 zMax rice) {j : ℕ}
    (before : j < pub.ops.size - 1) (level : ℕ) : pub.ops[j]? ≠ some (.terminal level) := by
  revert before
  unfold Public.ops
  rw [layout]
  cases scripted : script (layout20 zMax rice) pub.plan with
  | error _ => exact fun inside => absurd inside (by simp)
  | ok ops => exact fun inside => script20_free scripted inside level

/-- On a script of `layout20`, the work of each stage is before the terminal. -/
theorem stageIndex20_le_last (erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0))
    (k : ℕ) : stageIndex20 pub k ≤ pub.ops.size - 1 := by
  have size : pub.ops.size = 1522 := by rw [size_skeleton erased, skeleton20_length]
  by_cases inside : k < 473
  · rw [stageIndex20_lt pub inside]
    have step := workIndices20_mono (j := k) (k := 472) (by omega) (by omega)
    have last := foldDraws20.2.2.2.2.2.2.2.2.2
    omega
  · exact Nat.le_of_eq (ite_eq_right inside)

/-- For `layout20`, no operation before the work of a stage, and before the terminal for
`473 ≤ k`, is the terminal. -/
theorem free20 (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    ∀ x ∈ pub.ops.zipIdx.toList.take (stageIndex20 pub k), x.1.isTerminal = false := by
  refine take_free pub.ops _ fun j below level at_ => ?_
  rcases ops_scripted20 pub layout with empty | erased
  · rw [empty] at at_
    exact absurd at_ (by simp)
  · exact ops_free20 pub layout
      (Nat.lt_of_lt_of_le below (stageIndex20_le_last pub erased k)) level at_

end Free

section Run

variable {zMax rice : ℕ} (pub : Public) (digest : ByteArray → ByteArray) (proof : ByteArray)
  (messages : Array Message)

theorem Public.withDigest_layout : (pub.withDigest digest).layout = pub.layout := rfl

/-- The decoder of the wire format reads the operations only. -/
theorem decode_withDigest :
    decode (pub.withDigest digest).ops proof = decode pub.ops proof := rfl

theorem stageIndex20_withDigest (k : ℕ) :
    stageIndex20 (pub.withDigest digest) k = stageIndex20 pub k := rfl

theorem bits20_withDigest : Bits20 (pub.withDigest digest) ↔ Bits20 pub := Iff.rfl

/-- For `layout20`, the run before the work of the stage `k`, and before the terminal for
`473 ≤ k`, has the same result and the same state for each hash. -/
theorem run20_withDigest (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    runFrom (pub.withDigest digest) proof messages
        ((pub.withDigest digest).ops.zipIdx.toList.take (stageIndex20 (pub.withDigest digest) k))
        (channelStart (pub.withDigest digest)) =
      runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
        (channelStart pub) :=
  runFrom_withDigest pub digest proof messages _ _ (free20 pub layout k)

/-- For `layout20`, the run has no stop before the work of a stage for one hash when it has no
stop for each hash. -/
theorem ran20_withDigest (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    (runFrom (pub.withDigest digest) proof messages
        ((pub.withDigest digest).ops.zipIdx.toList.take (stageIndex20 (pub.withDigest digest) k))
        (channelStart (pub.withDigest digest))).1 = .ok () ↔
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take (stageIndex20 pub k))
        (channelStart pub)).1 = .ok () := by
  rw [run20_withDigest pub digest proof messages layout k]

/-- For `layout20`, the state before the work of the stage `k`, and before the terminal for
`473 ≤ k`, does not read the hash. -/
theorem stateAt20_withDigest (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    stateBefore (pub.withDigest digest) proof messages (pub.withDigest digest).ops
        (stageIndex20 (pub.withDigest digest) k) =
      stateBefore pub proof messages pub.ops (stageIndex20 pub k) :=
  congrArg Prod.snd (run20_withDigest pub digest proof messages layout k)

/-- For `layout20`, the events before the squeeze of the work of each stage do not read the
hash. -/
theorem eventsAt20_withDigest (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    eventsAt20 (pub.withDigest digest) proof messages k = eventsAt20 pub proof messages k := by
  unfold eventsAt20
  rw [stateAt20_withDigest pub digest proof messages layout k]
  rfl

theorem tDigestAt_withDigest (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    tDigestAt (pub.withDigest digest) proof messages k = tDigestAt pub proof messages k := by
  unfold tDigestAt
  rw [stateAt20_withDigest pub digest proof messages layout k]

theorem eDigestAt_withDigest (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    eDigestAt (pub.withDigest digest) proof messages k = eDigestAt pub proof messages k := by
  unfold eDigestAt
  rw [stateAt20_withDigest pub digest proof messages layout k]

theorem traceAt20_withDigest (k : ℕ) :
    traceAt20 (pub.withDigest digest) proof messages k = traceAt20 pub proof messages k := rfl

end Run

section Pages

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {zMax rice : ℕ} (pub : Public)
  (digest : ByteArray → ByteArray) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)

/-- The data of a wire message on a state do not read the hash. -/
theorem noteOf_withDigest (s : Replayer) :
    noteOf (E := E) (pub.withDigest digest) s tLast eLast = noteOf pub s tLast eLast := rfl

/-- The messages of the checks of a wire message on a state do not read the hash. -/
theorem recvs20_withDigest (s : Replayer) (k : ℕ) :
    recvs20 E (pub.withDigest digest) s k = recvs20 E pub s k := rfl

/-- For `layout20`, each page `k` does not read the hash. The pages `473 ≤ k` are on the state
before the terminal. -/
theorem pageAt20_withDigest (layout : pub.layout = layout20 zMax rice) (k : ℕ) :
    pageAt20 (E := E) (pub.withDigest digest) proof messages tLast eLast k =
      pageAt20 pub proof messages tLast eLast k := by
  unfold pageAt20 noteAt20 cellsAt20
  rw [stateAt20_withDigest pub digest proof messages layout k]
  rfl

theorem stagePage20_withDigest (layout : pub.layout = layout20 zMax rice) (j : ℕ) :
    stagePage20 (E := E) (pub.withDigest digest) proof messages tLast eLast j =
      stagePage20 pub proof messages tLast eLast j :=
  pageAt20_withDigest pub digest proof messages _ _ layout j

/-- For `layout20`, the key of each stage does not read the hash. -/
theorem addressAt20_withDigest (layout : pub.layout = layout20 zMax rice) (total k : ℕ) :
    addressAt20 (E := E) (pub.withDigest digest) proof messages tLast eLast total k =
      addressAt20 pub proof messages tLast eLast total k := by
  refine Address.ext (stagePage20_withDigest pub digest proof messages tLast eLast layout 0) ?_
  show (List.range (min k total)).map _ = (List.range (min k total)).map _
  exact List.map_congr_left fun j _ =>
    stagePage20_withDigest pub digest proof messages tLast eLast layout (j + 1)

end Pages

section Resolve

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {zMax rice : ℕ} (pub : Public)
  (digest : ByteArray → ByteArray)

/-- For `layout20`, the stages of a list of events are the same for each hash. -/
theorem stage20_withDigest (layout : pub.layout = layout20 zMax rice)
    (events : List Duplex.Event) (k : ℕ) (proof : ByteArray) (messages : Array Message) :
    Stage20 (pub.withDigest digest) events k proof messages ↔
      Stage20 pub events k proof messages := by
  unfold Stage20
  rw [run20_withDigest pub digest proof messages layout k,
    eventsAt20_withDigest pub digest proof messages layout k]
  exact Iff.rfl

theorem recordAddress20_withDigest (layout : pub.layout = layout20 zMax rice)
    (record : List (ByteArray × ByteArray)) (total k : ℕ) (proof : ByteArray)
    (messages : Array Message) :
    recordAddress20 (E := E) (pub.withDigest digest) record total k proof messages =
      recordAddress20 pub record total k proof messages := by
  unfold recordAddress20
  rw [tDigestAt_withDigest pub digest proof messages layout k,
    eDigestAt_withDigest pub digest proof messages layout k]
  exact addressAt20_withDigest pub digest proof messages _ _ layout total k

/-- For `layout20`, `resolve20` does not read the hash. -/
theorem resolve20_withDigest (layout : pub.layout = layout20 zMax rice)
    (record : List (ByteArray × ByteArray)) (events : List Duplex.Event) (total : ℕ) :
    resolve20 (E := E) total (pub.withDigest digest) record events =
      resolve20 total pub record events := by
  have stage : Stage20 (pub.withDigest digest) = Stage20 pub := by
    funext events k proof messages
    exact propext (stage20_withDigest pub digest layout events k proof messages)
  have address : recordAddress20 (E := E) (pub.withDigest digest) = recordAddress20 pub := by
    funext record total k proof messages
    exact recordAddress20_withDigest pub digest layout record total k proof messages
  unfold resolve20
  rw [stage, address]

/-- For `layout20`, `resolveWith20` does not read the hash. -/
theorem resolveWith20_withDigest (layout : pub.layout = layout20 zMax rice)
    (answer : List Duplex.Event → ℕ → ByteArray) (record : List (ByteArray × ByteArray))
    (events : List Duplex.Event) (total : ℕ) :
    resolveWith20 (E := E) total (pub.withDigest digest) answer record events =
      resolveWith20 total pub answer record events :=
  resolve20_withDigest (pub.reideal (some answer)) digest layout record events total

/-- For `layout20`, `back20` does not read the hash. -/
theorem back20_withDigest (layout : pub.layout = layout20 zMax rice) {total : ℕ}
    (address : Address (Key20.Wire E) total) :
    back20 (pub.withDigest digest) address = back20 pub address := by
  have presented : Presented20 (E := E) (total := total) (pub.withDigest digest) =
      Presented20 pub := by
    funext address proof messages tLast eLast
    unfold Presented20
    rw [run20_withDigest pub digest proof messages layout address.rest.length,
      addressAt20_withDigest pub digest proof messages tLast eLast layout total
        address.rest.length]
    rfl
  have events : eventsAt20 (pub.withDigest digest) = eventsAt20 pub := by
    funext proof messages k
    exact eventsAt20_withDigest pub digest proof messages layout k
  unfold back20
  rw [presented, events]

variable {L B C S D : ℕ}

/-- For `layout20`, the address function `resolveAt` of the recording prover does not read the
hash. -/
theorem resolveAt_withDigest (layout : pub.layout = layout20 zMax rice) (total : ℕ)
    (fs : Sponge.Base L B C → Sponge.Y S C) (record : List (Sponge.Str D × Bytes 32))
    (base : Sponge.Base L B C) :
    resolveAt (E := E) total (pub.withDigest digest) fs record base =
      resolveAt total pub fs record base :=
  resolveWith20_withDigest pub digest layout (answerOf fs) (recordOf record) base.events total

/-- For `layout20`, the key `backAt` of an address does not read the hash. -/
theorem backAt_withDigest (layout : pub.layout = layout20 zMax rice) {total : ℕ}
    (fs : Sponge.Base L B C → Sponge.Y S C) (address : Address (Key20.Wire E) total) :
    backAt (pub.withDigest digest) fs address = backAt pub fs address := by
  have back : back20 ((pub.withDigest digest).reideal (some (answerOf fs))) address =
      back20 (pub.reideal (some (answerOf fs))) address :=
    back20_withDigest (pub.reideal (some (answerOf fs))) digest layout address
  unfold backAt
  rw [back]

end Resolve

section Tower

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {Coord : Type} (pub : Public)
  (digest : ByteArray → ByteArray)

/-- The tower data of the replay read the setup prefixes only, and not the hash. -/
theorem tower20_withDigest (block : Key20.root.Block → Coord)
    (block₁ : Fin Key20.level₁.shape.blocks → Coord) (block₂ : Fin Key20.level₂.shape.blocks → Coord)
    (block₃ : Fin Key20.level₃.shape.blocks → Coord) (block₄ : Fin Key20.level₄.shape.blocks → Coord)
    (block₅ : Fin Key20.level₅.shape.blocks → Coord) (blockL : Fin Key20.terminal.blocks → Coord)
    (total stage : ℕ) :
    tower20 (E := E) (pub.withDigest digest) block block₁ block₂ block₃ block₄ block₅ blockL total
        stage =
      tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total stage := rfl

/-- `tower20_withDigest` for the public input without the setup prefix of the caller. -/
theorem tower20_sealed_withDigest (block : Key20.root.Block → Coord)
    (block₁ : Fin Key20.level₁.shape.blocks → Coord) (block₂ : Fin Key20.level₂.shape.blocks → Coord)
    (block₃ : Fin Key20.level₃.shape.blocks → Coord) (block₄ : Fin Key20.level₄.shape.blocks → Coord)
    (block₅ : Fin Key20.level₅.shape.blocks → Coord) (blockL : Fin Key20.terminal.blocks → Coord)
    (total stage : ℕ) :
    tower20 (E := E) (pub.withDigest digest).sealed block block₁ block₂ block₃ block₄ block₅ blockL
        total stage =
      tower20 pub.sealed block block₁ block₂ block₃ block₄ block₅ blockL total stage := rfl

end Tower

section Readings

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  {pub pub' : Public} {proof : ByteArray} {messages : Array Message}
  {answer : List Duplex.Event → ℕ → ByteArray}
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}
  {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}

omit [DecidableEq E] in
/-- `Readings20` reads the public input only through the events before the squeeze of the work
of each stage. -/
theorem Readings20.of_events
    (same : ∀ k, eventsAt20 pub' proof messages k = eventsAt20 pub proof messages k)
    (readings : Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅
      blockL point) :
    Readings20 pub' proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
      point := by
  have field : ∀ (k : ℕ) (d : Draw) (reply : Option Coord → CodeSlot E),
      FieldReads20 pub' proof messages answer k d reply =
        FieldReads20 pub proof messages answer k d reply := fun k d reply => by
    unfold FieldReads20
    rw [same]
  have fold : ∀ (k g : ℕ) {d n : ℕ} [NeZero d] (decode : CodeSlot E → Option (Rq d))
      (slots : Fin n → Coord) (reply : Option Coord → CodeSlot E),
      FoldReads20 pub' proof messages answer k g decode slots reply =
        FoldReads20 pub proof messages answer k g decode slots reply :=
    fun k g _ _ _ decode slots reply => by
      unfold FoldReads20
      rw [same]
  exact ⟨by simpa only [same] using readings.work, by simpa only [same] using readings.weights,
    by simpa only [field] using readings.field, by simpa only [fold] using readings.root,
    by simpa only [fold] using readings.level₁, by simpa only [fold] using readings.level₂,
    by simpa only [fold] using readings.level₃, by simpa only [fold] using readings.level₄,
    by simpa only [fold] using readings.level₅, by simpa only [fold] using readings.terminal⟩

omit [DecidableEq E] in
/-- For `layout20`, `Readings20` does not read the hash. -/
theorem readings20_withDigest {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (digest : ByteArray → ByteArray) :
    Readings20 (pub.withDigest digest) proof messages answer block block₁ block₂ block₃ block₄
        block₅ blockL point ↔
      Readings20 pub proof messages answer block block₁ block₂ block₃ block₄ block₅ blockL
        point :=
  ⟨Readings20.of_events fun k => (eventsAt20_withDigest pub digest proof messages layout k).symm,
    Readings20.of_events fun k => eventsAt20_withDigest pub digest proof messages layout k⟩

end Readings

section Last

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {zMax rice : ℕ} (pub : Public)
  (digest : ByteArray → ByteArray) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)

/-- For `layout20`, the state before the terminal does not read the hash. -/
theorem terminalState20_withDigest (layout : pub.layout = layout20 zMax rice) :
    terminalState20 (pub.withDigest digest) proof messages = terminalState20 pub proof messages :=
  stateBefore_withDigest20 pub layout digest proof messages (Nat.le_refl _)

/-- For `layout20`, `Digests20` with the hash `digest` says that `digest` gives the two digests
of the proof for the last blocks: `Digests20` reads the hash. -/
theorem digests20_withDigest (layout : pub.layout = layout20 zMax rice) :
    Digests20 (pub.withDigest digest) proof messages tLast eLast ↔
      ringDigest digest eLast =
          (terminalMessagesIn (terminalState20 pub proof messages) 6).eDigest ∧
        rowsDigestOf Key20.terminal digest tLast =
          (terminalMessagesIn (terminalState20 pub proof messages) 6).tDigest := by
  unfold Digests20
  rw [terminalState20_withDigest pub digest proof messages layout]
  exact Iff.rfl

/-- For `layout20`, `Last20` reads the hash only in its two digest equations: the bound of the
response and the checks of the terminal are those of the public input with each other hash. -/
theorem last20_withDigest (layout : pub.layout = layout20 zMax rice) :
    Last20 (E := E) (pub.withDigest digest) proof messages tLast eLast ↔
      ringDigest digest eLast =
          (terminalMessagesIn (terminalState20 pub proof messages) 6).eDigest ∧
        rowsDigestOf Key20.terminal digest tLast =
          (terminalMessagesIn (terminalState20 pub proof messages) 6).tDigest ∧
        (Extract.Route.euclidean Key20.normTerminal.operator Key20.normTerminal.limit).Short 0
          (zSource Key20.terminal (response20 pub proof messages)) ∧
        (terminalStatement20 (E := E) pub proof messages tLast).Accepts
          (partialsOf Key20.terminal
            (terminalMessagesIn (terminalState20 pub proof messages) 6) eLast)
          (challenge20L pub proof messages)
          (zSource Key20.terminal (response20 pub proof messages)) := by
  have state := terminalState20_withDigest pub digest proof messages layout
  have response : response20 (pub.withDigest digest) proof messages =
      response20 pub proof messages := congrArg (responseIn pub) state
  have statement : terminalStatement20 (E := E) (pub.withDigest digest) proof messages tLast =
      terminalStatement20 pub proof messages tLast :=
    congrArg (fun s => terminalStatementIn pub s tLast) state
  have challenge : challenge20L (pub.withDigest digest) proof messages =
      challenge20L pub proof messages :=
    congrArg (fun s => challengesOf Key20.terminal (terminalChallengesIn s 6)) state
  unfold Last20
  rw [state, response, statement, challenge]
  exact Iff.rfl

end Last

end NightstreamFPrime.Spec.Compression.Akita.Replay
