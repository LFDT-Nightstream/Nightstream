import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinPresent

/-! Theorem B of the join, direction "→", for `key20.aks` (Akita revision `d50f700`; ePrint
2026/1983, revision of 2026-11-23, Figure 4, Section 5.4 and Section 8.2;
`docs/akita-work/w16/design.md` section 2): on a transcript that presents the bytes of an accepted
proof, the verifier `Key20.Tower.accepts` of the tower data `tower20` accepts, one conjunct of
`towerLinks` at a time.

- `stepOk_of_accept`: each step of the replay of an accepted proof has no stop.
- `ofFn_getD_map`: the points of the public groups as lists of `E`.
- `path20_zero`, `zeroth_numbers`: the path and the numbers of the root fold of the model.
- `presents20_root`: the stage 0 and the link `RootLinked` of the batching root.
- `presents20_level0`: the link `Tier.Linked` from the root fold to the level 1. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

/-- Each step of the replay of an accepted proof has no stop. -/
theorem stepOk_of_accept {pub : Public} {proof : ByteArray}
    (scripted : script pub.layout pub.plan = .ok pub.ops) (ends : endsFit pub.ops.zipIdx.toList = true)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest)) (c : ℕ) :
    StepOk pub proof messages c := fun _ at_ =>
  (replay_step_ok pub proof messages
    (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 at_).2

/-- A list of `n` elements as the function of its entries. -/
theorem ofFn_getD_map {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {l : List Ext} {n : ℕ}
    (length : l.length = n) :
    List.ofFn (fun i : Fin n => ((l.getD i Ext.zero).into : E)) = l.map Ext.into := by
  subst length
  refine List.ext_getElem (by simp) fun i inside _ => ?_
  simp [List.getD_eq_getElem?_getD]

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]
  {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

omit [Fintype E] [DecidableEq E] [Fintype Coord] [DecidableEq Coord] in
/-- The path of the root fold is the path of its fold. -/
theorem path20_zero (pub : Public) (proof : ByteArray) (messages : Array Message) :
    path20 pub proof messages E 0 = foldPathIn E pub (fused20 pub proof messages 0) 0 (fold20 0)
      (setup20 pub 0) (fold20 0).vars (fold20 0).batch :=
  List.append_nil _

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "zeroth" =>
  Extract.liftBatch o.β o.key o.openKey o.claim Key20.rootVars (And.left Key20.root_sized) 8
    Key20.quotientDigits Key20.root_room Key20.rootBatch (And.left (And.right Key20.root_sized))
    o.block (Key20.rootDecode E) o.read Key20.rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (Extract.liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ Key20.level₁.vars
    (And.left Key20.level₁_sized) Key20.quotientDigits (And.left (And.right Key20.level₁_sized))
    Key20.level₁.batch (And.left (And.right (And.right Key20.level₁_sized))) o.block₁ code₁ o.read
    51 (Tier.finish zeroth)) Key20.level₁.tail

omit [Degree2 E] [Fintype Coord] [DecidableEq Coord] in
/-- The numbers of the root fold of the model: its fold stage 1, its 49 check stages, its end 51,
and its work bits are those of `bits20` at the stages 1 to 50. -/
theorem zeroth_numbers : (zeroth).start = 1 ∧ (zeroth).costs.length = 49 ∧ (zeroth).finish = 51 ∧
    (zeroth).foldBits = bits20.flatten.getD 1 0 ∧
    ∀ index < 49, (zeroth).bits.getD index 0 = bits20.flatten.getD (1 + 1 + index) 0 := by
  simp only [Extract.liftBatch, Tier.finish, Extract.foldCosts_length]
  decide

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem B "→" at the batching root: on a transcript that presents the bytes of an accepted
proof of `layout20`, the transcript has the stage 0 and the link `RootLinked` of `towerLinks`
holds (Akita Section 5.4, (72) with `φ = 1`): the payloads, the weights `ϑ` of the code, the target
`Σ_g ϑ_g·v_g`, the segments of the point of each group and the zero work bits of the stage 0.
`points` and `values`: the shape of the public statement. -/
theorem presents20_root {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    0 < point.rest.length ∧
      Extract.RootLinked (zeroth) 0 o.commit o.value o.place
        ((Extract.claimsRoot (Coord := Coord) (Slot := CodeSlot E) (total := total) 0 o.read
          o.value o.place).weights point) point := by
  subst tower
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨frame, -⟩ := verifyProof_frames20 layout accepted
  obtain ⟨at_, -, -, -⟩ := levels 0
  have linked := verifyProof_key20_rootLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded at_ (layout20_fold zMax rice 0) frame.weights (fun g => 3 + g.val) frame.pointAt
    frame.pointFresh frame.pointAfter frame.pointKept frame.openingsAt frame.openingsFresh
    frame.openingsAfter frame.openingsKept points values (setup20 pub 0)
  have data := presents.data
  have weighed := presents.weights
  refine ⟨by have := presents.long; omega, ?_, ?_, ?_, fun g => ?_, ?_⟩
  · show ((point.wire 1).data.claim).payload = (point.wire 0).data.commit
    rw [data, data]
    rfl
  · rw [weighed]
    show ((point.wire 1).data.claim).weight = _
    rw [data]
    rfl
  · rw [weighed]
    show ((point.wire 1).data.claim).value = ∑ g, _ * (point.wire 0).data.value g
    rw [data, data]
    rfl
  · obtain ⟨blocks, length, stride, degree, segments⟩ := linked.1 g
    refine ⟨_, _, _, _, blocks, length, stride, degree, ?_, ?_⟩
    · show ((point.wire 1).data.claim).point g = _
      rw [data]
      rfl
    · show Key20.Note.place (point.wire 0).data g = _
      rw [data, segments]
      exact ofFn_getD_map (by rw [Array.length_toList]; exact points g)
  · obtain ⟨⟨reply, moved⟩, found⟩ : ∃ entry, point.rest[0]? = some entry :=
      Option.isSome_iff_exists.mp (by
        rw [List.getElem?_eq_getElem (by have := presents.long; omega)]
        rfl)
    have answered : point.reply 0 = some reply := by simp [Prefix.reply, found]
    refine ⟨reply, answered, ?_⟩
    have worked := (presents.work 0 (by decide) reply answered).mpr
      (stepOk_of_accept scripted ends terminalSetup accepted decoded _)
    exact worked

/-- Theorem B "→" at the root fold (the level 0 of `towerLinks`, Akita Figure 4, steps 3 and 5): on
a transcript that presents the bytes of an accepted proof of `layout20`, the link `Tier.Linked`
from the root fold to the level 1 holds: the stages, the fold challenges and the work bits, the
game of the checks on the path of the code to the recorded `(fused.point, witness_evaluation)`,
and the claim of the level 1 from it. `expanded`: the setup prefix of the root fold exists. -/
theorem presents20_level0 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (expanded : (pub.setupFor (fold20 0).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    Tier.Linked (zeroth) (first) o.commit₁ point := by
  obtain ⟨started, length, finished, foldBits, bits⟩ := zeroth_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₀
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  have data := presents.data
  subst tower
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨rootFrame, frame, -⟩ := verifyProof_frames20 layout accepted
  have at_ : pub.ops[fusedIndex pub.layout 0]? =
      some (.check ⟨0, "fused.end", none⟩ (.fused 0 (fold20 0))) := (levels 0).at_
  have game : gameShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0)) 0 (fold20 0) = true :=
    (levels 0).game
  have found : pub.setupFor (fold20 0).setupCount = some (setup20 pub 0) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have rooted := (verifyProof_key20_rootLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded at_ (layout20_fold zMax rice 0) rootFrame.weights (fun g => 3 + g.val)
    rootFrame.pointAt rootFrame.pointFresh rootFrame.pointAfter rootFrame.pointKept
    rootFrame.openingsAt rootFrame.openingsFresh rootFrame.openingsAfter rootFrame.openingsKept
    points values (setup20 pub 0)).2
  have linked := verifyProof_key20_packLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded at_ (layout20_fold zMax rice 0) game frame.pointAt frame.openingsAt frame.pointBefore
    frame.openingsBefore frame.pointAfter frame.openingsAfter frame.pointHeld frame.openingsHeld
    frame.pointFresh frame.openingsFresh frame.pointKept frame.openingsKept
    (layout20_fold zMax rice 1) (setup20 pub 0)
  obtain ⟨fits, room, rowsFit, runs⟩ := verifyProof_key20_root_of (E := E) pub proof layout
    accepted terminalSetup decoded rfl found (fun _ => basis20 E) (fun _ t => basis20_apply t)
    (fun g i => challenge20₀ pub proof messages ⟨g, i⟩) (fun k i => by
      simp only [challenge20₀, Equiv.symm_apply_apply]) 0
  dsimp only at linked
  obtain ⟨segments, blocks, positioned, strided, degreed, -⟩ := linked
  refine ⟨by rw [finished]; have := presents.long; omega, ⟨⟨reply, answered, ?_, fun b => ?_⟩,
    fun index inside other found' => ?_, ?_⟩,
    ((recordedIn (fused20 pub proof messages 0) ⟨0, "fused.point", none⟩).toList.drop
      (4 + 6 + 7)).map Ext.into,
    (((recordedIn (fused20 pub proof messages 0) ⟨0, "fused.point", none⟩).toList.drop
      (4 + 6)).take 7).map Ext.into,
    ((recordedIn (fused20 pub proof messages 0) ⟨0, "fused.point", none⟩).toList.take 4).map
      Ext.into,
    (((recordedIn (fused20 pub proof messages 0) ⟨0, "fused.point", none⟩).toList.drop 4).take
      6).map Ext.into,
    (firstIn (fused20 pub proof messages 0) ⟨0, "witness_evaluation", none⟩).into, ?_, blocks,
    positioned, strided, degreed, ?_, ?_, ?_⟩
  · rw [foldBits]
    exact (presents.work 1 (by decide) reply answered).mpr (ok _)
  · rw [decodedAt b]
    rfl
  · rw [length] at inside
    rw [bits index inside]
    exact (presents.work (1 + 1 + index) (by omega) other found').mpr (ok _)
  · exact (presents.nonce 0 (by decide)).mpr (ok _)
  · have rooted' : batchStatementFor (E := E) Key20.root (setup20 pub 0)
        (claimPointIn (fused20 pub proof messages 0) 0) (dataIn pub (fused20 pub proof messages 0) 0)
        (fun k => .root (groupOrder 3 k).val) (rowsIn (fused20 pub proof messages 0) 0 (fold20 0))
        (claimsIn (fused20 pub proof messages 0) 0 (fold20 0)) =
        ((point.wire 1).data.claim).statement
          (fun g => Key.ofSetup (flatOf (setup20 pub 0)) (Key20.root.shape g))
          ⟨Setup.shared (flatOf (setup20 pub 0)) Key20.root.openDegree Key20.root.openRank _
            Key20.root.openSetupCode, PathKey.ofSetup (flatOf (setup20 pub 0)) Key20.root.chain⟩ := by
      rw [data]
      exact rooted
    rw [rooted'] at runs
    rw [path20_zero] at presented
    refine (Extract.Tier.game_eq_pure_iff_of_runs _ 0 point presented (result :=
      some ((recordedIn (fused20 pub proof messages 0) ⟨0, "fused.point", none⟩).toList.map Ext.into,
        (firstIn (fused20 pub proof messages 0) ⟨0, "witness_evaluation", none⟩).into)) ?_ _).mpr ?_
    · unfold Tier.challenge
      rw [challenged]
      exact runs
    · rw [← List.map_append, ← List.map_append, ← List.map_append, point_segments]
  · show ((point.wire _).data.claim₁).payload = (point.wire _).data.commit₁
    rw [data, data]
    rfl
  · show ((point.wire _).data.claim₁).point = _
    rw [data]
    rfl
  · show ((point.wire _).data.claim₁).value = _
    rw [data]
    rfl

end NightstreamFPrime.Spec.Compression.Akita.Replay
