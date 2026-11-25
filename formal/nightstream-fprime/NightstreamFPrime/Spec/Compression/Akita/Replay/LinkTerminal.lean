import NightstreamFPrime.Spec.Compression.Akita.Replay.LinkKey20
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalNorm

/-! The link from the last level of the replay to the terminal (Akita revision `d50f700`,
`fold/terminal.rs`; ePrint 2026/1983, revision of 2026-11-23, Section 8.2 with (127) to (131);
`Terminal.Linked` of `Extract/TraceLink.lean` and `Terminal.Passed` of
`Extract/TerminalSuffix.lean`).

The code runs the tensor reduction of the claim of the last level `ℓ` in `Build.terminal` of the
level `ℓ + 1`, with the keys `tensor.*` of the level `ℓ + 1`.

- `verifyProof_terminalLink`: on the state before the terminal, the statement `statementOf` of the
  replay is `TerminalStatement.ofPoint` for the segments of the tensor point `ρ`, the factor
  `θ = A_η(ρ)` and the final claim `ṽ`, with `A_term = matrixOf` of the setup and the state of the
  messages with the last block `tLast`.
- `layout20_terminal`: the terminal of `layout20`.
- `verifyProof_key20_terminal`: the terminal of `key20.aks` when `runProof` accepts: the bound
  `Route.Short` of the Euclidean route for `zSource z`, the fold challenges, and the checks (128)
  to (130) for `TerminalStatement.ofPoint` of the result of the level 5 with its tensor
  reduction.
- `terminalLink_before`: `verifyProof_terminalLink` when the run before the `tensor.end` and its
  step have no stop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

section Proof

variable (pub : Public) (proof : ByteArray) {ops : Array Op}
variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The link `Terminal.Linked` on the replay (Akita Section 8.2), when the run before the
`tensor.end` at `t` and the step of `t` have no stop: on the state before `c'` the statement
`statementOf` of the terminal at the level `level + 1` is `TerminalStatement.ofPoint` for
`A_term = matrixOf` of the setup, the state of the messages with the last block `tLast`, the
segments `ρ_pack`, `ρ_pos`, `ρ_blk` of the tensor point `ρ`, the factor `θ` and the final claim `ṽ`
of `tensorLink_before`. -/
theorem terminalLink_before {messages : Array Message}
    (aligned : ReadsAligned messages ops.zipIdx.toList 0) {level t c' ν : ℕ} {key : Key}
    (shape : roundsAt (ops.zipIdx.toList.take t) ⟨level + 1, "tensor.sumcheck", none⟩ ν 2 = true)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take t) (channelStart pub)).1 =
      .ok ())
    (stepped : (((step pub proof messages t (.check key (.tensor (level + 1) ν
      ⟨level, "fused.point", none⟩ ⟨level, "witness_evaluation", none⟩))).run).run
        (stateBefore pub proof messages ops t)).1 = .ok ())
    (after : t ≤ c')
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys level)
      [⟨level + 1, "tensor.sumcheck", none⟩] = true)
    {S : Shape} (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (width : 2 * (S.innerDegree / 4) = 2 ^ Nat.log 2 (S.innerDegree / 2))
    (length : S.blockLength = 2 ^ Nat.log 2 S.blockLength)
    (long : Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength ≤ ν)
    (blocks : S.blocks ≤ 2 ^ (ν - (Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength)))
    (setup : Array Fq) (tLast : ℕ → Rq S.innerDegree) :
    let s := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
    let ρ := (recordedIn s ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
    let η := firstIn s ⟨level + 1, "tensor.eta", none⟩
    let pack := (packPoint S ρ).map (Ext.into (E := E))
    let position := (positionPoint S ρ).map (Ext.into (E := E))
    let block := (blockPoint S ρ).map (Ext.into (E := E))
    pack ++ (position ++ block) = ρ.map Ext.into ∧ S.blocks ≤ 2 ^ block.length ∧
      S.blockLength = 2 ^ position.length ∧ 2 * (S.innerDegree / 4) = 2 ^ pack.length ∧
      statementOf (E := E) S quarter setup (terminalMessagesIn s (level + 1))
          (terminalChallengesIn s (level + 1)) tLast =
        Extract.TerminalStatement.ofPoint quarter (matrixOf S setup)
          (stateOf S (terminalMessagesIn s (level + 1)) tLast) block position pack
          (mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into))
          (firstIn s ⟨level + 1, "tensor.final_claims", none⟩).into := by
  obtain ⟨rhoLength, tailLength, -⟩ := tensorLink_before (E := E) pub proof aligned shape before
    stepped after kept (fun _ => 0)
  dsimp only at rhoLength tailLength ⊢
  obtain ⟨packLength, positionLength, blockLength⟩ := segments_length S _ (rhoLength ▸ long)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [← List.map_append, ← List.map_append, segments_append]
  · rw [List.length_map, blockLength, rhoLength]
    exact blocks
  · rw [List.length_map, positionLength]
    exact length
  · rw [List.length_map, packLength]
    exact width
  · have same : (recordedIn (stateBefore pub proof messages ops c')
        ⟨level + 1, "tensor.sumcheck.point", none⟩).toList.length =
        (recordedIn (stateBefore pub proof messages ops c')
          ⟨level + 1 - 1, "fused.point", none⟩).toList.tail.length := by
      rw [Nat.add_sub_cancel, rhoLength, tailLength]
    unfold statementOf terminalChallengesIn
    dsimp only
    rw [tensorFactor_into _ _ same, Nat.add_sub_cancel]
    rfl

/-- The link `Terminal.Linked` on the replay (Akita Section 8.2): when `runProof` accepts, on
the state before `c'` the statement `statementOf` of the terminal at the level `level + 1` is
`TerminalStatement.ofPoint` for `A_term = matrixOf` of the setup, the state of the messages with
the last block `tLast`, the segments `ρ_pack`, `ρ_pos`, `ρ_blk` of the tensor point `ρ`, the factor
`θ` and the final claim `ṽ` of `verifyProof_tensorLink`. The segments have the lengths of
`Terminal.Linked`. -/
theorem verifyProof_terminalLink (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {level t c' ν : ℕ} {key : Key}
    (at_ : ops[t]? = some (.check key (.tensor (level + 1) ν ⟨level, "fused.point", none⟩
      ⟨level, "witness_evaluation", none⟩)))
    (shape : roundsAt (ops.zipIdx.toList.take t) ⟨level + 1, "tensor.sumcheck", none⟩ ν 2 = true)
    (after : t ≤ c')
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys level)
      [⟨level + 1, "tensor.sumcheck", none⟩] = true)
    {S : Shape} (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (width : 2 * (S.innerDegree / 4) = 2 ^ Nat.log 2 (S.innerDegree / 2))
    (length : S.blockLength = 2 ^ Nat.log 2 S.blockLength)
    (long : Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength ≤ ν)
    (blocks : S.blocks ≤ 2 ^ (ν - (Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength)))
    (setup : Array Fq) (tLast : ℕ → Rq S.innerDegree) :
    let s := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
    let ρ := (recordedIn s ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
    let η := firstIn s ⟨level + 1, "tensor.eta", none⟩
    let pack := (packPoint S ρ).map (Ext.into (E := E))
    let position := (positionPoint S ρ).map (Ext.into (E := E))
    let block := (blockPoint S ρ).map (Ext.into (E := E))
    pack ++ (position ++ block) = ρ.map Ext.into ∧ S.blocks ≤ 2 ^ block.length ∧
      S.blockLength = 2 ^ position.length ∧ 2 * (S.innerDegree / 4) = 2 ^ pack.length ∧
      statementOf (E := E) S quarter setup (terminalMessagesIn s (level + 1))
          (terminalChallengesIn s (level + 1)) tLast =
        Extract.TerminalStatement.ofPoint quarter (matrixOf S setup)
          (stateOf S (terminalMessagesIn s (level + 1)) tLast) block position pack
          (mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into))
          (firstIn s ⟨level + 1, "tensor.final_claims", none⟩).into := by
  obtain ⟨before, stepped⟩ := replay_step_ok pub proof messages
    (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 at_
  exact terminalLink_before pub proof (decode_aligned decoded) shape before stepped after kept
    quarter width length long blocks setup tLast

end Proof

/-- The terminal of `layout20`: the shape `Key20.terminal`, the rounds `level₅.tail` of its tensor
reduction, the group of `B_term` fold coordinates of degree 64 with the filtered family
`(31, 11)`, the cap `normTerminal.limit` and the `M·δ·d` coefficients of `z`. -/
theorem layout20_terminal (zMax rice : ℕ) :
    (layout20 zMax rice).terminal.shape = Key20.terminal ∧
      (layout20 zMax rice).terminal.rounds = Key20.level₅.tail ∧
      (layout20 zMax rice).terminal.group = ⟨Key20.terminal.blocks, 64, 64, 31, 11, true⟩ ∧
      (layout20 zMax rice).terminal.limit = Key20.normTerminal.limit ∧
      (layout20 zMax rice).terminal.params.count =
        Key20.terminal.blockLength * Key20.terminal.sourceDigits * Key20.terminal.innerDegree :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

section Key20

variable (pub : Public) (proof : ByteArray) {ops : Array Op}
variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The terminal of `key20.aks` on the replay (Akita Section 8.2, (127) to (131);
`Terminal.Linked` and `Terminal.Passed`): when `runProof` accepts with the layout `layout20`,
the tensor reduction of the level 6 on the claim of the level 5 at `t` and the terminal at `c'`,
for fold challenges that the sampler drew for the terminal group and a payload of `z` that
decodes: `z` has the bound `Route.Short` of the Euclidean route of the terminal
(`‖z‖_2² ≤ S_max,term`), the fold challenges are valid, the segments of the tensor point have the
lengths of `Terminal.Linked`, and the checks (128) to (130) hold for `TerminalStatement.ofPoint`
of `A_term = matrixOf` of the setup, the state of the messages with a last block `tLast` of the
digest of the proof, and the result `(ρ, θ, ṽ)` of the level 5 with its tensor reduction. -/
theorem verifyProof_key20_terminal {zMax rice : ℕ}
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest))
    (layout : pub.layout = layout20 zMax rice) {t c' : ℕ} {key : Key}
    (tensorAt : ops[t]? = some (.check key (.tensor 6 Key20.level₅.tail ⟨5, "fused.point", none⟩
      ⟨5, "witness_evaluation", none⟩)))
    (rounds : roundsAt (ops.zipIdx.toList.take t) ⟨6, "tensor.sumcheck", none⟩ Key20.level₅.tail
      2 = true)
    (after : t ≤ c')
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys 5)
      [⟨6, "tensor.sumcheck", none⟩] = true)
    (terminalAt : ops[c']? = some (.terminal 6)) {bytes : ByteArray}
    (drawn : foldChallenges (layout20 zMax rice).terminal.group bytes =
      some (drawnIn (stateBefore pub proof messages ops c') 6 0))
    {z : Array ℤ}
    (payload : Golomb.decode (layout20 zMax rice).terminal.params
      (terminalMessagesIn (stateBefore pub proof messages ops c') 6).payload = some z)
    (quarter : 4 * (Key20.terminal.innerDegree / 4) = Key20.terminal.innerDegree) (bound : ℕ) :
    let s := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨5, "fused.point", none⟩).toList
    let ρ := (recordedIn s ⟨6, "tensor.sumcheck.point", none⟩).toList
    let η := firstIn s ⟨6, "tensor.eta", none⟩
    let pack := (packPoint Key20.terminal ρ).map (Ext.into (E := E))
    let position := (positionPoint Key20.terminal ρ).map (Ext.into (E := E))
    let block := (blockPoint Key20.terminal ρ).map (Ext.into (E := E))
    (Extract.Route.euclidean Key20.normTerminal.operator Key20.normTerminal.limit).Short bound
        (zSource Key20.terminal z) ∧
      ((terminalChallengesIn s 6).fold.length = Key20.terminal.blocks ∧
        (terminalChallengesIn s 6).fold.all (sparseValid Key20.terminal.innerDegree) = true) ∧
      pack ++ (position ++ block) = ρ.map Ext.into ∧ Key20.terminal.blocks ≤ 2 ^ block.length ∧
      Key20.terminal.blockLength = 2 ^ position.length ∧
      2 * (Key20.terminal.innerDegree / 4) = 2 ^ pack.length ∧
      ∃ (eLast : Rq Key20.terminal.innerDegree) (tLast : ℕ → Rq Key20.terminal.innerDegree),
        ringDigest pub.digest eLast = (terminalMessagesIn s 6).eDigest ∧
        rowsDigestOf Key20.terminal pub.digest tLast = (terminalMessagesIn s 6).tDigest ∧
        (Extract.TerminalStatement.ofPoint quarter (matrixOf Key20.terminal full)
          (stateOf Key20.terminal (terminalMessagesIn s 6) tLast) block position pack
          (mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into))
          (firstIn s ⟨6, "tensor.final_claims", none⟩).into).Accepts
          (partialsOf Key20.terminal (terminalMessagesIn s 6) eLast)
          (challengesOf Key20.terminal (terminalChallengesIn s 6)) (zSource Key20.terminal z) := by
  cases pub with
  | mk layout' plan session descriptor groups slot setupRows seed setup =>
  subst layout
  have numbers := key20_segments _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
    (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self))))
  obtain ⟨-, width, length, long, blocks⟩ := numbers
  obtain ⟨small, challenges, eLast, tLast, eDigest, tDigest, accepts⟩ :=
    verifyProof_terminal_accepts (E := E) _ proof scripted ends terminalSetup accepted decoded
      terminalAt rfl 5 (by show 2 ^ (5 + 1) = Key20.terminal.innerDegree; decide) quarter
      (cast_ne_zero (by show 0 < 2 * (Key20.terminal.innerDegree / 4); decide)
        (by show 2 * (Key20.terminal.innerDegree / 4) < fieldModulus; decide))
      (by show 1 ≤ Key20.terminal.blocks; decide) rfl (by show 0 < 31; decide) rfl drawn payload
  obtain ⟨segments, blocked, positioned, packed, linked⟩ := verifyProof_terminalLink (E := E)
    (S := Key20.terminal) _ proof scripted ends terminalSetup accepted decoded tensorAt rounds after
    kept quarter width length long blocks full tLast
  have short : vecNormSq (zSource Key20.terminal z) ≤ Key20.normTerminal.limit := by
    rw [← responseOf_decoded Key20.terminal payload]
    exact (decoded_normSq Key20.terminal payload rfl _).mp small
  refine ⟨short, challenges, segments, blocked, positioned, packed, eLast, tLast, eDigest, tDigest,
    ?_⟩
  rw [← linked]
  exact accepts

end Key20

end NightstreamFPrime.Spec.Compression.Akita.Replay
