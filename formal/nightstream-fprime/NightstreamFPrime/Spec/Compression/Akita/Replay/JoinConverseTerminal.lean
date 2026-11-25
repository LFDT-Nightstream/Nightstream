import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinTerminal

/-! Theorem B of the join, direction "←", at the terminal of `key20.aks` (Akita revision
`d50f700`, `fold/terminal.rs`, `VerifierFoldDraw`; ePrint 2026/1983, revision of 2026-11-23,
Section 8.2 with (127) to (131); `step_terminal_accepts` of `Replay/TerminalJoin.lean`).

- `coefficient_ne_zero`, `pairs_nonzero`, `sample_nonzero`, `sparseValid_of`,
  `foldChallenges_valid`: the fold challenges that the sampler of the code gives for a group of
  stride 1 are as many as the coordinates of the group, and each one is a valid sparse challenge
  of the terminal check (`sparseValid`).
- `Payload20`: the payload of `z` of the terminal decodes. The model does not have this check:
  `response20` is 0 for a payload that does not decode, and the code rejects such a payload
  (`terminalRun_decode`). `payload20_guard`: an accepted proof has `Payload20`.
- `Digests20`: the digests of the proof are the digests of the last blocks `tLast` and `eLast`.
  `digests20_of_accept`: an accepted proof has such last blocks. `last20_of_passed`: with `Digests20`, on a transcript that presents the bytes, the checks
  `Terminal.Passed` of the model give `Last20`.
- `terminal20_of_last`: after a run without a stop before the terminal, with `Payload20` and
  `Last20`, `terminalRun` accepts on the state before the terminal. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

section Sampler

theorem coefficient_ne_zero (ones signs i : ℕ) : coefficient ones signs i ≠ 0 := by
  unfold coefficient
  split_ifs <;> decide

theorem pairs_nonzero (ones signs : ℕ) (positions : List ℕ) :
    ∀ x ∈ pairs ones signs positions, x.2 ≠ 0 := by
  intro x member
  unfold pairs at member
  obtain ⟨i, inside, rfl⟩ := List.mem_mapIdx.mp member
  exact coefficient_ne_zero _ _ _

/-- Each coefficient of a challenge of the sampler is not zero. -/
theorem sample_nonzero {G : FoldGroup} {bytes : ByteArray} {cs : List Sparse}
    (sampled : sample G bytes = some cs) : ∀ c ∈ cs, ∀ x ∈ c, x.2 ≠ 0 := by
  have each : ∀ {value : ℕ} {c : Sparse}, candidate G Nat.choose value = some c →
      ∀ x ∈ c, x.2 ≠ 0 := by
    intro value c found
    obtain ⟨ones, twos, rfl, -, -⟩ := candidate_shape found
    exact pairs_nonzero _ _ _
  cases filtered : G.filtered with
  | true =>
      obtain ⟨rfl, -⟩ := sample_filtered filtered sampled
      intro c member
      obtain ⟨j, -, decoded⟩ := List.mem_filterMap.mp (List.mem_of_mem_take member)
      exact each (Option.filter_eq_some_iff.mp decoded).1
  | false =>
      simp only [sample, filtered, Bool.false_eq_true, ite_false, candidate_table] at sampled
      split_ifs at sampled
      cases sampled
      intro c member
      obtain ⟨j, -, found⟩ := List.mem_filterMap.mp member
      exact each found

/-- A valid sparse challenge with no zero coefficient passes `sparseValid` of the terminal. -/
theorem sparseValid_of {d : ℕ} {c : Sparse} (valid : Sparse.Valid d c)
    (nonzero : ∀ x ∈ c, x.2 ≠ 0) : sparseValid d c = true := by
  unfold sparseValid
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq]
  exact ⟨fun x member => ⟨valid.bounded x member, nonzero x member⟩, valid.distinct⟩

/-- The fold challenges of the sampler of the code for a group of stride 1: one for each
coordinate of the group, and each one passes `sparseValid` of the degree of the group. -/
theorem foldChallenges_valid {G : FoldGroup} (positive : 0 < G.degree) (same : G.ring = G.degree)
    {bytes : ByteArray} {cs : List Sparse} (drawn : foldChallenges G bytes = some cs) :
    cs.length = G.coordinates ∧ cs.all (sparseValid G.degree) = true := by
  have : NeZero G.degree := ⟨Nat.ne_of_gt positive⟩
  obtain ⟨cs₀, sampled, rfl⟩ := Option.map_eq_some_iff.mp drawn
  rw [List.map_congr_left fun c _ => ambient_one positive same c, List.map_id']
  exact ⟨(sample_mem sampled).1, List.all_eq_true.mpr fun c member =>
    sparseValid_of (sample_valid sampled c member) (sample_nonzero sampled c member)⟩

end Sampler

/-- The payload of `z` of the terminal of the proof decodes. The code rejects a payload that does
not decode (`terminal.z.decode`); the response `response20` of the model is 0 for it. -/
def Payload20 (pub : Public) (proof : ByteArray) (messages : Array Message) : Prop :=
  Golomb.decode pub.layout.terminal.params
    (terminalMessagesIn (terminalState20 pub proof messages) 6).payload ≠ none

/-- An accepted proof of `layout20` has `Payload20`. -/
theorem payload20_guard {pub : Public} {proof : ByteArray} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest)) :
    Payload20 pub proof messages := by
  have found := payload20_of_accept layout accepted terminalSetup decoded
  unfold Payload20
  rw [layout, found]
  exact Option.some_ne_none _

/-- The digests of the proof are the digests of the last blocks `tLast` of `t` and `eLast` of `e`
(difference (b): the proof has only the digests of these blocks). -/
def Digests20 (pub : Public) (proof : ByteArray) (messages : Array Message)
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) : Prop :=
  ringDigest pub.digest eLast =
      (terminalMessagesIn (terminalState20 pub proof messages) 6).eDigest ∧
    rowsDigestOf Key20.terminal pub.digest tLast =
      (terminalMessagesIn (terminalState20 pub proof messages) 6).tDigest

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- `Last20` has `Digests20`. -/
theorem Last20.digests {pub : Public} {proof : ByteArray} {messages : Array Message}
    {tLast : ℕ → Rq Key20.terminal.innerDegree} {eLast : Rq Key20.terminal.innerDegree}
    (last : Last20 (E := E) pub proof messages tLast eLast) :
    Digests20 pub proof messages tLast eLast :=
  ⟨last.1, last.2.1⟩

/-- An accepted proof of `layout20` has last blocks with `Digests20`. -/
theorem digests20_of_accept (K : Type) [Field K] [Fintype K] [DecidableEq K] [Algebra Zq K]
    [Degree2 K] {pub : Public} {proof : ByteArray} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest)) :
    ∃ tLast eLast, Digests20 pub proof messages tLast eLast :=
  let ⟨tLast, eLast, last⟩ := last20_of_accept (E := K) layout accepted terminalSetup decoded
  ⟨tLast, eLast, last.digests⟩

variable {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem B "←" at the terminal, the checks of the model: on a transcript that presents the
bytes of a proof, with the digests `Digests20` of the last blocks, the checks `Terminal.Passed` of
the model give `Last20`: the bound of the response and the checks (128) to (130) for the last
blocks, the fold challenges of the code and the response of the proof. -/
theorem last20_of_passed (digests : Digests20 pub proof messages tLast eLast)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point)
    (passed : o.terminal.Passed point) : Last20 (E := E) pub proof messages tLast eLast := by
  obtain ⟨reply, answered, decodedAt⟩ := presents.terminal
  subst tower
  have challenged := (Extract.Terminal.fold_of_reply _ answered decodedAt).1
  obtain ⟨-, accepts, short, -⟩ := passed
  refine ⟨digests.1, digests.2, ?_, ?_⟩
  · have short' : Extract.Route.Short _ _ ((point.wire (472 + 1)).data.response) := short
    rw [show ∀ k, k = 473 → (point.wire k).data.response =
        (note20 (E := E) pub proof messages tLast eLast).response from
      fun k at_ => by subst at_; exact presents.reads.response] at short'
    · exact short'
    · rfl
  · rw [challenged] at accepts
    have accepts' : ({ ((point.take 472).wire 472).data.statement with
        matrix := matrixOf Key20.terminal (full20 pub) }).Accepts
      ((point.take 472).wire 472).data.partials (challenge20L pub proof messages)
      ((point.wire (472 + 1)).data.response) := accepts
    rw [Prefix.wire_take le_rfl, presents.reads.statement, presents.reads.partials,
      show ∀ k, k = 473 → (point.wire k).data.response =
          (note20 (E := E) pub proof messages tLast eLast).response from
        fun k at_ => by subst at_; exact presents.reads.response] at accepts'
    · exact accepts'
    · rfl

omit [Fintype Coord] [DecidableEq Coord] in
/-- Theorem B "←" at the terminal (Akita Section 8.2, (128) to (131)): for a script of `layout20`,
after a run without a stop before the terminal, when the payload of `z` decodes and the last
blocks have `Last20`, `terminalRun` of the code accepts on the state before the terminal. The fold
challenges of the terminal are valid because the sampler of the code drew them
(`foldChallenges_valid`). -/
theorem terminal20_of_last {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (aligned : ReadsAligned messages pub.ops.zipIdx.toList 0)
    (payload : Payload20 pub proof messages)
    (last : Last20 (E := E) pub proof messages tLast eLast)
    (before : (runFrom pub proof messages (pub.ops.zipIdx.toList.take (pub.ops.size - 1))
      (channelStart pub)).1 = .ok ()) :
    ∃ out, terminalRun pub.layout.terminal.shape pub.digest pub.layout.terminal.limit
      pub.layout.terminal.params full
      (terminalMessagesIn (terminalState20 pub proof messages) 6)
      (terminalChallengesIn (terminalState20 pub proof messages) 6) = .ok out := by
  have scripted' : script (layout20 zMax rice) pub.plan = .ok pub.ops := layout ▸ scripted
  obtain ⟨bytes, drawn⟩ := drawn20_before layout scripted before
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted', skeleton20]
  have size : pub.ops.size = 1522 := by rw [size_skeleton erased, skeleton20_length]
  have tensor := (script20_frames scripted').2.2.2.2.2.2.1
  obtain ⟨z, payloadAt⟩ := Option.ne_none_iff_exists'.mp payload
  have response : response20 pub proof messages = z := by simp [response20, responseIn, payloadAt]
  have fullEq : full20 pub = full := by simp [full20, terminalSetup]
  subst fullEq
  obtain ⟨eDigest, tDigest, short, accepts⟩ := last
  rw [response] at short accepts
  obtain ⟨ran, stepped⟩ := before_step pub proof messages tensor.at_ (by rw [size]; decide) before
  obtain ⟨-, width, length, long, blocks⟩ := key20_segments _ (List.mem_cons_of_mem _
    (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self))))
  have linked := (terminalLink_before (E := E) (S := Key20.terminal) pub proof aligned tensor.rounds
    ran stepped tensor.after tensor.kept
    (And.right (And.right (And.right (And.right Key20.quarters)))) width length long blocks
    (full20 pub) tLast).2.2.2.2
  rw [layout] at payloadAt ⊢
  have norm : centeredNormSq z ≤ (layout20 zMax rice).terminal.limit :=
    (decoded_normSq Key20.terminal payloadAt rfl _).mpr (by
      rw [responseOf_decoded Key20.terminal payloadAt]
      exact short)
  obtain ⟨count, valid⟩ := foldChallenges_valid (G := ⟨Key20.terminal.blocks, 64, 64, 31, 11, true⟩)
    (by decide) rfl drawn
  have member : (drawnIn (terminalState20 pub proof messages) 6 0).getD
      (Key20.terminal.blocks - 1) [] ∈ drawnIn (terminalState20 pub proof messages) 6 0 := by
    rw [List.getD_eq_getElem _ _ (by rw [count]; decide)]
    exact List.getElem_mem _
  exact (terminalRun_ok_iff (E := E) (S := Key20.terminal) (by decide)
    (And.right (And.right (And.right (And.right Key20.quarters))))
    (cast_ne_zero (by decide) (by decide)) (by decide) _ _ _ _ _ payloadAt).mpr
    ⟨norm, ⟨count, valid⟩,
      isUnit_drawn (G := ⟨Key20.terminal.blocks, 64, 64, 31, 11, true⟩) (d := Key20.terminal.innerDegree) 5
        (by decide) rfl (by decide) rfl drawn member,
      eLast, tLast, eDigest, tDigest, by
        rw [show statementOf (E := E) Key20.terminal
          (And.right (And.right (And.right (And.right Key20.quarters)))) (full20 pub)
          (terminalMessagesIn (terminalState20 pub proof messages) 6)
          (terminalChallengesIn (terminalState20 pub proof messages) 6) tLast = _ from linked]
        exact accepts⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
