import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinLevels
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinDrawn

/-! Theorem B of the join, direction "→", at the level 5 and the terminal of `key20.aks`, and the
assembly (Akita revision `d50f700`, `fold/terminal.rs`; ePrint 2026/1983, revision of 2026-11-23,
Section 8.2 with (127) to (131); `Terminal.Linked` of `Extract/TraceLink.lean`, `Terminal.Passed`
of `Extract/TerminalSuffix.lean`, `Key20.Tower.accepts` of `Key20Opening.lean`).

- `Last20`: the last blocks `tLast` of `t` and `eLast` of `e` are those of the code: their
  digests are the digests of the proof, the response of the terminal has the bound of the
  Euclidean route, and the checks (128) to (130) hold for them (`verifyProof_key20_terminal`).
- `payload20_of_accept`, `last20_of_accept`: the payload of `z` of an accepted proof decodes, and
  last blocks with `Last20` exist.
- `fifth20_numbers`: the numbers of the level 5 of the model.
- `presents20_level5`: the link `Terminal.Linked` from the level 5 to the terminal.
- `presents20_passed`: the checks `Terminal.Passed` of the terminal.
- `presents20_accepts`: theorem B "→" for `key20.aks`: on a transcript of `tower20` that presents
  the bytes of an accepted proof with the last blocks `Last20`, `Key20.Tower.accepts` accepts.
- `verifyProof_presents20`: theorem B "→" with the last blocks of the code: an accepted proof has
  last blocks `Last20` such that each transcript that presents its bytes with them is accepted. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The last blocks `tLast` of `t` and `eLast` of `e` of the terminal of the code (difference (b):
the proof has only their digests): the digests are those of the proof, the response of the
terminal has `‖z‖_2² ≤ S_max,term`, and the checks (128) to (130) of `terminalStatement20` hold for
the clear partials with `eLast`, the fold challenges of the code and the response. -/
def Last20 (pub : Public) (proof : ByteArray) (messages : Array Message)
    (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree) : Prop :=
  ringDigest eLast = (terminalMessagesIn (terminalState20 pub proof messages) 6).eDigest ∧
    rowsDigestOf Key20.terminal tLast =
      (terminalMessagesIn (terminalState20 pub proof messages) 6).tDigest ∧
    (Extract.Route.euclidean Key20.normTerminal.operator Key20.normTerminal.limit).Short 0
      (zSource Key20.terminal (response20 pub proof messages)) ∧
    (terminalStatement20 (E := E) pub proof messages tLast).Accepts
      (partialsOf Key20.terminal (terminalMessagesIn (terminalState20 pub proof messages) 6) eLast)
      (challenge20L pub proof messages) (zSource Key20.terminal (response20 pub proof messages))

/-- The payload of `z` of an accepted proof of `layout20` decodes to `response20`. -/
theorem payload20_of_accept {pub : Public} {proof : ByteArray} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest)) :
    Golomb.decode (layout20 zMax rice).terminal.params
        (terminalMessagesIn (terminalState20 pub proof messages) 6).payload =
      some (response20 pub proof messages) := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have terminalAt := (verifyProof_frames20 layout accepted).2.2.2.2.2.2.2
  have stepped := (replay_step_ok pub proof messages
    (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 terminalAt).2
  rw [step_terminal_run pub proof messages _ 6 _ terminalSetup] at stepped
  rw [← layout]
  rcases found : Golomb.decode pub.layout.terminal.params
      (terminalMessagesIn (terminalState20 pub proof messages) 6).payload with _ | z
  · rw [terminalRun_decode _ _ found] at stepped
    cases stepped
  · simp [response20, responseIn, found]

/-- The last blocks of the terminal of an accepted proof of `layout20` (`verifyProof_key20_terminal`
with `verifyProof_drawn20` and `payload20_of_accept`): last blocks with `Last20` exist. -/
theorem last20_of_accept {pub : Public} {proof : ByteArray} {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} {rest : ℕ} (decoded : decode pub.ops proof = .ok (messages, rest)) :
    ∃ tLast eLast, Last20 (E := E) pub proof messages tLast eLast := by
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensor := frames.2.2.2.2.1
  have terminalAt := frames.2.2.2.2.2
  obtain ⟨bytes, drawnAt⟩ := verifyProof_drawn20 layout accepted terminalSetup decoded
  have fullEq : full20 pub = full := by simp [full20, terminalSetup]
  obtain ⟨short, -, -, -, -, -, eLast, tLast, eDigest, tDigest, accepts⟩ :=
    verifyProof_key20_terminal (E := E) pub proof scripted ends terminalSetup accepted decoded layout
      tensor.at_ tensor.rounds tensor.after tensor.kept terminalAt drawnAt
      (payload20_of_accept layout accepted terminalSetup decoded) (by decide) 0
  refine ⟨tLast, eLast, eDigest, tDigest, short, ?_⟩
  unfold terminalStatement20 terminalStatementIn
  rw [fullEq]
  exact accepts

variable {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "code₂" =>
  (CodeSlot.open Key20.open128 Key20.open128_fits :
    CodeSlot E → Option (Rq Key20.level₂.shape.innerDegree))

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

local notation "code₃" =>
  (CodeSlot.kept128 : CodeSlot E → Option (Rq Key20.level₃.shape.innerDegree))

local notation "code₄" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₄.shape.innerDegree))

local notation "code₅" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₅.shape.innerDegree))

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

omit [Fintype Coord] [DecidableEq Coord] in
/-- The numbers of the level 5 of the model: its fold stage 378, its 93 check stages (with the
tensor reduction of the terminal), its end 472, and its work bits are those of `bits20` at the
stages 378 to 471. -/
theorem fifth20_numbers : (fifth).start = 378 ∧ (fifth).costs.length = 93 ∧ (fifth).finish = 472 ∧
    (fifth).foldBits = bits20.flatten.getD 378 0 ∧
    ∀ index < 93, (fifth).bits.getD index 0 = bits20.flatten.getD (378 + 1 + index) 0 := by
  simp only [Extract.liftBatch, Extract.liftFold, Extract.committedTrace, Extract.committedTraceSq,
    Tier.tensored, Tier.finish, Extract.foldCosts_length, Extract.foldCostsSq_length,
    List.length_append, Extract.tensorCosts, List.length_cons, List.length_replicate]
  decide +kernel

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem B "→" at the level 5 (Akita Section 8.2): on a transcript that presents the bytes of
an accepted proof of `layout20`, the link `Terminal.Linked` from the level 5 with the tensor
reduction of its claim to the terminal holds. `expanded`: the setup prefix of the level 5
exists. -/
theorem presents20_level5 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 5).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    Extract.Terminal.Linked (fifth) o.terminal
      (And.right (And.right (And.right (And.right Key20.quarters)))) o.stateOf point := by
  obtain ⟨started, length, finished, foldBits, bits⟩ := fifth20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₅
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.2.2.2.1
  have tensor := frames.2.2.2.2.1
  have found : pub.setupFor (fold20 5).setupCount = some (setup20 pub 5) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
      List.mem_cons_self))) tensorBefore.at_ tensorBefore.rounds tensorBefore.after
    tensorBefore.kept (layout20_fold zMax rice 5) (by decide) (setup20 pub 5)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_traceSq_of (E := E) 5 Key20.level₅
    (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)) pub proof layout accepted
    terminalSetup decoded rfl found (challengeTrace20 pub proof messages 5 Key20.level₅)
    (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  obtain ⟨-, width, positionLength, long, blocksFit⟩ := key20_segments _
    (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
      (List.mem_cons_of_mem _ List.mem_cons_self))))
  have linked := verifyProof_terminalLink (E := E) (S := Key20.terminal) pub proof scripted ends
    terminalSetup accepted decoded tensor.at_ tensor.rounds tensor.after tensor.kept (by decide)
    width positionLength long blocksFit (full20 pub) tLast
  dsimp only at tensored linked before
  obtain ⟨-, -, runs'⟩ := tensored
  obtain ⟨segments, blocks, positioned, packedLength, -⟩ := linked
  have before' : traceStatementFor (E := E) Key20.level₅.opening quarter (setup20 pub 5)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ) (fold20 5))
      (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ) (fold20 5))
      (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((5 : Fin 6) : ℕ)))
        ((5 : Fin 6) : ℕ) (fold20 5)) =
      ((point.wire 378).data.claim₅).statement (Key.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 5)) Key20.level₅.opening) := by
    rw [presents.reads.claim₅]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 5 (by decide)] at presented
  have receiver : receiver20 pub ((5 : Fin 6) : ℕ) = pub.ops.size - 1 := rfl
  rw [receiver] at presented
  refine ⟨by rw [finished]; have := presents.long; omega, ⟨⟨reply, answered, ?_, fun b => ?_⟩,
    fun index inside other found' => ?_, ?_⟩, _, _, _,
    mle (((recordedIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
        ⟨5 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
      (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
        ⟨5, "fused.point", none⟩).toList.tail).map Ext.into)
        (firstIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
          ⟨5 + 1, "tensor.eta", none⟩).into),
    (firstIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
      ⟨5 + 1, "tensor.final_claims", none⟩).into, ?_, blocks, positioned, packedLength, ?_⟩
  · rw [foldBits]
    exact (presents.work 378 (by omega) reply (started ▸ answered)).mpr (ok _)
  · rw [decodedAt b]
    rfl
  · rw [length] at inside
    rw [bits index inside]
    rw [started] at found'
    exact (presents.work (378 + 1 + index) (by omega) other found').mpr (ok _)
  · rw [started]
    exact (presents.nonce 5 (by decide)).mpr (ok _)
  · refine (Extract.Tier.game_eq_pure_iff_of_runs _ 0 point presented (result :=
      some (((recordedIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
          ⟨5 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into,
        mle (((recordedIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
            ⟨5 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
          (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
            ⟨5, "fused.point", none⟩).toList.tail).map Ext.into)
            (firstIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
              ⟨5 + 1, "tensor.eta", none⟩).into),
        (firstIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1))
          ⟨5 + 1, "tensor.final_claims", none⟩).into)) ?_ _).mpr ?_
    · unfold Tier.challenge
      rw [challenged, started]
      exact runs'
    · rw [segments]
  · show { ((point.take 472).wire 472).data.statement with
        matrix := matrixOf Key20.terminal (full20 pub) } = _
    rw [Prefix.wire_take le_rfl, presents.reads.statement]
    show _ = Extract.TerminalStatement.ofPoint _ _ ((point.wire _).data.state) _ _ _ _ _
    rw [show ∀ k, k = 379 → (point.wire k).data.state = (note20 (E := E) pub proof messages tLast eLast).state from
      fun k at_ => by subst at_; exact presents.reads.state]
    · rfl
    · exact congrArg (· + 1) started

/-- Theorem B "→" at the terminal (Akita Section 8.2, (128) to (131)): on a transcript that
presents the bytes of an accepted proof of `layout20` with the last blocks `Last20` of the code,
the checks `Terminal.Passed` of the terminal hold: the work bits and the fold challenges of its
fold stage, the checks (128) to (130), the bound of the response, and the response nonce. -/
theorem presents20_passed {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (last : Last20 (E := E) pub proof messages tLast eLast)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) : o.terminal.Passed point := by
  obtain ⟨reply, answered, decodedAt⟩ := presents.terminal
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  obtain ⟨-, -, short, accepts⟩ := last
  have challenged := (Extract.Terminal.fold_of_reply _ answered decodedAt).1
  have bitsEq : Extract.Terminal.foldBits (tower20 (E := E) pub block block₁ block₂ block₃ block₄
      block₅ blockL total 472).terminal = bits20.flatten.getD 472 0 := by
    simp only [Extract.Terminal.foldBits]
    decide
  refine ⟨⟨reply, answered, ?_, fun i => ?_⟩, ?_, ?_, ?_⟩
  · rw [bitsEq]
    exact (presents.work 472 (by omega) reply answered).mpr (ok _)
  · show (Extract.Terminal.decode _ (reply (some (Extract.Terminal.block _ i)))).isSome
    rw [decodedAt i]
    rfl
  · rw [challenged]
    show ({ ((point.take 472).wire 472).data.statement with
        matrix := matrixOf Key20.terminal (full20 pub) }).Accepts
      ((point.take 472).wire 472).data.partials _ ((point.wire (472 + 1)).data.response)
    rw [Prefix.wire_take le_rfl, presents.reads.statement, presents.reads.partials,
      show ∀ k, k = 473 → (point.wire k).data.response = (note20 (E := E) pub proof messages tLast eLast).response from
        fun k at_ => by subst at_; exact presents.reads.response]
    · exact accepts
    · rfl
  · show Extract.Route.Short _ _ ((point.wire (472 + 1)).data.response)
    rw [show ∀ k, k = 473 → (point.wire k).data.response = (note20 (E := E) pub proof messages tLast eLast).response from
      fun k at_ => by subst at_; exact presents.reads.response]
    · exact short
    · rfl
  · exact (presents.nonce 6 (by decide)).mpr (ok _)

/-- Theorem B of the join, direction "→", for `key20.aks` (`docs/akita-work/w16/design.md`
section 2): on each transcript of `tower20` that presents the bytes of an accepted proof of
`layout20` (`Presents20`) with the last blocks `Last20` of the terminal of the code, the verifier
`Key20.Tower.accepts` of the claims accepts. `terminalSetup` and `expanded`: the setup prefixes of
the terminal and of the folds exist. `points` and `values`: the shape of the public statement. -/
theorem presents20_accepts {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3)
    (last : Last20 (E := E) pub proof messages tLast eLast)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅
      blockL point) :
    (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts point := by
  obtain ⟨long, rooted⟩ := presents20_root _ pub proof messages tLast eLast layout accepted
    terminalSetup decoded points values rfl presents
  exact ⟨long, rooted,
    presents20_level0 _ pub proof messages tLast eLast layout accepted terminalSetup decoded points
      values (expanded 0) rfl presents,
    presents20_level1 _ pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 1) rfl presents,
    presents20_level2 _ pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 2) rfl presents,
    presents20_level3 _ pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 3) rfl presents,
    presents20_level4 _ pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 4) rfl presents,
    presents20_level5 _ pub proof messages tLast eLast layout accepted terminalSetup decoded
      (expanded 5) rfl presents,
    presents20_passed _ pub proof messages tLast eLast layout accepted terminalSetup decoded last rfl
      presents⟩

/-- Theorem B "→" for `key20.aks` with the last blocks of the code: an accepted proof of
`layout20` has last blocks `tLast`, `eLast` with `Last20` (their digests are those of the proof,
and the terminal of the code accepts with them), and for these last blocks `Key20.Tower.accepts`
of `tower20` accepts each transcript that presents the bytes (`Presents20`). The hypotheses are
the layout, the setup prefixes (facts of the seed), the decoded messages and the shape of the
public statement (`points`, `values`). -/
theorem verifyProof_presents20 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (expanded : ∀ ℓ : Fin 6, (pub.setupFor (fold20 ℓ).setupCount).isSome) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (points : ∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g)
    (values : (pub.groups.flatMap (·.value)).size = 3) :
    ∃ (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree),
      Last20 (E := E) pub proof messages tLast eLast ∧
        ∀ point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total,
          Presents20 pub proof messages tLast eLast block block₁ block₂ block₃ block₄ block₅ blockL
            point →
          (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts point := by
  obtain ⟨tLast, eLast, last⟩ := last20_of_accept (E := E) layout accepted terminalSetup decoded
  exact ⟨tLast, eLast, last, fun point presents => presents20_accepts pub proof messages tLast eLast
    layout accepted terminalSetup expanded decoded points values last presents⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
