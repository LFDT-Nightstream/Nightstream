import NightstreamFPrime.Spec.Compression.Akita.Replay.RunStop

/-! The verifier on the public inputs and the bytes of an Akita opening proof, and its theorems
against the level games on the path of the code (Akita revision `d50f700`).

- `verifyProof pub proof` is the verifier: `reject 0 "statement"` when the statement does not have
  the shape of the schedule row (`Public.shaped`), and otherwise `runProof` on the public input
  without the setup prefix of the caller (`Public.sealed`). `verifyProof_accept_iff`.
- `runProof pub proof` is total: `reject 0 "plan"` when the layout does not take the plan
  (`script` fails; `VerifierGrinding::finish` of the code), and otherwise `verifyBytes` with the
  verdict of the replay of the script.
- `verifyProof_malformed`, `verifyProof_trailing`, `verifyProof_noWork`, `verifyProof_accept` and
  `verifyProof_reject` are the five statements of section 5.1 for `ops = script …`. `endsFit` of
  the script and the setup prefix of the terminal exclude a `mismatch` stop (`replay_matches`);
  `verdictOf` gives a `mismatch` stop the verdict `mismatch`, which rejects.
- `verifyProof_accept_of` is the converse of `verifyProof_accept`: the proof decodes with no byte
  left, the works hold, each check holds on the state before it and the terminal accepts.
- `verifyProof_trace_game`, `verifyProof_traceSq_game` and `verifyProof_tensor_game`: when
  `runProof` accepts, the game of the specification level (and of the tensor reduction) runs
  on the path of the recorded values to acceptance; the packing levels are in
  `Replay/FinalGame.lean`. `verifyProof_level`: the state hypotheses of
  the games and the level function at `fused.end`. `verifyProof_terminal_accepts`: the terminal
  statement of the specification accepts.
- `verifyProof_reject_fused`, `verifyProof_reject_level`, `verifyProof_reject_trace_game`,
  `verifyProof_reject_range`, `verifyProof_reject_energy`, `verifyProof_reject_stage3`,
  `verifyProof_reject_opening` and `verifyProof_reject_tensor_game`: at a reject of `fused.end`,
  `range.leaf.end` or `norm.end`, `norm.integer`, `stage3.end`, `tensor.opening` and `tensor.end`,
  the level (or tensor) function is `none`, and its game runs to `none`.
  `foldRunIn_of_energy_reject` covers `norm.integer`: `normRun` checks the same
  `E_resp` again. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

/-- The replay of the script of the layout on the public inputs and the bytes of a proof. -/
def runProof (pub : Public) (proof : ByteArray) : Outcome :=
  match script pub.layout pub.plan with
  | .error _ => .verdict (.reject 0 "plan")
  | .ok ops => verifyBytes (verdict pub ops proof fun _ => .accept) ops proof

/-- The verifier on the public inputs and the bytes of a proof: the checks of the statement
(`Public.shaped`), and then `runProof` with the setup prefix of the seed only. -/
def verifyProof (pub : Public) (proof : ByteArray) : Outcome :=
  if pub.shaped then runProof pub.sealed proof else .verdict (.reject 0 "statement")

/-- `verifyProof` accepts exactly when the statement has the shape of the schedule row and
`runProof` accepts without the setup prefix of the caller. -/
theorem verifyProof_accept_iff (pub : Public) (proof : ByteArray) :
    verifyProof pub proof = .verdict .accept ↔
      pub.shaped = true ∧ runProof pub.sealed proof = .verdict .accept := by
  unfold verifyProof
  cases pub.shaped <;> simp

/-- A statement without the shape of the schedule row is rejected before the proof is read. -/
theorem verifyProof_unshaped (pub : Public) (proof : ByteArray) (unshaped : pub.shaped = false) :
    verifyProof pub proof = .verdict (.reject 0 "statement") := by
  simp [verifyProof, unshaped]

open Replayer

section Proof

variable (pub : Public) (proof : ByteArray) {ops : Array Op}

theorem verifyProof_plan {reason : String} (failed : script pub.layout pub.plan = .error reason) :
    runProof pub proof = .verdict (.reject 0 "plan") := by
  simp only [runProof, failed]

theorem verifyProof_script (scripted : script pub.layout pub.plan = .ok ops) :
    runProof pub proof = verifyBytes (verdict pub ops proof fun _ => .accept) ops proof := by
  simp only [runProof, scripted]

theorem verifyProof_malformed (scripted : script pub.layout pub.plan = .ok ops)
    (site : Malformed) :
    runProof pub proof = .malformed site ↔ decode ops proof = .error site := by
  rw [verifyProof_script pub proof scripted]
  exact verifyBytes_malformed _ ops proof site

/-- With `endsFit` and the setup prefix of the terminal, the verdict of the replay accepts exactly
when the replay has no stop. -/
theorem verdict_accept_iff {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) (ends : endsFit ops.zipIdx.toList = true)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) :
    verdict pub ops proof (fun _ => .accept) messages = .accept ↔
      (replay pub ops proof messages).1 = none := by
  unfold verdict verdictOf
  rcases stop : (replay pub ops proof messages).1 with _ | (_ | _ | k)
  · simp
  · simp
  · simp
  · exact absurd stop (replay_matches pub proof messages ops decoded ends terminalSetup k)

theorem verifyProof_trailing (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) :
    runProof pub proof = .trailing ↔
      ∃ messages rest, decode ops proof = .ok (messages, rest) ∧ rest ≠ 0 ∧
        (replay pub ops proof messages).1 = none := by
  rw [verifyProof_script pub proof scripted, verifyBytes_trailing]
  constructor
  · rintro ⟨messages, rest, decoded, left, accepts⟩
    exact ⟨messages, rest, decoded, left,
      (verdict_accept_iff pub proof decoded ends terminalSetup).mp accepts⟩
  · rintro ⟨messages, rest, decoded, left, none_⟩
    exact ⟨messages, rest, decoded, left,
      (verdict_accept_iff pub proof decoded ends terminalSetup).mpr none_⟩

/-- `noWork` is the outcome exactly when the proof decodes and the replay stops first at a work. -/
theorem verifyProof_noWork_iff (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} :
    runProof pub proof = .verdict (.noWork i site) ↔
      ∃ messages rest, decode ops proof = .ok (messages, rest) ∧
        (replay pub ops proof messages).1 = some (.noWork i site) := by
  rw [verifyProof_script pub proof scripted]
  unfold verifyBytes
  rcases decode ops proof with failure | ⟨messages, rest⟩
  · simp
  · unfold verdict verdictOf
    rcases stop : (replay pub ops proof messages).1 with _ | (_ | _ | _) <;>
      by_cases empty : rest = 0 <;> simp [empty, stop]

/-- A `noWork` outcome: the proof decodes, the works do not hold, and the replay stops at the
operation `i`, a work, a squeeze of challenges, a response nonce or a draw of fold challenges,
after the operations before it without a stop. -/
theorem verifyProof_noWork (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (outcome : runProof pub proof = .verdict (.noWork i site)) :
    ∃ messages rest op, decode ops proof = .ok (messages, rest) ∧
      works pub ops proof messages = false ∧ ops[i]? = some op ∧ op.grinds = true ∧
      (runFrom pub proof messages (ops.zipIdx.toList.take i) (channelStart pub)).1 = .ok () ∧
      (((step pub proof messages i op).run).run (stateBefore pub proof messages ops i)).1 =
        .error (.noWork i site) := by
  obtain ⟨messages, rest, decoded, stop⟩ := (verifyProof_noWork_iff pub proof scripted).mp outcome
  obtain ⟨op, at_, before, stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at_ before stepped
  refine ⟨messages, rest, op, decoded, ?_, at_,
    step_noWork pub proof messages i op _ stepped, before, stepped⟩
  unfold works
  rw [stop]

/-- With `endsFit` and the setup prefix of the terminal, an `accept` outcome comes from a proof
that decodes with no byte left and whose replay has no stop. -/
theorem verifyProof_replay (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) :
    ∃ messages, decode ops proof = .ok (messages, 0) ∧ (replay pub ops proof messages).1 = none := by
  rw [verifyProof_script pub proof scripted] at accepted
  obtain ⟨messages, rest, decoded, empty, stops⟩ := (verifyBytes_accept pub ops proof).mp accepted
  subst empty
  refine ⟨messages, decoded, ?_⟩
  rcases stop : (replay pub ops proof messages).1 with _ | (op | op | k)
  · rfl
  · exact absurd stop (stops _ _).1
  · exact absurd stop (stops _ _).2.1
  · exact absurd stop (replay_matches pub proof messages ops decoded ends terminalSetup k)

theorem replay_of_decoded (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) :
    rest = 0 ∧ (replay pub ops proof messages).1 = none := by
  obtain ⟨m, decoded', none_⟩ := verifyProof_replay pub proof scripted ends terminalSetup accepted
  rw [decoded] at decoded'
  cases decoded'
  exact ⟨rfl, none_⟩

/-- When the replay has no stop, the run before the operation `c` and its step have no stop. -/
theorem replay_step_ok (messages : Array Message) (accepted : (replay pub ops proof messages).1 = none)
    {c : ℕ} {op : Op} (at_ : ops[c]? = some op) :
    (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () ∧
      (((step pub proof messages c op).run).run (stateBefore pub proof messages ops c)).1 =
        .ok () := by
  have ran := runFrom_take_ok pub proof messages _ (c + 1) _
    ((replay_none_iff pub ops proof messages).mp accepted)
  rw [take_succ_zipIdx ops at_, runFrom_append_fst, runFrom_cons_fst] at ran
  exact ⟨ran.1, ran.2.1⟩

theorem replay_ran (messages : Array Message) (accepted : (replay pub ops proof messages).1 = none)
    {c : ℕ} {op : Op} (at_ : ops[c]? = some op) :
    (runFrom pub proof messages (ops.zipIdx.toList.take c ++ [(op, c)]) (channelStart pub)).1 =
      .ok () := by
  have ran := runFrom_take_ok pub proof messages _ (c + 1) _
    ((replay_none_iff pub ops proof messages).mp accepted)
  rwa [take_succ_zipIdx ops at_] at ran

/-- The step of a check with a stage function accepts exactly when its result accepts. -/
theorem step_stage_ok (messages : Array Message) (i : ℕ) (key : Key) (check : Check)
    (s : Replayer) {out : StageOut} (value : stageValue pub s check = some out)
    (ran : (((step pub proof messages i (.check key check)).run).run s).1 = .ok ()) :
    out.accepts = true := by
  rw [step_stage pub proof messages i key check s value] at ran
  cases accepts : out.accepts
  · rw [accepts] at ran
    cases ran
  · rfl

/-- `verifyProof_accept` of section 5.1: with `endsFit` and the setup prefix of the terminal, an
`accept` outcome comes from a proof that decodes with no byte left, whose works hold, whose replay
has no stop, and on the state before each check: the stage function accepts (the range proof,
the tensor reduction, `fused.end`, `stage3.end`), the level function of each fold of the shape
`foldShape` is the recorded `(r_2, v_W)` (`offloadRunIn` with the result of Stage 3 at a fold of
the shape `offloadShape`), the tensor reduction accepts, and `terminalRun` accepts. -/
theorem verifyProof_accept (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) :
    ∃ messages, decode ops proof = .ok (messages, 0) ∧ works pub ops proof messages = true ∧
      (replay pub ops proof messages).1 = none ∧
      (∀ c key check out, ops[c]? = some (.check key check) →
        stageValue pub (stateBefore pub proof messages ops c) check = some out →
          out.accepts = true) ∧
      (∀ c key level F r a e, ops[c]? = some (.check key (.fused level F)) →
        foldShape (ops.zipIdx.toList.take c) level F r a e = true →
        foldRunIn pub (stateBefore pub proof messages ops c) level F =
          some ((recordedIn (stateBefore pub proof messages ops c)
              ⟨level, "fused.point", none⟩).toList,
            firstIn (stateBefore pub proof messages ops c) ⟨level, "witness_evaluation", none⟩)) ∧
      (∀ c key level F plan r a e b t, ops[c]? = some (.check key (.fused level F)) →
        offloadShape (ops.zipIdx.toList.take c) level F plan r a e b t = true →
        ∃ o, stage3In (stateBefore pub proof messages ops c) level plan = some o ∧
          offloadRunIn pub (stateBefore pub proof messages ops c) level F plan =
            some (((recordedIn (stateBefore pub proof messages ops c)
                ⟨level, "fused.point", none⟩).toList,
              firstIn (stateBefore pub proof messages ops c)
                ⟨level, "witness_evaluation", none⟩), o)) ∧
      (∀ c key level count point opening,
        ops[c]? = some (.check key (.tensor level count point opening)) →
        let s := stateBefore pub proof messages ops c
        (tensorRun count (recordedIn s point).toList (firstIn s opening)
          ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
          ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
          (firstIn s ⟨level, "tensor.eta", none⟩) (roundsIn s ⟨level, "tensor.sumcheck", none⟩)
          (firstIn s ⟨level, "tensor.final_claims", none⟩)).isSome = true) ∧
      ∀ c level, ops[c]? = some (.terminal level) →
        ∃ out, terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
          pub.layout.terminal.params full
          (terminalMessagesIn (stateBefore pub proof messages ops c) level)
          (terminalChallengesIn (stateBefore pub proof messages ops c) level) = .ok out := by
  obtain ⟨messages, decoded, none_⟩ :=
    verifyProof_replay pub proof scripted ends terminalSetup accepted
  have stage : ∀ c key check out, ops[c]? = some (.check key check) →
      stageValue pub (stateBefore pub proof messages ops c) check = some out →
        out.accepts = true := fun c key check out at_ value =>
    step_stage_ok pub proof messages c key check _ value
      (replay_step_ok pub proof messages none_ at_).2
  refine ⟨messages, decoded, by unfold works; rw [none_], none_, stage, ?_, ?_, ?_, ?_⟩
  · intro c key level F r a e at_ shape
    exact replay_level pub ops proof messages none_ at_ shape
  · intro c key level F plan r a e b t at_ shape
    obtain ⟨ran, some_⟩ := offloadRunIn_of_run pub proof messages _ (channelStart_fresh pub)
      (channelStart_empty pub) _ key c level F plan shape (replay_ran pub proof messages none_ at_)
    obtain ⟨o, found⟩ := Option.isSome_iff_exists.mp some_
    refine ⟨o, found, ?_⟩
    unfold stateBefore
    rw [ran, found]
    rfl
  · intro c key level count point opening at_
    exact stage c key _ _ at_ rfl
  · intro c level at_
    have stepped := (replay_step_ok pub proof messages none_ at_).2
    rw [step_terminal_run pub proof messages c level _ terminalSetup] at stepped
    split at stepped
    · rename_i out found
      exact ⟨out, found⟩
    · cases stepped

/-- The result of `holds` on a check with a stage function is the result of the stage function. -/
theorem holds_stage (s : Replayer) (check : Check) {out : StageOut}
    (value : stageValue pub s check = some out) :
    ((holds pub check).run).run s = (.ok out.accepts, s) := by
  have generic : ((holds pub check).run).run s =
      (.ok (((stageValue pub s check).map StageOut.accepts).getD false), s) := by
    cases check <;> first | rfl | (simp only [stageValue] at value; cases value)
  rw [generic, value]
  rfl

/-- The step of a check has no stop exactly when `holds` gives `true` on the state before it. -/
theorem step_check_iff (messages : Array Message) (i : ℕ) (key : Key) (check : Check)
    (s : Replayer) :
    (((step pub proof messages i (.check key check)).run).run s).1 = .ok () ↔
      (((holds pub check).run).run s).1 = .ok true := by
  cases value : stageValue pub s check with
  | some out =>
      rw [step_stage pub proof messages i key check s value, holds_stage pub s check value]
      cases out.accepts <;> simp
  | none =>
      rw [step_check, checkStep_run, value]
      simp only [checkWith]
      rw [run_bind]
      rcases found : ((holds pub check).run).run s with ⟨_ | b, s₁⟩
      · have := throws_holds (P := fun _ => False) pub check s _ (by rw [found])
        exact this.elim
      · cases b
        · simp
        · simp only [↓reduceIte, iff_true]
          rfl

/-- The converse of `verifyProof_accept`: with `endsFit` and the setup prefix of the terminal, a
proof that decodes with no byte left, whose works hold, each check of the script holds on the state
before it and the terminal accepts, is accepted. -/
theorem verifyProof_accept_of (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} (decoded : decode ops proof = .ok (messages, 0))
    (worked : works pub ops proof messages = true)
    (checks : ∀ c key check, ops[c]? = some (.check key check) →
      (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
      (((holds pub check).run).run (stateBefore pub proof messages ops c)).1 = .ok true)
    (terminal : ∀ c level, ops[c]? = some (.terminal level) →
      (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
      ∃ out, terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
        pub.layout.terminal.params full
        (terminalMessagesIn (stateBefore pub proof messages ops c) level)
        (terminalChallengesIn (stateBefore pub proof messages ops c) level) = .ok out) :
    runProof pub proof = .verdict .accept := by
  rw [verifyProof_script pub proof scripted, verifyBytes_verdict _ ops proof messages decoded]
  congr 1
  refine (verdict_accept_iff pub proof decoded ends terminalSetup).mpr ?_
  rcases stop : (replay pub ops proof messages).1 with _ | st
  · rfl
  · exfalso
    obtain ⟨op, at_, before, stepped⟩ := replay_stop_at pub proof messages ops stop
    rcases st with ⟨i, site⟩ | ⟨i, site⟩ | k
    · unfold works at worked
      rw [stop] at worked
      cases worked
    · simp only [Stop.op] at at_ before stepped
      rcases step_reject pub proof messages i op _ stepped with ⟨key, check, rfl, -⟩ |
        ⟨level, rfl, found⟩
      · have ok := (step_check_iff pub proof messages i key check _).mpr
          (checks i key check at_ before)
        rw [stepped] at ok
        cases ok
      · obtain ⟨out, accepts⟩ := terminal i level at_ before
        unfold terminalValue at found
        dsimp only at found
        rw [terminalSetup, Option.map_some, accepts] at found
        cases found
    · exact replay_matches pub proof messages ops decoded ends terminalSetup k stop

/-- `verifyProof_reject` of section 5.1: a `reject` outcome comes from a proof that decodes, and
the replay stops at the operation `i` after the operations before it without a stop: a check of
the name of the stop that does not hold on the state before it, or the terminal, whose
`terminalValue` is the error of the stop. -/
theorem verifyProof_reject (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site)) :
    ∃ messages rest, decode ops proof = .ok (messages, rest) ∧
      (replay pub ops proof messages).1 = some (.reject i site) ∧
      (runFrom pub proof messages (ops.zipIdx.toList.take i) (channelStart pub)).1 = .ok () ∧
      ((∃ key check, ops[i]? = some (.check key check) ∧ key.site = site ∧
          (((holds pub check).run).run (stateBefore pub proof messages ops i)).1 = .ok false) ∨
        ∃ level, ops[i]? = some (.terminal level) ∧
          terminalValue pub (stateBefore pub proof messages ops i) level = some (.error site)) := by
  rw [verifyProof_script pub proof scripted] at rejected
  unfold verifyBytes at rejected
  rcases decoded : decode ops proof with failure | ⟨messages, rest⟩
  · rw [decoded] at rejected
    cases rejected
  · rw [decoded] at rejected
    have stop : (replay pub ops proof messages).1 = some (.reject i site) := by
      revert rejected
      unfold verdict verdictOf
      rcases found : (replay pub ops proof messages).1 with _ | (_ | _ | _) <;>
        by_cases empty : rest = 0 <;> simp [empty, found]
    obtain ⟨op, at_, before, stepped⟩ := replay_stop_at pub proof messages ops stop
    simp only [Stop.op] at at_ before stepped
    refine ⟨messages, rest, rfl, stop, before, ?_⟩
    rcases step_reject pub proof messages i op _ stepped with ⟨key, check, rfl, named⟩ |
      ⟨level, rfl, found⟩
    · refine Or.inl ⟨key, check, at_, named, ?_⟩
      rcases held : ((holds pub check).run).run (stateBefore pub proof messages ops i) with
        ⟨_ | b, s₁⟩
      · have := throws_holds (P := fun _ => False) pub check _ _ (by rw [held])
        exact this.elim
      · cases b
        · rfl
        · have ok := (step_check_iff pub proof messages i key check _).mpr (by rw [held])
          rw [stepped] at ok
          cases ok
    · exact Or.inr ⟨level, at_, found⟩

/-- A reject at `fused.end` of a fold of the shape `foldShape`: the level function is `none` on
the state before it. -/
theorem verifyProof_reject_fused (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {F : Fold} (at_ : ops[i]? = some (.check key (.fused level F)))
    {r a e : ℕ} (shape : foldShape (ops.zipIdx.toList.take i) level F r a e = true) :
    foldRunIn pub (stateBefore pub proof messages ops i) level F = none := by
  obtain ⟨m, rest', decoded', stop, before, _⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  obtain ⟨op, at', before', stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at' before' stepped
  rw [at_] at at'
  cases at'
  refine foldRunIn_of_reject pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub)
    _ key i level F shape before ?_
  unfold stateBefore at stepped
  rw [stepped]
  simp

/-- A reject at the last check of the range proof (`range.leaf.end` or `norm.end`): the level
function is `none` on the state before it. -/
theorem verifyProof_reject_range (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {F : Fold} (at_ : ops[i]? = some (.check key (rangeCheckOf level F))) :
    foldRunIn pub (stateBefore pub proof messages ops i) level F = none := by
  obtain ⟨m, rest', decoded', stop, -, -⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  obtain ⟨op, at', -, stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at' stepped
  rw [at_] at at'
  cases at'
  exact foldRunIn_of_range_reject pub proof messages _ key i level F (by rw [stepped]; simp)

/-- `treeRun` with `normRun` is `none` when the energy does not fit. -/
theorem treeRun_normRun_none {leaf limit energy : ℕ} (unfit : energyFits energy limit = false)
    (merge : Ext) (rounds : List (List Ext × Ext)) (virtual image : Ext) :
    ∀ (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext),
      treeRun (normRun leaf limit energy merge rounds virtual image) stages nodes weights point
        claim = none
  | [], _, _, _, _ => by simp [treeRun, normRun, unfit]
  | S :: rest, nodes, weights, point, claim => by
      simp only [treeRun]
      split
      · rfl
      · split
        · exact treeRun_normRun_none unfit merge rounds virtual image rest _ _ _ _
        · rfl

/-- A stop at `norm.integer` on the Euclidean route: `normRun` checks
the same `E_resp` again, so `foldRunIn` is `none` on the state of the stop. -/
theorem foldRunIn_of_energy_reject (messages : Array Message) (s : Replayer) (key : Key)
    (i level : ℕ) (F : Fold) (euclidean : F.euclidean = true)
    (stopped : (((step pub proof messages i
      (.check key (.energy ⟨level, "norm.response_l2_sq", none⟩ F.limit))).run).run s).1 ≠
        .ok ()) :
    foldRunIn pub s level F = none := by
  have unfit : energyFits (normsIn s level).energy F.limit = false := by
    cases fits : energyFits (normsIn s level).energy F.limit
    · rfl
    · refine absurd ((step_check_iff pub proof messages i key _ s).mpr ?_) stopped
      change (Except.ok (energyFits (normsIn s level).energy F.limit) : Except Stop Bool) =
        .ok true
      rw [fits]
  unfold foldRunIn
  cases pub.setupFor F.setupCount with
  | none => rfl
  | some setup =>
      simp only [Option.bind_some, euclidean, ↓reduceIte]
      rw [foldRunSq_if, treeRun_normRun_none unfit]
      rfl

/-- A reject at `norm.integer` on the Euclidean route: the level function is `none` on the state
before it. -/
theorem verifyProof_reject_energy (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {F : Fold} (euclidean : F.euclidean = true)
    (at_ : ops[i]? = some (.check key (.energy ⟨level, "norm.response_l2_sq", none⟩ F.limit))) :
    foldRunIn pub (stateBefore pub proof messages ops i) level F = none := by
  obtain ⟨m, rest', decoded', stop, -, -⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  obtain ⟨op, at', -, stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at' stepped
  rw [at_] at at'
  cases at'
  exact foldRunIn_of_energy_reject pub proof messages _ key i level F euclidean
    (by rw [stepped]; simp)

/-- A reject at `stage3.end`: `stage3In` is `none` on the state before it. -/
theorem verifyProof_reject_stage3 (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {plan : SetupPlan}
    (at_ : ops[i]? = some (.check key (.stage3 level plan))) :
    stage3In (stateBefore pub proof messages ops i) level plan = none := by
  obtain ⟨m, rest', decoded', stop, before, held⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  rcases held with ⟨key', check, at', -, held⟩ | ⟨level', at', -⟩
  · rw [at_] at at'
    cases at'
    rw [holds_stage3] at held
    simp only [stageValue, Option.map_some, Option.getD_some, StageOut.accepts] at held
    cases found : stage3In (stateBefore pub proof messages ops i) level plan with
    | none => rfl
    | some o =>
        rw [found] at held
        cases held
  · rw [at_] at at'
    cases at'

/-- A reject at `tensor.opening`: `tensorRun` is `none` on the state before it, for each later
`η`, rounds and final claim. -/
theorem verifyProof_reject_opening (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key point value partials : Key}
    (at_ : ops[i]? = some (.check key (.tensorOpening point value partials)))
    (count : ℕ) (η : Ext) (rounds : List (List Ext × Ext)) (final : Ext) :
    let s := stateBefore pub proof messages ops i
    tensorRun count (recordedIn s point).toList (firstIn s value)
      ((recordedIn s partials).getD 0 Ext.zero) ((recordedIn s partials).getD 1 Ext.zero) η rounds
        final = none := by
  obtain ⟨m, rest', decoded', stop, before, _⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  obtain ⟨op, at', before', stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at' before' stepped
  rw [at_] at at'
  cases at'
  exact tensorRun_of_opening_reject pub proof messages i key _ count point value partials η
    rounds final (by rw [stepped]; simp)

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The tensor reduction at `tensor.end` with its sum-check of the shape `roundsAt`: when
`runProof` accepts, `tensorCheck` of the specification runs on the path of the recorded values
of the state before it to acceptance. -/
theorem verifyProof_tensor_game (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key}
    {level count : ℕ} {point opening : Key}
    (at_ : ops[c]? = some (.check key (.tensor level count point opening)))
    (shape : roundsAt (ops.zipIdx.toList.take c) ⟨level, "tensor.sumcheck", none⟩ count 2 = true)
    {s : Replayer} (state : stateBefore pub proof messages ops c = s) (w : Table) :
    let p0 := (recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero
    let p1 := (recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero
    let η := firstIn s ⟨level, "tensor.eta", none⟩
    let final := firstIn s ⟨level, "tensor.final_claims", none⟩
    ∃ o, tensorRun count (recordedIn s point).toList (firstIn s opening) p0 p1 η
        (roundsIn s ⟨level, "tensor.sumcheck", none⟩) final = some o ∧
      (tensorCheck count w ((recordedIn s point).toList.map (Ext.into (E := E)))
        (firstIn s opening).into).Runs
        (tensorPath E count (recordedIn s point).toList (firstIn s opening) p0 p1 η
          (roundsIn s ⟨level, "tensor.sumcheck", none⟩) final)
        (some (o.point.map Ext.into,
          mle (o.point.map Ext.into) (tensorWeight ((recordedIn s point).toList.tail.map Ext.into)
            η.into), o.claim.into)) := by
  obtain ⟨-, none_⟩ := replay_of_decoded pub proof scripted ends terminalSetup accepted decoded
  obtain ⟨before, stepped⟩ := replay_step_ok pub proof messages none_ at_
  obtain ⟨e, shaped⟩ := roundsAt_sound shape
  obtain ⟨roundCount, lengths, _⟩ := rounds_facts pub proof messages _ (channelStart_fresh pub)
    (channelStart_empty pub) _ _ count 2 e shaped
    (aligned_take messages _ c _ (decode_aligned decoded)) before
  unfold stateBefore at state stepped
  rw [state] at roundCount lengths stepped
  obtain ⟨run, game⟩ := step_tensor_runs (E := E) pub proof messages c key level count point
    opening s w
    roundCount lengths
  dsimp only at run game ⊢
  rw [run] at stepped
  cases found : tensorRun count (recordedIn s point).toList (firstIn s opening)
    ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
    ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
    (firstIn s ⟨level, "tensor.eta", none⟩) (roundsIn s ⟨level, "tensor.sumcheck", none⟩)
    (firstIn s ⟨level, "tensor.final_claims", none⟩) with
  | none =>
      simp only [found, Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at stepped
      cases stepped
  | some o =>
      refine ⟨o, rfl, ?_⟩
      have g := game
      simp only [found, Option.map_some] at g
      exact g

/-- A reject at `tensor.end` with its sum-check of the shape `roundsAt`: `tensorRun` is `none` on
the state before it and `tensorCheck` of the specification runs on the path of the recorded values
to `none`. -/
theorem verifyProof_reject_tensor_game (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level count : ℕ} {point opening : Key}
    (at_ : ops[i]? = some (.check key (.tensor level count point opening)))
    (shape : roundsAt (ops.zipIdx.toList.take i) ⟨level, "tensor.sumcheck", none⟩ count 2 = true)
    {s : Replayer} (state : stateBefore pub proof messages ops i = s) (w : Table) :
    let p0 := (recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero
    let p1 := (recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero
    let η := firstIn s ⟨level, "tensor.eta", none⟩
    let final := firstIn s ⟨level, "tensor.final_claims", none⟩
    tensorRun count (recordedIn s point).toList (firstIn s opening) p0 p1 η
        (roundsIn s ⟨level, "tensor.sumcheck", none⟩) final = none ∧
      (tensorCheck count w ((recordedIn s point).toList.map (Ext.into (E := E)))
        (firstIn s opening).into).Runs
        (tensorPath E count (recordedIn s point).toList (firstIn s opening) p0 p1 η
          (roundsIn s ⟨level, "tensor.sumcheck", none⟩) final) none := by
  obtain ⟨m, rest', decoded', stop, before, _⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  obtain ⟨op, at', before', stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at' before' stepped
  rw [at_] at at'
  cases at'
  obtain ⟨e, shaped⟩ := roundsAt_sound shape
  obtain ⟨roundCount, lengths, _⟩ := rounds_facts pub proof messages _ (channelStart_fresh pub)
    (channelStart_empty pub) _ _ count 2 e shaped
    (aligned_take messages _ i _ (decode_aligned decoded)) before
  unfold stateBefore at state stepped
  rw [state] at roundCount lengths stepped
  obtain ⟨run, game⟩ := step_tensor_runs (E := E) pub proof messages i key level count point
    opening s w
    roundCount lengths
  dsimp only at run game ⊢
  rw [run] at stepped
  cases found : tensorRun count (recordedIn s point).toList (firstIn s opening)
    ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
    ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
    (firstIn s ⟨level, "tensor.eta", none⟩) (roundsIn s ⟨level, "tensor.sumcheck", none⟩)
    (firstIn s ⟨level, "tensor.final_claims", none⟩) with
  | some o =>
      simp only [found, Option.isSome_some, ↓reduceIte] at stepped
      cases stepped
  | none =>
      refine ⟨rfl, ?_⟩
      have g := game
      simp only [found, Option.map_none] at g
      exact g

omit [Fintype E] [DecidableEq E] in
/-- The terminal: when `runProof` accepts, for a payload of `z` that decodes and fold
challenges that the sampler drew for the terminal group, `‖z‖_2² ≤ S_max`, the challenges are
valid and the terminal statement of the specification accepts on the state before the terminal
(Akita Section 8.2; `step_terminal_accepts`). -/
theorem verifyProof_terminal_accepts (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c level : ℕ}
    (at_ : ops[c]? = some (.terminal level)) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s)
    (n : ℕ) (power : 2 ^ (n + 1) = pub.layout.terminal.shape.innerDegree)
    (quarter : 4 * (pub.layout.terminal.shape.innerDegree / 4) =
      pub.layout.terminal.shape.innerDegree)
    (two : ((2 * (pub.layout.terminal.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (blocks : 1 ≤ pub.layout.terminal.shape.blocks)
    (degree : pub.layout.terminal.group.degree = pub.layout.terminal.shape.innerDegree)
    (ones : 0 < pub.layout.terminal.group.pm1)
    (same : pub.layout.terminal.group.ring = pub.layout.terminal.group.degree)
    {bytes : ByteArray}
    (drawn : foldChallenges pub.layout.terminal.group bytes = some (drawnIn s level 0))
    {z : Array ℤ}
    (payload : Golomb.decode pub.layout.terminal.params (terminalMessagesIn s level).payload =
      some z) :
    centeredNormSq z ≤ pub.layout.terminal.limit ∧
      ((terminalChallengesIn s level).fold.length = pub.layout.terminal.shape.blocks ∧
        (terminalChallengesIn s level).fold.all
          (sparseValid pub.layout.terminal.shape.innerDegree) = true) ∧
      ∃ (eLast : Rq pub.layout.terminal.shape.innerDegree)
        (tLast : ℕ → Rq pub.layout.terminal.shape.innerDegree),
        ringDigest eLast = (terminalMessagesIn s level).eDigest ∧
        rowsDigestOf pub.layout.terminal.shape tLast = (terminalMessagesIn s level).tDigest ∧
        (statementOf (E := E) pub.layout.terminal.shape quarter full (terminalMessagesIn s level)
          (terminalChallengesIn s level) tLast).Accepts
          (partialsOf pub.layout.terminal.shape (terminalMessagesIn s level) eLast)
          (challengesOf pub.layout.terminal.shape (terminalChallengesIn s level))
          (zSource pub.layout.terminal.shape z) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  have stepped := (replay_step_ok pub proof messages none_ at_).2
  rw [state] at stepped
  rcases found : ((step pub proof messages c (.terminal level)).run).run s with ⟨_ | _, s'⟩
  · rw [found] at stepped
    cases stepped
  · exact (step_terminal_accepts (E := E) pub proof messages c level s terminalSetup n power
      quarter two blocks degree ones same drawn payload).mp ⟨s', found⟩

/-- A trace level on the coefficient route: when `runProof` accepts, `foldCheck` of
`traceReduction` runs on the path of the code to acceptance with the recorded
`(fused.point, witness_evaluation)` on the state before its `fused.end`. -/
theorem verifyProof_trace_game (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s)
    {T : TraceShape} (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => T.shape) T.opening.openRank
      T.opening.openDegree T.opening.chain fun _ => .witness)
    (based : 2 ^ F.logBasis = T.shape.certifiedBase) (low : F.low = T.opening.low)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (challenge : Fin T.shape.blocks → Rq T.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq T.shape.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)).getD i
        (Poly.zero T.shape.innerDegree)))
    (fits : T.opening.cells T.fold ≤ 2 ^ F.vars)
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ F.batch) (w : Table)
    (plain : F.euclidean = false) (compressed : F.compressed = T.opening.compressed T.fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase)
    (leaf : leafDegree F.logBasis = rangeLeaf T.shape.certifiedBase) :
    (foldCheck (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) :=
  replay_trace_game pub ops proof messages decoded
    (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 at_ shape game state
    shaped expanded rows based low quarter two challenge challenged fits rowsFit w plain compressed
    arities leaf

/-- A trace level on the Euclidean route (`foldCheckSq` with the norm stage). -/
theorem verifyProof_traceSq_game (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s)
    {T : TraceShape} (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => T.shape) T.opening.openRank
      T.opening.openDegree T.opening.chain fun _ => .witness)
    (low : F.low = T.opening.low) (leaf : leafDegree F.logBasis = rangeLeaf T.shape.certifiedBase)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (challenge : Fin T.shape.blocks → Rq T.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq T.shape.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)).getD i
        (Poly.zero T.shape.innerDegree)))
    (fits : T.opening.cells T.fold ≤ 2 ^ F.vars)
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ F.batch) (w : Table)
    (positive : 1 ≤ rangeLeaf T.shape.certifiedBase) (small : F.limit < fieldModulus)
    (digits : 0 < T.fold.responseDigits) (euclidean : F.euclidean = true)
    (compressed : F.compressed = T.opening.compressed T.fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase) :
    (foldCheckSq (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) F.limit
        (T.shape.normKernel T.fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch)
      (some ((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn s ⟨level, "witness_evaluation", none⟩).into)) :=
  foldRunIn_traceSq_game pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
    key c level F shape game (aligned_take messages _ c _ (decode_aligned decoded))
    (replay_ran pub proof messages
      (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2 at_)
    state shaped expanded rows low leaf quarter two challenge challenged fits rowsFit w positive small
    digits euclidean compressed arities

/-- At `fused.end` of a fold of the shapes `foldShape` and `gameShape`, when `runProof`
accepts: the state before it has the state hypotheses of the games (`LevelState`) and the level
function is the recorded `(r_2, v_W)`, so each game theorem `foldRunIn_*_runs` of the level runs
to it. -/
theorem verifyProof_level (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F))) {r a e : ℕ}
    (shape : foldShape (ops.zipIdx.toList.take c) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true) :
    LevelState (stateBefore pub proof messages ops c) level F ∧
      foldRunIn pub (stateBefore pub proof messages ops c) level F =
        some ((recordedIn (stateBefore pub proof messages ops c)
            ⟨level, "fused.point", none⟩).toList,
          firstIn (stateBefore pub proof messages ops c) ⟨level, "witness_evaluation", none⟩) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  exact ⟨level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
      level F game (aligned_take messages _ c _ (decode_aligned decoded))
      (replay_step_ok pub proof messages none_ at_).1,
    replay_level pub ops proof messages none_ at_ shape⟩

/-- A reject at `fused.end` of a fold of the shapes `foldShape` and `gameShape`: the state before
it has the state hypotheses of the games (`LevelState`) and the level function is `none` there, so
each game theorem `foldRunIn_*_runs` of the level runs to `none`. -/
theorem verifyProof_reject_level (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {F : Fold} (at_ : ops[i]? = some (.check key (.fused level F)))
    {r a e : ℕ} (shape : foldShape (ops.zipIdx.toList.take i) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take i) level F = true) :
    LevelState (stateBefore pub proof messages ops i) level F ∧
      foldRunIn pub (stateBefore pub proof messages ops i) level F = none := by
  obtain ⟨m, rest', decoded', _, before, _⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  exact ⟨level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
      level F game (aligned_take messages _ i _ (decode_aligned decoded)) before,
    verifyProof_reject_fused pub proof scripted rejected decoded at_ shape⟩

/-- A reject at `fused.end` of a trace level on the coefficient route: `foldCheck` of
`traceReduction` runs on the path of the code to `none`. -/
theorem verifyProof_reject_trace_game (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : runProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {F : Fold} (at_ : ops[i]? = some (.check key (.fused level F)))
    {r a e : ℕ} (shape : foldShape (ops.zipIdx.toList.take i) level F r a e = true)
    (game : gameShape (ops.zipIdx.toList.take i) level F = true) {s : Replayer}
    (state : stateBefore pub proof messages ops i = s)
    {T : TraceShape} (shaped : F.fused = .trace T) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => T.shape) T.opening.openRank
      T.opening.openDegree T.opening.chain fun _ => .witness)
    (based : 2 ^ F.logBasis = T.shape.certifiedBase) (low : F.low = T.opening.low)
    (quarter : 4 * (T.shape.innerDegree / 4) = T.shape.innerDegree)
    (two : ((2 * (T.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (challenge : Fin T.shape.blocks → Rq T.shape.innerDegree)
    (challenged : ∀ i, challenge i = Poly.toRq T.shape.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)).getD i
        (Poly.zero T.shape.innerDegree)))
    (fits : T.opening.cells T.fold ≤ 2 ^ F.vars)
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ F.batch) (w : Table)
    (plain : F.euclidean = false) (compressed : F.compressed = T.opening.compressed T.fold)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase)
    (leaf : leafDegree F.logBasis = rangeLeaf T.shape.certifiedBase) :
    (foldCheck (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge F.vars fits F.batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) w).Runs
      (foldPathIn E pub s level F setup F.vars F.batch) none := by
  have none_ := verifyProof_reject_fused pub proof scripted rejected decoded at_ shape
  obtain ⟨m, rest', decoded', _, before, _⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  have st := level_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub)
    _ level F game (aligned_take messages _ i _ (decode_aligned decoded)) before
  unfold stateBefore at state none_
  rw [state] at st none_
  obtain ⟨virtual, count, lengths⟩ := st.leaf plain
  have runs := foldRunIn_trace_runs (E := E) pub s level F shaped expanded rows based low quarter
    two challenge challenged fits rowsFit w plain compressed virtual st.vars st.tau arities
    st.stages count (fun x member => (lengths x member).trans leaf) st.fusedCount st.fusedLengths
  rw [none_] at runs
  exact runs

end Proof

end NightstreamFPrime.Spec.Compression.Akita.Replay
