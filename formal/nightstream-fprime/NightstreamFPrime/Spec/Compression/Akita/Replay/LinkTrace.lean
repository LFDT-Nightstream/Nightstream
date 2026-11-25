import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalKey20
import NightstreamFPrime.Spec.Compression.Akita.Extract.TraceLink

/-! The link from a level of the replay to the next level with evaluation trace (Akita revision
`d50f700`, `fold/suffix.rs` and `stages/opening_claims/extension_claim.rs`; ePrint 2026/1983,
revision of 2026-11-23, Figure 4, steps 0, 3 and 5, Section 5.3 with (65), Section 7.1 with (95)
to (101); `Tier.tensored` and `Tier.TraceLinked` of `Extract/TraceLink.lean`).

The code runs the tensor reduction of the claim `(r_2, v_W)` of the level `ℓ` in the prefix of the
level `ℓ + 1` (`Build.prefix_`), so its keys `tensor.*` have the level `ℓ + 1`.

- `stateBefore_frees`: a key that no operation from `c` to `c'` writes has the same recorded values
  and field message before `c` and before `c'`, and a sum-check that none of them ends has the same
  rounds.
- `tensorRun_some`: an accepted tensor reduction has `ν` tail coordinates, its point is the point
  of its rounds and its final claim is the received one.
- `traceStatementFor_one`: the statement of a trace level with one claim is the folding statement
  of `TraceClaim.ofPoint` for the segments of the tensor point, the factor `θ = A_η(ρ)` and the
  final claim (`tensorFactor_into`).
- `verifyProof_tensorLink`: when `runProof` accepts, the tensor reduction of the level `ℓ + 1`
  on the claim of the level `ℓ` runs to `(ρ, θ, ṽ)` of the recorded values on a later state.
- `verifyProof_tensored`: the game of a level `ℓ` and then this tensor reduction (`Tier.tensored`)
  runs on the path of the fold and of the tensor reduction to `(ρ, θ, ṽ)`.
- `verifyProof_traceLink`: the statement `traceStatementFor` of the replay at the level `ℓ + 1` is
  the claim that `Tier.TraceLinked` takes from `(ρ, θ, ṽ)`, with the segments
  `ρ = ρ_pack ++ (ρ_pos ++ ρ_blk)` and the payload of the words `next_witness` of the level `ℓ`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

/-- The keys that the tensor reduction of the level `level + 1` and the link read: the point and
the value of the claim of the level `level`, and the partials, `η`, the final claims and the point
of the tensor reduction. -/
def tensorKeys (level : ℕ) : List Key :=
  [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩,
    ⟨level + 1, "tensor.partials", none⟩, ⟨level + 1, "tensor.eta", none⟩,
    ⟨level + 1, "tensor.final_claims", none⟩, ⟨level + 1, "tensor.sumcheck.point", none⟩]

theorem pathRun_source (chain : Option ChainShape) (image second degree count : ℕ)
    (source : RhsSource) : (pathRun chain image second degree count source).source = source := by
  cases chain <;> rfl

theorem dataIn_witness (pub : Public) (s : Replayer) (level : ℕ) {run : RhsRun}
    (source : run.source = .witness) :
    dataIn pub s (level + 1) run = (s.fields.get? ⟨level, "next_witness", none⟩).getD #[] := by
  unfold dataIn
  rw [source]
  rfl

theorem dataIn_payload (pub : Public) (s : Replayer) (level : ℕ) {run : RhsRun}
    (source : run.source = .payload) :
    dataIn pub s level run = (s.fields.get? ⟨level, "opening_payload", none⟩).getD #[] := by
  unfold dataIn
  rw [source]

theorem firstIn_congr {s s' : Replayer} {key : Key} (same : recordedIn s' key = recordedIn s key) :
    firstIn s' key = firstIn s key := by
  unfold firstIn
  rw [same]

section Frames

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A key that no operation from `c` to `c'` writes has the same recorded values and field
message before `c` and before `c'`; a sum-check that none of them ends has the same rounds. -/
theorem stateBefore_frees {ops : Array Op} {c c' : ℕ} (le : c ≤ c') {K R : List Key}
    (free : frees ((ops.zipIdx.toList.take c').drop c) K R = true) :
    (∀ k ∈ K, recordedIn (stateBefore pub proof messages ops c') k =
        recordedIn (stateBefore pub proof messages ops c) k ∧
      (stateBefore pub proof messages ops c').fields.get? k =
        (stateBefore pub proof messages ops c).fields.get? k) ∧
    ∀ k ∈ R, roundsIn (stateBefore pub proof messages ops c') k =
      roundsIn (stateBefore pub proof messages ops c) k := by
  have split : ops.zipIdx.toList.take c' =
      ops.zipIdx.toList.take c ++ (ops.zipIdx.toList.take c').drop c := by
    conv_lhs => rw [← List.take_append_drop c (ops.zipIdx.toList.take c')]
    rw [List.take_take, Nat.min_eq_left le]
  unfold stateBefore
  rw [split, runFrom_append]
  rcases runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub) with
    ⟨_ | _, s₁⟩
  · exact ⟨fun _ _ => ⟨rfl, rfl⟩, fun _ _ => rfl⟩
  · exact ⟨fun k inside => ⟨recorded_free pub proof messages _ s₁ k (frees_writes free inside),
      fields_free pub proof messages _ s₁ k (frees_writes free inside)⟩,
      fun k inside => roundsIn_free pub proof messages _ s₁ k (frees_ends free inside)⟩

end Frames

/-- An accepted tensor reduction: the claim point has `ν` tail coordinates, the point of the
result is the point of the rounds, and its final claim is the received final claim. -/
theorem tensorRun_some {ν : ℕ} {point : List Ext} {v p0 p1 η : Ext}
    {rounds : List (List Ext × Ext)} {final : Ext} {o : Ended}
    (ran : tensorRun ν point v p0 p1 η rounds final = some o) :
    point.tail.length = ν ∧ o.point = rounds.map (·.2) ∧ o.claim = final := by
  cases point with
  | nil => simp [tensorRun] at ran
  | cons head tail =>
      simp only [tensorRun, List.tail_cons] at ran ⊢
      by_cases long : tail.length = ν
      · rw [ite_eq_left long] at ran
        by_cases opened : v.same ((Ext.one - head) * p0 + head * p1) = true
        · rw [ite_eq_left opened] at ran
          by_cases ended : (plainRun (tensorInput p0 p1 η) rounds).claim.same final = true
          · rw [ite_eq_left ended] at ran
            obtain rfl := Option.some.inj ran
            exact ⟨long, plainRun_point _ _, rfl⟩
          · rw [ite_eq_right ended] at ran
            cases ran
        · rw [ite_eq_right opened] at ran
          cases ran
      · rw [ite_eq_right long] at ran
        cases ran

section Statement

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- With one claim, the value `Σ_i ρ_i·c_i` is the first claim. -/
theorem into_dot_one (claims : Array Ext) :
    ((dot #[Ext.one] claims).into : E) = ((claims[0]?).getD Ext.zero).into := by
  unfold dot
  rw [← Array.foldl_toList, Array.toList_zip]
  rcases claims with ⟨_ | ⟨c, l⟩⟩
  · simp
  · simp [Ext.into_add, Ext.into_mul, Ext.into_one, Ext.into_zero]

/-- The segments `(ρ_pack, ρ_pos, ρ_blk)` of a point are the point. -/
theorem segments_append (S : Shape) (ρ : List Ext) :
    packPoint S ρ ++ (positionPoint S ρ ++ blockPoint S ρ) = ρ := by
  unfold packPoint positionPoint blockPoint
  rw [← List.drop_drop, List.take_append_drop, List.take_append_drop]

/-- The lengths of the segments of a point with `ν` coordinates or more: `log_2(d/2)`, `log_2 M`
and the rest. -/
theorem segments_length (S : Shape) (ρ : List Ext)
    (long : Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength ≤ ρ.length) :
    (packPoint S ρ).length = Nat.log 2 (S.innerDegree / 2) ∧
      (positionPoint S ρ).length = Nat.log 2 S.blockLength ∧
      (blockPoint S ρ).length = ρ.length -
        (Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength) := by
  unfold packPoint positionPoint blockPoint
  refine ⟨?_, ?_, ?_⟩
  · rw [List.length_take]
    omega
  · rw [List.length_take, List.length_drop]
    omega
  · rw [List.length_drop]

variable {S : Shape} {O : OpeningShape S}

omit [Fintype E] [DecidableEq E] in
/-- The statement of a trace level with one claim is the folding statement of
`TraceClaim.ofPoint` for the matrices of the setup: the segments of the tensor point `ρ`, the
factor `θ = mle ρ (tensorWeight tail η)` of `tensorCheck` and the first claim. -/
theorem traceStatementFor_one (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (setup : Array Fq) (V : TraceValues) (data : RhsRun → Array Fq) (claims : Array Ext)
    (length : V.point.length = V.tail.length) :
    traceStatementFor (E := E) O quarter setup V data .witness #[Ext.one] claims =
      (Extract.TraceClaim.ofPoint quarter
        (payloadOf S.chain _ _
          (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => .witness) 0)))
        (payloadOf O.chain _ _
          (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)))
        ((blockPoint S V.point).map Ext.into) ((positionPoint S V.point).map Ext.into)
        ((packPoint S V.point).map Ext.into)
        (mle (V.point.map Ext.into) (tensorWeight (V.tail.map Ext.into) V.eta.into))
        ((claims[0]?).getD Ext.zero).into).statement
        (Key.ofSetup (flatOf setup) S) (OpeningKey.ofSetup (flatOf setup) O) := by
  unfold traceStatementFor traceStatementOf
  rw [tensorFactor_into _ _ length, into_dot_one]
  rfl

end Statement

section Proof

variable (pub : Public) (proof : ByteArray) {ops : Array Op}
variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The tensor reduction of the level `level + 1` at `tensor.end` (operation `t`) on the claim
`(fused.point, witness_evaluation)` of the level `level`: when `runProof` accepts and no
operation from `t` to `c'` writes `tensorKeys level` or ends `tensor.sumcheck`, on the state
before `c'` the tensor point `ρ` and the tail of `r_2` have `ν` coordinates, and `tensorCheck`
runs on the path of the recorded values to `(ρ, θ, ṽ)` with `θ = mle ρ (tensorWeight r_2.tail η)`
and the final claim `ṽ`. -/
theorem verifyProof_tensorLink (scripted : script pub.layout pub.plan = .ok ops)
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
    (w : Table) :
    let s := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
    let ρ := (recordedIn s ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
    let η := firstIn s ⟨level + 1, "tensor.eta", none⟩
    let final := firstIn s ⟨level + 1, "tensor.final_claims", none⟩
    ρ.length = ν ∧ r2.tail.length = ν ∧
      (tensorCheck ν w (r2.map (Ext.into (E := E)))
        (firstIn s ⟨level, "witness_evaluation", none⟩).into).Runs
        (tensorPath E ν r2 (firstIn s ⟨level, "witness_evaluation", none⟩)
          ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
          ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero) η
          (roundsIn s ⟨level + 1, "tensor.sumcheck", none⟩) final)
        (some (ρ.map Ext.into, mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into),
          final.into)) := by
  obtain ⟨-, none_⟩ := replay_of_decoded pub proof scripted ends terminalSetup accepted decoded
  obtain ⟨before, _⟩ := replay_step_ok pub proof messages none_ at_
  obtain ⟨e, shaped⟩ := roundsAt_sound shape
  obtain ⟨roundCount, _, points⟩ := rounds_facts pub proof messages _ (channelStart_fresh pub)
    (channelStart_empty pub) _ _ ν 2 e shaped
    (aligned_take messages _ t _ (decode_aligned decoded)) before
  obtain ⟨o, ran, game⟩ := verifyProof_tensor_game (E := E) pub proof scripted ends terminalSetup
    accepted decoded at_ shape rfl w
  obtain ⟨tailLength, oPoint, oClaim⟩ := tensorRun_some ran
  obtain ⟨frames, roundsFrame⟩ := stateBefore_frees pub proof messages after kept
  have frame := fun k (inside : k ∈ tensorKeys level) => (frames k inside).1
  have fp := frame ⟨level, "fused.point", none⟩ (by simp [tensorKeys])
  have we := frame ⟨level, "witness_evaluation", none⟩ (by simp [tensorKeys])
  have partials := frame ⟨level + 1, "tensor.partials", none⟩ (by simp [tensorKeys])
  have eta := frame ⟨level + 1, "tensor.eta", none⟩ (by simp [tensorKeys])
  have finals := frame ⟨level + 1, "tensor.final_claims", none⟩ (by simp [tensorKeys])
  have rho := frame ⟨level + 1, "tensor.sumcheck.point", none⟩ (by simp [tensorKeys])
  have rounds := roundsFrame ⟨level + 1, "tensor.sumcheck", none⟩ List.mem_cons_self
  have pointIs : o.point = (recordedIn (stateBefore pub proof messages ops t)
      ⟨level + 1, "tensor.sumcheck.point", none⟩).toList := by
    rw [oPoint]
    exact points
  dsimp only at game ⊢
  rw [fp, firstIn_congr we, partials, firstIn_congr eta, firstIn_congr finals, rho, rounds]
  refine ⟨?_, tailLength, ?_⟩
  · rw [← pointIs, oPoint, List.length_map]
    exact roundCount
  · rw [pointIs, oClaim] at game
    exact game

/-- `Tier.tensored` of a level `level` on the replay: when its game `g` runs on `path` to the
recorded `(fused.point, witness_evaluation)` on the state before `c` and `runProof` accepts,
`g` with the tensor reduction of the level `level + 1` at `t` (the keys `tensor.*` of the level of
the receiver) runs on `path` and the path of the tensor reduction to `(ρ, θ, ṽ)` of the recorded
values on the state before `c'`. -/
theorem verifyProof_tensored (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {level c t c' ν : ℕ} {key : Key}
    (at_ : ops[t]? = some (.check key (.tensor (level + 1) ν ⟨level, "fused.point", none⟩
      ⟨level, "witness_evaluation", none⟩)))
    (shape : roundsAt (ops.zipIdx.toList.take t) ⟨level + 1, "tensor.sumcheck", none⟩ ν 2 = true)
    (before : c ≤ t) (after : t ≤ c')
    (held : frees ((ops.zipIdx.toList.take t).drop c)
      [⟨level, "fused.point", none⟩, ⟨level, "witness_evaluation", none⟩] [] = true)
    (kept : frees ((ops.zipIdx.toList.take c').drop t) (tensorKeys level)
      [⟨level + 1, "tensor.sumcheck", none⟩] = true)
    {g : Game (Option (List E × E))} {path : List Move} (w : Table)
    (runs : g.Runs path
      (some ((recordedIn (stateBefore pub proof messages ops c)
          ⟨level, "fused.point", none⟩).toList.map Ext.into,
        (firstIn (stateBefore pub proof messages ops c)
          ⟨level, "witness_evaluation", none⟩).into))) :
    let s := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
    let ρ := (recordedIn s ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
    let η := firstIn s ⟨level + 1, "tensor.eta", none⟩
    let final := firstIn s ⟨level + 1, "tensor.final_claims", none⟩
    ρ.length = ν ∧ r2.tail.length = ν ∧
      (g.bind fun
        | none => .pure none
        | some (r2, v) => tensorCheck ν w r2 v).Runs
        (path ++ tensoredPath E ν (some (r2, firstIn s ⟨level, "witness_evaluation", none⟩))
          ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
          ((recordedIn s ⟨level + 1, "tensor.partials", none⟩).getD 1 Ext.zero) η
          (roundsIn s ⟨level + 1, "tensor.sumcheck", none⟩) final)
        (some (ρ.map Ext.into, mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into),
          final.into)) := by
  obtain ⟨long, tail, game⟩ := verifyProof_tensorLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded at_ shape after kept w
  obtain ⟨frames, -⟩ := stateBefore_frees pub proof messages before held
  obtain ⟨later, -⟩ := stateBefore_frees pub proof messages after kept
  have fp := (later ⟨level, "fused.point", none⟩ (by simp [tensorKeys])).1.trans
    (frames ⟨level, "fused.point", none⟩ List.mem_cons_self).1
  have we := (later ⟨level, "witness_evaluation", none⟩ (by simp [tensorKeys])).1.trans
    (frames ⟨level, "witness_evaluation", none⟩ (by simp)).1
  rw [← fp, ← firstIn_congr we] at runs
  dsimp only at long tail game ⊢
  exact ⟨long, tail, runs_bind runs game⟩

omit [Fintype E] [DecidableEq E] in
/-- The words `next_witness` of the level before and `opening_payload` of a fold with one group of
the shape `S` and the opening `O`. -/
theorem dataIn_runs (s : Replayer) (level : ℕ) (S : Shape) (O : OpeningShape S) :
    dataIn pub s (level + 1)
        (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => .witness) 0) =
      (s.fields.get? ⟨level, "next_witness", none⟩).getD #[] ∧
    dataIn pub s (level + 1) (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain) =
      (s.fields.get? ⟨level + 1, "opening_payload", none⟩).getD #[] :=
  ⟨dataIn_witness pub s level (pathRun_source _ _ _ _ _ _),
    dataIn_payload pub s (level + 1) (pathRun_source _ _ _ _ _ _)⟩

/-- The link `Tier.TraceLinked` on the replay (Akita Figure 4, steps 0, 3 and 5, (65), (95) to
(101)): when `runProof` accepts, on the state before `c'` the statement `traceStatementFor` of
the receiver `F` at the level `level + 1` (a fold with one group and evaluation trace) is the
folding statement of `TraceClaim.ofPoint` for the segments `ρ_pack`, `ρ_pos`, `ρ_blk` of the
tensor point `ρ` of `verifyProof_tensored`, its factor `θ` and its final claim `ṽ`, with the
payload of the words `next_witness` of the level `level` and the matrices of the setup. The
segments have the lengths of `Tier.TraceLinked`: `2·m = 2^|ρ_pack|`, `M = 2^|ρ_pos|` and
`B ≤ 2^|ρ_blk|`. `width`, `length`, `long` and `blocks` are numbers of the layout. -/
theorem verifyProof_traceLink (scripted : script pub.layout pub.plan = .ok ops)
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
    {F : Fold} (traced : F.trace.isSome = true) (single : F.groups.length = 1) {S : Shape}
    (O : OpeningShape S) (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (width : 2 * (S.innerDegree / 4) = 2 ^ Nat.log 2 (S.innerDegree / 2))
    (length : S.blockLength = 2 ^ Nat.log 2 S.blockLength)
    (long : Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength ≤ ν)
    (blocks : S.blocks ≤ 2 ^ (ν - (Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength)))
    (setup : Array Fq) :
    let s := stateBefore pub proof messages ops c'
    let r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
    let ρ := (recordedIn s ⟨level + 1, "tensor.sumcheck.point", none⟩).toList
    let η := firstIn s ⟨level + 1, "tensor.eta", none⟩
    let pack := (packPoint S ρ).map (Ext.into (E := E))
    let position := (positionPoint S ρ).map (Ext.into (E := E))
    let block := (blockPoint S ρ).map (Ext.into (E := E))
    pack ++ (position ++ block) = ρ.map Ext.into ∧ S.blocks ≤ 2 ^ block.length ∧
      S.blockLength = 2 ^ position.length ∧ 2 * (S.innerDegree / 4) = 2 ^ pack.length ∧
      traceStatementFor (E := E) O quarter setup (traceValuesIn s (level + 1) F)
          (dataIn pub s (level + 1)) .witness (rowsIn s (level + 1) F)
          (claimsIn s (level + 1) F) =
        (Extract.TraceClaim.ofPoint quarter
          (payloadOf S.chain _ _ ((s.fields.get? ⟨level, "next_witness", none⟩).getD #[]))
          (payloadOf O.chain _ _ ((s.fields.get? ⟨level + 1, "opening_payload", none⟩).getD #[]))
          block position pack (mle (ρ.map Ext.into) (tensorWeight (r2.tail.map Ext.into) η.into))
          (firstIn s ⟨level + 1, "tensor.final_claims", none⟩).into).statement
          (Key.ofSetup (flatOf setup) S) (OpeningKey.ofSetup (flatOf setup) O) := by
  obtain ⟨rhoLength, tailLength, -⟩ := verifyProof_tensorLink (E := E) pub proof scripted ends
    terminalSetup accepted decoded at_ shape after kept (fun _ => 0)
  dsimp only at rhoLength tailLength ⊢
  obtain ⟨packLength, positionLength, blockLength⟩ := segments_length S _ (rhoLength ▸ long)
  have rows : rowsIn (stateBefore pub proof messages ops c') (level + 1) F = #[Ext.one] := by
    simp [rowsIn, single]
  have claims : claimsIn (stateBefore pub proof messages ops c') (level + 1) F =
      recordedIn (stateBefore pub proof messages ops c') ⟨level + 1, "tensor.final_claims", none⟩ := by
    simp [claimsIn, traced]
  obtain ⟨witness, payload⟩ := dataIn_runs pub (stateBefore pub proof messages ops c') level S O
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [← List.map_append, ← List.map_append, segments_append]
  · rw [List.length_map, blockLength, rhoLength]
    exact blocks
  · rw [List.length_map, positionLength]
    exact length
  · rw [List.length_map, packLength]
    exact width
  · have same : (traceValuesIn (stateBefore pub proof messages ops c') (level + 1) F).point.length =
        (traceValuesIn (stateBefore pub proof messages ops c') (level + 1) F).tail.length := by
      show (recordedIn (stateBefore pub proof messages ops c')
          ⟨level + 1, "tensor.sumcheck.point", none⟩).toList.length =
        (recordedIn (stateBefore pub proof messages ops c') ⟨level, "fused.point", none⟩).toList.tail.length
      rw [rhoLength, tailLength]
    rw [rows, claims, traceStatementFor_one quarter setup _ _ _ same, witness, payload]
    rfl

end Proof

end NightstreamFPrime.Spec.Compression.Akita.Replay
