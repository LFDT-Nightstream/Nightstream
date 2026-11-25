import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinExists

/-! The `Note` of each wire message of the model from the state of the replay before the work of
its stage, for `key20.aks` (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23,
Figure 4; `docs/akita-work/w17/design.md` section 2, track N).

- `stageIndex20 pub k` is the index of the work of the stage `k` (`workIndices20`), and the index
  of the terminal for `k ≥ 473`. `noteAt20 … k` is `noteOf` on the state before it.
- `held20 c c' K`: no operation of the skeleton of `layout20` from `c` to `c'` writes a key of `K`
  (a Bool, closed computations of the kernel: `held20_claim` to `held20_partials`), and
  `works20` gives the indices of the works of the stages that `Key20.Tower.accepts` reads.
  `skeleton20_ordered`: each message of the prover is after the squeezes of the stage before its
  work.
- `held_of`: on a script of `layout20`, the keys of `held20 c c' K` have the same recorded values
  and field message before `c` and before `c'`.
- `rootClaimIn_congr`, `claimIn₁_congr`, `traceClaimIn_congr`, `terminalStatementIn_congr`,
  `stateIn_congr`, `partialsIn_congr`: each value of `noteOf` is a function of the recorded values
  of its keys.
- `noteAt20_reads`: for an accepted proof of `layout20`, the readings of `Key20.Tower.accepts` on
  the notes `noteAt20` are those of `note20` (`Reads20`): the code absorbs each value that the
  model reads in the wire message `k` before the work of the stage `k`, and no later operation
  writes it.
- `Prefix.mapWire f point` is the transcript with each wire message `k` replaced by `f k` of it.
  `SameRead read w w'`: the reader gives the same messages of the checks for `w` and `w'`.
  `pathOf_congr` and `Tier.Shows.mapWire`: a level shows the same challenges and path after a
  change of the wire messages that keeps the messages of the checks and the checks of the level.
- `renote note k wire` is `wire` with the data `note k`. `presents20_staged`: a transcript that
  presents the bytes of an accepted proof still presents them with the notes `noteAt20`.
- `exists_presents20_staged` is theorem A with the notes `noteAt20`: the `Note` of each wire
  message `k ≤ 473` is `noteOf` on the state before the work of the stage `k`.
  `verifyProof_exists_accepts20_staged` adds `Last20` and `Key20.Tower.accepts`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

section Skeleton

/-- No operation of the skeleton of `layout20` from `c` to `c'` writes a key of `K`. -/
def held20 (c c' : ℕ) (K : List Key) : Bool :=
  decide (c ≤ c') && frees ((sk20.take c').drop c) K []

/-- The keys that `traceClaimIn` reads at the level `level`. -/
def traceKeys (level : ℕ) : List Key :=
  [⟨level, "fused.point", none⟩, ⟨level + 1, "tensor.sumcheck.point", none⟩,
    ⟨level + 1, "tensor.eta", none⟩, ⟨level, "next_witness", none⟩,
    ⟨level + 1, "opening_payload", none⟩, ⟨level + 1, "tensor.final_claims", none⟩]

/-- The keys that `terminalStatementIn` reads. -/
def terminalKeys : List Key :=
  [⟨5, "fused.point", none⟩, ⟨6, "tensor.sumcheck.point", none⟩, ⟨6, "tensor.eta", none⟩,
    ⟨6, "tensor.final_claims", none⟩, ⟨5, "terminal_t", none⟩]

/-- The indices of the works of the stages whose wire message `Key20.Tower.accepts` reads. -/
theorem works20 : workIndices20.getD 1 0 = 11 ∧ workIndices20.getD 2 0 = 16 ∧
    workIndices20.getD 51 0 = 170 ∧ workIndices20.getD 52 0 = 173 ∧
    workIndices20.getD 120 0 = 389 ∧ workIndices20.getD 121 0 = 392 ∧
    workIndices20.getD 203 0 = 653 ∧ workIndices20.getD 204 0 = 656 ∧
    workIndices20.getD 284 0 = 912 ∧ workIndices20.getD 285 0 = 915 ∧
    workIndices20.getD 378 0 = 1214 ∧ workIndices20.getD 379 0 = 1218 ∧
    workIndices20.getD 472 0 = 1518 := by
  decide +kernel

theorem held20_claim :
    held20 11 165 [⟨0, "row_coefficients", none⟩, ⟨0, "opening_payload", none⟩] = true := by
  decide +kernel

theorem held20_commit₁ : held20 16 316 [⟨0, "next_witness", none⟩] = true := by decide +kernel

theorem held20_claim₁ :
    held20 165 170 [⟨0, "fused.point", none⟩, ⟨0, "witness_evaluation", none⟩] = true ∧
      held20 170 316 [⟨0, "next_witness", none⟩, ⟨1, "opening_payload", none⟩] = true := by
  decide +kernel

theorem held20_commit₂ : held20 173 586 [⟨1, "next_witness", none⟩] = true := by decide +kernel

theorem held20_claim₂ : held20 389 586 (traceKeys 1) = true := by decide +kernel

theorem held20_commit₃ : held20 392 848 [⟨2, "next_witness", none⟩] = true := by decide +kernel

theorem held20_claim₃ : held20 653 848 (traceKeys 2) = true := by decide +kernel

theorem held20_commit₄ : held20 656 1153 [⟨3, "next_witness", none⟩] = true := by decide +kernel

theorem held20_claim₄ : held20 912 1153 (traceKeys 3) = true := by decide +kernel

theorem held20_commit₅ : held20 915 1456 [⟨4, "next_witness", none⟩] = true := by decide +kernel

theorem held20_claim₅ : held20 1214 1456 (traceKeys 4) = true := by decide +kernel

theorem held20_state : held20 1218 1521 [⟨5, "terminal_t", none⟩] = true := by decide +kernel

theorem held20_statement : held20 1518 1521 terminalKeys = true := by decide +kernel

theorem held20_partials : held20 1518 1521 [⟨6, "terminal.e", none⟩] = true := by decide +kernel

/-- The operation squeezes challenges from the sponge. -/
def isDraw : Op → Bool
  | .challenge .. | .squeeze .. => true
  | _ => false

/-- The operations of the skeleton of `layout20` after the work of the stage `k - 1` and before
the work of the stage `k`. -/
def between20 (k : ℕ) : List (Op × ℕ) :=
  (sk20.take (workIndices20.getD k 0)).drop (if k = 0 then 0 else workIndices20.getD (k - 1) 0 + 1)

/-- No operation between the works of the stages `k - 1` and `k` is a work, and none squeezes
challenges after an operation that reads a message. -/
def ordered20 (k : ℕ) : Bool :=
  (between20 k).all (fun x => !isWork x.1) &&
    ((between20 k).dropWhile fun x => !x.1.reads).all fun x => !isDraw x.1

/-- Each message of the prover that the script of `layout20` reads is between the works of two
stages `k - 1` and `k`, after each squeeze of challenges between them: the code absorbs it
before the work of the stage `k`. -/
theorem skeleton20_ordered : (List.range 473).all ordered20 = true := by decide +kernel

end Skeleton

section Frames

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- On a script of `layout20`, a key that no operation of the skeleton from `c` to `c'` writes has
the same recorded values and field message before `c` and before `c'`. -/
theorem held_of {ops : Array Op} (erased : ops.toList.map Op.erase = skeleton (layout20 0 0))
    {c c' : ℕ} {K : List Key} (shaped : held20 c c' K = true) :
    ∀ k ∈ K, recordedIn (stateBefore pub proof messages ops c') k =
        recordedIn (stateBefore pub proof messages ops c) k ∧
      (stateBefore pub proof messages ops c').fields.get? k =
        (stateBefore pub proof messages ops c).fields.get? k := by
  unfold held20 at shaped
  simp only [Bool.and_eq_true, decide_eq_true_eq] at shaped
  exact (stateBefore_frees pub proof messages shaped.1 (joined_frees erased _ _ _ _ shaped.2)).1

end Frames

section Congr

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E] (pub : Public) {s s' : Replayer}

/-- `rootClaimIn` is a function of `row_coefficients` and of the opening payload of the level 0. -/
theorem rootClaimIn_congr
    (weights : recordedIn s' ⟨0, "row_coefficients", none⟩ =
      recordedIn s ⟨0, "row_coefficients", none⟩)
    (payload : s'.fields.get? ⟨0, "opening_payload", none⟩ =
      s.fields.get? ⟨0, "opening_payload", none⟩) :
    rootClaimIn (E := E) pub s' = rootClaimIn pub s := by
  simp only [rootClaimIn, weights, payload]

/-- `claimIn₁` is a function of `fused.point` and `witness_evaluation` of the level 0 of its first
state, and of `next_witness` of the level 0 and the opening payload of the level 1 of its second
state. -/
theorem claimIn₁_congr {t t' : Replayer}
    (point : recordedIn s' ⟨0, "fused.point", none⟩ = recordedIn s ⟨0, "fused.point", none⟩)
    (value : recordedIn s' ⟨0, "witness_evaluation", none⟩ =
      recordedIn s ⟨0, "witness_evaluation", none⟩)
    (witness : t'.fields.get? ⟨0, "next_witness", none⟩ = t.fields.get? ⟨0, "next_witness", none⟩)
    (payload : t'.fields.get? ⟨1, "opening_payload", none⟩ =
      t.fields.get? ⟨1, "opening_payload", none⟩) :
    claimIn₁ (E := E) s' t' = claimIn₁ s t := by
  simp only [claimIn₁, firstIn, point, value, witness, payload]

/-- The payload of `claimIn₁` is a function of `next_witness` of the level 0 of its second
state. -/
theorem claimIn₁_payload {t t' : Replayer}
    (witness : t'.fields.get? ⟨0, "next_witness", none⟩ =
      t.fields.get? ⟨0, "next_witness", none⟩) :
    (claimIn₁ (E := E) s' t').payload = (claimIn₁ (E := E) s t).payload := by
  simp only [claimIn₁, witness]

/-- `traceClaimIn` is a function of the recorded values and field messages of `traceKeys`. -/
theorem traceClaimIn_congr {level : ℕ} (L' : Key20.Level) {m : ℕ}
    (quarter : 4 * m = L'.shape.innerDegree)
    (same : ∀ k ∈ traceKeys level,
      recordedIn s' k = recordedIn s k ∧ s'.fields.get? k = s.fields.get? k) :
    traceClaimIn (E := E) s' level L' quarter = traceClaimIn s level L' quarter := by
  have point := (same ⟨level, "fused.point", none⟩ (by simp [traceKeys])).1
  have tensor := (same ⟨level + 1, "tensor.sumcheck.point", none⟩ (by simp [traceKeys])).1
  have eta := (same ⟨level + 1, "tensor.eta", none⟩ (by simp [traceKeys])).1
  have witness := (same ⟨level, "next_witness", none⟩ (by simp [traceKeys])).2
  have payload := (same ⟨level + 1, "opening_payload", none⟩ (by simp [traceKeys])).2
  have final := (same ⟨level + 1, "tensor.final_claims", none⟩ (by simp [traceKeys])).1
  simp only [traceClaimIn, firstIn, point, tensor, eta, witness, payload, final]

/-- The payload of `traceClaimIn` is a function of `next_witness` of its level. -/
theorem traceClaimIn_payload {level : ℕ} (L' : Key20.Level) {m : ℕ}
    (quarter : 4 * m = L'.shape.innerDegree)
    (witness : s'.fields.get? ⟨level, "next_witness", none⟩ =
      s.fields.get? ⟨level, "next_witness", none⟩) :
    (traceClaimIn (E := E) s' level L' quarter).payload =
      (traceClaimIn (E := E) s level L' quarter).payload := by
  simp only [traceClaimIn, Extract.TraceClaim.ofPoint, witness]

/-- The state `(t_i)_i` of the terminal is a function of the field message `terminal_t`. -/
theorem stateIn_congr (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (sent : s'.fields.get? ⟨5, "terminal_t", none⟩ = s.fields.get? ⟨5, "terminal_t", none⟩) :
    stateOf Key20.terminal (terminalMessagesIn s' 6) tLast =
      stateOf Key20.terminal (terminalMessagesIn s 6) tLast := by
  have same : (terminalMessagesIn s' 6).t = (terminalMessagesIn s 6).t :=
    congrArg (fun found => found.getD #[]) sent
  unfold stateOf
  rw [same]

/-- The partials `(e_i)_i` of the terminal are a function of the field message `terminal.e`. -/
theorem partialsIn_congr (eLast : Rq Key20.terminal.innerDegree)
    (sent : s'.fields.get? ⟨6, "terminal.e", none⟩ = s.fields.get? ⟨6, "terminal.e", none⟩) :
    partialsOf Key20.terminal (terminalMessagesIn s' 6) eLast =
      partialsOf Key20.terminal (terminalMessagesIn s 6) eLast := by
  have same : (terminalMessagesIn s' 6).e = (terminalMessagesIn s 6).e :=
    congrArg (fun found => found.getD #[]) sent
  unfold partialsOf
  rw [same]

/-- `terminalStatementIn` is a function of the recorded values and field messages of
`terminalKeys`. -/
theorem terminalStatementIn_congr (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (same : ∀ k ∈ terminalKeys,
      recordedIn s' k = recordedIn s k ∧ s'.fields.get? k = s.fields.get? k) :
    terminalStatementIn (E := E) pub s' tLast = terminalStatementIn pub s tLast := by
  have point := (same ⟨5, "fused.point", none⟩ (by simp [terminalKeys])).1
  have tensor := (same ⟨6, "tensor.sumcheck.point", none⟩ (by simp [terminalKeys])).1
  have eta := (same ⟨6, "tensor.eta", none⟩ (by simp [terminalKeys])).1
  have final := (same ⟨6, "tensor.final_claims", none⟩ (by simp [terminalKeys])).1
  have sent := stateIn_congr tLast (same ⟨5, "terminal_t", none⟩ (by simp [terminalKeys])).2
  simp only [terminalStatementIn, firstIn, point, tensor, eta, final, sent]

end Congr

section Stage

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]
  (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)

/-- The index of the operation before which the code has the wire message `k` of the model: the
work of the stage `k`, and the terminal for `k ≥ 473`. -/
def stageIndex20 (k : ℕ) : ℕ := if k < 473 then workIndices20.getD k 0 else pub.ops.size - 1

theorem stageIndex20_lt {k : ℕ} (inside : k < 473) :
    stageIndex20 pub k = workIndices20.getD k 0 := by
  simp only [stageIndex20, inside, ite_true]

/-- The `Note` of the wire message `k` of the model: `noteOf` on the state of the replay before
the work of the stage `k`. -/
noncomputable def noteAt20 (k : ℕ) : Key20.Note E :=
  noteOf pub (stateBefore pub proof messages pub.ops (stageIndex20 pub k)) tLast eLast

/-- The readings of `Key20.Tower.accepts` on the notes `noteAt20` of an accepted proof of
`layout20` are those of `note20`: each value that the model reads in the wire message `k` is
recorded before the work of the stage `k`, and no operation from there to the state of `note20`
writes its key. -/
theorem noteAt20_reads {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept) :
    Reads20 pub proof messages tLast eLast (noteAt20 (E := E) pub proof messages tLast eLast) := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    have scripted := verifyProof_scripted accepted
    rw [layout] at scripted
    rw [script_erase scripted, skeleton20]
  have last : pub.ops.size - 1 = 1521 := by rw [size_skeleton erased, skeleton20_length]
  obtain ⟨w1, w2, w51, w52, w120, w121, w203, w204, w284, w285, w378, w379, w472⟩ := works20
  obtain ⟨f0, f1, f2, f3, f4, f5⟩ := fusedIndex20_values
  have at_ : ∀ ℓ, fusedIndex pub.layout ℓ = fusedIndex (layout20 0 0) ℓ := fun ℓ => by
    rw [layout, fusedIndex20]
  have held := @held_of pub proof messages pub.ops erased
  refine ⟨rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · show rootClaimIn (E := E) pub (stateBefore pub proof messages pub.ops (stageIndex20 pub 1)) =
      rootClaimIn (E := E) pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0))
    rw [stageIndex20_lt pub (by decide), w1, at_, f0]
    have same := held held20_claim
    exact (rootClaimIn_congr pub (same _ (by simp)).1 (same _ (by simp)).2).symm
  · show (claimIn₁ (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 2))
        (stateBefore pub proof messages pub.ops (stageIndex20 pub 2))).payload =
      (claimIn₁ (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0))
        (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 1))).payload
    rw [stageIndex20_lt pub (by decide), w2, at_, at_, f0, f1]
    exact (claimIn₁_payload ((held held20_commit₁) _ (by simp)).2).symm
  · show claimIn₁ (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 51))
        (stateBefore pub proof messages pub.ops (stageIndex20 pub 51)) =
      claimIn₁ (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 0))
        (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 1))
    rw [stageIndex20_lt pub (by decide), w51, at_, at_, f0, f1]
    have before := held held20_claim₁.1
    have after := held held20_claim₁.2
    exact claimIn₁_congr (before _ (by simp)).1 (before _ (by simp)).1 (after _ (by simp)).2.symm
      (after _ (by simp)).2.symm
  · show (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 52)) 1
        Key20.level₂ Key20.quarters.1).payload =
      (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 1
        Key20.level₂ Key20.quarters.1).payload
    rw [stageIndex20_lt pub (by decide), w52, at_, f2]
    exact (traceClaimIn_payload _ _ ((held held20_commit₂) _ (by simp)).2).symm
  · show traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 120)) 1
        Key20.level₂ Key20.quarters.1 =
      traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 1
        Key20.level₂ Key20.quarters.1
    rw [stageIndex20_lt pub (by decide), w120, at_, f2]
    exact (traceClaimIn_congr _ _ (held held20_claim₂)).symm
  · show (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 121)) 2
        Key20.level₃ Key20.quarters.2.1).payload =
      (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3)) 2
        Key20.level₃ Key20.quarters.2.1).payload
    rw [stageIndex20_lt pub (by decide), w121, at_, f3]
    exact (traceClaimIn_payload _ _ ((held held20_commit₃) _ (by simp)).2).symm
  · show traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 203)) 2
        Key20.level₃ Key20.quarters.2.1 =
      traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3)) 2
        Key20.level₃ Key20.quarters.2.1
    rw [stageIndex20_lt pub (by decide), w203, at_, f3]
    exact (traceClaimIn_congr _ _ (held held20_claim₃)).symm
  · show (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 204)) 3
        Key20.level₄ Key20.quarters.2.2.1).payload =
      (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4)) 3
        Key20.level₄ Key20.quarters.2.2.1).payload
    rw [stageIndex20_lt pub (by decide), w204, at_, f4]
    exact (traceClaimIn_payload _ _ ((held held20_commit₄) _ (by simp)).2).symm
  · show traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 284)) 3
        Key20.level₄ Key20.quarters.2.2.1 =
      traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4)) 3
        Key20.level₄ Key20.quarters.2.2.1
    rw [stageIndex20_lt pub (by decide), w284, at_, f4]
    exact (traceClaimIn_congr _ _ (held held20_claim₄)).symm
  · show (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 285)) 4
        Key20.level₅ Key20.quarters.2.2.2.1).payload =
      (traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5)) 4
        Key20.level₅ Key20.quarters.2.2.2.1).payload
    rw [stageIndex20_lt pub (by decide), w285, at_, f5]
    exact (traceClaimIn_payload _ _ ((held held20_commit₅) _ (by simp)).2).symm
  · show traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (stageIndex20 pub 378)) 4
        Key20.level₅ Key20.quarters.2.2.2.1 =
      traceClaimIn (E := E) (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5)) 4
        Key20.level₅ Key20.quarters.2.2.2.1
    rw [stageIndex20_lt pub (by decide), w378, at_, f5]
    exact (traceClaimIn_congr _ _ (held held20_claim₅)).symm
  · show stateOf Key20.terminal
        (terminalMessagesIn (stateBefore pub proof messages pub.ops (stageIndex20 pub 379)) 6)
        tLast =
      stateOf Key20.terminal
        (terminalMessagesIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1)) 6) tLast
    rw [stageIndex20_lt pub (by decide), w379, last]
    exact (stateIn_congr tLast ((held held20_state) _ (by simp)).2).symm
  · show terminalStatementIn (E := E) pub (stateBefore pub proof messages pub.ops (stageIndex20 pub 472))
        tLast =
      terminalStatementIn (E := E) pub (stateBefore pub proof messages pub.ops (pub.ops.size - 1)) tLast
    rw [stageIndex20_lt pub (by decide), w472, last]
    exact (terminalStatementIn_congr pub tLast (held held20_statement)).symm
  · show partialsOf Key20.terminal
        (terminalMessagesIn (stateBefore pub proof messages pub.ops (stageIndex20 pub 472)) 6)
        eLast =
      partialsOf Key20.terminal
        (terminalMessagesIn (stateBefore pub proof messages pub.ops (pub.ops.size - 1)) 6) eLast
    rw [stageIndex20_lt pub (by decide), w472, last]
    exact (partialsIn_congr eLast ((held held20_partials) _ (by simp)).2).symm
  · rfl

end Stage

section MapWire

variable {Wire Reply : Type} {total : ℕ}

/-- The transcript with each wire message `k` replaced by `f k` of it. -/
def Prefix.mapWire (f : ℕ → Wire → Wire) (point : Prefix Wire Reply total) :
    Prefix Wire Reply total :=
  ⟨f 0 point.first, point.rest.zipIdx.map fun x => (x.1.1, f (x.2 + 1) x.1.2), by
    rw [List.length_map, List.length_zipIdx]
    exact point.short⟩

theorem Prefix.mapWire_length (f : ℕ → Wire → Wire) (point : Prefix Wire Reply total) :
    (Prefix.mapWire f point).rest.length = point.rest.length := by
  show (point.rest.zipIdx.map _).length = _
  rw [List.length_map, List.length_zipIdx]

theorem Prefix.mapWire_reply (f : ℕ → Wire → Wire) (point : Prefix Wire Reply total) (k : ℕ) :
    (Prefix.mapWire f point).reply k = point.reply k := by
  show ((point.rest.zipIdx.map _)[k]?).map Prod.fst = (point.rest[k]?).map Prod.fst
  rw [List.getElem?_map, List.getElem?_zipIdx]
  cases point.rest[k]? <;> rfl

theorem Prefix.mapWire_wire (f : ℕ → Wire → Wire) (point : Prefix Wire Reply total) {k : ℕ}
    (inside : k ≤ point.rest.length) : (Prefix.mapWire f point).wire k = f k (point.wire k) := by
  rcases k with _ | k
  · rfl
  · have found : point.rest[k]? = some point.rest[k] := List.getElem?_eq_getElem inside
    show (((point.rest.zipIdx.map _)[k]?).map Prod.snd).getD _ =
      f (k + 1) (((point.rest[k]?).map Prod.snd).getD point.first)
    rw [List.getElem?_map, List.getElem?_zipIdx, found]
    simp

/-- A list is related entry by entry to its image under a map of its entries with their
indices. -/
theorem forall₂_zipIdx_map {α β : Type} {R : β → α → Prop} (g : α × ℕ → β)
    (each : ∀ x i, R (g (x, i)) x) : ∀ (l : List α) (n : ℕ), List.Forall₂ R ((l.zipIdx n).map g) l
  | [], _ => .nil
  | x :: l, n => .cons (each x n) (forall₂_zipIdx_map g each l (n + 1))

/-- The reader gives the same messages of the checks for the wire messages `w` and `w'`. -/
def SameRead (read : Extract.Reader Wire) (w w' : Wire) : Prop :=
  ∀ (M : Type) [Nonempty M] (i : ℕ), read.message M w i = read.message M w' i

/-- The path of a game on a transcript segment is the same for wire messages with the same
messages of the checks and the same answers. -/
theorem pathOf_congr {α : Type} [Fintype Reply] (read : Extract.Reader Wire) (game : Game α) :
    ∀ (wire wire' : Wire) (index : ℕ) (moves moves' : List (Reply × Wire)),
      SameRead read wire wire' →
      List.Forall₂ (fun a b => a.1 = b.1 ∧ SameRead read a.2 b.2) moves moves' →
      Extract.pathOf read game wire index moves = Extract.pathOf read game wire' index moves' := by
  induction game with
  | pure a => exact fun _ _ _ _ _ _ _ => rfl
  | draw C bad next induction =>
      intro wire wire' index moves moves' first rest
      cases rest with
      | nil => rfl
      | @cons a b l l' head tail =>
          show Move.draw (Extract.sample Reply C a.1) ::
              Extract.pathOf read (next (Extract.sample Reply C a.1)) a.2 0 l =
            Move.draw (Extract.sample Reply C b.1) ::
              Extract.pathOf read (next (Extract.sample Reply C b.1)) b.2 0 l'
          rw [head.1]
          exact congrArg _ (induction _ _ _ _ _ _ head.2 tail)
  | recv M next induction =>
      intro wire wire' index moves moves' first rest
      show Move.recv (read.message M wire index) ::
          Extract.pathOf read (next (read.message M wire index)) wire (index + 1) moves =
        Move.recv (read.message M wire' index) ::
          Extract.pathOf read (next (read.message M wire' index)) wire' (index + 1) moves'
      rw [first M index]
      exact congrArg _ (induction _ _ _ _ _ _ first rest)

/-- Wire messages with the same cells have the same messages of the checks. -/
theorem sameRead_page {Data F : Type} {slots width cap : ℕ} [Zero F]
    {w w' : Extract.Page Data F slots width cap} (cells : w.cells = w'.cells) :
    SameRead Extract.Page.read w w' := by
  intro M _ i
  show (if inside : i < slots then (w.cells ⟨i, inside⟩).message M else Classical.arbitrary M) =
    (if inside : i < slots then (w'.cells ⟨i, inside⟩).message M else Classical.arbitrary M)
  rw [cells]

end MapWire

section Shows

variable {Block : Type} {Ring : Block → Type} [∀ b, Zero (Ring b)]
  {Stmt Brk Wit Weak Wire Coord Slot Result Source : Type} [Fintype Coord] [DecidableEq Coord]
  [Fintype Slot] [Nonempty Slot] {total : ℕ}

omit [Nonempty Slot] in
/-- A level shows the same challenges and path on the transcript with the wire messages `f k`,
when `f` keeps the messages of the checks and the checks of the level at its fold stage. -/
theorem Tier.Shows.mapWire {T : Tier Block Ring Stmt Brk Wit Weak Wire Coord Slot Result Source}
    {c : (b : Block) → Ring b} {path : List Move} {source : Source}
    {point : Prefix Wire (Option Coord → Slot) total} (f : ℕ → Wire → Wire)
    (long : T.finish ≤ point.rest.length)
    (check : T.check (f T.start (point.wire T.start)) = T.check (point.wire T.start))
    (reads : ∀ k w, SameRead T.read (f k w) w) (shows : Tier.Shows T c path source point) :
    Tier.Shows T c path source (Prefix.mapWire f point) := by
  obtain ⟨⟨reply, answered, decoded⟩, presented⟩ := shows
  have started : T.start + 1 ≤ point.rest.length :=
    (show T.start + 1 ≤ T.finish from Nat.le_add_right _ _).trans long
  have challenge : T.challenge (Prefix.mapWire f point) = T.challenge point := by
    funext b
    unfold Tier.challenge Extract.challengeAt
    rw [Prefix.mapWire_reply]
  refine ⟨⟨reply, (Prefix.mapWire_reply f point _).trans answered, decoded⟩, ?_⟩
  show Extract.pathOf T.read
    (T.check ((Prefix.mapWire f point).wire T.start) (T.challenge (Prefix.mapWire f point)) source)
    ((Prefix.mapWire f point).wire (T.start + 1)) 0
    ((((Prefix.mapWire f point).rest.take T.finish).drop (T.start + 1)).map
      fun move => (view move.1, move.2)) = path
  rw [challenge, Prefix.mapWire_wire f point (Nat.le_of_succ_le started),
    Prefix.mapWire_wire f point started, check]
  refine (pathOf_congr T.read _ _ _ 0 _ _ (reads _ _) ?_).trans presented
  refine List.forall₂_map_left_iff.mpr (List.forall₂_map_right_iff.mpr ?_)
  refine List.Forall₂.imp (fun a b same => ⟨congrArg view same.1, same.2⟩)
    (List.forall₂_drop _ (List.forall₂_take _ (forall₂_zipIdx_map
      (R := fun a b => a.1 = b.1 ∧ SameRead T.read a.2 b.2) _
      (fun x i => ⟨rfl, reads _ _⟩) point.rest 0)))

end Shows

section Staged

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

omit [Degree2 E] [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the root fold read a wire message only through its claim. -/
theorem zeroth_check {w w' : Key20.Wire E} (same : o.claim w = o.claim w') :
    (zeroth).check w = (zeroth).check w' := by
  simp only [Extract.liftBatch, same]

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 1 read a wire message only through its claim. -/
theorem first_check {w w' : Key20.Wire E} (same : o.claim₁ w = o.claim₁ w') :
    (first).check w = (first).check w' := by
  simp only [Tier.tensored, Extract.liftFold, same]

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 2 read a wire message only through its claim. -/
theorem second_check {w w' : Key20.Wire E} (same : o.claim₂ w = o.claim₂ w') :
    (second).check w = (second).check w' := by
  simp only [Tier.tensored, Extract.committedTrace, same]

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 3 read a wire message only through its claim. -/
theorem third_check {w w' : Key20.Wire E} (same : o.claim₃ w = o.claim₃ w') :
    (third).check w = (third).check w' := by
  simp only [Tier.tensored, Extract.committedTraceSq, same]

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 4 read a wire message only through its claim. -/
theorem fourth_check {w w' : Key20.Wire E} (same : o.claim₄ w = o.claim₄ w') :
    (fourth).check w = (fourth).check w' := by
  simp only [Tier.tensored, Extract.committedTraceSq, same]

omit [Fintype Coord] [DecidableEq Coord] in
/-- The checks of the level 5 read a wire message only through its claim. -/
theorem fifth_check {w w' : Key20.Wire E} (same : o.claim₅ w = o.claim₅ w') :
    (fifth).check w = (fifth).check w' := by
  simp only [Tier.tensored, Extract.committedTraceSq, same]

/-- The wire message `wire` with the data `note k`. -/
def renote (note : ℕ → Key20.Note E) (k : ℕ) (wire : Key20.Wire E) : Key20.Wire E :=
  ⟨note k, wire.cells, wire.nonce⟩

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

omit [Fintype E] [DecidableEq E] in
/-- Notes that agree with notes with the readings of `note20` up to the wire message 473 have
these readings. -/
theorem Reads20.congr {note note' : ℕ → Key20.Note E} (same : ∀ k ≤ 473, note' k = note k)
    (reads : Reads20 pub proof messages tLast eLast note) :
    Reads20 pub proof messages tLast eLast note' :=
  ⟨(congrArg Key20.Note.commit (same 0 (by decide))).trans reads.commit,
    (congrArg Key20.Note.value (same 0 (by decide))).trans reads.value,
    (congrArg Key20.Note.point (same 0 (by decide))).trans reads.place,
    (congrArg Key20.Note.claim (same 1 (by decide))).trans reads.claim,
    (congrArg Key20.Note.commit₁ (same 2 (by decide))).trans reads.commit₁,
    (congrArg Key20.Note.claim₁ (same 51 (by decide))).trans reads.claim₁,
    (congrArg Key20.Note.commit₂ (same 52 (by decide))).trans reads.commit₂,
    (congrArg Key20.Note.claim₂ (same 120 (by decide))).trans reads.claim₂,
    (congrArg Key20.Note.commit₃ (same 121 (by decide))).trans reads.commit₃,
    (congrArg Key20.Note.claim₃ (same 203 (by decide))).trans reads.claim₃,
    (congrArg Key20.Note.commit₄ (same 204 (by decide))).trans reads.commit₄,
    (congrArg Key20.Note.claim₄ (same 284 (by decide))).trans reads.claim₄,
    (congrArg Key20.Note.commit₅ (same 285 (by decide))).trans reads.commit₅,
    (congrArg Key20.Note.claim₅ (same 378 (by decide))).trans reads.claim₅,
    (congrArg Key20.Note.state (same 379 (by decide))).trans reads.state,
    (congrArg Key20.Note.statement (same 472 (by decide))).trans reads.statement,
    (congrArg Key20.Note.partials (same 472 (by decide))).trans reads.partials,
    (congrArg Key20.Note.response (same 473 (by decide))).trans reads.response⟩

/-- A transcript that presents the bytes of an accepted proof of `layout20` presents them also
with the notes `noteAt20` in place of the data of its wire messages: the `Note` of the wire
message `k ≤ 473` is then `noteOf` on the state of the replay before the work of the stage `k`. -/
theorem presents20_staged {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : runProof pub proof = .verdict .accept)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    PresentsAt20 o pub proof messages tLast eLast
        (Prefix.mapWire (renote (noteAt20 pub proof messages tLast eLast)) point) ∧
      ∀ k ≤ 473, ((Prefix.mapWire (renote (noteAt20 pub proof messages tLast eLast)) point).wire
        k).data = noteAt20 pub proof messages tLast eLast k := by
  have staged := noteAt20_reads (E := E) pub proof messages tLast eLast layout accepted
  have long := presents.long
  have old := presents.reads
  have data : ∀ k ≤ 473,
      ((Prefix.mapWire (renote (noteAt20 pub proof messages tLast eLast)) point).wire k).data =
        noteAt20 pub proof messages tLast eLast k := fun k le => by
    rw [Prefix.mapWire_wire _ _ (le.trans long)]
    rfl
  obtain ⟨started₀, -, finished₀, -⟩ := zeroth_numbers o
  obtain ⟨started₁, -, finished₁, -⟩ := first20_numbers o
  obtain ⟨started₂, -, finished₂, -⟩ := second20_numbers o
  obtain ⟨started₃, -, finished₃, -⟩ := third20_numbers o
  obtain ⟨started₄, -, finished₄, -⟩ := fourth20_numbers o
  obtain ⟨started₅, -, finished₅, -⟩ := fifth20_numbers o
  subst tower
  have reads : ∀ (k : ℕ) (w : Key20.Wire E), SameRead Extract.Page.read
      (renote (noteAt20 pub proof messages tLast eLast) k w) w := fun k w => sameRead_page rfl
  refine ⟨⟨by rw [Prefix.mapWire_length]; exact long,
    Reads20.congr pub proof messages tLast eLast data staged, fun j below => ?_,
    fun k below reply answered =>
      presents.work k below reply ((Prefix.mapWire_reply _ _ k).symm.trans answered), ?_,
    Tier.Shows.mapWire _ (by rw [finished₀]; omega) ?_ reads presents.level₀,
    Tier.Shows.mapWire _ (by rw [finished₁]; omega) ?_ reads presents.level₁,
    Tier.Shows.mapWire _ (by rw [finished₂]; omega) ?_ reads presents.level₂,
    Tier.Shows.mapWire _ (by rw [finished₃]; omega) ?_ reads presents.level₃,
    Tier.Shows.mapWire _ (by rw [finished₄]; omega) ?_ reads presents.level₄,
    Tier.Shows.mapWire _ (by rw [finished₅]; omega) ?_ reads presents.level₅, ?_⟩, data⟩
  · rw [Prefix.mapWire_wire _ _
      (((by decide : ∀ j < 7, starts20.getD j 0 ≤ 473) j below).trans long)]
    exact presents.nonce j below
  · refine Eq.trans ?_ presents.weights
    unfold Extract.Root.weights
    rw [Prefix.mapWire_reply]
  · rw [started₀]
    exact zeroth_check _ (staged.claim.trans old.claim.symm)
  · rw [started₁]
    exact first_check _ (staged.claim₁.trans old.claim₁.symm)
  · rw [started₂]
    exact second_check _ (staged.claim₂.trans old.claim₂.symm)
  · rw [started₃]
    exact third_check _ (staged.claim₃.trans old.claim₃.symm)
  · rw [started₄]
    exact fourth_check _ (staged.claim₄.trans old.claim₄.symm)
  · rw [started₅]
    exact fifth_check _ (staged.claim₅.trans old.claim₅.symm)
  · obtain ⟨reply, answered, decoded⟩ := presents.terminal
    exact ⟨reply, (Prefix.mapWire_reply _ _ _).trans answered, decoded⟩

/-- Theorem A of the join for `key20.aks` with the notes of the stages: for an accepted proof of
`layout20` and each last blocks `tLast`, `eLast` of the terminal, a transcript of `tower20`
presents the bytes (`Presents20`), and the `Note` of its wire message `k ≤ 473` is `noteAt20`: the
`Note` `noteOf` of the state of the replay before the work of the stage `k`. The hypotheses are
those of `exists_presents20`. -/
theorem exists_presents20_staged {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
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
          point ∧
        ∀ k ≤ 473, (point.wire k).data = noteAt20 pub proof messages tLast eLast k := by
  obtain ⟨point, presents⟩ := exists_presents20 (E := E) pub proof messages tLast eLast layout
    accepted terminalSetup expanded decoded points values injective injective₁ injective₂
    injective₃ injective₄ injective₅ injectiveL slots inside
  exact ⟨_, presents20_staged _ pub proof messages tLast eLast layout accepted rfl presents⟩

/-- Theorem A with the notes of the stages and theorem B "→" for `key20.aks`: an accepted proof
of `layout20` has last blocks `tLast`, `eLast` of the terminal with `Last20` and a transcript of
`tower20` that presents its bytes with them, whose wire message `k ≤ 473` has the `Note`
`noteAt20` of the state of the replay before the work of the stage `k`, and `Key20.Tower.accepts`
of `tower20` accepts this transcript. -/
theorem verifyProof_exists_accepts20_staged {zMax rice : ℕ}
    (layout : pub.layout = layout20 zMax rice)
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
        (∀ k ≤ 473, (point.wire k).data = noteAt20 pub proof messages tLast eLast k) ∧
        (tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472).accepts point := by
  obtain ⟨tLast, eLast, last⟩ := last20_of_accept (E := E) layout accepted terminalSetup decoded
  obtain ⟨point, presents, notes⟩ := exists_presents20_staged (E := E) pub proof messages tLast
    eLast layout accepted terminalSetup expanded decoded points values injective injective₁
    injective₂ injective₃ injective₄ injective₅ injectiveL slots inside
  exact ⟨tLast, eLast, point, last, presents, notes, presents20_accepts pub proof messages tLast
    eLast layout accepted terminalSetup expanded decoded points values last presents⟩

end Staged

end NightstreamFPrime.Spec.Compression.Akita.Replay
