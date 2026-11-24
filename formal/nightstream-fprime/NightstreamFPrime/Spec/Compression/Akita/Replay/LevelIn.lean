import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelPack

/-! The level functions of `Replay/LevelFold.lean` and `Replay/LevelPack.lean` on the recorded
values of a state of the replay (`Replay/Transcript.lean`; Akita revision `d50f700`, `verify_fold`
of `fold/mod.rs`, the operations of `Build.fold`).

- `dataIn`, `rowsIn`, `claimsIn` and `stage2In` are the words of the runs, the row coefficients,
  the claims and the claims of Stage 2 that `stage2Of` reads; `stage2Of_run`: `stage2Of` gives
  `stage2In` and keeps the state.
- `wordsIn` and `normsIn` are the values that the checks of a fold read, and `finishIn` is the
  end check of `fused.end` at a point (`traceFinish`, `packFinish`); `finishIn_end`: at the
  recorded point `fused.point` with the recorded end claim `fused.output` it is `fusedEndIn`.
- `foldRunIn` is `foldRun` (coefficient route) or `foldRunSq` (Euclidean route) on these values,
  and `offloadRunIn` is `offloadRun` with Stage 3 of the plan. `foldRunIn_eq` and
  `offloadRunIn_eq`: they are what the script computes, the stage function of the last check of
  the range proof (`rangeCheckOf`: `range.leaf.end` or `norm.end`), then `fused.end`, then
  `stage3.end`, when the rounds of `fused` end at the recorded point and end claim.
- `foldRunIn_trace_runs`, `foldRunIn_traceSq_runs`, `foldRunIn_pack_runs`, `foldRunIn_batch_runs`
  and `offloadRunIn_runs` are `traceFoldRun_runs`, `traceFoldRunSq_runs`, `packFoldRun_runs`,
  `batchFoldRun_runs` and `batchOffloadRun_runs` for these values: the game of the level, on the
  path `foldPathIn` (`offloadPathIn`) of the recorded values, ends in the result of `foldRunIn`
  (`offloadRunIn`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

namespace Replayer

/-- The words of a run of the relation of the level `level` (`stage2Of`): the commitment rows of
a root group, the setup rows, the next witness of the level before and the opening payload. -/
def dataIn (pub : Public) (s : Replayer) (level : ℕ) (run : RhsRun) : Array Fq :=
  match run.source with
  | .root g => commitmentRows ((pub.groups[g]?.map (·.commitment)).getD .empty)
      (run.degree * run.count)
  | .setup => pub.setupRows
  | .witness => (s.fields.get? ⟨level - 1, "next_witness", none⟩).getD #[]
  | .payload => (s.fields.get? ⟨level, "opening_payload", none⟩).getD #[]

/-- The row coefficients of the level (one claim: `1`). -/
def rowsIn (s : Replayer) (level : ℕ) (F : Fold) : Array Ext :=
  if 1 < F.groups.length then recordedIn s ⟨level, "row_coefficients", none⟩ else #[Ext.one]

/-- The claims of the level: the final claims of the tensor reduction at a level with evaluation
trace, and otherwise the openings. -/
def claimsIn (s : Replayer) (level : ℕ) (F : Fold) : Array Ext :=
  recordedIn s ⟨level, if F.trace.isSome then "tensor.final_claims" else "openings", none⟩

/-- `stage2Of` on the recorded values of `s`. -/
def stage2In (pub : Public) (s : Replayer) (level : ℕ) (F : Fold) : Stage2 :=
  stage2 (recordedIn s ⟨level, "tau1", none⟩).toList (firstIn s ⟨level, "alpha", none⟩)
    (dataIn pub s level) F.rhs F.openingRow (rowsIn s level F) (claimsIn s level F)
    (firstIn s ⟨level, "stage2_batch", none⟩ * firstIn s ⟨level, "range.image", none⟩ +
      if F.euclidean then
        firstIn s ⟨level, "norm.virtual_batch", none⟩ *
          batched (recordedIn s ⟨level, "norm.virtual_evaluations", none⟩).size
            (recordedIn s ⟨level, "norm.virtual_evaluations", none⟩).toList
            (firstIn s ⟨level, "norm.virtual_batch", none⟩)
      else Ext.zero)

/-- `stage2Of` gives `stage2In` of the state and keeps the state. -/
theorem stage2Of_run (pub : Public) (level : ℕ) (F : Fold) (s : Replayer) :
    ((stage2Of pub level F).run).run s = (.ok (stage2In pub s level F), s) := by
  unfold stage2Of stage2In rowsIn
  by_cases many : 1 < F.groups.length
  · simp only [ite_eq_left many]
    cases F.euclidean <;> rfl
  · simp only [ite_eq_right many]
    cases F.euclidean <;> rfl

/-- The values that the checks of the fold of the level `level` read (`Build.fold`,
`Build.range`): `α`, `τ_0`, `τ_1`, the product stages, `s_claim`, `ζ_bin`, `γ` and the rounds of
`fused`. -/
def wordsIn (s : Replayer) (level : ℕ) (F : Fold) : FoldWords :=
  ⟨firstIn s ⟨level, "alpha", none⟩, (recordedIn s ⟨level, "tau0", none⟩).toList,
    (recordedIn s ⟨level, "tau1", none⟩).toList,
    productsIn s level ((productStages F.logBasis).map (·.1)),
    firstIn s ⟨level, "range.image", none⟩, firstIn s ⟨level, "binary_batch", none⟩,
    firstIn s ⟨level, "stage2_batch", none⟩, roundsIn s ⟨level, "fused", none⟩⟩

/-- The values of the norm stage of the level `level`: `E_resp`, `ζ_norm`, the rounds,
`z̃(r_virt)` and `η_norm`. -/
def normsIn (s : Replayer) (level : ℕ) : NormWords :=
  ⟨(s.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0,
    firstIn s ⟨level, "norm.merge", none⟩, roundsIn s ⟨level, "norm.sumcheck", none⟩,
    firstIn s ⟨level, "norm.virtual_evaluations", none⟩,
    firstIn s ⟨level, "norm.virtual_batch", none⟩⟩

/-- The end check of `fused.end` of the level `level` at the point `r_2` with the setup prefix
`setup`: `traceFinish` at a level with evaluation trace, `packFinish` at a level with coefficient
packing. -/
def finishIn (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) : List Ext → FusedEnd :=
  match F.fused with
  | .trace T => traceFinish T.opening T.fold setup
      ((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)) (traceValuesIn s level F)
  | .packing P => packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
      (packingValuesIn s level F) (setupClaimIn s level F)

theorem finishIn_trace (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold)
    {T : TraceShape} (shaped : F.fused = .trace T) :
    finishIn setup s level F = traceFinish T.opening T.fold setup
      ((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)) (traceValuesIn s level F) := by
  simp only [finishIn, shaped]

theorem finishIn_packing (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold)
    {P : PackShape} (shaped : F.fused = .packing P) :
    finishIn setup s level F = packFinish (packInputsIn P setup s level F)
      (packGroupsIn P s level F) (packingValuesIn s level F) (setupClaimIn s level F) := by
  simp only [finishIn, shaped]

/-- The end check of `finishIn` at the recorded point `fused.point` with the recorded end claim
`fused.output` is the end check `fusedEndIn` of `fused.end`. -/
theorem finishIn_end (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) :
    (finishIn setup s level F (recordedIn s ⟨level, "fused.point", none⟩).toList).withClaim
      (firstIn s ⟨level, "fused.output", none⟩) = fusedEndIn setup s level F := by
  unfold finishIn fusedEndIn
  cases F.fused <;> rfl

theorem setupClaimIn_none (s : Replayer) (level : ℕ) (F : Fold) (direct : F.stage3 = none) :
    setupClaimIn s level F = none := by
  rw [setupClaimIn, direct, Option.map_none]

theorem setupClaimIn_some (s : Replayer) (level : ℕ) (F : Fold) {ν : ℕ}
    (offloaded : F.stage3 = some ν) :
    setupClaimIn s level F = some (firstIn s ⟨level, "stage3.setup_claim", none⟩) := by
  rw [setupClaimIn, offloaded, Option.map_some]

theorem productsIn_arities (s : Replayer) (level : ℕ) (arities : List ℕ) :
    (productsIn s level arities).map (·.arity) = arities := by
  unfold productsIn
  rw [List.map_map]
  exact (List.map_congr_left fun _ _ => rfl).trans (List.zipIdx_map_fst 0 arities)

theorem toList_firstIn (s : Replayer) (key : Key) (one : (recordedIn s key).size = 1) :
    (recordedIn s key).toList = [firstIn s key] := by
  unfold firstIn
  generalize recordedIn s key = a at one ⊢
  rcases a with ⟨_ | ⟨x, _ | ⟨y, l⟩⟩⟩ <;> simp_all

/-- The claim of Stage 2 on the coefficient route, for the runs `rel` of the relation. -/
theorem stage2In_plain (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
    (plain : F.euclidean = false) {rel : List RhsRun × ℕ} (rows : (F.rhs, F.openingRow) = rel) :
    (stage2In pub s level F).fused =
      (stage2 (recordedIn s ⟨level, "tau1", none⟩).toList (firstIn s ⟨level, "alpha", none⟩)
        (dataIn pub s level) rel.1 rel.2 (rowsIn s level F) (claimsIn s level F)
        (firstIn s ⟨level, "stage2_batch", none⟩ * firstIn s ⟨level, "range.image", none⟩ +
          Ext.zero)).fused := by
  subst rows
  rw [stage2In, plain]
  rfl

/-- The claim of Stage 2 on the Euclidean route with one virtual evaluation, for the runs `rel`
of the relation. -/
theorem stage2In_euclidean (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
    (euclidean : F.euclidean = true) {rel : List RhsRun × ℕ} (rows : (F.rhs, F.openingRow) = rel)
    (single : (recordedIn s ⟨level, "norm.virtual_evaluations", none⟩).size = 1) :
    (stage2In pub s level F).fused =
      (stage2 (recordedIn s ⟨level, "tau1", none⟩).toList (firstIn s ⟨level, "alpha", none⟩)
        (dataIn pub s level) rel.1 rel.2 (rowsIn s level F) (claimsIn s level F)
        (firstIn s ⟨level, "stage2_batch", none⟩ * firstIn s ⟨level, "range.image", none⟩ +
          firstIn s ⟨level, "norm.virtual_batch", none⟩ *
            batched 1 [firstIn s ⟨level, "norm.virtual_evaluations", none⟩]
              (firstIn s ⟨level, "norm.virtual_batch", none⟩))).fused := by
  subst rows
  rw [stage2In, euclidean, ite_eq_left rfl, toList_firstIn s _ single, single]

/-- The fold of the level `level` on the recorded values of `s` at a level without Stage 3: the
range proof with the norm stage on the Euclidean route, then the fused sum-check from the claim
of `stage2Of` with the end check of `fused.end`; `none` without the setup prefix. -/
def foldRunIn (pub : Public) (s : Replayer) (level : ℕ) (F : Fold) : Option (List Ext × Ext) :=
  (pub.setupFor F.setupCount).bind fun setup =>
    if F.euclidean then
      foldRunSq (leafDegree F.logBasis) F.limit F.low (wordsIn s level F) (normsIn s level)
        (stage2In pub s level F).fused (finishIn setup s level F)
    else
      foldRun (2 ^ F.logBasis) F.low (wordsIn s level F) (roundsIn s ⟨level, "range.leaf", none⟩)
        (stage2In pub s level F).fused (finishIn setup s level F)

/-- The offloaded fold of the level `level` on the recorded values of `s`: `offloadRun` with the
claim `σ` of `stage3.setup_claim` and the weight of the plan `plan` (`stage3In`). -/
def offloadRunIn (pub : Public) (s : Replayer) (level : ℕ) (F : Fold) (plan : SetupPlan) :
    Option ((List Ext × Ext) × (List Ext × Ext)) :=
  (pub.setupFor F.setupCount).bind fun setup =>
    offloadRun (2 ^ F.logBasis) F.low (wordsIn s level F) (roundsIn s ⟨level, "range.leaf", none⟩)
      (stage2In pub s level F).fused (finishIn setup s level F)
      (firstIn s ⟨level, "stage3.setup_claim", none⟩) (roundsIn s ⟨level, "stage3", none⟩)
      (firstIn s ⟨level, "stage3.prefix_evaluation", none⟩)
      (stage3Weight plan (firstIn s ⟨level, "alpha", none⟩)
        (recordedIn s ⟨level, "tau1", none⟩).toList)

/-- The last check of the range proof of the fold `F` at `level` (`Build.range`):
`range.leaf.end` on the coefficient route, `norm.end` on the Euclidean route. -/
def rangeCheckOf (level : ℕ) (F : Fold) : Check :=
  if F.euclidean then
    .rangeNorm level (leafDegree F.logBasis) F.limit ((productStages F.logBasis).map (·.1))
      (.turned ⟨level, "tau0", none⟩ F.low)
  else
    .range level (2 ^ F.logBasis) ((productStages F.logBasis).map (·.1))
      (.turned ⟨level, "tau0", none⟩ F.low)

theorem rangeCheckOf_plain (level : ℕ) (F : Fold) (plain : F.euclidean = false) :
    rangeCheckOf level F = .range level (2 ^ F.logBasis) ((productStages F.logBasis).map (·.1))
      (.turned ⟨level, "tau0", none⟩ F.low) := by
  rw [rangeCheckOf, plain, ite_eq_right Bool.false_ne_true]

theorem rangeCheckOf_euclidean (level : ℕ) (F : Fold) (euclidean : F.euclidean = true) :
    rangeCheckOf level F = .rangeNorm level (leafDegree F.logBasis) F.limit
      ((productStages F.logBasis).map (·.1)) (.turned ⟨level, "tau0", none⟩ F.low) := by
  rw [rangeCheckOf, euclidean, ite_eq_left rfl]

theorem stageValue_range (pub : Public) (s : Replayer) (level base : ℕ) (arities : List ℕ)
    (before : Before) :
    ((stageValue pub s (.range level base arities before)).map StageOut.accepts).getD false =
      (rangeRun base (productsIn s level arities) (roundsIn s ⟨level, "range.leaf", none⟩)
        (firstIn s ⟨level, "range.image", none⟩) (pointIn s before)).isSome :=
  rfl

theorem wordsIn_range (s : Replayer) (level b low : ℕ) (F : Fold) :
    rangeRun b (wordsIn s level F).stages (roundsIn s ⟨level, "range.leaf", none⟩)
        (wordsIn s level F).image ((wordsIn s level F).point low) =
      rangeRun b (productsIn s level ((productStages F.logBasis).map (·.1)))
        (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
        (pointIn s (.turned ⟨level, "tau0", none⟩ low)) :=
  rfl

/-- `fusedRun` at the recorded end of its rounds: the end check `finish r_2` with the end claim
`output`, and `(r_2, v_W)` when it holds. -/
theorem fusedRun_passing (claim : Ext) (rounds : List (List Ext × Ext))
    (finish : List Ext → FusedEnd) (r2 : List Ext) (output : Ext)
    (point : rounds.map (·.2) = r2) (ended : (plainRun claim rounds).claim = output) :
    fusedRun claim rounds finish =
      ((finish r2).withClaim output).passing.map fun f => (r2, f.value) := by
  rw [fusedRun_eq, point, ended]
  unfold FusedEnd.passing
  cases ((finish r2).withClaim output).holds <;> rfl

theorem foldRun_if (b low : ℕ) (W : FoldWords) (leaf : List (List Ext × Ext)) (claim : Ext)
    (finish : List Ext → FusedEnd) :
    foldRun b low W leaf claim finish =
      if (rangeRun b W.stages leaf W.image (W.point low)).isSome then
        fusedRun claim W.rounds finish
      else none := by
  unfold foldRun
  cases rangeRun b W.stages leaf W.image (W.point low) <;> rfl

theorem foldRunSq_if (leaf limit low : ℕ) (W : FoldWords) (N : NormWords) (claim : Ext)
    (finish : List Ext → FusedEnd) :
    foldRunSq leaf limit low W N claim finish =
      if (treeRun (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image) W.stages 1
          (fun _ => Ext.one) (W.point low) Ext.zero).isSome then
        fusedRun claim W.rounds finish
      else none := by
  unfold foldRunSq
  cases treeRun (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image) W.stages 1
    (fun _ => Ext.one) (W.point low) Ext.zero <;> rfl

theorem offloadRun_if (b low : ℕ) (W : FoldWords) (leaf : List (List Ext × Ext)) (claim : Ext)
    (finish : List Ext → FusedEnd) (σ : Ext) (setupRounds : List (List Ext × Ext))
    (prefixValue : Ext) (setupWeight : List Ext → List Ext → Ext) :
    offloadRun b low W leaf claim finish σ setupRounds prefixValue setupWeight =
      if (rangeRun b W.stages leaf W.image (W.point low)).isSome then
        (fusedRun claim W.rounds finish).bind fun c =>
          (stage3Run σ setupRounds prefixValue (setupWeight c.1)).map fun o => (c, o)
      else none := by
  unfold offloadRun
  cases rangeRun b W.stages leaf W.image (W.point low) <;> rfl

/-- `foldRunIn` is what the script computes at a level without Stage 3: `none` when the stage
function of the last check of the range proof or the end check of `fused.end` fails, and
otherwise the recorded point `fused.point` with `v_W`, when the rounds of `fused` end at the
recorded point and the recorded end claim. -/
theorem foldRunIn_eq (pub : Public) (s : Replayer) (level : ℕ) (F : Fold) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (point : (roundsIn s ⟨level, "fused", none⟩).map (·.2) =
      (recordedIn s ⟨level, "fused.point", none⟩).toList)
    (ended : (plainRun (stage2In pub s level F).fused (roundsIn s ⟨level, "fused", none⟩)).claim =
      firstIn s ⟨level, "fused.output", none⟩) :
    foldRunIn pub s level F =
      if ((stageValue pub s (rangeCheckOf level F)).map StageOut.accepts).getD false then
        (fusedEndIn setup s level F).passing.map fun f =>
          ((recordedIn s ⟨level, "fused.point", none⟩).toList, f.value)
      else none := by
  have fused := fusedRun_passing (stage2In pub s level F).fused (wordsIn s level F).rounds
    (finishIn setup s level F) _ _ point ended
  rw [finishIn_end] at fused
  cases euclidean : F.euclidean
  · rw [rangeCheckOf_plain level F euclidean]
    simp only [foldRunIn, expanded, Option.bind_some, euclidean, Bool.false_eq_true, ↓reduceIte]
    rw [foldRun_if, fused]
    rfl
  · rw [rangeCheckOf_euclidean level F euclidean]
    simp only [foldRunIn, expanded, Option.bind_some, euclidean, ↓reduceIte]
    rw [foldRunSq_if, fused]
    rfl

/-- `offloadRunIn` is what the script computes at an offloaded level: the last check of the range
proof, `fused.end` and `stage3.end` (`stage3In`), when the rounds of `fused` end at the recorded
point and end claim, and the absorbed claim `stage3.claim` of Stage 3 is the received
`stage3.setup_claim`. -/
theorem offloadRunIn_eq (pub : Public) (s : Replayer) (level : ℕ) (F : Fold) (plan : SetupPlan)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup)
    (plain : F.euclidean = false)
    (point : (roundsIn s ⟨level, "fused", none⟩).map (·.2) =
      (recordedIn s ⟨level, "fused.point", none⟩).toList)
    (ended : (plainRun (stage2In pub s level F).fused (roundsIn s ⟨level, "fused", none⟩)).claim =
      firstIn s ⟨level, "fused.output", none⟩)
    (claim : firstIn s ⟨level, "stage3.claim", none⟩ =
      firstIn s ⟨level, "stage3.setup_claim", none⟩) :
    offloadRunIn pub s level F plan =
      if ((stageValue pub s (rangeCheckOf level F)).map StageOut.accepts).getD false then
        ((fusedEndIn setup s level F).passing.map fun f =>
          ((recordedIn s ⟨level, "fused.point", none⟩).toList, f.value)).bind fun c =>
            (stage3In s level plan).map fun o => (c, o)
      else none := by
  have fused := fusedRun_passing (stage2In pub s level F).fused (wordsIn s level F).rounds
    (finishIn setup s level F) _ _ point ended
  rw [finishIn_end] at fused
  rw [rangeCheckOf_plain level F plain, stageValue_range]
  simp only [offloadRunIn, expanded, Option.bind_some]
  rw [offloadRun_if, fused, wordsIn_range]
  unfold stage3In
  rw [claim]
  cases (fusedEndIn setup s level F).passing
  · simp only [Option.map_none, Option.bind_none]
  · simp only [Option.map_some, Option.bind_some]

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E] in
/-- The path of the game of the fold of the level `level` for the recorded values of `s`:
`foldPathSq` on the Euclidean route, `foldPath` on the coefficient route. -/
noncomputable def foldPathIn (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
    (setup : Array Fq) (vars batch : ℕ) : List Move :=
  if F.euclidean then
    foldPathSq E vars batch F.low (leafDegree F.logBasis) F.limit F.compressed (wordsIn s level F)
      (normsIn s level) (stage2In pub s level F).fused (finishIn setup s level F)
  else
    foldPath E vars batch F.low (2 ^ F.logBasis) F.compressed (wordsIn s level F)
      (roundsIn s ⟨level, "range.leaf", none⟩) (stage2In pub s level F).fused
      (finishIn setup s level F)

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E] in
/-- The path of `offloadCheck` of the offloaded fold of the level `level` for the recorded values
of `s`. -/
noncomputable def offloadPathIn (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
    (setup : Array Fq) (vars batch : ℕ) : List Move :=
  offloadPath E vars batch F.low (2 ^ F.logBasis) F.compressed (wordsIn s level F)
    (roundsIn s ⟨level, "range.leaf", none⟩) (stage2In pub s level F).fused
    (finishIn setup s level F) (firstIn s ⟨level, "stage3.setup_claim", none⟩)
    (roundsIn s ⟨level, "stage3", none⟩) (firstIn s ⟨level, "stage3.prefix_evaluation", none⟩)

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- A trace level on the coefficient route (`traceFoldRun_runs`): `foldCheck` for
`traceReduction` of the statement of the recorded values, on the path of the recorded values,
ends in the result of `foldRunIn`. -/
theorem foldRunIn_trace_runs (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
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
    {μ : ℕ} (fits : T.opening.cells T.fold ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ batch) (w : Table)
    (plain : F.euclidean = false) (compressed : F.compressed = T.opening.compressed T.fold)
    (virtual : (recordedIn s ⟨level, "range.leaf.point", none⟩).toList =
      (roundsIn s ⟨level, "range.leaf", none⟩).map (·.2))
    (vars : (recordedIn s ⟨level, "tau0", none⟩).size = μ)
    (tau : (recordedIn s ⟨level, "tau1", none⟩).size = batch)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase)
    (stages : ∀ Q ∈ productsIn s level ((productStages F.logBasis).map (·.1)),
      Q.rounds.length = μ ∧ ∀ x ∈ Q.rounds, x.1.length = Q.arity)
    (count : (roundsIn s ⟨level, "range.leaf", none⟩).length = μ)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "range.leaf", none⟩,
      x.1.length = rangeLeaf T.shape.certifiedBase)
    (fusedCount : (roundsIn s ⟨level, "fused", none⟩).length = μ)
    (fusedLengths : ∀ x ∈ roundsIn s ⟨level, "fused", none⟩, x.1.length = 3) :
    (foldCheck (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge μ fits batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) w).Runs
      (foldPathIn E pub s level F setup μ batch)
      ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have key := traceFoldRun_runs (E := E) quarter two setup
    ((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)) (traceValuesIn s level F)
    (recordedIn s ⟨level, "tau0", none⟩).toList
    (productsIn s level ((productStages F.logBasis).map (·.1)))
    (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
    (roundsIn s ⟨level, "fused", none⟩) (dataIn pub s level) .witness (rowsIn s level F)
    (claimsIn s level F) challenge challenged fits rowsFit w plain compressed
    (by
      show (recordedIn s ⟨level, if F.euclidean then "norm.sumcheck.point" else "range.leaf.point",
        none⟩).toList = _
      rw [plain]
      exact virtual) vars tau
    ((productsIn_arities s level _).trans arities) stages count lengths fusedCount fusedLengths
  simp only [foldRunIn, foldPathIn, expanded, Option.bind_some, plain, Bool.false_eq_true,
    ↓reduceIte, finishIn_trace setup s level F shaped, stage2In_plain pub s level F plain rows,
    based, low, compressed]
  exact key

/-- A trace level on the Euclidean route (`traceFoldRunSq_runs`): `foldCheckSq` for
`traceReduction` of the statement of the recorded values, with the cap `F.limit` and the kernel
`Shape.normKernel`, on the path of the recorded values, ends in the result of `foldRunIn`. -/
theorem foldRunIn_traceSq_runs (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
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
    {μ : ℕ} (fits : T.opening.cells T.fold ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : T.opening.traceRowCount + 1 ≤ 2 ^ batch) (w : Table)
    (positive : 1 ≤ rangeLeaf T.shape.certifiedBase) (small : F.limit < fieldModulus)
    (digits : 0 < T.fold.responseDigits) (euclidean : F.euclidean = true)
    (compressed : F.compressed = T.opening.compressed T.fold)
    (single : (recordedIn s ⟨level, "norm.virtual_evaluations", none⟩).size = 1)
    (virtual : (recordedIn s ⟨level, "norm.sumcheck.point", none⟩).toList =
      (roundsIn s ⟨level, "norm.sumcheck", none⟩).map (·.2))
    (vars : (recordedIn s ⟨level, "tau0", none⟩).size = μ)
    (tau : (recordedIn s ⟨level, "tau1", none⟩).size = batch)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities T.shape.certifiedBase)
    (stages : ∀ Q ∈ productsIn s level ((productStages F.logBasis).map (·.1)),
      Q.rounds.length = μ ∧ ∀ x ∈ Q.rounds, x.1.length = Q.arity)
    (count : (roundsIn s ⟨level, "norm.sumcheck", none⟩).length = μ)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "norm.sumcheck", none⟩,
      x.1.length = rangeLeaf T.shape.certifiedBase + 1)
    (fusedCount : (roundsIn s ⟨level, "fused", none⟩).length = μ)
    (fusedLengths : ∀ x ∈ roundsIn s ⟨level, "fused", none⟩, x.1.length = 3) :
    (foldCheckSq (traceReduction (traceStatementFor (E := E) T.opening quarter setup
        (traceValuesIn s level F) (dataIn pub s level) .witness (rowsIn s level F)
        (claimsIn s level F)) challenge μ fits batch rowsFit) T.shape.certifiedBase
        (T.opening.binarySupport T.fold) (T.opening.compressed T.fold) F.limit
        (T.shape.normKernel T.fold) w).Runs
      (foldPathIn E pub s level F setup μ batch)
      ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have key := traceFoldRunSq_runs (E := E) quarter two setup
    ((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree)) (traceValuesIn s level F)
    F.limit (recordedIn s ⟨level, "tau0", none⟩).toList
    (productsIn s level ((productStages F.logBasis).map (·.1)))
    (firstIn s ⟨level, "range.image", none⟩)
    ((s.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0)
    (firstIn s ⟨level, "norm.merge", none⟩) (roundsIn s ⟨level, "norm.sumcheck", none⟩)
    (firstIn s ⟨level, "norm.virtual_evaluations", none⟩) (roundsIn s ⟨level, "fused", none⟩)
    (dataIn pub s level) .witness (rowsIn s level F) (claimsIn s level F) challenge challenged
    fits rowsFit w positive small digits euclidean compressed
    (by
      show (recordedIn s ⟨level, if F.euclidean then "norm.sumcheck.point" else "range.leaf.point",
        none⟩).toList = _
      rw [euclidean]
      exact virtual) vars tau
    ((productsIn_arities s level _).trans arities) stages count lengths fusedCount fusedLengths
  simp only [foldRunIn, foldPathIn, expanded, Option.bind_some, euclidean, ↓reduceIte,
    finishIn_trace setup s level F shaped, stage2In_euclidean pub s level F euclidean rows single,
    leaf, low, compressed]
  exact key

/-- A packing level with one group (`packFoldRun_runs`): `foldCheck` for `liftedReduction` of
the statement of the recorded values, on the path of the recorded values, ends in the result of
`foldRunIn`. `weighs` and `supported` are the weight and the binary-support weight of the end
check of `fused.end` at `r_2` (the weight theorem of the packing levels). -/
theorem foldRunIn_pack_runs (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
    {P : PackShape} (shaped : F.fused = .packing P) (direct : F.stage3 = none)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {S : Shape}
    {O : OpeningShape S} {fold : Descriptor}
    (rows : (F.rhs, F.openingRow) = relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain
      fun _ => .witness)
    (based : 2 ^ F.logBasis = S.certifiedBase) (low : F.low = O.low)
    (β : Module.Basis (Fin O.planes) Zq E) (challenge : Fin S.blocks → Rq O.carrierDegree)
    {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ) {digits : ℕ} (room : O.quotientRoom fold digits ≤ 2 ^ μ)
    {batch : ℕ} (rowsFit : O.rowCount + 1 ≤ 2 ^ batch) (w : Table) (plain : F.euclidean = false)
    (compressed : F.compressed = O.compressed fold)
    (virtual : (recordedIn s ⟨level, "range.leaf.point", none⟩).toList =
      (roundsIn s ⟨level, "range.leaf", none⟩).map (·.2))
    (vars : (recordedIn s ⟨level, "tau0", none⟩).size = μ)
    (tau : (recordedIn s ⟨level, "tau1", none⟩).size = batch)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities S.certifiedBase)
    (stages : ∀ Q ∈ productsIn s level ((productStages F.logBasis).map (·.1)),
      Q.rounds.length = μ ∧ ∀ x ∈ Q.rounds, x.1.length = Q.arity)
    (count : (roundsIn s ⟨level, "range.leaf", none⟩).length = μ)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "range.leaf", none⟩, x.1.length = rangeLeaf S.certifiedBase)
    (fusedCount : (roundsIn s ⟨level, "fused", none⟩).length = μ)
    (fusedLengths : ∀ x ∈ roundsIn s ⟨level, "fused", none⟩, x.1.length = 3)
    (weighs : ∀ p : List Ext, p.length = μ →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) none p).weight.into : E) =
      mle (p.map Ext.into) (((liftedReduction β (packStatementFor O setup (claimPointIn s level 0)
        (dataIn pub s level) .witness (rowsIn s level F) (claimsIn s level F)) challenge μ fits
          digits room batch rowsFit).rows (firstIn s ⟨level, "alpha", none⟩).into).batchWeight
            ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into)))
    (supported : ∀ p : List Ext, p.length = μ → O.compressed fold = true →
      ((foldSupport (Placement.of (packGroupsIn P s level F) P.path P.digits)
        (CellEq.half (recordedIn s ⟨level, "range.leaf.point", none⟩).toList).get
          (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
        if (O.gaps fold digits).support x then
          eqAt ((recordedIn s ⟨level, "range.leaf.point", none⟩).toList.map Ext.into) x
        else 0) :
    (foldCheck (liftedReduction β (packStatementFor O setup (claimPointIn s level 0)
        (dataIn pub s level) .witness (rowsIn s level F) (claimsIn s level F)) challenge μ fits
        digits room batch rowsFit) S.certifiedBase (O.gaps fold digits).support
        (O.compressed fold) w).Runs
      (foldPathIn E pub s level F setup μ batch)
      ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have key := packFoldRun_runs (E := E) β setup (claimPointIn s level 0)
    (packInputsIn P setup s level F) (packGroupsIn P s level F) (packingValuesIn s level F)
    (recordedIn s ⟨level, "tau0", none⟩).toList
    (productsIn s level ((productStages F.logBasis).map (·.1)))
    (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
    (roundsIn s ⟨level, "fused", none⟩) (dataIn pub s level) .witness (rowsIn s level F)
    (claimsIn s level F) challenge fits room rowsFit w compressed virtual vars tau
    ((productsIn_arities s level _).trans arities) stages count lengths fusedCount fusedLengths
    weighs supported
  simp only [foldRunIn, foldPathIn, expanded, Option.bind_some, plain, Bool.false_eq_true,
    ↓reduceIte, finishIn_packing setup s level F shaped, setupClaimIn_none s level F direct,
    stage2In_plain pub s level F plain rows, based, low, compressed]
  exact key

/-- A fold with opening groups without Stage 3 (`batchFoldRun_runs`): `foldCheck` for
`Batch.liftedReduction` of the statement of the recorded values, on the path of the recorded
values, ends in the result of `foldRunIn`. `weighs` and `supported` are the weight and the
binary-support weight of the end check of `fused.end` at `r_2`. -/
theorem foldRunIn_batch_runs (pub : Public) (s : Replayer) (level : ℕ) (F : Fold)
    {P : PackShape} (shaped : F.fused = .packing P) (direct : F.stage3 = none)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {n : ℕ}
    {B : Batch (Fin n)} (source : Fin n → RhsSource)
    (rows : (F.rhs, F.openingRow) =
      relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source)
    (base : ℕ) (based : 2 ^ F.logBasis = base) (low : F.low = B.low)
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (challenge : B.Challenge) {μ : ℕ}
    (fits : B.cells ≤ 2 ^ μ) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (w : Table) (plain : F.euclidean = false)
    (compressed : F.compressed = B.compressed)
    (virtual : (recordedIn s ⟨level, "range.leaf.point", none⟩).toList =
      (roundsIn s ⟨level, "range.leaf", none⟩).map (·.2))
    (vars : (recordedIn s ⟨level, "tau0", none⟩).size = μ)
    (tau : (recordedIn s ⟨level, "tau1", none⟩).size = batch)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities base)
    (stages : ∀ Q ∈ productsIn s level ((productStages F.logBasis).map (·.1)),
      Q.rounds.length = μ ∧ ∀ x ∈ Q.rounds, x.1.length = Q.arity)
    (count : (roundsIn s ⟨level, "range.leaf", none⟩).length = μ)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "range.leaf", none⟩, x.1.length = rangeLeaf base)
    (fusedCount : (roundsIn s ⟨level, "fused", none⟩).length = μ)
    (fusedLengths : ∀ x ∈ roundsIn s ⟨level, "fused", none⟩, x.1.length = 3)
    (weighs : ∀ p : List Ext, p.length = μ →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) none p).weight.into : E) =
      mle (p.map Ext.into) (((Batch.liftedReduction β (batchStatementFor B setup
        (claimPointIn s level) (dataIn pub s level) source (rowsIn s level F)
          (claimsIn s level F)) challenge μ fits base digits room batch rowsFit).rows
            (firstIn s ⟨level, "alpha", none⟩).into).batchWeight
              ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into)))
    (supported : ∀ p : List Ext, p.length = μ → B.compressed = true →
      ((foldSupport (Placement.of (packGroupsIn P s level F) P.path P.digits)
        (CellEq.half (recordedIn s ⟨level, "range.leaf.point", none⟩).toList).get
          (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
        if (B.gaps digits).support x then
          eqAt ((recordedIn s ⟨level, "range.leaf.point", none⟩).toList.map Ext.into) x
        else 0) :
    (foldCheck (Batch.liftedReduction β (batchStatementFor B setup (claimPointIn s level)
        (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)) challenge μ fits
        base digits room batch rowsFit) base (B.gaps digits).support B.compressed w).Runs
      (foldPathIn E pub s level F setup μ batch)
      ((foldRunIn pub s level F).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have key := batchFoldRun_runs (E := E) β setup (claimPointIn s level)
    (packInputsIn P setup s level F) (packGroupsIn P s level F) (packingValuesIn s level F)
    (recordedIn s ⟨level, "tau0", none⟩).toList
    (productsIn s level ((productStages F.logBasis).map (·.1)))
    (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
    (roundsIn s ⟨level, "fused", none⟩) (dataIn pub s level) source (rowsIn s level F)
    (claimsIn s level F) challenge fits base room rowsFit w compressed virtual vars tau
    ((productsIn_arities s level _).trans arities) stages count lengths fusedCount fusedLengths
    weighs supported
  simp only [foldRunIn, foldPathIn, expanded, Option.bind_some, plain, Bool.false_eq_true,
    ↓reduceIte, finishIn_packing setup s level F shaped, setupClaimIn_none s level F direct,
    stage2In_plain pub s level F plain rows, based, low, compressed]
  exact key

/-- The offloaded root of `key30.aks` (`batchOffloadRun_runs`): `offloadCheck` for
`Batch.liftedReduction` of the statement of the recorded values over the setup prefix `flat`, on
the path of the recorded values, ends in the result of `offloadRunIn`. `weighs` and `supported`
are the weight and the binary-support weight of the end check of `fused.end` at `r_2` with the
claim `σ` of `stage3.setup_claim`, and `setupWeighs` the weight of Stage 3 of the plan at `ρ`. -/
theorem offloadRunIn_runs (pub : Public) (s : Replayer) (level : ℕ) (F : Fold) (plan : SetupPlan)
    {P : PackShape} (shaped : F.fused = .packing P) {ν : ℕ} (offloaded : F.stage3 = some ν)
    {setup : Array Fq} (expanded : pub.setupFor F.setupCount = some setup) {n : ℕ}
    {B : Batch (Fin n)} (source : Fin n → RhsSource)
    (rows : (F.rhs, F.openingRow) =
      relation (fun k => B.shape (groupOrder n k)) B.openRank B.openDegree B.chain source)
    (base : ℕ) (based : 2 ^ F.logBasis = base) (low : F.low = B.low)
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (challenge : B.Challenge) {μ : ℕ}
    (fits : B.cells ≤ 2 ^ μ) {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (w : Table) (flat : Table)
    (scan : SetupScan (Batch.liftedReduction β ((batchStatementFor B setup (claimPointIn s level)
      (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)).over
        fun p : Fin (2 ^ ν) => flat p) challenge μ fits base digits room batch rowsFit) ν flat)
    (plain : F.euclidean = false) (compressed : F.compressed = B.compressed)
    (virtual : (recordedIn s ⟨level, "range.leaf.point", none⟩).toList =
      (roundsIn s ⟨level, "range.leaf", none⟩).map (·.2))
    (vars : (recordedIn s ⟨level, "tau0", none⟩).size = μ)
    (tau : (recordedIn s ⟨level, "tau1", none⟩).size = batch)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities base)
    (stages : ∀ Q ∈ productsIn s level ((productStages F.logBasis).map (·.1)),
      Q.rounds.length = μ ∧ ∀ x ∈ Q.rounds, x.1.length = Q.arity)
    (count : (roundsIn s ⟨level, "range.leaf", none⟩).length = μ)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "range.leaf", none⟩, x.1.length = rangeLeaf base)
    (fusedCount : (roundsIn s ⟨level, "fused", none⟩).length = μ)
    (fusedLengths : ∀ x ∈ roundsIn s ⟨level, "fused", none⟩, x.1.length = 3)
    (setupCount : (roundsIn s ⟨level, "stage3", none⟩).length = ν)
    (setupLengths : ∀ x ∈ roundsIn s ⟨level, "stage3", none⟩, x.1.length = 2)
    (weighs : ∀ p : List Ext, p.length = μ →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) (some (firstIn s ⟨level, "stage3.setup_claim", none⟩))
          p).weight.into : E) =
      scan.part (firstIn s ⟨level, "alpha", none⟩).into
          ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (p.map Ext.into) +
        scan.scale (firstIn s ⟨level, "alpha", none⟩).into
          ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (p.map Ext.into) *
          (firstIn s ⟨level, "stage3.setup_claim", none⟩).into)
    (supported : ∀ p : List Ext, p.length = μ → B.compressed = true →
      ((foldSupport (Placement.of (packGroupsIn P s level F) P.path P.digits)
        (CellEq.half (recordedIn s ⟨level, "range.leaf.point", none⟩).toList).get
          (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
        if (B.gaps digits).support x then
          eqAt ((recordedIn s ⟨level, "range.leaf.point", none⟩).toList.map Ext.into) x
        else 0)
    (setupWeighs : ∀ r2 ρ : List Ext, r2.length = μ → ρ.length = ν →
      ((stage3Weight plan (firstIn s ⟨level, "alpha", none⟩)
        (recordedIn s ⟨level, "tau1", none⟩).toList r2 ρ).into : E) =
      mle (ρ.map Ext.into) (scan.weight (firstIn s ⟨level, "alpha", none⟩).into
        ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (r2.map Ext.into))) :
    (offloadCheck (Batch.liftedReduction β ((batchStatementFor B setup (claimPointIn s level)
        (dataIn pub s level) source (rowsIn s level F) (claimsIn s level F)).over
          fun p : Fin (2 ^ ν) => flat p) challenge μ fits base digits room batch rowsFit) base
        (B.gaps digits).support B.compressed w ν flat scan).Runs
      (offloadPathIn E pub s level F setup μ batch)
      ((offloadRunIn pub s level F plan).map fun o =>
        ((o.1.1.map Ext.into, o.1.2.into), (o.2.1.map Ext.into, o.2.2.into))) := by
  have key := batchOffloadRun_runs (E := E) β setup (claimPointIn s level)
    (packInputsIn P setup s level F) (packGroupsIn P s level F) (packingValuesIn s level F)
    (recordedIn s ⟨level, "tau0", none⟩).toList
    (productsIn s level ((productStages F.logBasis).map (·.1)))
    (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
    (roundsIn s ⟨level, "fused", none⟩) (dataIn pub s level) source (rowsIn s level F)
    (claimsIn s level F) challenge fits base room rowsFit w ν flat scan
    (firstIn s ⟨level, "stage3.setup_claim", none⟩) (roundsIn s ⟨level, "stage3", none⟩)
    (firstIn s ⟨level, "stage3.prefix_evaluation", none⟩)
    (stage3Weight plan (firstIn s ⟨level, "alpha", none⟩)
      (recordedIn s ⟨level, "tau1", none⟩).toList) compressed virtual vars tau
    ((productsIn_arities s level _).trans arities) stages count lengths fusedCount fusedLengths
    setupCount setupLengths weighs supported setupWeighs
  simp only [offloadRunIn, offloadPathIn, expanded, Option.bind_some,
    finishIn_packing setup s level F shaped, setupClaimIn_some s level F offloaded,
    stage2In_plain pub s level F plain rows, based, low, compressed]
  exact key

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
