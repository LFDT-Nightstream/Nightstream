import NightstreamFPrime.Spec.Compression.Akita.Replay.Layout

/-! The transcript of the pinned Akita verifier as a list of operations, from the layout of a
schedule row and its grinding plan (Akita revision `d50f700`: `fold/root.rs`, `fold/suffix.rs`,
`fold/mod.rs`, `fold/terminal.rs`, `stages/`; the order is that of
`crates/ns-compress/tests/unit/akita_vectors/replay.rs`).

- `Site` is a grinding site of the code (`GrindingSite` of `akita-params`), and `Site.text` its
  text in the plan of a vector.
- `Op` is one operation of the transcript: the absorb of a public value, the receipt of a prover
  message, the proof of work of a site, the squeeze of field challenges, the response nonce of a
  fold, and the squeeze of the message of the fold challenges of one group.
- `Value` says how the verifier gets each absorbed public value: from the statement, from
  recorded messages and challenges, or by a computation of the code on them.
- `script` follows the layout and takes the bits of each site from the next run of the plan. It
  fails when a run has another site or another loss than the layout gives, when a run of the fold
  challenges does not have the count of the layout, or when runs are left (`next_entry`, `record_fold_challenges`, `finish` of
  `transcript_grinding/replay.rs`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The sum-check protocols of the code (`SumcheckProtocol`). -/
inductive Protocol
  | stage1
  | stage2
  | stage3
  | physicalL2
  | extensionOpeningReduction
deriving DecidableEq, Repr

def Protocol.text : Protocol → String
  | .stage1 => "Stage1"
  | .stage2 => "Stage2"
  | .stage3 => "Stage3"
  | .physicalL2 => "PhysicalL2"
  | .extensionOpeningReduction => "ExtensionOpeningReduction"

inductive Site
  | evaluationBatch (level : ℕ)
  | foldResponse (level : ℕ)
  | foldChallengeWork (level : ℕ)
  | foldChallengeGroup (level group : ℕ)
  | ringSwitchAlpha (level : ℕ)
  | tau0 (level : ℕ)
  | tau1 (level : ℕ)
  | round (protocol : Protocol) (level stage round : ℕ)
  | interstage (level stage : ℕ)
  | binary (level : ℕ)
  | stage2Batch (level : ℕ)
  | merge (level : ℕ)
  | virtualBatch (level : ℕ)
  | extensionPoint (level : ℕ)
  | claimBatch (level : ℕ)
deriving DecidableEq, Repr

/-- The text of a site in the `Debug` form of the code. -/
def Site.text : Site → String
  | .evaluationBatch l => s!"EvaluationBatch \{ level: {l} }"
  | .foldResponse l => s!"FoldResponse \{ level: {l} }"
  | .foldChallengeWork l => s!"FoldChallengeWork \{ level: {l} }"
  | .foldChallengeGroup l g => s!"FoldChallengeGroup \{ level: {l}, group: {g} }"
  | .ringSwitchAlpha l => s!"RingSwitchAlpha \{ level: {l} }"
  | .tau0 l => s!"Tau0Point \{ level: {l} }"
  | .tau1 l => s!"Tau1Point \{ level: {l} }"
  | .round p l s r =>
      s!"SumcheckRound \{ protocol: {p.text}, level: {l}, stage: {s}, round: {r} }"
  | .interstage l s => s!"Stage1InterstageBatch \{ level: {l}, stage: {s} }"
  | .binary l => s!"CompressionBinary \{ level: {l} }"
  | .stage2Batch l => s!"Stage2Batch \{ level: {l} }"
  | .merge l => s!"L2NormMerge \{ level: {l} }"
  | .virtualBatch l => s!"L2VirtualBatch \{ level: {l} }"
  | .extensionPoint l => s!"ExtensionOpeningPoint \{ level: {l} }"
  | .claimBatch l => s!"ExtensionOpeningClaimBatch \{ level: {l} }"

/-- The name of a transcript value: its level, its site in the event list and an index (the
round of a round message, the group of a root digest). -/
structure Key where
  level : ℕ
  site : String
  index : Option ℕ := none
deriving DecidableEq, Hashable, Repr, Inhabited

inductive Value
  /-- SHA3-256 of the canonical bytes of the commitment of a root group. -/
  | digest (group : ℕ)
  /-- The point of a root group. -/
  | point (group : ℕ)
  /-- The values of the root groups. -/
  | values
  /-- The encoded slot of the setup prefix. -/
  | slot
  /-- The concatenation of recorded messages and challenges. -/
  | recorded (keys : List Key)
  /-- The input claim of the first stage of the range proof. -/
  | zero
  /-- `Σ_i γ^i·c_i` of the child claims `c` of a stage and its batch challenge `γ`
  (`interstage_batch_weights`, `batch_claims`). -/
  | batched (children gamma : Key)
  /-- The input claim of the norm sum-check: the batched claim of the range proof, or zero, plus
  `merge · E_resp` (`stages/physical_l2_norm.rs`). -/
  | norm (batched : Option (Key × Key)) (merge energy : Key)
  /-- The input claim of the tensor reduction of one claim: `(1 − η)·row_0 + η·row_1` for the rows
  of the partials (`tensor_row_partials_from_columns`, `tensor_reduction_claim_from_rows`). -/
  | tensor (partials eta : Key)
  /-- The point of the next level: `ρ` with a zero coordinate at position `log_2 d − 1`
  (`ring_subfield_packed_extension_opening_point` for `k = 2`). -/
  | packed (degree : ℕ) (rho : Key)
  /-- The input claim of the fused sum-check of a fold: `β·image + relation + opening + η·virtual`
  (`stages/stage2.rs`, `Replay/Relation.lean`). -/
  | fused (F : Fold)
deriving Repr

inductive Op
  | absorb (key : Key) (value : Value)
  | fields (key : Key) (count : ℕ)
  | exts (key : Key) (count : ℕ)
  | integer (key : Key)
  | digest (key : Key)
  | bounded (key : Key) (max : ℕ)
  | work (site : Site) (level bits : ℕ)
  | challenge (key : Key) (count : ℕ)
  | foldNonce (level width : ℕ)
  | squeeze (level group bytes : ℕ)
deriving Repr

abbrev Build := StateT (ℕ × Array Op) (Except String)

namespace Build

def emit (op : Op) : Build Unit := modify fun (next, ops) => (next, ops.push op)

def take (plan : Array Run) (site : Site) (loss : ℕ) : Build Run := do
  let (next, ops) ← get
  let some run := plan[next]? | throw s!"the plan ends before {site.text}"
  unless run.site == site.text do throw s!"run {next} of the plan is {run.site}, not {site.text}"
  unless run.loss = loss do throw s!"{site.text} has the loss {run.loss}, the layout {loss}"
  set (next + 1, ops)
  return run

/-- The proof of work of a site: a nonce of `bits + 7` bits when `bits > 0` (`pow_bits`). -/
def work (plan : Array Run) (level : ℕ) (site : Site) (loss : ℕ) : Build Unit := do
  let run ← take plan site loss
  unless run.nonceBits = (if run.bits = 0 then 0 else run.bits + 7) do
    throw s!"{site.text} has {run.bits} bits and a nonce of {run.nonceBits} bits"
  emit (.work site level run.bits)

/-- `grinded_ext_challenges`: the work of the site, then `count` challenges. The loss of the site
is `loss`. -/
def challenges (plan : Array Run) (level : ℕ) (site : Site) (site' : String) (count loss : ℕ) :
    Build Unit := do
  work plan level site loss
  emit (.challenge ⟨level, site', none⟩ count)

/-- One sum-check (`akita-sumcheck/src/proof_stream.rs`): the input claim is absorbed, and each
round has its message of `degree` elements, the work of its site and its challenge. The round
challenges are the point `<name>.point`. The loss of a round is the degree of its polynomial:
`degree + 1` for the equality-factored rounds of Stage 1, whose message omits a factor, and
`degree` for the others. -/
def sumcheck (plan : Array Run) (level : ℕ) (name : String) (protocol : Protocol) (stage : ℕ)
    (claim : Value) (rounds degree : ℕ) : Build Unit := do
  emit (.absorb ⟨level, name ++ ".claim", none⟩ claim)
  for r in [0:rounds] do
    emit (.exts ⟨level, name ++ ".round", some r⟩ degree)
    challenges plan level (.round protocol level stage r) (name ++ ".point") 1
      (if protocol = .stage1 then degree + 1 else degree)

/-- The response nonce of a fold, its work, and the messages of the fold challenges of its
groups (`read_fold_response`, `VerifierFoldDraw`, `record_fold_challenges`). -/
def foldChallenges (plan : Array Run) (level : ℕ) (groups : List FoldGroup) : Build Unit := do
  let run ← take plan (.foldResponse level) 0
  emit (.foldNonce level run.nonceBits)
  work plan level (.foldChallengeWork level) (groups.map (·.coordinates)).sum
  for (G, g) in groups.zipIdx do
    emit (.squeeze level g G.messageBytes)
    let run ← take plan (.foldChallengeGroup level g) 0
    unless run.count = G.coordinates + 1 do
      throw s!"the plan has {run.count} entries for the {G.coordinates} challenges of group {g}"

/-- The tensor reduction of a level with one claim (`stages/opening_claims/extension_claim.rs`),
after the absorb of the openings. -/
def tensor (plan : Array Run) (level rounds : ℕ) (openings : List Key) : Build Unit := do
  let key (site : String) : Key := ⟨level, site, none⟩
  emit (.absorb (key "tensor.openings") (.recorded openings))
  emit (.exts (key "tensor.partials") 2)
  challenges plan level (.extensionPoint level) "tensor.eta" 1 1
  sumcheck plan level "tensor.sumcheck" .extensionOpeningReduction 0
    (.tensor (key "tensor.partials") (key "tensor.eta")) rounds 2
  emit (.exts (key "tensor.final_claims") 1)

/-- The stages of the range proof and the norm check of a fold (`stages/stage1.rs`). -/
def range (plan : Array Run) (level : ℕ) (F : Fold) : Build Unit := do
  let key (site : String) : Key := ⟨level, site, none⟩
  let stages := productStages F.logBasis
  let batched (k : ℕ) : Option (Key × Key) :=
    if k = 0 then none else some (key s!"range.{k - 1}.children", key s!"range.{k - 1}.batch")
  let claim (k : ℕ) : Value := match batched k with
    | some (children, gamma) => .batched children gamma
    | none => .zero
  for ((arity, children), k) in stages.zipIdx do
    sumcheck plan level s!"range.{k}" .stage1 k (claim k) F.vars arity
    emit (.exts (key s!"range.{k}.children") children)
    challenges plan level (.interstage level k) s!"range.{k}.batch" 1 (children - 1)
  let last := stages.length
  if F.euclidean then
    emit (.integer (key "norm.response_l2_sq"))
    challenges plan level (.merge level) "norm.merge" 1 1
    sumcheck plan level "norm.sumcheck" .physicalL2 0
      (.norm (batched last) (key "norm.merge") (key "norm.response_l2_sq")) F.vars
      (leafDegree F.logBasis + 1)
    emit (.exts (key "norm.virtual_evaluations") 1)
    emit (.exts (key "range.image") 1)
    challenges plan level (.virtualBatch level) "norm.virtual_batch" 1 1
  else
    sumcheck plan level "range.leaf" .stage1 last (claim last) F.vars (leafDegree F.logBasis)
    emit (.exts (key "range.image") 1)

/-- One fold (`fold/mod.rs`, `verify_fold`), after the absorbs of its points and values. -/
def fold (plan : Array Run) (level : ℕ) (F : Fold) : Build Unit := do
  let key (site : String) : Key := ⟨level, site, none⟩
  emit (.fields (key "opening_payload") F.payload)
  if 1 < F.groups.length then
    challenges plan level (.evaluationBatch level) "row_coefficients" F.groups.length
      F.groups.length
  foldChallenges plan level F.groups
  if F.last then
    emit (.fields (key "terminal_t") F.witness)
    emit (.digest (key "terminal_t_last_block_digest"))
  else
    emit (.fields (key "next_witness") F.witness)
  challenges plan level (.ringSwitchAlpha level) "alpha" 1
    (2 * (F.groups.map (·.ring)).foldl max 0 - 1)
  challenges plan level (.tau0 level) "tau0" F.vars F.vars
  challenges plan level (.tau1 level) "tau1" F.batch F.batch
  range plan level F
  if F.compressed then challenges plan level (.binary level) "binary_batch" 1 1
  challenges plan level (.stage2Batch level) "stage2_batch" 1 1
  sumcheck plan level "fused" .stage2 0 (.fused F) F.vars 3
  emit (.exts (key "witness_evaluation") 1)
  if let some rounds := F.stage3 then
    emit (.absorb (key "stage3.slot") .slot)
    emit (.exts (key "stage3.setup_claim") 1)
    sumcheck plan level "stage3" .stage3 0 (.recorded [key "stage3.setup_claim"]) rounds 2
    emit (.exts (key "stage3.prefix_evaluation") 1)

/-- The absorbs before the fold of a level: the statement at the root (`fold/root.rs`), the
points and the values of the level before at a level with packing (`fold/suffix.rs`), and the
tensor reduction with the point of the level at a level with evaluation trace. -/
def prefix_ (plan : Array Run) (level : ℕ) (F : Fold) (before : Option Fold) : Build Unit := do
  let key (site : String) : Key := ⟨level, site, none⟩
  let back (site : String) : Key := ⟨level - 1, site, none⟩
  let setup := (before.bind (·.stage3)).isSome
  match before, F.trace with
  | none, _ =>
      for g in [0:F.groups.length] do
        emit (.absorb ⟨0, "root.commitment_digest", some g⟩ (.digest g))
      for g in [0:F.groups.length] do
        emit (.absorb (key s!"root.point.{g}") (.point g))
      emit (.absorb (key "openings") .values)
  | some _, none =>
      let points := (if setup then [back "stage3.point"] else []) ++ [back "fused.point"]
      for (point, g) in points.zipIdx do
        emit (.absorb (key s!"point.{g}") (.recorded [point]))
      emit (.absorb (key "openings") (.recorded
        ((if setup then [back "stage3.prefix_evaluation"] else []) ++
          [back "witness_evaluation"])))
  | some _, some (rounds, degree) =>
      unless F.groups.length = 1 ∧ ¬setup do throw s!"level {level}: a trace level of one claim"
      tensor plan level rounds [back "witness_evaluation"]
      emit (.absorb (key "point.0") (.packed degree (key "tensor.sumcheck.point")))

/-- The terminal (`fold/terminal.rs`, `verify_terminal_suffix`). -/
def terminal (plan : Array Run) (level : ℕ) (T : Terminal) : Build Unit := do
  let key (site : String) : Key := ⟨level, site, none⟩
  tensor plan level T.rounds [⟨level - 1, "witness_evaluation", none⟩]
  emit (.absorb (key "terminal.point") (.packed T.degree (key "tensor.sumcheck.point")))
  emit (.fields (key "terminal.e") T.e)
  emit (.digest (key "terminal.e_last_block_digest"))
  foldChallenges plan level [T.group]
  emit (.bounded (key "terminal.z") T.zMax)

end Build

def script (layout : Layout) (plan : Array Run) : Except String (Array Op) := do
  let run : Build Unit := do
    let befores := none :: layout.folds.map some
    for ((F, before), level) in (layout.folds.zip befores).zipIdx do
      Build.prefix_ plan level F before
      Build.fold plan level F
    Build.terminal plan layout.folds.length layout.terminal
    let (next, _) ← get
    unless next = plan.size do throw s!"{plan.size - next} runs of the plan are left"
  let ((), (_, ops)) ← run.run (0, #[])
  return ops

end NightstreamFPrime.Spec.Compression.Akita.Replay
