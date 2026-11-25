import Std.Data.HashMap
import NightstreamFPrime.Spec.Compression.Akita.Replay.Wire
import NightstreamFPrime.Spec.Compression.Akita.Replay.Keccak
import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation
import NightstreamFPrime.Spec.Compression.Akita.Replay.Tensor
import NightstreamFPrime.Spec.Compression.Akita.Replay.Sampler
import NightstreamFPrime.Spec.Compression.Akita.Replay.Terminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPacking
import NightstreamFPrime.Spec.Compression.Akita.Replay.Stage3

/-! The Fiat–Shamir transcript of the pinned Akita verifier on the decoded messages of a proof
(Akita revision `d50f700`, jolt-transcript `25d87ba`, spongefish 0.8.0).

- The channel (`new_verifier_channel`, `bind_instance`): the duplex sponge absorbs the session
  identifier, SHAKE128 of `irtf-cfrg-fiat-shamir/session-id`, 136 zero bytes and the tag
  `akita-pcs/standalone-session/v8` (`derive_session_id`); then `BytesMsg(session)`; then
  `PROTOCOL_TAG ‖ BytesMsg(descriptor)`. `BytesMsg(b)` is the `u64` length, `b` and zeros to a
  word (`jolt-transcript/src/codec.rs`).
- A public value and a prover message are absorbed as their bytes with zeros to a word
  (`ProofChannel::public`, `receive_atoms`). The payload of `z` is its length as one `u64` word,
  then its bytes (`receive_bounded_bytes`).
- A challenge vector of `len` elements of `E` is one squeeze of `poolBytes (2·len)` bytes: 8-byte
  little-endian candidates, a candidate at or above `q` rejected, the first `2·len` accepted
  candidates the coordinates in order (`jolt-transcript/src/challenge.rs`, `pool_bytes`,
  `decode_pool`).
- The work of a site with `g > 0` bits (`VerifierTranscript::check_grind`): the nonce is canonical
  and below `2^(g + 7)`; it is absorbed as one `u64` word; 8 bytes are squeezed and their
  little-endian word has `g` low bits zero. The response nonce of a fold is canonical, below
  `2^width` and absorbed as one word (`read_fold_response`, `value_fits`). The message of the fold
  challenges of a group is squeezed, and `foldChallenges` (`Replay/Sampler.lean`) gives its
  challenges; `replay` records them in `Derived.drawn`, and it stops with `noWork` at the squeeze
  when the draw gives none (`VerifierFoldDraw`: the operator-norm pool runs out).
- `Value` of an absorb: `replay` computes it from the statement, the recorded messages and
  challenges, and the computations of `Value`. The rows of the relation of a root group are the
  last coefficients of its commitment bytes. The fused claim of a fold is `stage2`
  (`Replay/Relation.lean`); `replay` records its target, relation claim and opening claim.
- After the rounds of a sum-check, `replay` runs `sumcheckRun` (`Replay/SumCheck.lean`) on its
  absorbed claim, its round messages and its challenges, and records the end as `<name>.output`.
- At the last check of the range proof, of the norm check, of the tensor reduction, of the fused
  sum-check (`traceFused` or `foldFused`) and of Stage 3 of a level, `replay` runs the stage
  function on the recorded values (`stageValue`), records its result in `Derived.stages` and
  rejects when it is `none`. `Replayer.step_stage` states this for the step,
  and `step_range_runs`, `step_norm_runs` and `step_tensor_runs` state that the result is the
  result of `rangeCheck`, of `rangeTree` with `normLeafStage`, and of `tensorCheck` of the
  specification on the path of the recorded values, for recorded values of the shapes of the
  script (`rangeRun_runs`, `treeRun_runs` with `normRun_runs`, `tensorRun_runs`).
- At the terminal, `replay` runs `terminalRun` (`Replay/Terminal.lean`) on the recorded messages
  of the terminal, the point, `θ` (`tensorFactor`) and the final claim of its tensor reduction,
  the drawn fold challenges of the terminal group and the setup prefix of the seed
  (`Public.setupFor`); it records the values, and it rejects at the site of the first check of
  the code that fails.

The model of Akita is a random oracle; the transcript has no specification in Lean. The evidence
is the equality with the events of the vectors in `tests/AkitaVectors.lean`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

def tag (name : String) (length : ℕ) : ByteArray :=
  name.toUTF8 ++ ByteArray.mk (Array.replicate (length - name.utf8ByteSize) 0)

def zeros (count : ℕ) : ByteArray := ByteArray.mk (Array.replicate count 0)

def wordPadded (bytes : ByteArray) : ByteArray := bytes ++ zeros ((8 - bytes.size % 8) % 8)

def le64 (value : ℕ) : ByteArray :=
  ByteArray.mk ((Array.range 8).map fun i => (value / 2 ^ (8 * i) % 256).toUInt8)

/-- `BytesMsg(bytes)`. -/
def bytesMsg (bytes : ByteArray) : ByteArray := le64 bytes.size ++ wordPadded bytes

def extBytes (values : Array Ext) : ByteArray :=
  values.foldl (fun out e => out ++ le64 e.c0.word.toNat ++ le64 e.c1.word.toNat) ByteArray.empty

def sessionId : ByteArray :=
  Keccak.shake128 (tag "irtf-cfrg-fiat-shamir/session-id" 32 ++ zeros 136 ++
    tag "akita-pcs/standalone-session/v8" 32) 32

def protocolTag : ByteArray := tag "akita-pcs/transcript/v8/sha3" 32

/-- The digest of the canonical bytes of a root commitment (`CommitmentDigest`). -/
def commitmentDigest (commitment : ByteArray) : ByteArray :=
  Keccak.sha3_256 (tag "akita-pcs/commitment-digest/v8" 32 ++ commitment)

def channel (session descriptor : ByteArray) (start : Duplex := Duplex.empty) : Duplex :=
  ((start.absorb sessionId).absorb (bytesMsg session)).absorb (protocolTag ++ bytesMsg descriptor)

/-- `pool_bytes(len)` for `q = 2^64 − 59`: `R = 64 − ⌈log_2 59⌉ = 58`, the least `k` with
`(k + 1)·(R − ⌈log_2 (len + k)⌉) ≥ 161`, and 8 bytes for each of the `len + k` candidates. -/
def poolBytes (len : ℕ) : ℕ :=
  if len = 0 then 0 else
    let fits (k : ℕ) : Bool :=
      let L := Nat.size (len + k - 1)
      L < 58 && 161 ≤ (k + 1) * (58 - L)
    8 * (len + ((List.range 161).find? fits).getD 161)

def decodePool (bytes : ByteArray) (len : ℕ) : Option (Array Ext) := Id.run do
  let mut accepted : Array Fq := #[]
  for i in [0:bytes.size / 8] do
    if accepted.size = 2 * len then break
    if let some a := Fq.ofWord? (Wire.word bytes (8 * i)) then accepted := accepted.push a
  if accepted.size < 2 * len then return none
  return some ((Array.range len).map fun i => ⟨accepted[2 * i]!, accepted[2 * i + 1]!⟩)

def worked (check : ByteArray) (bits : ℕ) : Bool := (Wire.word check 0).toNat % 2 ^ bits = 0

/-- `ρ` with a zero coordinate at position `log_2 degree − 1`. -/
def packed (degree : ℕ) (rho : Array Ext) : Array Ext :=
  let at_ := Nat.log 2 degree - 1
  (rho.extract 0 at_).push Ext.zero ++ rho.extract at_ rho.size

/-- The coefficients of `count` elements at the end of the canonical bytes of a commitment. -/
def commitmentRows (commitment : ByteArray) (count : ℕ) : Array Fq :=
  (Array.range count).map fun j => ⟨Wire.word commitment (commitment.size - 8 * count + 8 * j)⟩

inductive Stop
  | noWork (op : ℕ) (site : String)
  | reject (op : ℕ) (site : String)
  | mismatch (op : ℕ)
deriving DecidableEq, Repr

/-- The result of a stage function on the recorded values at the last check of a stage group
(`Check.range`, `Check.rangeNorm`, `Check.tensor`). -/
inductive StageOut
  | range (out : Option (List Ext × Ext))
  | norm (out : Option (List Ext × Ext × Ext))
  | tensor (out : Option Ended)
  | fused (out : Option FusedEnd)
  | stage3 (out : Option (List Ext × Ext))
deriving Repr

def StageOut.accepts : StageOut → Bool
  | .range out => out.isSome
  | .norm out => out.isSome
  | .tensor out => out.isSome
  | .fused out => out.isSome
  | .stage3 out => out.isSome

structure Derived where
  challenges : Array (Key × Array Ext) := #[]
  works : Array (Site × ℕ × ℕ × Bool) := #[]
  absorbs : Array (Key × ByteArray) := #[]
  values : Array (Key × Ext) := #[]
  ends : Array (Key × Before × Ended) := #[]
  checks : Array Key := #[]
  rounds : Array (Key × List (List Ext × Ext)) := #[]
  recorded : Std.HashMap Key (Array Ext) := {}
  integers : Std.HashMap Key ℕ := {}
  fields : Std.HashMap Key (Array Fq) := {}
  squeezed : ℕ := 0
  foldMessages : Array (ℕ × ℕ × ByteArray) := #[]
  stages : Array (Key × StageOut) := #[]
  drawn : Array (ℕ × ℕ × List Sparse) := #[]
  bytes : Std.HashMap Key ByteArray := {}
  terminal : Option TerminalValues := none

/-- The payload of `z` of the terminal: `z_coords`, the Rice parameter, no `ℓ_∞` cap (the
Euclidean route) and `z_payload_bytes`. -/
def Terminal.params (T : Terminal) : Golomb.Params := ⟨T.zCoords, T.rice, none, T.zMax⟩

structure Replayer where
  sponge : Duplex
  next : ℕ
  env : Std.HashMap Key (Array Ext)
  fields : Std.HashMap Key (Array Fq)
  integers : Std.HashMap Key ℕ
  derived : Derived

abbrev ReplayM := ExceptT Stop (StateM Replayer)

namespace Replayer

/-- The recorded values of `key`. -/
def recordedIn (s : Replayer) (key : Key) : Array Ext := (s.env.get? key).getD #[]

def firstIn (s : Replayer) (key : Key) : Ext := ((recordedIn s key)[0]?).getD Ext.zero

/-- The rounds of the sum-check `key`: its messages and its challenges. -/
def roundsIn (s : Replayer) (key : Key) : List (List Ext × Ext) :=
  ((s.derived.rounds.find? (·.1 == key)).map (·.2)).getD []

def pointIn (s : Replayer) : Before → List Ext
  | .plain => []
  | .point k => (recordedIn s k).toList
  | .turned k low => (recordedIn s k).toList.rotate ((recordedIn s k).size - low)

/-- The product stages `range.<k>` of the level `level` with the arities `arities`. -/
def productsIn (s : Replayer) (level : ℕ) (arities : List ℕ) : List Product :=
  arities.zipIdx.map fun (arity, k) =>
    { arity, rounds := roundsIn s ⟨level, s!"range.{k}", none⟩
      children := (recordedIn s ⟨level, s!"range.{k}.children", none⟩).toList
      gamma := firstIn s ⟨level, s!"range.{k}.batch", none⟩ }

/-- The drawn fold challenges of a group. -/
def drawnIn (s : Replayer) (level group : ℕ) : List Sparse :=
  ((s.derived.drawn.find? fun x => x.1 == level && x.2.1 == group).map (·.2.2)).getD []

/-- The setup prefix of `count` elements for the replay: the first `count` elements of
`Public.setup` when it is long enough, and otherwise the prefix of the seed (`Setup.expand`). -/
def _root_.NightstreamFPrime.Spec.Compression.Akita.Replay.Public.setupFor (pub : Public)
    (count : ℕ) : Option (Array Fq) :=
  if count ≤ pub.setup.size then some (pub.setup.extract 0 count) else Setup.expand pub.seed count

/-- The values of the end of the fused sum-check of the trace level `level`: the point of the
level before without its first coordinate, the point and `η` of the tensor reduction, `τ_1`,
`r_2`, the point of the last sum-check of Stage 1 (`stage1_point`), `α`, the end claim, `v_W` and
the coefficients `γ`, `ζ_bin` and `η_norm`. -/
def traceValuesIn (s : Replayer) (level : ℕ) (F : Fold) : TraceValues where
  euclidean := F.euclidean
  compressed := F.compressed
  tail := (recordedIn s ⟨level - 1, "fused.point", none⟩).toList.tail
  point := (recordedIn s ⟨level, "tensor.sumcheck.point", none⟩).toList
  eta := firstIn s ⟨level, "tensor.eta", none⟩
  tau1 := (recordedIn s ⟨level, "tau1", none⟩).toList
  r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
  virtual := (recordedIn s ⟨level, if F.euclidean then "norm.sumcheck.point"
    else "range.leaf.point", none⟩).toList
  alpha := firstIn s ⟨level, "alpha", none⟩
  output := firstIn s ⟨level, "fused.output", none⟩
  value := firstIn s ⟨level, "witness_evaluation", none⟩
  batch := firstIn s ⟨level, "stage2_batch", none⟩
  binary := firstIn s ⟨level, "binary_batch", none⟩
  norm := firstIn s ⟨level, "norm.virtual_batch", none⟩

/-- The end check of the fused sum-check of a trace level with the shapes `T`, on the recorded
values, the drawn fold challenges of the level in `R_{q,d_A}` and the setup prefix. -/
def fusedIn (T : TraceShape) (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) :
    FusedEnd :=
  traceFused T.opening T.fold setup ((drawnIn s level 0).map (Poly.ofSparse T.shape.innerDegree))
    (traceValuesIn s level F)

/-- The point of the claim `g` of the level `level` as the replay absorbs it: `root.point.g` at
the root (`fold/root.rs`), `point.g` after it (`fold/suffix.rs`). -/
def claimPointIn (s : Replayer) (level g : ℕ) : List Ext :=
  (recordedIn s ⟨level, if level = 0 then s!"root.point.{g}" else s!"point.{g}", none⟩).toList

/-- The groups of a packing level in the order of `P`, each with its drawn fold challenges in its
subring `R_{q,d_car}` and in `R_{q,d_A}`, its point and its claim coefficient
(`row_coefficients`; one claim: `1`). -/
def packGroupsIn (P : PackShape) (s : Replayer) (level : ℕ) (F : Fold) : List PackGroup :=
  let rows := if 1 < F.groups.length then recordedIn s ⟨level, "row_coefficients", none⟩
    else #[Ext.one]
  P.slots.map fun g =>
    let cs := drawnIn s level g.index
    let stride := g.shape.innerDegree / g.opening.carrierDegree
    ⟨g.shape, g.opening, g.fold,
      cs.map fun c => Poly.ofSparse g.opening.carrierDegree (c.map fun x => (x.1 / stride, x.2)),
      cs.map (Poly.ofSparse g.shape.innerDegree), claimPointIn s level g.index,
      rows.getD g.index Ext.zero⟩

/-- The inputs of the weight of a packing level that do not depend on a group: the setup prefix,
`τ_1`, the quotient digits, the coefficient block `2^low`, the base `2^logBasis` and the path of
`D`, with the cells of the code. -/
def packInputsIn (P : PackShape) (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) :
    FoldInputs :=
  ⟨setup, (recordedIn s ⟨level, "tau1", none⟩).toList, P.digits, F.low, 2 ^ F.logBasis, P.path,
    .code, {}⟩

/-- The values of the end of the fused sum-check of a packing level: `τ_1`, `r_2`, the point of
the leaf stage of the range proof, `α`, the end claim, `v_W` and the coefficients `γ` and
`ζ_bin`. -/
def packingValuesIn (s : Replayer) (level : ℕ) (F : Fold) : PackingValues where
  compressed := F.compressed
  tau1 := (recordedIn s ⟨level, "tau1", none⟩).toList
  r2 := (recordedIn s ⟨level, "fused.point", none⟩).toList
  virtual := (recordedIn s ⟨level, "range.leaf.point", none⟩).toList
  alpha := firstIn s ⟨level, "alpha", none⟩
  output := firstIn s ⟨level, "fused.output", none⟩
  value := firstIn s ⟨level, "witness_evaluation", none⟩
  batch := firstIn s ⟨level, "stage2_batch", none⟩
  binary := firstIn s ⟨level, "binary_batch", none⟩

/-- The claim `σ` of Stage 3 at a level whose setup is offloaded to Stage 3. -/
def setupClaimIn (s : Replayer) (level : ℕ) (F : Fold) : Option Ext :=
  F.stage3.map fun _ => firstIn s ⟨level, "stage3.setup_claim", none⟩

/-- The end check of the fused sum-check of a packing level with the shapes `P` on the recorded
values, the drawn fold challenges and the setup prefix (`foldFused`). -/
def packingIn (P : PackShape) (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) :
    FusedEnd :=
  foldFused (packInputsIn P setup s level F) (packGroupsIn P s level F) (packingValuesIn s level F)
    (setupClaimIn s level F)

/-- The count of setup elements that the end check of the fused sum-check of `F` reads. -/
def _root_.NightstreamFPrime.Spec.Compression.Akita.Replay.Fold.setupCount (F : Fold) : ℕ :=
  match F.fused with
  | .trace T => traceSetupCount T.opening
  | .packing P => P.setupCount F.stage3.isSome

/-- The end check of the fused sum-check of the level `level` on the recorded values: `fusedIn`
at a level with evaluation trace, `packingIn` at a level with coefficient packing. -/
def fusedEndIn (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) : FusedEnd :=
  match F.fused with
  | .trace T => fusedIn T setup s level F
  | .packing P => packingIn P setup s level F

/-- The end of Stage 3 of the level `level` on the recorded values (`stage3.rs:62`). -/
def stage3In (s : Replayer) (level : ℕ) (plan : SetupPlan) : Option (List Ext × Ext) :=
  stage3Run (firstIn s ⟨level, "stage3.claim", none⟩) (roundsIn s ⟨level, "stage3", none⟩)
    (firstIn s ⟨level, "stage3.prefix_evaluation", none⟩)
    (stage3Weight plan (firstIn s ⟨level, "alpha", none⟩) (recordedIn s ⟨level, "tau1", none⟩).toList
      (recordedIn s ⟨level, "fused.point", none⟩).toList)

/-- The stage function of a last check of a stage group on the recorded values of `s`:
`rangeRun`, `treeRun` with `normRun`, `tensorRun`, `traceFused` or `foldFused` with the setup
prefix of `pub`, or `stage3Run`. -/
def stageValue (pub : Public) (s : Replayer) : Check → Option StageOut
  | .range level base arities before => some (.range (rangeRun base (productsIn s level arities)
      (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
      (pointIn s before)))
  | .rangeNorm level leaf limit arities before => some (.norm (treeRun
      (normRun leaf limit ((s.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0)
        (firstIn s ⟨level, "norm.merge", none⟩) (roundsIn s ⟨level, "norm.sumcheck", none⟩)
        (firstIn s ⟨level, "norm.virtual_evaluations", none⟩)
        (firstIn s ⟨level, "range.image", none⟩))
      (productsIn s level arities) 1 (fun _ => Ext.one) (pointIn s before) Ext.zero))
  | .tensor level count point opening => some (.tensor (tensorRun count
      (recordedIn s point).toList (firstIn s opening)
      ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
      ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
      (firstIn s ⟨level, "tensor.eta", none⟩) (roundsIn s ⟨level, "tensor.sumcheck", none⟩)
      (firstIn s ⟨level, "tensor.final_claims", none⟩)))
  | .fused level F => (pub.setupFor F.setupCount).map fun setup =>
      .fused (fusedEndIn setup s level F).passing
  | .stage3 level plan => some (.stage3 (stage3In s level plan))
  | _ => none

def bytesIn (s : Replayer) (key : Key) : ByteArray := (s.derived.bytes.get? key).getD .empty

/-- The messages of the terminal of the level `level` (`fold/terminal.rs`): the sent blocks of
`e`, its digest, the sent blocks of `t` and its digest from the level before, and the payload of
`z`. -/
def terminalMessagesIn (s : Replayer) (level : ℕ) : TerminalMessages :=
  ⟨(s.fields.get? ⟨level, "terminal.e", none⟩).getD #[],
    bytesIn s ⟨level, "terminal.e_last_block_digest", none⟩,
    (s.fields.get? ⟨level - 1, "terminal_t", none⟩).getD #[],
    bytesIn s ⟨level - 1, "terminal_t_last_block_digest", none⟩,
    bytesIn s ⟨level, "terminal.z", none⟩⟩

/-- The values of the terminal of the level `level`: the point `ρ` of its tensor reduction, the
factor `θ = tensorFactor` with the point of the level before without its first coordinate, the
final claim, and the drawn fold challenges of the terminal group (`final_relation`). -/
def terminalChallengesIn (s : Replayer) (level : ℕ) : TerminalChallenges :=
  let ρ := (recordedIn s ⟨level, "tensor.sumcheck.point", none⟩).toList
  ⟨ρ, tensorFactor (recordedIn s ⟨level - 1, "fused.point", none⟩).toList.tail
      (firstIn s ⟨level, "tensor.eta", none⟩) ρ,
    firstIn s ⟨level, "tensor.final_claims", none⟩, drawnIn s level 0⟩

/-- `terminalRun` of the terminal of the layout on the recorded values of `s`, with the setup
prefix of the seed (`Public.setupFor`); `none` when `Setup.expand` gives no prefix. -/
def terminalValue (pub : Public) (s : Replayer) (level : ℕ) :
    Option (Except String TerminalValues) :=
  let T := pub.layout.terminal
  (pub.setupFor T.setupCount).map fun setup =>
    terminalRun T.shape T.limit T.params setup (terminalMessagesIn s level)
      (terminalChallengesIn s level)

/-- The checks of the terminal of the code that the vectors name: the norm, the digest of the
last block of `e` and the trace row. -/
def terminalSites (level : ℕ) : Array Key :=
  #[⟨level, "terminal.z.norm", none⟩, ⟨level, "terminal.e_last_block.digest", none⟩,
    ⟨level, "terminal.trace", none⟩]

def recorded (key : Key) : ReplayM (Array Ext) := do
  return recordedIn (← get) key

def record (key : Key) (values : Array Ext) : ReplayM Unit :=
  modify fun s => { s with env := s.env.insert key (((s.env.get? key).getD #[]) ++ values) }

def absorb (bytes : ByteArray) : ReplayM Unit :=
  modify fun s => { s with sponge := s.sponge.absorb bytes }

def squeeze (count : ℕ) : ReplayM ByteArray := do
  let s ← get
  let (out, sponge) := s.sponge.squeeze count
  set { s with sponge, derived := { s.derived with squeezed := s.derived.squeezed + count } }
  return out

def message (i : ℕ) (messages : Array Message) : ReplayM Message := do
  let s ← get
  let some m := messages[s.next]? | throw (.mismatch i)
  set { s with next := s.next + 1 }
  return m

def fieldsOf (key : Key) : ReplayM (Array Fq) := do
  return ((← get).fields.get? key).getD #[]

def first (key : Key) : ReplayM Ext := do
  return firstIn (← get) key

/-- The claims of Stage 2 of the fold `F` at `level` (`fold/mod.rs`, `verify_fold`): the row
coefficients (one claim: `1`), the claims (the openings, or the final claims of the tensor
reduction at a level with evaluation trace), the public rows of the relation, and
`β·image + η·virtual` with `η·virtual` only on the Euclidean route
(`batch_l2_virtual_evaluations`). -/
def stage2Of (pub : Public) (level : ℕ) (F : Fold) : ReplayM Stage2 := do
  let at_ (site : String) : Key := ⟨level, site, none⟩
  let witness ← fieldsOf ⟨level - 1, "next_witness", none⟩
  let payload ← fieldsOf (at_ "opening_payload")
  let data (run : RhsRun) : Array Fq := match run.source with
    | .root g => commitmentRows ((pub.groups[g]?.map (·.commitment)).getD .empty)
        (run.degree * run.count)
    | .setup => pub.setupRows
    | .witness => witness
    | .payload => payload
  let rows ← if 1 < F.groups.length then recorded (at_ "row_coefficients") else pure #[Ext.one]
  let claims ← recorded (at_ (if F.trace.isSome then "tensor.final_claims" else "openings"))
  let virtual ← if F.euclidean then do
      let eta ← first (at_ "norm.virtual_batch")
      let virtuals ← recorded (at_ "norm.virtual_evaluations")
      pure (eta * batched virtuals.size virtuals.toList eta)
    else pure Ext.zero
  let rest := (← first (at_ "stage2_batch")) * (← first (at_ "range.image")) + virtual
  return stage2 (← recorded (at_ "tau1")).toList (← first (at_ "alpha")) data F.rhs F.openingRow
    rows claims rest

def value (pub : Public) (key : Key) : Value → ReplayM ByteArray
  | .digest g => return commitmentDigest ((pub.groups[g]?.map (·.commitment)).getD .empty)
  | .point g => return extBytes ((pub.groups[g]?.map (·.point)).getD #[])
  | .values => return extBytes (pub.groups.flatMap (·.value))
  | .slot => return pub.slot
  | .recorded keys => do
      return extBytes (← keys.foldlM (fun out k => return out ++ (← recorded k)) #[])
  | .zero => return extBytes #[Ext.zero]
  | .batched children gamma => do
      let c ← recorded children
      return extBytes #[batched c.size c.toList (← first gamma)]
  | .norm before merge energy => do
      let claim ← match before with
        | some (children, gamma) => do
            let c ← recorded children
            pure (batched c.size c.toList (← first gamma))
        | none => pure Ext.zero
      return extBytes #[claim + (← first merge) * (← first energy)]
  | .tensor partials eta => do
      let p ← recorded partials
      return extBytes #[tensorInput (p.getD 0 Ext.zero) (p.getD 1 Ext.zero) (← first eta)]
  | .packed degree rho => return extBytes (packed degree (← recorded rho))
  | .fused F => do
      let c ← stage2Of pub key.level F
      let at_ (site : String) : Key := ⟨key.level, site, none⟩
      modify fun s => { s with derived := { s.derived with values := s.derived.values ++
        #[(at_ "opening_target", c.target), (at_ "relation_claim", c.relation),
          (at_ "opening_claim", c.opening)] } }
      return extBytes #[c.fused]

def pointBefore (before : Before) : ReplayM (List Ext) := do
  return pointIn (← get) before

def weightsOf : Weights → ReplayM (ℕ → Ext)
  | .ones => pure fun _ => Ext.one
  | .powers gamma => return powExt (← first gamma)

/-- Whether an equation of the verifier holds on the recorded values. -/
def holds (pub : Public) : Check → ReplayM Bool
  | .tensorOpening point value partials => do
      let p ← recorded partials
      let head := (← recorded point).getD 0 Ext.zero
      return (← first value).same
        ((Ext.one - head) * p.getD 0 Ext.zero + head * p.getD 1 Ext.zero)
  | .stage nodes arity weights output children => do
      return (← first output).same
        (stageEnd nodes arity (← weightsOf weights) (← recorded children).toList)
  | .energy energy limit => return energyFits (((← get).integers.get? energy).getD 0) limit
  | check => do return ((stageValue pub (← get) check).map StageOut.accepts).getD false

/-- An equation of the verifier at the operation `i` for the result `staged` of its stage
function: the result is recorded and decides the check; with no stage function, `holds`. -/
def checkWith (pub : Public) (i : ℕ) (key : Key) (check : Check) : Option StageOut → ReplayM Unit
  | some out => do
      modify fun s => { s with derived := { s.derived with
        stages := s.derived.stages.push (key, out) } }
      unless out.accepts do throw (.reject i key.site)
      modify fun s => { s with derived := { s.derived with checks := s.derived.checks.push key } }
  | none => do
      unless ← holds pub check do throw (.reject i key.site)
      modify fun s => { s with derived := { s.derived with checks := s.derived.checks.push key } }

def checkStep (pub : Public) (i : ℕ) (key : Key) (check : Check) : ReplayM Unit := do
  checkWith pub i key check (stageValue pub (← get) check)

/-- The draw of the fold challenges of a group from its squeezed message at the operation `i`:
the message and the challenges are recorded, or the replay stops with `noWork` at the squeeze. -/
def draw (i level group : ℕ) (G : FoldGroup) (bytes : ByteArray) : ReplayM Unit := do
  modify fun s => { s with derived := { s.derived with
    foldMessages := s.derived.foldMessages.push (level, group, bytes) } }
  let some cs := foldChallenges G bytes
    | throw (.noWork i (Site.foldChallengeGroup level group).text)
  modify fun s => { s with derived := { s.derived with
    drawn := s.derived.drawn.push (level, group, cs) } }

/-- The terminal at the operation `i` for the result of `terminalValue`: the values and the
checks are recorded, or the replay rejects at the site of the check that fails. -/
def terminalWith (i level : ℕ) : Option (Except String TerminalValues) → ReplayM Unit
  | none => throw (.mismatch i)
  | some (.error site) => throw (.reject i site)
  | some (.ok out) => modify fun s => { s with derived := { s.derived with
      terminal := some out, checks := s.derived.checks ++ terminalSites level } }

def step (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ) :
    Op → ReplayM Unit
  | .absorb key v => do
      let bytes ← value pub key v
      absorb (wordPadded bytes)
      match v with
      | .digest _ | .slot => pure ()
      | _ => record key (← (decodeExts bytes))
      modify fun s => { s with derived := { s.derived with
        absorbs := s.derived.absorbs.push (key, bytes) } }
  | .fields key _ => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
      match m.data with
      | .fields values => modify fun s => { s with fields := s.fields.insert key values }
      | _ => throw (.mismatch i)
  | .digest key => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
      let .bytes value := m.data | throw (.mismatch i)
      modify fun s => { s with derived := { s.derived with
        bytes := s.derived.bytes.insert key value } }
  | .exts key _ => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
      match m.data with
      | .exts values => record key values
      | _ => throw (.mismatch i)
  | .integer key => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
      match m.data with
      | .integer value => do
          record key #[Ext.ofBase (Fq.ofNat value)]
          modify fun s => { s with integers := s.integers.insert key value }
      | _ => throw (.mismatch i)
  | .bounded key _ => do
      let m ← message i messages
      absorb (le64 (m.size - 4) ++ wordPadded (proof.extract (m.offset + 4) (m.offset + m.size)))
      let .bytes value := m.data | throw (.mismatch i)
      modify fun s => { s with derived := { s.derived with
        bytes := s.derived.bytes.insert key value } }
  | .work site _ bits => do
      if bits = 0 then return
      let m ← message i messages
      let .nonce nonce canonical := m.data | throw (.mismatch i)
      unless canonical ∧ nonce < 2 ^ (bits + 7) do throw (.noWork i site.text)
      absorb (le64 nonce)
      let passed := worked (← squeeze 8) bits
      modify fun s => { s with derived := { s.derived with
        works := s.derived.works.push (site, m.offset, nonce, passed) } }
      unless passed do throw (.noWork i site.text)
  | .challenge key count => do
      let some values := decodePool (← squeeze (poolBytes (2 * count))) count
        | throw (.noWork i key.site)
      record key values
      modify fun s => { s with derived := { s.derived with
        challenges := s.derived.challenges.push (key, values) } }
  | .foldNonce level width => do
      let m ← message i messages
      let .nonce nonce canonical := m.data | throw (.mismatch i)
      unless canonical ∧ (32 ≤ width ∨ nonce < 2 ^ width) do
        throw (.noWork i s!"FoldResponse \{ level: {level} }")
      absorb (le64 nonce)
  | .squeeze level group G => do draw i level group G (← squeeze G.messageBytes)
  | .ended key rounds before => do
      let at_ (suffix : String) (index : Option ℕ) : Key := ⟨key.level, key.site ++ suffix, index⟩
      let messages ← (List.range rounds).mapM fun r => recorded (at_ ".round" (some r))
      let form ← match before with
        | .plain => pure Form.plain
        | _ => do pure (Form.equality (← pointBefore before))
      let rounds := (messages.map (·.toList)).zip (← recorded (at_ ".point" none)).toList
      let some ended := sumcheckRun form (← first (at_ ".claim" none)) rounds
        | throw (.mismatch i)
      record (at_ ".output" none) #[ended.claim]
      modify fun s => { s with derived := { s.derived with
        ends := s.derived.ends.push (key, before, ended)
        rounds := s.derived.rounds.push (key, rounds) } }
  | .check key check => checkStep pub i key check
  | .terminal level => do terminalWith i level (terminalValue pub (← get) level)
where
  decodeExts (bytes : ByteArray) : ReplayM (Array Ext) :=
    return (Array.range (bytes.size / 16)).map fun j =>
      ⟨⟨Wire.word bytes (16 * j)⟩, ⟨Wire.word bytes (16 * j + 8)⟩⟩

end Replayer

/-- The replay of a decoded proof: the operations of the script from the sponge of the channel,
and what it derives. -/
def replay (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message) :
    Option Stop × Derived :=
  let run : ReplayM Unit := ops.zipIdx.forM fun (op, i) => Replayer.step pub proof messages i op
  let (result, s) := (run.run).run ⟨channel pub.session pub.descriptor pub.sponge, 0, {}, {}, {}, {}⟩
  (match result with | .ok () => none | .error stop => some stop,
    { s.derived with recorded := s.env, integers := s.integers, fields := s.fields })

/-- Each nonce of the messages is in range and each work predicate holds: the replay does not stop
at a work. -/
def works (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message) : Bool :=
  match (replay pub ops proof messages).1 with
  | some (.noWork _ _) => false
  | _ => true

/-- The verdict of a stop of the replay: `noWork`, `reject` or `mismatch` at the stop, and with no
stop the verdict `checks` of the equations that the replay does not have yet. -/
def verdictOf (stop : Option Stop) (checks : Array Message → Verdict) (messages : Array Message) :
    Verdict :=
  match stop with
  | some (.noWork op site) => .noWork op site
  | some (.reject op site) => .reject op site
  | some (.mismatch op) => .mismatch op
  | none => checks messages

/-- The verdict on the decoded messages: `noWork` at the first work of the replay that fails,
`reject` at the first equation of the replay that fails before each failed work, and otherwise
the verdict `checks` of the equations that the replay does not have yet. -/
def verdict (pub : Public) (ops : Array Op) (proof : ByteArray) (checks : Array Message → Verdict)
    (messages : Array Message) : Verdict :=
  verdictOf (replay pub ops proof messages).1 checks messages

/-- `verifyBytes_noWork`: when the
equations never give `noWork`, a `noWork` outcome comes from a proof that decodes and whose
messages do not have their work. Bytes after the last message do not change it: the code checks
the end of the proof after its last check. -/
theorem verifyBytes_noWork (pub : Public) (ops : Array Op) (proof : ByteArray)
    (checks : Array Message → Verdict) (equations : ∀ m op site, checks m ≠ .noWork op site)
    {op : ℕ} {site : String}
    (outcome : verifyBytes (verdict pub ops proof checks) ops proof = .verdict (.noWork op site)) :
    ∃ messages rest, decode ops proof = .ok (messages, rest) ∧
      works pub ops proof messages = false := by
  unfold verifyBytes at outcome
  rcases decoded : decode ops proof with failure | ⟨messages, rest⟩
  · simp [decoded] at outcome
  · refine ⟨messages, rest, rfl, ?_⟩
    simp only [decoded] at outcome
    have stopped : verdict pub ops proof checks messages = .noWork op site := by
      revert outcome
      cases verdict pub ops proof checks messages <;> by_cases empty : rest = 0 <;> simp [empty]
    unfold verdict verdictOf at stopped
    unfold works
    split at stopped
    · rename_i found
      rw [found]
    · cases stopped
    · cases stopped
    · exact absurd stopped (equations messages op site)

section Join

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

namespace Replayer

/-- The state after the stage function of the check `key` gave `out`. -/
def staged (s : Replayer) (key : Key) (out : StageOut) : Replayer :=
  { s with derived := { s.derived with stages := s.derived.stages.push (key, out) } }

/-- The state after the check `key` held. -/
def checked (s : Replayer) (key : Key) : Replayer :=
  { s with derived := { s.derived with checks := s.derived.checks.push key } }

theorem step_check (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (check : Check) :
    step pub proof messages i (.check key check) = checkStep pub i key check :=
  rfl

theorem bind_get_run (f : Replayer → ReplayM Unit) (s : Replayer) :
    ((get >>= f : ReplayM Unit).run).run s = ((f s).run).run s :=
  rfl

theorem checkStep_run (pub : Public) (i : ℕ) (key : Key) (check : Check) (s : Replayer) :
    ((checkStep pub i key check).run).run s =
      ((checkWith pub i key check (stageValue pub s check)).run).run s :=
  bind_get_run (fun x => checkWith pub i key check (stageValue pub x check)) s

theorem checkWith_run (pub : Public) (i : ℕ) (key : Key) (check : Check) (s : Replayer)
    (out : StageOut) :
    ((checkWith pub i key check (some out)).run).run s =
      if out.accepts then (.ok (), checked (staged s key out) key)
      else (.error (.reject i key.site), staged s key out) := by
  cases accepts : out.accepts
  · simp only [checkWith, accepts]
    rfl
  · simp only [checkWith, accepts]
    rfl

/-- The step of a check with a stage function records its result and rejects exactly when the
result is `none`. -/
theorem step_stage (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (check : Check) (s : Replayer) {out : StageOut}
    (value : stageValue pub s check = some out) :
    ((step pub proof messages i (.check key check)).run).run s =
      if out.accepts then (.ok (), checked (staged s key out) key)
      else (.error (.reject i key.site), staged s key out) := by
  rw [step_check, checkStep_run, value, checkWith_run]

end Replayer

open Replayer

section Range

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] [Algebra Zq K] {φ : Ext → K}
  (hom : ExtHom φ) (same : ∀ a b : Ext, a.same b = true ↔ φ a = φ b)
include hom same

/-- The range proof of a level on the coefficient route: the replay records `rangeRun` of the
recorded values and rejects when it is `none`, and `rangeCheck` of the specification ends in its
result on the path of the recorded values. -/
theorem step_range_runs (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (level base : ℕ) (arities : List ℕ) (before : Before) (s : Replayer) (μ : ℕ)
    (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (degrees : (productsIn s level arities).map (·.arity) = rangeArities base)
    (long : (pointIn s before).length = μ)
    (shaped : ∀ S ∈ productsIn s level arities,
      S.rounds.length = μ ∧ ∀ x ∈ S.rounds, x.1.length = S.arity)
    (count : (roundsIn s ⟨level, "range.leaf", none⟩).length = μ)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "range.leaf", none⟩, x.1.length = rangeLeaf base) :
    let out := rangeRun base (productsIn s level arities) (roundsIn s ⟨level, "range.leaf", none⟩)
      (firstIn s ⟨level, "range.image", none⟩) (pointIn s before)
    ((step pub proof messages i (.check key (.range level base arities before))).run).run s =
        (if out.isSome then (.ok (), checked (staged s key (.range out)) key)
        else (.error (.reject i key.site), staged s key (.range out))) ∧
      (rangeCheck base μ support w ((pointIn s before).map φ)).Runs
        (treePath φ (fun _ _ _ _ => leafPath φ (rangeLeaf base)
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩))
          (productsIn s level arities) 1 (fun _ => Ext.one) (pointIn s before) Ext.zero)
        (out.map fun o => (o.1.map φ, φ o.2)) := by
  intro out
  exact ⟨step_stage (out := .range out) pub proof messages i key _ s rfl,
    rangeRun_runs hom same base μ support w _ _ _ _ degrees long shaped count lengths⟩

variable (nat : ∀ n : ℕ, φ (Ext.ofBase (Fq.ofNat n)) = (n : K))
include nat

/-- The range proof of a level with the norm check on the Euclidean route: the replay records
`treeRun` with `normRun` of the recorded values and rejects when it is `none`, and `rangeTree`
with `normLeafStage` of the specification ends in its result on the path of the recorded
values. -/
theorem step_norm_runs (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (level leaf limit : ℕ) (arities : List ℕ) (before : Before) (s : Replayer)
    (μ : ℕ) (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (positive : 1 ≤ leaf) (small : limit < fieldModulus) (long : (pointIn s before).length = μ)
    (shaped : ∀ S ∈ productsIn s level arities,
      S.rounds.length = μ ∧ ∀ x ∈ S.rounds, x.1.length = S.arity)
    (count : (roundsIn s ⟨level, "norm.sumcheck", none⟩).length = μ)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "norm.sumcheck", none⟩, x.1.length = leaf + 1) :
    let energy := (s.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0
    let merge := firstIn s ⟨level, "norm.merge", none⟩
    let rounds := roundsIn s ⟨level, "norm.sumcheck", none⟩
    let virtual := firstIn s ⟨level, "norm.virtual_evaluations", none⟩
    let image := firstIn s ⟨level, "range.image", none⟩
    let out := treeRun (normRun leaf limit energy merge rounds virtual image)
      (productsIn s level arities) 1 (fun _ => Ext.one) (pointIn s before) Ext.zero
    ((step pub proof messages i (.check key (.rangeNorm level leaf limit arities before))).run).run
          s =
        (if out.isSome then (.ok (), checked (staged s key (.norm out)) key)
        else (.error (.reject i key.site), staged s key (.norm out))) ∧
      (rangeTree leaf μ w (fun nodes weights point claim =>
          normLeafStage leaf μ limit kernel support w nodes weights point claim)
        ((productsIn s level arities).map (·.arity)) 1 (fun _ => φ Ext.one)
        ((pointIn s before).map φ) (φ Ext.zero)).Runs
        (treePath φ (fun _ _ _ claim => normPath φ leaf limit energy merge rounds virtual image claim)
          (productsIn s level arities) 1 (fun _ => Ext.one) (pointIn s before) Ext.zero)
        (out.map fun o => (o.1.map φ, φ o.2.1, φ o.2.2)) := by
  intro energy merge rounds virtual image out
  exact ⟨step_stage (out := .norm out) pub proof messages i key _ s rfl,
    treeRun_runs hom same leaf μ w (fun o : List Ext × Ext × Ext => (o.1.map φ, φ o.2.1, φ o.2.2))
      _ _ _
      (fun nodes weights point claim long' => normRun_runs hom same nat leaf μ limit kernel support
        w _ _ _ _ _ nodes weights point claim positive count long' lengths small)
      _ 1 (fun _ => Ext.one) _ Ext.zero long shaped⟩

end Range

section Tensor

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The tensor reduction of a level: the replay records `tensorRun` of the recorded values and
rejects when it is `none`, and `tensorCheck` of the specification ends in its result on the path
of the recorded values. -/
theorem step_tensor_runs (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (level count : ℕ) (point opening : Key) (s : Replayer) (w : Table)
    (rounds : (roundsIn s ⟨level, "tensor.sumcheck", none⟩).length = count)
    (lengths : ∀ x ∈ roundsIn s ⟨level, "tensor.sumcheck", none⟩, x.1.length = 2) :
    let p0 := (recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero
    let p1 := (recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero
    let η := firstIn s ⟨level, "tensor.eta", none⟩
    let final := firstIn s ⟨level, "tensor.final_claims", none⟩
    let out := tensorRun count (recordedIn s point).toList (firstIn s opening) p0 p1 η
      (roundsIn s ⟨level, "tensor.sumcheck", none⟩) final
    ((step pub proof messages i (.check key (.tensor level count point opening))).run).run s =
        (if out.isSome then (.ok (), checked (staged s key (.tensor out)) key)
        else (.error (.reject i key.site), staged s key (.tensor out))) ∧
      (tensorCheck count w ((recordedIn s point).toList.map (Ext.into (E := E)))
        (firstIn s opening).into).Runs
        (tensorPath E count (recordedIn s point).toList (firstIn s opening) p0 p1 η
          (roundsIn s ⟨level, "tensor.sumcheck", none⟩) final)
        (out.map fun o => (o.point.map Ext.into,
          mle (o.point.map Ext.into) (tensorWeight ((recordedIn s point).toList.tail.map Ext.into)
            η.into), o.claim.into)) := by
  intro p0 p1 η final out
  exact ⟨step_stage (out := .tensor out) pub proof messages i key _ s rfl,
    tensorRun_runs count w _ _ _ _ _ _ _ rounds lengths⟩

end Tensor

end Join

end NightstreamFPrime.Spec.Compression.Akita.Replay
