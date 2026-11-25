import Lean.Data.Json
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules
import NightstreamFPrime.Spec.Compression.Akita.Replay.Wire
import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Replay.Plan
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalSetup
import NightstreamFPrime.Spec.Compression.Akita.Replay.Grows
import NightstreamFPrime.Spec.Compression.Akita.Replay.RunProof
import NightstreamFPrime.Spec.Compression.Akita.Replay.Statement

/-! The Lean replay of the vectors of real Akita openings in `tests/vectors/akita/`
(`crates/ns-compress/tests/unit/akita_vectors.rs`, Akita revision `d50f700`;
`docs/akita-work/match-design.md` section 7, items 1 to 4).

- Inputs. `Vector` has the public inputs of a positive file (`schedule`, `shape`, `session`,
  `descriptor`, `groups`, `setup_field_elements`, `grinding_plan`) and the `proof` bytes. The
  `events` of the file are in `Oracles` and are oracles for the comparison only.
- Gate 1. The count of the events of each `op` is the field `counts` of the file. The root groups
  have the variables of the root shapes of `Key20.lean` and `Key30.lean`, their commitment rows
  the count of the outer commitment of their shape, and the commitment bytes end with the rows.
- Gate 2. Products of `Ext` on values of the files (`gate2`) and `#guard`s.
- Gate 3. `decode` with the script of the layout of the Lean numbers reads each prover message of
  the proof with no byte left, equal to each `message`, `work` and `fold_nonce` event; the changes
  of the negative files with the stages `parse` and `trailing_bytes` are `malformed` and
  `trailing`.
- Gate 7. Each equation of the tensor reduction, the range proof and the norm check of the replay
  holds in the order of the `check` events of the file. The replay records the results of the
  stage functions at the last check of each group (`Replay/Transcript.lean`), and each accepts with the
  values of the replay; the factor `θ` of each tensor reduction is the value of the file.
- Gate 4. FIPS 202 vectors; the root digests; the work bits of the plan; the replay of the
  transcript from the proof and the public inputs only (`gate4`): each challenge, work, absorb and
  the values `opening_target`, `relation_claim` and `opening_claim` of each fold; the changes of the
  negative files with nothing from the events, each a reject with the outcome class of the Rust
  verifier at the failed site of the file, or a reject at an end check before it (`moved`).
- Gate 8. Each `fold_challenges` event equals the challenges that the replay draws
  (`foldChallenges` on the squeezed message).
- Gate 10. The three `check` events of the terminal are checks of the replay that pass, and the
  decoded `z`, its norm, the last block of `e` and the opening of the replay equal the events.
- The end checks. The replay evaluates `fused.end` at each level (`traceFused` at a level with
  evaluation trace, `foldFused` at a level with coefficient packing; `Replay/FusedJoin.lean`) and
  `stage3.end` at the root of `key30.aks` (`stage3Run`); the vectors have no event for them.
- The verdict. On each positive file `verifyBytes` gives `.verdict .accept` with no check left to
  the verdict `checks`: each check of the replay passes, the end checks and each check of the
  terminal included. A change of `witness_evaluation` at a level rejects at `fused.end` of that
  level when no work comes between them, and a change of `stage3.prefix_evaluation` at
  `stage3.end`: the code checks both before the next transcript operation
  (`validate_stage2_replay` and `verify_setup_sumcheck` in `verify_fold`), and the Rust test of the
  vectors cannot evaluate them, so the field `failed` of these changes is the next failure of its
  replay. The test lists each change whose Lean site differs from `failed` (`moved`).
- The statement. `publicOf` takes the shape of the statement from `root20` or `root30` of
  `Replay/Statement.lean`, and `Public.shaped` holds on each positive file (`shapedGate`): the
  profile headers of the Lean shapes are the headers of the commitments of the files. For each
  change of `key20-statement-negative.json`, `verifyProof` accepts exactly when the Rust verifier
  accepts (`statementNegative`).
- `Replayed` is a positive file with the one replay of its proof; `tests/AkitaAllVectors.lean`
  runs the replays and the gates of this file and of the five other vector tests on them. -/

namespace AkitaVectors

open Lean (Json)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Replay

def field (value : Json) (path : List String) : Except String Json :=
  path.foldlM (·.getObjVal? ·) value

def nat (value : Json) (path : List String) : Except String ℕ := do (← field value path).getNat?

def str (value : Json) (path : List String) : Except String String := do
  (← field value path).getStr?

def array (value : Json) (path : List String) : Except String (Array Json) := do
  (← field value path).getArr?

def hexDigit (c : UInt8) : Except String UInt8 :=
  if 48 ≤ c ∧ c ≤ 57 then pure (c - 48)
  else if 97 ≤ c ∧ c ≤ 102 then pure (c - 87)
  else throw s!"{c} is not a hex digit"

def hex (text : String) : Except String ByteArray := do
  let raw := text.toUTF8
  unless raw.size % 2 = 0 do throw "a hex string of odd length"
  let mut out := ByteArray.emptyWithCapacity (raw.size / 2)
  for i in [0:raw.size / 2] do
    out := out.push ((← hexDigit raw[2 * i]!) * 16 + (← hexDigit raw[2 * i + 1]!))
  return out

/-- A field element of the file: a decimal string below `q`. -/
def element (value : Json) : Except String Fq := do
  let n ← match (← value.getStr?).toNat? with
    | some n => pure n
    | none => throw s!"{value} is not a decimal string"
  unless n < NightstreamFPrime.Spec.fieldModulus do throw s!"{n} is not below q"
  return ⟨UInt64.ofNat n⟩

/-- An element of `E` of the file: `[c0, c1]`. -/
def ext (value : Json) : Except String Ext := do
  match ← value.getArr? with
  | #[c0, c1] => return ⟨← element c0, ← element c1⟩
  | _ => throw s!"{value} is not a pair"

def exts (value : Json) : Except String (Array Ext) := do (← value.getArr?).mapM ext

/-- A root group of a positive file. -/
structure Group where
  commitment : ByteArray
  digest : ByteArray
  rows : Array Fq
  point : Array Ext
  value : Array Ext

/-- The public inputs and the proof of a positive file. -/
structure Vector where
  name : String
  schedule : String
  shape : ℕ × ℕ × ℕ
  session : ByteArray
  descriptor : ByteArray
  groups : Array Group
  setupElements : ℕ
  plan : Array Run
  proof : ByteArray

/-- An event of a positive file. -/
structure Event where
  op : String
  level : ℕ
  site : String
  json : Json

/-- The events of a positive file and its counts: oracles for the comparison only. -/
structure Oracles where
  events : Array Event
  counts : List (String × ℕ)

def group (value : Json) : Except String Group := do
  return {
    commitment := ← hex (← str value ["commitment"])
    digest := ← hex (← str value ["commitment_digest"])
    rows := ← (← array value ["commitment_rows"]).mapM element
    point := ← exts (← field value ["point"])
    value := ← exts (← field value ["value"]) }

def run (value : Json) : Except String Run := do
  return {
    site := ← str value ["site"], loss := ← nat value ["loss"], bits := ← nat value ["bits"]
    nonceBits := ← nat value ["nonce_bits"], count := ← nat value ["count"]
    setBits := ← nat value ["set_bits"] }

def event (value : Json) : Except String Event := do
  return {
    op := ← str value ["op"], level := ← nat value ["level"]
    site := (str value ["site"]).toOption.getD "", json := value }

def vector (name : String) (value : Json) : Except String (Vector × Oracles) := do
  unless (← str value ["format"]) = "nightstream-akita-opening-v2" do throw "format"
  unless (← str value ["modulus"]) = toString NightstreamFPrime.Spec.fieldModulus do
    throw "modulus"
  unless (← str value ["result"]) = "accept" do throw "result"
  let counts ← match ← field value ["counts"] with
    | .obj entries => entries.toList.mapM fun (op, count) => do return (op, ← count.getNat?)
    | _ => throw "counts"
  return ({
      name
      schedule := ← str value ["schedule"]
      shape := (← nat value ["shape", "key"], ← nat value ["shape", "witness"],
        ← nat value ["shape", "inverse"])
      session := ← hex (← str value ["session"])
      descriptor := ← hex (← str value ["descriptor"])
      groups := ← (← array value ["groups"]).mapM group
      setupElements := ← nat value ["setup_field_elements"]
      plan := ← (← array value ["grinding_plan"]).mapM run
      proof := ← hex (← str value ["proof"]) },
    { events := ← (← array value ["events"]).mapM event, counts })

def directory : System.FilePath := "tests/vectors/akita"

def load (name : String) : IO (Vector × Oracles) := do
  let text ← IO.FS.readFile (directory / name)
  IO.ofExcept do vector name (← Json.parse text)

def positives : List String :=
  ["key20-opening.json", "key20-opening-retry.json", "key30-opening-26-26.json",
    "key30-opening-27-26.json"]

/-- The count of variables of a table of a shape. -/
def shapeVars (S : Shape) : ℕ := Nat.log 2 (S.blocks * S.blockLength * S.innerDegree)

/-- The root shapes of the schedule row of a vector. -/
def rootShapes (v : Vector) : Except String (List Shape) :=
  match v.schedule, v.shape with
  | "crates/ns-compress/artifacts/key20.aks", _ =>
      pure ((List.finRange 3).map Key20.root.shape)
  | "crates/ns-compress/artifacts/key30.aks", (_, 27, _) =>
      pure ((List.finRange 3).map (Key30.root Key30.w27).shape)
  | "crates/ns-compress/artifacts/key30.aks", (_, 26, _) =>
      pure ((List.finRange 3).map (Key30.root Key30.w26).shape)
  | schedule, _ => throw s!"no Lean row for {schedule}"

/-- The shape of the statement of the schedule row of a vector. -/
def rootOf (v : Vector) : Except String (Array RootGroup) :=
  match v.schedule, v.shape with
  | "crates/ns-compress/artifacts/key20.aks", _ => pure root20
  | "crates/ns-compress/artifacts/key30.aks", (_, 27, _) => pure (root30 Key30.w27)
  | "crates/ns-compress/artifacts/key30.aks", (_, 26, _) => pure (root30 Key30.w26)
  | schedule, _ => throw s!"no Lean row for {schedule}"

def fail (message : String) : IO Unit := throw <| IO.userError message

/-- Gate 1 for one positive file. -/
def gate1 (v : Vector) (o : Oracles) : IO Unit := do
  for (op, count) in o.counts do
    let found := (o.events.filter (·.op == op)).size
    unless found = count do fail s!"{v.name}: {found} events {op}, the file counts {count}"
  unless (o.counts.map (·.2)).sum = o.events.size do fail s!"{v.name}: an op without a count"
  let shapes ← IO.ofExcept (rootShapes v)
  unless v.groups.size = shapes.length do fail s!"{v.name}: {v.groups.size} root groups"
  for (g, S) in v.groups.toList.zip shapes do
    unless g.point.size = shapeVars S do
      fail s!"{v.name}: a point of {g.point.size} coordinates for a shape of {shapeVars S}"
    unless g.value.size = 1 do fail s!"{v.name}: a group with {g.value.size} values"
    unless g.rows.size = outer S do
      fail s!"{v.name}: {g.rows.size} commitment rows, the outer commitment has {outer S}"
    let tail := g.rows.foldl (fun bytes row => (List.range 8).foldl
      (fun bytes i => bytes.push (row.word >>> (8 * i).toUInt64).toUInt8) bytes) ByteArray.empty
    unless g.commitment.extract (g.commitment.size - tail.size) g.commitment.size == tail do
      fail s!"{v.name}: the commitment bytes do not end with the rows"
  IO.println s!"gate 1 {v.name}: {o.events.size} events in {o.counts.length} ops equal to counts; \
    {v.groups.size} root groups with points {v.groups.map (·.point.size)} = Lean shapes; \
    proof {v.proof.size} bytes; plan {v.plan.size} runs"

/-- The words of an event as elements of `E`. -/
def eventExts (e : Event) : Except String (Array Ext) := do exts (← field e.json ["words"])

/-- The first event of an `op` and a `site` at a level. -/
def find? (o : Oracles) (op : String) (level : ℕ) (site : String) : Option Event :=
  o.events.find? fun e => e.op == op && e.level == level && e.site == site

def u : Ext := ⟨Fq.zero, Fq.one⟩

#guard u * u == Ext.ofBase (Fq.ofNat 2)
#guard (Ext.ofBase (Fq.ofNat (NightstreamFPrime.Spec.fieldModulus - 1))) *
  (Ext.ofBase (Fq.ofNat (NightstreamFPrime.Spec.fieldModulus - 1))) == Ext.one
#guard (u + Ext.one) * (u - Ext.one) == Ext.one
#guard -(u * Ext.ofBase (Fq.ofNat 3)) + u * Ext.ofBase (Fq.ofNat 3) == Ext.zero

/-- Gate 2 for one positive file: at each level with the Euclidean norm, `norm.square` is the
square of `norm.virtual_evaluations`, and `norm.sumcheck.output` is
`norm.range_equality · norm.range_leaf + norm.merge · norm.square`. The values are those of the
file; the products are those of `Ext`. -/
def gate2 (v : Vector) (o : Oracles) : IO ℕ := do
  let mut levels := 0
  for e in o.events do
    unless e.op == "value" && e.site == "norm.square" do continue
    let get (op site : String) : IO Ext := do
      let some found := find? o op e.level site | throw <| IO.userError s!"{site} {e.level}"
      let values ← IO.ofExcept (eventExts found)
      let some value := values[0]? | throw <| IO.userError s!"{site} {e.level}"
      return value
    let square ← get "value" "norm.square"
    let virtual ← get "message" "norm.virtual_evaluations"
    unless virtual * virtual == square do fail s!"{v.name}: norm.square at level {e.level}"
    let output ← get "value" "norm.sumcheck.output"
    let equality ← get "value" "norm.range_equality"
    let leaf ← get "value" "norm.range_leaf"
    let merge ← get "challenge" "norm.merge"
    unless equality * leaf + merge * square == output do
      fail s!"{v.name}: norm.end at level {e.level}"
    levels := levels + 1
  return levels

/-- The terminal numbers of the schedule file of a vector. -/
structure TerminalFile where
  zMax : ℕ
  e : ℕ
  t : ℕ
  rice : ℕ
  zCoords : ℕ
  linf : Option ℕ
  cap : ℕ

/-- The terminal numbers of the schedule file of a vector: `z_payload_bytes`, `e_field_elems`,
`t_field_elems`, `z_rice_low_bits`, `z_coords` and `z_linf_cap` of `response_shape`, and the cap
`response_l2_sq_cap` of the Euclidean route. -/
def terminalNumbers (v : Vector) : IO TerminalFile := do
  let text ← IO.FS.readFile ("../.." / System.FilePath.mk v.schedule)
  IO.ofExcept do
    let rows ← array (← Json.parse text) ["rows"]
    let row ← match v.schedule, v.shape with
      | "crates/ns-compress/artifacts/key30.aks", (_, 26, _) => pure rows[1]!
      | _, _ => pure rows[0]!
    let terminal ← field row ["schedule", "terminal"]
    let group := (← array terminal ["response_shape", "layout", "groups"])[0]!
    return { zMax := ← nat group ["z_payload_bytes"], e := ← nat group ["e_field_elems"]
             t := ← nat group ["t_field_elems"], rice := ← nat group ["z_rice_low_bits"]
             zCoords := ← nat group ["z_coords"], linf := (nat group ["z_linf_cap"]).toOption
             cap := ← nat terminal ["inner", "matrix", "security_route", "L2",
               "response_l2_sq_cap"] }

/-- The layout of the schedule row of a vector, from the Lean numbers, `zMax` and the Rice
parameter of the file. The file has `e`, `t`, `z_coords` and the cap of the terminal, and they are
the numbers of the layout; it has no `ℓ_∞` cap of `z` (the Euclidean route). -/
def layoutWith (order : ∀ n, Equiv.Perm (Fin n)) (v : Vector) : IO Layout := do
  let file ← terminalNumbers v
  let layout ← match v.schedule, v.shape with
    | "crates/ns-compress/artifacts/key20.aks", _ => pure (layout20With order file.zMax file.rice)
    | "crates/ns-compress/artifacts/key30.aks", (_, 27, _) =>
        pure (layout30With order Key30.w27 file.zMax file.rice)
    | "crates/ns-compress/artifacts/key30.aks", (_, 26, _) =>
        pure (layout30With order Key30.w26 file.zMax file.rice)
    | schedule, _ => throw <| IO.userError s!"no Lean row for {schedule}"
  let T := layout.terminal
  unless T.e = file.e do fail s!"{v.name}: e has {file.e} elements, the layout {T.e}"
  let last := (layout.folds.getLast?.map (·.witness)).getD 0
  unless last = file.t do fail s!"{v.name}: t has {file.t} elements, the layout {last}"
  unless T.zCoords = file.zCoords ∧ T.limit = file.cap ∧ file.linf.isNone do
    fail s!"{v.name}: z_coords, the cap or z_linf_cap of the file differ from the layout"
  return layout

def layoutOf (v : Vector) : IO Layout := layoutWith groupOrder v

/-- The script without the end checks `fused.end` and `stage3.end`, for a replay of a change that
reads no end check: each is about 1 to 35 s in the interpreter of `#eval`. A check reads no
message and writes no recorded value, so `decode` and the recorded values are the same. -/
def withoutEnds (ops : Array Op) : Array Op :=
  ops.filter fun | .check _ (.fused ..) | .check _ (.stage3 ..) => false | _ => true

/-- The levels of a layout with evaluation trace and with coefficient packing. -/
def fusedLevels (layout : Layout) : List ℕ × List ℕ :=
  let trace (F : Fold) : Bool := match F.fused with | .trace _ => true | .packing _ => false
  (layout.folds.zipIdx.filterMap fun (F, level) => if trace F then some level else none,
    layout.folds.zipIdx.filterMap fun (F, level) => if trace F then none else some level)

/-- The count of setup elements that the end checks and the terminal of a layout read. -/
def setupCount (layout : Layout) : ℕ :=
  layout.folds.foldl (fun m F => max m F.setupCount) layout.terminal.setupCount

/-- A positive file with its script, its public input with the setup prefix, its decoded
messages, and the one replay of its proof (`replay`): its stop and what it derives. -/
structure Replayed where
  v : Vector
  o : Oracles
  layout : Layout
  ops : Array Op
  pub : Public
  messages : Array Message
  stop : Option Stop
  d : Derived

/-- The state at the end of the replay, for the readers of `Replayer`. -/
def Replayed.state (r : Replayed) : Replayer := ⟨Duplex.empty, 0, r.d.recorded, r.d.fields, r.d.integers, r.d⟩

/-- The operations of a script that read a prover message, in order. -/
def messageOps (ops : Array Op) : Array Op :=
  ops.filter fun
    | .fields .. | .exts .. | .integer .. | .digest .. | .bounded .. | .foldNonce .. => true
    | .work _ _ bits => bits != 0
    | _ => false

def decimal (value : Json) : Except String ℕ := do
  match (← value.getStr?).toNat? with
  | some n => pure n
  | none => throw s!"{value} is not a decimal string"

def hexOf (bytes : ByteArray) : String :=
  bytes.foldl (fun text b =>
    let digit (n : UInt8) := Char.ofNat (if n < 10 then 48 + n.toNat else 87 + n.toNat)
    (text.push (digit (b / 16))).push (digit (b % 16))) ""

/-- The events of a decoded message: its site, round, offset, length and words. -/
def expected (op : Op) (m : Message) : List (String × Option ℕ × ℕ × ℕ × String) :=
  match op, m.data with
  | .fields k _, .fields values => [(k.site, k.index, m.offset, m.size,
      toString (values.toList.map (·.word.toNat)))]
  | .exts k _, .exts values => [(k.site, k.index, m.offset, m.size,
      toString (values.toList.map fun e => [e.c0.word.toNat, e.c1.word.toNat]))]
  | .integer k, .integer value => [(k.site, k.index, m.offset, m.size, toString value)]
  | .digest k, .bytes value => [(k.site, k.index, m.offset, m.size, hexOf value)]
  | .bounded k _, .bytes value => [(k.site ++ ".length", none, m.offset, 4, toString value.size),
      (k.site, none, m.offset + 4, value.size, hexOf value)]
  | .work site _ bits, .nonce value _ =>
      [(site.text, none, m.offset, m.size, s!"{value} with {bits} bits")]
  | .foldNonce _ _, .nonce value _ => [("fold_nonce", none, m.offset, m.size, toString value)]
  | _, _ => [("?", none, m.offset, m.size, "?")]

/-- The same tuple from an event of the file. -/
def observed (e : Event) : Except String (String × Option ℕ × ℕ × ℕ × String) := do
  let round := (nat e.json ["round"]).toOption
  let offset ← nat e.json ["offset"]
  let size ← nat e.json ["bytes"]
  match e.op with
  | "work" => return (e.site, none, offset, size,
      s!"{← nat e.json ["nonce"]} with {← nat e.json ["bits"]} bits")
  | "fold_nonce" => return ("fold_nonce", none, offset, size, toString (← nat e.json ["nonce"]))
  | _ =>
    let words ← field e.json ["words"]
    let text ← match words with
      | .str s => pure s
      | .num _ => pure (toString (← words.getNat?))
      | .arr items =>
          if items.all (fun item => (item.getArr?).isOk) then
            pure (toString (← items.toList.mapM fun item => do
              (← item.getArr?).toList.mapM decimal))
          else pure (toString (← items.toList.mapM decimal))
      | _ => throw s!"words of {e.site}"
    return (e.site, round, offset, size, text)

/-- Gate 3 for one positive file: the script of its layout and plan, the messages that `decode`
reads from the proof with no byte left, and each `message`, `work` and `fold_nonce` event of the
file equal to a decoded message in order (site, round, offset, length, words, bits of a work). -/
def gate3 (v : Vector) (o : Oracles) : IO (Array Op) := do
  let layout ← layoutOf v
  let ops ← IO.ofExcept (script layout v.plan)
  let (messages, rest) ← match decode ops v.proof with
    | .ok result => pure result
    | .error site => throw <| IO.userError s!"{v.name}: malformed at {repr site}"
  unless rest = 0 do fail s!"{v.name}: {rest} trailing bytes"
  let derived := ((messageOps ops).toList.zip messages.toList).flatMap fun (op, m) => expected op m
  let events := o.events.filter fun e => e.op == "message" || e.op == "work" || e.op == "fold_nonce"
  unless (messageOps ops).size = messages.size do fail s!"{v.name}: message count"
  unless derived.length = events.size do
    fail s!"{v.name}: {derived.length} decoded message events, the file has {events.size}"
  for (d, e) in derived.zip events.toList do
    let found ← IO.ofExcept (observed e)
    unless d == found do
      fail s!"{v.name}: decoded {d.1} {d.2.1} at {d.2.2.1} ({d.2.2.2.1} bytes), the file has \
        {found.1} {found.2.1} at {found.2.2.1} ({found.2.2.2.1} bytes)"
  IO.println s!"gate 3 {v.name}: {ops.size} operations, {messages.size} messages to the end of \
    {v.proof.size} bytes; {events.size} events message/work/fold_nonce equal (site, round, offset, \
    bytes, words, work bits)"
  return ops

/-- Each key of the script has one operation that writes it (`Op.writes`), except the point of a
sum-check, which each round writes before the end of the sum-check: so by `lookup_later` each
value that a check reads is the value of its writer. -/
def writesGate (v : Vector) (ops : Array Op) : IO Unit := do
  let mut writers : Std.HashMap Key (List ℕ) := {}
  for (op, i) in ops.zipIdx do
    for key in op.writes do
      writers := writers.insert key (((writers.get? key).getD []) ++ [i])
  let mut points := 0
  for (key, indices) in writers.toList do
    if indices.length = 1 then continue
    let ended := ops.findIdx? fun
      | .ended k _ _ => k.level == key.level && k.site ++ ".point" == key.site
      | _ => false
    unless key.site.endsWith ".point" ∧ (ended.map fun e => indices.all (· < e)) == some true do
      fail s!"{v.name}: {indices.length} operations write {key.site} of level {key.level}"
    points := points + 1
  IO.println s!"writes {v.name}: {writers.size} keys; each has one writer except the points of \
    {points} sum-checks, whose rounds write them before the end"

example : ∀ x ∈ [(Op.work (.tau0 0) 0 0, 0)], (⟨0, "tau0", none⟩ : Key) ∉ x.1.writes := by
  simp [Op.writes]

/-- A change of a negative file applied to a proof. -/
def change (proof : ByteArray) (c : Json) : Except String ByteArray := do
  if let .ok offset := nat c ["offset"] then
    let mask := (← hex (← str c ["xor"]))[0]!
    return proof.set! offset (proof.get! offset ^^^ mask)
  if let .ok bytes := str c ["append"] then return proof ++ (← hex bytes)
  if let .ok count := nat c ["truncate"] then return proof.extract 0 (proof.size - count)
  throw "a change without offset, append or truncate"

/-- The byte layer on each change of a negative file: each change of the stage `parse` is
malformed and each change of the stage `trailing_bytes` has trailing bytes. Returns the outcome
classes of the byte layer for each stage. -/
def gate3Negative (name : String) (ops : Array Op) (proof : ByteArray) :
    IO (List (String × String × ℕ)) := do
  let text ← IO.FS.readFile (directory / name)
  let changes ← IO.ofExcept do array (← Json.parse text) ["changes"]
  let mut classes : List (String × String × ℕ) := []
  for c in changes do
    let stage ← IO.ofExcept (str c ["stage"])
    let changed ← IO.ofExcept (change proof c)
    let outcome := match decode ops changed with
      | .error _ => "malformed"
      | .ok (_, rest) => if rest = 0 then "parsed" else "trailing"
    if stage == "parse" && outcome != "malformed" then fail s!"{name}: a parse change is {outcome}"
    if stage == "trailing_bytes" && outcome != "trailing" then
      fail s!"{name}: a trailing change is {outcome}"
    classes := match classes.find? (fun (s, k, _) => s == stage && k == outcome) with
      | some _ => classes.map fun (s, k, n) => if s == stage && k == outcome then (s, k, n + 1)
          else (s, k, n)
      | none => classes ++ [(stage, outcome, 1)]
  IO.println s!"gate 3 {name}: {changes.size} changes; (stage, byte layer, count) = {classes}"
  return classes

/-! Gate 4. FIPS 202 vectors of SHAKE128 and SHA3-256, and the SHAKE128 of `Keccak` against the
list form of `AjtaiSetupV1/Shake128.lean` on inputs of one and more blocks. -/

#guard hexOf (Keccak.sha3_256 ByteArray.empty) ==
  "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
#guard hexOf (Keccak.sha3_256 "abc".toUTF8) ==
  "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
#guard hexOf (Keccak.shake128 ByteArray.empty 32) ==
  "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
#guard hexOf (Keccak.shake128 "abc".toUTF8 32) ==
  "5881092dd818bf5cf8a3ddb793fbcba74097d5c526a6d35f97b83351940f2cc8"
#guard [0, 1, 135, 136, 167, 168, 169, 400].all fun n =>
  let input := (List.range n).map (· * 7 % 256)
  (Keccak.shake128 (ByteArray.mk (input.map (·.toUInt8)).toArray) 400).toList.map (·.toNat) ==
    NightstreamFPrime.Spec.AjtaiSetupV1.Shake128.bytes input 400

/-- The challenges of a replay by key, the rounds of a sum-check joined. -/
def joined (d : Derived) : Std.HashMap Key (Array Ext) :=
  d.challenges.foldl (fun m (key, values) => m.insert key (((m.get? key).getD #[]) ++ values)) {}

/-- The absorbs of the file in order: level, site, and words or bytes. -/
def absorbEvents (o : Oracles) : Array (ℕ × String × String) :=
  (o.events.filter (·.op == "absorb")).map fun e =>
    (e.level, e.site, match field e.json ["words"] with
      | .ok _ => match eventExts e with
        | .ok values => hexOf (extBytes values)
        | .error _ => "?"
      | .error _ => (str e.json ["bytes"]).toOption.getD "?")

#guard [2, 3, 4, 5, 6].all fun logBasis =>
  (productStages logBasis).map (·.1) = rangeArities (2 ^ logBasis) ∧
    leafDegree logBasis = rangeLeaf (2 ^ logBasis)

#guard energyFits 5 5 && !energyFits 6 5 &&
  !energyFits NightstreamFPrime.Spec.fieldModulus (NightstreamFPrime.Spec.fieldModulus + 1)

/-- The results of the stage functions that the replay records at the last check of each stage
group (`Derived.stages`; `step_range_runs`, `step_norm_runs`, `step_tensor_runs` of
`Replay/Transcript.lean`): `rangeRun`, `treeRun` with `normRun`, and `tensorRun` accept with the point
and the values of the replay, at each level of the layout. The hypotheses of the three theorems
hold on the recorded values: the arities, the length of the range point, the count and the
length of the rounds, and a norm cap below `q`. -/
def stagesGate (v : Vector) (d : Derived) (layout : Layout) : IO Unit := do
  let s : Replayer := ⟨Duplex.empty, 0, d.recorded, d.fields, d.integers, d⟩
  let get (level : ℕ) (site : String) : Array Ext := Replayer.recordedIn s ⟨level, site, none⟩
  let one (level : ℕ) (site : String) : Ext := Replayer.firstIn s ⟨level, site, none⟩
  let shaped (rounds : List (List Ext × Ext)) (count width : ℕ) : Bool :=
    rounds.length == count && rounds.all (·.1.length == width)
  for (F, level) in layout.folds.zipIdx do
    let arities := (productStages F.logBasis).map (·.1)
    let point := Replayer.pointIn s (.turned ⟨level, "tau0", none⟩ F.low)
    let products := Replayer.productsIn s level arities
    unless products.map (·.arity) == rangeArities (2 ^ F.logBasis) && point.length == F.vars &&
        products.all (fun S => shaped S.rounds F.vars S.arity) do
      fail s!"{v.name}: the product stages of level {level} do not have the shapes of the script"
    if F.euclidean then
      unless F.limit < NightstreamFPrime.Spec.fieldModulus &&
          shaped (Replayer.roundsIn s ⟨level, "norm.sumcheck", none⟩) F.vars
            (leafDegree F.logBasis + 1) do
        fail s!"{v.name}: the norm check of level {level} does not have the shapes of the script"
    else
      unless shaped (Replayer.roundsIn s ⟨level, "range.leaf", none⟩) F.vars
          (rangeLeaf (2 ^ F.logBasis)) do
        fail s!"{v.name}: the leaf stage of level {level} does not have the shapes of the script"
  let mut ranges := 0
  let mut norms := 0
  let mut tensors := 0
  let mut fuseds := 0
  let mut stage3s := 0
  for (key, out) in d.stages do
    let level := key.level
    match out with
    | .fused out =>
        let some F := layout.folds[level]? | fail s!"{v.name}: no level {level}"
        match F.fused with
        | .trace T =>
            let V := Replayer.traceValuesIn s level F
            let d_A := T.shape.innerDegree
            unless out.isSome && V.r2.length == F.vars && V.virtual.length == F.vars &&
                V.tau1.length == F.batch && F.compressed == T.opening.compressed T.fold &&
                0 < T.fold.responseDigits && T.opening.cells T.fold ≤ 2 ^ F.vars &&
                T.opening.traceRowCount + 1 ≤ 2 ^ F.batch && 4 * (d_A / 4) == d_A &&
                (2 * (d_A / 4)) % NightstreamFPrime.Spec.fieldModulus != 0 do
              fail s!"{v.name}: fused.end at level {level}: the shapes of step_fused_runs"
        | .packing P =>
            unless out.isSome && !F.euclidean && P.slots.length = F.groups.length do
              fail s!"{v.name}: fused.end at level {level}: the shapes of step_packing_runs"
        fuseds := fuseds + 1
    | .stage3 out =>
        let rounds := Replayer.roundsIn s ⟨level, "stage3", none⟩
        unless out.isSome && rounds.all (·.1.length == 2) &&
            some rounds.length == (layout.folds[level]?.bind (·.stage3)) do
          fail s!"{v.name}: stage3.end at level {level}: the shapes of step_stage3_runs"
        stage3s := stage3s + 1
    | .range out =>
        unless out == some ((get level "range.leaf.point").toList, one level "range.image") do
          fail s!"{v.name}: rangeRun at level {level}"
        ranges := ranges + 1
    | .norm out =>
        unless out == some ((get level "norm.sumcheck.point").toList, one level "range.image",
            one level "norm.virtual_evaluations") do
          fail s!"{v.name}: treeRun with normRun at level {level}"
        norms := norms + 1
    | .tensor out =>
        unless shaped (Replayer.roundsIn s ⟨level, "tensor.sumcheck", none⟩)
            (Replayer.roundsIn s ⟨level, "tensor.sumcheck", none⟩).length 2 &&
          (out.map fun e => (e.point, e.claim)) ==
            some ((get level "tensor.sumcheck.point").toList, one level "tensor.final_claims") do
          fail s!"{v.name}: tensorRun at level {level}"
        tensors := tensors + 1
  let euclidean := (layout.folds.filter (·.euclidean)).length
  unless ranges = layout.folds.length - euclidean ∧ norms = euclidean ∧
      tensors = (layout.folds.filter (·.trace.isSome)).length + 1 ∧
      fuseds = layout.folds.length ∧
      stage3s = (layout.folds.filter (·.setup.isSome)).length do
    fail s!"{v.name}: the replay records {ranges}, {norms}, {tensors}, {fuseds}, {stage3s} stage \
      results"
  IO.println s!"gate 7 {v.name}: the replay records rangeRun at {ranges} levels, treeRun with \
    normRun at {norms} levels, tensorRun at {tensors} levels, traceFused or foldFused at {fuseds} \
    levels and stage3Run at {stage3s} (step_range_runs, step_norm_runs, step_tensor_runs, \
    step_fused_runs, step_packing_runs, step_stage3_runs), each accepting with the point and the \
    values of the replay; the shapes of the theorems hold"

/-- The factor `θ` of the tensor reduction of each level with evaluation trace and of the terminal
(`tensorFactor`, `tensorFactor_into`), from the values of the replay, is the value
`tensor.factor` of the file; with `ρ` in the reverse order it differs at some level. -/
def factorGate (v : Vector) (o : Oracles) (d : Derived) (layout : Layout) : IO Unit := do
  let get (level : ℕ) (site : String) : Array Ext := (d.recorded.get? ⟨level, site, none⟩).getD #[]
  let levels := (layout.folds.zipIdx.filterMap fun (F, level) => F.trace.map fun _ => level) ++
    [layout.folds.length]
  let factor (level : ℕ) (ρ : List Ext) : Ext :=
    tensorFactor (get (level - 1) "fused.point").toList.tail
      ((get level "tensor.eta").getD 0 Ext.zero) ρ
  let mut moved := false
  for level in levels do
    let ρ := (get level "tensor.sumcheck.point").toList
    let some e := find? o "value" level "tensor.factor"
      | fail s!"{v.name}: no tensor.factor at level {level}"
    let expected ← IO.ofExcept (eventExts e)
    unless #[factor level ρ] == expected do
      fail s!"{v.name}: tensor.factor of level {level} differs"
    moved := moved || #[factor level ρ.reverse] != expected
  unless moved do fail s!"{v.name}: tensor.factor does not depend on the order of ρ"
  IO.println s!"gate 7 {v.name}: tensorFactor = tensor.factor at {levels.length} levels; \
    with ρ reversed it differs"

/-- The challenges of a `fold_challenges` event. -/
def eventSparse (e : Event) : Except String (List Sparse) := do
  (← array e.json ["sparse"]).toList.mapM fun c => do
    (← c.getArr?).toList.mapM fun pair => do
      match ← pair.getArr? with
      | #[p, v] => return (← p.getNat?, ← v.getInt?)
      | _ => throw "a pair of a challenge"

/-- Gate 8: each `fold_challenges` event equals the challenges that the replay draws for its level
and group (`Derived.drawn`: `foldChallenges` of the squeezed message of the group), pair by pair
in the ring of `A`, in the order of the squeezes. Returns the count of challenges. -/
def foldGate (v : Vector) (o : Oracles) (d : Derived) : IO ℕ := do
  let events := o.events.filter (·.op == "fold_challenges")
  unless d.drawn.size = events.size ∧ d.foldMessages.size = events.size do
    fail s!"{v.name}: {d.drawn.size} draws, {events.size} fold_challenges events"
  let mut count := 0
  for ((level, group, cs), e) in d.drawn.toList.zip events.toList do
    unless level = e.level ∧ (nat e.json ["group"]).toOption == some group ∧
        (eventSparse e).toOption == some cs do
      fail s!"{v.name}: the fold challenges of group {group} of level {level} differ"
    count := count + cs.length
  return count

/-- Gate 10: the values of the terminal checks of the replay (`Derived.terminal`) equal the events:
the decoded `z` (`terminal.z.coefficients`), `‖z‖_2²` and the cap (`terminal.z.bounds`), the
recovered last block of `e` (`terminal.e_last_block`) and the opening (`terminal.opening`). -/
def terminalGate (v : Vector) (o : Oracles) (d : Derived) (layout : Layout) : IO Unit := do
  let T := layout.folds.length
  let some out := d.terminal | fail s!"{v.name}: the replay has no terminal values"
  let event (site : String) : IO Event := do
    let some e := find? o "value" T site | throw <| IO.userError s!"{v.name}: no event {site}"
    pure e
  let zEvent ← event "terminal.z.coefficients"
  let z ← IO.ofExcept do (← array zEvent.json ["words"]).mapM (·.getInt?)
  unless z == out.z do fail s!"{v.name}: z differs"
  let bounds ← event "terminal.z.bounds"
  unless (decimal (← IO.ofExcept (field bounds.json ["l2_sq"]))).toOption == some out.normSq ∧
      (decimal (← IO.ofExcept (field bounds.json ["cap"]))).toOption == some layout.terminal.limit do
    fail s!"{v.name}: the norm bounds differ"
  let eEvent ← event "terminal.e_last_block"
  let eLast ← IO.ofExcept do (← array eEvent.json ["words"]).mapM element
  unless eLast == Poly.canon layout.terminal.shape.innerDegree out.eLast do
    fail s!"{v.name}: the last block of e differs"
  unless (eventExts (← event "terminal.opening")).toOption == some #[out.opening] do
    fail s!"{v.name}: the opening differs"
  IO.println s!"gate 10 {v.name}: z ({out.z.size}), ‖z‖² = {out.normSq} ≤ {layout.terminal.limit},     the last block of e and the opening of the replay equal to the events"

/-- Gate 4 on a positive file: the replay from the bytes of the proof and the public inputs only.
Each `challenge` event of the file equals the derived challenge, each proof of work passes, each
absorb equals the absorb of the file, each value `opening_target`, `relation_claim` and
`opening_claim` of the file equals the value of the replay, each `fold_challenges` event equals
the drawn challenges, each `check` event is a check of the replay that passes, in order, `fused.end`
holds at each level and `stage3.end` at each level with Stage 3, and the values of the terminal
equal the events. `verifyBytes` runs with the verdict of this replay (`verdict` is `verdictOf` of
the stop of the replay of the decoded messages) and no check left to `checks`. -/
def gate4 (r : Replayed) : IO Unit := do
  let v := r.v
  let o := r.o
  let d := r.d
  if let some s := r.stop then fail s!"{v.name}: the replay stops: {repr s}"
  let derived := joined d
  let challengeEvents := o.events.filter (·.op == "challenge")
  for e in challengeEvents do
    let expected ← IO.ofExcept (eventExts e)
    unless derived.get? ⟨e.level, e.site, none⟩ == some expected do
      fail s!"{v.name}: the challenge {e.site} of level {e.level} differs"
  unless derived.size = challengeEvents.size do
    fail s!"{v.name}: {derived.size} derived challenge vectors, {challengeEvents.size} events"
  let works := o.events.filter (·.op == "work")
  unless d.works.size = works.size ∧ d.works.all (·.2.2.2) do fail s!"{v.name}: works"
  let absorbs := absorbEvents o
  unless d.absorbs.size = absorbs.size do
    fail s!"{v.name}: {d.absorbs.size} absorbs, the file has {absorbs.size}"
  for ((key, bytes), (level, site, value)) in d.absorbs.toList.zip absorbs.toList do
    unless key.level = level ∧ key.site = site ∧ hexOf bytes = value do
      fail s!"{v.name}: the absorb {key.site} of level {key.level} differs from {site} {level}"
  let names := ["opening_target", "relation_claim", "opening_claim"]
  let valueEvents := o.events.filter fun e => e.op == "value" && names.contains e.site
  unless d.values.size = valueEvents.size do
    fail s!"{v.name}: {d.values.size} derived values, the file has {valueEvents.size}"
  for ((key, value), e) in d.values.toList.zip valueEvents.toList do
    unless key.level = e.level ∧ key.site = e.site ∧ (← IO.ofExcept (eventExts e)) == #[value] do
      fail s!"{v.name}: the value {key.site} of level {key.level} differs from {e.site} {e.level}"
  let checkEvents := o.events.filter fun e => e.op == "check" && !e.site.endsWith ".claim_chain"
  let checked := d.checks.filter fun key => key.site != "fused.end" && key.site != "stage3.end"
  unless checked.size = checkEvents.size ∧
      (checked.zip checkEvents).all (fun (key, e) => key.level == e.level && key.site == e.site) do
    fail s!"{v.name}: the checks of the replay are not the check events of the file"
  let levels := List.range r.layout.folds.length
  for level in levels do
    unless d.checks.contains ⟨level, "fused.end", none⟩ do
      fail s!"{v.name}: fused.end of level {level} is not a check of the replay that passes"
  let stage3s := r.layout.folds.zipIdx.filterMap fun (F, level) => F.setup.map fun _ => level
  for level in stage3s do
    unless d.checks.contains ⟨level, "stage3.end", none⟩ do
      fail s!"{v.name}: stage3.end of level {level} is not a check of the replay that passes"
  let (traces, packings) := fusedLevels r.layout
  IO.println s!"end checks {v.name}: fused.end holds at each level, {traces} with evaluation \
    trace and {packings} with coefficient packing; stage3.end at {stage3s}"
  let terminal := (d.checks.filter (·.site.startsWith "terminal.")).size
  IO.println s!"gate 7 {v.name}: {d.checks.size} checks pass in the order of the {checkEvents.size} \
    check events (tensor.opening, tensor.end, range.<k>.end, range.leaf.end, norm.integer, \
    norm.end and {terminal} of the terminal: terminal.z.norm, terminal.e_last_block.digest, \
    terminal.trace) and the {d.checks.size - checkEvents.size} end checks"
  stagesGate v d r.layout
  factorGate v o d r.layout
  let drawn ← foldGate v o d
  IO.println s!"gate 8 {v.name}: {d.drawn.size} fold_challenges events equal to the draws of the \
    replay ({drawn} challenges)"
  terminalGate v o d r.layout
  let fused := (d.absorbs.filter (·.1.site == "fused.claim")).size
  IO.println s!"gate 4 {v.name}: {challengeEvents.size} challenge events equal, {d.works.size} \
    works pass, {d.absorbs.size} absorbs equal ({fused} fused.claim), {d.values.size} values \
    equal ({names}), {d.squeezed} bytes squeezed; the replay has no input from the events"
  match verifyBytes (verdictOf r.stop fun _ => .accept) r.ops v.proof with
  | .verdict .accept => IO.println s!"verdict {v.name}: accept; each check of the replay passes: \
      fused.end at the levels {levels}, stage3.end at {stage3s}, and terminalRun accepts (the \
      norm of z, the fold challenges and the unit, the last block of e, the rows of A, the \
      digests of the last blocks of e and t, the trace)"
  | other => fail s!"{v.name}: verifyBytes gives {repr other}"

/-- The order of the groups in the rows of the relation: with the groups in their own order in
place of `groupOrder` of the specification, the relation claim of the root differs from the file
and the replay stops at a work. -/
def orderGate (v : Vector) (o : Oracles) (pub : Public) : IO Unit := do
  let layout ← layoutWith (fun _ => Equiv.refl _) v
  let ops ← IO.ofExcept (script layout v.plan)
  let some (messages, _) := (decode ops v.proof).toOption | throw <| IO.userError "decode"
  let (stop, d) := replay { pub with layout } ops v.proof messages
  let some e := find? o "value" 0 "relation_claim" | throw <| IO.userError "relation_claim"
  let claim := (d.values.find? (·.1 == ⟨0, "relation_claim", none⟩)).map (#[·.2])
  if claim == some (← IO.ofExcept (eventExts e)) then
    fail s!"{v.name}: the relation claim does not depend on the order of the groups"
  let .some (.noWork _ site) := stop
    | fail s!"{v.name}: the replay with the groups in their own order does not stop at a work"
  IO.println s!"gate 4 {v.name}: with the groups in their own order the relation claim of the \
    root differs and the replay stops at the work {site}"

/-- The public input of a positive file. The slot of Stage 3 (the `stage3.slot` absorb) and the
commitment rows of the setup prefix (the value `setup_prefix.commitment_rows`) are statement
values of the file that Lean does not derive from the schedule and the seed yet. The seed of the
setup is `sample_akita_setup_seed` of the code (`Setup.sampleSeed`). -/
def publicOf (v : Vector) (o : Oracles) (layout : Layout) : IO Public := do
  let slot := (o.events.find? fun e => e.op == "absorb" && e.site == "stage3.slot").map
    fun e => (str e.json ["bytes"]).toOption.getD ""
  let setupRows ← match o.events.find? (·.site == "setup_prefix.commitment_rows") with
    | some e => IO.ofExcept do (← array e.json ["words"]).mapM element
    | none => pure #[]
  return { layout, plan := v.plan, session := v.session, descriptor := v.descriptor
           groups := v.groups.map fun g => ⟨g.commitment, g.point, g.value⟩
           slot := ← IO.ofExcept (hex (slot.getD "")), setupRows, seed := Setup.sampleSeed
           root := ← IO.ofExcept (rootOf v) }

/-- The statement of a positive file has the shape of its schedule row. -/
def shapedGate (v : Vector) (pub : Public) : IO Unit := do
  unless pub.shaped do fail s!"{v.name}: the statement does not have the shape of its row"
  IO.println s!"statement {v.name}: Public.shaped holds; {pub.root.size} commitments with the \
    profile headers of the Lean shapes, points of {pub.root.toList.map (·.vars)} coordinates"

/-- The statement of a change of a statement negative file: the group of the change with its
commitment, its point or its value. -/
def changedStatement (pub : Public) (c : Json) : Except String Public := do
  let group ← nat c ["group"]
  let some g := pub.groups[group]? | throw s!"no group {group}"
  let commitment ← match c.getObjVal? "commitment" with
    | .ok bytes => hex (← bytes.getStr?)
    | .error _ => pure g.commitment
  let point ← match c.getObjVal? "point" with
    | .ok words => exts words
    | .error _ => pure g.point
  let value ← match c.getObjVal? "value" with
    | .ok words => exts words
    | .error _ => pure g.value
  return { pub with groups := pub.groups.set! group ⟨commitment, point, value⟩ }

/-- The changes of the statement of a negative file with the proof of the positive file:
`verifyProof` accepts exactly when the Rust verifier accepts. The only change that the Rust
verifier accepts is no change, and the replay of the positive file gives its result (`gate4`).
`verifyProof` rejects each other change: the statement at `Public.shaped`, and a statement with
the shape of the row in the replay. -/
def statementNegative (name : String) (pub : Public) (proof : ByteArray) : IO Unit := do
  let start ← IO.monoMsNow
  let json ← IO.ofExcept (Json.parse (← IO.FS.readFile (directory / name)))
  unless (str json ["format"]).toOption == some "nightstream-akita-statement-negative-v1" do
    fail s!"{name}: format"
  let changes ← IO.ofExcept (array json ["changes"])
  let mut accepted := 0
  let mut classes : List (String × ℕ) := []
  for (c, index) in changes.toList.zipIdx do
    let kind ← IO.ofExcept (str c ["change"])
    let result ← IO.ofExcept (str c ["result"])
    let changed ← IO.ofExcept (changedStatement pub c)
    if result == "accept" then
      unless kind == "none" && changed.shaped do
        fail s!"{name}: change {index} {kind}: the file has accept"
      accepted := accepted + 1
    else
      let outcome := verifyProof changed proof
      if outcome == .verdict .accept then
        fail s!"{name}: change {index} {kind}: the file has {result}, verifyProof accepts"
      unless changed.shaped || outcome == .verdict (.reject 0 "statement") do
        fail s!"{name}: change {index} {kind}: {repr outcome} for a statement without the shape"
      let key := (if changed.shaped then "the replay: " else "Public.shaped: ") ++
        (if kind.startsWith "header." then "header" else kind)
      classes := match classes.find? (·.1 == key) with
        | some _ => classes.map fun (k, n) => if k == key then (k, n + 1) else (k, n)
        | none => classes ++ [(key, 1)]
  unless accepted = 1 do fail s!"{name}: {accepted} changes with accept"
  IO.println s!"statement {name}: {changes.size} statements, each with the result of the Rust \
    verifier; 1 accept (no change); verifyProof rejects at (place: change, count), \
    {(← IO.monoMsNow) - start} ms:"
  for (k, n) in classes do IO.println s!"  {k}: {n}"

/-- T3 for the root: the SHA3-256 digest of the canonical bytes of each root commitment is the
digest of the file. -/
def digests (v : Vector) : IO Unit := do
  for g in v.groups do
    unless commitmentDigest g.commitment == g.digest do fail s!"{v.name}: a root digest"
  IO.println s!"gate 4 {v.name}: {v.groups.size} root commitment digests = SHA3-256 of the bytes"

/-- The plan against the rule of the model and against the layout. -/
def planGate (v : Vector) (layout : Layout) : IO Unit := do
  let differences := planDifferences layout v.plan
  unless differences.isEmpty do fail s!"{v.name}: plan runs differ: {repr differences}"
  IO.println s!"gate 4 {v.name}: {v.plan.size} runs: each site and loss of the layout \
    (script), bits = Extract.workBits loss, FoldChallengeWork bits = foldWorkBits loss setBits"

/-- The site name of the code for a work that fails at an operation of the script
(`akita_vectors/replay.rs`): the challenge after the work, the round of a sum-check, the partials
of the tensor reduction, or the response nonce. -/
def failureOf (ops : Array Op) (i : ℕ) : ℕ × String :=
  match ops[i]? with
  | some (.work (.round _ level _ _) _ _) => match ops[i + 1]? with
    | some (.challenge key _) => (level, ((key.site.splitOn ".point").headD "") ++ ".round")
    | _ => (level, "?")
  | some (.work (.extensionPoint level) _ _) => (level, "tensor.partials")
  | some (.work (.foldChallengeWork level) _ _) => (level, "fold_nonce")
  | some (.foldNonce level _) => (level, "fold_nonce")
  | some (.work _ level _) => match ops[i + 1]? with
    | some (.challenge key _) => (level, key.site)
    | _ => (level, "?")
  | some (.challenge key _) => (key.level, key.site)
  | _ => (0, "?")

/-- The script operation of the failed site of the file: the work or the challenge of
`failureOf`, or the check of that name. -/
def failedAt (ops : Array Op) (failedLevel : ℕ) (failedSite : String) : Option ℕ :=
  (List.range ops.size).find? fun k => failureOf ops k == (failedLevel, failedSite) ||
    match ops[k]? with
    | some (.check key _) => key.level == failedLevel && key.site == failedSite
    | _ => false

/-- The step of the states pass: the end checks `fused.end` and `stage3.end` and the terminal are
skipped. They write only `Derived.stages`, `Derived.checks` and `Derived.terminal`, which no
operation reads, so each other part of a state is the state of `replay`. -/
def statesStep (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ) :
    Op → ReplayM Unit
  | .check _ (.fused ..) | .check _ (.stage3 ..) | .terminal _ => pure ()
  | op => Replayer.step pub proof messages i op

/-- The states of the states pass on `proof` before the operations `marks` (increasing), until it
stops. -/
def statesAt (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message)
    (marks : List ℕ) : Array (ℕ × Replayer) := Id.run do
  let mut s : Replayer := ⟨channel pub.session pub.descriptor, 0, {}, {}, {}, {}⟩
  let mut out : Array (ℕ × Replayer) := #[(0, s)]
  let mut at_ := 0
  for mark in marks do
    let run : ReplayM Unit := (ops.zipIdx.extract at_ mark).forM fun (op, i) =>
      statesStep pub proof messages i op
    match (run.run).run s with
    | (.ok (), s') => s := s'; at_ := mark; out := out.push (mark, s)
    | (.error _, _) => break
  return out

/-- The stop of the replay of the operations from `start` with the state `s`. -/
def stopFrom (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message)
    (start : ℕ) (s : Replayer) : Option Stop :=
  let run : ReplayM Unit := (ops.zipIdx.extract start ops.size).forM fun (op, i) =>
    Replayer.step pub proof messages i op
  match (run.run).run s with
  | (.ok (), _) => none
  | (.error stop, _) => some stop

/-- The states that a changed proof resumes from: before each `64`-th operation and after each
end check and the terminal, from the states pass of the positive proof. A state before the
operation `k` is a state of the replay of a changed proof with the same bytes before the end of
the messages that the operations before `k` read: `decode` reads from left to right, the replay
reads only these messages and their bytes, and the end checks before `k` pass on them. -/
structure Resume where
  states : Array (ℕ × Replayer)
  ends : Array ℕ

def resumeOf (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message) :
    Resume :=
  let ended (op : Op) : Bool := match op with
    | Op.check _ (Check.fused ..) | Op.check _ (Check.stage3 ..) | Op.terminal _ => true
    | _ => false
  let marks := (List.range ops.size).filter fun k =>
    k != 0 && (k % 64 == 0 || ((ops[k - 1]?).map ended).getD false)
  ⟨statesAt pub ops proof messages marks,
    messages.map fun m => m.offset + m.size⟩

/-- The stop of the replay of a changed proof whose bytes before `offset` are those of the
positive proof: from the last state whose messages end before `offset`. -/
def Resume.stop (r : Resume) (pub : Public) (ops : Array Op) (proof : ByteArray) (offset : ℕ)
    (messages : Array Message) : Option Stop :=
  let fits (s : Replayer) : Bool := s.next == 0 || r.ends.getD (s.next - 1) 0 ≤ offset
  match r.states.reverse.find? (fits ·.2) with
  | some (start, s) => stopFrom pub ops proof messages start s
  | none => (replay pub ops proof messages).1

/-- The level of an equation or of the terminal at the operation `i`. -/
def checkLevel (ops : Array Op) (i : ℕ) : Option ℕ :=
  match ops[i]? with
  | some (.check key _) => some key.level
  | some (.terminal level) => some level
  | _ => none

/-- The outcome of `verifyBytes` with the verdict of the replay on a changed proof: the class
(`malformed`, `trailing`, `noWork`, `reject` or `accept`) and the level and the site of the stop in
the names of the file (`failureOf` for a work). -/
def outcomeOf (ops : Array Op) (pub : Public) (resume : Resume) (changed : ByteArray)
    (offset : ℕ) : String × Option (ℕ × String) :=
  match verifyBytes (fun messages => verdictOf (resume.stop pub ops changed offset messages)
      (fun _ => .accept) messages) ops changed with
  | .malformed _ => ("malformed", none)
  | .trailing => ("trailing", none)
  | .verdict (.noWork i _) => ("noWork", some (failureOf ops i))
  | .verdict (.reject i site) => ("reject", (checkLevel ops i).map (·, site))
  | .verdict .accept => ("accept", none)
  | .verdict (.mismatch _) => ("mismatch", none)

/-- The class of the Rust verifier at the failed site of the file for the stage of a change: the
decoder of `z` is the check `terminal.z.decode` of the replay; the file names its stage `decode`
and its site `terminal.z`. -/
def stageClass : String → String
  | "parse" => "malformed"
  | "trailing_bytes" => "trailing"
  | "proof_of_work" | "fold_response" | "fold_challenges" => "noWork"
  | "equation" | "decode" => "reject"
  | _ => "?"

/-- The changes whose Lean site is not the field `failed` of the file, in the order of the file:
the index of the change, its stage, site and level, the failed site of the file and the Lean
site (level and site). Each is a reject at `fused.end` or `stage3.end` before the failed site. -/
def moved : List (String × List (ℕ × String × String × ℕ × String × ℕ × String × ℕ)) := [
  ("key20-opening-negative.json", [
    (4, "fold_response", "fused.round", 0, "fold_nonce", 1, "fused.end", 0),
    (15, "fold_response", "witness_evaluation", 0, "fold_nonce", 1, "fused.end", 0),
    (16, "fold_response", "witness_evaluation", 0, "fold_nonce", 1, "fused.end", 0),
    (23, "proof_of_work", "fused.round", 1, "tensor.partials", 2, "fused.end", 1),
    (34, "proof_of_work", "witness_evaluation", 1, "tensor.partials", 2, "fused.end", 1),
    (35, "proof_of_work", "witness_evaluation", 1, "tensor.partials", 2, "fused.end", 1),
    (67, "proof_of_work", "witness_evaluation", 2, "tensor.partials", 3, "fused.end", 2),
    (68, "equation", "witness_evaluation", 2, "tensor.opening", 3, "fused.end", 2),
    (75, "proof_of_work", "fused.round", 3, "tensor.partials", 4, "fused.end", 3),
    (104, "proof_of_work", "witness_evaluation", 3, "tensor.partials", 4, "fused.end", 3),
    (105, "proof_of_work", "witness_evaluation", 3, "tensor.partials", 4, "fused.end", 3),
    (111, "proof_of_work", "fused.round", 4, "tensor.partials", 5, "fused.end", 4),
    (147, "proof_of_work", "witness_evaluation", 4, "tensor.partials", 5, "fused.end", 4),
    (148, "proof_of_work", "witness_evaluation", 4, "tensor.partials", 5, "fused.end", 4),
    (155, "proof_of_work", "fused.round", 5, "tensor.partials", 6, "fused.end", 5),
    (192, "equation", "witness_evaluation", 5, "tensor.opening", 6, "fused.end", 5),
    (193, "equation", "witness_evaluation", 5, "tensor.opening", 6, "fused.end", 5)]),
  ("key30-opening-26-26-negative.json", [
    (0, "proof_of_work", "stage3.prefix_evaluation", 0, "row_coefficients", 1, "stage3.end", 0),
    (1, "fold_response", "stage3.prefix_evaluation", 0, "fold_nonce", 1, "stage3.end", 0)])]

/-- The changes of a negative file with nothing from the events, one task for each change. Each
change is a `reject` of the Rust verifier, and the replay rejects it too: at the failed site of
the file with the class of its stage (`parse` malformed, `trailing_bytes` trailing,
`proof_of_work`, `fold_response` and `fold_challenges` `noWork`, `equation` `reject` at the check,
`decode` `reject` at `terminal.z.decode`), or with a reject at `fused.end` or `stage3.end` before
the failed site. A change of `witness_evaluation` at a level with no work before `fused.end` must
reject at `fused.end` of its level, and a change of `stage3.prefix_evaluation` at `stage3.end`.
The changes at another site than `failed` are the list `moved`. -/
def gate4Negative (name : String) (ops : Array Op) (pub : Public) (proof : ByteArray) :
    IO Unit := do
  let start ← IO.monoMsNow
  let text ← IO.FS.readFile (directory / name)
  let changes ← IO.ofExcept do array (← Json.parse text) ["changes"]
  let some (messages, _) := (decode ops proof).toOption | throw <| IO.userError "decode"
  let resume := resumeOf pub ops proof messages
  let resumed ← IO.monoMsNow
  let tasks ← changes.mapM fun c => do
    let changed ← IO.ofExcept (change proof c)
    let offset := ((nat c ["offset"]).toOption.getD
      (min proof.size (proof.size - ((nat c ["truncate"]).toOption.getD 0))))
    return (c, Task.spawn fun _ => outcomeOf ops pub resume changed offset)
  let mut classes : List (String × String × ℕ) := []
  let mut found : List (ℕ × String × String × ℕ × String × ℕ × String × ℕ) := []
  for ((c, task), index) in tasks.toList.zipIdx do
    let stage ← IO.ofExcept (str c ["stage"])
    let result ← IO.ofExcept (str c ["result"])
    let failedLevel := (nat c ["failed", "level"]).toOption.getD 0
    let failedSite := (str c ["failed", "site"]).toOption.getD ""
    let level := (nat c ["level"]).toOption.getD 0
    let site := (str c ["site"]).toOption.getD ""
    let (klass, at_) := task.get
    let label := s!"{stage} ({site} {level})"
    if result != "reject" || klass == "accept" then
      fail s!"{name}: change {index} {label}: the file has {result}, the replay {klass}"
    let same := match at_ with
      | none => true
      | some (l, s) => l == failedLevel &&
          (s == failedSite || (s == "terminal.z.decode" && failedSite == "terminal.z"))
    let ended := (at_.map fun (_, s) => s == "fused.end" || s == "stage3.end").getD false
    let before := ((failedAt ops failedLevel failedSite).bind fun k =>
      (at_.bind fun (l, s) => ops.findIdx? fun
        | .check key _ => key == ⟨l, s, none⟩
        | _ => false).map fun j => decide (j < k)).getD false
    if same then
      unless klass == stageClass stage do
        fail s!"{name}: change {index} {label}: {klass} at the failed site, the stage {stage}"
    else
      unless klass == "reject" && ended && before do
        fail s!"{name}: change {index} {label}: {klass} at {at_}, the file {failedSite} {failedLevel}"
      let (l, s) := at_.getD (0, "")
      found := found ++ [(index, stage, site, level, failedSite, failedLevel, s, l)]
    let check := if site == "witness_evaluation" then some "fused.end"
      else if site == "stage3.prefix_evaluation" then some "stage3.end" else none
    let direct := ops.findIdx? fun | .exts key _ => key == ⟨level, site, none⟩ | _ => false
    let worked := (direct.bind fun k => check.bind fun site' =>
      ops.findIdx? (fun | .check key _ => key == ⟨level, site', none⟩ | _ => false) |>.map fun e =>
        (List.range (e - k)).any fun j => match ops[k + j]? with
          | some (.work _ _ bits) => bits != 0
          | _ => false)
    if let (some site', some false) := (check, worked) then
      unless at_ == some (level, site') do
        fail s!"{name}: change {index} {label} does not reject at {site'} {level}: {klass} {at_}"
    let key := if same then s!"{klass} at the failed site" else s!"{klass} at {at_.getD (0, "")}"
    classes := match classes.find? (fun (s, k, _) => s == stage && k == key) with
      | some _ => classes.map fun (s, k, n) => if s == stage && k == key then (s, k, n + 1)
          else (s, k, n)
      | none => classes ++ [(stage, key, 1)]
  IO.println s!"gate 4 {name}: {changes.size} changes, each a reject of the Rust verifier and of \
    the replay; the states pass {resumed - start} ms, the changes \
    {(← IO.monoMsNow) - resumed} ms; (stage, Lean, count):"
  for (s, k, n) in classes do IO.println s!"  {s}: {k}: {n}"
  IO.println s!"  {found.length} changes at another site than failed, each a reject:"
  for (i, stage, site, level, failedSite, failedLevel, at_, atLevel) in found do
    IO.println s!"    change {i}: {stage} of {site} at level {level}; failed {failedSite} \
      {failedLevel}; the replay rejects at {at_} {atLevel}"
  unless some found == (moved.find? (·.1 == name)).map (·.2) do
    fail s!"{name}: the changes at another site than failed are not the list moved"

end AkitaVectors
