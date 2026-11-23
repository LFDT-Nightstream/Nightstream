import Lean.Data.Json
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules
import NightstreamFPrime.Spec.Compression.Akita.Replay.Wire
import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Replay.Plan

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
  holds in the order of the `check` events of the file, and the stage functions with a theorem
  against the specification (`Replay/Range.lean`, `Replay/Tensor.lean`) accept on the values of
  the replay.
- Gate 4. FIPS 202 vectors; the root digests; the work bits of the plan; the replay of the
  transcript from the proof and the public inputs only (`gate4`): each challenge, work, absorb and
  the values `opening_target`, `relation_claim` and `opening_claim` of each fold; the changes of the
  negative files with nothing from the events.

Lake does not track the files, so `scripts/validate.sh` runs this file again. -/

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
  ["key20-opening.json", "key20-opening-retry.json", "key30-opening-26-28.json",
    "key30-opening-27-28.json"]

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

/-- The terminal numbers of the schedule file of a vector: `z_payload_bytes`, `e_field_elems` and
`t_field_elems` of `response_shape`. -/
def terminalNumbers (v : Vector) : IO (ℕ × ℕ × ℕ) := do
  let text ← IO.FS.readFile ("../.." / System.FilePath.mk v.schedule)
  IO.ofExcept do
    let rows ← array (← Json.parse text) ["rows"]
    let row ← match v.schedule, v.shape with
      | "crates/ns-compress/artifacts/key30.aks", (_, 26, _) => pure rows[1]!
      | _, _ => pure rows[0]!
    let layout ← array row ["schedule", "terminal", "response_shape", "layout", "groups"]
    let group := layout[0]!
    return (← nat group ["z_payload_bytes"], ← nat group ["e_field_elems"],
      ← nat group ["t_field_elems"])

/-- The layout of the schedule row of a vector, from the Lean numbers and `zMax` of the file. The
file has `e` and `t` of the terminal, and they are the counts of the layout. -/
def layoutWith (order : ∀ n, Equiv.Perm (Fin n)) (v : Vector) : IO Layout := do
  let (zMax, e, t) ← terminalNumbers v
  let layout ← match v.schedule, v.shape with
    | "crates/ns-compress/artifacts/key20.aks", _ => pure (layout20With order zMax)
    | "crates/ns-compress/artifacts/key30.aks", (_, 27, _) =>
        pure (layout30With order Key30.w27 zMax)
    | "crates/ns-compress/artifacts/key30.aks", (_, 26, _) =>
        pure (layout30With order Key30.w26 zMax)
    | schedule, _ => throw <| IO.userError s!"no Lean row for {schedule}"
  unless layout.terminal.e = e do fail s!"{v.name}: e has {e} elements, the layout {layout.terminal.e}"
  let last := (layout.folds.getLast?.map (·.witness)).getD 0
  unless last = t do fail s!"{v.name}: t has {t} elements, the layout {last}"
  return layout

def layoutOf (v : Vector) : IO Layout := layoutWith groupOrder v

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

/-- The stage functions that have a theorem against the specification, on the values of the
replay: `rangeRun` (`rangeRun_runs`), `treeRun` with `normRun` (`treeRun_runs`, `normRun_runs`) and
`tensorRun` (`tensorRun_runs`) accept, with the point and the values of the replay. -/
def stagesGate (v : Vector) (d : Derived) (layout : Layout) : IO Unit := do
  let get (level : ℕ) (site : String) : Array Ext := (d.recorded.get? ⟨level, site, none⟩).getD #[]
  let one (level : ℕ) (site : String) : Ext := (get level site).getD 0 Ext.zero
  let rounds (level : ℕ) (site : String) : List (List Ext × Ext) :=
    ((d.rounds.find? (·.1 == ⟨level, site, none⟩)).map (·.2)).getD []
  let mut ranges := 0
  let mut norms := 0
  let mut tensors := 0
  for (F, level) in layout.folds.zipIdx do
    let stages := (productStages F.logBasis).zipIdx.map fun ((arity, _), k) =>
      ({ arity, rounds := rounds level s!"range.{k}", children := (get level s!"range.{k}.children").toList
         gamma := one level s!"range.{k}.batch" } : Product)
    let tau0 := get level "tau0"
    let point := tau0.toList.rotate (tau0.size - F.low)
    let image := one level "range.image"
    if F.euclidean then
      let energy := (d.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0
      let virtual := one level "norm.virtual_evaluations"
      let out := treeRun (normRun (leafDegree F.logBasis) F.limit energy (one level "norm.merge")
        (rounds level "norm.sumcheck") virtual image) stages 1 (fun _ => Ext.one) point Ext.zero
      unless out == some ((get level "norm.sumcheck.point").toList, image, virtual) do
        fail s!"{v.name}: treeRun with normRun at level {level}"
      norms := norms + 1
    else
      let out := rangeRun (2 ^ F.logBasis) stages (rounds level "range.leaf") image point
      unless out == some ((get level "range.leaf.point").toList, image) do
        fail s!"{v.name}: rangeRun at level {level}"
      ranges := ranges + 1
  let tensorLevels := (layout.folds.zipIdx.filterMap fun (F, level) =>
    F.trace.map fun (count, _) => (level, count)) ++ [(layout.folds.length, layout.terminal.rounds)]
  for (level, count) in tensorLevels do
    let partials := get level "tensor.partials"
    let out := tensorRun count (get (level - 1) "fused.point").toList
      (one (level - 1) "witness_evaluation") (partials.getD 0 Ext.zero) (partials.getD 1 Ext.zero)
      (one level "tensor.eta") (rounds level "tensor.sumcheck")
    unless (out.map fun e => (e.point, e.claim)) ==
        some ((get level "tensor.sumcheck.point").toList, one level "tensor.final_claims") do
      fail s!"{v.name}: tensorRun at level {level}"
    tensors := tensors + 1
  IO.println s!"gate 7 {v.name}: rangeRun accepts at {ranges} levels, treeRun with normRun at \
    {norms} levels, tensorRun at {tensors} levels, each with the point and the values of the replay"

/-- Gate 4 on a positive file: the replay from the bytes of the proof and the public inputs only.
Each `challenge` event of the file equals the derived challenge, each proof of work passes, each
absorb equals the absorb of the file, and each value `opening_target`, `relation_claim` and
`opening_claim` of the file equals the value of the replay. The fold challenges are squeezed with
the length of the layout and not decoded (work item 8). -/
def gate4 (v : Vector) (o : Oracles) (ops : Array Op) (pub : Public) : IO Unit := do
  let some (messages, _) := (decode ops v.proof).toOption | throw <| IO.userError "decode"
  let (stop, d) := replay pub ops v.proof messages
  if let some s := stop then fail s!"{v.name}: the replay stops: {repr s}"
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
  let (have_, lack) := checkEvents.partition fun e => !e.site.startsWith "terminal."
  unless d.checks.size = have_.size ∧
      (d.checks.zip have_).all (fun (key, e) => key.level == e.level && key.site == e.site) do
    fail s!"{v.name}: the checks of the replay are not the check events of the file"
  IO.println s!"gate 7 {v.name}: {d.checks.size} equations hold in the order of the check events \
    (tensor.opening, tensor.end, range.<k>.end, range.leaf.end, norm.integer, norm.end); Lean has \
    no equation for {lack.size} check events ({(lack.map (·.site)).toList.eraseDups})"
  stagesGate v d pub.layout
  let fused := (d.absorbs.filter (·.1.site == "fused.claim")).size
  IO.println s!"gate 4 {v.name}: {challengeEvents.size} challenge events equal, {d.works.size} \
    works pass, {d.absorbs.size} absorbs equal ({fused} fused.claim), {d.values.size} values \
    equal ({names}), {d.squeezed} bytes squeezed; the replay has no input from the events"

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
values of the file that Lean does not derive from the schedule and the seed yet. -/
def publicOf (v : Vector) (o : Oracles) (layout : Layout) : IO Public := do
  let slot := (o.events.find? fun e => e.op == "absorb" && e.site == "stage3.slot").map
    fun e => (str e.json ["bytes"]).toOption.getD ""
  let setupRows ← match o.events.find? (·.site == "setup_prefix.commitment_rows") with
    | some e => IO.ofExcept do (← array e.json ["words"]).mapM element
    | none => pure #[]
  return { layout, plan := v.plan, session := v.session, descriptor := v.descriptor
           groups := v.groups.map fun g => ⟨g.commitment, g.point, g.value⟩
           slot := ← IO.ofExcept (hex (slot.getD "")), setupRows }

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

/-- The outcome of `verifyBytes` with the verdict of the replay on a changed proof, against the
stage and the `failed` site of the file. Lean has no equation of the fused sum-check, of Stage 3
and of the terminal yet, so `checks` rejects each proof whose works and equations pass. -/
def classify (ops : Array Op) (pub : Public) (changed : ByteArray) (failedLevel : ℕ)
    (failedSite : String) : String :=
  let checks : Array Message → Verdict := fun _ => .reject ops.size "no equation check in Lean"
  match verifyBytes (verdict pub ops changed checks) ops changed with
  | .malformed _ => "malformed"
  | .trailing => "trailing"
  | .verdict (.noWork i _) =>
      let (level, site) := failureOf ops i
      if level = failedLevel ∧ site = failedSite then "noWork at the failed site"
      else s!"noWork at {site} {level}, the file {failedSite} {failedLevel}"
  | .verdict (.reject i site) => match ops[i]? with
      | some (.check key _) =>
          if key.level = failedLevel ∧ site = failedSite then "reject at the failed check"
          else s!"reject at {site} {key.level}, the file {failedSite} {failedLevel}"
      | _ => s!"each work and check of Lean passes, the file {failedSite} {failedLevel}"
  | .verdict .accept => "accept"

/-- The changes of a negative file with nothing from the events, one task for each change: the
count of each pair of a stage and a Lean outcome. A change of the stage `parse`, `trailing_bytes`,
`proof_of_work` or `fold_response`, and a change of the stage `equation` whose check is not one of
the terminal, must have the outcome class of the file at the failed site of the file. -/
def gate4Negative (name : String) (ops : Array Op) (pub : Public) (proof : ByteArray) :
    IO Unit := do
  let text ← IO.FS.readFile (directory / name)
  let changes ← IO.ofExcept do array (← Json.parse text) ["changes"]
  let tasks ← changes.mapM fun c => do
    let stage ← IO.ofExcept (str c ["stage"])
    let changed ← IO.ofExcept (change proof c)
    let failedLevel := (nat c ["failed", "level"]).toOption.getD 0
    let failedSite := (str c ["failed", "site"]).toOption.getD ""
    return (stage, failedSite, Task.spawn fun _ => classify ops pub changed failedLevel failedSite)
  let mut classes : List (String × String × ℕ) := []
  for (stage, failedSite, task) in tasks do
    let outcome := task.get
    let expected := match stage with
      | "parse" => some "malformed"
      | "trailing_bytes" => some "trailing"
      | "proof_of_work" | "fold_response" => some "noWork at the failed site"
      | "equation" => if failedSite.startsWith "terminal." then none
          else some "reject at the failed check"
      | _ => none
    if let some class_ := expected then
      unless outcome == class_ do fail s!"{name}: a change of the stage {stage} is: {outcome}"
    classes := match classes.find? (fun (s, k, _) => s == stage && k == outcome) with
      | some _ => classes.map fun (s, k, n) => if s == stage && k == outcome then (s, k, n + 1)
          else (s, k, n)
      | none => classes ++ [(stage, outcome, 1)]
  IO.println s!"gate 4 {name}: {changes.size} changes; (stage, Lean, count):"
  for (s, k, n) in classes do IO.println s!"  {s}: {k}: {n}"

#eval show IO Unit from do
  let start ← IO.monoMsNow
  for name in positives do
    let (v, o) ← load name
    gate1 v o
    let levels ← gate2 v o
    IO.println s!"gate 2 {v.name}: norm.square = virtual² and norm.end on {levels} levels"
    let ops ← gate3 v o
    if name == "key20-opening.json" then
      discard <| gate3Negative "key20-opening-negative.json" ops v.proof
    if name == "key30-opening-26-28.json" then
      discard <| gate3Negative "key30-opening-26-28-negative.json" ops v.proof
    let layout ← layoutOf v
    let pub ← publicOf v o layout
    digests v
    planGate v layout
    gate4 v o ops pub
    orderGate v o pub
    if name == "key20-opening.json" then
      gate4Negative "key20-opening-negative.json" ops pub v.proof
    if name == "key30-opening-26-28.json" then
      gate4Negative "key30-opening-26-28-negative.json" ops pub v.proof
  IO.println s!"time {(← IO.monoMsNow) - start} ms"

end AkitaVectors
