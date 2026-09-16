import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiCCSPrefixCodeFold
import NightstreamFPrime.Export.Stage1.PiCCSNormSource

/-!
Retain the seventeen ordered norm sources after two Lean-derived challenges.
Each byte is an index into the proved 81-entry table. The reference check
compares every decoded field byte with two ordinary PrefixFold operations.
Rust messages and evaluations are not accepted by this executable.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiCCSPrefixReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Codec

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok value => pure value
  | .error error => throw (IO.userError error)

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def extensionValue (value : K) : Value :=
  .array [.atom value.c0.val, .atom value.c1.val]

private def decodeExtension (value : Lean.Json) : Except String K := do
  let words ← PiCCSInputCheck.decodeVector 2 PiCCSInputCheck.decodeField value
  return ⟨words.get 0, words.get 1⟩

private def decodePolynomial (value : Lean.Json) : Except String (FixedPolynomial K 9) := do
  let values ← PiCCSInputCheck.decodeVector 10 decodeExtension value
  return ⟨values.toList, Vector.length_toList⟩

private def readJson (path : System.FilePath) : IO Lean.Json := do
  checked (Lean.Json.parse (← IO.FS.readFile path))

private def decodeState (value : Lean.Json) : Except String Transcript.State := do
  (← value.getArr?).toList.mapM PiCCSInputCheck.decodeField

private structure Trace where
  coins : FiatShamir.PreSumcheck K Transcript.State productionShape
  state : Transcript.State
  claim : K
  challenges : List K

/-- Replay saved Lean rounds causally from the original public statement.
Every trace field is checked; a supplied state or challenge is never trusted. -/
private def readTrace (publicPath : System.FilePath) (roundPaths : List String) :
    IO Trace := do
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let mut trace : Trace := ⟨coins, coins.state, PiCCSPublicReplay.initialClaim input coins.gamma, []⟩
  for path in roundPaths do
    let saved ← checked ((← readJson path).getArr?)
    unless saved.size == 10 do throw (IO.userError "expected ten Lean round trace fields")
    unless (← checked (saved[0]!.getNat?)) == 1 do throw (IO.userError "unexpected Lean round schema")
    let alpha ← checked (PiCCSInputCheck.decodeVector cubeVariables decodeExtension saved[1]!)
    let gamma ← checked (decodeExtension saved[2]!)
    let before ← checked (decodeState saved[3]!)
    let polynomial ← checked (decodePolynomial saved[4]!)
    let claimedChallenge ← checked (decodeExtension saved[5]!)
    let after ← checked (decodeState saved[6]!)
    let initial ← checked (decodeExtension saved[7]!)
    let sum ← checked (decodeExtension saved[8]!)
    let nextClaim ← checked (decodeExtension saved[9]!)
    let endpoints := extensionOps.add
      (polynomial.evaluate extensionOps.toOps K.zero)
      (polynomial.evaluate extensionOps.toOps K.one)
    unless decide (alpha.toList = coins.alpha.coordinates ∧ gamma = coins.gamma ∧
        before = trace.state ∧ initial = trace.claim ∧ sum = endpoints ∧ endpoints = trace.claim) do
      throw (IO.userError s!"Lean round {trace.challenges.length} differs from the causal public transcript")
    if bound : trace.challenges.length < productionShape.cubeVariables then
      let index : Fin productionShape.cubeVariables := ⟨trace.challenges.length, bound⟩
      let absorbed := Transcript.piCcsOracle.transcript.absorbRound
        trace.state index polynomial.toMessage
      let (challenge, state) := Transcript.piCcsOracle.transcript.squeeze absorbed (.sumcheck index)
      let claim := polynomial.evaluate extensionOps.toOps challenge
      unless decide (claimedChallenge = challenge ∧ after = state ∧ nextClaim = claim) do
        throw (IO.userError s!"Lean round {index.val} challenge, state or claim differs")
      trace := { trace with state := state, claim := claim, challenges := trace.challenges ++ [challenge] }
    else throw (IO.userError "too many PiCCS rounds")
  return trace

private def groupCount : Nat := (PiCCSSourceImages.shape.carrierWidth + 3) / 4

private def metadata (trace : Trace) (first finish : Nat) : Value :=
  .array [.atom 1, .atom 2, .atom productionShape.sourceCount,
    .atom PiCCSSourceImages.shape.carrierWidth, .atom first, .atom finish,
    .array (trace.challenges.map extensionValue)]

private def sourceCode (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (column : Nat) : Fin 3 :=
  PiCCSNormSource.sourceCode (masks[column / ringDegree]?.getD #[]) source
    ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩

private def codeBytes (codes : Nat → Fin 3) (first count : Nat) : ByteArray :=
  ⟨(PiCCSPrefixCodeFold.quadCodes codes first count).map fun code => code.val.toUInt8⟩

private def sourcePath (directory : System.FilePath) (source : Nat) : System.FilePath :=
  directory / s!"source-{source}.bin"

private def validateRange (first finish chunkSize : Nat) : IO Unit := do
  unless first < finish && finish ≤ groupCount && chunkSize > 0 do
    throw (IO.userError "invalid norm group range or zero chunk size")

private def writeCodes (publicPath sourceInput roundZero roundOne directory : System.FilePath)
    (first finish chunkSize : Nat) : IO UInt32 := do
  validateRange first finish chunkSize
  unless !(← directory.pathExists) do throw (IO.userError "output directory already exists")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath [roundZero.toString, roundOne.toString]
  let (masks, records) ← SignedUnitSourceInput.read sourceInput PiCCSSourceImages.blockCount
  IO.FS.createDir directory
  report [("event", .str "norm_codes_sources_ready"), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let computeStarted ← IO.monoNanosNow
  for batch in [:(productionShape.sourceCount + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error Unit)) := #[]
    for source in [batch * workers:min productionShape.sourceCount ((batch + 1) * workers)] do
      if bound : source < productionShape.sourceCount then
        tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
          let output ← IO.FS.Handle.mk (sourcePath directory source) .write
          let mut next := first
          while next < finish do
            let count := min chunkSize (finish - next)
            let bytes := codeBytes (sourceCode masks ⟨source, bound⟩) next count
            unless bytes.size == count do throw (IO.userError "norm code size differs")
            output.write bytes
            next := next + count
          output.flush)
    for task in tasks do
      match ← IO.wait task with
      | .ok _ => pure ()
      | .error error => throw error
  IO.FS.writeFile (directory / "manifest.json") ((metadata trace first finish).render ++ "\n")
  report [("event", .str "norm_codes_complete"), ("first", Lean.toJson first),
    ("end", Lean.toJson finish), ("sources", Lean.toJson productionShape.sourceCount),
    ("payload_bytes", Lean.toJson ((finish - first) * productionShape.sourceCount)),
    ("workers", Lean.toJson workers), ("chunk_groups", Lean.toJson chunkSize),
    ("compute_write_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

/-- Canonical little-endian field words for the full-value comparison. -/
private def fieldBytes (values : Array K) : ByteArray := Id.run do
  let mut result := ByteArray.empty
  for value in values do
    for word in [value.c0.val.toUInt64, value.c1.val.toUInt64] do
      for byte in [:8] do
        result := result.push ((word >>> (8 * byte).toUInt64).toUInt8)
  return result


/-- The old scalar fold is evaluated on every possible signed four-scalar
input. This reference table does not use the new paired-table constructor. -/
private def referenceWords (firstChallenge secondChallenge : K) : Vector ByteArray 81 :=
  Vector.ofFn fun code =>
    let digit := fun index : Nat =>
      K.embed (PiCCSNormCache.signedValue
        ⟨(code.val / 3 ^ (3 - index)) % 3, Nat.mod_lt _ (by decide)⟩)
    let original := #[digit 0, digit 1, digit 2, digit 3]
    fieldBytes (PrefixFold.foldOne extensionOps
      (PrefixFold.foldOne extensionOps original firstChallenge) secondChallenge)

/-- Independently classify the existing scalar decoder's result. Its only
possible results are zero, one and minus one, by its definition. -/
private def scalarIndex (value : F) : Fin 3 :=
  if value.val == 0 then ⟨1, by decide⟩
  else if value.val == 1 then ⟨2, by decide⟩ else ⟨0, by decide⟩

private def originalQuadCode (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (group : Nat) : Fin 81 :=
  let digit := fun offset : Nat =>
    let column := 4 * group + offset
    scalarIndex (SignedUnitSourceInput.scalar (masks[column / ringDegree]?.getD #[])
      source ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩)
  let a := digit 0
  let b := digit 1
  let c := digit 2
  let d := digit 3
  ⟨27 * a.val + 9 * b.val + 3 * c.val + d.val, by
    have := a.isLt
    have := b.isLt
    have := c.isLt
    have := d.isLt
    omega⟩

private def readMetadata (directory : System.FilePath) (trace : Trace)
    (first finish : Nat) : IO Unit := do
  let actual ← readJson (directory / "manifest.json")
  let expected ← checked (Lean.Json.parse (metadata trace first finish).render)
  unless actual == expected do throw (IO.userError "norm code manifest differs from the selected source range and challenges")
  let entries ← directory.readDir
  unless entries.size == productionShape.sourceCount + 1 do
    throw (IO.userError "norm code directory has missing or extra files")

/-- Check every stored code and every decoded field byte against the original
scalar reader and two existing folds. Chunks start at absolute four-scalar
boundaries; no scalar is lost where a 54-lane source block ends. -/
private def checkCodes (publicPath sourceInput roundZero roundOne directory : System.FilePath)
    (first finish chunkSize : Nat) (direct : Bool := false) : IO UInt32 := do
  validateRange first finish chunkSize
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath [roundZero.toString, roundOne.toString]
  readMetadata directory trace first finish
  let (masks, records) ← SignedUnitSourceInput.read sourceInput PiCCSSourceImages.blockCount
  let firstChallenge := trace.challenges[0]?.getD K.zero
  let secondChallenge := trace.challenges[1]?.getD K.zero
  let table := PiCCSPrefixCodeFold.pairedTable
    (PiCCSPrefixNorm.values firstChallenge) secondChallenge
  let actualWords := table.map (fun value => fieldBytes #[value])
  let expectedWords := referenceWords firstChallenge secondChallenge
  for index in [:81] do
    if bound : index < 81 then
      unless actualWords.get ⟨index, bound⟩ == expectedWords.get ⟨index, bound⟩ do
        throw (IO.userError "norm table differs from two existing folds")
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let computeStarted ← IO.monoNanosNow
  report [("event", .str "norm_check_sources_ready"), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson (computeStarted - started))]
  for batch in [:(productionShape.sourceCount + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error Unit)) := #[]
    for source in [batch * workers:min productionShape.sourceCount ((batch + 1) * workers)] do
      if bound : source < productionShape.sourceCount then
        tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
          let input ← IO.FS.Handle.mk (sourcePath directory source) .read
          let mut next := first
          while next < finish do
            let count := min chunkSize (finish - next)
            let bytes ← input.read count.toUSize
            unless bytes.size == count do throw (IO.userError s!"truncated norm source {source}")
            let mut codes : Array (Fin 81) := #[]
            for byte in bytes do
              if codeBound : byte.toNat < 81 then codes := codes.push ⟨byte.toNat, codeBound⟩
              else throw (IO.userError s!"invalid norm code in source {source}")
            if direct then
              let decoded := PiCCSPrefixCodeFold.decode table codes
              let original := Array.ofFn fun entry : Fin (4 * count) =>
                let column := 4 * next + entry.val
                K.embed (SignedUnitSourceInput.scalar (masks[column / ringDegree]?.getD #[])
                  ⟨source, bound⟩ ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩)
              let reference := PrefixFold.foldOne extensionOps
                (PrefixFold.foldOne extensionOps original firstChallenge) secondChallenge
              unless fieldBytes decoded == fieldBytes reference do
                throw (IO.userError s!"norm field bytes differ: source={source} first={next} end={next + count}")
            else
              for index in [:count] do
                let actual := actualWords.get (codes[index]?.getD ⟨40, by decide⟩)
                let expected := expectedWords.get (originalQuadCode masks ⟨source, bound⟩ (next + index))
                unless actual == expected do
                  throw (IO.userError s!"norm field bytes differ: source={source} group={next + index}")
            next := next + count
          unless (← input.read 1).isEmpty do throw (IO.userError s!"extra bytes in norm source {source}"))
    for task in tasks do
      match ← IO.wait task with
      | .ok _ => pure ()
      | .error error => throw error
  report [("event", .str "norm_codes_byte_match"), ("first", Lean.toJson first),
    ("end", Lean.toJson finish), ("sources", Lean.toJson productionShape.sourceCount),
    ("values", Lean.toJson ((finish - first) * productionShape.sourceCount)),
    ("compared_bytes", Lean.toJson ((finish - first) * productionShape.sourceCount * 16)),
    ("direct", Lean.toJson direct),
    ("compute_read_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

end NightstreamFPrime.Export.PiCCSPrefixReplay

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | [mode, publicPath, sourceInput, roundZero, roundOne, directory, first, finish, chunkSize] =>
      match first.toNat?, finish.toNat?, chunkSize.toNat? with
      | some first, some finish, some chunkSize =>
          if mode == "norm-codes" then
            NightstreamFPrime.Export.PiCCSPrefixReplay.writeCodes
              publicPath sourceInput roundZero roundOne directory first finish chunkSize
          else if mode == "check-norm-codes" || mode == "check-norm-codes-direct" then
            NightstreamFPrime.Export.PiCCSPrefixReplay.checkCodes
              publicPath sourceInput roundZero roundOne directory first finish chunkSize
              (mode == "check-norm-codes-direct")
          else throw (IO.userError "expected norm-codes or check-norm-codes")
      | _, _, _ => throw (IO.userError "expected natural-number range and chunk size")
  | _ =>
      IO.eprintln "usage: replayPiCCSPrefix norm-codes|check-norm-codes[-direct] <public> <original-sources> <Lean-Q0> <Lean-Q1> <directory> <first-group> <end-group> <chunk-groups>"
      return 2
