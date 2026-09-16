import NightstreamFPrime.Export.PiCCSOriginalMerge
import NightstreamFPrime.Export.PiCCSOriginalEvaluation
import NightstreamFPrime.Export.PiCCSPrefixFiles
import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixPolynomial
import NightstreamFPrime.Export.Stage1.PiCCSCarriedMoments
import NightstreamFPrime.Export.Stage1.PiCCSTensorWeights
import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiCCSPrefixCodeFold
import NightstreamFPrime.Export.Stage1.PiCCSNormSource

/-!
Replay PiCCS contributions from retained ordered fresh, norm and carried
prefixes. The initial norm bytes index the proved 81-entry table; later
canonical field files use the existing PrefixFold operations.
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
  input : PiCCSPublicReplay.Input
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
  let mut trace : Trace := ⟨input, coins, coins.state, PiCCSPublicReplay.initialClaim input coins.gamma, []⟩
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

/-- Form the next inner norm cubic from the complete retained source files.
Each source keeps its own cubic and gamma exponent. Pairing uses absolute
indices; only the true missing final endpoint is zero. -/
private def normAfterTwo (publicPath roundZero roundOne directory outputPath : System.FilePath)
    (first finish chunkPairs : Nat) (direct : Bool) : IO UInt32 := do
  let pairs := (groupCount + 1) / 2
  unless first < finish && finish ≤ pairs && chunkPairs > 0 do
    throw (IO.userError "invalid third-round norm pair range or zero chunk size")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath [roundZero.toString, roundOne.toString]
  readMetadata directory trace 0 groupCount
  let table := PiCCSPrefixCodeFold.pairedTable
    (PiCCSPrefixNorm.values (trace.challenges[0]?.getD K.zero))
    (trace.challenges[1]?.getD K.zero)
  let tail := trace.coins.alpha.coordinates.drop 3
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let computeStarted ← IO.monoNanosNow
  let mut total := FixedPolynomial.zero extensionOps.toOps 3
  for batch in [:(productionShape.sourceCount + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error (FixedPolynomial K 3))) := #[]
    for source in [batch * workers:min productionShape.sourceCount ((batch + 1) * workers)] do
      tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
        let input ← IO.FS.Handle.mk (sourcePath directory source) .read
        let mut position := 0
        let mut buckets := PiCCSPrefixNormBuckets.empty 81
        let mut reference := FixedPolynomial.zero extensionOps.toOps 3
        while position < pairs do
          let count := min chunkPairs (pairs - position)
          let expectedBytes := min (2 * count) (groupCount - 2 * position)
          let bytes ← input.read expectedBytes.toUSize
          unless bytes.size == expectedBytes do
            throw (IO.userError s!"truncated norm source {source}")
          let lower := max first position
          let upper := min finish (position + count)
          let mut codes : Array (Fin 81) := #[]
          for byte in bytes do
            if bound : byte.toNat < 81 then
              if lower < upper then codes := codes.push ⟨byte.toNat, bound⟩
            else throw (IO.userError s!"invalid norm code in source {source}")
          if lower < upper then
            let low := fun index => codes[2 * (index - position)]?.getD ⟨40, by decide⟩
            let high := fun index => codes[2 * (index - position) + 1]?.getD ⟨40, by decide⟩
            if direct then
              reference := FixedPolynomial.add extensionOps.toOps reference
                (PiCCSPolynomialRange.range extensionOps lower (upper - lower) fun index =>
                  FixedPolynomial.scale extensionOps.toOps (weight index)
                    (PiCCSFirstRoundPair.normPair extensionOps
                      (table.get (low index)) (table.get (high index))))
            else
              buckets := PiCCSPrefixNormBuckets.accumulate buckets low high weight lower (upper - lower)
          position := position + count
        unless (← input.read 1).isEmpty do throw (IO.userError s!"extra bytes in norm source {source}")
        let value := if direct then reference else PiCCSPrefixNormBuckets.finish table buckets
        return FixedPolynomial.scale extensionOps.toOps
          (TargetPolynomial.power extensionOps.toOps trace.coins.gamma source) value)
    for task in tasks do
      let value ← match ← IO.wait task with
        | .ok value => pure value
        | .error error => throw error
      total := FixedPolynomial.add extensionOps.toOps total value
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom 2, .atom first, .atom finish,
    .array (trace.challenges.map extensionValue),
    .array (total.coefficients.map extensionValue)]).render ++ "\n")
  report [("event", .str "third_norm_complete"), ("first", Lean.toJson first),
    ("end", Lean.toJson finish), ("sources", Lean.toJson productionShape.sourceCount),
    ("validated_values", Lean.toJson (groupCount * productionShape.sourceCount)),
    ("direct", Lean.toJson direct), ("chunk_pairs", Lean.toJson chunkPairs),
    ("compute_read_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def matrixRows (_ : Unit) : Nat :=
  (PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application).rowCount

private def foldedCount (count consumed : Nat) : Nat :=
  (count + 2 ^ consumed - 1) / 2 ^ consumed

private def checkDepth (trace : Trace) : IO Unit := do
  unless 0 < trace.challenges.length && trace.challenges.length < cubeVariables do
    throw (IO.userError "prefix computation requires a nonterminal challenge prefix")

private def binaryRow (input : IO.FS.Handle) (width : Nat) : IO (Array K) := do
  let bytes ← input.read (width * 16).toUSize
  unless bytes.size == width * 16 do throw (IO.userError "truncated binary prefix row")
  checked (PiCCSPrefixFiles.decodeFields bytes)

private def freshVector (values : Array K) : IO (Vector K Spec.ProductionRelation.matrixCount) := do
  if bound : values.size = Spec.ProductionRelation.matrixCount then return ⟨values, bound⟩
  else throw (IO.userError "fresh binary prefix row has wrong width")

private def freshFromPrefix (publicPath directory outputPath : System.FilePath)
    (first finish : Nat) (roundPaths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath roundPaths
  checkDepth trace
  let consumed := trace.challenges.length
  let count := foldedCount (matrixRows ()) consumed
  unless first < finish && finish ≤ (count + 1) / 2 do throw (IO.userError "invalid fresh pair range")
  let chunks ← PiCCSPrefixFiles.read directory 2 Spec.ProductionRelation.matrixCount count trace.challenges
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps trace.coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps trace.coins.gamma powers
  let weights := PiCCSTensorWeights.prepare extensionOps (trace.coins.alpha.coordinates.drop (consumed + 1))
  let contribution (index : Nat) (low high : Vector K Spec.ProductionRelation.matrixCount) :
      IO (FixedPolynomial K 9) := do
    if first ≤ index && index < finish then
      if inside : index < 2 ^ (cubeVariables - consumed - 1) then
        let polynomial := PiCCSFreshPrefixPolynomial.contribution
          (PiCCSPublicReplay.verifierInput trace.input) trace.coins.alpha trace.challenges
          (NumericBooleanDomain.vertex (cubeVariables - consumed - 1) ⟨index, inside⟩)
          power weights low high
        return PiCCSPublicReplay.degree_eq trace.input ▸ polynomial
      else throw (IO.userError "fresh pair exceeds the remaining Boolean domain")
    else return FixedPolynomial.zero extensionOps.toOps 9
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut total := FixedPolynomial.zero extensionOps.toOps 9
  for batch in [:(chunks.size + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error (FixedPolynomial K 9))) := #[]
    for part in [batch * workers:min chunks.size ((batch + 1) * workers)] do
      if bound : part < chunks.size then
        let chunk := chunks[part]'bound
        let nextValues := (chunks[part + 1]?).map PiCCSPrefixFiles.Chunk.firstValues
        tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
          let input ← IO.FS.Handle.mk chunk.path .read
          let mut previous : Option (Vector K Spec.ProductionRelation.matrixCount) := none
          let mut subtotal := FixedPolynomial.zero extensionOps.toOps 9
          for index in [chunk.first:chunk.finish] do
            let values ← binaryRow input Spec.ProductionRelation.matrixCount
            if index == chunk.first then
              unless decide (values = chunk.firstValues) do throw (IO.userError "fresh first row changed")
            let row ← freshVector values
            if index % 2 == 0 then previous := some row
            else if let some low := previous then
              subtotal := FixedPolynomial.add extensionOps.toOps subtotal
                (← contribution (index / 2) low row)
              previous := none
          unless (← input.read 1).isEmpty do throw (IO.userError "extra fresh prefix bytes")
          if let some low := previous then
            let high ← freshVector (nextValues.getD
              (Array.replicate Spec.ProductionRelation.matrixCount K.zero))
            subtotal := FixedPolynomial.add extensionOps.toOps subtotal
              (← contribution ((chunk.finish - 1) / 2) low high)
          return subtotal)
    for task in tasks do
      let value ← match ← IO.wait task with
        | .ok value => pure value
        | .error error => throw error
      total := FixedPolynomial.add extensionOps.toOps total value
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom consumed, .atom first, .atom finish,
    .array (trace.challenges.map extensionValue), .array (total.coefficients.map extensionValue)]).render ++ "\n")
  report [("event", .str "prefix_fresh_complete"), ("consumed", Lean.toJson consumed),
    ("first", Lean.toJson first), ("end", Lean.toJson finish),
    ("validated_rows", Lean.toJson count), ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def carriedFromPrefix (kind : Nat) (publicPath directory outputPath : System.FilePath)
    (roundPaths : List String) : IO UInt32 := do
  unless kind ≤ 1 do throw (IO.userError "expected Pad or matrix family")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath roundPaths
  checkDepth trace
  let consumed := trace.challenges.length
  let count := foldedCount (if kind == 0 then PiCCSSourceImages.shape.carrierWidth else matrixRows ()) consumed
  let chunks ← PiCCSPrefixFiles.read directory kind 1 count trace.challenges
  let tail := (PiCCSPublicReplay.verifierInput trace.input).priorPoint.coordinates.drop (consumed + 1)
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut low := K.zero
  let mut high := K.zero
  for batch in [:(chunks.size + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error (K × K))) := #[]
    for part in [batch * workers:min chunks.size ((batch + 1) * workers)] do
      if bound : part < chunks.size then
        let chunk := chunks[part]'bound
        tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
          let input ← IO.FS.Handle.mk chunk.path .read
          let mut lo := K.zero
          let mut hi := K.zero
          for index in [chunk.first:chunk.finish] do
            let values ← binaryRow input 1
            if index == chunk.first then
              unless decide (values = chunk.firstValues) do throw (IO.userError "carried first row changed")
            let value := extensionOps.mul (weight (index / 2)) (values.getD 0 K.zero)
            if index % 2 == 0 then lo := extensionOps.add lo value
            else hi := extensionOps.add hi value
          unless (← input.read 1).isEmpty do throw (IO.userError "extra carried prefix bytes")
          return (lo, hi))
    for task in tasks do
      let value ← match ← IO.wait task with
        | .ok value => pure value
        | .error error => throw error
      low := extensionOps.add low value.1
      high := extensionOps.add high value.2
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom consumed, .atom kind, .atom 0, .atom count,
    .array (trace.challenges.map extensionValue), extensionValue low, extensionValue high]).render ++ "\n")
  report [("event", .str "prefix_carried_complete"), ("kind", Lean.toJson kind),
    ("consumed", Lean.toJson consumed), ("values", Lean.toJson count),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def prefixPolynomial (trace : Trace) (degree finish : Nat) (path : System.FilePath) :
    IO (FixedPolynomial K degree) := do
  let value ← readJson path
  checked (do
    match (← value.getArr?).toList with
    | [schema, consumed, first, endValue, coins, coefficients] =>
        unless (← schema.getNat?) == 1 && (← consumed.getNat?) == trace.challenges.length &&
            (← first.getNat?) == 0 && (← endValue.getNat?) == finish do
          throw "prefix polynomial does not cover the selected complete range"
        let coins ← PiCCSInputCheck.decodeVector trace.challenges.length decodeExtension coins
        unless decide (coins.toList = trace.challenges) do throw "prefix polynomial challenges differ"
        let coefficients ← PiCCSInputCheck.decodeVector (degree + 1) decodeExtension coefficients
        return ⟨coefficients.toList, Vector.length_toList⟩
    | _ => throw "expected six prefix polynomial fields")

private def prefixMoment (trace : Trace) (kind finish : Nat) (path : System.FilePath) :
    IO (K × K) := do
  let value ← readJson path
  checked (do
    match (← value.getArr?).toList with
    | [schema, consumed, family, first, endValue, coins, low, high] =>
        unless (← schema.getNat?) == 1 && (← consumed.getNat?) == trace.challenges.length &&
            (← family.getNat?) == kind && (← first.getNat?) == 0 && (← endValue.getNat?) == finish do
          throw "prefix moments do not cover the selected complete family"
        let coins ← PiCCSInputCheck.decodeVector trace.challenges.length decodeExtension coins
        unless decide (coins.toList = trace.challenges) do throw "prefix moment challenges differ"
        return (← decodeExtension low, ← decodeExtension high)
    | _ => throw "expected eight prefix moment fields")

private def composePrefix (publicPath freshPath normPath matrixPath padPath outputPath : System.FilePath)
    (roundPaths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let trace ← readTrace publicPath roundPaths
  checkDepth trace
  let consumed := trace.challenges.length
  let fresh ← prefixPolynomial trace 9 (foldedCount (matrixRows ()) (consumed + 1)) freshPath
  let norm ← prefixPolynomial trace 3
    (foldedCount PiCCSSourceImages.shape.carrierWidth (consumed + 1)) normPath
  let matrix ← prefixMoment trace 1 (foldedCount (matrixRows ()) consumed) matrixPath
  let pad ← prefixMoment trace 0 (foldedCount PiCCSSourceImages.shape.carrierWidth consumed) padPath
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps trace.coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps trace.coins.gamma powers
  let head := fun target : CubePoint K cubeVariables =>
    FixedPolynomial.scale extensionOps.toOps
      (PiCCSPrefixSelector.consumedFactor extensionOps trace.challenges target)
      (PiCCSCarriedMoments.headSelector extensionOps (PiCCSPrefixSelector.dropPoint target consumed))
  let normTerm : FixedPolynomial K 9 :=
    FixedPolynomial.scale extensionOps.toOps (power productionShape.constraintOffset)
      (FixedPolynomial.scale extensionOps.toOps (power productionShape.freshCount)
        (FixedPolynomial.widen extensionOps.toOps (by decide : 4 ≤ 9)
          (FixedPolynomial.mul extensionOps.toOps (head trace.coins.alpha) norm)))
  let carried : FixedPolynomial K 9 :=
    PiCCSCarriedMoments.carriedPair extensionOps (by decide : 2 ≤ 9)
      (head (PiCCSPublicReplay.verifierInput trace.input).priorPoint)
      (power productionShape.matrixEvaluationOffset) pad.1 pad.2 matrix.1 matrix.2
  let polynomial := FixedPolynomial.add extensionOps.toOps carried
    (FixedPolynomial.add extensionOps.toOps fresh normTerm)
  let endpoints := extensionOps.add (polynomial.evaluate extensionOps.toOps K.zero)
    (polynomial.evaluate extensionOps.toOps K.one)
  unless decide (endpoints = trace.claim) do throw (IO.userError "prefix round endpoint sum differs from prior claim")
  if bound : consumed < productionShape.cubeVariables then
    let index : Fin productionShape.cubeVariables := ⟨consumed, bound⟩
    let absorbed := Transcript.piCcsOracle.transcript.absorbRound trace.state index polynomial.toMessage
    let (challenge, state) := Transcript.piCcsOracle.transcript.squeeze absorbed (.sumcheck index)
    let stateValue := fun current : Transcript.State => Value.array (current.map (fun value => .atom value.val))
    IO.FS.writeFile outputPath ((Value.array [.atom 1,
      .array (trace.coins.alpha.coordinates.map extensionValue), extensionValue trace.coins.gamma,
      stateValue trace.state, .array (polynomial.coefficients.map extensionValue),
      extensionValue challenge, stateValue state, extensionValue trace.claim, extensionValue endpoints,
      extensionValue (polynomial.evaluate extensionOps.toOps challenge)]).render ++ "\n")
    report [("event", .str "prefix_round_composed"), ("round", Lean.toJson consumed),
      ("coefficients", Lean.toJson polynomial.coefficients.length)]
    return 0
  else throw (IO.userError "no remaining PiCCS round")

private def advanceFirst (kind : Nat) (publicPath roundZero roundOne outputDirectory : System.FilePath)
    (directories : List String) : IO UInt32 := do
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath [roundZero.toString, roundOne.toString]
  match trace.challenges with
  | [first, second] =>
      PiCCSPrefixFiles.foldLegacy kind (directories.map System.FilePath.mk) outputDirectory first second
      report [("event", .str "first_prefix_advanced"), ("kind", Lean.toJson kind),
        ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
      return 0
  | _ => throw (IO.userError "expected exactly two Lean-derived challenges")

private def normDirectory (directory : System.FilePath) (source : Nat) : System.FilePath :=
  directory / s!"source-{source}"

/-- Decode the retained signed-source tables and use the existing fold once.
Full chunks have an even number of inputs; only the true tail is odd. -/
private def normFieldsAfterTwo
    (publicPath roundZero roundOne roundTwo directory outputDirectory : System.FilePath)
    (chunkPairs : Nat) : IO UInt32 := do
  unless chunkPairs > 0 do throw (IO.userError "zero norm fold chunk size")
  unless !(← outputDirectory.pathExists) do throw (IO.userError "output directory already exists")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath [roundZero.toString, roundOne.toString, roundTwo.toString]
  readMetadata directory { trace with challenges := trace.challenges.take 2 } 0 groupCount
  let table := PiCCSPrefixCodeFold.pairedTable
    (PiCCSPrefixNorm.values (trace.challenges[0]?.getD K.zero))
    (trace.challenges[1]?.getD K.zero)
  let challenge := trace.challenges[2]?.getD K.zero
  let pairs := (groupCount + 1) / 2
  IO.FS.createDirAll outputDirectory
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  for batch in [:(productionShape.sourceCount + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error Unit)) := #[]
    for source in [batch * workers:min productionShape.sourceCount ((batch + 1) * workers)] do
      tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
        let input ← IO.FS.Handle.mk (sourcePath directory source) .read
        let output := normDirectory outputDirectory source
        IO.FS.createDir output
        let mut position := 0
        let mut ranges : Array (Nat × Nat) := #[]
        while position < pairs do
          let count := min chunkPairs (pairs - position)
          let expected := min (2 * count) (groupCount - 2 * position)
          let codes ← input.read expected.toUSize
          unless codes.size == expected do throw (IO.userError s!"truncated norm source {source}")
          let mut values : Array K := #[]
          for code in codes do
            if bound : code.toNat < 81 then values := values.push (table.get ⟨code.toNat, bound⟩)
            else throw (IO.userError s!"invalid norm code in source {source}")
          let folded := PrefixFold.foldOne extensionOps values challenge
          let bytes := PiCCSPrefixFiles.fieldBytes folded
          unless bytes.size == count * 16 do throw (IO.userError "norm fold byte count differs")
          IO.FS.writeBinFile (output / s!"{position}-{position + count}.bin") bytes
          ranges := ranges.push (position, position + count)
          position := position + count
        unless (← input.read 1).isEmpty do throw (IO.userError s!"extra bytes in norm source {source}")
        PiCCSPrefixFiles.writeManifest output (3 + source) 1 pairs trace.challenges ranges)
    for task in tasks do
      match ← IO.wait task with
      | .ok _ => pure ()
      | .error error => throw error
  report [("event", .str "norm_fields_after_three"), ("sources", Lean.toJson productionShape.sourceCount),
    ("values_per_source", Lean.toJson pairs), ("bytes", Lean.toJson (pairs * 16 * productionShape.sourceCount)),
    ("chunk_pairs", Lean.toJson chunkPairs), ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def checkNormDirectory (directory : System.FilePath) : IO Unit := do
  unless (← directory.readDir).size == productionShape.sourceCount do
    throw (IO.userError "norm prefix has missing or extra source directories")
  for source in [:productionShape.sourceCount] do
    unless (← (normDirectory directory source).isDir) do
      throw (IO.userError s!"missing norm prefix directory for source {source}")

private def advancePrefix (kind : Nat) (publicPath directory outputDirectory : System.FilePath)
    (roundPaths : List String) : IO UInt32 := do
  unless kind ≤ 2 do throw (IO.userError "expected Pad, matrix or fresh prefix family")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath roundPaths
  unless 3 ≤ trace.challenges.length do throw (IO.userError "binary prefix advancement needs at least three rounds")
  let previous := trace.challenges.dropLast
  let challenge := (trace.challenges.getLast?).getD K.zero
  let count := foldedCount (if kind == 0 then PiCCSSourceImages.shape.carrierWidth else matrixRows ())
    previous.length
  let width := if kind == 2 then Spec.ProductionRelation.matrixCount else 1
  PiCCSPrefixFiles.foldBinary kind width count directory outputDirectory previous challenge
  report [("event", .str "binary_prefix_advanced"), ("kind", Lean.toJson kind),
    ("consumed", Lean.toJson trace.challenges.length), ("rows", Lean.toJson ((count + 1) / 2)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def advanceNorm (publicPath directory outputDirectory : System.FilePath)
    (roundPaths : List String) : IO UInt32 := do
  unless !(← outputDirectory.pathExists) do throw (IO.userError "output directory already exists")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath roundPaths
  unless 4 ≤ trace.challenges.length do throw (IO.userError "norm field advancement needs at least four rounds")
  checkNormDirectory directory
  let previous := trace.challenges.dropLast
  let challenge := (trace.challenges.getLast?).getD K.zero
  let count := foldedCount PiCCSSourceImages.shape.carrierWidth previous.length
  IO.FS.createDirAll outputDirectory
  for source in [:productionShape.sourceCount] do
    PiCCSPrefixFiles.foldBinary (3 + source) 1 count
      (normDirectory directory source) (normDirectory outputDirectory source) previous challenge
  report [("event", .str "norm_fields_advanced"), ("sources", Lean.toJson productionShape.sourceCount),
    ("consumed", Lean.toJson trace.challenges.length), ("values_per_source", Lean.toJson ((count + 1) / 2)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

/-- Calculate each source cubic separately from complete canonical field files.
All records are decoded even when a requested arithmetic slice is smaller. -/
private def normFromPrefix (publicPath directory outputPath : System.FilePath)
    (first finish : Nat) (roundPaths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let trace ← readTrace publicPath roundPaths
  checkDepth trace
  unless 3 ≤ trace.challenges.length do throw (IO.userError "norm field prefix needs at least three challenges")
  checkNormDirectory directory
  let consumed := trace.challenges.length
  let count := foldedCount PiCCSSourceImages.shape.carrierWidth consumed
  unless first < finish && finish ≤ (count + 1) / 2 do throw (IO.userError "invalid norm field pair range")
  let tail := trace.coins.alpha.coordinates.drop (consumed + 1)
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let contribution (index : Nat) (low high : K) : FixedPolynomial K 3 :=
    if first ≤ index && index < finish then
      FixedPolynomial.scale extensionOps.toOps (weight index)
        (PiCCSFirstRoundPair.normPair extensionOps low high)
    else FixedPolynomial.zero extensionOps.toOps 3
  let mut jobs : Array (Nat × PiCCSPrefixFiles.Chunk × Option (Array K)) := #[]
  for source in [:productionShape.sourceCount] do
    let chunks ← PiCCSPrefixFiles.read (normDirectory directory source)
      (3 + source) 1 count trace.challenges
    for part in [:chunks.size] do
      if bound : part < chunks.size then
        jobs := jobs.push (source, chunks[part]'bound,
          (chunks[part + 1]?).map PiCCSPrefixFiles.Chunk.firstValues)
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut total := FixedPolynomial.zero extensionOps.toOps 3
  for batch in [:(jobs.size + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error (FixedPolynomial K 3))) := #[]
    for part in [batch * workers:min jobs.size ((batch + 1) * workers)] do
      if bound : part < jobs.size then
        let (source, chunk, nextValues) := jobs[part]'bound
        tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
          let input ← IO.FS.Handle.mk chunk.path .read
          let mut previous : Option K := none
          let mut subtotal := FixedPolynomial.zero extensionOps.toOps 3
          for index in [chunk.first:chunk.finish] do
            let values ← binaryRow input 1
            if index == chunk.first then
              unless decide (values = chunk.firstValues) do throw (IO.userError "norm first row changed")
            let value := values.getD 0 K.zero
            if index % 2 == 0 then previous := some value
            else if let some low := previous then
              subtotal := FixedPolynomial.add extensionOps.toOps subtotal
                (contribution (index / 2) low value)
              previous := none
          unless (← input.read 1).isEmpty do throw (IO.userError "extra norm prefix bytes")
          if let some low := previous then
            let high := (nextValues.getD #[K.zero]).getD 0 K.zero
            subtotal := FixedPolynomial.add extensionOps.toOps subtotal
              (contribution ((chunk.finish - 1) / 2) low high)
          return FixedPolynomial.scale extensionOps.toOps
            (TargetPolynomial.power extensionOps.toOps trace.coins.gamma source) subtotal)
    for task in tasks do
      let value ← match ← IO.wait task with
        | .ok value => pure value
        | .error error => throw error
      total := FixedPolynomial.add extensionOps.toOps total value
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom consumed, .atom first, .atom finish,
    .array (trace.challenges.map extensionValue),
    .array (total.coefficients.map extensionValue)]).render ++ "\n")
  report [("event", .str "prefix_norm_complete"), ("consumed", Lean.toJson consumed),
    ("first", Lean.toJson first), ("end", Lean.toJson finish),
    ("sources", Lean.toJson productionShape.sourceCount),
    ("validated_values", Lean.toJson (count * productionShape.sourceCount)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def finalPoint (publicPath : System.FilePath) (rounds : List String) :
    IO PaperAlgebra.Point := do
  let trace ← readTrace publicPath rounds
  if complete : trace.challenges.length = cubeVariables then
    return ⟨trace.challenges, complete⟩
  else throw (IO.userError "original final evaluations require all 28 Lean rounds")

private def originalMatrix (publicPath sourcePath : System.FilePath)
    (arguments : List String) (reference : Bool) : IO UInt32 := do
  let (rangeArguments, roundArguments) := arguments.span (fun argument => argument != "--")
  let "--" :: rounds := roundArguments
    | throw (IO.userError "original-matrix requires -- before all Lean round paths")
  let requests ← checked (PiCCSOriginalEvaluation.parseRangeRequests rangeArguments)
  PiCCSOriginalEvaluation.matrixRanges (← finalPoint publicPath rounds) sourcePath requests reference

private def originalPad (publicPath sourcePath : System.FilePath)
    (arguments : List String) (reference : Bool) : IO UInt32 := do
  let (rangeArguments, roundArguments) := arguments.span (fun argument => argument != "--")
  let "--" :: rounds := roundArguments
    | throw (IO.userError "original-pad requires -- before all Lean round paths")
  let requests ← checked (PiCCSOriginalEvaluation.parsePadRequests rangeArguments)
  PiCCSOriginalEvaluation.padRanges (← finalPoint publicPath rounds) sourcePath requests reference


private def mergeOriginalPad (publicPath outputPath : System.FilePath)
    (arguments : List String) : IO UInt32 := do
  let (pads, roundArguments) := arguments.span (fun argument => argument != "--")
  let "--" :: rounds := roundArguments
    | throw (IO.userError "merge-original-pad requires -- before all Lean round paths")
  PiCCSOriginalMerge.mergePad (← finalPoint publicPath rounds) outputPath pads

private def mergeOriginal (publicPath outputPath : System.FilePath)
    (arguments : List String) : IO UInt32 := do
  let (pads, rest) := arguments.span (fun argument => argument != "--")
  let "--" :: remaining := rest
    | throw (IO.userError "merge-original requires -- after Pad paths")
  let (matrices, roundArguments) := remaining.span (fun argument => argument != "--")
  let "--" :: rounds := roundArguments
    | throw (IO.userError "merge-original requires -- before all Lean round paths")
  PiCCSOriginalMerge.merge (← finalPoint publicPath rounds) outputPath pads matrices

private def family (name : String) : IO Nat :=
  match name with
  | "pad" => pure 0
  | "matrix" => pure 1
  | "fresh" => pure 2
  | _ => throw (IO.userError "expected pad, matrix or fresh")

end NightstreamFPrime.Export.PiCCSPrefixReplay

def main (arguments : List String) : IO UInt32 := do
  match arguments with
  | "merge-original-pad" :: publicPath :: outputPath :: rest =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.mergeOriginalPad publicPath outputPath rest
  | "merge-original" :: publicPath :: outputPath :: rest =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.mergeOriginal publicPath outputPath rest
  | "original-pad" :: publicPath :: sourcePath :: rest =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.originalPad publicPath sourcePath rest false
  | "original-pad-reference" :: publicPath :: sourcePath :: rest =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.originalPad publicPath sourcePath rest true
  | "original-matrix" :: publicPath :: sourcePath :: rest =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.originalMatrix publicPath sourcePath rest false
  | "original-matrix-reference" :: publicPath :: sourcePath :: rest =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.originalMatrix publicPath sourcePath rest true
  | ["norm-fields-after-two", publicPath, roundZero, roundOne, roundTwo,
      directory, outputDirectory, chunkPairs] =>
      match chunkPairs.toNat? with
      | some chunkPairs =>
          NightstreamFPrime.Export.PiCCSPrefixReplay.normFieldsAfterTwo
            publicPath roundZero roundOne roundTwo directory outputDirectory chunkPairs
      | none => throw (IO.userError "expected natural-number norm fold chunk size")
  | "advance-prefix" :: kind :: publicPath :: directory :: outputDirectory :: rounds =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.advancePrefix
        (← NightstreamFPrime.Export.PiCCSPrefixReplay.family kind)
        publicPath directory outputDirectory rounds
  | "advance-norm" :: publicPath :: directory :: outputDirectory :: rounds =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.advanceNorm publicPath directory outputDirectory rounds
  | "norm-prefix" :: publicPath :: directory :: outputPath :: first :: finish :: rounds =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSPrefixReplay.normFromPrefix
            publicPath directory outputPath first finish rounds
      | _, _ => throw (IO.userError "expected natural-number norm pair bounds")
  | "advance-first" :: kind :: publicPath :: roundZero :: roundOne :: outputDirectory :: directories =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.advanceFirst
        (← NightstreamFPrime.Export.PiCCSPrefixReplay.family kind)
        publicPath roundZero roundOne outputDirectory directories
  | "fresh-prefix" :: publicPath :: directory :: outputPath :: first :: finish :: rounds =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSPrefixReplay.freshFromPrefix
            publicPath directory outputPath first finish rounds
      | _, _ => throw (IO.userError "expected natural-number fresh pair bounds")
  | "carried-prefix" :: kind :: publicPath :: directory :: outputPath :: rounds =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.carriedFromPrefix
        (← NightstreamFPrime.Export.PiCCSPrefixReplay.family kind)
        publicPath directory outputPath rounds
  | "compose" :: publicPath :: fresh :: norm :: matrix :: pad :: output :: rounds =>
      NightstreamFPrime.Export.PiCCSPrefixReplay.composePrefix publicPath fresh norm matrix pad output rounds
  | mode :: rest =>
      if mode == "norm-after-two" || mode == "norm-after-two-direct" then
        match rest with
        | [publicPath, roundZero, roundOne, directory, outputPath, first, finish, chunkPairs] =>
            match first.toNat?, finish.toNat?, chunkPairs.toNat? with
            | some first, some finish, some chunkPairs =>
                NightstreamFPrime.Export.PiCCSPrefixReplay.normAfterTwo
                  publicPath roundZero roundOne directory outputPath first finish chunkPairs
                  (mode == "norm-after-two-direct")
            | _, _, _ => throw (IO.userError "expected natural-number norm pair range and chunk size")
        | _ => throw (IO.userError "expected public, Q0, Q1, norm directory, output and pair range")
      else
        match rest with
        | [publicPath, sourceInput, roundZero, roundOne, directory, first, finish, chunkSize] =>
            match first.toNat?, finish.toNat?, chunkSize.toNat? with
            | some first, some finish, some chunkSize =>
                if mode == "norm-codes" then
                  NightstreamFPrime.Export.PiCCSPrefixReplay.writeCodes
                    publicPath sourceInput roundZero roundOne directory first finish chunkSize
                else if mode == "check-norm-codes" || mode == "check-norm-codes-direct" then
                  NightstreamFPrime.Export.PiCCSPrefixReplay.checkCodes
                    publicPath sourceInput roundZero roundOne directory first finish chunkSize
                    (mode == "check-norm-codes-direct")
                else throw (IO.userError "unknown PiCCS prefix replay mode")
            | _, _, _ => throw (IO.userError "expected natural-number range and chunk size")
        | _ => throw (IO.userError "expected public, sources, Q0, Q1, norm directory and group range")
  | _ =>
      IO.eprintln "usage: replayPiCCSPrefix norm-codes|check-norm-codes[-direct] <public> <sources> <Q0> <Q1> <directory> <first-group> <end-group> <chunk-groups>"
      IO.eprintln "       replayPiCCSPrefix norm-after-two[-direct] <public> <Q0> <Q1> <norm-directory> <output> <first-pair> <end-pair> <chunk-pairs>"
      return 2
