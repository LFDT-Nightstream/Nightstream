import NightstreamFPrime.Export.SignedUnitSourceInput
import NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
import NightstreamFPrime.Export.Stage1.PiCCSFirstRound
import NightstreamFPrime.Export.Stage1.PiCCSFreshPolynomial
import NightstreamFPrime.Export.Stage1.PiCCSSourceImages
import NightstreamFPrime.Export.Stage1.PiCCSAggregatedImages
import NightstreamFPrime.Export.Stage1.PiCCSCarriedReadCache
import NightstreamFPrime.Export.Stage1.PiCCSCachedSelector
import NightstreamFPrime.Export.Stage1.PiCCSNormCache
import NightstreamFPrime.Export.Stage1.PiCCSNormScan
import NightstreamFPrime.Export.Stage1.PiCCSSignedFirstFold
import NightstreamFPrime.Export.Stage1.PiCCSPrefixNorm
import NightstreamFPrime.Export.Stage1.PiCCSPrefixNormBuckets
import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefix
import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixPolynomial
import NightstreamFPrime.Export.Stage1.PiCCSPadBlockMoment
import NightstreamFPrime.Export.Stage1.PiCCSPadPrefix
import NightstreamFPrime.Export.Stage1.PiCCSCarriedMoments
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache

/-!
Compute a first-round pair range from original witnesses and public claims.
Rust round messages and output evaluations are not inputs. A partial range
is a measurement/accumulation result, not a complete first round. The runner
aggregates linear carried reads before the existing numeric matrix interpreter.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiCCSFirstRoundReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
open NightstreamFPrime.Export.Codec

private abbrev Sources := Fin productionShape.sourceCount →
  Phi81Relation.Assignment PiCCSSourceImages.shape

private def layout := Folding.PiCCS.CanonicalRowLayout.layout cubeVariables
  PiCCSSourceImages.shape.carrierWidth
  (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits).cubeFits

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok value => pure value
  | .error error => throw (IO.userError error)

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def extensionValue (value : K) : Value :=
  .array [.atom value.c0.val, .atom value.c1.val]

private def sources (masks : Array (Array (Nat × Nat))) : Sources :=
  fun source column => SignedUnitSourceInput.scalar
    (masks[column.val / ringDegree]?.getD #[])
    ⟨source.val, source.isLt⟩ ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩

private def images (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (padBasis matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (powers : Nat → K)
    (assignments : Sources) (vertex : BooleanVertex cubeVariables) :
    IO (ProtocolPolynomial.OutputMessage K productionShape × K × K) := do
  let some result := PiCCSAggregatedImages.endpoint? program sourceRow layout assignments
      padBasis matrixBasis blocks powers vertex
    | throw (IO.userError "matrix row failed to load")
  return result

private structure CachedInvocation where
  firstRow : Nat
  fresh : Vector Spec.ProductionRelation.RowSemantics.PortValues 94
  carried : Vector (Vector K Spec.ProductionRelation.matrixCount) 94

private def invocationCache? (program : MatrixProgram.Program)
    (assignments : Sources) (matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (index : Nat) : Option CachedInvocation := Id.run do
  let mut start := 0
  for block in program.blocks do
    if index < start + block.rowCount then
      match block with
      | .poseidon poseidon =>
          let some (interface, row) := PiDECPoseidonNumericBlock.loadRow? poseidon
              PiCCSSourceImages.logicalWidth (index - start) | return none
          return some {
            firstRow := index - row.val
            fresh := PiDECPoseidonNumericRows.stored
              (PiCCSSourceImages.plainRead (assignments (freshSourceIndex ⟨0, by decide⟩))) interface
            carried := PiCCSCarriedReadCache.invocation matrixBasis blocks interface }
      | _ => return none
    start := start + block.rowCount
  return none

private def cachedImages (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (padBasis matrixBasis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree) (powers : Nat → K)
    (assignments : Sources) (cached : Option CachedInvocation)
    (vertex : BooleanVertex cubeVariables) :
    IO (Option CachedInvocation × (ProtocolPolynomial.OutputMessage K productionShape × K × K)) := do
  let index := NumericBooleanDomain.index vertex
  let reusable := cached.filter fun value =>
    value.firstRow ≤ index && index < value.firstRow + 94
  let ready := reusable.orElse fun _ => invocationCache? program assignments matrixBasis blocks index
  if let some value := ready then
    if within : value.firstRow ≤ index ∧ index < value.firstRow + 94 then
      let row : Fin 94 := ⟨index - value.firstRow, by omega⟩
      let fresh := Vector.ofFn fun port => ((value.fresh.get row).get port)
      return (ready, PiCCSAggregatedImages.fromRows layout assignments padBasis blocks powers
        vertex fresh (value.carried.get row))
  return (none, ← images program sourceRow padBasis matrixBasis blocks powers assignments vertex)

private def replay (publicPath sourcePath outputPath : System.FilePath)
    (first finish : Nat) : IO UInt32 := do
  unless first < finish && finish ≤ 2 ^ (cubeVariables - 1) do
    throw (IO.userError "invalid first-round pair range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let statementInput ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre statementInput)
  report [("event", .str "public_coins_ready"),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let sourceStarted ← IO.monoNanosNow
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let assignments := sources masks
  report [("event", .str "original_sources_ready"), ("records", Lean.toJson records),
    ("blocks", Lean.toJson masks.size),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - sourceStarted))]
  let preparationStarted ← IO.monoNanosNow
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let verifierInput := PiCCSPublicReplay.verifierInput statementInput
  let normTable := PiCCSNormCache.prepare power
  let alphaWeights := PiCCSTensorWeights.prepare extensionOps coins.alpha.coordinates.tail
  let priorWeights := PiCCSTensorWeights.prepare extensionOps verifierInput.priorPoint.coordinates.tail
  let basis ← IO.wait (Task.spawn fun _ => PiCCSAggregatedImages.prepare tables power)
  let blocks := PiCCSAggregatedImages.combinedBlock power assignments
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let cache ← IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  report [("event", .str "matrix_basis_ready"),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - preparationStarted))]
  let mut total : FixedPolynomial K 9 := FixedPolynomial.zero extensionOps.toOps 9
  let mut cached : Option CachedInvocation := none
  for index in [first:finish] do
    if within : index < 2 ^ 27 then
      let suffix := NumericBooleanDomain.vertex 27 ⟨index, within⟩
      let imageStarted ← IO.monoNanosNow
      let (next, low) ← cachedImages program (fun row => cache[row]?) basis.1 basis.2 blocks power
        assignments cached
        (PiCCSFirstRound.endpointVertex (by decide) false suffix)
      cached := next
      let (next, high) ← cachedImages program (fun row => cache[row]?) basis.1 basis.2 blocks power
        assignments cached
        (PiCCSFirstRound.endpointVertex (by decide) true suffix)
      cached := next
      let imageNs := (← IO.monoNanosNow) - imageStarted
      let kernelStarted ← IO.monoNanosNow
      let constructed ← IO.wait (Task.spawn fun _ =>
        PiCCSFirstRoundPair.pairPolynomialWithNorm extensionOps verifierInput power
          (PiCCSCachedSelector.equalitySelector extensionOps suffix coins.alpha alphaWeights)
          (PiCCSCachedSelector.equalitySelector extensionOps suffix verifierInput.priorPoint priorWeights)
          low.1 high.1 low.2.1 high.2.1 low.2.2 high.2.2
          (PiCCSNormCache.sourceNorm normTable power low.1 high.1))
      let term : FixedPolynomial K 9 :=
        PiCCSPublicReplay.degree_eq statementInput ▸ constructed
      total := FixedPolynomial.add extensionOps.toOps total term
      report [("event", .str "pair_complete"), ("pair", Lean.toJson index),
        ("image_ns", Lean.toJson imageNs),
        ("kernel_ns", Lean.toJson ((← IO.monoNanosNow) - kernelStarted))]
    else throw (IO.userError "pair exceeds the selected 27-bit suffix domain")
  let value := Value.array [.atom 1, .atom first, .atom finish,
    .array (total.coefficients.map extensionValue)]
  IO.FS.writeFile outputPath (value.render ++ "\n")
  report [("event", .str "first_round_range_complete"),
    ("first", Lean.toJson first), ("end", Lean.toJson finish),
    ("complete_round", Lean.toJson (first == 0 && finish == 2 ^ 27)),
    ("coefficients", Lean.toJson total.coefficients.length),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def replayNorm (publicPath sourcePath outputPath : System.FilePath)
    (first finish : Nat) : IO UInt32 := do
  unless first < finish && finish ≤ PiCCSSourceImages.blockCount do
    throw (IO.userError "invalid norm block range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let statementInput ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre statementInput)
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  report [("event", .str "norm_sources_ready"), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    productionShape.sourceCount
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let tail := coins.alpha.coordinates.tail
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let parts := min workers (finish - first)
  let computeStarted ← IO.monoNanosNow
  let mut tasks : Array (Task (Except IO.Error (FixedPolynomial K 3 × Nat))) := #[]
  for part in [:parts] do
    let start := first + (finish - first) * part / parts
    let stop := first + (finish - first) * (part + 1) / parts
    tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
      let rangeStarted ← IO.monoNanosNow
      let values ← IO.wait (Task.spawn fun _ => PiCCSNormBuckets.finish power
        (PiCCSNormScan.range weight masks start (stop - start)))
      let elapsed := (← IO.monoNanosNow) - rangeStarted
      return (values, elapsed))
  report [("event", .str "norm_ranges_queued"), ("ranges", Lean.toJson tasks.size),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted))]
  let mut total := FixedPolynomial.zero extensionOps.toOps 3
  let mut part := 0
  for task in tasks do
    let (values, rangeNs) ← match ← IO.wait task with
      | .ok result => pure result
      | .error error => throw error
    total := FixedPolynomial.add extensionOps.toOps total values
    report [("event", .str "norm_range_accumulated"), ("part", Lean.toJson part),
      ("first_block", Lean.toJson (first + (finish - first) * part / parts)),
      ("end_block", Lean.toJson (first + (finish - first) * (part + 1) / parts)),
      ("range_ns", Lean.toJson rangeNs),
      ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted))]
    part := part + 1
  let value := Value.array [.atom 1, .atom first, .atom finish,
    .array (coins.alpha.coordinates.map extensionValue), extensionValue coins.gamma,
    .array (total.coefficients.map extensionValue)]
  IO.FS.writeFile outputPath (value.render ++ "\n")
  report [("event", .str "norm_complete"), ("first_block", Lean.toJson first),
    ("end_block", Lean.toJson finish), ("workers", Lean.toJson parts),
    ("complete_carrier", Lean.toJson (first == 0 && finish == PiCCSSourceImages.blockCount)),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def freshRows (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (assignments : Sources)
    (cached : Option (Nat × Vector Spec.ProductionRelation.RowSemantics.PortValues 94))
    (vertex : BooleanVertex cubeVariables) :
    IO (Option (Nat × Vector Spec.ProductionRelation.RowSemantics.PortValues 94) ×
      Vector F Spec.ProductionRelation.matrixCount) := do
  let index := NumericBooleanDomain.index vertex
  if let some (firstRow, values) := cached then
    if within : firstRow ≤ index ∧ index < firstRow + 94 then
      let row : Fin 94 := ⟨index - firstRow, by omega⟩
      return (cached, Vector.ofFn ((values.get row).get))
  let read := PiCCSSourceImages.plainRead (assignments (freshSourceIndex ⟨0, by decide⟩))
  let mut firstRow := 0
  for block in program.blocks do
    if index < firstRow + block.rowCount then
      match block with
      | .poseidon poseidon =>
          let some (interface, row) := PiDECPoseidonNumericBlock.loadRow? poseidon
              PiCCSSourceImages.logicalWidth (index - firstRow)
            | throw (IO.userError "fresh invocation failed to load")
          let values := PiDECPoseidonNumericRows.stored read interface
          return (some (index - row.val, values), Vector.ofFn ((values.get row).get))
      | _ => break
    firstRow := firstRow + block.rowCount
  let some values := PiCCSSourceImages.freshMatrixImage? program sourceRow
      (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex
    | throw (IO.userError "fresh matrix row failed to load")
  return (none, values)

private def replayFresh (publicPath sourcePath outputPath : System.FilePath)
    (first finish : Nat) (reference : Bool := false) : IO UInt32 := do
  unless first < finish && finish ≤ 2 ^ (cubeVariables - 1) do
    throw (IO.userError "invalid fresh pair range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let statementInput ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre statementInput)
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let assignments := sources masks
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let input := PiCCSPublicReplay.verifierInput statementInput
  let weights := PiCCSTensorWeights.prepare extensionOps coins.alpha.coordinates.tail
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let sourceRows ← IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  report [("event", .str "fresh_sources_ready"), ("records", Lean.toJson records),
    ("rows", Lean.toJson program.rowCount),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let computeStarted ← IO.monoNanosNow
  let computeRange (first finish : Nat) : IO (FixedPolynomial K 9) := do
    let mut total : FixedPolynomial K 9 := FixedPolynomial.zero extensionOps.toOps 9
    let mut cached := none
    for index in [first:finish] do
      if within : index < 2 ^ 27 then
        let suffix := NumericBooleanDomain.vertex 27 ⟨index, within⟩
        let lowVertex := PiCCSFirstRound.endpointVertex (by decide) false suffix
        let highVertex := PiCCSFirstRound.endpointVertex (by decide) true suffix
        let (next, low) ← freshRows program (fun row => sourceRows[row]?) assignments cached lowVertex
        cached := next
        let (next, high) ← freshRows program (fun row => sourceRows[row]?) assignments cached highVertex
        cached := next
        let value :=
          FixedPolynomial.scale extensionOps.toOps (power productionShape.constraintOffset)
            (FixedPolynomial.widen extensionOps.toOps (Nat.le_max_left _ _)
              (if reference then
                PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps input power
                  (PiCCSCachedSelector.equalitySelector extensionOps suffix coins.alpha weights)
                  (PiCCSAggregatedImages.nonlinearMessage layout assignments lowVertex low)
                  (PiCCSAggregatedImages.nonlinearMessage layout assignments highVertex high)
              else PiCCSFreshPolynomial.ccsPolynomialWithPowers extensionOps input power
                  (PiCCSCachedSelector.equalitySelector extensionOps suffix coins.alpha weights)
                  (PiCCSAggregatedImages.nonlinearMessage layout assignments lowVertex low)
                  (PiCCSAggregatedImages.nonlinearMessage layout assignments highVertex high)))
        let term : FixedPolynomial K 9 := PiCCSPublicReplay.degree_eq statementInput ▸ value
        total := FixedPolynomial.add extensionOps.toOps total term
      else throw (IO.userError "fresh pair exceeds the selected domain")
    return total
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let parts := min workers (finish - first)
  let mut tasks : Array (Task (Except IO.Error (FixedPolynomial K 9 × Nat))) := #[]
  for part in [:parts] do
    let start := first + (finish - first) * part / parts
    let stop := first + (finish - first) * (part + 1) / parts
    tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
      let rangeStarted ← IO.monoNanosNow
      let values ← computeRange start stop
      return (values, (← IO.monoNanosNow) - rangeStarted))
  let mut total := FixedPolynomial.zero extensionOps.toOps 9
  let mut part := 0
  for task in tasks do
    let (values, rangeNs) ← match ← IO.wait task with
      | .ok result => pure result
      | .error error => throw error
    total := FixedPolynomial.add extensionOps.toOps total values
    report [("event", .str "fresh_range_accumulated"), ("part", Lean.toJson part),
      ("first", Lean.toJson (first + (finish - first) * part / parts)),
      ("end", Lean.toJson (first + (finish - first) * (part + 1) / parts)),
      ("range_ns", Lean.toJson rangeNs)]
    part := part + 1
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom first, .atom finish,
    .array (total.coefficients.map extensionValue)]).render ++ "\n")
  report [("event", .str "fresh_range_complete"), ("first", Lean.toJson first),
    ("end", Lean.toJson finish), ("reference", Lean.toJson reference),
    ("workers", Lean.toJson parts),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def carriedRows (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (basis : FixedArray (Vector K ringDegree) ringDegree)
    (blocks : Nat → Vector K ringDegree)
    (cached : Option (Nat × Array (Option (Vector K Spec.ProductionRelation.matrixCount))))
    (index : Nat) (cacheSparse : Bool := true) :
    IO (Option (Nat × Array (Option (Vector K Spec.ProductionRelation.matrixCount))) ×
      Vector K Spec.ProductionRelation.matrixCount) := do
  if let some (firstRow, values) := cached then
    if within : firstRow ≤ index ∧ index < firstRow + values.size then
      let some value := values[index - firstRow]'(by omega)
        | throw (IO.userError "carried retained row failed to load")
      return (cached, value)
  let mut firstRow := 0
  for block in program.blocks do
    if index < firstRow + block.rowCount then
      match block with
      | .poseidon poseidon =>
          let some (interface, row) := PiDECPoseidonNumericBlock.loadRow? poseidon
              PiCCSSourceImages.logicalWidth (index - firstRow)
            | throw (IO.userError "carried Poseidon row failed to load")
          let values := PiCCSCarriedReadCache.invocation basis blocks interface
          return (some (index - row.val, values.toArray.map some), values.get row)
      | .phi81Product product =>
          if cacheSparse then
            let localRow := index - firstRow
            let some descriptor := MatrixProgram.Phi81Product.ringDescriptor?
                product.families (localRow / 108)
              | throw (IO.userError "carried product descriptor failed to load")
            let some interface := PiDECProductInterface.interface? product
                PiCCSSourceImages.logicalWidth descriptor
              | throw (IO.userError "carried product interface failed to load")
            let values := PiCCSCarriedReadCache.productInvocation basis blocks interface
            let row : Fin 108 := ⟨localRow % 108, Nat.mod_lt _ (by decide)⟩
            let some value := values.get row
              | throw (IO.userError "carried product row failed to load")
            return (some (index - row.val, values.toArray), value)
          else
            let some values := PiCCSLinearRows.row? program
                (columns := PiCCSSourceImages.logicalWidth) sourceRow
                (PiCCSCarriedRead.read basis blocks) index
              | throw (IO.userError "carried product row failed to load")
            return (none, values)
      | .ordinary _ | .pin _ | .multiplicationGrid _ =>
          let loaded := if cacheSparse then
              PiCCSCarriedReadCache.row? program
                (columns := PiCCSSourceImages.logicalWidth) sourceRow basis blocks index
            else PiCCSLinearRows.row? program
              (columns := PiCCSSourceImages.logicalWidth) sourceRow
              (PiCCSCarriedRead.read basis blocks) index
          let some values := loaded
            | throw (IO.userError "carried sparse row failed to load")
          return (none, values)
    firstRow := firstRow + block.rowCount
  throw (IO.userError "carried matrix row exceeds active rows")

private def replayCarriedMatrix (publicPath sourcePath outputPath : System.FilePath)
    (first finish : Nat) (cacheSparse : Bool := true) : IO UInt32 := do
  unless first < finish do throw (IO.userError "invalid carried row range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let assignments := sources masks
  report [("event", .str "carried_sources_ready"), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let tail := (PiCCSPublicReplay.verifierInput input).priorPoint.coordinates.tail
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  let basis ← IO.wait (Task.spawn fun _ => (PiCCSAggregatedImages.prepare tables power).2)
  let blocks := PiCCSAggregatedImages.combinedBlock power assignments
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  unless finish ≤ program.rowCount do throw (IO.userError "carried range exceeds active rows")
  let cache ← IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  -- One full low-weight cycle was measured across the runtime workers.
  let extent := finish - first
  let rangeSize := max 1 ((2 * weights.1.size) / workers)
  let parts := min extent (max workers ((extent + rangeSize - 1) / rangeSize))
  let computeStarted ← IO.monoNanosNow
  let mut low := K.zero
  let mut high := K.zero
  for batch in [:(parts + workers - 1) / workers] do
    let batchFirst := batch * workers
    let batchEnd := min parts (batchFirst + workers)
    let mut tasks : Array (Task (Except IO.Error (K × K × Nat))) := #[]
    for part in [batchFirst:batchEnd] do
      let start := first + (finish - first) * part / parts
      let stop := first + (finish - first) * (part + 1) / parts
      tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
        let rangeStarted ← IO.monoNanosNow
        let mut low := K.zero
        let mut high := K.zero
        let mut retained := none
        for index in [start:stop] do
          if index / 2 < 2 ^ 27 then
            let (next, values) ← carriedRows program (fun row => cache[row]?) basis blocks retained index cacheSparse
            retained := next
            let combined := FiniteSumAlgebra.sumMap extensionOps
              (canonicalFinIndices Spec.ProductionRelation.matrixCount) fun slot =>
                extensionOps.mul (power (productionShape.runningCount * slot.val)) (values.get slot)
            let contribution := extensionOps.mul (weight (index / 2)) combined
            if index % 2 == 0 then low := extensionOps.add low contribution
            else high := extensionOps.add high contribution
          else throw (IO.userError "carried row exceeds Boolean domain")
        return (low, high, (← IO.monoNanosNow) - rangeStarted))
    let mut part := batchFirst
    for task in tasks do
      let (lo, hi, rangeNs) ← match ← IO.wait task with
        | .ok result => pure result
        | .error error => throw error
      low := extensionOps.add low lo
      high := extensionOps.add high hi
      report [("event", .str "carried_matrix_range_accumulated"),
        ("first", Lean.toJson (first + (finish - first) * part / parts)),
        ("end", Lean.toJson (first + (finish - first) * (part + 1) / parts)),
        ("range_ns", Lean.toJson rangeNs),
        ("low", .arr #[Lean.toJson lo.c0.val, Lean.toJson lo.c1.val]),
        ("high", .arr #[Lean.toJson hi.c0.val, Lean.toJson hi.c1.val])]
      part := part + 1
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom first, .atom finish,
    extensionValue low, extensionValue high]).render ++ "\n")
  report [("event", .str "carried_matrix_complete"), ("first", Lean.toJson first),
    ("end", Lean.toJson finish), ("workers", Lean.toJson workers),
    ("ranges", Lean.toJson parts),
    ("cached_sparse", Lean.toJson cacheSparse),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def replayCarriedPad (publicPath sourcePath outputPath : System.FilePath)
    (first finish : Nat) : IO UInt32 := do
  unless first < finish && finish ≤ PiCCSSourceImages.blockCount do
    throw (IO.userError "invalid carried Pad block range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let assignments := sources masks
  report [("event", .str "carried_pad_sources_ready"), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let tail := (PiCCSPublicReplay.verifierInput input).priorPoint.coordinates.tail
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  let basis ← IO.wait (Task.spawn fun _ => (PiCCSAggregatedImages.prepare tables power).1)
  let blocks := PiCCSAggregatedImages.combinedBlock power assignments
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  -- The measured 8192-block preflight is the existing low-weight table size.
  -- Batches retain the measured extent and existing dedicated worker count.
  -- The default runtime pool ran this workload on one CPU in the preflight.
  let extent := finish - first
  let rangeSize := max 1 weights.1.size
  let parts := min extent (max workers ((extent + rangeSize - 1) / rangeSize))
  let computeStarted ← IO.monoNanosNow
  let mut low := K.zero
  let mut high := K.zero
  for batch in [:(parts + workers - 1) / workers] do
    let batchFirst := batch * workers
    let batchEnd := min parts (batchFirst + workers)
    let mut tasks : Array (Task (Except IO.Error (K × K × Nat))) := #[]
    for part in [batchFirst:batchEnd] do
      let start := first + (finish - first) * part / parts
      let stop := first + (finish - first) * (part + 1) / parts
      tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
        let rangeStarted ← IO.monoNanosNow
        let mut low := K.zero
        let mut high := K.zero
        for block in [start:stop] do
          let source := blocks block
          let values := PiCCSPadBlockMoment.blockMoment basis weight block source
          low := extensionOps.add low values.1
          high := extensionOps.add high values.2
        return (low, high, (← IO.monoNanosNow) - rangeStarted))
    let mut part := batchFirst
    for task in tasks do
      let (lo, hi, rangeNs) ← match ← IO.wait task with
        | .ok result => pure result
        | .error error => throw error
      low := extensionOps.add low lo
      high := extensionOps.add high hi
      report [("event", .str "carried_pad_range_accumulated"),
        ("first_block", Lean.toJson (first + (finish - first) * part / parts)),
        ("end_block", Lean.toJson (first + (finish - first) * (part + 1) / parts)),
        ("range_ns", Lean.toJson rangeNs),
      ("low", .arr #[Lean.toJson lo.c0.val, Lean.toJson lo.c1.val]),
      ("high", .arr #[Lean.toJson hi.c0.val, Lean.toJson hi.c1.val])]
      part := part + 1
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom first, .atom finish,
    extensionValue low, extensionValue high]).render ++ "\n")
  report [("event", .str "carried_pad_complete"), ("first_block", Lean.toJson first),
    ("end_block", Lean.toJson finish), ("workers", Lean.toJson workers),
    ("ranges", Lean.toJson parts),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def composeExtension (value : Lean.Json) : Except String K := do
  let words ← PiCCSInputCheck.decodeVector 2 PiCCSInputCheck.decodeField value
  return ⟨words.get 0, words.get 1⟩

private def composePolynomial (degree : Nat) (value : Lean.Json) :
    Except String (FixedPolynomial K degree) := do
  let values ← PiCCSInputCheck.decodeVector (degree + 1) composeExtension value
  return ⟨values.toList, Vector.length_toList⟩

private def composeRead (path : System.FilePath) : IO Lean.Json := do
  checked (Lean.Json.parse (← IO.FS.readFile path))

private def composeFresh (expectedEnd : Nat) (value : Lean.Json) :
    Except String (FixedPolynomial K 9) := do
  match (← value.getArr?).toList with
  | [schema, first, finish, coefficients] =>
      unless (← schema.getNat?) == 1 && (← first.getNat?) == 0 &&
          (← finish.getNat?) == expectedEnd do
        throw "fresh contribution must cover the complete active pair prefix"
      composePolynomial 9 coefficients
  | _ => throw "expected four fresh contribution fields"

private def composeNorm (expectedEnd : Nat) (expectedAlpha : List K)
    (expectedGamma : K) (value : Lean.Json) : Except String (FixedPolynomial K 3) := do
  match (← value.getArr?).toList with
  | [schema, first, finish, alphaValue, gammaValue, coefficients] =>
      unless (← schema.getNat?) == 1 && (← first.getNat?) == 0 &&
          (← finish.getNat?) == expectedEnd do
        throw "norm contribution must cover the complete carrier"
      let alpha ← PiCCSInputCheck.decodeVector cubeVariables composeExtension alphaValue
      let gamma ← composeExtension gammaValue
      unless decide (alpha.toList = expectedAlpha ∧ gamma = expectedGamma) do
        throw "norm public coins differ from the original public input"
      composePolynomial 3 coefficients
  | _ => throw "expected six inner norm contribution fields"

private def composeMoment (value : Lean.Json) : Except String (Nat × Nat × K × K) := do
  match (← value.getArr?).toList with
  | [schema, first, finish, low, high] =>
      unless (← schema.getNat?) == 1 do throw "expected moment schema 1"
      return (← first.getNat?, ← finish.getNat?, ← composeExtension low, ← composeExtension high)
  | _ => throw "expected five carried moment fields"

/-- File order is the declared range order. No sorting or duplicate removal
can turn gaps, overlaps or repeated ranges into accepted coverage. -/
private def composeMoments (kind : String) (expectedEnd : Nat) (paths : List String) :
    IO (K × K) := do
  let mut next := 0
  let mut low := K.zero
  let mut high := K.zero
  for path in paths do
    let (first, finish, pieceLow, pieceHigh) ← checked (composeMoment (← composeRead path))
    unless first == next && first < finish && finish ≤ expectedEnd do
      throw (IO.userError s!"{kind} moments have a gap, overlap, reversed order or invalid extent: {path}")
    low := extensionOps.add low pieceLow
    high := extensionOps.add high pieceHigh
    next := finish
  unless next == expectedEnd do
    throw (IO.userError s!"{kind} moments do not cover the complete selected range")
  return (low, high)

/-- Compose independently computed Lean contributions. The file bindings to
original sources remain in external execution evidence; no proof target is read. -/
private def composeRound (publicPath freshPath normPath outputPath : System.FilePath)
    (matrixPaths padPaths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let statementInput ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre statementInput)
  let input := PiCCSPublicReplay.verifierInput statementInput
  let matrixRows := (PerApplicationMatrixProgram.matrixProgram
    Poseidon2HashChainV1Package.application).rowCount
  let carrierBlocks := PiCCSSourceImages.blockCount
  let fresh ← checked (composeFresh ((matrixRows + 1) / 2) (← composeRead freshPath))
  let norm ← checked (composeNorm carrierBlocks coins.alpha.coordinates coins.gamma
    (← composeRead normPath))
  let matrix ← composeMoments "matrix" matrixRows matrixPaths
  let pad ← composeMoments "Pad" carrierBlocks padPaths
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let alphaHead := PiCCSCarriedMoments.headSelector extensionOps coins.alpha
  let priorHead := PiCCSCarriedMoments.headSelector extensionOps input.priorPoint
  let normTerm : FixedPolynomial K 9 :=
    FixedPolynomial.scale extensionOps.toOps (power productionShape.constraintOffset)
      (FixedPolynomial.scale extensionOps.toOps (power productionShape.freshCount)
        (FixedPolynomial.widen extensionOps.toOps (by decide : 4 ≤ 9)
          (FixedPolynomial.mul extensionOps.toOps alphaHead norm)))
  let carried : FixedPolynomial K 9 :=
    PiCCSCarriedMoments.carriedPair extensionOps (by decide : 2 ≤ 9) priorHead
      (power productionShape.matrixEvaluationOffset) pad.1 pad.2 matrix.1 matrix.2
  let polynomial := FixedPolynomial.add extensionOps.toOps carried
    (FixedPolynomial.add extensionOps.toOps fresh normTerm)
  let initial := PiCCSPublicReplay.initialClaim statementInput coins.gamma
  let endpoints := extensionOps.add
    (polynomial.evaluate extensionOps.toOps K.zero)
    (polynomial.evaluate extensionOps.toOps K.one)
  unless decide (endpoints = initial) do
    throw (IO.userError s!"first-round endpoint sum differs from the initial claim: sum={((extensionValue endpoints).render)}, initial={((extensionValue initial).render)}")
  let (challenge, nextState) := PiCCSPublicReplay.firstRound coins.state polynomial
  let nextClaim := polynomial.evaluate extensionOps.toOps challenge
  let stateValue := fun state : Transcript.State => Value.array (state.map (fun word => .atom word.val))
  -- Trace schema 1: alpha, gamma, pre-state, ten q coefficients, challenge,
  -- post-squeeze state, initial claim, q(0)+q(1), and q(challenge).
  let encoded := Value.array [.atom 1,
    .array (coins.alpha.coordinates.map extensionValue), extensionValue coins.gamma,
    stateValue coins.state, .array (polynomial.coefficients.map extensionValue),
    extensionValue challenge, stateValue nextState, extensionValue initial,
    extensionValue endpoints, extensionValue nextClaim]
  IO.FS.writeFile outputPath (encoded.render ++ "\n")
  report [("event", .str "first_round_composed"), ("coefficients", Lean.toJson polynomial.coefficients.length),
    ("matrix_rows", Lean.toJson matrixRows), ("carrier_blocks", Lean.toJson carrierBlocks),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def savedFirstChallenge
    (coins : FiatShamir.PreSumcheck K Transcript.State productionShape)
    (roundPath : System.FilePath) : IO K := do
  let saved ← checked ((← composeRead roundPath).getArr?)
  unless saved.size == 10 do throw (IO.userError "expected complete Lean round-zero result")
  let some encoded ← pure saved[4]? | throw (IO.userError "missing Lean polynomial")
  let polynomial ← checked (composePolynomial 9 encoded)
  let (challenge, _) := PiCCSPublicReplay.firstRound coins.state polynomial
  let some encodedChallenge ← pure saved[5]? | throw (IO.userError "missing saved challenge")
  let savedChallenge ← checked (composeExtension encodedChallenge)
  unless decide (challenge = savedChallenge) do
    throw (IO.userError "saved challenge differs from the independently replayed transcript")
  return challenge

private def saveFreshPrefix
    (publicPath sourcePath roundPath outputDirectory : System.FilePath)
    (first finish : Nat) (reference : Bool := false) : IO UInt32 := do
  unless !(← outputDirectory.pathExists) do throw (IO.userError "output directory already exists")
  let started ← IO.monoNanosNow
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let challenge ← savedFirstChallenge coins roundPath
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let pairs := (program.rowCount + 1) / 2
  unless first < finish && finish ≤ pairs do throw (IO.userError "invalid fresh prefix pair range")
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let assignments := sources masks
  let sourceRows ← IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  let loadRow cached vertex := do
    if reference then
      let some values := PiCCSSourceImages.freshMatrixImage? program (fun row => sourceRows[row]?)
          (assignments (freshSourceIndex ⟨0, by decide⟩)) vertex
        | throw (IO.userError "fresh prefix reference row failed to load")
      pure (none, values)
    else freshRows program (fun row => sourceRows[row]?) assignments cached vertex
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let parts := min workers (finish - first)
  IO.FS.createDirAll outputDirectory
  report [("event", .str "fresh_prefix_sources_ready"), ("records", Lean.toJson records),
    ("rows", Lean.toJson program.rowCount),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let computeStarted ← IO.monoNanosNow
  let mut tasks : Array (Task (Except IO.Error Nat)) := #[]
  for part in [:parts] do
    let start := first + (finish - first) * part / parts
    let stop := first + (finish - first) * (part + 1) / parts
    tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
      let rangeStarted ← IO.monoNanosNow
      let output ← IO.FS.Handle.mk (outputDirectory / s!"{start}-{stop}.jsonl") .write
      output.putStrLn ((Value.array [.atom 1, .atom 1, .atom Spec.ProductionRelation.matrixCount,
        .atom program.rowCount, .atom start, .atom stop, extensionValue challenge]).render)
      let mut cached := none
      for index in [start:stop] do
        if within : index < 2 ^ 27 then
          let suffix := NumericBooleanDomain.vertex 27 ⟨index, within⟩
          let (next, low) ← loadRow cached
            (PiCCSFirstRound.endpointVertex (by decide) false suffix)
          cached := next
          let (next, high) ← loadRow cached
            (PiCCSFirstRound.endpointVertex (by decide) true suffix)
          cached := next
          let values := PiCCSFreshPrefix.pairRow (low.map K.embed) (high.map K.embed) challenge
          output.putStrLn ((Value.array [.atom index,
            .array (values.toList.map extensionValue)]).render)
        else throw (IO.userError "fresh prefix index exceeds Boolean domain")
      output.putStrLn "[]"
      return (← IO.monoNanosNow) - rangeStarted)
  let mut part := 0
  for task in tasks do
    let elapsed ← match ← IO.wait task with
      | .ok elapsed => pure elapsed
      | .error error => throw error
    report [("event", .str "fresh_prefix_range_written"),
      ("first", Lean.toJson (first + (finish - first) * part / parts)),
      ("end", Lean.toJson (first + (finish - first) * (part + 1) / parts)),
      ("elapsed_ns", Lean.toJson elapsed)]
    part := part + 1
  report [("event", .str "fresh_prefix_complete"), ("first", Lean.toJson first),
    ("end", Lean.toJson finish), ("matrices", Lean.toJson Spec.ProductionRelation.matrixCount),
    ("reference", Lean.toJson reference),
    ("consumed_challenges", Lean.toJson (1 : Nat)), ("workers", Lean.toJson parts),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def saveCarriedPrefix
    (publicPath sourcePath roundPath outputDirectory : System.FilePath)
    (first finish : Nat) (pad : Bool) : IO UInt32 := do
  unless first < finish do throw (IO.userError "empty or reversed carried prefix range")
  unless !(← outputDirectory.pathExists) do throw (IO.userError "output directory already exists")
  let started ← IO.monoNanosNow
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let challenge ← savedFirstChallenge coins roundPath
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let bound := if pad then PiCCSSourceImages.blockCount else (program.rowCount + 1) / 2
  unless finish ≤ bound do throw (IO.userError "carried prefix range exceeds selected extent")
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let assignments := sources masks
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let tail := (PiCCSPublicReplay.verifierInput input).priorPoint.coordinates.drop 2
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  let basis ← IO.wait (Task.spawn fun _ =>
    if pad then (PiCCSAggregatedImages.prepare tables power).1
    else (PiCCSAggregatedImages.prepare tables power).2)
  let blocks := PiCCSAggregatedImages.combinedBlock power assignments
  let sourceRows : Std.HashMap Nat R1CS.Row ← if pad then pure {} else IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  -- Reuse the measured Pad block and matrix row extents of the original scans.
  let rangeSize := max 1 (if pad then weights.1.size else weights.1.size / workers)
  let extent := finish - first
  let parts := min extent (max workers ((extent + rangeSize - 1) / rangeSize))
  let width := if pad then 27 else 1
  let kind := if pad then 0 else 1
  IO.FS.createDirAll outputDirectory
  report [("event", .str "carried_prefix_sources_ready"), ("pad", Lean.toJson pad),
    ("records", Lean.toJson records), ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let computeStarted ← IO.monoNanosNow
  let mut low := K.zero
  let mut high := K.zero
  for batch in [:(parts + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error (K × K × Nat))) := #[]
    for part in [batch * workers:min parts ((batch + 1) * workers)] do
      let start := first + (finish - first) * part / parts
      let stop := first + (finish - first) * (part + 1) / parts
      tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
        let partStarted ← IO.monoNanosNow
        let output ← IO.FS.Handle.mk (outputDirectory / s!"{start}-{stop}.jsonl") .write
        output.putStrLn ((Value.array [.atom 1, .atom 1, .atom kind, .atom width,
          .atom bound, .atom start, .atom stop, extensionValue challenge]).render)
        let mut cached := none
        let mut low := K.zero
        let mut high := K.zero
        for index in [start:stop] do
          let values ← if pad then pure (PiCCSPadPrefix.foldedBlock basis (blocks index) challenge)
            else do
              let mut pair := #[]
              for row in [2 * index:2 * index + 2] do
                if row < program.rowCount then
                  let (next, values) ← carriedRows program (fun current => sourceRows[current]?)
                    basis blocks cached row
                  cached := next
                  pair := pair.push (FiniteSumAlgebra.sumMap extensionOps
                    (canonicalFinIndices Spec.ProductionRelation.matrixCount) fun slot =>
                      extensionOps.mul (power (productionShape.runningCount * slot.val)) (values.get slot))
                else pair := pair.push K.zero
              pure (PrefixFold.foldOne extensionOps pair challenge)
          unless values.size == width do throw (IO.userError "carried prefix width differs from selected family")
          output.putStrLn ((Value.array [.atom index, .array (values.toList.map extensionValue)]).render)
          for lane in [:values.size] do
            let position := width * index + lane
            let contribution := extensionOps.mul (weight (position / 2)) (values.getD lane K.zero)
            if position % 2 == 0 then low := extensionOps.add low contribution
            else high := extensionOps.add high contribution
        output.putStrLn "[]"
        return (low, high, (← IO.monoNanosNow) - partStarted))
    let mut part := batch * workers
    for task in tasks do
      let (lo, hi, elapsed) ← match ← IO.wait task with
        | .ok value => pure value
        | .error error => throw error
      low := extensionOps.add low lo
      high := extensionOps.add high hi
      report [("event", .str "carried_prefix_range_written"),
        ("first", Lean.toJson (first + (finish - first) * part / parts)),
        ("end", Lean.toJson (first + (finish - first) * (part + 1) / parts)),
        ("elapsed_ns", Lean.toJson elapsed)]
      part := part + 1
  IO.FS.writeFile (outputDirectory / "moments.json") ((Value.array [.atom 1, .atom 1,
    .atom kind, .atom (width * first), .atom (width * finish), extensionValue challenge,
    extensionValue low, extensionValue high]).render ++ "\n")
  report [("event", .str "carried_prefix_complete"), ("pad", Lean.toJson pad),
    ("first", Lean.toJson first), ("end", Lean.toJson finish),
    ("values", Lean.toJson (width * (finish - first))), ("workers", Lean.toJson workers), ("ranges", Lean.toJson parts),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private structure FreshPrefixChunk where
  path : System.FilePath
  first : Nat
  finish : Nat
  firstValues : Vector K Spec.ProductionRelation.matrixCount

private def prefixRow (value : Lean.Json) (expected : Nat) :
    Except String (Vector K Spec.ProductionRelation.matrixCount) := do
  match (← value.getArr?).toList with
  | [index, row] =>
      unless (← index.getNat?) == expected do throw "fresh prefix row index is not consecutive"
      PiCCSInputCheck.decodeVector Spec.ProductionRelation.matrixCount composeExtension row
  | _ => throw "expected indexed fresh prefix row"

private def prefixChunks (directory : System.FilePath) (rowCount : Nat) (challenge : K)
    (expectedFirst expectedEnd : Nat) :
    IO (Array FreshPrefixChunk) := do
  let mut chunks := #[]
  for entry in ← directory.readDir do
    let input ← IO.FS.Handle.mk entry.path .read
    let value ← checked (Lean.Json.parse (← input.getLine))
    let (first, finish) ← checked (do
      match (← value.getArr?).toList with
      | [schema, consumed, ports, rows, first, finish, encodedChallenge] =>
          unless (← schema.getNat?) == 1 && (← consumed.getNat?) == 1 &&
              (← ports.getNat?) == Spec.ProductionRelation.matrixCount &&
              (← rows.getNat?) == rowCount do throw "fresh prefix profile differs from the selected input"
          unless decide ((← composeExtension encodedChallenge) = challenge) do
            throw "fresh prefix challenge differs from the Lean transcript"
          pure (← first.getNat?, ← finish.getNat?)
      | _ => throw "expected seven fresh prefix header fields")
    unless first < finish do throw (IO.userError "empty or reversed fresh prefix chunk")
    let firstValues ← checked (prefixRow (← checked (Lean.Json.parse (← input.getLine))) first)
    chunks := chunks.push ⟨entry.path, first, finish, firstValues⟩
  chunks := chunks.qsort (fun left right => decide (left.first < right.first))
  let mut next := expectedFirst
  for chunk in chunks do
    unless chunk.first == next && chunk.finish ≤ expectedEnd do
      throw (IO.userError "fresh prefix chunks have a gap, overlap or excessive range")
    next := chunk.finish
  unless next == expectedEnd do throw (IO.userError "fresh prefix is incomplete")
  return chunks

private def freshAfterFirst (publicPath roundPath directory outputPath : System.FilePath)
    (interval : Option (Nat × Nat) := none) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let challenge ← savedFirstChallenge coins roundPath
  let rowCount := (PerApplicationMatrixProgram.matrixProgram
    Poseidon2HashChainV1Package.application).rowCount
  let fullEnd := (rowCount + 1) / 2
  let first := (interval.map Prod.fst).getD 0
  let finish := (interval.map Prod.snd).getD fullEnd
  unless first < finish && finish ≤ fullEnd && first % 2 == 0 &&
      (finish % 2 == 0 || finish == fullEnd) do
    throw (IO.userError "fresh scan range must contain complete adjacent pairs")
  let chunks ← prefixChunks directory rowCount challenge first finish
  let challenges := [challenge]
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let weights := PiCCSTensorWeights.prepare extensionOps
    (PiCCSPrefixSelector.dropPoint coins.alpha challenges.length).coordinates.tail
  let contribution (index : Nat) (low high : Vector K Spec.ProductionRelation.matrixCount) :
      IO (FixedPolynomial K 9) := do
    if inside : index < 2 ^ 26 then
      let value := PiCCSFreshPrefixPolynomial.contribution (PiCCSPublicReplay.verifierInput input)
        coins.alpha challenges (NumericBooleanDomain.vertex 26 ⟨index, inside⟩) power weights low high
      pure (PiCCSPublicReplay.degree_eq input ▸ value)
    else throw (IO.userError "fresh prefix pair exceeds the second-round domain")
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let computeStarted ← IO.monoNanosNow
  let mut total := FixedPolynomial.zero extensionOps.toOps 9
  for batch in [:(chunks.size + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error (FixedPolynomial K 9 × Nat))) := #[]
    for part in [batch * workers:min chunks.size ((batch + 1) * workers)] do
      if found : part < chunks.size then
        let chunk := chunks[part]'found
        let nextValues := (chunks[part + 1]?).map FreshPrefixChunk.firstValues
        tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
          let partStarted ← IO.monoNanosNow
          let stream ← IO.FS.Handle.mk chunk.path .read
          let _ ← stream.getLine
          let mut previous := none
          let mut subtotal := FixedPolynomial.zero extensionOps.toOps 9
          for index in [chunk.first:chunk.finish] do
            let values ← checked (prefixRow (← checked (Lean.Json.parse (← stream.getLine))) index)
            if index == chunk.first then
              unless decide (values = chunk.firstValues) do
                throw (IO.userError "fresh prefix first row changed after header validation")
            if index % 2 == 0 then previous := some values
            else if let some low := previous then
              subtotal := FixedPolynomial.add extensionOps.toOps subtotal
                (← contribution (index / 2) low values)
              previous := none
          unless ((← stream.getLine).trimAscii.toString == "[]") && (← stream.getLine).isEmpty do
            throw (IO.userError "fresh prefix has missing terminator or extra data")
          if let some low := previous then
            let high := nextValues.getD (Vector.replicate Spec.ProductionRelation.matrixCount K.zero)
            subtotal := FixedPolynomial.add extensionOps.toOps subtotal
              (← contribution ((chunk.finish - 1) / 2) low high)
          return (subtotal, (← IO.monoNanosNow) - partStarted))
    for task in tasks do
      let (subtotal, elapsed) ← match ← IO.wait task with
        | .ok value => pure value
        | .error error => throw error
      total := FixedPolynomial.add extensionOps.toOps total subtotal
      report [("event", .str "second_fresh_chunk_complete"), ("elapsed_ns", Lean.toJson elapsed)]
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom 1, .atom (first / 2),
    .atom ((finish + 1) / 2), extensionValue challenge,
    .array (total.coefficients.map extensionValue)]).render ++ "\n")
  report [("event", .str (if first == 0 && finish == fullEnd then
    "second_fresh_complete" else "second_fresh_range_complete")),
    ("first_pair", Lean.toJson (first / 2)), ("end_pair", Lean.toJson ((finish + 1) / 2)),
    ("row_count", Lean.toJson rowCount),
    ("chunks", Lean.toJson chunks.size), ("workers", Lean.toJson workers),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def prefixPolynomial (path : System.FilePath) (degree expectedEnd : Nat)
    (challenge : K) : IO (FixedPolynomial K degree) := do
  let value ← composeRead path
  checked (do
    match (← value.getArr?).toList with
    | [schema, consumed, first, finish, coin, coefficients] =>
        unless (← schema.getNat?) == 1 && (← consumed.getNat?) == 1 &&
            (← first.getNat?) == 0 && (← finish.getNat?) == expectedEnd do
          throw "prefix polynomial does not cover the selected extent"
        unless decide ((← composeExtension coin) = challenge) do
          throw "prefix polynomial challenge differs from the Lean transcript"
        composePolynomial degree coefficients
    | _ => throw "expected six prefix polynomial fields")

private def prefixMoments (paths : List String) (kind expectedEnd : Nat)
    (challenge : K) : IO (K × K) := do
  let mut ranges : Array (Nat × Nat × K × K) := #[]
  for path in paths do
    let value ← composeRead path
    let entry ← checked (do
      match (← value.getArr?).toList with
      | [schema, consumed, family, first, finish, coin, low, high] =>
          unless (← schema.getNat?) == 1 && (← consumed.getNat?) == 1 &&
              (← family.getNat?) == kind do throw "carried prefix family or schema differs"
          unless decide ((← composeExtension coin) = challenge) do
            throw "carried prefix challenge differs from the Lean transcript"
          pure (← first.getNat?, ← finish.getNat?, ← composeExtension low, ← composeExtension high)
      | _ => throw "expected eight carried prefix moment fields")
    ranges := ranges.push entry
  ranges := ranges.qsort (fun left right => decide (left.1 < right.1))
  let mut next := 0
  let mut low := K.zero
  let mut high := K.zero
  for (first, finish, lo, hi) in ranges do
    unless first == next && first < finish && finish ≤ expectedEnd do
      throw (IO.userError "carried prefix coverage has a gap, overlap or excessive range")
    low := extensionOps.add low lo
    high := extensionOps.add high hi
    next := finish
  unless next == expectedEnd do throw (IO.userError "carried prefix coverage is incomplete")
  return (low, high)

private def composeSecond (publicPath roundPath freshPath normPath outputPath : System.FilePath)
    (matrixPaths padPaths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let publicInput ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre publicInput)
  let challenge ← savedFirstChallenge coins roundPath
  let saved ← checked ((← composeRead roundPath).getArr?)
  let previous ← checked (composePolynomial 9 (saved[4]?.getD .null))
  let (_, state) := PiCCSPublicReplay.firstRound coins.state previous
  let input := PiCCSPublicReplay.verifierInput publicInput
  let rows := (PerApplicationMatrixProgram.matrixProgram
    Poseidon2HashChainV1Package.application).rowCount
  let fresh ← prefixPolynomial freshPath 9 (((rows + 1) / 2 + 1) / 2) challenge
  let norm ← prefixPolynomial normPath 3 ((PiCCSSourceImages.shape.carrierWidth + 3) / 4) challenge
  let matrix ← prefixMoments matrixPaths 1 ((rows + 1) / 2) challenge
  let pad ← prefixMoments padPaths 0 (27 * PiCCSSourceImages.blockCount) challenge
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  let head := fun target : CubePoint K cubeVariables =>
    FixedPolynomial.scale extensionOps.toOps
      (PiCCSPrefixSelector.consumedFactor extensionOps [challenge] target)
      (PiCCSCarriedMoments.headSelector extensionOps (PiCCSPrefixSelector.dropPoint target 1))
  let normTerm : FixedPolynomial K 9 :=
    FixedPolynomial.scale extensionOps.toOps (power productionShape.constraintOffset)
      (FixedPolynomial.scale extensionOps.toOps (power productionShape.freshCount)
        (FixedPolynomial.widen extensionOps.toOps (by decide : 4 ≤ 9)
          (FixedPolynomial.mul extensionOps.toOps (head coins.alpha) norm)))
  let carried : FixedPolynomial K 9 :=
    PiCCSCarriedMoments.carriedPair extensionOps (by decide : 2 ≤ 9) (head input.priorPoint)
      (power productionShape.matrixEvaluationOffset) pad.1 pad.2 matrix.1 matrix.2
  let polynomial := FixedPolynomial.add extensionOps.toOps carried
    (FixedPolynomial.add extensionOps.toOps fresh normTerm)
  let initial := previous.evaluate extensionOps.toOps challenge
  let endpoints := extensionOps.add (polynomial.evaluate extensionOps.toOps K.zero)
    (polynomial.evaluate extensionOps.toOps K.one)
  unless decide (endpoints = initial) do throw (IO.userError "second-round sum differs from the first-round claim")
  let index : Fin productionShape.cubeVariables := ⟨1, by decide⟩
  let absorbed := Transcript.piCcsOracle.transcript.absorbRound state index polynomial.toMessage
  let (nextChallenge, nextState) := Transcript.piCcsOracle.transcript.squeeze absorbed (.sumcheck index)
  let stateValue := fun current : Transcript.State => Value.array (current.map (fun word => .atom word.val))
  IO.FS.writeFile outputPath ((Value.array [.atom 1,
    .array (coins.alpha.coordinates.map extensionValue), extensionValue coins.gamma,
    stateValue state, .array (polynomial.coefficients.map extensionValue),
    extensionValue nextChallenge, stateValue nextState, extensionValue initial,
    extensionValue endpoints, extensionValue (polynomial.evaluate extensionOps.toOps nextChallenge)]).render ++ "\n")
  report [("event", .str "second_round_composed"), ("coefficients", .num 10),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def replayFirstNormFold
    (publicPath sourcePath roundPath outputPath : System.FilePath)
    (first finish : Nat) (reference : Bool) : IO UInt32 := do
  unless first < finish && finish ≤ PiCCSSourceImages.blockCount do
    throw (IO.userError "invalid norm fold block range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let challenge ← savedFirstChallenge coins roundPath
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let table := PiCCSSignedFirstFold.prepare challenge
  let count := (finish - first) * ringDegree
  let width := ringDegree / 2
  let output ← IO.FS.Handle.mk outputPath .write
  output.putStrLn ((Value.array [.atom 1, .atom productionShape.sourceCount,
    .atom first, .atom finish, extensionValue challenge]).render)
  report [("event", .str "first_norm_fold_sources_ready"), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let mut computeNs := 0
  for sourceIndex in [:productionShape.sourceCount] do
    if sourceBound : sourceIndex < productionShape.sourceCount then
      let source : Fin productionShape.sourceCount := ⟨sourceIndex, sourceBound⟩
      let code := fun index : Nat =>
        let column := first * ringDegree + index
        PiCCSNormSource.sourceCode (masks[column / ringDegree]?.getD #[]) source
          ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩
      let computeStarted ← IO.monoNanosNow
      let values ← IO.wait (Task.spawn fun _ =>
        if reference then
          PrefixFold.foldOne extensionOps
            (Array.ofFn fun index : Fin count =>
              let column := first * ringDegree + index.val
              K.embed (SignedUnitSourceInput.scalar
                (masks[column / ringDegree]?.getD #[]) ⟨sourceIndex, sourceBound⟩
                ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩)) challenge
        else PiCCSSignedFirstFold.foldOne table count code)
      let elapsed := (← IO.monoNanosNow) - computeStarted
      computeNs := computeNs + elapsed
      unless values.size == (finish - first) * width do
        throw (IO.userError "unexpected folded source width")
      for block in [:(finish - first)] do
        let coefficients := (values.extract (block * width) ((block + 1) * width)).toList
        output.putStrLn ((Value.array [.atom sourceIndex, .atom ((first + block) * width),
          .array (coefficients.map extensionValue)]).render)
      report [("event", .str "first_norm_fold_source_written"),
        ("source", Lean.toJson sourceIndex), ("values", Lean.toJson values.size),
        ("compute_ns", Lean.toJson elapsed)]
  output.putStrLn "[]"
  report [("event", .str "first_norm_fold_complete"), ("reference", Lean.toJson reference),
    ("first_block", Lean.toJson first), ("end_block", Lean.toJson finish),
    ("sources", Lean.toJson productionShape.sourceCount),
    ("values", Lean.toJson (productionShape.sourceCount * (finish - first) * width)),
    ("compute_ns", Lean.toJson computeNs),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def replayNormAfterFirst
    (publicPath sourcePath roundPath outputPath : System.FilePath)
    (first finish : Nat) (reference : Bool) : IO UInt32 := do
  let groups := (PiCCSSourceImages.blockCount * ringDegree + 3) / 4
  unless first < finish && finish ≤ groups do
    throw (IO.userError "invalid second-round norm pair range")
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let started ← IO.monoNanosNow
  let input ← checked (PiCCSPublicReplay.parse (← IO.FS.readFile publicPath))
  let coins ← IO.wait (Task.spawn fun _ => PiCCSPublicReplay.pre input)
  let challenge ← savedFirstChallenge coins roundPath
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  let tail := coins.alpha.coordinates.drop 2
  let weights := PiCCSTensorWeights.prepare extensionOps tail
  let weight := PiCCSTensorWeights.lookup extensionOps tail weights
  let powers := PiCCSGammaPowers.prepare extensionOps.toOps coins.gamma
    (PiCCSFirstRoundPair.powerCount productionShape)
  let power := PiCCSGammaPowers.lookup extensionOps.toOps coins.gamma powers
  report [("event", .str "second_norm_sources_ready"), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let computeStarted ← IO.monoNanosNow
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let mut total := FixedPolynomial.zero extensionOps.toOps 3
  for batch in [:(productionShape.sourceCount + workers - 1) / workers] do
    let mut tasks : Array (Task (Except IO.Error (FixedPolynomial K 3 × Nat))) := #[]
    for sourceIndex in [batch * workers:min productionShape.sourceCount ((batch + 1) * workers)] do
      if sourceBound : sourceIndex < productionShape.sourceCount then
        let source : Fin productionShape.sourceCount := ⟨sourceIndex, sourceBound⟩
        tasks := tasks.push (← IO.asTask (prio := Task.Priority.dedicated) do
          let sourceStarted ← IO.monoNanosNow
          let code := fun column : Nat =>
            PiCCSNormSource.sourceCode (masks[column / ringDegree]?.getD #[]) source
              ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩
          let scalar := fun column : Nat =>
            K.embed (SignedUnitSourceInput.scalar (masks[column / ringDegree]?.getD #[])
              source ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩)
          let cubic := if reference then
              PiCCSPolynomialRange.range extensionOps first (finish - first) fun group =>
                FixedPolynomial.scale extensionOps.toOps (weight group)
                  (PiCCSFirstRoundPair.normPair extensionOps
                    (PrefixFold.interpolate extensionOps challenge (scalar (4 * group)) (scalar (4 * group + 1)))
                    (PrefixFold.interpolate extensionOps challenge (scalar (4 * group + 2)) (scalar (4 * group + 3))))
            else
              PiCCSPrefixNorm.range challenge code weight first (finish - first)
          return (FixedPolynomial.scale extensionOps.toOps (power source.val) cubic,
            (← IO.monoNanosNow) - sourceStarted))
    let mut sourceIndex := batch * workers
    for task in tasks do
      let (cubic, elapsed) ← match ← IO.wait task with
        | .ok result => pure result
        | .error error => throw error
      total := FixedPolynomial.add extensionOps.toOps total cubic
      report [("event", .str "second_norm_source_complete"),
        ("source", Lean.toJson sourceIndex), ("compute_ns", Lean.toJson elapsed)]
      sourceIndex := sourceIndex + 1
  IO.FS.writeFile outputPath ((Value.array [.atom 1, .atom 1, .atom first, .atom finish,
    extensionValue challenge, .array (total.coefficients.map extensionValue)]).render ++ "\n")
  report [("event", .str "second_norm_complete"), ("first", Lean.toJson first),
    ("end", Lean.toJson finish), ("reference", Lean.toJson reference),
    ("sources", Lean.toJson productionShape.sourceCount), ("workers", Lean.toJson workers),
    ("compute_ns", Lean.toJson ((← IO.monoNanosNow) - computeStarted)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

end NightstreamFPrime.Export.PiCCSFirstRoundReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | "compose-second" :: publicPath :: roundPath :: freshPath :: normPath :: outputPath :: paths =>
      let (matrixPaths, rest) := paths.span (fun path => path != "pad")
      match rest with
      | "pad" :: padPaths =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.composeSecond
            publicPath roundPath freshPath normPath outputPath matrixPaths padPaths
      | _ => throw (IO.userError "compose-second requires matrix moment paths, pad, and Pad moment paths")
  | ["carried-pad-prefix", publicPath, sourcePath, roundPath, directory, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.saveCarriedPrefix
            publicPath sourcePath roundPath directory first finish true
      | _, _ => throw (IO.userError "block bounds must be natural numbers")
  | ["carried-matrix-prefix", publicPath, sourcePath, roundPath, directory, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.saveCarriedPrefix
            publicPath sourcePath roundPath directory first finish false
      | _, _ => throw (IO.userError "row-pair bounds must be natural numbers")
  | ["fresh-after-first", publicPath, roundPath, directory, outputPath] =>
      NightstreamFPrime.Export.PiCCSFirstRoundReplay.freshAfterFirst
        publicPath roundPath directory outputPath
  | ["fresh-after-first-range", publicPath, roundPath, directory, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.freshAfterFirst
            publicPath roundPath directory outputPath (some (first, finish))
      | _, _ => throw (IO.userError "prefix bounds must be natural numbers")
  | ["fresh-prefix", publicPath, sourcePath, roundPath, outputDirectory, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.saveFreshPrefix
            publicPath sourcePath roundPath outputDirectory first finish
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | ["fresh-prefix-reference", publicPath, sourcePath, roundPath, outputDirectory, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.saveFreshPrefix
            publicPath sourcePath roundPath outputDirectory first finish true
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | ["norm-after-first", publicPath, sourcePath, roundPath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayNormAfterFirst
            publicPath sourcePath roundPath outputPath first finish false
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | ["norm-after-first-reference", publicPath, sourcePath, roundPath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayNormAfterFirst
            publicPath sourcePath roundPath outputPath first finish true
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | ["fold-norm-prefix", publicPath, sourcePath, roundPath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayFirstNormFold
            publicPath sourcePath roundPath outputPath first finish false
      | _, _ => throw (IO.userError "block bounds must be natural numbers")
  | ["fold-norm-prefix-reference", publicPath, sourcePath, roundPath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayFirstNormFold
            publicPath sourcePath roundPath outputPath first finish true
      | _, _ => throw (IO.userError "block bounds must be natural numbers")
  | "compose" :: publicPath :: freshPath :: normPath :: outputPath :: momentPaths =>
      let (matrixPaths, rest) := momentPaths.span (fun path => path != "pad")
      match rest with
      | "pad" :: padPaths =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.composeRound
            publicPath freshPath normPath outputPath matrixPaths padPaths
      | _ => throw (IO.userError "compose requires matrix moment paths followed by pad and Pad moment paths")

  | ["carried-pad", publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayCarriedPad
            publicPath sourcePath outputPath first finish
      | _, _ => throw (IO.userError "block bounds must be natural numbers")
  | ["carried-matrix-reference", publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayCarriedMatrix
            publicPath sourcePath outputPath first finish false
      | _, _ => throw (IO.userError "row bounds must be natural numbers")
  | ["carried-matrix", publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayCarriedMatrix
            publicPath sourcePath outputPath first finish
      | _, _ => throw (IO.userError "row bounds must be natural numbers")
  | ["fresh-reference", publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayFresh
            publicPath sourcePath outputPath first finish true
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | ["fresh", publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayFresh
            publicPath sourcePath outputPath first finish
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | ["norm", publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replayNorm
            publicPath sourcePath outputPath first finish
      | _, _ => throw (IO.userError "block bounds must be natural numbers")
  | [publicPath, sourcePath, outputPath, first, finish] =>
      match first.toNat?, finish.toNat? with
      | some first, some finish =>
          NightstreamFPrime.Export.PiCCSFirstRoundReplay.replay
            publicPath sourcePath outputPath first finish
      | _, _ => throw (IO.userError "pair bounds must be natural numbers")
  | _ =>
      IO.eprintln "usage: replayPiCCSFirstRound <public-input> <original-sources> <new-output> <first-pair> <end-pair>"
      IO.eprintln "       replayPiCCSFirstRound norm <public-input> <original-sources> <new-output> <first-block> <end-block>"
      IO.eprintln "       replayPiCCSFirstRound fresh <public-input> <original-sources> <new-output> <first-pair> <end-pair>"
      IO.eprintln "       replayPiCCSFirstRound fresh-reference <public-input> <original-sources> <new-output> <first-pair> <end-pair>"
      IO.eprintln "       replayPiCCSFirstRound carried-matrix <public-input> <original-sources> <new-output> <first-row> <end-row>"
      IO.eprintln "       replayPiCCSFirstRound carried-matrix-reference <public-input> <original-sources> <new-output> <first-row> <end-row>"
      IO.eprintln "       replayPiCCSFirstRound carried-pad <public-input> <original-sources> <new-output> <first-block> <end-block>"
      IO.eprintln "       replayPiCCSFirstRound compose <public-input> <fresh> <norm> <new-output> <matrix-moment>... pad <Pad-moment>..."
      IO.eprintln "       replayPiCCSFirstRound fold-norm-prefix[-reference] <public-input> <original-sources> <Lean-round-zero> <new-output> <first-block> <end-block>"
      IO.eprintln "       replayPiCCSFirstRound norm-after-first[-reference] <public-input> <original-sources> <Lean-round-zero> <new-output> <first-pair> <end-pair>"
      IO.eprintln "       replayPiCCSFirstRound fresh-prefix[-reference] <public-input> <original-sources> <Lean-round-zero> <new-output-directory> <first-pair> <end-pair>"
      IO.eprintln "       replayPiCCSFirstRound fresh-after-first <public-input> <Lean-round-zero> <fresh-prefix-directory> <new-output>"
      IO.eprintln "       replayPiCCSFirstRound fresh-after-first-range <public-input> <Lean-round-zero> <fresh-prefix-directory> <new-output> <first-row> <end-row>"
      return 2
