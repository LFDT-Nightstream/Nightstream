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

end NightstreamFPrime.Export.PiCCSFirstRoundReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
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
      return 2
