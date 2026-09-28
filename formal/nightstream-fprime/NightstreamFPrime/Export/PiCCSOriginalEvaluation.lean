import NightstreamFPrime.Export.Stage1.PiCCSOriginalPad
import NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixBatch
import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported
import NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PiDECProductRow
import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows
import NightstreamFPrime.Export.Codec

/-!
Original-source final matrix evaluations at the causally replayed PiCCS point.
The caller supplies that Lean-derived point. Every source, port and ring lane
is retained. This module never accepts Rust results or mixed parent witnesses.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiCCSOriginalEvaluation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)
open NightstreamFPrime.Export.Codec

structure RangeRequest where
  outputPath : System.FilePath
  blockIndex : Nat
  firstRow : Nat
  lastRow : Nat

def parseRangeRequests : List String → Except String (List RangeRequest)
  | [] => pure []
  | output :: block :: first :: last :: rest => do
      let (some block, some first, some last) := (block.toNat?, first.toNat?, last.toNat?)
        | throw "block and row bounds must be natural numbers"
      let remaining ← parseRangeRequests rest
      pure ({ outputPath := output, blockIndex := block, firstRow := first, lastRow := last } ::
        remaining)
  | _ => throw "expected complete output/block/first/last groups"

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private def referenceBatch
    (evaluate : Fin productionShape.sourceCount →
      Vector MaterializedRingK matrixCount) : PiCCSOriginalMatrixBatch.Batch :=
  let bySource := Vector.ofFn evaluate
  Vector.ofFn fun code =>
    let pair : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
    (bySource.get pair.1).get pair.2

def matrixRanges (point : PaperAlgebra.Point) (sourcePath : System.FilePath)
    (requests : List RangeRequest) (reference : Bool) : IO UInt32 := do
  unless !requests.isEmpty do throw (IO.userError "expected at least one original matrix range")
  let mut outputNames : Array String := #[]
  for request in requests do
    let name := request.outputPath.toString
    if outputNames.contains name then throw (IO.userError "duplicate matrix output path")
    unless !(← request.outputPath.pathExists) do throw (IO.userError "output already exists")
    outputNames := outputNames.push name
  let started ← IO.monoNanosNow
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  for request in requests do
    let some selected := program.blocks[request.blockIndex]?
      | throw (IO.userError "invalid selected matrix block")
    unless request.firstRow < request.lastRow ∧ request.lastRow ≤ selected.rowCount do
      throw (IO.userError "invalid selected matrix row range")
    match selected with
    | .poseidon _ =>
        unless request.firstRow % 86 = 0 && request.lastRow % 86 = 0 do
          throw (IO.userError "Poseidon range must contain complete 86-row invocations")
    | .phi81Product _ =>
        unless request.firstRow % 108 = 0 && request.lastRow % 108 = 0 do
          throw (IO.userError "Phi81 range must contain complete 108-row invocations")
    | _ => pure ()
  let sourceStarted ← IO.monoNanosNow
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  report [("event", .str "original_sources_ready"), ("records", Lean.toJson records),
    ("blocks", Lean.toJson masks.size),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - sourceStarted))]
  let supportStarted ← IO.monoNanosNow
  let zeroSources ← IO.wait (Task.spawn fun _ =>
    Vector.ofFn fun source : Fin productionShape.sourceCount =>
      PiCCSOriginalSupport.isZero masks source)
  report [("event", .str "original_source_support"),
    ("zero_sources", Lean.toJson (List.ofFn fun source => zeroSources.get source)),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - supportStarted))]
  let tablesStarted ← IO.monoNanosNow
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  report [("event", .str "basis_ready"),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - tablesStarted))]
  let sharedLoadNs := (← IO.monoNanosNow) - started
  report [("event", .str "shared_load_complete"), ("ranges", Lean.toJson requests.length),
    ("shared_load_ns", Lean.toJson sharedLoadNs), ("reference", .bool reference)]
  let logicalWidth := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
  let read := fun (source : Fin productionShape.sourceCount) (output : Fin ringDegree)
      (column : Fin logicalWidth) => PiCCSOriginalReads.read tables masks source output column
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  for request in requests do
    let rangeStarted ← IO.monoNanosNow
    let some selected := program.blocks[request.blockIndex]?
      | throw (IO.userError "invalid selected matrix block")
    if bounds : request.firstRow < request.lastRow ∧ request.lastRow ≤ selected.rowCount then
      let firstRow := request.firstRow
      let lastRow := request.lastRow
      let count := lastRow - firstRow
      let blockStart := ((program.blocks.take request.blockIndex).map MatrixProgram.Block.rowCount).sum
      let first := blockStart + firstRow
      let finish := blockStart + lastRow
      unless finish ≤ program.rowCount do throw (IO.userError "range exceeds matrix row domain")
      let loadStarted ← IO.monoNanosNow
      let (unitCount, evaluate) : Nat × (Nat → Nat → PiCCSOriginalMatrixBatch.Batch) ←
        match selectedEq : selected with
        | .poseidon block => do
            let invocations := lastRow / 86 - firstRow / 86
            have blockBound : lastRow ≤ block.invocationCount * 86 := by
              simpa only [selectedEq, MatrixProgram.Block.rowCount,
                MatrixProgram.Poseidon.Block.rowCount] using bounds.2
            let interfaces ← Vector.ofFnM fun (index : Fin invocations) => do
              have invBound : firstRow / 86 + index.val < block.invocationCount := by
                dsimp only [invocations] at index
                omega
              let some interface := PiDECPoseidonNumericBlock.loadInvocation?
                  block logicalWidth ⟨firstRow / 86 + index.val, invBound⟩
                | throw (IO.userError "selected invocation interface rejected")
              pure interface
            pure (invocations, fun lo hi =>
              if reference then referenceBatch fun source =>
                PiDECMatrixInvocationRange.sum (first + 86 * lo) point (read source)
                  (interfaces.extract lo hi)
              else PiCCSOriginalMatrixSupported.invocations zeroSources.get (first + 86 * lo) point read
                (interfaces.extract lo hi))
        | .phi81Product block => do
            if aligned : firstRow % 108 = 0 ∧ lastRow % 108 = 0 then
              let invocations := lastRow / 108 - firstRow / 108
              let interfaces ← Vector.ofFnM fun (index : Fin invocations) => do
                let some descriptor := MatrixProgram.Phi81Product.ringDescriptor?
                    block.families (firstRow / 108 + index.val)
                  | throw (IO.userError "selected product descriptor rejected")
                let some interface := PiDECProductInterface.interface? block logicalWidth descriptor
                  | throw (IO.userError "selected product interface rejected")
                pure interface
              let forms ← Vector.ofFnM fun (index : Fin count) => do
                have groupBound : index.val / 108 < invocations := by
                  dsimp only [count] at index
                  dsimp only [invocations]
                  omega
                let some row := PiDECProductRow.row?
                    (interfaces.get ⟨index.val / 108, groupBound⟩) (index.val % 108)
                  | throw (IO.userError "selected product row rejected")
                pure row.meaningfulForm
              pure (count, fun lo hi =>
                if reference then referenceBatch fun source =>
                  PiDECMatrixSparseRange.sum (first + lo) point (read source) (forms.extract lo hi)
                else PiCCSOriginalMatrixSupported.sparse zeroSources.get (first + lo) point read (forms.extract lo hi))
            else throw (IO.userError "Phi81 range must contain complete 108-row invocations")
        | other => do
            let cache ← IO.wait (Task.spawn fun _ =>
              PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
            let source := fun row => cache[row]?
            let forms ← Vector.ofFnM fun (index : Fin count) => do
              let some row := other.row? logicalWidth source (firstRow + index.val)
                | throw (IO.userError "selected sparse row rejected")
              pure row
            pure (count, fun lo hi =>
              if reference then referenceBatch fun source =>
                PiDECMatrixSparseRange.sum (first + lo) point (read source) (forms.extract lo hi)
              else PiCCSOriginalMatrixSupported.sparse zeroSources.get (first + lo) point read (forms.extract lo hi))
      report [("event", .str "range_begin"), ("block", Lean.toJson request.blockIndex),
        ("start", Lean.toJson first), ("end", Lean.toJson finish),
        ("load_ns", Lean.toJson ((← IO.monoNanosNow) - loadStarted))]
      unless 0 < unitCount do throw (IO.userError "empty selected matrix unit range")
      let parts := min unitCount workers
      let arithmeticStarted ← IO.monoNanosNow
      let mut tasks := #[]
      for slice in [:parts] do
        let lo := unitCount * slice / parts
        let hi := unitCount * (slice + 1) / parts
        tasks := tasks.push (Task.spawn (prio := Task.Priority.dedicated) fun _ => evaluate lo hi)
      let mut total := PiDECEvaluationBatch.zero (productionShape.sourceCount * matrixCount)
      for task in tasks do
        let values ← IO.wait task
        total := PiDECEvaluationBatch.add total values
      let computeNs := (← IO.monoNanosNow) - arithmeticStarted
      let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
      let output := Value.array [.atom 1, .atom program.rowCount, .atom first, .atom finish,
        .array (point.coordinates.map encodeK),
        .array (List.ofFn fun source : Fin productionShape.sourceCount =>
          .array (List.ofFn fun port : Fin matrixCount =>
            .array (List.ofFn fun lane : Fin ringDegree =>
              encodeK ((total.get (Fin.encodeProd (source, port))).toRing lane))))]
      unless !(← request.outputPath.pathExists) do throw (IO.userError "output already exists")
      IO.FS.writeFile request.outputPath (output.render ++ "\n")
      report [("event", .str "range_complete"), ("output", .str request.outputPath.toString),
        ("block", Lean.toJson request.blockIndex), ("start", Lean.toJson first),
        ("end", Lean.toJson finish), ("sources", Lean.toJson productionShape.sourceCount),
        ("field_words", Lean.toJson (productionShape.sourceCount * matrixCount * ringDegree * 2)),
        ("workers", Lean.toJson workers), ("tasks", Lean.toJson parts),
        ("compute_ns", Lean.toJson computeNs),
        ("range_ns", Lean.toJson ((← IO.monoNanosNow) - rangeStarted))]
    else throw (IO.userError "invalid selected matrix row range")
  report [("event", .str "original_matrix_complete"),
    ("shared_load_ns", Lean.toJson sharedLoadNs),
    ("total_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0


structure PadRequest where
  outputPath : System.FilePath
  firstBlock : Nat
  lastBlock : Nat

def parsePadRequests : List String → Except String (List PadRequest)
  | [] => pure []
  | output :: first :: last :: rest => do
      let (some first, some last) := (first.toNat?, last.toNat?)
        | throw "Pad block bounds must be natural numbers"
      let remaining ← parsePadRequests rest
      pure ({ outputPath := output, firstBlock := first, lastBlock := last } :: remaining)
  | _ => throw "expected complete output/first/last Pad groups"

def padRanges (point : PaperAlgebra.Point) (sourcePath : System.FilePath)
    (requests : List PadRequest) (reference : Bool) : IO UInt32 := do
  unless !requests.isEmpty do throw (IO.userError "expected at least one original Pad range")
  let mut outputNames : Array String := #[]
  for request in requests do
    unless request.firstBlock < request.lastBlock &&
        request.lastBlock ≤ PiCCSSourceImages.blockCount do
      throw (IO.userError "invalid original Pad block range")
    if outputNames.contains request.outputPath.toString then
      throw (IO.userError "duplicate Pad output path")
    unless !(← request.outputPath.pathExists) do throw (IO.userError "output already exists")
    outputNames := outputNames.push request.outputPath.toString
  let started ← IO.monoNanosNow
  let (masks, records) ← SignedUnitSourceInput.read sourcePath PiCCSSourceImages.blockCount
  report [("event", .str "original_sources_ready"), ("records", Lean.toJson records),
    ("blocks", Lean.toJson masks.size),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  let sharedLoadNs := (← IO.monoNanosNow) - started
  let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
  let referenceRows := fun index =>
    if live : index < PiCCSSourceImages.shape.carrierWidth then
      Vector.ofFn fun source : Fin productionShape.sourceCount =>
        Vector.ofFn fun output : Fin ringDegree =>
          PiCCSSourceImages.preparedRead tables (PiCCSOriginalReads.assignment masks source)
            output (⟨index, live⟩ : Fin PiCCSSourceImages.shape.carrierWidth)
    else Vector.replicate productionShape.sourceCount (Vector.replicate ringDegree (0 : F))
  for request in requests do
    let count := request.lastBlock - request.firstBlock
    let parts := min count workers
    let computeStarted ← IO.monoNanosNow
    let mut tasks := #[]
    for slice in [:parts] do
      let lo := count * slice / parts
      let hi := count * (slice + 1) / parts
      tasks := tasks.push (Task.spawn (prio := Task.Priority.dedicated) fun _ =>
        if reference then
          PiDECEvaluationBatch.range ((request.firstBlock + lo) * ringDegree)
            ((hi - lo) * ringDegree) point referenceRows
        else PiCCSOriginalPad.range (request.firstBlock + lo) (hi - lo) point masks)
    let mut total := PiDECEvaluationBatch.zero productionShape.sourceCount
    for task in tasks do
      total := PiDECEvaluationBatch.add total (← IO.wait task)
    let computeNs := (← IO.monoNanosNow) - computeStarted
    let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
    let output := Value.array [.atom 1, .atom PiCCSSourceImages.blockCount,
      .atom request.firstBlock, .atom request.lastBlock,
      .array (point.coordinates.map encodeK),
      .array (List.ofFn fun source : Fin productionShape.sourceCount =>
        .array (List.ofFn fun lane : Fin ringDegree =>
          encodeK ((total.get source).toRing lane)))]
    unless !(← request.outputPath.pathExists) do throw (IO.userError "output already exists")
    IO.FS.writeFile request.outputPath (output.render ++ "\n")
    report [("event", .str "pad_range_complete"), ("output", .str request.outputPath.toString),
      ("first", Lean.toJson request.firstBlock), ("end", Lean.toJson request.lastBlock),
      ("sources", Lean.toJson productionShape.sourceCount),
      ("field_words", Lean.toJson (productionShape.sourceCount * ringDegree * 2)),
      ("workers", Lean.toJson workers), ("tasks", Lean.toJson parts),
      ("reference", .bool reference), ("compute_ns", Lean.toJson computeNs)]
  report [("event", .str "original_pad_complete"), ("shared_load_ns", Lean.toJson sharedLoadNs),
    ("total_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

end NightstreamFPrime.Export.PiCCSOriginalEvaluation
