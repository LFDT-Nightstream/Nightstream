import NightstreamFPrime.Export.PiDECParentInput
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Stage1.PiDECParentIntRead
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
import NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange
import NightstreamFPrime.Export.Stage1.PiDECParentMagnitude
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PiDECProductRow
import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows
import NightstreamFPrime.Export.Codec

/-!
Execute a contiguous selected matrix-row range from the independent
Lean parent. The accepted C execution supplies the point. Every child, matrix
port and Phi81 output coefficient is computed; native outputs are not inputs.
This is a staged matrix replay result, not a complete matrix-family result.
The guard's equality proofs are in PiDECMatrixZeroRead. Importing those
reference-plan proofs here would initialize the full plan at startup.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECMatrixReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)
open NightstreamFPrime.Export.Codec

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def report (fields : List (String × Lean.Json)) : IO Unit := do
  IO.println (Lean.Json.mkObj fields).compress
  (← IO.getStdout).flush

private structure RangeRequest where
  outputPath : System.FilePath
  blockIndex : Nat
  firstRow : Nat
  lastRow : Nat

private def parseRangeRequests : List String → Except String (List RangeRequest)
  | [] => pure []
  | output :: block :: first :: last :: rest => do
      let (some block, some first, some last) := (block.toNat?, first.toNat?, last.toNat?)
        | throw "block and row bounds must be natural numbers"
      let remaining ← parseRangeRequests rest
      pure ({ outputPath := output, blockIndex := block, firstRow := first, lastRow := last } ::
        remaining)
  | _ => throw "expected complete output/block/first/last groups"

private def ranges (ccsPath : System.FilePath) (requests : List RangeRequest)
    (parentPaths : List String) (batch : Bool) : IO UInt32 := do
  unless !requests.isEmpty do throw (IO.userError "expected at least one matrix range")
  let mut outputNames : Array String := #[]
  for request in requests do
    let name := request.outputPath.toString
    if outputNames.contains name then throw (IO.userError "duplicate matrix output path")
    unless !(← request.outputPath.pathExists) do throw (IO.userError "output already exists")
    outputNames := outputNames.push name
  let started ← IO.monoNanosNow
  let ccs ← checked (PiCCSInputCheck.parse (← IO.FS.readFile ccsPath))
  let phase ← IO.wait (Task.spawn fun _ => PiCCSInputCheck.execute ccs)
  unless phase.accepted do throw (IO.userError "C input rejected")
  report [("event", .str "point_ready"),
    ("independence", .str "given Lean-verified Rust PiCCS output"),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  let parentStarted ← IO.monoNanosNow
  let (parents, records) ← PiDECParentInput.read parentPaths
  report [("event", .str "parent_ready"), ("storage", .str "centered_integer"),
    ("blocks", Lean.toJson parents.size), ("records", Lean.toJson records),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - parentStarted))]
  let magnitudeStarted ← IO.monoNanosNow
  let maximum ← IO.wait (Task.spawn fun _ =>
    PiDECParentMagnitude.maximumMagnitude parents)
  report [("event", .str "parent_magnitude_ready"), ("maximum", Lean.toJson maximum),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - magnitudeStarted))]
  let tablesStarted ← IO.monoNanosNow
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  report [("event", .str "basis_ready"),
    ("retained_coefficients", Lean.toJson ((List.ofFn fun basis : Fin ringDegree =>
      ((List.ofFn fun output : Fin ringDegree =>
        ((tables.get basis).get output).entries.length)).sum)).sum),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - tablesStarted))]
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  if batch then
    for request in requests do
      let some selected := program.blocks[request.blockIndex]?
        | throw (IO.userError "invalid selected matrix block")
      unless request.firstRow < request.lastRow ∧ request.lastRow ≤ selected.rowCount do
        throw (IO.userError "invalid selected matrix row range")
      let blockStart :=
        ((program.blocks.take request.blockIndex).map MatrixProgram.Block.rowCount).sum
      unless blockStart + request.lastRow ≤ program.rowCount do
        throw (IO.userError "range exceeds matrix row domain")
      match selected with
      | .poseidon _ =>
          unless request.firstRow % 94 = 0 && request.lastRow % 94 = 0 do
            throw (IO.userError "Poseidon range must contain complete 94-row invocations")
      | .phi81Product _ =>
          unless request.firstRow % 108 = 0 && request.lastRow % 108 = 0 do
            throw (IO.userError "Phi81 range must contain complete 108-row invocations")
      | _ => pure ()
  let sharedLoadNs := (← IO.monoNanosNow) - started
  if batch then
    report [("event", .str "shared_load_complete"), ("ranges", Lean.toJson requests.length),
      ("shared_load_ns", Lean.toJson sharedLoadNs)]
  let runRange : RangeRequest → IO Unit := fun request => do
    let rangeStarted ← IO.monoNanosNow
    let outputPath := request.outputPath
    let blockIndex := request.blockIndex
    let firstRow := request.firstRow
    let lastRow := request.lastRow
    unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
    let some selected := program.blocks[blockIndex]?
      | throw (IO.userError "invalid selected matrix block")
    if bounds : firstRow < lastRow ∧ lastRow ≤ selected.rowCount then
      let count := lastRow - firstRow
      let logicalWidth := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
      let blockStart := ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum
      let first := blockStart + firstRow
      let finish := blockStart + lastRow
      unless finish ≤ program.rowCount do throw (IO.userError "range exceeds matrix row domain")
      let readChild := fun (child : Fin productionGlobalParams.k)
          (output : Fin ringDegree) (column : Fin logicalWidth) =>
        if live : column.val / ringDegree < Poseidon2HashChainV1Setup.messageColumns then
          PiDECParentIntRead.sparseRead tables
            (parents.get ⟨column.val / ringDegree, live⟩)
            ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ child output
        else 0
      let loadStarted ← IO.monoNanosNow
      let (unitCount, evaluate) : Nat × (Nat → Nat →
          Fin productionGlobalParams.k →
          Vector MaterializedRingK matrixCount) ← match selectedEq : selected with
        | .poseidon block => do
            unless firstRow % 94 = 0 && lastRow % 94 = 0 do
              throw (IO.userError "Poseidon range must contain complete 94-row invocations")
            let invocations := lastRow / 94 - firstRow / 94
            have blockBound : lastRow ≤ block.invocationCount * 94 := by
              simpa only [selectedEq, MatrixProgram.Block.rowCount,
                MatrixProgram.Poseidon.Block.rowCount] using bounds.2
            let interfaces ← Vector.ofFnM fun (index : Fin invocations) => do
              have invBound : firstRow / 94 + index.val < block.invocationCount := by
                dsimp only [invocations] at index
                omega
              let some interface := PiDECPoseidonNumericBlock.loadInvocation?
                  block logicalWidth ⟨firstRow / 94 + index.val, invBound⟩
                | throw (IO.userError "selected invocation interface rejected")
              pure interface
            pure (invocations, fun lo hi child =>
              PiDECMatrixInvocationRange.sum (first + 94 * lo) phase.point (readChild child)
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
              pure (count, fun lo hi child =>
                PiDECMatrixSparseRange.sum (first + lo) phase.point (readChild child)
                  (forms.extract lo hi))
            else throw (IO.userError "Phi81 range must contain complete 108-row invocations")
        | other => do
            let cache ← IO.wait (Task.spawn fun _ =>
              PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
            let source := fun row => cache[row]?
            let forms ← Vector.ofFnM fun (index : Fin count) => do
              let some row := other.row? logicalWidth source (firstRow + index.val)
                | throw (IO.userError "selected sparse row rejected")
              pure row
            pure (count, fun lo hi child =>
              PiDECMatrixSparseRange.sum (first + lo) phase.point (readChild child)
                (forms.extract lo hi))
      report ([("event", .str "range_begin"), ("block", Lean.toJson blockIndex),
        ("block_rows", Lean.toJson selected.rowCount),
        ("first_local_row", Lean.toJson firstRow), ("last_local_row_exclusive", Lean.toJson lastRow),
        ("start", Lean.toJson first), ("end", Lean.toJson finish),
        ("load_ns", Lean.toJson ((← IO.monoNanosNow) - loadStarted))] ++
        if batch then [("output", .str outputPath.toString)] else [])
      unless 0 < unitCount do throw (IO.userError "empty selected matrix unit range")
      let workers := max 1 (((← IO.getEnv "LEAN_NUM_THREADS").bind String.toNat?).getD 1)
      let children := List.finRange productionGlobalParams.k
      let activeCount := (children.filter fun child =>
        decide (¬ maximum < 2 ^ child.val)).length
      let arithmeticStarted ← IO.monoNanosNow
      let mut tasks := #[]
      let mut activeRank := 0
      let mut taskCount := 0
      for child in children do
        let mut childTasks := #[]
        unless maximum < 2 ^ child.val do
          let parts := min unitCount (max 1
            (workers / activeCount + if activeRank < workers % activeCount then 1 else 0))
          for slice in [:parts] do
            let lo := unitCount * slice / parts
            let hi := unitCount * (slice + 1) / parts
            childTasks := childTasks.push (← IO.asTask do
              let sliceStarted ← IO.monoNanosNow
              let values ← IO.wait (Task.spawn fun _ => evaluate lo hi child)
              let sliceFinished ← IO.monoNanosNow
              return (values, sliceStarted, sliceFinished))
          activeRank := activeRank + 1
          taskCount := taskCount + parts
        tasks := tasks.push childTasks
      report [("event", .str "slices_queued"), ("workers", Lean.toJson workers),
        ("active_children", Lean.toJson activeCount), ("units", Lean.toJson unitCount),
        ("tasks", Lean.toJson taskCount),
        ("queue_ns", Lean.toJson ((← IO.monoNanosNow) - arithmeticStarted))]
      let mut allValues : Array (Vector MaterializedRingK matrixCount) := #[]
      for child in children do
        let childTasks := tasks[child.val]!
        let mut total := PiDECEvaluationBatch.zero matrixCount
        let mut firstCompute : Option Nat := none
        let mut lastCompute := 0
        let mut sliceComputeNs := 0
        let mut mergeNs := 0
        for slice in [:childTasks.size] do
          let result ← IO.wait childTasks[slice]!
          let (partValues, sliceStarted, sliceFinished) ← match result with
            | .ok value => pure value
            | .error error => throw error
          let mergeStarted ← IO.monoNanosNow
          total := PiDECEvaluationBatch.add total partValues
          let mergeElapsed := (← IO.monoNanosNow) - mergeStarted
          firstCompute := some (match firstCompute with
            | none => sliceStarted
            | some earlier => min earlier sliceStarted)
          lastCompute := max lastCompute sliceFinished
          sliceComputeNs := sliceComputeNs + (sliceFinished - sliceStarted)
          mergeNs := mergeNs + mergeElapsed
          report [("event", .str "slice_complete"), ("child", Lean.toJson child.val),
            ("slice", Lean.toJson slice),
            ("first_unit", Lean.toJson (unitCount * slice / childTasks.size)),
            ("last_unit_exclusive", Lean.toJson (unitCount * (slice + 1) / childTasks.size)),
            ("start_ns", Lean.toJson (sliceStarted - arithmeticStarted)),
            ("compute_ns", Lean.toJson (sliceFinished - sliceStarted)),
            ("merge_ns", Lean.toJson mergeElapsed)]
        let values := PiDECParentMagnitude.ifActive maximum child (fun _ => total)
        let computeNs := match firstCompute with
          | none => 0
          | some earliest => lastCompute - earliest
        allValues := allValues.push values
        report [("event", .str "child_complete"), ("child", Lean.toJson child.val),
          ("zero_from_parent_bound", .bool (decide (maximum < 2 ^ child.val))),
          ("compute_ns", Lean.toJson computeNs), ("slices", Lean.toJson childTasks.size),
          ("slice_compute_ns", Lean.toJson sliceComputeNs), ("merge_ns", Lean.toJson mergeNs)]
      let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
      let output := Value.array [.atom 1, .atom program.rowCount, .atom first, .atom finish,
        .array (phase.point.coordinates.map encodeK),
        .array (allValues.toList.map fun matrices =>
          .array (List.ofFn fun matrix : Fin matrixCount =>
            .array (List.ofFn fun lane : Fin ringDegree =>
              encodeK ((matrices.get matrix).toRing lane))))]
      IO.FS.writeFile outputPath (output.render ++ "\n")
      let completed ← IO.monoNanosNow
      report ([("event", .str "range_complete"),
        ("children", Lean.toJson allValues.size), ("matrices", Lean.toJson matrixCount),
        ("field_words", Lean.toJson (productionGlobalParams.k * matrixCount * ringDegree * 2)),
        ("total_ns", Lean.toJson (completed - if batch then rangeStarted else started))] ++
        if batch then [
          ("output", .str outputPath.toString), ("block", Lean.toJson blockIndex),
          ("first_local_row", Lean.toJson firstRow), ("last_local_row_exclusive", Lean.toJson lastRow),
          ("start", Lean.toJson first), ("end", Lean.toJson finish),
          ("timing_scope", .str "range_after_shared_load")]
        else [])
      pure ()
    else throw (IO.userError "invalid selected matrix row range")
  for request in requests do
    runRange request
  if batch then
    report [("event", .str "batch_complete"), ("ranges", Lean.toJson requests.length),
      ("shared_load_ns", Lean.toJson sharedLoadNs),
      ("total_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
  return 0

private def range (ccsPath outputPath : System.FilePath)
    (blockIndex firstRow lastRow : Nat) (parentPaths : List String) : IO UInt32 :=
  ranges ccsPath [{ outputPath, blockIndex, firstRow, lastRow }] parentPaths false

private def rangesCommand (ccsPath : System.FilePath) (arguments : List String) : IO UInt32 := do
  let (requestArguments, parentArguments) := arguments.span (fun argument => argument != "--")
  let "--" :: parents := parentArguments
    | throw (IO.userError "ranges requires -- before the Lean parent paths")
  unless !parents.isEmpty do throw (IO.userError "ranges requires Lean parent paths")
  let requests ← checked (parseRangeRequests requestArguments)
  ranges ccsPath requests parents true

end NightstreamFPrime.Export.PiDECMatrixReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | "ranges" :: ccsPath :: remaining =>
      NightstreamFPrime.Export.PiDECMatrixReplay.rangesCommand ccsPath remaining
  | ccsPath :: outputPath :: block :: first :: last :: parents =>
      match block.toNat?, first.toNat?, last.toNat? with
      | some block, some first, some last =>
          NightstreamFPrime.Export.PiDECMatrixReplay.range
            ccsPath outputPath block first last parents
      | _, _, _ => throw (IO.userError "block and row bounds must be natural numbers")
  | _ => throw (IO.userError "usage: replayPiDECMatrix <C-input> <new-output> <block> <first-local-row> <last-exclusive> <Lean-parent-ranges>...")
