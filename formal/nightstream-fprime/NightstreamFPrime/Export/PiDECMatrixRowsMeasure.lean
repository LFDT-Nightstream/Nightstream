import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBlockSupport
import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PiDECProductRow

/-!
Measure existing matrix-row evaluation at the first and last valid row of
every compact program block. Poseidon rows use the proved numeric evaluator.
The coordinate-derived read is a feasibility probe, not witness conformance.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECMatrixRowsMeasure

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1

private def blockTag : MatrixProgram.Block → String
  | .ordinary _ => "ordinary"
  | .multiplicationGrid _ => "multiplicationGrid"
  | .phi81Product _ => "phi81Product"
  | .pin _ => "pin"
  | .poseidon _ => "poseidon"

private def emit (output : IO.FS.Handle) (value : Lean.Json) : IO Unit := do
  let line := value.compress
  output.putStrLn line
  output.flush
  IO.println line
  (← IO.getStdout).flush

private def measureRow (output : IO.FS.Handle) (program : MatrixProgram.Program)
    (logicalWidth : Nat) (sourceRow : Nat → Option R1CS.Row)
    (block : MatrixProgram.Block) (blockIndex blockStart ordinal : Nat)
    (boundary : String) : IO Unit := do
  emit output (Lean.Json.mkObj [
    ("event", .str "row_begin"), ("block", Lean.toJson blockIndex),
    ("row", Lean.toJson ordinal), ("boundary", .str boundary)])
  let read := fun column : Fin logicalWidth => Spec.Poseidon2.ofNat column.val
  if let .poseidon poseidon := block then
    let started ← IO.monoNanosNow
    let result ← IO.wait (Task.spawn fun _ =>
      PiDECPoseidonNumericBlock.row? poseidon read (ordinal - blockStart))
    let finished ← IO.monoNanosNow
    let some values := result
      | throw (IO.userError s!"numeric Poseidon row rejected at block {blockIndex}, row {ordinal}")
    emit output (Lean.Json.mkObj [
      ("event", .str "row_complete"), ("block", Lean.toJson blockIndex),
      ("row", Lean.toJson ordinal), ("boundary", .str boundary),
      ("mode", .str "proved_numeric_poseidon"),
      ("evaluation_task_ns", Lean.toJson (finished - started)),
      ("values", Lean.toJson (List.ofFn fun port => (values.get port).val)),
      ("ports", Lean.toJson Spec.ProductionRelation.matrixCount)])
    return
  let beforeLookup ← IO.monoNanosNow
  let result ← match block with
    | .phi81Product product => do
        let localRow := ordinal - blockStart
        let descriptor ← IO.wait (Task.spawn fun _ =>
          MatrixProgram.Phi81Product.descriptor? product.families (localRow / 34))
        let descriptorAt ← IO.monoNanosNow
        emit output (Lean.Json.mkObj [
          ("event", .str "product_descriptor"), ("block", Lean.toJson blockIndex),
          ("row", Lean.toJson ordinal),
          ("elapsed_ns", Lean.toJson (descriptorAt - beforeLookup))])
        let some descriptor := descriptor | throw (IO.userError "product descriptor rejected")
        let interface ← IO.wait (Task.spawn fun _ =>
          PiDECProductInterface.interface? product logicalWidth descriptor)
        let interfaceAt ← IO.monoNanosNow
        emit output (Lean.Json.mkObj [
          ("event", .str "product_interface"), ("block", Lean.toJson blockIndex),
          ("row", Lean.toJson ordinal),
          ("elapsed_ns", Lean.toJson (interfaceAt - descriptorAt))])
        let some interface := interface | throw (IO.userError "product interface rejected")
        IO.wait (Task.spawn fun _ =>
          (PiDECProductRow.row? interface (localRow % 34)).map
            Layout.ProductionRelation.ProductSumPlan.Row.meaningfulForm)
    | _ => IO.wait (Task.spawn fun _ => program.row? logicalWidth sourceRow ordinal)
  let afterLookup ← IO.monoNanosNow
  let some forms := result | do
    emit output (Lean.Json.mkObj [
      ("event", .str "row_unsupported"), ("block", Lean.toJson blockIndex),
      ("row", Lean.toJson ordinal),
      ("lookup_task_ns", Lean.toJson (afterLookup - beforeLookup))])
    throw (IO.userError s!"canonical row? returned none at block {blockIndex}, row {ordinal}")
  let mut generationNs := afterLookup - beforeLookup
  let mut statisticsNs := 0
  for port in List.finRange Spec.ProductionRelation.matrixCount do
    emit output (Lean.Json.mkObj [
      ("event", .str "port_begin"), ("block", Lean.toJson blockIndex),
      ("row", Lean.toJson ordinal), ("port", Lean.toJson port.val)])
    let beforePort ← IO.monoNanosNow
    let (form, entryCount, value) ← IO.wait (Task.spawn fun _ =>
      let form := match meaningfulPort? port with
        | some meaningful => forms meaningful
        | none => SparseForm.empty
      (form, form.entries.length, form.evalSparse read))
    let afterPort ← IO.monoNanosNow
    let distinctBlocks ← IO.wait (Task.spawn fun _ =>
      (PiDECEvaluationBlockSupport.blockIndices form).length)
    let afterStatistics ← IO.monoNanosNow
    let generated := afterPort - beforePort
    let statistics := afterStatistics - afterPort
    generationNs := generationNs + generated
    statisticsNs := statisticsNs + statistics
    emit output (Lean.Json.mkObj [
      ("event", .str "port"), ("block", Lean.toJson blockIndex),
      ("row", Lean.toJson ordinal), ("port", Lean.toJson port.val),
      ("entry_count", Lean.toJson entryCount),
      ("value", Lean.toJson value.val),
      ("distinct_block_count", Lean.toJson distinctBlocks),
      ("materialize_and_count_task_ns", Lean.toJson generated),
      ("support_statistics_task_ns", Lean.toJson statistics)])
  emit output (Lean.Json.mkObj [
    ("event", .str "row_complete"), ("block", Lean.toJson blockIndex),
    ("row", Lean.toJson ordinal), ("boundary", .str boundary),
    ("lookup_task_ns", Lean.toJson (afterLookup - beforeLookup)),
    ("generation_and_entry_count_task_ns", Lean.toJson generationNs),
    ("support_statistics_task_ns", Lean.toJson statisticsNs),
    ("ports", Lean.toJson Spec.ProductionRelation.matrixCount)])

private def measure (outputPath : System.FilePath) (firstBlock lastBlock : Nat) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let output ← IO.FS.Handle.mk outputPath .write
  emit output (Lean.Json.mkObj [
    ("event", .str "initialization_begin"), ("schema", Lean.toJson (3 : Nat)),
    ("first_block", Lean.toJson firstBlock), ("last_block_exclusive", Lean.toJson lastBlock),
    ("read", .str "column_index_mod_Goldilocks"),
    ("scope", .str "endpoint feasibility probe; not actual witness matrix replay")])
  let beforeProgram ← IO.monoNanosNow
  let program ← IO.wait (Task.spawn fun _ => PerApplicationMatrixProgram.matrixProgram
    Poseidon2HashChainV1Package.application)
  let afterProgram ← IO.monoNanosNow
  unless firstBlock < lastBlock && lastBlock ≤ program.blocks.length do
    throw (IO.userError "invalid selected block interval")
  emit output (Lean.Json.mkObj [
    ("event", .str "source_accessor_begin"),
    ("program_task_ns", Lean.toJson (afterProgram - beforeProgram))])
  let beforeSource ← IO.monoNanosNow
  let cache ← IO.wait (Task.spawn fun _ =>
    PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
  let sourceRow := fun source => cache[source]?
  let afterSource ← IO.monoNanosNow
  let logicalWidth := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
  emit output (Lean.Json.mkObj [
    ("event", .str "program"), ("logical_width", Lean.toJson logicalWidth),
    ("block_count", Lean.toJson program.blocks.length),
    ("row_count", Lean.toJson program.rowCount),
    ("stored_source_rows", Lean.toJson cache.size),
    ("ports", Lean.toJson Spec.ProductionRelation.matrixCount),
    ("source_accessor_task_ns", Lean.toJson (afterSource - beforeSource))])
  let mut blockIndex := 0
  let mut start := 0
  let mut measuredRows := 0
  for block in program.blocks do
    let count := block.rowCount
    if firstBlock ≤ blockIndex && blockIndex < lastBlock then
      emit output (Lean.Json.mkObj [
        ("event", .str "block"), ("block", Lean.toJson blockIndex),
        ("kind", .str (blockTag block)), ("start", Lean.toJson start),
        ("end_exclusive", Lean.toJson (start + count)), ("row_count", Lean.toJson count)])
      if count = 0 then
        emit output (Lean.Json.mkObj [
          ("event", .str "empty_block"), ("block", Lean.toJson blockIndex)])
      else if count = 1 then
        measureRow output program logicalWidth sourceRow block blockIndex start start "first_last"
        measuredRows := measuredRows + 1
      else
        measureRow output program logicalWidth sourceRow block blockIndex start start "first"
        measureRow output program logicalWidth sourceRow block blockIndex start (start + count - 1) "last"
        measuredRows := measuredRows + 2
    start := start + count
    blockIndex := blockIndex + 1
  emit output (Lean.Json.mkObj [
    ("event", .str "complete"), ("blocks", Lean.toJson blockIndex),
    ("rows", Lean.toJson start), ("measured_rows", Lean.toJson measuredRows),
    ("first_block", Lean.toJson firstBlock), ("last_block_exclusive", Lean.toJson lastBlock)])
  return 0

end NightstreamFPrime.Export.PiDECMatrixRowsMeasure

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | [outputPath, firstBlock, lastBlock] =>
      match firstBlock.toNat?, lastBlock.toNat? with
      | some first, some last =>
          NightstreamFPrime.Export.PiDECMatrixRowsMeasure.measure outputPath first last
      | _, _ => throw (IO.userError "block endpoints must be natural numbers")
  | _ => throw (IO.userError "usage: measurePiDECMatrixRows <new-output.jsonl> <first-block> <last-block-exclusive>")
