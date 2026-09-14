import NightstreamFPrime.Export.PiDECParentInput
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Stage1.PiDECParentIntRead
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
import NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange
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

private def range (ccsPath outputPath : System.FilePath)
    (blockIndex firstRow lastRow : Nat) (parentPaths : List String) : IO UInt32 := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
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
  let tablesStarted ← IO.monoNanosNow
  let tables ← IO.wait (Task.spawn fun _ => PiDECParentSparseRead.prepare ())
  report [("event", .str "basis_ready"),
    ("retained_coefficients", Lean.toJson ((List.ofFn fun basis : Fin ringDegree =>
      ((List.ofFn fun output : Fin ringDegree =>
        ((tables.get basis).get output).entries.length)).sum)).sum),
    ("elapsed_ns", Lean.toJson ((← IO.monoNanosNow) - tablesStarted))]
  let program ← IO.wait (Task.spawn fun _ =>
    PerApplicationMatrixProgram.matrixProgram Poseidon2HashChainV1Package.application)
  let some selected := program.blocks[blockIndex]?
    | throw (IO.userError "invalid selected matrix block")
  if bounds : firstRow < lastRow ∧ lastRow ≤ selected.rowCount then
    let count := lastRow - firstRow
    let logicalWidth := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
    let blockStart := ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum
    let first := blockStart + firstRow
    let finish := blockStart + lastRow
    unless finish ≤ program.rowCount do throw (IO.userError "range exceeds matrix row domain")
    let loadStarted ← IO.monoNanosNow
    let evaluate : (Fin ringDegree → Fin logicalWidth → F) →
        Vector MaterializedRingK matrixCount ← match selectedEq : selected with
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
          pure fun read => PiDECMatrixInvocationRange.sum first phase.point read interfaces
      | .phi81Product block => do
          if aligned : firstRow % 34 = 0 ∧ lastRow % 34 = 0 then
            let invocations := lastRow / 34 - firstRow / 34
            let interfaces ← Vector.ofFnM fun (index : Fin invocations) => do
              let some descriptor := MatrixProgram.Phi81Product.descriptor?
                  block.families (firstRow / 34 + index.val)
                | throw (IO.userError "selected product descriptor rejected")
              let some interface := PiDECProductInterface.interface? block logicalWidth descriptor
                | throw (IO.userError "selected product interface rejected")
              pure interface
            let forms ← Vector.ofFnM fun (index : Fin count) => do
              have groupBound : index.val / 34 < invocations := by
                dsimp only [count] at index
                dsimp only [invocations]
                omega
              let some row := PiDECProductRow.row?
                  (interfaces.get ⟨index.val / 34, groupBound⟩) (index.val % 34)
                | throw (IO.userError "selected product row rejected")
              pure row.meaningfulForm
            pure fun read => PiDECMatrixSparseRange.sum first phase.point read forms
          else throw (IO.userError "Phi81 range must contain complete 34-row invocations")
      | other => do
          let cache ← IO.wait (Task.spawn fun _ =>
            PiDECCanonicalSourceCache.stored Poseidon2HashChainV1Package.application)
          let source := fun row => cache[row]?
          let forms ← Vector.ofFnM fun (index : Fin count) => do
            let some row := other.row? logicalWidth source (firstRow + index.val)
              | throw (IO.userError "selected sparse row rejected")
            pure row
          pure fun read => PiDECMatrixSparseRange.sum first phase.point read forms
    report [("event", .str "range_begin"), ("block", Lean.toJson blockIndex),
      ("block_rows", Lean.toJson selected.rowCount),
      ("first_local_row", Lean.toJson firstRow), ("last_local_row_exclusive", Lean.toJson lastRow),
      ("start", Lean.toJson first), ("end", Lean.toJson finish),
      ("load_ns", Lean.toJson ((← IO.monoNanosNow) - loadStarted))]
    let mut tasks := #[]
    for child in List.finRange productionGlobalParams.k do
      tasks := tasks.push (← IO.asTask do
        let read := fun (output : Fin ringDegree) (column : Fin logicalWidth) =>
          if live : column.val / ringDegree < Poseidon2HashChainV1Setup.messageColumns then
            PiDECParentIntRead.sparseRead tables
              (parents.get ⟨column.val / ringDegree, live⟩)
              ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ child output
          else 0
        let started ← IO.monoNanosNow
        let values ← IO.wait (Task.spawn fun _ =>
          evaluate read)
        return (values, (← IO.monoNanosNow) - started))
    let mut allValues : Array (Vector MaterializedRingK matrixCount) := #[]
    for child in [:tasks.size] do
      let result ← IO.wait tasks[child]!
      let (values, computeNs) ← match result with
        | .ok value => pure value
        | .error error => throw error
      allValues := allValues.push values
      report [("event", .str "child_complete"), ("child", Lean.toJson child),
        ("compute_ns", Lean.toJson computeNs)]
    let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
    let output := Value.array [.atom 1, .atom program.rowCount, .atom first, .atom finish,
      .array (phase.point.coordinates.map encodeK),
      .array (allValues.toList.map fun matrices =>
        .array (List.ofFn fun matrix : Fin matrixCount =>
          .array (List.ofFn fun lane : Fin ringDegree =>
            encodeK ((matrices.get matrix).toRing lane))))]
    IO.FS.writeFile outputPath (output.render ++ "\n")
    report [("event", .str "range_complete"),
      ("children", Lean.toJson allValues.size), ("matrices", Lean.toJson matrixCount),
      ("field_words", Lean.toJson (productionGlobalParams.k * matrixCount * ringDegree * 2)),
      ("total_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
    return 0
  else throw (IO.userError "invalid selected matrix row range")

end NightstreamFPrime.Export.PiDECMatrixReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | ccsPath :: outputPath :: block :: first :: last :: parents =>
      match block.toNat?, first.toNat?, last.toNat? with
      | some block, some first, some last =>
          NightstreamFPrime.Export.PiDECMatrixReplay.range
            ccsPath outputPath block first last parents
      | _, _, _ => throw (IO.userError "block and row bounds must be natural numbers")
  | _ => throw (IO.userError "usage: replayPiDECMatrix <C-input> <new-output> <block> <first-local-row> <last-exclusive> <Lean-parent-ranges>...")
