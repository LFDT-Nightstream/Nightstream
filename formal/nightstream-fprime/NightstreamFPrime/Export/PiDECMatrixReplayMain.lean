import NightstreamFPrime.Export.PiDECParentInput
import NightstreamFPrime.Export.Stage1.PiCCSInputCheck
import NightstreamFPrime.Export.Stage1.PiDECParentSparseRead
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation
import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows
import NightstreamFPrime.Export.Codec

/-!
Execute one complete selected Poseidon matrix invocation from the independent
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

private def invocation (ccsPath outputPath : System.FilePath)
    (blockIndex invocationIndex : Nat) (parentPaths : List String) : IO UInt32 := do
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
  report [("event", .str "parent_ready"),
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
  let some (.poseidon block) := program.blocks[blockIndex]?
    | throw (IO.userError "selected program block is not a Poseidon block")
  if invBound : invocationIndex < block.invocationCount then
    let logicalWidth := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
    let some interface := PiDECPoseidonNumericBlock.loadInvocation?
        block logicalWidth ⟨invocationIndex, invBound⟩
      | throw (IO.userError "selected invocation interface rejected")
    let blockStart := ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum
    let first := blockStart + (Fin.encodeProd
      (⟨invocationIndex, invBound⟩, (⟨0, by decide⟩ : Fin 94))).val
    let finish := first + 94
    unless finish ≤ program.rowCount do throw (IO.userError "invocation exceeds matrix row domain")
    report [("event", .str "invocation_begin"), ("block", Lean.toJson blockIndex),
      ("invocation", Lean.toJson invocationIndex),
      ("start", Lean.toJson first), ("end", Lean.toJson finish)]
    let mut tasks := #[]
    for child in List.finRange productionGlobalParams.k do
      tasks := tasks.push (← IO.asTask do
        let read := fun (output : Fin ringDegree) (column : Fin logicalWidth) =>
          if live : column.val / ringDegree < Poseidon2HashChainV1Setup.messageColumns then
            PiDECParentSparseRead.read tables
              (parents.get ⟨column.val / ringDegree, live⟩)
              ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ child output
          else 0
        let preparing ← IO.monoNanosNow
        let prepared ← IO.wait (Task.spawn fun _ => PiDECMatrixInvocation.prepare read interface)
        let summing ← IO.monoNanosNow
        let values ← IO.wait (Task.spawn fun _ => PiDECMatrixInvocation.sum first phase.point prepared)
        let finished ← IO.monoNanosNow
        return (values, summing - preparing, finished - summing))
    let mut allValues : Array (Vector MaterializedRingK matrixCount) := #[]
    for child in [:tasks.size] do
      let result ← IO.wait tasks[child]!
      let (values, prepareNs, sumNs) ← match result with
        | .ok value => pure value
        | .error error => throw error
      allValues := allValues.push values
      report [("event", .str "child_complete"), ("child", Lean.toJson child),
        ("prepare_ns", Lean.toJson prepareNs), ("sum_ns", Lean.toJson sumNs)]
    let encodeK := fun value : K => Value.array [.atom value.c0.val, .atom value.c1.val]
    let output := Value.array [.atom 1, .atom program.rowCount, .atom first, .atom finish,
      .array (phase.point.coordinates.map encodeK),
      .array (allValues.toList.map fun matrices =>
        .array (List.ofFn fun matrix : Fin matrixCount =>
          .array (List.ofFn fun lane : Fin ringDegree =>
            encodeK ((matrices.get matrix).toRing lane))))]
    IO.FS.writeFile outputPath (output.render ++ "\n")
    report [("event", .str "invocation_complete"),
      ("children", Lean.toJson allValues.size), ("matrices", Lean.toJson matrixCount),
      ("field_words", Lean.toJson (productionGlobalParams.k * matrixCount * ringDegree * 2)),
      ("total_ns", Lean.toJson ((← IO.monoNanosNow) - started))]
    return 0
  else throw (IO.userError "invalid selected invocation index")

end NightstreamFPrime.Export.PiDECMatrixReplay

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | ccsPath :: outputPath :: block :: invocation :: parents =>
      match block.toNat?, invocation.toNat? with
      | some block, some invocation =>
          NightstreamFPrime.Export.PiDECMatrixReplay.invocation
            ccsPath outputPath block invocation parents
      | _, _ => throw (IO.userError "block and invocation must be natural numbers")
  | _ => throw (IO.userError "usage: replayPiDECMatrix <C-input> <new-output> <block> <invocation> <Lean-parent-ranges>...")
