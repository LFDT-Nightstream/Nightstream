import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
Measure the first two stored blocks of a Lean parent range to separate
initialization from repeated work. Compute all fixed-key row contributions
for their sixteen digits. This neither accumulates a complete commitment
nor consumes expected outputs.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECCommitmentBlockMeasurement

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Export.Stage1

private def checked {Alpha : Type} (value : Except String Alpha) : IO Alpha :=
  match value with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def readRange (input : IO.FS.Handle) : IO (Nat × Nat) := do
  let line ← input.getLine
  let header ← checked do
    (← (← Lean.Json.parse line).getArr?).toList.mapM Lean.Json.getNat?
  match header with
    | [1, blocks, start, finish] =>
        if blocks = Poseidon2HashChainV1Setup.messageColumns && start < finish && finish ≤ blocks then
          pure (start, finish)
        else throw (IO.userError "invalid selected parent range")
    | _ => throw (IO.userError "expected a Lean parent range header")

private def readBlock (input : IO.FS.Handle) (start finish : Nat) :
    IO (Nat × StoredAssignment ringDegree) := do
  let line ← input.getLine
  let (block, words) ← checked do
    let fields ← (← Lean.Json.parse line).getArr?
    match fields.toList with
    | [block, values] => return (← block.getNat?, ← values.getArr?)
    | _ => throw "expected a stored parent block"
  unless start ≤ block && block < finish do
    throw (IO.userError "parent block is outside its range")
  let mut values : Array F := #[]
  for word in words do
    let value ← checked word.getNat?
    unless value < goldilocksModulus do throw (IO.userError "noncanonical parent coefficient")
    values := values.push (Radix.fieldOfNat value)
  if size : values.size = ringDegree then return (block, ⟨values, size⟩)
  else throw (IO.userError "expected 54 parent coefficients")

private def measureBlock (block : Nat) (parent : StoredAssignment ringDegree)
    (outputPath : System.FilePath) : IO Unit := do
  let keyPath : System.FilePath := outputPath.toString ++ ".keys.json"
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  unless !(← keyPath.pathExists) do throw (IO.userError "key output already exists")
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError "parent exceeds the strict B bound")
  if live : block < Poseidon2HashChainV1Setup.messageColumns then
    let started ← IO.monoMsNow
    let keyTasks := Vector.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      Task.spawn fun _ =>
        PiDECCommitmentBlock.keyBlock Poseidon2HashChainV1Setup.productionSetup row ⟨block, live⟩
    let keys ← keyTasks.mapM fun task => IO.wait task
    let keysComputed ← IO.monoMsNow
    let keyValues := Value.array (List.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      .array (List.ofFn fun lane : Fin ringDegree => .atom ((keys.get row).get lane).val))
    IO.FS.writeFile keyPath (keyValues.render ++ "\n")
    let keysFinished ← IO.monoMsNow
    let productTasks := Vector.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      Task.spawn fun _ => PiDECCommitmentBlock.products (keys.get row) children
    let results ← productTasks.mapM fun task => IO.wait task
    let productsComputed ← IO.monoMsNow
    let value := Value.array [.atom 1, .atom block,
      .array (List.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
        .array (List.ofFn fun child : Fin productionGlobalParams.k =>
          .array (List.ofFn fun lane : Fin ringDegree =>
            .atom (((results.get row).get child).get lane).val)))]
    IO.FS.writeFile outputPath (value.render ++ "\n")
    let finished ← IO.monoMsNow
    IO.println s!"pidec_commitment_block=measured block={block} rows={Poseidon2HashChainV1Setup.verifierRows} children={productionGlobalParams.k} product_coefficients={Poseidon2HashChainV1Setup.verifierRows * productionGlobalParams.k * ringDegree} key_compute_ms={keysComputed - started} key_encode_write_ms={keysFinished - keysComputed} products_compute_ms={productsComputed - keysFinished} products_encode_write_ms={finished - productsComputed} compute_encode_write_ms={finished - started}"
  else throw (IO.userError "block is outside the selected fixed key")

private def measure (parentPath outputPath : System.FilePath) : IO UInt32 := do
  let input ← IO.FS.Handle.mk parentPath .read
  let (start, finish) ← readRange input
  let (first, firstParent) ← readBlock input start finish
  let (second, secondParent) ← readBlock input start finish
  unless first < second do throw (IO.userError "stored blocks must be ordered")
  let nextPath : System.FilePath := outputPath.toString ++ ".next.json"
  measureBlock first firstParent outputPath
  measureBlock second secondParent nextPath
  return 0

end NightstreamFPrime.Export.PiDECCommitmentBlockMeasurement

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | [parentPath, outputPath] =>
      NightstreamFPrime.Export.PiDECCommitmentBlockMeasurement.measure parentPath outputPath
  | _ =>
      IO.eprintln "usage: measurePiDECCommitmentBlock <Lean-parent-range> <new-output>"
      return 2
