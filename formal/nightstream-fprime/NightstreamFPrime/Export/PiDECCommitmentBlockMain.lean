import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiDECCommitmentBlock
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
Measure the first actual block of a Lean parent range. Compute all fixed-key
row contributions for its sixteen digits. This is a feasibility measurement;
it neither accumulates a complete commitment nor consumes expected outputs.
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

private def firstBlock (input : IO.FS.Handle) : IO (Nat × StoredAssignment ringDegree) := do
  let line ← input.getLine
  let header ← checked do
    (← (← Lean.Json.parse line).getArr?).toList.mapM Lean.Json.getNat?
  let (start, finish) ← match header with
    | [1, blocks, start, finish] =>
        if blocks = Poseidon2HashChainV1Setup.messageColumns && start < finish && finish ≤ blocks then
          pure (start, finish)
        else throw (IO.userError "invalid selected parent range")
    | _ => throw (IO.userError "expected a Lean parent range header")
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

private def measure (parentPath outputPath : System.FilePath) : IO UInt32 := do
  let keyPath : System.FilePath := outputPath.toString ++ ".keys.json"
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  unless !(← keyPath.pathExists) do throw (IO.userError "key output already exists")
  let input ← IO.FS.Handle.mk parentPath .read
  let (block, parent) ← firstBlock input
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError "parent exceeds the strict B bound")
  if live : block < Poseidon2HashChainV1Setup.messageColumns then
    let started ← IO.monoMsNow
    let keys := Vector.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      PiDECCommitmentBlock.keyBlock Poseidon2HashChainV1Setup.productionSetup row ⟨block, live⟩
    let keyValues := Value.array (List.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      .array (List.ofFn fun lane : Fin ringDegree => .atom ((keys.get row).get lane).val))
    IO.FS.writeFile keyPath (keyValues.render ++ "\n")
    let keysFinished ← IO.monoMsNow
    let results := Vector.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
      PiDECCommitmentBlock.products (keys.get row) children
    let value := Value.array [.atom 1, .atom block,
      .array (List.ofFn fun row : Fin Poseidon2HashChainV1Setup.verifierRows =>
        .array (List.ofFn fun child : Fin productionGlobalParams.k =>
          .array (List.ofFn fun lane : Fin ringDegree =>
            .atom (((results.get row).get child).get lane).val)))]
    IO.FS.writeFile outputPath (value.render ++ "\n")
    let finished ← IO.monoMsNow
    IO.println s!"pidec_commitment_block=measured block={block} rows={Poseidon2HashChainV1Setup.verifierRows} children={productionGlobalParams.k} product_coefficients={Poseidon2HashChainV1Setup.verifierRows * productionGlobalParams.k * ringDegree} key_compute_encode_write_ms={keysFinished - started} products_compute_encode_write_ms={finished - keysFinished} compute_encode_write_ms={finished - started}"
    return 0
  else throw (IO.userError "block is outside the selected fixed key")

end NightstreamFPrime.Export.PiDECCommitmentBlockMeasurement

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | [parentPath, outputPath] =>
      NightstreamFPrime.Export.PiDECCommitmentBlockMeasurement.measure parentPath outputPath
  | _ =>
      IO.eprintln "usage: measurePiDECCommitmentBlock <Lean-parent-range> <new-output>"
      return 2
