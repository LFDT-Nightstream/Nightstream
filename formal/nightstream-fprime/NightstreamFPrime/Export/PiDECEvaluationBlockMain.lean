import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBlock
import NightstreamFPrime.Export.Stage1.PiDECEvaluationPadBlock
import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit

/-!
Measure selected Pad rows 1 and 2 on actual Lean parent block 0.
Compare every child coefficient with the existing basis kernel. This is a
single-block calculation; full row and point accumulation remain separate
obligations. No Rust expected output is an input.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.PiDECEvaluationBlockMeasurement

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
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

private def measureRow (block : Nat)
    (children : Vector (StoredAssignment ringDegree) productionGlobalParams.k)
    (basis : Fin ringDegree) (outputPath : System.FilePath) : IO Unit := do
  unless !(← outputPath.pathExists) do throw (IO.userError "output already exists")
  let layout := Folding.PiCCS.CanonicalRowLayout.layout Lifecycle.cubeVariables
    (Phi81CarrierLayout.carrierWidth PiDECInputCheck.logicalWidth)
    PiDECInputCheck.relation.cubeFits
  let vertex := Folding.PiCCS.PaperJoint.NumericBooleanDomain.vertex
    Lifecycle.cubeVariables ⟨basis.val, Nat.lt_trans basis.isLt (by decide)⟩
  let form := PiDECEvaluationPadBlock.form layout vertex
  let started ← IO.monoMsNow
  let computed ← IO.wait (Task.spawn fun _ => PiDECEvaluationBlock.rowBlock form block children)
  let computedAt ← IO.monoMsNow
  let reference ← IO.wait (Task.spawn fun _ => Vector.ofFn fun child : Fin productionGlobalParams.k =>
    Vector.ofFn fun output : Fin ringDegree =>
      CarrierAction.kernelImage basis (children.get child).get output)
  let referenceAt ← IO.monoMsNow
  for child in List.finRange productionGlobalParams.k do
    for lane in List.finRange ringDegree do
      unless ((computed.get child).get lane) = ((reference.get child).get lane) do
        throw (IO.userError s!"Pad block mismatch at child {child.val}, lane {lane.val}")
  let value := Value.array [.atom 1, .atom block, .atom basis.val,
    .array (List.ofFn fun child : Fin productionGlobalParams.k =>
      .array (List.ofFn fun lane : Fin ringDegree =>
        .atom (((computed.get child).get lane).val)))]
  IO.FS.writeFile outputPath (value.render ++ "\n")
  let finished ← IO.monoMsNow
  IO.println s!"pidec_evaluation_block=passed block={block} pad_row={basis.val} children={productionGlobalParams.k} coefficients={productionGlobalParams.k * ringDegree} kernel_ms={computedAt - started} reference_ms={referenceAt - computedAt} compare_write_ms={finished - referenceAt}"
  return ()

private def measure (parentPath outputPath : System.FilePath) : IO UInt32 := do
  let input ← IO.FS.Handle.mk parentPath .read
  let (start, finish) ← readRange input
  let (block, parent) ← readBlock input start finish
  unless start = 0 && block = 0 do throw (IO.userError "expected actual parent block zero")
  let some children := StoredSplit.splitChecked parent
    | throw (IO.userError "parent exceeds the strict B bound")
  -- A distinct second row separates initialization from repeated work.
  measureRow block children ⟨1, by decide⟩ outputPath
  measureRow block children ⟨2, by decide⟩ (outputPath.toString ++ ".next.json")
  return 0

end NightstreamFPrime.Export.PiDECEvaluationBlockMeasurement

def main (arguments : List String) : IO UInt32 := do
  let arguments := if arguments.head? == some "--" then arguments.tail else arguments
  match arguments with
  | [parentPath, outputPath] =>
      NightstreamFPrime.Export.PiDECEvaluationBlockMeasurement.measure parentPath outputPath
  | _ =>
      IO.eprintln "usage: measurePiDECEvaluationBlock <Lean-parent-range> <new-output>"
      return 2
