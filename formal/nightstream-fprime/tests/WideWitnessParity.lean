import NightstreamFPrime.Export.WitnessEncoding
import NightstreamFPrime.Export.Package
import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Program
import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintExecution
import NightstreamFPrime.Layout.PiRlcWideSampler.Witness

/-! Emit the proved scalar's exact witness batches and retained coordinates.
The array evaluator keeps the temporary helper computation linear. Every
case is checked against all gadget rows before it is emitted. -/

namespace NightstreamFPrime.Tests.WideWitnessParity

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export NightstreamFPrime.Export.Codec
open NightstreamFPrime.Gadgets.Sampling
open Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

private def operations (_ : Unit) : List Op :=
  WideReduction.Program.operations PiRlcWideSampler.RangePlan.interface 4

private def batches (_ : Unit) : List WitnessBatch := witnesses (operations ())

private def retained (_ : Unit) : Value :=
  let block {count : Nat} (kind : Nat) (source : Fin count → Nat) :=
    Value.array [.atom kind, (list nat).encode (List.ofFn source)]
  .array [
    block 0 (fun index => (PiRlcWideSampler.Retained.canonicalBits.source index).val),
    block 2 (fun index => (PiRlcWideSampler.Retained.canonicalFields.source index).val),
    block 0 (fun index => (PiRlcWideSampler.Retained.resultBits.source index).val)]

private def boundary (integer : Nat) : IO Value := do
  let draw : Draw := drawIndex.symm ⟨integer % drawCount, Nat.mod_lt _ (by decide)⟩
  let base := PiRlcWideSampler.Witness.inputEnv draw
  let mut values : Array F := #[]
  for batch in batches () do
    unless batch.start = 4 + values.size do
      throw (IO.userError "non-contiguous wide witness batches")
    for recipe in batch.recipes do
      values := values.push (recipe.eval (WideReduction.HintExecution.read base 4 values))
    for hint in batch.hints do
      values := values.push (hint.eval (WideReduction.HintExecution.read base 4 values))
  unless values.size = WideReduction.Program.privateCount do
    throw (IO.userError "wide witness private count")
  let completed := WideReduction.HintExecution.read base 4 values
  for row in flatConstraints (operations ()) do
    unless row.eval completed = 0 do
      throw (IO.userError "wide witness row failed")
  let coordinates := List.ofFn fun index : Fin 937 =>
    (PiRlcWideSampler.Witness.rangeCoordinate completed index).val
  pure (.array [
    (list nat).encode (List.ofFn fun lane => (draw lane).val),
    (list nat).encode (values.toList.map Fin.val),
    (list nat).encode coordinates])

private def fixture : IO Value := do
  let boundaries := [0, 1, scalarCount - 1, scalarCount, scalarCount + 1,
    goldilocksModulus - 1, goldilocksModulus, goldilocksModulus + 1,
    drawCount - scalarCount, drawCount - 2, drawCount - 1]
  let cases ← boundaries.mapM boundary
  pure (.array [
    .atom 1, .atom 4, .atom WideReduction.Program.privateCount,
    (list Package.WitnessBatch.format).encode ((batches ()).map WitnessEncoding.batch),
    retained (), .array cases])

def run (arguments : List String) : IO UInt32 :=
  ParityEmitter.runIO "wide_witness_parity" fixture arguments

end NightstreamFPrime.Tests.WideWitnessParity

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Tests.WideWitnessParity.run arguments
