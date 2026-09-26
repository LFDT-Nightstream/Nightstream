import NightstreamFPrime.Export.Stage1.Wide.MatrixProgram
import NightstreamFPrime.Export.Stage1.Wide.HashChainCounts
import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.ParityEmitter

/-! Emit only compact matrix operands for independent cost measurement.
The selected source-row archive and production packages are not regenerated. -/

namespace NightstreamFPrime.Tests.WideMatrixCost

open NightstreamFPrime.Layout NightstreamFPrime.Spec NightstreamFPrime.Export
open Stage1

private def data (compiled : PiRlcWideSampler.RangePlan.Compiled) : Codec.Value :=
  let application := Poseidon2HashChainV1Package.application
  .array [
    .atom 1,
    MatrixProgram.Program.format.encode (Wide.MatrixProgram.program application compiled),
    MatrixProgram.Program.format.encode (PerApplicationMatrixProgram.matrixProgram application),
    .atom (Wide.RetainedLayout.logicalWidth application),
    .atom (Wide.MatrixProgram.program application compiled).rowCount,
    (Codec.list (Codec.list Codec.nat)).encode Poseidon2.initialConstants,
    (Codec.list Codec.nat).encode Poseidon2.internalConstants,
    (Codec.list (Codec.list Codec.nat)).encode Poseidon2.terminalConstants,
    (Codec.list Codec.nat).encode Poseidon2.internalDiagonal]

def run (arguments : List String) : IO UInt32 := do
  let some compiled := PiRlcWideSampler.RangePlan.compile?
    | throw (IO.userError "range compiler rejected the candidate")
  ParityEmitter.run "wide_matrix_cost_operands" (data compiled) arguments

end NightstreamFPrime.Tests.WideMatrixCost

def main (arguments : List String) : IO UInt32 := NightstreamFPrime.Tests.WideMatrixCost.run arguments
