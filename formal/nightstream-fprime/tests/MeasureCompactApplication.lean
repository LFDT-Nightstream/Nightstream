import NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixExecution
import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.ParityEmitter
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1

/-! Export exact matrix operands for a normalized sparse cost calculation. -/

open NightstreamFPrime.Layout
open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Stage1

namespace NightstreamFPrime.Tests.MeasureCompactApplication

private def data (_delay : Unit) : Codec.Value :=
  let application := NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1.program
  let certificate : ApplicationPoseidonRetainedBlock.Certificate application := ⟨rfl, fun _ => rfl⟩
  let geometry : ApplicationPoseidonRetainedGeometry.Geometry application certificate 149292999 :=
    ⟨by rw [ApplicationPoseidonRetainedGeometry.completeLogicalWidth_eq]⟩
  .array [
    MatrixProgram.Program.format.encode (ApplicationPoseidonMatrixProgram.matrixProgram geometry),
    (Codec.list (Codec.list Codec.nat)).encode Poseidon2.initialConstants,
    (Codec.list Codec.nat).encode Poseidon2.internalConstants,
    (Codec.list (Codec.list Codec.nat)).encode Poseidon2.terminalConstants,
    (Codec.list Codec.nat).encode Poseidon2.internalDiagonal]

end NightstreamFPrime.Tests.MeasureCompactApplication

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.ParityEmitter.run "application_matrix_cost_operands"
    (NightstreamFPrime.Tests.MeasureCompactApplication.data ()) arguments
