import NightstreamFPrime.Layout.MatrixProgram.Poseidon
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.MatrixProgram.PoseidonInput

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.Poseidon

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def Block.format : Format Block where
  encode := fun block => .array [
    .atom block.invocationCount,
    .atom block.oneColumn,
    RetainedBlock.format.encode block.retained,
    PoseidonInput.Program.format.encode block.input]
  decode
    | .array [.atom invocationCount, .atom oneColumn, retained, input] => do
        pure ⟨invocationCount, oneColumn,
          ← RetainedBlock.format.decode retained,
          ← PoseidonInput.Program.format.decode input⟩
    | _ => .error "invalid Poseidon2 matrix block"
  decode_encode := by
    intro block
    cases block
    simp [RetainedBlock.format.decode_encode,
      PoseidonInput.Program.format.decode_encode]
    rfl

end NightstreamFPrime.Layout.MatrixProgram.Poseidon
