import NightstreamFPrime.Layout.MatrixProgram.Program
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram.MultiplicationGrid
import NightstreamFPrime.Export.MatrixProgram.Ordinary
import NightstreamFPrime.Export.MatrixProgram.Phi81Product
import NightstreamFPrime.Export.MatrixProgram.Pin
import NightstreamFPrime.Export.MatrixProgram.Poseidon

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def Block.format : Format Block where
  encode
    | .ordinary block =>
        .array [.atom 0, Ordinary.Block.format.encode block]
    | .multiplicationGrid block =>
        .array [.atom 4, MultiplicationGrid.Block.format.encode block]
    | .phi81Product block =>
        .array [.atom 3, Phi81Product.Block.format.encode block]
    | .pin block =>
        .array [.atom 1, Pin.Block.format.encode block]
    | .poseidon block =>
        .array [.atom 2, Poseidon.Block.format.encode block]
  decode
    | .array [.atom 0, block] => do
        pure (.ordinary (← Ordinary.Block.format.decode block))
    | .array [.atom 4, block] => do
        pure (.multiplicationGrid
          (← MultiplicationGrid.Block.format.decode block))
    | .array [.atom 3, block] => do
        pure (.phi81Product (← Phi81Product.Block.format.decode block))
    | .array [.atom 1, block] => do
        pure (.pin (← Pin.Block.format.decode block))
    | .array [.atom 2, block] => do
        pure (.poseidon (← Poseidon.Block.format.decode block))
    | _ => .error "invalid production matrix block"
  decode_encode := by
    intro block
    cases block <;>
      simp [Ordinary.Block.format.decode_encode,
        MultiplicationGrid.Block.format.decode_encode,
        Phi81Product.Block.format.decode_encode,
        Pin.Block.format.decode_encode, Poseidon.Block.format.decode_encode]

def Program.format : Format Program where
  encode := fun program => (list Block.format).encode program.blocks
  decode := fun value => do
    pure ⟨← (list Block.format).decode value⟩
  decode_encode := by
    intro program
    cases program
    simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram
