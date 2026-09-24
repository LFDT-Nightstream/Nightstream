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

private def blockEncode (block : Block) : Value :=
  match block with
  | .ordinary block => .array [.atom 0, Ordinary.Block.format.encode block]
  | .ordinaryTemplate block rows => .array [.atom 6, Ordinary.Block.format.encode block, Affine.Table.format.encode rows]
  | .multiplicationGrid block => .array [.atom 4, MultiplicationGrid.Block.format.encode block]
  | .phi81Product block => .array [.atom 3, Phi81Product.Block.format.encode block]
  | .pin block => .array [.atom 1, Pin.Block.format.encode block]
  | .poseidon block => .array [.atom 2, Poseidon.Block.format.encode block]
  | .mapped sourceWidth projection block =>
      .array [.atom 5, .atom sourceWidth, SourceProjection.format.encode projection, blockEncode block]

private def blockDecode (value : Value) : Except String Block :=
  match value with
  | .array [.atom 0, block] => do
      pure (.ordinary (← Ordinary.Block.format.decode block))
  | .array [.atom 6, block, rows] => do
      pure (.ordinaryTemplate (← Ordinary.Block.format.decode block) (← Affine.Table.format.decode rows))
  | .array [.atom 4, block] => do
      pure (.multiplicationGrid (← MultiplicationGrid.Block.format.decode block))
  | .array [.atom 3, block] => do
      pure (.phi81Product (← Phi81Product.Block.format.decode block))
  | .array [.atom 1, block] => do
      pure (.pin (← Pin.Block.format.decode block))
  | .array [.atom 2, block] => do
      pure (.poseidon (← Poseidon.Block.format.decode block))
  | .array [.atom 5, .atom sourceWidth, projection, block] => do
      pure (.mapped sourceWidth (← SourceProjection.format.decode projection) (← blockDecode block))
  | _ => .error "invalid production matrix block"
  termination_by sizeOf value

def Block.format : Format Block where
  encode := blockEncode
  decode := blockDecode
  decode_encode := by
    intro block
    induction block with
    | ordinary block => simp [blockEncode, blockDecode, Format.decode_encode]
    | ordinaryTemplate block rows =>
      simp [blockEncode, blockDecode, Format.decode_encode]
      rfl
    | multiplicationGrid block => simp [blockEncode, blockDecode, Format.decode_encode]
    | phi81Product block => simp [blockEncode, blockDecode, Format.decode_encode]
    | pin block => simp [blockEncode, blockDecode, Format.decode_encode]
    | poseidon block => simp [blockEncode, blockDecode, Format.decode_encode]
    | mapped width projection block ih =>
      simp [blockEncode, blockDecode, Format.decode_encode, ih]
      rfl

def Program.format : Format Program where
  encode := fun program => (list Block.format).encode program.blocks
  decode := fun value => do
    pure ⟨← (list Block.format).decode value⟩
  decode_encode := by
    intro program
    cases program
    simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram
