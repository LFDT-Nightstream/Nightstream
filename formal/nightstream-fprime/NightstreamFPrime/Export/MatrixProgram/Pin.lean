import NightstreamFPrime.Layout.MatrixProgram.Pin
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.Pin

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def Block.format : Format Block where
  encode := fun block => (list WireForm.format).encode block.values
  decode := fun value => do
    pure ⟨← (list WireForm.format).decode value⟩
  decode_encode := by
    intro block
    cases block
    simp [Format.decode_encode]

end NightstreamFPrime.Layout.MatrixProgram.Pin
